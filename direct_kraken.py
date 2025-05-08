import uuid
import os
import dataclasses
from functools import partial
import time

from threadpoolctl import threadpool_limits
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
from typing import List, Optional, Iterator, Tuple, Callable, Union

from kraken import rpred
from kraken.containers import ProcessingStep
from kraken.lib import models
from kraken import serialization
from kraken.lib.xml import XMLPage
from tqdm import tqdm
from PIL import Image
from rtk.task import Task, InputType, InputListType, _sbmsg


def ocr(
        input_file: str,
        model: str,
        device: str = "cpu",
        text_direction: str = "L",
        pad: int = 16,
        custom_template: bool = False,
        template: str = "alto",
        sub_line_segmentation: bool = False
) -> Optional[str]:

    with threadpool_limits(limits=1):
        nm = models.load_any(model, device=device)
        step = ProcessingStep(id=f'_{uuid.uuid4()}',
                              category='processing',
                              description='Text line recognition',
                              settings={
                                  'text_direction': text_direction, 'models': model, 'pad': 16, 'bidi_reordering': True
                              })
        try:
            doc = XMLPage(input_file)
            bidi_reordering = doc.base_dir
            bounds = doc.to_container()
        except Exception as E:
            print("[ERROR] Kraken Parsing Issue")
            print(f"File {input_file}: {E}")
            return None

        try:
            im = Image.open(doc.imagename)
            imsize = im.size
        except IOError as E:
            print("[ERROR] Kraken Pillow Image Issue")
            print(f"File {input_file}: {E}")
            return None

        try:
            # Ensure correct bounds
            lines = bounds.lines
            def minmax(x: int, threshold: int):
                return int(min(max(x, 0), threshold))
            def normalize(l: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
                return list([
                    (
                        minmax(x, imsize[0]-1),
                        minmax(y, imsize[1]-1)
                    )
                    for x, y in l
                ])

            for line in lines:
                line.boundary = normalize(line.boundary)
                line.baseline = normalize(line.baseline)
        except Exception as E:
            print("[ERROR] Kraken normalizing lines")
            print(f"File {input_file}: {E}")
            return None


        try:
            preds = []
            it = rpred.rpred(
                nm, im, bounds, pad, bidi_reordering=bidi_reordering, no_legacy_polygons=False
            )
            for pred in it:
                preds.append(pred)
            results = dataclasses.replace(it.bounds, lines=preds, imagename=doc.imagename)
        except Exception as E:
            print("[ERROR] Kraken Pred Issue")
            print(f"File {input_file}: {E}")
            return None

        try:
            with open(input_file, "w") as fp:
                fp.write(
                    serialization.serialize(
                        results=results,
                        image_size=imsize,
                        writing_mode=text_direction,
                        scripts=None,
                        template=template,
                        template_source='custom' if custom_template else 'native',
                        processing_steps=[step],
                        sub_line_segmentation=sub_line_segmentation
                    )
                )
                return input_file
        except Exception as E:
            print("[ERROR] Kraken Serialization Issue")
            print(f"File {input_file}: {E}")
            return None


class KrakenDirectTask(Task):
    """ Runs a Kraken Like command (Kraken, YALTAi)

    KrakenLikeCommand expect `$out` in its command
    """

    def __init__(
            self,
            *args,
            model: str,
            template: Optional[str] = "alto",
            subline_segmentation: bool = False,
            desc: Optional[str] = "direct-kraken",
            check_content: Union[Callable[[str], bool], bool] = False,
            workers: int = 1,
            custom_rename: Callable[[str], str] = None,
            max_time_per_op: int = 120,  # Seconds
            **kwargs
    ):
        super(KrakenDirectTask, self).__init__(*args, **kwargs)
        self.model_path = model
        self.template: str = template
        self.check_content: Union[Callable[[str], bool], bool] = check_content
        self._output_files: List[str] = []
        self.max_time_per_op: int = max_time_per_op
        self.desc: str = desc
        self.subline_segmentation: bool = subline_segmentation
        self.rename: Callable[[str], str] = custom_rename or self._rename
        self.workers: int = workers
        self.check_fn: Callable[[str], bool] = check_content or self._base_check

    def _rename(self, inp: str) -> str:
        return os.path.splitext(inp)[0] + ".xml"

    @property
    def output_files(self) -> List[InputType]:
        return list([
            self.rename(file)
            for file in self._output_files
        ])

    def _base_check(self, inp: str) -> Tuple[str, bool]:
        out = self.rename(inp)
        if os.path.exists(out):
            return inp, self.check_content(out)
        else:
            return inp, False

    def check(self) -> bool:
        all_done: bool = True
        pbar = tqdm(desc=_sbmsg("Checking prior processed documents"), total=len(self.input_files))
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            for (inp, status) in pool.map(self.check_fn, self.input_files):
                self._checked_files[inp] = status
                if not status:
                    all_done = False
                pbar.update(1)
        self._output_files.extend([self.rename(inp) for inp, status in self._checked_files.items() if status])
        return all_done

    def _process(self, inputs: InputListType) -> bool:
        """ Use parallel """
        # Group inputs into the number of workers
        total_texts = len(inputs)
        bar = tqdm(desc=_sbmsg(f"Processing {self.desc} command"), total=total_texts)

        ocr_fn = partial(
            ocr,
            model=self.model_path,
            template=self.template, custom_template=self.template not in {"alto", "pagexml"},
            sub_line_segmentation=self.subline_segmentation
        )
        if self.workers <= 1:
            print("Warning: Workers set to 1. This won't enable parallelism.")
        with ProcessPoolExecutor(max_workers=self.workers) as pool:
            futures = {pool.submit(ocr_fn, inp) for inp in inputs}

            for future in as_completed(futures):
                try:
                    result = future.result(timeout=self.max_time_per_op*3)
                    if isinstance(result, str):
                        self._output_files.append(result)
                        bar.update(1)
                except Exception as e:
                    print(f"Task failed: {e}")
        bar.close()
        if len(self._output_files) == len(self.input_files):
            return True
        return False