import uuid
import os
import dataclasses
from functools import partial

from threadpoolctl import threadpool_limits
from concurrent.futures.thread import ThreadPoolExecutor
from typing import List, Optional, Iterator, Tuple, Callable, Union

from kraken import rpred
from kraken.containers import ProcessingStep
from kraken.lib import models
from kraken import serialization
from kraken.lib.xml import XMLPage
from tqdm import tqdm
from PIL import Image
from rtk import utils
from rtk.task import Task, InputType, InputListType, _sbmsg

def ocr(
        files: List[str],
        model: str,
        device: str = "cpu",
        pbar: Optional[tqdm] = None,
        text_direction: str = "L",
        pad: int = 16,
        custom_template: bool = False,
        template: str = "alto",
        sub_line_segmentation: bool = False
) -> Iterator[str]:
    with threadpool_limits(limits=1):
        nm = models.load_any(model, device=device)
        step = ProcessingStep(id=f'_{uuid.uuid4()}',
                              category='processing',
                              description='Text line recognition',
                              settings={
                                  'text_direction': text_direction, 'models': model, 'pad': 16, 'bidi_reordering': True
                              })
        for inp in files:
            try:
                doc = XMLPage(inp)
                bidi_reordering = doc.base_dir
                bounds = doc.to_container()
            except Exception as E:
                print("[ERROR] Kraken Parsing Issue")
                print(E)
                continue
            try:
                im = Image.open(doc.imagename)
                imsize = im.size
            except IOError as E:
                print("[ERROR] Kraken Pillow Image Issue")
                print(E)
                continue

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
                print(E)
                continue

            try:
                with open(inp, "w") as fp:
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
            except Exception as E:
                print("[ERROR] Kraken Serialization Issue")
                print(E)
                continue

            pbar.update(1)
            yield inp




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
        inputs = utils.split_batches(inputs, self.workers)
        tp = ThreadPoolExecutor(len([batches for batches in inputs if len(batches)]))
        bar = tqdm(desc=_sbmsg(f"Processing {self.desc} command"), total=total_texts)
        ocr_fn = partial(
            ocr,
            model=self.model_path, pbar=bar,
            template=self.template, custom_template=self.template not in {"alto", "pagexml"},
            sub_line_segmentation=self.subline_segmentation
        )
        for gen in tp.map(ocr_fn, inputs, timeout=max([len(b) for b in inputs])*self.max_time_per_op):
            for elem in gen:
                if isinstance(elem, str):
                    self._output_files.append(elem)
        bar.close()
        if len(self._output_files) == len(self.input_files):
            return True
        return False