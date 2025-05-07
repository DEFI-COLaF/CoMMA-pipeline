import dataclasses
import json
from typing import List, Optional, Iterator, Tuple
from threadpoolctl import threadpool_limits
import uuid

from kraken import rpred
from kraken.containers import BBoxLine, Segmentation, ProcessingStep
from kraken.lib import models
from kraken.lib.progress import KrakenProgressBar
from kraken import serialization
from kraken.lib.xml import XMLPage
from tqdm import tqdm
from PIL import Image

def ocr(
        model: str,
        files: List[Tuple[str, str]],
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
        for inp, out in files:
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
                with open(out, "w") as fp:
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
            yield out


with tqdm() as pbar:
    for i in ocr(
    "catmus-medieval-1.6.0.mlmodel",
    [("bn-f-departement-des-manuscrits-latin-6887/f10-np.xml", "bn-f-departement-des-manuscrits-latin-6887/f10-np.xml")],
        pbar=pbar,
        template="template.xml",
        sub_line_segmentation=False,
        custom_template=True
    ):
        print(i)