# Allow running from anywhere: put the repo root (parent of scripts/) on sys.path
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))


import glob
from lib.rtk_adapt import Manifest
from rtk.utils import download_iiif_image
import csv
import os
import anycase as cases
from typing import List, Tuple
from rtk.task import DownloadIIIFManifestTask


dl_manifests = DownloadIIIFManifestTask(
    [],
    output_directory="output",
    naming_function=cases.to_kebab,
    multiprocess=1
)


# [(Url, Target)]
require_download: List[Tuple[str, str, str]] = []


for file in glob.glob(f"{os.getenv('DATA_DIR', 'data-in-process')}/*/.manifest.json") + glob.glob("./*/.manifest.json"):
    manifest = Manifest.from_json(file)
    images = manifest.image_order
    existing_images = [os.path.basename(f) for f in glob.glob(f"{manifest.directory}/*.jpg")]
    missings: List[Tuple[int, str]] = []
    for idx, image in enumerate(images):
        if f"{image}.jpg" not in existing_images:
            missings.append((idx, image))
    if missings and len(glob.glob(f"{manifest.directory}/*.xml")):
        data = dl_manifests.parse_cache(uri=manifest.manifest_id)
        for image_idx, image_name in missings:
            require_download.append(
                (manifest.manifest_id, data[image_idx][0], f"{manifest.directory}/{image_name}.jpg")
            )

with open("missing.csv", "w") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow(["Manifest", "URI", "Path"])
    w.writerows(require_download)