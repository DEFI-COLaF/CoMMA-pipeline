""" This is a sample script for using RTK (Release the krakens)

It takes a file with a list of manifests to download from IIIF (See manifests.txt) and passes it in a suit of commands:

0. It downloads manifests and transform them into CSV files
1. It downloads images from the manifests
2. It applies YALTAi segmentation with line segmentation
3. It fixes up the image PATH of XML files
4. It processes the text as well through Kraken
5. It removes the image files (from the one hunder object that were meant to be done in group)

The batch file should be lower if you want to keep the space used low, specifically if you use DownloadIIIFManifest.

"""
from collections import defaultdict

from rtk.task import DownloadIIIFImageTask, KrakenAltoCleanUpCommand, ClearFileCommand, \
    DownloadIIIFManifestTask, YALTAiCommand, KrakenRecognizerCommand, ExtractZoneAltoCommand
from rtk_adapt import YaltoCommand, create_tar_gz_archives
from rtk import utils
import pandas as pd
import cases
from typing import Tuple, Dict, List
from pathlib import Path
import shutil
import time

BATCH_SIZE = 16
YOLO_BATCH_SIZE = 64
KRAKEN_BATCH_SIZE = 20
SLEEP_TIME = 120
DOWNLOAD_BATCH_SIZE = 10

df = pd.read_csv("extraction_biblissima_20250410.csv", delimiter=";")["manifest_url"].tolist()
df = [
    uri.replace("https://gallica.bnf.fr/iiif/ark:/12148/", "https://openapi.bnf.fr/iiif/presentation/v3/ark:/12148/")
    for uri in df
]

with open("done.txt", "r") as f:
    done = set(f.read().split())

df = [x for x in df if x not in done]

batches = [df[i:i+BATCH_SIZE] for i in range(0, len(df), BATCH_SIZE)][1:]

kebab = cases.to_kebab

def process_batch(batch) -> Tuple[List[str], List[str]]:
    print("[Task] Download manifests")
    dl_manifests = DownloadIIIFManifestTask(
        batch,
        output_directory="output",
        naming_function=lambda x: kebab(x),
        multiprocess=DOWNLOAD_BATCH_SIZE
    )
    dl_manifests.process()

    file_map: Dict[Tuple[str, str, str], str] = dl_manifests.output_files_map.copy()

    print("[Task] Download JPG")
    dl_images = DownloadIIIFImageTask(
        dl_manifests.output_files,
        max_height=2500,
        multiprocess=DOWNLOAD_BATCH_SIZE,
        downstream_check=DownloadIIIFImageTask.check_downstream_task("xml", utils.check_parsable)
    )
    dl_images.process()

    # Retry once if image download count is off
    if len(dl_manifests.output_files) != len(dl_images.output_files):
        print("[Warning] Image download incomplete. Waiting 60 seconds before retry...")
        time.sleep(SLEEP_TIME)
        dl_images.process()

        if len(dl_manifests.output_files) != len(dl_images.output_files):
            print("[Warning] Still mismatch in manifest/image file count after retry.")
            print(f"Manifests: {len(dl_manifests.output_files)}, Images: {len(dl_images.output_files)}")

    # Update file_map for renamed images
    file_map: Dict[str, str] = {
        dl_images.rename_download(file): uri
        for file, uri in file_map.items()
    }

    xmls = YaltoCommand(
        list(set(dl_images.output_files)),
        binary="yolalto",
        model_path="medieval-yolov11x.pt",
        batch_size=YOLO_BATCH_SIZE,
        check_content=utils.check_parsable
    )
    xmls.process()

    print("[Task] OCR")
    kraken = KrakenRecognizerCommand(
        xmls.output_files,
        binary="kraken",
        device="cuda:0",
        template="template.xml",
        model="catmus-medieval-1.6.0.mlmodel",
        raise_on_error=True,
        multiprocess=KRAKEN_BATCH_SIZE,
        check_content=True
    )
    kraken.process()

    outs = defaultdict(list)
    for file in set(kraken.output_files):
        jpg = file.replace(".xml", ".jpg")
        if jpg in file_map:
            outs[file_map[jpg]].append(Path(file))

    mss_lengths = defaultdict(lambda: 0)
    for file in file_map:
        mss_lengths[file_map[file]] += 1

    complete, incomplete = [], []
    for key in outs:
        if len(outs[key]) == mss_lengths[key]:
            complete.append(key)
        else:
            incomplete.append(key)

    ordered = {
        uri: [Path(dl_images.rename_download(row)) for row in dl_manifests.parse_cache(uri)]
        for uri in complete
    }

    print("COMP:", complete)
    print("INCOMP:", incomplete)

    with open("done.txt", "r") as f:
        done = f.read().split()
    with open("done.txt", "w") as f:
        f.write("\n".join(sorted(list(set(done + complete)))))

    print("[Task] GZIPING")
    create_tar_gz_archives(
        uri_to_files={k: v for k, v in outs.items() if k in complete},
        ordering_dict={k: v for k, v in ordered.items() if k in complete},
        naming_func=lambda x: Path("targz") / Path(str(ordered[x][0].parent.name) + ".tar.gz")
    )

    for uri in complete:
        dirname = Path("./" + str(ordered[uri][0].parent.name))
        shutil.rmtree(dirname, ignore_errors=True)

    return complete, incomplete


# --- RUN LOOP WITH RETRY ---
for batch in batches:
    _, incomplete = process_batch(batch)

    # Retry once if there are incomplete files
    if incomplete:
        print(f"[Retry] Re-processing incomplete items in batch: {incomplete}")
        _, retry_incomplete = process_batch(incomplete)

        if retry_incomplete:
            print(f"[Warning] Still incomplete after retry: {retry_incomplete}")
