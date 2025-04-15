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
from typing import Tuple, Dict
from pathlib import Path
import shutil

BATCH_SIZE = 1
df = pd.read_csv("extraction_biblissima_20250410.csv", delimiter=";")["manifest_url"].tolist()
df = [
    uri.replace("https://gallica.bnf.fr/iiif/ark:/12148/", "https://openapi.bnf.fr/iiif/presentation/v3/ark:/12148/")
    for uri in df
]

with open("done.txt", "r") as f:
    done = set(f.read().split())

done = [] # DEBUG MODE
df = [x for x in df if x not in done]

batches = [df[i:i+BATCH_SIZE] for i in range(0, len(df), BATCH_SIZE)]

kebab = cases.to_kebab


for batch in batches:
    # Add a logic to check if this document was wholy processed

    # Download Manifests
    print("[Task] Download manifests")
    dl_manifests = DownloadIIIFManifestTask(
        batch,
        output_directory="output",
        naming_function=lambda x: kebab(x), multiprocess=10
    )
    dl_manifests.process()

    # Tuple to Manifest URI
    file_map: Dict[Tuple[str, str, str], str] = dl_manifests.output_files_map.copy()

    # Download Files
    print("[Task] Download JPG")
    dl_images = DownloadIIIFImageTask(
        dl_manifests.output_files,
        max_height=2500,
        multiprocess=4,
        downstream_check=DownloadIIIFImageTask.check_downstream_task("xml", utils.check_parsable)
    )
    dl_images.process()

    file_map: Dict[str, str] = {
        dl_images.rename_download(file): uri
        for file, uri in file_map.items()
    }

    xmls = YaltoCommand(
        dl_images.output_files,
        binary="env/bin/yolalto",
        model_path="medieval-yolov11x.pt",
        batch_size=1,
        check_content=utils.check_parsable
    )
    xmls.process()


    # Apply Kraken
    print("[Task] OCR")
    kraken = KrakenRecognizerCommand(
        xmls.output_files,
        binary="env/bin/kraken",
        device="cpu",
        template="template.xml",
        model="catmus-medieval-1.6.0.mlmodel",
        raise_on_error=True,
        multiprocess=8,  # GPU Memory // 3gb
        check_content=True
    )
    kraken.process()


    outs = defaultdict(list)
    for file in set(kraken.output_files):
        # Naming is known here, no need to make magical stuff...
        jpg = file.replace(".xml", ".jpg")
        if jpg in file_map:
            outs[file_map[jpg]].append(Path(file))

    mss_lengths = defaultdict(lambda: 0)
    for file in file_map:
        mss_lengths[file_map[file]] += 1

    complete = []
    incomplete = []
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
        f.write("\n".join(sorted(list(set(done+complete)))))
    print("[Task] GZIPING")

    create_tar_gz_archives(
        uri_to_files={
            manifest_uri: file_list
            for manifest_uri, file_list in outs.items()
            if manifest_uri in complete
        },
        ordering_dict={
            manifest_uri: file_list
            for manifest_uri, file_list in ordered.items()
            if manifest_uri in complete
        },
        naming_func=lambda x: Path("targz") / Path(str(ordered[x][0].parent.name)+".tar.gz")
    )

    for uri in complete:
        dirname = Path("./"+str(ordered[uri][0].parent.name))
        shutil.rmtree(dirname)
