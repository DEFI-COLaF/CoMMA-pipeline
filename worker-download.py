import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Set
from collections import defaultdict

import glob
import pandas as pd
from rtk.task import DownloadIIIFImageTask, DownloadIIIFManifestTask
from rtk import utils
from rtk_adapt import Manifest
import cases

# Constants
DOWNLOAD_BATCH_SIZE = 15         # Number of manifests to download in parallel
PROCESS_BATCH_SIZE = 1000        # Number of images to process per queue job
RETRY_LIMIT = 1                  # How many times to retry a manifest
RETRY_DELAY = 10                 # Seconds to wait before retrying a failed manifest
MAX_QUEUE_SIZE = 1240*4          # Number of batch that we can keep without processing
SLEEP_TIME_BETWEEN_POOL_CHECK = 20

# Represents a successfully downloaded image and which manifest it belongs to
@dataclass
class DownloadedImage:
    manifest_id: str
    image_path: Path

# Tracks manifest completeness and coordinates GZIP/completion logic
class ManifestTracker:
    def __init__(self):
        self.expected: Dict[str, int] = defaultdict(int)        # manifest_id → expected image count
        self.completed: Dict[str, Set[Path]] = defaultdict(set)  # manifest_id → list of completed files
        self.retry_counts: Dict[str, int] = defaultdict(int)    # manifest_id → how many times we’ve retried
        self.done: Set[str] = set(self._load_done())            # already processed manifests (from done.txt)
        self.manifest_to_directory: Dict[str, str] = {}
        self.directory_to_manifest: Dict[str, str] = {}
        self.order: Dict[str, List] = defaultdict(list)

    def _load_done(self) -> List[str]:
        try:
            with open("done.txt", "r") as f:
                return f.read().split()
        except FileNotFoundError:
            return []

    def is_complete(self, manifest_id: str):
        return len(self.completed[manifest_id]) == self.expected[manifest_id]

    def mark_done(self, manifest_id: str):
        # Add manifest to the done list and persist to disk
        self.done.add(manifest_id)
        with open("done.txt", "w") as f:
            f.write("\n".join(sorted(self.done)))

    def add_expected(self, manifest_id: str, count: int):
        # Record how many images we expect to process for this manifest
        self.expected[manifest_id] = count

    def record_image_order(self, manifest_id: str, record_image: str):
        self.order[manifest_id].append(record_image)

    def register_dir(self, manifest_id: str, directory: str):
        self.manifest_to_directory[manifest_id] = directory
        self.directory_to_manifest[directory] = manifest_id

    def add_completed(self, manifest_id: str, path: Path):
        # Add a successfully processed XML to the manifest's completed set
        self.completed[manifest_id].add(path)

# Downloads manifests and images, puts image batches on queue for processing
def download_worker(tracker: ManifestTracker, manifest_urls: List[str]):
    kebab = cases.to_kebab

    # Break up manifest list into smaller download groups
    manifest_batches = [manifest_urls[i:i+DOWNLOAD_BATCH_SIZE] for i in range(0, len(manifest_urls), DOWNLOAD_BATCH_SIZE)]

    for batch in manifest_batches:
        print("[Downloader] Downloading manifests")
        dl_manifests = DownloadIIIFManifestTask(
            batch,
            output_directory="output",
            naming_function=kebab,
            multiprocess=DOWNLOAD_BATCH_SIZE // 2
        )
        dl_manifests.process()

        # We make a copy of the list of image to download
        # Tuple[Image URI, Directory, Filename]
        images_to_download: List[Tuple[str, str, str]] = dl_manifests.output_files.copy()

        # Compute how many images we have to do per manuscript
        image_count = defaultdict(lambda: 0)
        for ((_, output_directory, filename), uri) in dl_manifests.output_files_map.items():
            image_count[uri] += 1
            tracker.register_dir(uri, output_directory)
            tracker.record_image_order(uri, filename)

        # Then register this expectation
        for uri in image_count:
            tracker.add_expected(uri, image_count[uri])

        for manifest_uri in tracker.manifest_to_directory:
            m = Manifest(
                manifest_id=manifest_uri,
                directory=tracker.manifest_to_directory[manifest_uri],
                image_order=tracker.order[manifest_uri],
                total_images=tracker.expected[manifest_uri]
            )
            m.to_json()

        # To allow for a more frequent download -> process system, we treat image download
        #   in a separate batching approach
        for i in range(0, len(images_to_download), PROCESS_BATCH_SIZE):
            # We simplify the variable
            image_download_batch = images_to_download[i:i+PROCESS_BATCH_SIZE]
            print("[Downloader] Downloading images")
            dl_images = DownloadIIIFImageTask(
                image_download_batch,
                max_height=2500,
                retries_no_options=RETRY_LIMIT,
                retries=RETRY_LIMIT,
                time_between_retries=RETRY_DELAY,
                multiprocess=DOWNLOAD_BATCH_SIZE,
                #downstream_check=DownloadIIIFImageTask.check_downstream_task("xml", utils.check_parsable)
            )
            dl_images.process()
            #retries = 0
            #while len(set(dl_images.output_files)) < len(image_download_batch) and retries < RETRY_LIMIT:
            #    if retries > 0:
            #        print("[Downloader] Incomplete image download, retrying after sleep...")
            #        time.sleep(RETRY_DELAY)
            #    dl_images.process()
            #    retries += 1

            while (len(glob.glob("./*/*.jpg"))-len(glob.glob("./*/*.xml"))) >= MAX_QUEUE_SIZE:
                print("Waiting for some queue space")
                time.sleep(SLEEP_TIME_BETWEEN_POOL_CHECK)

        for manifest_uri in tracker.manifest_to_directory:
            m = Manifest(
                manifest_id=manifest_uri,
                directory=tracker.manifest_to_directory[manifest_uri],
                image_order=tracker.order[manifest_uri],
                total_images=tracker.expected[manifest_uri]
            )
            if len(m.image_order) > 1 and len(m.found_images()) == 0:
                tracker.mark_done(manifest_uri)
                print(f"Giving up on {manifest_uri}")
    with open(".totally_done", "w") as f:
        f.write("")


# Entrypoint
if __name__ == "__main__":
    tracker = ManifestTracker()

    # Load manifests and filter out already completed ones
    df = pd.read_csv("extraction_biblissima_20250410.csv", delimiter=";")["manifest_url"].tolist()
    df = [
        uri.replace("https://gallica.bnf.fr/iiif/ark:/12148/", "https://openapi.bnf.fr/iiif/presentation/v3/ark:/12148/")
        for uri in df
    ]
    df = [uri for uri in df if uri not in tracker.done]
    # Launch producer and consumer
    print("[Main] Starting downloader")
    download_worker(tracker, df)
