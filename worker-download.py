import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict

import pandas as pd
from rtk.task import DownloadIIIFImageTask, DownloadIIIFManifestTask, KrakenRecognizerCommand
from rtk_adapt import YaltoCommand, create_tar_gz_archives
from rtk import utils
import shutil
import cases

# Constants
DOWNLOAD_BATCH_SIZE = 10          # Number of manifests to download in parallel
YOLO_BATCH_SIZE = 64              # Batch size for YALTAi segmentation
KRAKEN_BATCH_SIZE = 20            # Batch size for Kraken OCR
PROCESS_BATCH_SIZE = 5          # Number of images to process per queue job
RETRY_LIMIT = 3                   # How many times to retry a manifest
RETRY_DELAY = 120                 # Seconds to wait before retrying a failed manifest
MAX_QUEUE_SIZE = 3 #1000 // PROCESS_BATCH_SIZE  # Number of batch that we can keep without processing


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
async def download_worker(queue: asyncio.Queue, tracker: ManifestTracker, manifest_urls: List[str]):
    kebab = cases.to_kebab

    loop = asyncio.get_event_loop()

    # Break up manifest list into smaller download groups
    manifest_batches = [manifest_urls[i:i+DOWNLOAD_BATCH_SIZE] for i in range(0, len(manifest_urls), DOWNLOAD_BATCH_SIZE)]

    for batch in manifest_batches:
        print("[Downloader] Downloading manifests")
        dl_manifests = DownloadIIIFManifestTask(
            batch,
            output_directory="output",
            naming_function=kebab,
            multiprocess=DOWNLOAD_BATCH_SIZE
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

        # To allow for a more frequent download -> process system, we treat image download
        #   in a separate batching approach
        for i in range(0, len(images_to_download), PROCESS_BATCH_SIZE):
            # We simplify the variable
            image_download_batch = images_to_download[i:i+PROCESS_BATCH_SIZE]
            print("[Downloader] Downloading images")
            dl_images = DownloadIIIFImageTask(
                image_download_batch,
                max_height=2500,
                multiprocess=DOWNLOAD_BATCH_SIZE,
                downstream_check=DownloadIIIFImageTask.check_downstream_task("xml", utils.check_parsable)
            )

            # We have a maximum set of retries for downloading images
            #   And we check that everything was downloaded
            retries = 0
            while len(set(dl_images.output_files)) < len(image_download_batch) and retries < RETRY_LIMIT:
                if retries > 0:
                    print("[Downloader] Incomplete image download, retrying after sleep...")
                    await asyncio.sleep(RETRY_DELAY)
                dl_images.process()
                retries += 1

            # Recorde the image download
            downloaded_images = sorted(list(set(dl_images.output_files)))
            # We record for each image to download a Filepath<>Manifest URI link
            #    img_tuple[1] is the output directory
            image_path_to_uri: Dict[str, str] = {
                dl_images.rename_download(image_tuple): tracker.directory_to_manifest[image_tuple[1]]
                for image_tuple in image_download_batch
            }

            # Now, we look into what was actually downloaded
            #   and map image to a DownloadedImage type
            downloaded_batch = [
                DownloadedImage(manifest_id=image_path_to_uri[image_path], image_path=Path(image_path))
                for image_path in downloaded_images if image_path in image_path_to_uri
            ]
            await queue.put(downloaded_batch)
    await queue.put("END")
    return


# Orchestrates downloader and processor, manages manifest list and queue
async def main():
    queue = asyncio.Queue(maxsize=MAX_QUEUE_SIZE)  # Backpressure: max number of pending batches
    tracker = ManifestTracker()

    # Load manifests and filter out already completed ones
    df = pd.read_csv("extraction_biblissima_20250410.csv", delimiter=";")["manifest_url"].tolist()
    df = [
        uri.replace("https://gallica.bnf.fr/iiif/ark:/12148/", "https://openapi.bnf.fr/iiif/presentation/v3/ark:/12148/")
        for uri in df if uri not in tracker.done
    ]

    # Launch producer and consumer
    print("[Main] Starting producer and consumer")
    producer = asyncio.create_task(download_worker(queue, tracker, df))
    consumer = asyncio.create_task(process_worker(queue, tracker))

    # Wait for producer to finish and queue to empty
    await producer  # Await producer completion
    await queue.join()  # Ensure the queue has been fully processed
    consumer.cancel()  # Cancel consumer once queue is empty

# Entrypoint
if __name__ == "__main__":
    asyncio.run(main())