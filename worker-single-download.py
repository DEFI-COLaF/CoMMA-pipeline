import asyncio
import os
import time
import datetime
import glob
import csv
import argparse
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Set
from collections import defaultdict, deque
from urllib.parse import urlparse
from PIL import Image

import pandas as pd
import tqdm
import unidecode
from rtk.task import DownloadIIIFImageTask, DownloadIIIFManifestTask
from rtk import utils
from lib.rtk_adapt import Manifest
import cases


# Constants
DOWNLOAD_BATCH_SIZE = int(os.getenv("BATCH_SIZE", 5))   # Number of manifests to download in parallel
RETRY_LIMIT = 1                                         # How many times to retry a manifest
RETRY_NO_OPTIONS = 2
RETRY_DELAY = 10                                        # Seconds to wait before retrying a failed manifest
MAX_QUEUE_SIZE = 1240*60                                # Number of batch that we can keep without processing
SLEEP_TIME_BETWEEN_POOL_CHECK = 20
MANIFEST_DIRECTORY: str = "output"


# Represents a successfully downloaded image and which manifest it belongs to
@dataclass
class DownloadedImage:
    manifest_id: str
    image_path: Path

# Tracks manifest completeness and coordinates GZIP/completion logic
class ManifestTracker:
    def __init__(self, worker: int):
        self.worker: int = worker
        self.expected: Dict[str, int] = defaultdict(int)               # manifest_id → expected image count
        self.completed: Dict[str, Set[Path]] = defaultdict(set)        # manifest_id → list of completed files
        self.retry_counts: Dict[str, int] = defaultdict(int)           # manifest_id → how many times we’ve retried
        self.done: Set[str] = set(self._load("done.txt"))              # already processed manifests (from done.txt)
        for done in glob.glob("done-w*.txt"):
            self.done = self.done.union(self._load(done))
        self.shamelist: Set[str] = set(self._load("shame-list.txt"))   # already processed manifests (from done.txt)
        for shamelist in glob.glob("shame-list-w*.txt"):
            self.shamelist = self.shamelist.union(self._load(shamelist))
        self.manifest_to_directory: Dict[str, str] = {}
        self.directory_to_manifest: Dict[str, str] = {}
        self.order: Dict[str, List] = defaultdict(list)

    def _load(self, file) -> List[str]:
        try:
            with open(file, "r") as f:
                return f.read().split()
        except FileNotFoundError:
            return []

    def is_complete(self, manifest_id: str):
        return len(self.completed[manifest_id]) == self.expected[manifest_id]

    def mark_done(self, manifest_id: str):
        # Add manifest to the done list and persist to disk
        self.done.add(manifest_id)
        with open(f"done-w{self.worker}.txt", "w") as f:
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


def alternate_by_domain(url_series: pd.Series) -> pd.Series:
    # Extract domain names
    domains = url_series.apply(lambda u: urlparse(u).netloc)

    domain_counts = domains.value_counts(normalize=True) * 100
    print("Domain ratios (%):")
    for domain, ratio in domain_counts.items():
        print(f"  {domain}: {ratio:.2f}%")

    # Group URLs by domain, preserving order
    domain_groups = defaultdict(deque)
    for url, domain in zip(url_series, domains):
        domain_groups[domain].append(url)

    # Alternate between domains
    output = []
    domain_keys = list(domain_groups.keys())
    while any(domain_groups.values()):
        for domain in domain_keys:
            if domain_groups[domain]:
                output.append(domain_groups[domain].popleft())

    return pd.Series(output, name="manifest_url")


def split_work(items: list, max_workers: int, index: int) -> list:
    """
    Splits the list of items so that each worker gets every `max_workers`-th item starting from its index.
    """
    index -= 1
    if index < 0 or index >= max_workers:
        raise ValueError("index must be between 0 and max_workers - 1")
    return items[index::max_workers]

def rename_manifest_download(
        uri: str,
        naming_function: Callable[[str], str],
        output_directory: str = MANIFEST_DIRECTORY
) -> str:
    return os.path.join(
        output_directory, utils.change_ext(naming_function(uri), "csv"))



def parse_manifest(file: str) -> List[Tuple[str, str, str]]:
    with open(file) as f:
        files = list([tuple(row) for row in csv.reader(f)])
    return files


def rename_image_download(image_detail: Tuple[str, str, str]) -> str:
    return os.path.join(image_detail[1], f"{image_detail[2]}.jpg")


def kebab_with_fallback(string: str) -> str:
    if "BSB " in string or "bsb " in string.lower():
        return cases.to_kebab(unidecode.unidecode(string.split("-")[-1].strip()))
    else:
        return cases.to_kebab(unidecode.unidecode(string))


def single_download(tracker: ManifestTracker, manifests: List[str]):
    for manifest_uri in manifests:
        print(f"[Downloader] Downloading manifest {manifest_uri}")
        print(f"[TIME] {datetime.datetime.now()}")
        manifest_csv = rename_manifest_download(manifest_uri, cases.to_kebab)
        requires_download = not os.path.exists(manifest_csv)

        if requires_download:
            try:
                result = utils.download_iiif_manifest(manifest_uri, manifest_csv, naming_function=kebab_with_fallback)
                if not result:
                    print("\t[Details] Manifest undownloadable")
                    with open(f"shame-list-w{tracker.worker}.txt", "a") as f:
                        f.writelines([str(manifest_uri) + "\n"])
                    continue
            except Exception as E:
                print(f"\t[ERROR]{E}")
                continue

        images_details = parse_manifest(manifest_csv)

        # Compute how many images we have to do per manuscript
        image_count = len(images_details)
        # We still record stuff in the tracker, just in case
        for (_, output_directory, filename) in images_details:
            tracker.register_dir(manifest_uri, output_directory)
            tracker.record_image_order(manifest_uri, filename)
        # Then register this expectation
        tracker.add_expected(manifest_uri, image_count)

        cased = Path(tracker.manifest_to_directory[manifest_uri]).name

        if len(glob.glob(f"targz/**/{cased}.tar.gz", recursive=True)):
            print(f"\ttargz/**/{cased}.tar.gz exists")
            tracker.mark_done(manifest_uri)
            continue

        # We rewrite the json just in case
        m = Manifest(
            manifest_id=manifest_uri,
            directory=tracker.manifest_to_directory[manifest_uri],
            image_order=tracker.order[manifest_uri],
            total_images=tracker.expected[manifest_uri]
        )
        m.to_json()

        # Now we prepare the images
        print(f"\t[Details] {len(images_details)} in the manifest")
        if len(images_details):
            print(f"\t[Details] {images_details[0][1]} is the directory")
        # We avoid webp because it's not cool
        images_details: List[Tuple[str, str, str]] = [
            (el[0], *el[1:]) for el in images_details
        ]
        print(f"\t[Details] [TIME] {datetime.datetime.now()}")
        print("Checking preprocessed")
        images_to_download: List[Tuple[str, str, str]] = []
        for image in tqdm.tqdm(images_details):
            needs_downloading = True
            image_path = rename_image_download(image)
            if os.path.exists(image_path):
                try:
                    _ = Image.open(image_path).tobytes()
                    needs_downloading = False
                except Exception as E:
                    needs_downloading = True
            if needs_downloading:
                images_to_download.append(image)

        print(f"\t[Details] {len(images_to_download)} images to process remaining")
        errors = 0
        aborted = False
        failed = []
        for image in tqdm.tqdm(images_to_download):
            result = utils.download_iiif_image(
                image[0],
                rename_image_download(image),
                options = {"max_height": 2500},
                retries = RETRY_LIMIT,
                retries_no_options = RETRY_NO_OPTIONS,
                time_between_retries = RETRY_DELAY
            )
            if not result:
                errors += 1
                m.add_errors(image[0])
                # At a maximum of 10% of errors for 50 images or more, we forget about this manuscript
                if len(images_details) > 30 and errors / (len(images_details)) > .1:
                    print("\t[ERROR] Too much errors (>10% of 4xx/5xx), moving to next manuscript.")
                    m.to_json()
                    aborted = True
                    continue
        if aborted:
            with open(f"shame-list-w{tracker.worker}.txt", "a") as f:
                f.writelines([str(manifest_uri)+"\n"])
            #tracker.mark_done(manifest_uri)
        print(f"MANIFEST {manifest_uri} ==> ({len(m.found_images())}/{len(m.image_order)}")
        print(f"\t[Details] Directory is {m.directory}")

        # Now check if pause !

        while (len(glob.glob("./*/*.jpg")) - len(glob.glob("./*/*.xml")) - 1000) >= MAX_QUEUE_SIZE:
            print("[WAIT] Waiting for some queue space")
            time.sleep(SLEEP_TIME_BETWEEN_POOL_CHECK)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split work among workers.")
    parser.add_argument('--max', type=int, required=True, help='Total number of workers')
    parser.add_argument('--index', type=int, required=True, help='Index of this worker')

    args = parser.parse_args()

    tracker = ManifestTracker(args.index)

    # Load manifests and filter out already completed ones
    df = pd.read_csv("extraction_biblissima_20250410.csv", delimiter=";")["manifest_url"]
    df = df.unique().tolist()
    df = df + pd.read_csv("biblissima_bodleian.csv", delimiter=";")["manifest_url"].unique().tolist()
    uri_renamer = lambda u: u.replace("https://gallica.bnf.fr/iiif/ark:/12148/", "https://openapi.bnf.fr/iiif/presentation/v3/ark:/12148/")
    df = [
        uri_renamer(uri) if uri_renamer(uri) not in tracker.shamelist else uri # Keep good old URIs
        for uri in df
    ]
    df = [uri for uri in df if uri not in tracker.done and uri not in tracker.shamelist]
    df = alternate_by_domain(pd.Series(df)).tolist()


    assigned_items = split_work(df, args.max, args.index)

    # print(df)
    # Launch producer and consumer
    print("[Main] Starting downloader")
    #df = pd.read_csv("extraction_biblissima_20250410.csv", delimiter=";")["manifest_url"]
    single_download(tracker, assigned_items)
