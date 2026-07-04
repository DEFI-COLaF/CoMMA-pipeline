import os
import time
import datetime
import glob
import csv
import argparse
import tarfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Set, Any, Optional
from collections import defaultdict, deque
from urllib.parse import urlparse
from PIL import Image
import shutil

import pandas as pd
import tqdm
import unidecode
from rtk import utils
from lib.rtk_adapt import Manifest
from lib.targz_index import archive_exists
from lib.uris import to_openapi
import anycase as cases


# Constants
DOWNLOAD_BATCH_SIZE = int(os.getenv("BATCH_SIZE", 5))   # Number of manifests to download in parallel
RETRY_LIMIT = 1                                         # How many times to retry a manifest
RETRY_NO_OPTIONS = 2
RETRY_DELAY = 10                                        # Seconds to wait before retrying a failed manifest
MAX_QUEUE_SIZE = 1240*60                                # Number of batch that we can keep without processing
SLEEP_TIME_BETWEEN_POOL_CHECK = 20
MANIFEST_DIRECTORY: str = "output"
DATA_DIR: str = os.getenv("DATA_DIR", "data-in-process")  # Where manuscript image directories are created


def check_sleep():
    while (len(glob.glob(f"{DATA_DIR}/*/*.jpg")) + len(glob.glob("./*/*.jpg"))
           - len(glob.glob(f"{DATA_DIR}/*/*.xml")) - len(glob.glob("./*/*.xml")) - 1000) >= MAX_QUEUE_SIZE:
        print("[WAIT] Waiting for some queue space")
        time.sleep(SLEEP_TIME_BETWEEN_POOL_CHECK)


def count_xml_in_targz(path: str) -> int:
    """
    Count the number of .xml files in each given .tar.gz archive.

    Args:
        path (Path): List of paths to .tar.gz files.

    Returns:
        Dict[Path, int]: Mapping from archive path to the number of .xml files found.
    """
    path = Path(path)

    count = 0
    try:
        with tarfile.open(path, "r:gz") as tar:
            for member in tar.getmembers():
                if member.isfile() and member.name.lower().endswith(".xml"):
                    count += 1
    except tarfile.TarError as e:
        print(f"Error reading {path}: {e}")
        count = 0

    return count


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
    # Legacy CSVs recorded directories at the repo root: keep one only if a
    # partial download already sits there, otherwise remap it under DATA_DIR.
    remapped = []
    for url, directory, filename in files:
        if os.path.dirname(directory) != DATA_DIR and not os.path.isdir(directory):
            directory = os.path.join(DATA_DIR, os.path.basename(directory))
        remapped.append((url, directory, filename))
    return remapped


def rename_image_download(image_detail: Tuple[str, str, str]) -> str:
    return os.path.join(image_detail[1], f"{image_detail[2]}.jpg")


def get_identifier(iiif_object: Dict[str, Any]) -> Optional[str]:
    identifiers = [None]
    for entry in iiif_object.get("metadata", []):
        labels = entry.get("label", None)
        if isinstance(labels, list):
            labels = {
                (lbl.get("@value", "").lower() if isinstance(lbl, dict) else lbl.lower())
                for lbl in entry.get("label")
            }
        else:
            labels = {labels.lower()}
        if "shelfmark" in labels:
            return entry["value"]
        elif "identifier" in labels:
            return entry["value"]

def kebab_with_fallback(string: str, fallback: Dict[str, Any] = None) -> str:
    if fallback:# and fallback.get("@id") in Constant_Shelfmark:
        _id = fallback.get("@id", fallback.get("id"))
        if _id in Constant_Shelfmark:
            string = Constant_Shelfmark[_id]

    return cases.to_kebab(unidecode.unidecode(string))


def data_dir_naming(string: str, fallback: Dict[str, Any] = None) -> str:
    """Directory recorded in the manifest CSV: DATA_DIR/<kebab-label>.

    The kebab basename must never change (resume logic matches archives by
    it); only the parent folder is added.
    """
    return os.path.join(DATA_DIR, kebab_with_fallback(string, fallback))


def single_download(tracker: ManifestTracker, manifests: List[str], max_download: int):
    downloaded = 0
    for manifest_uri in manifests:
        print(f"[Downloader] Downloading manifest {manifest_uri}")
        print(f"[TIME] {datetime.datetime.now()}")
        manifest_csv = rename_manifest_download(manifest_uri, cases.to_kebab)
        requires_download = not os.path.exists(manifest_csv)

        if requires_download:
            try:
                result = utils.download_iiif_manifest(manifest_uri, manifest_csv, naming_function=data_dir_naming)
                if not result:
                    print("\t[Details] Manifest undownloadable")
                    with open(f"shame-list-w{tracker.worker}.txt", "a") as f:
                        f.writelines([str(manifest_uri) + "\n"])
                    continue
            except Exception as E:
                print(f"\t[ERROR] {E}")
                print(E)
                continue

        images_details = parse_manifest(manifest_csv)

        current_manifest = Manifest.from_csv(manifest_uri, images_details, tracker=tracker)
        current_directory = tracker.manifest_to_directory[manifest_uri]
        current_directory_name = Path(current_directory).name

        override_targz_exist = False
        # Only check mark if the current manifest is not the local one
        if not current_manifest.check_mark():
            if os.path.exists(current_directory):
                try:
                    other_manifest = Manifest.from_json(str(Path(current_directory) / ".manifest.json"))
                    if len(other_manifest.images) < len(current_manifest.images):
                        override_targz_exist = True
                        print("\t[Info] Another manifest exist for the same manuscript but with less images,"
                              " we overwrite the folder")
                        shutil.rmtree(current_directory)
                        tracker.mark_done(other_manifest.manifest_id)
                    else:
                        tracker.mark_done(current_manifest.manifest_id)
                        print("\t[Info] Another manifest exist for the same manuscript but with more images, passing")
                        check_sleep()
                        continue
                except Exception as e:
                    print(f"\t[Error] No manifest in pre-existing directory {current_directory_name}")

            done = False
            if not override_targz_exist:
                # targz/index.csv first (URI-aware, collision-proof), then the
                # legacy name-glob for archives that predate the index
                indexed = archive_exists(manifest_uri, current_directory_name)
                targz_found = [indexed] if indexed else []
                for targz_path in targz_found:
                    xml_in_targs: int = count_xml_in_targz(targz_path)
                    if xml_in_targs >= len(current_manifest.images):
                        print(f"\t{targz_path} exists and is larger/same size as current manifest")
                        tracker.mark_done(current_manifest.manifest_id)
                        done = True
                        break
            if done:
                check_sleep()
                continue
        current_manifest.to_json()
        current_manifest.mark()

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
            if not current_manifest.check_mark():
                break
            result = utils.download_iiif_image(
                image[0],
                rename_image_download(image),
                options = {"max_height": 2500},
                retries = RETRY_LIMIT,
                retries_no_options = RETRY_NO_OPTIONS,
                time_between_retries = RETRY_DELAY
            )
            # In case time worked against you...
            if not current_manifest.check_mark():
                shutil.rmtree(rename_image_download(image), ignore_errors=True)
                break
            if not result:
                errors += 1
                current_manifest.add_errors(image[0])
                # At a maximum of 10% of errors for 50 images or more, we forget about this manuscript
                if len(images_details) > 30 and errors / (len(images_details)) > .1:
                    print("\t[ERROR] Too much errors (>10% of 4xx/5xx), moving to next manuscript.")
                    current_manifest.to_json()
                    aborted = True
                    continue
            downloaded += 1
            if max_download != -1 and downloaded >= max_download:
                print(f"Stopping the run here, reached maximum downloads {downloaded}")
                return
        if aborted:
            with open(f"shame-list-w{tracker.worker}.txt", "a") as f:
                f.writelines([str(manifest_uri)+"\n"])
        print(f"MANIFEST {manifest_uri} ==> ({len(current_manifest.found_images())}/{len(current_manifest.image_order)}")
        print(f"\t[Details] Directory is {current_manifest.directory}")
        print(f"Total download: {downloaded}")
        # Now check if pause !

        check_sleep()


def load_biblissima_data(csv_files: List[Tuple[str, str]]) -> Tuple[List[str], dict]:
    """
    Load multiple CSV files containing 'cote' and 'manifest_url' columns,
    build a shelfmark mapping, and return a DataFrame of unique manifest URLs
    and a dictionary mapping manifest URLs to shelfmarks.

    Parameters
    ----------
    csv_files : List[str]
        List of CSV file paths to read.
    delimiter : str, optional
        CSV delimiter (default: ';').

    Returns
    -------
    df : pd.DataFrame
        DataFrame with a single column 'manifest_url' containing unique values.
    Constant_Shelfmark : dict
        Mapping from manifest_url to cote.
    """
    all_dfs = []
    shelfmark_mappings = {}

    for file, delimiter in csv_files:
        df_shelfmark = pd.read_csv(file, delimiter=delimiter)[["cote", "manifest_url"]]
        mapping = {value: key for _, (key, value) in df_shelfmark.iterrows()}
        shelfmark_mappings.update(mapping)
        all_dfs.append(df_shelfmark[["manifest_url"]])

    # Combine all DataFrames and get unique manifest URLs
    df = pd.concat(all_dfs, ignore_index=True)
    df = pd.DataFrame({"manifest_url": df["manifest_url"].unique()})["manifest_url"].tolist()

    return df, shelfmark_mappings


def parse_file_sep(arg: str) -> Tuple[str, str]:
    """
    Parse a string in the format 'filename=separator'.
    Example: 'data.csv=;' -> ('data.csv', ';')
    """
    if '=' not in arg:
        raise argparse.ArgumentTypeError(
            "Each file must be specified as 'filename=separator', e.g. 'data.csv=;'"
        )
    filename, sep = arg.split('=', 1)
    if not filename or not sep:
        raise argparse.ArgumentTypeError(f"Invalid format for file argument: '{arg}'")
    return filename, sep

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Split work among workers.")
    parser.add_argument('--max', type=int, required=True, help='Total number of workers')
    parser.add_argument('--index', type=int, required=True, help='Index of this worker')
    parser.add_argument(
        "--files",
        nargs="+",
        type=parse_file_sep,
        metavar="FILE=SEP",
        help="List of CSV files with their separators, e.g. 'file.csv=;' or 'data.csv=$'",
        required=True,
    )
    parser.add_argument('--queue', type=int, default=MAX_QUEUE_SIZE, help='Total number of workers')
    parser.add_argument("--max_download", type=int, required=False, default=-1)
    args = parser.parse_args()

    MAX_QUEUE_SIZE = args.queue

    tracker = ManifestTracker(args.index)

    # Load manifests and filter out already completed ones
    df, Constant_Shelfmark = load_biblissima_data(args.files)
    for key in list(Constant_Shelfmark.keys()):
        Constant_Shelfmark[to_openapi(key)] = Constant_Shelfmark[key]

    Constant_Max_Download: int = args.max_download
    df = [
        to_openapi(uri) if to_openapi(uri) not in tracker.shamelist else uri # Keep good old URIs
        for uri in df
    ]
    df = [uri for uri in df if uri not in tracker.done and uri not in tracker.shamelist]
    df = alternate_by_domain(pd.Series(df)).tolist()
    if args.max > 1:
        assigned_items = split_work(df, args.max, args.index)
    else:
        assigned_items = df
    # print(df)
    # Launch producer and consumer
    print("[Main] Starting downloader")
    single_download(tracker, assigned_items, max_download=args.max_download)
