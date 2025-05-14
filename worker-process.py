import os.path
import time
from pathlib import Path
from typing import Set, List, Dict, Optional, Tuple
from PIL import Image
from rtk import utils
from rtk.task import KrakenAltoCleanUpCommand
import shutil
import random
import re
import glob
import lxml.etree as et
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from rtk_adapt import YaltoCommand, Manifest, create_tar_gz_archives
from direct_kraken import KrakenDirectTask


def naming_func(name: Path, base_dir: Path = Path("./targz")) -> Path:
    all_tars = set(Path().glob("targz-batch-*/*.tar.gz"))
    name = Path(name).name  # ensure only filename part is used

    batch_num = len(all_tars) // 1000
    batch_folder = f"batch_{batch_num:03d}"

    return base_dir / batch_folder / name


def filter_valid_jpgs(images: Set[str], max_workers: int = 8) -> Set[str]:
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(check_image_file, images)
    return {j for j, is_valid in zip(images, results) if is_valid}


def print_current_time():
    """Prints the current time in HH:MM:SS format."""
    now = datetime.now()
    print(now.strftime("%H:%M:%S"))

WATCH_DIR = "."
YOLO_BATCH_SIZE: int = 1
KRAKEN_BATCH_SIZE: int = int(os.getenv("KRAKEN_BATCH_SIZE", 40))
TARGET_COUNT: int = 64 # 4 * YOLO_BATCH_SIZE # Number of jpg to reach to run produce
TIME_BETWEEN_CHECK: int = 10
CACHED_DONE = {}
CACHED_PARSABLE = {}
REVERSE: bool = bool(int(os.getenv("REVERSE", 0))) # REVERSE=1 start from the end

if REVERSE:
    print("GOING REVERSE")

def check_image_file(filepath: Path) -> bool:
    try:
        Image.open(filepath).tobytes()
    except Exception:
        return False
    return True


def custom_layout_check(filepath) -> bool:
    if utils.check_parsable(filepath):
        return True
    return False

def custom_ocr_check(filepath: str, ratio: int = 1) -> bool:
    if utils.check_content(filepath, ratio):
        return True
    try:
        xml = et.parse(filepath)
        for element in xml.xpath("//a:processingCategory", namespaces={"a": "http://www.loc.gov/standards/alto/ns-v4#"}):
            if element.text.strip() == "contentGeneration":
                return True
    except Exception:
        return False
    return False


def custom_ocr_check_with_inp(filepath: str, ratio: int = 1) -> Tuple[str, bool]:
    return filepath, custom_ocr_check(filepath, ratio)


def archive(directories_with_processed_files: List[Path], manifests: Dict[Path, Manifest]):
    print_current_time()
    print(f"Checking {len(directories_with_processed_files)} for archival process")
    for directory in directories_with_processed_files:
        manifest: Optional[Manifest] = manifests.get(directory)
        if not manifest:
            print(f"Houston we got a problem for {directory} (No Manifest)")
        else:
            if manifest.is_complete(checking_function=custom_ocr_check, log=True):
                print(f"[Processor] Archiving complete manifest {manifest.directory}")
                paths = list([Path(directory) / Path(image).with_suffix(".xml") for image in manifest.image_order])

                if not paths:
                    continue

                def _get_order(_path: Path) -> int:
                    return manifest.image_order.index(_path.with_suffix("").name)

                # Create ordering based on JPGs with same stem
                ordering = sorted(paths, key=_get_order)

                # Archive
                create_tar_gz_archives(
                    uri_to_files={manifest.manifest_id: paths},
                    ordering_dict={manifest.manifest_id: ordering},
                    naming_func=lambda x: naming_func(Path(manifest.directory).name + ".tar.gz")
                )

                # Cleanup
                shutil.rmtree(manifest.directory, ignore_errors=True)
                try:
                    with open("done.txt", "r") as f:
                        done = f.read().split()
                except FileNotFoundError:
                    done = []
                done.append(manifest.manifest_id)
                with open("done.txt", "w") as f:
                    f.write("\n".join(done))


def clean_up_archives():
    print("Running clean-up")
    manifests: Dict[Path, Manifest] = {}
    directories: List[Path] = list(map(Path, find_manifest_dirs(".")))
    for directory in directories:
        manifests[directory] = Manifest.from_json(str(directory / ".manifest.json"))
    archive(directories, manifests)


# Consumes batches of images from queue, performs segmentation, OCR, and GZIP when ready
def process_worker(batches: List[Path]):
    print("Processing")
    print_current_time()
    # Deduplicate paths and reconstruct manifest mapping

    def split_into_batches(lst: List[Path], n: int) -> List[List[Path]]:
        """Split a list into batches of size n."""
        return [lst[i:i + n] for i in range(0, len(lst), n)]

    batches = sorted(batches)
    print(f"{len(batches)} files to process")
    if REVERSE:
        batches = batches[::-1]

    for batch in split_into_batches(batches, KRAKEN_BATCH_SIZE*10):
        images = [str(item) for item in batch]
        images = [file for file in images if os.path.exists(file)]
        if not len(images):
            continue
        manifests: Dict[Path, Manifest] = {}
        kept = []
        for image in batch:
            if image.parent not in manifests:
                if os.path.exists(str(image.parent / ".manifest.json")):
                    manifests[image.parent] = Manifest.from_json(str(image.parent / ".manifest.json"))
                else:
                    print(f"{image.parent} has no manifests...")
                    continue
            kept.append(image)
        images = [str(img) for img in kept]
        random.shuffle(images)

        print("[Processor] Segmenting with YALTAi")
        xmls = YaltoCommand(
            images,
            binary="yolalto",
            model_path="medieval-yolov11x.pt",
            batch_size=YOLO_BATCH_SIZE,
            check_content=custom_layout_check
        )
        try:
            xmls.process()
        except Exception as E:
            print(E)


        files = [] + xmls.output_files
        random.shuffle(files)
        files = [f for f in files if custom_layout_check(f)]
        print(f"{len(files)}/{len(xmls.output_files)} have correct XML. Filtered the wrong ones")

        cleanup = KrakenAltoCleanUpCommand(files)
        cleanup.process()

        files = [f for f in list(cleanup.output_files) if custom_layout_check(f)]
        print(f"{len(files)}/{len(xmls.output_files)} have correct XML. Filtered the wrong ones")

        files = [f for f in files if os.path.exists(f.replace(".xml", ".jpg"))]
        print(f"{len(files)}/{len(xmls.output_files)} have their JPGs")

        print("[Processor] OCR with Kraken")
        kraken = KrakenDirectTask(
            files,
            device="cpu",
            template="template.xml",
            model="catmus-medieval-1.6.0.mlmodel",
            workers=KRAKEN_BATCH_SIZE,
            check_content=custom_ocr_check_with_inp,
            subline_segmentation=False,
            max_time_per_op=600  #
        )
        kraken.process()

        cleanup = KrakenAltoCleanUpCommand(kraken.output_files)
        cleanup.process()


        # Register all successful results in the tracker
        directories_with_processed_files = list(set([
            Path(xml_path).parent
            for xml_path in kraken.output_files
        ]))
        archive(directories_with_processed_files, manifests)


def find_manifest_dirs(root_dir: str) -> List[str]:
    """
    Recursively finds all directories under root_dir that contain a '.manifest.json' file.

    Args:
        root_dir (str): Path to start the search from.

    Returns:
        List[str]: List of directory paths containing a '.manifest.json' file.
    """
    manifest_dirs = []
    for manifest_file in map(Path, glob.glob(f"{root_dir}/*/.manifest.json")):
        try:
            with open(str(manifest_file)) as f:
                json.load(f)
            # print(f"Found {manifest_file.parent}")
            manifest_dirs.append(str(manifest_file.parent))
        except Exception:
            print(f"Unparsable {manifest_file}")

    return manifest_dirs

# Watch for changes in the watch directory
def watch_directory():

    def get_unprocessed():
        print("Checking files for processing...")
        print_current_time()
        jpgs = set()
        for directory in find_manifest_dirs("."):
            jpgs = jpgs.union(
                set([
                    file
                    for file in map(Path, glob.glob(f"{directory}/*.jpg"))
                    if not file.with_suffix(".xml").exists()
                ])
            )
            # Check all xml without jpgs
            for file in sorted(glob.glob(f"./{directory}/*.xml")):
                # If OCR was not done, it means it needs to be done :)
                if not custom_ocr_check(file):
                    if os.path.exists(Path(file).with_suffix(".jpg")):
                        jpgs.add(Path(file).with_suffix(".jpg"))

        jpgs = filter_valid_jpgs(jpgs, max_workers=KRAKEN_BATCH_SIZE)

        if len(jpgs) >= TARGET_COUNT:
            process_worker(list(jpgs))
            return True
        elif Path(".totally_done").exists():
            process_worker(list(jpgs))
            print("Finished")
            return False

        print("Nothing found, waiting")
        return True

    while get_unprocessed():
        time.sleep(TIME_BETWEEN_CHECK)

# Run the producer (watcher)
if __name__ == "__main__":
    print("Hello, let's go")
    print_current_time()
    clean_up_archives()
    watch_directory()
