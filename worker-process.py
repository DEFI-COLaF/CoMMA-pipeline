import time
from pathlib import Path
from typing import Set, List, Dict, Optional
from rtk_adapt import YaltoCommand, Manifest, create_tar_gz_archives
from rtk import utils
from rtk.task import KrakenRecognizerCommand
import shutil
import random
import glob
import lxml.etree as et
import json


WATCH_DIR = "."
YOLO_BATCH_SIZE: int = 32
KRAKEN_BATCH_SIZE: int = 24
TARGET_COUNT: int = 64 # 4 * YOLO_BATCH_SIZE # Number of jpg to reach to run produce
TIME_BETWEEN_CHECK: int = 10

CACHED_DONE = {}
CACHED_PARSABLE = {}


def custom_layout_check(filepath) -> bool:
    if CACHED_PARSABLE.get(filepath):
        return True
    if utils.check_parsable(filepath):
        CACHED_PARSABLE[filepath] = True
        return True
    return False

def custom_ocr_check(filepath: str, ratio: int = 1) -> bool:
    if CACHED_DONE.get(filepath):
        return True
    if utils.check_content(filepath, ratio):
        CACHED_DONE[filepath] = True
        return True
    try:
        xml = et.parse(filepath)
        for element in xml.xpath("//a:processingCategory", namespaces={"a": "http://www.loc.gov/standards/alto/ns-v4#"}):
            if element.text.strip() == "contentGeneration":
                CACHED_DONE[filepath] = True
                return True
    except Exception:
        return False
    return False


def archive(directories_with_processed_files: List[Path], manifests: Dict[Path, Manifest]):
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
                    naming_func=lambda x: Path("targz") / (Path(manifest.directory).name + ".tar.gz")
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
def process_worker(batch: List[Path]):
    # Deduplicate paths and reconstruct manifest mapping
    images = [str(item) for item in batch]
    manifests: Dict[Path, Manifest] = {}
    for image in batch:
        if image.parent not in manifests:
            manifests[image.parent] = Manifest.from_json(str(image.parent / ".manifest.json"))

    print("[Processor] Segmenting with YALTAi")
    xmls = YaltoCommand(
        images,
        binary="yolalto",
        model_path="medieval-yolov11x.pt",
        batch_size=YOLO_BATCH_SIZE,
        check_content=custom_layout_check
    )
    xmls.process()

    files = [] + xmls.output_files
    random.shuffle(files)
    print("[Processor] OCR with Kraken")
    kraken = KrakenRecognizerCommand(
        xmls.output_files,
        binary="kraken",
        device="cpu",
        template="template.xml",
        model="catmus-medieval-1.6.0.mlmodel",
        raise_on_error=False,
        multiprocess=KRAKEN_BATCH_SIZE,
        check_content=True,
        custom_check_function=custom_ocr_check,
        other_options=" --no-subline-segmentation ",
        max_time_per_op=240  #
    )
    kraken.process()

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
        print("Checking...")
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
                    jpgs.add(Path(file).with_suffix(".jpg"))
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
    clean_up_archives()
    watch_directory()
