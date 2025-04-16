import time
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from typing import Set, List, Dict, Optional
from rtk_adapt import YaltoCommand, Manifest, create_tar_gz_archives
from rtk import utils
from rtk.task import KrakenRecognizerCommand
import shutil
import glob


WATCH_DIR = "."
YOLO_BATCH_SIZE: int = 32
KRAKEN_BATCH_SIZE: int = 24
TARGET_COUNT: int = 64 # 4 * YOLO_BATCH_SIZE # Number of jpg to reach to run produce
TIME_BETWEEN_CHECK: int = 10

# Consumes batches of images from queue, performs segmentation, OCR, and GZIP when ready
def process_worker(batch: List[Path]):
    # Deduplicate paths and reconstruct manifest mapping
    images = [str(item) for item in batch]
    manifests: Dict[str, Manifest] = {}
    for image in batch:
        if image.parent not in manifests:
            manifests[image.parent] = Manifest.from_json(image.parent / ".manifest.json")

    print("[Processor] Segmenting with YALTAi")
    xmls = YaltoCommand(
        images,
        binary="yolalto",
        model_path="medieval-yolov11x.pt",
        batch_size=YOLO_BATCH_SIZE,
        check_content=utils.check_parsable
    )
    xmls.process()


    print("[Processor] OCR with Kraken")
    kraken = KrakenRecognizerCommand(
        xmls.output_files,
        binary="kraken",
        device="cpu",
        template="template.xml",
        model="catmus-medieval-1.6.0.mlmodel",
        raise_on_error=True,
        multiprocess=KRAKEN_BATCH_SIZE,
        check_content=True,
        other_options=" --no-subline-segmentation ",
        max_time_per_op=240  #
    )
    kraken.process()

    # Register all successful results in the tracker
    directories_with_processed_files = list(set([
        Path(xml_path).parent
        for xml_path in kraken.output_files
    ]))
    for directory in directories_with_processed_files:
        manifest: Optional[Manifest] = manifests.get(directory)
        if not manifest:
            print("Houston we got a problem")
        else:
            if manifest.is_complete():
                print(f"[Processor] Archiving complete manifest {manifest.directory}")
                paths = list([Path(directory) / Path(image).with_suffix(".xml") for image in manifest.image_order])
                print(paths)
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

def find_manifest_dirs(root_dir: str) -> List[str]:
    """
    Recursively finds all directories under root_dir that contain a '.manifest.json' file.

    Args:
        root_dir (str): Path to start the search from.

    Returns:
        List[str]: List of directory paths containing a '.manifest.json' file.
    """
    manifest_dirs = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if ".manifest.json" in filenames:
            manifest_dirs.append(dirpath)
    return manifest_dirs

# Watch for changes in the watch directory
def watch_directory():
    def get_unprocessed():
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
                if utils.check_parsable(file) == False or utils.check_content(file) == False:
                    print(f"{Path(file)} needs to be reworked")
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
    watch_directory()
