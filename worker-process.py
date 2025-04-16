import time
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from typing import Set, List, Dict
from rtk_adapt import YaltoCommand, Manifest, create_tar_gz_archives
from rtk import utils
from rtk.task import KrakenRecognizerCommand
import shutil


WATCH_DIR = "."
JPG_FILES_DETECTED: Set[Path] = set()
TARGET_COUNT: int = 10 # Number of jpg to reach to run produce
YOLO_BATCH_SIZE: int = 32
KRAKEN_BATCH_SIZE: int = 2

# Consumes batches of images from queue, performs segmentation, OCR, and GZIP when ready
async def process_worker(batch: List[Path]):
    # Deduplicate paths and reconstruct manifest mapping
    images = [str(item) for item in batch]
    manifests: Dict[str, Manifest] = {}
    manifest_map = {}
    for image in batch:
        if image.parent not in manifests:
            manifests[image.parent] = Manifest.from_json(image.parent / ".manifest.json")
        manifest_map[image.image_path] = manifests[image.parent].manifest_id

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
        device="cuda:0",
        template="template.xml",
        model="catmus-medieval-1.6.0.mlmodel",
        raise_on_error=True,
        multiprocess=KRAKEN_BATCH_SIZE,
        check_content=True
    )
    kraken.process()

    # Register all successful results in the tracker
    for xml_path in kraken.output_files:
        img_path = Path(xml_path).with_suffix(".jpg")
        manifest_id = manifest_map.get(img_path)
        if not manifest_id:
            print("Houston we got a problem")
        else:
            manifest = manifests[manifest_id]
            if manifest.is_complete():
                print(f"[Processor] Archiving complete manifest {manifest_id}")
                paths = list([Path(image).with_suffix(".xml") for image in manifest.image_order])
                if not paths:
                    continue

                # Get the staging folder from the parent of any file
                stage_folder = manifest.directory

                def _get_order(_path: Path) -> int:
                    return manifest.image_order.index(_path.with_suffix(".jpg").name)

                # Create ordering based on JPGs with same stem
                ordering = sorted(paths, key=_get_order)

                # Archive
                create_tar_gz_archives(
                    uri_to_files={manifest_id: paths},
                    ordering_dict={manifest_id: ordering},
                    naming_func=lambda x: Path("targz") / (stage_folder.name + ".tar.gz")
                )

                # Cleanup
                shutil.rmtree(stage_folder, ignore_errors=True)
                try:
                    with open("done.txt", "r") as f:
                        done = f.read().split()
                except FileNotFoundError:
                    done = []
                done.append(manifest.manifest_id)
                with open("done.txt", "w") as f:
                    f.write("\n".join(done))


# Watchdog event handler to trigger the producer when a new .jpg file is added
class JPGFileHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return
        global JPG_FILES_DETECTED

        file_path = Path(event.src_path)
        if file_path.suffix == ".jpg":
            print(f"Detected .jpg file: {file_path}")
            # Add to the set of .jpg files without .xml
            if not file_path.with_suffix(".xml").exists():
                JPG_FILES_DETECTED.add(file_path)
            # Trigger the producer if there are 100 .jpg files without .xml files
            if len(JPG_FILES_DETECTED) >= TARGET_COUNT:
                print(f"Found {TARGET_COUNT} images to process, processing")
                process_worker(list(JPG_FILES_DETECTED))
                JPG_FILES_DETECTED = set()

# Watch for changes in the watch directory
def watch_directory():
    event_handler = JPGFileHandler()
    observer = Observer()
    observer.schedule(event_handler, WATCH_DIR, recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
    if JPG_FILES_DETECTED:
        process_worker(list(JPG_FILES_DETECTED))



# Run the producer (watcher)
if __name__ == "__main__":
    watch_directory()