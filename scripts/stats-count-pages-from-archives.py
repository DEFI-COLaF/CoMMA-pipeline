# Allow running from anywhere: put the repo root (parent of scripts/) on sys.path
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))


from lib.tar_utils import get_manifest_and_xmls
import glob
import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


def process_tar(file: str):
    try:
        _, xmls = get_manifest_and_xmls(file)
        return len(xmls)
    except Exception as e:
        return file, f"error: {e}"


if __name__ == "__main__":
    files = glob.glob("targz/**/*.tar.gz", recursive=True)

    results = []
    with ProcessPoolExecutor(max_workers=12) as executor:
        futures = {executor.submit(process_tar, file): file for file in files}
        with tqdm.tqdm(total=len(futures)) as bar:
            for future in as_completed(futures):
                cnt = future.result()
                results.append(cnt)
                bar.update(1)
                bar.set_description(f"Pages {sum(results)}")
