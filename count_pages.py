from convert import get_manifest_and_xmls
import glob
import os.path
from typing import List, Union, Optional
import tqdm
import lxml.etree as et
import dataclasses
import tarfile
from typing import List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed


def process_tar(file: str):
    try:
        _, xmls = get_manifest_and_xmls(file)
        return len(xmls)
    except Exception as e:
        return file, f"error: {e}"


if __name__ == "__main__":
    files = glob.glob("/home/tclerice/targez/targz/*.tar.gz")

    results = []
    with ProcessPoolExecutor(max_workers=12) as executor:
        futures = {executor.submit(process_tar, file): file for file in files}
        with tqdm.tqdm(total=len(futures)) as bar:
            for future in as_completed(futures):
                cnt = future.result()
                results.append(cnt)
                bar.update(1)
                bar.set_description(f"Pages {sum(results)}")
