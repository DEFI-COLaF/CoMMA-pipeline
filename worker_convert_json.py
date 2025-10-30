import glob
import os.path
from typing import List, Union, Optional, Dict, Any
import tqdm
import re
import json
import lxml.etree as et
import dataclasses
import tarfile
from typing import List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd

_DF = pd.concat([
    pd.read_csv("biblissima_bodleian.csv", delimiter=";"),
    pd.read_csv("extraction_biblissima_20250410.csv", delimiter=";")
])
_RUBRICATED = {"rend": "rubricated"}
_INCIPIT = {"rend": "rubricated", "type": "incipit"}

XSL = et.XSLT(et.parse("to-json.xsl"))

def flint(string: str) -> int:
    return int(float(string))

def get_pages_in_sequence(directory: str):
    mets = et.parse(f"{directory}/METS.xml")
    pages = mets.xpath("/m:mets/m:fileSec/m:fileGrp/m:file/m:FLocat/@x:href", namespaces={
        "m": "http://www.loc.gov/METS/",
        "x": "http://www.w3.org/1999/xlink"
    })
    return list(map(lambda x: f"{directory}/{x}", pages))


def simplify_and_lines(tar_gz, alto_path: str) -> Dict[str, Any]:
    try:
        with tarfile.open(tar_gz, 'r:gz') as tar:
            member = tar.getmember(alto_path.replace(".jpg", ".xml"))
            with tar.extractfile(member) as f:
                xml = et.parse(f)
    except Exception as E:
        print(tar_gz, alto_path, E)
        return {"alto": alto_path, "zones": [], "wh": (0, 0)}
    xml = XSL(xml)
    zones: List[Dict] = []
    for zone in xml.xpath("//region"):
        zones.append({
            "type": zone.attrib["type"],
            "lines": [],
            "wh": (flint(zone.attrib["width"] or 0), flint(zone.attrib["height"] or 0)),
            "xy": (flint(zone.attrib["x"] or 0), flint(zone.attrib["y"] or 0)),
        })
        for line in zone.xpath("./line"):
            zones[-1]["lines"].append(
                {
                    "type": str(line.attrib["type"]),
                    "content": str(line.text or "").strip(),
                    "wh": (flint(line.attrib["width"] or 0), flint(line.attrib["height"] or 0)),
                    "xy": (flint(line.attrib["x"] or 0), flint(line.attrib["y"] or 0)),
                }
            )
    return {
        "zones": zones,
        "alto": alto_path,
        "wh": (flint(xml.xpath("/doc")[0].attrib["width"] or 0), flint(xml.xpath("/doc")[0].attrib["height"] or 0))
    }


def to_tei(file_order: List[str], tar_gz: str, output, metadata):
    out = []
    for page in file_order:
        out.append(simplify_and_lines(tar_gz, page))
    out = {
        **metadata, "files": out
    }
    with open(f"{output}/{os.path.basename(tar_gz)}.json", "w") as f:
        json.dump(out, f)
    return f"{output}/{os.path.basename(tar_gz)}.json"


def get_manifest_and_xmls(tar_gz_path: str) -> Tuple[str | None, List[str]]:
    """
    Extracts the path to 'manifest.txt' and a list of all other .xml files in a .tar.gz archive.

    Returns:
        A tuple: (manifest_path, list_of_xml_paths)
    """
    manifest = None
    xml_files = []

    with tarfile.open(tar_gz_path, 'r:gz') as tar:
        for member in tar.getmembers():
            if member.isfile():
                if member.name.endswith('manifest.txt'):
                    manifest = member.name
                elif member.name.endswith('.xml'):
                    xml_files.append(member.name)

    return manifest, xml_files


def read_file_from_tar(tar_gz_path: str, file_name: str) -> str:
    """
    Reads the content of 'manifest.txt' from a .tar.gz archive.

    Args:
        tar_gz_path: Path to the .tar.gz archive.
        manifest_name: The path to the manifest file inside the archive.

    Returns:
        The content of the manifest file as a string.
    """
    with tarfile.open(tar_gz_path, 'r:gz') as tar:
        member = tar.getmember(file_name)
        with tar.extractfile(member) as f:
            return f.read().decode('utf-8')


def find_tar_gz_files_recursive(dir_path: str) -> List[str]:
    """
    Explicitly recursive version to find all .tar.gz files in a directory tree.

    Args:
        dir_path (str): Root directory path to start the search.

    Returns:
        List[str]: A list of paths to found .tar.gz files.
    """
    matches = []
    for entry in os.scandir(dir_path):
        if entry.is_dir(follow_symlinks=False):
            matches.extend(find_tar_gz_files_recursive(entry.path))
        elif entry.is_file() and entry.name.endswith(".tar.gz"):
            matches.append(entry.path)
    return matches



def process_tar(file: str):
    if len(glob.glob(f"json/*/{os.path.basename(file)}.json")):
        return file, "success"
    try:
        batch = f"batch-{len(glob.glob('json/**/*.json')) // 1000:03d}"
        os.makedirs(f"json/{batch}", exist_ok=True)

        _, xmls = get_manifest_and_xmls(file)
        manifest = read_file_from_tar(file, "manifest.txt")
        manifest_uri, *files = manifest.split("\n")
        metadata = _DF[
          (_DF.manifest_url == manifest_uri) |
          (_DF.manifest_url == manifest_uri.replace("https://openapi.bnf.fr/iiif/presentation/v3/ark:/12148/", "https://gallica.bnf.fr/iiif/ark:/12148/"))
        ].fillna(value="").to_dict(orient="records")[0]
        files = [os.path.basename(f) for f in files]
        to_tei(files, file, f"json/{batch}", metadata)
        return file, "success"
    except Exception as e:
        return file, f"error: {e}"


if __name__ == "__main__":
    files = find_tar_gz_files_recursive("targz")
    l = len(files)
    results = []
    with ProcessPoolExecutor(max_workers=12) as executor:
        futures = {executor.submit(process_tar, file): file for file in files}
        with tqdm.tqdm(total=len(futures)) as bar:
            for future in as_completed(futures):
                file, status = future.result()
                results.append((file, status))
                bar.update(1)

    # Optional: Print or log errors
    for file, status in results:
        if status != "success":
            print(f"{file} failed with {status}")[tclerice@cleps download-bnf]