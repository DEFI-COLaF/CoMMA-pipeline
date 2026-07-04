import glob
import os.path
from typing import List, Dict, Any
import tqdm
import json
import tarfile
import lxml.etree as et
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd

from lib.tar_utils import find_tar_gz_files_recursive, read_archive_metadata
from lib.uris import uri_variants

_DF = pd.concat([
    pd.read_csv("biblissima_bodleian.csv", delimiter=";"),
    pd.read_csv("extraction_biblissima_20250410.csv", delimiter=";")
])

XSL = et.XSLT(et.parse("assets/03-to-json.xsl"))

def flint(string: str) -> int:
    return int(float(string))


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


def to_json(file_order: List[str], tar_gz: str, output, metadata, image_urls: Dict[str, str]):
    out = []
    for page in file_order:
        page_dict = simplify_and_lines(tar_gz, page)
        page_dict["image_url"] = image_urls.get(os.path.splitext(os.path.basename(page))[0], "")
        out.append(page_dict)
    out = {
        **metadata, "files": out
    }
    with open(f"{output}/{os.path.basename(tar_gz)}.json", "w") as f:
        json.dump(out, f)
    return f"{output}/{os.path.basename(tar_gz)}.json"


def process_tar(file: str):
    if len(glob.glob(f"json/*/{os.path.basename(file)}.json")):
        return file, "success"
    try:
        batch = f"batch-{len(glob.glob('json/**/*.json')) // 1000:03d}"
        os.makedirs(f"json/{batch}", exist_ok=True)

        provenance = read_archive_metadata(file)
        manifest_uri = provenance["manifest_id"]
        files = provenance["files"]

        variants = set(uri_variants(manifest_uri)) | set(
            uri_variants(provenance.get("source_manifest_id", manifest_uri))
        )
        rows = _DF[_DF.manifest_url.isin(variants)].fillna(value="").to_dict(orient="records")
        metadata = rows[0] if rows else {"manifest_url": manifest_uri}
        metadata["manifest_id"] = manifest_uri
        metadata["source_manifest_id"] = provenance.get("source_manifest_id", manifest_uri)

        image_urls = {
            img["filename"]: img.get("image_url", "")
            for img in provenance.get("images", [])
        }
        to_json(files, file, f"json/{batch}", metadata, image_urls)
        return file, "success"
    except Exception as e:
        return file, f"error: {e}"


if __name__ == "__main__":
    files = find_tar_gz_files_recursive("targz")
    l = len(files)
    results = []
    with ProcessPoolExecutor(max_workers=int(os.getenv("CONVERT_WORKERS", 12))) as executor:
        futures = {executor.submit(process_tar, file): file for file in files}
        with tqdm.tqdm(total=len(futures)) as bar:
            for future in as_completed(futures):
                file, status = future.result()
                results.append((file, status))
                bar.update(1)

    # Optional: Print or log errors
    for file, status in results:
        if status != "success":
            print(f"{file} failed with {status}")
