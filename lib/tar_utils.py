"""Helpers to create and read the per-manuscript tar.gz archives.

These are deliberately free of heavy dependencies (rtk, kraken) so they can be
imported and tested anywhere.

Archives come in two flavours:
- legacy: only a ``manifest.txt`` member (first line = manifest URI, following
  lines = file paths — possibly full local paths, so consumers must basename
  them);
- v2 (``schema_version: 2``): additionally a ``manifest.json`` member with the
  full provenance (source URI, per-image source URLs, page order, errors).
"""
import csv
import datetime
import io
import json
import os
import tarfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from lib.uris import to_gallica


def get_manifest_and_xmls(tar_gz_path: str) -> Tuple[Optional[str], List[str]]:
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
    Reads the content of a member file from a .tar.gz archive.

    Args:
        tar_gz_path: Path to the .tar.gz archive.
        file_name: The path of the file inside the archive.

    Returns:
        The content of the file as a string.
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


def read_archive_metadata(tar_gz_path: str) -> Dict[str, Any]:
    """Returns the provenance record of an archive.

    Prefers the v2 ``manifest.json`` member; falls back to ``manifest.txt``
    for legacy archives. In both cases the returned dict contains at least
    ``schema_version``, ``manifest_id`` and ``files`` (the ordered .xml member
    basenames).
    """
    try:
        meta = json.loads(read_file_from_tar(tar_gz_path, "manifest.json"))
        meta.setdefault(
            "files", [f"{stem}.xml" for stem in meta.get("image_order", [])]
        )
        return meta
    except KeyError:
        pass
    manifest = read_file_from_tar(tar_gz_path, "manifest.txt")
    manifest_uri, *files = manifest.split("\n")
    return {
        "schema_version": 1,
        "manifest_id": manifest_uri,
        "files": [os.path.basename(f) for f in files if f.strip()],
    }


def build_archive_metadata(
    manifest_id: str,
    directory: str,
    image_order: List[str],
    total_images: int,
    errors: List[str],
    csv_path: Optional[str] = None,
    image_uris: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Builds the v2 ``manifest.json`` provenance record for an archive.

    Per-image source URLs come from ``image_uris`` (aligned with
    ``image_order``, as stored by ``Manifest.uris``) when available, else from
    ``csv_path`` (manifest download CSV rows: image_url, directory, filename);
    when both are missing the record is still produced, just without URLs.
    """
    url_by_filename: Dict[str, str] = {}
    if image_uris:
        url_by_filename = dict(zip(image_order, image_uris))
    elif csv_path:
        try:
            with open(csv_path) as f:
                for row in csv.reader(f):
                    if len(row) >= 3:
                        url_by_filename[row[2]] = row[0]
        except FileNotFoundError:
            print(f"[WARNING] {csv_path} not found; archiving without per-image source URLs")
    return {
        "schema_version": 2,
        "manifest_id": manifest_id,
        "source_manifest_id": to_gallica(manifest_id),
        "directory": directory,
        "image_order": image_order,
        "total_images": total_images,
        "errors": errors,
        "images": [
            {"filename": name, "image_url": url_by_filename.get(name, "")}
            for name in image_order
        ],
        "archived_at": datetime.datetime.now().isoformat(),
    }


def create_tar_gz_archives(
    uri_to_files: Dict[str, List[Path]],
    naming_func: Callable[[str], Path],
    ordering_dict: Dict[str, List[Path]],
    metadata: Optional[Dict[str, Dict[str, Any]]] = None,
) -> None:
    """
    Creates a .tar.gz archive for each URI with a manifest and files.

    Args:
        uri_to_files: A dictionary mapping URIs to lists of local file Paths.
        naming_func: A function that takes a URI and returns the target tar.gz file path.
        ordering_dict: A dictionary mapping URIs to an ordered list of file Paths.
        metadata: Optional URI → provenance dict, written as a `manifest.json` member
            (see build_archive_metadata).
    """
    def add_member(tar: tarfile.TarFile, name: str, data: bytes) -> None:
        info = tarfile.TarInfo(name=name)
        info.size = len(data)
        tar.addfile(info, fileobj=io.BytesIO(data))

    for uri, files in uri_to_files.items():
        archive_path = naming_func(uri)
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        ordered_files = ordering_dict.get(uri, files)
        # Basenames match the arcnames used below (legacy archives stored local paths)
        manifest_content = [uri] + [Path(path).name for path in ordered_files]

        with tarfile.open(archive_path, "w:gz") as tar:
            add_member(tar, "manifest.txt", "\n".join(manifest_content).encode("utf-8"))
            if metadata and uri in metadata:
                add_member(tar, "manifest.json", json.dumps(metadata[uri]).encode("utf-8"))

            # Add each file to the archive
            for file_path in files:
                if Path(file_path).is_file():
                    tar.add(file_path, arcname=Path(file_path).name)
