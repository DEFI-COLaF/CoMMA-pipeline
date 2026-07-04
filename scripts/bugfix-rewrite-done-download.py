# Allow running from anywhere: put the repo root (parent of scripts/) on sys.path
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))


import os
import glob
import argparse

from typing import Tuple, List, Dict
from pathlib import Path
from collections import defaultdict

import anycase as cases
import tqdm


def list_tar_gz_names(directory: str) -> Dict[str, List[str]]:
    targz = defaultdict(list)
    for value in Path(directory).rglob("*.tar.gz"):
        targz[str(value.name)].append(str(value))
    return targz


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
    from worker_single_download import (load_biblissima_data, rename_manifest_download, parse_manifest,
                                        count_xml_in_targz)
    from lib.uris import to_openapi as uri_renamer
    parser = argparse.ArgumentParser(description="Parse source CSV to detect down work.")
    parser.add_argument(
        "--files",
        nargs="+",
        type=parse_file_sep,
        metavar="FILE=SEP",
        help="List of CSV files with their separators, e.g. 'file.csv=;' or 'data.csv=$'",
        required=True,
    )
    parser.add_argument("--tar-gz-directory", type=str, help="Directory where targz are", default="targz")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Perform a dry run without modifying files or extracting archives."
    )
    args = parser.parse_args()
    manifest_list, Constant_Shelfmark = load_biblissima_data(args.files)
    targz = list_tar_gz_names(args.tar_gz_directory)
    # First we find what we downloaded as far as manifest go
    manifest_exist = []
    for manifest_uri in tqdm.tqdm(manifest_list):
        manifest_csv = rename_manifest_download(manifest_uri, cases.to_kebab)
        if os.path.exists(manifest_csv):
            # This in fact bypass the need for Constant_Shelfmark because we get the directory
            manifest_exist.append((manifest_csv, manifest_uri))
            continue

        manifest_uri = uri_renamer(manifest_uri)
        if os.path.exists(manifest_csv):
            # This in fact bypass the need for Constant_Shelfmark because we get the directory
            manifest_exist.append((manifest_csv, manifest_uri))
            continue

    print(f"{len(manifest_exist)/len(manifest_list)*100:.2f}% of manifests downloaded ({len(manifest_exist)})")
    downloaded = []
    # We are also gonna apply a remanifesting here, to ensure that the right manifest is in the right folder...
    for (manifest_csv, manifest_uri) in tqdm.tqdm(manifest_exist):
        current_manifest = parse_manifest(manifest_csv)
        _, directory, _ = current_manifest[0]
        current_manifest = len(current_manifest)
        tar = f"{directory}.tar.gz"
        if tar in targz:
            for targz_path in targz[tar]:
                xml_in_targs: int = count_xml_in_targz(targz_path)
                if xml_in_targs >= current_manifest:
                    downloaded.append(manifest_uri)
                    break

    print(f"{len(downloaded)/len(manifest_exist)*100:.2f}% of manifests downloaded have been fully processed ({len(downloaded)})")

    if not args.dry_run:
        with open("done.txt", "w") as f:
            f.write("\n".join(downloaded))
