# Allow running from anywhere: put the repo root (parent of scripts/) on sys.path
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))


import argparse
from collections import defaultdict

import tqdm
import os
import anycase as cases
from typing import List, Dict, Tuple
from pathlib import Path
from worker_single_download import (load_biblissima_data, rename_manifest_download, parse_manifest,
                                    count_xml_in_targz, Manifest, parse_file_sep)
from lib.uris import to_openapi as uri_renamer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse source CSV to detect down work.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Perform a dry run without modifying files or extracting archives."
    )
    parser.add_argument(
        "--files",
        nargs="+",
        type=parse_file_sep,
        metavar="FILE=SEP",
        help="List of CSV files with their separators, e.g. 'file.csv=;' or 'data.csv=$'",
        required=True,
    )
    args = parser.parse_args()
    manifest_list, Constant_Shelfmark = load_biblissima_data(args.files)

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

    directory_matches: Dict[str, List[Tuple[str, Manifest]]] = defaultdict(list)

    for (manifest_csv, manifest_uri) in tqdm.tqdm(manifest_exist):
        current_manifest = parse_manifest(manifest_csv)
        manifest_obj = Manifest.from_csv(manifest_uri, current_manifest)
        if os.path.exists(manifest_obj.directory):
            directory_matches[manifest_obj.directory].append((manifest_csv, manifest_obj))
            continue

    for directory, manifests in directory_matches.items():
        current_manifest = Manifest.from_json(f"{directory}/.manifest.json")
        # ToDo complete here
        # List images
        images = set([str(file.stem) for file in Path(directory).glob("*.jpg")])
        if not len(images) or len(manifests) == 1:
            current_manifest.mark()
            continue

        print(f"{directory}")
        percent = {
            "current": len(images.intersection(set(current_manifest.image_order))) / len(images)
        }

        for csv_file, other_manifest in manifests:
            percent[csv_file] = len(images.intersection(set(other_manifest.image_order))) / len(images)

        # The simplest might be to take simply the biggest manifest, which is what we do in worker_single_download
        (largest_manifest_csv, largest_manifest), *needs_cleaning = sorted(
            manifests, key=lambda x: len(x[1].images), reverse=True
        )

        if largest_manifest.images == current_manifest.images:
            print("\t[INFO] ✅ Current manifest is the largest")
            if sorted(largest_manifest.image_order) == sorted(list(images)):
                print("\t 🔚 This manuscript is good, no need for clean up")
                continue
        else:
            print("\t[INFO] 📦 Requires moving to best manifest")


        clean_up_images = images.difference(set(largest_manifest.image_order))
        print(f"\t[INFO] 🗑️ Images that needs to be clean up: {len(clean_up_images)}")

        # Check that there are no overwriting issues...
        common_images = images.intersection(set(largest_manifest.image_order).intersection([
            image
            for _, loc_manifest in needs_cleaning
            for image in loc_manifest.image_order
        ]))
        print(f"\t[INFO] ⛓️ Common images: {len(common_images)}")

        for file in common_images.union(clean_up_images):
            os.remove(f"{directory}/{file}.jpg")
            if os.path.exists(f"{directory}/{file}.xml"):
                os.remove(f"{directory}/{file}.xml")

        print(f"\t[INFO] 🧹 Removed {len(common_images.union(clean_up_images))}")
        largest_manifest.to_json()
        images = set([str(file.stem) for file in Path(directory).glob("*.jpg")])
        print(f"\t🎯 Coverage: {len(images.intersection(set(largest_manifest.image_order))) / len(images)}")
        print("\t🤝 Done")
        largest_manifest.mark()