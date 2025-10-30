import glob
import csv
import os
import pandas as pd
import anycase as cases
import tqdm
from typing import Tuple
import argparse

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


def file_exists_recursive(filename: str, root_dir: str = ".") -> bool:
    """
    Check if a file with the given name exists anywhere under the specified directory.

    Parameters:
        filename (str): The name of the file to search for (not a path).
        root_dir (str): The directory to start searching from. Defaults to current directory.

    Returns:
        bool: True if the file is found, False otherwise.
    """
    for dirpath, _, filenames in os.walk(root_dir):
        if filename in filenames:
            return True
    return False
#
# df = pd.read_csv("extraction_biblissima_20250410.csv", delimiter=";")["manifest_url"]
# df = [
#     uri.replace("https://gallica.bnf.fr/iiif/ark:/12148/", "https://openapi.bnf.fr/iiif/presentation/v3/ark:/12148/")
#     for uri in df
# ]
# downloaded = []
# for uri in tqdm.tqdm(df):
#     name = f"output/{cases.to_kebab(uri)}.csv"
#     if os.path.exists(name):
#         with open(name) as f:
#             reader = csv.reader(f)
#             data = next(iter(reader))[1]+".tar.gz"
#             exists = file_exists_recursive(data)
#         if exists:
#             #print(uri, data, exists)
#             downloaded.append(uri)
#
#
# with open("done.txt", "w") as f:
#     f.write("\n".join(downloaded))

if __name__ == "__main__":
    from worker_single_download import load_biblissima_data, rename_manifest_download, parse_manifest
    parser = argparse.ArgumentParser(description="Parse source CSV to detect down work.")
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
    for manifest_uri in manifest_list:
        manifest_csv = rename_manifest_download(manifest_uri, cases.to_kebab)
        if os.path.exists(manifest_csv):
            manifest_exist.append((manifest_csv, manifest_uri))

    print(f"{len(manifest_exist)/len(manifest_list)*100:.2f}% of manifests downloaded")

    for (manifest_csv, manifest_uri) in manifest_exist:
        images = parse_manifest(manifest_csv)
        print(images[0])