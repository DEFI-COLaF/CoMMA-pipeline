import glob
import csv
import os
import pandas as pd
import cases
import tqdm

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

df = pd.read_csv("extraction_biblissima_20250410.csv", delimiter=";")["manifest_url"]
df = [
    uri.replace("https://gallica.bnf.fr/iiif/ark:/12148/", "https://openapi.bnf.fr/iiif/presentation/v3/ark:/12148/")
    for uri in df
]
downloaded = []
for uri in tqdm.tqdm(df):
    name = f"output/{cases.to_kebab(uri)}.csv"
    if os.path.exists(name):
        with open(name) as f:
            reader = csv.reader(f)
            data = next(iter(reader))[1]+".tar.gz"
            exists = file_exists_recursive(data)
        if exists:
            #print(uri, data, exists)
            downloaded.append(uri)


with open("done.txt", "w") as f:
    f.write("\n".join(downloaded))
