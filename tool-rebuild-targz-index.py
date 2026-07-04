"""Backfills targz/index.csv from the archives already on disk.

Walks ``targz/**/*.tar.gz``, reads each archive's provenance (v2
``manifest.json`` member, or legacy ``manifest.txt``) and appends an index row
for every archive path not yet indexed. Idempotent: a second run adds nothing.

Run once in production after deploying the index code, then whenever archives
were produced by an older pipeline version.
"""
import os
import sys

import tqdm

from lib.tar_utils import find_tar_gz_files_recursive, read_archive_metadata
from lib.targz_index import DEFAULT_INDEX_PATH, append_index, load_index


def main(targz_root: str = "targz", index_path: str = DEFAULT_INDEX_PATH) -> None:
    indexed_paths = {os.path.abspath(path) for path in load_index(index_path).values()}
    archives = find_tar_gz_files_recursive(targz_root)
    added, failed = 0, 0
    for archive in tqdm.tqdm(archives):
        if os.path.abspath(archive) in indexed_paths:
            continue
        try:
            meta = read_archive_metadata(archive)
        except Exception as exc:
            print(f"[ERROR] {archive}: {exc}")
            failed += 1
            continue
        dirname = os.path.basename(archive)[: -len(".tar.gz")]
        append_index(
            meta["manifest_id"], archive, dirname, len(meta["files"]),
            index_path=index_path,
        )
        added += 1
    print(f"{added} archives added to {index_path} ({len(archives)} on disk, {failed} unreadable)")


if __name__ == "__main__":
    main(*sys.argv[1:])
