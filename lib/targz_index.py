"""Append-only index mapping manifest URIs to their tar.gz archives.

``targz/index.csv`` is the authoritative URI ⇄ archive map, written each time
an archive is produced. Legacy archives that predate the index are still found
through a name-glob fallback; run ``tool-rebuild-targz-index.py`` once to
backfill them.
"""
import csv
import datetime
import fcntl
import glob
import hashlib
import os
from typing import Dict, Optional

from lib.uris import uri_variants

DEFAULT_INDEX_PATH = "targz/index.csv"
INDEX_HEADER = ["manifest_url", "archive_path", "dirname", "n_files", "archived_at"]


def append_index(
    manifest_uri: str,
    archive_path: str,
    dirname: str,
    n_files: int,
    index_path: str = DEFAULT_INDEX_PATH,
) -> None:
    os.makedirs(os.path.dirname(index_path) or ".", exist_ok=True)
    with open(index_path, "a", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(INDEX_HEADER)
            writer.writerow([
                manifest_uri, str(archive_path), dirname, n_files,
                datetime.datetime.now().isoformat(),
            ])
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def load_index(index_path: str = DEFAULT_INDEX_PATH) -> Dict[str, str]:
    """manifest_url → archive_path (last write wins)."""
    index: Dict[str, str] = {}
    try:
        with open(index_path, newline="") as f:
            for row in csv.DictReader(f):
                index[row["manifest_url"]] = row["archive_path"]
    except FileNotFoundError:
        pass
    return index


def archive_exists(
    manifest_uri: str,
    dirname: str,
    index_path: str = DEFAULT_INDEX_PATH,
    targz_root: str = "targz",
) -> Optional[str]:
    """Path of the archive holding this manifest, or None.

    Checks the index first (under every known URI variant), then falls back to
    the legacy name-glob — unless the index shows that the name belongs to a
    different manifest (kebab-label collision).
    """
    index = load_index(index_path)
    for variant in uri_variants(manifest_uri):
        path = index.get(variant)
        if path and os.path.exists(path):
            return path
    owned_by_other = {
        os.path.abspath(path)
        for uri, path in index.items()
        if uri not in uri_variants(manifest_uri)
    }
    for hit in glob.glob(f"{targz_root}/**/{dirname}.tar.gz", recursive=True):
        if os.path.abspath(hit) not in owned_by_other:
            return hit
    return None


def resolve_archive_basename(
    dirname: str,
    manifest_uri: str,
    index_path: str = DEFAULT_INDEX_PATH,
    targz_root: str = "targz",
) -> str:
    """Basename to archive this manifest under.

    Normally ``<dirname>.tar.gz``; when a tar with that name already exists
    and belongs to a *different* manifest (same kebab label), a short URI-hash
    suffix is added so the existing archive is not overwritten.
    """
    default = f"{dirname}.tar.gz"
    hits = glob.glob(f"{targz_root}/**/{default}", recursive=True)
    if not hits:
        return default
    index = load_index(index_path)
    variants = set(uri_variants(manifest_uri))
    for hit in hits:
        owners = {uri for uri, path in index.items() if os.path.abspath(path) == os.path.abspath(hit)}
        if not owners or owners & variants:
            # Unindexed (legacy, assume ours) or indexed under this manifest
            return default
    suffix = hashlib.sha1(manifest_uri.encode("utf-8")).hexdigest()[:8]
    return f"{dirname}-{suffix}.tar.gz"
