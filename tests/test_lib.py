"""Tests for the rtk-free lib modules (uris, tar_utils, targz_index) and the
index rebuild tool. Only needs the light deps (tqdm); run from the repo root:

    python -m unittest discover -s tests
"""
import importlib.util
import io
import json
import os
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from lib.uris import to_gallica, to_openapi, uri_variants
from lib.tar_utils import (
    build_archive_metadata, create_tar_gz_archives, get_manifest_and_xmls,
    read_archive_metadata, read_file_from_tar,
)
from lib.targz_index import (
    append_index, load_index, archive_exists, resolve_archive_basename,
)

GALLICA = "https://gallica.bnf.fr/iiif/ark:/12148/btv1b8451103g"
OPENAPI = "https://openapi.bnf.fr/iiif/presentation/v3/ark:/12148/btv1b8451103g"


class UriTests(unittest.TestCase):
    def test_rewrites(self):
        self.assertEqual(to_openapi(GALLICA), OPENAPI)
        self.assertEqual(to_gallica(OPENAPI), GALLICA)
        self.assertEqual(to_gallica(GALLICA), GALLICA)
        self.assertEqual(to_openapi(OPENAPI), OPENAPI)

    def test_variants(self):
        gallica_http = GALLICA.replace("https://", "http://")
        self.assertEqual(set(uri_variants(GALLICA)), {GALLICA, OPENAPI, gallica_http})
        self.assertEqual(set(uri_variants(OPENAPI)), {OPENAPI, GALLICA, gallica_http})
        self.assertEqual(set(uri_variants(gallica_http)), {gallica_http, GALLICA, OPENAPI})
        foreign = "https://iiif.bodleian.ox.ac.uk/iiif/manifest/x.json"
        self.assertEqual(uri_variants(foreign), [foreign])

    def test_http_rewrites(self):
        gallica_http = GALLICA.replace("https://", "http://")
        self.assertEqual(to_openapi(gallica_http), OPENAPI)
        self.assertEqual(to_gallica(gallica_http), GALLICA)


class ArchiveScenarioTests(unittest.TestCase):
    """Full scenario in a temp dir: build metadata, archive, index, rebuild."""

    def setUp(self):
        self._cwd = os.getcwd()
        self._tmp = tempfile.TemporaryDirectory()
        os.chdir(self._tmp.name)

        # Fake manuscript: two ALTO files, deliberately unsorted page order
        self.ms_dir = Path("my-manuscript")
        self.ms_dir.mkdir()
        self.order = ["p0002", "p0001"]
        for stem in ["p0001", "p0002"]:
            (self.ms_dir / f"{stem}.xml").write_text(f"<alto>{stem}</alto>")
        os.makedirs("output")
        with open("output/fake.csv", "w") as f:
            f.write("https://img/p0001.jpg,my-manuscript,p0001\n")
            f.write("https://img/p0002.jpg,my-manuscript,p0002\n")

        self.meta = build_archive_metadata(
            manifest_id=OPENAPI, directory=str(self.ms_dir),
            image_order=self.order, total_images=2, errors=[],
            csv_path="output/fake.csv",
        )
        paths = [self.ms_dir / f"{s}.xml" for s in ["p0001", "p0002"]]
        ordered = sorted(paths, key=lambda p: self.order.index(p.stem))
        self.tar_path = Path("targz/batch_000/my-manuscript.tar.gz")
        create_tar_gz_archives(
            uri_to_files={OPENAPI: paths}, ordering_dict={OPENAPI: ordered},
            naming_func=lambda x: self.tar_path, metadata={OPENAPI: self.meta},
        )

        # Legacy archive: manifest.txt only, full local paths
        self.legacy = Path("targz/batch_000/legacy-ms.tar.gz")
        with tarfile.open(self.legacy, "w:gz") as t:
            data = (
                f"{GALLICA}\n/cluster/data/legacy-ms/p0001.xml"
                f"\n/cluster/data/legacy-ms/p0002.xml"
            ).encode()
            info = tarfile.TarInfo("manifest.txt")
            info.size = len(data)
            t.addfile(info, io.BytesIO(data))

    def tearDown(self):
        os.chdir(self._cwd)
        self._tmp.cleanup()

    def test_metadata(self):
        self.assertEqual(self.meta["schema_version"], 2)
        self.assertEqual(self.meta["source_manifest_id"], GALLICA)
        self.assertEqual(
            {i["filename"]: i["image_url"] for i in self.meta["images"]},
            {"p0001": "https://img/p0001.jpg", "p0002": "https://img/p0002.jpg"},
        )

    def test_archive_members_and_manifest_txt(self):
        with tarfile.open(self.tar_path) as t:
            names = set(t.getnames())
        self.assertEqual(
            names, {"manifest.txt", "manifest.json", "p0001.xml", "p0002.xml"}
        )
        lines = read_file_from_tar(str(self.tar_path), "manifest.txt").split("\n")
        self.assertEqual(lines[0], OPENAPI)
        # Basenames, in page order
        self.assertEqual(lines[1:], ["p0002.xml", "p0001.xml"])

    def test_read_archive_metadata_v2_and_legacy(self):
        v2 = read_archive_metadata(str(self.tar_path))
        self.assertEqual(v2["schema_version"], 2)
        self.assertEqual(v2["manifest_id"], OPENAPI)
        self.assertEqual(v2["files"], ["p0002.xml", "p0001.xml"])

        legacy = read_archive_metadata(str(self.legacy))
        self.assertEqual(
            legacy,
            {"schema_version": 1, "manifest_id": GALLICA,
             "files": ["p0001.xml", "p0002.xml"]},
        )

    def test_get_manifest_and_xmls(self):
        mpath, xmls = get_manifest_and_xmls(str(self.tar_path))
        self.assertEqual(mpath, "manifest.txt")
        self.assertEqual(set(xmls), {"p0001.xml", "p0002.xml"})

    def test_index_and_resume(self):
        append_index(OPENAPI, str(self.tar_path), "my-manuscript", 2)
        self.assertEqual(load_index().get(OPENAPI), str(self.tar_path))
        # Found under either URI variant
        self.assertEqual(archive_exists(GALLICA, "my-manuscript"), str(self.tar_path))
        # Legacy archive found through the name-glob fallback
        self.assertEqual(archive_exists("https://x/unindexed", "legacy-ms"), str(self.legacy))
        self.assertIsNone(archive_exists("https://x/nope", "no-such-dir"))
        # Same dirname but the archive is owned by another URI: no false resume
        self.assertIsNone(archive_exists("https://x/other", "my-manuscript"))

    def test_resolve_archive_basename(self):
        append_index(OPENAPI, str(self.tar_path), "my-manuscript", 2)
        self.assertEqual(resolve_archive_basename("brand-new", "https://x/a"), "brand-new.tar.gz")
        self.assertEqual(resolve_archive_basename("my-manuscript", GALLICA), "my-manuscript.tar.gz")
        collided = resolve_archive_basename("my-manuscript", "https://x/other")
        self.assertTrue(collided.startswith("my-manuscript-"))
        self.assertNotEqual(collided, "my-manuscript.tar.gz")
        # Unindexed legacy archive with the same name is assumed to be ours
        self.assertEqual(resolve_archive_basename("legacy-ms", "https://whatever"), "legacy-ms.tar.gz")

    def test_missing_csv_keeps_archiving(self):
        meta = build_archive_metadata(
            manifest_id=OPENAPI, directory=str(self.ms_dir),
            image_order=self.order, total_images=2, errors=[],
            csv_path="output/does-not-exist.csv",
        )
        self.assertEqual([i["image_url"] for i in meta["images"]], ["", ""])

    def test_rebuild_tool_backfills_and_is_idempotent(self):
        append_index(OPENAPI, str(self.tar_path), "my-manuscript", 2)
        spec = importlib.util.spec_from_file_location(
            "rebuild", REPO / "scripts" / "tool-rebuild-targz-index.py"
        )
        rebuild = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rebuild)

        rebuild.main()
        index = load_index()
        self.assertEqual(index.get(GALLICA), str(self.legacy))
        n_rows = len(Path("targz/index.csv").read_text().splitlines())
        rebuild.main()
        self.assertEqual(len(Path("targz/index.csv").read_text().splitlines()), n_rows)


if __name__ == "__main__":
    unittest.main()
