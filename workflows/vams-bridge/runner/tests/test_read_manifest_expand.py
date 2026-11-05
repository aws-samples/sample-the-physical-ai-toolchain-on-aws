"""The whole-asset / folder manifest entry is expanded into one entry per object before staging."""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
for name, value in {"VAMS_DATA_ROLE_ARN": "arn:aws:iam::000000000000:role/data",
                    "VAMS_ROLE_EXTERNAL_ID": "x", "DATASETS_BUCKET": "datasets",
                    "CHECKPOINTS_BUCKET": "checkpoints"}.items():
    os.environ.setdefault(name, value)
import read_manifest  # noqa: E402

ROOT = "assets/x42/"
OBJECTS = [ROOT, ROOT + "meta/info.json", ROOT + "meta/", ROOT + "data/chunk-000/episode_000000.parquet",
           ROOT + "videos/chunk-000/observation.images.wrist/episode_000000.mp4"]


def _lister(entry):
    assert entry["bucket"] == "asset-bucket"
    return [k for k in OBJECTS if k.startswith(entry["key"])]


def _entry(key, relative_path, version_id="v1"):
    return {"relativePath": relative_path, "bucket": "asset-bucket", "key": key, "versionId": version_id,
            "bucketRegion": "", "databaseId": "db", "assetId": "x42"}


class ExpandFolderEntries(unittest.TestCase):
    def test_whole_asset_entry_becomes_one_entry_per_object(self):
        out = read_manifest._expand_folder_entries([_entry(ROOT, "/")], _lister)
        self.assertEqual(sorted(e["relativePath"] for e in out),
                         ["/data/chunk-000/episode_000000.parquet", "/meta/info.json",
                          "/videos/chunk-000/observation.images.wrist/episode_000000.mp4"])
        self.assertTrue(all(e["key"].startswith(ROOT) and not e["key"].endswith("/") for e in out))
        self.assertTrue(all(e["versionId"] == "" for e in out), "a prefix cannot pin a version")
        self.assertTrue(all(e["bucket"] == "asset-bucket" and e["assetId"] == "x42" for e in out))

    def test_subfolder_entry_keeps_its_relative_root(self):
        out = read_manifest._expand_folder_entries([_entry(ROOT + "meta/", "/meta/")], _lister)
        self.assertEqual([e["relativePath"] for e in out], ["/meta/info.json"])

    def test_file_entries_pass_through_untouched(self):
        entry = _entry(ROOT + "meta/info.json", "/meta/info.json")
        out = read_manifest._expand_folder_entries([entry], lambda e: self.fail("must not list"))
        self.assertEqual(out, [entry])

    def test_empty_folder_is_rejected(self):
        with self.assertRaises(ValueError):
            read_manifest._expand_folder_entries([_entry(ROOT + "empty/", "/empty/")], lambda e: [])


if __name__ == "__main__":
    unittest.main()
