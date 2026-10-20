"""Tests for runner/checks.py. Run with ``python3 -m unittest discover -s runner/tests`` from
``workflows/vams-bridge`` (or ``pytest runner/tests``). No AWS access needed."""
from __future__ import annotations

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import checks  # noqa: E402


def _fixture(episodes=2, dims=6):
    info = {
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.state": {"shape": [dims]},
            "action": {"shape": [dims]},
            "observation.images.ego_view": {"dtype": "video"},
        },
    }
    modality = {
        "state": {"arm": {"start": 0, "end": 4}, "gripper": {"start": 4, "end": dims}},
        "action": {"arm": {"start": 0, "end": 4}, "gripper": {"start": 4, "end": dims}},
        "video": {"ego_view": {"original_key": "observation.images.ego_view"}},
        "annotation": {"human.task_description": {"original_key": "task_index"}},
    }
    eps = [{"episode_index": i, "length": 100} for i in range(episodes)]
    tasks = [{"task_index": 0, "task": "pick"}]
    meta = {
        "meta/info.json": json.dumps(info),
        "meta/modality.json": json.dumps(modality),
        "meta/episodes.jsonl": "\n".join(json.dumps(e) for e in eps),
        "meta/tasks.jsonl": "\n".join(json.dumps(t) for t in tasks),
    }
    keys = set(meta)
    for i in range(episodes):
        keys.add(f"data/chunk-000/episode_{i:06d}.parquet")
        keys.add(f"videos/chunk-000/observation.images.ego_view/episode_{i:06d}.mp4")
    return meta, keys, info, modality


class ValidateDataset(unittest.TestCase):
    def test_valid_fixture_passes(self):
        meta, keys, _, _ = _fixture()
        summary = checks.validate_dataset(meta, keys)
        self.assertEqual(summary, {"episodes": 2, "tasks": 1, "videoKeys": ["ego_view"]})

    def test_missing_meta_file_is_check_a(self):
        meta, keys, _, _ = _fixture()
        meta["meta/modality.json"] = None
        with self.assertRaises(checks.DatasetInvalid) as cm:
            checks.validate_dataset(meta, keys)
        self.assertIn("meta/modality.json is missing", str(cm.exception))

    def test_group_gap_is_check_b(self):
        meta, keys, _, modality = _fixture()
        modality["state"]["gripper"]["start"] = 5
        meta["meta/modality.json"] = json.dumps(modality)
        with self.assertRaises(checks.DatasetInvalid) as cm:
            checks.validate_dataset(meta, keys)
        self.assertIn("'gripper' starts at 5 but the previous group ended at 4", str(cm.exception))

    def test_original_key_on_state_group_is_check_c(self):
        meta, keys, _, modality = _fixture()
        modality["action"]["arm"]["original_key"] = "action"
        meta["meta/modality.json"] = json.dumps(modality)
        with self.assertRaises(checks.DatasetInvalid) as cm:
            checks.validate_dataset(meta, keys)
        self.assertIn("sets original_key", str(cm.exception))

    def test_video_key_not_in_features_is_check_e(self):
        meta, keys, _, modality = _fixture()
        modality["video"]["wrist"] = {}
        meta["meta/modality.json"] = json.dumps(modality)
        with self.assertRaises(checks.DatasetInvalid) as cm:
            checks.validate_dataset(meta, keys)
        self.assertIn("observation.images.wrist", str(cm.exception))

    def test_missing_parquet_is_check_g(self):
        meta, keys, _, _ = _fixture()
        keys.discard("data/chunk-000/episode_000001.parquet")
        with self.assertRaises(checks.DatasetInvalid) as cm:
            checks.validate_dataset(meta, keys)
        self.assertIn("episode 1: missing data/chunk-000/episode_000001.parquet", str(cm.exception))

    def test_episode_without_length_is_check_g(self):
        meta, keys, _, _ = _fixture()
        meta["meta/episodes.jsonl"] = json.dumps({"episode_index": 0})
        with self.assertRaises(checks.DatasetInvalid) as cm:
            checks.validate_dataset(meta, keys)
        self.assertIn("no 'length' field", str(cm.exception))

    def test_many_missing_episodes_are_summarised(self):
        meta, keys, _, _ = _fixture(episodes=9)
        keys = {k for k in keys if not k.endswith(".parquet")}
        with self.assertRaises(checks.DatasetInvalid) as cm:
            checks.validate_dataset(meta, keys)
        self.assertIn("4 further episode(s)", str(cm.exception))


class BatchDividesGpus(unittest.TestCase):
    def test_unreadable_gpu_count_is_advisory(self):
        checks.check_batch_divides_gpus(3, None)
        checks.check_batch_divides_gpus(3, 0)

    def test_divisible_passes(self):
        checks.check_batch_divides_gpus(8, 4)
        checks.check_batch_divides_gpus(2, 1)

    def test_not_divisible_or_too_small_is_refused(self):
        with self.assertRaises(ValueError):
            checks.check_batch_divides_gpus(6, 4)
        with self.assertRaises(ValueError):
            checks.check_batch_divides_gpus(2, 4)


if __name__ == "__main__":
    unittest.main()
