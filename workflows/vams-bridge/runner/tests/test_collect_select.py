"""Collect copies the deployable model, not the state needed to resume training (D12)."""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
for name, value in {"VAMS_DATA_ROLE_ARN": "arn:aws:iam::000000000000:role/data",
                    "VAMS_CONTROL_ROLE_ARN": "arn:aws:iam::000000000000:role/control",
                    "VAMS_ROLE_EXTERNAL_ID": "x", "CHECKPOINTS_BUCKET": "checkpoints",
                    "VAMS_BRIDGE_LOG_GROUP": "/aws/vendedlogs/Pipelines/pai-test",
                    "AWS_DEFAULT_REGION": "us-east-1"}.items():
    os.environ.setdefault(name, value)
import collect  # noqa: E402

# The GR00T N1.7 output of a 100-step batch-n17 run, as the container uploaded it (38.1 GB).
CHECKPOINT = ["config.json", "embodiment_id.json", "experiment_cfg/conf.yaml", "experiment_cfg/config.yaml",
              "experiment_cfg/dataset_statistics.json", "experiment_cfg/final_model_config.json",
              "experiment_cfg/final_processor_config.json", "model-00001-of-00003.safetensors",
              "model-00002-of-00003.safetensors", "model-00003-of-00003.safetensors",
              "model.safetensors.index.json", "optimizer.pt", "processor_config.json", "rng_state.pth",
              "scheduler.pt", "statistics.json", "trainer_state.json", "training_args.bin", "wandb_config.json"]
FINAL_MODEL = ["config.json", "experiment_cfg/conf.yaml", "experiment_cfg/config.yaml",
               "experiment_cfg/dataset_statistics.json", "experiment_cfg/final_model_config.json",
               "experiment_cfg/final_processor_config.json", "model-00001-of-00003.safetensors",
               "model-00002-of-00003.safetensors", "model-00003-of-00003.safetensors",
               "model.safetensors.index.json", "processor/embodiment_id.json", "processor/processor_config.json",
               "processor/statistics.json", "training_args.bin", "training_metadata.json",
               "ur3_embodiment/ur3_config.py", "wandb_config.json"]
RUN7 = FINAL_MODEL + ["ur3_embodiment/__pycache__/ur3_config.cpython-312.pyc"] + [
    "checkpoint-100/" + k for k in CHECKPOINT]


class SelectDeployable(unittest.TestCase):
    def test_final_model_at_root_drops_every_intermediate_checkpoint(self):
        keep, skip = collect._select_deployable(RUN7)
        self.assertEqual(keep, sorted(FINAL_MODEL))
        self.assertTrue(all(k.startswith("checkpoint-100/") or "__pycache__" in k for k in skip))
        self.assertEqual(len(keep) + len(skip), len(RUN7))

    def test_without_a_final_model_the_latest_checkpoint_is_kept_minus_its_training_state(self):
        keys = ["checkpoint-50/" + k for k in CHECKPOINT] + ["checkpoint-100/" + k for k in CHECKPOINT]
        keep, skip = collect._select_deployable(keys)
        self.assertTrue(keep and all(k.startswith("checkpoint-100/") for k in keep))
        self.assertIn("checkpoint-100/model-00001-of-00003.safetensors", keep)
        for state in ("optimizer.pt", "scheduler.pt", "rng_state.pth"):
            self.assertNotIn("checkpoint-100/" + state, keep)
        state = {"optimizer.pt", "scheduler.pt", "rng_state.pth"}
        self.assertEqual(sorted(skip), sorted(["checkpoint-50/" + k for k in CHECKPOINT]
                                              + ["checkpoint-100/" + k for k in state]))

    def test_numeric_not_lexical_ordering_picks_the_latest_step(self):
        keep, _ = collect._select_deployable(["checkpoint-900/model.safetensors", "checkpoint-1000/model.safetensors"])
        self.assertEqual(keep, ["checkpoint-1000/model.safetensors"])

    def test_a_flat_output_without_checkpoint_dirs_is_copied_except_training_state(self):
        keep, skip = collect._select_deployable(["model.tar.gz", "optimizer.pt", "notes/readme.txt"])
        self.assertEqual(keep, ["model.tar.gz", "notes/readme.txt"])
        self.assertEqual(skip, ["optimizer.pt"])

    def test_a_directory_named_like_a_checkpoint_below_the_root_is_not_treated_as_one(self):
        keep, _ = collect._select_deployable(["model.safetensors", "eval/checkpoint-5/report.json"])
        self.assertEqual(keep, ["eval/checkpoint-5/report.json", "model.safetensors"])


if __name__ == "__main__":
    unittest.main()
