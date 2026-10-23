#!/usr/bin/env python3
"""Tests for prepare-g1-job.py; no AWS account is used."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent
SCRIPT = HERE.parent / "prepare-g1-job.py"
MANIFEST = HERE.parent / "cosmos3-job.yaml"
SPEC = spec_from_file_location("prepare_g1_job", SCRIPT)
MODULE = module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class PrepareG1JobTest(unittest.TestCase):
    def setUp(self) -> None:
        self.outputs = {
            "EksClusterName": "physical-ai-dev-cosmos3",
            "DataBucketName": "physical-ai-data-123456789012",
            "CosmosRuntimeImageUri": "vllm/vllm-omni:cosmos3",
            "Cosmos3ArchiveS3Uri": (
                "s3://physical-ai-data-123456789012/"
                "cosmos3-assets/archives/cosmos3-nano.tar"
            ),
            "Cosmos3ArchiveSha256": MODULE.EXPECTED_ARCHIVE_SHA256,
        }

    def test_renders_validated_nano_g1_job(self) -> None:
        rendered = MODULE.render_job(MANIFEST.read_text(), "us-east-2", self.outputs)
        self.assertIn("name: cosmos3-g1-generate", rendered)
        self.assertIn('value: "nano"', rendered)
        self.assertEqual(rendered.count('nvidia.com/gpu: "1"'), 2)
        self.assertIn("observation.images.cam_high/episode_000000.mp4", rendered)
        self.assertIn("g1_pick_blue_box_generated.mp4", rendered)
        self.assertIn("both hands approach slowly", rendered)
        self.assertIn("firm symmetric grasp", rendered)
        self.assertIn("brief stabilization pause", rendered)
        self.assertIn("slow constant speed", rendered)
        self.assertIn("motion in one continuous shot", rendered)
        self.assertIn("negative_prompt=", rendered)
        self.assertIn("jittery, jerky motion", rendered)
        self.assertIn("deformed hands", rendered)
        self.assertIn("slipping grasp", rendered)
        self.assertIn("models--Qwen--Qwen3Guard-Gen-0.6B", rendered)
        self.assertIn("-c:v libx264", rendered)
        self.assertNotRegex(rendered, r"<(?:REGION|DATASETS_BUCKET|DATA_BUCKET)>")

    def test_renders_custom_prompt_and_unique_run_settings(self) -> None:
        prompt = (
            "A fixed robot camera observes both hands.\n"
            "The hands establish a stable grasp, then lift the box slowly."
        )
        rendered = MODULE.render_job(
            MANIFEST.read_text(),
            "us-east-2",
            self.outputs,
            prompt=prompt,
            job_name="cosmos3-g1-smooth-pickup-v2",
            output_filename="g1_pick_blue_box_smooth-pickup-v2.mp4",
            reference_filename="g1_pick_blue_box_smooth-pickup-v2_reference.mp4",
            seed=200,
        )
        self.assertIn("name: cosmos3-g1-smooth-pickup-v2", rendered)
        self.assertIn("app: cosmos3-g1-smooth-pickup-v2", rendered)
        self.assertIn("A fixed robot camera observes both hands.", rendered)
        self.assertIn("establish a stable grasp", rendered)
        self.assertIn(
            'name: OUTPUT_FILENAME, value: "g1_pick_blue_box_smooth-pickup-v2.mp4"',
            rendered,
        )
        self.assertIn(
            'name: REFERENCE_FILENAME, value: '
            '"g1_pick_blue_box_smooth-pickup-v2_reference.mp4"',
            rendered,
        )
        self.assertIn('name: SEED, value: "200"', rendered)

    def test_rejects_invalid_run_settings(self) -> None:
        with self.assertRaisesRegex(ValueError, "Kubernetes DNS label"):
            MODULE.render_job(
                MANIFEST.read_text(),
                "us-east-2",
                self.outputs,
                job_name="Invalid_Name",
            )
        with self.assertRaisesRegex(ValueError, "simple .mp4 filename"):
            MODULE.render_job(
                MANIFEST.read_text(),
                "us-east-2",
                self.outputs,
                output_filename="../output.mp4",
            )
        with self.assertRaisesRegex(ValueError, "prompt must not be empty"):
            MODULE.render_job(
                MANIFEST.read_text(), "us-east-2", self.outputs, prompt="  "
            )

    def test_rejects_stale_runtime_bundle_checksum(self) -> None:
        self.outputs["Cosmos3ArchiveSha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "stale"):
            MODULE.validate_outputs(self.outputs)

    def test_rejects_missing_stack_output(self) -> None:
        del self.outputs["CosmosRuntimeImageUri"]
        with self.assertRaisesRegex(ValueError, "CosmosRuntimeImageUri"):
            MODULE.validate_outputs(self.outputs)


if __name__ == "__main__":
    unittest.main()
