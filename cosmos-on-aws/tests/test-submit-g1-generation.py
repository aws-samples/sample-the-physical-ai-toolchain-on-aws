#!/usr/bin/env python3
"""Tests for submit-g1-generation.py; no AWS or network calls are used."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import subprocess
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
SCRIPT = HERE.parent / "submit-g1-generation.py"
SPEC = spec_from_file_location("submit_g1_generation", SCRIPT)
MODULE = module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class SubmitG1GenerationTest(unittest.TestCase):
    def test_request_contains_prompt_negative_prompt_and_video(self) -> None:
        video = Path("/work/reference-h264.mp4")
        command = MODULE.build_create_command(
            "http://127.0.0.1:8000",
            "/models/cosmos3-nano",
            "A smooth pickup under warm lighting.",
            "jittery, deformed hands",
            200,
            video,
        )
        joined = "\n".join(command)
        self.assertIn("http://127.0.0.1:8000/v1/videos", joined)
        self.assertIn("prompt=A smooth pickup under warm lighting.", joined)
        self.assertIn("negative_prompt=jittery, deformed hands", joined)
        self.assertIn("input_reference=@/work/reference-h264.mp4;type=video/mp4", joined)
        self.assertIn("seed=200", joined)
        self.assertIn("condition_frame_indexes_vision", joined)

    def test_output_names_are_unique_by_run_id(self) -> None:
        self.assertEqual(
            MODULE.output_names("warm-lighting-v1"),
            (
                "g1_pick_blue_box_warm-lighting-v1.mp4",
                "g1_pick_blue_box_warm-lighting-v1_reference.mp4",
            ),
        )

    def test_s3_output_preflight_distinguishes_missing_and_existing(self) -> None:
        with mock.patch.object(
            MODULE.subprocess,
            "run",
            return_value=subprocess.CompletedProcess([], 0, "", ""),
        ):
            self.assertTrue(MODULE.object_exists("bucket", "output.mp4", "us-east-2"))

        with mock.patch.object(
            MODULE.subprocess,
            "run",
            return_value=subprocess.CompletedProcess([], 254, "", "404 Not Found"),
        ):
            self.assertFalse(MODULE.object_exists("bucket", "output.mp4", "us-east-2"))

        with mock.patch.object(
            MODULE.subprocess,
            "run",
            return_value=subprocess.CompletedProcess([], 254, "", "AccessDenied"),
        ):
            with self.assertRaisesRegex(RuntimeError, "AccessDenied"):
                MODULE.object_exists("bucket", "output.mp4", "us-east-2")

    def test_rejects_non_loopback_server(self) -> None:
        with self.assertRaisesRegex(ValueError, "127.0.0.1"):
            MODULE.validate_settings(
                "warm-lighting-v1",
                "prompt",
                100,
                "http://cosmos3.example.com:8000",
            )

    def test_rejects_invalid_run_or_empty_prompt(self) -> None:
        with self.assertRaisesRegex(ValueError, "run ID"):
            MODULE.validate_settings(
                "Invalid_Run", "prompt", 100, "http://127.0.0.1:8000"
            )
        with self.assertRaisesRegex(ValueError, "prompt"):
            MODULE.validate_settings(
                "valid-run", "  ", 100, "http://127.0.0.1:8000"
            )


if __name__ == "__main__":
    unittest.main()
