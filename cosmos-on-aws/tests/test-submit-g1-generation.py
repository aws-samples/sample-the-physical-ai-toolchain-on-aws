#!/usr/bin/env python3
"""Tests for submit-g1-generation.py; no AWS or network calls are used."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import subprocess
import tempfile
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

    def test_edge_request_contains_source_control_and_recorded_settings(self) -> None:
        command = MODULE.build_create_command(
            "http://127.0.0.1:8000",
            "/models/cosmos3-nano",
            "Preserve the source motion under warm lighting.",
            "jittery, deformed hands",
            2026,
            Path("/work/source.mp4"),
            Path("/work/control-edge.mp4"),
        )
        joined = "\n".join(command)
        self.assertIn("input_reference=@/work/source.mp4;type=video/mp4", joined)
        self.assertIn(
            "control_reference=@/work/control-edge.mp4;type=video/mp4", joined
        )
        self.assertIn("control_type=edge", joined)
        self.assertIn("size=1104x832", joined)
        self.assertIn("num_frames=121", joined)
        self.assertIn("fps=30", joined)
        self.assertIn("num_inference_steps=50", joined)
        self.assertIn("guidance_scale=3.0", joined)
        self.assertIn('"control_guidance":2.0', joined)
        self.assertIn('"control_weight":1.0', joined)

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

    def test_missing_local_ffmpeg_uses_cosmos_pod(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.mp4"
            destination = Path(directory) / "normalized.mp4"
            source.write_bytes(b"source")

            def fake_run(command: list[str], *, capture: bool = False):
                if command[:3] == ["kubectl", "get", "pod"]:
                    return subprocess.CompletedProcess(command, 0, "cosmos-pod", "")
                if command[:2] == ["kubectl", "cp"] and ":" in command[2]:
                    Path(command[3]).write_bytes(b"normalized")
                return subprocess.CompletedProcess(command, 0, "", "")

            def fake_which(name: str):
                return None if name == "ffmpeg" else f"/usr/bin/{name}"

            with mock.patch.object(MODULE.shutil, "which", side_effect=fake_which), \
                 mock.patch.object(MODULE, "run", side_effect=fake_run), \
                 mock.patch.object(MODULE.subprocess, "run"):
                method = MODULE.normalize_video(
                    source,
                    destination,
                    "cosmos3-g1-server",
                    "warm-lighting-v1",
                )

            self.assertEqual(method, "Cosmos Pod cosmos-pod")
            self.assertEqual(destination.read_bytes(), b"normalized")

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
