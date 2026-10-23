#!/usr/bin/env python3
"""Submit one G1 V2V generation to a port-forwarded Cosmos 3 service."""

from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import urlopen

DEFAULT_INPUT_KEY = (
    "teleoperation-data/pick_place_data/videos/chunk-000/"
    "observation.images.cam_high/episode_000000.mp4"
)
DEFAULT_NEGATIVE_PROMPT = (
    "blurry, distorted, low quality, jittery, jerky motion, sudden acceleration, "
    "deformed hands, fused fingers, extra fingers, floating objects, deforming box, "
    "slipping grasp, camera shake, camera movement, scene cuts, flicker"
)
RUN_ID = re.compile(r"^[a-z0-9](?:[-a-z0-9]{0,47}[a-z0-9])?$")


def run(command: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=True,
        text=True,
        capture_output=capture,
    )


def load_stack_outputs(stack_name: str, region: str) -> dict[str, str]:
    try:
        result = run(
            [
                "aws",
                "cloudformation",
                "describe-stacks",
                "--stack-name",
                stack_name,
                "--region",
                region,
                "--output",
                "json",
            ],
            capture=True,
        )
        rows = json.loads(result.stdout)["Stacks"][0]["Outputs"]
    except FileNotFoundError as exc:
        raise RuntimeError("AWS CLI is not installed or is not on PATH") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(exc.stderr.strip() or "AWS CLI returned an error") from exc
    except (json.JSONDecodeError, KeyError, IndexError) as exc:
        raise RuntimeError("CloudFormation returned an unexpected response") from exc
    return {row["OutputKey"]: row["OutputValue"] for row in rows}


def validate_settings(run_id: str, prompt: str, seed: int, server_url: str) -> None:
    if not RUN_ID.fullmatch(run_id):
        raise ValueError(
            "run ID must use lowercase letters, numbers, and hyphens (maximum 49 characters)"
        )
    if not prompt.strip():
        raise ValueError("prompt must not be empty")
    if not 0 <= seed <= 4_294_967_295:
        raise ValueError("seed must be between 0 and 4294967295")
    parsed = urlparse(server_url)
    if parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or not parsed.port:
        raise ValueError("server URL must be an explicit http://127.0.0.1:<port> URL")
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise ValueError("server URL must not contain a path, query, or fragment")


def output_names(run_id: str) -> tuple[str, str]:
    return (
        f"g1_pick_blue_box_{run_id}.mp4",
        f"g1_pick_blue_box_{run_id}_reference.mp4",
    )


def object_exists(bucket: str, key: str, region: str) -> bool:
    result = subprocess.run(
        [
            "aws",
            "s3api",
            "head-object",
            "--bucket",
            bucket,
            "--key",
            key,
            "--region",
            region,
        ],
        text=True,
        capture_output=True,
    )
    if result.returncode == 0:
        return True
    detail = result.stderr
    if "404" in detail or "Not Found" in detail or "NoSuchKey" in detail:
        return False
    raise RuntimeError(detail.strip() or f"unable to inspect s3://{bucket}/{key}")


def get_json(url: str, timeout: int = 15) -> dict:
    try:
        with urlopen(url, timeout=timeout) as response:
            return json.load(response)
    except (HTTPError, URLError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cosmos service request failed: {exc}") from exc


def get_model_id(server_url: str) -> str:
    document = get_json(server_url.rstrip("/") + "/v1/models")
    models = document.get("data", [])
    if not models or not models[0].get("id"):
        raise RuntimeError("Cosmos service returned no model ID")
    return str(models[0]["id"])


def build_create_command(
    server_url: str,
    model_id: str,
    prompt: str,
    negative_prompt: str,
    seed: int,
    video_path: Path,
) -> list[str]:
    return [
        "curl",
        "-fsS",
        "-X",
        "POST",
        server_url.rstrip("/") + "/v1/videos",
        "-H",
        "Accept: application/json",
        "--form-string",
        f"model={model_id}",
        "--form-string",
        f"prompt={prompt}",
        "--form-string",
        f"negative_prompt={negative_prompt}",
        "--form-string",
        "size=1280x720",
        "--form-string",
        "num_frames=189",
        "--form-string",
        "fps=24",
        "--form-string",
        "num_inference_steps=35",
        "--form-string",
        "guidance_scale=6.0",
        "--form-string",
        "max_sequence_length=4096",
        "--form-string",
        "flow_shift=10.0",
        "--form-string",
        'extra_params={"condition_frame_indexes_vision":[0,1],"condition_video_keep":"first"}',
        "--form-string",
        f"seed={seed}",
        "-F",
        f"input_reference=@{video_path};type=video/mp4",
    ]


def wait_for_completion(server_url: str, video_id: str, timeout_minutes: int) -> None:
    deadline = time.monotonic() + timeout_minutes * 60
    poll = 0
    while time.monotonic() < deadline:
        poll += 1
        document = get_json(server_url.rstrip("/") + f"/v1/videos/{video_id}")
        status = document.get("status", "unknown")
        print(f"[{poll}] status={status}")
        if status == "completed":
            return
        if status in {"failed", "cancelled"}:
            raise RuntimeError(f"Cosmos generation {status}: {document}")
        time.sleep(10)
    raise RuntimeError(f"Cosmos generation exceeded {timeout_minutes} minutes")


def download_content(server_url: str, video_id: str, destination: Path) -> None:
    url = server_url.rstrip("/") + f"/v1/videos/{video_id}/content"
    try:
        with urlopen(url, timeout=120) as response, destination.open("wb") as target:
            shutil.copyfileobj(response, target)
    except (HTTPError, URLError, OSError) as exc:
        raise RuntimeError(f"Unable to download generated video: {exc}") from exc
    if destination.stat().st_size == 0:
        raise RuntimeError("Generated video is empty")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit one sequential G1 V2V request to the reusable Cosmos service"
    )
    parser.add_argument("--region", required=True)
    parser.add_argument("--stack-name", default="physical-ai-toolchain")
    parser.add_argument(
        "--data-bucket",
        help="Validated participant data bucket; otherwise read from stack outputs",
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--server-url", default="http://127.0.0.1:8000")
    parser.add_argument("--input-key", default=DEFAULT_INPUT_KEY)
    parser.add_argument("--output-prefix", default="cosmos3-generated")
    parser.add_argument("--timeout-minutes", type=int, default=30)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        validate_settings(args.run_id, args.prompt, args.seed, args.server_url)
        if args.timeout_minutes < 1 or args.timeout_minutes > 60:
            raise ValueError("timeout must be between 1 and 60 minutes")
        for executable in ("aws", "curl", "ffmpeg"):
            if shutil.which(executable) is None:
                raise RuntimeError(f"required command is not installed: {executable}")

        lock_path = Path.home() / ".cache" / "cosmos3-generation.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock = lock_path.open("w")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                "another Cosmos generation client is already running on this workstation"
            ) from exc

        if args.data_bucket:
            bucket = args.data_bucket
        else:
            outputs = load_stack_outputs(args.stack_name, args.region)
            bucket = outputs.get("DataBucketName")
            if not bucket:
                raise RuntimeError("DataBucketName was not found in stack outputs")
        generated_name, reference_name = output_names(args.run_id)
        for name in (generated_name, reference_name):
            key = f"{args.output_prefix}/{name}"
            if object_exists(bucket, key, args.region):
                raise RuntimeError(
                    f"run ID '{args.run_id}' already has output at s3://{bucket}/{key}"
                )

        model_id = get_model_id(args.server_url)
        print(f"PASS - Cosmos service ready: {model_id}")
        print(f"Run ID: {args.run_id}")
        print(f"Generated object: s3://{bucket}/{args.output_prefix}/{generated_name}")

        with tempfile.TemporaryDirectory(prefix=f"cosmos3-{args.run_id}-") as directory:
            work = Path(directory)
            source = work / "reference-source.mp4"
            normalized = work / "reference-h264.mp4"
            generated = work / "generated.mp4"

            run(
                [
                    "aws",
                    "s3",
                    "cp",
                    f"s3://{bucket}/{args.input_key}",
                    str(source),
                    "--region",
                    args.region,
                    "--only-show-errors",
                ]
            )
            run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(source),
                    "-an",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    str(normalized),
                ]
            )
            response = run(
                build_create_command(
                    args.server_url,
                    model_id,
                    args.prompt,
                    args.negative_prompt,
                    args.seed,
                    normalized,
                ),
                capture=True,
            )
            try:
                video_id = json.loads(response.stdout)["id"]
            except (json.JSONDecodeError, KeyError) as exc:
                raise RuntimeError(
                    f"Cosmos returned an unexpected submission response: {response.stdout}"
                ) from exc
            print(f"Submitted video ID: {video_id}")
            wait_for_completion(args.server_url, video_id, args.timeout_minutes)
            download_content(args.server_url, video_id, generated)

            destination = f"s3://{bucket}/{args.output_prefix}"
            run(
                [
                    "aws",
                    "s3",
                    "cp",
                    str(generated),
                    f"{destination}/{generated_name}",
                    "--region",
                    args.region,
                    "--sse",
                    "AES256",
                    "--only-show-errors",
                ]
            )
            run(
                [
                    "aws",
                    "s3",
                    "cp",
                    str(source),
                    f"{destination}/{reference_name}",
                    "--region",
                    args.region,
                    "--sse",
                    "AES256",
                    "--only-show-errors",
                ]
            )
            print(f"PASS - Generated video bytes: {generated.stat().st_size}")
            print(f"PASS - Uploaded reference: {reference_name}")
            print("GENERATION_COMPLETE")
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        detail = exc.stderr.strip() if isinstance(exc, subprocess.CalledProcessError) and exc.stderr else str(exc)
        print(f"FAIL - {detail}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
