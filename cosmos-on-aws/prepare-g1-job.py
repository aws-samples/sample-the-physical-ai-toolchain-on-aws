#!/usr/bin/env python3
"""Render a workshop Cosmos 3 Nano/G1 Job from CloudFormation outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

EXPECTED_ARCHIVE_SHA256 = (
    "a0660a0494018fff848fda34ecf0e8a823a360502a0254b5f5068f2e1c7f82ff"
)
REQUIRED_OUTPUTS = (
    "EksClusterName",
    "DataBucketName",
    "CosmosRuntimeImageUri",
    "Cosmos3ArchiveS3Uri",
    "Cosmos3ArchiveSha256",
)
DEFAULT_PROMPT = (
    "A fixed head-mounted camera view from a Unitree G1 humanoid robot shows a "
    "white tabletop and its two open three-fingered Dex3 hands on opposite sides "
    "of a rigid blue rectangular box. At first, both hands approach slowly while "
    "the box remains still. The fingertips align with the box's side faces, then "
    "all fingers close simultaneously to form a firm symmetric grasp. After a "
    "brief stabilization pause, the robot lifts the level box straight upward at "
    "a slow constant speed while maintaining continuous contact. The table, box "
    "shape, lighting, and camera remain stable throughout. Photorealistic, fluid "
    "mechanical motion in one continuous shot."
)
KUBERNETES_NAME = re.compile(r"^[a-z0-9](?:[-a-z0-9]{0,61}[a-z0-9])?$")
MP4_FILENAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*\.mp4$")


def load_stack_outputs(stack_name: str, region: str) -> dict[str, str]:
    command = [
        "aws",
        "cloudformation",
        "describe-stacks",
        "--stack-name",
        stack_name,
        "--region",
        region,
        "--output",
        "json",
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        document = json.loads(result.stdout)
        rows = document["Stacks"][0]["Outputs"]
    except FileNotFoundError as exc:
        raise RuntimeError("AWS CLI is not installed or is not on PATH") from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() or "AWS CLI returned an error"
        raise RuntimeError(detail) from exc
    except (json.JSONDecodeError, KeyError, IndexError) as exc:
        raise RuntimeError("CloudFormation returned an unexpected response") from exc
    return {row["OutputKey"]: row["OutputValue"] for row in rows}


def validate_outputs(outputs: dict[str, str]) -> None:
    missing = [name for name in REQUIRED_OUTPUTS if not outputs.get(name)]
    if missing:
        raise ValueError("missing stack outputs: " + ", ".join(missing))
    actual = outputs["Cosmos3ArchiveSha256"].lower()
    if actual != EXPECTED_ARCHIVE_SHA256:
        raise ValueError(
            "Cosmos3ArchiveSha256 is stale. Expected "
            f"{EXPECTED_ARCHIVE_SHA256}, got {actual}. Ask the facilitator to update "
            "the participant CloudFormation stack before submitting a Job."
        )


def validate_run_settings(
    job_name: str,
    output_filename: str,
    reference_filename: str,
    prompt: str,
    seed: int,
) -> None:
    if not KUBERNETES_NAME.fullmatch(job_name):
        raise ValueError(
            "job name must be a lowercase Kubernetes DNS label of at most 63 characters"
        )
    for label, filename in (
        ("output filename", output_filename),
        ("reference filename", reference_filename),
    ):
        if not MP4_FILENAME.fullmatch(filename):
            raise ValueError(f"{label} must be a simple .mp4 filename")
    if not prompt.strip():
        raise ValueError("prompt must not be empty")
    if not 0 <= seed <= 4_294_967_295:
        raise ValueError("seed must be between 0 and 4294967295")


def render_prompt_block(prompt: str) -> str:
    lines = [line.strip() for line in prompt.strip().splitlines() if line.strip()]
    return "\n".join(f"                {line}" for line in lines) + "\n"


def render_job(
    source: str,
    region: str,
    outputs: dict[str, str],
    *,
    prompt: str = DEFAULT_PROMPT,
    job_name: str = "cosmos3-g1-generate",
    output_filename: str = "g1_pick_blue_box_generated.mp4",
    reference_filename: str = "g1_pick_blue_box_reference.mp4",
    seed: int = 100,
) -> str:
    validate_outputs(outputs)
    validate_run_settings(
        job_name, output_filename, reference_filename, prompt, seed
    )
    bucket = outputs["DataBucketName"]
    image = outputs["CosmosRuntimeImageUri"]
    archive_uri = outputs["Cosmos3ArchiveS3Uri"]
    archive_sha = outputs["Cosmos3ArchiveSha256"].lower()

    replacements = {
        "image: vllm/vllm-omni:cosmos3": f"image: {image}",
        "cosmos3-vllm-omni": job_name,
        'value: "super"': 'value: "nano"',
        "activeDeadlineSeconds: 2700": "activeDeadlineSeconds: 3600",
        "groot-data/ur3/dataset/videos/chunk-000/observation.images.wrist":
            "teleoperation-data/pick_place_data/videos/chunk-000/observation.images.cam_high",
        "cosmos-samples": "cosmos3-generated",
        "augmented_episode_000000_eks_selfcontained.mp4": output_filename,
        "original_episode_000000_eks_selfcontained.mp4": reference_filename,
        '- { name: SEED, value: "100" }':
            f'- {{ name: SEED, value: "{seed}" }}',
        'nvidia.com/gpu: "8"': 'nvidia.com/gpu: "1"',
        '- { name: REQUIRE_LOCAL_WEIGHTS, value: "false" }':
            '- { name: REQUIRE_LOCAL_WEIGHTS, value: "true" }\n'
            '            - { name: HF_HUB_OFFLINE, value: "1" }\n'
            '            - { name: TRANSFORMERS_OFFLINE, value: "1" }',
        '- { name: COSMOS_ARCHIVE_S3_URI, value: "" }':
            f'- {{ name: COSMOS_ARCHIVE_S3_URI, value: "{archive_uri}" }}',
        '- { name: COSMOS_ARCHIVE_SHA256, value: "" }':
            f'- {{ name: COSMOS_ARCHIVE_SHA256, value: "{archive_sha}" }}',
    }
    text = source
    for old, new in replacements.items():
        if old not in text:
            raise ValueError(f"expected manifest setting not found: {old}")
        text = text.replace(old, new)

    text = text.replace("<REGION>", region)
    text = text.replace("<DATASETS_BUCKET>", bucket)
    text = text.replace("<DATA_BUCKET>", bucket)

    text, count = re.subn(
        r"(?ms)(^\s*- name: PROMPT\s*\n\s*value: >-\s*\n).*?(?=^\s*# GPU count)",
        lambda match: match.group(1) + render_prompt_block(prompt),
        text,
    )
    if count != 1:
        raise ValueError(f"expected one PROMPT block; found {count}")

    prompt_lines = [line.strip() for line in prompt.strip().splitlines() if line.strip()]
    checks = {
        "Job name": f"name: {job_name}" in text and f"app: {job_name}" in text,
        "Nano model": 'value: "nano"' in text,
        "runtime image": f"image: {image}" in text,
        "one GPU": text.count('nvidia.com/gpu: "1"') == 2,
        "Region": f'name: AWS_REGION, value: "{region}"' in text,
        "G1 input": "observation.images.cam_high/episode_000000.mp4" in text,
        "S3 output": f"s3://{bucket}/cosmos3-generated" in text,
        "output filename": f'name: OUTPUT_FILENAME, value: "{output_filename}"' in text,
        "reference filename":
            f'name: REFERENCE_FILENAME, value: "{reference_filename}"' in text,
        "seed": f'name: SEED, value: "{seed}"' in text,
        "prompt": all(line in text for line in prompt_lines),
        "negative prompt": "negative_prompt=" in text and "slipping grasp" in text,
        "local weights": 'name: REQUIRE_LOCAL_WEIGHTS, value: "true"' in text,
        "archive URI": f'name: COSMOS_ARCHIVE_S3_URI, value: "{archive_uri}"' in text,
        "archive checksum": f'name: COSMOS_ARCHIVE_SHA256, value: "{archive_sha}"' in text,
        "Hugging Face offline mode": 'name: HF_HUB_OFFLINE, value: "1"' in text,
        "Transformers offline mode": 'name: TRANSFORMERS_OFFLINE, value: "1"' in text,
        "Qwen3Guard": "models--Qwen--Qwen3Guard-Gen-0.6B" in text,
        "AV1 normalization": "-c:v libx264" in text,
        "no placeholders": not re.search(
            r"<(?:REGION|DATASETS_BUCKET|DATA_BUCKET)>", text
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError("manifest validation failed: " + ", ".join(failed))
    return text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a validated Cosmos 3 Nano/G1 workshop Job"
    )
    parser.add_argument("--region", required=True, help="AWS Region containing the stack")
    parser.add_argument("--stack-name", default="physical-ai-toolchain")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--job-name", default="cosmos3-g1-generate")
    parser.add_argument("--output-filename", default="g1_pick_blue_box_generated.mp4")
    parser.add_argument(
        "--reference-filename", default="g1_pick_blue_box_reference.mp4"
    )
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(__file__).with_name("cosmos3-job.yaml"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("cosmos3-g1-generate-job.yaml")
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        outputs = load_stack_outputs(args.stack_name, args.region)
        source = args.source.read_text()
        rendered = render_job(
            source,
            args.region,
            outputs,
            prompt=args.prompt,
            job_name=args.job_name,
            output_filename=args.output_filename,
            reference_filename=args.reference_filename,
            seed=args.seed,
        )
        if args.output.resolve() == args.source.resolve():
            raise ValueError("output must not overwrite the reference manifest")
        args.output.write_text(rendered)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"FAIL - {exc}", file=sys.stderr)
        return 1

    print(f"PASS - Cluster: {outputs['EksClusterName']}")
    print(f"PASS - Data bucket: {outputs['DataBucketName']}")
    print(f"PASS - Runtime bundle SHA-256: {outputs['Cosmos3ArchiveSha256']}")
    print(f"PASS - Job name: {args.job_name}")
    print(f"PASS - Generated object: {args.output_filename}")
    print(f"PASS - Wrote validated Job: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
