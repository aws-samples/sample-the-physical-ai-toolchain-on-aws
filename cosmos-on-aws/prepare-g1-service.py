#!/usr/bin/env python3
"""Render the private reusable Cosmos 3 Nano/G1 workshop service."""

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
    "CosmosRuntimeImageUri",
    "Cosmos3ArchiveS3Uri",
    "Cosmos3ArchiveSha256",
)
KUBERNETES_NAME = re.compile(r"^[a-z0-9](?:[-a-z0-9]{0,61}[a-z0-9])?$")


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
        rows = json.loads(result.stdout)["Stacks"][0]["Outputs"]
    except FileNotFoundError as exc:
        raise RuntimeError("AWS CLI is not installed or is not on PATH") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(exc.stderr.strip() or "AWS CLI returned an error") from exc
    except (json.JSONDecodeError, KeyError, IndexError) as exc:
        raise RuntimeError("CloudFormation returned an unexpected response") from exc
    return {row["OutputKey"]: row["OutputValue"] for row in rows}


def validate_inputs(
    outputs: dict[str, str], deployment_name: str, service_name: str
) -> None:
    missing = [name for name in REQUIRED_OUTPUTS if not outputs.get(name)]
    if missing:
        raise ValueError("missing stack outputs: " + ", ".join(missing))
    actual = outputs["Cosmos3ArchiveSha256"].lower()
    if actual != EXPECTED_ARCHIVE_SHA256:
        raise ValueError(
            "Cosmos3ArchiveSha256 is stale. Expected "
            f"{EXPECTED_ARCHIVE_SHA256}, got {actual}."
        )
    for label, name in (
        ("deployment name", deployment_name),
        ("service name", service_name),
    ):
        if not KUBERNETES_NAME.fullmatch(name):
            raise ValueError(
                f"{label} must be a lowercase Kubernetes DNS label of at most 63 characters"
            )


def render_service(
    source: str,
    region: str,
    outputs: dict[str, str],
    *,
    deployment_name: str = "cosmos3-g1-server",
    service_name: str = "cosmos3-g1-server",
) -> str:
    validate_inputs(outputs, deployment_name, service_name)
    image = outputs["CosmosRuntimeImageUri"]
    archive_uri = outputs["Cosmos3ArchiveS3Uri"]
    archive_sha = outputs["Cosmos3ArchiveSha256"].lower()

    replacements = {
        "image: vllm/vllm-omni:cosmos3": f"image: {image}",
        "cosmos3-vllm-omni": deployment_name,
        'value: "super"': 'value: "nano"',
        'nvidia.com/gpu: "8"': 'nvidia.com/gpu: "1"',
        '- { name: REQUIRE_LOCAL_WEIGHTS, value: "false" }':
            '- { name: REQUIRE_LOCAL_WEIGHTS, value: "true" }\n'
            '            - { name: HF_HUB_OFFLINE, value: "1" }\n'
            '            - { name: TRANSFORMERS_OFFLINE, value: "1" }',
        '- { name: COSMOS_ARCHIVE_S3_URI, value: "" }':
            f'- {{ name: COSMOS_ARCHIVE_S3_URI, value: "{archive_uri}" }}',
        '- { name: COSMOS_ARCHIVE_SHA256, value: "" }':
            f'- {{ name: COSMOS_ARCHIVE_SHA256, value: "{archive_sha}" }}',
        'value: "<REGION>"': f'value: "{region}"',
    }
    text = source
    for old, new in replacements.items():
        if old not in text:
            raise ValueError(f"expected service setting not found: {old}")
        text = text.replace(old, new)

    # Deployment and Service use the same placeholder in the base manifest. If a
    # distinct Service name is requested, change only the second document metadata.
    marker = "kind: Service\nmetadata:\n  name: " + deployment_name
    if marker not in text:
        raise ValueError("rendered Service metadata was not found")
    text = text.replace(
        marker,
        "kind: Service\nmetadata:\n  name: " + service_name,
        1,
    )

    checks = {
        "Deployment": "kind: Deployment" in text,
        "one replica": "replicas: 1" in text,
        "Recreate strategy": "type: Recreate" in text,
        "runtime image": f"image: {image}" in text,
        "one GPU": text.count('nvidia.com/gpu: "1"') == 2,
        "Nano model": 'name: COSMOS_MODEL, value: "nano"' in text,
        "archive URI": f'name: COSMOS_ARCHIVE_S3_URI, value: "{archive_uri}"' in text,
        "archive checksum": f'name: COSMOS_ARCHIVE_SHA256, value: "{archive_sha}"' in text,
        "offline mode": 'name: HF_HUB_OFFLINE, value: "1"' in text
        and 'name: TRANSFORMERS_OFFLINE, value: "1"' in text,
        "startup probe": "startupProbe:" in text and "path: /v1/models" in text,
        "Service": "kind: Service" in text and f"name: {service_name}" in text,
        "ClusterIP only": "type: ClusterIP" in text
        and "type: LoadBalancer" not in text
        and "type: NodePort" not in text,
        "port 8000": "port: 8000" in text and "containerPort: 8000" in text,
        "no placeholders": "<REGION>" not in text,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError("service validation failed: " + ", ".join(failed))
    return text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the private reusable Cosmos 3 Nano workshop service"
    )
    parser.add_argument("--region", required=True)
    parser.add_argument("--stack-name", default="physical-ai-toolchain")
    parser.add_argument("--deployment-name", default="cosmos3-g1-server")
    parser.add_argument("--service-name", default="cosmos3-g1-server")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(__file__).with_name("cosmos3-service.yaml"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("cosmos3-g1-service.yaml")
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        outputs = load_stack_outputs(args.stack_name, args.region)
        rendered = render_service(
            args.source.read_text(),
            args.region,
            outputs,
            deployment_name=args.deployment_name,
            service_name=args.service_name,
        )
        if args.output.resolve() == args.source.resolve():
            raise ValueError("output must not overwrite the reference manifest")
        args.output.write_text(rendered)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"FAIL - {exc}", file=sys.stderr)
        return 1

    print(f"PASS - Cluster: {outputs['EksClusterName']}")
    print(f"PASS - Runtime bundle SHA-256: {outputs['Cosmos3ArchiveSha256']}")
    print(f"PASS - Deployment: {args.deployment_name}")
    print(f"PASS - ClusterIP Service: {args.service_name}")
    print(f"PASS - Wrote validated service: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
