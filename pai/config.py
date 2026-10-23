"""Configuration loader for config.json — single source of truth.

config.json lives in the repo root and is NOT gitignored. All launchers read
region, workstation settings, and batch defaults from this file. The CLI's
config commands can show and edit it.
"""

from __future__ import annotations

import json
import os
from functools import reduce
from pathlib import Path


def _find_repo_root() -> Path:
    """Walk up from this file to find the directory containing config.json."""
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "config.json").exists():
            return current
        current = current.parent
    raise FileNotFoundError(
        "config.json not found. Run from within the aws-physical-ai-toolchain repo."
    )


REPO_ROOT = _find_repo_root()
CONFIG_PATH = REPO_ROOT / "config.json"


def load(path: str | Path | None = None) -> dict:
    """Load the shared JSON config, or an explicitly selected file."""
    with open(path if path is not None else CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


def save(config: dict) -> None:
    """Write config.json with 2-space indent."""
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
        f.write("\n")


def get(dotted_key: str, default=None):
    """Get a nested config value. E.g., get('aws.region') -> 'us-west-2'.

    Returns *default* if any key in the path is missing or None.
    """
    try:
        return reduce(lambda d, k: d[k], dotted_key.split("."), load())
    except (KeyError, TypeError):
        return default


def resolve_region() -> str:
    """Resolve region with precedence: AWS_DEFAULT_REGION env → config.aws.region → us-west-2.

    NOTE: The validated region for this toolchain is us-west-2 (the ECR/Secrets
    Manager region). cdk/bin/app.ts falls back to us-east-1 for CDK deploy; that
    divergence is intentional (CDK default vs validated runtime region). The CLI
    prioritizes the runtime region (us-west-2) for consistency with the Python
    launchers.
    """
    env_region = os.environ.get("AWS_DEFAULT_REGION")
    if env_region:
        return env_region
    config_region = get("aws.region")
    if config_region:
        return config_region
    return "us-west-2"


def batch_defaults() -> dict:
    """Return batch job defaults from config (instanceType, numNodes, maxvCpus)."""
    return {
        "instanceType": get("batch.instanceType", "g6.12xlarge"),
        "numNodes": get("batch.numNodes", 2),
        "maxvCpus": get("batch.maxvCpus", 96),
    }
