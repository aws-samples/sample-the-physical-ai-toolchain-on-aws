"""Identify committed runtime/build inputs while permitting notebook outputs and user configuration."""
from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess

COMPONENT = Path(__file__).resolve().parents[2]
REPOSITORY = COMPONENT.parent

# These paths contain the submitted code, build recipes and lifecycle helpers.
# Configured values are recorded separately with each deployment/run. Notebook
# source and outputs are retained as evidence, not mounted into GPU containers.
RUNTIME_PATHS = (
    "isaac-lab-arena-on-aws/src",
    "isaac-lab-arena-on-aws/entrypoints",
    "isaac-lab-arena-on-aws/scripts",
    "isaac-lab-arena-on-aws/config",
    "isaac-lab-arena-on-aws/docker",
    "isaac-lab-arena-on-aws/infra",
    "isaac-lab-arena-on-aws/pyproject.toml",
    "pai", "foundation/infra", "containers/isaac-lab-arena",
    "pyproject.toml",
)


def identity(repository=REPOSITORY):
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all", "--", *RUNTIME_PATHS],
        cwd=repository, text=True).strip()
    if dirty:
        raise ValueError(
            "Commit the runtime/build inputs before deployment or execution. "
            "Notebook outputs and root config.json may change; executable inputs must remain clean.\n" + dirty)
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip()


def input_digest(repository=REPOSITORY):
    paths = subprocess.check_output(
        ["git", "ls-files", "-z", "--", *RUNTIME_PATHS], cwd=repository).decode().split("\0")
    digest = hashlib.sha256()
    for name in sorted(filter(None, paths)):
        digest.update(name.encode() + b"\0" + (Path(repository) / name).read_bytes())
    return digest.hexdigest()
