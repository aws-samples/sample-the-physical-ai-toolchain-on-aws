"""Read the toolchain config and check local prerequisites without AWS calls."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


def check_setup(repo_root, *, tools=True):
    """Return the Arena section using the existing pai.config loader."""
    repo_root = Path(repo_root).resolve()
    errors = []

    def check(label, passed, detail=""):
        print(f"{'PASS' if passed else 'FAIL'}  {label}" + (f": {detail}" if detail else ""))
        if not passed:
            errors.append(label)

    commands = {
        "Python 3.11": ["python3.11", "--version"],
        "Git": ["git", "--version"],
        "Git LFS": ["git", "lfs", "version"],
        "AWS CLI": ["aws", "--version"],
        "Terraform": ["terraform", "version", "-json"],
        "Bash": ["bash", "--version"],
    }
    for label, command in commands.items() if tools else []:
        if not shutil.which(command[0]):
            check(label, False, "not installed or not on PATH")
            continue
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        valid = result.returncode == 0
        if label == "Python 3.11":
            valid = valid and result.stdout.startswith("Python 3.11.")
        if label == "AWS CLI":
            valid = valid and result.stdout.startswith("aws-cli/2.")
        if label == "Terraform" and valid:
            try:
                version = json.loads(result.stdout)["terraform_version"]
                valid = tuple(map(int, version.split(".")[:2])) >= (1, 9)
            except (ValueError, KeyError, TypeError):
                valid = False
        check(label, valid, "available" if valid else "install/fix the required version")

    sys.path.insert(0, str(repo_root))
    from pai.config import CONFIG_PATH, load

    if CONFIG_PATH.resolve() != repo_root / "config.json":
        raise RuntimeError("Another clone's pai.config is loaded; restart this notebook's kernel.")
    try:
        config = load()
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Fix {CONFIG_PATH}: it must exist and contain valid JSON.") from exc
    arena = config.get("arena")
    if not isinstance(arena, dict):
        raise RuntimeError(f"Add the Arena configuration section to {CONFIG_PATH}.")
    check("Shared configuration", True, str(CONFIG_PATH))
    check("arena.account_id", bool(re.fullmatch(r"\d{12}", str(arena.get("account_id", "")))),
          "must be your twelve-digit workload account")
    for key in ("hf_token_file", "ngc_token_file"):
        value = arena.get(key)
        path = Path(value).expanduser() if isinstance(value, str) and value else None
        valid = bool(path and path.is_absolute() and path.is_file()
                     and not path.resolve().is_relative_to(repo_root))
        if valid:
            try:
                size = path.stat().st_size
                valid = 0 < size <= 65536
                if valid:
                    content = path.read_text().strip()
                    valid = bool(content) and not content.startswith("{")
                    del content
            except (OSError, UnicodeError):
                valid = False
        check("arena." + key, valid, "requires a readable plaintext token file outside the clone")
    check("arena.region", arena.get("region") == "us-east-1", "current application region: us-east-1")
    check("arena.deployment_name",
          bool(re.fullmatch(r"[a-z][a-z0-9-]{1,21}", str(arena.get("deployment_name", "")))),
          "2–22 lowercase letters/digits/hyphens")
    profile = arena.get("profile")
    check("arena.profile", profile is None or isinstance(profile, str) and bool(profile.strip()),
          "null uses existing credentials; a string selects a configured AWS profile")
    prepared = arena.get("prepare_cells")
    check("arena.prepare_cells", isinstance(prepared, list) and bool(prepared)
          and all(isinstance(cell, str) and bool(cell) for cell in prepared),
          "nonempty list; supported choices are checked after CLI installation")
    for mode in ("local", "managed"):
        section = arena.get(mode)
        section = section if isinstance(section, dict) else {}
        cell = section.get("cell")
        check(f"arena.{mode}.cell", isinstance(cell, str) and isinstance(prepared, list) and cell in prepared,
              "must be included in prepare_cells")
        fields = ("instance_type",) if mode == "local" else ("train_instance_type", "eval_instance_type")
        pattern = r"[a-z0-9]+\.[a-z0-9]+" if mode == "local" else r"ml\.[a-z0-9]+\.[a-z0-9]+"
        for field in fields:
            check(f"arena.{mode}.{field}", bool(re.fullmatch(pattern, str(section.get(field, "")))),
                  "instance type format; capacity and compatibility are checked by deployment")
    run = arena.get("run")
    run = run if isinstance(run, dict) else {}
    seconds = run.get("max_runtime_seconds")
    check("arena.run.max_runtime_seconds", type(seconds) is int and 0 < seconds < 1000000,
          "positive explicit GPU-job deadline")
    recording = run.get("record_video")
    check("arena.run.record_video", recording == "auto" or type(recording) is bool,
          'use "auto", true or false')
    state = Path(os.environ.get("VLA_STATE_DIR", str(Path.home() / ".local/state/vla"))).expanduser()
    for label, path in (("Clone and client tools", repo_root), ("Operation records and evidence", state)):
        existing = next(p for p in (path, *path.parents) if p.exists())
        free_gib = shutil.disk_usage(existing).free / 1024**3
        check(label, free_gib >= 40,
              f"{path}: {free_gib:.1f} GiB free; requires at least 40 GiB")
    if errors:
        raise RuntimeError(
            "Fix the failed tools/settings before proceeding: " + ", ".join(errors)
            + f". Edit the arena section in {CONFIG_PATH}; do not put token values in it."
        )
    print("Notebook client setup passed. AWS identity, model access, permissions and quotas still require deployment preflight.")
    return arena
