"""Small persistent records for CLI deployments and executions."""
from __future__ import annotations

import datetime as dt
import contextlib
import json
import os
import re
import shlex
import uuid
from pathlib import Path
from .cli_context import command_name


def timestamp():
    return dt.datetime.now(dt.timezone.utc).isoformat()


class OperationBusy(ValueError):
    """Another process currently owns this operation's record."""


def validate_name(name):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]{0,39}", name):
        raise ValueError("Names must contain 1–40 letters, digits or hyphens")
    return name


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with open(temporary, "x", opener=lambda p, f: os.open(p, f, 0o600)) as stream:
            json.dump(value, stream, indent=2, default=str)
            stream.write("\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class Store:
    def __init__(self, root=None):
        self.root = Path(root or os.environ.get(
            "VLA_STATE_DIR", str(Path.home() / ".local/state/vla")
        )).expanduser().resolve()

    def follow_command(self, run_id):
        command = shlex.split(command_name())
        if self.root != (Path.home() / ".local/state/vla").resolve():
            command += ["--state-dir", str(self.root)]
        return shlex.join([*command, "status", validate_name(run_id), "--follow"])

    @contextlib.contextmanager
    def lock(self, kind, name, *, wait=False):
        """Lock the complete reload/change/save operation, including observations."""
        import fcntl

        path = self.path(kind, name).parent
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
        with open(path / ".lock", "a+") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | (0 if wait else fcntl.LOCK_NB))
            except BlockingIOError as exc:
                raise OperationBusy(
                    f"Another command is coordinating {name}; use {command_name()} status"
                ) from exc
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def path(self, kind, name):
        return self.root / kind / validate_name(name) / "record.json"

    def load(self, kind, name):
        path = self.path(kind, name)
        try:
            data = json.loads(path.read_text())
        except FileNotFoundError as exc:
            raise ValueError(f"No saved {kind[:-1]} {name!r} at {path}") from exc
        if not isinstance(data, dict) or data.get("schema_version") != 1 or data.get("id") != name:
            raise ValueError(f"Unsupported or mismatched operation record: {path}")
        return data

    def save(self, kind, data):
        data.update(schema_version=1, updated_at=timestamp())
        write_json(self.path(kind, data["id"]), data)

    def create_run(self, data):
        path = self.path("runs", data["id"])
        try:
            path.parent.mkdir(parents=True, exist_ok=False, mode=0o700)
        except FileExistsError as exc:
            raise ValueError(
                f"Run {data['id']} already exists; inspect it with "
                f"{command_name()} status, or use a new ID"
            ) from exc
        self.save("runs", data)
        return path.parent

    def list(self):
        rows = []
        if self.root.exists():
            for kind in ("deployments", "runs"):
                for path in sorted((self.root / kind).glob("*/record.json")):
                    try:
                        data = self.load(kind, path.parent.name)
                    except (OSError, ValueError) as exc:
                        data = {"id": path.parent.name, "status": "UnreadableRecord",
                                "record_path": str(path), "failure_reason": str(exc)}
                    rows.append({"kind": kind[:-1], **data})
        return rows


def new_run_id():
    return dt.datetime.now(dt.timezone.utc).strftime("run-%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
