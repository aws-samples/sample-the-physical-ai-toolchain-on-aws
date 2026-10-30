"""Finish one recorded run without removing its supplied infrastructure."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time

from .deployment import aws_session
from .operations import timestamp


def _stop_local_service(unit, timeout_seconds):
    """Stop a live worker; completed transient units may already be collected."""
    def state():
        output = subprocess.check_output(
            ["systemctl", "show", unit, "--property=LoadState,ActiveState"],
            text=True, timeout=30)
        values = dict(line.split("=", 1) for line in output.splitlines() if "=" in line)
        if not values.get("LoadState") or not values.get("ActiveState"):
            raise ValueError("Could not determine the local worker service state")
        return values

    before = state()
    if before["ActiveState"] in {"inactive", "failed"}:
        return
    result = subprocess.run(["sudo", "-n", "systemctl", "stop", unit],
                            capture_output=True, text=True, timeout=timeout_seconds)
    after = state()
    if result.returncode and after["LoadState"] != "not-found":
        raise RuntimeError(f"Could not stop the local worker: {result.stderr.strip()}")
    if after["ActiveState"] not in {"inactive", "failed"}:
        raise ValueError("The local worker service is still active")


def cleanup(record, store, *, execute=False, timeout_seconds=3600):
    plan = {
        "run_id": record["id"],
        "stop": "Only this recorded execution, if it is still active",
        "remove": ("Successful local containers after archive/readback" if record["mode"] == "local"
                   else "No persistent SageMaker metadata"),
        "retain": ["Supplied host and deployment", "S3 artifacts and model registrations",
                   "Pipeline/job metadata and logs", "Failed/stopped local containers and working files"],
    }
    if not execute:
        return plan
    if timeout_seconds <= 0:
        raise ValueError("Cleanup timeout must be positive")
    if record["request"].get("transport") == "ssm":
        from .remote_execution import exchange
        return exchange(record, "cleanup", store)
    from .execution import refresh, require_execution_host, stop

    deadline = time.monotonic() + timeout_seconds
    session = aws_session(record.get("profile") if record["mode"] == "managed" else None,
                          record["region"])
    if session.client("sts").get_caller_identity()["Account"] != record["account_id"]:
        raise ValueError("Cleanup credentials select a different account")
    current = refresh(record, store)
    if current["status"] not in {"Succeeded", "Failed", "Stopped", "VerificationFailed"}:
        stop(current, store)
        while True:
            current = refresh(current, store)
            if current["status"] in {"Succeeded", "Failed", "Stopped", "VerificationFailed"}:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("Cancellation is not yet terminal; repeat cleanup for this run")
            time.sleep(min(10, max(0, deadline - time.monotonic())))
    with store.lock("runs", record["id"], wait=True):
        current = store.load("runs", record["id"])
        current["cleanup"] = {**plan, "status": "Running", "started_at": timestamp()}
        store.save("runs", current)
        try:
            if current["mode"] == "local":
                require_execution_host(current)
                from .operations import validate_name
                unit = "vla-local-" + validate_name(current["id"])
                _stop_local_service(unit, max(1, deadline - time.monotonic()))
                # The worker has stopped. Check the actual Docker state before
                # claiming that this run no longer consumes GPU compute.
                ids = subprocess.check_output([
                    "docker", "ps", "-aq", "--no-trunc",
                    "--filter", "label=vla.local.run=" + current["id"],
                ], text=True, timeout=30).split()
                for container_id in ids:
                    item = json.loads(subprocess.check_output(
                        ["docker", "inspect", container_id], text=True, timeout=30))[0]
                    if item["Config"].get("Labels", {}).get("vla.local.run") != current["id"]:
                        raise ValueError("Container ownership changed; cleanup stopped")
                    if item["State"]["Running"]:
                        subprocess.run(["docker", "stop", "--time", "15", item["Id"]],
                                       check=True, capture_output=True, timeout=60)
                active = subprocess.check_output([
                    "docker", "ps", "-q", "--filter", "label=vla.local.run=" + current["id"],
                ], text=True, timeout=30).strip()
                if active:
                    raise ValueError("Owned containers remain running")
                if current.get("independently_verified"):
                    script = Path(current["component"]) / "scripts/local/archive_run.py"
                    command = [sys.executable, str(script), "--run-dir", current["run_dir"],
                               "--remove-containers"]
                    if not Path(current["run_dir"], "manifest.json").is_file():
                        raise ValueError("Recorded local evidence directory is missing")
                    result = subprocess.run(command, text=True, capture_output=True, check=True,
                                            timeout=max(1, deadline - time.monotonic()))
                    current["cleanup"]["archive"] = json.loads(result.stdout)
                else:
                    current["cleanup"]["retained_container_ids"] = ids
            current["cleanup"].update(status="Complete", finished_at=timestamp())
        except BaseException as exc:
            current["cleanup"].update(status="Incomplete", error=f"{type(exc).__name__}: {exc}")
            store.save("runs", current)
            raise
        store.save("runs", current)
        return current
