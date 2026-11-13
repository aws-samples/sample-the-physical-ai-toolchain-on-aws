"""Observe a saved deployment without taking over its coordinator or submitting work."""
from __future__ import annotations

import time
import os
import socket
import sys

from .deployment import aws_session
from .remote_commands import observe
from .cli_context import command_name


def follow_deployment(store, name, *, as_json, timeout_seconds):
    from .cli import show

    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        record = store.load("deployments", name)
        session = aws_session(record.get("profile"), record["region"])
        active = []
        for entry in record.get("builds", {}).values():
            if entry.get("id") and entry.get("status") == "IN_PROGRESS":
                build = session.client("codebuild").batch_get_builds(ids=[entry["id"]])["builds"][0]
                active.append(f"{entry['id']}: {build['buildStatus']}, {build.get('currentPhase')}")
        for entry in record.get("host_commands", {}).values():
            if entry.get("command_id") and entry.get("status") in {"Pending", "InProgress", "Delayed"}:
                result = observe(session.client("ssm", region_name=record["local"]["host_region"]),
                                 record["local"]["instance_id"], dict(entry))
                active.append(f"Host command {entry['command_id']}: {result['Status']}")
        if active:
            record["activity"] = "; ".join(active)
        show(record, as_json)
        # This observer never rewrites the coordinator's record. A disconnected
        # deploy process needs an explicit --resume to advance to its next phase.
        if record["status"] in {"Ready", "Selected", "Planned"}:
            return 0
        if record["status"] in {"Failed", "Interrupted"}:
            print(f"Resume preparation with {command_name()} deploy --name {name} --resume", file=sys.stderr)
            return 1
        coordinator = record.get("coordinator", {})
        if coordinator.get("host") == socket.gethostname() and coordinator.get("pid"):
            try:
                os.kill(coordinator["pid"], 0)
            except ProcessLookupError:
                print("The local preparation coordinator has exited. Cloud work may still be running; "
                      f"continue with {command_name()} deploy --name {name} --resume.", file=sys.stderr)
                return 1
        time.sleep(min(30, max(0, deadline - time.monotonic())))
    print(f"Stopped following preparation. Inspect {command_name()} status {name}; "
          f"use {command_name()} deploy --name {name} --resume if its coordinator disconnected.",
          file=sys.stderr)
    return 2
