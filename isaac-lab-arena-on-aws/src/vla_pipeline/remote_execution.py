"""Carry ordinary local CLI requests to an explicitly prepared EC2 host."""
from __future__ import annotations

import json
import hashlib
import shlex
import time
import uuid

from .deployment import aws_session
from .host import download_command, put_verified
from .operations import timestamp
from . import remote_commands


def apply_host_observations(record, remote):
    """Carry verified host results while preserving the initiating profile and request."""
    for key in ("status", "unit", "run_dir", "activity", "failure_reason", "receipt_uri",
                "elapsed_seconds", "requested_steps", "complete_training_workflow",
                "independently_verified", "verification_status", "verification_scope",
                "contract_checks", "outputs", "cancellation_requested", "cleanup",
                 "service_state", "last_log_age_seconds", "episodes", "success_rate", "evidence_archive"):
        if key in remote:
            record[key] = remote[key]


def exchange(record, action, store, *, wait=True):
    with store.lock("runs", record["id"]):
        current = store.load("runs", record["id"])
        record.clear()
        record.update(current)
        return _exchange(record, action, store, wait=wait)


def _exchange(record, action, store, *, wait):
    target = record["request"]["local_target"]
    session = aws_session(record.get("profile"), record["region"])
    ssm = session.client("ssm", region_name=target["host_region"])
    def save():
        store.save("runs", record)

    actions = record.setdefault("remote_actions", {})
    entry = actions.get(action)
    # Mutating run/stop and expensive verification retain their original command.
    # Read-only status gets a new observation once its last request has completed.
    terminal_failure = entry and entry.get("status") in {
        "Failed", "TimedOut", "Cancelled", "DeliveryTimedOut", "ExecutionTimedOut",
    }
    if entry is None or (action in {"status", "cleanup"} and
                         (entry.get("response_received") or terminal_failure)):
        if terminal_failure:
            record.setdefault("remote_action_history", []).append({"action": action, **entry})
        token = uuid.uuid4().hex
        bucket = target["development_bucket"]
        key = f"localdev/cli/{record['deployment']}/runs/{record['id']}/{token}"
        entry = {"operation_token": token, "response": {"bucket": bucket, "key": key + "/response.json"}}
        payload = {"action": action, "deployment": target["deployment_id"], "run_id": record["id"],
                   "source_commit": record["source_commit"], "operation_token": token,
                   "response": entry["response"], "request": record["request"]}
        entry["payload"] = put_verified(session, bucket, key + "/request.json",
                                        (json.dumps(payload) + "\n").encode())
        actions[action] = entry
        save()
    destination = f"{target['state_dir']}/transport/{entry['operation_token']}.json"
    command = "\n".join([
        "set -euo pipefail", "umask 077",
        "unset AWS_PROFILE AWS_DEFAULT_PROFILE AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN AWS_SECURITY_TOKEN",
        "export AWS_DEFAULT_REGION=us-east-1 AWS_REGION=us-east-1",
        shlex.join(["install", "-d", "-m", "700", target["state_dir"] + "/transport"]),
        download_command(entry["payload"], destination),
        shlex.join([target["component"] + "/.venv/bin/python", "-m", "vla_pipeline.remote_worker",
                    "exchange", "--state-dir", target["state_dir"], "--payload", destination]),
    ])
    remote_commands.submit(ssm, target["instance_id"], entry,
                           ["bash -c " + shlex.quote(command)], save,
                            timeout=43200 if action == "verify" else 3600 if action in {"cleanup", "report"} else 600,
                           bucket=target["development_bucket"])
    while True:
        observed = remote_commands.observe(ssm, target["instance_id"], entry)
        if observed["Status"] not in remote_commands.ACTIVE:
            save()
            break
        record["transport_activity"] = f"Host {action}: SSM {entry['command_id']} {observed['Status']}"
        if action == "run":
            record.update(status="Submitting", failure_reason=None)
        save()
        if not wait:
            return record
        time.sleep(5)
    if observed["Status"] != "Success":
        raise RuntimeError(
            f"Host {action} command {entry['command_id']} {observed['Status']}: "
            f"{observed.get('StandardErrorContent', '')}. Inspect the saved SSM output; "
            "an interrupted submission may already have started the run.")
    response = entry["response"]
    s3 = session.client("s3")
    acknowledgments = []
    for line in observed.get("StandardOutputContent", "").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "response_sha256" in value:
            acknowledgments.append(value)
    if len(acknowledgments) != 1:
        raise ValueError("SSM did not return one response identity; inspect its recorded output")
    acknowledgment = acknowledgments[0]
    version = acknowledgment.get("response_version")
    if version in (None, "", "null"):
        raise ValueError("Host response is unversioned")
    head = s3.head_object(Bucket=response["bucket"], Key=response["key"], VersionId=version)
    if version in (None, "", "null") or head["ContentLength"] > 4 * 1024 * 1024:
        raise ValueError("Host response must be a bounded, versioned JSON object")
    stream = s3.get_object(Bucket=response["bucket"], Key=response["key"], VersionId=version)["Body"]
    try:
        body = stream.read(4 * 1024 * 1024 + 1)
    finally:
        stream.close()
    if len(body) != head["ContentLength"] or hashlib.sha256(body).hexdigest() != acknowledgment["response_sha256"]:
        raise ValueError("Downloaded host response differs from its SSM acknowledgment")
    result = json.loads(body)
    if (result.get("operation_token"), result.get("run_id"), result.get("source_commit")) != (
        entry["operation_token"], record["id"], record["source_commit"],
    ):
        raise ValueError("Host response does not belong to this request")
    entry.update(response_received=timestamp(), response_version=version)
    save()
    if not result["ok"]:
        raise RuntimeError(result["error"])
    remote = result["record"]
    if (remote["source_commit"] != record["source_commit"]
            or remote["request"]["parameters"] != record["request"]["parameters"]
            or remote["request"]["steps"] != record["request"]["steps"]):
        raise ValueError("Host result differs from the submitted request")
    # Keep the initiating profile, paths and transport request; copy observations.
    apply_host_observations(record, remote)
    record.pop("transport_activity", None)
    save()
    return record
