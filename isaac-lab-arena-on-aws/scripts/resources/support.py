"""Shared utilities for recorded commands and Terraform-owned resource inventories.

The environment CLI uses these alongside normal component operations.
Notebook cells invoke the CLI rather than implementing cloud operations.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time
import uuid


def save(path, value):
    save_bytes(path, (json.dumps(value, indent=2, default=str) + "\n").encode())


def save_bytes(path, body):
    """Replace a complete file atomically, without sharing another writer's temporary file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_bytes(body)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load(path):
    return json.loads(Path(path).read_text())


def run(arguments, *, repo, logs, name, env=None):
    """Run the displayed public command and retain stdout/stderr and its receipt."""
    arguments = list(map(str, arguments))
    logs = Path(logs)
    logs.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    prefix = logs / f"{name}-{stamp}"
    print("$", shlex.join(arguments), flush=True)
    receipt = {"command": arguments, "cwd": str(repo), "started_at": stamp,
               "log": str(prefix.with_suffix(".log"))}
    with prefix.with_suffix(".log").open("w") as log:
        process = subprocess.Popen(arguments, cwd=repo, env=env, text=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        receipt["pid"] = process.pid
        save(prefix.with_suffix(".json"), receipt)
        output = []
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
                output.append(line)
            code = process.wait()
        except KeyboardInterrupt:
            import signal
            process.send_signal(signal.SIGINT)
            # Terraform must get time to preserve its state. Do not kill it blindly.
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                print("Process still exiting:", process.pid, "Inspect it before retrying.")
            raise
        finally:
            receipt.update(exit_code=process.poll(),
                           finished_at=datetime.now(timezone.utc).isoformat())
            save(prefix.with_suffix(".json"), receipt)
    if code:
        raise subprocess.CalledProcessError(code, arguments, output="".join(output))
    return "".join(output)


def put_version(s3, account, bucket, key, body):
    checksum = hashlib.sha256(body).hexdigest()
    reply = s3.put_object(Bucket=bucket, Key=key, Body=body,
                          ExpectedBucketOwner=account, ServerSideEncryption="AES256")
    version = reply.get("VersionId")
    if version in (None, "", "null"):
        raise ValueError("Runbook inputs and evidence require S3 versioning.")
    response = s3.get_object(Bucket=bucket, Key=key, VersionId=version,
                             ExpectedBucketOwner=account)
    with response["Body"] as stream:
        if hashlib.sha256(stream.read()).hexdigest() != checksum:
            raise ValueError("S3 readback differs from the uploaded input.")
    return {"bucket": bucket, "key": key, "version": version, "sha256": checksum}


def get_version(s3, account, reference, limit=16 * 1024 * 1024):
    if reference.get("version") in (None, "", "null"):
        raise ValueError("A real S3 version is required.")
    response = s3.get_object(Bucket=reference["bucket"], Key=reference["key"],
                             VersionId=reference["version"], ExpectedBucketOwner=account)
    with response["Body"] as stream:
        body = stream.read(limit + 1)
    if len(body) > limit or hashlib.sha256(body).hexdigest() != reference["sha256"]:
        raise ValueError("Versioned input failed size/hash verification.")
    return body


def ssm_command(ssm, request, receipt):
    """Record the intent before submission; never blindly repeat an uncertain request."""
    receipt = Path(receipt)
    if receipt.exists():
        saved = load(receipt)
        if saved["request"] != request:
            raise ValueError("This receipt belongs to another request; preserve it.")
        if saved.get("command_id"):
            return saved["command_id"]
        # The comment is unique to this recorded request. Reconcile a lost reply.
        matches = []
        for page in ssm.get_paginator("list_commands").paginate():
            matches.extend(c for c in page["Commands"]
                           if c.get("Comment") == request["Comment"]
                           and c.get("DocumentName") == request["DocumentName"]
                           and c.get("InstanceIds") == request["InstanceIds"]
                           and c.get("Parameters") == request["Parameters"])
        if len(matches) != 1:
            raise ValueError("Resolve this uncertain SSM submission before retrying: " + str(receipt))
        saved["command_id"] = matches[0]["CommandId"]
        save(receipt, saved)
        return saved["command_id"]
    saved = {"request": request, "started_at": datetime.now(timezone.utc).isoformat()}
    save(receipt, saved)
    response = ssm.send_command(**request)
    saved["command_id"] = response["Command"]["CommandId"]
    save(receipt, saved)
    return saved["command_id"]


def wait_ssm(ssm, command, instance, output, seconds=3600):
    """Wait for setup/inspection while retaining the SSM result for diagnosis."""
    deadline = time.monotonic() + seconds
    while True:
        try:
            result = ssm.get_command_invocation(CommandId=command, InstanceId=instance)
        except ssm.exceptions.InvocationDoesNotExist:
            result = {"Status": "Pending"}
        save(output, result)
        print(command, result["Status"], flush=True)
        if result["Status"] not in {"Pending", "InProgress", "Delayed", "Cancelling"}:
            if result["Status"] != "Success" or result.get("ResponseCode") != 0:
                raise RuntimeError("SSM operation failed; see " + str(output) + "\n" +
                                   result.get("StandardErrorContent", "")[-3000:])
            return result
        if time.monotonic() >= deadline:
            raise TimeoutError("Observer timed out. Inspect/reconnect to the saved SSM ID.")
        time.sleep(min(30, max(0, deadline - time.monotonic())))


def terraform_inventory(states):
    result = {}
    for name, directory in states.items():
        path = Path(directory) / "terraform.tfstate"
        if not path.exists():
            continue
        state = load(path)
        resources = []
        for resource in state.get("resources", []):
            if resource["mode"] != "managed":
                continue
            for item in resource["instances"]:
                address = resource["type"] + "." + resource["name"]
                if resource.get("module"):
                    address = resource["module"] + "." + address
                if "index_key" in item:
                    address += "[" + json.dumps(item["index_key"]) + "]"
                resources.append({"address": address, "type": resource["type"],
                                  "attributes": item["attributes"]})
        result[name] = {"directory": str(Path(directory).resolve()),
                        "lineage": state["lineage"], "resources": resources}
    return result


def review_destroy_plan(plan, recorded):
    """Reject replacement/create operations or resources outside this recorded state."""
    known = {r["address"]: r["attributes"] for r in recorded["resources"]}
    removals = []
    for resource in plan.get("resource_changes", []):
        if resource["mode"] != "managed":
            continue
        actions = resource["change"]["actions"]
        if actions == ["no-op"]:
            continue
        if actions != ["delete"] or resource["address"] not in known:
            raise ValueError("Destruction plan exceeds the recorded ownership: " + resource["address"])
        before = resource["change"]["before"]
        if before.get("id") != known[resource["address"]].get("id"):
            raise ValueError("Resource identity changed since inventory: " + resource["address"])
        removals.append(resource["address"])
    return removals


def exact_version_removals(current, archived):
    """Allow a partial deletion retry, but never a version added after the archive."""
    expected = {(item["key"], item["version"]): item
                for item in archived["objects"]}
    markers = {(item["key"], item["version"]) for item in archived["delete_markers"]}
    removals = []
    for item in current["objects"]:
        key = (item["key"], item["version"])
        if key not in expected or any(item[field] != expected[key][field]
                                      for field in ("bytes", "etag")):
            raise ValueError("Unarchived source version remains; preserve and archive it first.")
        removals.append({"Key": key[0], "VersionId": key[1]})
    for item in current["delete_markers"]:
        key = (item["key"], item["version"])
        if key not in markers:
            raise ValueError("An unrecorded delete marker appeared after archival.")
        removals.append({"Key": key[0], "VersionId": key[1]})
    return removals
