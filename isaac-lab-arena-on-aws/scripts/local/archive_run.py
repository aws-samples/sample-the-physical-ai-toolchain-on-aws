#!/usr/bin/env python3
"""Retain verified local evidence in S3 before optionally removing its stopped containers."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import boto3

EVIDENCE_FILES = (
    "manifest.json", "parameters.json", "execution-parameters.json",
    "status.json", "execution.json", "steps.json",
    "negative-control.json", "verified-receipt.json", "independent-verification.json",
    "independent-s3-proof.json", "container-exits.json", "systemd-exit.json",
    "timings.json", "training-summary.json", "training-metrics.csv",
    "evaluation-summary.json", "scratch-mounts.json", "disk-usage.json",
    "pipeline-definition.redacted.json",
    "run.log", "train.log", "eval.log", "validate.log",
)
OPTIONAL_EVIDENCE_FILES = ("negative-validation.json", "negative-validation.log", "disk-checks.json")


def remove_verified_containers(root, manifest, ids):
    """Reconcile a prior removal without treating an unexplained absence as success."""
    path = root / "container-cleanup.json"
    receipt = json.loads(path.read_text()) if path.exists() else {
        "run_id": manifest["run_id"], "canonical_commit": manifest["canonical_commit"],
        "ids": ids, "removal_requested": [], "removed": [],
    }
    if (receipt["run_id"], receipt["canonical_commit"], receipt["ids"]) != (
            manifest["run_id"], manifest["canonical_commit"], ids):
        raise RuntimeError("Container cleanup receipt belongs to a different run or container set")

    def save():
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(receipt, indent=2) + "\n")
        temporary.replace(path)

    def inspect(container_id):
        result = subprocess.run(["docker", "inspect", container_id],
                                capture_output=True, text=True, timeout=30)
        if result.returncode:
            if "no such object:" in result.stderr.lower() or "no such container:" in result.stderr.lower():
                return None
            raise RuntimeError(f"Could not inspect container: {result.stderr}")
        return json.loads(result.stdout)[0]

    # Validate the whole set before deleting any currently present container.
    present = []
    for container_id in ids:
        container = inspect(container_id)
        if container is None:
            if container_id not in receipt["removal_requested"]:
                raise RuntimeError("Container is missing without a recorded removal request")
            continue
        if container["Config"].get("Labels", {}).get("vla.local.run") != manifest["run_id"]:
            raise RuntimeError("Container does not belong to this run")
        if container["State"]["Running"] or container["State"]["ExitCode"] != 0:
            raise RuntimeError("Container is active or did not exit successfully")
        present.append(container_id)
    for container_id in present:
        if container_id not in receipt["removal_requested"]:
            receipt["removal_requested"].append(container_id)
        save()
        subprocess.run(["docker", "rm", container_id], check=True, capture_output=True, timeout=60)
    for container_id in ids:
        if inspect(container_id) is not None:
            raise RuntimeError("Container still exists after removal")
    receipt["removed"] = ids
    save()
    return ids


def archive_run(root, remove_containers=False):
    def load(name):
        return json.loads((root / name).read_text())

    manifest = load("manifest.json")
    verification = load("independent-verification.json")
    if load("status.json")["status"] != "Succeeded" or verification["status"] != "Succeeded":
        raise RuntimeError("Archive/removal requires an independently verified successful run")
    if verification["canonical_commit"] != manifest["canonical_commit"]:
        raise RuntimeError("Verification refers to a different code revision")
    if verification["run_id"] != manifest["run_id"]:
        raise RuntimeError("Verification refers to a different run")
    session = boto3.Session(region_name=manifest["region"])
    caller = session.client("sts").get_caller_identity()
    if caller["Account"] != manifest["caller"]["Account"]:
        raise RuntimeError("Archive identity is in a different AWS account")
    s3 = session.client("s3")
    bucket = manifest["development_bucket"]
    records = []

    def retain(name, data):
        digest = hashlib.sha256(data).hexdigest()
        key = f"localdev/run-evidence/{manifest['run_id']}/{digest}/{name}"
        created = s3.put_object(Bucket=bucket, Key=key, Body=data)
        version = created.get("VersionId")
        if version in (None, "", "null"):
            raise RuntimeError(f"Evidence upload has no real S3 version: {name}")
        stored = s3.get_object(Bucket=bucket, Key=key, VersionId=version)
        if hashlib.sha256(stored["Body"].read()).hexdigest() != digest:
            raise RuntimeError(f"Evidence read-back mismatch: {name}")
        return {
            "name": name, "sha256": digest, "bytes": len(data),
            "uri": f"s3://{bucket}/{key}", "version_id": version,
        }

    required_steps = manifest.get("requested_steps", ["FineTune", "SimEval", "Validate", "SuccessGate"])
    kinds = {kind for step, kind in (("FineTune", "train"), ("SimEval", "eval"),
                                    ("Validate", "validate")) if step in required_steps}
    files = [name for name in EVIDENCE_FILES
             if (name not in {"train.log", "eval.log", "validate.log"} or name[:-4] in kinds)
             and (name != "verified-receipt.json" or "Validate" in required_steps)]
    if verification.get("contract_checks"):
        if load("contract-checks.json") != verification["contract_checks"]:
            raise RuntimeError("Named check report differs from the independently verified result")
        files.append("contract-checks.json")
    if manifest.get("resolved_request_sha256"):
        files.extend(["resolved-request.json", "job-outputs.json"])
        if "SimEval" in required_steps and "Validate" not in required_steps:
            files.append("raw-evaluation.json")
    for name in (*files, *OPTIONAL_EVIDENCE_FILES):
        path = root / name
        if not path.exists():
            if name in OPTIONAL_EVIDENCE_FILES:
                continue
            if name in {"scratch-mounts.json", "disk-usage.json"} and not manifest.get("scratch"):
                continue
            raise RuntimeError(f"Required evidence is absent: {path}")
        records.append(retain(name, path.read_bytes()))
    archive = {"run_id": manifest["run_id"], "status": "ArchivedAndReadBack", "files": records}
    archive["index"] = retain("archive-index.json", (json.dumps(archive, indent=2) + "\n").encode())
    (root / "archive-manifest.json").write_text(json.dumps(archive, indent=2) + "\n")
    if remove_containers:
        ids = [row["id"] for row in load("container-exits.json")]
        if len(ids) != len(kinds):
            raise RuntimeError("Expected exactly the containers for the verified requested steps")
        archive["removed_containers"] = remove_verified_containers(root, manifest, ids)
        (root / "archive-manifest.json").write_text(json.dumps(archive, indent=2) + "\n")
    return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--remove-containers", action="store_true")
    args = parser.parse_args()
    result = archive_run(args.run_dir.resolve(), args.remove_containers)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
