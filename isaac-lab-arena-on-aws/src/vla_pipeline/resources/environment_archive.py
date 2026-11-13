"""Download verified evidence before removing the environment's archive bucket."""
from __future__ import annotations

import hashlib
import shutil
import time
from pathlib import Path

from botocore.exceptions import ClientError

from .environment import load, module, save, support


def file_digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def verify_local(env):
    folder = env.root / "evidence"
    index = load(folder / "index.json")
    expected = {(v["key"], v["version"]) for v in index["inventory"]["objects"]}
    observed = {(v["key"], v["version"]) for v in index["files"].values()}
    if expected != observed or len(index["files"]) != len(expected):
        raise ValueError("Local evidence does not cover every archived object version")
    for name, item in index["files"].items():
        if not name.isalnum() or len(name) != 64:
            raise ValueError("Invalid local evidence filename")
        path = folder / "objects" / name
        if path.is_symlink() or path.stat().st_size != item["bytes"] or file_digest(path) != item["sha256"]:
            raise ValueError("Local evidence failed size/SHA256 verification: " + str(path))
    return {"directory": str(folder), "versions": len(expected),
            "bytes": sum(v["bytes"] for v in index["files"].values()),
            "size_and_sha256_verified": True}


def preserve_locally(env, bucket):
    archive = module("isaac-lab-arena-on-aws/scripts/resources/archive.py")
    s3, account = env.client("s3"), env.config["account_id"]
    folder = env.root / "evidence"
    objects = folder / "objects"
    objects.mkdir(parents=True, exist_ok=True)
    path = folder / "index.json"
    if path.exists():
        index = load(path)
    else:
        index = {"inventory": archive.inventory(s3, account, [bucket])[0], "files": {}}
        save(path, index)
    if index["inventory"]["bucket"] != bucket:
        raise ValueError("Local evidence belongs to another bucket")
    frozen = load(env.root / "teardown/storage-archive.json")
    expected_digests = {(v["key"], v["version"]): v["sha256"] for v in frozen["objects"].values()}
    required = sum(v["bytes"] for v in index["inventory"]["objects"]
                   if hashlib.sha256(archive.encoded(v)).hexdigest() not in index["files"])
    free = shutil.disk_usage(folder).free
    if free < required + 2 * 1024**3:
        raise ValueError(f"Local evidence needs {required / 1024**3:.1f} GiB plus 2 GiB headroom; "
                         f"{free / 1024**3:.1f} GiB available. Archive retained; free space and resume.")
    for number, item in enumerate(index["inventory"]["objects"], 1):
        name = hashlib.sha256(archive.encoded(item)).hexdigest()
        target = objects / name
        if name in index["files"]:
            previous = index["files"][name]
            if (target.is_file() and not target.is_symlink() and target.stat().st_size == item["bytes"]
                    and file_digest(target) == previous["sha256"]):
                continue
            raise ValueError("Previously saved evidence changed; preserve the S3 archive and inspect " + str(target))
        print(f"Saving local evidence {number}/{len(index['inventory']['objects'])}: "
              f"{item['key']} ({item['bytes']} bytes)", flush=True)
        temporary, checksum, size, last = target.with_suffix(".part"), hashlib.sha256(), 0, time.monotonic()
        with temporary.open("wb") as output:
            for block in archive.read_version(s3, account, bucket, item["key"], item["version"], item["bytes"]):
                output.write(block)
                checksum.update(block)
                size += len(block)
                if time.monotonic() - last > 30:
                    print(f"Local evidence download: {size}/{item['bytes']} bytes", flush=True)
                    last = time.monotonic()
        digest = checksum.hexdigest()
        if size != item["bytes"] or digest != expected_digests.get((item["key"], item["version"]), digest):
            raise ValueError("Downloaded evidence differs from the verified archive")
        temporary.replace(target)
        index["files"][name] = {**item, "sha256": digest}
        save(path, index)
    return verify_local(env)


def finish_evidence(env):
    if not env.record.get("remove_evidence"):
        return
    folder = env.root / "teardown"
    owned_path = folder / "archive-ownership.json"
    state_dir = env.root / "retained-evidence"
    if not owned_path.exists():
        owned = support.terraform_inventory({"retained-evidence": state_dir})
        save(owned_path, owned)
    owned = load(owned_path)["retained-evidence"]
    buckets = [r["attributes"]["id"] for r in owned["resources"] if r["type"] == "aws_s3_bucket"]
    if len(buckets) != 1:
        raise ValueError("Expected exactly one owned archive bucket")
    bucket, s3, account = buckets[0], env.client("s3"), env.config["account_id"]
    current_state = support.terraform_inventory({"retained-evidence": state_dir})["retained-evidence"]
    if current_state["lineage"] != owned["lineage"]:
        raise ValueError("Archive Terraform state was replaced")
    # A receipt is written only after complete local readback. A retry after an
    # interrupted deletion uses that receipt and still rehashes the local files.
    receipt = folder / "local-evidence-verified.json"
    if receipt.exists():
        local = verify_local(env)
    else:
        local = preserve_locally(env, bucket)
        save(receipt, local)
    archive = module("isaac-lab-arena-on-aws/scripts/resources/archive.py")
    frozen = load(env.root / "evidence/index.json")["inventory"]
    try:
        current = archive.inventory(s3, account, [bucket])[0]
    except s3.exceptions.NoSuchBucket:
        current = None
    if current is not None:
        removals = support.exact_version_removals(current, frozen)
        for start in range(0, len(removals), 1000):
            result = s3.delete_objects(Bucket=bucket, ExpectedBucketOwner=account,
                                      Delete={"Objects": removals[start:start + 1000], "Quiet": True})
            if result.get("Errors"):
                raise ValueError("Temporary archive removal failed: " + str(result["Errors"]))
    env.terraform("retained-evidence", ["plan", "-destroy", "-input=false", "-out=retire.tfplan"],
                  "destroy-plan-archive")
    plan = env.terraform_json("retained-evidence", "show", "-json", "retire.tfplan")
    support.review_destroy_plan(plan, owned)
    save(folder / "archive-destroy-plan.json", plan)
    env.terraform("retained-evidence", ["apply", "-input=false", "retire.tfplan"], "destroy-apply-archive")
    check_removed(env, bucket)
    result = load(env.root / "result.json")
    result.update(retained_archive=None, removed_archive=bucket, local_evidence=local)
    save(env.root / "result.json", result)
    print("Temporary archive removed; verified evidence saved at:", local["directory"], flush=True)


def check_removed(env, bucket):
    states = support.terraform_inventory({"retained-evidence": env.root / "retained-evidence"})
    if any(s["resources"] for s in states.values()):
        raise ValueError("Archive Terraform resources remain")
    try:
        env.client("s3").head_bucket(Bucket=bucket, ExpectedBucketOwner=env.config["account_id"])
    except ClientError as exc:
        if exc.response["Error"]["Code"] not in {"404", "NoSuchBucket"}:
            raise
    else:
        raise ValueError("Temporary archive bucket remains")
