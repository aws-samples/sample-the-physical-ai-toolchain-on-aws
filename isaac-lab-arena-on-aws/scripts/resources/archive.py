#!/usr/bin/env python3
"""Inventory and preserve versioned deployment evidence before infrastructure teardown.

This command copies data; it never deletes source objects, policies or resources.
Stop writers first. Supply only buckets owned by the deployment being retired.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
from pathlib import Path
import re
import sys
import time

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.exceptions import (ClientError, ConnectionClosedError, EndpointConnectionError,
                                 IncompleteReadError, ReadTimeoutError, ResponseStreamingError)


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(encoded(value))
    temporary.replace(path)


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def session(args, account, region):
    value = boto3.Session(profile_name=args.profile, region_name=region)
    if value.client("sts").get_caller_identity()["Account"] != account:
        raise ValueError("AWS identity does not match the manifest account")
    return value.client("s3")


def inventory(client, account, buckets):
    result = []
    for bucket in sorted(buckets):
        versions, markers = [], []
        for page in client.get_paginator("list_object_versions").paginate(
                Bucket=bucket, ExpectedBucketOwner=account):
            for item in page.get("Versions", []):
                if item["VersionId"] == "null":
                    raise ValueError(
                        f"{bucket}/{item['Key']} has no immutable version; copy or rewrite it "
                        "into a versioned location first. Enabling versioning does not replace old null versions.")
                versions.append({"key": item["Key"], "version": item["VersionId"],
                                 "bytes": item["Size"], "etag": item["ETag"],
                                 "last_modified": item["LastModified"].isoformat(),
                                 "is_latest": item["IsLatest"]})
            markers.extend({"key": item["Key"], "version": item["VersionId"],
                            "last_modified": item["LastModified"].isoformat(),
                            "is_latest": item["IsLatest"]}
                           for item in page.get("DeleteMarkers", []))
        for page in client.get_paginator("list_multipart_uploads").paginate(
                Bucket=bucket, ExpectedBucketOwner=account):
            if page.get("Uploads"):
                raise ValueError(f"{bucket} has unfinished uploads; finish or explicitly abandon them first")
        result.append({"bucket": bucket,
                       "objects": sorted(versions, key=lambda v: (v["key"], v["version"])),
                       "delete_markers": sorted(markers, key=lambda v: (v["key"], v["version"]))})
    return result


def read_version(client, account, bucket, key, version, expected_bytes):
    """Resume interrupted reads of one immutable version; never skip or repeat yielded bytes."""
    offset, retries = 0, 0
    while True:
        start = offset
        try:
            response = client.get_object(
                Bucket=bucket, Key=key, VersionId=version, ExpectedBucketOwner=account,
                **({"Range": f"bytes={start}-"} if start else {}))
            stream = response["Body"]
            try:
                if response.get("VersionId") != version or response["ContentLength"] != expected_bytes - start:
                    raise ValueError(f"Version or size changed for {bucket}/{key}")
                if start and response.get("ContentRange") != f"bytes {start}-{expected_bytes - 1}/{expected_bytes}":
                    raise ValueError(f"Incorrect resumed byte range for {bucket}/{key}")
                for block in stream.iter_chunks(8 * 1024 * 1024):
                    if offset + len(block) > expected_bytes:
                        raise ValueError(f"Read exceeds expected size for {bucket}/{key}")
                    offset += len(block)
                    yield block
                    if offset == expected_bytes:
                        return
                if offset == expected_bytes:
                    return
                raise IncompleteReadError(actual_bytes=offset - start, expected_bytes=expected_bytes - start)
            finally:
                stream.close()
        except (ReadTimeoutError, ResponseStreamingError, IncompleteReadError,
                ConnectionClosedError, EndpointConnectionError) as exc:
            if retries >= 3:
                raise
            retries += 1
            print(f"Read interrupted ({type(exc).__name__}); retry {retries}/3 "
                  f"at byte {offset}/{expected_bytes}: {bucket}/{key}", flush=True)
            time.sleep(2 ** retries)


def digest(client, account, bucket, key, version, expected_bytes):
    checksum, count, last = hashlib.sha256(), 0, time.monotonic()
    for block in read_version(client, account, bucket, key, version, expected_bytes):
        checksum.update(block)
        count += len(block)
        if time.monotonic() - last >= 30:
            print(f"Readback {bucket}/{key}: {count}/{expected_bytes} bytes", flush=True)
            last = time.monotonic()
    if count != expected_bytes:
        raise ValueError(f"Incomplete read of {bucket}/{key}")
    return checksum.hexdigest()


def create_manifest(args):
    if not re.fullmatch(r"\d{12}", args.account_id):
        raise ValueError("--account-id must contain twelve digits")
    buckets = sorted(set(args.source_bucket))
    if args.destination_bucket in buckets:
        raise ValueError("The archive bucket must be outside the source/deletion scope")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9/_-]{0,180}", args.prefix):
        raise ValueError("--prefix must be a short, explicit archive prefix")
    client = session(args, args.account_id, args.region)
    versioning = client.get_bucket_versioning(
        Bucket=args.destination_bucket, ExpectedBucketOwner=args.account_id)
    if versioning.get("Status") != "Enabled":
        raise ValueError("The archive bucket must have versioning enabled")
    manifest = {
        "schema_version": 1, "created_at": now(), "account_id": args.account_id,
        "region": args.region, "destination": {"bucket": args.destination_bucket,
                                             "prefix": args.prefix.rstrip("/")},
        "sources": inventory(client, args.account_id, buckets),
        "scope": "All selected object versions; delete markers are recorded as metadata",
    }
    # Creating a different inventory must not overwrite an earlier review.
    with args.manifest.open("xb") as output:
        output.write(encoded(manifest))
    objects = [v for source in manifest["sources"] for v in source["objects"]]
    print(f"Inventoried {len(objects)} versions, {sum(v['bytes'] for v in objects)} bytes.")
    print(f"Review {args.manifest} before copy. No source data was changed.")


def copy_archive(args):
    manifest = json.loads(args.manifest.read_text())
    if manifest["schema_version"] != 1:
        raise ValueError("Unsupported archive manifest")
    account, region = manifest["account_id"], manifest["region"]
    client = session(args, account, region)
    manifest_sha = hashlib.sha256(encoded(manifest)).hexdigest()
    state = (json.loads(args.state.read_text()) if args.state.exists() else
             {"schema_version": 1, "manifest": manifest, "manifest_sha256": manifest_sha,
              "objects": {}, "status": "Copying"})
    if state["manifest_sha256"] != manifest_sha or state["manifest"] != manifest:
        raise ValueError("Saved archive belongs to another inventory")
    destination = manifest["destination"]
    buckets = [source["bucket"] for source in manifest["sources"]]
    if destination["bucket"] in buckets:
        raise ValueError("Archive destination overlaps the deletion scope")
    if client.get_bucket_versioning(Bucket=destination["bucket"],
                                   ExpectedBucketOwner=account).get("Status") != "Enabled":
        raise ValueError("Archive bucket versioning is not enabled")
    if inventory(client, account, buckets) != manifest["sources"]:
        raise ValueError("Sources changed after inventory; stop writers and create a new manifest")
    transfer = TransferConfig(multipart_threshold=64 * 1024 * 1024,
                              multipart_chunksize=64 * 1024 * 1024, max_concurrency=4)
    for source in manifest["sources"]:
        for item in source["objects"]:
            identity = {"bucket": source["bucket"], **item}
            object_id = hashlib.sha256(encoded(identity)).hexdigest()
            target_key = destination["prefix"] + "/objects/" + object_id
            previous = state["objects"].get(object_id)
            print(f"Preserving {source['bucket']}/{item['key']} ({item['bytes']} bytes)", flush=True)
            source_sha = (previous["sha256"] if previous else
                          digest(client, account, source["bucket"], item["key"],
                                 item["version"], item["bytes"]))
            if previous:
                target_version = previous["version"]
            else:
                # A previous process may have copied the object but lost its receipt.
                try:
                    existing = client.head_object(Bucket=destination["bucket"], Key=target_key,
                                                  ExpectedBucketOwner=account)
                except ClientError as exc:
                    if exc.response["Error"]["Code"] not in {"404", "NoSuchKey"}:
                        raise
                    existing = None
                if existing and existing.get("Metadata", {}).get("archive-source-sha256") != source_sha:
                    raise ValueError("Archive key already exists with different provenance")
                if not existing:
                    client.copy(
                        {"Bucket": source["bucket"], "Key": item["key"], "VersionId": item["version"]},
                        destination["bucket"], target_key, Config=transfer,
                        # Source ownership was checked by version listing and
                        # GetObject. S3Transfer exposes only destination ownership.
                        ExtraArgs={"ExpectedBucketOwner": account,
                                   "ServerSideEncryption": "AES256", "MetadataDirective": "REPLACE",
                                   "Metadata": {"archive-source-sha256": source_sha}})
                    existing = client.head_object(Bucket=destination["bucket"], Key=target_key,
                                                  ExpectedBucketOwner=account)
                target_version = existing["VersionId"]
                if target_version == "null":
                    raise ValueError("Archive copy has no immutable version")
            actual = digest(client, account, destination["bucket"], target_key,
                            target_version, item["bytes"])
            if actual != source_sha:
                raise ValueError("Archive content does not match its source")
            state["objects"][object_id] = {
                "source": identity, "key": target_key, "version": target_version,
                "bytes": item["bytes"], "sha256": source_sha,
            }
            save(args.state, state)
    if inventory(client, account, buckets) != manifest["sources"]:
        raise ValueError("Sources changed during copying; retain the sources and review a new inventory")
    state.update(status="Verified", verified_at=now())
    save(args.state, state)
    key = destination["prefix"] + "/index-" + manifest_sha + ".json"
    body = encoded(state)
    version = client.put_object(Bucket=destination["bucket"], Key=key, Body=body,
                                ExpectedBucketOwner=account, ServerSideEncryption="AES256")["VersionId"]
    actual = digest(client, account, destination["bucket"], key, version, len(body))
    if actual != hashlib.sha256(body).hexdigest():
        raise ValueError("Archive index readback failed")
    reference = {"bucket": destination["bucket"], "key": key, "version": version,
                 "sha256": actual, "bytes": len(body)}
    save(args.state.with_suffix(".reference.json"), reference)
    print(json.dumps(reference, indent=2))
    print("Archive verified. Sources, resources and their policies remain unchanged.")


def verify_archive(args):
    state = json.loads(args.state.read_text())
    manifest = state["manifest"]
    if (state.get("schema_version") != 1 or state.get("status") != "Verified"
            or hashlib.sha256(encoded(manifest)).hexdigest() != state["manifest_sha256"]):
        raise ValueError("Archive index is incomplete or inconsistent")
    expected = {
        hashlib.sha256(encoded({"bucket": source["bucket"], **item})).hexdigest()
        for source in manifest["sources"] for item in source["objects"]
    }
    if set(state["objects"]) != expected:
        raise ValueError("Archive index does not cover every inventoried object version")
    client = session(args, manifest["account_id"], manifest["region"])
    for item in state["objects"].values():
        actual = digest(client, manifest["account_id"], manifest["destination"]["bucket"],
                        item["key"], item["version"], item["bytes"])
        if actual != item["sha256"]:
            raise ValueError(f"Archive verification failed: {item['key']}")
    print(f"Verified {len(expected)} archived versions. Source buckets were not required.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", help="Named AWS profile; omit for an assigned role")
    commands = parser.add_subparsers(dest="operation", required=True)
    command = commands.add_parser("inventory", help="Record exact source versions without modifying them")
    command.add_argument("--account-id", required=True)
    command.add_argument("--region", required=True)
    command.add_argument("--source-bucket", action="append", default=[],
                         help="Repeat for owned source buckets; omit if setup failed before any bucket was created")
    command.add_argument("--destination-bucket", required=True)
    command.add_argument("--prefix", required=True)
    command.add_argument("--manifest", type=Path, required=True)
    command.set_defaults(function=create_manifest)
    command = commands.add_parser("copy", help="Copy selected versions and verify their contents")
    command.add_argument("--manifest", type=Path, required=True)
    command.add_argument("--state", type=Path, required=True)
    command.set_defaults(function=copy_archive)
    command = commands.add_parser("verify", help="Recheck archived data without the original buckets")
    command.add_argument("--state", type=Path, required=True)
    command.set_defaults(function=verify_archive)
    args = parser.parse_args()
    path = args.manifest if args.operation == "inventory" else args.state
    # One writer/reader per local operation record. A different remote operator
    # must use a different archive prefix.
    with path.with_suffix(path.suffix + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        args.function(args)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, ClientError, OSError, ConnectionClosedError, EndpointConnectionError,
            IncompleteReadError, ReadTimeoutError, ResponseStreamingError) as exc:
        print(f"Archive incomplete: {exc}", file=sys.stderr)
        raise SystemExit(1)
