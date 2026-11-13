#!/usr/bin/env python3
"""Preserve explicitly selected host record directories; never collect credentials or caches."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile

import boto3

from support import load, put_version, save


def inventory(roots):
    files = {}
    for root in roots:
        root = root.resolve(strict=True)
        for path in root.rglob("*"):
            if path.is_symlink():
                # Local runs link to scratch source/container directories.
                # Preserve the reference, never follow it into caches or data.
                stat = path.lstat()
                files[str(path)] = {"path": str(path), "bytes": 0,
                                    "mtime_ns": stat.st_mtime_ns,
                                    "symlink_target": os.readlink(path)}
                continue
            if not path.is_file():
                continue
            if any(part in {".aws", ".docker", ".venv", ".env", "credentials"}
                   for part in path.parts):
                raise ValueError(f"Credential/cache directory is outside record archival: {path}")
            stat = path.stat()
            files[str(path)] = {"path": str(path), "bytes": stat.st_size,
                                "mtime_ns": stat.st_mtime_ns}
    if sum(item["bytes"] for item in files.values()) > 1024 ** 3:
        raise ValueError("Host records exceed 1 GiB. Archive model artifacts through S3 separately.")
    return sorted(files.values(), key=lambda item: item["path"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--account-id", required=True)
    parser.add_argument("--region", required=True)
    args = parser.parse_args()
    session = boto3.Session(region_name=args.region)
    if session.client("sts").get_caller_identity()["Account"] != args.account_id:
        raise ValueError("Host selected another account.")
    args.output.mkdir(parents=True, exist_ok=True)
    entries = inventory(args.root)
    saved = args.output / "inventory.json"
    if saved.exists() and load(saved) != entries:
        raise ValueError("Source records changed after capture. Preserve this attempt and use a new output.")
    save(saved, entries)
    members, links = [], []
    archive = args.output / "records.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for item in entries:
            if "symlink_target" in item:
                links.append(item)
                continue
            path = Path(item["path"])
            body = path.read_bytes()
            if len(body) != item["bytes"] or path.stat().st_mtime_ns != item["mtime_ns"]:
                raise ValueError("Record changed during capture: " + str(path))
            name = str(path).lstrip("/")
            info = tar.gettarinfo(str(path), arcname=name)
            tar.addfile(info, io.BytesIO(body))
            members.append({"member": name, "bytes": len(body),
                            "sha256": hashlib.sha256(body).hexdigest()})
    if inventory(args.root) != entries:
        raise ValueError("Records changed during capture; stop writers before trying again.")
    index = args.output / "index.json"
    save(index, {"files": members, "links": links})
    s3 = session.client("s3")
    references = {}
    for path in (archive, index):
        references[path.name] = put_version(
            s3, args.account_id, args.bucket, args.prefix + "/" + path.name, path.read_bytes())
    save(args.output / "references.json", references)
    reference = put_version(s3, args.account_id, args.bucket, args.prefix + "/references.json",
                            json.dumps(references).encode())
    print(json.dumps(reference))
    print(f"Preserved {len(members)} files and {len(links)} link references. Sources remain in place.")


if __name__ == "__main__":
    main()
