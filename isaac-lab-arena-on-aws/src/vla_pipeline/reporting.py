"""Export one recorded experiment's actual evidence and videos for CLI/notebook use."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tempfile
from urllib.parse import urlparse

from .backend import script
from .deployment import aws_session
from .operations import write_json

EVALUATION_LIMIT = 512 * 1024 * 1024


def archive_local(record):
    """Publish verified worker diagnostics without removing containers or rerunning work."""
    from .execution import require_execution_host
    require_execution_host(record)
    if not record.get("independently_verified"):
        raise ValueError("The run must pass independent verification before exporting verified worker evidence")
    root = Path(record["run_dir"])
    saved = root / "archive-manifest.json"
    if saved.is_file():
        value = json.loads(saved.read_text())
        if value.get("run_id") != record["id"]:
            raise ValueError("Saved evidence archive belongs to a different run")
        return value
    completed = subprocess.run(
        [sys.executable, str(Path(record["component"]) / "scripts/local/archive_run.py"),
         "--run-dir", str(root)], capture_output=True, text=True, check=True, timeout=3600)
    return json.loads(completed.stdout)


def download(s3, account, reference, destination, *, maximum):
    """Read the recorded object version, checking its size and available hash."""
    from .resources.environment import module
    archive = module("isaac-lab-arena-on-aws/scripts/resources/archive.py")
    uri = urlparse(reference["uri"])
    version, size = reference["version_id"], reference["bytes"]
    if (uri.scheme != "s3" or not uri.netloc or not uri.path.strip("/")
            or uri.query or uri.fragment or version in (None, "", "null")
            or type(size) is not int or not 0 <= size <= maximum):
        raise ValueError("Expected a bounded, versioned evidence object")
    if destination.is_symlink():
        raise ValueError("Evidence destination must not be a symbolic link")
    head = s3.head_object(Bucket=uri.netloc, Key=uri.path.lstrip("/"), VersionId=version,
                          ExpectedBucketOwner=account)
    if head["ContentLength"] != size or (
            reference.get("etag") and head["ETag"] != reference["etag"]):
        raise ValueError("The evidence object differs from the recorded version")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    if temporary.is_symlink():
        raise ValueError("Evidence temporary destination must not be a symbolic link")
    digest, actual = hashlib.sha256(), 0
    with temporary.open("wb") as stream:
        for chunk in archive.read_version(s3, account, uri.netloc, uri.path.lstrip("/"), version, size):
            stream.write(chunk)
            digest.update(chunk)
            actual += len(chunk)
    if actual != size or reference.get("sha256", digest.hexdigest()) != digest.hexdigest():
        raise ValueError("Evidence download failed size/SHA256 verification")
    temporary.replace(destination)
    return digest.hexdigest()


def export_simulation(record, session, directory, *, include_video):
    """Extract bounded metrics and MP4 files from the exact verified SimEval artifact."""
    from .common.capped_reader import capped_tar_open
    output = record.get("outputs", {}).get("SimEval")
    requested = str(record["request"]["parameters"].get("EvalRecordVideo", "false")).lower() == "true"
    supported = record["request"]["selection"]["simulator"] == "isaac_arena"
    index = {"run_id": record["id"], "videos": [],
             "status": "not_requested" if supported else "not_supported",
             "reason": "Recording was not requested." if supported else "Recording is available for Arena evaluation."}
    if output is None:
        index["reason"] = "This execution has no completed SimEval artifact."
        write_json(directory / "videos.json", index)
        return
    if output.get("bytes", 0) <= 0:
        raise ValueError("The SimEval artifact is empty")
    with tempfile.TemporaryDirectory(prefix="evaluation-", dir=directory) as temporary:
        packed = Path(temporary) / "model.tar.gz"
        checksum = download(session.client("s3"), record["account_id"], output, packed,
                            maximum=EVALUATION_LIMIT)
        members_seen, expanded, metrics = set(), 0, None
        with capped_tar_open(str(packed), "r:gz", max_bytes=EVALUATION_LIMIT) as tar:
            for number, member in enumerate(tar):
                name = PurePosixPath(member.name)
                expanded += member.size
                if (number >= 10000 or expanded > EVALUATION_LIMIT or member.size < 0
                        or name.is_absolute() or ".." in name.parts
                        or not (member.isfile() or member.isdir()) or str(name) in members_seen):
                    raise ValueError("Unsafe or oversized evaluation evidence archive")
                members_seen.add(str(name))
                if not member.isfile():
                    continue
                if str(name) in {"metrics.json", "checkpoint_manifest.json"}:
                    if member.size > 8 * 1024 * 1024:
                        raise ValueError("Evaluation JSON exceeds its inspection limit")
                    value = json.loads(tar.extractfile(member).read())
                    write_json(directory / str(name), value)
                    if str(name) == "metrics.json":
                        metrics = value
                elif include_video and name.parts[0] == "videos" and name.suffix.lower() == ".mp4":
                    target = directory / str(name)
                    if target.is_symlink() or not target.resolve().is_relative_to(directory.resolve()):
                        raise ValueError("Video destination points outside this run's evidence")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    stream, digest, size = tar.extractfile(member), hashlib.sha256(), 0
                    with target.open("wb") as saved:
                        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                            saved.write(chunk)
                            digest.update(chunk)
                            size += len(chunk)
                    if size != member.size or size == 0:
                        raise ValueError("The recorded video is empty or incomplete")
                    index["videos"].append({"path": str(name), "bytes": size, "sha256": digest.hexdigest(),
                                             "source_artifact": output})
        if metrics is None:
            raise ValueError("The simulation artifact has no metrics.json")
        for key in ("episodes", "success_rate"):
            if record.get(key) is not None and metrics.get(key) != record[key]:
                raise ValueError("Exported metrics differ from the independently verified run")
        write_json(directory / "evaluation-summary.json", metrics)
        index.update(artifact=output, artifact_sha256=checksum)
    if include_video and index["videos"]:
        index.update(status="recorded", reason=None)
    elif requested and include_video:
        raise ValueError("Recording was requested, but this run's artifact contains no usable video")
    elif requested:
        index.update(status="not_exported", reason="Use report --include-video to export the recorded footage.")
    write_json(directory / "videos.json", index)


def managed_training(record, session, directory):
    """Read this training job's CloudWatch logs; never infer observed steps from requested steps."""
    parameters = record["request"]["parameters"]
    trained = "FineTune" in record["request"]["steps"]
    text = ""
    if trained:
        output = record["outputs"]["FineTune"]
        events = []
        for page in session.client("logs").get_paginator("filter_log_events").paginate(
                logGroupName="/aws/sagemaker/TrainingJobs", logStreamNamePrefix=output["job_name"] + "/"):
            events.extend(page["events"])
        events.sort(key=lambda item: (item["timestamp"], item.get("ingestionTime", 0), item.get("eventId", "")))
        write_json(directory / "training-log-events.json", events)
        text = "\n".join(event["message"] for event in events)
        (directory / "train.log").write_text(text)
    records, totals = script("local/local_metrics.py").training_records(text)
    explicit = re.findall(r"log confirms global_step=(\d+)", text)
    observed = [row["optimizer_step"] for row in records if type(row["optimizer_step"]) is int]
    summary = {
        "evidence_class": "trainer_reported_diagnostics", "training_performed": trained,
        "requested_optimizer_steps": parameters.get("TrainSteps") if trained else None,
        "observed_final_step": int(explicit[-1]) if explicit else observed[-1] if observed else None,
        "observed_step_basis": "trainer_log" if explicit or observed else "not_extracted",
        "logged_loss_records": len(records), "trainer_summary": totals,
        "source": "train.log" if trained else None,
        "source_sha256": hashlib.sha256(text.encode()).hexdigest() if trained else None,
    }
    write_json(directory / "training-summary.json", summary)


def report(record, store, *, output_dir=None, include_video=False):
    from .execution import refresh
    directory = Path(output_dir).expanduser().resolve() if output_dir else (
        store.path("runs", record["id"]).parent / "report")
    if (directory / "run.json").exists():
        saved = json.loads((directory / "run.json").read_text())
        if saved["id"] != record["id"]:
            raise ValueError("Output directory already belongs to a different run")
    directory.mkdir(parents=True, exist_ok=True)
    record = refresh(record, store, verify=True)
    session = aws_session(record.get("profile"), record["region"])
    if session.client("sts").get_caller_identity()["Account"] != record["account_id"]:
        raise ValueError("Report credentials select a different account")
    write_json(directory / "run.json", record)
    if not record.get("independently_verified"):
        raise ValueError(
            f"Run {record['id']}: {record['status']}; "
            f"{record.get('failure_reason') or record.get('activity')}. "
            "Follow the run until independent verification completes before exporting its report.")
    if record["mode"] == "local":
        if record["request"].get("transport") == "ssm":
            from .remote_execution import exchange
            record = exchange(record, "report", store)
        else:
            record["evidence_archive"] = archive_local(record)
            store.save("runs", record)
        archived = record["evidence_archive"]
        if archived["run_id"] != record["id"]:
            raise ValueError("Worker evidence belongs to another run")
        write_json(directory / "archive-index.json", archived)
        for item in archived["files"]:
            name = item["name"]
            if PurePosixPath(name).name != name or name in {"run.json", "videos.json"}:
                raise ValueError("Invalid worker evidence filename")
            download(session.client("s3"), record["account_id"], item, directory / name,
                     maximum=64 * 1024 * 1024)
    else:
        managed_training(record, session, directory)
        for path in store.path("runs", record["id"]).parent.glob("*.json"):
            if path.name != "record.json":
                write_json(directory / path.name, json.loads(path.read_text()))
        package = directory / "managed-registration.json"
        if package.exists():
            record["model_approval_status"] = json.loads(package.read_text()).get("ModelApprovalStatus")
    export_simulation(record, session, directory, include_video=include_video)
    record["report_directory"] = str(directory)
    write_json(directory / "run.json", record)
    store.save("runs", record)
    training = json.loads((directory / "training-summary.json").read_text())
    evaluation_path = directory / "evaluation-summary.json"
    evaluation = json.loads(evaluation_path.read_text()) if evaluation_path.exists() else {}
    print(f"Run {record['id']}: {record['status']}; independent verification passed.", flush=True)
    if training["training_performed"]:
        print(f"Training: requested {training['requested_optimizer_steps']} optimizer steps; "
              f"observed {training['observed_final_step']} in the trainer log.", flush=True)
    else:
        print("Training: skipped; this execution reused the supplied checkpoint.", flush=True)
    if evaluation.get("episodes") is not None:
        counts = evaluation.get("per_task", [])
        successes = sum(row["successes"] for row in counts) if counts else None
        print(f"Evaluation: {successes}/{evaluation['episodes']} robot successes; "
              f"suite {evaluation['suite']}.", flush=True)
    print("Saved evidence and recordings:", directory, flush=True)
    return record
