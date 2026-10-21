"""Collect: runs after the job SUCCEEDED. Moves results onto the VAMS asset and returns the
outcome summary the runner hands to SendTaskSuccess.

Input::

    {"resolved": <ReadManifest output>, "engine": "...",
     "pipelineExecutionArn" | "trainingJobArn" | "jobId": "..."}

Rules this follows (from the VAMS bridge contract):

* Files go to ``outputS3AssetFilesPath`` keeping their relative path; VAMS versions them onto the
  asset. Do NOT create a per-job folder here: the workflow's
  ``defaultOutputFileBaseExecutionPathExtension`` (``/{{jobName}}/``) already separates runs.
* Metrics and the model-registry reference go to ``outputS3AssetMetadataPath/asset.metadata.json``
  as the VAMS metadata envelope. ``asset.metadata.json`` is a reserved basename.
* The job's log TAIL is copied into the VAMS-account bridge log group (readable in VAMS today);
  it is NOT returned, because the SendTaskSuccess output lands in the durable execution record.
* The summary returned is outcome only: counts and identifiers, no job stdout.

Status: implemented. The copy streams each object through this function (two principals, so no
managed CopyObject); a single invocation is bounded by the Lambda timeout. A Batch job's output is
reduced to the deployable model first (see _select_deployable): intermediate checkpoint-<N>/
directories and optimizer/scheduler/RNG state stay in the toolchain checkpoints bucket. A run whose
deployable model still exceeds what 15 minutes can move needs a pre-arranged output location.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config

LOGGER = logging.getLogger("vams-bridge.collect")
LOGGER.setLevel(logging.INFO)

RETRY_CONFIG = Config(retries={"max_attempts": 5, "mode": "adaptive"})
TRANSFER_CONFIG = TransferConfig(multipart_chunksize=64 * 1024 * 1024, max_concurrency=4)

VAMS_DATA_ROLE_ARN = os.environ["VAMS_DATA_ROLE_ARN"]
VAMS_CONTROL_ROLE_ARN = os.environ["VAMS_CONTROL_ROLE_ARN"]
VAMS_ROLE_EXTERNAL_ID = os.environ["VAMS_ROLE_EXTERNAL_ID"]
CHECKPOINTS_BUCKET = os.environ["CHECKPOINTS_BUCKET"]
VAMS_BRIDGE_LOG_GROUP = os.environ["VAMS_BRIDGE_LOG_GROUP"]   # /aws/vendedlogs/Pipelines/pai-<component>
LOG_TAIL_MAX_EVENTS = int(os.environ.get("LOG_TAIL_MAX_EVENTS", "200"))

sagemaker = boto3.client("sagemaker", config=RETRY_CONFIG)
batch = boto3.client("batch", config=RETRY_CONFIG)
logs = boto3.client("logs", config=RETRY_CONFIG)


def _assume(role_arn: str, session_name: str):
    creds = boto3.client("sts", config=RETRY_CONFIG).assume_role(
        RoleArn=role_arn, RoleSessionName=session_name,
        ExternalId=VAMS_ROLE_EXTERNAL_ID, DurationSeconds=900)["Credentials"]
    return boto3.Session(aws_access_key_id=creds["AccessKeyId"],
                         aws_secret_access_key=creds["SecretAccessKey"],
                         aws_session_token=creds["SessionToken"])


def _job_outputs(event: dict) -> tuple[list[str], dict]:
    """Return (list of s3:// URIs to copy, metadata dict) for the finished job."""
    engine = event["engine"]
    resolved = event["resolved"]
    if engine == "sagemaker-pipeline":
        arn = event["pipelineExecutionArn"]
        steps = sagemaker.list_pipeline_execution_steps(PipelineExecutionArn=arn)["PipelineExecutionSteps"]
        meta = {"pipelineExecutionArn": arn}
        uris = []
        for step in steps:
            md = step.get("Metadata", {})
            if "TrainingJob" in md:
                tj = sagemaker.describe_training_job(
                    TrainingJobName=md["TrainingJob"]["Arn"].split("/")[-1])
                uris.append(tj["ModelArtifacts"]["S3ModelArtifacts"])
                meta["trainingJobName"] = tj["TrainingJobName"]
                meta["trainingSeconds"] = tj.get("TrainingTimeInSeconds")
            if "RegisterModel" in md:
                meta["modelPackageArn"] = md["RegisterModel"]["Arn"]
        return uris, meta
    if engine == "sagemaker-training":
        tj = sagemaker.describe_training_job(TrainingJobName=resolved["jobName"])
        return [tj["ModelArtifacts"]["S3ModelArtifacts"]], {
            "trainingJobArn": tj["TrainingJobArn"], "trainingSeconds": tj.get("TrainingTimeInSeconds")}
    if engine == "batch":
        prefix = f"s3://{CHECKPOINTS_BUCKET}/vams/{resolved['jobName']}/"
        return [prefix], {"batchJobId": event["jobId"]}
    raise ValueError(f"unknown engine {engine}")


def _copy_outputs(uris: list[str], dest_prefix: str, vams_session) -> int:
    """Copy each output object, or every object under an output prefix, to dest_prefix, preserving
    the path relative to the job's output root. Source objects are read with this function's own
    role (toolchain buckets); destination writes use the VAMS data role. Returns the object count.

    Two principals, so no managed CopyObject: each object is streamed through this function with a
    multipart upload. The reserved basename asset.metadata.json is never written from here, because
    VAMS would read it as the asset-level metadata envelope that _write_metadata_envelope owns.
    """
    own_s3 = boto3.client("s3", config=RETRY_CONFIG)
    dst_s3 = vams_session.client("s3", config=RETRY_CONFIG)
    dst_bucket, dst_root = _split(dest_prefix)
    dst_root = dst_root.rstrip("/") + "/"
    copied = 0

    def put(src_bucket, src_key, rel):
        nonlocal copied
        if rel.rsplit("/", 1)[-1] == "asset.metadata.json":
            LOGGER.warning("skipping %s: reserved basename", rel)
            return
        body = own_s3.get_object(Bucket=src_bucket, Key=src_key)["Body"]
        dst_s3.upload_fileobj(body, dst_bucket, dst_root + rel, Config=TRANSFER_CONFIG)
        copied += 1

    for uri in uris:
        bucket, key = _split(uri)
        if uri.endswith("/"):
            root = key
            sizes = {}
            for page in own_s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=root):
                for obj in page.get("Contents", []):
                    if not obj["Key"].endswith("/"):
                        sizes[obj["Key"][len(root):]] = obj["Size"]
            keep, skip = _select_deployable(list(sizes))
            LOGGER.info("output %s: copying %d objects (%d bytes), leaving %d objects (%d bytes) of "
                        "training state in place", uri, len(keep), sum(sizes[r] for r in keep),
                        len(skip), sum(sizes[r] for r in skip))
            for rel in keep:
                put(bucket, root + rel, rel)
        else:
            put(bucket, key, key.rsplit("/", 1)[-1])
    LOGGER.info("copied %d objects to %s", copied, dest_prefix)
    return copied


_CHECKPOINT_DIR = re.compile(r"^checkpoint-(\d+)/")
_TRAINING_STATE_FILES = ("optimizer.pt", "scheduler.pt", "scaler.pt")


def _select_deployable(rel_keys: list[str]) -> tuple[list[str], list[str]]:
    """Split a job's output (keys relative to its root) into (copy, leave) lists.

    The copy is the model a consumer loads, not the state needed to resume training. When the root
    holds a final model (a safetensors shard or index at the top level), every intermediate
    checkpoint-<N>/ directory is left out: each one duplicates the model shards and adds the
    optimizer state, which together are about three times the model's size. When the root holds no
    final model, the highest-numbered checkpoint-<N>/ directory is copied instead. Optimizer,
    scheduler and RNG state and Python bytecode caches are never copied. Everything left out stays
    in the toolchain checkpoints bucket, so a run can still be resumed from there.
    """
    def is_state(rel: str) -> bool:
        name = rel.rsplit("/", 1)[-1]
        return (name in _TRAINING_STATE_FILES or name.startswith("rng_state")
                or "/__pycache__/" in f"/{rel}" or name.endswith(".pyc"))

    root_has_model = any("/" not in r and r.endswith((".safetensors", ".safetensors.index.json"))
                         for r in rel_keys)
    steps = sorted({int(m.group(1)) for r in rel_keys if (m := _CHECKPOINT_DIR.match(r))})
    kept_dir = None if root_has_model or not steps else f"checkpoint-{steps[-1]}/"

    keep, skip = [], []
    for rel in sorted(rel_keys):
        in_checkpoint = _CHECKPOINT_DIR.match(rel) is not None
        if is_state(rel) or (in_checkpoint and not (kept_dir and rel.startswith(kept_dir))):
            skip.append(rel)
        else:
            keep.append(rel)
    return keep, skip


def _write_metadata_envelope(meta: dict, metadata_prefix: str, vams_session) -> None:
    envelope = {
        "type": "metadata",
        "updateType": "update",
        "metadata": [{"metadataKey": f"pai_{k}", "metadataValue": json.dumps(v) if not isinstance(v, str) else v}
                     for k, v in meta.items() if v is not None],
    }
    bucket, key = _split(metadata_prefix)
    vams_session.client("s3", config=RETRY_CONFIG).put_object(
        Bucket=bucket, Key=f"{key.rstrip('/')}/asset.metadata.json",
        Body=json.dumps(envelope).encode("utf-8"), ContentType="application/json")


def _source_log_streams(event: dict, meta: dict) -> list[tuple[str, str]]:
    """(logGroupName, logStreamName) pairs holding the finished job's output, per engine."""
    engine = event["engine"]
    if engine == "batch":
        job = batch.describe_jobs(jobs=[event["jobId"]])["jobs"]
        stream = (job[0].get("container") or {}).get("logStreamName") if job else None
        return [("/aws/batch/job", stream)] if stream else []
    job_name = meta.get("trainingJobName") or event["resolved"]["jobName"]
    streams = logs.describe_log_streams(logGroupName="/aws/sagemaker/TrainingJobs",
                                        logStreamNamePrefix=f"{job_name}/", limit=10)["logStreams"]
    return [("/aws/sagemaker/TrainingJobs", s["logStreamName"]) for s in streams]


def _ship_log_tail(event: dict, meta: dict, vams_session) -> int:
    """Copy the last LOG_TAIL_MAX_EVENTS events of the job's CloudWatch stream(s) into the
    VAMS-account bridge log group, stream = the VAMS pipelineExecutionId (the location RegisterRunner
    advertised to VAMS). The tail is for an operator reading the execution in VAMS; it is never
    returned in the task output. Returns the event count written."""
    pipeline_execution_id = event["resolved"]["pipelineExecutionId"]
    events = []
    for group, stream in _source_log_streams(event, meta):
        page = logs.get_log_events(logGroupName=group, logStreamName=stream,
                                   limit=LOG_TAIL_MAX_EVENTS, startFromHead=False)
        for ev in page.get("events", []):
            events.append({"timestamp": ev["timestamp"], "message": f"[{stream}] {ev['message']}"})
    events.sort(key=lambda e: e["timestamp"])
    events = events[-LOG_TAIL_MAX_EVENTS:]
    if not events:
        events = [{"timestamp": int(time.time() * 1000),
                   "message": "vams-bridge: no job log events were found to copy"}]

    vams_logs = vams_session.client("logs", config=RETRY_CONFIG)
    try:
        vams_logs.create_log_stream(logGroupName=VAMS_BRIDGE_LOG_GROUP, logStreamName=pipeline_execution_id)
    except vams_logs.exceptions.ResourceAlreadyExistsException:
        pass
    # <= LOG_TAIL_MAX_EVENTS events and far under the 1 MiB batch limit, so one call.
    vams_logs.put_log_events(logGroupName=VAMS_BRIDGE_LOG_GROUP, logStreamName=pipeline_execution_id,
                             logEvents=events)
    return len(events)


def _split(s3_uri: str) -> tuple[str, str]:
    without = s3_uri[len("s3://"):]
    bucket, _, key = without.partition("/")
    return bucket, key


def lambda_handler(event, context):
    resolved = event["resolved"]
    data = _assume(VAMS_DATA_ROLE_ARN, "pai-vams-bridge-collect")
    control = _assume(VAMS_CONTROL_ROLE_ARN, "pai-vams-bridge-collect-logs")

    uris, meta = _job_outputs(event)
    copied = _copy_outputs(uris, resolved["outputS3AssetFilesPath"], data)
    _write_metadata_envelope(meta, resolved["outputS3AssetMetadataPath"], data)
    log_events = _ship_log_tail(event, meta, control)

    # Outcome only. No job stdout, no ARNs with account ids beyond what VAMS already stores.
    return {
        "status": "SUCCEEDED",
        "engine": event["engine"],
        "jobName": resolved["jobName"],
        "filesWritten": copied,
        "metadataKeys": sorted(f"pai_{k}" for k in meta),
        "logEventsShipped": log_events,
        "completedAt": int(time.time()),
    }
