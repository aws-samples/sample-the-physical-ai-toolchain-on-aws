"""ReadManifest: first state of every VAMS bridge runner.

Input (from the runner state machine)::

    {"body": <VAMS task body: TaskToken, inputManifestS3Location, inputConfigurationS3Location, ...>,
     "engine": "sagemaker-pipeline" | "sagemaker-training" | "batch",
     "runnerExecutionArn": "..."}

Output: the job request the runner's Submit state needs, plus the VAMS identifiers Collect needs
later. Any validation failure RAISES; the runner's Catch routes to NotifyFailure, which calls
SendTaskFailure under the control role, so a rejected run never leaves the VAMS step pending.

This Lambda runs in the toolchain account. Reading the VAMS manifest and staging the input files
requires the VAMS-account DATA role (assumed here with the ExternalId); the task token is never
read or logged by this function.

Status: the staging copy, the structural dataset validation and the GPU-divisibility gate are
implemented. Not covered: checks that read parquet columns (the container's loader asserts them).
"""
from __future__ import annotations

import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config

import checks
import manifestHelper

LOGGER = logging.getLogger("vams-bridge.read-manifest")
LOGGER.setLevel(logging.INFO)

RETRY_CONFIG = Config(retries={"max_attempts": 5, "mode": "adaptive"})

VAMS_DATA_ROLE_ARN = os.environ["VAMS_DATA_ROLE_ARN"]
VAMS_ROLE_EXTERNAL_ID = os.environ["VAMS_ROLE_EXTERNAL_ID"]
DATASETS_BUCKET = os.environ["DATASETS_BUCKET"]          # toolchain Foundation datasets bucket
CHECKPOINTS_BUCKET = os.environ["CHECKPOINTS_BUCKET"]    # toolchain Foundation checkpoints bucket
HF_TOKEN_SECRET_ID = os.environ.get("HF_TOKEN_SECRET_ID", "physical-ai/hf-token")
BATCH_JOB_DEFINITION = os.environ.get("BATCH_JOB_DEFINITION", "")  # name, never the ARN
SAGEMAKER_TRAINING_IMAGE = os.environ.get("SAGEMAKER_TRAINING_IMAGE", "")
STAGE_WORKERS = int(os.environ.get("STAGE_WORKERS", "8"))
STAGE_RESERVE_MS = 60_000  # stop submitting copies when this little invocation time remains

# Job names must be unique per VAMS pipeline execution so an at-least-once event cannot start a
# second job: SageMaker rejects a duplicate TrainingJobName; the Pipeline uses ClientRequestToken.
_JOB_NAME_SAFE = re.compile(r"[^A-Za-z0-9-]")


def _job_name(pipeline_execution_id: str, prefix: str) -> str:
    stem = _JOB_NAME_SAFE.sub("-", pipeline_execution_id)[:40]
    return f"{prefix}-{stem}"[:63]


def _data_role_session():
    sts = boto3.client("sts", config=RETRY_CONFIG)
    creds = sts.assume_role(
        RoleArn=VAMS_DATA_ROLE_ARN,
        RoleSessionName="pai-vams-bridge-read-manifest",
        ExternalId=VAMS_ROLE_EXTERNAL_ID,
        DurationSeconds=900,
    )["Credentials"]
    return boto3.Session(
        aws_access_key_id=creds["AccessKeyId"],
        aws_secret_access_key=creds["SecretAccessKey"],
        aws_session_token=creds["SessionToken"],
    )


# One VAMS-side client per asset-bucket Region (an asset bucket may sit outside the VAMS Region).
def _vams_client_for(session, region: str, cache: dict):
    key = region or ""
    if key not in cache:
        cache[key] = session.client("s3", region_name=region or None, config=RETRY_CONFIG)
    return cache[key]


def _copy_one(src_client, src_bucket, src_key, version_id, dst_client, dst_bucket, dst_key) -> int:
    """Stream one object from the VAMS bucket (read with the data role) into the datasets bucket
    (written with this function's own role). Two principals, so a managed CopyObject is not
    possible; upload_fileobj multiparts the stream and never buffers the whole object."""
    kwargs = {"Bucket": src_bucket, "Key": src_key}
    if version_id:
        kwargs["VersionId"] = version_id
    obj = src_client.get_object(**kwargs)
    dst_client.upload_fileobj(obj["Body"], dst_bucket, dst_key,
                              Config=TransferConfig(multipart_chunksize=64 * 1024 * 1024, max_concurrency=4))
    return int(obj.get("ContentLength") or 0)


def _stage_inputs(resolved: dict, dataset_prefix: str, vams_session, deadline_ms) -> set[str]:
    """Copy every manifest input file from the VAMS asset bucket into
    s3://DATASETS_BUCKET/<dataset_prefix>/dataset/<relativePath>, preserving the asset-relative path
    so the LeRobot layout (meta/, data/, videos/) survives. Returns the set of staged relative keys.

    Bounded by the Lambda's own timeout: the pool stops submitting when less than STAGE_RESERVE_MS
    remain, and the run is rejected rather than started on a partial copy. A dataset too large for
    one 15-minute invocation is an operator problem to solve with a pre-staged copy, not something
    this function should paper over.
    """
    entries = resolved.get("inputFiles", [])
    if not entries:
        raise ValueError("The manifest lists no input files; a LeRobot dataset needs at least meta/info.json")

    own_s3 = boto3.client("s3", config=RETRY_CONFIG)
    clients: dict = {}
    staged: set[str] = set()
    total = 0

    def work(entry):
        rel = entry["relativePath"].lstrip("/")
        src = _vams_client_for(vams_session, entry.get("bucketRegion", ""), clients)
        size = _copy_one(src, entry["bucket"], entry["key"], entry.get("versionId"),
                         own_s3, DATASETS_BUCKET, f"{dataset_prefix}/dataset/{rel}")
        return rel, size

    with ThreadPoolExecutor(max_workers=STAGE_WORKERS) as pool:
        futures = []
        for entry in entries:
            if deadline_ms() < STAGE_RESERVE_MS:
                raise TimeoutError(f"staging stopped with {len(futures)}/{len(entries)} files submitted: "
                                   f"not enough invocation time left; pre-stage this dataset")
            futures.append(pool.submit(work, entry))
        for fut in as_completed(futures):
            rel, size = fut.result()  # a failed copy raises here and rejects the run
            staged.add(rel)
            total += size

    LOGGER.info("staged %d files (%d bytes) under s3://%s/%s/dataset/", len(staged), total,
                DATASETS_BUCKET, dataset_prefix)
    return staged


def _validate_lerobot_dataset(dataset_prefix: str, staged: set[str]) -> dict:
    """Run the ported structural checks (checks.validate_dataset) against the staged copy: the four
    meta files are read back from the datasets bucket, every other path is checked against the set
    of keys that were actually staged. Raises before any compute is started."""
    own_s3 = boto3.client("s3", config=RETRY_CONFIG)
    texts: dict = {}
    for name in checks.META_FILES:
        if name not in staged:
            texts[name] = None
            continue
        obj = own_s3.get_object(Bucket=DATASETS_BUCKET, Key=f"{dataset_prefix}/dataset/{name}")
        texts[name] = obj["Body"].read().decode("utf-8")
    summary = checks.validate_dataset(texts, staged)
    LOGGER.info("dataset validated", extra=summary)
    return summary


def _job_definition_gpus(job_definition: str):
    """GPU count the ACTIVE job definition requests, or None when unreadable (advisory, as upstream)."""
    try:
        batch = boto3.client("batch", config=RETRY_CONFIG)
        revisions = batch.describe_job_definitions(
            jobDefinitionName=job_definition.split(":")[0], status="ACTIVE")["jobDefinitions"]
        if not revisions:
            return None
        latest = max(revisions, key=lambda d: d["revision"])
        for req in latest.get("containerProperties", {}).get("resourceRequirements", []):
            if req.get("type") == "GPU":
                return int(req["value"])
    except Exception:  # noqa: BLE001 - advisory, mirrors submit_batch_training_n17.py
        return None
    return None


def _config(resolved: dict, s3_client) -> dict:
    """The template's rendered configBody (json), fetched via the manifest helper.
    A present-but-unparseable body is an error, not a fallback to defaults."""
    body = manifestHelper.fetch_input_configuration(s3_client, resolved.get("inputConfigurationS3Location"))
    if body is None:
        raise ValueError("No input configuration: the pipeline requires a template")
    return body


def _build_sagemaker_pipeline(resolved, cfg, dataset_prefix, job_name):
    params = [
        {"Name": "DatasetPrefix", "Value": dataset_prefix},
        {"Name": "MaxSteps", "Value": str(int(cfg["maxSteps"]))},
        {"Name": "BatchSize", "Value": str(int(cfg["batchSize"]))},
        {"Name": "BaseModel", "Value": str(cfg["baseModel"])},
        {"Name": "InstanceType", "Value": str(cfg["instanceType"])},
        # HFToken is resolved by the pipeline step from Secrets Manager; it is never a VAMS tag.
        {"Name": "HFTokenSecretId", "Value": HF_TOKEN_SECRET_ID},
    ]
    return {"pipelineParameters": params,
            "idempotencyToken": resolved["pipelineExecutionId"][:36]}


def _build_batch(resolved, cfg, dataset_prefix, job_name):
    if not BATCH_JOB_DEFINITION:
        raise ValueError("BATCH_JOB_DEFINITION is not configured on this runner")
    env = [
        {"name": "DATASET_S3_URI", "value": f"s3://{DATASETS_BUCKET}/{dataset_prefix}/dataset/"},
        {"name": "CHECKPOINT_PREFIX", "value": f"s3://{CHECKPOINTS_BUCKET}/vams/{job_name}/"},
        {"name": "MAX_STEPS", "value": str(int(cfg["maxSteps"]))},
        {"name": "GLOBAL_BATCH_SIZE", "value": str(int(cfg["batchSize"]))},
        {"name": "EMBODIMENT", "value": str(cfg["embodiment"])},
        {"name": "TUNE_VISION_TOWER", "value": "1" if cfg.get("tuneVisionTower") else "0"},
        {"name": "HF_TOKEN_SECRET_ID", "value": HF_TOKEN_SECRET_ID},
    ]
    # Port of submit_batch_training_n17.py's pre-check: a global batch the job's GPU count cannot
    # divide fails at container startup, after the instance has been provisioned. Refuse it here.
    checks.check_batch_divides_gpus(int(cfg["batchSize"]), _job_definition_gpus(BATCH_JOB_DEFINITION))
    return {"jobDefinition": BATCH_JOB_DEFINITION, "jobDefinitionName": BATCH_JOB_DEFINITION,
            "environment": env}


def _build_sagemaker_training(resolved, cfg, dataset_prefix, job_name):
    # Isaac Lab RL shape (training/scripts/launch_rl.py): hyperparameters as environment variables.
    return {
        "algorithmSpecification": {"TrainingImage": SAGEMAKER_TRAINING_IMAGE, "TrainingInputMode": "File"},
        "inputDataConfig": [{"ChannelName": "training", "DataSource": {"S3DataSource": {
            "S3DataType": "S3Prefix", "S3Uri": f"s3://{DATASETS_BUCKET}/{dataset_prefix}/dataset/",
            "S3DataDistributionType": "FullyReplicated"}}}],
        "outputDataConfig": {"S3OutputPath": f"s3://{CHECKPOINTS_BUCKET}/vams/{job_name}/"},
        "resourceConfig": {"InstanceType": str(cfg["instanceType"]), "InstanceCount": 1, "VolumeSizeInGB": 100},
        "stoppingCondition": {"MaxRuntimeInSeconds": int(cfg.get("maxRuntimeSeconds", 43200))},
        "environment": {k: str(v) for k, v in cfg.get("environment", {}).items()},
    }


_BUILDERS = {
    "sagemaker-pipeline": ("pai-gr00t", _build_sagemaker_pipeline),
    "sagemaker-training": ("pai-train", _build_sagemaker_training),
    "batch": ("pai-batch", _build_batch),
}


def lambda_handler(event, context):
    engine = event["engine"]
    prefix, build = _BUILDERS[engine]
    body = event["body"]
    deadline_ms = getattr(context, "get_remaining_time_in_millis", lambda: 900_000)

    session = _data_role_session()
    vams_s3 = session.client("s3", config=RETRY_CONFIG)  # VAMS Region: manifest + config live there

    resolved = manifestHelper.resolve_pipeline_inputs(body, vams_s3)
    pipeline_execution_id = manifestHelper.pipeline_execution_id_from_event_prefix(
        resolved["orchestrationEventPrefix"])
    job_name = _job_name(pipeline_execution_id, prefix)
    dataset_prefix = f"vams/{job_name}"

    cfg = _config(resolved, vams_s3)
    staged = _stage_inputs(resolved, dataset_prefix, session, deadline_ms)
    if engine != "sagemaker-training":
        _validate_lerobot_dataset(dataset_prefix, staged)

    request = build(resolved, cfg, dataset_prefix, job_name)
    request.update({
        "jobName": job_name,
        "pipelineExecutionId": pipeline_execution_id,
        "orchestrationEventPrefix": resolved["orchestrationEventPrefix"],
        "datasetPrefix": dataset_prefix,
        # What Collect needs to write results back; copied from the manifest, never derived.
        "outputS3AssetFilesPath": resolved["outputS3AssetFilesPath"],
        "outputS3AssetMetadataPath": resolved["outputS3AssetMetadataPath"],
        "assetId": resolved["assetId"],
        "databaseId": resolved["databaseId"],
    })
    LOGGER.info("job request built", extra={"engine": engine, "jobName": job_name,
                                             "pipelineExecutionId": pipeline_execution_id})
    return request
