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

Status: skeleton. The engine builders return the request shape; the staging copy and the dataset
validator call are marked TODO.
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path

import boto3
from botocore.config import Config

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


def _stage_inputs(resolved: dict, dataset_prefix: str, vams_s3) -> int:
    """Copy every manifest input file from the VAMS asset bucket into
    s3://DATASETS_BUCKET/<dataset_prefix>/dataset/<relativePath>, preserving the asset-relative
    path so the LeRobot layout (meta/, data/, videos/) survives. Returns the file count.

    TODO: implement with a managed copy per object (source read via the VAMS data role, destination
    write via this function's own role); parallelise with a bounded pool for large datasets.
    """
    count = 0
    for entry in resolved.get("inputFiles", []):
        rel = entry["relativePath"].lstrip("/")
        LOGGER.info("stage %s -> s3://%s/%s/dataset/%s", entry["fileKey"], DATASETS_BUCKET,
                    dataset_prefix, rel)
        count += 1
    if count == 0:
        raise ValueError("The manifest lists no input files; a LeRobot dataset needs at least meta/info.json")
    return count


def _validate_lerobot_dataset(dataset_prefix: str) -> None:
    """Run the toolchain's validate_lerobot_gr00t_dataset.py against the staged copy and reject
    the run on a non-zero report, before any compute is started.

    TODO: the validator reads a local directory; sync meta/ + a sample of episodes to /tmp first.
    """
    LOGGER.info("validate staged dataset under s3://%s/%s", DATASETS_BUCKET, dataset_prefix)


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
    # TODO: port _check_batch_divides_gpus from training/gr00t/submit_batch_training_n17.py so a
    # global batch the job definition's GPU count cannot divide is rejected here.
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

    session = _data_role_session()
    vams_s3 = session.client("s3", config=RETRY_CONFIG)

    resolved = manifestHelper.resolve_pipeline_inputs(body, vams_s3)
    pipeline_execution_id = manifestHelper.pipeline_execution_id_from_event_prefix(
        resolved["orchestrationEventPrefix"])
    job_name = _job_name(pipeline_execution_id, prefix)
    dataset_prefix = f"vams/{job_name}"

    cfg = _config(resolved, vams_s3)
    _stage_inputs(resolved, dataset_prefix, vams_s3)
    if engine != "sagemaker-training":
        _validate_lerobot_dataset(dataset_prefix)

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
