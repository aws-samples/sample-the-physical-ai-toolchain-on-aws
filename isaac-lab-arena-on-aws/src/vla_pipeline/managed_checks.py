"""Named checks over actual SageMaker descriptions and downloaded evidence.

The managed verifier supplies service responses and exact S3 versions. These
functions make its assertions readable and replayable; they do not launch jobs.
"""
from __future__ import annotations

import hashlib
import math
from urllib.parse import urlparse


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def check_environment(job, params, fields, label):
    environment = job.get("Environment", {})
    for variable, parameter in fields.items():
        if parameter in params:
            require(environment.get(variable) == str(params[parameter]),
                    f"{label} {variable} differs from requested {parameter}")


def check_verification_account(actual, expected):
    """The verifier is reading the requested AWS account."""
    require(actual == expected, "Verification credentials belong to another account")


def check_requested_managed_steps_succeeded(execution, rows, steps):
    """The pipeline and exactly the requested stages completed successfully."""
    from .execution import check_managed_steps
    passed, reason = check_managed_steps(execution, rows, steps)
    require(passed, reason or "Requested managed steps did not succeed")


def check_executed_parameters_match_request(actual, params):
    """SageMaker executed the parameters selected by the CLI."""
    require(all(actual.get(key) == str(value) for key, value in params.items()),
            "Executed parameters differ from the CLI request")


def check_gpu_job(job, params, step, image_key, type_key, volume_key):
    require(job["TrainingJobStatus"] == "Completed", f"{step} job did not complete")
    require(job["AlgorithmSpecification"]["TrainingImage"] == params[image_key],
            f"{step} used a different image")
    require(job["ResourceConfig"]["InstanceType"] == params[type_key]
            and job["ResourceConfig"]["VolumeSizeInGB"] == params[volume_key],
            f"{step} resources differ from the request")
    require(job["StoppingCondition"]["MaxRuntimeInSeconds"] == params["MaxRuntimeSeconds"],
            f"{step} runtime budget differs from the request")
    check_environment(job, params, {
        "VLA_MAX_RUNTIME_SECONDS": "MaxRuntimeSeconds",
        **({
            "TRAIN_MODEL_FAMILY": "ModelFamily", "TRAIN_MAX_STEPS": "TrainSteps",
            "TRAIN_SAVE_STEPS": "TrainSaveSteps", "TRAIN_SUITE": "TrainSuite",
            "TRAIN_DATASET_S3URI": "DatasetS3Uri", "TRAIN_DATASET_REVISION": "DatasetRevision",
            "GR00T_VERSION": "Gr00tVersion",
        } if step == "FineTune" else {
            "EVAL_MODEL_FAMILY": "ModelFamily", "EVAL_SUITE": "Suite",
            "EVAL_SEED": "EvalSeed", "EVAL_TRIALS": "EvalTrials",
            "EVAL_TASK_IDS": "EvalTaskIds", "EVAL_DOSE_STEPS": "EvalDoseSteps",
            "EVAL_GR00T_VERSION": "Gr00tVersion", "EVAL_SIM_CONFIG": "EvalSimConfig",
            "USE_GROOT_SERVER": "UseGrootServer", "ARENA_CONNECTOR": "ArenaConnector",
        }),
    }, step)


def check_finetune_job_completed_as_requested(job, params):
    """Training completed with the requested image, GPU, disk, budget and training parameters."""
    check_gpu_job(job, params, "FineTune", "TrainImageUri", "TrainInstanceType", "VolumeSizeInGB")


def check_simulation_job_completed_as_requested(job, params):
    """Simulation completed with the requested image, GPU, disk, budget and evaluation parameters."""
    check_gpu_job(job, params, "SimEval", "EvalImageUri", "EvalInstanceType", "EvalVolumeSizeInGB")


def check_simulation_consumed_selected_checkpoint(job, checkpoint):
    """The simulation job's model input is this run's trained or explicitly supplied checkpoint."""
    channels = [item for item in job["InputDataConfig"] if item["ChannelName"] == "model"]
    require(len(channels) == 1 and channels[0]["DataSource"]["S3DataSource"]["S3Uri"] == checkpoint,
            "SimEval did not consume this execution's selected checkpoint")


def check_validation_job_completed_with_selected_inputs(processing, params, steps, checkpoint, output, arn):
    """Validation completed with the selected checkpoint, evaluation output and expected parameters."""
    require(processing["ProcessingJobStatus"] == "Completed", "Validate job did not complete")
    check_environment(processing, params, {
        "EXPECTED_EVAL_SEED": "EvalSeed", "EXPECTED_EVAL_TRIALS": "EvalTrials",
        "EXPECTED_EVAL_TASK_IDS": "EvalTaskIds", "EXPECTED_MODEL_FAMILY": "ModelFamily",
        "EXPECTED_SUITE": "Suite", "EVALUATOR_IMAGE_URI": "EvalImageUri",
        "EXPECTED_FAMILY_VERSION": "Gr00tVersion",
        "EXPECTED_EMBODIMENT_TAG": "ExpectedEmbodimentTag",
        "EXPECTED_ARENA_EMBODIMENT": "ExpectedArenaEmbodiment",
        "EXPECTED_ARENA_OBJECT": "ExpectedArenaObject",
        "EXPECTED_POLICY_CONFIG": "ExpectedPolicyConfig",
        "EXPECTED_SUCCESS_THRESHOLD": "SuccessThreshold",
        **({
            "EXPECTED_TRAIN_STEPS": "TrainSteps", "EXPECTED_TRAIN_SUITE": "TrainSuite",
            "EXPECTED_DATASET_S3URI": "DatasetS3Uri", "EXPECTED_DATASET_REVISION": "DatasetRevision",
        } if "FineTune" in steps else {}),
    }, "Validate")
    require(processing.get("Environment", {}).get("EXPECTED_TRAIN_PATH") ==
            ("train" if "FineTune" in steps else "eval_only"),
            "Validate training/checkpoint-only mode differs from the request")
    require(processing["Environment"].get("EXPECTED_MODEL_SOURCE_URI") == checkpoint,
            "Validate expected source differs from the selected checkpoint")
    inputs = {item["S3Input"]["LocalPath"]: item["S3Input"]["S3Uri"]
              for item in processing["ProcessingInputs"] if "S3Input" in item}
    require(inputs.get("/opt/ml/processing/checkpoint") == checkpoint
            and inputs.get("/opt/ml/processing/eval_output") == output,
            "Validate input linkage differs from this execution")
    destinations = [item["S3Output"]["S3Uri"] for item in processing[
        "ProcessingOutputConfig"]["Outputs"] if item["OutputName"] == "validated"]
    require(len(destinations) == 1 and destinations[0].rstrip("/").endswith("/" + arn.rsplit("/", 1)[1]),
            "Validate output must be scoped to this execution")
    return destinations[0].rstrip("/") + "/validated_metrics.json"


def check_validated_metrics_match_simulation_and_request(receipt, params, report):
    """Validation reports the requested model, tasks, episodes and seed, matching the simulation result."""
    from .registry import resolve_suite
    require(receipt.get("validation_passed") is True and receipt.get("policy_type") == "checkpoint",
            "Validate did not publish successful real-checkpoint evidence")
    for field, parameter in (("model_family", "ModelFamily"), ("suite", "Suite"),
                             ("eval_seed", "EvalSeed"), ("eval_trials", "EvalTrials")):
        require(receipt.get(field) == params[parameter],
                f"Validate receipt {field} differs from requested {parameter}")
    suite = resolve_suite(params["Suite"])
    require(receipt.get("task_ids") == list(suite.canonical_task_ids)
            and receipt.get("episodes") == params["EvalTrials"] * len(suite.canonical_task_ids),
            "Validate did not report the requested episodes/tasks")
    require(receipt.get("success_rate") == report["success_rate"]
            and receipt.get("episodes") == report["episodes"],
            "Validate and simulation report different results")


def check_training_lineage_matches_requested_mode(receipt, params, steps):
    """Full training records the requested training dose; checkpoint reuse claims no training in this run."""
    from .backend import script
    lineage = script("local/local_profiles.py").check_training_contract(receipt, params)
    if "FineTune" in steps:
        comparisons = [item for item in lineage["fields"] if item["field"] == "train_steps"]
        require(len(comparisons) == 1
                and comparisons[0]["expected"] == comparisons[0]["observed"] == str(params["TrainSteps"]),
                "Training lineage does not contain the requested training-step count")
    return lineage


def check_validated_checkpoint_identity(identity, checkpoint, expected):
    """Validation identifies the exact selected checkpoint URI, S3 version and ETag."""
    source = urlparse(checkpoint)
    require((identity["bucket"], identity["key"], identity["version_id"], identity["etag"].strip('"')) ==
            (source.netloc, source.path.lstrip("/"), expected["version_id"], expected["etag"].strip('"')),
            "Validate's checkpoint identity differs from the selected checkpoint")


def check_validation_used_packaged_sdk(runtime):
    """The validation receipt identifies the expected packaged SDK bytes and versions."""
    from .validation_sdk import sdk_bundle
    digest = hashlib.sha256(sdk_bundle()).hexdigest()
    require(runtime.get("boto3_version") == runtime.get("botocore_version") == "1.42.97"
            and runtime.get("sdk_bundle_sha256") == digest, "Validate SDK identity differs")


def check_conditional_publication_is_real(promotion):
    """Publication used conditional creation or identical reuse, without a test stub."""
    require(not promotion.get("local_test_publication"), "Managed receipt contains a publication stub")
    require(promotion.get("artifact_publication") in {"created_multipart", "already_present_identical"}
            and promotion.get("attestation_publication") in {"created", "already_present_identical"},
            "Unexpected conditional-publication evidence")


def check_downloaded_attestation_matches_execution(attestation, digest, receipt, lineage, checkpoint, arn):
    """Downloaded attestation bytes bind the full validated receipt and checkpoint to this execution."""
    promotion, runtime, identity = receipt["promotion"], receipt["validation_runtime"], receipt["model_artifact_identity"]
    require(promotion["attestation_content_digest"] == "sha256:" + digest
            and promotion["attestation_sha256"] == digest, "Downloaded attestation digest mismatch")
    require(attestation["lineage"] == lineage
            and attestation["validated_metrics"]["validation_runtime"] == runtime
            and attestation["execution"]["pipeline_execution_id"] == arn.rsplit("/", 1)[1],
            "Attestation is not bound to this validation/execution")
    require(attestation.get("attestation_version") == 1 and attestation["validated_metrics"] ==
            {key: value for key, value in receipt.items() if key != "promotion"},
            "Attestation does not contain the complete validated receipt")
    require(attestation["source_artifact"]["identity"] == identity
            and attestation["source_artifact"]["expected_source_uri"] == checkpoint,
            "Attestation identifies a different source checkpoint")


def check_promoted_checkpoint_matches_validated_artifact(promoted, checkpoint_identity, attestation, receipt):
    """Published model size, content-addressed path and validated digest identities agree."""
    artifact, promotion = attestation["promoted_artifact"], receipt["promotion"]
    require(promoted["bytes"] == checkpoint_identity["bytes"] == artifact["size_bytes"]
            and artifact["archive_sha256"] == promotion["archive_sha256"]
            and artifact["weights_digest_recomputed_by_validate"] == receipt["weights_digest_recomputed_by_validate"]
            and promoted["key"] == f"artifacts/v1/sha256/{promotion['archive_sha256']}/model.tar.gz",
            "Promoted artifact linkage differs")


def check_success_gate_threshold(receipt, params):
    """Measured success meets the explicit threshold; this does not establish general model quality."""
    rate = receipt["success_rate"]
    require(type(rate) in (int, float) and math.isfinite(rate)
            and 1 >= rate >= params["SuccessThreshold"] >= 0, "Success threshold was not met")


def check_registered_model_matches_publication(package, params, steps, promotion, digest, checkpoint, reuse):
    """The pending-approval registration points to this model, image and attestation, with correct training mode."""
    containers = package["InferenceSpecification"]["Containers"]
    metadata = package.get("CustomerMetadataProperties", {})
    metrics = package.get("ModelMetrics", {}).get("ModelQuality", {}).get("Statistics", {})
    require(package["ModelPackageStatus"] == "Completed"
            and package["ModelApprovalStatus"] == "PendingManualApproval"
            and package["ModelPackageGroupName"] == params["ModelPackageGroupName"]
            and len(containers) == 1 and containers[0]["ModelDataUrl"] == promotion["model_uri"]
            and containers[0]["Image"] == params["TrainImageUri" if "FineTune" in steps else "EvalImageUri"],
            "Registered package differs from this execution's promoted model")
    require(metrics.get("S3Uri") == promotion["attestation_uri"]
            and metrics.get("ContentDigest") == "sha256:" + digest,
            "Registered metrics do not identify the downloaded attestation")
    require((metadata.get("eval_only") == "true" and metadata.get("input_checkpoint_uri") == checkpoint)
            if reuse else "eval_only" not in metadata, "Registered training/evaluation-only mode differs")
