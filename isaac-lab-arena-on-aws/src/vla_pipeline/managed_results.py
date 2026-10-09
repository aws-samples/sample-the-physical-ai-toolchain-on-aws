"""Verify managed job/output linkage and downloaded validation evidence.

This independently reads the service records and S3 evidence. Validate remains
the actor that recomputes checkpoint contents; this client does not download and
rehash a multi-gigabyte checkpoint or claim to have reconstructed its training.
"""
from __future__ import annotations

import hashlib
import json
from urllib.parse import urlparse

from . import managed_checks as checks
from .backend import script
from .deployment import aws_session, discover
from .managed_checks import require, check_environment
from .operations import timestamp, write_json

VERIFICATION_VERSION = 4


class CheckResults:
    """Record every expected check, including checks not reached after a failure."""

    def __init__(self, directory, steps):
        self.path = directory / "contract-checks.json"
        selected = set(steps)
        functions = [
            (checks.check_verification_account, None),
            (checks.check_requested_managed_steps_succeeded, None),
            (checks.check_executed_parameters_match_request, None),
            (checks.check_finetune_job_completed_as_requested, "FineTune"),
            (checks.check_simulation_job_completed_as_requested, "SimEval"),
            (check_versioned_job_outputs, None),
            (checks.check_simulation_consumed_selected_checkpoint, "SimEval"),
            (checks.check_validation_job_completed_with_selected_inputs, "Validate"),
            (checks.check_validated_metrics_match_simulation_and_request, "Validate"),
            (checks.check_training_lineage_matches_requested_mode, "Validate"),
            (checks.check_validated_checkpoint_identity, "Validate"),
            (check_checkpoint_version_readback, "Validate"),
            (checks.check_validation_used_packaged_sdk, "Validate"),
            (checks.check_conditional_publication_is_real, "Validate"),
            (checks.check_downloaded_attestation_matches_execution, "Validate"),
            (checks.check_promoted_checkpoint_matches_validated_artifact, "Validate"),
            (checks.check_success_gate_threshold, "SuccessGate"),
            (checks.check_registered_model_matches_publication, "RegisterModel"),
        ]
        self.rows = {
            fn.__name__: {"name": fn.__name__, "description": fn.__doc__,
                          "status": "not_run" if step is None or step in selected else "not_applicable"}
            for fn, step in functions
        }
        self.save()

    def summary(self):
        states = {row["status"] for row in self.rows.values()}
        return {"scope": "Actual SageMaker descriptions and independently read S3 evidence; "
                         "checkpoint-content and training claims retain Validate's stated scope",
                "status": "failed" if "failed" in states else "pending" if "not_run" in states else "passed",
                "checks": list(self.rows.values())}

    def save(self):
        write_json(self.path, self.summary())

    def run(self, function, *args, **kwargs):
        row = self.rows[function.__name__]
        try:
            result = function(*args, **kwargs)
        except Exception as exc:
            row.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
            raise
        else:
            row["status"] = "passed"
            return result
        finally:
            self.save()
            print(f"{row['status'].upper():14} {row['name']}" +
                  (": " + row["reason"] if row.get("reason") else ""), flush=True)


def check_versioned_job_outputs(directory, s3, manifest, params, rows, output_buckets):
    """Re-read nonempty job output versions and validate the actual simulation archive and checkpoint linkage."""
    return script("local/local_outputs.py").verify_outputs(
        directory, s3, manifest, params, rows, expected_output_buckets=output_buckets)


def check_checkpoint_version_readback(s3, checkpoint, identity):
    """S3 still serves the nonempty checkpoint version identified by validation."""
    actual = object_identity(s3, checkpoint, version=identity["version_id"])
    require(actual["etag"].strip('"') == identity["etag"].strip('"'),
            "S3 checkpoint version differs from validation evidence")
    return actual


def object_identity(s3, uri, *, bucket=None, version=None):
    location = urlparse(uri)
    require(location.scheme == "s3" and location.netloc and location.path.strip("/")
            and not location.query and not location.fragment, f"Invalid output URI: {uri}")
    require(bucket is None or location.netloc == bucket, "Output is outside its configured bucket")
    head = s3.head_object(Bucket=location.netloc, Key=location.path.lstrip("/"),
                          **({"VersionId": version} if version else {}))
    require(head.get("VersionId") not in (None, "", "null") and head["ContentLength"] > 0,
            "Output must be nonempty and versioned")
    return {"uri": uri, "bucket": location.netloc, "key": location.path.lstrip("/"),
            "version_id": head["VersionId"], "etag": head["ETag"], "bytes": head["ContentLength"]}


def read_json(s3, identity):
    require(identity["bytes"] <= 8 * 1024 * 1024, "Evidence JSON exceeds 8 MiB")
    stream = s3.get_object(Bucket=identity["bucket"], Key=identity["key"],
                           VersionId=identity["version_id"])["Body"]
    try:
        body = stream.read(8 * 1024 * 1024 + 1)
    finally:
        stream.close()
    require(len(body) == identity["bytes"], "Evidence download length differs from S3 metadata")
    return json.loads(body), hashlib.sha256(body).hexdigest()


def verify(record, store):
    directory = store.path("runs", record["id"]).parent
    results = CheckResults(directory, record["request"]["steps"])
    try:
        return verify_and_record(record, directory, results)
    except Exception as exc:
        failure = results.summary()
        failure.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write_json(results.path, failure)
        record["contract_checks"] = failure
        raise


def verify_and_record(record, directory, results):
    session = aws_session(record.get("profile"), record["region"])
    results.run(checks.check_verification_account,
                session.client("sts").get_caller_identity()["Account"], record["account_id"])
    cfg = discover(session, record["project"])
    sm, s3 = session.client("sagemaker"), session.client("s3")
    arn, request = record["execution_arn"], record["request"]
    execution = sm.describe_pipeline_execution(PipelineExecutionArn=arn)
    rows = [row for page in sm.get_paginator("list_pipeline_execution_steps").paginate(
        PipelineExecutionArn=arn) for row in page["PipelineExecutionSteps"]]
    results.run(checks.check_requested_managed_steps_succeeded, execution, rows, request["steps"])
    actual = {item["Name"]: item["Value"] for page in sm.get_paginator(
        "list_pipeline_parameters_for_execution").paginate(PipelineExecutionArn=arn)
        for item in page["PipelineParameters"]}
    results.run(checks.check_executed_parameters_match_request, actual, request["parameters"])
    write_json(directory / "managed-execution.json",
               {"execution": execution, "steps": rows, "parameters": actual})
    params = request["parameters"]
    by_name = {row["StepName"]: row for row in rows}
    outputs, jobs = {}, {}
    output_buckets = {"FineTune": cfg.bucket, "SimEval": cfg.handoff_bucket}
    for step, function in (
        ("FineTune", checks.check_finetune_job_completed_as_requested),
        ("SimEval", checks.check_simulation_job_completed_as_requested),
    ):
        if step not in request["steps"]:
            continue
        job_arn = by_name[step]["Metadata"]["TrainingJob"]["Arn"]
        name = job_arn.rsplit("/", 1)[1]
        job = sm.describe_training_job(TrainingJobName=name)
        results.run(function, job, params)
        outputs[step] = {**object_identity(
            s3, job["ModelArtifacts"]["S3ModelArtifacts"], bucket=output_buckets[step]), "job_name": name}
        jobs[step] = job
    write_json(directory / "managed-jobs.json", jobs)
    write_json(directory / "job-outputs.json", outputs)
    manifest = {"development_bucket": cfg.bucket,
                "checkpoint_identity": request.get("checkpoint_identity")}
    adapted_rows = json.loads(json.dumps(rows, default=str))
    for row in adapted_rows:
        if row["StepName"] in outputs:
            row["Metadata"]["TrainingJob"]["Arn"] = outputs[row["StepName"]]["job_name"]
    proof, report = results.run(
        check_versioned_job_outputs, directory, s3, manifest, params, adapted_rows, output_buckets)
    result = {"independently_verified": True, "verification_status": "passed",
              "verification_version": VERIFICATION_VERSION,
              "verification_scope": proof["scope"], "outputs": outputs,
              "source_commit": record["source_commit"], "execution_arn": arn,
              "verified_at": timestamp(), "requested_steps": request["steps"]}
    if report:
        result.update(episodes=report["episodes"], success_rate=report["success_rate"])
        checkpoint = request.get("checkpoint_s3") or outputs["FineTune"]["uri"]
        results.run(checks.check_simulation_consumed_selected_checkpoint, jobs["SimEval"], checkpoint)
    if "Validate" in request["steps"]:
        processing = sm.describe_processing_job(
            ProcessingJobName=by_name["Validate"]["Metadata"]["ProcessingJob"]["Arn"].rsplit("/", 1)[1])
        write_json(directory / "managed-validation-job.json", processing)
        receipt_uri = results.run(checks.check_validation_job_completed_with_selected_inputs,
            processing, params, request["steps"], checkpoint, outputs["SimEval"]["uri"], arn)
        receipt_identity = object_identity(s3, receipt_uri, bucket=cfg.handoff_bucket)
        receipt, receipt_sha = read_json(s3, receipt_identity)
        results.run(checks.check_validated_metrics_match_simulation_and_request, receipt, params, report)
        lineage = results.run(checks.check_training_lineage_matches_requested_mode,
                              receipt, params, request["steps"])
        identity = receipt["model_artifact_identity"]
        expected = request.get("checkpoint_identity") or outputs["FineTune"]
        results.run(checks.check_validated_checkpoint_identity, identity, checkpoint, expected)
        checkpoint_identity = results.run(check_checkpoint_version_readback, s3, checkpoint, identity)
        runtime = receipt["validation_runtime"]
        results.run(checks.check_validation_used_packaged_sdk, runtime)
        promotion = receipt["promotion"]
        results.run(checks.check_conditional_publication_is_real, promotion)
        attestation_identity = object_identity(s3, promotion["attestation_uri"], bucket=cfg.trust_bucket)
        attestation, digest = read_json(s3, attestation_identity)
        results.run(checks.check_downloaded_attestation_matches_execution,
                    attestation, digest, receipt, lineage, checkpoint, arn)
        promoted = object_identity(s3, promotion["model_uri"], bucket=cfg.trust_bucket)
        results.run(checks.check_promoted_checkpoint_matches_validated_artifact,
                    promoted, checkpoint_identity, attestation, receipt)
        if "SuccessGate" in request["steps"]:
            results.run(checks.check_success_gate_threshold, receipt, params)
        if "RegisterModel" in request["steps"]:
            package_arn = by_name["RegisterModel-RegisterModel"]["Metadata"]["RegisterModel"]["Arn"]
            package = sm.describe_model_package(ModelPackageName=package_arn)
            write_json(directory / "managed-registration.json", package)
            results.run(checks.check_registered_model_matches_publication,
                        package, params, request["steps"], promotion, digest, checkpoint,
                        bool(request.get("checkpoint_s3")))
            result["model_package_arn"] = package_arn
        proof.update(receipt=receipt, receipt_identity=receipt_identity, receipt_sha256=receipt_sha,
                     attestation=attestation, attestation_identity=attestation_identity,
                     promoted_identity=promoted, checkpoint_identity=checkpoint_identity)
        result.update(receipt_uri=receipt_uri, attestation_sha256=digest,
                      verification_scope="Requested managed jobs, versioned evaluation evidence, "
                      "Validate publication and requested registration. Checkpoint-content digest "
                      "and training lineage retain Validate's stated verification/attestation scope.")
    proof["output_verification_scope"] = proof["scope"]
    proof["scope"] = result["verification_scope"]
    result["contract_checks"] = results.summary()
    require(result["contract_checks"]["status"] == "passed", "A required managed check was not completed")
    write_json(directory / "managed-evidence.json", proof)
    write_json(directory / "independent-verification.json", result)
    return result
