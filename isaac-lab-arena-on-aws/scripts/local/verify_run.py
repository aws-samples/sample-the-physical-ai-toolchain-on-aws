#!/usr/bin/env python3
"""Independent completed-run check; does not import the launcher's verifier."""
import argparse
import datetime
import hashlib
import json
import re
import subprocess
import sys
import time
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote, urlparse
from urllib.request import urlopen

import boto3


def read_dataset_reference(url):
    """Keep the independent lookup, allowing at most five minutes for HTTP 429."""
    deadline = time.monotonic() + 300
    for attempt in range(6):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError("Dataset verification retry allowance exhausted; retry the verifier "
                               "against this completed run. Training does not need to restart.")
        try:
            with urlopen(url, timeout=min(30, remaining)) as response:
                return json.load(response)
        except HTTPError as exc:
            if exc.code != 429:
                raise
            delay = min(30 * 2**attempt, 60)
            retry_after = exc.headers.get("Retry-After", "")
            try:
                delay = max(1, int(retry_after))
            except ValueError:
                try:
                    delay = max(1, (parsedate_to_datetime(retry_after)
                                   - datetime.datetime.now(datetime.timezone.utc)).total_seconds())
                except (TypeError, ValueError, OverflowError):
                    pass
            if attempt == 5 or delay >= deadline - time.monotonic():
                raise RuntimeError(
                    "Hugging Face could not serve the independent dataset lookup (HTTP 429). "
                    "Verification is incomplete; retry the verifier against this completed run "
                    "after the service recovers. Training does not need to restart."
                ) from exc
            print(f"Hugging Face dataset lookup returned HTTP 429; verification retry "
                  f"{attempt + 1}/5 in {delay:.0f}s. Completed training is retained.",
                  file=sys.stderr, flush=True)
            time.sleep(delay)


def verify_run(root):
    verification_started = time.monotonic()
    def load(name):
        return json.loads((root / name).read_text())


    def save(name, value):
        if name == "independent-verification.json":
            from check_run_evidence import run_checks
            checks = run_checks(root)
            value["contract_checks"] = checks
            if checks["status"] != "passed":
                value["status"] = "VerificationFailed"
                (root / name).write_text(json.dumps(value, indent=2, default=str) + "\n")
                raise RuntimeError("Named pipeline checks failed; see contract-checks.json")
        (root / name).write_text(json.dumps(value, indent=2, default=str) + "\n")


    status, execution, manifest, parameters = (
        load("status.json"), load("execution.json"), load("manifest.json"), load("parameters.json"))
    rows = load("steps.json")["PipelineExecutionSteps"]
    ordered = ["FineTune", "SimEval", "Validate", "SuccessGate"]
    requested = manifest.get("requested_steps", ordered)
    if manifest.get("resolved_request_sha256"):
        raw_request = (root / "resolved-request.json").read_bytes()
        assert hashlib.sha256(raw_request).hexdigest() == manifest["resolved_request_sha256"]
        request = json.loads(raw_request)
        assert request["steps"] == requested
        assert all(parameters.get(key) == value for key, value in request["parameters"].items())
    else:
        assert requested == ordered, "Legacy runs must retain the complete four-step contract"
    start = 1 if parameters.get("CheckpointS3Uri") else 0
    assert requested and requested == ordered[start:ordered.index(requested[-1]) + 1]
    required = set(requested)
    assert status["status"] == execution["PipelineExecutionStatus"] == "Succeeded"
    assert len(rows) == len(required) and {row["StepName"] for row in rows} == required
    assert all(row["StepStatus"] == "Succeeded" for row in rows)
    if "SuccessGate" in required:
        assert next(row for row in rows if row["StepName"] == "SuccessGate")["Metadata"]["Condition"]["Outcome"] is True
    unit = manifest.get("systemd_unit") or "vla-local-" + manifest["run_id"]
    service = subprocess.check_output(
        ["systemctl", "show", unit, "--property=LoadState,ActiveState,Result,ExecMainStatus"], text=True)
    properties = dict(line.split("=", 1) for line in service.splitlines())
    assert properties["ActiveState"] != "active", properties
    # A collected transient service returns default Result=success/ExecMainStatus=0
    # even with LoadState=not-found. Require the manager's real completion event.
    started_us = int(datetime.datetime.fromisoformat(status["started_at"]).timestamp() * 1_000_000)
    events = [
        json.loads(line) for line in subprocess.check_output(
            ["journalctl", "-u", unit, "-o", "json", "--no-pager"], text=True).splitlines()
    ]
    success_events = [
        event for event in events
        if event.get("MESSAGE_ID") == "7ad2d189f7e94e70a38c781354912448"
        and event.get("_PID") == "1" and event.get("UNIT") == unit + ".service"
        and int(event["__REALTIME_TIMESTAMP"]) >= started_us
    ]
    assert success_events, "No systemd manager event confirms successful launcher completion"
    launcher_exit_code = None
    if properties["LoadState"] != "not-found":
        assert properties["ExecMainStatus"] == "0" and properties["Result"] == "success", properties
        launcher_exit_code = int(properties["ExecMainStatus"])
    save("systemd-exit.json", {
        "properties": properties,
        "success_event": {key: success_events[-1].get(key) for key in (
            "_PID", "UNIT", "MESSAGE_ID", "MESSAGE", "__REALTIME_TIMESTAMP", "INVOCATION_ID")},
        "numeric_exit_code_observed": launcher_exit_code,
    })

    control = load("negative-control.json")
    assert control["description"]["PipelineExecutionStatus"] == "Failed"
    assert control["checker_rejected"] is True and control["required_steps_present"] is True
    assert required <= {row["StepName"] for row in control["steps"]["PipelineExecutionSteps"]}
    assert "Missing required steps" not in control["checker_rejection"]

    ids = subprocess.check_output(
        ["docker", "ps", "-aq", "--filter", "label=vla.local.run=" + manifest["run_id"]],
        text=True).split()
    expected_kinds = {kind for step, kind in (
        ("FineTune", "train"), ("SimEval", "eval"), ("Validate", "validate")) if step in required}
    assert len(ids) == len(expected_kinds), ids
    raw_containers = json.loads(subprocess.check_output(["docker", "inspect", *ids], text=True))
    containers = []
    validation_log = None
    for container in raw_containers:
        state = container["State"]
        assert not state["Running"] and state["ExitCode"] == 0 and not state["OOMKilled"]
        reference = container["Config"]["Image"]
        environment = dict(item.split("=", 1) for item in container["Config"]["Env"] if "=" in item)
        kind = ("train" if "TRAIN_MODEL_FAMILY" in environment else
                "eval" if "EVAL_MODEL_FAMILY" in environment else "validate")
        if manifest.get("resolved_request_sha256"):
            bindings = {
                "train": {"TRAIN_MODEL_FAMILY": "ModelFamily", "TRAIN_MAX_STEPS": "TrainSteps",
                          "TRAIN_SUITE": "TrainSuite", "GR00T_VERSION": "Gr00tVersion"},
                "eval": {"EVAL_MODEL_FAMILY": "ModelFamily", "EVAL_SUITE": "Suite",
                         "EVAL_TRIALS": "EvalTrials", "EVAL_SEED": "EvalSeed",
                         "EVAL_GR00T_VERSION": "Gr00tVersion"},
                "validate": {"EXPECTED_MODEL_FAMILY": "ModelFamily", "EXPECTED_SUITE": "Suite",
                             "EXPECTED_EVAL_TRIALS": "EvalTrials", "EXPECTED_EVAL_SEED": "EvalSeed"},
            }[kind]
            for variable, parameter in bindings.items():
                assert environment.get(variable) == str(parameters[parameter]), (
                    f"{kind} container {variable} differs from the requested {parameter}")
            if kind == "validate":
                assert environment.get("EXPECTED_TRAIN_PATH") == (
                    "eval_only" if parameters.get("CheckpointS3Uri") else "train")
        assert reference == manifest["images"][kind]
        assert container["Image"] == manifest["image_ids"][kind]
        containers.append({
            "id": container["Id"], "name": container["Name"], "state": state,
            "kind": kind,
            "image": container["Image"], "image_reference": container["Config"]["Image"],
            "mounts": container["Mounts"],
        })
        result = subprocess.run(
            ["docker", "logs", container["Id"]], capture_output=True, text=True, check=True)
        log = result.stdout + "\n" + result.stderr
        token = environment.get("HF_TOKEN")
        if token:
            log = log.replace(token, "[REDACTED_HF_TOKEN]")
        (root / f"{kind}.log").write_text(log)
        if kind == "validate":
            validation_log = log
    assert {container["kind"] for container in containers} == expected_kinds
    save("container-exits.json", containers)
    if "Validate" in required:
        assert validation_log is not None
        (root / "validate.log").write_text(validation_log)
        assert "sdk_repair" not in validation_log
        assert re.search(r"\bpip(?:3)?\s+install\b", validation_log) is None
        assert "packaged SDK ready: boto3=1.42.97 botocore=1.42.97" in validation_log

    session = boto3.Session(region_name=manifest["region"])
    caller = session.client("sts").get_caller_identity()
    assert caller["Account"] == manifest["caller"]["Account"]
    s3 = session.client("s3")
    bucket = manifest["development_bucket"]

    if "Validate" not in required:
        from local_outputs import verify_outputs
        from local_metrics import write_metrics

        proof, report = verify_outputs(root, s3, manifest, parameters, rows)
        save("independent-s3-proof.json", proof)
        result = {
            "status": "Succeeded", "verified_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "run_id": manifest["run_id"], "canonical_commit": manifest["canonical_commit"],
            "pipeline_execution_id": execution["PipelineExecutionArn"],
            "pipeline_steps": {row["StepName"]: row["StepStatus"] for row in rows},
            "complete_training_workflow": False,
            "verification_scope": proof["scope"],
            "launcher_result": "success", "launcher_exit_code": launcher_exit_code,
            "launcher_observation": "systemd_manager_success_event",
            "container_exit_codes": [container["state"]["ExitCode"] for container in containers],
            "negative_control_rejected": True, "negative_control_had_all_required_names": True,
            "validation_performed": False, "conditional_publication_checked": False,
            "outputs": proof["outputs"],
            "episodes": report["episodes"] if report else None,
            "success_rate": report["success_rate"] if report else None,
            "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
        write_metrics(root, status, rows, containers, parameters, report,
                      time.monotonic() - verification_started)
        save("independent-verification.json", result)
        return result

    def read(uri):
        parsed = urlparse(uri)
        assert parsed.scheme == "s3" and parsed.netloc == bucket, uri
        result = s3.get_object(Bucket=bucket, Key=parsed.path.lstrip("/"))
        return result["Body"].read()


    receipt_uri = (
        f"s3://{bucket}/validated/v1/{execution['PipelineExecutionArn']}/validated_metrics.json")
    receipt = json.loads(read(receipt_uri))
    assert status["receipt_uri"] == receipt_uri
    assert receipt["validation_passed"] is True
    expected_tasks = list(range(10)) if parameters["Suite"] == "libero_spatial" else [0]
    assert parameters["Suite"] in {"libero_spatial", "arena_gr1_fridge"}
    assert parameters["EvalTaskIds"] == "all"
    assert receipt["task_ids"] == expected_tasks
    assert receipt["episodes"] == parameters["EvalTrials"] * len(expected_tasks)
    assert receipt["policy_type"] == "checkpoint"
    assert receipt["suite"] == parameters["Suite"]
    assert receipt["model_family"] == parameters["ModelFamily"]
    runtime = receipt["validation_runtime"]
    assert runtime["boto3_version"] == runtime["botocore_version"] == "1.42.97"
    assert runtime["sdk_bundle_sha256"] == manifest["validation_sdk_bundle_sha256"]
    lineage = receipt["training_contract"]
    fields = {row["field"]: row for row in lineage["fields"]}
    repo_id, dataset_reference = None, None
    if parameters.get("CheckpointS3Uri"):
        assert lineage == {
            "train_path": "eval_only", "fields": [], "matched": [], "verified": [], "attested": [],
            "reason": "checkpoint_training_is_outside_this_execution",
        }, "A checkpoint-input run cannot vouch for earlier training"
    elif parameters["ModelFamily"] == "gr00t":
        assert lineage["matched"] == ["train_steps"]
        assert lineage["verified"] == ["dataset_revision"]
        assert set(lineage["attested"]) == {
            "dataset_source", "dataset_subdirectory", "dataset_content_digest", "train_suite"}
        for field in lineage["attested"]:
            assert fields[field]["status"] == "attested"
            assert fields[field]["binding"] == "checkpoint_digest"
        assert fields["dataset_content_digest"]["dataset_recomputed_by_validate"] is False
        assert fields["train_suite"]["observed"] == parameters["TrainSuite"]
        revision = fields["dataset_revision"]
        assert revision["verification_scope"] == "revision_identity_only"
        assert revision["verification_method"] in (
            "independent_huggingface_resolution", "pinned_commit_parameter")
        # The external lookup belongs to the selected suite, not a hardcoded Arena repo.
        repo_id = (
            "IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot"
            if parameters["Suite"] == "libero_spatial"
            else "nvidia/Arena-GR1-Manipulation-PlaceItemCloseDoor-Task")
        requested_revision = parameters["DatasetRevision"]
        if requested_revision == "__FROM_SUITE_MANIFEST__":
            requested_revision = "main"
        url = ("https://huggingface.co/api/datasets/" + repo_id + "/revision/"
               + quote(requested_revision, safe="") + "?expand=sha")
        dataset_reference = read_dataset_reference(url)
        assert dataset_reference["sha"] == revision["observed"]
    else:
        assert lineage["matched"] == ["train_steps"]
        assert not lineage.get("verified") and not lineage.get("attested")
        assert {field: row["status"] for field, row in fields.items()} == {
            "train_steps": "matched", "dataset_source": "recorded_only",
            "dataset_revision": "recorded_only", "train_suite": "unavailable"}

    promotion = receipt["promotion"]
    assert promotion["artifact_publication"] in {"created_multipart", "already_present_identical"}
    assert promotion["attestation_publication"] in {"created", "already_present_identical"}
    if promotion["artifact_publication"] == "created_multipart":
        assert "conditional create succeeded: CompleteMultipartUpload IfNoneMatch=*" in validation_log
    if promotion["attestation_publication"] == "created":
        assert "conditional create succeeded: PutObject IfNoneMatch=*" in validation_log
    assert not promotion.get("local_test_publication")
    attestation_bytes = read(promotion["attestation_uri"])
    attestation_digest = "sha256:" + hashlib.sha256(attestation_bytes).hexdigest()
    assert attestation_digest == promotion["attestation_content_digest"]
    attestation = json.loads(attestation_bytes)
    assert attestation["lineage"] == lineage
    assert attestation["validated_metrics"]["validation_runtime"] == runtime
    assert attestation["execution"]["pipeline_execution_id"] == execution["PipelineExecutionArn"]
    identity = receipt["model_artifact_identity"]
    source = urlparse(parameters["CheckpointS3Uri"]) if parameters.get("CheckpointS3Uri") else None
    assert identity["bucket"] == (source.netloc if source else bucket)
    assert identity["version_id"] not in ("", "null", None)
    if source:
        assert identity["key"] == source.path.lstrip("/")
        selected = manifest["checkpoint_identity"]
        assert identity["version_id"] == selected["version_id"]
        assert identity["etag"].strip('"') == selected["etag"].strip('"')
    checkpoint_head = s3.head_object(
        Bucket=identity["bucket"], Key=identity["key"], VersionId=identity["version_id"])
    assert checkpoint_head["VersionId"] == identity["version_id"]
    assert checkpoint_head["ETag"].strip('"') == identity["etag"].strip('"')
    artifact = urlparse(promotion["model_uri"])
    assert artifact.netloc == bucket
    promoted_head = s3.head_object(Bucket=bucket, Key=artifact.path.lstrip("/"))
    assert promoted_head["ContentLength"] == checkpoint_head["ContentLength"]
    assert promotion["archive_sha256"] in artifact.path
    if promotion["artifact_publication"] == "already_present_identical":
        # A deterministic retry can legitimately reuse the content-addressed artifact.
        # Independently hash the existing bytes before accepting this publication state.
        body = s3.get_object(Bucket=bucket, Key=artifact.path.lstrip("/"))["Body"]
        digest = hashlib.sha256()
        try:
            for chunk in iter(lambda: body.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        finally:
            body.close()
        assert digest.hexdigest() == promotion["archive_sha256"]
    save("independent-s3-proof.json", {
        "receipt": receipt, "attestation": attestation,
        "checkpoint_head": checkpoint_head, "promoted_head": promoted_head,
        "dataset_reference": (
            {"repo": repo_id, "sha": dataset_reference["sha"]} if dataset_reference else None),
    })
    result = {
        "status": "Succeeded",
        "verified_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "run_id": manifest["run_id"], "canonical_commit": manifest["canonical_commit"],
        "pipeline_execution_id": execution["PipelineExecutionArn"],
        "pipeline_steps": {row["StepName"]: row["StepStatus"] for row in rows},
        "complete_training_workflow": requested == ordered,
        "validation_performed": True,
        "verification_scope": "Requested steps and validated artifact publication",
        "launcher_result": "success",
        "launcher_exit_code": launcher_exit_code,
        "launcher_observation": "systemd_manager_success_event",
        "container_exit_codes": [container["state"]["ExitCode"] for container in containers],
        "negative_control_rejected": True, "negative_control_had_all_required_names": True,
        "sdk_packaged": True, "runtime_pip_absent": True,
        "conditional_publication_checked": True,
        "artifact_publication": promotion["artifact_publication"],
        "attestation_publication": promotion["attestation_publication"],
        "lineage_verified": lineage.get("verified", []),
        "lineage_attested": lineage.get("attested", []),
        "dataset_commit": dataset_reference["sha"] if dataset_reference else None,
        "receipt_uri": receipt_uri,
        "attestation_digest": attestation_digest,
        "episodes": receipt["episodes"], "success_rate": receipt["success_rate"],
        "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    from local_metrics import write_metrics

    write_metrics(root, status, rows, containers, parameters, receipt,
                  time.monotonic() - verification_started)
    save("independent-verification.json", result)
    return result


def main():
    if not __debug__:
        raise SystemExit("Verification requires assertions; do not use python -O or PYTHONOPTIMIZE")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    result = verify_run(args.run_dir.resolve())
    print(json.dumps(result, indent=2))
    from local_metrics import completion_summary

    print(completion_summary(args.run_dir.resolve(), result))


if __name__ == "__main__":
    main()
