#!/usr/bin/env python3
"""Readable regression checks over real local-run evidence.

These checks supplement the live independent verifier. Replaying saved evidence
does not establish current AWS state or rerun training. Every check has a name,
an explanation and a result; missing evidence fails the applicable check.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re

CHECKS = []


def check(*steps):
    def register(function):
        CHECKS.append((function, set(steps)))
        return function
    return register


def require(condition, explanation):
    if not condition:
        raise AssertionError(explanation)


class Evidence:
    def __init__(self, root):
        self.root = Path(root)

    def json(self, name):
        return json.loads((self.root / name).read_text())

    @property
    def parameters(self):
        return self.json("parameters.json")

    @property
    def steps(self):
        return self.json("manifest.json").get(
            "requested_steps", ["FineTune", "SimEval", "Validate", "SuccessGate"])

    @property
    def receipt(self):
        return self.json("independent-s3-proof.json")["receipt"]


@check()
def check_request_digest_and_effective_parameters(e):
    """The saved request is unchanged and the executed parameters match it."""
    manifest = e.json("manifest.json")
    if not manifest.get("resolved_request_sha256"):
        require(manifest.get("parameter_source") in {
            "managed_arena_n16_sample", "registry_and_run_libero_train_default"},
            "CLI request digest is missing")
        require(e.steps == ["FineTune", "SimEval", "Validate", "SuccessGate"],
                "The manual runner requires the complete four-step contract")
        return "not_applicable", "Manual runner parameters have no frozen CLI request; full-step verification remains required"
    raw = (e.root / "resolved-request.json").read_bytes()
    require(hashlib.sha256(raw).hexdigest() == manifest["resolved_request_sha256"], "Request digest changed")
    request = json.loads(raw)
    require(request["steps"] == e.steps, "Executed step selection differs")
    require(all(e.parameters.get(k) == v for k, v in request["parameters"].items()),
            "An executed parameter differs from the request")


@check()
def check_all_requested_steps_succeeded(e):
    """Exactly the requested steps exist, and every one reports success."""
    rows = e.json("steps.json")["PipelineExecutionSteps"]
    require(len(rows) == len(e.steps) and {r["StepName"] for r in rows} == set(e.steps),
            "Missing, duplicate or unexpected pipeline step")
    require(all(r["StepStatus"] == "Succeeded" for r in rows), "A requested step did not succeed")
    require(e.json("status.json")["status"] == e.json("execution.json")["PipelineExecutionStatus"] == "Succeeded",
            "Launcher and pipeline terminal states disagree")


@check()
def check_worker_success_event(e):
    """The system manager recorded successful worker completion for this run."""
    result = e.json("systemd-exit.json")
    event = result["success_event"]
    unit = e.json("manifest.json").get("systemd_unit") or "vla-local-" + e.json("manifest.json")["run_id"]
    require(event["_PID"] == "1" and event["UNIT"] == unit + ".service"
            and event["MESSAGE_ID"] == "7ad2d189f7e94e70a38c781354912448",
            "No matching system-manager success event")
    require(result["properties"]["ActiveState"] != "active", "Worker is still running")


@check()
def check_containers_exited_without_oom(e):
    """Each requested workload container exited zero without an out-of-memory kill."""
    rows = e.json("container-exits.json")
    expected = {kind for step, kind in [("FineTune", "train"), ("SimEval", "eval"), ("Validate", "validate")]
                if step in e.steps}
    require(len(rows) == len(expected) and {r["kind"] for r in rows} == expected,
            "Missing, duplicate or unexpected workload container")
    require(all(not r["state"]["Running"] and r["state"]["ExitCode"] == 0
                and not r["state"]["OOMKilled"] for r in rows), "Container failed, is running, or was OOM-killed")


@check()
def check_container_image_digests(e):
    """GPU image references are digest-pinned; all containers match their recorded image IDs."""
    manifest = e.json("manifest.json")
    for row in e.json("container-exits.json"):
        reference = manifest["images"][row["kind"]]
        if row["kind"] in {"train", "eval"}:
            require(re.search(r"@sha256:[0-9a-f]{64}$", reference) is not None, "GPU image is not digest-pinned")
        require(re.fullmatch(r"sha256:[0-9a-f]{64}", row["image"]) is not None,
                "Actual image content ID is missing")
        require(row["image_reference"] == reference and row["image"] == manifest["image_ids"][row["kind"]],
                "Container image identity differs")


@check("FineTune")
def check_training_job_success_log(e):
    """The GR00T trainer printed completion with the requested and observed step counts."""
    if e.parameters["ModelFamily"] != "gr00t":
        return "not_applicable", "This exact completion marker belongs to the GR00T trainer"
    text = (e.root / "train.log").read_text()
    matches = re.findall(r"<<< finetune OK: requested (\d+) steps, log confirms global_step=(\d+)", text)
    expected = int(e.parameters["TrainSteps"])
    require(len(matches) == 1 and tuple(map(int, matches[0])) == (expected, expected),
            "Missing or conflicting trainer completion message")
    require("<<< finetune FAIL:" not in text, "Trainer logged a failed fine-tune")


@check("FineTune")
def check_training_step_count(e):
    """Observed optimizer steps equal the requested budget; exit zero alone is insufficient."""
    training = e.json("training-summary.json")
    require(training["training_performed"] is True, "Training was not performed")
    require(training["observed_final_step"] == training["requested_optimizer_steps"] ==
            int(e.parameters["TrainSteps"]), "Training stopped early or reported a different step count")


@check("FineTune")
def check_training_loss_is_finite(e):
    """Training emitted loss records, and every parsed loss and summary value is finite."""
    from local_metrics import training_records
    records, summary = training_records((e.root / "train.log").read_text())
    require(bool(records), "No training loss records were observed")
    require(all(math.isfinite(r["loss"]) for r in records), "Non-finite training loss")
    require(all(math.isfinite(v) for v in summary.values()), "Non-finite training summary")


@check("FineTune")
def check_training_summary_matches_log(e):
    """The summary refers to the exact saved training log and its parsed loss records."""
    from local_metrics import training_records
    body = (e.root / "train.log").read_bytes()
    records, _ = training_records(body.decode())
    summary = e.json("training-summary.json")
    require(summary["source_sha256"] == hashlib.sha256(body).hexdigest(), "Training log changed")
    require(summary["logged_loss_records"] == len(records), "Loss-record count differs")
    require(summary["first_logged_loss"] == records[0]["loss"] and
            summary["last_logged_loss"] == records[-1]["loss"], "Summary loss values differ")


@check("Validate")
def check_checkpoint_is_nonempty_and_versioned(e):
    """The selected checkpoint is a nonempty S3 object with an immutable version."""
    proof = e.json("independent-s3-proof.json")
    identity, head = e.receipt["model_artifact_identity"], proof["checkpoint_head"]
    require(identity["version_id"] not in (None, "", "null"), "Checkpoint has no immutable version")
    require(head["VersionId"] == identity["version_id"] and head["ContentLength"] > 0,
            "Checkpoint is empty or the wrong version")
    require(head["ETag"].strip('"') == identity["etag"].strip('"'), "Checkpoint ETag differs")


@check()
def check_requested_step_outputs_are_versioned(e):
    """Each requested GPU step produced a nonempty, versioned output for its recorded job."""
    outputs = e.json("status.json")["outputs"]
    rows = {r["StepName"]: r for r in e.json("steps.json")["PipelineExecutionSteps"]}
    for step in ("FineTune", "SimEval"):
        if step not in e.steps:
            continue
        output = outputs[step]
        require(output["job_name"] == rows[step]["Metadata"]["TrainingJob"]["Arn"],
                f"{step} output belongs to another job")
        require(output["version_id"] not in (None, "", "null") and output["bytes"] > 0,
                f"{step} output is empty or unversioned")


@check("Validate")
def check_evaluation_uses_selected_checkpoint(e):
    """Validation identifies this run's trained checkpoint or the explicitly supplied checkpoint."""
    identity = e.receipt["model_artifact_identity"]
    selected = e.parameters.get("CheckpointS3Uri")
    if selected:
        require(identity["s3_uri"] == selected, "Evaluation used another supplied checkpoint")
        require(identity["version_id"] == e.json("manifest.json")["checkpoint_identity"]["version_id"],
                "Evaluation used another checkpoint version")
    else:
        outputs = e.json("status.json")["outputs"]
        require(identity["s3_uri"] == outputs["FineTune"]["uri"] and
                identity["version_id"] == outputs["FineTune"]["version_id"],
                "Evaluation did not use this run's FineTune output version")
    require(e.receipt["policy_type"] == "checkpoint", "Evaluation did not use a checkpoint policy")


@check("Validate")
def check_training_and_checkpoint_reuse_are_distinct(e):
    """Checkpoint reuse makes no claim that this execution trained the model."""
    lineage = e.receipt["training_contract"]
    if e.parameters.get("CheckpointS3Uri"):
        require("FineTune" not in e.steps and lineage["train_path"] == "eval_only",
                "Checkpoint reuse was reported as training")
        require(not lineage["fields"] and not lineage["verified"] and not lineage["attested"],
                "This execution claims unobserved earlier training")
    else:
        require("FineTune" in e.steps and lineage["train_path"] == "train", "Training lineage is missing")
        fields = [v for v in lineage["fields"] if v["field"] == "train_steps"]
        require(len(fields) == 1 and fields[0]["expected"] == fields[0]["observed"] ==
                str(e.parameters["TrainSteps"]), "Checkpoint lineage has a different training dose")


@check("Validate")
def check_evaluation_task_and_episode_counts(e):
    """Every selected task has exactly the requested number of episodes."""
    receipt = e.receipt
    expected = list(range(10)) if e.parameters["Suite"] == "libero_spatial" else [0]
    require(receipt["task_ids"] == expected, "Unexpected task selection")
    require(receipt["episodes"] == len(expected) * int(e.parameters["EvalTrials"]), "Episode count differs")
    rows = receipt["per_task"]
    require(len(rows) == len(expected) and {r["task_id"] for r in rows} == set(expected),
            "Missing, duplicate or unexpected per-task result")
    require(all(r["episodes"] == int(e.parameters["EvalTrials"]) for r in rows), "Per-task episodes differ")


@check("Validate")
def check_evaluation_seed_model_and_suite(e):
    """The result records the requested seed, model family and simulation suite."""
    for key, parameter in [("eval_seed", "EvalSeed"), ("eval_trials", "EvalTrials"),
                           ("model_family", "ModelFamily"), ("suite", "Suite")]:
        require(e.receipt[key] == e.parameters[parameter], f"Evaluation {key} differs from the request")


@check("Validate")
def check_success_rate_arithmetic(e):
    """The reported success rates equal integer successes divided by actual episodes."""
    rows = e.receipt["per_task"]
    for row in rows:
        require(type(row["successes"]) is int and type(row["episodes"]) is int
                and 0 <= row["successes"] <= row["episodes"] and row["episodes"] > 0,
                "Invalid success or episode count")
        require(math.isclose(row["success_rate"], row["successes"] / row["episodes"],
                             rel_tol=1e-9, abs_tol=1e-9), "Per-task success-rate arithmetic differs")
    episodes, successes = sum(r["episodes"] for r in rows), sum(r["successes"] for r in rows)
    require(episodes == e.receipt["episodes"] and episodes > 0, "Total episode count differs")
    require(math.isclose(e.receipt["success_rate"], successes / episodes, rel_tol=1e-9, abs_tol=1e-9),
            "Overall success-rate arithmetic differs")


@check("Validate")
def check_validation_summary_matches_receipt(e):
    """The human-readable evaluation summary agrees with the validated result."""
    summary = e.json("evaluation-summary.json")
    for key in ("model_family", "suite", "task_ids", "episodes", "success_rate", "per_task",
                "policy_type", "model_artifact_identity", "training_contract"):
        require(summary[key] == e.receipt[key], f"Evaluation summary differs in {key}")


@check("Validate")
def check_validation_sdk_was_packaged(e):
    """Validation used the packaged SDK without installing or repairing it at runtime."""
    text = (e.root / "validate.log").read_text()
    runtime = e.receipt["validation_runtime"]
    require("packaged SDK ready: boto3=1.42.97 botocore=1.42.97" in text, "Packaged SDK readiness absent")
    require("sdk_repair" not in text and re.search(r"\bpip(?:3)?\s+install\b", text) is None,
            "Validation repaired dependencies at runtime")
    require(runtime["sdk_bundle_sha256"] == e.json("manifest.json")["validation_sdk_bundle_sha256"],
            "SDK bundle differs")


@check("Validate")
def check_attestation_binds_this_execution(e):
    """Published provenance identifies this execution, checkpoint and validated metrics."""
    proof = e.json("independent-s3-proof.json")
    attestation = proof["attestation"]
    require(attestation["execution"]["pipeline_execution_id"] ==
            e.json("execution.json")["PipelineExecutionArn"], "Attestation belongs to another execution")
    require(attestation["lineage"] == e.receipt["training_contract"], "Attestation lineage differs")
    require(attestation["validated_metrics"] ==
            {k: v for k, v in e.receipt.items() if k != "promotion"}, "Attestation metrics differ")


@check("Validate")
def check_promoted_checkpoint_identity(e):
    """The published checkpoint has the selected size and a content-addressed output path."""
    proof = e.json("independent-s3-proof.json")
    promotion = e.receipt["promotion"]
    require(proof["checkpoint_head"]["ContentLength"] == proof["promoted_head"]["ContentLength"],
            "Published checkpoint size differs")
    require(promotion["archive_sha256"] in promotion["model_uri"], "Publication path is not content-addressed")
    require(not promotion.get("local_test_publication"), "Publication was a test stub")


@check("Validate")
def check_conditional_publication_logged(e):
    """New publications logged conditional creation; identical retries are explicitly identified."""
    promotion, text = e.receipt["promotion"], (e.root / "validate.log").read_text()
    for field, created, marker in [
        ("artifact_publication", "created_multipart", "CompleteMultipartUpload IfNoneMatch=*"),
        ("attestation_publication", "created", "PutObject IfNoneMatch=*"),
    ]:
        require(promotion[field] in {created, "already_present_identical"}, "Unexpected publication result")
        if promotion[field] == created:
            require("conditional create succeeded: " + marker in text, "Conditional publication log missing")


@check("SuccessGate")
def check_success_gate_matches_threshold(e):
    """The gate passed only when the measured success rate met the explicit threshold."""
    rate, threshold = e.receipt["success_rate"], float(e.parameters["SuccessThreshold"])
    require(math.isfinite(rate) and 0 <= threshold <= rate <= 1, "Success threshold was not met")
    gate = next(r for r in e.json("steps.json")["PipelineExecutionSteps"] if r["StepName"] == "SuccessGate")
    require(gate["Metadata"]["Condition"]["Outcome"] is True, "Gate outcome was not true")


@check()
def check_failed_pipeline_negative_control(e):
    """The checker rejected a failed pipeline even though all required step names existed."""
    control = e.json("negative-control.json")
    require(control["description"]["PipelineExecutionStatus"] == "Failed", "Negative control was not failed")
    require(control["checker_rejected"] is True and control["required_steps_present"] is True,
            "Failed pipeline was accepted or lacked the intended negative-control conditions")
    require(set(e.steps) <= {r["StepName"] for r in control["steps"]["PipelineExecutionSteps"]},
            "Negative control is missing required steps")
    require("Missing required steps" not in control["checker_rejection"], "Checker rejected for the wrong cause")


@check()
def check_step_timings_are_nonnegative(e):
    """Recorded step and container durations are finite and nonnegative."""
    timings = e.json("timings.json")
    require(set(timings["step_wall_seconds"]) == set(e.steps), "Step timing coverage differs")
    for value in [timings["launcher_wall_seconds"], *timings["step_wall_seconds"].values(),
                  *timings["container_wall_seconds"].values()]:
        require(type(value) in (int, float) and math.isfinite(value) and value >= 0, "Invalid duration")


def run_checks(root):
    e, results = Evidence(root), []
    for function, required_steps in CHECKS:
        row = {"name": function.__name__, "description": function.__doc__}
        try:
            if required_steps and not required_steps <= set(e.steps):
                row.update(status="not_applicable", reason="Step not requested: " + ", ".join(sorted(required_steps)))
            else:
                result = function(e)
                row.update(status="passed" if result is None else result[0])
                if result is not None:
                    row["reason"] = result[1]
        except Exception as exc:
            row.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
        print(f"{row['status'].upper():14} {row['name']}" +
              (": " + row["reason"] if row.get("reason") else ""), flush=True)
        results.append(row)
    report = {"scope": "Regression checks of saved local-run evidence; live independent verification remains required",
              "status": "failed" if any(r["status"] == "failed" for r in results) else "passed",
              "checks": results}
    (e.root / "contract-checks.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(0 if run_checks(args.run_dir)["status"] == "passed" else 1)
