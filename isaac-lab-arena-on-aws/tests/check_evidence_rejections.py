#!/usr/bin/env python3
"""Regression checks that damage copies of real evidence and require rejection.

No AWS calls, job submissions, checkpoint downloads or source-evidence edits.
Supply a verified full GR00T local run, or managed run. These
tests check verifier behavior, not present-day AWS state or model quality.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from copy import deepcopy
import datetime as dt
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT),
    str(ROOT / "scripts/local"),
    str(ROOT / "src"),
]


def read_json(path):
    return json.loads(path.read_text())


class NamedResult(unittest.TextTestResult):
    """Save the same individual outcomes printed by unittest."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.checks = []

    def addSuccess(self, test):
        super().addSuccess(test)
        self.checks.append({"name": test.id(), "status": "passed"})

    def addFailure(self, test, error):
        super().addFailure(test, error)
        self.checks.append({"name": test.id(), "status": "failed", "reason": str(error[1])})

    def addError(self, test, error):
        super().addError(test, error)
        self.checks.append({"name": test.id(), "status": "error", "reason": str(error[1])})


class ArenaEvidenceRejections(unittest.TestCase):
    """A complete local GR00T sample must reject each damaged contract."""

    @classmethod
    def setUpClass(cls):
        from check_run_evidence import run_checks
        cls.checker = staticmethod(run_checks)
        cls.files = {p.name: p.read_bytes() for p in cls.source.iterdir()
                     if p.is_file() and not p.is_symlink() and p.suffix in {".json", ".log"}}
        cls.source_hashes = {name: hashlib.sha256(body).hexdigest() for name, body in cls.files.items()}
        manifest = json.loads(cls.files["manifest.json"])
        parameters = json.loads(cls.files["parameters.json"])
        if (manifest.get("requested_steps") != ["FineTune", "SimEval", "Validate", "SuccessGate"]
                or parameters["ModelFamily"] != "gr00t" or not manifest.get("resolved_request_sha256")):
            raise ValueError("Use a complete, CLI-launched GR00T local run for these rejection checks")
        with tempfile.TemporaryDirectory(dir=cls.output) as folder:
            root = Path(folder)
            for name, body in cls.files.items():
                (root / name).write_bytes(body)
            with redirect_stdout(io.StringIO()):
                report = cls.checker(root)
            (cls.output / "arena-baseline.json").write_text(json.dumps(report, indent=2) + "\n")
            if report["status"] != "passed":
                raise ValueError("Original Arena evidence failed; inspect arena-baseline.json first")

    @classmethod
    def tearDownClass(cls):
        for name, digest in cls.source_hashes.items():
            if hashlib.sha256((cls.source / name).read_bytes()).hexdigest() != digest:
                raise AssertionError("Source evidence changed during replay: " + name)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=self.output)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for name, body in self.files.items():
            (self.root / name).write_bytes(body)

    def edit(self, name, change):
        path = self.root / name
        value = read_json(path)
        change(value)
        path.write_text(json.dumps(value))

    def reject(self, expected):
        with redirect_stdout(io.StringIO()):
            report = self.checker(self.root)
        failures = [row for row in report["checks"] if row["status"] == "failed"]
        self.assertIn(expected, [row["name"] for row in failures], failures)

    def test_missing_cli_request_hash(self):
        self.edit("manifest.json", lambda v: v.update(resolved_request_sha256=None))
        self.reject("check_request_digest_and_effective_parameters")

    def test_changed_request_bytes(self):
        with (self.root / "resolved-request.json").open("a") as stream:
            stream.write("\n")
        self.reject("check_request_digest_and_effective_parameters")

    def test_failed_pipeline_with_all_step_names(self):
        self.edit("execution.json", lambda v: v.update(PipelineExecutionStatus="Failed"))
        self.reject("check_all_requested_steps_succeeded")

    def test_missing_pipeline_step(self):
        self.edit("steps.json", lambda v: v["PipelineExecutionSteps"].pop())
        self.reject("check_all_requested_steps_succeeded")

    def test_success_event_belongs_to_another_worker(self):
        self.edit("systemd-exit.json", lambda v: v["success_event"].update(UNIT="another-worker.service"))
        self.reject("check_worker_success_event")

    def test_container_was_killed_for_memory(self):
        self.edit("container-exits.json", lambda v: v[0]["state"].update(OOMKilled=True))
        self.reject("check_containers_exited_without_oom")

    def test_container_used_another_image(self):
        self.edit("container-exits.json", lambda v: v[0].update(image="not-the-selected-image"))
        self.reject("check_container_image_digests")

    def test_missing_training_success_message(self):
        path = self.root / "train.log"
        path.write_text(path.read_text().replace("<<< finetune OK:", "removed completion marker:"))
        self.reject("check_training_job_success_log")

    def test_training_stopped_one_step_early(self):
        self.edit("training-summary.json", lambda v: v.update(
            observed_final_step=int(v["requested_optimizer_steps"]) - 1))
        self.reject("check_training_step_count")

    def test_nonfinite_training_loss(self):
        with (self.root / "train.log").open("a") as stream:
            stream.write("\n{'loss': nan}\n")
        self.reject("check_training_loss_is_finite")

    def test_training_summary_disagrees_with_log(self):
        self.edit("training-summary.json", lambda v: v.update(logged_loss_records=v["logged_loss_records"] + 1))
        self.reject("check_training_summary_matches_log")

    def test_empty_checkpoint(self):
        self.edit("independent-s3-proof.json", lambda v: v["checkpoint_head"].update(ContentLength=0))
        self.reject("check_checkpoint_is_nonempty_and_versioned")

    def test_checkpoint_has_no_immutable_version(self):
        self.edit("independent-s3-proof.json",
                  lambda v: v["receipt"]["model_artifact_identity"].update(version_id="null"))
        self.reject("check_checkpoint_is_nonempty_and_versioned")

    def test_output_belongs_to_another_training_job(self):
        self.edit("status.json", lambda v: v["outputs"]["FineTune"].update(job_name="another-job"))
        self.reject("check_requested_step_outputs_are_versioned")

    def test_evaluation_used_another_checkpoint(self):
        self.edit("independent-s3-proof.json",
                  lambda v: v["receipt"]["model_artifact_identity"].update(s3_uri="s3://wrong/checkpoint"))
        self.reject("check_evaluation_uses_selected_checkpoint")

    def test_training_is_mislabeled_as_checkpoint_reuse(self):
        self.edit("independent-s3-proof.json", lambda v: v["receipt"]["training_contract"].update(train_path="eval_only"))
        self.reject("check_training_and_checkpoint_reuse_are_distinct")

    def test_evaluation_returned_one_fewer_episode(self):
        self.edit("independent-s3-proof.json",
                  lambda v: v["receipt"].update(episodes=v["receipt"]["episodes"] - 1))
        self.reject("check_evaluation_task_and_episode_counts")

    def test_evaluation_used_another_seed(self):
        self.edit("independent-s3-proof.json",
                  lambda v: v["receipt"].update(eval_seed=str(int(v["receipt"]["eval_seed"]) + 1)))
        self.reject("check_evaluation_seed_model_and_suite")

    def test_success_rate_arithmetic_is_wrong(self):
        self.edit("independent-s3-proof.json",
                  lambda v: v["receipt"].update(success_rate=0.5 if v["receipt"]["success_rate"] != 0.5 else 0.25))
        self.reject("check_success_rate_arithmetic")

    def test_readable_summary_disagrees_with_receipt(self):
        self.edit("evaluation-summary.json", lambda v: v.update(episodes=v["episodes"] + 1))
        self.reject("check_validation_summary_matches_receipt")

    def test_missing_evaluation_summary(self):
        (self.root / "evaluation-summary.json").unlink()
        self.reject("check_validation_summary_matches_receipt")

    def test_validator_repaired_dependencies_at_runtime(self):
        with (self.root / "validate.log").open("a") as stream:
            stream.write("\npip install unexpected-runtime-repair\n")
        self.reject("check_validation_sdk_was_packaged")

    def test_attestation_belongs_to_another_execution(self):
        self.edit("independent-s3-proof.json",
                  lambda v: v["attestation"]["execution"].update(pipeline_execution_id="another-execution"))
        self.reject("check_attestation_binds_this_execution")

    def test_published_model_is_a_stub(self):
        self.edit("independent-s3-proof.json", lambda v: v["receipt"]["promotion"].update(local_test_publication=True))
        self.reject("check_promoted_checkpoint_identity")

    def test_conditional_publication_message_is_missing(self):
        self.edit("independent-s3-proof.json",
                  lambda v: v["receipt"]["promotion"].update(artifact_publication="created_multipart"))
        path = self.root / "validate.log"
        path.write_text(path.read_text().replace("conditional create succeeded:", "removed publication marker:"))
        self.reject("check_conditional_publication_logged")

    def test_success_rate_is_below_threshold(self):
        self.edit("parameters.json", lambda v: v.update(SuccessThreshold=0.5))
        self.edit("independent-s3-proof.json", lambda v: v["receipt"].update(success_rate=0.0))
        self.reject("check_success_gate_matches_threshold")

    def test_failed_pipeline_negative_control_was_accepted(self):
        self.edit("negative-control.json", lambda v: v.update(checker_rejected=False))
        self.reject("check_failed_pipeline_negative_control")

    def test_negative_run_duration(self):
        self.edit("timings.json", lambda v: v.update(launcher_wall_seconds=-1))
        self.reject("check_step_timings_are_nonnegative")


class ManagedEvidenceRejections(unittest.TestCase):
    """Managed assertions replay saved service responses without calling AWS."""

    @classmethod
    def setUpClass(cls):
        from vla_pipeline import managed_checks
        cls.checks = managed_checks
        cls.original = {name: read_json(cls.source / filename) for name, filename in {
            "record": "record.json", "jobs": "managed-jobs.json", "proof": "managed-evidence.json",
            "package": "managed-registration.json", "execution": "managed-execution.json",
            "raw": "raw-evaluation.json", "validation": "managed-validation-job.json",
        }.items()}
        if cls.original["record"]["request"]["steps"] != [
                "FineTune", "SimEval", "Validate", "SuccessGate", "RegisterModel"]:
            raise ValueError("Use the saved verifier directory of a complete managed training run")

    def setUp(self):
        self.data = deepcopy(self.original)
        self.params = self.data["record"]["request"]["parameters"]
        self.receipt = self.data["proof"]["receipt"]

    def reject(self, function, arguments, change):
        # The unchanged real response must pass this exact assertion first.
        function(*arguments)
        copied = deepcopy(arguments)
        change(copied)
        with self.assertRaises(ValueError):
            function(*copied)

    def test_verification_uses_another_account(self):
        self.reject(self.checks.check_verification_account,
                    [self.data["record"]["execution_arn"].split(":")[4], self.data["record"]["account_id"]],
                    lambda a: a.__setitem__(0, "000000000000"))

    def test_training_service_reported_failure(self):
        self.reject(self.checks.check_finetune_job_completed_as_requested,
                    [self.data["jobs"]["FineTune"], self.params],
                    lambda a: a[0].update(TrainingJobStatus="Failed"))

    def test_training_used_another_instance_type(self):
        self.reject(self.checks.check_finetune_job_completed_as_requested,
                    [self.data["jobs"]["FineTune"], self.params],
                    lambda a: a[0]["ResourceConfig"].update(InstanceType="not-the-requested-instance"))

    def test_training_used_another_step_budget(self):
        self.reject(self.checks.check_finetune_job_completed_as_requested,
                    [self.data["jobs"]["FineTune"], self.params],
                    lambda a: a[0]["Environment"].update(TRAIN_MAX_STEPS=str(int(self.params["TrainSteps"]) - 1)))

    def test_training_used_another_disk_size(self):
        self.reject(self.checks.check_finetune_job_completed_as_requested,
                    [self.data["jobs"]["FineTune"], self.params],
                    lambda a: a[0]["ResourceConfig"].update(VolumeSizeInGB=self.params["VolumeSizeInGB"] + 1))

    def test_simulation_used_another_image(self):
        self.reject(self.checks.check_simulation_job_completed_as_requested,
                    [self.data["jobs"]["SimEval"], self.params],
                    lambda a: a[0]["AlgorithmSpecification"].update(TrainingImage="another-image"))

    def test_simulation_used_another_checkpoint(self):
        self.reject(self.checks.check_simulation_consumed_selected_checkpoint,
                    [self.data["jobs"]["SimEval"], self.data["proof"]["outputs"]["FineTune"]["uri"]],
                    lambda a: a.__setitem__(1, "s3://wrong/checkpoint"))

    def validation_arguments(self):
        return [self.data["validation"], self.params, self.data["record"]["request"]["steps"],
                self.data["proof"]["outputs"]["FineTune"]["uri"],
                self.data["proof"]["outputs"]["SimEval"]["uri"], self.data["record"]["execution_arn"]]

    def test_validation_job_failed(self):
        self.reject(self.checks.check_validation_job_completed_with_selected_inputs,
                    self.validation_arguments(), lambda a: a[0].update(ProcessingJobStatus="Failed"))

    def test_validation_used_another_checkpoint(self):
        self.reject(self.checks.check_validation_job_completed_with_selected_inputs,
                    self.validation_arguments(), lambda a: a.__setitem__(3, "s3://wrong/checkpoint"))

    def test_validation_writes_under_another_execution(self):
        self.reject(self.checks.check_validation_job_completed_with_selected_inputs,
                    self.validation_arguments(),
                    lambda a: a.__setitem__(5, a[5].rsplit("/", 1)[0] + "/another-execution"))

    def test_validation_reported_fewer_episodes(self):
        self.reject(self.checks.check_validated_metrics_match_simulation_and_request,
                    [self.receipt, self.params, self.data["raw"]],
                    lambda a: a[0].update(episodes=a[0]["episodes"] - 1))

    def test_validation_identifies_another_checkpoint_version(self):
        output = self.data["proof"]["outputs"]["FineTune"]
        self.reject(self.checks.check_validated_checkpoint_identity,
                    [self.receipt["model_artifact_identity"], output["uri"], output],
                    lambda a: a[0].update(version_id="another-version"))

    def test_training_lineage_claims_another_step_budget(self):
        def change(arguments):
            fields = arguments[0]["training_contract"]["fields"]
            next(row for row in fields if row["field"] == "train_steps")["observed"] = str(
                int(arguments[1]["TrainSteps"]) - 1)
        self.reject(self.checks.check_training_lineage_matches_requested_mode,
                    [self.receipt, self.params, self.data["record"]["request"]["steps"]], change)

    def test_validator_sdk_bundle_is_different(self):
        self.reject(self.checks.check_validation_used_packaged_sdk,
                    [self.receipt["validation_runtime"]],
                    lambda a: a[0].update(sdk_bundle_sha256="0" * 64))

    def attestation_arguments(self):
        return [self.data["proof"]["attestation"], self.receipt["promotion"]["attestation_sha256"],
                self.receipt, self.receipt["training_contract"],
                self.data["proof"]["outputs"]["FineTune"]["uri"], self.data["record"]["execution_arn"]]

    def test_downloaded_attestation_belongs_to_another_execution(self):
        self.reject(self.checks.check_downloaded_attestation_matches_execution,
                    self.attestation_arguments(),
                    lambda a: a[0]["execution"].update(pipeline_execution_id="another-execution"))

    def test_downloaded_attestation_digest_differs_from_publication(self):
        self.reject(self.checks.check_downloaded_attestation_matches_execution,
                    self.attestation_arguments(), lambda a: a.__setitem__(1, "0" * 64))

    def test_promoted_checkpoint_has_another_size(self):
        self.reject(self.checks.check_promoted_checkpoint_matches_validated_artifact,
                    [self.data["proof"]["promoted_identity"], self.data["proof"]["checkpoint_identity"],
                     self.data["proof"]["attestation"], self.receipt],
                    lambda a: a[0].update(bytes=a[0]["bytes"] - 1))

    def test_publication_is_a_stub(self):
        self.reject(self.checks.check_conditional_publication_is_real,
                    [self.receipt["promotion"]], lambda a: a[0].update(local_test_publication=True))

    def test_success_rate_is_below_threshold(self):
        self.reject(self.checks.check_success_gate_threshold, [self.receipt, self.params],
                    lambda a: (a[0].update(success_rate=0.0), a[1].update(SuccessThreshold=0.5)))

    def registration_arguments(self):
        return [self.data["package"], self.params, self.data["record"]["request"]["steps"],
                self.receipt["promotion"], self.receipt["promotion"]["attestation_sha256"],
                self.data["proof"]["outputs"]["FineTune"]["uri"], False]

    def test_model_registration_was_automatically_approved(self):
        self.reject(self.checks.check_registered_model_matches_publication, self.registration_arguments(),
                    lambda a: a[0].update(ModelApprovalStatus="Approved"))

    def test_registration_identifies_another_attestation(self):
        self.reject(self.checks.check_registered_model_matches_publication, self.registration_arguments(),
                    lambda a: a[0]["ModelMetrics"]["ModelQuality"]["Statistics"].update(ContentDigest="sha256:wrong"))

    def test_executed_parameters_differ_from_request(self):
        self.reject(self.checks.check_executed_parameters_match_request,
                    [self.data["execution"]["parameters"], self.params],
                    lambda a: a[0].update(TrainSteps=str(int(self.params["TrainSteps"]) - 1)))

    def test_failed_pipeline_still_contains_all_steps(self):
        self.reject(self.checks.check_requested_managed_steps_succeeded,
                    [self.data["execution"]["execution"], self.data["execution"]["steps"],
                     self.data["record"]["request"]["steps"]],
                    lambda a: a[0].update(PipelineExecutionStatus="Failed"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arena-run-dir", type=Path, help="Local run directory containing verifier JSON and logs")
    parser.add_argument("--managed-run-dir", type=Path, help="Managed operation directory containing verifier downloads")
    parser.add_argument("--output-dir", type=Path, default=Path("local-dev/evidence-regression"),
                        help="Parent for a new timestamped result folder and temporary copies")
    args = parser.parse_args()
    if not any((args.arena_run_dir, args.managed_run_dir)):
        parser.error("Supply at least one evidence input")
    output = args.output_dir.resolve() / dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True)
    suite = unittest.TestSuite()
    inputs = {}
    for kind, source, cls in [
        ("arena", args.arena_run_dir, ArenaEvidenceRejections),
        ("managed", args.managed_run_dir, ManagedEvidenceRejections),
    ]:
        if source:
            cls.source, cls.output = source.resolve(), output
            inputs[kind] = str(cls.source)
            suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(cls))
    print("Testing rejection of damaged evidence; no cloud operations. Results:", output, flush=True)
    result = unittest.TextTestRunner(verbosity=2, resultclass=NamedResult).run(suite)
    summary = {
        "scope": "Verifier regression using copies of saved evidence; no new workload or AWS-state acceptance",
        "inputs": inputs, "tests": result.testsRun, "passed": result.wasSuccessful(),
        "checks": result.checks,
        "failures": [{"test": str(test), "detail": error} for test, error in result.failures + result.errors],
    }
    (output / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
