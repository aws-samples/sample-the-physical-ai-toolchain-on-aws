#!/usr/bin/env python3
"""Execute the supported local VLA test profiles with versioned development S3 handoffs."""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from string import Template
from urllib.parse import urlparse

EXPECTED_STEPS = {"FineTune", "SimEval", "Validate", "SuccessGate"}


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


def git(component, *args):
    repo = component.parent
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={repo}", "-C", str(repo), *args], text=True,
        env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"}
    ).strip()


def stop_containers(run_id):
    proc = subprocess.run(
        ["docker", "ps", "-q", "--filter", f"label=vla.local.run={run_id}"],
        capture_output=True, text=True, timeout=20,
    )
    ids = proc.stdout.split()
    if ids:
        subprocess.run(["docker", "stop", "--time", "15", *ids], timeout=60, check=False)


def assert_success(execution, required=EXPECTED_STEPS):
    required = set(required)
    description = execution.describe()
    rows = execution.list_steps()["PipelineExecutionSteps"]
    present = {row["StepName"] for row in rows}
    failed = [
        {k: row.get(k) for k in ("StepName", "StepStatus", "FailureReason")}
        for row in rows if row["StepStatus"] != "Succeeded"
    ]
    if description["PipelineExecutionStatus"] != "Succeeded" or failed:
        reasons = [
            f"{row['StepName']} {row['StepStatus']}: "
            f"{row['FailureReason'] or description.get('FailureReason') or 'no reason reported'}"
            for row in failed
        ]
        if not reasons:
            reasons = [f"Pipeline {description['PipelineExecutionStatus']}: "
                       f"{description.get('FailureReason') or 'no reason reported'}"]
        if required - present:
            reasons.append("Downstream steps not reached: " + ", ".join(sorted(required - present)))
        raise RuntimeError("; ".join(reasons))
    if not required.issubset(present):
        raise RuntimeError(f"Missing required steps: {sorted(required - present)}")
    if present != required:
        raise RuntimeError(f"Executed steps differ from the request: {sorted(present)}")
    if "SuccessGate" in required:
        gate = next(row for row in rows if row["StepName"] == "SuccessGate")
        if gate.get("Metadata", {}).get("Condition", {}).get("Outcome") not in (True, "True", "true"):
            raise RuntimeError("SuccessGate did not evaluate to true")
    return description, rows


def load_parameters(path, declarations, ecr_registry):
    exported = json.loads(path.read_text())
    raw = {item["Name"]: Template(item["Value"]).substitute(ECR_REGISTRY=ecr_registry)
           for item in exported["PipelineParameters"]}
    if len(raw) != len(exported["PipelineParameters"]):
        raise RuntimeError("Baseline contains duplicate parameter names")
    declared = {p.name: p for p in declarations.values()}
    unknown = set(raw) - set(declared)
    if unknown:
        raise RuntimeError(f"Baseline contains undeclared parameters: {sorted(unknown)}")
    params = {}
    for name, parameter in declared.items():
        if name not in raw:
            if parameter.default_value is None:
                raise RuntimeError(f"Baseline lacks required parameter {name}")
            continue
        kind = parameter.parameter_type.value
        params[name] = {"Integer": int, "Float": float, "String": str}[kind](raw[name])
    for name, value in {
        "ModelFamily": "gr00t", "Suite": "arena_gr1_fridge",
        "TrainSuite": "arena_gr1_fridge", "Gr00tVersion": "n16",
        "UseGrootServer": "true",
    }.items():
        if params.get(name) != value:
            raise RuntimeError(f"Unexpected baseline {name}={params.get(name)!r}")
    if not params.get("ExpectedPolicyConfig"):
        raise RuntimeError("ExpectedPolicyConfig is required")
    if not isinstance(json.loads(params["EvalSimConfig"]), dict):
        raise RuntimeError("EvalSimConfig must encode a JSON object")
    return params


def negative_control(session, role, run_id, required=None):
    from sagemaker.workflow.condition_step import ConditionStep
    from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
    from sagemaker.workflow.fail_step import FailStep
    from sagemaker.workflow.pipeline import Pipeline

    required = required or ["FineTune", "SimEval", "Validate", "SuccessGate"]
    steps = []
    for name in required:
        is_gate = name == required[-1]
        steps.append(ConditionStep(
            name=name,
            conditions=[ConditionGreaterThanOrEqualTo(
                left=0.0 if is_gate else 1.0, right=1.0 if is_gate else 0.0)],
            if_steps=[],
            else_steps=[FailStep(name="ExpectedFailure", error_message="local negative control")]
                if is_gate else [],
            depends_on=[steps[-1]] if steps else [],
        ))
    pipeline = Pipeline(name=f"local-negative-{run_id}", steps=steps,
                        sagemaker_session=session)
    pipeline.create(role_arn=role)
    execution = pipeline.start()
    description = execution.describe()
    actual_steps = execution.list_steps()
    present = {row["StepName"] for row in actual_steps["PipelineExecutionSteps"]}
    if not set(required).issubset(present):
        raise RuntimeError("Negative control did not exercise all required step names")
    if description["PipelineExecutionStatus"] != "Failed":
        raise RuntimeError("Negative control failed to detect an SDK execution failure")
    try:
        assert_success(execution, required)
    except RuntimeError as error:
        reason = str(error)
        if "Missing required steps" in reason:
            raise RuntimeError("Negative control only tested missing names") from error
    else:
        raise RuntimeError("Runner incorrectly accepted the negative control")
    return {"description": description, "steps": actual_steps,
            "required_steps_present": True, "checker_rejected": True,
            "checker_rejection": reason, "control_only": True}


def read_s3(s3, uri, bucket):
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or parsed.netloc != bucket:
        raise RuntimeError(f"Artifact points outside development storage: {uri}")
    return s3.get_object(Bucket=parsed.netloc, Key=parsed.path.lstrip("/"))["Body"].read()


def validated_output_uri(description):
    outputs = description["ProcessingOutputConfig"]["Outputs"]
    # The local pipeline executor mutates this list into a dictionary keyed by
    # OutputName while resolving step properties. Managed describe retains a list.
    if isinstance(outputs, dict):
        destinations = [outputs["validated"]["S3Output"]["S3Uri"]] if "validated" in outputs else []
    else:
        destinations = [
            item["S3Output"]["S3Uri"] for item in outputs if item["OutputName"] == "validated"
        ]
    if len(destinations) != 1:
        raise RuntimeError("Validate did not expose exactly one receipt output")
    return destinations[0].rstrip("/") + "/validated_metrics.json"


def verify_receipt(session, s3, steps, bucket, params):
    step = next(row for row in steps if row["StepName"] == "Validate")
    job = step["Metadata"]["ProcessingJob"]["Arn"]
    output = session.sagemaker_client.describe_processing_job(ProcessingJobName=job)
    return verify_receipt_uri(s3, validated_output_uri(output), bucket, params)


def verify_receipt_uri(s3, receipt_uri, bucket, params):
    from local_profiles import check_training_contract, expected_episodes

    receipt = json.loads(read_s3(s3, receipt_uri, bucket))
    if receipt.get("validation_passed") is not True:
        raise RuntimeError("Validation receipt does not report validation_passed=true")
    if receipt.get("episodes") != expected_episodes(params) or receipt.get("policy_type") != "checkpoint":
        raise RuntimeError("Receipt does not describe the requested real checkpoint episodes")
    check_training_contract(receipt, params)
    runtime = receipt["validation_runtime"]
    if runtime.get("boto3_version") != "1.42.97" or runtime.get("botocore_version") != "1.42.97":
        raise RuntimeError("Validate did not use the packaged SDK")
    if not runtime.get("sdk_bundle_sha256"):
        raise RuntimeError("Validate receipt omitted the SDK bundle identity")
    promotion = receipt["promotion"]
    if promotion.get("local_test_publication"):
        raise RuntimeError("Unexpected test-publication stub")
    artifact_uri = promotion["model_uri"]
    parsed = urlparse(artifact_uri)
    if parsed.scheme != "s3" or parsed.netloc != bucket:
        raise RuntimeError("Promoted artifact is outside development storage")
    head = s3.head_object(Bucket=parsed.netloc, Key=parsed.path.lstrip("/"))
    attestation_bytes = read_s3(s3, promotion["attestation_uri"], bucket)
    actual = "sha256:" + hashlib.sha256(attestation_bytes).hexdigest()
    if actual != promotion["attestation_content_digest"]:
        raise RuntimeError("Attestation digest differs from the validation receipt")
    identity = receipt["model_artifact_identity"]
    source = urlparse(params["CheckpointS3Uri"]) if params.get("CheckpointS3Uri") else None
    if identity["bucket"] != (source.netloc if source else bucket) or identity.get(
        "version_id"
    ) in (None, "", "null"):
        raise RuntimeError("Checkpoint does not have the required S3 version identity")
    if source and identity["key"] != source.path.lstrip("/"):
        raise RuntimeError("Receipt names a different supplied checkpoint")
    return {"receipt_uri": receipt_uri, "receipt": receipt,
            "attestation": json.loads(attestation_bytes), "promoted_artifact_head": head}


class RedactedStream:
    def __init__(self, stream, secret):
        self.stream, self.secret = stream, secret

    def write(self, value):
        self.stream.write(value.replace(self.secret, "[REDACTED_HF_TOKEN]"))
        return len(value)

    def flush(self):
        self.stream.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


def development_config(cfg, bucket, run_id):
    """Reject shared storage before any code staging or preflight writes."""
    if bucket in {cfg.bucket, cfg.handoff_bucket, cfg.trust_bucket}:
        raise RuntimeError("Local development bucket must differ from all managed buckets")
    if not re.fullmatch(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]", bucket):
        raise RuntimeError("Supply a bucket name, not an S3 URI or prefix")
    return dataclasses.replace(
        cfg, bucket=bucket, handoff_bucket=bucket, trust_bucket=bucket,
        pipeline_name=f"vla-local-arena-{run_id}", prefix="localdev/vla-pipeline",
    )


def ensure_images(boto, images, pull, timeout_seconds=7200):
    """Pull missing ECR images only when explicitly requested."""
    import base64

    logged_in = set()
    identities = {}
    for name, image in images.items():
        inspect = subprocess.run(
            ["docker", "image", "inspect", "--format", "{{.Id}}", image],
            capture_output=True, text=True,
        )
        if inspect.returncode:
            if not pull:
                raise RuntimeError(f"Missing {name} image {image}; rerun with --pull-images")
            registry = image.split("/", 1)[0]
            match = re.fullmatch(r"(\d{12})\.dkr\.ecr\.([a-z0-9-]+)\.amazonaws\.com", registry)
            if not match:
                raise RuntimeError(f"Expected a private ECR image, got {image}")
            if registry not in logged_in:
                auth = boto.client("ecr", region_name=match[2]).get_authorization_token(
                    registryIds=[match[1]])["authorizationData"][0]
                username, password = base64.b64decode(auth["authorizationToken"]).decode().split(":", 1)
                subprocess.run(
                    ["docker", "login", "--username", username, "--password-stdin", registry],
                    input=password, text=True, capture_output=True, check=True,
                )
                logged_in.add(registry)
            print(f"[local] Downloading image {image} (limit {timeout_seconds}s)", flush=True)
            subprocess.run(["docker", "pull", image], check=True, timeout=timeout_seconds)
            inspect = subprocess.run(
                ["docker", "image", "inspect", "--format", "{{.Id}}", image],
                capture_output=True, text=True, check=True,
            )
        identities[name] = inspect.stdout.strip()
    return identities


def check_container_identity(image, region, caller, expected_role):
    """Check actual container IMDS credentials before starting either GPU step."""
    probe = (
        "import boto3,json;"
        f"s=boto3.Session(region_name={region!r});"
        "assert s.get_credentials().method == 'iam-role', 'Container must use EC2 instance role';"
        "print(json.dumps(s.client('sts').get_caller_identity()))"
    )
    result = subprocess.run(
        ["docker", "run", "--rm", "--entrypoint", "python3", image, "-c", probe],
        capture_output=True, text=True, timeout=90,
    )
    if result.returncode:
        raise RuntimeError(
            "Container cannot use the expected EC2 role; check IMDSv2 hop limit and network access. "
            + result.stderr[-2000:])
    identity = json.loads(result.stdout)
    expected = f"arn:aws:sts::{caller['Account']}:assumed-role/{expected_role}/"
    if identity["Account"] != caller["Account"] or not identity["Arn"].startswith(expected):
        raise RuntimeError(f"Unexpected container identity: {identity['Arn']}")
    return identity


def check_compose(image, run_dir, run_id, preparation_seconds=600):
    """Probe the installed CLI before adapting the pinned SDK's v2-only detector."""
    from sagemaker.local.image import _SageMakerContainer

    version = subprocess.check_output(
        ["docker", "compose", "version", "--short"], text=True, timeout=30).strip()
    if not re.fullmatch(r"v?(2|5)\.\d+\.\d+(?:[-+].*)?", version):
        raise RuntimeError(f"Untested Docker Compose version: {version}")
    probe = run_dir / "compose-probe.json"
    save(probe, {"services": {"probe": {
        "image": image,
        "entrypoint": ["python", "-c", "print('local Compose probe passed')"],
        "labels": {"vla.local.probe": run_id},
    }}})
    prefix = ["docker", "compose", "--project-name", "vla-probe-" + run_id.lower(),
              "-f", str(probe)]
    report = {"project": "vla-probe-" + run_id.lower(), "phases": [],
              "cleanup_command": shlex.join(["sudo", *prefix, "down"])}

    def phase(name, command, timeout):
        started = time.monotonic()
        row = {"phase": name, "command": command, "timeout_seconds": timeout,
               "status": "Running"}
        report["phases"].append(row)
        stdout_path = run_dir / f"compose-{name}.stdout.log"
        stderr_path = run_dir / f"compose-{name}.stderr.log"
        row.update(stdout=str(stdout_path), stderr=str(stderr_path))
        save(run_dir / "compose-probe-status.json", report)
        print(f"[local] Compose {name}: limit {timeout}s; output in {stdout_path}", flush=True)
        try:
            with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
                with subprocess.Popen(command, stdout=stdout, stderr=stderr) as proc:
                    try:
                        while True:
                            remaining = timeout - (time.monotonic() - started)
                            if remaining <= 0:
                                raise TimeoutError(f"Compose {name} exceeded {timeout}s")
                            try:
                                row["exit_code"] = proc.wait(timeout=min(30, remaining))
                                break
                            except subprocess.TimeoutExpired:
                                print(f"[local] Compose {name}: waiting, "
                                      f"{time.monotonic() - started:.0f}s elapsed", flush=True)
                    except BaseException:
                        # Terminating the CLI does not cancel an in-flight Docker daemon request.
                        proc.terminate()
                        try:
                            proc.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                            proc.wait()
                        raise
            if row["exit_code"]:
                raise RuntimeError(f"Compose {name} exited {row['exit_code']}")
            row["status"] = "Succeeded"
        except BaseException as exc:
            row.update(status="Failed", failure_reason=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            row["elapsed_seconds"] = round(time.monotonic() - started, 3)
            save(run_dir / "compose-probe-status.json", report)

    failure = None
    try:
        phase("prepare", [*prefix, "create"], preparation_seconds)
        phase("cpu-check", [
            *prefix, "up", "--no-recreate", "--abort-on-container-exit",
            "--exit-code-from", "probe"], 120)
        if "local Compose probe passed" not in (run_dir / "compose-cpu-check.stdout.log").read_text():
            raise RuntimeError("Compose probe did not execute its CPU command")
    except Exception as exc:
        failure = exc
    finally:
        try:
            phase("cleanup", [*prefix, "down"], 60)
        except Exception as exc:
            report["cleanup_error"] = str(exc)
            if failure is None:
                failure = exc
        if failure is not None:
            # A timed-out create can complete AFTER down returns. An empty inventory
            # is an observation, not proof that the daemon has cancelled creation.
            report["late_creation_possible"] = report["phases"][0]["status"] != "Succeeded"
            try:
                inventory = subprocess.run(
                    ["docker", "ps", "-a", "--filter",
                     f"label=com.docker.compose.project={report['project']}",
                     "--format", "{{json .}}"],
                    text=True, capture_output=True, timeout=20, check=True)
                report["remaining_containers_observed"] = [
                    json.loads(line) for line in inventory.stdout.splitlines()]
            except Exception as exc:
                report["inventory_error"] = str(exc)
            save(run_dir / "compose-probe-status.json", report)
            print(f"[local] Probe evidence: {run_dir / 'compose-probe-status.json'}. "
                  "After Docker settles, reconcile this exact project with:\n"
                  f"{report['cleanup_command']}", file=sys.stderr, flush=True)
    if failure is not None:
        raise RuntimeError(
            f"{failure}; see {run_dir / 'compose-probe-status.json'} and compose-*.log") from failure
    # The 2.257.6 detector rejects a working v5 plugin because it searches for
    # the literal 'v2'. Keep the actual version and probe in the run evidence.
    _SageMakerContainer._get_compose_cmd_prefix = staticmethod(lambda: ["docker", "compose"])
    return {"version": version, "cpu_probe_exit": 0, "phases": report["phases"],
             "sdk_command_prefix": ["docker", "compose"]}


def main():
    from local_profiles import PROFILES

    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=sorted(PROFILES), default="arena-gr1")
    parser.add_argument("--component", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--parameters", type=Path,
                        default=Path(__file__).with_name("arena-gr1-smoke.json"))
    parser.add_argument("--request", type=Path,
                        help="Resolved vla CLI request; replaces legacy sample recipe selection")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--region", default=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
    parser.add_argument("--development-bucket", required=True)
    parser.add_argument("--expected-role", required=True, help="EC2 instance role name, without ARN/path")
    parser.add_argument("--train-image", help="Override the selected profile's training ECR image")
    parser.add_argument("--eval-image", help="Override the selected profile's evaluation ECR digest URI")
    parser.add_argument("--train-steps", type=int)
    parser.add_argument("--eval-trials", type=int)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--pull-images", action="store_true",
                        help="Authenticate to ECR and pull any missing images")
    parser.add_argument("--timeout-seconds", type=int, default=10800)
    parser.add_argument("--container-preparation-seconds", type=int, default=600,
                        help="Compose container creation budget, separate from its 120s CPU check")
    parser.add_argument("--image-pull-timeout-seconds", type=int, default=7200,
                        help="Per-image download budget; the overall deadline still applies")
    parser.add_argument("--max-runtime-seconds", type=int,
                        help="Per-job training/evaluation runtime budget")
    parser.add_argument("--scratch-root", type=Path,
                        help="Directory on a separate mounted scratch filesystem")
    args = parser.parse_args()
    if args.request and any(value is not None for value in (
        args.train_image, args.eval_image, args.train_steps, args.eval_trials, args.max_runtime_seconds,
    )):
        parser.error("--request is already resolved; do not combine it with recipe overrides")
    if args.region != "us-east-1":
        parser.error("This sample supports only us-east-1; set --region us-east-1.")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]{0,39}", args.run_id):
        parser.error("--run-id must be 1–40 alphanumeric/hyphen characters")
    if args.timeout_seconds < 1 or any(
            value is not None and value < 1
            for value in (args.train_steps, args.eval_trials, args.max_runtime_seconds,
                          args.container_preparation_seconds, args.image_pull_timeout_seconds)):
        parser.error("Timeout, training steps and evaluation trials must be positive")
    os.umask(0o077)
    component = args.component.resolve()
    run_dir = args.run_dir.resolve()
    args.parameters = args.parameters.resolve()
    # systemd opens run.log before invoking Python; no execution evidence may exist yet.
    if run_dir.exists() and {entry.name for entry in run_dir.iterdir()} - {"run.log"}:
        parser.error("Run directory is not empty; choose a fresh run ID/directory")
    run_dir.mkdir(parents=True, exist_ok=True)
    os.environ["AWS_DEFAULT_REGION"] = args.region
    os.environ["VLA_REGION"] = args.region
    os.environ["SAGEMAKER_TELEMETRY_OPT_OUT"] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.path.insert(0, str(component / "src"))
    os.chdir(component)
    sys.dont_write_bytecode = True
    started = time.time()
    state = {"run_id": args.run_id, "profile": args.profile,
             "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
             "status": "Preparing", "development_only": True}
    save(run_dir / "status.json", state)

    def cancel(signum, _frame):
        stop_containers(args.run_id)
        raise TimeoutError(f"Local execution cancelled by signal {signum}")

    signal.signal(signal.SIGTERM, cancel)
    signal.signal(signal.SIGINT, cancel)
    signal.signal(signal.SIGALRM, cancel)
    signal.alarm(args.timeout_seconds)
    execution = None
    disk_measurements = None
    token = ""
    host_lock = None
    try:
        import fcntl
        host_lock = open("/var/lock/vla-local-gpu.lock", "a+")
        try:
            fcntl.flock(host_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("This host is preparing or running another VLA execution; "
                               "inspect that operation before retrying") from exc
        import boto3
        from local_artifacts import install as install_local_artifact_compression
        from local_profiles import execution_parameters, libero_parameters
        from sagemaker.workflow.pipeline_context import LocalPipelineSession

        import vla_pipeline
        from vla_pipeline.common.sourcedir import stage
        from vla_pipeline.config import load_config
        from vla_pipeline.pipeline import build_parameters, build_pipeline
        from vla_pipeline.registry import require_image_capability, resolve, resolve_image_digest
        from vla_pipeline.runner import stage_validate_code, upload_code, upload_directory
        from vla_pipeline.validation_sdk import sdk_bundle

        actual_module = Path(vla_pipeline.__file__).resolve()
        if not actual_module.is_relative_to(component):
            raise RuntimeError(f"Wrong package imported: {actual_module}")
        from vla_pipeline.source import identity as runtime_source_identity
        commit = runtime_source_identity(component.parent)
        if commit != args.expected_commit:
            raise RuntimeError(f"Unexpected canonical commit {commit}")
        versions = {name: importlib.metadata.version(name)
                    for name in ("sagemaker", "boto3", "botocore", "docker")}
        if versions["sagemaker"] != "2.257.6" or versions["boto3"] != "1.43.73":
            raise RuntimeError(f"Unexpected SDK versions: {versions}")
        if not Path("/usr/bin/pigz").is_file():
            raise RuntimeError("Local artifact compression requires /usr/bin/pigz")
        boto = boto3.Session(region_name=args.region)
        caller = boto.client("sts").get_caller_identity()
        if boto.get_credentials().method != "iam-role":
            raise RuntimeError("Use EC2 instance-role credentials; unset AWS_PROFILE and AWS_* keys")
        if not caller["Arn"].startswith(
                f"arn:aws:sts::{caller['Account']}:assumed-role/{args.expected_role}/"):
            raise RuntimeError(f"Unexpected AWS identity {caller['Arn']}")
        bucket = args.development_bucket
        cfg = development_config(load_config(region=args.region), bucket, args.run_id)
        s3 = boto.client("s3")
        if s3.get_bucket_versioning(Bucket=bucket).get("Status") != "Enabled":
            raise RuntimeError("Development bucket versioning is not enabled")
        marker = f"preflight/{args.run_id}/{time.time_ns()}.json"
        created = s3.put_object(Bucket=bucket, Key=marker,
                               Body=b'{"purpose":"local-preflight"}', IfNoneMatch="*")
        if created.get("VersionId") in (None, "", "null"):
            raise RuntimeError("Development bucket did not return a real VersionId")
        s3.get_object(Bucket=bucket, Key=marker, VersionId=created["VersionId"])["Body"].close()
        upload = s3.create_multipart_upload(Bucket=bucket, Key=marker + ".multipart")
        s3.abort_multipart_upload(Bucket=bucket, Key=marker + ".multipart",
                                  UploadId=upload["UploadId"])
        token = os.environ.get("HF_TOKEN") or boto.client("secretsmanager").get_secret_value(
            SecretId=cfg.hf_secret_name)["SecretString"].strip()
        if not token:
            raise RuntimeError("Hugging Face token is empty")
        sys.stdout, sys.stderr = RedactedStream(sys.stdout, token), RedactedStream(sys.stderr, token)

        session = LocalPipelineSession(boto_session=boto, default_bucket=bucket)
        if not session.sagemaker_client.__class__.__module__.startswith("sagemaker.local"):
            raise RuntimeError("Expected a local SageMaker client")
        scratch_layout = None
        if args.scratch_root:
            from local_scratch import DiskMeasurements, install_mounts, prepare

            scratch_layout = prepare(args.scratch_root, run_dir, args.profile)
            install_mounts(scratch_layout, run_dir)
            disk_measurements = DiskMeasurements(scratch_layout, run_dir).start()
        elif args.profile != "arena-gr1":
            raise RuntimeError("The LIBERO sample profiles require --scratch-root")
        container_root = run_dir / "containers"
        container_root.mkdir(exist_ok=True)
        session.config = {"local": {
            "local_code": True, "container_root": str(container_root),
            "container_config": {"shm_size": "8g", "labels": {"vla.local.run": args.run_id}},
        }}
        ecr_registry = f"{caller['Account']}.dkr.ecr.{args.region}.amazonaws.com"
        family, simulator, _ = PROFILES[args.profile]
        request_digest = None
        request = None
        requested_steps = ["FineTune", "SimEval", "Validate", "SuccessGate"]
        graph_options = {}
        if args.request:
            from vla_pipeline.launch_request import local_parameters

            request_bytes = args.request.read_bytes()
            request = json.loads(request_bytes)
            request_digest = hashlib.sha256(request_bytes).hexdigest()
            save(run_dir / "resolved-request.json", request)
            if request.get("schema_version") != 1:
                raise RuntimeError("Unsupported resolved request schema")
            if request["selection"]["local_profile"] != args.profile:
                raise RuntimeError("Resolved request does not match the local profile")
            if request["account_id"] != caller["Account"]:
                raise RuntimeError("Resolved request targets a different AWS account")
            requested_steps = request["steps"]
            graph_options = {"through": requested_steps[-1],
                             "checkpoint_s3_uri": request.get("checkpoint_s3")}
            params = local_parameters(request, build_parameters(**graph_options))
        elif args.profile == "arena-gr1":
            params = load_parameters(args.parameters, build_parameters(), ecr_registry)
        else:
            params = libero_parameters(
                args.profile, build_parameters(), ecr_registry, resolve_image_digest,
                train_image=args.train_image, eval_image=args.eval_image)
        for name, value in {
            "TrainImageUri": args.train_image, "EvalImageUri": args.eval_image,
            "TrainSteps": args.train_steps, "EvalTrials": args.eval_trials,
            "MaxRuntimeSeconds": args.max_runtime_seconds,
        }.items():
            if value is not None:
                params[name] = value
        save(run_dir / "negative-control.json", negative_control(
            session, cfg.role_arn, args.run_id, requested_steps))
        if "SimEval" in requested_steps and "@sha256:" not in params["EvalImageUri"]:
            raise RuntimeError("--eval-image must identify an immutable ECR digest")
        if "FineTune" in requested_steps and params["Gr00tVersion"] == "n16" and (
            params["TrainSteps"] > params["TrainSaveSteps"]
        ):
            raise RuntimeError("N1.6 supports a final checkpoint only; increase TrainSaveSteps")
        for step, key in (("FineTune", "TrainInstanceType"), ("SimEval", "EvalInstanceType")):
            if step in requested_steps:
                params[key] = "local_gpu"
        from vla_pipeline.local_deadline import install as install_job_deadline

        install_job_deadline(session.sagemaker_client, params["MaxRuntimeSeconds"])
        print("[local] Effective recipe: " + json.dumps({
            name: params[name] for name in (
                "ModelFamily", "Gr00tVersion", "Suite", "TrainSuite", "TrainSteps",
                "EvalTrials", "EvalSeed", "SuccessThreshold", "TrainImageUri", "EvalImageUri")
            if name in params
        }), flush=True)
        sources = {}
        for step, label, parameter in (("FineTune", "train", "TrainSourceDirUri"),
                                       ("SimEval", "eval", "EvalSourceDirUri")):
            if step not in requested_steps:
                continue
            source_link = run_dir / f"source-{label}"
            destination = ((Path(scratch_layout["work"]) / f"source-{label}")
                           if scratch_layout else source_link)
            if destination.exists() or source_link.exists() or source_link.is_symlink():
                raise RuntimeError(f"Use a fresh run directory: {destination} already exists")
            shutil.move(stage(family, repo_root=str(component)), destination)
            if scratch_layout:
                source_link.symlink_to(destination, target_is_directory=True)
            sources[label] = upload_directory(cfg, str(destination))
            params[parameter] = json.dumps("file://" + str(destination))
        validate_uri = None
        if "Validate" in requested_steps:
            validate_path = stage_validate_code(repo_root=str(component))
            validate_uri = upload_code(cfg, validate_path)
            shutil.rmtree(Path(validate_path).parent)
        pipeline = build_pipeline(cfg, session=session, validate_code_uri=validate_uri, **graph_options)
        for step in pipeline.steps:
            if step.name == "SuccessGate":
                step.if_steps = []
            estimator = getattr(step, "estimator", None)
            if estimator is not None:
                estimator.environment["HF_TOKEN"] = token
                if not estimator.output_path.startswith(f"s3://{bucket}/"):
                    raise RuntimeError(f"Step {step.name} writes outside development storage")
        definition = pipeline.definition()
        save(run_dir / "pipeline-definition.redacted.json",
             json.loads(definition.replace(token, "[REDACTED_HF_TOKEN]")))
        if "RegisterModel-RegisterModel" in definition:
            raise RuntimeError("Registration was not removed from the local graph")
        image_references = {}
        for step, label, key in (("FineTune", "train", "TrainImageUri"),
                                  ("SimEval", "eval", "EvalImageUri")):
            if step in requested_steps:
                image_references[label] = params[key]
        if "Validate" in requested_steps:
            validate_definition = next(
                step for step in json.loads(definition)["Steps"] if step["Name"] == "Validate")
            image_references["validate"] = validate_definition["Arguments"]["AppSpecification"]["ImageUri"]
        probe_image = image_references.get("validate") or next(iter(image_references.values()))
        state["activity"] = "Checking/downloading images"
        save(run_dir / "status.json", state)
        image_ids = ensure_images(boto, image_references, args.pull_images,
                                  args.image_pull_timeout_seconds)
        if scratch_layout:
            from local_scratch import check_space, initial_budget

            check_space(scratch_layout, phase="before starting the requested pipeline",
                        minimums=initial_budget(args.profile, requested_steps), run_dir=run_dir)
        elif shutil.disk_usage(container_root).free < 180 * 1024**3:
            raise RuntimeError("Require 180 GiB free on the single working filesystem "
                               "before starting the requested pipeline")
        state["activity"] = "Preparing Compose container and checking CPU execution"
        save(run_dir / "status.json", state)
        compose_probe = check_compose(probe_image, run_dir, args.run_id,
                                      args.container_preparation_seconds)
        state["activity"] = "Checking image capability, container identity and GPU"
        save(run_dir / "status.json", state)
        capability = resolve(family, simulator).required_image_capability
        if capability and "SimEval" in requested_steps:
            require_image_capability(params["EvalImageUri"], capability)
        container_identity = check_container_identity(
            probe_image, args.region, caller, args.expected_role)
        gpu_probe = subprocess.run(
            ["docker", "run", "--rm", "--gpus", "all", "--entrypoint", "nvidia-smi",
             image_references.get("train") or image_references["eval"],
             "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=60, check=True,
        ).stdout.strip()
        install_local_artifact_compression(run_dir)
        start_parameters = execution_parameters(params, build_parameters(**graph_options))
        save(run_dir / "parameters.json", params)
        save(run_dir / "execution-parameters.json", start_parameters)
        save(run_dir / "manifest.json", {
            "run_id": args.run_id, "profile": args.profile,
            "requested_steps": requested_steps,
            "complete_training_workflow": set(requested_steps) == EXPECTED_STEPS,
            "checkpoint_identity": request.get("checkpoint_identity") if request else None,
            "canonical_commit": commit, "package_file": str(actual_module),
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "sdk_versions": versions, "caller": caller, "development_bucket": bucket,
            "region": args.region, "container_identity": container_identity,
            "container_gpu": gpu_probe,
            "source_archives": sources, "validator_source": validate_uri,
            "validation_sdk_bundle_sha256": (
                hashlib.sha256(sdk_bundle()).hexdigest() if "Validate" in requested_steps else None),
            "baseline_parameters_sha256": (
                hashlib.sha256(args.parameters.read_bytes()).hexdigest()
                if args.profile == "arena-gr1" and not args.request else None),
            "resolved_request_sha256": request_digest,
            "parameter_source": (
                "vla_cli_request" if args.request else
                "managed_arena_n16_sample" if args.profile == "arena-gr1"
                else "registry_and_run_libero_train_default"),
            "profile_code_sha256": hashlib.sha256(
                Path(__file__).with_name("local_profiles.py").read_bytes()).hexdigest(),
            "systemd_unit": os.environ.get("VLA_LOCAL_SYSTEMD_UNIT"),
            "compose_probe": compose_probe,
            "images": image_references, "image_ids": image_ids,
            "arena_overlays": [], "shared_memory": "8g",
            "artifact_compression": {"implementation": "pigz", "level": 1, "workers": 8},
            "local_artifact_helper_sha256": hashlib.sha256(
                (Path(__file__).parent / "local_artifacts.py").read_bytes()).hexdigest(),
            "timeout_seconds": args.timeout_seconds,
            "container_preparation_seconds": args.container_preparation_seconds,
            "image_pull_timeout_seconds": args.image_pull_timeout_seconds,
            "scratch": scratch_layout,
        })
        print(f"[local] canonical={commit} caller={caller['Arn']} bucket={bucket}", flush=True)
        print(f"[local] sources={sources} validator={validate_uri}", flush=True)
        print("[local] Negative-control failure, configuration, S3 version reads and multipart cleanup passed", flush=True)
        if args.preflight_only:
            state["status"] = "PreflightSucceeded"
            state["free_gib"] = round(shutil.disk_usage(container_root).free / 1024**3, 1)
            state["disk_budget_gib"] = (
                initial_budget(args.profile, requested_steps) if scratch_layout else {"root": 180})
            return 0
        active = subprocess.check_output(["docker", "ps", "-q"], text=True).strip()
        if active:
            raise RuntimeError("Unexpected existing running containers; refusing a competing GPU run")
        pipeline.create(role_arn=cfg.role_arn)
        state["status"] = "Running"
        state["activity"] = " → ".join(requested_steps)
        state["preparation_seconds"] = round(time.time() - started, 1)
        save(run_dir / "status.json", state)
        print("[local] Starting requested pipeline: " + state["activity"], flush=True)
        execution = pipeline.start(parameters=start_parameters)
        state["pipeline_seconds"] = round(time.time() - started - state["preparation_seconds"], 1)
        save(run_dir / "execution.json", execution.describe())
        save(run_dir / "steps.json", execution.list_steps())
        _, steps = assert_success(execution, requested_steps)
        outputs = {}
        for step in steps:
            if step["StepName"] not in {"FineTune", "SimEval"}:
                continue
            job = step["Metadata"]["TrainingJob"]["Arn"]
            description = session.sagemaker_client.describe_training_job(TrainingJobName=job)
            uri = description["ModelArtifacts"]["S3ModelArtifacts"]
            parsed = urlparse(uri)
            if parsed.scheme != "s3" or parsed.netloc != bucket:
                raise RuntimeError(f"Job output is outside development storage: {uri}")
            head = s3.head_object(Bucket=bucket, Key=parsed.path.lstrip("/"))
            if head.get("VersionId") in (None, "", "null") or head["ContentLength"] <= 0:
                raise RuntimeError(f"Job output is empty or unversioned: {uri}")
            outputs[step["StepName"]] = {
                "job_name": job, "uri": uri, "version_id": head["VersionId"],
                "etag": head["ETag"], "bytes": head["ContentLength"],
            }
        save(run_dir / "job-outputs.json", outputs)
        if "Validate" in requested_steps:
            checked = verify_receipt(session, s3, steps, bucket, params)
            expected_sdk = hashlib.sha256(sdk_bundle()).hexdigest()
            if checked["receipt"]["validation_runtime"]["sdk_bundle_sha256"] != expected_sdk:
                raise RuntimeError("Receipt SDK bundle differs from the staged validator")
            if request and request.get("checkpoint_identity"):
                actual = checked["receipt"]["model_artifact_identity"]
                expected = request["checkpoint_identity"]
                if actual["version_id"] != expected["version_id"] or actual["etag"].strip('"') != expected["etag"].strip('"'):
                    raise RuntimeError("Supplied checkpoint changed after CLI preparation")
            save(run_dir / "verified-receipt.json", checked)
            state.update(receipt_uri=checked["receipt_uri"],
                         success_rate=checked["receipt"]["success_rate"])
        state.update(status="Succeeded", outputs=outputs, requested_steps=requested_steps,
                     complete_training_workflow=set(requested_steps) == EXPECTED_STEPS)
        print("[local] REQUESTED STEPS SUCCEEDED: " + " → ".join(requested_steps), flush=True)
        return 0
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        if token:
            reason = reason.replace(token, "[REDACTED_HF_TOKEN]")
        state.update(status="Failed", failure_reason=reason)
        if execution is not None:
            save(run_dir / "execution.json", execution.describe())
            save(run_dir / "steps.json", execution.list_steps())
        print(f"[local] FAILED: {state['failure_reason']}", file=sys.stderr, flush=True)
        return 1
    finally:
        signal.alarm(0)
        if disk_measurements is not None:
            disk_measurements.stop()
        try:
            stop_containers(args.run_id)
        except Exception as cleanup_error:
            state["container_cleanup_error"] = str(cleanup_error)
        state.update(ended_at=dt.datetime.now(dt.timezone.utc).isoformat(),
                     elapsed_seconds=round(time.time() - started, 1))
        save(run_dir / "status.json", state)


if __name__ == "__main__":
    raise SystemExit(main())
