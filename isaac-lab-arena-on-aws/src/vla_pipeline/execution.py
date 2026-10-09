"""CLI adapters over the existing managed pipeline and detached local worker."""
from __future__ import annotations

import dataclasses
import contextlib
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from urllib.parse import urlparse

from .deployment import aws_session, discover, image_digest, local_instance
from .operations import OperationBusy, Store, new_run_id, timestamp, validate_name, write_json
from .cli_context import command_name

COMPONENT = Path(__file__).resolve().parents[2]


def source_identity():
    from .source import identity
    return identity()


def prepare(request, deployment, check_remote=True):
    """Select immutable images; retain the same dictionary for display and execution."""
    from .registry import check_storage_fits, require_image_capability, resolve

    if deployment.get("request") and deployment.get("status") != "Ready":
        raise ValueError("Deployment preparation is not complete; inspect it and resume before running")
    request = json.loads(json.dumps(request))
    images = {**deployment.get("images", {}).get(request["cell"], {}),
              **request.pop("image_overrides")}
    local = deployment.get("local", {})
    on_host = False
    if request["mode"] == "local":
        missing = [key for key in ("development_bucket", "expected_role", "scratch_root",
                                  "instance_id", "host_region") if not local.get(key)]
        if missing:
            raise ValueError("Local deployment lacks " + ", ".join(missing)
                             + f"; select these with {command_name()} deploy --use-existing")
        if check_remote and sys.platform == "linux":
            try:
                identity = local_instance()
                on_host = (identity["accountId"], identity["instanceId"], identity["region"]) == (
                    deployment["account_id"], local["instance_id"], local["host_region"])
            except (OSError, ValueError):
                pass
        if not on_host and local.get("remote_ready"):
            request["transport"] = "ssm"
        elif check_remote and not on_host:
            raise ValueError("Run this command on the selected EC2 host, or prepare it with "
                             f"{command_name()} deploy --prepare-host for remote execution")
        else:
            request.pop("transport", None)
        request["local_target"] = dict(local)
    session = None
    if check_remote:
        # A local worker must use its instance role, not the provisioning profile.
        session = aws_session(
            deployment.get("profile") if request["mode"] == "managed" or not on_host else None,
            deployment["region"],
        )
        caller = session.client("sts").get_caller_identity()
        if caller["Account"] != deployment["account_id"]:
            raise ValueError(f"Wrong caller account {caller['Account']}; expected "
                             f"{deployment['account_id']}")
        if request["mode"] == "local" and on_host:
            role = local["expected_role"]
            if session.get_credentials().method != "iam-role" or not role or not caller["Arn"].startswith(
                f"arn:aws:sts::{caller['Account']}:assumed-role/{role}/"
            ):
                raise ValueError("Run locally on the prepared EC2 host using its configured instance role")
            identity = local_instance()
            if (identity["accountId"], identity["instanceId"], identity["region"]) != (
                deployment["account_id"], local["instance_id"], local["host_region"],
            ):
                raise ValueError("This host differs from the selected EC2 instance/account/host region")
            request["host"] = identity
        elif request.get("transport") == "ssm":
            instances = session.client("ec2", region_name=local["host_region"]).describe_instances(
                InstanceIds=[local["instance_id"]])["Reservations"]
            instance = instances[0]["Instances"][0]
            if instance["State"]["Name"] != "running":
                raise ValueError("Selected EC2 host is not running; start it before submitting")
        cfg = discover(session, deployment["project"])
        if cfg.account_id != deployment["account_id"]:
            raise ValueError("Discovered deployment belongs to another account")
    params = request["parameters"]
    for step, parameter in (("FineTune", "TrainImageUri"), ("SimEval", "EvalImageUri")):
        if step in request["steps"]:
            if step not in images:
                raise ValueError(f"No {step} image selected for {request['cell']}; use {command_name()} deploy")
            if check_remote:
                images[step] = image_digest(session, images[step])
            else:
                from .registry import parse_ecr_reference
                ref = parse_ecr_reference(images[step])
                if not ref.account or ref.region != "us-east-1" or not ref.digest:
                    raise ValueError("Offline plans require full us-east-1 ECR digest URIs; "
                                     f"select images with {command_name()} deploy")
            params[parameter] = images[step]
    if check_remote and "SimEval" in request["steps"]:
        spec = resolve(request["selection"]["model"], request["selection"]["simulator"])
        if spec.required_image_capability:
            require_image_capability(images["SimEval"], spec.required_image_capability,
                                     ecr_client=session.client("ecr"))
        if request.get("checkpoint_s3"):
            uri = urlparse(request["checkpoint_s3"])
            head = session.client("s3").head_object(Bucket=uri.netloc, Key=uri.path.lstrip("/"))
            if head.get("VersionId") in (None, "", "null"):
                raise ValueError("The supplied checkpoint requires a real S3 VersionId")
            request["checkpoint_identity"] = {
                "version_id": head["VersionId"], "etag": head["ETag"],
            }
    if check_remote and request["mode"] == "managed":
        for step, instance_key, volume_key in (
            ("FineTune", "TrainInstanceType", "VolumeSizeInGB"),
            ("SimEval", "EvalInstanceType", "EvalVolumeSizeInGB"),
        ):
            if step in request["steps"]:
                check_storage_fits(params[instance_key], params[volume_key], session.client("ec2"),
                                   require_gpu=True)
    from .pipeline import build_parameters

    declarations = build_parameters(through=request["steps"][-1],
                                    checkpoint_s3_uri=request.get("checkpoint_s3"))
    declared = {parameter.name for parameter in declarations.values()}
    request["parameters"] = {key: value for key, value in params.items() if key in declared}
    request["complete_training_workflow"] = request["steps"] == (
        ["FineTune", "SimEval", "Validate", "SuccessGate"]
        + (["RegisterModel"] if request["mode"] == "managed" else []))
    request.update(account_id=deployment["account_id"], region=deployment["region"],
                   deployment=deployment["id"], images=images,
                   checks=("Identity, deployment parameters and published images checked; "
                           "container/runtime checks happen in the worker"
                           if check_remote else "offline; AWS not checked"))
    return request


def submit(request, deployment, store, run_id=None):
    run_id = validate_name(run_id or new_run_id())
    commit = source_identity()
    if request.get("transport") == "ssm" and (
        request["local_target"].get("source_commit") != commit
    ):
        raise ValueError("Prepared host has a different commit; use deploy --prepare-host "
                         "with a new deployment name for this checkout before running")
    if request["mode"] == "local" and (COMPONENT / "local-dev/runs" / run_id).exists():
        raise ValueError(f"Local run directory already exists for {run_id}; use a new run ID")
    record = {"id": run_id, "cell": request["cell"], "mode": request["mode"],
              "deployment": deployment["id"], "account_id": deployment["account_id"],
              "region": deployment["region"], "profile": deployment.get("profile"),
              "project": deployment["project"],
              "component": str(COMPONENT), "source_commit": commit,
              "request": request, "group": request.get("group"),
              "status": "Submitting", "created_at": timestamp()}
    from .cli_settings import configuration_receipt
    record["configuration_receipt"] = configuration_receipt(request)
    if deployment.get("lifecycle"):
        with store.lock("deployments", deployment["id"], wait=True):
            current_deployment = store.load("deployments", deployment["id"])
            if current_deployment["status"] != "Ready":
                raise ValueError("The deployment is no longer Ready; no new run was submitted")
            directory = store.create_run(record)
            current_deployment.setdefault("run_ids", []).append(run_id)
            store.save("deployments", current_deployment)
            # Teardown takes this same lock. Keep it until the external
            # submission response (or its recovery receipt) has been saved.
            return _submit_record(request, current_deployment, record, directory, store)
    else:
        directory = store.create_run(record)
    return _submit_record(request, deployment, record, directory, store)


def _submit_record(request, deployment, record, directory, store):
    run_id = record["id"]
    if request.get("transport") == "ssm":
        try:
            from .remote_execution import exchange
            return exchange(record, "run", store)
        except BaseException as exc:
            with store.lock("runs", run_id, wait=True):
                record = store.load("runs", run_id)
                record["submission_error"] = f"{type(exc).__name__}: {exc}"
                if record["status"] == "Submitting":
                    record.update(status="SubmissionError", failure_reason=record["submission_error"],
                                  activity="Submission was interrupted. Follow this run to check "
                                           "its saved SSM command; do not submit a duplicate.")
                store.save("runs", record)
            raise
    with store.lock("runs", run_id, wait=True):
        record = store.load("runs", run_id)
        try:
            if request["mode"] == "local":
                record.update(unit=f"vla-local-{run_id}",
                              run_dir=str(COMPONENT / "local-dev/runs" / run_id))
                store.save("runs", record)
                submit_local(request, deployment, record, directory)
            else:
                submit_managed(request, deployment, record, directory, store)
        except BaseException as exc:
            record.update(status="SubmissionError", failure_reason=f"{type(exc).__name__}: {exc}",
                          activity="Submission did not finish normally. Inspect this recorded run "
                                   "before retrying; remote work may have started.")
            store.save("runs", record)
            raise
        store.save("runs", record)
    return record


def submit_local(request, deployment, record, directory):
    local = deployment.get("local", {})
    for name in ("development_bucket", "expected_role", "scratch_root"):
        if not local.get(name):
            raise ValueError(f"Local deployment lacks {name}; select it with {command_name()} deploy --use-existing")
    if sys.platform != "linux":
        raise ValueError("The local worker must execute on its selected EC2 host")
    request_path = directory / "request.json"
    write_json(request_path, request)
    # Total guard includes both bounded GPU jobs plus preparation and CPU work.
    limits = request["local_limits"]
    total = limits["total_timeout_seconds"]
    command = [
        "bash", str(COMPONENT / "scripts/local/launch.sh"), record["id"],
        "--request", str(request_path), "--profile", request["selection"]["local_profile"],
        "--region", deployment["region"], "--development-bucket", local["development_bucket"],
        "--expected-role", local["expected_role"], "--scratch-root", local["scratch_root"],
        "--pull-images", "--timeout-seconds", str(total),
        "--image-pull-timeout-seconds", str(limits["image_pull_timeout_seconds"]),
        "--container-preparation-seconds", str(limits["container_preparation_seconds"]),
    ]
    environment = {**os.environ, "VLA_LOCAL_PYTHON": sys.executable,
                   "VLA_FOUNDATION_PROJECT": deployment["project"]}
    result = subprocess.run(command, env=environment, text=True, capture_output=True)
    (directory / "submission.log").write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    record.update(status="Submitted", local_total_timeout_seconds=total)


def submit_managed(request, deployment, record, directory, store):
    import boto3
    from sagemaker.workflow.pipeline_context import PipelineSession

    from .common.sourcedir import stage
    from .pipeline import build_pipeline
    from .runner import stage_validate_code, upload_code, upload_directory, upsert_versioned

    session = aws_session(deployment.get("profile"), deployment["region"])
    cfg = dataclasses.replace(discover(session, deployment["project"]),
                              pipeline_name=f"vla-{record['id']}")
    sm = session.client("sagemaker")
    owned = bool(deployment.get("lifecycle"))
    if owned:
        # A native run gets a new pipeline. Never update a similarly named
        # pipeline belonging to another machine or deployment.
        existing = [row for page in sm.get_paginator("list_pipelines").paginate(
            PipelineNamePrefix=cfg.pipeline_name) for row in page["PipelineSummaries"]
                    if row["PipelineName"] == cfg.pipeline_name]
        if existing:
            raise ValueError("A pipeline already uses this run ID. Choose another --run-id; it was not changed.")
        record["resource_owner_token"] = uuid.uuid4().hex
        record["pipeline_name"] = cfg.pipeline_name
        store.save("runs", record)
    tags = [{"Key": "pai-arena-owner", "Value": record["resource_owner_token"]}] if owned else []
    # Existing upload helpers use boto3.client; bind them to this explicit session.
    previous = boto3.DEFAULT_SESSION
    boto3.DEFAULT_SESSION = session
    try:
        source_directory = stage(request["selection"]["model"], repo_root=str(COMPONENT))
        try:
            source_uri = upload_directory(cfg, source_directory)
        finally:
            shutil.rmtree(source_directory)
        validate_uri = None
        if "Validate" in request["steps"]:
            validate_file = stage_validate_code(repo_root=str(COMPONENT))
            try:
                validate_uri = upload_code(cfg, validate_file)
            finally:
                shutil.rmtree(Path(validate_file).parent)
        pipeline = build_pipeline(
            cfg, session=PipelineSession(boto_session=session, default_bucket=cfg.bucket),
            validate_code_uri=validate_uri, checkpoint_s3_uri=request.get("checkpoint_s3"),
            through=request["steps"][-1],
        )
        if "RegisterModel" in request["steps"]:
            group = request["parameters"]["ModelPackageGroupName"]
            try:
                sm.describe_model_package_group(ModelPackageGroupName=group)
            except sm.exceptions.ClientError as exc:
                error = exc.response["Error"]
                if error["Code"] not in {
                    "ResourceNotFound", "ResourceNotFoundException", "ValidationException",
                } or (
                    error["Code"] == "ValidationException"
                    and "does not exist" not in error["Message"].lower()
                ):
                    raise
                if owned:
                    record["created_model_group"] = {
                        "name": group, "creation_requested": True, "confirmed": False,
                        "arn": f"arn:aws:sagemaker:{record['region']}:{record['account_id']}:model-package-group/{group}"}
                    store.save("runs", record)
                sm.create_model_package_group(ModelPackageGroupName=group, **({"Tags": tags} if owned else {}))
                if owned:
                    record["created_model_group"]["confirmed"] = True
                    store.save("runs", record)
        params = dict(request["parameters"])
        if "SimEval" in request["steps"]:
            params["EvalSourceDirUri"] = source_uri
        if "FineTune" in request["steps"]:
            params["TrainSourceDirUri"] = source_uri
        declared = {parameter.name for parameter in pipeline.parameters}
        unknown = params.keys() - declared
        if unknown:
            raise ValueError(f"Resolved request contains undeclared pipeline parameters: {sorted(unknown)}")
        missing = {p.name for p in pipeline.parameters if p.default_value is None} - params.keys()
        if missing:
            raise ValueError(f"Resolved request lacks required pipeline parameters: {sorted(missing)}")
        if owned:
            record["pipeline_creation_requested"] = True
            store.save("runs", record)
            version = pipeline.create(role_arn=cfg.role_arn, tags=tags)
            if "PipelineVersionId" not in version:
                version = pipeline.update(role_arn=cfg.role_arn)
        else:
            version = upsert_versioned(pipeline, cfg.role_arn)
        values = [{"Name": key, "Value": str(value)} for key, value in params.items()]
        write_json(directory / "submitted-parameters.json", values)
        record.update(pipeline_name=cfg.pipeline_name, pipeline_version=version["PipelineVersionId"],
                      source_uri=source_uri, validate_uri=validate_uri,
                      client_request_token=uuid.uuid4().hex)
        store.save("runs", record)
        result = sm.start_pipeline_execution(
            PipelineName=cfg.pipeline_name, PipelineVersionId=version["PipelineVersionId"],
            PipelineParameters=values, ClientRequestToken=record["client_request_token"],
            PipelineExecutionDisplayName=record["id"],
        )
        record.update(status="Submitted", execution_arn=result["PipelineExecutionArn"])
    finally:
        boto3.DEFAULT_SESSION = previous


def read_local(path):
    try:
        return json.loads(path.read_text())
    except PermissionError:
        return json.loads(subprocess.check_output(
            ["sudo", "-n", "cat", str(path)], text=True, stderr=subprocess.PIPE))


def require_execution_host(record):
    expected = record["request"].get("host")
    if expected is None or local_instance() != expected:
        raise ValueError("Use the EC2 host recorded for this execution to inspect or cancel it")


def check_managed_steps(execution, rows, required):
    """A green API status must contain the requested graph, including its gate."""
    if execution["PipelineExecutionStatus"] != "Succeeded":
        return False, execution.get("FailureReason")
    by_name = {row["StepName"]: row for row in rows}
    expected_names = {"RegisterModel-RegisterModel" if name == "RegisterModel" else name
                      for name in required}
    if set(by_name) != expected_names or len(rows) != len(expected_names):
        return False, "Executed step names differ from the requested graph"
    missing = []
    for name in required:
        actual = "RegisterModel-RegisterModel" if name == "RegisterModel" else name
        if actual not in by_name or by_name[actual]["StepStatus"] != "Succeeded":
            missing.append(name)
    if missing:
        return False, "Requested steps did not all succeed: " + ", ".join(missing)
    if "SuccessGate" in required:
        outcome = by_name["SuccessGate"].get("Metadata", {}).get("Condition", {}).get("Outcome")
        # The managed API uses the string enum "True"/"False"; the local SDK
        # uses a boolean. These are deliberately separate status adapters.
        if outcome != "True":
            return False, f"SuccessGate did not report a true outcome: {outcome!r}"
    return True, None


def local_activity(record, root):
    """Read bounded new log data using the existing waiter's observed markers."""
    from .backend import script

    log = root / "run.log"
    progress = dict(record.get("log_observation", {}))
    try:
        size = log.stat().st_size
        position = progress.get("position", 0)
        if position > size:
            progress, position = {}, 0
        with log.open("rb") as stream:
            stream.seek(position)
            data = stream.read(512 * 1024)
        boundary = data.rfind(b"\n")
        if boundary < 0:
            return
        progress["position"] = position + boundary + 1
        for line in data[:boundary].decode(errors="replace").splitlines():
            if ("[local] Starting FineTune → SimEval" in line
                    or "[local] Starting requested pipeline:" in line):
                progress["pipeline_started"] = True
            if not progress.get("pipeline_started"):
                continue
            match = re.search(r"Starting pipeline step: '(FineTune|SimEval|Validate|SuccessGate)'", line)
            if match:
                progress.update(step=match.group(1), reported=None)
            else:
                activity = script("local/wait_run.py").reported_activity(line)
                if activity:
                    progress["reported"] = activity
        record["log_observation"] = progress
        if record["status"] == "Running" and progress.get("step"):
            record["activity"] = "Latest step started: " + progress["step"]
            if progress.get("reported"):
                record["activity"] += "; last reported: " + progress["reported"]
    except OSError:
        # A non-root observer may only read status through sudo; its follower
        # retains the existing privileged waiter and full log handling.
        return


def observe(record, store=None):
    if record["request"].get("transport") == "ssm":
        if store is None:
            raise ValueError("Remote observations require the saved operation store")
        from .remote_execution import exchange
        # Reconcile an interrupted submission before attempting another action.
        submission = record.get("remote_actions", {}).get("run", {})
        if (record.get("status") in {"Submitting", "SubmissionError"}
                and not submission.get("response_received")
                and submission.get("status") not in {"Failed", "TimedOut", "Cancelled",
                                                     "DeliveryTimedOut", "ExecutionTimedOut"}):
            return exchange(record, "run", store, wait=False)
        return exchange(record, "status", store, wait=False)
    result = dict(record)
    result["requested_steps"] = record["request"]["steps"]
    result["complete_training_workflow"] = record["request"]["steps"] == (
        ["FineTune", "SimEval", "Validate", "SuccessGate"]
        + (["RegisterModel"] if record["mode"] == "managed" else []))
    if record.get("created_at"):
        result["elapsed_seconds"] = round(max(
            0, time.time() - dt.datetime.fromisoformat(record["created_at"]).timestamp()), 1)
    if record["mode"] == "local" and record.get("run_dir"):
        require_execution_host(record)
        root = Path(record["run_dir"])
        state = {}
        try:
            state = read_local(root / "status.json")
            result.update(status=state["status"], activity=state.get("activity"),
                          failure_reason=state.get("failure_reason"), receipt_uri=state.get("receipt_uri"),
                          elapsed_seconds=state.get("elapsed_seconds", result.get("elapsed_seconds")))
        except (FileNotFoundError, subprocess.CalledProcessError, json.JSONDecodeError):
            result["activity"] = "Worker starting; waiting for its first status. Service: " + record["unit"]
        service = subprocess.run(
            ["systemctl", "show", record["unit"], "--property=ActiveState", "--value"],
            text=True, capture_output=True, timeout=15)
        result["service_state"] = service.stdout.strip() if not service.returncode else "unknown"
        if state.get("status") == "Succeeded" and result["service_state"] in {
            "active", "activating", "deactivating", "reloading",
        }:
            result.update(status="Finalizing", activity="Pipeline finished; waiting for its worker to exit")
        elif state.get("status") not in {"Succeeded", "Failed"}:
            if (result["service_state"] in {"inactive", "failed"}
                    and time.time() - dt.datetime.fromisoformat(record["created_at"]).timestamp() > 30):
                result.update(status="Failed", failure_reason=state.get("failure_reason") or
                              f"Worker service is {result['service_state']} without a successful result")
        log = root / "run.log"
        try:
            result["last_log_age_seconds"] = round(max(0, time.time() - log.stat().st_mtime))
        except OSError:
            pass
        local_activity(result, root)
        try:
            proof = read_local(root / "independent-verification.json")
            result["independently_verified"] = (
                proof.get("status") == "Succeeded" and proof.get("run_id") == record["id"]
                and proof.get("canonical_commit") == record["source_commit"]
                and set(proof.get("pipeline_steps", {})) == set(record["request"]["steps"])
            )
            result["verification_scope"] = proof.get("verification_scope")
            if proof.get("contract_checks"):
                result["contract_checks"] = proof["contract_checks"]
                if proof["contract_checks"]["status"] != "passed":
                    result.update(independently_verified=False, status="VerificationFailed",
                                  failure_reason="Named pipeline checks failed; inspect contract-checks.json")
            result["outputs"] = proof.get("outputs") or state.get("outputs", {})
            for key in ("episodes", "success_rate"):
                if key in proof:
                    result[key] = proof[key]
            if result["independently_verified"]:
                actual = read_local(root / "parameters.json")
                mismatched = [key for key, value in record["request"]["parameters"].items()
                              if actual.get(key) != value]
                if mismatched:
                    result.update(independently_verified=False, status="VerificationFailed",
                                  failure_reason="Worker parameters differ from the CLI request: "
                                                 + ", ".join(mismatched))
        except (FileNotFoundError, subprocess.CalledProcessError, json.JSONDecodeError):
            result["independently_verified"] = False
        if record.get("verification_error") and not result.get("independently_verified"):
            result.update(status="VerificationFailed", failure_reason=record["verification_error"])
    elif record["mode"] == "managed":
        sm = aws_session(record.get("profile"), record["region"]).client("sagemaker")
        if not record.get("execution_arn"):
            if not record.get("client_request_token"):
                return result
            matches = [row for page in sm.get_paginator("list_pipeline_executions").paginate(
                PipelineName=record["pipeline_name"])
                for row in page["PipelineExecutionSummaries"]
                if row.get("PipelineExecutionDisplayName") == record["id"]]
            if len(matches) != 1:
                result["activity"] = (
                    f"Submission needs reconciliation: found {len(matches)} executions for "
                    f"{record['pipeline_name']}/{record['id']}; no duplicate submitted")
                return result
            record["execution_arn"] = result["execution_arn"] = matches[0]["PipelineExecutionArn"]
        execution = sm.describe_pipeline_execution(PipelineExecutionArn=record["execution_arn"])
        steps = []
        token = None
        while True:
            page = sm.list_pipeline_execution_steps(
                PipelineExecutionArn=record["execution_arn"], **({"NextToken": token} if token else {})
            )
            steps.extend(page["PipelineExecutionSteps"])
            token = page.get("NextToken")
            if not token:
                break
        result.update(status=execution["PipelineExecutionStatus"], steps=steps,
                      failure_reason=execution.get("FailureReason"),
                      independently_verified=False)
        if execution.get("CreationTime"):
            end = execution.get("LastModifiedTime") if execution["PipelineExecutionStatus"] in {
                "Succeeded", "Failed", "Stopped"} else dt.datetime.now(dt.timezone.utc)
            result["elapsed_seconds"] = round((end - execution["CreationTime"]).total_seconds(), 1)
        complete, reason = check_managed_steps(execution, steps, record["request"]["steps"])
        result["requested_steps_succeeded"] = complete
        if execution["PipelineExecutionStatus"] == "Succeeded" and not complete:
            result.update(status="VerificationFailed", failure_reason=reason)
        elif complete and store is not None:
            proof_path = store.path("runs", record["id"]).parent / "independent-verification.json"
            if proof_path.exists():
                from .managed_results import VERIFICATION_VERSION
                proof = json.loads(proof_path.read_text())
                if (proof.get("verification_version") == VERIFICATION_VERSION
                        and proof.get("source_commit") == record["source_commit"]
                        and proof.get("execution_arn") == record["execution_arn"]
                        and proof.get("requested_steps") == record["request"]["steps"]):
                    result.update(proof)
        active = [row for row in steps if row["StepStatus"] in {"Starting", "Executing", "Stopping"}]
        if active:
            details = []
            for row in active:
                description = row["StepName"]
                job_arn = row.get("Metadata", {}).get("TrainingJob", {}).get("Arn")
                if job_arn:
                    job = sm.describe_training_job(TrainingJobName=job_arn.rsplit("/", 1)[1])
                    description += ": " + job.get("SecondaryStatus", job["TrainingJobStatus"])
                details.append(description)
            result["activity"] = "Active steps: " + "; ".join(details)
        elif result["status"] in {"Executing", "Submitted"}:
            result["activity"] = "Execution submitted; waiting for a job to start"
    if result.get("status") == "Succeeded":
        result["activity"] = ("Requested steps completed; independent verification passed"
                              if result.get("independently_verified") else
                              "Requested steps completed; independent verification pending")
    elif result.get("status") in {"Failed", "Stopped", "VerificationFailed"}:
        result["activity"] = "Execution " + result["status"]
    if result.get("independently_verified"):
        result["verification_status"] = "passed"
    elif result.get("status") == "VerificationFailed":
        result["verification_status"] = "failed"
    elif result.get("status") in {"Failed", "Stopped", "SubmissionError"}:
        result["verification_status"] = "not completed; execution did not succeed"
    elif result.get("status") == "Succeeded":
        result["verification_status"] = "pending; follow this run to verify"
    else:
        result["verification_status"] = "pending; execution is still in progress"
    return result


def refresh(record, store, *, verify=False):
    """Reload, observe and persist under one lock; busy observers read saved state."""
    try:
        if record["request"].get("transport") == "ssm":
            # Each exchange reloads and saves under its own lock.
            state = observe(store.load("runs", record["id"]), store=store)
            if verify and state["status"] == "Succeeded" and not state.get("independently_verified"):
                from .remote_execution import exchange
                state = exchange(state, "verify", store, wait=False)
            return state
        with store.lock("runs", record["id"], wait=verify):
            state = observe(store.load("runs", record["id"]), store=store)
            if verify and state["status"] == "Succeeded" and not state.get("independently_verified"):
                state["activity"] = "Checking completed-run evidence"
                store.save("runs", state)
                try:
                    if state["mode"] == "managed":
                        from .managed_results import verify as verify_managed
                        state.update(verify_managed(state, store))
                    else:
                        log = store.path("runs", record["id"]).parent / "verification.log"
                        with log.open("a") as stream:
                            attempt_start = stream.tell()
                            completed = subprocess.run([
                                sys.executable, str(COMPONENT / "scripts/local/verify_run.py"),
                                "--run-dir", state["run_dir"],
                            ], stdout=stream, stderr=subprocess.STDOUT)
                        if completed.returncode:
                            with log.open("rb") as stream:
                                stream.seek(max(attempt_start, log.stat().st_size - 4000))
                                detail = stream.read(4000).decode(errors="replace").strip()
                            raise ValueError(
                                f"Independent verification exited {completed.returncode}: "
                                f"{detail or 'No diagnostic output was written.'}\nFull log: {log}")
                        state = observe(state, store=store)
                        if not state.get("independently_verified"):
                            raise ValueError("Independent verification did not establish the requested result")
                    state["activity"] = "Requested steps completed; independent verification passed"
                except Exception as exc:
                    state.update(status="VerificationFailed", independently_verified=False,
                                 verification_status="failed", activity="Independent verification failed",
                                 failure_reason=f"{type(exc).__name__}: {exc}")
                    state["verification_error"] = state["failure_reason"]
            store.save("runs", state)
            return state
    except OperationBusy:
        # Never write this snapshot back: the coordinator may still be saving identifiers.
        return store.load("runs", record["id"])


def follow(record, *, as_json=False, timeout_seconds=86400, store=None):
    store = store or Store()
    reconnect = store.follow_command(record["id"])
    if record["mode"] == "local" and record["request"].get("transport") != "ssm":
        require_execution_host(record)
        if not record.get("run_dir"):
            raise ValueError("No local run directory was recorded")
        result = subprocess.run([
            "sudo", sys.executable, str(Path(record["component"]) / "scripts/local/wait_run.py"),
            "--run-dir", record["run_dir"], "--timeout-seconds", str(timeout_seconds),
            "--reconnect-command", reconnect,
        ], stdout=sys.stderr if as_json else None).returncode
        with contextlib.redirect_stdout(sys.stderr):
            state = refresh(record, store)
        if as_json:
            print(json.dumps(state, default=str))
        if result == 0 and not state.get("independently_verified"):
            print(state.get("failure_reason") or "Independent verification was not established",
                  file=sys.stderr)
            return 1
        return result
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        from .cli import show

        with contextlib.redirect_stdout(sys.stderr):
            state = refresh(record, store, verify=True)
        show(state, as_json)
        sys.stdout.flush()
        record = state
        if state["status"] == "Succeeded" and state.get("independently_verified"):
            return 0
        if state["status"] in {"Failed", "Stopped", "SubmissionError", "VerificationFailed"}:
            return 1
        time.sleep(min(30, max(0, deadline - time.monotonic())))
    print("Watch timeout reached; the execution may still be running. "
          f"Reconnect with {reconnect}.", file=sys.stderr)
    return 2


def stop(record, store=None):
    store = store or Store()
    if record["request"].get("transport") == "ssm":
        from .remote_execution import exchange
        return exchange(record, "stop", store)
    with store.lock("runs", record["id"], wait=True):
        record = store.load("runs", record["id"])
        if record["mode"] == "local":
            require_execution_host(record)
            from .cleanup import _stop_local_service
            _stop_local_service(f"vla-local-{validate_name(record['id'])}", 120)
        else:
            if not record.get("execution_arn"):
                raise ValueError("No recorded execution ARN; reconcile the submission before cancelling")
            aws_session(record.get("profile"), record["region"]).client("sagemaker").stop_pipeline_execution(
                PipelineExecutionArn=record["execution_arn"]
            )
        record["cancellation_requested"] = True
        store.save("runs", record)
        return record
