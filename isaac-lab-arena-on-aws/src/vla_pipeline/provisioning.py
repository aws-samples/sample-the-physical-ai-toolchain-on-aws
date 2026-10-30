"""Coordinate the existing deployment recipes with durable state and explicit plans."""
from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
from pathlib import Path

from .backend import COMPONENT
from .deployment import aws_session, discover
from .operations import timestamp, validate_name, write_json
from .cli_context import command_name

COMPONENT_REPOSITORIES = ("vla/gr00t", "vla/openvla", "vla/molmoact2", "vla/isaac-arena")


def secret_check(session, names):
    """Read access and nonempty plaintext values, without returning or logging tokens."""
    results = {}
    for name in names:
        response = session.client("secretsmanager").get_secret_value(
            SecretId=name, VersionStage="AWSCURRENT")
        value = response.pop("SecretString", "").strip()
        response.pop("SecretBinary", None)
        if not value or value.startswith("{"):
            raise ValueError(f"Secret {name} must contain a nonempty plaintext token")
        results[name] = {key: response[key] for key in ("ARN", "VersionId")}
    return results


def model_access_check(session, request):
    """Check selected GR00T model/data access before starting paid image builds."""
    from urllib.error import HTTPError
    from urllib.parse import quote, urlparse
    from urllib.request import HTTPRedirectHandler, Request, build_opener
    from .registry import named_cells, resolve_suite

    # Redirects to a storage CDN must not receive the caller's HF token.
    class Redirect(HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
            if redirected is not None and urlparse(newurl).hostname != "huggingface.co":
                redirected.remove_header("Authorization")
            return redirected

    token = session.client("secretsmanager").get_secret_value(
        SecretId=request["hf_secret_name"], VersionStage="AWSCURRENT")["SecretString"].strip()
    opener = build_opener(Redirect())
    paths = {"https://huggingface.co/api/whoami-v2"}
    defaults = json.loads((COMPONENT / "entrypoints/train/gr00t/defaults.json").read_text())
    for name in request["cells"]:
        cell = named_cells()[name]
        if cell["model"] != "gr00t":
            continue
        suffix = "_n16" if cell["version"] == "n16" else ""
        repo = defaults["base_checkpoint" + suffix]
        revision = defaults["base_checkpoint_hf_revision" + suffix]
        paths.add(f"https://huggingface.co/{repo}/resolve/{revision}/config.json")
        dataset = resolve_suite(cell["suite"]).dataset
        if dataset:
            file = "/".join(filter(None, (dataset.subdir, "meta/info.json")))
            paths.add("https://huggingface.co/datasets/" + dataset.repo_id + "/resolve/"
                      + quote(dataset.revision or "main", safe="") + "/" + file)
    checked = []
    for url in sorted(paths):
        method = "GET" if url.endswith("/whoami-v2") else "HEAD"
        try:
            with opener.open(Request(url, method=method, headers={"Authorization": "Bearer " + token}),
                             timeout=30) as response:
                if response.status != 200:
                    raise ValueError("Hugging Face did not confirm access to " + url)
        except HTTPError as exc:
            # Report the endpoint/status, never headers or the supplied token.
            raise ValueError(f"Hugging Face access check returned HTTP {exc.code}: {url}. "
                             "Check the token and accept the selected model/dataset terms before resuming.") from None
        checked.append(url)
        print("Model-access check passed:", url, flush=True)
    return {"checked": checked, "scope": "HF token and selected GR00T configuration/dataset metadata. "
            "Other model downloads and NGC registry authentication are checked by their build/runtime recipes."}


def input_digest():
    from .source import input_digest as runtime_digest
    return runtime_digest()


def environment(profile, region):
    # Select a profile; never copy credential material into a command or state file.
    env = dict(os.environ, AWS_DEFAULT_REGION=region, AWS_REGION=region, TF_IN_AUTOMATION="1")
    if profile:
        for name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN",
                     "AWS_SECURITY_TOKEN", "AWS_DEFAULT_PROFILE"):
            env.pop(name, None)
        env["AWS_PROFILE"] = profile
    return env


def confirm(message, yes):
    print(message, flush=True)
    if yes:
        return
    if not os.isatty(0):
        raise ValueError("Review the displayed plan, then repeat with --yes for noninteractive apply")
    if input("Apply this plan? [y/N] ").strip().lower() != "y":
        raise ValueError("Plan was not approved; no apply for this phase was started")


def require_fresh_namespace(session, request):
    """Refuse taking ownership of an existing deployment before creating a NAT gateway."""
    project = request["project"]
    parameters = session.client("ssm").describe_parameters(
        ParameterFilters=[{"Key": "Name", "Option": "BeginsWith", "Values": [f"/{project}/"]}],
        MaxResults=5,
    )["Parameters"]
    supplied = set(request.get("existing_ecr_repos", []))
    unknown = supplied - set(COMPONENT_REPOSITORIES)
    if unknown:
        raise ValueError("Unsupported existing image repositories: " + ", ".join(sorted(unknown)))
    repositories = [name for name in COMPONENT_REPOSITORIES if name not in supplied]
    repositories += [f"{project}/{name}" for name in (
        "gr00t-training", "gr00t-inference", "isaac-lab", "isaac-lab-arena",
        "cosmos-transfer", "cosmos3")]
    collisions = [item["Name"] for item in parameters]
    ecr = session.client("ecr")
    # Explicit references must exist. They remain data sources in Terraform,
    # never imported resources or lifecycle policies owned by this deployment.
    for name in sorted(supplied):
        try:
            ecr.describe_repositories(repositoryNames=[name])
        except ecr.exceptions.RepositoryNotFoundException as exc:
            raise ValueError(f"Selected existing repository {name} does not exist") from exc
    for name in repositories:
        try:
            ecr.describe_repositories(repositoryNames=[name])
        except ecr.exceptions.RepositoryNotFoundException:
            continue
        collisions.append("ECR " + name)
    prefix = f"{project}-{request['environment']}"
    buckets = {row["Name"] for row in session.client("s3").list_buckets()["Buckets"]}
    collisions += ["S3 " + name for purpose in ("datasets", "models", "checkpoints", "trust", "handoff")
                   if (name := f"{prefix}-{purpose}-{request['account_id']}") in buckets]
    iam = session.client("iam")
    for suffix in ("sagemaker-role", "cosmos-role", "training", "workload", "validation",
                   "vla-codebuild-role"):
        try:
            iam.get_role(RoleName=f"{prefix}-{suffix}")
        except iam.exceptions.NoSuchEntityException:
            continue
        collisions.append("IAM " + prefix + "-" + suffix)
    if collisions:
        raise ValueError(
            "Existing resources have another state owner: " + ", ".join(collisions)
            + ". Use --use-existing to select them, then --prepare-host if needed. "
              "For a new project sharing only image repositories, explicitly select "
              "each --existing-ecr-repo. "
              "The CLI does not import or overwrite their Terraform state."
        )


def run_process(command, directory, env, record, save, label):
    log = directory / (label + ".log")
    record.update(activity=label, process={"pid": None, "command": command,
                                         "log": str(log), "started_at": timestamp()})
    save()
    with log.open("a") as stream:
        proc = subprocess.Popen(command, cwd=directory, env=env, stdout=stream,
                                stderr=subprocess.STDOUT, start_new_session=True)
        record["process"]["pid"] = proc.pid
        save()
        try:
            while True:
                try:
                    result = proc.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    record["activity"] = f"{label}: running; output {log}"
                    save()
                    print(record["activity"], flush=True)
        except BaseException:
            # Terraform receives one graceful interrupt and gets a chance to save state.
            import signal
            if proc.poll() is None:
                proc.send_signal(signal.SIGINT)
            try:
                record["process"]["exit_code"] = proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                record["activity"] = f"{label} still exiting; inspect PID {proc.pid} before resume"
            save()
            raise
    record["process"]["exit_code"] = result
    save()
    if result:
        tail = log.read_text(errors="replace")[-6000:]
        raise RuntimeError(f"{label} exited {result}; log: {log}\n{tail}")


def terraform_phase(name, request, work, env, record, save, *, yes, plan_only):
    phases = record.setdefault("phases", {})
    if phases.get(name, {}).get("status") == "Succeeded":
        return
    directory = work / name
    source = COMPONENT.parent / "foundation/infra" if name == "foundation" else COMPONENT / "infra"
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    for path in source.iterdir():
        if path.suffix == ".tf" or path.name == ".terraform.lock.hcl":
            target = directory / path.name
            if not target.exists():
                shutil.copy2(path, target)
    variables = {"aws_region": request["region"], "project_name": request["project"],
                 "environment": request["environment"]}
    if name == "component":
        variables.update(codebuild_project_name=f"{request['project']}-{request['environment']}-vla-image-build",
                         hf_secret_name=request["hf_secret_name"], ngc_secret_name=request["ngc_secret_name"],
                         additional_input_s3_arns=request.get("additional_input_s3_arns", []))
        supplied = request.get("existing_ecr_repos", [])
        variables.update(existing_ecr_repos=supplied,
                         ecr_repos=[name for name in COMPONENT_REPOSITORIES if name not in supplied])
    write_json(directory / "deployment.auto.tfvars.json", variables)
    phases[name] = {"status": "Planning", "directory": str(directory)}
    save()
    run_process(["terraform", "init", "-input=false", "-no-color"], directory, env, record, save, name + "-init")
    run_process(["terraform", "plan", "-input=false", "-no-color", "-lock-timeout=30s",
                 "-out=deployment.tfplan"], directory, env, record, save, name + "-plan")
    plan = subprocess.check_output(
        ["terraform", "show", "-no-color", "deployment.tfplan"], cwd=directory, env=env, text=True)
    print(plan, flush=True)
    phases[name].update(status="Planned", plan=str(directory / "deployment.tfplan"))
    save()
    if plan_only:
        return
    confirm(f"Apply {name} from the saved plan above. State stays at {directory}.", yes)
    phases[name]["status"] = "Applying"
    save()
    run_process(["terraform", "apply", "-input=false", "-no-color", "deployment.tfplan"],
                directory, env, record, save, name + "-apply")
    outputs = json.loads(subprocess.check_output(
        ["terraform", "output", "-json"], cwd=directory, env=env, text=True))
    phases[name].update(status="Succeeded", outputs=outputs, completed_at=timestamp())
    save()


def deploy(args, store):
    from .execution import source_identity
    from .launch_request import assignments
    from .registry import named_cells

    validate_name(args.name)
    source_identity()
    with store.lock("deployments", args.name):
        previous = store.load("deployments", args.name) if store.path("deployments", args.name).exists() else {}
        if args.resume:
            if not previous.get("request"):
                raise ValueError("No interrupted deployment request exists under this name")
            request = previous["request"]
            if args.profile and args.profile != request.get("profile"):
                raise ValueError("Resume uses the saved profile; refresh that profile first")
            record = previous
            if record["source_commit"] != source_identity() or record["input_digest"] != input_digest():
                raise ValueError("Deployment/build inputs changed; resume from the recorded source revision")
        else:
            if previous.get("request"):
                raise ValueError("A deployment operation already exists; use --resume")
            if args.region != "us-east-1":
                raise ValueError("The application region must be us-east-1")
            supplied = json.loads(Path(args.selection).read_text()) if args.selection else {}
            if supplied and supplied.get("schema_version") != 1:
                raise ValueError("Selection files must have schema_version 1")
            selected = {**previous, **supplied}
            project = args.project or selected.get("project") or "physical-ai"
            environment_name = args.environment or selected.get("request", {}).get("environment", "dev")
            if not re.fullmatch(r"[a-z][a-z0-9-]{1,29}", project):
                raise ValueError("--project must be 2–30 lowercase letters/digits/hyphens")
            if not re.fullmatch(r"[a-z][a-z0-9-]{0,11}", environment_name):
                raise ValueError("--environment must be 1–12 lowercase letters/digits/hyphens")
            cells = args.cell or list(selected.get("images", {}))
            if not cells or any(cell not in named_cells() for cell in cells):
                raise ValueError(f"Select supported --cell values; inspect {command_name()} cells")
            images = dict(selected.get("images", {}))
            overrides = assignments(args.image, "--image")
            if overrides:
                if len(cells) != 1:
                    raise ValueError("--image overrides require one selected --cell")
                images[cells[0]] = {**images.get(cells[0], {}), **overrides}
            local = dict(selected.get("local", {}))
            if args.create_local_host:
                if not re.fullmatch(r"[a-z][a-z0-9-]{1,25}", args.name):
                    raise ValueError("With --create-local-host, use a 2–26-character lowercase deployment name")
                if local.get("instance_id"):
                    raise ValueError("The imported selection already supplies a host; omit --create-local-host")
            for key, value in {"instance_id": args.local_host, "host_region": args.host_region,
                               "development_bucket": args.development_bucket,
                               "expected_role": args.expected_role, "scratch_root": args.scratch_root}.items():
                if value is not None:
                    local[key] = value
            if local.get("instance_id"):
                local.setdefault("host_region", args.region)
                local.setdefault("scratch_root", "/opt/dlami/nvme/vla-tests")
            if args.prepare_host and not local.get("instance_id"):
                raise ValueError("--prepare-host requires a supplied or saved --local-host")
            request = {
                "profile": args.profile or selected.get("profile"), "region": args.region,
                "project": project, "environment": environment_name, "cells": list(dict.fromkeys(cells)),
                "images": images, "local": local, "prepare_existing": args.prepare_host,
                "hf_secret_name": args.hf_secret_name or (
                    f"vla-pipeline/{args.name}/hf-token" if args.hf_token_file else "vla-pipeline/hf-token"),
                "ngc_secret_name": args.ngc_secret_name or (
                    f"vla-pipeline/{args.name}/ngc-token" if args.ngc_token_file else "vla-pipeline/ngc-token"),
                "hf_token_file": args.hf_token_file, "ngc_token_file": args.ngc_token_file,
                "create_local_host": args.create_local_host, "gpu_zone": args.gpu_zone,
                "additional_input_s3_arns": args.input_s3_arn or [],
                "existing_ecr_repos": sorted(set(args.existing_ecr_repo)),
            }
            record = {
                "id": args.name, "request": request, "created_at": timestamp(), "status": "Planning",
                "source_commit": source_identity(), "input_digest": input_digest(),
                "profile": request["profile"], "region": request["region"], "project": request["project"],
                "images": {}, "local": local, "infrastructure_owned": not args.prepare_host,
                "host_preparation_verified": False,
            }
        session = aws_session(request["profile"], request["region"])
        from .cli_settings import require_account
        caller = require_account(session, args.account_id or record.get("account_id"))
        if record.get("account_id", caller["Account"]) != caller["Account"]:
            raise ValueError("Selected profile belongs to a different account than the saved operation")
        record["account_id"] = request["account_id"] = caller["Account"]
        if args.resume and record.get("status") == "Ready":
            return record
        if not args.resume:
            for other in (previous, supplied):
                if other and (other.get("account_id") != caller["Account"]
                              or other.get("region") != request["region"]):
                    raise ValueError("Saved selection differs from the chosen account/region")
        work = store.path("deployments", args.name).parent / "work"
        work.mkdir(parents=True, exist_ok=True, mode=0o700)
        record["work_directory"] = str(work)
        record["coordinator"] = {"pid": os.getpid(), "host": socket.gethostname(), "started_at": timestamp()}
        def save():
            store.save("deployments", record)
        save()
        try:
            process = record.get("process", {})
            if args.resume and process.get("pid") and process.get("exit_code") is None:
                try:
                    os.kill(process["pid"], 0)
                except ProcessLookupError:
                    pass
                else:
                    raise ValueError(f"Recorded Terraform process PID {process['pid']} may still be active; "
                                     "inspect it and the state lock before resume. No force-unlock is performed.")
            record.update(status="Preparing", activity="Checking required token secrets")
            record.pop("failure_reason", None)
            save()
            from .deployment import image_digest
            from .registry import require_image_capability, resolve
            for cell, images in request["images"].items():
                if cell not in request["cells"]:
                    continue
                for step, uri in images.items():
                    if step not in {"FineTune", "SimEval"}:
                        raise ValueError(f"Unexpected image step {step}")
                    images[step] = image_digest(session, uri)
                definition = named_cells()[cell]
                capability = resolve(definition["model"], definition["simulator"]).required_image_capability
                if capability and images.get("SimEval"):
                    require_image_capability(images["SimEval"], capability, ecr_client=session.client("ecr"))
            if record["local"].get("instance_id"):
                local = record["local"]
                ec2 = session.client("ec2", region_name=local["host_region"])
                host = ec2.describe_instances(InstanceIds=[local["instance_id"]])["Reservations"][0]["Instances"][0]
                if host["State"]["Name"] != "running" or host.get("Architecture") != "x86_64":
                    raise ValueError("Supply a running x86-64 GPU instance before deployment")
                metadata = ec2.describe_instance_types(InstanceTypes=[host["InstanceType"]])["InstanceTypes"][0]
                if not any(gpu.get("Manufacturer") == "NVIDIA"
                           for gpu in metadata.get("GpuInfo", {}).get("Gpus", [])):
                    raise ValueError("The supplied instance does not report an NVIDIA GPU")
            if request["prepare_existing"]:
                cfg = discover(session, request["project"])
                record["secrets"] = secret_check(session, [cfg.hf_secret_name])
                # Existing resource ownership is preserved; this mode never builds images.
                for cell in request["cells"]:
                    selected = request["images"].get(cell, {})
                    if set(selected) != {"FineTune", "SimEval"}:
                        raise ValueError("--prepare-host requires both published image selections for each cell")
                    record["images"][cell] = {step: image_digest(session, uri) for step, uri in selected.items()}
            else:
                if not shutil.which("terraform"):
                    raise ValueError(
                        f"Terraform is required for new deployment; install it before {command_name()} deploy")
                if not args.plan:
                    from .lifecycle import prepare_environment
                    confirm("Prepare this deployment's token secrets and resource ownership records.", args.yes)
                    prepare_environment(record, store)
                if args.plan and (request.get("hf_token_file") or request.get("ngc_token_file")):
                    record["secret_plan"] = {
                        "names": [request["hf_secret_name"], request["ngc_secret_name"]],
                        "action": "Create missing secret entries from supplied token files during apply",
                    }
                    print(json.dumps(record["secret_plan"]), flush=True)
                else:
                    record["secrets"] = secret_check(
                        session, [request["hf_secret_name"], request["ngc_secret_name"]])
                if not args.plan:
                    record["model_access"] = model_access_check(session, request)
                    save()
                if not (work / "foundation/terraform.tfstate").exists():
                    require_fresh_namespace(session, request)
                env = environment(request["profile"], request["region"])
                foundation_ready = record.get("phases", {}).get("foundation", {}).get("status") == "Succeeded"
                terraform_phase("foundation", request, work, env, record, save,
                                yes=args.yes, plan_only=args.plan)
                if args.plan and not foundation_ready:
                    record.update(status="Planned", activity="Foundation plan saved. Component plan follows "
                                  "after Foundation exists; no apply or image build was performed.")
                    save()
                    return record
                terraform_phase("component", request, work, env, record, save,
                                yes=args.yes, plan_only=args.plan)
                if args.plan:
                    record.update(status="Planned", activity="Component plan saved; no apply, "
                                  "image build or host preparation was performed.")
                    save()
                    return record
                cfg = discover(session, request["project"])
                from .image_builds import prepare_images
                record["activity"] = "Preparing selected images"
                save()
                prepare_images(session, cfg, record, save, retry_failed=args.resume)
                if request.get("create_local_host"):
                    from .lifecycle import create_host
                    create_host(record, store)
            if record["local"].get("instance_id"):
                from .host import prepare_host
                prepare_host(session, cfg, record, save, yes=args.yes, plan_only=args.plan)
            record.update(status="Planned" if args.plan else "Ready",
                          activity="Preparation plan only" if args.plan else "Selected cells are ready to run",
                          completed_at=timestamp())
            save()
            return record
        except BaseException as exc:
            record.update(status="Interrupted" if isinstance(exc, KeyboardInterrupt) else "Failed",
                          failure_reason=f"{type(exc).__name__}: {exc}",
                          next_action=f"Inspect this operation, then {command_name()} deploy "
                                      f"--name {args.name} --resume")
            save()
            raise
