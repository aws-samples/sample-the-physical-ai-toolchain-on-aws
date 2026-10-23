"""Archive and retire only the resources recorded by an Arena deployment."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import shlex
import tarfile
import time

from botocore.exceptions import ClientError
from ..source import REPOSITORY as REPO_ROOT
from .environment import load, module, save, support

PHASES = ["finish-workloads", "capture-hosts", "archive-storage",
          "retire-outputs", "destroy-infrastructure", "verify-teardown", "finish-evidence"]


def inventory(env):
    directory = env.root / "teardown"
    directory.mkdir(exist_ok=True)
    current = support.terraform_inventory(env.states)
    path = directory / "ownership.json"
    if not path.exists():
        save(path, current)
    recorded = load(path)
    for name, state in current.items():
        if name not in recorded or state["lineage"] != recorded[name]["lineage"]:
            raise ValueError("Terraform state was replaced; inspect recorded ownership")
        ids = {r["address"]: r["attributes"]["id"] for r in recorded[name]["resources"]}
        for resource in state["resources"]:
            if ids.get(resource["address"]) != resource["attributes"]["id"]:
                raise ValueError("New or unowned resource appeared during teardown")
    return recorded


def resources(owned, kind):
    return [r["attributes"] for state in owned.values() for r in state["resources"]
            if r["type"] == kind]


def finish_workloads(env):
    return env.finish_workloads()


def administrator(env, instance, shell, label):
    folder = env.root / "teardown"
    request = {"InstanceIds": [instance], "DocumentName": "AWS-RunShellScript",
               "Comment": env.name + "-" + label, "Parameters": {
                   "commands": ["bash -c " + shlex.quote("set -euo pipefail\numask 077\n" + shell)],
                   "executionTimeout": ["3600"]}}
    command = support.ssm_command(env.client("ssm"), request, folder / (label + "-submission.json"))
    return support.wait_ssm(env.client("ssm"), command, instance, folder / (label + "-result.json"))


def capture_hosts(env):
    folder = env.root / "teardown"
    if (folder / "hosts-captured.json").exists():
        return
    context = env.host_capture_context()
    if context is None:
        save(folder / "hosts-captured.json", {"scope": "No prepared sample hosts; no workload was submitted"})
        return
    gpu, targets = context["gpu"], context["targets"]
    path = folder / "capture-utility.json"
    if not path.exists():
        files = {}
        for filename in ("capture.py", "support.py"):
            body = (REPO_ROOT / "isaac-lab-arena-on-aws/scripts/resources" / filename).read_bytes()
            digest = hashlib.sha256(body).hexdigest()
            files[filename] = support.put_version(env.client("s3"), env.config["account_id"],
                gpu["development_bucket"], "runbook/teardown-code/" + digest + "/" + filename, body)
        save(path, {"candidate": env.record["candidate"], "files": files})
    utility = load(path)
    helper = "/opt/pai-runbook/" + env.name + "/capture-code/" + utility["candidate"]
    delivery = "\n".join([
        "import boto3,hashlib,pathlib",
        "root=pathlib.Path(" + repr(helper) + ");root.mkdir(parents=True,exist_ok=True)",
        "s3=boto3.client('s3',region_name=" + repr(env.config["region"]) + ")",
        "for name,ref in " + repr(utility["files"]) + ".items():",
        "    response=s3.get_object(Bucket=ref['bucket'],Key=ref['key'],VersionId=ref['version'],ExpectedBucketOwner=" + repr(env.config["account_id"]) + ")",
        "    with response['Body'] as stream: body=stream.read()",
        "    assert hashlib.sha256(body).hexdigest()==ref['sha256']",
        "    (root/name).write_bytes(body)",
    ])
    captured = {}
    for label, (instance, python, roots) in targets.items():
        arguments = [python, helper + "/capture.py",
                     "--output", "/opt/pai-runbook/" + env.name + "/capture-" + label,
                     "--bucket", gpu["development_bucket"], "--prefix", "runbook/host-records/" + label,
                     "--account-id", env.config["account_id"], "--region", env.config["region"]]
        for root in roots:
            arguments += ["--root", root]
        result = administrator(env, instance,
            shlex.join([python, "-c", delivery]) + "\n" + shlex.join(arguments), "capture-" + label)
        refs = [json.loads(line) for line in result["StandardOutputContent"].splitlines()
                if line.startswith('{"bucket":')]
        if len(refs) != 1:
            raise ValueError("Read the saved host capture output before retrying")
        s3, account = env.client("s3"), env.config["account_id"]
        references = json.loads(support.get_version(s3, account, refs[0]))
        index = json.loads(support.get_version(s3, account, references["index.json"]))
        body = support.get_version(s3, account, references["records.tar.gz"], limit=1024**3)
        (folder / (label + "-records.tar.gz")).write_bytes(body)
        save(folder / (label + "-index.json"), index)
        with tarfile.open(fileobj=io.BytesIO(body), mode="r:gz") as archive:
            expected = {item["member"]: item for item in index["files"]}
            if len(archive.getmembers()) != len(expected) or set(archive.getnames()) != set(expected):
                raise ValueError("Host archive member index differs")
            for member in archive:
                if not member.isfile() or member.name.startswith("/") or ".." in Path(member.name).parts:
                    raise ValueError("Unsafe host archive member")
                data = archive.extractfile(member).read()
                item = expected[member.name]
                if len(data) != item["bytes"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
                    raise ValueError("Host archive member differs from its index")
        captured[label] = {"reference": refs[0], "files": len(index["files"]),
                           "link_references": len(index.get("links", []))}
    save(folder / "hosts-captured.json", captured)
    print("Host evidence verified:", json.dumps(captured), flush=True)


def archive_storage(env):
    import shutil
    folder, owned = env.root / "teardown", inventory(env)
    account, region = env.config["account_id"], env.config["region"]
    path = env.deployment_path
    deployment = load(path) if path.exists() else {}
    attempts = [*deployment.get("builds", {}).values(), *deployment.get("build_history", [])]
    ids = list(dict.fromkeys(b["id"] for b in attempts if b.get("id")))
    if ids:
        codebuild = env.client("codebuild")
        builds = codebuild.batch_get_builds(ids=ids)["builds"]
        if len(builds) != len(ids):
            raise ValueError("A recorded image build could not be found")
        for build in builds:
            if not build.get("buildComplete"):
                codebuild.stop_build(id=build["id"])
        deadline = time.monotonic() + 600
        while not all(b.get("buildComplete") for b in builds):
            if time.monotonic() >= deadline:
                raise TimeoutError("Recorded image builds are still stopping; retain evidence and resume teardown")
            print("Waiting for recorded image builds to stop before archiving", flush=True)
            time.sleep(15)
            builds = codebuild.batch_get_builds(ids=ids)["builds"]
            if len(builds) != len(ids):
                raise ValueError("A recorded image build disappeared during teardown")
        save(folder / "image-builds.json", builds)
        capture_build_logs(env, builds, owned)
    instances = [r["id"] for r in resources(owned, "aws_instance")]
    if instances:
        ec2 = env.client("ec2")
        def states():
            return {i["InstanceId"]: i["State"]["Name"] for reservation in
                    ec2.describe_instances(InstanceIds=instances)["Reservations"]
                    for i in reservation["Instances"]}
        current = states()
        pending = [i for i, state in current.items() if state in {"running", "pending"}]
        if pending:
            ec2.stop_instances(InstanceIds=pending)
        deadline = time.monotonic() + 900
        while not all(state in {"stopped", "terminated"} for state in current.values()):
            if time.monotonic() >= deadline:
                raise TimeoutError("Owned hosts are not stopped; inspect before archiving")
            print("Stopping owned writers:", current, flush=True)
            time.sleep(15)
            current = states()
        save(folder / "writers-stopped.json", current)
    retained = env.root / "retained-evidence"
    retained.mkdir(exist_ok=True)
    if not (retained / "main.tf").exists():
        shutil.copyfile(REPO_ROOT / "isaac-lab-arena-on-aws/infra/resources/evidence/main.tf", retained / "main.tf")
        shutil.copyfile(REPO_ROOT / "isaac-lab-arena-on-aws/infra/.terraform.lock.hcl",
                        retained / ".terraform.lock.hcl")
        save(retained / "environment.auto.tfvars.json",
             {"account_id": account, "profile": env.config.get("profile"), "region": region, "name": env.name})
    env.terraform("retained-evidence", ["init", "-input=false"], "archive-init")
    env.terraform("retained-evidence", ["plan", "-input=false", "-out=evidence.tfplan"], "archive-plan")
    env.terraform("retained-evidence", ["apply", "-input=false", "evidence.tfplan"], "archive-apply")
    bucket = env.outputs("retained-evidence")["bucket"]
    buckets = [r["id"] for r in resources(owned, "aws_s3_bucket")]
    if bucket in buckets:
        raise ValueError("Archive bucket overlaps the deletion scope")
    cli = [env.python, "isaac-lab-arena-on-aws/scripts/resources/archive.py", *env.profile_args]
    manifest, state = folder / "storage-manifest.json", folder / "storage-archive.json"
    if not manifest.exists():
        args = [*cli, "inventory", "--account-id", account, "--region", region,
                "--destination-bucket", bucket, "--prefix", "application", "--manifest", manifest]
        for source_bucket in buckets:
            args += ["--source-bucket", source_bucket]
        env.run(args, "archive-inventory")
    if not state.exists() or load(state).get("status") != "Verified":
        env.run([*cli, "copy", "--manifest", manifest, "--state", state], "archive-copy")
    env.run([*cli, "verify", "--state", state], "archive-verify")
    files = [
        (p, str(Path(label) / p.relative_to(root)))
        for label, root in env.evidence_roots.items() for p in root.rglob("*")
        if p.is_file() and not p.is_symlink()
        and not any(part in {"client-tools", ".terraform", ".venv", "evidence"} for part in p.relative_to(root).parts)
        and p.suffix in {".json", ".tf", ".tfstate", ".hcl", ".log", ".sh"}
    ]
    bundle = folder / "caller-records.tar.gz"
    with tarfile.open(bundle, "w:gz") as tar:
        for path, archive_name in files:
            tar.add(path, arcname=archive_name, recursive=False)
        if env.notebook_path.exists():
            tar.add(env.notebook_path, arcname="notebook/" + env.notebook_path.name, recursive=False)
    reference = support.put_version(env.client("s3"), account, bucket,
                                    "caller/records.tar.gz", bundle.read_bytes())
    save(folder / "caller-archive-reference.json", reference)
    print("Evidence retained in:", bucket, flush=True)


def capture_build_logs(env, builds, owned):
    """Capture only the streams returned by this deployment's recorded builds."""
    folder = env.root / "teardown"
    baseline = load(env.root / "baseline.json")
    projects = {r["name"] for r in resources(owned, "aws_codebuild_project")}
    logs, groups = env.client("logs"), {}
    for build in builds:
        detail = build.get("logs", {})
        group, stream = detail.get("groupName"), detail.get("streamName")
        if not group or not stream:
            continue
        if build["projectName"] not in projects or group != "/aws/codebuild/" + build["projectName"]:
            raise ValueError("Build logs are outside the recorded project; inspect before retirement")
        path = folder / "build-logs" / (hashlib.sha256((group + "/" + stream).encode()).hexdigest() + ".json")
        if not path.exists():
            events, token = [], None
            while True:
                page = logs.get_log_events(logGroupName=group, logStreamName=stream,
                    startFromHead=True, **({"nextToken": token} if token else {}))
                events.extend(page["events"])
                if page["nextForwardToken"] == token:
                    break
                token = page["nextForwardToken"]
            save(path, {"build": build["id"], "group": group, "stream": stream, "events": events})
        groups.setdefault(group, {"streams": [], "owned": group not in baseline.get("log_groups", [])})
        groups[group]["streams"].append(stream)
    save(folder / "build-log-groups.json", groups)


def absent(function, arguments, missing):
    try:
        return function(**arguments)
    except ClientError as exc:
        if exc.response["Error"]["Code"] not in missing:
            raise
        return None


def package_exists(sm, arn):
    try:
        sm.describe_model_package(ModelPackageName=arn)
    except ClientError as exc:
        error = exc.response["Error"]
        if error["Code"] in {"ResourceNotFound", "ResourceNotFoundException"} or (
                error["Code"] == "ValidationException" and
                any(text in error["Message"].lower() for text in ("does not exist", "could not find"))):
            return False
        raise
    return True


def pipeline_exists(sm, arn):
    return any(item["PipelineArn"] == arn for page in sm.get_paginator("list_pipelines").paginate(
        PipelineNamePrefix=arn.rsplit("/", 1)[-1]) for item in page["PipelineSummaries"])


def retire_outputs(env):
    folder = env.root / "teardown"
    if load(folder / "storage-archive.json")["status"] != "Verified":
        raise ValueError("Archive must be verified before output retirement")
    sm, account, region = env.client("sagemaker"), env.config["account_id"], env.config["region"]
    path = folder / "managed-outputs.json"
    retired = load(path) if path.exists() else {}
    for arn, expected_pipeline in env.managed_executions().items():
        pipeline_arn = arn.split("/execution/")[0]
        if pipeline_arn != expected_pipeline or not arn.startswith(
                f"arn:aws:sagemaker:{region}:{account}:pipeline/"):
            raise ValueError("Execution differs from its recorded pipeline/account/region")
        if arn not in retired:
            detail = sm.describe_pipeline_execution(PipelineExecutionArn=arn)
            if detail["PipelineExecutionStatus"] not in {"Succeeded", "Failed", "Stopped"}:
                raise ValueError("Managed execution remains active")
            steps = [step for page in sm.get_paginator("list_pipeline_execution_steps").paginate(
                PipelineExecutionArn=arn) for step in page["PipelineExecutionSteps"]]
            packages = [s["Metadata"]["RegisterModel"]["Arn"] for s in steps
                        if s.get("Metadata", {}).get("RegisterModel", {}).get("Arn")]
            retired[arn] = {"pipeline": pipeline_arn, "steps": steps, "packages": packages,
                            "execution": detail, "removed_packages": [], "pipeline_removed": False}
            save(path, retired)
        item = retired[arn]
        for package in item["packages"]:
            if not package.startswith(f"arn:aws:sagemaker:{region}:{account}:model-package/"):
                raise ValueError("Registered package is outside this account/region")
            if package not in item["removed_packages"]:
                if package_exists(sm, package):
                    sm.delete_model_package(ModelPackageName=package)
                item["removed_packages"].append(package)
                save(path, retired)
        if not item["pipeline_removed"]:
            if pipeline_exists(sm, pipeline_arn):
                executions = [e for page in sm.get_paginator("list_pipeline_executions").paginate(
                    PipelineName=pipeline_arn) for e in page["PipelineExecutionSummaries"]]
                if {e["PipelineExecutionArn"] for e in executions} != {arn}:
                    raise ValueError("Pipeline gained another execution; preserve it")
                sm.delete_pipeline(PipelineName=pipeline_arn.rsplit("/", 1)[-1])
            item["pipeline_removed"] = True
            save(path, retired)
    save(path, retired)
    ecr = env.client("ecr")
    path = folder / "retired-build-images.json"
    images = load(path) if path.exists() else {}
    operation = env.deployment_path
    deployment = load(operation) if operation.exists() else {}
    baseline = load(env.root / "baseline.json") if (env.root / "baseline.json").exists() else {}
    attempts = {**{f"prior-{n}": b for n, b in enumerate(deployment.get("build_history", []))},
                **deployment.get("builds", {})}
    for name, build in attempts.items():
        if not build.get("id") or name in images:
            continue
        repository, tag = build["repository"].split("/", 1)[1], build["tag"]
        # A build can push successfully and fail before the CLI records its
        # digest. Its unique tag still belongs to this recorded build attempt.
        if build.get("digest"):
            digest = build["digest"].split("@", 1)[1]
        else:
            current = absent(ecr.describe_images, {"repositoryName": repository,
                             "imageIds": [{"imageTag": tag}]}, {"ImageNotFoundException"})
            if not current:
                continue
            digest = current["imageDetails"][0]["imageDigest"]
        old = baseline.get("external_ecr", {}).get(repository, [])
        if any(i.get("imageTag") == tag for i in old):
            raise ValueError("Build tag existed before this deployment")
        arguments = {"repositoryName": repository, "imageIds": [{"imageDigest": digest}]}
        current = absent(ecr.describe_images, arguments, {"ImageNotFoundException"})
        if current:
            if tag in current["imageDetails"][0].get("imageTags", []):
                reply = ecr.batch_delete_image(repositoryName=repository, imageIds=[{"imageTag": tag}])
                if reply.get("failures"):
                    raise ValueError("Image tag retirement failed: " + str(reply["failures"]))
            current = absent(ecr.describe_images, arguments, {"ImageNotFoundException"})
            if (current and not current["imageDetails"][0].get("imageTags")
                    and not any(i.get("imageDigest") == digest for i in old)):
                reply = ecr.batch_delete_image(**arguments)
                if reply.get("failures"):
                    raise ValueError("Image digest retirement failed: " + str(reply["failures"]))
        images[name] = {"repository": repository, "digest": digest, "tag": tag}
        save(path, images)


def destroy_infrastructure(env):
    folder, owned = env.root / "teardown", inventory(env)
    archived = load(folder / "storage-archive.json")
    if archived["status"] != "Verified" or not (folder / "caller-archive-reference.json").exists():
        raise ValueError("Verified archives are required before deletion")
    allow_owned_repository_deletion(env, owned)
    s3, account = env.client("s3"), env.config["account_id"]
    archive = module("isaac-lab-arena-on-aws/scripts/resources/archive.py")
    buckets = {r["id"] for r in resources(owned, "aws_s3_bucket")}
    policy_buckets = {r["bucket"] for r in resources(owned, "aws_s3_bucket_policy")}
    for source_bucket in archived["manifest"]["sources"]:
        bucket = source_bucket["bucket"]
        if bucket not in buckets or bucket == archived["manifest"]["destination"]["bucket"]:
            raise ValueError("Source bucket is outside the deletion scope")
        try:
            current = archive.inventory(s3, account, [bucket])[0]
        except s3.exceptions.NoSuchBucket:
            continue
        removals = support.exact_version_removals(current, source_bucket)
        if bucket in policy_buckets:
            s3.delete_bucket_policy(Bucket=bucket, ExpectedBucketOwner=account)
        for start in range(0, len(removals), 1000):
            reply = s3.delete_objects(Bucket=bucket, ExpectedBucketOwner=account,
                Delete={"Objects": removals[start:start+1000], "Quiet": True})
            if reply.get("Errors"):
                raise ValueError("Source deletion failed: " + str(reply["Errors"]))
        if archive.inventory(s3, account, [bucket])[0]["objects"]:
            raise ValueError("Source bucket still contains object versions")
    gpu_file = env.root / "gpu-outputs.json"
    if gpu_file.exists():
        role = load(gpu_file)["runtime_role"]
        if role not in {r["name"] for r in resources(owned, "aws_iam_role")}:
            raise ValueError("GPU runtime role is outside the owned state")
        absent(env.client("iam").delete_role_policy,
               {"RoleName": role, "PolicyName": env.host_policy_name}, {"NoSuchEntity"})
    for name in ("gpu", "arena", "foundation", "secrets"):
        if name not in owned:
            continue
        env.terraform(name, ["plan", "-destroy", "-input=false", "-out=retire.tfplan"], "destroy-plan-" + name)
        plan = env.terraform_json(name, "show", "-json", "retire.tfplan")
        deletions = support.review_destroy_plan(plan, owned[name])
        save(folder / (name + "-destroy-plan.json"), plan)
        print(name, "owned resources to remove:", len(deletions), flush=True)
        env.terraform(name, ["apply", "-input=false", "retire.tfplan"], "destroy-apply-" + name)
    path = folder / "build-log-groups.json"
    if path.exists():
        logs = env.client("logs")
        for group, record in load(path).items():
            if not record["owned"]:
                continue
            try:
                streams = {s["logStreamName"] for page in logs.get_paginator("describe_log_streams").paginate(
                    logGroupName=group) for s in page["logStreams"]}
            except logs.exceptions.ResourceNotFoundException:
                continue
            if streams - set(record["streams"]):
                raise ValueError("Unarchived build log streams appeared; preserve the group and inspect " + group)
            logs.delete_log_group(logGroupName=group)


def allow_owned_repository_deletion(env, owned):
    """Allow Terraform to delete only this deployment's newly owned, empty ECR repositories."""
    repositories = [r for r in owned.get("arena", {}).get("resources", [])
                    if r["type"] == "aws_ecr_repository"]
    if not repositories:
        return
    baseline = load(env.root / "baseline.json")["external_ecr"]
    ecr = env.client("ecr")
    for resource in repositories:
        name = resource["attributes"]["name"]
        if (resource["address"] != f'aws_ecr_repository.repos["{name}"]'
                or name in baseline or resource["attributes"].get("force_delete")):
            raise ValueError("Repository is outside the guarded teardown scope: " + name)
        try:
            images = [image for page in ecr.get_paginator("list_images").paginate(
                repositoryName=name) for image in page.get("imageIds", [])]
        except ecr.exceptions.RepositoryNotFoundException:
            continue
        if images:
            raise ValueError("Repository still contains images; preserve it and inspect: " + name)
    # Leave the shipped protection intact. This override belongs only to this
    # deployment's recorded Terraform directory, after archive and image retirement.
    # force_delete remains false, so a concurrent image push still blocks deletion.
    path = env.states["arena"] / "retire_owned_override.tf.json"
    save(path, {"resource": {"aws_ecr_repository": {"repos": {
        "lifecycle": {"prevent_destroy": False},
    }}}})
    print("Allow deletion of recorded empty repositories:", ", ".join(
        r["attributes"]["name"] for r in repositories), flush=True)


def wait_for_secret_absence(client, secret_arn, *, timeout=300):
    """Wait for an owned secret's accepted deletion to become observable."""
    deadline = time.monotonic() + timeout
    while True:
        secret = absent(client.describe_secret, {"SecretId": secret_arn},
                        {"ResourceNotFoundException"})
        if secret is None:
            return
        if not secret.get("DeletedDate"):
            raise ValueError("Owned token secret is not marked for deletion: " + secret_arn)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Owned token secret deletion is not yet complete: " + secret_arn)
        print("Waiting for confirmed secret deletion:", secret_arn, flush=True)
        time.sleep(min(5, remaining))


def verify_absence(env):
    owned = load(env.root / "teardown/ownership.json")
    remaining = support.terraform_inventory(env.states)
    if any(state["resources"] for state in remaining.values()):
        raise ValueError("Managed Terraform resources remain")
    checked = {"empty_terraform_states": list(remaining), "instances": [], "buckets": [],
               "roles": [], "volumes": [], "nat_gateways": [], "elastic_ips": [], "secrets": [],
               "build_log_groups": []}
    sm, ec2, account = env.client("sagemaker"), env.client("ec2"), env.config["account_id"]
    for item in load(env.root / "teardown/managed-outputs.json").values():
        if pipeline_exists(sm, item["pipeline"]) or any(package_exists(sm, p) for p in item["packages"]):
            raise ValueError("Recorded pipeline or registration remains")
    instances = [r["id"] for r in resources(owned, "aws_instance")]
    if instances:
        for instance in instances:
            reply = absent(ec2.describe_instances, {"InstanceIds": [instance]}, {"InvalidInstanceID.NotFound"})
            rows = [i for r in (reply or {}).get("Reservations", []) for i in r["Instances"]]
            if not all(i["State"]["Name"] == "terminated" for i in rows):
                raise ValueError("Owned EC2 instance remains: " + instance)
        checked["instances"] = instances
    for r in resources(owned, "aws_s3_bucket"):
        try:
            env.client("s3").head_bucket(Bucket=r["id"], ExpectedBucketOwner=account)
        except ClientError as exc:
            if exc.response["Error"]["Code"] not in {"404", "NoSuchBucket"}:
                raise
        else:
            raise ValueError("Owned bucket remains: " + r["id"])
        checked["buckets"].append(r["id"])
    for r in resources(owned, "aws_iam_role"):
        if absent(env.client("iam").get_role, {"RoleName": r["name"]}, {"NoSuchEntity"}) is not None:
            raise ValueError("Owned role remains")
        checked["roles"].append(r["name"])
    volumes = [r["id"] for r in resources(owned, "aws_ebs_volume")]
    volumes += [disk["volume_id"] for r in resources(owned, "aws_instance") for disk in r.get("root_block_device", [])]
    for volume in volumes:
        if absent(ec2.describe_volumes, {"VolumeIds": [volume]}, {"InvalidVolume.NotFound"}) is not None:
            raise ValueError("Owned volume remains")
        checked["volumes"].append(volume)
    for r in resources(owned, "aws_nat_gateway"):
        rows = ec2.describe_nat_gateways(NatGatewayIds=[r["id"]])["NatGateways"]
        if not all(row["State"] == "deleted" for row in rows):
            raise ValueError("Owned NAT gateway remains")
        checked["nat_gateways"].append(r["id"])
    for r in resources(owned, "aws_eip"):
        if absent(ec2.describe_addresses, {"AllocationIds": [r["id"]]}, {"InvalidAllocationID.NotFound"}) is not None:
            raise ValueError("Owned Elastic IP remains")
        checked["elastic_ips"].append(r["id"])
    for r in resources(owned, "aws_secretsmanager_secret"):
        wait_for_secret_absence(env.client("secretsmanager"), r["arn"])
        checked["secrets"].append(r["name"])
    path = env.root / "teardown/build-log-groups.json"
    if path.exists():
        for group, record in load(path).items():
            if not record["owned"]:
                continue
            if any(g["logGroupName"] == group for page in env.client("logs").get_paginator(
                "describe_log_groups").paginate(logGroupNamePrefix=group) for g in page["logGroups"]):
                raise ValueError("Owned build log group remains: " + group)
            checked["build_log_groups"].append(group)
    from .environment_verify import verify
    checked.update(verify(env, owned))
    return checked


def verify_teardown(env):
    checks = verify_absence(env)
    folder = env.root / "teardown"
    env.run([env.python, "isaac-lab-arena-on-aws/scripts/resources/archive.py", *env.profile_args, "verify",
             "--state", folder / "storage-archive.json"], "verify-after-teardown")
    reference = load(folder / "storage-archive.reference.json")
    result = {"name": env.name, "candidate": env.record["candidate"], "teardown": "passed",
              "absence_checks": checks, "archive_verified_after_deletion": True,
              "retained_archive": reference["bucket"], "samples": {}}
    for report in env.cleanup_reports():
        result["samples"][report["id"]] = report["acceptance"]
    result["complete_acceptance"] = bool(result["samples"]) and all(
        value == "passed" for value in result["samples"].values())
    save(env.root / "result.json", result)
    saved = support.put_version(env.client("s3"), env.config["account_id"], reference["bucket"],
                                "caller/result.json", json.dumps(result, indent=2).encode())
    save(env.root / "result-reference.json", saved)
    print(json.dumps(result, indent=2), flush=True)


def actions():
    from .environment_archive import finish_evidence
    return zip(PHASES, (finish_workloads, capture_hosts, archive_storage,
                        retire_outputs, destroy_infrastructure, verify_teardown, finish_evidence))
