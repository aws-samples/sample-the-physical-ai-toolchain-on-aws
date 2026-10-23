"""Prepare one explicitly supplied EC2 host; preserve its profile and existing data."""
from __future__ import annotations

import hashlib
import json
import shlex
import subprocess
import time
from pathlib import Path

from botocore.exceptions import ClientError

from .backend import COMPONENT
from .operations import timestamp
from .remote_commands import ACTIVE, observe, submit, wait

IDLE_PROBE = """set -eu
if command -v docker >/dev/null && [ -n "$(docker ps -q)" ]; then
  echo "Host has running containers; no preparation performed" >&2; exit 1
fi
if pgrep -f '[r]un_local_pipeline.py' >/dev/null; then
  echo "Host has an active VLA worker" >&2; exit 1
fi
command -v nvidia-smi >/dev/null || {
  echo "NVIDIA driver missing. Use the README GPU DLAMI or install its NVIDIA driver before host preparation; bootstrap installs container tools, not GPU drivers." >&2
  exit 1
}
nvidia-smi --query-gpu=name,memory.total --format=csv
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
"""


def put_verified(session, bucket, key, body):
    s3 = session.client("s3")
    digest = hashlib.sha256(body).hexdigest()
    result = s3.put_object(Bucket=bucket, Key=key, Body=body)
    version = result.get("VersionId")
    if version in (None, "", "null"):
        raise RuntimeError("Host delivery requires versioned development storage")
    stored = s3.get_object(Bucket=bucket, Key=key, VersionId=version)["Body"]
    try:
        if hashlib.sha256(stored.read()).hexdigest() != digest:
            raise RuntimeError("Host delivery readback mismatch")
    finally:
        stored.close()
    return {"bucket": bucket, "key": key, "version_id": version, "sha256": digest}


def download_command(item, destination, *, wait_for_access=False):
    command = ["aws", "s3api", "get-object", "--region", "us-east-1",
               "--bucket", item["bucket"], "--key", item["key"],
               "--version-id", item["version_id"], destination]
    download = shlex.join(command) + " >/dev/null"
    if wait_for_access:
        # The just-added inline role policy may not yet be effective on EC2.
        # Retry this read only; never weaken the version/hash or IAM checks.
        error = shlex.quote(destination + ".download-error")
        download = f"""for attempt in {{1..13}}; do
  if {download} 2>{error}; then
    rm -f {error}
    break
  fi
  cat {error} >&2
  if [ "$attempt" -eq 13 ] || ! grep -q 'AccessDenied' {error}; then exit 1; fi
  echo "Waiting for the new instance-role S3 grant (attempt $attempt/13)" >&2
  sleep 10
done"""
    return (download + "\n" +
            shlex.join(["printf", "%s  %s\\n", item["sha256"], destination]) + " | sha256sum -c -")


def runtime_policy(cfg, bucket, secret_arn):
    return {"Version": "2012-10-17", "Statement": [
        {"Effect": "Allow", "Action": ["ssm:GetParameter"],
         "Resource": f"arn:aws:ssm:{cfg.region}:{cfg.account_id}:parameter/{cfg.foundation_project}/*"},
        {"Effect": "Allow", "Action": ["secretsmanager:GetSecretValue"], "Resource": secret_arn},
        {"Effect": "Allow", "Action": ["s3:GetBucketVersioning", "s3:GetBucketLocation",
                                      "s3:ListBucket", "s3:ListBucketMultipartUploads"],
         "Resource": f"arn:aws:s3:::{bucket}"},
        {"Effect": "Allow", "Action": ["s3:GetObject", "s3:GetObjectVersion", "s3:PutObject",
                                      "s3:AbortMultipartUpload", "s3:ListMultipartUploadParts"],
         "Resource": f"arn:aws:s3:::{bucket}/*"},
    ]}


def mark_prepared(session, record, save):
    local = record["local"]
    local.update(remote_ready=True, source_commit=record["source_commit"], prepared_at=timestamp())
    record["host_preparation_verified"] = True
    selection = {key: record[key] for key in
                 ("id", "account_id", "region", "project", "images", "local", "source_commit")}
    selection.update(schema_version=1, profile=None, status="Selected", host_preparation_verified=True)
    key = f"localdev/cli/{record['id']}/{record['source_commit']}/client-selection.json"
    reference = put_verified(session, local["development_bucket"], key,
                             (json.dumps(selection, indent=2) + "\n").encode())
    record["selection_uri"] = f"s3://{reference['bucket']}/{reference['key']}"
    record["selection_version"] = reference["version_id"]
    record["activity"] = "Host prerequisites, checkout and selected images are prepared"
    save()


def prepare_host(session, cfg, record, save, *, yes, plan_only=False):
    from .provisioning import confirm

    local = record["local"]
    host, region = local["instance_id"], local["host_region"]
    ec2, iam = session.client("ec2", region_name=region), session.client("iam")
    instance = ec2.describe_instances(InstanceIds=[host])["Reservations"][0]["Instances"][0]
    if instance["State"]["Name"] != "running":
        raise ValueError(f"Supplied host {host} is {instance['State']['Name']}; start it explicitly")
    if instance.get("Architecture") != "x86_64":
        raise ValueError("The supplied GPU host must be x86-64")
    profile_arn = instance.get("IamInstanceProfile", {}).get("Arn")
    if profile_arn:
        profile = iam.get_instance_profile(InstanceProfileName=profile_arn.rsplit("/", 1)[1])["InstanceProfile"]
        if len(profile["Roles"]) != 1:
            raise ValueError("Expected exactly one role in the supplied host profile")
        role = profile["Roles"][0]["RoleName"]
        if local.get("expected_role") and local["expected_role"] != role:
            raise ValueError(f"Host uses {role}; preserve its profile and select that role explicitly")
    else:
        role = local.get("expected_role") or f"vla-{record['id']}-local"
    bucket = local.get("development_bucket") or (
        f"{record['project']}-{record['request']['environment']}-localdev-{cfg.account_id}")
    if bucket in {cfg.bucket, cfg.handoff_bucket, cfg.trust_bucket}:
        raise ValueError("Local development storage must differ from every managed bucket")
    secret = session.client("secretsmanager").describe_secret(SecretId=cfg.hf_secret_name)
    policy = runtime_policy(cfg, bucket, secret["ARN"])
    local.update(expected_role=role, development_bucket=bucket, instance_type=instance["InstanceType"],
                 host_supplied=not bool(record.get("owned_gpu")))
    prefix = f"/opt/vla-cli/{record['id']}"
    checkout = f"{prefix}/checkouts/{record['source_commit']}/repo"
    local.update(component=f"{checkout}/{COMPONENT.name}", state_dir=f"{prefix}/state",
                 deployment_id=record["id"], source_commit=record["source_commit"],
                 remote_ready=False)
    local.pop("prepared_at", None)
    plan = {"host": host, "region": region, "instance_type": instance["InstanceType"],
            "existing_instance_profile": profile_arn, "runtime_role": role, "development_bucket": bucket,
            "runtime_policy": policy, "scratch_root": local["scratch_root"], "checkout": checkout,
            "actions": ["Preserve existing profile; add scoped runtime policy and ECR/SSM read access",
                        "Use IMDSv2 with hop limit 2", "Verify idle host before system setup",
                        "Install documented Ubuntu prerequisites if missing",
                        "Deliver this Git commit and nonsecret selection; pull selected published images"],
            "no_actions": ["No instance acquisition, formatting, pruning or disk resizing",
                           "No managed trust-policy change or image build"]}
    record["host_plan"] = plan
    save()
    print(json.dumps(plan, indent=2), flush=True)
    if plan_only:
        return
    confirm("Prepare this supplied host using the plan above?", yes)
    ssm = session.client("ssm", region_name=region)
    phases = record.setdefault("host_commands", {})
    setup_key = "bootstrap-" + record["source_commit"]
    previous = phases.get(setup_key)
    if previous and previous.get("submitted_at") and not previous.get("command_id"):
        # This branch only reconciles the unique prior comment; submit() will
        # not send a new command when submitted_at is already present.
        submit(ssm, host, previous, [], save)
    if previous and previous.get("command_id"):
        record["activity"] = "Following existing host preparation: " + previous["command_id"]
        result = observe(ssm, host, previous)
        save()
        if result["Status"] in ACTIVE:
            result = wait(ssm, host, previous, save)
        if result["Status"] == "Success":
            mark_prepared(session, record, save)
            return
        record.setdefault("host_command_history", []).append({"phase": setup_key, **previous})
        phases.pop(setup_key)
        save()

    # A role-less DLAMI needs SSM access before its idle state can be inspected.
    if not profile_arn:
        try:
            existing_role = iam.get_role(RoleName=role)["Role"]
        except iam.exceptions.NoSuchEntityException:
            trust = {"Version": "2012-10-17", "Statement": [{
                "Effect": "Allow", "Principal": {"Service": "ec2.amazonaws.com"},
                "Action": "sts:AssumeRole"}]}
            existing_role = iam.create_role(
                RoleName=role, AssumeRolePolicyDocument=json.dumps(trust),
                Tags=[{"Key": "VlaDeployment", "Value": record["id"]}])["Role"]
            record.setdefault("created_resources", []).append(existing_role["Arn"])
            save()
        if not any(tag["Key"] == "VlaDeployment" and tag["Value"] == record["id"]
                   for tag in existing_role.get("Tags", [])):
            raise ValueError(f"Role {role} already exists without this operation's ownership tag")
        iam.attach_role_policy(RoleName=role,
                               PolicyArn="arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore")
        try:
            profile = iam.get_instance_profile(InstanceProfileName=role)["InstanceProfile"]
        except iam.exceptions.NoSuchEntityException:
            profile = iam.create_instance_profile(
                InstanceProfileName=role, Tags=[{"Key": "VlaDeployment", "Value": record["id"]}])["InstanceProfile"]
        if not profile["Roles"]:
            iam.add_role_to_instance_profile(InstanceProfileName=role, RoleName=role)
        elif [item["RoleName"] for item in profile["Roles"]] != [role]:
            raise ValueError("Existing instance profile contains a different role")
        for attempt in range(12):
            try:
                ec2.associate_iam_instance_profile(InstanceId=host, IamInstanceProfile={"Name": role})
                break
            except ClientError as exc:
                if exc.response["Error"]["Code"] != "InvalidParameterValue" or attempt == 11:
                    raise
                time.sleep(5)
    else:
        attached = {row["PolicyArn"] for page in iam.get_paginator(
            "list_attached_role_policies").paginate(RoleName=role) for row in page["AttachedPolicies"]}
        ssm_core = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
        if ssm_core not in attached:
            iam.attach_role_policy(RoleName=role, PolicyArn=ssm_core)
    deadline = time.monotonic() + 600
    while True:
        rows = ssm.describe_instance_information(
            Filters=[{"Key": "InstanceIds", "Values": [host]}])["InstanceInformationList"]
        if rows and rows[0]["PingStatus"] == "Online":
            break
        if time.monotonic() >= deadline:
            raise RuntimeError("Supplied host is not SSM Online; check its agent, egress and SSM instance-role access")
        record["activity"] = f"Waiting for SSM agent on {host}"
        save()
        print(record["activity"], flush=True)
        time.sleep(30)
    # Always inspect current idleness, even when resuming completed preparation.
    record["activity"] = f"SSM online; checking idle host and GPU on {host}"
    probe = {}
    phases["idle-probe"] = probe
    submit(ssm, host, probe, [IDLE_PROBE], save, timeout=120)
    wait(ssm, host, probe, save)

    s3 = session.client("s3")
    try:
        s3.head_bucket(Bucket=bucket, ExpectedBucketOwner=cfg.account_id)
    except ClientError as exc:
        if exc.response["ResponseMetadata"]["HTTPStatusCode"] != 404:
            raise
        s3.create_bucket(Bucket=bucket)
        record.setdefault("created_resources", []).append(f"arn:aws:s3:::{bucket}")
        save()
    s3.put_bucket_versioning(Bucket=bucket, VersioningConfiguration={"Status": "Enabled"})
    s3.put_public_access_block(Bucket=bucket, PublicAccessBlockConfiguration={
        "BlockPublicAcls": True, "IgnorePublicAcls": True,
        "BlockPublicPolicy": True, "RestrictPublicBuckets": True})
    policy_name = "VlaCli-" + record["id"]
    try:
        existing_policy = iam.get_role_policy(RoleName=role, PolicyName=policy_name)["PolicyDocument"]
    except iam.exceptions.NoSuchEntityException:
        existing_policy = None
    if existing_policy and existing_policy != policy:
        raise ValueError(f"Policy {policy_name} already has different permissions; preserve it and inspect ownership")
    if existing_policy != policy:
        iam.put_role_policy(RoleName=role, PolicyName=policy_name, PolicyDocument=json.dumps(policy))
    attached = {row["PolicyArn"] for page in iam.get_paginator("list_attached_role_policies").paginate(
        RoleName=role) for row in page["AttachedPolicies"]}
    for name in ("AmazonEC2ContainerRegistryReadOnly", "AmazonSSMManagedInstanceCore"):
        arn = "arn:aws:iam::aws:policy/" + name
        if arn not in attached:
            iam.attach_role_policy(RoleName=role, PolicyArn=arn)
    ec2.modify_instance_metadata_options(
        InstanceId=host, HttpEndpoint="enabled", HttpTokens="required", HttpPutResponseHopLimit=2)

    work = Path(record["work_directory"])
    bundle = work / f"{record['source_commit']}.bundle"
    if not bundle.exists():
        subprocess.run(["git", "bundle", "create", str(bundle), "HEAD"], cwd=COMPONENT, check=True)
    delivery_prefix = f"localdev/cli/{record['id']}/{record['source_commit']}"
    bundle_ref = put_verified(session, bucket, delivery_prefix + "/source.bundle", bundle.read_bytes())
    bootstrap = put_verified(session, bucket, delivery_prefix + "/bootstrap.sh",
                             (COMPONENT / "scripts/local/bootstrap.sh").read_bytes())
    selection = {key: record[key] for key in ("id", "account_id", "region", "project", "images", "local")}
    selection.update(schema_version=1, profile=None, status="Ready", host_preparation_verified=True,
                     source_commit=record["source_commit"])
    selection_ref = put_verified(session, bucket, delivery_prefix + "/selection.json",
                                 (json.dumps(selection, indent=2) + "\n").encode())
    setup = phases.setdefault(setup_key, {})
    command = "\n".join([
        "set -euo pipefail", "umask 077",
        "unset AWS_PROFILE AWS_DEFAULT_PROFILE AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN AWS_SECURITY_TOKEN",
        "export AWS_DEFAULT_REGION=us-east-1 AWS_REGION=us-east-1 GIT_LFS_SKIP_SMUDGE=1",
        "export VLA_SCRATCH_ROOT=" + shlex.quote(local["scratch_root"]),
        shlex.join(["install", "-d", "-m", "700", prefix]),
        f'test "$(aws sts get-caller-identity --query Account --output text)" = {shlex.quote(cfg.account_id)}',
        download_command(bootstrap, prefix + "/bootstrap.sh", wait_for_access=True),
        shlex.join(["bash", prefix + "/bootstrap.sh"]),
        # Hold the host lock across clone, dependency setup and image pulls too.
        "exec 9>/var/lock/vla-local-gpu.lock",
        'flock -n 9 || { echo "A VLA worker started during preparation"; exit 1; }',
        IDLE_PROBE,
        download_command(bundle_ref, prefix + "/source.bundle"),
        shlex.join(["mkdir", "-p", str(Path(checkout).parent)]),
        f"if [ ! -d {shlex.quote(checkout + '/.git')} ]; then "
        + shlex.join(["git", "clone", "--no-checkout", prefix + "/source.bundle", checkout]) + "; fi",
        shlex.join(["git", "-C", checkout, "checkout", "--detach", record["source_commit"]]),
        f'test "$(git -C {shlex.quote(checkout)} rev-parse HEAD)" = {record["source_commit"]}',
        f'test -z "$(git -C {shlex.quote(checkout)} status --porcelain --untracked-files=no)"',
        shlex.join(["bash", local["component"] + "/scripts/local/setup.sh"]),
        shlex.join(["install", "-d", "-m", "700", local["state_dir"] + "/deployments/" + record["id"]]),
        download_command(selection_ref, local["state_dir"] + "/deployments/" + record["id"] + "/record.json"),
        shlex.join([local["component"] + "/.venv/bin/python", "-m", "vla_pipeline.remote_worker",
                    "prepare", "--state-dir", local["state_dir"], "--deployment", record["id"]]),
    ])
    record["activity"] = "Preparing host prerequisites, checkout and selected image downloads"
    submit(ssm, host, setup, ["bash -c " + shlex.quote(command)], save, timeout=43200, bucket=bucket)
    wait(ssm, host, setup, save)
    mark_prepared(session, record, save)
