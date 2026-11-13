"""Fresh-account prerequisites before image builds or GPU instance creation."""
from __future__ import annotations

from pathlib import Path
import shutil
import uuid

from ..source import REPOSITORY as REPO_ROOT
from .environment import load, save

TOKEN_NAMES = {"hf": "vla-pipeline/hf-token", "ngc": "vla-pipeline/ngc-token"}


def check_capacity(env):
    ec2 = env.client("ec2")
    selected = env.config["gpu_instance"]
    if not selected.startswith("g6e."):
        raise ValueError("This sample environment supports g6e GPU hosts; choose an explicit g6e instance")
    types = {selected}
    running = [i for page in ec2.get_paginator("describe_instances").paginate(Filters=[
        {"Name": "instance-state-name", "Values": ["pending", "running", "stopping"]}])
        for r in page["Reservations"] for i in r["Instances"]]
    types.update(i["InstanceType"] for i in running)
    details = {}
    ordered = sorted(types)
    for start in range(0, len(ordered), 100):
        details.update({i["InstanceType"]: i for i in ec2.describe_instance_types(
            InstanceTypes=ordered[start:start + 100])["InstanceTypes"]})
    required = details[selected]["VCpuInfo"]["DefaultVCpus"]
    used = sum(details[i["InstanceType"]]["VCpuInfo"]["DefaultVCpus"] for i in running
               if i["InstanceType"].startswith(("g", "vt")) and i.get("InstanceLifecycle") != "spot")
    service = env.client("service-quotas")
    quotas = [q for page in service.get_paginator("list_service_quotas").paginate(ServiceCode="ec2")
              for q in page["Quotas"]]
    matches = [q for q in quotas if q["QuotaName"] == "Running On-Demand G and VT instances"]
    if len(matches) != 1:
        raise ValueError("Cannot identify the EC2 G/VT quota; inspect Service Quotas before building")
    quota = matches[0]
    result = {"instance_type": selected, "requested_vcpus": required,
              "existing_g_vt_vcpus": used, "quota": quota["Value"],
              "quota_code": quota["QuotaCode"],
              "scope": "Quota is checked; actual GPU capacity is established only by a successful launch"}
    save(env.root / "capacity-check.json", result)
    print(f"EC2 G/VT quota: {used} in use + {required} requested / {quota['Value']:g} vCPUs", flush=True)
    if used + required > quota["Value"]:
        raise ValueError("GPU quota is insufficient. Arrange the required quota in AWS, "
                         "then resume this deployment. No quota request, image build or "
                         "instance launch was performed.")
    offerings = ec2.describe_instance_type_offerings(LocationType="availability-zone", Filters=[
        {"Name": "instance-type", "Values": [selected]}])["InstanceTypeOfferings"]
    zones = sorted({i["Location"] for i in offerings})
    if not zones or (env.config.get("gpu_zone") and env.config["gpu_zone"] not in zones):
        raise ValueError(f"{selected} is not offered in the selected zone; available zones: {', '.join(zones)}")
    print("GPU type offered in:", ", ".join(zones), flush=True)


def read_token(env, kind):
    path = env.config.get(kind + "_token_file")
    if path:
        path = Path(path).expanduser().resolve()
        if path.is_relative_to(env.root) or path.is_relative_to(REPO_ROOT):
            raise ValueError("Keep token files outside the checkout and environment state/evidence directory")
        value = path.read_text().strip()
    else:
        raise ValueError(f"Provide --{kind}-token-file or arena.{kind}_token_file in config.json "
                         "to create this deployment's missing token secret.")
    if not value or value.startswith("{") or len(value.encode()) > 65536:
        raise ValueError(f"The supplied {kind.upper()} token must be a nonempty plaintext token")
    return value


def token_secrets(env):
    secrets = env.client("secretsmanager")
    names = {kind: env.config[kind + "_secret_name"] for kind in TOKEN_NAMES}
    folder = env.states["secrets"]
    variables = folder / "environment.auto.tfvars.json"
    # Once ownership is recorded, a partially completed creation remains ours.
    if variables.exists():
        created_names = load(variables)["secret_names"]
    else:
        created_names = []
        for name in names.values():
            try:
                detail = secrets.describe_secret(SecretId=name)
            except secrets.exceptions.ResourceNotFoundException:
                created_names.append(name)
            else:
                if detail.get("DeletedDate"):
                    raise ValueError(f"Secret {name} is pending deletion; finish its recovery/removal before setup")
    values = {}
    for kind, name in names.items():
        if name in created_names:
            values[kind] = read_token(env, kind)
        else:
            # Reuse supplied secrets without changing their value or ownership.
            response = secrets.get_secret_value(SecretId=name, VersionStage="AWSCURRENT")
            value = response.pop("SecretString", "").strip()
            response.pop("SecretBinary", None)
            if not value or value.startswith("{"):
                raise ValueError(f"Existing {name} must contain a nonempty plaintext token")
    if created_names:
        folder.mkdir(parents=True, exist_ok=True)
        if not (folder / "main.tf").exists():
            shutil.copyfile(REPO_ROOT / "isaac-lab-arena-on-aws/infra/resources/secrets/main.tf", folder / "main.tf")
            shutil.copyfile(REPO_ROOT / "isaac-lab-arena-on-aws/infra/.terraform.lock.hcl",
                            folder / ".terraform.lock.hcl")
        if not variables.exists():
            save(variables, {"account_id": env.config["account_id"], "region": env.config["region"],
                             "profile": env.config.get("profile"), "name": env.name,
                             "secret_names": created_names})
        env.terraform("secrets", ["init", "-input=false"], "secrets-init")
        env.terraform("secrets", ["plan", "-input=false", "-out=secrets.tfplan"], "secrets-plan")
        env.terraform("secrets", ["apply", "-input=false", "secrets.tfplan"], "secrets-apply")
        arns = env.outputs("secrets")["secret_arns"]
        receipts_path = env.root / "token-secret-receipts.json"
        receipts = load(receipts_path) if receipts_path.exists() else {}
        for kind, value in values.items():
            name = names[kind]
            if name not in receipts:
                receipts[name] = {"arn": arns[name], "request_token": str(uuid.uuid4()), "status": "Pending"}
                save(receipts_path, receipts)
            receipt = receipts[name]
            if receipt["status"] != "Ready":
                reply = secrets.put_secret_value(SecretId=arns[name],
                    ClientRequestToken=receipt["request_token"], SecretString=value)
                receipt.update(status="Ready", version=reply["VersionId"])
                save(receipts_path, receipts)
            check = secrets.get_secret_value(SecretId=arns[name], VersionId=receipt["version"])
            if check.pop("SecretString", None) != value:
                raise ValueError(f"{name} readback differs; preserve and inspect the recorded setup")
    save(env.root / "token-secret-ownership.json",
         {"created": created_names, "supplied": sorted(set(names.values()) - set(created_names)),
          "token_values_saved": False})
    print(f"Token secrets ready: {len(created_names)} created; "
          f"{len(names) - len(created_names)} supplied secrets preserved. No token values logged.", flush=True)
