"""Record shared resources and provision an Arena GPU host in the deployed VPC."""
from __future__ import annotations

import shutil

from ..source import REPOSITORY as REPO_ROOT
from .environment import load, save


def record_baseline(env):
    """Record existing shared images/log groups before acquiring any resource ownership."""
    ecr = env.client("ecr")
    path = env.root / "baseline.json"
    if not path.exists():
        logs = env.client("logs")
        log_groups = [g["logGroupName"] for page in logs.get_paginator("describe_log_groups").paginate(
            logGroupNamePrefix="/aws/codebuild/" + env.name) for g in page["logGroups"]]
        references = {}
        for repository in ("vla/gr00t", "vla/openvla", "vla/molmoact2", "vla/isaac-arena"):
            try:
                ecr.describe_repositories(repositoryNames=[repository])
            except ecr.exceptions.RepositoryNotFoundException:
                continue
            references[repository] = [item for page in ecr.get_paginator("list_images").paginate(
                repositoryName=repository) for item in page["imageIds"]]
        save(path, {"account_id": env.config["account_id"], "candidate": env.record["candidate"],
                    "name": env.name, "external_ecr": references, "log_groups": log_groups})
    baseline = load(path)
    if (baseline["account_id"], baseline["candidate"], baseline["name"]) != (
            env.config["account_id"], env.record["candidate"], env.name):
        raise ValueError("Saved baseline belongs to another deployment")
    return baseline


def network(env):
    ssm, ec2 = env.client("ssm"), env.client("ec2")
    project = getattr(env, "project", env.name)
    vpc = ssm.get_parameter(Name=f"/{project}/vpc-id")["Parameter"]["Value"]
    ids = ssm.get_parameter(Name=f"/{project}/private-subnet-ids")["Parameter"]["Value"].split(",")
    return vpc, ids, ec2.describe_subnets(SubnetIds=ids)["Subnets"]


def gpu_host(env):
    vpc, ids, subnets = network(env)
    ec2 = env.client("ec2")
    folder = env.states["gpu"]
    folder.mkdir(parents=True, exist_ok=True)
    if not (folder / "main.tf").exists():
        shutil.copyfile(REPO_ROOT / "isaac-lab-arena-on-aws/infra/resources/gpu-host/main.tf", folder / "main.tf")
        shutil.copyfile(REPO_ROOT / "isaac-lab-arena-on-aws/infra/.terraform.lock.hcl",
                        folder / ".terraform.lock.hcl")
    variables = folder / "environment.auto.tfvars.json"
    if not variables.exists():
        images = ec2.describe_images(Owners=["amazon"], Filters=[
            {"Name": "name", "Values": ["Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*"]},
            {"Name": "architecture", "Values": ["x86_64"]},
            {"Name": "state", "Values": ["available"]}])["Images"]
        if not images:
            raise ValueError("No supported GPU DLAMI found")
        config = {"account_id": env.config["account_id"], "profile": env.config.get("profile"),
                  "region": env.config["region"], "project": env.name, "name": env.name + "-gpu",
                  "vpc_id": vpc, "ami_id": max(images, key=lambda i: i["CreationDate"])["ImageId"],
                  "instance_type": env.config["gpu_instance"], "root_volume_gib": 800}
        zone = env.config.get("gpu_zone")
        matching = [s for s in subnets if zone is None or s["AvailabilityZone"] == zone]
        if matching:
            config["subnet_id"] = sorted(matching, key=lambda s: s["AvailabilityZone"])[0]["SubnetId"]
        else:
            routes = ec2.describe_route_tables(Filters=[
                {"Name": "association.subnet-id", "Values": [ids[0]]}])["RouteTables"]
            if len(routes) != 1:
                raise ValueError("Cannot identify the deployed private subnet's route table")
            config["extra_subnet"] = {"availability_zone": zone, "cidr_block": "10.1.20.0/24",
                                      "route_table_id": routes[0]["RouteTableId"]}
        save(variables, config)
    env.terraform("gpu", ["init", "-input=false"], "gpu-init")
    env.terraform("gpu", ["plan", "-input=false", "-out=host.tfplan"], "gpu-plan")
    env.terraform("gpu", ["apply", "-input=false", "host.tfplan"], "gpu-apply")
    outputs = env.outputs("gpu")
    save(env.root / "gpu-outputs.json", outputs)
    print("GPU host:", outputs["instance_id"], env.config["gpu_instance"], flush=True)
