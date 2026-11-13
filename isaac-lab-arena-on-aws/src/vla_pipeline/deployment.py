"""Discover an existing installation and preserve explicit image selections."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from .operations import timestamp, validate_name


def aws_session(profile, region):
    import boto3
    return boto3.Session(profile_name=profile, region_name=region)


def discover(session, project="physical-ai"):
    from .config import load_config
    return load_config(project_name=project, boto_session=session)


def local_instance():
    """Read this host's identity through IMDSv2; no workstation credentials."""
    from urllib.request import ProxyHandler, Request, build_opener

    if sys.platform != "linux":
        raise ValueError("Run local commands on the prepared EC2 GPU host")
    endpoint = "http://169.254.169.254/latest/"
    opener = build_opener(ProxyHandler({}))
    with opener.open(Request(
        endpoint + "api/token", method="PUT",
        headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"},
    ), timeout=5) as response:
        token = response.read().decode()
    with opener.open(Request(
        endpoint + "dynamic/instance-identity/document",
        headers={"X-aws-ec2-metadata-token": token},
    ), timeout=5) as response:
        identity = json.load(response)
    return {key: identity[key] for key in ("accountId", "instanceId", "instanceType", "region")}


def image_digest(session, image):
    from .registry import parse_ecr_reference
    reference = parse_ecr_reference(image)
    if not reference.account or not reference.region:
        raise ValueError("Use a full private ECR image URI, including account and region")
    if reference.region != "us-east-1":
        raise ValueError("Application images must be selected from us-east-1")
    client = session.client("ecr", region_name=reference.region)
    rows = client.describe_images(
        registryId=reference.account, repositoryName=reference.repository,
        imageIds=[reference.image_id],
    )["imageDetails"]
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one published image for {image}")
    return (f"{reference.account}.dkr.ecr.{reference.region}.amazonaws.com/"
            f"{reference.repository}@{rows[0]['imageDigest']}")


def select_existing(args, store):
    from .launch_request import assignments
    from .registry import named_cells, require_image_capability, resolve

    validate_name(args.name)
    if args.region != "us-east-1":
        raise ValueError("The application region must be us-east-1; host region is separate")
    cells = named_cells()
    supplied = json.loads(Path(args.selection).read_text()) if args.selection else {}
    if supplied and supplied.get("schema_version") != 1:
        raise ValueError("Selection files must use schema_version 1")
    previous = (store.load("deployments", args.name)
                if store.path("deployments", args.name).exists() else {})
    if previous.get("request"):
        raise ValueError("This name belongs to a coordinated deployment; use a new selection name "
                         "or --resume to continue that operation")
    selected_cells = args.cell or list(supplied.get("images", previous.get("images", {})))
    if not selected_cells or any(name not in cells for name in selected_cells):
        raise ValueError(f"Select supported --cell values. Available: {', '.join(cells)}")
    project = args.project or supplied.get("project") or previous.get("project") or "physical-ai"
    profile = args.profile or previous.get("profile")
    session = aws_session(profile, args.region)
    from .cli_settings import require_account
    require_account(session, getattr(args, "account_id", None))
    cfg = discover(session, project)
    for other in (previous, supplied):
        if other and (other.get("account_id") != cfg.account_id or other.get("region") != cfg.region):
            raise ValueError("Deployment account/region differs; use its actual profile or another name")
    images = {**previous.get("images", {}), **supplied.get("images", {})}
    overrides = assignments(args.image, "--image")
    if overrides and len(selected_cells) != 1:
        raise ValueError("Per-step image overrides require one --cell; add other cells in another call")
    for name in selected_cells:
        selected = {**images.get(name, {}), **overrides}
        if set(selected) != {"FineTune", "SimEval"}:
            raise ValueError(
                f"{name} requires published --image FineTune=URI and --image SimEval=URI, "
                "or their saved selection. Historical image tags are not automatic defaults."
            )
        images[name] = {step: image_digest(session, uri) for step, uri in selected.items()}
        cell = cells[name]
        spec = resolve(cell["model"], cell["simulator"])
        if spec.required_image_capability:
            require_image_capability(images[name]["SimEval"], spec.required_image_capability,
                                     ecr_client=session.client("ecr", region_name=cfg.region))
    local = {**previous.get("local", {}), **supplied.get("local", {})}
    for key, value in {
        "development_bucket": args.development_bucket, "expected_role": args.expected_role,
        "scratch_root": args.scratch_root, "instance_id": args.local_host,
        "host_region": args.host_region,
    }.items():
        if value is not None:
            local[key] = value
    if local.get("development_bucket"):
        bucket = local["development_bucket"]
        if bucket in {cfg.bucket, cfg.handoff_bucket, cfg.trust_bucket}:
            raise ValueError("Local development storage must differ from all managed buckets")
        if session.client("s3").get_bucket_versioning(Bucket=bucket).get("Status") != "Enabled":
            raise ValueError("The development bucket must have S3 versioning enabled")
    if local.get("instance_id"):
        if not local.get("host_region"):
            raise ValueError("--local-host requires --host-region")
        on_selected_host = False
        if session.get_credentials().method == "iam-role":
            identity = local_instance()
            on_selected_host = (identity["accountId"], identity["instanceId"], identity["region"]) == (
                cfg.account_id, local["instance_id"], local["host_region"],
            )
        if on_selected_host:
            local["instance_type"] = identity["instanceType"]
            local["observed_state"] = "running"
        else:
            result = session.client("ec2", region_name=local["host_region"]).describe_instances(
                InstanceIds=[local["instance_id"]]
            )
            instance = result["Reservations"][0]["Instances"][0]
            local["instance_type"] = instance["InstanceType"]
            local["observed_state"] = instance["State"]["Name"]
    record = {
        "id": args.name, "account_id": cfg.account_id, "region": cfg.region,
        "profile": profile, "project": project, "images": images, "local": local,
        "status": "Selected", "created_at": previous.get("created_at", timestamp()),
        "infrastructure_owned": False,
        "host_preparation_verified": bool(local.get("remote_ready")),
    }
    store.save("deployments", record)
    return record
