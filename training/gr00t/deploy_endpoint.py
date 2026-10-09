"""
Deploy a fine-tuned GR00T model as a SageMaker real-time endpoint.

Takes the model.tar.gz a Lab 1 training job produced and stands up an endpoint
that serves action predictions: create_model -> create_endpoint_config ->
create_endpoint, using the groot-inference container (BYOC /ping + /invocations).

Usage:
    # Deploy the endpoint (ECR image + role auto-resolved from the Foundation stack):
    python training/gr00t/deploy_endpoint.py \
        --model-s3 s3://<BUCKET>/groot-data/ur3/output/<JOB>/output/model.tar.gz \
        --endpoint-name groot-ur3

    # Show exactly what would be created, without touching AWS:
    python training/gr00t/deploy_endpoint.py --model-s3 s3://... --dry-run

    # Call a live endpoint with a wrist image + robot state:
    python training/gr00t/deploy_endpoint.py --invoke --endpoint-name groot-ur3 \
        --image wrist.jpg --state 0,0,0,0,0,0,0 --task "pick up the cube"

    # Tear it down (endpoint + config + model) when finished:
    python training/gr00t/deploy_endpoint.py --delete --endpoint-name groot-ur3

GR00T inference needs a GPU endpoint (default ml.g5.2xlarge, ~$1.5/hr) and the
model loads slowly, so the container startup health-check timeout is 30 min.
"""

import argparse
import base64
import json
import os
import sys
import time

import boto3


def _account(region):
    return boto3.client("sts", region_name=region).get_caller_identity()["Account"]


def _defaults(region):
    """Resolve ECR image + execution role from account/region (override via flags/env)."""
    acct = _account(region)
    project, env = os.environ.get("PROJECT_NAME", "physical-ai"), os.environ.get("ENVIRONMENT", "dev")
    image = os.environ.get("GROOT_INFERENCE_IMAGE",
                           f"{acct}.dkr.ecr.{region}.amazonaws.com/{project}/gr00t-inference:latest")
    role = os.environ.get("SAGEMAKER_ROLE_ARN",
                          f"arn:aws:iam::{acct}:role/{project}-{env}-sagemaker-role")
    return image, role


def deploy(model_s3, endpoint_name, instance_type, region, image=None, role=None, dry_run=False):
    img, rolearn = _defaults(region)
    image = image or img
    role = role or rolearn
    ts = time.strftime("%Y%m%d-%H%M%S")
    model_name = f"{endpoint_name}-model-{ts}"
    config_name = f"{endpoint_name}-config-{ts}"

    create_model = {
        "ModelName": model_name,
        "PrimaryContainer": {
            "Image": image,
            "ModelDataUrl": model_s3,
            "Environment": {
                "GROOT_MODEL_PATH": "/opt/ml/model",
                "GROOT_EMBODIMENT": "NEW_EMBODIMENT",
                "GROOT_ACTION_DIM": "7",
                "GROOT_STATE_DIM": "7",
                "GROOT_CAMERAS": "wrist",
                **({"HF_TOKEN": os.environ["HF_TOKEN"]} if os.environ.get("HF_TOKEN") else {}),
            },
        },
        "ExecutionRoleArn": role,
    }
    create_config = {
        "EndpointConfigName": config_name,
        "ProductionVariants": [{
            "VariantName": "default",
            "ModelName": model_name,
            "InstanceType": instance_type,
            "InitialInstanceCount": 1,
            # GR00T loads slowly — give the container 30 min before health checks must pass.
            "ContainerStartupHealthCheckTimeoutInSeconds": 1800,
        }],
    }
    create_endpoint = {"EndpointName": endpoint_name, "EndpointConfigName": config_name}

    if dry_run:
        print(json.dumps({"create_model": create_model, "create_endpoint_config": create_config,
                          "create_endpoint": create_endpoint}, indent=2, default=str))
        print("\n[dry-run] No AWS calls made.")
        return {"dry_run": True, "endpoint_name": endpoint_name}

    sm = boto3.client("sagemaker", region_name=region)
    print(f"  create_model: {model_name}")
    sm.create_model(**create_model)
    print(f"  create_endpoint_config: {config_name}")
    sm.create_endpoint_config(**create_config)
    print(f"  create_endpoint: {endpoint_name}")
    sm.create_endpoint(**create_endpoint)
    print(f"\n  Endpoint '{endpoint_name}' creating (~10-30 min for GR00T to load).")
    print(f"  Status: aws sagemaker describe-endpoint --endpoint-name {endpoint_name} "
          f"--query EndpointStatus --output text")
    return {"status": "creating", "endpoint_name": endpoint_name, "model_data": model_s3}


def invoke(endpoint_name, image_path, state, task, region):
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()
    payload = {"image": img_b64, "state": [float(x) for x in state.split(",")], "task": task}
    rt = boto3.client("sagemaker-runtime", region_name=region)
    resp = rt.invoke_endpoint(EndpointName=endpoint_name, ContentType="application/json",
                              Body=json.dumps(payload))
    print(resp["Body"].read().decode())


def delete(endpoint_name, region):
    sm = boto3.client("sagemaker", region_name=region)
    # Resolve the config + model the endpoint points at, then delete all three.
    try:
        ep = sm.describe_endpoint(EndpointName=endpoint_name)
        config_name = ep["EndpointConfigName"]
        cfg = sm.describe_endpoint_config(EndpointConfigName=config_name)
        model_name = cfg["ProductionVariants"][0]["ModelName"]
    except sm.exceptions.ClientError as e:
        print(f"  (could not resolve endpoint chain: {e})")
        config_name = model_name = None
    sm.delete_endpoint(EndpointName=endpoint_name)
    print(f"  deleted endpoint {endpoint_name}")
    if config_name:
        sm.delete_endpoint_config(EndpointConfigName=config_name)
        print(f"  deleted endpoint-config {config_name}")
    if model_name:
        sm.delete_model(ModelName=model_name)
        print(f"  deleted model {model_name}")


def main():
    p = argparse.ArgumentParser(description="Deploy a fine-tuned GR00T model as a SageMaker endpoint")
    p.add_argument("--model-s3", help="S3 URI of the training job's model.tar.gz")
    p.add_argument("--endpoint-name", default="groot-ur3")
    p.add_argument("--instance-type", default="ml.g5.2xlarge")
    p.add_argument("--region", default="us-west-2")
    p.add_argument("--image", default=None, help="Override the inference ECR image URI")
    p.add_argument("--role-arn", default=None, help="Override the SageMaker execution role ARN")
    p.add_argument("--dry-run", action="store_true", help="Show the API calls; make none")
    p.add_argument("--invoke", action="store_true", help="Call a live endpoint instead of deploying")
    p.add_argument("--delete", action="store_true", help="Delete the endpoint + config + model")
    p.add_argument("--image-path", "--image-file", dest="image_path", default=None,
                   help="(--invoke) path to a wrist image file")
    p.add_argument("--state", default="0,0,0,0,0,0,0", help="(--invoke) comma-separated 7D state")
    p.add_argument("--task", default="", help="(--invoke) task description string")
    args = p.parse_args()

    if args.delete:
        delete(args.endpoint_name, args.region)
        return
    if args.invoke:
        if not args.image_path:
            p.error("--invoke requires --image-path <file>")
        invoke(args.endpoint_name, args.image_path, args.state, args.task, args.region)
        return
    if not args.model_s3:
        p.error("--model-s3 is required to deploy (or use --invoke / --delete)")

    result = deploy(args.model_s3, args.endpoint_name, args.instance_type, args.region,
                    image=args.image, role=args.role_arn, dry_run=args.dry_run)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
