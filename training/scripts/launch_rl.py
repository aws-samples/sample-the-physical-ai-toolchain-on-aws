"""
Standalone Isaac Lab RL launcher (no GR00T required).

Launches an Isaac Lab reinforcement-learning training job on SageMaker using the
`physical-ai/isaac-lab` container. RL is a standalone pipeline — use this to train
(or smoke-test) an RL policy in simulation with no GR00T / imitation-learning stage
and no Lab 1 dependency.

The validated path uses Isaac Lab's built-in
`Isaac-Velocity-Flat-G1-v0` task. The custom UR3
pick-and-place task (`PickAndPlaceUR3-v0`) IS gym-registered (training/envs/__init__.py)
but is NOT yet wired into the container's training entrypoint and is GPU-unvalidated
— see docs/ROADMAP.md (Feature 2) — so it will not resolve in a training job until
that work lands.

Account/region/role/image are resolved from the caller — nothing hardcoded.

Usage:
    # Preview the job (no AWS writes):
    python training/scripts/launch_rl.py --dry-run

    # Smoke test: 50 iterations on a single A10G (~5-10 min, a few dollars):
    python training/scripts/launch_rl.py --max-iterations 50 --instance-type ml.g5.xlarge

    # Larger run:
    python training/scripts/launch_rl.py --task Isaac-Velocity-Flat-Anymal-D-v0 \
        --num-envs 4096 --max-iterations 1500 --instance-type ml.g5.12xlarge
"""

import argparse
import json
import os
import sys

import boto3

REGION = os.environ.get("AWS_DEFAULT_REGION", "us-west-2")
PROJECT_NAME = os.environ.get("PROJECT_NAME", "physical-ai")
ENVIRONMENT = os.environ.get("ENVIRONMENT", "dev")


def _account() -> str:
    return boto3.client("sts", region_name=REGION).get_caller_identity()["Account"]


def _ssm_get(name: str) -> str | None:
    """Try to read an SSM parameter. Returns None if not found."""
    try:
        ssm = boto3.client("ssm", region_name=REGION)
        return ssm.get_parameter(Name=name)["Parameter"]["Value"]
    except Exception:
        return None


def _role_arn() -> str:
    return os.environ.get(
        "SAGEMAKER_ROLE_ARN",
        _ssm_get(f"/{PROJECT_NAME}/sagemaker-role-arn")
        or f"arn:aws:iam::{_account()}:role/{PROJECT_NAME}-{ENVIRONMENT}-sagemaker-role",
    )


def _isaac_lab_image() -> str:
    if os.environ.get("ISAAC_LAB_IMAGE"):
        return os.environ["ISAAC_LAB_IMAGE"]
    ecr_uri = _ssm_get(f"/{PROJECT_NAME}/ecr/isaac-lab")
    if ecr_uri:
        return f"{ecr_uri}:3.0.0-beta2-g1" if ":" not in ecr_uri.rsplit("/", 1)[-1] else ecr_uri
    return f"{_account()}.dkr.ecr.{REGION}.amazonaws.com/{PROJECT_NAME}/isaac-lab:3.0.0-beta2-g1"


def _bucket() -> str:
    return os.environ.get(
        "DATASETS_BUCKET",
        _ssm_get(f"/{PROJECT_NAME}/datasets-bucket")
        or f"{PROJECT_NAME}-{ENVIRONMENT}-datasets-{_account()}",
    )


def _checkpoints_bucket() -> str:
    return os.environ.get(
        "CHECKPOINTS_BUCKET",
        _ssm_get(f"/{PROJECT_NAME}/checkpoints-bucket")
        or f"{PROJECT_NAME}-{ENVIRONMENT}-checkpoints-{_account()}",
    )


def launch(task: str, num_envs: int, max_iterations: int, framework: str,
           instance_type: str, runtime_min: int, instance_count: int, dry_run: bool):
    job_name = f"isaac-lab-rl-{int(__import__('time').time())}"
    bucket = _checkpoints_bucket()
    role = _role_arn()
    image = _isaac_lab_image()
    output = f"s3://{bucket}/isaac-lab/"

    hyperparams = {
        "task": task,
        "num_envs": str(num_envs),
        "max_iterations": str(max_iterations),
        "framework": framework,
        # mode defaults to "train" in the container entrypoint
    }

    # Multi-node NCCL communication is handled by the container entrypoint
    # (containers/isaac-lab/sm-train-entrypoint.sh), which parses SageMaker's
    # resourceconfig.json and launches torchrun with --nnodes/--node_rank/--rdzv_endpoint.
    job_request = {
        "TrainingJobName": job_name,
        "RoleArn": role,
        "AlgorithmSpecification": {"TrainingImage": image, "TrainingInputMode": "File"},
        "OutputDataConfig": {"S3OutputPath": output},
        "ResourceConfig": {
            "InstanceType": instance_type,
            "InstanceCount": instance_count,
            "VolumeSizeInGB": 100,
        },
        "StoppingCondition": {"MaxRuntimeInSeconds": runtime_min * 60},
        "HyperParameters": hyperparams,
    }

    print(f"{'='*60}")
    print(f"  Standalone Isaac Lab RL training (no GR00T)")
    print(f"  Job:        {job_name}")
    print(f"  Task:       {task}")
    print(f"  Envs/iters: {num_envs} envs, {max_iterations} iterations ({framework})")
    print(f"  Instance:   {instance_type} x {instance_count}")
    print(f"  Image:      {image}")
    print(f"  Output:     {output}")
    print(f"{'='*60}")

    if task.startswith("PickAndPlaceUR3") or task.startswith("PickAndPlace-UR3"):
        print("\n  WARNING: the UR3 task is not yet GPU-validated and won't resolve in a "
              "training job. Use the validated Isaac-Velocity-Flat-G1-v0 task.\n")

    if instance_count > 1:
        print(f"  NOTE: Multi-node training ({instance_count} instances) is wired but "
              "not yet validated on hardware.\n")

    if dry_run:
        print("[dry-run] Would call sagemaker.create_training_job with:\n")
        print(json.dumps(job_request, indent=2))
        print("\n[dry-run] No AWS calls made.")
        return job_name

    sm = boto3.client("sagemaker", region_name=REGION)
    sm.create_training_job(**job_request)
    print(f"  Launched. Monitor:")
    print(f"    aws sagemaker describe-training-job --training-job-name {job_name} --region {REGION}")
    return job_name


def main():
    p = argparse.ArgumentParser(description="Standalone Isaac Lab RL launcher (no GR00T)")
    p.add_argument("--task", default="Isaac-Velocity-Flat-G1-v0",
                   help="Isaac Lab task id (G1 flat locomotion is the validated default)")
    p.add_argument("--num-envs", type=int, default=4096)
    p.add_argument("--max-iterations", type=int, default=50)
    p.add_argument("--framework", default="rsl_rl", choices=["rsl_rl", "skrl", "rl_games"])
    p.add_argument("--instance-type", default="ml.g5.xlarge",
                   help="Use a G-family GPU (G5/G6/G6e). P-family lacks RT Cores → Isaac Sim crashes.")
    p.add_argument("--instance-count", type=int, default=1,
                   help="Number of instances for multi-node training (default: 1)")
    p.add_argument("--runtime-min", type=int, default=60, help="Max runtime in minutes")
    p.add_argument("--dry-run", action="store_true", help="Preview the job; make no AWS calls")
    args = p.parse_args()

    launch(args.task, args.num_envs, args.max_iterations, args.framework,
           args.instance_type, args.runtime_min, args.instance_count, args.dry_run)


if __name__ == "__main__":
    main()
