"""
Launch GR00T N1.7 Fine-Tuning on SageMaker

Submits a SageMaker Training Job that fine-tunes NVIDIA GR00T N1.7-3B
on a customer's LeRobot-format dataset stored in S3. Checkpoints are written
to the checkpoints bucket (same destination pattern as the N1.6 Batch path),
NOT the datasets bucket used by launch_training.py's N1.6 SageMaker path.

N1.7 requires a multi-GPU instance (ml.g6e.12xlarge, 4x L40S) — DeepSpeed ZeRO
only activates when num_gpus>1, and the default fine-tuning recipe's ~2B
trainable params OOM on a single 48GB GPU regardless of batch size. See
isaac-gr00t-on-aws/n17-sagemaker-training-guide.md for the full rationale.

Usage:
    python launch_training_n17.py \
        --datasets-bucket physical-ai-dev-datasets-123456789 \
        --dataset-prefix groot-data/ur3 \
        --checkpoints-bucket physical-ai-dev-checkpoints-123456789 \
        --role-arn arn:aws:iam::123456789:role/physical-ai-dev-sagemaker-role \
        --ecr-image 123456789.dkr.ecr.us-east-2.amazonaws.com/physical-ai/gr00t-training:n17 \
        --max-steps 10000

    # Dry run (shows config, doesn't launch):
    python launch_training_n17.py --dry-run ...
"""

import argparse
import json
import os
from datetime import datetime

import boto3

# Validated on ml.g6e.12xlarge (4x L40S). ml.g6e.4xlarge (1 GPU) OOMs at the
# optimizer step regardless of batch size — DeepSpeed ZeRO sharding requires
# num_gpus>1 to activate in the training code.
MIN_VALIDATED_INSTANCE = "ml.g6e.12xlarge"


def launch_training_job(
    datasets_bucket: str,
    dataset_prefix: str,
    checkpoints_bucket: str,
    role_arn: str,
    ecr_image: str,
    max_steps: int = 10000,
    batch_size: int = 8,
    gradient_accumulation_steps: int = 2,
    learning_rate: float = 1e-4,
    base_model: str = "nvidia/GR00T-N1.7-3B",
    instance_type: str = MIN_VALIDATED_INSTANCE,
    region: str = "us-east-2",
    dry_run: bool = False,
) -> dict:
    """Submit a SageMaker training job for GR00T N1.7 fine-tuning."""

    if instance_type == "ml.g6e.4xlarge":
        print(
            "  WARNING: ml.g6e.4xlarge (1 GPU) is NOT validated for N1.7 — it OOMs "
            "at the optimizer step regardless of batch size. Use ml.g6e.12xlarge "
            "(4 GPUs) or larger so DeepSpeed ZeRO can shard optimizer memory."
        )

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    job_name = f"gr00t-n17-finetune-{timestamp}"

    training_config = {
        "TrainingJobName": job_name,
        "RoleArn": role_arn,
        "AlgorithmSpecification": {
            "TrainingImage": ecr_image,
            "TrainingInputMode": "File",
        },
        "HyperParameters": {
            "base_model": base_model,
            "max_steps": str(max_steps),
            "batch_size": str(batch_size),
            "gradient_accumulation_steps": str(gradient_accumulation_steps),
            "learning_rate": str(learning_rate),
        },
        "InputDataConfig": [
            {
                "ChannelName": "training",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": f"s3://{datasets_bucket}/{dataset_prefix}/dataset/",
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
            }
        ],
        # Checkpoints go to the checkpoints bucket — same destination pattern
        # as the Batch entrypoint (CHECKPOINT_BUCKET), not the datasets bucket.
        "OutputDataConfig": {
            "S3OutputPath": f"s3://{checkpoints_bucket}/sagemaker-output/",
        },
        "ResourceConfig": {
            "InstanceType": instance_type,
            "InstanceCount": 1,
            "VolumeSizeInGB": 200,
        },
        "StoppingCondition": {
            "MaxRuntimeInSeconds": 86400,  # 24 hours max
        },
    }

    # Inject the HuggingFace token from the environment (both names the SDK reads).
    # Never bake a token or a placeholder into the request.
    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token:
        training_config["Environment"] = {
            "HF_TOKEN": hf_token,
            "HUGGING_FACE_HUB_TOKEN": hf_token,
        }
    else:
        print("  WARNING: HF_TOKEN not set — the N1.7 backbone nvidia/Cosmos-Reason2-2B is a "
              "gated repo, so its download will fail with HTTP 401.")

    # Cost estimation (ml.g6e.12xlarge: ~$8.06/hr on-demand, 4x L40S)
    hourly_rate = {"ml.g6e.12xlarge": 8.06, "ml.g6e.48xlarge": 32.24, "ml.g6e.4xlarge": 2.24}.get(
        instance_type, 8.06
    )
    estimated_hours = (max_steps * 9) / 3600  # ~9 sec/step observed on g6e.12xlarge
    estimated_cost = estimated_hours * hourly_rate

    model_output = f"s3://{checkpoints_bucket}/sagemaker-output/{job_name}/output/model.tar.gz"

    if dry_run:
        return {
            "dry_run": True,
            "job_name": job_name,
            "config": training_config,
            "estimated_duration_hours": round(estimated_hours, 1),
            "estimated_cost_usd": round(estimated_cost, 0),
            "model_output": model_output,
            "note": "Drop --dry-run to launch.",
        }

    # Launch the job
    sm_client = boto3.client("sagemaker", region_name=region)
    sm_client.create_training_job(**training_config)

    print(f"  Job submitted: {job_name}")
    print(f"  Instance: {instance_type}")
    print(f"  Estimated duration: ~{estimated_hours:.1f} hours")
    print(f"  Estimated cost: ~${estimated_cost:.0f}")
    print(f"  Checkpoints will be written to: {model_output}")
    print(f"  Monitor: aws sagemaker describe-training-job --training-job-name {job_name}")

    return {
        "status": "launched",
        "job_name": job_name,
        "estimated_duration_hours": round(estimated_hours, 1),
        "estimated_cost_usd": round(estimated_cost, 0),
        "model_output": model_output,
    }


def main():
    parser = argparse.ArgumentParser(description="Launch GR00T N1.7 fine-tuning on SageMaker")
    parser.add_argument("--datasets-bucket", required=True, help="S3 bucket with training data")
    parser.add_argument("--dataset-prefix", required=True, help="S3 prefix for the dataset")
    parser.add_argument(
        "--checkpoints-bucket", required=True, help="S3 bucket for checkpoint output"
    )
    parser.add_argument("--role-arn", required=True, help="SageMaker execution role ARN")
    parser.add_argument("--ecr-image", required=True, help="ECR URI for the :n17 training container")
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=8, help="Global batch size (pre-accumulation)")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--base-model", default="nvidia/GR00T-N1.7-3B")
    parser.add_argument(
        "--instance-type",
        default=MIN_VALIDATED_INSTANCE,
        help=f"Default {MIN_VALIDATED_INSTANCE} (4 GPUs) — required for DeepSpeed ZeRO sharding",
    )
    parser.add_argument("--region", default="us-east-2")
    parser.add_argument("--dry-run", action="store_true", help="Show config without launching")
    args = parser.parse_args()

    result = launch_training_job(
        datasets_bucket=args.datasets_bucket,
        dataset_prefix=args.dataset_prefix,
        checkpoints_bucket=args.checkpoints_bucket,
        role_arn=args.role_arn,
        ecr_image=args.ecr_image,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        base_model=args.base_model,
        instance_type=args.instance_type,
        region=args.region,
        dry_run=args.dry_run,
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
