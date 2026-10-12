# SageMaker GR00T Training Guide

Fine-tune GR00T N1.6 on Amazon SageMaker. SageMaker provisions GPU instances on demand, runs training, and tears them down automatically.

---

## Prerequisites

| Requirement | How to verify |
|-------------|---------------|
| AWS CLI configured | `aws sts get-caller-identity` |
| Python 3.11+ with boto3 | `python3 -c "import boto3"` |
| Foundation deployed | `aws ssm get-parameter --name /physical-ai/sagemaker-role-arn --region us-east-2` |
| Container image in ECR | `aws ecr describe-images --repository-name physical-ai/gr00t-training --region us-east-2` |
| HuggingFace token | For base model download (set as `HF_TOKEN` env var) |
| SageMaker GPU quota for ml.g6e.4xlarge | **Must request before first use** |

> **Important — Service Quota Request:**
>
> The `ml.g6e.4xlarge` (48 GB L40S GPU) is required for GR00T N1.6-3B fine-tuning. It is NOT available by default.
>
> **Request quota increase before launching jobs:**
> 1. Go to **AWS Console** → **Service Quotas** → **Amazon SageMaker**
> 2. Search for `ml.g6e.4xlarge for training job usage`
> 3. Request increase to at least **1 instance**
> 4. Wait for approval (typically 15-30 minutes)
>
> Without this, jobs fail immediately with `ResourceLimitExceeded`.
>
> **Why g6e.4xlarge?** GR00T N1.6-3B is a 3.2B parameter model that requires ~40 GB GPU memory with gradient checkpointing. The 24 GB A10G (ml.g5.xlarge) OOMs even with batch_size=1. The 48 GB L40S (ml.g6e.4xlarge) fits comfortably.

---

## Step 1: Upload Training Data to S3

Follow [Step 4 in the Batch guide](batch-training-guide.md#step-4-prepare-and-stage-the-dataset) to prepare and upload the UR3 dataset.

---

## Step 2: Launch Training

### Preview (no cost)

```bash
AWS_DEFAULT_REGION=us-east-2 python3 training/gr00t/launch_training.py \
  --s3-bucket physical-ai-dev-datasets-<ACCOUNT_ID> \
  --dataset-prefix groot-data/ur3 \
  --role-arn arn:aws:iam::<ACCOUNT_ID>:role/physical-ai-dev-sagemaker-role \
  --ecr-image <ACCOUNT_ID>.dkr.ecr.us-east-2.amazonaws.com/physical-ai/gr00t-training:latest \
  --base-model nvidia/GR00T-N1.6-3B \
  --max-steps 10 \
  --batch-size 2 \
  --instance-type ml.g6e.4xlarge \
  --region us-east-2 \
  --dry-run
```

### Launch for real

```bash
HF_TOKEN=<YOUR_HF_TOKEN> \
AWS_DEFAULT_REGION=us-east-2 python3 training/gr00t/launch_training.py \
  --s3-bucket physical-ai-dev-datasets-<ACCOUNT_ID> \
  --dataset-prefix groot-data/ur3 \
  --role-arn arn:aws:iam::<ACCOUNT_ID>:role/physical-ai-dev-sagemaker-role \
  --ecr-image <ACCOUNT_ID>.dkr.ecr.us-east-2.amazonaws.com/physical-ai/gr00t-training:latest \
  --base-model nvidia/GR00T-N1.6-3B \
  --max-steps 5000 \
  --batch-size 2 \
  --instance-type ml.g6e.4xlarge \
  --region us-east-2
```

---

## Step 3: Monitor Training

```bash
aws sagemaker describe-training-job \
  --training-job-name <JOB_NAME> \
  --region us-east-2 \
  --query '{Status:TrainingJobStatus,Secondary:SecondaryStatus}'
```

Or view in the [SageMaker console](https://us-east-2.console.aws.amazon.com/sagemaker/home?region=us-east-2#/jobs).

---

## Step 4: Retrieve Checkpoints

SageMaker packages the model output into `model.tar.gz`:

```bash
aws s3 cp \
  s3://physical-ai-dev-datasets-<ACCOUNT_ID>/groot-data/ur3/output/<JOB_NAME>/output/model.tar.gz .

tar -xzf model.tar.gz
```

---

## How It Works

1. `launch_training.py` resolves your ECR image, IAM role, and S3 paths
2. Calls `sagemaker.create_training_job()` with the dataset as an input channel
3. SageMaker provisions `ml.g6e.4xlarge`, pulls the container, mounts S3 data at `/opt/ml/input/data/training/`
4. The container's entrypoint runs `train_entrypoint.py` which:
   - Registers the UR3 embodiment config
   - Downloads the GR00T N1.6-3B base model from HuggingFace
   - Runs fine-tuning with gradient checkpointing
   - Saves checkpoint to `/opt/ml/model/`
5. SageMaker packages `/opt/ml/model/` → `model.tar.gz` → uploads to S3
6. Instance terminates automatically

---

## Cost

| Scenario | Instance | Time | Cost |
|----------|----------|------|------|
| Smoke test (10 steps) | ml.g6e.4xlarge | ~5 min | ~$1 |
| Full training (5000 steps) | ml.g6e.4xlarge | ~6 hrs | ~$15 |

SageMaker auto-terminates — no idle cost.

---

## Troubleshooting

| Problem | Cause | Solution |
|---------|-------|----------|
| `ResourceLimitExceeded` | No quota for ml.g6e.4xlarge | Request quota increase in Service Quotas |
| `CapacityError` | No instances available in region | Retry later or try a different AZ |
| `CUDA out of memory` | Instance too small (ml.g5.xlarge = 24GB) | Use ml.g6e.4xlarge (48GB) or ml.g5.12xlarge (4×24GB) |
| Exit code 127 | Container entrypoint misconfigured | Ensure Dockerfile has proper ENTRYPOINT |
| HF rate limit | HF_TOKEN not set | Export HF_TOKEN before launching |

---

## Validation Status

| Step | Status |
|------|--------|
| SageMaker job launch | ✅ Validated |
| Data mount from S3 | ✅ Validated |
| GR00T N1.6-3B training (ml.g6e.4xlarge) | ✅ Validated |
| Checkpoint to S3 (model.tar.gz) | ✅ Validated |
