# AWS Batch GR00T Training Guide

Fine-tune GR00T N1.6 on AWS Batch with GPU instances. This guide walks through deploying the infrastructure with Terraform, building the training container, and submitting fine-tuning jobs.

---

## Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│ 1. Foundation (Terraform)                                                │
│    S3 buckets · ECR repos · IAM roles · VPC                              │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 2. GR00T Infra (Terraform)                                               │
│    CodeBuild project · Batch compute env · Job queue                     │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 3. Container Build (CodeBuild)                                           │
│    PyTorch DLC base + Isaac-GR00T N1.6 + flash-attn → ECR                │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 4. Upload Training Data to S3                                            │
│    Convert teleop recordings → LeRobot v2 → S3                           │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 5. Training Job (AWS Batch)                                              │
│    g5.12xlarge (4× A10G) · GR00T fine-tuning · Checkpoints → S3         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Time:** ~2-3 hours (setup + container build + first training run)
**Cost:** ~$56 for full training (5000 steps on g5.12xlarge EC2 pricing)

---

## Prerequisites

| Requirement | How to verify |
|-------------|---------------|
| AWS CLI configured | `aws sts get-caller-identity` |
| Terraform >= 1.5 | `terraform --version` |
| Foundation deployed | `aws ssm get-parameter --name /physical-ai/vpc-id --region us-east-2` |
| HuggingFace token | Create at https://huggingface.co/settings/tokens |
| EC2 GPU quota for g6e.4xlarge | Check EC2 Service Quotas |

---

## Step 1: Deploy Foundation (if not already done)

If you already deployed foundation for Isaac Lab, skip this step — it's shared.

```bash
cd foundation/infra
terraform init
terraform plan -var="aws_region=us-east-2"
terraform apply -var="aws_region=us-east-2"
```

---

## Step 2: Deploy GR00T Infrastructure

```bash
cd gr00t-training-on-aws/infra
terraform init
terraform plan -var="aws_region=us-east-2" -var="enable_batch=true"
terraform apply -var="aws_region=us-east-2" -var="enable_batch=true"
```

**What this creates:**

| Resource | Purpose |
|----------|---------|
| CodeBuild: `physical-ai-dev-groot-training-build` | Builds the GR00T training container |
| CodeBuild: `physical-ai-dev-groot-inference-build` | Builds the GR00T inference container |
| Batch compute env: `physical-ai-dev-groot-compute` | g5.12xlarge (4× A10G, 512 GiB disk) |
| Batch job queue: `physical-ai-dev-groot-queue` | Job scheduling |
| Security group | Outbound access for ECR/S3/HuggingFace |
| IAM roles | CodeBuild + Batch instance access |

**Verify:**
```bash
terraform output
# batch_compute_environment = "physical-ai-dev-groot-compute"
# batch_job_queue = "physical-ai-dev-groot-queue"
```

---

## Step 3: Build the Training Container

### 3a. Package and upload the source

```bash
# From repo root
zip -r /tmp/source.zip . \
  -x ".git/*" \
  -x "*/node_modules/*" \
  -x "*/.terraform/*" \
  -x "*/terraform.tfstate*"

aws s3 cp /tmp/source.zip \
  s3://physical-ai-dev-datasets-<ACCOUNT_ID>/codebuild-source/source.zip \
  --region us-east-2
```

### 3b. Trigger the build

```bash
aws codebuild start-build \
  --project-name "physical-ai-dev-groot-training-build" \
  --region us-east-2 \
  --source-type-override S3 \
  --source-location-override \
    "physical-ai-dev-datasets-<ACCOUNT_ID>/codebuild-source/source.zip"
```

> **Note:** The GR00T training image builds from `nvcr.io/nvidia/pytorch`, which NGC serves to anonymous pulls, so an NGC API key is optional. If `physical-ai/ngc-api-key` exists and the build role can read it, the buildspec logs in to `nvcr.io`; otherwise it pulls anonymously.

### 3c. Monitor progress

```bash
aws codebuild batch-get-builds \
  --ids "<BUILD_ID>" \
  --region us-east-2 \
  --query 'builds[0].{status:buildStatus,phase:currentPhase}'
```

**Build time:** ~15-30 minutes.

---

## Step 4: Upload Training Data

GR00T expects data in LeRobot v2.0 format in S3.

### Option A: Use the bundled UR3 sample dataset

The repo includes 27 real UR3 pick-and-place teleoperation episodes. Download via Git LFS, convert, and upload:

```bash
# 1. Pull the dataset from Git LFS (~1.7 GB)
git lfs pull --include="training/data/ur3_episodes_001_027.zip"

# 2. Extract the Zarr episodes
unzip -o training/data/ur3_episodes_001_027.zip -d training/data/episodes

# 3. Install conversion dependencies (if not already installed)
pip install opencv-python-headless zarr pyarrow pandas

# 4. Convert Zarr episodes to LeRobot v2.0 format
python3 training/gr00t/convert_zarr_to_lerobot.py \
  --episodes-dir training/data/episodes/episodes \
  --output-dir training/data/ur3_lerobot_dataset

# 5. Upload to S3
aws s3 sync training/data/ur3_lerobot_dataset/ \
  s3://physical-ai-dev-datasets-<ACCOUNT_ID>/groot-data/ur3/dataset/ \
  --region us-east-2
```

### Option B: Use your own teleop data

If you have your own robot teleoperation data in Zarr format, convert and upload using the same pipeline:

```bash
python3 training/gr00t/convert_zarr_to_lerobot.py \
  --episodes-dir /path/to/your/episodes \
  --output-dir /tmp/my_lerobot_dataset

aws s3 sync /tmp/my_lerobot_dataset/ \
  s3://physical-ai-dev-datasets-<ACCOUNT_ID>/groot-data/custom/dataset/ \
  --region us-east-2
```

### Required dataset structure (LeRobot v2.0)

```
dataset/
├── data/chunk-000/                          # Parquet files (actions, states)
│   ├── episode_000000.parquet
│   └── ...
├── meta/
│   ├── info.json                            # Dataset metadata (robot_type, fps)
│   ├── episodes.jsonl                       # Episode lengths and tasks (REQUIRED)
│   ├── modality.json                        # GR00T embodiment config (arm/gripper mapping)
│   ├── tasks.jsonl                          # Task vocabulary
│   └── stats.json                           # Dataset statistics
└── videos/chunk-000/
    └── observation.images.wrist/            # Wrist camera MP4s (one per episode)
```

---

## Step 5: Store HuggingFace Token

The training job needs to download the GR00T N1.6 base model weights:

```bash
aws secretsmanager create-secret \
  --name "physical-ai/hf-token" \
  --secret-string "<YOUR_HF_TOKEN>" \
  --region us-east-2 \
  --description "HuggingFace token for downloading GR00T base model"
```

---

## Step 6: Register a Job Definition

```bash
aws batch register-job-definition \
  --job-definition-name "physical-ai-dev-groot-training" \
  --type container \
  --region us-east-2 \
  --container-properties '{
    "image": "<ACCOUNT_ID>.dkr.ecr.us-east-2.amazonaws.com/physical-ai/gr00t-training:latest",
    "command": ["/opt/ml/code/batch-train"],
    "resourceRequirements": [
      {"type": "VCPU", "value": "48"},
      {"type": "MEMORY", "value": "180000"},
      {"type": "GPU", "value": "4"}
    ],
    "environment": [
      {"name": "BASE_MODEL", "value": "nvidia/GR00T-N1.6-3B"},
      {"name": "DATASET_S3_URI", "value": "s3://physical-ai-dev-datasets-<ACCOUNT_ID>/groot-data/ur3/dataset/"},
      {"name": "MAX_STEPS", "value": "5000"},
      {"name": "BATCH_SIZE", "value": "8"},
      {"name": "LEARNING_RATE", "value": "1e-4"},
      {"name": "GRAD_ACCUM_STEPS", "value": "4"},
      {"name": "CHECKPOINT_BUCKET", "value": "physical-ai-dev-checkpoints-<ACCOUNT_ID>"},
      {"name": "HF_TOKEN", "value": "<YOUR_HF_TOKEN>"}
    ]
  }'
```

> **Security note:** For production, store HF_TOKEN in Secrets Manager and reference it via the job definition's `secrets` field instead of plaintext environment variables.

---

## Step 7: Submit a Training Job

### Smoke test (100 steps, ~15 min)

```bash
aws batch submit-job \
  --job-name "groot-finetune-smoke-test" \
  --job-queue "physical-ai-dev-groot-queue" \
  --job-definition "physical-ai-dev-groot-training" \
  --region us-east-2 \
  --container-overrides '{
    "environment": [
      {"name": "MAX_STEPS", "value": "100"}
    ]
  }'
```

### Full training (5000 steps, ~11 hrs)

```bash
aws batch submit-job \
  --job-name "groot-finetune-full" \
  --job-queue "physical-ai-dev-groot-queue" \
  --job-definition "physical-ai-dev-groot-training" \
  --region us-east-2
```

---

## Step 8: Monitor and Retrieve Checkpoints

### Monitor job status

```bash
aws batch describe-jobs \
  --jobs "<JOB_ID>" \
  --region us-east-2 \
  --query 'jobs[0].{status:status,reason:statusReason}'
```

### Retrieve checkpoints

After the job completes:

```bash
aws s3 ls s3://physical-ai-dev-checkpoints-<ACCOUNT_ID>/checkpoints/groot/<JOB_ID>/ \
  --region us-east-2 --recursive
```

Download the fine-tuned model:
```bash
aws s3 cp \
  s3://physical-ai-dev-checkpoints-<ACCOUNT_ID>/checkpoints/groot/<JOB_ID>/ \
  ./groot-checkpoint/ --recursive --region us-east-2
```

---

## What You'll See During Training

```
=== GR00T N1.6-3B Fine-Tuning ===
  Base model:       nvidia/GR00T-N1.6-3B
  Dataset:          /opt/ml/input/data/training
  Max steps:        5000
  Global batch:     8
  Num GPUs:         4
  [OPT] gradient_checkpointing = True
  Per-device batch size: 2
  Effective batch size: 8 (grad_accum=4)
```

Training logs show loss decreasing over steps. A well-converging run shows loss dropping from ~2-3 down to ~0.5-1.0 over 5000 steps.

---

## Managing Compute Costs

### Keep an instance warm

```bash
aws batch update-compute-environment \
  --compute-environment "physical-ai-dev-groot-compute" \
  --compute-resources 'minvCpus=48,desiredvCpus=48' \
  --region us-east-2
```

> **Warning:** g5.12xlarge costs ~$5.67/hr. Only keep warm during active iteration sessions.

### Scale back to zero

```bash
aws batch update-compute-environment \
  --compute-environment "physical-ai-dev-groot-compute" \
  --compute-resources 'minvCpus=0,desiredvCpus=0' \
  --region us-east-2
```

---

## Configuration Variables (Terraform)

| Variable | Default | Description |
|----------|---------|-------------|
| `enable_batch` | `false` | Set to `true` to deploy Batch resources |
| `batch_instance_type` | `g5.12xlarge` | 4× A10G GPUs (24 GB each) |
| `batch_max_vcpus` | `96` | Maximum vCPUs for auto-scaling |

---

## Validation Status

The following has been validated end-to-end on AWS Batch (g6e.4xlarge, us-east-2):

| Step | Status | Notes |
|------|--------|-------|
| Terraform deploy (Foundation + GR00T infra) | ✅ Validated | VPC, Batch compute, job queue, CodeBuild |
| Container build (CodeBuild) | ✅ Validated | NGC base + GR00T N1.6 SDK + all deps |
| GR00T N1.6-3B model loading | ✅ Validated | 3.2B params, Eagle backbone, embodiment config |
| S3 dataset download (`aws s3 sync`) | ✅ Validated | UR3 LeRobot v2.0 data (27 episodes) |
| Training (10 steps) | ✅ Validated | Fine-tuning completed successfully |
| Checkpoint upload to S3 | ✅ Validated | Checkpoints synced to checkpoints bucket |

---

## Troubleshooting

| Problem | Cause | Solution |
|---------|-------|----------|
| Job stuck in `RUNNABLE` | No GPU capacity or quota | Check EC2 g5 quota; verify private subnets cover GPU AZs |
| `No space left on device` | Disk too small for model + container | Launch template provides 512 GiB (should be sufficient) |
| `HF_TOKEN` errors | Token not set or invalid | Verify token at https://huggingface.co/settings/tokens |
| `CUDA out of memory` | Batch size too large for 24 GB | Reduce `BATCH_SIZE` or increase `GRAD_ACCUM_STEPS` |
| `aws: command not found` | AWS CLI not in container | Checkpoint upload uses boto3 (Python), not CLI |
| Build fails pulling DLC base | CodeBuild can't auth to 763104351884 | Buildspec includes DLC ECR login step |
| Foundation VPC not found | SSM parameter missing | Deploy Foundation first (`cd foundation/infra && terraform apply`) |
| `episodes.jsonl not found` | Dataset uses LeRobot v2.1+ format | GR00T N1.6 requires v2.0 format with `meta/episodes.jsonl`. Convert using `pai groot convert` or use a v2.0 dataset |
| Container runs `train_entrypoint.py` instead of `batch-train` | Docker ENTRYPOINT overrides Batch command | Dockerfile must use `ENTRYPOINT []` with `CMD` (not `ENTRYPOINT ["python", "..."]`) |
| Cached image on warm Batch instance | Old image not pulled after rebuild | Scale compute to 0, wait for instance termination, scale back up |

---

## Teardown

```bash
# 1. Scale compute to zero
aws batch update-compute-environment \
  --compute-environment "physical-ai-dev-groot-compute" \
  --compute-resources 'minvCpus=0,desiredvCpus=0' \
  --region us-east-2

# 2. Wait for instances to terminate (~2 min)

# 3. Destroy GR00T infra
cd gr00t-training-on-aws/infra
terraform destroy -var="aws_region=us-east-2" -var="enable_batch=true"
```

---

## Next Steps

- Deploy the trained model for inference (see `containers/gr00t-inference/`)
- Try different datasets and hyperparameters
- Scale to multi-node training for larger models
- Compare with the [SageMaker path](sagemaker-training-guide.md) for managed training
