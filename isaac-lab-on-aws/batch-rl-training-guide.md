# AWS Batch RL Training Guide

Train Isaac Lab reinforcement learning policies on AWS Batch with GPU instances. This guide walks through deploying the infrastructure with Terraform, building the training container, and submitting jobs.

---

## Overview

This guide uses **Terraform** (Path A) to deploy infrastructure for RL training on AWS Batch. The pipeline:

```
┌─────────────────────────────────────────────────────────────────────────┐
│ 1. Foundation (Terraform)                                                │
│    S3 buckets · ECR repos · IAM roles · SSM parameters                   │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 2. Isaac Lab Infra (Terraform)                                           │
│    CodeBuild project · VPC · Batch compute env · Job queue               │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 3. Container Build (CodeBuild)                                           │
│    Pull NGC base (16 GB) · Install RL deps · Push to ECR                 │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 4. Training Job (AWS Batch)                                              │
│    g6e.4xlarge · Isaac Lab PPO · 1024 parallel envs · Checkpoints → S3   │
└─────────────────────────────────────────────────────────────────────────┘
```

**Time:** ~2 hours (mostly waiting for container build + training)
**Cost:** ~$5-10 per training run (g6e.4xlarge for 15-30 min)

---

## Prerequisites

| Requirement | How to verify |
|-------------|---------------|
| AWS CLI configured | `aws sts get-caller-identity` |
| Terraform >= 1.5 | `terraform --version` |
| NVIDIA NGC API key | Generate at https://ngc.nvidia.com/setup/api-key |
| GPU quota for g6e instances | Check EC2 service quotas in your target region |

---

## Step 1: Deploy Foundation Infrastructure

The foundation creates shared resources that all other components depend on.

```bash
cd foundation/infra
terraform init
terraform plan -var="aws_region=us-east-2"
terraform apply -var="aws_region=us-east-2"
```

**What this creates:**

| Resource | Purpose |
|----------|---------|
| S3: `physical-ai-dev-datasets-<ACCOUNT_ID>` | Training datasets |
| S3: `physical-ai-dev-models-<ACCOUNT_ID>` | Trained models (ONNX, TensorRT) |
| S3: `physical-ai-dev-checkpoints-<ACCOUNT_ID>` | Training checkpoints (14-day expiry) |
| ECR: `physical-ai/isaac-lab` | Isaac Lab training container |
| IAM: `physical-ai-dev-sagemaker-role` | Execution role for training |
| SSM Parameters | Discovery mechanism for downstream components |

**Verify:**
```bash
terraform output
# Should show bucket names, role ARN, instance profile
```

---

## Step 2: Store NGC API Key

CodeBuild needs your NGC key to pull the Isaac Lab base image from `nvcr.io`.

```bash
aws secretsmanager create-secret \
  --name "physical-ai/ngc-api-key" \
  --secret-string "<YOUR_NGC_API_KEY>" \
  --region us-east-2 \
  --description "NVIDIA NGC API key for pulling container images from nvcr.io"
```

---

## Step 3: Deploy Isaac Lab Infrastructure

This creates the CodeBuild project, VPC, and AWS Batch resources.

```bash
cd isaac-lab-on-aws/infra
terraform init
terraform plan -var="aws_region=us-east-2" -var="enable_batch=true"
terraform apply -var="aws_region=us-east-2" -var="enable_batch=true"
```

**What this creates:**

| Resource | Purpose |
|----------|---------|
| CodeBuild: `physical-ai-dev-isaac-lab-build` | Builds the Isaac Lab container image |
| VPC: `physical-ai-dev-isaac-lab-vpc` | Networking for Batch compute (10.1.0.0/16) |
| 2 public + 3 private subnets | NAT for outbound, GPU instances in private subnets |
| NAT Gateway | Outbound internet for private subnets (ECR/S3 access) |
| Batch compute env: `physical-ai-dev-rl-compute` | g6e.4xlarge with 512 GiB disk |
| Batch job queue: `physical-ai-dev-rl-queue` | Job scheduling |
| Launch template | 512 GiB gp3 root volume (Isaac Lab image is ~16 GB) |
| Security group | Self-referencing rules for inter-node NCCL |
| IAM roles | Batch instance access to ECR, S3, SSM |

**Verify:**
```bash
terraform output
# batch_compute_environment = "physical-ai-dev-rl-compute"
# batch_job_queue = "physical-ai-dev-rl-queue"
```

### Configuration Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `enable_batch` | `false` | Set to `true` to deploy Batch resources |
| `batch_instance_type` | `g6e.4xlarge` | GPU instance type (1 L40S GPU) |
| `batch_max_vcpus` | `96` | Maximum vCPUs for auto-scaling |

---

## Step 4: Build the Training Container

The container is built entirely in AWS CodeBuild — no local Docker needed.

### 4a. Package and upload the source

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

### 4b. Trigger the build

```bash
aws codebuild start-build \
  --project-name "physical-ai-dev-isaac-lab-build" \
  --region us-east-2 \
  --environment-variables-override \
    "name=NGC_SECRET_NAME,value=physical-ai/ngc-api-key,type=PLAINTEXT" \
  --source-type-override S3 \
  --source-location-override \
    "physical-ai-dev-datasets-<ACCOUNT_ID>/codebuild-source/source.zip"
```

### 4c. Monitor progress

```bash
# Check status
aws codebuild batch-get-builds \
  --ids "<BUILD_ID>" \
  --region us-east-2 \
  --query 'builds[0].{status:buildStatus,phase:currentPhase}'
```

Or watch in the [CodeBuild console](https://us-east-2.console.aws.amazon.com/codesuite/codebuild/projects/physical-ai-dev-isaac-lab-build/history?region=us-east-2).

**Build time:** 30-60 minutes (pulling 16 GB NGC base image is the bottleneck).

**What it produces:** A ~16 GB container image at `<ACCOUNT_ID>.dkr.ecr.us-east-2.amazonaws.com/physical-ai/isaac-lab:3.0.0-beta2-g1`

---

## Step 5: Register a Job Definition

Define how Batch should run the training container:

```bash
aws batch register-job-definition \
  --job-definition-name "physical-ai-dev-isaac-lab-rl" \
  --type container \
  --region us-east-2 \
  --container-properties '{
    "image": "<ACCOUNT_ID>.dkr.ecr.us-east-2.amazonaws.com/physical-ai/isaac-lab:3.0.0-beta2-g1",
    "command": ["/opt/ml/code/batch-train"],
    "resourceRequirements": [
      {"type": "VCPU", "value": "16"},
      {"type": "MEMORY", "value": "60000"},
      {"type": "GPU", "value": "1"}
    ],
    "environment": [
      {"name": "TASK", "value": "Isaac-Velocity-Flat-Anymal-D-v0"},
      {"name": "NUM_ENVS", "value": "1024"},
      {"name": "MAX_ITERATIONS", "value": "100"},
      {"name": "PROC_PER_NODE", "value": "1"},
      {"name": "FRAMEWORK", "value": "rsl_rl"},
      {"name": "CHECKPOINT_BUCKET", "value": "physical-ai-dev-checkpoints-<ACCOUNT_ID>"}
    ]
  }'
```

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TASK` | `Isaac-Velocity-Flat-Anymal-D-v0` | Isaac Lab task name |
| `NUM_ENVS` | `4096` | Parallel simulation environments |
| `MAX_ITERATIONS` | `100` | PPO training iterations |
| `PROC_PER_NODE` | `4` | Number of GPUs (set to 1 for g6e.4xlarge) |
| `FRAMEWORK` | `rsl_rl` | RL framework: `rsl_rl`, `skrl`, or `rl_games` |
| `CHECKPOINT_BUCKET` | — | S3 bucket for checkpoint upload (required) |
| `CHECKPOINT_PREFIX` | `checkpoints/<job_id>` | S3 key prefix for checkpoints |

---

## Step 6: Submit a Training Job

```bash
aws batch submit-job \
  --job-name "isaac-lab-rl-training" \
  --job-queue "physical-ai-dev-rl-queue" \
  --job-definition "physical-ai-dev-isaac-lab-rl" \
  --region us-east-2
```

### Monitor the job

```bash
aws batch describe-jobs \
  --jobs "<JOB_ID>" \
  --region us-east-2 \
  --query 'jobs[0].{status:status,reason:statusReason}'
```

Job lifecycle: `SUBMITTED` → `PENDING` → `RUNNABLE` → `STARTING` → `RUNNING` → `SUCCEEDED`

Or monitor in the [AWS Batch console](https://us-east-2.console.aws.amazon.com/batch/home?region=us-east-2#jobs).

---

## Step 7: Verify Checkpoints in S3

After the job completes, checkpoints are uploaded to S3:

```bash
aws s3 ls s3://physical-ai-dev-checkpoints-<ACCOUNT_ID>/checkpoints/<JOB_ID>/ \
  --region us-east-2 --recursive
```

Checkpoint files include model weights (`.pt`), training logs, and tensorboard data.

---

## What You'll See During Training

Isaac Lab logs each PPO iteration to CloudWatch:

```
Learning iteration 50/100
Computation: 31367 steps/s (collection: 0.698s, learning 0.086s)
Mean reward: -0.30
Mean episode length: 73.70
```

- **steps/s** — training throughput across all parallel envs
- **Mean reward** — should climb from negative toward 0 and above over iterations
- **Mean episode length** — grows as the robot survives longer

A 100-iteration smoke test takes ~2 minutes of training time. A full 1500-iteration run takes 20-30 minutes.

---

## Managing Compute Costs

### Keep an instance warm (faster job starts)

By default, the compute environment scales to zero when idle. To keep one instance warm for faster iterations:

```bash
aws batch update-compute-environment \
  --compute-environment "physical-ai-dev-rl-compute" \
  --compute-resources 'minvCpus=16,desiredvCpus=16' \
  --region us-east-2
```

This keeps a g6e.4xlarge running (~$1.86/hr). Jobs start immediately instead of waiting 3-5 min for provisioning.

### Scale back to zero when done

```bash
aws batch update-compute-environment \
  --compute-environment "physical-ai-dev-rl-compute" \
  --compute-resources 'minvCpus=0,desiredvCpus=0' \
  --region us-east-2
```

---

## Troubleshooting

| Problem | Cause | Solution |
|---------|-------|----------|
| Job stuck in `RUNNABLE` | No GPU capacity in the configured AZs | Ensure private subnets cover the AZ with GPU availability; check EC2 quota |
| `No space left on device` | Default 30 GiB EBS too small for 16 GB image | Increase launch template volume (default is 512 GiB in our config) |
| `aws: command not found` in container | AWS CLI not installed in Isaac Lab image | Use boto3 (Python) for AWS operations; the entrypoint handles S3 upload via boto3 |
| S3 bucket tag errors | Parentheses/commas in tag values | Use dashes instead of special characters in tag values |
| CodeBuild can't download source | Missing S3 permissions on CodeBuild role | Add `s3:GetObject` and `s3:ListBucket` to the CodeBuild IAM policy |
| Build fails at `COPY training/` | Missing source directories | Ensure `training/` and `workflows/` exist in the repo root |
| Checkpoints not uploaded to S3 | Old container image without boto3 upload | Rebuild the container after updating `batch-train-entrypoint.sh` |
| NCCL errors (multi-node) | Network interface not detected | Set `NCCL_SOCKET_IFNAME` or let NCCL autodetect |

---

## Teardown

```bash
# 1. Scale compute to zero
aws batch update-compute-environment \
  --compute-environment "physical-ai-dev-rl-compute" \
  --compute-resources 'minvCpus=0,desiredvCpus=0' \
  --region us-east-2

# 2. Wait for instances to terminate (~2 min)

# 3. Destroy Isaac Lab infra
cd isaac-lab-on-aws/infra
terraform destroy -var="aws_region=us-east-2" -var="enable_batch=true"

# 4. Destroy Foundation (optional — other components may depend on it)
cd foundation/infra
terraform destroy -var="aws_region=us-east-2"
```

---

## Architecture Notes

- **Why private subnets?** Batch GPU instances don't need inbound internet access. Private subnets + NAT gateway give outbound access (ECR image pulls, S3 uploads) without exposing instances publicly.
- **Why 512 GiB disk?** The Isaac Lab container image is ~16 GB. Docker needs to decompress and store layers, and training writes checkpoints locally before upload. 512 GiB provides comfortable headroom.
- **Why boto3 instead of AWS CLI?** The Isaac Lab container is based on NVIDIA's NGC image which includes Python + pip packages but not the AWS CLI. boto3 is installed for S3 integration.
- **Why `set -e` in the entrypoint?** Fails fast on any error during training setup. The S3 upload runs after training completes regardless of training exit code.

---

## Next Steps

- Increase `MAX_ITERATIONS` to 1500+ for a converged policy
- Try different tasks: `Isaac-Velocity-Rough-Anymal-D-v0` (terrain), `Isaac-Reach-Franka-v0` (arm)
- Export the trained policy to ONNX for edge deployment (see [Lab 5 Step 4](../workshop/lab-5-rl-refinement-with-isaac.md))
- Add your own custom environment in `training/envs/`
