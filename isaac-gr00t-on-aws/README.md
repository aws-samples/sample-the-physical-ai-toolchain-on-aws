# GR00T Training on AWS

Fine-tune [NVIDIA GR00T](https://developer.nvidia.com/isaac/gr00t) — a 3B-parameter Vision-Language-Action (VLA) model for humanoid robots — on your own teleoperation data using AWS GPU infrastructure. Supports both **N1.6** and **N1.7** (Cosmos-Reason2-2B backbone, structured reasoning).

GR00T learns to predict motor commands from camera images by watching human demonstrations. Fine-tune it on 27+ episodes of your robot performing a task, and it generalizes to new situations.

> **Choosing a version?** N1.7 requires a bigger/more expensive instance (`ml.g6e.12xlarge`, 4 GPUs) than N1.6 (`ml.g6e.4xlarge`, 1 GPU) — see [EC2 / SageMaker Instance Recommendation](n17-sagemaker-training-guide.md#ec2--sagemaker-instance-recommendation) before picking a version.

---

## What You'll Build

```
┌──────────────────────────────────────────────────────────────┐
│  Foundation (shared base)                                     │
│  S3 buckets · ECR repos · IAM roles · VPC                     │
└───────────────────────────┬──────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│  Container Build (CodeBuild)                                  │
│  PyTorch DLC base + Isaac-GR00T N1.6 + flash-attn → ECR       │
└───────────────────────────┬──────────────────────────────────┘
                            │
                ┌───────────┴───────────┐
                │                       │
                ▼                       ▼
┌──────────────────────┐  ┌──────────────────────────┐
│  Option A: SageMaker │  │  Option B: AWS Batch      │
│  Managed training    │  │  Self-managed compute     │
│  Pay per job         │  │  VPC + GPU fleet          │
│  Auto-teardown       │  │  Full control             │
└──────────────────────┘  └──────────────────────────┘
                │                       │
                └───────────┬───────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│  Checkpoints → S3                                             │
│  s3://physical-ai-dev-checkpoints-<ACCOUNT_ID>/               │
└──────────────────────────────────────────────────────────────┘
```

Both options use the **same container image** and produce the same output — fine-tuned GR00T checkpoints in S3.

---

## Which Option Should I Use?

| | **SageMaker** | **AWS Batch** |
|---|---|---|
| **Best for** | Quick experiments, workshops, one-off runs | Repeated runs, persistent fleet, cost control |
| **Setup effort** | Minimal — IAM role + ECR image | Moderate — Batch compute env + job queue |
| **GPU provisioning** | Managed by SageMaker | You control instance type, scaling |
| **Cost model** | Pay per second of training | EC2 pricing (can keep instances warm) |
| **Teardown** | Automatic when job completes | Manual (scale to zero or destroy) |
| **Multi-GPU** | Automatic (entrypoint detects GPUs) | Same entrypoint, same behavior |
| **VPC required** | No | Yes (uses Foundation VPC) |
| **Checkpoint storage** | S3 (via SageMaker output path) | S3 (via boto3 upload in entrypoint) |

**Choose SageMaker when:**
- You want the fastest path to a fine-tuned model
- You're running workshop labs or experimenting
- You don't need persistent GPU infrastructure

**Choose AWS Batch when:**
- You want full control over the compute fleet
- You need to keep GPU instances warm between runs
- You're iterating rapidly on training configs

---

## Shared Prerequisites

Both paths require:

1. **Foundation infrastructure deployed** — S3 buckets, ECR repos, IAM roles, VPC
2. **Container image built** — GR00T training container pushed to ECR
3. **HuggingFace token** — for downloading the base model weights
4. **Training data** — LeRobot v2 format uploaded to S3

---

## Training Parameters

### N1.6

| Parameter | Default | Description |
|-----------|---------|-------------|
| `base_model` | `nvidia/GR00T-N1.6-3B` | HuggingFace model name |
| `max_steps` | `5000` | Training steps (~11 hrs on g5.12xlarge) |
| `batch_size` | `8` | Global batch size |
| `learning_rate` | `1e-4` | Learning rate |
| `gradient_accumulation_steps` | `4` | Gradient accumulation |
| `CHECKPOINT_BUCKET` | — | S3 bucket for checkpoint upload (Batch only) |

### N1.7

| Parameter | Default | Description |
|-----------|---------|-------------|
| `base_model` | `nvidia/GR00T-N1.7-3B` | HuggingFace model name |
| `max_steps` | `10000` | Training steps |
| `batch_size` | `8` | Global batch size (pre-accumulation) |
| `gradient_accumulation_steps` | `2` | Gradient accumulation |
| **Instance** | `ml.g6e.12xlarge` (4 GPUs) | **Required** — single-GPU OOMs regardless of batch size |

---

## Deployment Guides

### [N1.6 SageMaker Guide →](sagemaker-training-guide.md)

SageMaker deployment for N1.6 on `ml.g6e.4xlarge` (single GPU).

### [N1.7 SageMaker Guide →](n17-sagemaker-training-guide.md)

SageMaker deployment for N1.7 on `ml.g6e.12xlarge` (4 GPUs, DeepSpeed ZeRO required). Includes EC2 instance sizing rationale and OOM troubleshooting.

### [AWS Batch Guide →](batch-training-guide.md)

Terraform-based Batch deployment for N1.6: Foundation → VPC → Batch compute → Container build → Job submission → S3 checkpoints. **Fully validated end-to-end** with UR3 teleoperation data.

### [Run from VAMS (optional) →](../workflows/vams-bridge/README.md)

Launch fine-tuning from the [Visual Asset Management System (VAMS)](https://github.com/awslabs/visual-asset-management-system) on a dataset asset and get the model written back to it. Deploys on top of the Batch (or SageMaker Pipeline) setup above; nothing here needs it.

---

## GR00T N1.6 vs N1.7

| | N1.6 | N1.7 |
|---|---|---|
| Backbone | Eagle-Block2A-2B | Cosmos-Reason2-2B (Qwen3-VL) |
| Base model | `nvidia/GR00T-N1.6-3B` | `nvidia/GR00T-N1.7-3B` |
| **Minimum validated instance** | `ml.g6e.4xlarge` (1× L40S, 48GB) | `ml.g6e.12xlarge` (4× L40S, 192GB) |
| Reasoning | Basic | Structured (task + subtask level) |
| Training API | `experiment.run(config)` | `launch_finetune.py` CLI |
| Install | pip + manual venv | Native `uv sync` |
| Python | 3.11 | 3.12 |
| Action horizon | Up to 16 | Up to 40 |
| DROID support | No | Yes (out of box) |
| Multi-step tasks | Limited | Improved coherence |

---

## Data Format

GR00T expects data in **LeRobot v2** format (HuggingFace's standard for robot training):

- **Parquet files** — numeric data (joint positions, actions, rewards)
- **MP4 files** — camera video (wrist cam, overhead cam)
- **meta/modality.json** — describes the action space and sensor layout

Use `training/gr00t/convert_zarr_to_lerobot.py` to convert from Zarr/ROS bag format.

---

## Estimated Costs

### N1.6

| Scenario | Instance | Time | Cost |
|----------|----------|------|------|
| Smoke test (100 steps) | ml.g5.xlarge | ~15 min | ~$2 |
| Full training (5000 steps) | ml.g5.12xlarge | ~11 hrs | ~$79 |
| Full training (5000 steps) | g5.12xlarge (Batch) | ~11 hrs | ~$56 (EC2 pricing) |

### N1.7

| Scenario | Instance | Time | Cost |
|----------|----------|------|------|
| Smoke test (100 steps) | ml.g6e.12xlarge | ~15 min | ~$2 |
| Full training (10,000 steps) | ml.g6e.12xlarge | ~8-10 hrs (est.) | ~$65-80 |

---

## Drive the Trained Policy with an Agent (Strands Agents)

Once GR00T is fine-tuned, [**strands-robots**](https://github.com/strands-labs/robots)
turns the checkpoint into closed-loop robot behavior driven by natural language.
It ships a first-class GR00T policy provider (`strands_robots.policies.groot`), so the
same `model.tar.gz` this pipeline produces runs on a MuJoCo sim twin (no hardware) or
a real arm (`mode="real"`) behind one interface:

```python
from strands import Agent
from strands_robots import Robot

robot = Robot("so101")                 # MuJoCo sim by default; mode="real" for hardware
robot.run_policy(                      # roll out the fine-tuned GR00T checkpoint
    policy_provider="groot",
    # Only use checkpoints you trust — a policy artifact can execute arbitrary code on load.
    policy_config={"pretrained_name_or_path": "s3://.../output/model.tar.gz"},
)
Agent(tools=[robot])("pick up the red cube")   # agent supervises the rollout in NL
```

strands-robots also **records demos in LeRobot v2** (the exact format this pipeline
trains on) via teleoperation, closing the data loop. See
[strands-agents-on-aws](../strands-agents-on-aws/) for the agentic orchestration layer
and `pai agent sim --policy s3://.../model.tar.gz` to run it from the CLI.
