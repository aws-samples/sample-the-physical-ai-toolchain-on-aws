# SageMaker GR00T N1.7 Training Guide

Fine-tune [GR00T N1.7](https://huggingface.co/blog/nvidia/gr00t-n1-7) on Amazon SageMaker. N1.7 replaces N1.6's Eagle backbone with **Cosmos-Reason2-2B** (Qwen3-VL) and adds task/subtask-level reasoning, but it has materially higher memory requirements — plan the instance size accordingly.

---

## EC2 / SageMaker Instance Recommendation

> **Use `ml.g6e.12xlarge` (4× L40S, 192 GB total VRAM). Do NOT use `ml.g6e.4xlarge` (1× L40S, 48 GB) — it will OOM.**

This differs from N1.6, which fits on a single 48 GB GPU. N1.7's default fine-tuning recipe (`--tune-visual --tune-projector --tune-diffusion-model`) has ~2.03B trainable parameters (64% of the 3.14B total). The Adam optimizer keeps two fp32 moment buffers per trainable parameter, which alone consumes ~41 GB — before activations, gradients, or KV cache. A single L40S has 44.4 GB usable, so **any batch size** (we validated both 12 and 4) OOMs identically at the optimizer step:

```
torch.OutOfMemoryError: CUDA out of memory. ... GPU 0 has a total capacity of 44.40 GiB
of which 9.31 MiB is free. ... this process has 44.38 GiB memory in use.
```

The `launch_finetune.py` CLI has no `--gradient-checkpointing` flag to work around this on a single GPU. The actual fix is **more GPUs**: the training code auto-enables DeepSpeed ZeRO whenever `num_gpus > 1`, which shards the optimizer states across GPUs instead of replicating them. That's what `ml.g6e.12xlarge` (4 GPUs) gives you.

| Instance | GPUs | VRAM (total) | N1.7 fine-tune? | Notes |
|----------|------|---------------|------------------|-------|
| `ml.g6e.4xlarge` | 1× L40S | 48 GB | ❌ OOMs | Works fine for N1.6, not N1.7 |
| `ml.g6e.12xlarge` | 4× L40S | 192 GB | ✅ Validated | DeepSpeed ZeRO auto-enabled at `num_gpus>1` |
| `ml.g6e.48xlarge` | 8× L40S | 384 GB | ✅ (untested) | Overkill for a 3B model; use for larger batch/throughput |
| `ml.p4d.24xlarge` | 8× A100 40GB | 320 GB | ✅ (untested) | Alternative if L40S quota unavailable |

**Cost note:** `ml.g6e.12xlarge` is ~3x the hourly cost of `ml.g6e.4xlarge`, but it's the smallest instance that actually completes N1.7 fine-tuning. There's no cheaper path with the current CLI.

---

## Prerequisites

| Requirement | How to verify |
|-------------|---------------|
| AWS CLI configured | `aws sts get-caller-identity` |
| Foundation deployed | `aws ssm get-parameter --name /physical-ai/sagemaker-role-arn --region us-east-2` |
| Container image in ECR (`:n17` tag) | `aws ecr describe-images --repository-name physical-ai/gr00t-training --region us-east-2 --image-ids imageTag=n17` |
| HuggingFace token | Stored in Secrets Manager (`physical-ai/hf-token`) — needed to download `nvidia/GR00T-N1.7-3B` |
| SageMaker quota for `ml.g6e.12xlarge` | **Must request before first use** |

> **Service Quota Request:**
> 1. AWS Console → **Service Quotas** → **Amazon SageMaker**
> 2. Search for `ml.g6e.12xlarge for training job usage`
> 3. Request increase to at least **1 instance**
> 4. Wait for approval (typically 15-30 minutes)
>
> Without this, jobs fail with `ResourceLimitExceeded`. Note this is a **separate quota** from `ml.g6e.4xlarge` used by N1.6.

---

## Step 1: Build the N1.7 Container

The N1.7 container is a separate Dockerfile from N1.6 (native `uv sync` install, Cosmos-Reason2-2B backbone, Python 3.12):

```bash
# Package just the container source (small, fast upload)
cd sample-aws-physical-ai-toolchain
zip -r /tmp/source-n17.zip containers/gr00t-training/ -x "*/__pycache__/*"
aws s3 cp /tmp/source-n17.zip \
  s3://physical-ai-dev-datasets-<ACCOUNT_ID>/codebuild-source/source-n17.zip \
  --region us-east-2

NGC_KEY=$(aws secretsmanager get-secret-value --secret-id physical-ai/ngc-api-key \
  --region us-east-2 --query 'SecretString' --output text)

aws codebuild start-build \
  --project-name "physical-ai-dev-gr00t-training-build" \
  --region us-east-2 \
  --source-type-override S3 \
  --source-location-override "physical-ai-dev-datasets-<ACCOUNT_ID>/codebuild-source/source-n17.zip" \
  --buildspec-override "containers/gr00t-training/buildspec-n17.yml" \
  --compute-type-override BUILD_GENERAL1_2XLARGE \
  --environment-variables-override "[{\"name\":\"NGC_API_KEY\",\"value\":\"$NGC_KEY\",\"type\":\"PLAINTEXT\"}]"
```

**Build time:** ~15-20 min. Pushes to `physical-ai/gr00t-training:n17` (and `:latest`).

> **Compute type note:** Use `BUILD_GENERAL1_2XLARGE` (72 GB RAM). The default project compute (`BUILD_GENERAL1_LARGE`, 8 GB) is not enough for `uv sync` + flash-attn compilation.

---

## Step 2: Upload Training Data to S3

Same LeRobot v2.0 UR3 dataset as N1.6 — see [Step 4 in the Batch guide](batch-training-guide.md#step-4-prepare-and-stage-the-dataset). No format changes for N1.7.

---

## Step 3: Submit the Training Job

### Option A: `launch_training_n17.py` (recommended)

Checkpoints are written to the **checkpoints bucket** (`sagemaker-output/` prefix), matching the same destination convention as the N1.6 Batch path — not the datasets bucket.

```bash
HF_TOKEN=$(aws secretsmanager get-secret-value \
  --secret-id physical-ai/hf-token --region us-east-2 --query 'SecretString' --output text)
export HF_TOKEN

python3 training/gr00t/launch_training_n17.py \
  --datasets-bucket physical-ai-dev-datasets-<ACCOUNT_ID> \
  --dataset-prefix groot-data/ur3 \
  --checkpoints-bucket physical-ai-dev-checkpoints-<ACCOUNT_ID> \
  --role-arn arn:aws:iam::<ACCOUNT_ID>:role/physical-ai-dev-sagemaker-role \
  --ecr-image <ACCOUNT_ID>.dkr.ecr.us-east-2.amazonaws.com/physical-ai/gr00t-training:n17 \
  --max-steps 100 \
  --region us-east-2
```

Preview first with `--dry-run` to see the full job config and cost estimate without launching.

### Option B: Raw `create-training-job` CLI

```bash
aws sagemaker create-training-job \
  --training-job-name "groot-n17-finetune-$(date +%Y%m%d-%H%M%S)" \
  --role-arn "arn:aws:iam::<ACCOUNT_ID>:role/physical-ai-dev-sagemaker-role" \
  --algorithm-specification '{
    "TrainingImage": "<ACCOUNT_ID>.dkr.ecr.us-east-2.amazonaws.com/physical-ai/gr00t-training:n17",
    "TrainingInputMode": "File"
  }' \
  --input-data-config '[{
    "ChannelName": "training",
    "DataSource": {
      "S3DataSource": {
        "S3DataType": "S3Prefix",
        "S3Uri": "s3://physical-ai-dev-datasets-<ACCOUNT_ID>/groot-data/ur3/dataset/",
        "S3DataDistributionType": "FullyReplicated"
      }
    }
  }]' \
  --output-data-config '{
    "S3OutputPath": "s3://physical-ai-dev-checkpoints-<ACCOUNT_ID>/sagemaker-output/"
  }' \
  --resource-config '{
    "InstanceType": "ml.g6e.12xlarge",
    "InstanceCount": 1,
    "VolumeSizeInGB": 200
  }' \
  --hyper-parameters '{
    "base_model": "nvidia/GR00T-N1.7-3B",
    "max_steps": "10000",
    "batch_size": "8",
    "learning_rate": "1e-4",
    "gradient_accumulation_steps": "2"
  }' \
  --environment "{\"HF_TOKEN\": \"$HF_TOKEN\", \"HUGGING_FACE_HUB_TOKEN\": \"$HF_TOKEN\"}" \
  --stopping-condition '{"MaxRuntimeInSeconds": 21600}' \
  --region us-east-2
```

> **Checkpoint destination:** Both options write `model.tar.gz` to `s3://physical-ai-dev-checkpoints-<ACCOUNT_ID>/sagemaker-output/<job-name>/output/model.tar.gz` — the same checkpoints bucket used by N1.6's Batch path (`CHECKPOINT_BUCKET`).

### Smoke test first (recommended)

Set `max_steps=100` for a ~15 minute validation run before committing to a full training job.

---

## Step 4: Monitor Training

```bash
aws sagemaker describe-training-job \
  --training-job-name <JOB_NAME> \
  --region us-east-2 \
  --query '{Status:TrainingJobStatus,Secondary:SecondaryStatus}'
```

Tail logs (once in `Training` secondary status):

```bash
LOG_STREAM=$(aws logs describe-log-streams \
  --log-group-name /aws/sagemaker/TrainingJobs \
  --log-stream-name-prefix <JOB_NAME> \
  --region us-east-2 --query 'logStreams[0].logStreamName' --output text)

aws logs get-log-events \
  --log-group-name /aws/sagemaker/TrainingJobs \
  --log-stream-name "$LOG_STREAM" \
  --region us-east-2 --query 'events[*].message' --output text | tail -50
```

Look for `{'loss': ..., 'grad_norm': ..., 'learning_rate': ...}` lines — loss should trend downward.

---

## Step 5: Retrieve Checkpoints

```bash
aws s3 cp \
  s3://physical-ai-dev-checkpoints-<ACCOUNT_ID>/sagemaker-output/<JOB_NAME>/output/model.tar.gz .
tar -xzf model.tar.gz
```

---

## How It Works

1. `train_entrypoint.py` writes the UR3 embodiment config (`register_modality_config`, `new_embodiment` tag — same pattern as N1.6)
2. Deletes stale `stats.json` files (format can differ between SDK versions)
3. Launches `gr00t/experiment/launch_finetune.py` via `torch.distributed.run --nproc_per_node <num_gpus>`
4. With `num_gpus > 1`, the training code (`experiment.py`) auto-builds a DeepSpeed config and passes it to the HF `Trainer` — this is what shards optimizer memory across GPUs
5. Fine-tunes `--tune-visual --tune-projector --tune-diffusion-model` (LLM frozen, matching NVIDIA's reference recipe)
6. Saves checkpoints + `training_metadata.json` to `/opt/ml/model/` → SageMaker uploads `model.tar.gz` to S3

---

## Cost

| Scenario | Instance | Time | Cost (on-demand) |
|----------|----------|------|-------------------|
| Smoke test (100 steps) | ml.g6e.12xlarge | ~15 min | ~$2 |
| Full training (10,000 steps) | ml.g6e.12xlarge | ~8-10 hrs (est.) | ~$65-80 |

SageMaker auto-terminates the instance on job completion — no idle cost.

---

## Troubleshooting

| Problem | Cause | Solution |
|---------|-------|----------|
| `torch.OutOfMemoryError` at optimizer step, any batch size | Single-GPU instance (`ml.g6e.4xlarge`) — optimizer states alone exceed 44 GB | Use `ml.g6e.12xlarge` (4 GPUs) so DeepSpeed shards optimizer memory |
| `Unrecognized options: --gradient-checkpointing` | This CLI version has no such flag | Don't pass it — use a multi-GPU instance instead |
| `ResourceLimitExceeded` | No quota for `ml.g6e.12xlarge` | Request quota increase (see Prerequisites) — separate from the `ml.g6e.4xlarge` quota |
| Training job stuck in `Pending` / "waiting for capacity" | `ml.g6e.12xlarge` is less commonly available than `.4xlarge` | Wait — SageMaker retries automatically; typically resolves within 10-30 min |
| `CUDA out of memory` even on 4 GPUs | Batch size too aggressive | Lower `batch_size` or raise `gradient_accumulation_steps` |
| Build fails with OOM/disk errors in CodeBuild | Default compute type too small for `uv sync` | Use `--compute-type-override BUILD_GENERAL1_2XLARGE` |
| `COPY ... not found` in Docker build | Dockerfile COPY paths assume `containers/gr00t-training/` as build context, but zip root is the actual context | Prefix COPY paths with `containers/gr00t-training/` |

---

## Validation Status

| Step | Status | Notes |
|------|--------|-------|
| Container build (`Dockerfile.n17`) | ✅ Validated | NGC PyTorch base + `uv sync` + Isaac-GR00T `1a1837f` (N1.7 General Release) |
| GR00T N1.7-3B model loading | ✅ Validated | 3.14B params, Cosmos-Reason2-2B backbone, 2.03B trainable (64%) |
| Single-GPU training (`ml.g6e.4xlarge`) | ❌ Confirmed OOM | Optimizer states exceed 44 GB regardless of batch size |
| Multi-GPU training (`ml.g6e.12xlarge`, DeepSpeed) | ✅ Validated | 100 steps completed in 917s, loss 1.35→1.25 |
| Checkpoint upload to S3 | ✅ Validated | `model.tar.gz` via SageMaker output path |

---

## N1.6 vs N1.7 — Practical Differences

| | N1.6 | N1.7 |
|---|---|---|
| Backbone | Eagle-Block2A-2B | Cosmos-Reason2-2B (Qwen3-VL) |
| Base model | `nvidia/GR00T-N1.6-3B` | `nvidia/GR00T-N1.7-3B` |
| Minimum instance | `ml.g6e.4xlarge` (1× L40S, 48GB) | `ml.g6e.12xlarge` (4× L40S, 192GB) |
| Why | Fits with gradient checkpointing on 1 GPU | No CLI gradient-checkpointing flag; needs DeepSpeed ZeRO across GPUs |
| Training API | `gr00t.experiment.experiment.run(config)` | `gr00t/experiment/launch_finetune.py` CLI |
| Install | pip + manual venv | Native `uv sync` (pyproject.toml) |
| Python | 3.11 | 3.12 |
| Action horizon | Up to 16 | Up to 40 |
| Reasoning | Basic | Structured (task + subtask level) |
