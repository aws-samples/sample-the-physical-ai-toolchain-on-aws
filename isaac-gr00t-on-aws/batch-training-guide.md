# AWS Batch GR00T Training Guide (N1.7 + Unitree G1 Edu)

Fine-tune **GR00T N1.7-3B** on AWS Batch against the 201-episode **Unitree G1 Edu
(Dex3 hands, 28 DOF)** pick-and-place dataset. This guide covers the Terraform
infrastructure, the container rebuild that makes the image embodiment-selectable,
the dataset preparation, and the job submission.

Every number quoted below was measured in this account. Anything that has not
been observed is labelled **untested** rather than estimated — see
[Validation Status](#validation-status). The one thing this procedure has **not**
yet produced is a completed training run: `g6e.12xlarge` capacity could not be
obtained in us-east-2 (details in the Validation Status and Troubleshooting
sections).

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
│ 2. GR00T Infra (Terraform, isaac-gr00t-on-aws/infra)                     │
│    CodeBuild project · Batch compute env · Job queue · Job definition    │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 3. Container Build (CodeBuild, buildspec-n17.yml)                        │
│    NGC PyTorch 25.04 + Isaac-GR00T N1.7 + flash-attn → ECR :n17          │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 4. Dataset prep (LeRobot v2.1 as-is + meta/modality.json)                │
│    prepare_g1_dex3_dataset.py → validate → stage to datasets bucket      │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 5. Training Job (AWS Batch)                                              │
│    g6e.12xlarge (4× L40S, 48 GB each) · DeepSpeed ZeRO · ckpts → S3      │
└─────────────────────────────────────────────────────────────────────────┘
```

**Hardware:** compute environment `physical-ai-dev-gr00t-compute` launches a
single `g6e.12xlarge` — 48 vCPU, 384 GiB RAM, 4× NVIDIA L40S (48 GB each), with a
512 GiB gp3 root volume from the launch template (the container image alone is
22.6 GB).

**Why 4 GPUs:** at `num_gpus > 1` the trainer enables DeepSpeed ZeRO-2, which
shards the AdamW moments and gradients across ranks. At one GPU DeepSpeed is
switched off entirely and the ~2.03B trainable parameters plus their fp32
optimizer state (~36.9 GB static) do not leave enough headroom on a single 48 GB
card. A single-GPU `g6e.4xlarge` run is **not** a supported configuration for
this image.

---

## Prerequisites

| Requirement | How to verify |
|-------------|---------------|
| AWS CLI configured | `aws sts get-caller-identity` |
| Terraform >= 1.5 (`~/bin/terraform` here) | `terraform version` |
| Foundation deployed | `aws ssm get-parameter --name /physical-ai/vpc-id --region us-east-2` |
| EC2 quota: `Running On-Demand G and VT instances` >= 48 vCPU | Service Quotas, code `L-DB2E81BA` |
| `g6e.12xlarge` **capacity** in at least one private-subnet AZ | see Troubleshooting — quota is not the same as capacity |
| NGC API key in Secrets Manager (`physical-ai/ngc-api-key`) | `aws secretsmanager describe-secret --secret-id physical-ai/ngc-api-key --region us-east-2` |
| HuggingFace token in Secrets Manager (`physical-ai/hf-token`) | `aws secretsmanager describe-secret --secret-id physical-ai/hf-token --region us-east-2` |
| Python 3.10+ with `pyarrow` (dataset prep only) | `python3 -c "import pyarrow"` |

> `nvidia/GR00T-N1.7-3B` is **not** a gated repo, so the HuggingFace token is not
> functionally required to download the base weights. It is still configured, and
> an unresolvable lookup is a warning rather than a failure.

---

## Step 1: Deploy Foundation (if not already done)

Shared with Isaac Lab — skip if already deployed.

```bash
cd foundation/infra
terraform init
terraform plan -var="aws_region=us-east-2"
terraform apply -var="aws_region=us-east-2"
```

---

## Step 2: Deploy GR00T Infrastructure

```bash
cd isaac-gr00t-on-aws/infra
terraform init
terraform plan -var="aws_region=us-east-2" -var="enable_batch=true" -var="batch_instance_type=g6e.12xlarge"
terraform apply -var="aws_region=us-east-2" -var="enable_batch=true" -var="batch_instance_type=g6e.12xlarge"
```

**What this creates:**

| Resource | Purpose |
|----------|---------|
| CodeBuild: `physical-ai-dev-gr00t-training-build` | Builds the GR00T training container |
| CodeBuild: `physical-ai-dev-gr00t-inference-build` | Builds the GR00T inference container |
| Batch compute env: `physical-ai-dev-gr00t-compute` | `g6e.12xlarge` (4× L40S, 512 GiB gp3 root), min/desired 0 vCPU, max 96 |
| Batch job queue: `physical-ai-dev-gr00t-queue` | Job scheduling |
| Batch job definition: `physical-ai-dev-gr00t-n17-training` | Image `:n17`, 48 VCPU / 180000 MEMORY / 4 GPU |
| Security group | Outbound access for ECR/S3/HuggingFace |
| IAM roles | Batch service role, Batch instance role (S3 + ECR pull + one secret) |

**Verify:**
```bash
terraform output
# batch_compute_environment = "physical-ai-dev-gr00t-compute"
# batch_job_queue           = "physical-ai-dev-gr00t-queue"
# batch_job_definition      = "physical-ai-dev-gr00t-n17-training"
```

> ### ⚠️ Do not use the job definition `physical-ai-dev-gr00t-training`
>
> **Every revision** of `physical-ai-dev-gr00t-training` (revs 1-10 in this
> account) points at the ECR repository `physical-ai/groot-training` — note the
> misspelling. That repository **does not exist**; the real one is
> `physical-ai/gr00t-training`. Any job submitted against those revisions dies at
> image pull with a `CannotPullContainerError`, after the full capacity wait and
> instance launch. Always submit against **`physical-ai-dev-gr00t-n17-training`**.
> `training/gr00t/submit_batch_training_n17.py` refuses any `--job-definition`
> whose name does not contain `n17` for exactly this reason.

### Pending apply: the HF_TOKEN → Secrets Manager migration

The Terraform in `isaac-gr00t-on-aws/infra/batch.tf` now codifies the job
definition **without** a plaintext `HF_TOKEN` entry. The live revision 1 in this
account still carries one, because the Terraform has been written and validated
but **not applied** — applying registers a new job-definition revision, which is
a separately approved action.

What the config does:

- `data "aws_secretsmanager_secret" "hf_token"` looks up `physical-ai/hf-token`
  and references only its `.arn`. The value never enters Terraform state.
- The instance role's inline policy gains one statement
  (`Sid = "ReadHuggingFaceTokenSecretOnly"`): `secretsmanager:GetSecretValue`
  scoped to that single secret ARN. The existing S3 scope
  (`arn:aws:s3:::physical-ai-dev-*`) is unchanged.
- The job definition sets `HF_TOKEN_SECRET_ID=physical-ai/hf-token`; the
  container's `batch-train` script resolves it at runtime under the instance
  role, and only when `HF_TOKEN` is unset.

The alternative — Batch's native `containerProperties.secrets` injection —
additionally needs an `executionRoleArn` on the job definition plus the same
scoped `GetSecretValue` on *that* role, because the ECS agent resolves secrets
with the execution role rather than the container-instance role
([AWS Batch: specifying sensitive data](https://docs.aws.amazon.com/batch/latest/userguide/specifying-sensitive-data-secrets.html)).
The runtime-fetch shape was chosen because it needs no new role and the container
was being rebuilt anyway.

Validate the HCL without touching state:

```bash
cd isaac-gr00t-on-aws/infra
~/bin/terraform fmt -check
~/bin/terraform init -backend=false
~/bin/terraform validate
```

---

## Step 3: Rebuild the Training Container

### 3a. Why a rebuild is mandatory

The image previously tagged `:n17` **cannot train this dataset**, and no
environment variable or `--container-overrides` setting can change that:

- `containers/gr00t-training/train_entrypoint_n17.py` used to inline a **UR3
  end-effector** modality config as a string literal, unconditionally: video key
  `wrist`, state/action groups `arm` + `gripper`, both action configs
  `ActionType.EEF` with `ActionFormat.XYZ_ROTVEC`. Against a 28-dim joint-space
  dataset, `EEF` routes through `Rotation.from_rotvec(data[3:])`, which requires
  exactly 3 rotation components and raises inside SciPy on a 7-element joint
  group.
- `Dockerfile.n17` declares `ENTRYPOINT ["/opt/ml/code/entrypoint.sh"]`. A Batch
  `containerOverrides.command` replaces only Docker `CMD`, which arrives as
  `"$@"` to that entrypoint, and Batch job definitions have **no entrypoint
  override**. So the hardcoded path ran no matter what was passed in.

The rebuilt entrypoint is embodiment-selectable:

| variable | meaning |
|---|---|
| `EMBODIMENT_CONFIG` | `ur3` (default) or `g1_dex3`. `g1_dex3` selects `/opt/ml/code/g1_dex3_modality_config.py`. An unknown value fails immediately. |
| `MODALITY_CONFIG_PATH` | Optional absolute path inside the container; wins over `EMBODIMENT_CONFIG`. Set-but-missing is a hard failure, not a silent fallback. |
| `HF_TOKEN_SECRET_ID` | Secrets Manager id, read only when `HF_TOKEN` is unset. |

Both configs register under `EmbodimentTag.NEW_EMBODIMENT`, so
`--embodiment-tag new_embodiment` is passed either way — the G1's 28-dim action
space gets a freshly initialised projector head while the Cosmos-Reason2-2B
backbone, vision tower and diffusion head start from pretrained N1.7 weights.

### 3b. Package and upload the source

```bash
# From repo root
zip -rq /tmp/source-n17.zip . \
  -x ".git/*" -x "*/.git/*" \
  -x ".worktrees/*" \
  -x "*/.terraform/*" -x "*/terraform.tfstate*" \
  -x "*/__pycache__/*"

aws s3 cp /tmp/source-n17.zip \
  s3://physical-ai-dev-datasets-<ACCOUNT_ID>/codebuild-source/source-n17.zip \
  --region us-east-2
```

**Measured:** `zip` 1 s, upload 4 s, archive 2.5 MB.

### 3c. Trigger the build

The CodeBuild project is `NO_SOURCE` with an **inline N1.6 buildspec**, so the
N1.7 buildspec has to be supplied as an override. The default `LARGE` compute has
too little memory for `uv sync` + flash-attn, hence the compute override.

```bash
aws codebuild start-build \
  --project-name "physical-ai-dev-gr00t-training-build" \
  --region us-east-2 \
  --source-type-override S3 \
  --source-location-override \
    "physical-ai-dev-datasets-<ACCOUNT_ID>/codebuild-source/source-n17.zip" \
  --buildspec-override containers/gr00t-training/buildspec-n17.yml \
  --compute-type-override BUILD_GENERAL1_2XLARGE
```

Do **not** pass the NGC key with `--environment-variables-override`: a
`PLAINTEXT` override is stored in the build record. `buildspec-n17.yml` reads it
from Secrets Manager (`physical-ai/ngc-api-key`) instead, and the CodeBuild role
is already scoped for that one secret. A grep for `nvapi-` across the full build
log of the validated build returns 0 matches.

The buildspec pushes `:n17` (the tag the job definition consumes) and an
immutable dated `:n17-YYYYMMDD`. It deliberately does **not** touch `:latest`,
which belongs to the N1.6 inline buildspec.

### 3d. Monitor progress

```bash
aws codebuild batch-get-builds \
  --ids "<BUILD_ID>" \
  --region us-east-2 \
  --query 'builds[0].{status:buildStatus,phase:currentPhase}'
```

**Measured build phases** (`BUILD_GENERAL1_2XLARGE`, from-scratch, no layer
cache):

| phase | duration |
|---|---|
| QUEUED | 0 s |
| PROVISIONING | 173 s |
| DOWNLOAD_SOURCE / INSTALL | 0 s |
| PRE_BUILD (ECR + NGC login) | 2 s |
| BUILD (NGC base pull, `uv sync` incl. flash-attn, layer export) | 436 s (7 m 16 s) |
| POST_BUILD (push 22.6 GB, two tags) | 583 s (9 m 43 s) |
| **total wall clock** | **19 m 57 s** |

Resulting image: 22,597,572,749 B (22.6 GB).

> **Budget for the queue.** A first attempt on `BUILD_GENERAL1_2XLARGE` sat in
> `QUEUED` for **1502 s (25 m 2 s)** before provisioning — a capacity wait, not a
> misconfiguration (the account's Linux/2XLarge concurrency quota is 100 and only
> that one build was in flight). The successful second attempt queued for 0 s.

### 3e. Confirm the image actually changed

```bash
aws ecr describe-images --repository-name physical-ai/gr00t-training \
  --image-ids imageTag=n17 --region us-east-2 \
  --query 'imageDetails[0].{digest:imageDigest,pushed:imagePushedAt,tags:imageTags}'
```

The digest **must** differ from before the build. The ECR repository is
`MUTABLE`, so re-pushing `:n17` lets the existing job definition pick up the new
image with no new revision. The superseded image is not deleted and stays
reachable by digest, so the tag move is reversible.

---

## Step 4: Prepare and Stage the Dataset

### 4a. What the dataset is

Canonical read-only copy:
`s3://physical-ai-workshop-assets-<ACCOUNT_ID>/datasets/pick_place_data/` — 407
objects, 128 MiB. **Never write to that prefix.**

| field | value |
|---|---|
| `codebase_version` | `v2.1` |
| `robot_type` | `Unitree_G1_Dex3` |
| episodes / frames / tasks | 201 / 29,105 / 1 |
| `fps` | 30 |
| `observation.state`, `action` | float32 `[28]` |
| `observation.images.cam_high` | video `[3, 480, 640]`, codec **av1** |

The 28 joints decompose into four contiguous 7-element groups: `left_arm`,
`right_arm`, `left_hand`, `right_hand`.

### 4b. No LeRobot conversion is needed

**Ship the dataset as-is.** The N1.7 episode loader
(`gr00t/data/dataset/lerobot_episode_loader.py` at the commit baked into the
image) **never reads `codebase_version`** — a full grep for it returns nothing.
It cares only about structure: `meta/info.json`, `meta/episodes.jsonl`,
`meta/tasks.jsonl`, `meta/modality.json`, and the `data_path` / `video_path` /
`chunks_size` / `features` entries of `info.json`. The only conversion script
upstream ships is **v3 → v2**, because LeRobot v3 dropped `meta/episodes.jsonl`.
This dataset has `meta/episodes.jsonl` with the required `length` field, so the
failure mode the old "requires v2.0" note described cannot occur here.

The only genuine gap is the GR00T-specific `meta/modality.json`, which
`prepare_g1_dex3_dataset.py` authors.

### 4c. `meta/stats.json` is generated in-container — do not ship one

`gr00t/data/dataset/factory.py` calls `generate_stats(dataset_path)` and
`generate_rel_stats(...)` inside a rank-0 barrier **before** the loader asserts
that `meta/stats.json` exists. `generate_stats` globs `data/*/*.parquet` and
computes `mean/std/min/max/q01/q99` for every float feature;
`generate_rel_stats` writes `meta/relative_stats.json` for each `RELATIVE` action
group (here `left_arm` and `right_arm`). Reading 201 parquet files totalling
~8 MB takes seconds.

`train_entrypoint_n17.py` **deletes** any shipped `meta/stats.json`,
`meta/relative_stats.json` and `meta/*.stats.json` before launching, so a
hand-written copy is thrown away. `meta/episodes_stats.jsonl` is ignored entirely
by GR00T (and lacks `q01`/`q99` anyway) — harmless, but not a stats source.

### 4d. The `meta/modality.json` this container consumes

Written verbatim by `prepare_g1_dex3_dataset.py`:

```json
{
    "state": {
        "left_arm": { "start": 0, "end": 7 },
        "right_arm": { "start": 7, "end": 14 },
        "left_hand": { "start": 14, "end": 21 },
        "right_hand": { "start": 21, "end": 28 }
    },
    "action": {
        "left_arm": { "start": 0, "end": 7 },
        "right_arm": { "start": 7, "end": 14 },
        "left_hand": { "start": 14, "end": 21 },
        "right_hand": { "start": 21, "end": 28 }
    },
    "video": {
        "cam_high": { "original_key": "observation.images.cam_high" }
    },
    "annotation": {
        "human.task_description": { "original_key": "task_index" }
    }
}
```

Three constraints worth knowing before editing it:

- **No `original_key` inside the state/action groups.** The loader treats a
  present `original_key` as the key to look up in `stats.json`, and `left_arm` is
  not a key there. Omitting it resolves to `observation.state` / `action`, which
  are the keys `generate_stats` writes.
- **`annotation.human.task_description` must map to `task_index`.** The parquet
  has no `annotation.*` column; values are resolved through `tasks.jsonl`.
- **Action group names must match state group names**, because a `RELATIVE`
  action looks up its reference state by the same key.

### 4e. Run the prep and the validator

```bash
# 1. Pull a local working copy (never modify the workshop-assets original)
mkdir -p /tmp/g1work
aws s3 sync s3://physical-ai-workshop-assets-<ACCOUNT_ID>/datasets/pick_place_data/ \
  /tmp/g1work/pick_place_data/ --region us-east-2

# 2. Preview the modality.json, then write it
python3 training/gr00t/prepare_g1_dex3_dataset.py \
  --dataset-dir /tmp/g1work/pick_place_data --dry-run

python3 training/gr00t/prepare_g1_dex3_dataset.py \
  --dataset-dir /tmp/g1work/pick_place_data

# 3. Validate against every assertion the N1.7 loader makes
python3 training/gr00t/validate_lerobot_gr00t_dataset.py \
  --dataset-dir /tmp/g1work/pick_place_data \
  --sample-episodes 201

# 4. Stage to the dev datasets bucket
python3 training/gr00t/prepare_g1_dex3_dataset.py \
  --dataset-dir /tmp/g1work/pick_place_data \
  --upload \
  --s3-uri s3://physical-ai-dev-datasets-<ACCOUNT_ID>/groot-data/pick_place_data/
```

**Measured:** the validator passes with **0 problems, 2 warnings** — 201/201
episodes resolved (parquet + one mp4 each), 29,105 rows read, ~2 s. The two
warnings are the expected zero-range action dims 14 (`kLeftHandThumb0`) and 21
(`kRightHandThumb0`), both constant at 0.0; the normalization denominator is
clamped at 1e-8, so they cannot produce NaN.

Staged result: **409 objects, 134,298,177 bytes** under
`s3://physical-ai-dev-datasets-<ACCOUNT_ID>/groot-data/pick_place_data/` — the 407
source objects plus `meta/modality.json` plus a provenance copy of
`meta/g1_dex3_modality_config.py`. The source bucket re-verified at 407 objects,
untouched.

> **The dataset must be staged to a `physical-ai-dev-*` bucket.** The Batch
> instance role grants S3 only on `arn:aws:s3:::physical-ai-dev-*`, so pointing
> `DATASET_S3_URI` at `physical-ai-workshop-assets-...` fails with
> `AccessDenied`. The fix is staging, not widening the policy;
> `submit_batch_training_n17.py` rejects any `--dataset-s3-uri` containing
> `workshop-assets`.

### Required dataset structure (LeRobot v2.x + GR00T)

```
pick_place_data/
├── data/chunk-000/episode_000000.parquet … episode_000200.parquet
├── meta/
│   ├── info.json                 # features, shapes, fps, data_path, video_path
│   ├── episodes.jsonl            # REQUIRED: episode_index, tasks, length
│   ├── tasks.jsonl               # REQUIRED: task_index, task
│   ├── modality.json             # REQUIRED: authored by prepare_g1_dex3_dataset.py
│   ├── episodes_stats.jsonl      # ignored by GR00T
│   └── (stats.json, relative_stats.json generated in-container)
└── videos/chunk-000/observation.images.cam_high/episode_000000.mp4 … 000200.mp4
```

---

## Step 5: Store the HuggingFace Token

The secret already exists in this account. To create it elsewhere:

```bash
aws secretsmanager create-secret \
  --name "physical-ai/hf-token" \
  --secret-string "<YOUR_HF_TOKEN>" \
  --region us-east-2 \
  --description "HuggingFace token, read at runtime by the GR00T Batch container"
```

The container reads it via `HF_TOKEN_SECRET_ID` using the Batch instance role, so
the token never appears in the job definition, in CloudWatch, or in this repo.
Because `nvidia/GR00T-N1.7-3B` is not gated, a failed lookup only prints
`HuggingFace credential: NOT resolved (continuing; base model is not gated)`.

---

## Step 6: Submit a Training Job

Use `--container-overrides` with **environment entries only**. Do not register a
new job-definition revision for a configuration change.

### Recommended: the submitter script

```bash
# Preview the exact payload, call no AWS API
python3 training/gr00t/submit_batch_training_n17.py \
  --dataset-s3-uri s3://physical-ai-dev-datasets-<ACCOUNT_ID>/groot-data/pick_place_data/ \
  --max-steps 15 \
  --embodiment-config g1_dex3 \
  --dry-run

# Submit and watch (streams the /aws/batch/job log, prints phase dwell times)
python3 -u training/gr00t/submit_batch_training_n17.py \
  --dataset-s3-uri s3://physical-ai-dev-datasets-<ACCOUNT_ID>/groot-data/pick_place_data/ \
  --max-steps 15 \
  --embodiment-config g1_dex3 \
  --job-name g1-dex3-smoke

# Re-attach to an in-flight job (the watcher is just a poller; killing it is harmless)
python3 -u training/gr00t/submit_batch_training_n17.py --watch-only <JOB_ID>
```

The payload it sends (verified by `--dry-run`):

```json
{
  "jobQueue": "physical-ai-dev-gr00t-queue",
  "jobDefinition": "physical-ai-dev-gr00t-n17-training",
  "containerOverrides": {
    "environment": [
      { "name": "DATASET_S3_URI", "value": "s3://physical-ai-dev-datasets-<ACCOUNT_ID>/groot-data/pick_place_data/" },
      { "name": "MAX_STEPS", "value": "15" },
      { "name": "EMBODIMENT_CONFIG", "value": "g1_dex3" },
      { "name": "BATCH_SIZE", "value": "8" },
      { "name": "GRAD_ACCUM_STEPS", "value": "2" }
    ]
  }
}
```

No `command`, no `resourceRequirements` — environment only.

### Equivalent raw CLI

```bash
aws batch submit-job \
  --job-name "g1-dex3-smoke" \
  --job-queue "physical-ai-dev-gr00t-queue" \
  --job-definition "physical-ai-dev-gr00t-n17-training" \
  --region us-east-2 \
  --container-overrides '{
    "environment": [
      {"name": "DATASET_S3_URI", "value": "s3://physical-ai-dev-datasets-<ACCOUNT_ID>/groot-data/pick_place_data/"},
      {"name": "MAX_STEPS", "value": "15"},
      {"name": "EMBODIMENT_CONFIG", "value": "g1_dex3"},
      {"name": "BATCH_SIZE", "value": "8"},
      {"name": "GRAD_ACCUM_STEPS", "value": "2"}
    ]
  }'
```

Keep `BATCH_SIZE=8` / `GRAD_ACCUM_STEPS=2` on 4× L40S: those are the values
recorded as measured-working on the same hardware in
[n17-sagemaker-training-guide.md](n17-sagemaker-training-guide.md). `BATCH_SIZE`
is the global, pre-accumulation batch across all GPUs, so 8 means per-device 2,
and the effective optimizer batch is 16. If OOM appears, drop to `BATCH_SIZE=4`
with `GRAD_ACCUM_STEPS=4` to hold the effective batch. There is no
`--gradient-checkpointing` lever in this SDK revision.

---

## Step 7: Watch the Job

```bash
# Status and phase
aws batch describe-jobs --jobs "<JOB_ID>" --region us-east-2 \
  --query 'jobs[0].{status:status,reason:statusReason,created:createdAt,started:startedAt,stopped:stoppedAt}'

# Find the log stream, then tail it
STREAM=$(aws batch describe-jobs --jobs "<JOB_ID>" --region us-east-2 \
  --query 'jobs[0].container.logStreamName' --output text)
aws logs get-log-events --log-group-name /aws/batch/job \
  --log-stream-name "$STREAM" --start-from-head --region us-east-2 \
  --query 'events[].message' --output text | tail -50
```

`container.logStreamName` is `null` until the container starts, so a job that
never leaves `RUNNABLE` has no logs at all — that is a capacity signal, not a
missing-log bug.

Expected log sequence once a container runs (**untested — see Validation
Status**): the `batch-train` banner echoing `Embodiment:` and `Modality config:`;
`Video pre-flight: OK - episode_000000.mp4 codec=… frame=…`; `Generating stats
for /opt/ml/input/data/training`; `Generating relative stats` for `left_arm` /
`right_arm`; the `launch_finetune.py` command line with `--embodiment-tag
new_embodiment` and the `g1_dex3` modality path; the 3.14B-parameter model load
(~2.03B trainable); DeepSpeed ZeRO at `num_gpus=4`; then
`{'loss': …, 'grad_norm': …, 'learning_rate': …}` lines.

### The three success checks

1. `aws batch describe-jobs --jobs <JOB_ID> --query 'jobs[0].status'` →
   `SUCCEEDED`.
2. At least one `'loss':` line in the `/aws/batch/job` stream, finite and not
   NaN. At 15 steps with a freshly initialised `new_embodiment` projector, only
   "finite and decreasing-ish" is meaningful — the starting value will not match
   the UR3 run's 1.35.
3. Objects under `s3://physical-ai-dev-checkpoints-<ACCOUNT_ID>/`, at the
   `batch-train` default prefix `checkpoints/groot-n17/<AWS_BATCH_JOB_ID>/`:

```bash
aws s3 ls s3://physical-ai-dev-checkpoints-<ACCOUNT_ID>/ --recursive --region us-east-2
```

---

## Step 8: Retrieve Checkpoints

```bash
aws s3 cp \
  s3://physical-ai-dev-checkpoints-<ACCOUNT_ID>/checkpoints/groot-n17/<JOB_ID>/ \
  ./groot-n17-checkpoint/ --recursive --region us-east-2
```

The prefix also contains `training_metadata.json`, which records the resolved
`embodiment_config`, `modality_config_path` and `embodiment_tag` — the quickest
way to confirm which modality config a checkpoint was trained with.

---

## Managing Compute Costs

The compute environment sits at `minvCpus=0` / `desiredvCpus=0` and scales to
zero on its own. A job in `RUNNABLE` costs nothing — no instance exists yet.
`desiredvCpus` rises to 48 while a job is queued and returns to 0 once the job
reaches a terminal state.

```bash
# Keep an instance warm during an active iteration session
aws batch update-compute-environment \
  --compute-environment "physical-ai-dev-gr00t-compute" \
  --compute-resources 'minvCpus=48,desiredvCpus=48' \
  --region us-east-2

# Scale back to zero when done
aws batch update-compute-environment \
  --compute-environment "physical-ai-dev-gr00t-compute" \
  --compute-resources 'minvCpus=0,desiredvCpus=0' \
  --region us-east-2
```

> **Warning:** a warm `g6e.12xlarge` bills continuously. Check current on-demand
> pricing for your region before leaving one up, and scale back to zero as soon
> as the iteration session ends.

---

## Configuration Variables (Terraform)

| Variable | Default | Description |
|----------|---------|-------------|
| `enable_batch` | `false` | Set to `true` to deploy the Batch resources |
| `batch_instance_type` | `g6.12xlarge` | Pass `g6e.12xlarge` — the validated 4× L40S (48 GB) shape. The deployed CE is `g6e.12xlarge`. |
| `batch_max_vcpus` | `96` | Maximum vCPUs for auto-scaling (two 48-vCPU instances) |

---

## Validation Status

Measured in account `<ACCOUNT_ID>`, region `us-east-2`.

| Step | Status | Notes |
|------|--------|-------|
| Terraform deploy (Foundation + GR00T infra) | ✅ Validated | VPC, Batch CE, queue, CodeBuild, instance role all live and `VALID` |
| Secrets Manager HCL (`batch.tf`, `outputs.tf`) | ⚠️ Written, **not applied** | `fmt -check` + `init -backend=false` + `validate` all exit 0. No `plan`/`apply` against real state; live job-definition rev 1 still carries the plaintext `HF_TOKEN`. |
| Container rebuild (CodeBuild, `buildspec-n17.yml`) | ✅ Validated | 19 m 57 s, ECR `:n17` digest changed, both new modality configs present in the image |
| Dataset prep + `meta/modality.json` | ✅ Validated | Validator: 0 problems, 2 expected warnings, 201/201 episodes, 29,105 rows |
| Dataset staged to the dev datasets bucket | ✅ Validated | 409 objects / 134,298,177 B; workshop-assets source untouched at 407 |
| Batch job submission + queue admission | ✅ Validated | `SUBMITTED` → `RUNNABLE` in **30 s** |
| `g6e.12xlarge` instance launch | ❌ **Blocked — no EC2 capacity** | Job held `RUNNABLE` for **2 h 38 m 29 s**; 30/30 ASG launch attempts failed `InsufficientInstanceCapacity` across all three AZs (2a ×15, 2c ×9, 2b ×6) |
| ECR image pull onto a Batch instance | ⬜ Untested | `STARTING` never reached; no measured pull time exists |
| In-container dataset download | ⬜ Untested | `RUNNING` never reached |
| AV1 video decode (torchcodec pre-flight) | ⬜ Untested | The largest remaining risk. The pre-flight turns a silent late failure into an immediate, legible one. |
| GR00T N1.7-3B model load + DeepSpeed ZeRO at 4 GPUs | ⬜ Untested | No container ever started |
| Training steps / loss / per-step timing | ⬜ Untested | **No training timing is quoted anywhere in this guide** because none has been measured |
| Checkpoint upload to S3 | ⬜ Untested | `s3://physical-ai-dev-checkpoints-<ACCOUNT_ID>/` is still empty (0 objects) |

Nothing was billed for the blocked run: no `g6e.12xlarge` was ever launched. AWS
Batch keeps retrying on its own, so a job left in `RUNNABLE` starts by itself
when capacity frees up — it does not need resubmitting.

### Capacity, for workshop planning

Per-account quota was not the constraint: `L-DB2E81BA` (G and VT on-demand) is
144 vCPU in this region with 64 vCPU in use by unrelated instances, so the 48
needed were well within it. Regional **pool depth** was the constraint — one
`g6e.12xlarge` could not be obtained across all three us-east-2 AZs in 2 h 38 m
of retries roughly every 4-8 minutes. An event needing many of them concurrently
in one region cannot rely on on-demand `g6e.12xlarge`; plan for capacity
reservations, another region, or another instance family.

---

## Troubleshooting

| Problem | Cause | Solution |
|---------|-------|----------|
| Job stuck in `RUNNABLE`, `statusReason` null, no log stream | No `g6e.12xlarge` **capacity** (not quota, not config) — the CE is `VALID` and `desiredvCpus` did rise | Inspect the backing ASG: `aws autoscaling describe-scaling-activities --auto-scaling-group-name <ce-asg> --region us-east-2 --query 'Activities[].{code:StatusCode,cause:StatusMessage}'`. `InsufficientInstanceCapacity` means wait — Batch retries by itself. Check the quota separately with `aws service-quotas get-service-quota --service-code ec2 --quota-code L-DB2E81BA`. |
| Tempted to pin the CE to the AZ the error message names | The "you can currently get capacity in us-east-2b/c" hint is a point-in-time pool estimate | **Do not.** us-east-2b was recommended by both the 2a and 2c failure messages and itself failed 6/6. Pinning reduces the number of pools the ASG can try and fixes nothing. |
| `CannotPullContainerError` / repository does not exist | Job submitted against `physical-ai-dev-gr00t-training` (any revision), which points at the misspelled repo `physical-ai/groot-training` | Submit against `physical-ai-dev-gr00t-n17-training` |
| Container trains a UR3 end-effector config on G1 data, or SciPy raises in `Rotation.from_rotvec` | `EMBODIMENT_CONFIG` not set, so the default `ur3` config ran | Pass `EMBODIMENT_CONFIG=g1_dex3` in `--container-overrides`. A Batch `command` override cannot fix this: `Dockerfile.n17` sets an `ENTRYPOINT` and Batch has no entrypoint override. |
| Video pre-flight fails with a torchcodec/ffmpeg error on `episode_000000.mp4` | AV1 decode (`video.codec: "av1"`) unavailable in the image's ffmpeg/torchcodec combination | Transcode all 201 videos to H.264 (`ffmpeg -i in.mp4 -c:v libx264 -pix_fmt yuv420p out.mp4`), re-stage the dataset to S3, resubmit. This is a data fix, not a container change. |
| `AccessDenied` reading `DATASET_S3_URI` | The instance role grants S3 only on `arn:aws:s3:::physical-ai-dev-*`, and the canonical dataset lives in `physical-ai-workshop-assets-<ACCOUNT_ID>` | Stage a prepared copy to `s3://physical-ai-dev-datasets-<ACCOUNT_ID>/groot-data/pick_place_data/`. Do **not** widen the policy. |
| `HF_TOKEN` still appears in plaintext in the job definition | The Secrets Manager migration is written in `batch.tf` but **not applied** — applying registers a new revision | Apply `isaac-gr00t-on-aws/infra` deliberately (`enable_batch=true`), then confirm the new revision has `HF_TOKEN_SECRET_ID` and no `HF_TOKEN` |
| `HuggingFace credential: NOT resolved` in the logs | The instance role has no `secretsmanager:GetSecretValue` yet (see the row above) | Harmless: `nvidia/GR00T-N1.7-3B` is not gated and training continues. Fix by applying the Terraform. |
| `CUDA out of memory` | Batch geometry too large for 48 GB cards | `BATCH_SIZE=4` with `GRAD_ACCUM_STEPS=4` keeps the effective batch at 16. No `--gradient-checkpointing` flag exists in this SDK revision. |
| Single-GPU run OOMs at the optimizer step | At `num_gpus=1` DeepSpeed is disabled outright, so there is no ZeRO sharding and no offload path in the shipped configs | Use a 4-GPU instance. Making 1 GPU work requires source edits inside the image, not configuration. |
| `episodes.jsonl not found` | A LeRobot **v3** dataset — v3 dropped `meta/episodes.jsonl` | N1.7 needs `meta/episodes.jsonl` with a `length` field. It does **not** care about `codebase_version`, so **no v2.1 → v2.0 conversion is needed**; for a v3 source use the upstream `scripts/lerobot_conversion/convert_v3_to_v2.py`. |
| Stats look wrong / normalization is odd | A hand-written `meta/stats.json` was shipped | It is deleted on entry and regenerated in-container by `gr00t.data.stats` on rank 0. Do not author one. |
| `No space left on device` | Image is 22.6 GB | The launch template provides a 512 GiB gp3 root volume; confirm the CE is using it |
| CodeBuild fails with `Expected Commands[N] to be of string type: found subkeys instead` | A buildspec command containing `": "` as an unquoted plain YAML scalar parses as a mapping | Single-quote the whole command. Note this surfaces only *after* queue + provisioning, so a typo can cost ~28 minutes. |
| CodeBuild runs out of memory during `uv sync` / flash-attn | Default `BUILD_GENERAL1_LARGE` compute | Pass `--compute-type-override BUILD_GENERAL1_2XLARGE` |
| NGC key visible in the build record | Passed with `--environment-variables-override ... PLAINTEXT` | Let `buildspec-n17.yml` read it from Secrets Manager instead |
| Cached image on a warm Batch instance after a `:n17` re-push | Old layers still on the instance | Scale the CE to 0, wait for termination, scale back up |
| Foundation VPC not found | SSM parameter missing | Deploy Foundation first (`cd foundation/infra && terraform apply`) |

---

## Teardown

```bash
# 1. Scale compute to zero (it is already 0 unless you warmed it)
aws batch update-compute-environment \
  --compute-environment "physical-ai-dev-gr00t-compute" \
  --compute-resources 'minvCpus=0,desiredvCpus=0' \
  --region us-east-2

# 2. Wait for instances to terminate (~2 min), confirm:
aws batch describe-compute-environments \
  --compute-environments "physical-ai-dev-gr00t-compute" \
  --region us-east-2 --query 'computeEnvironments[0].computeResources.desiredvCpus'

# 3. Destroy the GR00T infra
cd isaac-gr00t-on-aws/infra
terraform destroy -var="aws_region=us-east-2" -var="enable_batch=true" -var="batch_instance_type=g6e.12xlarge"
```

Staged datasets and checkpoints live in the Foundation buckets and are not
removed by this destroy. Delete them explicitly if you want them gone — and leave
`s3://physical-ai-workshop-assets-<ACCOUNT_ID>/datasets/pick_place_data/` alone.

---

## Next Steps

- Re-run the smoke test when `g6e.12xlarge` capacity is available, and fill in
  the untested rows of the Validation Status table with measured numbers
- Apply the Secrets Manager Terraform to retire the plaintext `HF_TOKEN`
- Deploy the trained model for inference (see `containers/gr00t-inference/`)
- Compare with the [SageMaker N1.7 path](n17-sagemaker-training-guide.md) for
  managed training on the same 4× L40S hardware
