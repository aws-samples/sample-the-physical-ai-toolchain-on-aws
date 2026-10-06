# AWS Batch GR00T Training Guide (N1.7 + Unitree G1 Edu)

Fine-tune **GR00T N1.7-3B** on AWS Batch against the 201-episode **Unitree G1 Edu
(Dex3 hands, 28 DOF)** pick-and-place dataset. This guide covers the Terraform
infrastructure, the container rebuild that makes the image embodiment-selectable,
the dataset preparation, and the job submission.

**The primary target is a single GPU**: `g6e.4xlarge` (1× L40S 48 GB). Leaving the
vision tower frozen — the upstream default — drops the trainable set to
877,747,328 parameters, which fits one card with room to spare. The 4-GPU
`g6e.12xlarge` path is kept as an alternative, but it could not be scheduled at
all: 30 of 30 instance-launch attempts failed across all three us-east-2 AZs
during a 2 h 38 m wait.

Every number quoted below was either measured in this account or explicitly
labelled **derived** (calculated from source) or **untested** — see
[Validation Status](#validation-status). Two things this procedure has **not**
yet produced: a completed training run, and any measured GPU memory figure. No
container has started yet, so nothing in the training path is validated.

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
│    g6e.4xlarge (1× L40S 48 GB) · frozen vision tower · ckpts → S3        │
└─────────────────────────────────────────────────────────────────────────┘
```

**Hardware (primary):** `g6e.4xlarge` — 16 vCPU, 128 GiB RAM, 1× NVIDIA L40S
(48 GB), 512 GiB gp3 root volume from the launch template (the container image
alone is 22.6 GB).

**Why one GPU is enough: `TUNE_VISUAL=0`.** Upstream defaults `tune_visual` to
`False` (`gr00t/configs/finetune_config.py:52` and
`gr00t/configs/model/gr00t_n1d7.py:46`); this container's entrypoint used to force
it on. With the vision tower frozen, the trainable set is the projector plus the
DiT action head — **877,747,328 parameters**, derived from source — and the frozen
backbone retains no activation graph at all, because no backbone parameter
requires grad. Turning it on adds roughly 1.15B trainable parameters, their fp32
gradients and AdamW moments, *and* the whole ViT/LLM activation graph, which is
what previously forced 4 GPUs.

Derived memory budget at `TUNE_VISUAL=0` (fp32 params + fp32 grads + two AdamW
moments; the model loads in fp32 because `launch_finetune.py` forces
`load_bf16 = False`):

| term | size |
|---|---|
| params, 3.14B × 4 B | 11,978 MiB |
| grads, 877,747,328 × 4 B | 3,348 MiB |
| AdamW moments, 877,747,328 × 8 B | 6,697 MiB |
| **static total** | **22,023 MiB** |
| card | 45,776 MiB |
| **headroom for activations, workspaces, fragmentation** | **23,753 MiB** |

> **This budget is DERIVED, not measured.** No container has run. The first real
> run must report peak VRAM; until then treat the batch geometry below as a
> conservative starting point rather than a validated setting.

**The 4-GPU alternative (`g6e.12xlarge`, `TUNE_VISUAL=1`)** still works in
principle — DeepSpeed ZeRO-2 engages at `num_gpus > 1` and shards gradients and
optimizer state — but:

> ⚠️ **`g6e.12xlarge` could not be obtained in us-east-2.** One instance, 30
> launch attempts, all three AZs (2a ×15, 2c ×9, 2b ×6), every one
> `InsufficientInstanceCapacity`, over a 2 h 38 m 29 s `RUNNABLE` dwell. Quota was
> not the constraint. Do not plan a workshop around it; the only guaranteed
> capacity here is an on-demand reservation of `g6e.4xlarge`.

---

## Prerequisites

| Requirement | How to verify |
|-------------|---------------|
| AWS CLI configured | `aws sts get-caller-identity` |
| Terraform >= 1.5 (`~/bin/terraform` here) | `terraform version` |
| Foundation deployed | `aws ssm get-parameter --name /physical-ai/vpc-id --region us-east-2` |
| EC2 quota: `Running On-Demand G and VT instances` >= 16 vCPU (48 for the 4-GPU path) | Service Quotas, code `L-DB2E81BA` |
| Capacity for the instance type you target | see Troubleshooting — quota is not the same as capacity. `g6e.4xlarge` is reservation-backed here; `g6e.12xlarge` was unobtainable |
| A compute environment that can place the job's shape | **Not yet true for `g6e.4xlarge`**: the deployed CE offers `g6e.12xlarge` only, so a 16 vCPU / 1 GPU job stays `RUNNABLE`. Creating a single-GPU CE is a pending decision, not something this guide has done |
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
# Single-GPU (primary). This is the default shape — instance type g6e.4xlarge
# and job-definition 16 / 122880 / 1 — so the -var below is redundant and is
# passed only to make the shape explicit in the command you run.
terraform plan  -var="aws_region=us-east-2" -var="enable_batch=true" -var="batch_instance_type=g6e.4xlarge"
terraform apply -var="aws_region=us-east-2" -var="enable_batch=true" -var="batch_instance_type=g6e.4xlarge"

# 4-GPU alternative: the instance type AND the job shape must both change.
# terraform apply -var="aws_region=us-east-2" -var="enable_batch=true" \
#   -var="batch_instance_type=g6e.12xlarge" \
#   -var="batch_job_vcpus=48" -var="batch_job_memory_mib=180000" -var="batch_job_gpus=4"
```

> The **deployed** compute environment in this account is `g6e.12xlarge`, created
> before the single-GPU finding. Changing it is a pending decision, so nothing in
> this account can currently place a 1-GPU job.

**What this creates:**

| Resource | Purpose |
|----------|---------|
| CodeBuild: `physical-ai-dev-gr00t-training-build` | Builds the GR00T training container |
| CodeBuild: `physical-ai-dev-gr00t-inference-build` | Builds the GR00T inference container |
| Batch compute env: `physical-ai-dev-gr00t-compute` | `var.batch_instance_type` — the **deployed** one is `g6e.12xlarge` (4× L40S, 512 GiB gp3 root), min/desired 0 vCPU, max 96. A single-GPU run needs a CE offering `g6e.4xlarge` |
| Batch job queue: `physical-ai-dev-gr00t-queue` | Job scheduling |
| Batch job definition: `physical-ai-dev-gr00t-n17-training` | Image `:n17`, command `batch-train`, 16 VCPU / 122880 MEMORY / 1 GPU by default (variables — `48 / 180000 / 4` for the 4-GPU path), plus the **mandatory** `EMBODIMENT_CONFIG=g1_dex3` and `TUNE_VISUAL=0` (see [Required job-definition fields](#required-job-definition-fields)) |
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

### Required job-definition fields

Whether the job definition comes from Terraform, `aws batch
register-job-definition`, or the **AWS console form** (the path a workshop
attendee follows), these fields are all required. Nothing here is optional, and
two of them fail *silently* rather than loudly when they are wrong.

| Console field | Value | Why it matters |
|---|---|---|
| Job definition name | `physical-ai-dev-gr00t-n17-training` | Any name is fine, but it must contain `n17` for `submit_batch_training_n17.py` to accept it |
| Platform / type | EC2, container | Not Fargate: this needs GPUs |
| Image | `<ACCOUNT_ID>.dkr.ecr.us-east-2.amazonaws.com/physical-ai/gr00t-training:n17` | Note the spelling `gr00t`, not `groot` |
| Command | `batch-train` | Selects the Batch branch of the image entrypoint. `/opt/ml/code/batch-train` and `./batch-train` work too — the entrypoint matches on the basename. **An empty command runs the SageMaker path, which downloads no dataset and uploads no checkpoint.** |
| vCPUs | `16` | `g6e.4xlarge`. Use `48` for `g6e.12xlarge` |
| Memory | `122880` MiB | Leaves ~8 GiB of the 128 GiB instance to the ECS agent and kernel. Use `180000` for `g6e.12xlarge` |
| GPUs | `1` | 1× L40S. Use `4` only together with `TUNE_VISUAL=1` |
| Execution role | *leave empty* | Not needed: the token is fetched at runtime with the instance role |
| **Env: `EMBODIMENT_CONFIG`** | **`g1_dex3`** | **Mandatory.** The image default is `ur3`. Omitting it does not error — the job trains the UR3 end-effector config against 28-DOF joint data and dies inside `scipy`'s `Rotation.from_rotvec`, or worse, trains something meaningless |
| Env: `DATASET_S3_URI` | `s3://physical-ai-dev-datasets-<ACCOUNT_ID>/groot-data/pick_place_data/` | Must be a `physical-ai-dev-*` bucket; the instance role cannot read `physical-ai-workshop-assets-*` |
| Env: `CHECKPOINT_BUCKET` | `physical-ai-dev-checkpoints-<ACCOUNT_ID>` | Omitting it means the checkpoints are lost when the container exits |
| **Env: `TUNE_VISUAL`** | **`0`** | **Mandatory for a 1-GPU job.** `0` freezes the vision tower (upstream default) and is what makes a single 48 GB card viable: 877,747,328 trainable parameters instead of ~2.03B. `1` is only valid with 4 GPUs, where DeepSpeed ZeRO shards the optimizer state. A bad value (e.g. `ture`) is a hard failure, not a silent default |
| Env: `BASE_MODEL` | `nvidia/GR00T-N1.7-3B` | |
| Env: `BATCH_SIZE` / `GRAD_ACCUM_STEPS` | `2` / `8` | `BATCH_SIZE` is the **global, pre-accumulation** batch and the trainer computes `per_device = BATCH_SIZE / num_gpus`, so on 1 GPU it *is* the per-device batch. 2 × 8 keeps the effective optimizer batch at 16. On 4 GPUs use `8` / `2` for the same effective batch |
| Env: `MAX_STEPS` | `100` (or `15` for a smoke test) | |
| Env: `AWS_DEFAULT_REGION` | `us-east-2` | Used by the in-container `aws s3`/`secretsmanager` calls |
| Env: `HF_TOKEN_SECRET_ID` | `physical-ai/hf-token` | Optional in practice — `nvidia/GR00T-N1.7-3B` is not gated, so an unresolved lookup is a warning |

If the compute environment is also being created by hand, reuse the existing
CloudFormation stack outputs rather than inventing new ones:
`BatchPrivateSubnetId`, `BatchLaunchTemplateId` (the 512 GiB gp3 root volume the
22.6 GB image needs) and `BatchInstanceProfileArn` (the role scoped to
`physical-ai-dev-*` S3 and ECR pull).

> **The job's shape must be placeable on the compute environment's instance
> type.** AWS Batch does not reject an unplaceable job — it parks it in
> `RUNNABLE` indefinitely with a null `statusReason`, which looks exactly like a
> capacity shortage. A 16 vCPU / 1 GPU job cannot run on a CE that offers only
> `g6e.12xlarge`, and a 48 vCPU / 4 GPU job cannot run on one that offers only
> `g6e.4xlarge`. Check both before blaming capacity.

> **Do not put a HuggingFace token, or any other secret, in a job-definition
> environment variable.** Environment values are visible to anyone who can call
> `describe-job-definitions`. Pass `HF_TOKEN_SECRET_ID` instead.

### Pending apply: the HF_TOKEN → Secrets Manager migration

The Terraform in `isaac-gr00t-on-aws/infra/batch.tf` now codifies the job
definition **without** a plaintext `HF_TOKEN` entry. The live revision 1 in this
account still carries one, because the Terraform has been written and validated
but **not applied** — applying registers a new job-definition revision, which is
a separately approved action.

Two more written-not-applied changes ride along in the same resource: the
command is now `["batch-train"]` and the environment carries
`EMBODIMENT_CONFIG=g1_dex3`. Live revision 1 keeps `["/opt/ml/code/batch-train"]`
and no `EMBODIMENT_CONFIG`; the first is harmless after the container rebuild
(the entrypoint matches on basename), the second is **not** — until the apply
lands, every submission against revision 1 must pass `EMBODIMENT_CONFIG=g1_dex3`
in `--container-overrides`, which `submit_batch_training_n17.py` always does.

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
| `EMBODIMENT_CONFIG` | **Mandatory for this dataset: `g1_dex3`.** The image default is `ur3`, so leaving it out does not error — it silently trains the UR3 end-effector config on 28-DOF joint data and raises inside SciPy. `g1_dex3` selects `/opt/ml/code/g1_dex3_modality_config.py`. An unknown value fails immediately. |
| `MODALITY_CONFIG_PATH` | Optional absolute path inside the container; wins over `EMBODIMENT_CONFIG`. Set-but-missing is a hard failure, not a silent fallback. |
| `TUNE_VISUAL` | `0` (default, matching upstream) or `1`. `0` freezes the vision tower: 877,747,328 trainable parameters, fits one L40S. `1` restores the ~2.03B-parameter configuration and needs 4 GPUs. An unparseable value is a hard failure. |
| `HF_TOKEN_SECRET_ID` | Secrets Manager id, read only when `HF_TOKEN` is unset. |

The same rebuild also fixed the **command dispatch**. The entrypoint used to test
`$1` against the exact string `batch-train`, but the job definition (and anyone
typing the real path) passes `/opt/ml/code/batch-train`, which did not match — so
the container fell through to the SageMaker branch, which does **not** download
`DATASET_S3_URI`, does not resolve `HF_TOKEN_SECRET_ID` and does not upload
checkpoints. It now dispatches on the **basename** of `$1`, so `batch-train`,
`/opt/ml/code/batch-train` and `./batch-train` all reach the Batch path, and an
argument-free invocation still takes the SageMaker path.

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

**Measured build phases** — the first of this guide's two rebuilds, the one that
made the container embodiment-selectable (`BUILD_GENERAL1_2XLARGE`,
from-scratch, no layer cache):

| phase | duration |
|---|---|
| QUEUED | 0 s |
| PROVISIONING | 173 s |
| DOWNLOAD_SOURCE / INSTALL | 0 s |
| PRE_BUILD (ECR + NGC login) | 2 s |
| BUILD (NGC base pull, `uv sync` incl. flash-attn, layer export) | 436 s (7 m 16 s) |
| POST_BUILD (push 22.6 GB, two tags) | 583 s (9 m 43 s) |
| **total wall clock** | **19 m 57 s** |

That image measured 22,597,572,749 B. A second rebuild on the same buildspec and
compute type — adding the basename command dispatch and `TUNE_VISUAL` — took
**20 m 59 s** with the same phase shape, so budget ~20 min per rebuild. The
image now carried by `:n17` is from that second build: **22,597,509,931 B
(22.6 GB)**. The superseded image keeps its dated tag.

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

> **`EMBODIMENT_CONFIG=g1_dex3` is mandatory on every submission against live
> revision 1**, which ships no `EMBODIMENT_CONFIG` of its own. The image default
> is `ur3`, and the UR3 config is an end-effector config: against this 28-DOF
> joint-space dataset it raises inside SciPy's `Rotation.from_rotvec`. Omitting it
> produces a failed job, not a warning. `submit_batch_training_n17.py` always
> sends it; a hand-written `aws batch submit-job` must too. The same applies to
> `DATASET_S3_URI` (the live revision still points at the old UR3 prefix unless
> the Terraform has been applied).

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
      { "name": "BATCH_SIZE", "value": "2" },
      { "name": "GRAD_ACCUM_STEPS", "value": "8" },
      { "name": "TUNE_VISUAL", "value": "0" }
    ]
  }
}
```

No `command`, no `resourceRequirements` — environment only. The command is left
alone on purpose: the entrypoint dispatches on the basename of `$1`, so an
override that is not `…/batch-train` silently selects the SageMaker path.

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
      {"name": "BATCH_SIZE", "value": "2"},
      {"name": "GRAD_ACCUM_STEPS", "value": "8"},
      {"name": "TUNE_VISUAL", "value": "0"}
    ]
  }'
```

### Batch geometry

`BATCH_SIZE` is the **global, pre-accumulation** batch. The trainer derives
`per_device_train_batch_size = global_batch_size // num_gpus`
(`gr00t/experiment/experiment.py:263-266`, with
`assert global_batch_size % num_gpus == 0`), so the same number means different
things on different hardware:

| instance | GPUs | `BATCH_SIZE` | per-device | `GRAD_ACCUM_STEPS` | effective batch |
|---|---|---|---|---|---|
| **`g6e.4xlarge` (primary)** | 1 | **2** | **2** | **8** | 16 |
| `g6e.12xlarge` (alternative) | 4 | 8 | 2 | 2 | 16 |

> **`BATCH_SIZE` must be divisible by the job definition's GPU count.** The
> trainer asserts `global_batch_size % num_gpus == 0` before dividing, so the
> single-GPU default `2` fails at startup on the 4-GPU job definition that is live
> in this account. `submit_batch_training_n17.py` reads the job definition's GPU
> count and refuses the mismatch up front, naming the pair to use instead; a
> hand-written `aws batch submit-job` gets no such check and fails after the
> capacity wait and the 22.6 GB image pull.

The 4-GPU pair `8` / `2` on one GPU would mean a **per-device batch of 8**, not 2.
`2` / `8` is deliberately conservative: it holds the per-device batch at 2 while
halving the optimizer round trips of a `1` / `16` run, and the derived peak
(~22.0 GiB static plus activations, workspaces and fragmentation) leaves well over
40% of the card unused. `4` / `4` is the next step up — take it once a real run
reports peak VRAM. If OOM appears first, go to `1` / `16`. There is no
`--gradient-checkpointing` lever in this SDK revision, and the custom modules
never call a checkpoint function, so it would not help anyway.

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
Status**): the `batch-train` banner echoing `Embodiment:`, `Modality config:` and
`Tune visual:`; `Video pre-flight: OK - episode_000000.mp4 codec=… frame=…`;
`Generating stats for /opt/ml/input/data/training`; `Generating relative stats`
for `left_arm` / `right_arm`; the `launch_finetune.py` command line with
`--embodiment-tag new_embodiment`, the `g1_dex3` modality path and
`--no-tune-visual`; the 3.14B-parameter model load (877,747,328 trainable with
the vision tower frozen — this is the first chance to confirm the derived figure);
then `{'loss': …, 'grad_norm': …, 'learning_rate': …}` lines. On a 1-GPU run there
is **no** DeepSpeed line: ZeRO only engages at `num_gpus > 1`.

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
While a job is queued, `desiredvCpus` rises to the job's vCPU request and
returns to 0 once the job reaches a terminal state. **Match the number below to
the compute environment you actually deployed** — one instance's worth of vCPUs,
not more:

| compute environment | vCPUs per instance | `desiredvCpus` for one warm instance |
|---|---|---|
| `g6e.4xlarge` (single-GPU, primary) | 16 | `16` |
| `g6e.12xlarge` (4-GPU alternative) | 48 | `48` |

The 0 → **48** transition recorded in this guide's validation evidence was
measured on the **deployed `g6e.12xlarge`** compute environment with the 48-vCPU
job shape of live revision 1. On the `g6e.4xlarge` CE that Step 2 creates, the
equivalent figure is 16, and passing 48 there scales to **three** instances
(~$9/hr) instead of one.

```bash
# Keep ONE instance warm during an active iteration session.
# 16 = g6e.4xlarge (the Step 2 default). Use 48 only on a g6e.12xlarge CE.
# minvCpus stays 0 so the environment can still fall back to zero on its own.
aws batch update-compute-environment \
  --compute-environment "physical-ai-dev-gr00t-compute" \
  --compute-resources 'minvCpus=0,desiredvCpus=16' \
  --region us-east-2

# Scale back to zero when done
aws batch update-compute-environment \
  --compute-environment "physical-ai-dev-gr00t-compute" \
  --compute-resources 'minvCpus=0,desiredvCpus=0' \
  --region us-east-2
```

> Raising `minvCpus` above 0 pins the instance up: Batch will not scale it down
> on an empty queue, and it keeps billing until a second command sets it back.
> If you do raise it, you must return it to 0 explicitly.

On-demand Linux pricing in us-east-2, from the AWS Pricing API:

| instance | GPUs | $/hr |
|---|---|---|
| `g6e.4xlarge` (primary) | 1× L40S | **$3.00424** |
| `g6e.12xlarge` (alternative) | 4× L40S | **$10.49264** |

A run's cost is that rate × wall clock, and no training wall clock has been
measured yet — so this guide quotes no per-run figure. Note the single-GPU path is
~3.5× cheaper per hour, which only matters if it is not more than 3.5× slower; at
877,747,328 trainable parameters versus ~2.03B there is no measurement to say
either way yet.

> **Warning:** a warm instance bills continuously whether or not a job is
> running. Scale back to zero as soon as the iteration session ends. A job sitting
> in `RUNNABLE` bills nothing, because no instance exists yet.

---

## Configuration Variables (Terraform)

| Variable | Default | Description |
|----------|---------|-------------|
| `enable_batch` | `false` | Set to `true` to deploy the Batch resources |
| `batch_instance_type` | `g6e.4xlarge` | Matches the single-GPU job shape below. The currently deployed CE is `g6e.12xlarge`, so changing this replaces the compute environment |
| `batch_max_vcpus` | `96` | Maximum vCPUs for auto-scaling |
| `batch_job_vcpus` | `16` | Job-definition vCPU request. `48` for `g6e.12xlarge` |
| `batch_job_memory_mib` | `122880` | Job-definition memory request. `180000` for `g6e.12xlarge` |
| `batch_job_gpus` | `1` | Job-definition GPU request. `4` only with `batch_instance_type=g6e.12xlarge` |

The job definition carries two `lifecycle` preconditions that reject an
inconsistent pair at plan time: `batch_job_gpus` must equal the GPU count of
`batch_instance_type`, and that instance type's cards must be 48 GB-class. Both
mismatches are otherwise silent — too many GPUs parks the job in `RUNNABLE` with
no `statusReason`, and a 24 GB L4 (`g6.12xlarge`) lets the job launch and then
OOM mid-load.

---

## Validation Status

Measured in account `<ACCOUNT_ID>`, region `us-east-2`.

| Step | Status | Notes |
|------|--------|-------|
| Terraform deploy (Foundation + GR00T infra) | ✅ Validated | VPC, Batch CE, queue, CodeBuild, instance role all live and `VALID` |
| Secrets Manager HCL (`batch.tf`, `outputs.tf`) | ⚠️ Written, **not applied** | `fmt -check` + `init -backend=false` + `validate` all exit 0. No `plan`/`apply` against real state; live job-definition rev 1 still carries the plaintext `HF_TOKEN`. |
| Single-GPU job shape HCL (`batch_job_vcpus` / `_memory_mib` / `_gpus` = 16 / 122880 / 1) | ⚠️ Written, **not applied** | `validate` exits 0. Live rev 1 still requests 48 / 180000 / 4, which no `g6e.4xlarge` can place. |
| Single-GPU compute environment (`g6e.4xlarge`) | ⬜ **Does not exist** | The deployed CE offers `g6e.12xlarge` only. Creating one is a pending decision; until then a 1-GPU job cannot be scheduled at all. |
| Container rebuild (CodeBuild, `buildspec-n17.yml`) | ✅ Validated | 19 m 57 s for the embodiment-selectable build, 20 m 59 s for the second build that added the basename command dispatch and `TUNE_VISUAL`. ECR `:n17` digest changed each time. |
| Entrypoint command dispatch (`batch-train` / `/opt/ml/code/batch-train` / `./batch-train`) | ✅ Validated | All three argv forms reach the Batch branch, and an empty command still takes the SageMaker path — exercised against the entrypoint extracted from the pushed image. |
| Single-GPU memory budget at `TUNE_VISUAL=0` | ⚠️ **Derived, not measured** | 22,023 MiB static of 45,776 MiB, computed from source parameter counts. No container has run, so no peak-VRAM figure exists. |
| Dataset prep + `meta/modality.json` | ✅ Validated | Validator: 0 problems, 2 expected warnings, 201/201 episodes, 29,105 rows |
| Dataset staged to the dev datasets bucket | ✅ Validated | 409 objects / 134,298,177 B; workshop-assets source untouched at 407 |
| Batch job submission + queue admission | ✅ Validated | `SUBMITTED` → `RUNNABLE` in **30 s** |
| `g6e.12xlarge` instance launch | ❌ **Blocked — no EC2 capacity** | Job held `RUNNABLE` for **2 h 38 m 29 s**; 30/30 ASG launch attempts failed `InsufficientInstanceCapacity` across all three AZs (2a ×15, 2c ×9, 2b ×6) |
| `g6e.4xlarge` instance launch | ⬜ Untested | No single-GPU compute environment exists yet, so nothing has been submitted against one |
| ECR image pull onto a Batch instance | ⬜ Untested | `STARTING` never reached; no measured pull time exists |
| In-container dataset download | ⬜ Untested | `RUNNING` never reached |
| AV1 video decode (torchcodec pre-flight) | ⬜ Untested | The largest remaining risk. The pre-flight turns a silent late failure into an immediate, legible one. |
| GR00T N1.7-3B model load (1 GPU, frozen vision tower) | ⬜ Untested | No container ever started; the 877,747,328 trainable-parameter figure is derived from source, not read from a log |
| DeepSpeed ZeRO at 4 GPUs (`TUNE_VISUAL=1`) | ⬜ Untested | No container ever started |
| Training steps / loss / per-step timing | ⬜ Untested | **No training timing is quoted anywhere in this guide** because none has been measured |
| Checkpoint upload to S3 | ⬜ Untested | `s3://physical-ai-dev-checkpoints-<ACCOUNT_ID>/` is still empty (0 objects) |

Nothing was billed for the blocked run: no `g6e.12xlarge` was ever launched. AWS
Batch keeps retrying on its own, so a job left in `RUNNABLE` starts by itself
when capacity frees up — it does not need resubmitting.

### The job definition was never functional, independent of the capacity problem

Worth stating plainly, because it changes how much the blocked run proved: the
`:n17` job definition could not have trained anything even with capacity. Its
command `/opt/ml/code/batch-train` did not match the entrypoint's exact-string
test, so the container would have taken the SageMaker branch, found
`/opt/ml/input/data/training` empty (that branch performs no `aws s3 sync`) and
failed with no dataset — and it would never have uploaded a checkpoint. The
capacity shortage hid the defect rather than causing it. The basename dispatch
fixes it, but **the fix is unexercised against a real job**: a smoke test must
actually run before this guide is treated as a validated procedure.

The same applies to the resource shape. Live revision 1 requests 48 vCPU /
180000 MiB / 4 GPU, which no `g6e.4xlarge` can place — so even with reserved
single-GPU capacity, a job against revision 1 would sit in `RUNNABLE` with a null
`statusReason` and look like another capacity shortage. That is why the shape is
now a Terraform variable rather than a literal.

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
| Job stuck in `RUNNABLE`, `statusReason` null, **and `desiredvCpus` never rises** | The job's `resourceRequirements` cannot be placed on the CE's instance type — e.g. 16 vCPU / 1 GPU against a `g6e.12xlarge`-only CE, or 48 vCPU / 4 GPU against a `g6e.4xlarge` one. Batch parks the job instead of rejecting it | Compare `aws batch describe-job-definitions --job-definition-name <name> --query 'jobDefinitions[-1].containerProperties.resourceRequirements'` against `aws batch describe-compute-environments --compute-environments <ce> --query 'computeEnvironments[0].computeResources.instanceTypes'`. Fix the shape (`batch_job_vcpus` / `batch_job_memory_mib` / `batch_job_gpus`) or the CE — do not wait for capacity. |
| Job stuck in `RUNNABLE`, `statusReason` null, no log stream | No **capacity** for the instance type (not quota, not config) — the CE is `VALID` and `desiredvCpus` did rise | Inspect the backing ASG: `aws autoscaling describe-scaling-activities --auto-scaling-group-name <ce-asg> --region us-east-2 --query 'Activities[].{code:StatusCode,cause:StatusMessage}'`. `InsufficientInstanceCapacity` means wait — Batch retries by itself. Check the quota separately with `aws service-quotas get-service-quota --service-code ec2 --quota-code L-DB2E81BA`. |
| Tempted to pin the CE to the AZ the error message names | The "you can currently get capacity in us-east-2b/c" hint is a point-in-time pool estimate | **Do not.** us-east-2b was recommended by both the 2a and 2c failure messages and itself failed 6/6. Pinning reduces the number of pools the ASG can try and fixes nothing. |
| `CannotPullContainerError` / repository does not exist | Job submitted against `physical-ai-dev-gr00t-training` (any revision), which points at the misspelled repo `physical-ai/groot-training` | Submit against `physical-ai-dev-gr00t-n17-training` |
| Container trains a UR3 end-effector config on G1 data, or SciPy raises in `Rotation.from_rotvec` | `EMBODIMENT_CONFIG` not set, so the image default `ur3` ran. Live job-definition revision 1 ships no `EMBODIMENT_CONFIG` | Pass `EMBODIMENT_CONFIG=g1_dex3` — in `--container-overrides` for every submission, or as a job-definition environment entry (see [Required job-definition fields](#required-job-definition-fields)). A Batch `command` override cannot fix this: `Dockerfile.n17` sets an `ENTRYPOINT` and Batch has no entrypoint override. |
| Logs show the `=== GR00T N1.7-3B Fine-Tuning ===` banner but no `Downloading dataset from S3`, and the job dies on an empty dataset directory | The container took the SageMaker branch: the `command` basename is not `batch-train` (e.g. empty command, or `bash -c …`). That branch performs no `aws s3 sync` and no checkpoint upload | Set the command to `batch-train` (`/opt/ml/code/batch-train` and `./batch-train` also work on images built after the basename-dispatch fix). Confirm the first log line is the `batch-train` banner with `Embodiment:` and `Modality config:`. |
| Video pre-flight fails with a torchcodec/ffmpeg error on `episode_000000.mp4` | AV1 decode (`video.codec: "av1"`) unavailable in the image's ffmpeg/torchcodec combination | Transcode all 201 videos to H.264 (`ffmpeg -i in.mp4 -c:v libx264 -pix_fmt yuv420p out.mp4`), re-stage the dataset to S3, resubmit. This is a data fix, not a container change. |
| `AccessDenied` reading `DATASET_S3_URI` | The instance role grants S3 only on `arn:aws:s3:::physical-ai-dev-*`, and the canonical dataset lives in `physical-ai-workshop-assets-<ACCOUNT_ID>` | Stage a prepared copy to `s3://physical-ai-dev-datasets-<ACCOUNT_ID>/groot-data/pick_place_data/`. Do **not** widen the policy. |
| `HF_TOKEN` still appears in plaintext in the job definition | The Secrets Manager migration is written in `batch.tf` but **not applied** — applying registers a new revision | Apply `isaac-gr00t-on-aws/infra` deliberately (`enable_batch=true`), then confirm the new revision has `HF_TOKEN_SECRET_ID` and no `HF_TOKEN` |
| `HuggingFace credential: NOT resolved` in the logs | The instance role has no `secretsmanager:GetSecretValue` yet (see the row above) | Harmless: `nvidia/GR00T-N1.7-3B` is not gated and training continues. Fix by applying the Terraform. |
| `AssertionError: global_batch_size must be divisible by num_gpus` at startup | `BATCH_SIZE` does not divide the job definition's GPU count — e.g. the single-GPU default `2` against a 4-GPU job definition | Match the pair to the shape: `2`/`8` for 1 GPU, `8`/`2` for 4 GPUs. `submit_batch_training_n17.py` refuses the mismatch before submitting. |
| `CUDA out of memory` on one GPU | Either `TUNE_VISUAL=1` (which needs 4 GPUs for ZeRO sharding), or the per-device batch is too large — remember `BATCH_SIZE` *is* the per-device batch at `num_gpus=1` | Confirm the log banner says `Tune visual: False`. Then step down `BATCH_SIZE=1` with `GRAD_ACCUM_STEPS=16` to hold the effective batch at 16. No `--gradient-checkpointing` flag exists in this SDK revision, and the custom modules never call a checkpoint function, so it would not help. |
| `TUNE_VISUAL=1` on a 1-GPU instance OOMs | At `num_gpus=1` DeepSpeed is disabled outright (`experiment.py:258-262`), so the ~2.03B trainable parameters carry unsharded fp32 grads and AdamW moments, and the ViT/LLM activation graph is retained | Leave `TUNE_VISUAL=0` on one GPU. Visual tuning is a 4-GPU configuration. |
| Log banner shows `Tune visual: True` when you expected frozen | `TUNE_VISUAL` was set to `1` somewhere — job definition environment, or `--tune-visual` on the submitter | The submitter always sends `TUNE_VISUAL` explicitly, so the submitted value wins over the job definition. Check the payload it printed. |
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

# 3. Destroy the GR00T infra. Pass the SAME -var values you applied with;
#    g6e.4xlarge is the Step 2 default, so change it only if you deployed the
#    4-GPU alternative (then also pass the 48 / 180000 / 4 job shape).
cd isaac-gr00t-on-aws/infra
terraform destroy -var="aws_region=us-east-2" -var="enable_batch=true" -var="batch_instance_type=g6e.4xlarge"
```

Staged datasets and checkpoints live in the Foundation buckets and are not
removed by this destroy. Delete them explicitly if you want them gone — and leave
`s3://physical-ai-workshop-assets-<ACCOUNT_ID>/datasets/pick_place_data/` alone.

---

## Next Steps

- Create a single-GPU (`g6e.4xlarge`) compute environment against the reserved
  capacity, then run the smoke test and fill in the untested rows of the
  Validation Status table with measured numbers — including **peak VRAM**, which
  is the one number that turns the derived memory budget into a measured one
- Apply the Terraform so the job definition carries the single-GPU shape,
  `EMBODIMENT_CONFIG`, `TUNE_VISUAL` and the Secrets Manager reference (this
  registers a new job-definition revision)
- Apply the Secrets Manager Terraform to retire the plaintext `HF_TOKEN`
- Deploy the trained model for inference (see `containers/gr00t-inference/`)
- Compare with the [SageMaker N1.7 path](n17-sagemaker-training-guide.md) for
  managed training on the same 4× L40S hardware
