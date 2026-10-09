# Isaac Lab Arena Evaluation on AWS

Fine-tune a vision-language-action (VLA) model, run it in simulation, check the
checkpoint and evaluation evidence, and register the result in SageMaker.
The same `pai arena` commands run this pipeline on **managed SageMaker** or in
**SageMaker local mode on your EC2 GPU host**.
The `vla` compatibility command uses the same backend and saved records;
existing scripts and the supplementary examples using it continue to work.

The main walkthrough uses **GR00T N1.6 with Isaac Lab Arena**: a GR1 humanoid
places an item in a fridge and closes the door. Additional cells pair GR00T,
OpenVLA and MolmoAct2 with LIBERO.

**Start with [notebooks/runbook.ipynb](notebooks/runbook.ipynb).** Edit the `arena`
section of [`config.json` at the repository root](../config.json), then run its
cells from setup through teardown. It uses the same `pai arena` commands shown
below and displays the actual results and, for Arena, the run's recorded video.
Use a CPU SageMaker AI notebook instance (`notebook-al2023-v1`, `ml.t3.xlarge`,
150 GiB volume) so the client tools and retained evidence stay in AWS. Clone
onto its data volume, configure the account and token-file paths, and choose
Run All. The notebook installs missing client tools without sudo and creates
its virtual environment. Training and evaluation run on the separate GPU EC2
host and SageMaker jobs. The supplied notebook remains available for viewing
results after application teardown; stop that notebook instance when finished.

The bootstrap supports Linux x86_64. A Studio JupyterLab environment can use the
same setup, but acceptance records name the exact environment actually tested.
Other clients can run the notebook with the documented tools already installed.
The committed notebook has no saved execution outputs; see the run's executed
copy before claiming an end-to-end pass.

- [Pipeline overview](#pipeline-overview)
- [1. Setting up and running](#1-setting-up-and-running)
- [2. Supplementary information](#2-supplementary-information)

## Pipeline overview

<a id="what-youll-build"></a>

```text
                         Your training dataset
                                  │
                                  ▼
                             FineTune (GPU)
                                  │ checkpoint
            Existing checkpoint ──┤
                                  ▼
                             SimEval (GPU)
                                  │ episode results and metrics
                                  ▼
                             Validate (CPU)
                                  │ checked evidence and attestation
                                  ▼
                             SuccessGate
                                  │ threshold passed
                  ┌───────────────┴──────────────────┐
                  ▼                                  ▼
        Local: development result          Managed: RegisterModel
                                           PendingManualApproval
```

### The five pipeline steps

| Step | What it does | Output |
| --- | --- | --- |
| **FineTune** | Trains the selected model on the cell's dataset. | Checkpoint and training metadata. |
| **SimEval** | Runs the trained policy in closed-loop simulation. | Per-episode outcomes and success rate. |
| **Validate** | Checks checkpoint integrity, evaluation evidence and the selected task's contract. | Validation receipt and attestation; invalid evidence fails the step. |
| **SuccessGate** | Compares the validated success rate with your threshold. | A pass/fail decision. |
| **RegisterModel** | Records the accepted model and evidence in SageMaker Model Registry. Managed mode only. | A model package awaiting manual approval. |

Both execution modes use S3 for checkpoint handoff and ECR for images.
Managed jobs use separate step roles and the deployed publication buckets.
Local containers share the selected EC2 host and publish to a separate,
versioned development bucket. Local execution does not register a model.
For debugging, we recommend **local mode on an EC2 GPU instance**.

<a id="compatibility-matrix"></a>

### Supported cells

A **cell** selects the model version, simulator and task suite together.
Choose one explicitly when launching; the complete training-to-result workflow
runs unless you request a shorter path.

| Cell name | Model | Simulator and task | CLI execution modes |
| --- | --- | --- | --- |
| `gr00t-n16-arena` | GR00T N1.6 | Isaac Lab Arena, GR1 fridge task — main walkthrough | Local, managed |
| `gr00t-n17-arena` | GR00T N1.7 | Isaac Lab Arena, GR1 fridge task | Managed |
| `gr00t-n17-libero` | GR00T N1.7 | LIBERO spatial, ten tasks | Local, managed |
| `molmoact2-libero` | MolmoAct2 | LIBERO spatial, ten tasks | Local, managed |
| `openvla-libero` | OpenVLA | LIBERO spatial, ten tasks | Local, managed |

OpenVLA/MolmoAct2 with Arena and N1.6 with LIBERO are not supported.
See [model and embodiment compatibility](#model-and-embodiment-compatibility)
for the reasons. A supported cell is an available application path, not a claim
that every version or deployment has passed; the
[recorded runs](#accepted-reference-runs) identify verified executions.

### GPU, time, cost and storage at a glance

| Plan for | Practical starting point |
| --- | --- |
| Local GPU host | Tested `g6e.8xlarge`: one L40S with 48 GB nominal GPU memory, 32 vCPUs and 256 GiB RAM. |
| Local storage | The host recipe starts with 800 GiB root EBS and separate 900 GB NVMe scratch. After images are present, the Arena training sample requires **90 GiB free on root and 250 GiB free on scratch**. Retained runs need additional space. |
| First image build | Three Arena images took **92m35s** in the recorded separate-account deployment; downloads and host preparation are additional. Reuse compatible builds afterward. |
| Small local Arena run | 200 training steps and three episodes took **about 27–32 minutes**, roughly **$2.07–$2.39** of host compute. |
| Longer local Arena run | The recorded 20,000-step, 20-episode experiment took **5h38m12s**, roughly **$25.53** of host compute. This is an observed experiment, not a prescribed benchmark protocol. |
| Managed sample | Choose a GPU instance for each of FineTune and SimEval. Actual training can finish well before a capacity queue clears; recorded full pipelines ranged from about an hour to many hours. |

Local estimates use the **$4.52856/hour Linux On-Demand rate checked on
2026-09-17** for us-east-1/us-east-2. They exclude builds, storage, networking,
failed attempts and time the host sits idle; a running idle host is still billed.
See the [per-cell instance matrix](#cell-instance-and-storage-matrix) and
[dated measurements and pricing](#gpu-runtime-and-cost-planning) for managed
rates, other models and the evidence behind these numbers.

## 1. Setting up and running

Follow this section from prerequisites to a result. The examples use GR00T N1.6
and Arena; the cell table above lists the other choices.

**Contents**

- [Required setup](#required-setup)
- [Install and discover](#install-and-discover)
- [Common tasks](#common-tasks)
  - [Deploy once](#deploy-once)
  - [Run on managed SageMaker](#run-the-full-pipeline-on-managed-sagemaker)
  - [Run locally on EC2](#run-the-full-local-pipeline-on-ec2)
  - [Train only](#train-only)
  - [Evaluate an existing checkpoint](#evaluate-an-existing-checkpoint)
  - [Inspect, reconnect and finish](#inspect-reconnect-and-finish)
- [CLI reference](#cli-reference)
  - [Commands and shared options](#commands-and-shared-options)
  - [Deployment arguments](#deployment-arguments)
  - [Run arguments](#run-arguments)
  - [Status and cancellation arguments](#status-and-cancellation-arguments)

<a id="setup"></a>
<a id="shared-prerequisites"></a>

### Required setup

Start with the [AWS Physical AI Toolchain setup](../README.md#prerequisites)
and its complete local checkout. Run the commands below on a laptop or other
**provisioning machine**; it needs no GPU. Arena's requirements are listed below.
`pai arena deploy` creates Foundation resources, application buckets and images if
you are starting with a new account.

| Requirement | What to prepare |
| --- | --- |
| Tools | Bash, Git, **Python 3.11–3.12 for the shared CLI**, AWS CLI v2, **Terraform ≥1.9**, and outbound HTTPS. See the toolchain's [Python setup](../README.md#python-environment). The separate EC2 worker also supports Python 3.10. |
| Target AWS account | A working provisioning/admin profile able to create infrastructure and grant the runtime roles access. Application deployment is supported in **us-east-1 only**. |
| Hugging Face token | A read token from [HF settings](https://huggingface.co/settings/tokens), with the [GR00T N1.6 model terms](https://huggingface.co/nvidia/GR00T-N1.6-3B) accepted and access to the [Arena dataset](#prerequisite--huggingface-token-required-for-every-gr00t-run). Keep it in a private file outside the repository and set `arena.hf_token_file`. |
| NVIDIA NGC key | A personal key with NGC Catalog access from [NGC API Keys](https://org.ngc.nvidia.com/setup/api-keys), with the required NVIDIA terms accepted. Keep it in a private file outside the repository and set `arena.ngc_token_file`. |
| Managed execution | SageMaker **Training** GPU quota for both GPU steps, plus CPU Processing quota for Validate. Quota does not guarantee capacity. |
| Local execution | Use `deploy --create-local-host g6e.8xlarge` to create the GPU EC2 host and prepare it. A supplied host is also supported. Training and evaluation run on that host, not on the machine running the notebook. |

The shared configuration requires the target `account_id`, token-file paths and,
when using named credentials, `profile`. Defaults select the region, deployment
name, cells, instances and runtime. Explicit flags override those defaults.
The CLI checks the real AWS account before provisioning and creates the missing
deployment-specific Secrets Manager entries from those files. It does not request
quota increases. NGC registry authentication is checked by CodeBuild; selected
GR00T model and dataset access is checked before builds start.

The token commands below are an alternative for administrators supplying existing
Secrets Manager entries. With token files configured, `pai arena deploy` handles
their creation; do not also create duplicate entries manually.

In a Bash shell, choose the **target** profile and check the account before
creating anything. Use a Python executable in the supported range if your
default `python3` is newer:

```bash
export TARGET_PROFILE=your-target-account-profile
git --version
python3 --version
aws --version
terraform version
aws sts get-caller-identity --profile "$TARGET_PROFILE" \
  --region us-east-1 --query Account --output text --no-cli-pager
```

Store both tokens as **plaintext values**, not JSON key/value objects, in
Secrets Manager in that account and us-east-1. Existing secrets can be reused.
For new secrets, use the console's **Other type of secret → Plaintext**, or
these commands with private files containing only the token:

```bash
chmod 600 /secure/path/hf-token /secure/path/ngc-token
aws secretsmanager create-secret --profile "$TARGET_PROFILE" --region us-east-1 \
  --name vla-pipeline/hf-token --secret-string file:///secure/path/hf-token
aws secretsmanager create-secret --profile "$TARGET_PROFILE" --region us-east-1 \
  --name vla-pipeline/ngc-token --secret-string file:///secure/path/ngc-token
```

The files above are your secure input files, not repository files. Do not
re-create or overwrite secrets already supplied by an administrator. Confirm
both reads below succeed; they print identifiers, **not token values**:

```bash
(
  set -e
  for secret_name in vla-pipeline/hf-token vla-pipeline/ngc-token; do
    aws secretsmanager get-secret-value --profile "$TARGET_PROFILE" \
      --region us-east-1 --secret-id "$secret_name" \
      --query '{ARN:ARN,VersionId:VersionId}' --output json
  done
)
```

Resolve any missing secret or denied read before deploying. These checks confirm
AWS access; your token must also have the provider permissions described above.
Terraform grants consuming roles access but does not obtain token values.
[Detailed token setup and errors](#provision-and-check-both-tokens-before-infrastructure)
are covered later.

**For local execution:** `--create-local-host` uses the existing GPU DLAMI and
Terraform host recipe, with no separate CPU controller. Alternatively,
`--local-host` prepares a running instance you supply; its ownership stays with
you. A supplied host can be in another region (us-east-2 has been used), while
S3, ECR, secrets and SageMaker remain in us-east-1. Cross-region transfers can
incur charges. [Manual host setup](#prepare-the-ec2-host-and-aws-resources)
remains available for that supplied-host path.

### Install and discover

From the root of your existing toolchain checkout:

```bash
cd isaac-lab-arena-on-aws
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e .. -e .
pai arena --help
pai arena cells
pai arena cells --details
```

This installs both the parent toolchain CLI and the Arena backend in one
environment. Arena reads the root configuration's `arena` section; other
components' settings remain independent. Reconnecting uses the saved
deployment's account, region and images.

Discovery is offline: it needs no deployment or GPU. `pai arena cells --details`
shows declared instances, disk sizes and episode counts, not measured minimum
requirements. `pai arena run --help` and `pai arena deploy --help` list their arguments.

All following commands run **from this component directory on the provisioning
machine**, with its virtual environment active. Keep runtime/build inputs at a
clean, committed revision. Notebook outputs and the root configuration may
change; resolved configuration is recorded with each run. After reconnecting, return here and run
`source .venv/bin/activate`; restore `TARGET_PROFILE` when a command needs it.
Saved deployment records retain the selected profile, images and host.

<a id="command-guide"></a>

### Common tasks

These examples use `gr00t-n16-arena` and a deployment named `arena-review`.
The sample runs the **complete workflow** with a small budget: 200 training
steps, three episodes and seed 100. Its zero threshold tests that the workflow
functions; it is not a model-quality target. For a larger experiment, choose
training/evaluation counts, threshold and runtime deliberately.

#### Deploy once

Create Foundation and Arena infrastructure, build the required images and
save their immutable digests. No training starts during deployment:

```bash
pai arena deploy --create-local-host g6e.8xlarge --yes
pai arena status arena-notebook
```

These commands use `arena.deployment_name` (default `arena-notebook`), credentials,
token files and `prepare_cells` from `config.json`. Use the configured name in
`status`. Cold setup takes hours: a recorded Arena-only setup took 2h11m;
building all families adds work. Shared images are built once. Status checks
published images, the GPU host and current disk reserves before training.

For managed-only use, omit `--create-local-host`. For a supplied host, replace it
with `--local-host INSTANCE_ID --host-region HOST_REGION`; that host will not be
deleted by deployment teardown.

Choose one deployment command and wait for `Ready`. If preparation is
interrupted, continue the same operation with
`pai arena deploy --resume --yes`.
If infrastructure/images already exist, follow
[reuse an existing deployment](#reuse-an-existing-deployment-or-host) instead.

With the shared configuration set, the ordinary short-run commands are:

```bash
pai arena run --cell gr00t-n16-arena --mode local --sample --run-id my-local-sample --wait
pai arena report my-local-sample --output-dir local-dev/my-local-sample --include-video
pai arena run --cell gr00t-n16-arena --mode managed --sample --run-id my-managed-sample \
  --instance FineTune=ml.g6e.12xlarge --instance SimEval=ml.g6e.12xlarge --wait
pai arena report my-managed-sample --output-dir local-dev/my-managed-sample --include-video
```

`--sample` supplies 200 training steps, three trials and threshold zero.
Explicit budgets override those values. It runs the full graph unless you
explicitly provide a checkpoint or a stopping step. The longer examples below
show every choice explicitly; replace `arena-review` with your saved deployment.

After inspecting the results, preserve the evidence and remove owned resources:

```bash
pai arena destroy arena-notebook --remove-evidence --yes
pai arena status arena-notebook
```

Destroy includes all recorded runs under that deployment, including optional
experiments. Verified evidence is retained locally before its temporary AWS
archive is removed. Supplied hosts, shared image repositories and pre-existing
model package groups retain their original ownership. A model package group
created here is also retained if another execution has added packages to it.
SageMaker job history and shared TrainingJobs/ProcessingJobs log groups remain;
stopping jobs and removing their pipeline does not erase that history.

#### Run the full pipeline on managed SageMaker

Submit FineTune → SimEval → Validate → SuccessGate → RegisterModel:

```bash
pai arena run --deployment arena-review --mode managed --cell gr00t-n16-arena \
  --instance FineTune=ml.g6e.8xlarge --instance SimEval=ml.g6e.8xlarge \
  --train-steps 200 --eval-trials 3 --eval-seed 100 --threshold 0 \
  --max-runtime-seconds 21600 --run-id arena-managed-01
pai arena status arena-managed-01 --follow
```

The result is checked evidence and a model package in PendingManualApproval.
A submitted execution may still be waiting for capacity.

#### Run the full local pipeline on EC2

After host preparation, submit FineTune → SimEval → Validate → SuccessGate
from the same provisioning machine:

```bash
pai arena run --deployment arena-review --mode local --cell gr00t-n16-arena \
  --train-steps 200 --eval-trials 3 --eval-seed 100 --threshold 0 \
  --max-runtime-seconds 21600 --run-id arena-local-01
pai arena status arena-local-01 --follow
```

The result is verified development evidence in S3 and saved run diagnostics.
The host runs the containers; your laptop follows them through SSM.

#### Train only

Produce a checkpoint on the prepared host without running simulation:

```bash
pai arena run --deployment arena-review --mode local --cell gr00t-n16-arena \
  --through FineTune --train-steps 200 --max-runtime-seconds 21600 \
  --run-id arena-train-01
pai arena status arena-train-01 --follow
```

The status output and saved `job-outputs.json` identify the checkpoint URI
and S3 version.

#### Evaluate an existing checkpoint

Skip training and run SimEval → Validate → SuccessGate on EC2. Replace the S3
URI with the compatible checkpoint produced above:

```bash
pai arena run --deployment arena-review --mode local --cell gr00t-n16-arena \
  --checkpoint-s3 s3://YOUR_VERSIONED_BUCKET/path/model.tar.gz \
  --eval-trials 3 --eval-seed 100 --threshold 0 --max-runtime-seconds 21600 \
  --run-id arena-checkpoint-01
pai arena status arena-checkpoint-01 --follow
```

For the managed equivalent, use `--mode managed` and add
`--instance SimEval=ml.g6e.8xlarge`; registration is included.
For simulation alone, add `--through SimEval` and omit `--threshold`.
The [step selection reference](#step-selection-and-checkpoint-inputs) covers
all supported stopping points and checkpoint requirements.

#### Inspect, reconnect and finish

| Task | Command |
| --- | --- |
| List saved operations | `pai arena status` |
| Reconnect to the same local run | `pai arena status arena-local-01 --follow` |
| Follow deployment preparation | `pai arena status arena-review --follow` |
| Resume interrupted deployment | `pai arena deploy --name arena-review --resume --yes` |
| Cancel a managed execution | `pai arena stop arena-managed-01` |
| Inspect a run's cleanup scope | `pai arena cleanup arena-local-01` |
| Finish that run and remove accepted local containers | `pai arena cleanup arena-local-01 --yes` |

Ctrl-C stops watching without cancelling the run. A completed sample should
report both successful requested steps and independent verification, followed
by the robot outcome. For example, “workflow verified; 1 of 3 episodes
succeeded” proves the pipeline worked, not that the policy is production-ready.
After verification, `cleanup --yes` archives successful local evidence, checks
the S3 readback and removes that run's stopped containers. A run launched over
SSM uses the same connection for cleanup; an interactive host shell is optional.
Failed or cancelled runs are stopped with diagnostics retained. Host, caches,
working files, S3 output and managed registrations remain. See the
[archive and storage instructions](#archive-an-accepted-local-run) before
reclaiming working files, then [finish the session](#finish-a-local-session).
The host is not stopped automatically.

The next tables list the available commands and arguments. Longer explanations
and manual procedures are in [Supplementary information](#2-supplementary-information).

### CLI reference

#### Commands and shared options

| Command | Purpose |
| --- | --- |
| `pai arena cells [--cell NAME] [--details]` | List available cells, or inspect one cell's declared hardware and task counts. |
| `pai arena deploy --name NAME ...` | Prepare infrastructure/images and optionally a host; alternatively select existing resources. |
| `pai arena run --deployment NAME ...` | Submit one complete or explicitly shortened execution. It never builds images. |
| `pai arena status [ID] [--follow]` | List saved operations, inspect one, or follow that same operation. |
| `pai arena stop RUN_ID` | Request cancellation of one recorded execution; preserve host and evidence. |
| `pai arena cleanup RUN_ID [--yes]` | Show cleanup scope; with `--yes`, stop owned work and archive/remove verified local containers. |
| `pai arena report RUN_ID [--output-dir DIR] [--include-video]` | Export that independently verified run's evidence, training/evaluation summary and optional actual recordings. No new experiment. |
| `pai arena destroy DEPLOYMENT [--yes] [--remove-evidence]` | Show teardown scope, or preserve evidence and remove deployment-owned resources. |

Each command accepts `--help`. Put global options **before** the command,
for example `pai arena --json cells --details`.

| Global option | Meaning |
| --- | --- |
| `--state-dir DIRECTORY` | Persistent deployment/run records. Overrides `VLA_STATE_DIR`; otherwise defaults to `~/.local/state/vla`. Reuse the same directory across sessions. |
| `--json` | Machine-readable output. Managed/SSM followers emit successive JSON records; direct EC2 following retains verifier diagnostics on stderr and emits its final record on stdout. |
| `--debug` | Print a traceback on command failure, for example `pai arena --debug status RUN_ID`. Normal errors remain concise. |
| `--config FILE` | Use this shared toolchain configuration instead of the root `config.json`. Put it before the command. |

#### Deployment arguments

| Argument | Meaning |
| --- | --- |
| `--name NAME` | Saved deployment name; defaults to `arena.deployment_name`. Runs select it with `--deployment` or that same default. |
| `--account-id ACCOUNT` | Expected twelve-digit AWS account; defaults to `arena.account_id`. Credentials must resolve to that account before writes. |
| `--profile PROFILE` | Target provisioning profile. Omit on a host using its EC2 instance role. |
| `--region us-east-1` | Application region; default and only supported value. |
| `--project NAME` | Foundation/SSM namespace; new configured deployments use their deployment name. Legacy unconfigured commands default to `physical-ai`. Saved selections retain their namespace. |
| `--environment NAME` | Resource environment suffix; default `dev`. Changing this alone does not isolate the project-wide SSM namespace. |
| `--cell NAME` | Cell to prepare; repeat for several. Required on a new operation unless cells come from an imported/existing selection. |
| `--image STEP=ECR_URI` | Select a published image for `FineTune` or `SimEval`; full private ECR URI required, resolved to a digest. Shared images are reused when preparing multiple cells. |
| `--use-existing` | Check/save existing resources and images. No infrastructure changes, grants, builds or host installation. |
| `--selection FILE` | Import a schema-version-1 nonsecret selection, including prepared image/host references. |
| `--local-host INSTANCE_ID` | Supplied GPU instance to prepare or select. Does not acquire capacity. |
| `--create-local-host TYPE` | Create and prepare an owned `g6e` GPU EC2 instance; omit for managed-only deployment. Cannot combine with a supplied host or existing-selection modes. |
| `--gpu-zone ZONE` | Optional availability zone for a newly created GPU host. Quotas are checked; capacity is established by the launch. |
| `--hf-token-file FILE` / `--ngc-token-file FILE` | Private plaintext files outside the checkout. Create missing deployment-specific token secrets; existing secrets remain supplied resources. Default paths come from the Arena configuration. |
| `--host-region REGION` | That instance's region; may differ from the application region. |
| `--scratch-root PATH` | Working directory on the host's separate mounted scratch filesystem. |
| `--development-bucket NAME` | Versioned bucket for local checkpoints and evidence. |
| `--expected-role NAME` | Expected EC2 runtime role; host preparation preserves/adds access to the attached profile and checks the resulting identity. |
| `--prepare-host` | Prepare a supplied host from saved/imported resources and published images; no Terraform or image builds. |
| `--hf-secret-name NAME` / `--ngc-secret-name NAME` | Existing plaintext secrets; default `vla-pipeline/hf-token` / `vla-pipeline/ngc-token`. |
| `--input-s3-arn ARN` | Additional managed input bucket/object ARN; repeat for the input locations that need access. |
| `--existing-ecr-repo NAME` | For a new project, reference a supplied `vla/gr00t`, `vla/openvla`, `vla/molmoact2` or `vla/isaac-arena` repository. Repeat for each; its lifecycle stays with its existing owner. |
| `--plan` | Save/display preparation plans without applying infrastructure, building images or changing the host. |
| `--yes` | Approve the displayed plans noninteractively. Without it, applicable phases request confirmation. |
| `--resume` | Continue the recorded preparation using its saved choices and source. Do not supply new cells, images, host settings or provisioning choices. |

`--use-existing` cannot be combined with provisioning/confirmation flags such as
`--prepare-host`, `--plan`, `--resume` or `--yes`. Use a separate host-preparation
command after selecting existing infrastructure. Keep the committed Terraform
provider lock; do not upgrade it simply to follow the walkthrough.

#### Run arguments

The full graph is the default: four local steps, five managed steps.
Choose a cell, mode, budgets and managed GPU instances explicitly.
The shared configuration can supply the cell, deployment, hardware and runtime;
`--sample` can supply the short-run budgets.

| Argument | Meaning |
| --- | --- |
| `--deployment NAME` | Saved deployment, including profile, resources, images and optional host. Defaults to `arena.deployment_name`. |
| `--mode local` / `--mode managed` | Required executor choice. Local uses the prepared host; managed allocates SageMaker jobs. |
| `--sample` | Real 200-step training, three trials and threshold zero for selected steps. Explicit budgets win; an explicit checkpoint still skips training. |
| `--record-video` / `--no-record-video` | Enable/disable Arena rollout recording. Config `arena.run.record_video` accepts `true`, `false` or `"auto"` (Arena only). Without a preference, recording is off. Unsupported cells reject an explicit recording request. |
| `--cell NAME` | One supported cell from the table above or `pai arena cells`. |
| `--model NAME`, `--model-version n16\|n17`, `--simulator isaac_arena\|libero`, `--suite NAME` | Alternative to `--cell`: provide model, simulator and suite, plus version for GR00T. Must match an exposed cell; do not combine with `--cell`. |
| `--instance STEP=TYPE` | Managed GPU choice, e.g. `FineTune=ml.g6e.8xlarge`. Repeat for SimEval; omitted choices come from `arena.managed`. Rejected in local mode. |
| `--volume-gb STEP=GB` | Optional per-GPU-step requested volume override; otherwise uses the cell declaration. Does not resize local disks or enlarge fixed instance storage. |
| `--image STEP=ECR_URI` | Optional per-GPU-step published-image override; otherwise uses the deployment's saved digest. No build occurs. |
| `--train-steps N` | Positive training budget when FineTune runs; `--sample` defaults to 200. |
| `--eval-trials N` | Positive episode count when SimEval runs; `--sample` defaults to three. **Total for Arena, per task for LIBERO** (ten spatial tasks). OpenVLA allows at most 50 per task. |
| `--eval-seed N` | Evaluation seed; defaults to 100 for Arena and 1000 for LIBERO. |
| `--threshold RATE` | Finite rate from 0 to 1 when SuccessGate runs; `--sample` defaults to zero. Zero is a workflow check, not a model-quality bar. |
| `--max-runtime-seconds N` | Positive deadline for each selected GPU job; can come from `arena.run`. Separate from the watcher allowance. |
| `--save-steps N` | Training checkpoint interval; default 1,000,000. N1.6 requires it to be at least the training budget (final checkpoint only). |
| `--checkpoint-s3 URI` | Use an existing versioned `s3://bucket/key` checkpoint and omit FineTune. |
| `--through STEP` | Stop after FineTune, SimEval, Validate, SuccessGate or RegisterModel, including preceding dependencies. RegisterModel requires managed mode. |
| `--run-id ID` | Optional explicit run identifier; existing IDs are rejected to prevent duplicate submission. |
| `--group NAME` | Label independent runs for `pai arena status --group NAME`. Does not launch a campaign itself. |
| `--dry-run` | Resolve the request and perform AWS prerequisite reads without submitting work or making AWS writes. |
| `--offline` | Only with `--dry-run`: use saved selection without AWS reads. Does not prove prerequisites are ready. |
| `--wait` | Follow after submitting. Ctrl-C stops watching without cancelling. |
| `--watch-timeout-seconds N` | Positive following allowance, default 86,400; expiry does not cancel jobs. |
| `--image-pull-timeout-seconds N` | Local only: positive per-image pull allowance, default 7,200. |
| `--container-preparation-seconds N` | Local only: positive Compose creation allowance, default 600. The separate CPU probe retains 120 seconds. |
| `--local-timeout-seconds N` | Local only: overall preparation/execution deadline. Default is selected GPU-job budgets + three image-pull allowances + Compose allowance + 3,600 seconds. Must be positive and below 1,000,000 seconds. |

For example, `--cell gr00t-n16-arena` is equivalent to
`--model gr00t --model-version n16 --simulator isaac_arena --suite arena_gr1_fridge`.
Each `STEP=VALUE` option accepts only `FineTune` and `SimEval`, once each;
assignments to omitted steps are rejected.

#### Status and cancellation arguments

| Argument | Meaning |
| --- | --- |
| `status ID` | Inspect a saved deployment or run; without ID, list saved operations and last observed states. |
| `status ID --follow` | Follow that same operation; it never resubmits. |
| `status --group NAME` | List runs with this label; do not combine with an ID. |
| `status ID --follow --watch-timeout-seconds N` | Change the positive following allowance; default 86,400 seconds. |
| `stop RUN_ID` | Request cancellation of that run. Does not stop the EC2 host or erase evidence. |
| `cleanup RUN_ID` | Show what cleanup removes and retains, without changing resources. |
| `cleanup RUN_ID --yes` | Apply the saved run's cleanup. Active work is stopped first; successful local containers are removed only after evidence archive/readback. |
| `cleanup RUN_ID --yes --timeout-seconds N` | Local/managed cleanup allowance, default 3,600 seconds. Remote SSM cleanup has its own 3,600-second worker allowance. |
| `report RUN_ID --output-dir DIRECTORY` | Export the exact run's verified results. The directory cannot belong to another run. |
| `report RUN_ID --include-video` | Include recordings from the run's versioned SimEval artifact; unsupported simulators report that recording is unavailable. |
| `destroy DEPLOYMENT` | Show owned resources and runs without deleting them. |
| `destroy DEPLOYMENT --yes` | Archive evidence, remove owned resources, and independently check removal. The AWS archive remains unless explicitly removed. |
| `destroy DEPLOYMENT --yes --remove-evidence` | First verify a complete local archive, then remove the temporary AWS archive. Repeating the same command resumes incomplete teardown. Local results remain. |

**That completes the setup and command guide.** The rest of this README is
supplementary reference; you do not need to read it sequentially to run a sample.

## 2. Supplementary information

Read these sections as needed. They explain saved configuration, execution
behavior and the individual preparation steps behind the commands above.
Manual deployment procedures are alternatives to `vla deploy`; use the existing
state when resources are already owned by a CLI deployment.

**Contents**

- Deployment and execution details
  - [Preparation, state and resuming](#preparation-state-and-resuming)
  - [Reuse an existing deployment or host](#reuse-an-existing-deployment-or-host)
  - [Resource checks and deadlines](#resource-checks-and-deadlines)
  - [Step selection and checkpoint inputs](#step-selection-and-checkpoint-inputs)
  - [Status, verification and cancellation](#status-verification-and-cancellation)
- Manual walkthroughs
  - [Choose where to run](#choose-where-to-run)
  - [Deploy infrastructure](#deploy)
  - [Build the images](#build-the-container-images)
  - [Prepare and run on EC2](#run-locally--ec2-gpu-debugging)
  - [Run on managed SageMaker](#run-in-cloud--managed-sagemaker)
  - [Optional workflows](#optional-workflows-and-reference)
- Resources and implementation
  - [GPU, runtime, storage and cost planning](#gpu-runtime-and-cost-planning)
  - [What code runs where](#what-code-runs-where)
  - [Arena images and connector contract](#arena-images-and-connector-contract)
  - [GR00T AV1 decoding](#gr00t-av1-decoding)
  - [Validation runtime and training lineage](#validation-runtime-and-training-lineage)
  - [Model compatibility](#model-and-embodiment-compatibility)
  - [Repository layout](#layout)
  - [Arena internals](#reading-the-arena-internals-without-pulling-the-image)
- Development and maintenance
  - [Tests and development workflow](#tests)
  - [Notes and limitations](#notes-and-limitations)
  - [Resource ownership and permanent teardown](#resource-ownership-and-permanent-teardown)
  - [License](#license)

### Preparation, state and resuming

Fresh preparation checks token access, applies the existing Foundation and Arena
Terraform modules, builds the selected images with CodeBuild, then optionally
prepares the host. The Arena base precedes its connector. Build IDs, image
digests, logs and completed phases are saved as they become available.
If another deployment owns the fixed `vla/*` image repositories, create a
separately named project and add `--existing-ecr-repo REPOSITORY` for each
existing repository. Accepted names are `vla/gr00t`, `vla/openvla`,
`vla/molmoact2` and `vla/isaac-arena`. The CLI checks those repositories and
references them through Terraform data sources; it does not import them or
change their lifecycle policies. The new project's roles, buckets, network and
Foundation image repositories remain separately owned. Unlisted collisions
still stop deployment. The builder can publish new image tags to the selected
existing repositories; supply published `--image` choices to reuse images.
Resume uses the saved ownership choices; do not pass new repository selections.

In a fresh account, `--plan` stops at the Foundation dependency boundary if
the component cannot yet be planned; `--resume --yes` continues from there.

The coordinator runs in your terminal. Ctrl-C leaves submitted CodeBuild/SSM
work running and gracefully interrupts Terraform to preserve its state.
`status --follow` watches only; `deploy --resume` advances subsequent phases.
Resume reuses successful builds, reconnects to running work and records a new
attempt for failed builds. Ambiguous submissions stop with identifiers for
investigation. Refresh expired AWS credentials under the same profile name.

If an SSM submission has no recorded command ID, inspect **Systems Manager →
Run Command → Command history** in the **host region**, using the reported host,
comment and submission time. Follow/resume the saved operation again once the
command appears; it reconnects without resending. A missing result or truncated
history scan does not prove nothing was submitted. Do not edit the record or
start a second preparation/run to bypass this check. Resolve the command's
outcome before deliberately starting a replacement operation.

Records live outside the clone under `~/.local/state/vla`; Terraform state,
configuration and plans live at `deployments/NAME/work/` within it. Preserve
them after deleting a checkout. A selection import transfers resource/image
references, **not Terraform ownership**. Do not apply a second Terraform state
against already-owned resources; the CLI does not import or force-unlock them.

Host preparation preserves an attached instance profile, adds scoped runtime
and SSM access, installs dependencies, delivers the committed source and pulls
images. It can create/attach a runtime role if none exists. Active containers
or GPU work stop preparation. It does not prune, format or resize disks.
A later source revision needs a new preparation name and checkout; never
replace source used by an active run.

### Reuse an existing deployment or host

Use this when infrastructure and images are already prepared—for example, when
moving to another laptop or using a host prepared by a colleague. A deployment
selection is a JSON file recording resource names, image digests and the host
location. It contains no AWS credentials or token values. Importing it lets the
CLI reuse that setup.

Host preparation reports the file's S3 location as `selection_uri` and its
version as `selection_version`. Download the file below. `local-dev/` is a local
working folder excluded from Git, so machine-specific settings are not committed
with the application; it contains no additional application code required to run:

```bash
mkdir -p local-dev
aws s3api get-object --profile "$TARGET_PROFILE" --region us-east-1 \
  --bucket YOUR_DEV_BUCKET --key REPORTED_SELECTION_KEY \
  --version-id REPORTED_SELECTION_VERSION local-dev/deployment-selection.json
vla deploy --name arena-review --use-existing --profile "$TARGET_PROFILE" \
  --region us-east-1 --selection local-dev/deployment-selection.json
```

Use the reported `selection_version` in the command above so the imported bytes
match the prepared selection. The imported selection identifies the prepared
checkout and host state; import does not reinstall the host or start a run.
For direct execution on EC2, omit `--profile` and use its instance role.

Without an exported selection, first
[select the published image digests](#select-published-image-digests), then:

```bash
vla deploy --name arena-review --use-existing \
  --profile "$TARGET_PROFILE" --region us-east-1 --cell gr00t-n16-arena \
  --image FineTune="$LOCAL_TRAIN_IMAGE" --image SimEval="$LOCAL_ARENA_EVAL_IMAGE"
```

The two variables are full ECR URIs from that image-selection step. To add
local capability using those existing images:

```bash
vla deploy --name arena-review --prepare-host \
  --local-host INSTANCE_ID --host-region HOST_REGION \
  --scratch-root /opt/dlami/nvme/vla-tests --yes
```

Host setup briefly retries the initial source download while its new role
permission propagates. Persistent denials retain the original S3 error; inspect
the selected runtime role before `deploy --resume`. Download integrity checks
remain required.

For a manually prepared EC2 host, restore the
[host selection](#clone-and-install-on-ec2) and register it from that host:

```bash
vla deploy --name arena-review --use-existing \
  --region us-east-1 --cell gr00t-n16-arena \
  --image FineTune="$LOCAL_TRAIN_IMAGE" --image SimEval="$LOCAL_ARENA_EVAL_IMAGE" \
  --development-bucket "$ARENA_DEV_BUCKET" --expected-role "$ARENA_EC2_ROLE" \
  --local-host "$INSTANCE_ID" --host-region "$EC2_HOST_REGION" \
  --scratch-root "$VLA_SCRATCH_ROOT"
```

Selections contain names and identifiers, not credentials. The CLI checks
account, image and host identity; containers use the host's role, never copied
provisioning credentials. A remotely submitted run requires the same source
commit as its prepared host checkout. Keep imported files outside tracked
source or in the `local-dev/` working folder so the clean-source check passes.

### Resource checks and deadlines

Hardware choices are checked against the installed SageMaker API and, where
readable, EC2 GPU/storage metadata. Those checks do not reserve capacity or
establish model compatibility. A choice that differs from the cell's declared
instance prints a notice and remains allowed; declarations are not measured
minimums. See the [instance matrix](#cell-instance-and-storage-matrix).
The local worker checks [both filesystem reserves](#local-storage-and-retention)
after pulls and before training. A volume flag is not a disk-space fix.

Local per-job deadlines include synchronous SDK setup and artifact handling;
the total deadline also covers preparation and validation. Watching, image
pulling, container creation and GPU execution have different allowances.
Increasing a watcher timeout does not extend a job's deadline.

<a id="two-pipeline-entry-modes"></a>

### Step selection and checkpoint inputs

| Selection | Steps run |
| --- | --- |
| Neither `--through` nor `--checkpoint-s3` | FineTune → SimEval → Validate → SuccessGate; managed also registers. |
| `--through FineTune` | FineTune only. |
| `--through SimEval` | FineTune → SimEval. |
| `--checkpoint-s3 URI --through SimEval` | SimEval only, using the supplied checkpoint. |
| `--checkpoint-s3 URI` | SimEval → Validate → SuccessGate; managed also registers. |
| `--through Validate` | Stop after validation/publication, with or without preceding training. |
| `--through SuccessGate` | Stop after the threshold decision; no registration. |
| `--through RegisterModel` | Complete the managed graph, with or without preceding training. |

Only supply budgets and instances for selected steps. Training-only requests
omit evaluation arguments and threshold. Checkpoint inputs omit all training
arguments. Threshold is required only when SuccessGate runs.

The checkpoint must be a real object in versioned S3 with the format and
metadata expected by the selected cell. Use the matching model version; arbitrary
raw weights are not automatically compatible. The managed workload and validation
roles need read access to it; a caller's successful HEAD alone does not grant
those roles access. Re-evaluation checks integrity of the supplied bytes, not
independent proof of earlier training. A registry approver still reviews lineage.

### Status, verification and cancellation

For SSM-submitted work, use the saved record on the provisioning machine.
For direct EC2 submission, run status/stop on that host. Reuse the original
state directory after reconnecting.

Several status watchers can follow the same run. Updates use the latest saved
record under a lock; a busy observer displays the last saved state until the
coordinator finishes. `transport_activity` describes SSM communication separately
from the last observed workload activity. A failed read-only status command is
preserved in the record's history; the next status invocation requests a new
observation. Reconnecting to an interrupted launch follows its original command.
Ctrl-C stops the watcher, not the GPU job.

Local following invokes the independent verifier after worker completion:
requested steps, container exits, the applicable negative control and outputs.
Keep the containers until verification and [archiving](#archive-an-accepted-local-run)
finish. Managed following checks requested steps, job settings, exact S3 evaluation
versions, receipt/attestation hashes and registry linkage when selected.
Validate remains responsible for recomputing checkpoint content integrity.
The managed verifier also compares the complete receipt with its attestation,
the source/promoted artifact identities, actual job environments and the
registered image with the request. Old cached proofs are recomputed when the
verifier's evidence format changes. This reads existing results; it never starts
new training or changes a registered package.

Routine SageMaker SDK configuration messages are diagnostics, not execution
failures. With `--json`, diagnostics and watch/recovery instructions go to stderr;
stdout contains only result records. While running, managed GPU steps show the
job's observed secondary status (for example `Pending` or `Downloading`).
`Pending` alone does not establish why allocation is waiting. Once finished, the
summary distinguishes completed steps, independent verification and robot success.
An unreadable local operation record is listed with its path and error alongside
healthy records; preserve it for diagnosis.

Partial runs report their actual scope. Training-only verification confirms
completion and a nonempty versioned checkpoint, not its contents. Simulation-only
verification additionally reads/checks the evaluation archive, without performing
Validate's digest/publication checks. A supplied-checkpoint run does not claim
to verify earlier training lineage. Submission, workflow verification and robot
task success are separate outcomes.

One `run` command submits one execution. Repeat with distinct IDs and explicit
instances for a managed capacity campaign; `--group` only labels those runs.
FineTune and SimEval both consume **Training-job GPU quota**. Other executions
can occupy the next step's slot, so account for overlapping jobs across the
campaign rather than assuming one free training type is enough.

The legacy `python -m vla_pipeline.cli definition [-o FILE]` export remains
available for inspecting a pipeline definition using AWS configuration discovery.

<a id="walkthrough-and-setup-reference"></a>

### Choose where to run

| | Managed SageMaker in the cloud | SageMaker local mode on EC2 |
| --- | --- | --- |
| Start here | [Cloud instructions](#run-in-cloud--managed-sagemaker) | [EC2 instructions below](#run-locally--ec2-gpu-debugging) |
| Entry point | `vla run --mode managed` (legacy: `scripts/run_arena.py`) | `vla run --mode local` from the provisioning machine after host preparation, or directly on EC2 |
| Compute | Managed training jobs for FineTune/SimEval; CPU Processing for Validate | Docker containers on your EC2 GPU host; no managed job allocation |
| Accepted graph | FineTune → SimEval → Validate → SuccessGate → RegisterModel | FineTune → SimEval → Validate → SuccessGate |
| Storage and identity | Deployed models/handoff/trust buckets; separate step roles | Dedicated versioned dev bucket; checked EC2 role on host and containers |
| Result | Registry package in PendingManualApproval | Development receipt/attestation and local execution evidence |
| Inspect | Execution ARN, SageMaker APIs, CloudWatch and registered artifacts | `local-dev/runs/<run-id>/`, systemd and Docker; local IDs are not cloud ARNs |

The small N1.6/Arena sample uses **200 training steps and three episodes**.
Managed acceptance requires all five steps including registration. Local
acceptance requires all four local steps and verified development artifacts.
Optional LIBERO compatibility checks are not prerequisites for either route.

Both use S3 checkpoint handoffs and ECR images. Local mode needs network access
to AWS and Hugging Face; EC2/storage charges still apply. The new `vla run`
selects the executor with `--mode`. The legacy `scripts/run_arena.py` remains
managed-only; the detailed local walkthrough below also retains its original
runner commands for direct debugging.

For training diagnostics, run full training/evaluation **locally** and inspect
the loss history, completed steps, throughput and measured task successes.
Use a **small managed end-to-end sample** to verify registration; a full managed
training run is not required for registration acceptance.

#### Discovery used by the manual commands

The commands below use the target AWS profile and the shell selection saved in
[Select the deployment and check identity](#select-the-deployment-and-check-identity).
The CLI saves the equivalent choices in its own deployment record; the manual
shell selection is only needed when following these lower-level commands.

SSM discovery is mandatory. Set `VLA_FOUNDATION_PROJECT` to the same `project_name`
used for Foundation and component Terraform (default: `physical-ai`). All launchers
and builders use that namespace. `load_config()` reads the Foundation's
`/<project>/sagemaker-role-arn` and `/<project>/models-bucket`, plus the component's
`/<project>/isaac-lab-arena/hf-secret-name`. The latter contains the configured
Secrets Manager name, never the token value; deploy component Terraform before
using the updated scripts. Both FineTune and SimEval receive this reference.
Arena builds and launches also read `/<project>/ecr/isaac-lab-arena`.
Missing or inaccessible parameters fail without trying another namespace.
`VLA_USE_SSM`, `VLA_ROLE_ARN`, `VLA_BUCKET`, and a local pipeline YAML are not used.
Select `us-east-1` through the AWS profile or `VLA_REGION`. Builders' `--project-name`
selects the CodeBuild project, not the Foundation namespace.

### Deploy

A **new AWS account starts without application tokens, images or discovery parameters**.
Select the target identity, provision and check both tokens, deploy Foundation
and this component, then build the full images using the next section.
Nothing is automatically copied from another account.
For an existing deployment, reuse its Terraform state and selected image releases.

Use an administrator/provisioning identity for the **target account** here.
The EC2 testing role is a separate identity, configured later.

#### Select the deployment and check identity

Choose these values once. **Every later section uses this same selection.**
Save the selection below and restore it in subsequent shells. It contains no
authentication profile or secret value. On a fresh EC2 host, first use the
minimal [before-clone host values](#initialize-the-new-ubuntu-host); restore the
full selection **after cloning**. Keep the printed selection with your evidence.
Use a distinct project for an isolated deployment: SSM discovery is scoped by
project, not environment. Changing only `VLA_ENVIRONMENT` does not isolate SSM.

On the provisioning machine first select the target profile:

```bash
unset AWS_DEFAULT_PROFILE AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY \
  AWS_SESSION_TOKEN AWS_SECURITY_TOKEN
export AWS_PROFILE=your-target-account-profile
```

Then set the deployment selection, from this component directory:

```bash
set -euo pipefail
export AWS_DEFAULT_REGION=us-east-1
export AWS_REGION="$AWS_DEFAULT_REGION"
export VLA_REGION="$AWS_DEFAULT_REGION"
export TARGET_ACCOUNT_ID=your-target-account-id
export VLA_FOUNDATION_PROJECT=physical-ai  # choose a distinct project if needed
export VLA_ENVIRONMENT=dev
export VLA_PIPELINE_NAME="$VLA_FOUNDATION_PROJECT-$VLA_ENVIRONMENT-arena"
export HF_SECRET_NAME=vla-pipeline/hf-token
export NGC_SECRET_NAME=vla-pipeline/ngc-token
export CODEBUILD_PROJECT_NAME="$VLA_FOUNDATION_PROJECT-$VLA_ENVIRONMENT-vla-image-build"
export ARENA_DEV_BUCKET="$VLA_FOUNDATION_PROJECT-$VLA_ENVIRONMENT-localdev-$TARGET_ACCOUNT_ID"
export ARENA_EC2_ROLE="$VLA_FOUNDATION_PROJECT-$VLA_ENVIRONMENT-local-debug"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
if [ "$ACCOUNT_ID" != "$TARGET_ACCOUNT_ID" ]; then
  echo "Selected AWS identity is not the intended deployment account." >&2
  exit 1
fi
mkdir -p local-dev
declare -p TARGET_ACCOUNT_ID AWS_DEFAULT_REGION AWS_REGION VLA_REGION \
  VLA_FOUNDATION_PROJECT \
  VLA_ENVIRONMENT VLA_PIPELINE_NAME HF_SECRET_NAME NGC_SECRET_NAME \
  CODEBUILD_PROJECT_NAME ARENA_DEV_BUCKET ARENA_EC2_ROLE \
  > local-dev/deployment-selection.txt
cat local-dev/deployment-selection.txt
TF_ARGS=(-var="aws_region=$AWS_DEFAULT_REGION"
         -var="project_name=$VLA_FOUNDATION_PROJECT"
         -var="environment=$VLA_ENVIRONMENT")
```

The selection file contains names and IDs, never credentials or token values.
For a later shell in this same component directory, select the intended AWS
identity and restore the saved values rather than retyping them:

```bash
set -euo pipefail
source local-dev/deployment-selection.txt
test "$AWS_DEFAULT_REGION" = us-east-1
test "$AWS_REGION" = us-east-1
test "$VLA_REGION" = us-east-1
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
test "$ACCOUNT_ID" = "$TARGET_ACCOUNT_ID"
TF_ARGS=(-var="aws_region=$AWS_DEFAULT_REGION"
         -var="project_name=$VLA_FOUNDATION_PROJECT"
         -var="environment=$VLA_ENVIRONMENT")
```

Use the target provisioning profile on your laptop and the instance role on
EC2. Restoring this file does not switch AWS identities. To transfer the
selection to the new host, use the bundle/bootstrap instructions below.

Keep project/bucket/role names within AWS naming limits. The component's ECR
names (`vla/gr00t`, `vla/openvla`, `vla/molmoact2`, `vla/isaac-arena`) are fixed;
changing the project does not rename them. Foundation's ECR repository names
use the selected project prefix, such as `<project>/isaac-lab-arena`.
The Arena registry group is also manifest-selected (`vla-arena-gr00t`), independently
of `VLA_PIPELINE_NAME`. Inspect existing ownership before reusing any of these
names; changing the pipeline name alone does not isolate a registry group.

#### Provision and check both tokens before infrastructure

##### Prerequisite — HuggingFace token (required for every GR00T run)

GR00T fine-tuning pulls a **gated** base model from HuggingFace, so a token is mandatory:
without it, FineTune fails on the very first download. Set this up **once**, before running anything.
The main N1.6/Arena recipe uses `nvidia/GR00T-N1.6-3B` and dataset
`nvidia/Arena-GR1-Manipulation-PlaceItemCloseDoor-Task`, revision
`arena_v0.2_lab_v3.0`, subdirectory
`ranch_bottle_into_fridge/ranch_bottle_into_fridge_generated_100/lerobot`.
The named dataset revision is resolved and recorded by the run; it is not itself
an immutable commit. The selected token must have access to these resources.

1. **Get a token and accept the model license.** Create a *read* token at
   [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens), then open the gated
   GR00T base-model repo you are fine-tuning (e.g.
   [nvidia/GR00T-N1.6-3B](https://huggingface.co/nvidia/GR00T-N1.6-3B)) and click **Agree** to accept
   its license — the token only works after the license is accepted.

2. **Store it in the target AWS account** at the name the pipeline reads (default
   `vla-pipeline/hf-token`, in the pipeline's region). The secure-file commands
   in this section create both secrets before **Foundation** Terraform.
   To use a different secret name, set `HF_SECRET_NAME` in the deployment selection
   above. Component Terraform publishes it at
   `/<project>/isaac-lab-arena/hf-secret-name`; the launcher resolves that value.

3. **Grant read access.** Managed FineTune and SimEval use the component's training
   and workload roles; both need `secretsmanager:GetSecretValue` on that secret.
   Component Terraform supplies these grants. Local execution needs the same
   secret permission on the EC2 instance role.

Resolution order in the entry scripts is **`HF_TOKEN` env → the Secrets Manager secret above → fail**.
The detached local launcher retrieves the secret itself and injects the token
into both GPU containers. A host export is not forwarded by that launcher.
The token is never baked into code or an image.

##### NVIDIA NGC API key — required for fresh image builds

Sign in to [NVIDIA NGC](https://ngc.nvidia.com/), open
[Setup → API Keys](https://org.ngc.nvidia.com/setup/api-keys), and generate a
personal key with **NGC Catalog** access. Follow NVIDIA's
[NGC account and personal-key instructions](https://docs.nvidia.com/ngc/gpu-cloud/ngc-user-guide/index.html)
if account or organization setup is required before key creation. Ensure the
account can access the Isaac Sim container and accept the applicable NVIDIA
terms. Save the key securely when it is displayed.

Store the key as the plaintext value of `NGC_SECRET_NAME` (default
`vla-pipeline/ngc-token`) using the token setup below. CodeBuild uses it to
authenticate to `nvcr.io` when building the base image. An HF token cannot
replace it. Both tokens are needed for a fresh deployment, including a local
EC2 run that first builds its own images.

After obtaining both tokens, store them below. This requires no Foundation or
component resources.
Use the **target** identity and region selected in the preceding block.

If you already have the two target-owned secrets, select their names in the
deployment selection and skip creation. Do not overwrite an existing token
merely to follow this walkthrough. Otherwise, create each in the Secrets
Manager console using **Other type of secret → Plaintext**, or supply files
readable only by the provisioning user. Each file must contain just the token,
without a JSON wrapper:

```bash
# Replace these with your own secure files; these are not files in the repo.
chmod 600 /secure/path/hf-token /secure/path/ngc-token
aws secretsmanager create-secret --name "$HF_SECRET_NAME" \
  --secret-string file:///secure/path/hf-token
aws secretsmanager create-secret --name "$NGC_SECRET_NAME" \
  --secret-string file:///secure/path/ngc-token
```

Never put token values in chat, Git, shell history, deployment-selection files
or pipeline parameters. A secret name or ARN alone does not supply its value.
If a credential owner supplies secrets for an agent, agree on the secure
provisioning method **before** the agent starts infrastructure creation.

Run this check even when the secrets already exist. It reads their current
versions but prints **only ARNs and version IDs**, never token values:

```bash
check_token_secrets() (
  set -euo pipefail
  : "${TARGET_ACCOUNT_ID:?Repeat the deployment selection first}"
  : "${HF_SECRET_NAME:?Repeat the deployment selection first}"
  : "${NGC_SECRET_NAME:?Repeat the deployment selection first}"
  test "$(aws sts get-caller-identity --query Account --output text)" = "$TARGET_ACCOUNT_ID"
  token_check_failed=0
  for token_secret_name in "$HF_SECRET_NAME" "$NGC_SECRET_NAME"; do
    if ! aws secretsmanager get-secret-value \
      --region "$AWS_DEFAULT_REGION" \
      --secret-id "$token_secret_name" --version-stage AWSCURRENT \
      --query '{ARN:ARN,VersionId:VersionId}' --output json; then
      printf 'Required token unavailable: %s. Resolve this before Terraform apply.\n' \
        "$token_secret_name" >&2
      token_check_failed=1
    fi
  done
  test "$token_check_failed" -eq 0
)
check_token_secrets
```

Both reads must succeed. `ResourceNotFoundException` means the secret has not
been provisioned at the selected name/account/region. `AccessDeniedException`
requires the credential owner to grant the provisioning identity
`secretsmanager:GetSecretValue` and, for a customer-managed encryption key,
the needed KMS decrypt access. Do not continue to infrastructure creation
until both reads pass. These checks establish AWS storage/read access;
successful image builds and model downloads establish the tokens' provider
authorization. The credential owner must still accept the required terms.

Keep this function in the current Bash shell; repeat the selection and check
block after opening a new shell. The apply commands below invoke it again.
Component Terraform resolves the exact secret ARNs and grants the build and
workload roles access; it does not create or copy token values.

#### Check resource ownership and quotas

Before any apply, inspect target SSM parameters, buckets, ECR repositories,
CodeBuild projects and IAM roles for collisions with this selection. Existing
resources must keep their owning Terraform state. For example:

```bash
aws ssm describe-parameters \
  --parameter-filters "Key=Name,Option=BeginsWith,Values=/$VLA_FOUNDATION_PROJECT/"
aws ecr describe-repositories --query 'repositories[].repositoryName'
aws codebuild list-projects
aws iam list-roles --query 'Roles[].RoleName'
aws s3api list-buckets --query 'Buckets[].Name'
aws sagemaker list-pipelines --query 'PipelineSummaries[].PipelineName'
aws sagemaker list-model-package-groups \
  --query 'ModelPackageGroupSummaryList[].ModelPackageGroupName'
```

Review the intended quotas too: CodeBuild Linux `LARGE` and `2XLARGE` builds,
SageMaker **training** jobs for the instance types you pass to the CLI (the
common-task Arena example uses `ml.g6e.8xlarge` for both GPU steps), and
`ml.m5.large` Processing for Validate. The later manual Arena example uses
`ml.g6e.xlarge`; quota for that type does not cover `.8xlarge`.
The later reference EC2 host requires
32 On-Demand G-family vCPUs. Request any missing quota through Service Quotas
before allocating compute; quota alone does not establish available capacity.

```bash
aws service-quotas list-service-quotas --service-code codebuild \
  --query 'Quotas[].{Name:QuotaName,Code:QuotaCode,Value:Value}' --output table
aws service-quotas list-service-quotas --service-code sagemaker \
  --query "Quotas[?contains(QuotaName, 'ml.g6e.') || contains(QuotaName, 'ml.m5.large')].{Name:QuotaName,Code:QuotaCode,Value:Value}" \
  --output table
aws service-quotas list-service-quotas --service-code ec2 \
  --query "Quotas[?contains(QuotaName, 'On-Demand G')].{Name:QuotaName,Code:QuotaCode,Value:Value}" \
  --output table
```

#### Apply Foundation and component infrastructure

The Foundation apply below creates shared storage, IAM, SSM and ECR resources,
plus a **VPC, two public and three private subnets, an Internet gateway, and a
NAT gateway with an Elastic IP**. Include those network resources in the
deployment budget and retention inventory. Review the plan and preserve its
Terraform state; the later component apply consumes this Foundation.
Foundation also provisions shared Cosmos/Isaac Lab/GR00T roles and repositories
for the wider toolchain. They appear in the plan even for this Arena sample.
The first fresh-account walkthrough created 54 Foundation resources; review
your actual plan rather than treating that observation as a fixed count.

```bash
: "${TARGET_ACCOUNT_ID:?Repeat the deployment selection in this shell}"
: "${VLA_FOUNDATION_PROJECT:?Repeat the deployment selection in this shell}"
source .venv/bin/activate
test "$AWS_DEFAULT_REGION" = us-east-1
test "$VLA_REGION" = us-east-1
TF_ARGS=(-var="aws_region=us-east-1"
         -var="project_name=$VLA_FOUNDATION_PROJECT"
         -var="environment=$VLA_ENVIRONMENT")

terraform -chdir=../foundation/infra init
terraform -chdir=../foundation/infra plan "${TF_ARGS[@]}" \
  -out="$PWD/local-dev/foundation.tfplan"
terraform -chdir=../foundation/infra show -no-color "$PWD/local-dev/foundation.tfplan"
```

After reviewing that exact plan and its resource counts, apply the saved plan:

```bash
check_token_secrets || exit 1
terraform -chdir=../foundation/infra apply "$PWD/local-dev/foundation.tfplan"
```

With Foundation deployed and both tokens checked, plan the component:

```bash
terraform -chdir=infra init
terraform -chdir=infra plan "${TF_ARGS[@]}" \
  -var="codebuild_project_name=$CODEBUILD_PROJECT_NAME" \
  -var="hf_secret_name=$HF_SECRET_NAME" -var="ngc_secret_name=$NGC_SECRET_NAME" \
  -out="$PWD/local-dev/component.tfplan"
terraform -chdir=infra show -no-color "$PWD/local-dev/component.tfplan"
```

After reviewing the component plan and resource counts:

```bash
check_token_secrets || exit 1
terraform -chdir=infra apply "$PWD/local-dev/component.tfplan"
```

`TF_ARGS` supplies the same project, environment and region to both deployments.
Keep their state files and the selected variable values for future updates.

The component creates training/base ECR repositories, the image-build CodeBuild
project and role, **two S3 buckets**, and **three SageMaker execution roles**.
Foundation owns the Arena connector ECR repository and shared VPC. The component
creates no additional VPC.

The buckets and roles exist to separate identities that must not be able to
forge each other's evidence:

| Resource | Purpose |
|---|---|
| `*-trust-<account>` bucket | validator code, promoted artifacts, attestations |
| `*-handoff-<account>` bucket | the SimEval to Validate evidence handoff |
| `training` role | FineTune only: writes `train/`, never `eval/` |
| `workload` role | SimEval only: writes the raw evidence Validate judges |
| `validation` role | Validate only: publishes the gate's receipt |

The HF-secret grant is on the two worker roles, not the Foundation role. An
earlier design attached it to Foundation; that is no longer accurate.

Keep externally managed resources with their existing owner. Use
`existing_ecr_repos` for repositories owned by another deployment, and exclude
those names from `ecr_repos`. Set `codebuild_project_name` to an unused name when
the default builder already belongs to another stack, then pass that name with
`--project-name` to both image-build scripts. Preserve the deployed variable
values on subsequent plans. See [resource ownership and permanent teardown](#resource-ownership-and-permanent-teardown).
Build or verify all three required images below before launching a pipeline.

### Build the container images

**This is how images get into a fresh account's ECR.** Component Terraform creates
the repositories, CodeBuild project and its publishing role. The build scripts
upload source from this clone and start CodeBuild; CodeBuild builds and pushes
using that role's ECR layer-upload/`PutImage` permissions. No Mac publisher or
pre-existing `vla/gr00t:1.2` image is required.

The submitting identity needs SSM discovery, build-source upload/read under the
component trust bucket's `code/v1/*`, CodeBuild start/status and ECR inspection.
The EC2 runtime role only needs image-read access when using these published images.
Builds use CodeBuild compute, independently of SageMaker GPU capacity.
The connector includes both pinned **N1.6 and N1.7** server environments,
so building it for the N1.6 sample also downloads N1.7 dependencies. Neither
environment is installed afresh for every evaluation.

From this component directory, after both Terraform deployments and secret setup:

```bash
set -euo pipefail
: "${TARGET_ACCOUNT_ID:?Repeat the deployment selection in this shell}"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
test "$ACCOUNT_ID" = "$TARGET_ACCOUNT_ID"
ECR_REGISTRY="$ACCOUNT_ID.dkr.ecr.$AWS_DEFAULT_REGION.amazonaws.com"
BUILD_PROJECT=$(aws ssm get-parameter \
  --name "/$VLA_FOUNDATION_PROJECT/component/codebuild-project" \
  --query Parameter.Value --output text)
BUILD_TAG="dev-$(git rev-parse --short HEAD)-$(date -u +%Y%m%dT%H%M%SZ)"
TRAIN_BUILD_TAG="$BUILD_TAG"
ARENA_BASE_BUILD_TAG="$BUILD_TAG"
ARENA_BUILD_TAG="$BUILD_TAG"
declare -p BUILD_PROJECT ECR_REGISTRY TRAIN_BUILD_TAG ARENA_BASE_BUILD_TAG ARENA_BUILD_TAG \
  >> local-dev/deployment-selection.txt
mkdir -p local-dev/builds
PYTHONPATH=src python -u scripts/build_images.py --family gr00t --tag "$TRAIN_BUILD_TAG" \
  2>&1 | tee "local-dev/builds/train-$TRAIN_BUILD_TAG.log"
```

Require that command to succeed. It builds the **full** GR00T image from the AWS
Deep Learning Container base, including its environments and the AV1 decoder
check, then pushes `vla/gr00t:$TRAIN_BUILD_TAG`. For all three optional LIBERO families,
use `--family all` instead; each family is published with that same chosen tag.
The source archive includes the actual Dockerfile inputs. GR00T's tiny generated
AV1 fixture is ordinary Git content and works with the LFS-skipping clone above.

Arena additionally needs its base and connector. Build the base first:

```bash
BASE_BUILD_ID=$(aws codebuild start-build --project-name "$BUILD_PROJECT" \
  --source-type-override NO_SOURCE \
  --compute-type-override BUILD_GENERAL1_2XLARGE \
  --timeout-in-minutes-override 180 \
  --buildspec-override "$(cat entrypoints/eval/isaac_arena/base/buildspec_base.yml)" \
  --environment-variables-override \
    name=ECR_REGISTRY,value="$ECR_REGISTRY" \
    name=IMAGE_URI,value="$ECR_REGISTRY/vla/isaac-arena:$ARENA_BASE_BUILD_TAG" \
    name=NGC_SECRET_ID,value="$NGC_SECRET_NAME" \
  --query build.id --output text)
printf '%s\n' "$BASE_BUILD_ID" > "local-dev/builds/base-$ARENA_BASE_BUILD_TAG.id"
aws codebuild batch-get-builds --ids "$BASE_BUILD_ID" \
  --query 'builds[0].{status:buildStatus,phase:currentPhase}'
```

Repeat the status query until terminal. Require **`SUCCEEDED`**, not merely a
returned ID, before building the connector:

```bash
PYTHONPATH=src python -u scripts/build_arena_connector.py --connector gr00t \
  --base-image "vla/isaac-arena:$ARENA_BASE_BUILD_TAG" --tag "$ARENA_BUILD_TAG" --wait \
  2>&1 | tee "local-dev/builds/connector-$ARENA_BUILD_TAG.log"
```

#### Select published image digests

Use this block after the builds. In a new shell, select your target AWS profile,
activate the clone's environment and restore `local-dev/deployment-selection.txt`.
It now includes your build tags. They start equal but can differ after a retry.
Preserve the base tag used by the connector build too.
These are read-only ECR/SSM queries and work under the checked EC2 runtime role too.

```bash
set -euo pipefail
: "${TARGET_ACCOUNT_ID:?Repeat the deployment selection in this shell}"
: "${VLA_FOUNDATION_PROJECT:?Repeat the deployment selection in this shell}"
: "${TRAIN_BUILD_TAG:?Set the tag from the successful GR00T build}"
: "${ARENA_BUILD_TAG:?Set the tag from the successful Arena connector build}"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
test "$ACCOUNT_ID" = "$TARGET_ACCOUNT_ID"
ECR_REGISTRY="$ACCOUNT_ID.dkr.ecr.$AWS_DEFAULT_REGION.amazonaws.com"
ARENA_REPOSITORY=$(aws ssm get-parameter \
  --name "/$VLA_FOUNDATION_PROJECT/ecr/isaac-lab-arena" \
  --query Parameter.Value --output text)
case "$ARENA_REPOSITORY" in
  "$ECR_REGISTRY/"*) ;;
  *) echo "Arena repository is outside the selected account/region." >&2; exit 1 ;;
esac
ARENA_REPO_NAME="${ARENA_REPOSITORY#"$ECR_REGISTRY/"}"
TRAIN_DIGEST=$(aws ecr describe-images --repository-name vla/gr00t \
  --image-ids imageTag="$TRAIN_BUILD_TAG" --query 'imageDetails[0].imageDigest' --output text)
ARENA_DIGEST=$(aws ecr describe-images --repository-name "$ARENA_REPO_NAME" \
  --image-ids imageTag="$ARENA_BUILD_TAG" --query 'imageDetails[0].imageDigest' --output text)
if ! [[ "$TRAIN_DIGEST" =~ ^sha256:[0-9a-f]{64}$ ]] \
    || ! [[ "$ARENA_DIGEST" =~ ^sha256:[0-9a-f]{64}$ ]]; then
  echo "ECR did not return both required image digests." >&2
  exit 1
fi
# Managed launchers take repository-relative references.
TRAIN_IMAGE="vla/gr00t@$TRAIN_DIGEST"
ARENA_EVAL_IMAGE="$ARENA_REPO_NAME@$ARENA_DIGEST"
# The local runner takes full ECR URIs.
LOCAL_TRAIN_IMAGE="$ECR_REGISTRY/$TRAIN_IMAGE"
LOCAL_ARENA_EVAL_IMAGE="$ECR_REGISTRY/$ARENA_EVAL_IMAGE"
printf 'Train tag: %s\nArena tag: %s\nTrain: %s\nArena eval: %s\n' \
  "$TRAIN_BUILD_TAG" "$ARENA_BUILD_TAG" "$LOCAL_TRAIN_IMAGE" "$LOCAL_ARENA_EVAL_IMAGE"
declare -p TRAIN_BUILD_TAG ARENA_BASE_BUILD_TAG ARENA_BUILD_TAG \
  TRAIN_IMAGE ARENA_EVAL_IMAGE LOCAL_TRAIN_IMAGE LOCAL_ARENA_EVAL_IMAGE \
  >> local-dev/deployment-selection.txt
```

These references are saved with the deployment selection for transfer to EC2.
Appending a later selection preserves the history; Bash restores the last value
for each name. The file holds no AWS credentials or token values. Restore it
when reconnecting, rather than rebuilding or selecting a historical example tag.
ECR tags are immutable: use a fresh tag for a rebuild. Build only GR00T plus the
Arena images for the main path; the other families are needed only for LIBERO.

| Image | Source recipe | Published repository |
| --- | --- | --- |
| GR00T / OpenVLA / MolmoAct2 | `docker/<family>/Dockerfile` with its family directory as context | `vla/<family>` |
| Arena base | `entrypoints/eval/isaac_arena/base/buildspec_base.yml` | `vla/isaac-arena` |
| Arena connector | Root `containers/isaac-lab-arena/` plus component source packaged by its builder | Foundation's Arena ECR URI from SSM |
| Validate | Pipeline-resolved AWS sklearn image plus the packaged SDK in staged source | No separate application image build |

The family buildspec authenticates to the same `DLC_REGISTRY` passed to Docker.
The supplied default is AWS's DLC registry in us-east-1. If deliberately overriding
it in CodeBuild, the build role also needs `ecr:BatchGetImage`,
`ecr:GetDownloadUrlForLayer` and `ecr:BatchCheckLayerAvailability` on the exact
replacement base-image repository, plus access allowed by its owner. Changing
the registry variable alone does not change IAM permissions.

The small incremental AV1 repair used during development is not a substitute
for these full builds in a fresh account.

The 2026-09-17 separate-account us-east-1 review built all three images in
**92m35s**, including publication. The resulting GR00T training image and Arena
connector completed the N1.6/GR1 fridge sample together. Select the digests from
**your** builds; the sample does not supply a public, prebuilt image release.
Keep successful images when changing only host tooling, training entry source or
Validate source. Rebuild the affected image when its Dockerfile, dependencies or
baked evaluator code changes. Export and ECR push can take minutes with quiet
intervals after the Docker build steps finish.

Budget for three sequential builds: GR00T uses the project's 60-minute limit on
`LARGE`; the Arena base requests 180 minutes on `2XLARGE`, and the connector
requests 120 minutes on `2XLARGE`. These are timeouts. A timeout requires
diagnosis and a new build; it is not a successful
publication.

#### Inspect, cancel or retry an image build

The commands above save build IDs as soon as AWS returns them: training and
connector IDs are in their unbuffered submission logs; the base ID is in its
`.id` file. Disconnecting a launcher does not stop a submitted CodeBuild job.
On the provisioning machine, restore the selection and inspect that same build:

```bash
source local-dev/deployment-selection.txt
# Select the build you were watching; this example resumes the connector.
BUILD_ID=$(sed -n 's/^Build started: //p' "local-dev/builds/connector-$ARENA_BUILD_TAG.log" | tail -1)
: "${BUILD_ID:?No saved build ID; inspect the submission log before retrying}"
aws codebuild batch-get-builds --ids "$BUILD_ID" \
  --query 'builds[0].{Status:buildStatus,Phase:currentPhase,Phases:phases,Logs:logs,Source:source,Environment:environment.computeType}'
```

The returned `logs.deepLink` opens its CloudWatch stream. With CLI log access,
set `LOG_GROUP` and `LOG_STREAM` to that response's `logs.groupName` and
`logs.streamName`, then run
`aws logs get-log-events --log-group-name "$LOG_GROUP" --log-stream-name "$LOG_STREAM" --limit 100 --no-paginate`.
Wait until a stream exists. A build timeout or missing dependency is a failed
build even if an earlier stage pushed an image.
For training, extract its ID with
`sed -n 's/^  Build started: //p' "local-dev/builds/train-$TRAIN_BUILD_TAG.log"`;
for the base, read `local-dev/builds/base-$ARENA_BASE_BUILD_TAG.id`.
If submission failed before an ID was saved, inspect CodeBuild's recent builds
and source/tag before submitting again; a missing local ID does not prove AWS
rejected the request.

If the log reports an HTTP 500 or timeout downloading a pinned dependency
(for example `flash-attn` from GitHub), retain that failed build ID and use
the connector-only retry below after the dependency host recovers. The
connector allows up to ten HTTP retries with a 120-second read timeout.
It still uses the pinned lockfile and fails if the download cannot complete.
Do not remove the dependency or change its version to bypass a download error.

To cancel that build, use `aws codebuild stop-build --id "$BUILD_ID"` and query
until terminal. Stopping the Python launcher alone does not cancel CodeBuild.
Preserve the ID, logs and source hash; fix the cause and use a **new tag** for
the retry. An already successful base can be reused by explicitly passing its
recorded `vla/isaac-arena:<base-tag>` to the connector builder's `--base-image`.
For example, if only the connector failed, leave `TRAIN_BUILD_TAG` and
`ARENA_BASE_BUILD_TAG` unchanged, set a new `ARENA_BUILD_TAG`, and rerun the
connector command. Select the resulting training and connector tags separately;
do not rebuild a successful training image just to make their tags match.
Save the new tag to the selection file before retrying:
`declare -p ARENA_BUILD_TAG >> local-dev/deployment-selection.txt`.
Changes to the baked Arena evaluator/helpers require a connector rebuild.
Changes to training or Validate source are staged by the next pipeline launch.

#### Rebuild only the connector after a code update

On the **provisioning machine**, update to the reviewed source commit, activate
its environment and restore your existing deployment selection. Reuse the
successful training and base images. This does not require Terraform:

```bash
set -euo pipefail
source .venv/bin/activate
source local-dev/deployment-selection.txt
test "$(aws sts get-caller-identity --query Account --output text)" = "$TARGET_ACCOUNT_ID"
: "${TRAIN_BUILD_TAG:?Recover the successful training tag from your build record}"
: "${ARENA_BASE_BUILD_TAG:?Recover the successful base tag from your build record}"
ARENA_BUILD_TAG="dev-$(git rev-parse --short HEAD)-$(date -u +%Y%m%dT%H%M%SZ)"
declare -p ARENA_BUILD_TAG >> local-dev/deployment-selection.txt
mkdir -p local-dev/builds
PYTHONPATH=src python -u scripts/build_arena_connector.py --connector gr00t \
  --base-image "vla/isaac-arena:$ARENA_BASE_BUILD_TAG" --tag "$ARENA_BUILD_TAG" --wait \
  2>&1 | tee "local-dev/builds/connector-$ARENA_BUILD_TAG.log"
```

After `SUCCEEDED`, repeat [image digest selection](#select-published-image-digests)
and [transfer the selection to EC2](#connect-to-the-host). Update the host checkout
to the same reviewed commit using its Git remote or the
[verified bundle update](#obtain-the-source-and-install-the-application).
Then run a new preflight with `--pull-images` and a new local sample.
The previous connector build took about 34 minutes; downloading the changed
image layers and the roughly 28-minute sample are additional.

### Run locally — EC2 GPU debugging

This section contains the complete EC2 setup and run workflow. All executable
tooling is tracked under `scripts/local/`; only generated runs, private settings
and caches live under ignored `local-dev/`.

Local acceptance is FineTune → SimEval → Validate → SuccessGate. RegisterModel
is omitted; use the [managed sample](#run-a-sample-first) to check real registration. Both paths
use S3 and ECR. Local mode creates no managed training/processing jobs, but EC2
and storage charges still apply.

#### Prepare the EC2 host and AWS resources

Use an x86-64 Ubuntu GPU instance with a working NVIDIA driver, Docker Engine,
Docker Compose v2-compatible CLI (`docker compose`), NVIDIA Container Toolkit,
Python 3.10–3.12 and systemd. The reference machine is a `g6e.8xlarge` with one
L40S and Ubuntu/Python 3.10. A GPU-ready AWS Deep Learning AMI is a starting point;
verify the tools below rather than assuming every AMI contains them.
The pinned SageMaker SDK's Compose detector recognizes only v2. The local runner
supports Compose v2/v5 by first executing a real CPU Compose probe, then selecting
the tested `docker compose` command. It records the actual version, including
v5.5.1 on the reference box; it does not falsify the version string.

Provision enough disk for the training image, large Arena connector, model/data
downloads and temporary checkpoint copies. After images are present, the Arena
sample requires **90 GiB free on root and 250 GiB free on the separate scratch
filesystem** before training starts. Other cells have different
[planning reserves](#local-storage-and-retention). An 800 GiB root volume is the
reference size, not a guarantee of available space: images, stopped containers
and package caches can fill it. Free space on scratch does not compensate for a
full root disk.

**Disk throughput also affects startup.** In the supplied-host review, cold image
pulls saturated the root gp3 volume at 125 MiB/s. Docker then spent 5m33s creating
the small CPU probe container. There was ample free space. The launch recipe
below retains that baseline throughput and the runner allows ten minutes for
Compose preparation; this allowance still needs a cold-host verification.
`--scratch-root` moves application scratch and caches, **not Docker/containerd's
image stores**. Do not relocate those stores or prune images on a working host
just to retry preflight. Increasing EBS throughput is a separate measured tuning
option; it is not required by the successful warm-host run.

On the reference DLAMI, `/opt/dlami/nvme` has a separate mounted scratch filesystem.
Check `df -hT / /opt/dlami/nvme` and `lsblk` before using it. The inventory found
783 GiB free there, while root had 138 GiB free. Passing
`--scratch-root /opt/dlami/nvme/vla-tests` places staged source working directories,
container working directories, temporary files and selected data/model caches
on that disk. N1.6 installs its runtime and downloads beneath `/opt/ml/code`;
that source mount therefore also needs scratch space. The runner refuses a path
that is on root because the intended scratch disk is absent or unmounted.
It preserves the baked environments under `/opt/vla`; see
[storage and retention](#local-storage-and-retention).
Keep the instance running while relying on NVMe caches or unarchived evidence:
instance-store data is lost on stop/termination. Closing SSH or your terminal
does not stop the detached systemd pipeline.

The Foundation and component infrastructure must already be deployed; local mode
still calls `load_config()` through SSM. For a new AWS account, first follow
[fresh-account deployment and full image builds](#deploy). Use the image digests built or selected in that account, as shown below. Attach an EC2 instance profile with:

| Access | Purpose |
| --- | --- |
| `ssm:GetParameter` on the project's Foundation/component discovery names | Load roles, bucket names, prefix and HF-secret reference; `GetParametersByPath` alone is insufficient |
| ECR `GetAuthorizationToken`, `DescribeImages`, `BatchGetImage`, `GetDownloadUrlForLayer`, `BatchCheckLayerAvailability` | Resolve tags and pull the runtime images; an eval digest override must exist in this account/region |
| S3 `GetBucketVersioning`, `GetBucketLocation`, `ListBucket`, `ListBucketMultipartUploads` on the dedicated development bucket | Discover and check scratch storage |
| S3 `GetObject`, `GetObjectVersion`, `PutObject`, `AbortMultipartUpload`, `ListMultipartUploadParts` on its objects | Checkpoints, staged code, evaluation outputs and conditional publication |
| Secrets Manager `GetSecretValue` on the configured HF secret | The detached launcher retrieves the token through the instance role |

The SSM names are defined in [`config.py`](src/vla_pipeline/config.py):
`/<project>/sagemaker-role-arn`, `models-bucket`, `isaac-lab-arena/hf-secret-name`,
and `/<project>/component/{training-role-arn,workload-role-arn,validation-role-arn,
trust-bucket,handoff-bucket,pipeline-prefix}`. The default project is `physical-ai`.
Encrypted resources may additionally require their configured KMS key permissions.

Use **instance-role credentials** for both the host launcher and containers.
Set IMDSv2's HTTP PUT response hop limit to **2** for Docker bridge access.
The runner checks the actual role inside a CPU container before GPU work; an
assumed-role session or host environment credentials do not substitute for this.
Unset static AWS keys, `AWS_PROFILE` and `AWS_DEFAULT_PROFILE` on the EC2 host;
the bootstrap commands below also clear both session-token aliases.

Have an operator provision a **separate, versioned development S3 bucket** and grant
the permissions above. Do not grant the box access to production trust publication
or change the SageMaker roles' trust policies. The runner refuses a development
bucket equal to any discovered managed `bucket`, `handoff_bucket` or `trust_bucket`.

For a **new** bucket in the supported `us-east-1` region, run these with the
provisioning identity before connecting to EC2:

```bash
# Run with the provisioning identity; skip creation for an existing bucket.
: "${ARENA_DEV_BUCKET:?Repeat the deployment selection in this shell}"
test "$AWS_DEFAULT_REGION" = us-east-1
test "$VLA_REGION" = us-east-1
test "$(aws sts get-caller-identity --query Account --output text)" = "$TARGET_ACCOUNT_ID"
aws s3api create-bucket --bucket "$ARENA_DEV_BUCKET" --region us-east-1
aws s3api put-bucket-versioning --bucket "$ARENA_DEV_BUCKET" \
  --versioning-configuration Status=Enabled
aws s3api put-public-access-block --bucket "$ARENA_DEV_BUCKET" \
  --public-access-block-configuration \
  BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
```

This command intentionally uses the us-east-1 bucket-creation form. Do not
change it to another region: substituting us-west-2 without regional bucket
configuration failed in the earlier review. Other deployment regions are
outside this sample's supported setup.

For a **new EC2 role and instance profile**, the following creates the runtime
permissions from the table. Use an existing role only after checking its grants;
do not rerun creation against the current box's role.

```bash
: "${ARENA_EC2_ROLE:?Repeat the deployment selection in this shell}"
HF_SECRET_NAME=$(aws ssm get-parameter \
  --name "/$VLA_FOUNDATION_PROJECT/isaac-lab-arena/hf-secret-name" \
  --query Parameter.Value --output text)
HF_SECRET_ARN=$(aws secretsmanager describe-secret \
  --secret-id "$HF_SECRET_NAME" --query ARN --output text)
mkdir -p local-dev
python - "$ACCOUNT_ID" "$AWS_DEFAULT_REGION" "$VLA_FOUNDATION_PROJECT" \
  "$ARENA_DEV_BUCKET" "$HF_SECRET_ARN" <<'PY'
import json, pathlib, sys
account, region, project, bucket, secret = sys.argv[1:]
directory = pathlib.Path("local-dev")
trust = {"Version": "2012-10-17", "Statement": [{
    "Effect": "Allow", "Principal": {"Service": "ec2.amazonaws.com"},
    "Action": "sts:AssumeRole"}]}
policy = {"Version": "2012-10-17", "Statement": [
    {"Effect": "Allow", "Action": ["ssm:GetParameter"],
     "Resource": f"arn:aws:ssm:{region}:{account}:parameter/{project}/*"},
    {"Effect": "Allow", "Action": ["secretsmanager:GetSecretValue"],
     "Resource": secret},
    {"Effect": "Allow", "Action": [
        "s3:GetBucketVersioning", "s3:GetBucketLocation",
        "s3:ListBucket", "s3:ListBucketMultipartUploads"],
     "Resource": f"arn:aws:s3:::{bucket}"},
    {"Effect": "Allow", "Action": [
        "s3:GetObject", "s3:GetObjectVersion", "s3:PutObject",
        "s3:AbortMultipartUpload", "s3:ListMultipartUploadParts"],
     "Resource": f"arn:aws:s3:::{bucket}/*"}
]}
(directory / "ec2-trust.json").write_text(json.dumps(trust))
(directory / "ec2-runtime-policy.json").write_text(json.dumps(policy))
PY
aws iam create-role --role-name "$ARENA_EC2_ROLE" \
  --assume-role-policy-document file://local-dev/ec2-trust.json
aws iam put-role-policy --role-name "$ARENA_EC2_ROLE" --policy-name VlaLocalRuntime \
  --policy-document file://local-dev/ec2-runtime-policy.json
aws iam attach-role-policy --role-name "$ARENA_EC2_ROLE" \
  --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly
aws iam attach-role-policy --role-name "$ARENA_EC2_ROLE" \
  --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore
aws iam create-instance-profile --instance-profile-name "$ARENA_EC2_ROLE"
aws iam add-role-to-instance-profile --instance-profile-name "$ARENA_EC2_ROLE" \
  --role-name "$ARENA_EC2_ROLE"
```

The managed ECR policy supplies image-read access, including AWS's sklearn
repository. CodeBuild's separate role handles publishing. The SSM managed policy
enables Session Manager access to the instance. The source-transfer procedure
below uses your existing toolchain checkout and the development S3 bucket.

##### Use an already acquired host

If an operator supplied a new, idle GPU host in your target account, keep it and
skip the new-instance launch recipe. First deploy the application, build its
images, and prepare the development bucket and runtime role/profile above.
Record the supplied instance ID and its actual region on the provisioning
machine. `EC2_HOST_REGION` selects EC2 and Session Manager operations only;
keep `AWS_DEFAULT_REGION`, `AWS_REGION` and `VLA_REGION` set to `us-east-1`.
The host needs outbound HTTPS to the application region and download services.

```bash
INSTANCE_ID=your-supplied-instance-id
export EC2_HOST_REGION=your-supplied-instance-region
test "$(aws sts get-caller-identity --query Account --output text)" = "$TARGET_ACCOUNT_ID"
aws ec2 describe-instances --region "$EC2_HOST_REGION" --instance-ids "$INSTANCE_ID" \
  --query 'Reservations[0].Instances[0].{ID:InstanceId,State:State.Name,Type:InstanceType,AMI:ImageId,Role:IamInstanceProfile.Arn,Metadata:MetadataOptions,Volumes:BlockDeviceMappings}'
```

Check the supplied AMI, storage and idle status against the host requirements
above. If no instance profile is attached, associate the runtime profile created
above. Preserve an existing profile; the selected `ARENA_EC2_ROLE` must identify
its role and have the listed runtime grants.

```bash
HOST_PROFILE_ARN=$(aws ec2 describe-instances --region "$EC2_HOST_REGION" \
  --instance-ids "$INSTANCE_ID" \
  --query 'Reservations[0].Instances[0].IamInstanceProfile.Arn' --output text)
if [ -z "$HOST_PROFILE_ARN" ] || [ "$HOST_PROFILE_ARN" = None ]; then
  aws ec2 associate-iam-instance-profile --region "$EC2_HOST_REGION" \
    --instance-id "$INSTANCE_ID" --iam-instance-profile "Name=$ARENA_EC2_ROLE"
else
  HOST_ROLE=$(aws iam get-instance-profile \
    --instance-profile-name "${HOST_PROFILE_ARN##*/}" \
    --query 'InstanceProfile.Roles[0].RoleName' --output text)
  if [ "$HOST_ROLE" != "$ARENA_EC2_ROLE" ]; then
    echo "Existing profile uses $HOST_ROLE; preserve it and resolve the runtime-role selection/grants." >&2
    exit 1
  fi
fi
aws ec2 modify-instance-metadata-options --region "$EC2_HOST_REGION" \
  --instance-id "$INSTANCE_ID" --http-endpoint enabled \
  --http-tokens required --http-put-response-hop-limit 2
printf '%s\n' "$INSTANCE_ID" > local-dev/ec2-instance-id.txt
printf '%s\n' "$EC2_HOST_REGION" > local-dev/ec2-host-region.txt
```

An instance without a role is not ready for the local runner or Session Manager.
Allow profile propagation, then continue at [Connect to the host](#connect-to-the-host).
Do not run the new-instance commands for a supplied host.

##### Select and launch a new host

For a new host, the following uses an AWS-published x86-64
Ubuntu 22.04 Base GPU AMI, one Foundation private subnet with NAT egress, a new
security group with no inbound rules, and Session Manager access. It requests
one `g6e.8xlarge` with an encrypted **800 GiB gp3 root volume**, plus the instance's
NVMe scratch. Run with the target provisioning identity. Use a new deployment
label and preserve it with the launch request for retries.

```bash
set -euo pipefail
: "${TARGET_ACCOUNT_ID:?Repeat the deployment selection with provisioning credentials}"
if [ "$(aws sts get-caller-identity --query Account --output text)" != "$TARGET_ACCOUNT_ID" ]; then
  echo "Selected AWS identity is not the intended deployment account." >&2
  exit 1
fi
: "${VLA_FOUNDATION_PROJECT:?Set the deployed Foundation project name}"
: "${ARENA_EC2_ROLE:?Set the new runtime role/profile created above}"
export EC2_HOST_REGION="$AWS_DEFAULT_REGION"
HOST_LABEL="arena-local-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p local-dev
VPC_ID=$(aws ssm get-parameter --name "/$VLA_FOUNDATION_PROJECT/vpc-id" \
  --query Parameter.Value --output text)
PRIVATE_SUBNETS=$(aws ssm get-parameter \
  --name "/$VLA_FOUNDATION_PROJECT/private-subnet-ids" \
  --query Parameter.Value --output text)
IFS=',' read -r -a SUBNET_IDS <<< "$PRIVATE_SUBNETS"
aws ec2 describe-subnets --subnet-ids "${SUBNET_IDS[@]}" \
  --query 'Subnets[].{Subnet:SubnetId,VPC:VpcId,AZ:AvailabilityZone,FreeIPs:AvailableIpAddressCount}' \
  --output table
aws ec2 describe-instance-type-offerings --location-type availability-zone \
  --filters Name=instance-type,Values=g6e.8xlarge \
  --query 'InstanceTypeOfferings[].Location' --output table
```

Select a listed **Foundation private subnet** in an Availability Zone that offers
the instance type. Offerings and sufficient quota do not reserve capacity.
Keep the selected AMI ID in the saved request; the public SSM parameter advances
as AWS publishes releases.

```bash
SUBNET_ID=your-selected-private-subnet-id
case ",$PRIVATE_SUBNETS," in
  *",$SUBNET_ID,"*) ;;
  *) echo "Subnet is outside the discovered Foundation private subnet list." >&2; exit 1 ;;
esac
if [ "$(aws ec2 describe-subnets --subnet-ids "$SUBNET_ID" \
  --query 'Subnets[0].VpcId' --output text)" != "$VPC_ID" ]; then
  echo "Selected subnet belongs to a different VPC." >&2
  exit 1
fi
AMI_ID=$(aws ssm get-parameter \
  --name /aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id \
  --query Parameter.Value --output text)
aws ec2 describe-images --owners amazon --image-ids "$AMI_ID" \
  --query 'Images[0].{ID:ImageId,Name:Name,Owner:OwnerId,State:State,Arch:Architecture,Root:RootDeviceName,Devices:BlockDeviceMappings}' \
  --output json > local-dev/ec2-ami.json
python - <<'PY'
import json, pathlib
image = json.loads(pathlib.Path("local-dev/ec2-ami.json").read_text())
if not image or image["State"] != "available" or image["Arch"] != "x86_64":
    raise SystemExit("The selected AMI is not an available Amazon x86-64 image")
print(json.dumps(image, indent=2))
PY
EXISTING_GROUPS=$(aws ec2 describe-security-groups \
  --filters "Name=vpc-id,Values=$VPC_ID" "Name=group-name,Values=$HOST_LABEL" \
  --query 'length(SecurityGroups)' --output text)
if [ "$EXISTING_GROUPS" != "0" ]; then
  echo "Host label already names a security group; inspect its owner before proceeding." >&2
  exit 1
fi
```

Review the account, AMI, subnet, IAM profile, quota and resource inventory before
creating the host resources. The security-group lookup must be empty for a new
label. The next commands create one group and save the exact instance request.
The group's default outbound rule permits downloads; no inbound rule or public
IP is requested.

```bash
SECURITY_GROUP_ID=$(aws ec2 create-security-group \
  --group-name "$HOST_LABEL" --description "Arena local debugging via Session Manager" \
  --vpc-id "$VPC_ID" --query GroupId --output text)
python - "$AMI_ID" "$SUBNET_ID" "$SECURITY_GROUP_ID" "$ARENA_EC2_ROLE" \
  "$HOST_LABEL" "$VLA_FOUNDATION_PROJECT" <<'PY'
import json, pathlib, sys
ami, subnet, group, profile, label, project = sys.argv[1:]
image = json.loads(pathlib.Path("local-dev/ec2-ami.json").read_text())
if image["ID"] != ami or not image["Root"]:
    raise SystemExit("AMI selection changed or has no root device")
request = {
    "ImageId": ami, "InstanceType": "g6e.8xlarge",
    "MinCount": 1, "MaxCount": 1, "ClientToken": label,
    "IamInstanceProfile": {"Name": profile},
    "MetadataOptions": {"HttpTokens": "required", "HttpEndpoint": "enabled",
                        "HttpPutResponseHopLimit": 2},
    "NetworkInterfaces": [{"DeviceIndex": 0, "SubnetId": subnet, "Groups": [group],
                           "AssociatePublicIpAddress": False, "DeleteOnTermination": True}],
    "BlockDeviceMappings": [{"DeviceName": image["Root"], "Ebs": {
        "VolumeSize": 800, "VolumeType": "gp3", "Encrypted": True,
        "Iops": 3000, "Throughput": 125, "DeleteOnTermination": True}}],
    "TagSpecifications": [
        {"ResourceType": kind, "Tags": [
            {"Key": "Name", "Value": label}, {"Key": "Project", "Value": project}]}
        for kind in ("instance", "volume")],
}
path = pathlib.Path("local-dev/ec2-launch.json")
path.write_text(json.dumps(request, indent=2) + "\n")
print(path.read_text())
PY
if aws ec2 run-instances --cli-input-json file://local-dev/ec2-launch.json \
  --dry-run > local-dev/ec2-dry-run.log 2>&1; then
  echo "Unexpected successful exit from a dry run; inspect the response." >&2
  exit 1
else
  cat local-dev/ec2-dry-run.log
  if ! grep -q '(DryRunOperation)' local-dev/ec2-dry-run.log; then
    exit 1
  fi
fi
```

`DryRunOperation` is the expected nonzero response for an authorized dry run;
other errors must be resolved. It does not test available capacity. After
reviewing the saved request, submit it once:

```bash
INSTANCE_ID=$(aws ec2 run-instances --cli-input-json file://local-dev/ec2-launch.json \
  --query 'Instances[0].InstanceId' --output text)
printf '%s\n' "$INSTANCE_ID" > local-dev/ec2-instance-id.txt
printf '%s\n' "$EC2_HOST_REGION" > local-dev/ec2-host-region.txt
aws ec2 describe-instances --instance-ids "$INSTANCE_ID" \
  --query 'Reservations[0].Instances[0].{ID:InstanceId,State:State.Name,Subnet:SubnetId,AMI:ImageId,Role:IamInstanceProfile.Arn,Metadata:MetadataOptions,Volumes:BlockDeviceMappings}'
```

##### Connect to the host

Both host routes continue here on the provisioning machine. Restore the recorded
host identity if this is a new shell:

```bash
INSTANCE_ID=$(cat local-dev/ec2-instance-id.txt)
export EC2_HOST_REGION=$(cat local-dev/ec2-host-region.txt)
aws ssm describe-instance-information --region "$EC2_HOST_REGION" \
  --filters "Key=InstanceIds,Values=$INSTANCE_ID" \
  --query 'InstanceInformationList[].{ID:InstanceId,Status:PingStatus,Platform:PlatformName}'
```

Wait for the instance to be running and Session Manager to report `Online`.
Use the EC2 console's **Connect → Session Manager**, or install the ordinary
[Session Manager CLI plugin](https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager-working-with-install-plugin.html)
on the provisioning machine, then connect using the host's region:

```bash
aws ssm start-session --region "$EC2_HOST_REGION" --target "$INSTANCE_ID"
```

The AWS CLI alone does not include that plugin; verify
`session-manager-plugin --version` (the documented minimum is 1.2.764.0).
The connecting identity needs `ssm:StartSession`, `ssm:ResumeSession` and
`ssm:TerminateSession` permission for its own sessions and target instance.
No SSH key or inbound SSH rule is needed for this Session Manager path.
For the **manual host setup below**, run `sudo -iu ubuntu`, then `bash`.
If `vla deploy --prepare-host` already prepared this host, keep using the CLI
from your provisioning machine. Its checkout and records under `/opt/vla-cli/`
are owned by root. For diagnostics in Session Manager, use `sudo -i`, then
`cd` to the component path in the deployment record's `local.component`;
the `ubuntu` shell cannot access that private checkout. Do not change its ownership.
If the node stays offline, inspect its attached profile/SSM agent and outbound
network access; a running EC2 state alone does not prove a usable connection.
For a newly launched host, preserve `ec2-launch.json` and its client token while
resolving an ambiguous submission; creating a new token can create another instance.
The launch recipe deletes the root volume on termination, so archive verified
run evidence before teardown.
Inspect the new host's actual driver, Docker and scratch mount in the next steps.
This launch recipe still requires execution in the target account before being
counted as a tested cold-start path.

Keep the project, development bucket, instance profile and image digests selected
above for the EC2 shell. The saved selections below carry them between machines.

On the provisioning machine, save the nonsecret deployment selection to your
development bucket so the new host can restore it after cloning:

```bash
declare -p INSTANCE_ID EC2_HOST_REGION ARENA_EC2_ROLE \
  >> local-dev/deployment-selection.txt
aws s3 cp local-dev/deployment-selection.txt \
  "s3://$ARENA_DEV_BUCKET/localdev/bootstrap/deployment-selection.txt"
```

#### Clone and install on EC2

##### Initialize the new Ubuntu host

Run this once on the **new, idle Ubuntu 22.04 host**, before any workloads.
The selected DLAMI must provide a working NVIDIA driver and the instance-store
mount; `nvidia-smi` must succeed before Docker setup. AWS CLI v2 is also an
ordinary prerequisite on this host; verify `aws --version` and use the official
[AWS CLI installer](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)
if the AMI lacks it. The source bundle route needs the CLI before cloning.

```bash
set -euo pipefail
unset AWS_PROFILE AWS_DEFAULT_PROFILE AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY \
  AWS_SESSION_TOKEN AWS_SECURITY_TOKEN
export AWS_DEFAULT_REGION=us-east-1
export AWS_REGION="$AWS_DEFAULT_REGION"
aws --version
```

**Before cloning**, copy the exact account ID and development bucket name from
the provisioning machine's selection. No repository or component directory is
needed for this block; it uses the new host's instance role:

```bash
export TARGET_ACCOUNT_ID=the-account-selected-on-the-provisioning-machine
export ARENA_DEV_BUCKET=the-development-bucket-created-above
test "$(aws sts get-caller-identity --query Account --output text)" = "$TARGET_ACCOUNT_ID"
```

Keep these values for the bundle download. Project, runtime role and the remaining
deployment settings are restored **after** source acquisition, from the component
directory. Continue the new host's system prerequisites:

```bash
nvidia-smi
sudo apt-get update
sudo apt-get install -y git python3-venv pigz ca-certificates curl gnupg
python3 --version
```

If Docker Engine or its Compose plugin is absent on this new host, install them
from Docker's Ubuntu repository. The following removes only conflicting Docker
distribution packages that are actually installed. This is initial setup for
the new idle host, not an upgrade procedure for a host running workloads:

```bash
if ! command -v docker >/dev/null || ! docker compose version >/dev/null 2>&1; then
  DOCKER_CONFLICTS=()
  for package in docker.io docker-compose docker-compose-v2 docker-buildx docker-doc \
      podman-docker containerd runc; do
    if dpkg-query -W -f='${db:Status-Status}' "$package" 2>/dev/null | grep -qx installed; then
      DOCKER_CONFLICTS+=("$package")
    fi
  done
  if [ "${#DOCKER_CONFLICTS[@]}" -gt 0 ]; then
    sudo apt-get remove -y "${DOCKER_CONFLICTS[@]}"
  fi
  sudo install -m 0755 -d /etc/apt/keyrings
  sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
    -o /etc/apt/keyrings/docker.asc
  sudo chmod a+r /etc/apt/keyrings/docker.asc
  sudo tee /etc/apt/sources.list.d/docker.sources >/dev/null <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: jammy
Components: stable
Architectures: amd64
Signed-By: /etc/apt/keyrings/docker.asc
EOF
  sudo apt-get update
  sudo apt-get install -y docker-ce docker-ce-cli containerd.io \
    docker-buildx-plugin docker-compose-plugin
fi
```

If NVIDIA Container Toolkit is absent, install it, then configure Docker's
NVIDIA runtime. The Docker restart below is part of initial host preparation:

```bash
if ! command -v nvidia-ctk >/dev/null; then
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
  sudo apt-get update
  sudo apt-get install -y nvidia-container-toolkit
fi
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl enable --now docker
sudo systemctl restart docker
sudo docker info
docker compose version
nvidia-smi
```

Require successful commands, then verify the scratch filesystem before creating
the application directory:

```bash
lsblk -o NAME,SIZE,TYPE,FSTYPE,MOUNTPOINTS
findmnt -T /opt/dlami/nvme
df -hT / /opt/dlami/nvme
python3 - <<'PY'
import os, pathlib
scratch = pathlib.Path("/opt/dlami/nvme")
if not scratch.is_dir() or os.stat(scratch).st_dev == os.stat("/").st_dev:
    raise SystemExit("Expected separate DLAMI NVMe mount is absent; stop host preparation")
PY
export VLA_SCRATCH_ROOT=/opt/dlami/nvme/vla-tests
sudo install -d -m 755 -o ubuntu -g ubuntu "$VLA_SCRATCH_ROOT"
```

The selected DLAMI normally prepares that mount at boot. If it is absent, stop
and resolve the AMI/storage prerequisite; do not format an unidentified disk.
If using an operator-provisioned separate filesystem instead, inspect it and
set `VLA_SCRATCH_ROOT` to its application directory. Record the AMI ID, driver,
Docker/Compose versions and `df`/`lsblk` output with the test evidence. A later
GPU container probe verifies actual GPU passthrough.
`setup.sh` installs Python dependencies; it does not perform the host setup above.

##### Obtain the source and install the application

The CLI's `deploy --prepare-host` delivers source automatically. For manual
host setup, transfer your existing toolchain checkout as a Git bundle using the
procedure below. The bundle preserves its commit identity without requiring
repository credentials on EC2.

On the provisioning machine, start in this component directory. Commit tracked
edits first: a Git bundle transfers committed content. Select the target
provisioning profile and the development bucket created above:

```bash
set -euo pipefail
unset AWS_DEFAULT_PROFILE AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY \
  AWS_SESSION_TOKEN AWS_SECURITY_TOKEN
export AWS_PROFILE=your-target-account-profile
: "${TARGET_ACCOUNT_ID:?Repeat the deployment selection with the same target account}"
if [ "$(aws sts get-caller-identity --query Account --output text)" != "$TARGET_ACCOUNT_ID" ]; then
  echo "Selected AWS identity is not the intended bundle destination account." >&2
  exit 1
fi
: "${ARENA_DEV_BUCKET:?Set the target development bucket created above}"
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "Commit tracked changes before creating the source bundle." >&2
  exit 1
fi
BOOTSTRAP_BRANCH=$(git branch --show-current)
BOOTSTRAP_COMMIT=$(git rev-parse HEAD)
mkdir -p local-dev
BOOTSTRAP_BUNDLE="local-dev/toolchain-$BOOTSTRAP_COMMIT.bundle"
git bundle create "$BOOTSTRAP_BUNDLE" "$BOOTSTRAP_BRANCH"
git bundle verify "$BOOTSTRAP_BUNDLE"
BOOTSTRAP_SHA256=$(python -c \
  'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' \
  "$BOOTSTRAP_BUNDLE")
aws s3 cp "$BOOTSTRAP_BUNDLE" \
  "s3://$ARENA_DEV_BUCKET/localdev/bootstrap/$BOOTSTRAP_COMMIT/toolchain.bundle"
printf 'Branch: %s\nCommit: %s\nBundle SHA256: %s\n' \
  "$BOOTSTRAP_BRANCH" "$BOOTSTRAP_COMMIT" "$BOOTSTRAP_SHA256"
```

On the new EC2 host, use its target instance role. Set the branch, commit and
checksum to the values printed on the provisioning machine, then clone:

```bash
set -euo pipefail
unset AWS_PROFILE AWS_DEFAULT_PROFILE AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY \
  AWS_SESSION_TOKEN AWS_SECURITY_TOKEN
export AWS_DEFAULT_REGION=us-east-1
export AWS_REGION="$AWS_DEFAULT_REGION"
: "${TARGET_ACCOUNT_ID:?Set the before-clone host account value}"
: "${ARENA_DEV_BUCKET:?Set the before-clone host development bucket value}"
test "$(aws sts get-caller-identity --query Account --output text)" = "$TARGET_ACCOUNT_ID"
BOOTSTRAP_BRANCH=the-branch-printed-above
BOOTSTRAP_COMMIT=the-full-commit-printed-above
BOOTSTRAP_SHA256=the-bundle-checksum-printed-above
aws s3 cp \
  "s3://$ARENA_DEV_BUCKET/localdev/bootstrap/$BOOTSTRAP_COMMIT/toolchain.bundle" \
  "$HOME/toolchain.bundle"
python3 - "$HOME/toolchain.bundle" "$BOOTSTRAP_SHA256" <<'PY'
import hashlib, pathlib, sys
actual = hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest()
if actual != sys.argv[2]:
    raise SystemExit("Source bundle checksum mismatch")
PY
```

For a **new EC2 checkout**, clone the verified bundle and install:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --branch "$BOOTSTRAP_BRANCH" \
  "$HOME/toolchain.bundle" "$HOME/canonical-dev"
if [ "$(git -C "$HOME/canonical-dev" rev-parse HEAD)" != "$BOOTSTRAP_COMMIT" ]; then
  echo "Cloned commit differs from the selected source commit." >&2
  exit 1
fi
cd "$HOME/canonical-dev/isaac-lab-arena-on-aws"
bash scripts/local/setup.sh
source .venv/bin/activate
```

For an **existing EC2 checkout receiving a reviewed update**, use this block
instead of the clone block. First recreate/upload the bundle from the updated
provisioning-machine checkout, and repeat the host download/hash verification
above with its new commit and checksum. Wait until the previous run has finished.
This preserves the existing ignored run directories, selections and caches:

```bash
cd "$HOME/canonical-dev"
test -z "$(git status --porcelain --untracked-files=no)"
test "$(git branch --show-current)" = "$BOOTSTRAP_BRANCH"
git fetch "$HOME/toolchain.bundle" "$BOOTSTRAP_BRANCH"
test "$(git rev-parse FETCH_HEAD)" = "$BOOTSTRAP_COMMIT"
git merge --ff-only FETCH_HEAD
test "$(git rev-parse HEAD)" = "$BOOTSTRAP_COMMIT"
cd isaac-lab-arena-on-aws
bash scripts/local/setup.sh
source .venv/bin/activate
```

Require every command to succeed before continuing. The bundle carries Git source
and history; images still come from the full builds in the target account. This
route uses no source-account credentials on EC2. Its `origin` points at the local
bundle: before pushing development changes, set `origin` to an authenticated,
writable Git repository with `git remote set-url origin <your-writable-remote>`.

After installing the source on EC2, restore the selection saved by your provisioning
machine. Run this in the component directory on EC2, keeping `AWS_PROFILE`
and static credential aliases unset:

```bash
mkdir -p local-dev
# On an existing host, preserve its previously saved scratch path in this shell.
if [ -f local-dev/deployment-selection.txt ]; then
  source local-dev/deployment-selection.txt
fi
aws s3 cp "s3://$ARENA_DEV_BUCKET/localdev/bootstrap/deployment-selection.txt" \
  local-dev/deployment-selection.txt
cat local-dev/deployment-selection.txt  # names and IDs only; check your selection
source local-dev/deployment-selection.txt
test "$AWS_DEFAULT_REGION" = us-east-1
test "$AWS_REGION" = us-east-1
test "$VLA_REGION" = us-east-1
test "$(aws sts get-caller-identity --query Account --output text)" = "$TARGET_ACCOUNT_ID"
```

The downloaded selection includes the host identity and the selected image
digests. It does not transfer the laptop's credentials. If you selected different
builds later, repeat the provisioning-machine upload and this download.
On EC2, save the scratch path you checked during host preparation:

```bash
: "${VLA_SCRATCH_ROOT:?Set the application path on the verified separate scratch mount}"
declare -p VLA_SCRATCH_ROOT >> local-dev/deployment-selection.txt
: "${LOCAL_TRAIN_IMAGE:?Restore the selected training image digest}"
: "${LOCAL_ARENA_EVAL_IMAGE:?Restore the selected evaluator image digest}"
```

Before pulling images or starting a sample, confirm that **this EC2 role** can
read this deployment's discovery parameter and HF secret. A successful read
using the provisioning profile does not establish host access:

```bash
aws sts get-caller-identity
HF_SECRET_NAME=$(aws ssm get-parameter \
  --name "/$VLA_FOUNDATION_PROJECT/isaac-lab-arena/hf-secret-name" \
  --query Parameter.Value --output text)
aws secretsmanager get-secret-value --region us-east-1 \
  --secret-id "$HF_SECRET_NAME" --version-stage AWSCURRENT \
  --query '{ARN:ARN,VersionId:VersionId}' --output json
```

The last command prints only secret metadata. If it returns `AccessDenied`,
check the attached role against the permissions table above, including the
actual project prefix and secret ARN. Region and project are part of those
ARNs. The new-role recipe grants the selected deployment's permissions.
If an operator supplied a shared role, have them add the missing scoped grants
under a separate policy name; do not replace its existing policies.

`setup.sh` installs the package plus SageMaker **2.257.6**, host boto3 **1.43.73**,
Docker Python SDK **7.2.0** and the local-mode extras in this clone's `.venv`.
The separate Validate SDK is hash-pinned and packaged at source staging; see
[validation runtime](#validation-runtime-and-training-lineage). The first staging needs package-index
access to download its locked wheels; subsequent staging reuses verified cache files.

#### Preflight and pull the selected images

**On the EC2 host**, from the component directory. Preparation downloads missing
images, creates the Compose probe container, then runs its small CPU command.
These have separate limits; a six-hour overall budget does not replace the
shorter phase limits:

| Setting | Sample walkthrough | Scope |
| --- | --- | --- |
| `--image-pull-timeout-seconds` | 7200 (default) | Each missing image download |
| `--container-preparation-seconds` | 600 (default) | Compose probe container creation |
| CPU probe | 120 seconds | CPU command after preparation |
| `--timeout-seconds` | 21600 below | Whole launcher, including preparation and pipeline |
| `--max-runtime-seconds` | 21600 below | Each training/evaluation job; overall limit still wins |
| Waiter `--timeout-seconds` | 21900 below | Observation only; never extends or cancels the run |

The 2026-09-17 review used the six-hour overall/job settings successfully.
Image downloads took tens of minutes; the retained cold-start failure spent
25m25s in preflight, including downloads and the old two-minute Compose timeout.
Ten minutes for container preparation is a proposed recovery improvement,
not a guarantee for every disk. Retain each attempt's directory.

```bash
: "${VLA_FOUNDATION_PROJECT:?Repeat the deployment selection in this EC2 shell}"
: "${ARENA_DEV_BUCKET:?Repeat the deployment selection in this EC2 shell}"
: "${ARENA_EC2_ROLE:?Repeat the deployment selection in this EC2 shell}"
: "${VLA_SCRATCH_ROOT:?Verify the mounted scratch filesystem and set its application path}"
: "${LOCAL_TRAIN_IMAGE:?Set the full training ECR digest printed by the build section}"
: "${LOCAL_ARENA_EVAL_IMAGE:?Set the full Arena ECR digest printed by the build section}"
PREFLIGHT_ID="preflight-$(date -u +%Y%m%dT%H%M%SZ)"

bash scripts/local/launch.sh "$PREFLIGHT_ID" \
  --region us-east-1 \
  --development-bucket "$ARENA_DEV_BUCKET" \
  --expected-role "$ARENA_EC2_ROLE" \
  --scratch-root "$VLA_SCRATCH_ROOT" \
  --train-image "$LOCAL_TRAIN_IMAGE" --eval-image "$LOCAL_ARENA_EVAL_IMAGE" \
  --preflight-only --pull-images \
  --timeout-seconds 21600 --container-preparation-seconds 600

sudo .venv/bin/python scripts/local/wait_run.py \
  --run-dir "local-dev/runs/$PREFLIGHT_ID" --timeout-seconds 21900
```

The launcher detaches using systemd. Its shell returning zero means submission
only. The waiter observes completion and returns nonzero on a failed preflight.
Every 30 seconds the waiter prints elapsed time, service state and the age of the
last log write. Those observations show activity; they do not prove forward
progress or completion. Compose preparation also prints elapsed time and retains
its command output. Use
`sudo tail -f "local-dev/runs/$PREFLIGHT_ID/run.log"` for more detail. Ctrl-C
stops following, not the service. After the service exits, inspect:

```bash
sudo cat "local-dev/runs/$PREFLIGHT_ID/status.json"
sudo journalctl -u "vla-local-$PREFLIGHT_ID" --no-pager -n 30
```

Require `status: PreflightSucceeded`. This checks the clean Git commit, package
versions, SSM, separate versioned S3 reads/writes, multipart cleanup, HF-secret
delivery, staged source, image availability and actual container IAM identity.
It also runs a CPU container through Compose and cleans up that probe's resources.
On failure, `compose-probe-status.json` and `compose-*.stdout.log` /
`compose-*.stderr.log` retain the phase, command, deadline and output. Cleanup
failure is recorded separately from the original error. If creation timed out,
Docker can finish it after the client exits: an empty immediate inventory does
not prove cleanup. Wait for disk activity to settle, run the exact project-specific
cleanup command printed in the report, inspect that project's containers, then
launch a new preflight ID. See [preflight recovery](#debug-cancel-retry-and-retain-evidence).
It also executes a real failing **Condition/Fail-step local pipeline** and requires
the success checker to reject it. No training or evaluation GPU step runs.

The preflight checks the selected cell's initial disk reserve after image pulls,
before any training. `disk-checks.json` records free/required GiB for **both**
filesystems; a refusal names the insufficient disk. The full run repeats this
check, because space may have changed since preflight. These reserves cover the
sample's observed growth with margin, not every possible training duration,
checkpoint count or dataset. Image pulls can be large.
`--pull-images` pulls only absent images and authenticates through the instance
role. Subsequent runs can omit it. Check GPU passthrough against the pulled image:

```bash
PREFLIGHT_TRAIN_IMAGE=$(sudo .venv/bin/python -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["images"]["train"])' \
  "local-dev/runs/$PREFLIGHT_ID/manifest.json")
sudo docker run --rm --gpus all --entrypoint nvidia-smi "$PREFLIGHT_TRAIN_IMAGE"
df -h .
sudo docker ps
```

The tracked recipe derives from the complete passing managed export, including
`ExpectedPolicyConfig`, with `EvalTrials` increased from one to three.
It templates the account/region in ECR locations and
replaces source paths during staging. The commands above explicitly override its historical image references with this
account's images. `--eval-image` must be an `@sha256:` URI. Do not remove these
overrides in a new account just because the historical recipe contains a digest.

An alternative complete managed parameter export can be supplied with
`--parameters /absolute/path/parameters.json`. Use
`aws sagemaker list-pipeline-parameters-for-execution` with JSON output and automatic
pagination; do not hand-select a subset. That export requires a caller with the
managed execution read permission. The shipped recipe avoids requiring it on the box.

#### Run the four-step pipeline

Use a **new run ID**, after preflight has finished and the GPU is idle:

```bash
RUN_ID="arena-$(date -u +%Y%m%dT%H%M%SZ)"
bash scripts/local/launch.sh "$RUN_ID" \
  --region us-east-1 \
  --development-bucket "$ARENA_DEV_BUCKET" \
  --expected-role "$ARENA_EC2_ROLE" \
  --scratch-root "$VLA_SCRATCH_ROOT" \
  --train-image "$LOCAL_TRAIN_IMAGE" --eval-image "$LOCAL_ARENA_EVAL_IMAGE" \
  --timeout-seconds 21600 --max-runtime-seconds 21600

sudo .venv/bin/python scripts/local/wait_run.py \
  --run-dir "local-dev/runs/$RUN_ID" --timeout-seconds 21900
```

The shipped recipe selects N1.6, GR1, the fridge task, 200 steps, three episodes,
seed 100 and threshold 0.0. The runner prints the effective recipe before GPU work.
Change training dose with `--train-steps`;
the current N1.6 path supports a final checkpoint, not intermediate dose curves.
The three LIBERO spatial profiles below select their own family, images, training
suite and seed from the registry/managed-launcher contract.
The waiter performs independent verification after successful execution. Inspect
the result and archive it using the general acceptance section below, even when
you are skipping the optional LIBERO section.

`FineTune` includes dependency setup, dataset/model downloads and checkpoint
packaging as well as training. The N1.6 path also pauses for two minutes before
its base-model download. Follow `run.log` to distinguish these phases while the
waiter continues to report `FineTune`. Validate can also spend several quiet
minutes reading and publishing large model archives; inspect the logs and
elapsed time before treating a quiet interval as a stalled process.

**Reconnect without starting another run.** Reconnect to the saved instance
using [Session Manager](#connect-to-the-host), become `ubuntu`, and return to the
same component checkout. This command follows its last submitted run, including
preflight, and invokes independent verification when a full run finishes:

```bash
sudo .venv/bin/python scripts/local/wait_run.py --latest --timeout-seconds 21900
```

For subsequent inspection/archive commands, restore the same full-run ID in this
shell. Use the ID printed by the waiter if another launch has since changed
`latest-run-id.txt`:

```bash
RUN_ID=$(sudo cat local-dev/latest-run-id.txt)
sudo cat "local-dev/runs/$RUN_ID/status.json"
```

Continue to the result/archive commands only for a completed four-step run.
`PreflightSucceeded` means no training/evaluation ran; follow the launch block
above with a new full-run ID.

For a particular older run, use `--run-dir local-dev/runs/<run-id>` instead.
The launcher saves the latest ID only after systemd accepts the submission.
Resume **watching** with the waiter; restarting training requires an explicit
new launch.

The connector disables the base image's interactive-service `AppReady` healthcheck
for this batch workload. It uses the existing server readiness probe, rollout
watchdog, process exit and result validation. Absence of a Docker health status
does not mean the simulator is ready. Older images may show `unhealthy` while
valid rollouts progress; the historical sample did. Check timestamped activity and
[process evidence](#debug-cancel-retry-and-retain-evidence), then require independent
verification. Rebuild the connector to pick up this change.

The launcher pins the current Git HEAD, and the runner rejects tracked local
changes. For a debug iteration, commit the edits locally, then run a fresh ID;
push after verification. Each run records its commit, runner/helper hashes,
parameters, source archive identities, image IDs and SDK bundle digest.
Image-baked Arena evaluator changes require a rebuilt image and
`--eval-image <new-digest>`; editing the training sourcedir does not overlay
`/workspace/eval_entry.py`.

The launcher uses root for Docker and private run directories. Its AWS identity
remains the checked EC2 instance role. It retrieves the configured Secrets Manager
HF token and injects it into **both FineTune and SimEval containers**. A host
`export HF_TOKEN` is not forwarded by this detached launcher.

Storage is redirected **before staging**: all three config bucket references
point to the development bucket. Checkpoints still travel through **S3**, because
Arena requires a real S3 VersionId and ETag. Only source directories use `file://`.
Validate publishes real conditional-create artifacts/attestations to dev storage.
RegisterModel is removed from the gate branch.

#### Verify the completed run before cleanup

`wait_run.py` already runs the independent verifier. After it succeeds, inspect
the saved result for that exact run:

```bash
sudo cat "local-dev/runs/$RUN_ID/status.json"
sudo cat "local-dev/runs/$RUN_ID/steps.json"
sudo cat "local-dev/runs/$RUN_ID/independent-verification.json"
```

If you did not use the waiter, run the verifier once while the completed
containers and systemd journal are still present:

```bash
sudo .venv/bin/python scripts/local/verify_run.py \
  --run-dir "local-dev/runs/$RUN_ID"
```

The independent dataset lookup requests only the commit ID and retries temporary
Hugging Face queue or rate-limit responses (HTTP 429) within a five-minute
allowance, reporting each wait. If that allowance is exhausted, verification
remains incomplete. After the service recovers, run
the same verifier command on the GPU host against the completed run; training
does not need to restart. Preserve its containers, journal and failed
verification log until verification succeeds. Then refresh the saved run's
status so observers pick up the new proof.

Require the verifier to exit zero and create
`local-dev/runs/$RUN_ID/independent-verification.json` with `status: Succeeded`.
It independently checks:

- All four required steps succeeded and the gate evaluated true.
- The systemd manager recorded successful completion; missing transient-unit
  state is not treated as an observed exit code.
- All three workload containers exited zero, without OOM, using recorded images.
- The real negative control failed and was rejected.
- Validate used the packaged SDK for conditional publication, with no runtime
  `sdk_repair` or pip-install path.
- The S3 receipt and attestation bytes/digest and versioned checkpoint identity agree.
  GR00T revision identity is verified and other lineage is attested. OpenVLA and
  MolmoAct2 keep their existing `recorded_only`/`unavailable` qualifications.

Successful verification also writes `timings.json`, `training-metrics.csv`,
`training-summary.json` and `evaluation-summary.json`, together with redacted
per-container logs. The timing report separates container time from complete
step time, which includes SDK copying/compression/upload. Loss dictionaries with
no optimizer step retain a blank step field; log order is not an invented step.
It prints a readable summary alongside its detailed JSON, for example:

```text
Local workflow: independently verified
Training: 200/200 steps (trainer log)
Robot trials: 1/3 successful; success rate 33.3%
Run time: 0h 27m 32s, including preparation
Managed execution: not checked by this local verifier
Host: still running; archive results before stopping it
Results: <absolute run directory>
```

That is the measured sample outcome, not an expected score. A submitted managed
sample must be checked separately; local verification cannot certify registration.

The verifier needs the completed containers and systemd journal to still exist.
Retain them until verification and log/inspection backup are complete. Historical
reports from already-cleaned runs remain evidence; live inspection cannot recreate
their removed containers.

##### Reach the host from a CLI run

If you launched through `vla` on your provisioning machine, stay on that machine
to retrieve the saved run:

```bash
RUN_ID=arena-local-01
vla --json status "$RUN_ID"
```

Use your actual run ID. The record contains the AWS `profile` and, under
`request.local_target`, the host's `instance_id`, `host_region`, `component`
directory and `scratch_root`. Use those recorded values, not the provisioning
machine's top-level `component` path. The host region can differ from the
application region.

To open an interactive shell, set these values from that record:

```bash
TARGET_PROFILE=your-recorded-profile
INSTANCE_ID=your-recorded-instance-id
EC2_HOST_REGION=your-recorded-host-region
aws ssm start-session --profile "$TARGET_PROFILE" \
  --region "$EC2_HOST_REGION" --target "$INSTANCE_ID"
```

This interactive command needs the AWS Session Manager plugin on the
provisioning machine. The normal `vla` run/status commands use SSM Run Command
and do not need that plugin. An existing SSH connection to the same host also
works.

**In the EC2 shell**, set the recorded component directory, run ID and scratch
root. Then use the verification/archive commands below:

```bash
cd /your/recorded/host/component
RUN_ID=arena-local-01
VLA_SCRATCH_ROOT=/your/recorded/scratch/root
sudo cat "local-dev/runs/$RUN_ID/status.json"
```

The placeholders must be replaced; preparation normally uses a component path
under `/opt/vla-cli/`. Cleanup runs on EC2 because it inspects that host's
containers and disks. Do not rerun training to obtain cleanup records. If you
already launched directly on EC2 from this component directory, continue with
your existing `RUN_ID` and `VLA_SCRATCH_ROOT`.

##### Archive an accepted local run

**Optional cleanup:** Save your debugging logs and run records to S3, then
remove the completed run's stopped containers to reclaim disk space.

For a CLI-launched run, `pai arena cleanup RUN_ID --yes` performs the archive
and container-removal operation below, including over SSM. Repeat that same
command if its completion reply was interrupted; recorded removal intent
allows it to reconcile containers already removed. The manual commands remain
useful for runs launched directly with the lower-level scripts.

This is not required to train, evaluate or inspect results. For either Arena
or LIBERO, run these commands **in the EC2 component directory**, after
verification succeeds. CLI users can recover that directory
[from the saved run](#reach-the-host-from-a-cli-run).

For Arena, optionally [retrieve the effective policy configuration](#the-policy-config-that-actually-runs-is-not-the-file-on-disk)
**before removing containers**. That inspection command copies the file from the
completed evaluation container; the file also remains in SimEval's S3 model artifact.

```bash
: "${RUN_ID:?Set the completed run ID}"
sudo cat "local-dev/runs/$RUN_ID/timings.json"
sudo cat "local-dev/runs/$RUN_ID/training-summary.json"
sudo cat "local-dev/runs/$RUN_ID/evaluation-summary.json"
sudo .venv/bin/python scripts/local/archive_run.py \
  --run-dir "local-dev/runs/$RUN_ID" --remove-containers
```

Images and caches are kept. Save the printed S3 archive location with your
run notes.

Container removal does **not** remove bind-mounted temporary files. To recover
scratch space from this accepted, archived run, first inspect its recorded path:

```bash
if SCRATCH_RUN=$(sudo .venv/bin/python - "$RUN_ID" <<'PY'
import json, pathlib, re, sys
assert re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]{0,39}", sys.argv[1])
root = pathlib.Path("local-dev/runs") / sys.argv[1]
manifest = json.loads((root / "manifest.json").read_text())
archive = json.loads((root / "archive-manifest.json").read_text())
proof = json.loads((root / "independent-verification.json").read_text())
assert manifest["run_id"] == archive["run_id"] == proof["run_id"] == sys.argv[1]
assert proof["status"] == "Succeeded"
assert archive["status"] == "ArchivedAndReadBack" and archive.get("removed_containers")
work = pathlib.Path(manifest["scratch"]["work"]).resolve()
assert work == pathlib.Path(manifest["scratch"]["root"]).resolve() / "runs" / sys.argv[1]
print(work)
PY
); then
  sudo du -sh -- "$SCRATCH_RUN"
  printf 'Validated scratch path: %s\n' "$SCRATCH_RUN"
else
  unset SCRATCH_RUN
  printf 'Archive/path checks failed; nothing was removed. Preserve the run for inspection.\n' >&2
fi
# After checking the printed path, remove only that completed run's working directory:
if [ -n "${SCRATCH_RUN:-}" ]; then sudo rm -r -- "$SCRATCH_RUN"; fi
df -hT / "$VLA_SCRATCH_ROOT"
```

This keeps the shared `cache/` directory, other runs, published S3 artifacts and
the small records under `local-dev/runs/$RUN_ID`. Its source/container symlinks
will no longer resolve; inspect or copy anything needed for debugging first.
Older runners stored real `source-train`/`source-eval` directories under the
checkout, so their source working data can still consume root after scratch
cleanup. Review those exact old-run directories separately after archiving.
For a **failed** run, preserve diagnostics and any useful checkpoint before
reclaiming its data; it cannot use the successful-run archiver.

**Optional maintainer check:** when changing validation behavior, run the
bad-seed CPU replay before archiving and removing containers:

```bash
sudo .venv/bin/python scripts/local/check_bad_eval_seed.py \
  --run-dir "local-dev/runs/$RUN_ID"
```

It runs the actual staged Validate code with networking disabled and must
reject the deliberately wrong seed. The archiver includes
`negative-validation.json` and its log when present. This additional replay
is not required to use or archive the sample; the normal pipeline validation,
runner failure control and independent verification remain required.

##### Finish a local session

After verification and archiving, decide whether to keep the GPU host running
for another run. If no workload is active and all needed data is archived, you
can stop **your selected host** from the provisioning machine. For a CLI run,
reuse `TARGET_PROFILE`, `INSTANCE_ID` and `EC2_HOST_REGION`
[from its saved record](#reach-the-host-from-a-cli-run). For a host created by
the manual walkthrough, restore its deployment selection and read the saved IDs:

```bash
# Manual walkthrough only; CLI users already have these recorded values.
INSTANCE_ID=$(cat local-dev/ec2-instance-id.txt)
EC2_HOST_REGION=$(cat local-dev/ec2-host-region.txt)
```

Check the selected host before stopping it:

```bash
: "${TARGET_PROFILE:?Set the selected workload profile}"
: "${INSTANCE_ID:?Set the selected host ID}"
: "${EC2_HOST_REGION:?Set the host region}"
aws sts get-caller-identity --profile "$TARGET_PROFILE"
aws ec2 describe-instances --profile "$TARGET_PROFILE" \
  --region "$EC2_HOST_REGION" --instance-ids "$INSTANCE_ID" \
  --query 'Reservations[0].Instances[0].{ID:InstanceId,State:State.Name,Tags:Tags}'
# After confirming this is your idle test host:
aws ec2 stop-instances --profile "$TARGET_PROFILE" \
  --region "$EC2_HOST_REGION" --instance-ids "$INSTANCE_ID"
aws ec2 wait instance-stopped --profile "$TARGET_PROFILE" \
  --region "$EC2_HOST_REGION" --instance-ids "$INSTANCE_ID"
```

Stopping loses NVMe instance-store data and ends running-instance compute
charges. EBS, S3/ECR, secrets, logs and the Foundation NAT/network resources
remain; stopping EC2 is not permanent teardown. Do not stop a shared host with
another active workload or unarchived data.

**End of the local walkthrough.** Once your run is verified, its evidence is saved,
and you have decided whether to keep the host, you can stop reading. The next
section is troubleshooting/reference. Managed execution is a separate option
with [its own instructions](#run-in-cloud--managed-sagemaker).

#### Debug, cancel, retry and retain evidence

For a failed **Compose preflight**, inspect the retained report first:

```bash
PREFLIGHT_ID=the-failed-preflight-id
sudo cat "local-dev/runs/$PREFLIGHT_ID/compose-probe-status.json"
sudo bash -c 'cat "$1"/compose-*.stderr.log' bash "local-dev/runs/$PREFLIGHT_ID"
PROBE_PROJECT=$(sudo .venv/bin/python -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["project"])' \
  "local-dev/runs/$PREFLIGHT_ID/compose-probe-status.json")
sudo docker ps -a --filter "label=com.docker.compose.project=$PROBE_PROJECT"
```

After the failed launcher's service has stopped and Docker's pending creation has
settled, use the exact `cleanup_command` saved in that report. It addresses only
the failed probe's project. Inspect again for late-created containers; do not
use global prune. Preserve the failed attempt and repeat the preflight command
with a new ID. Successful preflight does not start training.

Docker's `healthy`/`unhealthy` field is the result of an image's healthcheck,
not the pipeline verdict. The current connector sets `HEALTHCHECK NONE` because
the base check targets an interactive service. Historical evaluator images
inherited a check that
looks for `AppReady` in Isaac Sim `kit_*.log` files under
`/isaac-sim/.nvidia-omniverse/logs/Kit`; real rollouts can progress while that
marker check reports `unhealthy`. Inspect the check in the selected container
and compare actual, timestamped progress with `run.log`, tracked step status and
`FailureReason`. Neither a healthy marker nor an unhealthy marker establishes
the workload's final result. Require final container exits and the independent
verifier above; do not dismiss other healthcheck failures as harmless.

```bash
sudo docker ps -a --filter "label=vla.local.run=$RUN_ID"
CONTAINER_ID=the-container-id-for-this-run
sudo docker inspect --format '{{json .Config.Healthcheck}}' "$CONTAINER_ID"
sudo docker inspect --format '{{json .State}}' "$CONTAINER_ID"
sudo docker logs --tail 100 "$CONTAINER_ID"
```

To inspect the service or explicitly cancel this run:

```bash
# Inspect only this run's containers, including completed ones.
sudo docker ps -a --filter "label=vla.local.run=$RUN_ID"
sudo journalctl -u "vla-local-$RUN_ID" --no-pager -n 100

# Cancel the launcher; its signal handler stops this run's containers.
sudo systemctl stop "vla-local-$RUN_ID"
sudo docker ps --filter "label=vla.local.run=$RUN_ID"
```

If containers remain after a forced stop, stop only IDs with this run's label.
The Python runner defaults to three hours; the service adds 120 seconds for
shutdown to the supplied `--timeout-seconds`. Set `--max-runtime-seconds` as well
for a longer per-job training/evaluation budget. A longer wait command alone
does not increase either execution budget.

For a larger Arena training/evaluation experiment, keep the selected images,
dataset, task and seed fixed. Set the desired dose and both execution budgets
explicitly, after checking the sample's measured disk/time requirements:

```bash
TRAIN_STEPS=20000          # example diagnostic dose; choose your experiment budget
EVAL_TRIALS=20             # example episode count, not a reproduced NVIDIA protocol
RUN_TIMEOUT_SECONDS=172800 # example 48-hour launcher budget
JOB_TIMEOUT_SECONDS=172800
RUN_ID="arena-full-$(date -u +%Y%m%dT%H%M%SZ)"
bash scripts/local/launch.sh "$RUN_ID" \
  --profile arena-gr1 --development-bucket "$ARENA_DEV_BUCKET" \
  --expected-role "$ARENA_EC2_ROLE" --scratch-root "$VLA_SCRATCH_ROOT" \
  --train-image "$LOCAL_TRAIN_IMAGE" --eval-image "$LOCAL_ARENA_EVAL_IMAGE" \
  --train-steps "$TRAIN_STEPS" --eval-trials "$EVAL_TRIALS" \
  --timeout-seconds "$RUN_TIMEOUT_SECONDS" --max-runtime-seconds "$JOB_TIMEOUT_SECONDS"
sudo .venv/bin/python scripts/local/wait_run.py \
  --run-dir "local-dev/runs/$RUN_ID" --timeout-seconds "$((RUN_TIMEOUT_SECONDS + 300))"
```

Use the same verification and evidence-retention commands as the sample.
The current N1.6 path saves a final checkpoint; its default `TrainSaveSteps=1000000`
must remain larger than the requested dose. `SuccessThreshold=0.0` permits collecting
diagnostic results for manual interpretation; it does not establish model quality.
FineTune gives subprocesses the remaining time within 95% of the job budget,
including setup/download time. Arena's baked rollout also has a separate hang
watchdog: by default `max(3600, episodes * 900 + 1800)` seconds, or 5.5 hours for 20 episodes.
Increasing the launcher budget alone does not change that watchdog.
Accept the local sample before choosing a larger local experiment. A separately
submitted managed sample checks registration; waiting for its capacity is not
a prerequisite for local training. LIBERO checks are optional compatibility work.

| Failure | Next action |
| --- | --- |
| Wrong identity or container cannot reach IMDS | Check attached instance profile, remove host credential overrides, and check IMDSv2 hop limit 2 |
| `AccessDenied` during config/staging | Check the exact SSM `GetParameter` names and development bucket/version-read grants |
| HF authorization error | Check the secret value and model access accepted by that token's owner |
| Image missing | Run a fresh preflight with `--pull-images`, or provide the correct ECR image URI |
| Compose preparation/CPU timeout | Inspect its phase logs and disk activity, reconcile its exact project after Docker settles, then retry preflight with a new ID |
| Initial disk reserve not met | Read the root/scratch free and required values in `disk-checks.json`; reclaim archived data on the affected disk or add storage before launching training |
| Dirty tracked files | Commit the debug changes locally, then run a new ID |
| Nonzero step/container exit or failed gate | Read that step's log and `FailureReason`; fix and use a new run ID |

Each retry starts a new four-step execution; there is no local resume guarantee.
Never overwrite a previous `status.json` to mark a retry successful. Back up logs,
`container-exits.json`, source/parameter manifests and S3 identities before removing
completed containers. Do not run broad Docker pruning while a run is active or
delete a checkpoint that has no verified durable copy.
For failed runs, retain the run directory, `run.log`, status/steps and owned
containers until diagnosis; use a new run ID after the fix. Do not stop or
terminate the instance while relying on unarchived NVMe evidence.
When reconnecting, enter the same clone and run `sudo .venv/bin/python
scripts/local/wait_run.py --latest`, or select the original run with `--run-dir`;
the detached systemd execution continues independently of the session.

To share a proven change from EC2:

```bash
git switch -c feat/arena-local-your-topic  # do this before making debug edits
# Edit and commit the changes, then stage/rebuild as needed and run the affected path.
# Verify the fresh run and archive its evidence before pushing.
export PATH="$PWD/.venv/bin:$PATH"
git push -u origin HEAD
```

A later dev → staging → production promotion is separate from this local run.
The local pipeline is named `vla-local-arena-<run-id>` by the runner; the explicit
`VLA_PIPELINE_NAME` selection above applies to the managed pipeline.

Disk observations are retained in `disk-usage.json` at 15-second intervals,
including the minimum observed free space on root and scratch. Shorter-lived
peaks may fall between samples. The evidence archiver also retains this report.


### Run in cloud — managed SageMaker

Run from this component directory after [deployment](#deploy) and
[image preparation](#build-the-container-images). A laptop or CPU machine can
submit this workflow; it does not need Docker or a local GPU.

```bash
python3 -m venv .venv  # use Python 3.10–3.12
source .venv/bin/activate
python -m pip install -e .

: "${TARGET_ACCOUNT_ID:?Repeat the deployment selection in this shell}"
: "${VLA_FOUNDATION_PROJECT:?Repeat the deployment selection in this shell}"
: "${VLA_PIPELINE_NAME:?Repeat the deployment selection in this shell}"
test "$(aws sts get-caller-identity --query Account --output text)" = "$TARGET_ACCOUNT_ID"
# Managed --train-image/--eval-image take repository-relative references.
# Preserve an explicit selection made after a new build.
: "${TRAIN_IMAGE:?Set the repository-relative training digest from the build section}"
: "${ARENA_EVAL_IMAGE:?Set the repository-relative Arena digest from the build section}"
```

In a new provisioning-machine shell, select the intended AWS profile, activate
`.venv`, and run `source local-dev/deployment-selection.txt` to restore both the
deployment and its selected image digests.
`VLA_PIPELINE_NAME` is the explicit managed pipeline to create/update. Do not
change it to a shared default when testing an isolated deployment.

These must identify the images built or selected in the target registry,
without the registry hostname. `run_arena.py` adds the
configured account's registry itself; passing a full URI would double that
prefix. Explicit selection avoids using the pair manifest's older
build-target tag. The Validate image is resolved by the pipeline's sklearn
processor; the current AWS SDK is packaged in its staged source.

The submitter needs SSM discovery, ECR image inspection, code staging in the
deployed trust bucket, pipeline/model-package APIs, and permission to pass the
configured roles. The managed step roles come from component SSM discovery.
The launcher stages the current checkout, creates/updates a pipeline version,
starts that exact version and prints its execution ARN.

#### Run a sample first

Always run a sample before a full run. A sample is a real evaluation at reduced
size, not a different kind of evaluation, and it is the only way to find plumbing
faults without paying for a full run to find them.

**A sample reduces counts and nothing else.** Fewer training steps, fewer
episodes, fewer tasks. It never alters the data, shortens an episode, or relaxes a
correctness check. Shortening `episode_length_s` would change what the policy is
asked to do, so a "success" would not mean the same thing — that is a different
experiment, not a smaller one.

Both sample commands and the shipped local recipe select **three episodes**.
The older one-episode result remains historical evidence; reproduce it only by
explicitly setting `--eval-trials 1` on the local runner.

**Sample and full are one axis; the pipeline entry point is a separate one.** They
are independent:

| | Sample scale | Full scale |
|---|---|---|
| **Train-default** (`--train-steps`) | plumbing check of all five steps | the real result |
| **Eval-only** (`--checkpoint-s3`) | plumbing check of the eval path | re-scoring existing weights |

A sample train-default run is the one that exercises the whole graph, so start
there.

##### Step 1 — run it

```bash
set -euo pipefail
MANAGED_SUBMISSION_ID="managed-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p local-dev/managed
printf '%s\n' "$MANAGED_SUBMISSION_ID" > local-dev/latest-managed-submission.txt
PYTHONPATH=src python -u scripts/run_arena.py --family gr00t --gr00t-version n16 \
    --suite arena_gr1_fridge --train-steps 200 --eval-trials 3 \
    --eval-seed 100 --threshold 0.0 \
    --train-instance ml.g6e.xlarge --eval-instance ml.g6e.xlarge \
    --acknowledge-instance-override \
    --train-image "$TRAIN_IMAGE" --eval-image "$ARENA_EVAL_IMAGE" \
    2>&1 | tee "local-dev/managed/$MANAGED_SUBMISSION_ID.log"
```

Why each value:

- `--train-steps 200` — the proven short training dose, producing a real checkpoint.
- `--eval-trials 3` — the floor. Three episodes can express 0, 1/3, 2/3, or 1.
- `--eval-seed 100` — pinned so the run is reproducible and the seed appears in
  the recorded evidence.
- `--threshold 0.0` — capability does not decide this workflow check. Report the
  observed successes honestly; every integrity check must still fail loudly.
- The instance overrides use the one-GPU types from the passing managed recipe.
  The explicit acknowledgement is required because family defaults select
  `ml.g6e.12xlarge` for both jobs. Confirm available quota/capacity; changing
  instance size can change actual GPU count and training behavior.

The family defaults request 150 GiB for training/Validate and 150 GiB for
evaluation. `MaxRuntimeSeconds` defaults to 28,800 seconds (eight hours) for each
GPU job; it is not a pipeline-wide wall-clock limit or a capacity reservation.
The poll below waits until terminal status. Inspect capacity and cancel explicitly
if the wait exceeds your allocation budget.

##### Step 2 — know what may and may not pass

A sample proves the machinery, so it must still fail on: a missing or malformed
manifest, a digest mismatch, absent or non-boolean success telemetry, an episode
count that differs from the budget, a success rate that is not realizable from
the episodes, a report that fails the shared schema, or an artifact that is not
identified by version. If any of those pass quietly, the sample has told you
nothing.

Only the capability threshold is relaxed.

##### Step 3 — wait and verify from AWS

The launcher prints an execution ARN and exits; that means **submitted**.
The submission log preserves it. `Executing`/`InProgress` can include a training
job whose secondary status is `Pending` with “waiting for capacity.”
Use [managed diagnostics](#managed-diagnostics-cancellation-and-evidence) to see
the child job's actual status. You can continue with local execution while that
job waits; do not submit duplicates just because it has not started.

In a later provisioning-machine shell, restore the target profile and deployment
selection, then recover the ARN of **this** submission and poll its terminal state:

```bash
MANAGED_SUBMISSION_ID=$(cat local-dev/latest-managed-submission.txt)
EXECUTION_ARN=$(sed -n 's/^  Execution ARN: //p' "local-dev/managed/$MANAGED_SUBMISSION_ID.log" | tail -1)
: "${EXECUTION_ARN:?Inspect the saved submission log; do not submit again to resume}"
python - "$EXECUTION_ARN" <<'PY'
import sys, time
import boto3
sm = boto3.client("sagemaker")
while True:
    result = sm.describe_pipeline_execution(PipelineExecutionArn=sys.argv[1])
    state = result["PipelineExecutionStatus"]
    print(state, result.get("FailureReason", ""), flush=True)
    if state == "Succeeded":
        break
    if state in {"Failed", "Stopped"}:
        raise SystemExit(1)
    time.sleep(30)
PY

aws sagemaker list-pipeline-execution-steps \
  --pipeline-execution-arn "$EXECUTION_ARN" \
  --query 'PipelineExecutionSteps[].{Step:StepName,Status:StepStatus,Failure:FailureReason}' \
  --output table
PYTHONPATH=src python scripts/verify_evaluation_cell.py "$EXECUTION_ARN"
```

The verifier must exit zero. Stopping the poll with Ctrl-C does not cancel cloud
jobs. To cancel the execution, use
`aws sagemaker stop-pipeline-execution --pipeline-execution-arn "$EXECUTION_ARN"`.
Inspect the training/processing job names in the step metadata to find their
CloudWatch streams under `/aws/sagemaker/TrainingJobs` and
`/aws/sagemaker/ProcessingJobs`.

Then confirm by hand, against this execution and not the newest thing in the
account:

1. All five steps reached `Succeeded` in **this** execution.
2. FineTune, SimEval and Validate have timestamped CloudWatch job activity.
   SuccessGate has a true outcome and RegisterModel has successful package
   metadata; these two control-plane steps do not have workload-container logs.
3. The registered attestation's downloaded bytes match its registered digest,
   and its validated episode count matches the requested task/episode budget.
   Use the read-only command below after saving the execution metadata.
4. The registered package points at **this** execution's promoted artifact and
   its attestation, with `eval_only` absent.

A partial run, or a number read out of a log line, does not count. If any step
above cannot be shown, the sample failed regardless of what the console says.

##### Managed diagnostics, cancellation and evidence

While waiting or investigating failure, save the exact execution's metadata and
inspect its child jobs. These commands do not require a second GPU allocation:

```bash
: "${EXECUTION_ARN:?Set the ARN printed by this submission}"
MANAGED_RUN_DIR="local-dev/managed/${EXECUTION_ARN##*/}"
mkdir -p "$MANAGED_RUN_DIR"
aws sagemaker describe-pipeline-execution --pipeline-execution-arn "$EXECUTION_ARN" \
  > "$MANAGED_RUN_DIR/execution.json"
aws sagemaker list-pipeline-execution-steps --pipeline-execution-arn "$EXECUTION_ARN" \
  > "$MANAGED_RUN_DIR/steps.json"
aws sagemaker list-pipeline-parameters-for-execution --pipeline-execution-arn "$EXECUTION_ARN" \
  > "$MANAGED_RUN_DIR/parameters.json"
python - "$MANAGED_RUN_DIR/steps.json" <<'PY'
import json, sys
import boto3
sm = boto3.client("sagemaker")
for step in json.load(open(sys.argv[1]))["PipelineExecutionSteps"]:
    print(step["StepName"], step["StepStatus"], step.get("FailureReason", ""))
    metadata = step.get("Metadata", {})
    if "TrainingJob" in metadata:
        name = metadata["TrainingJob"]["Arn"].rsplit("/", 1)[1]
        job = sm.describe_training_job(TrainingJobName=name)
        print(name, job["TrainingJobStatus"], job.get("SecondaryStatus"),
              job.get("FailureReason", ""), job.get("SecondaryStatusTransitions", []))
    if "ProcessingJob" in metadata:
        name = metadata["ProcessingJob"]["Arn"].rsplit("/", 1)[1]
        job = sm.describe_processing_job(ProcessingJobName=name)
        print(name, job["ProcessingJobStatus"], job.get("FailureReason", ""))
    if "RegisterModel" in metadata:
        package = sm.describe_model_package(ModelPackageName=metadata["RegisterModel"]["Arn"])
        print(package["ModelPackageArn"], package["ModelPackageStatus"],
              package["ModelApprovalStatus"])
PY
```

`Starting`/capacity-waiting is not completed computation. For a job name printed
above, find its stream using
`aws logs describe-log-streams --log-group-name /aws/sagemaker/TrainingJobs --log-stream-name-prefix "$JOB_NAME/"`;
use `/aws/sagemaker/ProcessingJobs` for Validate. Set `JOB_NAME` first, then
`LOG_GROUP` and `LOG_STREAM` from that response and retrieve events with the
CloudWatch command in the build diagnostics section.

Cancel with `aws sagemaker stop-pipeline-execution --pipeline-execution-arn "$EXECUTION_ARN"`,
then refresh the execution and child-job status until terminal. If a child job
remains active, explicitly stop **that execution's** named job with
`aws sagemaker stop-training-job --training-job-name "$JOB_NAME"` or
`aws sagemaker stop-processing-job --processing-job-name "$JOB_NAME"`.
Interrupting the wait script does not cancel managed compute.

After a successful terminal state, refresh the saved metadata and preserve the
verifier result with the selected source commit:

```bash
set -euo pipefail
git rev-parse HEAD > "$MANAGED_RUN_DIR/checkout-commit.txt"
PYTHONPATH=src python scripts/verify_evaluation_cell.py "$EXECUTION_ARN" \
  | tee "$MANAGED_RUN_DIR/verification.log"
```

That verifier establishes execution and registration linkage. The following
reads the small registered attestation by S3 version, compares its actual SHA256
with the package's `ContentDigest`, and checks the sample's task/episode budget.
It supports the Arena fridge and LIBERO spatial examples in this README; it
does not rerun simulation or independently rehash checkpoint contents.

```bash
python - "$EXECUTION_ARN" "$MANAGED_RUN_DIR" <<'PY'
import hashlib, json, pathlib, sys
from urllib.parse import urlparse
import boto3

arn, output = sys.argv[1], pathlib.Path(sys.argv[2])
sm, s3 = boto3.client("sagemaker"), boto3.client("s3")
steps = [step for page in sm.get_paginator("list_pipeline_execution_steps").paginate(
    PipelineExecutionArn=arn) for step in page["PipelineExecutionSteps"]]
registrations = [step["Metadata"]["RegisterModel"]["Arn"] for step in steps
                 if "RegisterModel" in step.get("Metadata", {})]
assert len(registrations) == 1, "Expected one package from this execution"
package = sm.describe_model_package(ModelPackageName=registrations[0])
source = package["ModelMetrics"]["ModelQuality"]["Statistics"]
location = urlparse(source["S3Uri"])
assert location.scheme == "s3" and location.netloc
key = location.path.lstrip("/")
head = s3.head_object(Bucket=location.netloc, Key=key)
version = head.get("VersionId")
assert version and version.lower() not in {"null", "none"}, "Attestation is not versioned"
response = s3.get_object(Bucket=location.netloc, Key=key, VersionId=version)
try:
    body = response["Body"].read()
finally:
    response["Body"].close()
digest = "sha256:" + hashlib.sha256(body).hexdigest()
assert digest == source["ContentDigest"].lower(), "Registered attestation digest mismatch"
attestation = json.loads(body)
assert attestation["execution"]["pipeline_execution_id"] == arn.rsplit("/", 1)[1]
params = {item["Name"]: item["Value"] for page in sm.get_paginator(
    "list_pipeline_parameters_for_execution").paginate(PipelineExecutionArn=arn)
    for item in page["PipelineParameters"]}
canonical_tasks = {"arena_gr1_fridge": [0], "libero_spatial": list(range(10))}
expected_tasks = (canonical_tasks[params["Suite"]] if params["EvalTaskIds"] == "all"
                  else json.loads(params["EvalTaskIds"]))
metrics = attestation["validated_metrics"]
assert metrics["suite"] == params["Suite"]
assert metrics["model_family"] == params["ModelFamily"]
assert metrics["task_ids"] == expected_tasks
assert metrics["eval_seed"] == int(params["EvalSeed"])
assert metrics["episodes"] == int(params["EvalTrials"]) * len(expected_tasks)
output.mkdir(parents=True, exist_ok=True)
(output / "registered-attestation.json").write_bytes(body)
evidence = {"execution_arn": arn, "package_arn": registrations[0],
            "attestation_uri": source["S3Uri"], "version_id": version,
            "sha256": digest, "episodes": metrics["episodes"],
            "success_rate": metrics["success_rate"]}
(output / "registered-attestation-check.json").write_text(json.dumps(evidence, indent=2) + "\n")
print(json.dumps(evidence, indent=2))
PY
```

Retain those files, build IDs/image digests, CloudWatch logs and the versioned S3
artifact/receipt/attestation referenced by this execution. The verifier checks
the registered package against that evidence. Do not delete its S3 objects or
model-package version to make a rerun look successful. A failed attempt remains
failed: diagnose it, fix/commit the source or rebuild the image as appropriate,
and submit a fresh **continuous** sample with the same deployment selection.
An eval-only rerun is a different contract and does not replace the five-step
train-to-registration acceptance.

**End of the managed walkthrough.** A verified five-step sample and saved
registration evidence complete this route. The larger training experiment and
the remaining sections are optional reference.

##### Optional — collect the full training numbers locally

The managed sample above is sufficient to test registration mechanics.
Run the larger training/evaluation experiment on EC2 following the
[local instructions](#run-locally--ec2-gpu-debugging).
Set both `--timeout-seconds` and `--max-runtime-seconds` explicitly for a long job.

Keep the model, data and evaluation recipe fixed and report loss history,
observed optimizer steps, throughput and raw success counts. Threshold 0.0
lets the diagnostic workflow finish for manual interpretation; it does not
turn a poor policy result into a capability pass. NVIDIA's task recipes use
different GPU/batch configurations, so 20,000 steps and 20 episodes are local
sanity-check counts, not a claim of reproducing its benchmark.

### Optional workflows and reference

The recommended sample ends with verification and session cleanup above.
Use the following sections for other models, larger experiments, pricing detail
or implementation work. They are not extra deployment prerequisites.

| Reference | Use it when |
| --- | --- |
| [GPU times, cost and storage](#gpu-runtime-and-cost-planning) | Sizing runs or reclaiming space |
| [Image and connector contract](#arena-images-and-connector-contract) | Changing baked code or selecting a base image |
| [GR00T AV1 decoding](#gr00t-av1-decoding) | Diagnosing an older LIBERO image's video decoder |
| [Validation runtime and lineage](#validation-runtime-and-training-lineage) | Inspecting SDK packaging or receipt claims |
| [Tests and development workflow](#tests) | Maintaining the component or reviewing a deployment |
| [Rollout video capture](#rollout-video-capture) | Seeing what the robot actually did in an episode |
| [Resource ownership and teardown](#resource-ownership-and-permanent-teardown) | Transferring or permanently removing resources |

#### Rollout video capture

Evaluation reports a success rate and per-subtask rates; by default it saves no
imagery, so a 1/3 result does not show which episode succeeded. Isaac Lab Arena
records rollouts itself through `gymnasium.wrappers.RecordVideo`, wired in
`isaaclab_arena/evaluation/policy_runner.py`. This component forwards the
`--video` and `--video_dir` flags and poses the recorded camera; there is no
capture code here to maintain.

**Off by default, and that default is deliberate.** `--video` makes Arena build the
scene with `render_mode="rgb_array"` instead of `None`. That is a different
environment construction from the argv behind the
[accepted reference runs](#accepted-reference-runs), and rendering costs time.
Seeding happens before the `RecordVideo` wrap, so RNG ordering is unaffected, but
treat a recorded rollout as a diagnostic rather than a reference measurement.

Enable it on the managed pipeline with `--record-video`:

```bash
vla run --deployment arena-review --cell gr00t-n16-arena --mode managed \
  --record-video \
  --max-runtime-seconds 14400 --train-steps 200 --eval-trials 3 --threshold 0.0 \
  --instance FineTune=ml.g6e.12xlarge --instance SimEval=ml.g6e.12xlarge
```

Or on the direct Arena submitter:

```bash
PYTHONPATH=src python scripts/submit_simeval.py --record-video true \
  --eval-image "$EVAL_IMAGE" --checkpoint-s3 "$CHECKPOINT_S3"
```

Both set `EVAL_RECORD_VIDEO=true` in the SimEval container. The pipeline parameter
is `EvalRecordVideo` (`"true"`/`"false"`), so a hand-started
`StartPipelineExecution` can set it too.

The mp4 is written to `$SM_MODEL_DIR/videos/`, which SageMaker tars into the
SimEval `ModelArtifacts` beside `metrics.json`. Retrieve it from the handoff
bucket:

```bash
aws s3 cp "s3://<handoff-bucket>/eval/v1/<simeval-job-name>/output/model.tar.gz" .
tar -xzf model.tar.gz && ls videos/
```

#### Why the recorded camera is posed here

Headless, the pinned Isaac Lab never applies the task's `ViewerCfg` to the camera
`RecordVideo` records (`/OmniverseKit_Persp`).
`ViewportCameraController.update_view_location()` resolves the right eye and lookat,
then hands them to `sim.set_camera_view()`, which only updates visualizers, and a
headless run has none. Requesting `--visualizer kit` doesn't help, because
`KitVisualizer` skips viewport setup when headless. The first recorded rollout
showed exactly this: a valid 1305-frame mp4 in which the kitchen was a few grey
pixels, even though the log showed a correctly framed
`viewer=ViewerCfg(eye=(2.55, -2.08, 2.52), lookat=<the ranch bottle>)`.

So when recording, `eval_entry.py` launches policy_runner through
`entrypoints/eval/isaac_arena/gr00t/arena_video_runner.py`. Just before `RecordVideo`
wraps the env, that wrapper authors the env's own resolved eye and lookat onto the
camera prim, then runs policy_runner unchanged. It backports upstream Isaac Lab's
later `set_kit_renderer_camera_view()`. The framing still comes from the task:
nothing in the wrapper is scene-specific. Unrecorded runs don't use the wrapper.

Two limits worth knowing before relying on this:

- **One continuous mp4, not one file per episode.** Arena wires
  `step_trigger=(step == 0)` with `video_length = num_episodes * max_episode_length`,
  so a single recording spans every episode and you scrub to find boundaries.
  Per-episode files would need `episode_trigger`, an upstream Arena change.
- **Editing the eval entry needs an image rebuild.** The connector's ENTRYPOINT runs
  the baked `/workspace/eval_entry.py`, so changing how these flags are passed
  requires a new connector image and digest. See
  [image and connector contract](#arena-images-and-connector-contract).

#### Other managed entry points

These are optional alternatives to the main N1.6 train-default sample, not
additional commands required for a fresh deployment. Run only the alternative
you intend. The suite supplies the task, embodiment, object selection, and policy-config path. The
entrypoint patches the config's example model path to the mounted checkpoint;
it does not search for an alternative config when one is missing.

Use a distinct pipeline name for each alternative. The inline selections below
leave the main `VLA_PIPELINE_NAME` unchanged. These examples explicitly select
one-GPU `ml.g6e.8xlarge` instances; check their Training quota and capacity
separately from the main sample's `ml.g6e.xlarge` quota. The launchers return an
execution ARN immediately. Inspect and verify that exact execution using the
managed diagnostics above.

**N1.7/Arena train-default:**

```bash
VLA_PIPELINE_NAME="$VLA_FOUNDATION_PROJECT-$VLA_ENVIRONMENT-arena-n17" \
  PYTHONPATH=src python scripts/run_arena.py --family gr00t --gr00t-version n17 \
    --suite arena_gr1_fridge --train-steps 200 --eval-trials 3 \
    --train-instance ml.g6e.8xlarge --eval-instance ml.g6e.8xlarge \
    --acknowledge-instance-override \
    --train-image "$TRAIN_IMAGE" --eval-image "$ARENA_EVAL_IMAGE"
```

**N1.6/Arena eval-only:** first set `CHECKPOINT_S3_URI` to the exact checkpoint
object in a versioned S3 bucket.

```bash
: "${CHECKPOINT_S3_URI:?Set the S3 model.tar.gz URI to evaluate}"
VLA_PIPELINE_NAME="$VLA_FOUNDATION_PROJECT-$VLA_ENVIRONMENT-arena-eval-only" \
  PYTHONPATH=src python scripts/run_arena.py --family gr00t --gr00t-version n16 \
    --suite arena_gr1_fridge --checkpoint-s3 "$CHECKPOINT_S3_URI" \
    --eval-trials 3 \
    --train-instance ml.g6e.8xlarge --eval-instance ml.g6e.8xlarge \
    --acknowledge-instance-override \
    --train-image "$TRAIN_IMAGE" --eval-image "$ARENA_EVAL_IMAGE"
```

Eval-only checkpoints must include the manifest expected by this component and
match the selected model version. Eval-only is not evidence of a continuous
train-to-registration run.
Old checkpoints without `training_lineage.json` are permitted only through this
explicit eval-only contract; new GR00T train-default validations require lineage.
For a checkpoint outside Foundation's models bucket, configure the component's
`additional_input_s3_arns` with both the input bucket ARN and its allowed object
prefix ARN, then review/apply the component plan with the same deployment variables.
External bucket/KMS owners must also permit those step-role reads. No extra input
grant is needed for the main train-default sample.

##### Standalone Arena evaluation

For a faster managed debugging iteration, `submit_simeval.py` submits **only
SimEval** against an existing fine-tuned checkpoint. It does not run Validate,
the gate or registration. Run from the provisioning machine after restoring
the deployment selection and activating `.venv`.

Use `CHECKPOINT_S3_URI` from the previous FineTune job's
`ModelArtifacts.S3ModelArtifacts` (available through `describe-training-job`).
The exact object must exist in a versioned bucket readable by the workload role;
the external-input permission guidance above also applies here. Stage the current
source, then submit:

```bash
set -euo pipefail
: "${CHECKPOINT_S3_URI:?Set the exact trained model.tar.gz S3 URI}"
: "${LOCAL_ARENA_EVAL_IMAGE:?Restore the full connector digest URI}"
mkdir -p local-dev
PYTHONPATH=src python - <<'PY'
from pathlib import Path
from vla_pipeline.common.sourcedir import stage
from vla_pipeline.config import load_config
from vla_pipeline.runner import upload_directory
uri = upload_directory(load_config(), stage("gr00t"))
Path("local-dev/standalone-source-uri.txt").write_text(uri + "\n")
PY
EVAL_SOURCE_DIR_URI=$(cat local-dev/standalone-source-uri.txt)
python scripts/submit_simeval.py --gr00t-version n16 \
  --suite arena_gr1_fridge --eval-trials 3 --eval-seed 100 \
  --eval-image "$LOCAL_ARENA_EVAL_IMAGE" \
  --checkpoint-s3 "$CHECKPOINT_S3_URI" \
  --eval-source-dir "$EVAL_SOURCE_DIR_URI"
```

Omitting `--instance` uses the family manifest's evaluation default, currently
`ml.g6e.12xlarge` (four GPUs). To select another instance, supply
`--instance ml.g6e.8xlarge --acknowledge-instance-override` after checking quota
and capacity. The script prints the resolved instance and exact job name.
Output goes under the deployed `pipeline_prefix/eval` in the configured models
bucket. The prefix comes from SSM; keep Terraform's policy and SSM configuration
in agreement when deploying with a custom prefix.

Save the printed job name as `SIMEVAL_JOB`. Submission alone is not completion:

```bash
aws sagemaker describe-training-job --region us-east-1 \
  --training-job-name "$SIMEVAL_JOB" \
  --query '{status:TrainingJobStatus,reason:FailureReason,instance:ResourceConfig.InstanceType,output:OutputDataConfig.S3OutputPath,artifact:ModelArtifacts.S3ModelArtifacts}'
```

For a completed job, inspect its artifact and verify that it was published under
the selected prefix. A failed or Pending job does not establish that output
publication works.

##### Published-checkpoint diagnostic on the local GPU

This optional N1.6 positive control evaluates NVIDIA's published fridge checkpoint.
It performs no training and produces diagnostic metrics without the pipeline's
validation/promotion chain. Run it when the GPU is free, after the normal host
setup and connector selection. A new directory forces the first-download path
without deleting another run's cache. Keep at least the normal scratch headroom.

On EC2, restore the deployment selection, activate `.venv`, and run:

```bash
set -euo pipefail
: "${VLA_SCRATCH_ROOT:?Restore the verified scratch directory}"
: "${LOCAL_ARENA_EVAL_IMAGE:?Restore the rebuilt connector digest URI}"
: "${HF_SECRET_NAME:?Restore the configured HF secret name}"
POSCTRL_ID="arena-posctrl-$(date -u +%Y%m%dT%H%M%SZ)"
POSCTRL_ROOT="$VLA_SCRATCH_ROOT/$POSCTRL_ID"
mkdir "$POSCTRL_ROOT"
mkdir "$POSCTRL_ROOT/cache" "$POSCTRL_ROOT/output"
git rev-parse HEAD > "$POSCTRL_ROOT/source-commit.txt"
printf '%s\n' "$LOCAL_ARENA_EVAL_IMAGE" > "$POSCTRL_ROOT/image.txt"
export HF_TOKEN
HF_TOKEN=$(aws secretsmanager get-secret-value --region us-east-1 \
  --secret-id "$HF_SECRET_NAME" --query SecretString --output text)
# Resolve once, then pin the diagnostic to that exact external revision.
export N16_POSCTRL_CKPT_REPO=nvidia/GN1.6-Tuned-Arena-GR1-PlaceItemCloseDoor-Task
export N16_POSCTRL_CKPT_REV
N16_POSCTRL_CKPT_REV=$(python - <<'PY'
import json, os, re
from urllib.request import Request, urlopen
request = Request("https://huggingface.co/api/models/" + os.environ["N16_POSCTRL_CKPT_REPO"],
                  headers={"Authorization": "Bearer " + os.environ["HF_TOKEN"]})
with urlopen(request, timeout=60) as response:
    sha = json.load(response)["sha"]
assert re.fullmatch(r"[0-9a-f]{40}", sha), sha
print(sha)
PY
)
export EVAL_SIM_CONFIG
EVAL_SIM_CONFIG=$(PYTHONPATH=src python - <<'PY'
import json
from vla_pipeline.arena import resolve_runtime
from vla_pipeline.registry import resolve_suite
print(json.dumps(resolve_runtime(resolve_suite("arena_gr1_fridge"), "gr00t", "n16")))
PY
)
printf '%s\n' "$N16_POSCTRL_CKPT_REPO@$N16_POSCTRL_CKPT_REV" > "$POSCTRL_ROOT/checkpoint.txt"
printf '%s\n' "$EVAL_SIM_CONFIG" > "$POSCTRL_ROOT/config.json"
sudo --preserve-env=HF_TOKEN,N16_POSCTRL_CKPT_REPO,N16_POSCTRL_CKPT_REV,EVAL_SIM_CONFIG \
  docker run -d --name "$POSCTRL_ID" --gpus all --shm-size 8g \
  --mount "type=bind,source=$POSCTRL_ROOT/cache,target=/tmp/gn16_posctrl_ckpt" \
  --mount "type=bind,source=$POSCTRL_ROOT/output,target=/opt/ml/model" \
  --env HF_TOKEN --env N16_POSCTRL_CKPT_REPO --env N16_POSCTRL_CKPT_REV \
  --env EVAL_SIM_CONFIG --env ARENA_CONNECTOR=groot \
  --env EVAL_GR00T_VERSION=n16 --env EVAL_POSCTRL_N16=true \
  --env USE_GROOT_SERVER=true --env SM_HP_USE_GROOT_SERVER=true \
  --env EVAL_SUITE=arena_gr1_fridge --env EVAL_TRIALS=3 --env EVAL_SEED=100 \
  "$LOCAL_ARENA_EVAL_IMAGE"
unset HF_TOKEN
declare -p POSCTRL_ID POSCTRL_ROOT >> local-dev/deployment-selection.txt
sudo docker logs --follow "$POSCTRL_ID"
```

Ctrl-C stops following logs while the detached container continues. Reconnect by
restoring the selection and repeating `docker logs --follow "$POSCTRL_ID"`.
When it finishes, inspect its exit state and saved output:

```bash
sudo docker inspect "$POSCTRL_ID" --format '{{json .State}}'
sudo cat "$POSCTRL_ROOT/cache/.resolved_commit"
sudo cat "$POSCTRL_ROOT/output/metrics.json"
sudo docker logs "$POSCTRL_ID" > "$POSCTRL_ROOT/run.log" 2>&1
```

Require exit zero, the requested repository/revision in `.resolved_commit`,
three completed episodes and the reported task/configuration. An empty cache
must download and resolve correctly on its first attempt. Save failed attempts;
do not describe a later warm-cache retry as a successful cold download. The
diagnostic writes schema-v3 metrics with `policy_type: positive_control`. The
normal Validate/Register path requires `policy_type: checkpoint`, so these
metrics are diagnostic only and must not be submitted for registration.
No task-success threshold is implied by the positive-control name.

##### Optional managed LIBERO

`scripts/run_libero.py` accepts the same repository-relative `--train-image`
and `--eval-image` references as the Arena launcher. Select the published
image explicitly; no manifest edit is needed. Both launchers resolve supplied
tags to digests before submitting train/eval jobs. For the N1.7/LIBERO spatial
sample, reuse the full GR00T image selected in the build section:

```bash
VLA_PIPELINE_NAME="$VLA_FOUNDATION_PROJECT-$VLA_ENVIRONMENT-gr00t-libero" \
  PYTHONPATH=src python scripts/run_libero.py --family gr00t \
    --suite libero_spatial --train-steps 200 --eval-trials 3 \
    --eval-seed 1000 --eval-task-ids all --threshold 0.0 \
    --train-instance ml.g6e.8xlarge --eval-instance ml.g6e.8xlarge \
    --train-image "$TRAIN_IMAGE" --eval-image "$TRAIN_IMAGE" \
    --acknowledge-instance-override --no-wait
```

This requests 150 GiB per GPU job and three episodes for each of ten tasks.
Without the explicit instance overrides, the family selects four-GPU
`ml.g6e.12xlarge` instances.
Wait and verify its printed execution ARN using the same AWS diagnostics above.
For another LIBERO family, set `LIBERO_FAMILY` to `molmoact2` or `openvla` and
`LIBERO_BUILD_TAG` to its successful build tag, then resolve its image:

```bash
: "${LIBERO_FAMILY:?Select molmoact2 or openvla}"
: "${LIBERO_BUILD_TAG:?Set the successful build tag for this family}"
LIBERO_DIGEST=$(aws ecr describe-images --repository-name "vla/$LIBERO_FAMILY" \
  --image-ids imageTag="$LIBERO_BUILD_TAG" --query 'imageDetails[0].imageDigest' --output text)
[[ "$LIBERO_DIGEST" =~ ^sha256:[0-9a-f]{64}$ ]] || { echo "Image digest not found" >&2; exit 1; }
LIBERO_IMAGE="vla/$LIBERO_FAMILY@$LIBERO_DIGEST"
VLA_PIPELINE_NAME="$VLA_FOUNDATION_PROJECT-$VLA_ENVIRONMENT-$LIBERO_FAMILY-libero" \
  PYTHONPATH=src python scripts/run_libero.py --family "$LIBERO_FAMILY" \
    --suite libero_spatial --train-steps 200 --eval-trials 3 \
    --eval-seed 1000 --eval-task-ids all --threshold 0.0 \
    --train-image "$LIBERO_IMAGE" --eval-image "$LIBERO_IMAGE" --no-wait
```

Family manifests supply instance and volume defaults. To select other capacity,
add `--train-instance`, `--eval-instance` and `--acknowledge-instance-override`.
MolmoAct2 requests 250 GiB for each GPU job; OpenVLA requests 300 GiB for training
and 200 GiB for evaluation. The launcher checks the selected instance's storage
limit, not free space inside every container path. See the
[instance matrix and MolmoAct2 storage-failure retest](#cell-instance-and-storage-matrix)
before selecting an override. These compatibility runs are separate from the
main Arena acceptance.

#### Run the three LIBERO samples

After the clone/setup instructions above, use **one profile at a time**:
`openvla-libero`, then `molmoact2-libero`, then `gr00t-libero`.
Each defaults to 200 training steps, three episodes per each of ten spatial tasks,
seed 1000 and threshold 0.0. GR00T uses N1.7. MolmoAct2 retains its managed
`TrainSuite=unified` contract; the evaluation suite remains spatial.
`parameters.json` records the complete effective recipe. The start request in
`execution-parameters.json` omits unchanged empty defaults: SageMaker rejects
empty API overrides, while declared pipeline defaults remain valid.

All three profiles have independently verified four-step local reference runs,
listed in the runtime table below. The full GR00T image build includes its AV1
decoder, and the staged evaluator audits the backbone actually loaded for
inference. Use both the selected source and its compatible images.

From the component directory:

```bash
PROFILE=openvla-libero  # repeat later with molmoact2-libero, then gr00t-libero
: "${ARENA_DEV_BUCKET:?Repeat the deployment selection in this EC2 shell}"
: "${ARENA_EC2_ROLE:?Repeat the deployment selection in this EC2 shell}"
: "${VLA_SCRATCH_ROOT:?Set the verified scratch directory}"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
test "$ACCOUNT_ID" = "$TARGET_ACCOUNT_ID"
PREFLIGHT_ID="pf-$PROFILE-$(date -u +%Y%m%dT%H%M%SZ)"
# Build all desired families first, using the image-build section above.
BUILD_TAG=your-published-build-tag
FAMILY="${PROFILE%-libero}"
ECR_REGISTRY="$ACCOUNT_ID.dkr.ecr.${AWS_DEFAULT_REGION:-us-east-1}.amazonaws.com"
LIBERO_DIGEST=$(aws ecr describe-images --repository-name "vla/$FAMILY" \
  --image-ids imageTag="$BUILD_TAG" --query 'imageDetails[0].imageDigest' --output text)
if ! [[ "$LIBERO_DIGEST" =~ ^sha256:[0-9a-f]{64}$ ]]; then
  echo "ECR did not return the selected LIBERO image digest." >&2
  exit 1
fi
LIBERO_IMAGE="$ECR_REGISTRY/vla/$FAMILY@$LIBERO_DIGEST"
IMAGE_ARGS=(--train-image "$LIBERO_IMAGE" --eval-image "$LIBERO_IMAGE")

bash scripts/local/launch.sh "$PREFLIGHT_ID" \
  --profile "$PROFILE" --development-bucket "$ARENA_DEV_BUCKET" \
  --expected-role "$ARENA_EC2_ROLE" --scratch-root "$VLA_SCRATCH_ROOT" \
  "${IMAGE_ARGS[@]}" --preflight-only --pull-images
sudo .venv/bin/python scripts/local/wait_run.py \
  --run-dir "local-dev/runs/$PREFLIGHT_ID"

RUN_ID="$PROFILE-$(date -u +%Y%m%dT%H%M%SZ)"
bash scripts/local/launch.sh "$RUN_ID" \
  --profile "$PROFILE" --development-bucket "$ARENA_DEV_BUCKET" \
  --expected-role "$ARENA_EC2_ROLE" --scratch-root "$VLA_SCRATCH_ROOT" \
  "${IMAGE_ARGS[@]}" --train-steps 200 --eval-trials 3
sudo .venv/bin/python scripts/local/wait_run.py \
  --run-dir "local-dev/runs/$RUN_ID"
```

Require each command to succeed before the next. `wait_run.py` observes the
detached service and runs the independent verifier after successful completion.
It does not accept a launch submission as proof of a completed pipeline.
Use another terminal to follow `run.log` if desired.

The newly built family image supplies both training and LIBERO evaluation.
GR00T's full build includes the AV1 decoder fix. Explicit overrides are applied
before any historical default-tag lookup. Stop on a failed ECR lookup.

For each profile, follow the shared [verification](#verify-the-completed-run-before-cleanup)
and [archive](#archive-an-accepted-local-run) sections **once**, then return
here to select the next profile. Do not archive/remove its containers here and
then attempt live verification again. Record timings and publish any verified
fix before the next cell. If a cell fails, retain the evidence, fix the cause
and use a new run ID.

These three compatibility profiles are optional for a fresh Arena deployment.
For a compatibility campaign that includes them, review their results and publish
verified fixes before starting the larger Arena experiment. Keep managed
registration acceptance separate from these local GPU checks.

### GPU, runtime and cost planning

**The full local Arena experiment took 5h38m12s and approximately $25.53 in EC2
compute on one `g6e.8xlarge`.** This was GR00T N1.6, 20,000 training steps and
20 fridge-task episodes, including validation. Budget additional time and cost
for deployment, image builds/downloads, verification, retries and idle time.
The separate-account review measured all three builds and the local sample
below. Its supplied EC2 host bypassed the new-host launch recipe, and preflight
needed assistance; this does not prove every clean-account setup path.

#### Cell instance and storage matrix

Use these selections for the documented **200-step samples**, not as measured
minimum hardware requirements. A passing historical run establishes that specific
source, image and storage configuration; it does not validate the current checkout.
Local rows use one EC2 host with the scratch layout below. Managed entries use
one SageMaker instance for each GPU step.

| Cell | Local debugging | Managed FineTune → SimEval | Requested train / eval volume (GiB) | Evidence and limits |
| --- | --- | --- | --- | --- |
| GR00T N1.6 / Arena GR1 fridge | `g6e.8xlarge` | `ml.g6e.xlarge` → `ml.g6e.xlarge` | 150 / 150 | Historical local samples/full experiment and managed five-step samples passed. This is the main walkthrough. |
| GR00T N1.7 / LIBERO spatial | `g6e.8xlarge` | `ml.g6e.8xlarge` → `ml.g6e.8xlarge` in the example | 150 / 150 | Local passed. The recorded managed success used `ml.g6e.16xlarge` → `ml.g6.2xlarge`; the example's pair is not established by that result. |
| MolmoAct2 / LIBERO spatial | `g6e.8xlarge` | `ml.g6e.12xlarge` → `ml.g6e.12xlarge` (family defaults used by the example) | 250 / 250 | Local passed. Managed FineTune completed on `.16xlarge` and uploaded its checkpoint; downstream SimEval was quota-blocked. Full managed completion and the default pair remain unverified. |
| OpenVLA / LIBERO spatial | `g6e.8xlarge` | `ml.g6e.12xlarge` → `ml.g5.2xlarge` | 300 / 200 | Local passed. Managed values are the family defaults, not a newly verified result. |
| GR00T N1.7 / Arena GR1 fridge | No accepted local run listed here | `ml.g6e.8xlarge` → `ml.g6e.8xlarge` in the optional example | 150 / 150 | Optional supported path; the N1.6 results do not validate it. |
| Standalone GR00T N1.6 / Arena | Use the normal local pipeline for local acceptance | No FineTune → `ml.g6e.12xlarge` by default | — / 150 | Evaluates an existing checkpoint and publishes evaluation output. It does not run Validate or RegisterModel; a queued job is not a pass. |

The family manifests select `.12xlarge` for both GR00T GPU steps when no
overrides are supplied. That size has **four** L40S GPUs; `.xlarge`, `.8xlarge`
and `.16xlarge` each have **one**. GPU memory is per device, not one combined
pool. MolmoAct2 currently launches one training process even on a four-GPU host.
Use the [measured managed alternatives](#additional-measured-managed-samples)
when selecting capacity, and record the actual instance for each step.

<a id="molmoact2-managed-disk-failure-and-next-run"></a>

##### MolmoAct2 managed disk failure and follow-up

An observed 2026-09-17 managed execution used `ml.g6e.8xlarge`,
requested 250 GiB, and reached step 80 of 200. CloudWatch recorded
`Checkpoint policy after step 80`, followed by:

```text
safetensors_rust.SafetensorError: Error while serializing: I/O error: No space left on device (os error 28)
```

The executed source and logged command both selected `/opt/vla/train-out`.
The final model object was absent. This confirms a checkpoint-write storage
failure; the generic `train_entry.py` exit code alone did not explain it.
No filesystem headroom snapshot was captured in that job, so it does **not**
establish that this instance type lacks enough total disk or GPU memory.

**Follow-up checked on 2026-09-18:** corrected source completed
FineTune with a 200-step request on `ml.g6e.16xlarge` and uploaded a
29,437,358,968-byte checkpoint. AWS recorded 1,484 billable seconds (**24m44s**).
The execution then failed to create SimEval because the account's
single `ml.g6e.12xlarge` Training slot was occupied. This establishes the
corrected training/upload path on that instance; it does not establish a full
managed MolmoAct2 pipeline or a minimum instance size.

The correction writes intermediate checkpoints to `/tmp/molmoact2-train-out`;
the final model still goes to `/opt/ml/model`. The launcher stages the trainer
from source, so this storage fix does not require rebuilding the MolmoAct2 image.
For the next full sample, use 200 training steps, all ten spatial tasks, three
episodes per task, seed 1000, threshold 0 and 250 GiB requested per GPU step.
Choose available instances using the matrix and check quota for both steps.
With `vla run`, these are the ordinary `--instance FineTune=...`,
`--instance SimEval=...` and `--volume-gb STEP=250` choices.

Retain the trainer's `[storage]` records for `/`, `/opt/vla`, `/tmp` and the
final model directory. Full acceptance still requires downstream steps passing.
Do not mark `.8xlarge` unsupported or `.12xlarge` sufficient based on a disk-path
error or a quota refusal.

AWS documents `/tmp` as training scratch. On NVMe-backed training instances,
`VolumeSizeInGB` does not add EBS capacity: available instance storage is fixed.
The G6e specifications list 900 GB for `.8xlarge` and 3,800 GB for `.12xlarge`,
but those totals do not prove headroom at a particular container path.
See [SageMaker storage guidance](https://docs.aws.amazon.com/sagemaker/latest/dg/model-train-storage-tips-considerations.html),
[training resource configuration](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_ResourceConfig.html)
and [G6e specifications](https://aws.amazon.com/ec2/instance-types/g6e/).

#### Instance and storage requirements

The tested local host has **one NVIDIA L40S, 48 GB nominal GPU memory, 32 vCPUs
and 256 GiB system RAM**. FineTune and SimEval run sequentially on that GPU;
Validate uses CPU on the same host, so its duration still incurs the EC2 host
rate. This is a tested configuration, not a measured minimum VRAM requirement.
Smaller-memory GPUs have not been accepted for the full local Arena recipe.

Prices below are **USD, Linux On-Demand, `us-east-1` (N. Virginia), checked
2026-09-17**, before tax, discounts or credits. EC2 rates assume shared tenancy.
SageMaker prices are for the indicated job component, not Studio.

| Use | Instance | GPUs / nominal GPU memory | vCPUs / system RAM | Compute rate |
| --- | --- | --- | --- | --- |
| Tested local host; [EC2 launch instructions](#select-and-launch-a-new-host) | `g6e.8xlarge` | 1 × L40S / 48 GB | 32 / 256 GiB | $4.52856/hour (about $4.53) |
| Managed sample's explicit FineTune and SimEval overrides | `ml.g6e.xlarge` | 1 × L40S / 48 GB | 4 / 32 GiB | $2.61/hour per Training job instance |
| GR00T family defaults when no instance overrides are supplied | `ml.g6e.12xlarge` | **4 × L40S / 192 GB total** | 48 / 384 GiB | $13.12/hour per Training job instance |
| Managed Validate | `ml.m5.large` | None | 2 / 8 GiB | $0.115/hour per Processing job instance |

The managed sample command explicitly selects the one-GPU instances and
acknowledges the override. Leaving the family defaults unchanged requests four
GPUs per job; neither the one-GPU timings nor their costs apply to that choice.

For local disk, the launch recipe provisions an **800 GiB gp3 root volume** and
uses the instance's **900 GB total NVMe instance storage** for scratch. Keep
the selected cell's [initial disk reserve](#local-storage-and-retention) on both
filesystems after images are present. GPU
memory and disk capacity are separate requirements. Follow the
[EC2 storage instructions](#run-locally--ec2-gpu-debugging); archive evidence
before stopping the host because NVMe instance storage is ephemeral.

#### Measured workload times and approximate compute cost

All local rows below used one `g6e.8xlarge` with images already present.
The separate-account supplied host was in us-east-2; earlier hosts were in us-east-1.
Model/data cache state varied; the
[reference results below](#accepted-reference-runs) describe the
accepted recipes and cache details. These are individual observations,
not cold-start guarantees or comparable model-quality benchmarks.

| Local cell | Training steps / evaluation episodes | FineTune | SimEval | Validate | Pipeline elapsed | Approx. EC2 compute |
| --- | --- | --- | --- | --- | --- | --- |
| GR00T N1.6 / Arena, September 17 review | 200 / **3** | 11m48s | 9m56s | 9m45s | **31m38s** | **$2.39** |
| GR00T N1.7 / LIBERO spatial, September 17 review | 200 / 3 × 10 tasks | 14m42s | 18m19s | 8m04s | **41m12s** | **$3.11** |
| MolmoAct2 / LIBERO spatial, September 17 review | 200 / 3 × 10 tasks | 25m42s | 19m15s | 31m01s | **1h16m06s** | **$5.74** |
| GR00T N1.6 / Arena sample, rebuilt connector on supplied host | 200 / **3** | 11m52s | 9m09s | 6m17s | **27m27s** | **$2.07** |
| GR00T N1.6 / Arena sample, separate-account supplied host | 200 / **3** | 10m38s | 9m34s | 7m14s | **27m32s** | **$2.08** |
| GR00T N1.6 / Arena sample | 200 / **1** | 14m59s | 8m25s | 7m43s | **31m12s** | **$2.35** |
| GR00T N1.6 / Arena full experiment | 20,000 / 20 | 5h08m08s | 23m19s | 6m45s | **5h38m12s** | **$25.53** |
| OpenVLA-OFT / LIBERO spatial | 200 / 3 × 10 tasks | 18m31s | 12m17s | 10m13s | **41m01s** | **$3.10** |
| MolmoAct2 / LIBERO spatial | 200 / 3 × 10 tasks | 24m04s | 19m19s | 23m40s | **67m04s** | **$5.07** |
| GR00T N1.7 / LIBERO spatial | 200 / 3 × 10 tasks | 10m28s | 16m42s | 7m26s | **34m37s** | **$2.62** |

FineTune includes setup, downloads and checkpoint packaging, not just optimizer
steps. The full Arena trainer reported **5h00m51s** of optimization within its
5h08m08s FineTune step. Pipeline elapsed includes all four local steps; separately
rounded step times need not sum exactly. Cost is elapsed host hours × $4.52856,
including 6–8 seconds of launcher preparation where recorded. The historical
Arena sample uses pipeline time because that preparation measurement is absent.
Neither estimate is a billing-invoice measurement.

The September 17 review rows used rebuilt connector/MolmoAct2
images and existing host caches. All four local steps and independent verification
passed; the observed robot successes were 1/3, 10/30 and 30/30 respectively.
These are workflow samples with threshold zero, not comparative model-quality
benchmarks. Their separate new-image preflights took about 8m16s for Arena and
6m30s for MolmoAct2. They precede the subsequent MolmoAct2 intermediate-checkpoint
storage correction; retain that distinction when comparing disk measurements.

The earlier three LIBERO checks totaled **about 2h23m and $10.80** in workload
compute; adding the full Arena experiment gives **about eight hours and $36.33**.
These totals exclude failed attempts, time between runs and setup. The earlier Arena sample used **one episode**. The separate-account **three-episode**
sample measured 27m32s after preflight recovery. Its trainer reported 3m48s of
optimization within the 10m38s FineTune step.
The updated connector sample measured 27m27s on the same supplied host, with
3m48s of trainer computation within the 11m52s FineTune step. Its separate
preflight took 7m52s, including a new connector pull and 153s of container
preparation; the 600-second preparation allowance succeeded without retry.
Existing training/validation images and caches were retained, so this is not a
completely fresh-host measurement.

The supplied-host costs use the current us-east-2 `g6e.8xlarge` rate,
which was also $4.52856/hour on 2026-09-17. They cover pipeline time only;
the rebuilt connector's separate 7m52s preflight adds approximately $0.59.

For a **managed** reference, the previously accepted five-step N1.6/Arena sample
(200 steps, one episode, one `ml.g6e.xlarge` per GPU job) took **2h25m15s** from
execution creation through registration. AWS recorded **23m00s FineTune +
37m10s SimEval billable GPU time** and **12m32s Validate Processing elapsed**.
At the rates above, those job durations imply **about $2.64 compute**, excluding
job storage and other charges; the CPU portion uses elapsed time as an estimate.
The GPU jobs also spent **69m24s in Pending/provisioning**, outside their reported
billable time. That status alone does not identify how much was capacity wait.
This historical sample is a scheduling reference; it does not establish the
runtime of a fresh-account deployment, the three-episode sample or full managed
training.

#### Additional measured managed samples

AWS records checked on 2026-09-17 show the following **seven individual successful
five-step executions launched on 2026-09-16**, including registration. These
historical runs used their recorded source and image versions; they do not
validate later edits in this checkout. Each uses one instance per job and
`ml.m5.large` for Validate. Arena uses the N1.6 fridge recipe, seed 100 and
three episodes; LIBERO uses N1.7/spatial, seed 1000 and three episodes per each
of ten tasks. All train for 200 steps with threshold 0.0.

| Cell (200 training steps) | FineTune instance → SimEval instance | FineTune billable | SimEval billable | Validate elapsed | Pipeline wall time | Approx. compute |
| --- | --- | --- | --- | --- | --- | --- |
| N1.6 / Arena, 3 episodes | `ml.g6e.16xlarge` → `ml.g6.2xlarge` | 21m00s | 22m00s | 14m38s | 1h10m08s | $3.79 |
| N1.6 / Arena, 3 episodes | `ml.g6e.16xlarge` → `ml.g6.4xlarge` | 16m59s | 19m29s | 15m23s | 2h49m41s | $3.25 |
| N1.6 / Arena, 3 episodes | `ml.g6e.2xlarge` → `ml.g5.2xlarge` | 19m44s | 38m06s | 13m07s | 11h27m23s | $1.91 |
| N1.6 / Arena, 3 episodes | `ml.g6e.8xlarge` → `ml.g5.8xlarge` | 17m18s | 21m25s | 13m37s | 3h44m43s | $2.75 |
| N1.6 / Arena, 3 episodes | `ml.p4d.24xlarge` → `ml.g5.4xlarge` | 12m20s | 33m20s | 9m06s | 4h54m16s | $6.34 |
| N1.6 / Arena, 3 episodes | `ml.p4de.24xlarge` → `ml.g5.2xlarge` | 11m22s | 26m00s | 8m46s | 6h04m53s | $6.65 |
| N1.7 / LIBERO spatial, 30 episodes | `ml.g6e.16xlarge` → `ml.g6.2xlarge` | 26m45s | 33m08s | 15m08s | 4h23m23s | $4.93 |

FineTune and SimEval columns are AWS **billable seconds**, including image/data
transfer and output packaging within that interval. They are not optimizer or
rollout-only times. Validate uses Processing start-to-end elapsed time; pipeline
wall time runs from execution creation through the final step. Compute estimates
use the matching us-east-1 Training and Processing On-Demand price-list rates
checked on 2026-09-17, multiplied by instance count and each job's measured
usage. They exclude storage, builds, networking, idle resources and failed
attempts, and are not invoiced charges.

These runs spent **9m17s to 10h14m00s in Pending/provisioning** across their two
GPU jobs, outside reported GPU billable time. The recorded status does not
separate capacity waiting from all other provisioning work. Different hardware,
source versions and download conditions make these individual observations,
not a hardware speed ranking or an averaged performance promise.

The `g5` evaluation instances have one A10G with 24 GB nominal GPU memory;
`g6` has one 24 GB L4.
The listed `g6e.2xlarge`, `.8xlarge` and `.16xlarge` training instances each
have one 48 GB L40S. `p4d.24xlarge` and `p4de.24xlarge` each have eight A100s
(40 GB and 80 GB nominal per GPU respectively). These managed evaluation observations
do not establish that the full local training-plus-simulation workflow fits a
24 GB GPU. The recommended local host remains the tested one-L40S setup above.

#### Local storage and retention

FineTune, SimEval and Validate run sequentially. Keep the images cached between
model runs; the three model families do not need to occupy GPU memory at once.

| Storage | Purpose |
| --- | --- |
| Root EBS / Docker and containerd image stores | Checkout, tooling, cached images, container writable layers and small durable run records |
| Separate filesystem selected by `--scratch-root` | Staged source working directories, selected model/data downloads, temporary files and SageMaker local working directories |
| Versioned development S3 bucket | Checkpoints, evaluation outputs, receipts, attestations and optional log archives |

Inspect both Docker and containerd: changing Docker's `data-root` alone did not
move the reference host's roughly 308 GiB store under `/var/lib/containerd`.
The runner's explicit scratch mounts move selected runtime paths instead. A host
`TMPDIR` does not redirect every trainer cache or checkpoint. Do not mount an
empty directory over `/opt/vla` or `/workspace`, which would hide baked code and
environments. The SDK also replaces a plain `container_config["volumes"]` entry;
inspect the generated Compose mounts when changing the layout.

The runner checks both disks after pulling images, **before training**:

| Requested local workflow | Root free before starting | Scratch free before starting |
| --- | --- | --- |
| N1.6 / Arena, including FineTune | 90 GiB | 250 GiB |
| MolmoAct2 / LIBERO, including FineTune | 60 GiB | 350 GiB |
| GR00T N1.7 or OpenVLA / LIBERO, including FineTune | 60 GiB | 220 GiB |
| Supplied checkpoint, FineTune omitted | 60 GiB | 220 GiB |

These are conservative planning floors for the sample, not measured hardware
minimums or a promise that an arbitrary dose will fit. Training-only requests
keep their cell's reserve. Larger datasets or retained checkpoint counts need
additional space. Every subsequent container still requires **40 GiB on root
and 100 GiB on scratch**; do not reduce that check to get past a refusal.
The legacy Arena invocation without separate scratch retains its 180 GiB
single-filesystem floor; the CLI requires separate scratch.

Accepted 200-step runs observed about 40/73 GiB of root/scratch
growth for Arena, 1/87 GiB for GR00T LIBERO and 4/218 GiB for MolmoAct2.
The initial floors preserve the later 40/100 GiB guard with margin. Arena's
source working files now go on scratch, so budget for that transfer as well.
Package caches and container writable layers still consume root.

Free scratch space does not compensate for a full root disk. One observed
200-step run exhausted the root reserve before SimEval while scratch still had
245.9 GiB free. The runner checks both disks before training and reports the
failing filesystem when refusing to proceed.

Earlier accepted 200-step LIBERO runs measured:

| Cell | Minimum root free | Root free after container removal | Minimum scratch free | Scratch free afterward |
| --- | --- | --- | --- | --- |
| OpenVLA / LIBERO | 119 GiB | 135.6 GiB | 689.8 GiB | 704.0 GiB |
| MolmoAct2 / LIBERO | 62.5 GiB | 135.6 GiB | 499.9 GiB | 530.9 GiB |

Those earlier MolmoAct2 runs retained five trainer checkpoints on the root
overlay, using about 73 GiB of root headroom. The current trainer writes these
intermediates to `/tmp/molmoact2-train-out`: SageMaker's documented temporary
storage path, also mounted on scratch by the local runner. Budget that checkpoint
growth on scratch as well as space for downloads and the final model. The trainer
logs the actual filesystem device, total size and free bytes before downloads,
after the dataset download, and after training or a trainer failure. A requested
training-volume size alone does not establish available space at every path.
The final policy/base archive in those runs was 29.6 GB. These observations include
other host activity and retained download caches; `disk-usage.json` samples every
15 seconds and can miss shorter peaks.

Between runs, save the verification results, logs and timings; confirm the
required outputs reached S3. Remove completed-run containers or disposable
intermediates only when space is needed, using the
[accepted-run cleanup instructions](#archive-an-accepted-local-run). Keep useful pinned
caches and other runs' resources. Give outputs and temporary directories unique
run paths, then check both filesystems before the next launch.

NVMe instance-store data is lost on stop, hibernate or termination; keep final
artifacts in versioned S3 and small records on EBS. If scratch is too small, a
separate temporary EBS data volume can be removed after results are retained.
A larger root volume cannot simply be shrunk later; that requires migration to
another volume. See AWS's
[instance-store persistence](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/instance-store-lifetime.html)
and [EBS modification limits](https://docs.aws.amazon.com/ebs/latest/userguide/ebs-modify-volume.html).

#### Costs outside the workload estimate

- **Host lifetime:** local mode leaves EC2 running after a pipeline ends.
  Twenty-four running hours on this host cost about **$108.69 compute**, including
  idle hours. After archiving, stopping the host ends running-instance compute
  charges; retained EBS and other resources continue to cost money.
- **Persistent disk:** gp3 capacity is **$0.08/GB-month**, approximately
  **$64/month for the provisioned 800 GiB volume**, prorated for its lifetime.
  The launch recipe uses the included 3,000 IOPS / 125 MiB/s baseline; extra
  provisioned IOPS/throughput and snapshots cost more.
- **Image builds:** the documented Linux CodeBuild jobs cost **$0.02/minute**
  for GR00T on `BUILD_GENERAL1_LARGE`, and **$0.20/minute** for the Arena base
  and connector on `BUILD_GENERAL1_2XLARGE`. Builds are billed in whole minutes.
  The us-east-1 separate-account build sequence took **92m35s**: GR00T
  **24m52s**, base **33m44s**, connector **33m58s**. At the dated rates above,
  rounding each build to whole minutes gives approximately **$14.10** for those
  three successful builds. The 60/180/120-minute limits are timeouts, not expected
  completion times; retries and downloads are additional.
- **Network:** the Foundation creates a NAT gateway and its public IPv4 address.
  In this region those cost **$0.045/hour + $0.045/GB processed** for NAT and
  **$0.005/address-hour** for IPv4: about **$1.20/day fixed**, before traffic.
  These resources remain chargeable after the EC2 host stops. Downloads through
  NAT can be substantial; compressed transfer bytes have not been measured.
- **Other services:** budget for ECR images, versioned S3 checkpoints/evidence,
  requests, logs, secrets and applicable data transfer. The compute estimates
  above exclude these, deployment work, negative controls and evidence archiving.

Hardware was checked with EC2 `DescribeInstanceTypes` and the
[G6e specification](https://aws.amazon.com/ec2/instance-types/g6e/).
Rates were checked with the
[AWS Price List Query API](https://docs.aws.amazon.com/awsaccountbilling/latest/aboutv2/price-changes.html):
EC2 `BoxUsage:g6e.8xlarge`; SageMaker `USE1-Train:ml.g6e.xlarge`,
`USE1-Train:ml.g6e.12xlarge` and `USE1-Processing:ml.m5.large`.
Recheck the region and current [EC2](https://aws.amazon.com/ec2/pricing/on-demand/),
[SageMaker](https://aws.amazon.com/sagemaker-ai/pricing/),
[EBS](https://aws.amazon.com/ebs/pricing/),
[CodeBuild](https://aws.amazon.com/codebuild/pricing/) and
[VPC](https://aws.amazon.com/vpc/pricing/) prices before allocating a budget.

#### Collect timing evidence for each new run

The tracked local verifier already writes `timings.json` with launcher,
pipeline, step and container durations, plus trainer runtime when available.
Run the [verification and archive commands](#verify-the-completed-run-before-cleanup) before deleting
containers. Preserve the run's commit, image digests, parameters, hardware and
cache conditions with those timings; keep failed attempts in the campaign cost.
After verification, estimate this run's EC2 compute from its timing file:

```bash
: "${RUN_ID:?Set the completed, verified run ID}"
EC2_HOURLY_USD=4.52856 # us-east-1 g6e.8xlarge, checked 2026-09-17; update for your host
sudo .venv/bin/python - "local-dev/runs/$RUN_ID/timings.json" "$EC2_HOURLY_USD" <<'PY'
import json
import sys
from decimal import Decimal

with open(sys.argv[1]) as source:
    timings = json.load(source, parse_float=Decimal)
hours = Decimal(str(timings["launcher_wall_seconds"])) / 3600
estimate = hours * Decimal(sys.argv[2])
print(f"Launcher: {hours:.3f} hours; estimated EC2 compute: ${estimate:.2f}")
print("Add setup, idle time, other attempts, storage, builds and network separately.")
PY
```

For managed runs, retain `describe-pipeline-execution`, `list-pipeline-execution-steps`
and each job's `describe-training-job` / `describe-processing-job` response.
Report execution elapsed separately from `BillableTimeInSeconds`, multiply job
usage by its instance count and matching **Training/Processing** rate, and retain
the pricing date. Use billing data for actual charges.

#### Accepted reference runs

These are historical results from development accounts, not artifacts expected
in a new account. The [timing and cost tables](#gpu-runtime-and-cost-planning)
provide the measured instance and step durations. Verify your own execution.

| Local observation | Recipe | Validated task successes |
| --- | --- | --- |
| Initial Arena sample | GR00T N1.6/Arena, 200 steps, 1 episode | 0/1; workflow check |
| OpenVLA compatibility sample | OpenVLA/LIBERO spatial, 200 steps, 3 episodes × 10 tasks | 0/30 |
| MolmoAct2 compatibility sample | MolmoAct2/LIBERO spatial, 200 steps, 3 episodes × 10 tasks | 30/30 |
| GR00T compatibility sample | GR00T N1.7/LIBERO spatial, 200 steps, 3 episodes × 10 tasks | 14/30 |
| Full Arena experiment, September 16 | GR00T N1.6/Arena, 20,000 steps, 20 episodes | 15/20 (75%) |
| Separate-account sample, September 17 | GR00T N1.6/Arena, 200 steps, 3 episodes | 1/3 (33.3%); 27m32s after preflight recovery |
| Connector update, September 17 | Same supplied host, 200 steps, 3 episodes | 1/3 (33.3%); 27m27s, no retry |

All seven accepted local runs had four successful steps, three zero container exits,
independent verification and version-read evidence archives. These runs used
`SuccessThreshold=0.0`. They measure different starting models and budgets;
they establish neither a model ranking nor before/after training improvement.

The full Arena trainer reached 20,000 steps and reported 1.108 steps/s, with
2,000 loss records from 0.9852 to 0.0020. Per-record optimizer-step indices were
absent. Saved simulator records independently matched 15/20 successes. The run
used the unchanged v15 evaluator, cached images, 48-hour runtime budgets and
only a final checkpoint. It is a diagnostic experiment, not an NVIDIA benchmark
reproduction; a historical comparison to NVIDIA's microwave result was not a
valid comparison for this fridge task.

The separate-account sample built its images in us-east-1:
training digest `sha256:a8e7c257d7af15faca2f99ecdb941cefdb6cd63dfd513f92c7850ba2effe1f3a`,
base digest `sha256:2f2acf9588a357d1455eac275da862ffd2f12fccbfa6fb6d632415e64273c152`,
connector digest `sha256:ace5ee23fe60de8a33701d78dbfa31c74649f3659b724d4b82092b0c54c1574b`.
These identify the tested combination; they are not publicly distributed images.
Its supplied `g6e.8xlarge` was in us-east-2. Container preparation initially
exceeded the old 120-second deadline; retrying after disk writes settled passed.
These local observations do not certify a managed submission. See the
[managed measurements](#additional-measured-managed-samples) for completed
SageMaker executions.

The subsequent update rebuilt only the connector in **28m05s**, publishing digest
`sha256:4d82d678c54a4a3376edd7e759cab593365bce67bce77f0d6880d79deb491daf`.
Training and base images were reused. Disconnecting and reconnecting followed
the same local execution. The saved effective policy YAML matched its validated
digest, and the rebuilt image disabled the inherited interactive healthcheck.
This review exercised the existing-host update route; it did not exercise a new
host launch, a fresh infrastructure deployment, or a managed run of this revision.

Earlier GR00T/LIBERO attempts failed on AV1 decoding and consumed-backbone
attribution. The current training build includes the decoder fix, and staged
LIBERO source includes the loader-audit fix. Those failed attempts remain
failures; their raw rollout numbers are not accepted results. The three accepted
LIBERO pipelines totaled 2h22m42s, excluding builds, retries and verification.

### What code runs where

There is one repository, with two ways of delivering code into containers:

| Location | What runs there | How an edit reaches execution |
| --- | --- | --- |
| Provisioning machine, such as your laptop | Terraform, build submission scripts and the managed launcher | Run the commands from the selected checkout; GPU work does not run here |
| AWS CodeBuild → ECR | Docker builds of the training environment, Arena base and connector | Build and publish a new tag, then select its digest |
| EC2 host in local mode | `scripts/local/launch.sh`, the Python runner, pipeline orchestration and verification tools | Clone/pull the selected commit and install it in that clone's `.venv` |
| FineTune container | Training environment from ECR plus the clone's staged `train_entry.py` and helpers | Source edits are staged on the next launch; environment changes need an image rebuild |
| Arena SimEval container | Isaac Sim, Arena and evaluator/server code baked into the connector image | Rebuild the connector and select its new digest; pulling Git alone does not update baked code |
| Validate container | Generated validator source, shared modules and the packaged SDK | Source staging delivers the selected checkout on the next launch |
| SuccessGate | A pipeline condition evaluated by the local SDK or managed service | Uses the pipeline definition; it is not a fourth workload container |

Training, simulation and validation run sequentially in three containers on the
EC2 host. Checkpoints pass through versioned S3. Managed execution uses the same
staged-source and image contracts on SageMaker compute, then registers the model.
Pushing Git changes does not update an EC2 checkout or an already-running job.

#### Review the same source and images locally and on SageMaker

All build, launch and verification commands are in this repository. A review
handoff needs the source branch and commit, deployment/account access, selected
image digests, and the exact cells to run. It does not need a private launcher,
pre-generated AWS request files or a separately prepared pipeline.

1. Commit and push the reviewed source. Build only images whose installed
   dependencies or baked files changed, then save their digests using
   [image selection](#select-published-image-digests). Unchanged images can be reused.
2. Update the EC2 checkout to that commit, run its tracked setup, and select
   those images. Use the normal [local commands](#run-the-four-step-pipeline)
   for Arena and the [LIBERO profiles](#run-the-three-libero-samples) for affected
   compatibility paths. Keep generated settings and run evidence in `local-dev/`.
3. In parallel with local verification, use the same commit and image digests with
   [the managed Arena launcher](#step-1--run-it) or
   [the managed LIBERO launcher](#optional-managed-libero). The launcher stages
   source from that checkout and creates/updates and starts its pipeline version
   automatically. This is normal launch preparation, not a code or image rebuild.
4. For parallel capacity attempts, keep the cell, counts, seeds and images fixed.
   Select instance types through the documented override flags and use a unique
   `VLA_PIPELINE_NAME` per attempt. Save the returned execution ARN and parameters,
   then follow that execution's status and verification. A successful different
   cell does not validate the requested one. If a local run exposes a candidate
   defect, cancel affected managed attempts rather than continuing to test it.

Local verification covers the application on the selected EC2 host. Managed
execution additionally exercises SageMaker storage, the separate runtime roles,
and model registration. Record both outcomes against their actual source and
image identities; changing either after review creates a new candidate to verify.

Choose verification according to the paths changed. For changes affecting
training, images or publication, the following small samples cover distinct
behaviors:

| Where | Cell or behavior | Acceptance |
| --- | --- | --- |
| Existing EC2 host, sequentially | N1.6/Arena; N1.7/LIBERO spatial; MolmoAct2/LIBERO spatial | Each completes 200 training steps and all four local steps, with independent verification. Arena has 3 episodes/seed 100; LIBERO has 3 per task/seed 1000. |
| Managed, in parallel with local review | N1.6/Arena | Same sample through all five steps and model registration. |
| Managed | MolmoAct2/LIBERO spatial | Same sample, including final checkpoint upload and all five steps. |
| Managed standalone | N1.6/Arena on an existing checkpoint | Three episodes/seed 100 and actual evaluation output under the selected prefix. This does not substitute for the full pipeline. |

All samples use threshold 0.0 and report robot successes separately. Include
OpenVLA or N1.7/Arena when a change affects those paths; do not repeat unrelated
GPU work for an editorial change. Runners can reuse published images when their
installed dependencies and baked files are unchanged. Ordinary source staging
and pipeline submission still occur on managed launch.

### Arena images and connector contract

This section is reference for image changes. Follow
[Build the container images](#build-the-container-images) for the normal
fresh-account sequence.

| Image | Recipe / builder | Repository owner |
| --- | --- | --- |
| Base: Isaac Sim and Isaac Lab Arena | `entrypoints/eval/isaac_arena/base/buildspec_base.yml` | Component Terraform, `vla/isaac-arena` |
| Connector: base plus GR00T servers and evaluator | `scripts/build_arena_connector.py`, toolchain-root `containers/isaac-lab-arena/Dockerfile` | Foundation, discovered from `/<project>/ecr/isaac-lab-arena` |

The pair manifest's `eval_base_image` becomes `ARENA_BASE_IMAGE` in the
connector build. The builder verifies the selected repository/tag before
submission. `--base-image` selects another compatible base; a full registry URI
currently requires `--skip-base-check`, which bypasses that availability check,
not Docker's pull or IAM enforcement. `--tag` defaults to the pair manifest's
`eval_image` tag (`v11` in the manifest). This is a build-target default, not a
release selector: historical `v15` references describe earlier measurements,
and the walkthrough supplies its own build tag. If you override the tag, select
the resulting image explicitly for the run. Use the published digest, as in the
walkthrough.

The model family, source directory and builder's `--connector` choice are
spelled `gr00t`. The runtime selector (`ArenaConnector` / `ARENA_CONNECTOR`)
is spelled `groot`. These are different configuration fields, not
interchangeable aliases.

#### Base requirements

| Required path or capability | Consumer |
| --- | --- |
| `/isaac-sim/python.sh` and `/isaac-sim/setup_python_env.sh` | Connector build and container entrypoint |
| `/workspace/isaaclab_arena/evaluation/policy_runner.py` | Baked evaluator |
| Runner flags `--policy_type`, `--num_episodes`, `--num_envs`, `--remote_host`, `--remote_port`, `--policy_config_yaml_path`, `--enable_cameras`, plus a positional task and its `--object` / `--embodiment` flags | Closed-loop rollout |
| GR00T remote-policy protocol over ZMQ, default port 5555 | Arena client and GR00T server |
| NVIDIA libraries including `libGLX_nvidia.so.0`, and `OMNI_KIT_ACCEPT_EULA` | Container startup |
| An embodiment-matching closed-loop policy YAML and its referenced assets | Suite's `arena.policy_config` |

An unset policy path is refused at submission unless supplied with
`--policy-config-yaml`. An unreadable file fails in the evaluator; it does not
trigger discovery of a substitute. The fridge suite names the pinned GR1
configuration. See [effective policy configuration](#the-policy-config-that-actually-runs-is-not-the-file-on-disk)
for how the checkpoint path is patched and the consumed YAML is saved.

The connector builds separate `/opt/gr00t-n16/.venv` and
`/opt/gr00t-n17/.venv` environments. Both are present even for the N1.6 sample.
Isaac Sim retains its own numerical/runtime packages; installing GR00T's
transitive dependencies into that interpreter can break extension imports.

#### Baked inputs and rebuilds

The Arena entrypoint does not execute SageMaker's staged evaluator source.
Rebuild the connector after changing its Dockerfile, dependencies or any of
these copied inputs (paths are relative to this component):

| Copied input | Purpose |
| --- | --- |
| `entrypoints/eval/isaac_arena/_shared/docker_entrypoint_multi.sh` | Container entrypoint |
| `entrypoints/eval/isaac_arena/gr00t/eval_entry.py` | Rollout orchestration and schema-v3 report |
| `entrypoints/eval/isaac_arena/gr00t/_verify_gr00t.py` | Build-time probe |
| `entrypoints/eval/isaac_arena/gr00t/gr00t_seeded_server.py` | Seeded server startup |
| `entrypoints/eval/isaac_arena/gr00t/gr00t_n17_n16_action_adapter.py` | N1.7 action-format adapter |
| `entrypoints/eval/isaac_arena/gr00t/gr00t_n17_n16_action_adapter.pth` | Adapter import hook |
| `entrypoints/eval/isaac_arena/_shared/digest.py` | Checkpoint digest, copied from the mirror |
| `src/vla_pipeline/common/validator.py` | Report validation |
| `src/vla_pipeline/common/capped_reader.py` | Bounded reads |
| `src/vla_pipeline/common/source_identity.py` | Source identity |
| `src/vla_pipeline/common/checkpoint_compat.py` | Checkpoint compatibility |
| `entrypoints/train/gr00t/defaults.json` | GR00T defaults |
| `entrypoints/train/gr00t/arena_gr1_data_config.py` | GR1 modality configuration |

Only `digest.py` comes from the `_shared/` mirror; the four shared modules
listed under `src/` are copied from that location. The modality COPY selects
the GR1 file, not every `arena_*_data_config.py`.

Suite manifests are delivered through `EvalSimConfig`, so changing a suite does
not itself require an image rebuild. Adding a training suite also requires
updating the container-side suite literals described in
[Notes and limitations](#notes-and-limitations). Attribute a run to its actual
image digest and source snapshot; a Git pull does not replace baked code.

#### GR00T remote-policy protocol

The evaluator launches the selected GR00T server and drives Arena's client over
ZMQ. Messages use `MsgSerializer([payload, info_dict])`:

| Direction | Payload |
| --- | --- |
| Observation → server | `video.<cam>`: uint8 array `(n_envs, T, H, W, C)`; `state.<joint>`: float32 array; `annotation.human.task_description`: string |
| Action → Arena | `action.<joint>`: float32 array `(1, horizon, DOF)` |

N1.6 uses its native GR1 server. For N1.7, the adapter converts per-modality
msgpack dictionaries to arrays for the older Arena client in the native GR1
action space. Its `.pth` hook loads in Isaac Sim's interpreter, including the
policy-runner subprocess. It does not retarget Franka actions to GR1. Changes to
this adapter require a connector rebuild and runtime validation.

### GR00T AV1 decoding

The LIBERO spatial LeRobot dataset contains AV1 video. A decoder can import
successfully yet fail on `get_frames_at` with
`Could not push packet to decoder: Function not implemented`.
`docker/gr00t/build_ffmpeg.sh` enables libdav1d CPU decoding in FFmpeg 7.0.2,
retaining H.264/H.265 support. Both the full Dockerfile and `Dockerfile.av1`
run `check_video_decode.py`, which checks actual pixels, frame order and repeated
indices using a bundled two-frame AV1 fixture without a GPU or dataset download.

**Fresh accounts use the full build in the walkthrough**, which already includes
this correction. The optional incremental recipe below repairs an older
GR00T image's decoder without re-syncing its Python environment. It requires a
published base digest, Docker, ECR read access and permission to publish a new
tag. An EC2 runtime role's image-read grants alone do not authorize publishing;
an administrator must grant repository-scoped layer upload and `ecr:PutImage`
permissions to the chosen builder, or use the existing CodeBuild path.

From the component directory on the Docker host, using this deployment's
credentials and registry:

```bash
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REGISTRY="$ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com"
BASE_DIGEST=$(aws ecr describe-images --region us-east-1 \
  --repository-name vla/gr00t --image-ids imageTag=1.2 \
  --query 'imageDetails[0].imageDigest' --output text)
GR00T_IMAGE_TAG="av1-repair-$(date -u +%Y%m%dT%H%M%SZ)"
aws ecr get-login-password --region us-east-1 |
  sudo docker login --username AWS --password-stdin "$ECR_REGISTRY"
git -C .. archive HEAD:isaac-lab-arena-on-aws docker/gr00t |
  sudo docker build --progress=plain -f docker/gr00t/Dockerfile.av1 \
    --build-arg "BASE_IMAGE=$ECR_REGISTRY/vla/gr00t@$BASE_DIGEST" \
    -t "$ECR_REGISTRY/vla/gr00t:$GR00T_IMAGE_TAG" -
sudo docker push "$ECR_REGISTRY/vla/gr00t:$GR00T_IMAGE_TAG"
REPAIRED_DIGEST=$(aws ecr describe-images --region us-east-1 \
  --repository-name vla/gr00t --image-ids "imageTag=$GR00T_IMAGE_TAG" \
  --query 'imageDetails[0].imageDigest' --output text)
printf '%s\n' "$ECR_REGISTRY/vla/gr00t@$REPAIRED_DIGEST"
```

Pass that immutable URI with `--train-image` to the local `gr00t-libero`
profile. The incremental recipe uses the **component root** as its context;
the full CodeBuild recipe uses **`docker/gr00t/`**. The archive includes committed
build files only, excluding generated runs and secrets.

### Validation runtime and training lineage

`stage_validate_code()` packages the validator, shared modules and a hash-locked
AWS SDK into a content-addressed source file. The first staging downloads the
seven pure-Python wheels in `src/vla_pipeline/validation_sdk.lock`, verifies their
hashes and caches them under `~/.cache/vla-validation-sdk`.
`VLA_VALIDATION_SDK_CACHE` selects another host cache. The closure supports the
Validate image's Python 3.9; it is separate from the host SDK.

Validate extracts those wheels into a private temporary directory before
importing boto3. It does not run pip or contact a package index.
`PutObject` and `CompleteMultipartUpload` must support `IfNoneMatch`; missing
support fails before publication. Receipts record the actual SDK versions and
bundle digest. Refreshing dependencies requires updating the complete
hash-locked closure and exercising the staged validator in its processing image.

GR00T writes `training_lineage.json` before computing the checkpoint's normative
digest. The downloader resolves the dataset ref once and uses that commit for
both listing and download. The sidecar records the requested suite/dataset,
resolved Hugging Face commit, consumed subdirectory and measured dataset-tree
digest. It remains separate from `checkpoint_manifest.json`, preserving the
manifest-v1 contract with existing Arena images.

Validate reads the sidecar from its own downloaded checkpoint. It compares the
suite/source/subdirectory with the requested contract and the content digest
with the checkpoint manifest. For a named dataset ref, it independently resolves
the Hugging Face metadata and requires the recorded commit to match; a moved or
unavailable ref fails. An exact 40-character commit in `DatasetRevision` avoids
that metadata lookup during validation.

| Field | Receipt status | What is established |
| --- | --- | --- |
| Dataset revision | `verified` | The requested ref resolves to the recorded commit, or it equals the explicitly pinned execution parameter |
| Dataset source, subdirectory, training suite | `attested` | Producer assertions match the requested contract and are digest-covered |
| Dataset content digest | `attested` | Sidecar and manifest agree; Validate does not recompute the dataset tree |
| Training steps | `matched` | The checkpoint recipe agrees with the requested step count |

Validate does not replay training or download the training dataset. Digest
coverage makes attestations tamper-evident; it does not independently establish
training consumption. **New GR00T training executions require the sidecar.**
Older checkpoints remain usable through explicit eval-only execution, where
training lineage is outside this run. Other model families retain their existing
lineage behavior.

### Model and embodiment compatibility

The [supported-cell table](#supported-cells) lists the CLI's available paths.
GR00T N1.6/Arena uses the native GR1 head and the `GR00T-N1.6-3B` base model.
N1.7/Arena uses `GR00T-N1.7-3B` with the N1.7-to-N1.6 action adapter during
evaluation. The `vla` CLI requires an explicit cell/version; the legacy
`scripts/run_arena.py` defaults to N1.7 unless `--gr00t-version n16` is supplied.

OpenVLA/MolmoAct2 are Franka-arm policies, while the Arena task uses the Fourier
GR1 humanoid's joint actions. This application provides no retargeting adapter
between those embodiments, so those pairs are not exposed. N1.6/LIBERO is also
excluded: its checkpoint is incompatible with the shipped N1.7 evaluator.
These are cell boundaries, not additional datasets to select.

### Layout

```
isaac-lab-arena-on-aws/
├── infra/                 Terraform: ECR + CodeBuild + step roles + scoped grants
├── src/vla_pipeline/      the pipeline SDK: pipeline.py, config.py, runner.py, registry.py, common/
├── entrypoints/           SageMaker container entrypoints
│   ├── train/<family>/    train_entry.py + defaults.json (+ GR00T Arena modality configs)
│   ├── eval/libero/<family>/     LIBERO eval entries (sourcedir-delivered)
│   ├── eval/isaac_arena/<family>/ Arena eval entries (baked by the root-level connector recipe)
│   └── validate_entry.py  the trust-chain validation step
├── scripts/               run_libero.py, run_arena.py, submit_simeval.py, build_arena_connector.py, …
├── config/                registry manifests
│   ├── suites/            what a suite IS: dataset + embodiment + task ids + Arena runtime contract
│   ├── families/          model-family adapter identity (train image, defaults.json)
│   ├── simulators/        simulator adapter identity (delivery mode, result protocol)
│   └── pairs/             (family × simulator) → eval image, connector, registry group
├── docker/                LIBERO train image Dockerfiles
└── tests/                 offline unit tests (no AWS)
```

#### The suite defines the cell

`--suite` selects a manifest under [`config/suites/`](config/suites) that declares
**everything that must agree between FineTune and SimEval**: the training dataset
(repo, subdir, revision), the per-family embodiment tag and modality config, the
canonical task ids, and — for Arena — the closed-loop runtime contract
(`policy_runner` task, `--embodiment`, `--object`, `--policy_config_yaml_path`).
The per-run `EvalTrials` parameter supplies `--num_episodes`; the evaluator uses
one environment. `num_steps` is retired and rejected in `EvalSimConfig`.

Adding an Arena task is therefore a manifest, not a flag incantation.

**What makes a cell coherent, precisely.** Two different mechanisms, worth not
conflating:

1. **The launcher.** `run_arena.py` has no `--task-name`: the task *is* the suite,
   derived from the same manifest FineTune's dataset comes from. This is what makes
   a launcher-submitted cell coherent by construction.
2. **Validate, as an independent backstop** — for hand-started
   `StartPipelineExecution` calls, where `Suite` and `EvalSimConfig` are set
   independently and can disagree:
   - **suite/task**: the report's task *label* must equal the suite's `arena.task`.
     Note this checks a label the evaluator echoes back, not the simulator's own
     record of what it instantiated.
   - **suite/embodiment**: the `embodiment_tag` FineTune recorded in
     `checkpoint_manifest.input_config` must equal the one the suite declares for
     that family and version. This one reads a value *FineTune wrote*, so it catches
     a policy trained for one embodiment and evaluated under a different declared
     tag — which the digest chain cannot see, because the bytes really are the
     trained bytes.

Both gates fail closed: if the suite table fails to reach the Validate container,
the step errors rather than skipping the checks.

**What Validate checks for Arena.** `effective_eval_config` must report
`budget_type=fixed_trials`, `num_episodes` equal to the requested `EvalTrials`,
and integer `num_envs=1`. It also requires `embodiment_tag`, `arena_embodiment`,
`arena_object`, `policy_config` and `gr00t_version`, checked against the expected
pipeline values. Missing fields, mismatches or a retired `num_steps` field fail.
The consumed policy-config digest must be present and correctly formatted;
Validate cannot independently recompute that evaluator-reported digest.
Overrides such as `--embodiment-tag` are checked against the requested value.

**What Validate does *not* check.** Validate compares the evaluator's
*self-reported* runtime configuration against the *requested* values. It does
not independently observe the simulator scene or verify that the declared
`policy_config` file was semantically correct for the embodiment. The
embodiment gate compares declared tags, not the embodiment the simulator
actually instantiated. See also
[training-lineage boundaries](#validation-runtime-and-training-lineage).

Arena runtime flags on `run_arena.py` (`--embodiment-tag`, `--arena-embodiment`,
`--object`, `--policy-config-yaml`) are **overrides**: unset means
"use the manifest". A suite marked `status: experimental` (`arena_g1`) needs
`--allow-experimental` and cannot pass Validate.

### Reading the Arena internals without pulling the image

The suite manifest records **container paths** (`arena.policy_config`,
`policy_runner.py`), which has repeatedly led readers to believe the only way to
inspect them is to pull the large connector image. It is not. Isaac Lab-Arena is
public and Apache-2.0, and the base image is built from a **pinned commit**, so
every file the manifest names can be read directly on GitHub.

**Pinned Arena commit:** `8b4a3a47fc53de23e8205089d71109a2e2348acd`
([browse the tree at that commit](https://github.com/isaac-sim/IsaacLab-Arena/tree/8b4a3a47fc53de23e8205089d71109a2e2348acd))

The base image is built with `WORKDIR=/workspace`, so the mapping is mechanical —
**strip the `/workspace/` prefix to get the upstream repository path**:

| Container path (what the manifest records) | Upstream path at the pinned commit |
|---|---|
| `/workspace/isaaclab_arena/evaluation/policy_runner.py` | [`isaaclab_arena/evaluation/policy_runner.py`](https://github.com/isaac-sim/IsaacLab-Arena/blob/8b4a3a47fc53de23e8205089d71109a2e2348acd/isaaclab_arena/evaluation/policy_runner.py) |
| `/workspace/isaaclab_arena_gr00t/policy/config/gr1_manip_ranch_bottle_gr00t_closedloop_config.yaml` | [`isaaclab_arena_gr00t/policy/config/gr1_manip_ranch_bottle_gr00t_closedloop_config.yaml`](https://github.com/isaac-sim/IsaacLab-Arena/blob/8b4a3a47fc53de23e8205089d71109a2e2348acd/isaaclab_arena_gr00t/policy/config/gr1_manip_ranch_bottle_gr00t_closedloop_config.yaml) |

#### The policy config that actually runs is not the file on disk

The upstream YAML ships a **placeholder checkpoint**:

```yaml
model_path: /models/isaaclab_arena/sequential_static_manipulation_tutorial/checkpoint-20000
```

Before starting the rollout, the eval entrypoint rewrites `model_path` to the
mounted checkpoint (`_patch_policy_config_model_path`) and passes the rewritten
copy to `policy_runner.py` via `--policy_config_yaml_path`. So when comparing runs,
three things are distinct and must not be conflated:

1. **The upstream file** — the task contract (instruction, `embodiment_tag: GR1`,
   `action_horizon`, camera/joint configs). Identical for every run of this cell.
2. **The suite manifest's `arena.policy_config`** — *which* upstream file was
   selected. This is what Validate's coherence gate compares.
3. **The effective config** — the patched copy that `policy_runner.py` actually
   read, differing from (1) only in `model_path`.

The connector preserves the final checkpoint's patched config as
`policy_config.yaml`, beside `metrics.json` in SimEval's `/opt/ml/model` output.
It is included in that job's S3 `model.tar.gz` artifact. Its bytes must match
`effective_eval_config.policy_config_digest` in the report. Historical images
did not save this file; rebuild the connector to enable it.

For optional inspection after local verification, while its containers still
exist, copy the file and compare it with the validated receipt:

```bash
RUN_DIR="local-dev/runs/$RUN_ID"
EVAL_CONTAINER=$(sudo .venv/bin/python -c \
  'import json,sys; print(next(row["id"] for row in json.load(open(sys.argv[1])) if row["kind"] == "eval"))' \
  "$RUN_DIR/container-exits.json")
sudo docker cp "$EVAL_CONTAINER:/opt/ml/model/policy_config.yaml" "$RUN_DIR/policy_config.yaml"
sudo .venv/bin/python - "$RUN_DIR" <<'PY'
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
receipt = json.loads((root / "verified-receipt.json").read_text())["receipt"]
actual = "sha256:" + hashlib.sha256((root / "policy_config.yaml").read_bytes()).hexdigest()
expected = receipt["effective_eval_config"]["policy_config_digest"]
if actual != expected:
    raise SystemExit(f"Saved config digest differs from receipt: {actual} != {expected}")
print(f"Saved config matches the recorded digest: {actual}")
PY
```

This preserves the evaluator's configuration for debugging; it does not add an
independent observation of the simulator scene. Its `model_path` names the
checkpoint inside that job and is not a portable input for a different run.
For a dose curve, it describes the canonical final checkpoint's report.

#### Step budget vs episode count

The component already uses `--num_episodes=EvalTrials` with one environment and
checks the observed count. The following explains the upstream API and the
historical step-budget failure; it is not additional setup. Do not add
`num_steps` to a suite's Arena block or to `EvalSimConfig`.

`--num_steps` is a **step budget**, not an episode count, and confusing the two
produces an evaluation that reports nothing. An episode ends only when the
environment signals `terminated` (task success, or a failure condition such as the
object being dropped) or `truncated` (the time limit is reached). Arena's rollout
loop counts an episode on either signal, so a timed-out episode **is** counted:

```python
obs, _, terminated, truncated, _ = env.step(actions)
if terminated.any() or truncated.any():
    env_ids = (terminated | truncated).nonzero().flatten()
    num_episodes_completed += env_ids.shape[0]
```

The time limit is set by the environment, not by the step budget:

```
max_episode_length = episode_length_s / (sim.dt * decimation)
```

For `put_item_in_fridge_and_close_door` that is `10.0 / (0.005 * 4)` = **500 policy
steps**. Consequently:

- `--num_steps` **below** `max_episode_length` can only ever record an episode that
  ends *early* (success or dropped object). A policy that neither succeeds nor fails
  early records **zero** episodes, and every rate is computed over an empty sample.
- To collect `k` complete episodes with a step budget, `--num_steps` must be at
  least `k * max_episode_length`.

Because the step budget and the requested episode count are two different stopping
criteria for the same rollout, they can silently disagree — a step budget only
guarantees *enough room* for `k` episodes, and early termination can produce more.
`policy_runner.py` accepts
[`--num_episodes`](https://github.com/isaac-sim/IsaacLab-Arena/blob/8b4a3a47fc53de23e8205089d71109a2e2348acd/isaaclab_arena/evaluation/policy_runner_cli.py)
as an alternative to `--num_steps`, which removes the arithmetic by asking for episodes
directly. Three caveats that matter when using it:

- **It is not mutually exclusive at the CLI.** `main()` prefers `--num_steps` and clears
  `num_episodes` before the rollout assertion, so passing both silently uses the step
  budget rather than erroring. Pass exactly one.
- **Episode mode has no independent step ceiling.** Each episode is bounded by
  `max_episode_length` *provided truncation fires*; if termination flags are suppressed
   the loop does not stop on its own. The component supplies the separate
   wall-clock watchdog described in the local timeout section.
- **`k` is exact only with a single environment.** The rollout counts every environment
  that ends in the same vectorized step, so `--num_envs 2` with `k=3` can complete four
  episodes. The reported `num_episodes` is also read from the recorder dataset rather
  than the loop counter, so the observed count must be checked against the requested one.

### Tests

#### Development workflow

During development, prioritize a fresh user's experience: clone, follow this
README, deploy, build and run without private setup knowledge. Keep changes on
a feature branch and submit them for review; do not push directly to the default
branch. A deployment reviewer should clone the designated branch and record its
exact commit, including any assistance or corrections needed.

1. Start from the next README action or an observed execution failure.
2. Make the smallest useful change and review the files selected for its commit.
   Record the source snapshot before an attributed execution; rebuild when baked
   code or dependencies change.
3. Run the affected container or pipeline step, inspect its real outputs and
   exit status, fix failures and repeat that step.
4. Connect the working steps into the small local pipeline; use a managed sample
   for registration. Update this README with what a fresh user needed and retain
   run identity, failure details, timing and cost.

Choose the smallest execution that resolves the uncertainty. An unrelated
documentation edit does not justify repeating long training. Focused existing
checks can support diagnosis. Avoid expanding a small usability fix into a
coverage project or repeating unrelated tests. Preserve existing tests,
application validation and truthful results:
partial numbers or a failed step must not become an end-to-end pass.

Once the working snapshot is approved for integration, freeze the intended
behavior and conduct a dedicated regression pass around the demonstrated user
workflows. That stabilization supports later changes by multiple developers.

#### Existing diagnostic tools

These existing tools are available for focused diagnosis and the later dedicated
regression stabilization pass. They are **not prerequisites for cloning,
deploying or running this walkthrough**. During development, follow the actual
container/pipeline workflow, fix observed errors and rerun the affected path.
A test count is not deployment evidence or a coverage target.

The following manual tools supplement the normal `vla` commands. Run them from
the component directory with `PYTHONPATH=src`; tools that submit jobs use your
selected AWS account and incur charges.

| Tool | Purpose | AWS activity |
| --- | --- | --- |
| `scripts/run_matrix.py --manifest config/matrix_manifest.json --deploy` | Run the manifest's small LIBERO grid; `--max-concurrent` controls concurrency. | Creates/updates pipelines and submits GPU jobs. |
| `scripts/run_plumbing.py --run all` | Exercise CPU-only gate-true, gate-false and intentionally invalid evidence workflows. | Deploys diagnostic pipelines and runs three executions. |
| `scripts/verify_failure.py EXECUTION_ARN STEP VIOLATION` | Check that an existing execution failed at the expected step for the expected reason. | Reads pipeline execution/step records; submits no jobs. |

- **Unit / offline** (`tests/`, no AWS; measured 125 seconds on the reference box): pipeline-shape, registry
  resolution, digest, validator, sourcedir staging, safe-extract. Run:

  ```bash
  python3 -m venv local-dev/test-env
  source local-dev/test-env/bin/activate
  python -m pip install -e '.[dev]'
  python -m pytest tests/ -q --strict-markers -rs
  python scripts/gate_injected_audit.py
  python scripts/gate_telemetry_patch_order.py
  python scripts/gate_reported_equals_executed.py
  ```

  Keep this test environment separate from the launcher's `.venv`; the dev
  dependencies include Torch and TensorFlow and take additional disk/download
  time. Place it on the selected scratch filesystem if root is tight.
  [`buildspec_tests.yml`](buildspec_tests.yml) runs from the toolchain root:
  it installs `./isaac-lab-arena-on-aws[dev]` and runs
  `python -m pytest isaac-lab-arena-on-aws/tests -q --strict-markers -rs`.
  Supplying this specification does not configure a required repository CI check.

- **Local integration**: follow the [EC2 workflow in this README](#run-locally--ec2-gpu-debugging)
  for the Arena sample; run the three LIBERO samples only when checking those
  optional cells. It uses the deployed Foundation and
  development S3/ECR, with compute on the EC2 GPU.

- **Managed integration** (requires AWS, the Foundation and managed capacity):
  `scripts/run_*.py` submits real SageMaker pipeline executions. A small managed
  sample checks registration separately from the local training/evaluation runs.

  The optional [deployment review checklist](#deployment-review-checklist) describes
  evidence to retain. This README remains the deployment procedure.

#### Deployment review checklist

Record the cloned commit, account and deployment region (`us-east-1`), plus the
GPU host's region when different. Identify what was created from scratch versus
supplied: tokens, infrastructure, images and host. Retain Terraform plans and
owning state, build IDs, published digests, actual step/container results,
artifact identities and measured times.

Keep a running usability log: the exact command, redacted error or confusing
instruction, its effect on progress, assistance needed and eventual resolution.
Preserve failed attempts after a retry succeeds. Prepared inputs and historical
successes are not proof of a fresh-account deployment.

Local acceptance requires FineTune → SimEval → Validate → SuccessGate and the
[independent verifier](#run-locally--ec2-gpu-debugging). Managed acceptance adds
RegisterModel and verification of the registered artifacts. If only submission
was requested, record the execution ARN/state and leave completion explicitly
unverified. Infrastructure and image builds alone are partial progress.
Threshold zero checks pipeline operation, not model quality.

#### MolmoAct2 policy-audit diagnostics

The LIBERO evaluator's `_POLICY_LOAD_PROBE` reconstructs the policy using the
normal factory and checks expected keys, live PEFT adapter state and effective
LoRA deltas. Inspecting saved tensor names or a nonzero `lora_B` alone would miss
keys ignored by a non-strict loader. The probe needs Torch, PEFT, safetensors and
LeRobot in the evaluation image; offline source checks do not execute it.

The accepted `molmoact2-libero-01` run exercised the normal checkpoint path with
the same evaluator source. It does **not** establish execution of all the
mutation cases below. These are optional targeted maintainer diagnostics, not
additional attendee setup:

1. **a real produced checkpoint passes** — use an unmodified output from the
   pinned trainer as the positive control.
2. **a renamed LoRA key fails despite nonzero B** — rename a saved `lora_B`
   tensor; expect the unexpected-key error.
3. **a missing action-expert or base key fails** — delete an expected
   non-adapter tensor; expect the missing-key error.
4. **omitted manifest and saved-mode fields cannot bypass the audit** — remove
   `train_mode_vlm` from policy config and recipe; the upstream `lora` default
   must still trigger the audit.
5. **manifest/saved-mode disagreement fails** — provide conflicting modes.
6. **nonzero B with zero A fails the effective-delta assertion** — zero
   `lora_A` while retaining nonzero `lora_B`; the effective adapter is a no-op.

Existing offline startup checks cover the mode-selection cases (4–5); retain
separate container evidence before calling a mutation case runtime-verified.
Passing the probe establishes reconstruction and adapter state. It does not
prove training occurred or improved the policy, that every adapter contributes
to an action, that deltas were not baked into base weights, or that preprocessing,
normalization and non-adapter values are numerically correct. It also does not
prove a later CUDA process uses identical bytes/dependencies or executes graphs
correctly.

### Notes and limitations

- **GR00T image pairing.** Both N1.6 and N1.7 use the component's GR00T training
  image with glibc 2.35 and CUDA 12.4. Runtime setup selects the pinned GR00T
  revision for the requested version. The Arena connector is shared: N1.6 uses
  its native GR1 server, while N1.7 uses the action-format adapter.
- **Arena base build.** The pinned upstream recipe resolved Isaac Sim
  `6.0.0-dev2`, not final 6.0.0. Connector v15 is a historical N1.6/Arena
  reference; the separate-account builds and later connector rebuild are recorded
  under [accepted reference runs](#accepted-reference-runs). Historical results
  do not certify a fresh run of this checkout or account.
- **Policy configuration.** The fridge suite declares its explicit GR1 config
  path. Other suites without a policy config are refused at submission unless
  the operator supplies one. A declared path alone does not prove runtime
  compatibility; the file and its referenced assets must exist in the image.
- **`dummy` family / `dummy_sim` / `dummy_suite`.** An offline example for
  registry resolution and pipeline construction (`status: experimental`). Its
  entrypoints deliberately raise `NotImplementedError`; it cannot run an
  end-to-end sample. A new model needs working train/eval entrypoints as well
  as its manifests.
- **Experimental Arena suites.** `arena_g1` (G1 loco-manipulation) and
  `arena_gr1` (GR1 open-microwave) both lack a configured policy YAML and remain
  `status: experimental`. The low-level launcher's `--allow-experimental` flag
  permits selection only; it does not supply the missing policy or add a suite
  to Validate's supported allowlist. Neither is a ready-to-run CLI cell.
- **Suite definitions in containers.** Validate receives the resolved suite table
  embedded by `stage_validate_code()` and uses it for its allowlist and coherence
  checks. GR00T training still carries `SUITE_DATASETS`/`SUITE_META` and modality
  selection as literals: adding a training suite requires updating both the
  manifests and these copies. The standalone validator has a literal fallback
  for offline callers. Existing
  suite-consistency tests are available; do not infer that repository CI is
  configured or that a new suite has been validated merely from their presence.
- **Modality delivery.** Both pinned NVIDIA inference policies load modality
  configuration from the checkpoint's processor. The baked GR1 Python config
  and server's modality flag do not override that checkpoint-policy path.
  Training-side modality selection is separate. Supporting another embodiment
  requires compatible training data, checkpoint processor and Arena policy YAML,
  followed by execution validation.
- **Checkpoint input settings.** New Arena GR00T checkpoints record their
  embodiment without LIBERO's `n_action_steps` and `max_episode_steps` fields.
  Those positive integer strings remain required for GR00T/LIBERO. Older Arena
  artifacts containing the fields remain readable; Arena's effective policy
  YAML describes its actual rollout settings.
- **Training reproducibility.** The pinned GR00T trainer supports
  `TRAIN_SEED=42`; requests for a different training seed fail explicitly.
  `EvalSeed` is separate. Named dataset refs can move; pin an exact revision
  when repeatable identity is required. Neither pinning nor recorded seeds
  guarantee deterministic training or a particular success rate.
- **MolmoAct2 dataset downloads.** The trainer downloads the resolved Hugging
  Face dataset revision, or reads an explicitly supplied S3 dataset. Hugging
  Face's local cache remains available. The former opt-in `VLA_BUCKET` S3 cache
  and its dataset-write grant have been removed; ordinary pipelines did not
  enable it. Re-running still performs training and evaluation, with no pipeline
  result-cache reuse.
- **Dataset selection.** `__LIBERO_DEFAULT__` also selects Arena datasets.
  It is shared by multiple training entrypoints; renaming it requires updating
  all consumers together. Family overrides remain suite-keyed.
- **Reference-policy validation mode.** `scripts/submit_simeval.py --posctrl-n16`
  serves NVIDIA's published N1.6 checkpoint directly instead of a fine-tuned one
  (a positive-control / base-anchor measurement). It **deliberately bypasses the
  train→eval trust chain**, so it is a diagnostic, not a registrable pipeline run;
  the standard train-default path keeps the full digest chain intact.
- **IAM identities.** The Foundation role orchestrates the managed pipeline.
  FineTune, SimEval and Validate use the component's separate training, workload
  and validation roles. The local path uses its checked EC2 role and a separate
  dev bucket; it does not assume those managed roles or establish their trust boundary.
- **Terraform state is local** (no remote backend) — fine for a sample/dev deploy;
  configure a remote backend for shared/production use.
- **ECR teardown is guarded.** Direct `terraform destroy` refuses the ECR repos
  (`prevent_destroy = true`). `pai arena destroy --yes` removes that protection
  only in its recorded deployment directory, after archiving evidence and
  checking that its newly owned repositories are empty. Supplied repositories
  remain outside deletion scope. Retaining ECR alone does not make destruction safe:
  registered packages also depend on the versioned evidence buckets. Review
  [resource ownership and retention](#resource-ownership-and-permanent-teardown) before
  decommissioning; deployment acceptance does not require teardown.
- **Eval seed** is always set explicitly by the launchers (`--eval-seed`; Arena
  uses 100, LIBERO 1000). The pipeline's `EvalSeed` parameter default is a fallback
  only — Validate compares against whatever seed the run actually recorded.
  (`run_arena.py` previously defaulted to 1000, silently contradicting the
  documented Arena protocol on any run that did not pass `--eval-seed`; it now
  defaults to 100.)
- **RegisterModel** uses `sagemaker.workflow.model_step.ModelStep` (fed by
  `Model.register(...)` under a `PipelineSession`), not the deprecated
  `RegisterModel` step collection — the registration step emits **no** deprecation
  warning. The register `Model` intentionally carries no `entry_point`/`source_dir`
  so `ModelStep` does not inject a repack step and alter the pipeline graph.

### Resource ownership and permanent teardown

This is optional decommissioning reference. For ordinary debugging, use
[Finish a local session](#finish-a-local-session); a successful deployment does
not require deleting its infrastructure.

Foundation owns shared discovery, its models bucket, orchestration role and
connector ECR repository. Component Terraform owns its configured training/base
ECR repositories, image builder and role, versioned trust/handoff buckets and
training/workload/validation roles. The HF-secret grant belongs to the training
and workload roles. Keep both Terraform states and the saved deployment variables.
Destroying component roles does not transfer their permissions to Foundation;
externally supplied secrets also have a separate lifecycle.

#### Existing resources and imports

Repositories managed by CloudFormation or another Terraform deployment must
remain with that owner. Exclude them from `ecr_repos` and list them in
`existing_ecr_repos` to grant builder access without taking over their lifecycle.
The lists must not overlap. Select a distinct `codebuild_project_name` when
necessary and pass it through `--project-name` to both builders.

Import only after the previous owner has explicitly relinquished management.
For each agreed repository, the component's Terraform address is
`aws_ecr_repository.repos["<repository-name>"]`. Use that deployment's state and
the same saved variables for import, plan and later apply; inspect the ownership
diff before applying. Do not import Foundation's connector repository into
component state or apply an empty state over resources already managed elsewhere.

#### Retaining images and evidence

Component ECR repositories have `prevent_destroy = true`. The CLI's owned-resource
teardown uses a local Terraform override after archiving evidence and retiring
its images. It refuses pre-existing repositories, nonempty repositories and
forced image deletion. The shipped Terraform protection remains intact.

For manual lifecycle management, a reviewed plan can
retain them under another owner, including by removing their repository and
lifecycle-policy addresses from this state. Record that transfer and update the
declarations before any later apply, which would otherwise try to recreate them.
Retaining ECR alone does not preserve registered-model provenance.

| Component bucket | Contents | Consequence of deletion |
| --- | --- | --- |
| `<prefix>-trust-<account>` | Code under `code/v1/`, promoted artifacts under `artifacts/v1/`, attestations under `evidence/v1/` | Registered packages lose their referenced artifacts and provenance |
| `<prefix>-handoff-<account>` | Raw evaluation evidence under `eval/v1/`, gate receipts under `validated/v1/` | Past registration decisions lose their supporting evidence |

Both buckets are versioned. Their policies deny `DeleteObject` and
`DeleteObjectVersion` on protected namespaces to every principal, including the
writer. Runtime credentials cannot empty those prefixes. S3 also refuses to
delete a non-empty bucket, so `terraform destroy` can stop after deleting other
resources. Removing registrations does not automatically remove evidence, and
removing evidence does not unregister packages.

Before permanent decommissioning, decide whether registered packages must remain
usable. If they do, retain their referenced locations and evidence. Otherwise,
retire the packages and preserve the required artifact, attestation and receipt
versions in durable storage before planning deletion. Any policy changes and
version removal must be explicit parts of that reviewed plan, scoped to the
selected deployment. Stopping compute does not require deleting either bucket.

### License

This component is part of the AWS Physical AI Toolchain and is licensed under
the Apache License 2.0 — see the repository-root `LICENSE`. Third-party
components integrated or vendored by this component are attributed in
[`NOTICE`](NOTICE).
