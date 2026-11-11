# Lab 1: Train a Robot Policy from Demonstrations

**Goal:** Fine-tune NVIDIA GR00T (a Vision-Language-Action model — takes camera images + text instructions in, outputs motor commands) on teleoperation data → get a model that predicts robot joint actions from camera images
**Time:** 2 hours (30 min hands-on + training runs in background)
**Cost:** ~$2 for smoke test, ~$79 for full training

> **New to Physical AI?** See the [terminology guide](README.md#physical-ai-terminology)
> for definitions of key concepts like policy, teleoperation, VLA, and fine-tuning.

---

## What You're Building

A robot manipulation policy trained via **imitation learning**. Here's what that means:

1. A human teleoperated a robot arm to perform a task (pick and place) — 27 demonstrations were recorded. Teleoperation is typically done using a game controller (we used Xbox), a VR headset (Apple Vision Pro is increasingly popular), or a 3D Space Mouse. The operator controls the robot's end-effector while camera and joint data are recorded automatically.
2. Each demo captured: wrist camera images + joint positions + Cartesian velocity commands at every timestep
3. You fine-tune GR00T (a 3B parameter vision-language-action model) to learn the pattern: "given this camera image → predict these motor commands"
4. The result is a model that can control the robot autonomously for that task

**What GR00T fine-tuning actually trains:**
- The **language model** backbone stays **frozen** — NVIDIA already trained it on millions of robot videos
- The **vision tower**, the **projector** (maps vision features to action space), and the **diffusion action head** (predicts action sequences) get updated
- This is why 50 demos are enough — you're not training from scratch, you're teaching an existing model a new task

**The limitation of imitation learning alone:**
- The model can replicate what it saw in the demos (~70-80% success)
- It struggles with variations it hasn't seen (different object positions, lighting changes)
- Lab 4 fixes this with RL refinement in simulation

---

## The Pipeline

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Demo Data   │────▶│  SageMaker   │────▶│   Evaluate   │────▶│   Model      │
│  (LeRobot    │     │  Training    │     │   (action    │     │   Registry   │
│   parquet +  │     │  Job         │     │    error)    │     │              │
│   video)     │     │              │     │              │     │  groot-models│
│              │     │  ml.g5.      │     │              │     │  version N   │
│  in S3       │     │  12xlarge    │     │  in S3       │     │              │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘

Orchestrated by: SageMaker Pipeline (groot-finetune-pipeline)
```

**Key details:**
- **Instance:** `ml.g5.12xlarge` — 4× NVIDIA A10G GPUs, **24 GB per GPU (96 GB total)**, 192 GB system RAM. GR00T fine-tuning fits 24 GB here via gradient checkpointing + DeepSpeed ZeRO-2 + gradient accumulation (a single A10G OOMs — the 4-GPU box is required).
- **Container:** `groot-training` (7.1 GB, based on `nvidia/cuda:12.4.1-devel-ubuntu22.04`)
- **Dataset format:** LeRobot v2 — parquet files (actions, states, episode metadata) + MP4 video (camera observations)
- **Training time:** ~8 sec/step. 100 steps = 15 min (smoke test). 5000 steps = 11 hrs (full)
- **Cost:** SageMaker on-demand pricing for ml.g5.12xlarge is ~$7.09/hr. No idle cost — you only pay while training runs.
- **Output:** `model.tar.gz` in S3 containing: the fine-tuned model checkpoint + `eval_report.json` (dataset baselines — see Step 7)

---

## Prerequisites

- Foundation stack deployed (`pai deploy foundation` — see Lab 0). This already
  triggered the CodeBuild job that builds the training container in the cloud.
- The `pai` CLI installed (`pip install -e .` from the repo root — see Lab 0). Every
  step below leads with `pai groot ...`; the raw `python training/gr00t/...` commands
  are in the "Under the hood" drop-downs if you prefer them.
- A HuggingFace token for the GR00T base-model download (avoids anonymous rate
  limits during the multi-GB download): either `export HF_TOKEN=...` in the terminal
  you launch from, or store it once as the `physical-ai/hf-token` secret (Lab 0
  Step 6), which `pai groot launch` falls back to
  - Get one at https://huggingface.co/settings/tokens
  - Review the model card / license at https://huggingface.co/nvidia/GR00T-N1.6-3B
- Python 3.11+ with `boto3` and `huggingface-hub` installed

---

## Step 1: Verify Infrastructure

```bash
# Preflight: credentials, region, Foundation stack, training image in ECR
pai doctor
```

`pai groot launch`/`deploy` resolve the bucket, role, and ECR image from the
Foundation stack automatically — you don't have to pass them. To see the raw
outputs yourself:

```bash
aws cloudformation describe-stacks --stack-name PhysicalAi-dev-Foundation \
  --query 'Stacks[0].Outputs[*].[OutputKey,OutputValue]' --output table
```

You should see:
| Key | Value |
|-----|-------|
| DatasetsBucketName | `physical-ai-dev-datasets-<ACCOUNT_ID>` |
| ModelsBucketName | `physical-ai-dev-models-<ACCOUNT_ID>` |
| SageMakerRoleArn | `arn:aws:iam::<ACCOUNT_ID>:role/physical-ai-dev-sagemaker-role` |
| GrootTrainingRepoUri | `<ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com/physical-ai/gr00t-training` |

The `pai` commands resolve these for you, but the "Under the hood" raw commands
and Steps 7 and 9 use shell variables. Set them with one command:

```bash
eval "$(pai env)"   # exports AWS_REGION, BUCKET, ROLE_ARN, ECR_URI (+ JOB_NAME, MODEL_S3 once a job has run)
```

> **New terminal?** Shell variables don't carry over. Run
> `source .venv/bin/activate && eval "$(pai env)"` first, in any step below.

---

## Step 2: Prepare the Training Data

The training data is 27 episodes of UR3 pick-and-place, recorded via Xbox controller teleoperation. The raw data is in Zarr format (how our recording tools capture it) and needs to be converted to LeRobot v2 format (what GR00T reads).

**No robot needed** — the 27 episodes ship with the repo. Pull and convert them below. If you *do* have a UR3 and want to record your own demonstrations first, expand the optional section, then rejoin at conversion.

<details>
<summary>🤖 Optional: record your own demonstrations (requires a physical UR3)</summary>

This is the front bookend of the full hardware loop. It needs a real UR3 (URScript
port 30002 + dashboard port 29999 reachable), a wrist camera (Intel RealSense D405, or
any UVC/RTSP camera via the OpenCV fallback), and a Robotiq 2F-85 gripper. Nothing
else in this lab requires hardware.

```bash
export ROBOT_IP=192.168.1.100    # your UR3's IP (127.0.0.1 targets a local URSim)

# Gamepad teleop (opens a browser UI that reads an HTML5 game controller):
pai groot record --task "pick up the red cube" --mode gamepad

# ...or keyboard teleop (WASD+IJKL; works over an SSH tunnel, no browser/gamepad):
pai groot record --task "pick up the red cube" --mode keyboard

# Preview without moving the arm:
pai groot record --task "pick up the red cube" --dry-run
```

Controls (gamepad): left stick = X/Y, right stick = Z + wrist rotation, triggers =
gripper, **A** = start/stop recording, **B** = emergency stop.

Each demonstration lands in `training/data/episodes/episodes/episode_NNN_<task>/` —
exactly where `pai groot convert` looks below, so recorded data flows straight into
the pipeline (see [docs/zarr-schema.md](../docs/zarr-schema.md) for the layout).
Record **50–100** demonstrations with natural variation for a policy that generalizes.
Check what you captured with `python -m robot.ur3.recorder list`.

> **Safety:** the arm moves during teleop. Clear the workspace and keep the
> teach-pendant e-stop within reach.
>
> **Status:** the teleop capture path is ported from a validated GR00T reference
> but has not been re-verified on physical hardware in this repo — smoke-test it on
> your arm before a long recording session.

</details>

```bash
# 2a. Pull the dataset from Git LFS and extract it.
#     The zip's internal root is `episodes/`, so this yields
#     training/data/episodes/episodes/episode_*.
#     Needs Git LFS (Lab 0 prerequisites): without it the zip is a 135-byte pointer.
git lfs pull --include="training/data/ur3_episodes_001_027.zip"
unzip -o training/data/ur3_episodes_001_027.zip -d training/data/episodes

# 2b. Convert Zarr episodes → LeRobot v2 format
#     (conversion deps come with `pip install -e .` from Lab 0)
pai groot convert
```

`pai groot convert` defaults to the bundled UR3 episodes
(`training/data/episodes/episodes` → `training/data/ur3_lerobot_dataset`);
pass `--episodes-dir`/`--output-dir` to point it elsewhere, or `--dry-run` to
see the exact command first.

<details>
<summary>Under the hood (raw command)</summary>

```bash
python training/gr00t/convert_zarr_to_lerobot.py \
  --episodes-dir training/data/episodes/episodes \
  --output-dir training/data/ur3_lerobot_dataset
```

</details>

**What the conversion does:**
- Reads each Zarr episode (wrist camera frames + joint states + velocity commands)
- Aligns camera frames to telemetry timestamps (camera runs at 5Hz, telemetry at 10Hz)
- Extracts action vectors from URScript `speedl` commands (6D Cartesian velocity + gripper)
- Encodes camera frames to MP4 video
- Writes LeRobot v2 parquet + metadata files

**Output structure:**
```
training/data/ur3_lerobot_dataset/
├── data/chunk-000/          # Parquet files: 7D actions (6 velocity + gripper), 7D states (6 joints + gripper)
│   ├── episode_000000.parquet
│   ├── episode_000001.parquet
│   └── ... (27 episodes)
├── meta/
│   ├── info.json            # Dataset metadata (robot_type: ur3, fps: 5)
│   ├── modality.json        # GR00T embodiment config (arm start/end, gripper start/end)
│   ├── episodes.jsonl       # Episode lengths and task descriptions
│   └── tasks.jsonl          # Task vocabulary
└── videos/chunk-000/
    └── observation.images.wrist/  # Wrist camera MP4s (one per episode)
```

**Bringing your own data:** If you have Zarr episodes from your own teleop setup,
the one-command ingestion path converts → uploads → (optionally) trains:

```bash
pai groot ingest \
  --episodes-dir ./my_robot_episodes \
  --prefix groot-data/myrobot \
  --train --max-steps 100
```

<details>
<summary>Under the hood (raw command)</summary>

```bash
python training/gr00t/ingest_customer_data.py \
  --episodes-dir ./my_robot_episodes \
  --prefix groot-data/myrobot \
  --train --max-steps 100
```

</details>

The expected Zarr schema (`observations/joints`, `observations/gripper_position`,
`images/wrist`, `commands.json`, and the `zarr.json` attrs) is documented in full in
[docs/zarr-schema.md](../docs/zarr-schema.md). For a non-UR3 robot you also update the
state/action dimensions in `convert_zarr_to_lerobot.py` and the GR00T modality config
(`containers/gr00t-training/ur3_modality_config.py`) — both are explained there.

---

## Step 3: Upload Dataset to S3

```bash
pai groot upload
```

`pai groot upload` resolves the datasets bucket from the Foundation stack and
syncs `training/data/ur3_lerobot_dataset/` to `s3://<bucket>/groot-data/ur3/dataset/`
— where `pai groot launch` expects it.

<details>
<summary>Under the hood (raw command)</summary>

```bash
aws s3 sync training/data/ur3_lerobot_dataset/ "s3://$BUCKET/groot-data/ur3/dataset/"
```

</details>

---

## Step 4: Get the Training Container

**You don't build this locally.** When you deployed the Foundation stack (Lab 0),
CDK kicked off an AWS CodeBuild job that builds the `groot-training` image in the
cloud and pushes it to ECR. This means **no multi-GB `docker build` on your
laptop** — and it works even on Apple Silicon, where the image can't be built
locally at all.

Check that the image is ready:

```bash
# Was the build triggered? (it runs automatically on cdk deploy)
aws codebuild list-builds-for-project --project-name physical-ai-groot-training-build \
  --query 'ids[0]' --output text

# Is the image in ECR yet? (build takes ~10 min)
aws ecr describe-images --repository-name physical-ai/gr00t-training \
  --image-ids imageTag=latest --query 'imageDetails[0].imagePushedAt' --output text
```

If the image isn't there yet, watch the build in the [CodeBuild console](https://console.aws.amazon.com/codesuite/codebuild/projects)
(the `BuildConsole` Foundation stack output links straight to it). Re-run a build
any time with:

```bash
aws codebuild start-build --project-name physical-ai-groot-training-build
```

**What's in the container:**
- AWS SageMaker PyTorch DLC base (PyTorch 2.5.1 / CUDA 12.4 / Python 3.11)
- NVIDIA Isaac-GR00T (N1.6) installed from source, with flash-attn + decord
- `transformers==4.51.3` (GR00T N1.6 / Eagle backbone compatibility)
- pandas + pyarrow (dataset baselines for the eval report)
- `train_entrypoint.py` — the script SageMaker runs

<details>
<summary><strong>Optional:</strong> build locally instead (only if you're iterating on the Dockerfile)</summary>

The `groot-training` image builds from a public CUDA base, so you *can* build it
on an x86 machine with Docker if you want a faster edit/rebuild loop:

```bash
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin $ECR_URI
cd containers/gr00t-training
docker build --platform linux/amd64 -t groot-training .
docker tag groot-training:latest $ECR_URI:latest
docker push $ECR_URI:latest
cd ../..
```

For everyone else, the CodeBuild image above is all you need.
</details>

---

## Step 5: Launch Training (Smoke Test)

The training runs as a **SageMaker Pipeline** (`groot-finetune-pipeline`) that
trains and then registers the model. `pai groot launch` resolves your account's
bucket, role, and ECR image from the Foundation stack, creates the pipeline if it
doesn't exist yet, and starts an execution — all in one command.

**First, preview it (free — makes no AWS calls):**

```bash
pai groot launch --dry-run
```

This prints the resolved bucket/role/image and the pipeline parameters
(dataset prefix, max-steps, instance type, and the `groot-models` registry it
registers to) so you can sanity-check before spending anything. If you've set
`HF_TOKEN`, it's forwarded to the pipeline but never echoed (shown as `<redacted>`).

**Then launch a 100-step smoke run** (~15 min, ~$2):

```bash
pai groot launch --max-steps 100
```

<details>
<summary>Under the hood (raw commands)</summary>

`pai groot launch` is the two-step pipeline create + execute:

```bash
# Create the pipeline (one-time; safe to re-run — it updates in place):
python training/gr00t/pipeline.py --create \
  --s3-bucket $BUCKET \
  --role-arn $ROLE_ARN \
  --ecr-image $ECR_URI:latest \
  --region us-west-2

# Execute a 100-step smoke run:
python training/gr00t/pipeline.py --execute \
  --max-steps 100 \
  --dataset-prefix groot-data/ur3 \
  --region us-west-2
```

</details>

The pipeline handles:
1. Provisioning an ml.g5.12xlarge instance (4× A10G GPUs)
2. Pulling your container from ECR
3. Downloading the UR3 dataset from S3
4. Downloading GR00T N1.6-3B base model from HuggingFace
5. Fine-tuning the vision tower + projector + action head (LLM backbone frozen) for 100 steps
6. Generating an eval report (action prediction error)
7. Registering the trained model to the `groot-models` Model Registry
8. Terminating the instance (no idle charges)

**What the pipeline adds over a raw training job:** automatic model registration
to the `groot-models` registry, versioned model packages with an approval workflow,
a DAG view in SageMaker Studio, and re-execution with new parameters without editing code.

> **GPU-memory note:** GR00T N1.6 fine-tuning fits the 24 GB A10Gs in
> `ml.g5.12xlarge` via gradient checkpointing + DeepSpeed ZeRO-2 + gradient
> accumulation (the training core is ported from a validated reference). A single
> A10G OOMs — the 4-GPU box is required. If a run still OOMs, lower `batch_size` or
> raise `gradient_accumulation_steps` (see Troubleshooting).

---

## Step 6: Monitor

```bash
# List recent pipeline executions and their status
pai groot runs
```

The execution kicks off a training job named `groot-finetune-<timestamp>`. Check it
directly with `pai rl status <job-name>` (works for any SageMaker training job).

<details>
<summary>Under the hood (raw command)</summary>

```bash
# `eval "$(pai env)"` sets this too. The pipeline's training jobs contain "TrainGR00T".
export JOB_NAME=$(aws sagemaker list-training-jobs --name-contains TrainGR00T \
  --sort-by CreationTime --sort-order Descending --max-results 1 \
  --query 'TrainingJobSummaries[0].TrainingJobName' --output text)

aws sagemaker describe-training-job --training-job-name $JOB_NAME \
  --query '{Status:TrainingJobStatus,Secondary:SecondaryStatus}'
```

</details>

Status progression: `Pending` → `Downloading` → `Training` → `Uploading` → `Completed`

---

## Step 7: Review Results

```bash
eval "$(pai env)"   # BUCKET + JOB_NAME (latest GR00T training job)

# The eval report and training metadata are also uploaded as a small
# output.tar.gz (a few KB) next to the multi-GB model.tar.gz:
aws s3 cp "s3://$BUCKET/pipeline-output/$JOB_NAME/output/output.tar.gz" /tmp/
mkdir -p /tmp/model-output
tar -xzf /tmp/output.tar.gz -C /tmp/model-output/

# View eval report
cat /tmp/model-output/eval_report.json
```

> Jobs trained before this change have the report only inside `model.tar.gz`, as
> its last file. To read it without downloading the archive, stream it in AWS
> CloudShell (same region):
> `aws s3 cp "s3://$BUCKET/pipeline-output/$JOB_NAME/output/model.tar.gz" - | tar -xzO --occurrence=1 --wildcards '*eval_report.json'`

**What the eval report shows:**
- `baseline_mse_overall` — error if you always predicted the mean action (worst reasonable)
- `naive_prediction_mse_overall` — error if you just repeated the previous timestep
- `status` — currently `dataset_baselines_only`. **These are dataset reference lines,
  not a model evaluation.** Scoring the trained checkpoint open-loop (loading it with
  `Gr00tPolicy` and comparing predicted vs ground-truth actions) is a deliberately
  deferred follow-up — we don't ship checkpoint-inference code we can't validate on a
  GPU. Until then, judge training from the loss curve in the CloudWatch/TensorBoard logs.

---

## Step 7b: List recent runs

```bash
pai groot runs
```

<details>
<summary>Under the hood (raw command)</summary>

```bash
python training/gr00t/pipeline.py --list-runs --region us-west-2
```

</details>

---

## Step 8 (Optional): Full Training Run

Once the smoke test passes, kick off the real training (same command, more steps):

```bash
pai groot launch --max-steps 5000
```

<details>
<summary>Under the hood (raw command)</summary>

```bash
python training/gr00t/pipeline.py --execute --max-steps 5000 \
  --dataset-prefix groot-data/ur3 --region us-west-2
```

</details>

This runs for several hours (~11 hrs, ~$79 at the smoke-test config). With more steps
the training loss should drop substantially; the exact curve depends on the data and the
24 GB-fit settings, and **the numbers in this lab have not yet been validated on a real
g5 run** — treat them as expectations, not measured results.

---

## Step 9: Deploy the Fine-Tuned Model

Serve the trained policy as a SageMaker real-time endpoint so an application (or a
robot) can ask it for actions. The endpoint runs the `groot-inference` container
(built in Lab 0 alongside the training image) and loads your `model.tar.gz`.

```bash
# The model.tar.gz of the latest completed training job. `pai env` reads it from
# the job (ModelArtifacts.S3ModelArtifacts), under s3://$BUCKET/pipeline-output/:
eval "$(pai env)" && echo "$MODEL_S3"

# Preview exactly what gets created (no AWS calls):
pai groot deploy --model-s3 "$MODEL_S3" --endpoint-name groot-ur3 --dry-run

# Deploy (creates model → endpoint-config → endpoint; ~10–30 min to come InService):
pai groot deploy --model-s3 "$MODEL_S3" --endpoint-name groot-ur3

# Wait until it can take requests. While it's still "Creating", invoke fails with
# a misleading "Endpoint groot-ur3 ... not found":
aws sagemaker wait endpoint-in-service --endpoint-name groot-ur3
```

If the endpoint ends up `Failed` with `InsufficientInstanceCapacity`, the region had
no free capacity for that GPU type: `pai groot delete --endpoint-name groot-ur3`,
then redeploy with another type (`--instance-type ml.g6.2xlarge` or `ml.g5.4xlarge`)
or try again later.

The endpoint runs on `ml.g5.2xlarge` (GR00T inference needs a GPU). GR00T loads
slowly, so the container's startup health-check timeout is set to 30 minutes.

**Call the endpoint** — give it a wrist image + the 7D robot state + the task.
Take a real one from your converted dataset (episode 0, first frame), so the input
looks like what the model trained on:

```bash
eval "$(python - <<'PY'
import cv2, json, shlex, pandas as pd
root = "training/data/ur3_lerobot_dataset"
ok, frame = cv2.VideoCapture(f"{root}/videos/chunk-000/observation.images.wrist/episode_000000.mp4").read()
cv2.imwrite("wrist.jpg", frame)
df = pd.read_parquet(f"{root}/data/chunk-000/episode_000000.parquet")
tasks = [json.loads(line)["task"] for line in open(f"{root}/meta/tasks.jsonl")]
print("export STATE=" + ",".join(f"{v:.4f}" for v in df["observation.state"].iloc[0]))
print("export TASK=" + shlex.quote(tasks[int(df["task_index"].iloc[0])]))
PY
)"
echo "$STATE | $TASK"
```

Then call it:

```bash
pai groot invoke --endpoint-name groot-ur3 \
  --image-path wrist.jpg \
  --state "$STATE" \
  --task "$TASK"
# → {"actions": [[vx, vy, vz, rx, ry, rz, gripper], ...], "action_dim": 7}
```

After a 100-step smoke run the actions only show the endpoint works; judge policy
quality after the full run (Step 8).

**Tear it down** when finished (a running GPU endpoint bills continuously):

```bash
pai groot delete --endpoint-name groot-ur3
```

<details>
<summary>Under the hood (raw commands)</summary>

```bash
python training/gr00t/deploy_endpoint.py --model-s3 "$MODEL_S3" \
  --endpoint-name groot-ur3 --dry-run
python training/gr00t/deploy_endpoint.py --model-s3 "$MODEL_S3" --endpoint-name groot-ur3

python training/gr00t/deploy_endpoint.py --invoke --endpoint-name groot-ur3 \
  --image-path wrist.jpg --state "$STATE" --task "$TASK"

python training/gr00t/deploy_endpoint.py --delete --endpoint-name groot-ur3
```

</details>

> **Status:** the deploy/serve path is wired against the proven GR00T N1.6 serving
> API. Standing up a live endpoint needs a GPU instance and is billed hourly — run
> it when you're ready to serve, and `pai groot delete` when done.

---

## Step 10 (Optional): Close the Loop on a Physical UR3

This is the back bookend of the full hardware loop: let the deployed endpoint drive a
real arm. Like the optional recording in Step 2, it needs a physical UR3 + wrist
camera. Everything above this point is cloud-only.

`pai groot control` captures the wrist camera, asks the endpoint for an action chunk,
executes the first few actions on the arm, then re-queries — receding-horizon control
at the rate the model trained on (5 Hz).

```bash
export ROBOT_IP=192.168.1.100    # your UR3's IP

# Preview (no motion, no endpoint call):
pai groot control --task "pick up the red cube" --endpoint-name groot-ur3 --dry-run

# Run it for real — THE ARM WILL MOVE:
pai groot control --task "pick up the red cube" --endpoint-name groot-ur3 --max-queries 20
```

Each cycle: grab a wrist frame + read the 7D state → POST to the endpoint → get a
16-step action chunk (`[vx, vy, vz, rx, ry, rz, gripper]`) → execute the first 4 steps
via URScript `speedl` (clamped by the safety controller) → re-query. Add
`--save-images` to dump the frames the policy saw to `/tmp` for debugging.

> **Safety:** the arm moves autonomously here. Clear the workspace, keep the e-stop in
> hand, and start with `--max-queries` small. `SafeUR3Controller` clamps joint,
> workspace, and velocity limits and trips on force — but it is not a substitute for
> the physical e-stop.
>
> **Status:** the closed-loop control path is ported from a validated GR00T reference
> and its endpoint contract is verified against a live SageMaker endpoint, but the
> on-arm motion has not been re-verified on physical hardware in this repo — smoke-test
> carefully before relying on it.

<details>
<summary>Under the hood (raw command)</summary>

```bash
ROBOT_IP=192.168.1.100 python -m robot.ur3.control "pick up the red cube" \
  --endpoint groot-ur3 --max-queries 20
```

</details>

---

## ✅ Lab 1 Checkpoint

You've completed Lab 1 if you can answer:
- [ ] What dataset did we train on? (27 real UR3 pick-and-place episodes, recorded via Xbox controller teleop)
- [ ] What format does GR00T expect? (LeRobot v2 — parquet + MP4, with modality.json for embodiment config)
- [ ] What parts of GR00T get trained? (vision tower + projector + diffusion action head; the LLM backbone stays frozen)
- [ ] What instance type did we use? (ml.g5.12xlarge — 4× A10G GPUs)
- [ ] Where is the trained model? (S3 bucket + Model Registry `groot-models`)
- [ ] What's the limitation? (imitation only — fails on unseen variations → Lab 4 fixes this with RL)
- [ ] Which steps need a physical robot, and which don't? (only the optional record/control bookends need a UR3; convert → train → deploy are cloud-only)

---

## What's Next: Lab 2

Lab 1 gave you a trained policy. Next, you'll set up a visual development workstation to see what's happening inside the Isaac Lab simulation — essential for debugging RL environments before running headless training at scale.

**Lab 2 deploys an Isaac Sim workstation** where you can:
- Visually watch the robot attempt tasks in simulation
- Iterate on reward functions and environment design with instant feedback
- Test training containers locally before sending to SageMaker
- Debug physics issues you can't diagnose from logs alone

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ResourceLimitExceeded` on ml.g5.12xlarge | Request GPU quota increase in Service Quotas console |
| Container pull fails | Ensure ECR URI is correct and in same region as training job |
| `HF_TOKEN` error during training | Set `HF_TOKEN` env var (the base model is ungated, but a token avoids download rate limits) |
| Training job stays in `Pending` for >20 min | GPU capacity shortage — try a different AZ or instance type |
| Eval report missing from model.tar.gz | Check CloudWatch logs for errors in the eval step |
| CUDA out-of-memory during training | A10G is 24 GB. The container already uses gradient checkpointing + ZeRO-2 + grad accumulation; lower the `batch_size` hyperparameter (e.g. 4 or 2) or raise `gradient_accumulation_steps` |
| Job fails with "GR00T SDK not available" | Expected if the container wasn't built with the SDK — the job now fails loudly (non-zero exit) instead of silently producing an empty model. Rebuild the container (Step 4) |

---

**Previous:** [← Lab 0: Prerequisites](lab-0-prerequisites.md)
**Next:** [Lab 2: Isaac Sim Workstation →](lab-2-isaac-workstation.md)
