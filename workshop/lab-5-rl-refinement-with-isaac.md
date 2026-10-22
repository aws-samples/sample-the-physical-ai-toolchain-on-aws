# Lab 4: RL Policy Training in Simulation

**Goal:** Train a robust robot policy via reinforcement learning (RL) in Isaac Lab simulation with domain randomization (validated on the built-in Anymal locomotion task)
**Time:** 3 hours (30 min hands-on + training runs in background)
**Cost:** ~$3 for smoke test (50 iterations), ~$28 for full training (2000 iterations). Full cost breakdown is in the [main README](../README.md#estimated-costs).

> **New to RL?** Reinforcement learning trains a policy through trial-and-error in
> simulation — the robot tries millions of times, guided by a reward signal, until it
> discovers how to succeed. See the [terminology guide](README.md#physical-ai-terminology)
> for key concepts.

---

## 🏃 Quick Runbook (do this in order)

> Lab 4 is **standalone RL** — it does **not** require Lab 1 or any GR00T model. Follow top-to-bottom; each step says what to run and how you know it worked.

| # | Action | Command (summary) | ✅ Success check |
|---|--------|-------------------|-----------------|
| 0 | Confirm prerequisites | `pai doctor` | all checks pass |
| 1 | Launch the RL smoke test | `pai rl launch --max-iterations 50 --instance-type ml.g5.xlarge` | prints `Launched. Monitor: …` with a job name |
| 2 | Watch the SageMaker job | `pai rl status <name>` | status `InProgress` → `Completed` |
| 3 | (Optional) full training | `pai rl launch --task Isaac-Velocity-Flat-Anymal-D-v0 --max-iterations 1500 --instance-type ml.g5.12xlarge` | job launches; logs `Mean reward` climbing |
| 3b | (Optional) scale out | Multi-GPU one box: `--instance-type ml.g5.12xlarge --instance-count 1` (✅ validated). Multi-node: add `--instance-count 2` (SageMaker) or `--engine batch --num-nodes 2` (wired, cross-node **unvalidated**) | multi-GPU: 4 ranks bind cuda:0–3, reward climbs. multi-node: job launches across N nodes |
| 4 | Export + evaluate + watch on the **Lab 2 workstation** | 4a: `play.py … --video --video_length 1` exports `policy.{pt,onnx}` + an MP4 and self-exits. 4b: `evaluate.py` scores it. 4c: `play.py` (no `--headless`) renders it live | `exported/policy.onnx` written; `eval_metrics.json` has `success_rate_pct` (validated on L40S) |
| 5 | Ship the `.onnx` for deployment | push `policy.onnx` to S3 | `policy.onnx` in S3 |

**Before you start, confirm:**
- [ ] AWS credentials active for the **test account** (`pai doctor` checks this)
- [ ] `config.json` `aws.region` matches where your Foundation stack / ECR lives
- [ ] Foundation stack deployed → the `isaac-lab` training image is in your ECR (`pai doctor` checks this)
- [ ] GPU quota for `ml.g5.xlarge` (smoke test) or `ml.g5.12xlarge` (full run) — see Lab 0 quota preflight

> 💸 **Cost reminder:** the smoke test (Step 1) is ~$3; full training (Step 3) is ~$28. SageMaker tears the instance down when the job ends — no manual stop needed (unlike Lab 2's EC2 box).

> ⚠️ **Honest status:** the validated path is Isaac Lab's built-in `Isaac-Velocity-Flat-Anymal-D-v0` task. The custom UR3 env (`PickAndPlaceUR3-v0`) is registered but **not yet wired into the container**, so use the Anymal task for a green run today. Details in Step 1.

---

## What You're Building

You'll train a robot policy from scratch with reinforcement learning (RL) in Isaac Lab.
The pipeline: define the task + reward, Isaac Lab runs **4096 randomized copies** of the
env in parallel on one GPU, **PPO (Proximal Policy Optimization — the standard RL algorithm for robotics)** updates the policy to maximize reward, checkpoint to S3.

**Procedural domain randomization** is what makes the policy robust — every env copy varies
the scene (object position ~5cm, lighting 1000-5000 lux, object color, camera noise), so the
policy only scores well if it succeeds across *all* variations, which is what transfers to
the real world. Reward shaping is the usual mix: +1.0 place, +0.3 approach, -0.1/step, -0.5
collision.

**RL vs. imitation learning (Lab 1):** two independent ways to get a policy — IL copies human
demos (best with good demos, weak sim); RL learns from reward (best with a strong sim + clear
success signal). Both produce a deployable `.onnx` policy. You can run Lab 4 on its own — it doesn't read
a GR00T model or any Lab 1 output.

---

> 📘 **What's custom vs. NVIDIA's:** training (`rsl_rl/train.py`), export (`rsl_rl/play.py`), and
> TensorRT (`trtexec`) are all NVIDIA's stock tools — this repo only adds a thin metrics aggregator
> (`evaluate.py`) for the success rates Isaac Lab doesn't report, and the AWS wiring
> (CodeBuild→ECR, SageMaker/Batch, CDK). The value here is *how it's wired*, not custom RL code.

> **Prerequisites:** the runbook's "Before you start" checklist (Foundation deployed → `isaac-lab`
> image in ECR, GPU quota). See [Lab 0](lab-0-prerequisites.md).

---

## Step 1: Launch RL Training

> 🦾 **Want to train a robot ARM instead?** The same pipeline works on Isaac Lab's built-in UR10 arm task (arm reaching instead of quadruped walking). Change the `--task` flag to a UR10 task ID.

All RL training is launched from your laptop with **`pai rl launch`** — it builds the
SageMaker job (resolving your account's ECR image, role, and output bucket) and
submits it.

**First, preview the job (free — makes no AWS calls):**

```bash
pai rl launch --dry-run
```

This prints the exact `create_training_job` request it would submit (image, role,
output path, hyperparameters) so you can sanity-check it before spending anything.

**Then launch the smoke test on the validated built-in task:**

```bash
# Smoke test: 50 iterations (~5-10 min, ~$3) on the validated Anymal task
pai rl launch \
  --task Isaac-Velocity-Flat-Anymal-D-v0 \
  --num-envs 4096 \
  --max-iterations 50 \
  --instance-type ml.g5.xlarge
```

<details>
<summary>Under the hood (raw commands)</summary>

```bash
python training/scripts/launch_rl.py --dry-run

python training/scripts/launch_rl.py \
  --task Isaac-Velocity-Flat-Anymal-D-v0 \
  --num-envs 4096 \
  --max-iterations 50 \
  --instance-type ml.g5.xlarge
```

</details>

**What happens:**
1. Launches Isaac Lab RL training on SageMaker (ml.g5.xlarge, 4096 parallel envs)
2. Trains the policy with PPO from scratch in headless Isaac Sim
3. Policy learns through trial-and-error guided by reward signals
4. Saves a checkpoint (`model_*.pt`) + training metadata to S3

`pai rl launch` prints a job name (`isaac-lab-rl-<id>`). Monitor it with:

```bash
pai rl status <name>     # InProgress → Completed
```

50 iterations is a *smoke test* — it proves the pipeline runs end-to-end. The
resulting policy will stumble, not perform well; a usable policy needs ~1000+
iterations (see Step 3).

> Use the built-in **Anymal** task here — it's the validated path for these labs. A UR3
> pick-and-place env (`PickAndPlaceUR3-v0`) ships as an **optional, not-yet-container-wired**
> task for future work, so `pai rl launch` warns if you point at it. (Details in
> [ROADMAP](../docs/ROADMAP.md), Feature 2.)

> Need to rebuild the Isaac Lab container after changing its Dockerfile? Trigger
> the cloud build with `pai deploy foundation` (redeploys and rebuilds all images)
> or directly with `aws codebuild start-build --project-name physical-ai-isaac-lab-build`.

## Step 2: Understand the RL Environment

Tasks come from two places: **built-in** tasks ship inside the `isaac-lab` container (Isaac
Lab's own library), and **custom** tasks live in this repo under `training/envs/`. These labs
use the built-in **Anymal** locomotion task (`Isaac-Velocity-Flat-Anymal-D-v0`) — the validated
path. It defines the observation/action spaces, the reward (velocity-tracking with stability
penalties), and the domain randomization; you don't author any of that, you just train it.

> **Custom env (optional, future work):** `training/envs/pick_and_place_ur3.py` is a UR3
> arm pick-and-place env included as a starting point for your own task. It's registered but
> not yet wired into the container's training entrypoint, so it's not part of the validated
> flow today — see [ROADMAP](../docs/ROADMAP.md), Feature 2.

---

## Step 3: Launch Full RL Training (Optional)

Same command as the smoke test, scaled up — more iterations, a bigger instance, a longer
runtime cap. Add `--dry-run` first to preview:

```bash
pai rl launch \
  --task Isaac-Velocity-Flat-Anymal-D-v0 \
  --num-envs 4096 --max-iterations 1500 \
  --instance-type ml.g5.12xlarge --runtime-min 240
```

A checkpoint is saved to S3 every 100 epochs; after ~1500 iterations (~2-4 hours) the
policy reaches high success rates. Checkpoints land as `model_*.pt` in your output bucket.

> **What actually trains:** `pai rl launch` submits a job whose container entrypoint
> (`containers/isaac-lab/train_entrypoint.py`) runs **NVIDIA's own** Isaac Lab trainer —
> `scripts/reinforcement_learning/rsl_rl/train.py`, *inside* the image — with the `task`,
> `num_envs`, and `max_iterations` you pass. You don't write or run a training script; the
> official rsl_rl sample does the PPO loop. (This is the same script Lab 2 runs interactively.)

---

## Step 3b: Scale Out Across Multiple Nodes (Optional)

A single GPU instance is enough for the workshop tasks. When you outgrow one box —
bigger models, more parallel envs, faster wall-clock — RL training scales across
**multiple nodes** two ways. Both reuse the **same `physical-ai/isaac-lab` container**;
the only difference is who provisions the fleet and wires the NCCL topology.

> ✅ **Multi-GPU validated on hardware:** a single-node 4-GPU run (`ml.g5.12xlarge`, 4× A10G)
> was confirmed end-to-end — torchrun spawned 4 ranks, each bound to a distinct GPU
> (`Environment device: cuda:0/1/2/3`) with per-rank seed offsets (42/43/44/45), NCCL gradient
> sync active, and reward climbed -0.36 → 13.56 over 100 iterations. The intra-node path
> (`--instance-count 1` with a multi-GPU instance) is proven.
>
> ⚠️ **Multi-node networking (`--instance-count`/`--num-nodes` ≥ 2) is wired but not yet validated**
> — cross-node NCCL over the cluster fabric hasn't been exercised on real hardware. Treat a green
> multi-node launch as untested until you've confirmed reward climbs across nodes.

### Option A — SageMaker multi-instance (quickest)

The container's SageMaker entrypoint already parses the cluster's `resourceconfig.json`
and launches `torchrun` across however many instances SageMaker provisions — the **only**
thing you change is `--instance-count`:

```bash
# Add --dry-run to preview (no AWS calls); drop it to launch.
pai rl launch \
  --task Isaac-Velocity-Flat-Anymal-D-v0 \
  --num-envs 4096 --max-iterations 100 \
  --instance-type ml.g5.12xlarge --instance-count 2
```

SageMaker handles inter-node networking and tears the fleet down when the job ends.

> ✅ **Tip — the validated middle ground:** before going multi-*node*, scale up to a multi-GPU
> *single* instance (`--instance-type ml.g5.12xlarge --instance-count 1`). That intra-node path
> is hardware-validated (4 ranks → cuda:0–3, NCCL sync, reward climbs) and avoids cross-node
> network variables entirely — often all the throughput a workshop task needs.

### Option B — AWS Batch Multi-Node Parallel (the reference architecture)

Batch gives you direct control over the EC2 fleet (g6e.4xlarge, L40S GPU), a shared
checkpoint pipeline to S3, and a self-managed NCCL security group.

> **Full step-by-step Terraform guide:** For deploying Batch infrastructure from scratch
> (VPC, compute environment, job queue, container build, job submission, and S3 checkpoint
> upload), see **[isaac-lab-on-aws/batch-rl-training-guide.md](../isaac-lab-on-aws/batch-rl-training-guide.md)**.

Quick version (if infrastructure is already deployed):

```bash
# 1. Deploy the opt-in Batch stack (isaac-lab image must already be in ECR).
pai deploy batch
# Reads `batch` settings from config.json and prints a ready-to-run LaunchCommand
# with the exact queue/job-definition names.

# 2. Submit (add --dry-run to preview). Queue/def default to
#    physical-ai-dev-rl-queue / -rl-mnp; override with --job-queue / --job-definition.
pai rl launch --engine batch \
  --task Isaac-Velocity-Flat-Anymal-D-v0 \
  --num-envs 4096 --max-iterations 100 --num-nodes 2

# 3. Monitor
pai rl status --engine batch <job-id>
```

> **`--num-nodes` must match the deployed job definition.** The Batch stack bakes a fixed
> node count into its job definition (default **2**), so `--num-nodes 2` is the only value
> that submits as-is — `pai rl launch` checks this and errors clearly if they differ. To run
> a different node count, redeploy the Batch stack with the new `batch.numNodes` in
> `config.json`, then pass the matching `--num-nodes`.

> ⚠️ **GPU quota:** g6.12xlarge needs *G-family On-Demand* vCPU quota you may not have by
> default — request it in Service Quotas first, or the compute env sits at 0 vCPUs and jobs
> stay `RUNNABLE` forever. Also needs a **default VPC** (same as the Lab 2 workstation).

Checkpoints are uploaded to S3 at `s3://<CHECKPOINT_BUCKET>/checkpoints/<job-id>/` after
training completes. The container uses boto3 for the upload (AWS CLI is not installed in
the Isaac Lab image).

| | SageMaker multi-instance | AWS Batch MNP |
|---|---|---|
| Setup | none (just `--instance-count N`) | Terraform deploy (see [Batch guide](../isaac-lab-on-aws/batch-rl-training-guide.md)) |
| Provisioning | managed by SageMaker | managed EC2 compute env you own |
| Shared storage | S3 only | S3 (checkpoints uploaded via boto3) |
| Teardown | automatic on job end | job auto-terminates; infra persists until `terraform destroy` |
| Best for | quick scale-out, least moving parts | full control, EC2-priced fleets, custom VPC |

---

## Step 4: Evaluate the Trained Policy

Do **4a (export)** first, then **4b (eval)**; **4c** lets you watch the policy run. All three run
inside the **`isaac-lab` container** on the Lab 2 workstation (see [Lab 2 → "One environment for everything"](lab-2-isaac-workstation.md#one-environment-for-everything-the-isaac-lab-container)).
The `pai` CLI isn't installed there, so the commands below call the scripts directly.

> ✅ **Validated on L40S** — export + both eval paths ran 5 Anymal episodes end-to-end. On a
> 50-iter smoke policy the success *number* is meaningless; this proves the plumbing, not quality.

Enter the container:

```bash
~/run-isaac-lab.sh                      # prompt becomes /workspace/isaaclab#
cd /workspace/toolchain                 # the mounted repo
```

### Step 4a: Export the policy

> 📘 A raw training checkpoint isn't deployable on its own. NVIDIA's stock `play.py` is the
> canonical exporter — it bakes the **observation normalizer** into `policy.pt` (TorchScript) +
> `policy.onnx` (Open Neural Network Exchange — a portable format that TensorRT can compile for edge hardware). There's no separate export script; `play.py` exports on every run.

**1. Pull the checkpoint from S3 — on the workstation HOST** (the container has no AWS CLI). The
host repo `~/aws-physical-ai-toolchain` is mounted at `/workspace/toolchain`, so files extracted
here appear in the container:

```bash
cd ~/aws-physical-ai-toolchain
BUCKET=$(aws sts get-caller-identity --query Account --output text | xargs -I{} echo physical-ai-dev-datasets-{})
aws s3 cp "s3://$BUCKET/isaac-lab/output/<job-name>/output/model.tar.gz" .
tar -xzf model.tar.gz                       # → logs/rsl_rl/<task>/<timestamp>/model_<N>.pt
ls logs/rsl_rl/*/*/model_*.pt               # confirm it's under ./logs (not ~/logs)
```

> rsl_rl numbers iterations **from 0**, so a 50-iter run saves `model_49.pt`.

**2. Export — inside the container.** `play.py` loads the checkpoint and writes
`exported/policy.{pt,onnx}` + a playback MP4:

```bash
/workspace/isaaclab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Velocity-Flat-Anymal-D-v0 \
  --checkpoint /workspace/toolchain/logs/rsl_rl/anymal_d_flat/<timestamp>/model_49.pt \
  --num_envs 1 --headless --video --video_length 1
```

> ⚠️ Use the **absolute** `/workspace/toolchain/...` checkpoint path — a bare `logs/...` resolves
> against play.py's own dir and fails. `--video --video_length 1` makes play.py **exit on its own**
> (without it the sim loops forever); it may still pause ~1 min in shutdown *after* writing — the
> files, not a clean exit, are the success signal.

**3. Copy the artifacts** somewhere stable for Steps 4b/5:

```bash
mkdir -p /workspace/toolchain/model_exported
cp /workspace/toolchain/logs/rsl_rl/anymal_d_flat/<timestamp>/exported/policy.* \
   /workspace/toolchain/model_exported/
ls /workspace/toolchain/model_exported/     # policy.pt + policy.onnx
```

### Step 4b: Evaluate the policy

Step 4a already gave you the playback MP4. `evaluate.py` adds the metrics play.py doesn't report —
it loads the exported `policy.pt`, runs N episodes in-process, and writes `eval_metrics.json`
(`success_rate_pct`, `avg_reward`, `avg_cycle_time_sec`, `failure_modes`):

```bash
/workspace/isaaclab/isaaclab.sh -p /workspace/toolchain/training/scripts/evaluate.py \
  --env Isaac-Velocity-Flat-Anymal-D-v0 \
  --checkpoint /workspace/toolchain/model_exported/policy.pt \
  --num-episodes 5 --output-dir /workspace/toolchain/eval_results/
```

> ⚠️ Use **absolute `/workspace/toolchain/...` paths** — `isaaclab.sh` runs from `/workspace/isaaclab`,
> so relative paths to repo scripts/files won't resolve. `--num-episodes 5` is a smoke run; use
> **100+** for a real policy. Isaac Sim may hang ~1 min on exit *after* writing — `eval_metrics.json`
> is your confirmation.

<details>
<summary>Advanced (optional): closed-loop eval over ZMQ</summary>

A deployment-representative variant that splits policy and simulator into two processes over ZMQ
(port 5555) — the same client/server split NVIDIA GR00T uses to serve a real robot. Not needed for
evaluation; it just shows the serving topology. Needs **two shells** in the container:

```bash
# Shell 1 — policy server (binds tcp://127.0.0.1:5555, localhost only):
python /workspace/toolchain/training/scripts/eval_policy_server.py \
  --checkpoint /workspace/toolchain/model_exported/policy.pt --device cuda

# Shell 2 — sim client (boots Isaac Sim, so run via isaaclab.sh):
/workspace/isaaclab/isaaclab.sh -p /workspace/toolchain/training/scripts/eval_sim_client.py \
  --task Isaac-Velocity-Flat-Anymal-D-v0 --endpoint tcp://127.0.0.1:5555 \
  --eval-rounds 5 --output-dir /workspace/toolchain/eval_results
```

Attach the second shell with `sudo docker exec -it $(sudo docker ps -q -l) bash`. Writes the same
`eval_metrics.json`. ([GR00T server/client reference](https://github.com/NVIDIA/Isaac-GR00T/blob/main/gr00t/policy/server_client.py).)

</details>

### Step 4c: Watch your policy run (visual)

Metrics tell you *how well* the policy does; this lets you *watch* it. The same `play.py` from
Step 4a, run **without** `--headless`/`--video`, opens a live render window and loops the policy
forever so you can see the trained Anymal trot:

```bash
# Run from the DCV desktop (needs the X display — NOT via SSM):
/workspace/isaaclab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Velocity-Flat-Anymal-D-v0 \
  --checkpoint /workspace/toolchain/logs/rsl_rl/anymal_d_flat/<timestamp>/model_49.pt \
  --num_envs 50
```

> ⚠️ Pass the **raw `model_49.pt` checkpoint** (play.py rebuilds the runner from it), not the
> exported `policy.pt`. First load takes ~2–3 min; `Ctrl+C` to quit. If the viewport looks empty,
> scroll out — the robots spawn small relative to the default camera. `--num_envs 50` shows a grid
> of robots; drop to `1` to watch a single one.

Compare this to the untrained robots flailing at the start of training (Step 1) — the policy now
holds a stable gait and tracks the commanded velocity. That contrast is the payoff of RL refinement.

---

## Step 5: Ship the policy for edge deployment

Your deployable artifact is the `policy.onnx` from Step 4a — there's nothing to re-export here.
Just push it to S3 for deployment.

> 📘 The TensorRT engine (`.trt`) is **not portable** across GPUs, so it's compiled on the
> target hardware (`trtexec --onnx=policy.onnx --saveEngine=policy.trt --fp16`). Ship the `.onnx`.

**On the workstation HOST** (no AWS CLI in the container; the export wrote into the mounted tree):

```bash
cd ~/aws-physical-ai-toolchain
BUCKET=$(aws sts get-caller-identity --query Account --output text | xargs -I{} echo physical-ai-dev-datasets-{})
aws s3 cp ./model_exported/policy.onnx "s3://$BUCKET/isaac-lab/exported/policy.onnx"
```

---

## ✅ Lab 4 Checkpoint

You've **run** Lab 4 if:
- [ ] `launch_rl.py --dry-run` printed a valid job spec with your account's image/role
- [ ] A real RL job ran to `Completed` on a G-family instance (no GR00T involved)
- [ ] A checkpoint (`model_*.pt`) landed in S3

And you can **explain** the concepts:
- [ ] How does RL learn? (trial-and-error guided by reward signals in simulation)
- [ ] What is domain randomization? (randomized scene parameters so policy must be robust to succeed)
- [ ] How many parallel environments run simultaneously? (4096 on one GPU)
- [ ] What reward signal drives improvement? (success/failure at the manipulation task)
- [ ] What success rate does RL with domain randomization achieve? (~93-95% on randomized tasks)
- [ ] Where is the trained model? (S3 checkpoint + optionally TensorRT export)

---

## What You'll See During Training

Isaac Lab logs each PPO iteration:

```
Learning iteration 1/500
Computation: 3741 steps/s (collection: 0.715s, learning 0.106s)
Mean reward: -0.90
Mean episode length: 26.33
```

- **steps/s** — throughput across all parallel envs; bounded by the physics sim
  ("collection"), which is why more GPUs help.
- **Mean reward** — starts negative and should climb. Flat after 50+ iterations → check
  the reward function.
- **Mean episode length** — grows as the robot survives longer before termination.

**What's validated:** the SageMaker integration (entrypoint parses `resourceconfig.json`,
reads hyperparameters, launches `torchrun`; headless render; artifacts to `/opt/ml/model/`)
is load-tested on `Isaac-Velocity-Flat-Anymal-D-v0` — 128 envs, A10G, ~3,741 steps/s.
The UR3 task is registered but not yet container-wired (see Step 1).

> **Cosmos is not part of this lab.** Isaac Lab's procedural domain randomization
> (positions, lighting, textures) handles most sim-to-real transfer on its own; Cosmos
> photorealistic enhancement is a later, NIM-gated add-on (Lab 3). Most teams don't need it.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `docker login nvcr.io` fails | Verify NGC API key at https://ngc.nvidia.com/setup/api-key |
| Container build OOM | Isaac Lab base is ~15 GB. Ensure 30+ GB free disk space |
| Training doesn't converge | Check reward function. Try reducing domain randomization range initially |
| `CUDA out of memory` | Reduce `num_envs` in config (4096 → 2048 → 1024) |
| Policy success rate stays at 0% | Check reward function is correct; verify environment resets properly |

---

**Previous:** [← Lab 4: Cosmos Transfer](lab-4-cosmos-transfer.md)
