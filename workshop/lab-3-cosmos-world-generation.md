# Lab 3: Cosmos World Generation

**Goal:** Use NVIDIA Cosmos 3 Super (a World Foundation Model) to generate synthetic robot demonstrations using the Predict capability — expanding your training dataset without additional teleoperation
**Time:** 1-2 hours (15 min deploy + ~20 min model load + generation)
**Cost:** Capacity Block pricing varies (~$37/hr for p5.48xlarge; minimum block is typically 8-13 hrs). You pay for the full block regardless of usage — generate as many videos as you can.

> **New to World Foundation Models?** See the [terminology guide](README.md#physical-ai-terminology)
> for definitions of Cosmos, VLA, and related concepts.

> **Workshop Note:** This lab requires `p5.48xlarge` (8x H100 80GB) for the Cosmos 3
> generation server. P5 capacity is scarce — this lab uses an **EC2 Capacity Block**
> (reserved GPU allocation) to guarantee availability. In a live workshop:
> - **Instructor-led demonstration** — instructor runs from a pre-approved account
> - **Self-paced** — requires P5 quota + Capacity Block purchase
> - **Pre-baked fallback** — download pre-generated samples from S3 (no GPU needed)
>
> Labs 1, 2, and 4 work on standard instances (g5/g6 family). This lab is optional —
> Lab 4 (RL Policy Training) works without Cosmos using built-in domain randomization.

---

## What You're Building

You have 27 demonstrations from Lab 1. You need hundreds to train a robust policy.
Recording more teleop data is expensive and slow. **Cosmos 3 generates new synthetic
demonstrations** — plausible novel trajectories of the same task — from a text prompt
and a reference video of your starting scene.

**What Cosmos 3 does (Predict mode — video generation):**
- Takes the first frames of your input video as the **starting state** (the table with blocks)
- Generates a **new 8-second video** (189 frames, 1280x720, 24fps) of the robot completing the task
- Each generation with a different seed produces a **different trajectory** (different approach angles, timing, motion style)
- The prompt controls **what happens** (which block, what target, what conditions)

**The value:** From one reference video + varied prompts/seeds, you generate dozens of
plausible demonstrations. This is synthetic data generation for scaling robot learning.

---

## The Demo

```
Lab 1 wrist camera clip (episode_000000.mp4 — the starting scene)
    ↓
Cosmos 3 V2V with task prompt + seed 100
    ↓
Output: 8-sec video of UR3 picking red block and placing on yellow target (novel trajectory)
    ↓
Same prompt + seed 200 → different approach angle
Same prompt + seed 300 → different timing
Different prompt → different block color, lighting conditions
```

One input reference video → many synthetic demonstrations.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  Cosmos 3 Synthetic Demonstration Generation                                 │
│                                                                              │
│  ┌──────────────┐    ┌────────────────────┐    ┌───────────────┐            │
│  │ Your laptop  │    │  EC2 p5.48xlarge    │    │  Output       │            │
│  │              │    │  (Capacity Block)   │    │               │            │
│  │ SSM port-    │    │  ┌──────────────┐  │    │  Synthetic    │            │
│  │ forward      │───▶│  │ vLLM-Omni    │  │───▶│  demo MP4s    │            │
│  │ :8000        │    │  │ Cosmos3-Super│  │    │  (189 frames  │            │
│  │              │    │  │ (8x H100)    │  │    │   1280x720)   │            │
│  │ curl -F      │    │  └──────────────┘  │    │               │            │
│  │ + prompt     │    │                    │    │  Each is a new│            │
│  └──────────────┘    └────────────────────┘    │  trajectory   │            │
│                                                └───────────────┘            │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## How It Works

**Input you provide:**
1. A reference video (your Lab 1 wrist camera clip) — provides the starting scene
2. A detailed text prompt — describes the robot, the action, and the conditions
3. A seed — different seeds produce different trajectories for the same prompt

**What Cosmos 3 generates:**
- A new 189-frame, 1280x720, 24fps video
- Starting from the first frames of your reference (table with blocks visible)
- Showing the robot completing the described task with a novel trajectory
- Photorealistic quality — looks like real footage

**How this scales your dataset:**
```
1 reference video × 5 prompts × 5 seeds = 25 synthetic demonstrations
```

Each is a unique, plausible execution of the pick-and-place task under different conditions.

> **Important:** These synthetic videos do NOT come with action labels (joint positions/velocities).
> They're useful for vision pre-training (teach the model what "pick and place" looks like across
> many variations) and evaluation (does a generated rollout look plausible?). For action-paired
> training data, see the note on Cosmos Transfer 2.5 at the bottom of this lab.

---

## Prerequisites

> **Bring your own data:** The example uses our UR3 pick-and-place video as a reference,
> but you can use any wrist camera MP4 from your own robot or simulation. Write prompts
> that describe YOUR task and environment — the pipeline is the same regardless of robot or use case.

- **Lab 1 completed** — you have a LeRobot v2 dataset with wrist camera MP4s in S3
- **Hugging Face account + token** — with gated model access:
  1. Create an account at https://huggingface.co if you don't have one
  2. Accept the licenses for **both** gated Cosmos models (both are auto-approval):
     - [`nvidia/Cosmos-Guardrail1`](https://huggingface.co/nvidia/Cosmos-Guardrail1) → "Expand to review and access" → accept
     - [`nvidia/Cosmos-1.0-Guardrail`](https://huggingface.co/nvidia/Cosmos-1.0-Guardrail) → "Expand to review and access" → accept
     - (**Critical:** vLLM-Omni downloads BOTH at startup. Missing either → 403 → server crashes.)
  3. Create a **Read** access token:
     - Go to https://huggingface.co/settings/tokens → **"+ Create new token"**
     - Name: `cosmos-aws`, Type: **Read**
     - Copy the `hf_...` value (shown only once)
  4. Store the token in AWS Secrets Manager:
     ```bash
     # Set this once to the region your foundation stack is deployed in — every
     # command in this lab uses it.
     export REGION="<your-region>"

     aws secretsmanager create-secret --name physical-ai/hf-token \
       --secret-string "hf_YOUR_TOKEN_HERE" --region $REGION
     ```
- **P5 quota** — at least 192 vCPUs of "Running On-Demand P instances":
  ```bash
  aws service-quotas get-service-quota --service-code ec2 \
    --quota-code L-417A185B --region $REGION --query 'Quota.Value'
  ```
- **IAM role + instance profile** — with SSM, ECR read, and S3 access (see `docs/cosmos3-validated-runbook.md`)
- **vllm-omni:cosmos3 image in ECR** — Step 1 below

---

## Step 1: Mirror the vLLM-Omni Image to ECR

The Cosmos 3 server uses the official `vllm/vllm-omni:cosmos3` image (~30 GB compressed).
Mirror it to your ECR using CodeBuild (avoids pulling 30 GB locally):

```bash
# Create the ECR repo
aws ecr create-repository --repository-name vllm-omni --region $REGION

# Run the mirror script (creates a CodeBuild project, starts the build).
# It reads AWS_REGION, so pass the same region as above.
AWS_REGION=$REGION bash scripts/mirror-vllm-omni.sh
# → Takes ~15 min. Image lands at: <ACCOUNT>.dkr.ecr.<REGION>.amazonaws.com/vllm-omni:cosmos3
```

---

## Step 2: Launch the Server (Automated)

One script handles everything — Capacity Block purchase, instance launch, image pull,
server start, and readiness check:

```bash
# Preview what it will do (no AWS calls, no cost):
bash scripts/cosmos3-launch.sh --dry-run

# Full deploy (interactive — asks you to confirm the Capacity Block purchase):
bash scripts/cosmos3-launch.sh
```

The script:
1. Creates the IAM role + instance profile (idempotent)
2. Finds the cheapest available Capacity Block and shows the price
3. Asks for confirmation before purchasing
4. Waits for the block to activate (may be immediate or up to 30 min)
5. Launches p5.48xlarge into the block (8x H100, 500 GB EBS)
6. Pulls the image from ECR (~5 min)
7. Starts the Cosmos 3 server and waits for model loading (~15-20 min first time)
8. Prints `COSMOS 3 SERVER IS READY` with your instance ID

> **Manual scan (if you want to check availability before running the script):**
> ```bash
> for REGION in us-east-1 us-east-2 us-west-2; do
>   echo "=== $REGION ==="
>   aws ec2 describe-capacity-block-offerings \
>     --instance-type p5.48xlarge \
>     --capacity-duration-hours 24 \
>     --instance-count 1 \
>     --region $REGION \
>     --query 'CapacityBlockOfferings[*].{Hours:CapacityBlockDurationHours,Cost:UpfrontFee,AZ:AvailabilityZone,Start:StartDate}' \
>     --output table
> done
> ```
> P5 instances are rarely available on-demand (at the time of this writing).
> Block durations and pricing vary by region and time of day.

**Total time from script start to ready:** ~25 min (mostly model weight download).

> **Cost:** Capacity Blocks are priced at ~$37/hr for p5.48xlarge. Minimum block
> durations vary by availability (typically 8-24 hrs). You pay for the entire block
> upfront — but once running, generate as many videos as you want. Each video takes
> ~5 min, so in an 8-hr block you could generate ~90 synthetic demonstrations.
>
> **Tip:** The block runs whether you're generating or not. Plan your prompts in advance
> and batch-generate during the block window to maximize value.

---

## Step 3: Generate Your First Synthetic Demonstration

Copy a reference clip from your Lab 1 dataset to the instance, then generate:

```bash
# Copy reference video to the instance
aws ssm send-command --instance-ids $INSTANCE_ID --document-name AWS-RunShellScript \
  --parameters '{"commands":["aws s3 cp s3://<BUCKET>/groot-data/ur3/dataset/videos/chunk-000/observation.images.wrist/episode_000000.mp4 /tmp/episode_000000.mp4 --region '"$REGION"' && docker cp /tmp/episode_000000.mp4 cosmos3:/tmp/episode_000000.mp4"]}' \
  --region $REGION
```

Generate a synthetic pick-and-place demonstration:

```bash
aws ssm send-command --instance-ids $INSTANCE_ID --document-name AWS-RunShellScript \
  --parameters '{"commands":["docker exec cosmos3 curl -sS -X POST http://localhost:8000/v1/videos/sync -H \"Accept: video/mp4\" -F \"model=nvidia/Cosmos3-Super\" -F \"prompt=A UR3 robot arm with a Robotiq gripper reaches down to a dark matte table, grasps a small red wooden block, lifts it slowly, and places it onto a yellow sticky note approximately 6 inches away. Top-down wrist camera view. Colorful wooden blocks are scattered on the table. Smooth deliberate motion.\" -F \"size=1280x720\" -F \"num_frames=189\" -F \"fps=24\" -F \"num_inference_steps=35\" -F \"guidance_scale=6.0\" -F \"max_sequence_length=4096\" -F \"flow_shift=10.0\" -F \"extra_params={\\\"condition_frame_indexes_vision\\\":[0,1],\\\"condition_video_keep\\\":\\\"first\\\"}\" -F \"seed=100\" -F \"input_reference=@/tmp/episode_000000.mp4;type=video/mp4\" -o /tmp/output.mp4 -w \"\\nHTTP:%{http_code} SIZE:%{size_download}\" --max-time 600"]}' \
  --timeout-seconds 900 --region $REGION
```

**Generation takes ~5-8 min on 8x H100.** HTTP 200 + ~7 MB output = success.

Upload the result:

```bash
aws ssm send-command --instance-ids $INSTANCE_ID --document-name AWS-RunShellScript \
  --parameters '{"commands":["docker cp cosmos3:/tmp/output.mp4 /tmp/output.mp4 && aws s3 cp /tmp/output.mp4 s3://<BUCKET>/cosmos-samples/generated_demo_seed100.mp4 --region '"$REGION"'"]}' \
  --region $REGION
```

---

## Step 3b: Validate the Output in S3

Don't just trust the `HTTP:200` from the generation call — confirm the file actually
landed in S3 and is a real, playable video before moving on.

**1. Confirm the object exists and check its size:**

```bash
aws s3 ls s3://<BUCKET>/cosmos-samples/ --region $REGION --human-readable
```

You should see both the original reference clip and the generated output, e.g.:
```
2026-08-05 14:49:19  861.7 KiB original_episode_000000.mp4
2026-08-05 14:49:19    5.9 MiB augmented_episode_000000.mp4
```

A generated video should be **several MB** (189 frames at 1280x720). If it's only a
few KB, the generation likely failed silently — check `docker logs cosmos3` on the
instance for errors before continuing.

**2. Download both files locally to inspect:**

```bash
mkdir -p ./cosmos-output
aws s3 cp s3://<BUCKET>/cosmos-samples/original_episode_000000.mp4 ./cosmos-output/ --region $REGION
aws s3 cp s3://<BUCKET>/cosmos-samples/generated_demo_seed100.mp4 ./cosmos-output/ --region $REGION
ls -lh ./cosmos-output/
```

**3. Verify it's a valid video (not a truncated/corrupt file):**

```bash
# ffprobe reports duration, resolution, and codec — a corrupt file will error out here
ffprobe -v error -show_entries format=duration,size -show_entries stream=width,height,codec_name \
  ./cosmos-output/generated_demo_seed100.mp4
```

Expect `width=1280`, `height=720`, `duration≈7.9` (189 frames at 24fps), and a valid
`codec_name` (e.g. `h264`). If `ffprobe` errors out or reports `0x0`, the download or
generation was corrupted — re-run Step 3.

**4. Visually compare original vs. generated:**

Play both files side by side (VS Code's built-in preview, `vlc`, `mpv`, or any video
player). You're checking:
- The generated video **starts from the same scene** as the original (same table, same blocks — Cosmos 3 conditions on your input's first frames)
- The action described in your prompt is **visibly happening** (the right block, moving toward the right target)
- No obvious artifacts: flickering, extra/missing objects, physically implausible motion

If the generated video doesn't match your prompt (wrong block color, no motion, hallucinated objects), revisit your prompt — see [Prompting Best Practices](#prompting-best-practices) below. This is expected with vague prompts, not a pipeline bug.

---

## Step 4: Try Different Prompts (~5 min each)

Each generation takes **~5 min on 8x H100**. Change the seed for different trajectories
of the same prompt, or change the prompt for different conditions. Here are tested
variations — try them all while your block is running:

**Variation A — Different block color (blue):**
```
A UR3 robot arm with a Robotiq gripper reaches down to a dark matte table, grasps a
small blue wooden block, lifts it slowly, and places it onto a yellow sticky note.
Top-down wrist camera view. Colorful wooden blocks scattered on the table.
Smooth deliberate motion.
```
*Result: gripper picks up blue block instead of red. Same coherent trajectory.*

**Variation B — Different lighting (dim workshop):**
```
A UR3 robot arm with a Robotiq gripper picks up a small red wooden block from a dark
table and places it on a yellow sticky note. Top-down wrist camera view. Dim overhead
lighting with strong shadows. Colorful blocks on the table.
```
*Result: darker scene, more dramatic shadows. Same task completion.*

**Variation C — Different approach (from the left):**
```
A UR3 robot arm with a Robotiq gripper approaches from the left side, grasps a small
red wooden block from a dark matte table, and places it on a yellow sticky note to the
right. Top-down wrist camera view. Multiple colorful blocks visible. Smooth motion.
```

**Variation D — Different speed (fast):**
```
A UR3 robot arm with a Robotiq gripper quickly grasps a red wooden block from a dark
table, lifts it high, then precisely places it on a yellow sticky note. Top-down wrist
camera view. Colorful blocks on table. Swift confident motion.
```

**Variation E — Failure case (for negative training data):**
```
A UR3 robot arm with a Robotiq gripper attempts to grasp a small red wooden block but
the block slips from the gripper and falls back onto the dark table. Top-down wrist
camera view. Colorful blocks on table. The grasp fails.
```

> **Timing (validated on p5.48xlarge, 8x H100):**
> - First generation after model load: ~8 min (includes JIT warmup)
> - Subsequent generations: ~5 min each
> - All generations produce 189 frames at 24fps (1280x720) = 8-second video, ~7 MB

See `docs/cosmos3-prompt-catalog.md` for the complete catalog with results and lessons learned.

---

## Step 5: Terminate When Done

```bash
aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region $REGION
```

The capacity block expires at its end time regardless. Instance charges stop on termination.

---

## ✅ Lab 3 Checkpoint

- [ ] vLLM-Omni image mirrored to ECR
- [ ] Capacity Block purchased and activated
- [ ] Cosmos 3 server running (8x H100, `/v1/models` returns `nvidia/Cosmos3-Super`)
- [ ] Generated at least one synthetic demonstration (HTTP 200, ~7 MB MP4 output)
- [ ] Confirmed the output in S3 (`aws s3 ls`, correct file size, `ffprobe` reports valid resolution/duration)
- [ ] Visually verified: output shows coherent pick-and-place trajectory
- [ ] (Optional) Generated multiple variations (different seeds/prompts)
- [ ] (Optional) Uploaded all outputs to S3
- [ ] Terminated the instance

---

## Cost Estimation

| Resource | Cost | Notes |
|----------|------|-------|
| Capacity Block (p5.48xlarge) | ~$37/hr | Minimum block varies (8-24 hrs); pay upfront for full block |
| S3 storage (generated videos) | ~$0.001 | ~7 MB per video |
| CodeBuild (image mirror, one-time) | ~$0.50 | 15 min on BUILD_GENERAL1_LARGE |

**Example:** An 8-hr block at $37/hr = ~$296. In that time you can generate ~90 videos.
That's ~$3.30 per synthetic demonstration.

> **Budget tip:** Plan your prompt list in advance. Generate as many variations as you
> can during the block window — block cost is fixed whether you generate 1 or 90 videos.

---

## Prompting Best Practices

Based on our testing (full catalog in `docs/cosmos3-prompt-catalog.md`):

1. **Be specific about the robot:** "UR3 robot arm with a Robotiq gripper"
2. **Describe the complete action:** "reaches → grasps → lifts → places"
3. **Specify the target precisely:** "yellow sticky note approximately 6 inches away"
4. **State the camera angle:** "Top-down wrist camera view"
5. **Describe the scene context:** "Colorful wooden blocks scattered on table"
6. **Specify motion style:** "Smooth deliberate motion"
7. **Vary one element at a time** to build diverse datasets (block color, lighting, speed)

**What does NOT work:**
- Vague prompts → model hallucinates random content
- Editing instructions ("add scratches") → this is a generator, not an editor
- Environment-only descriptions without action → model invents its own action

---

## Pre-Generated Samples (No GPU Needed)

If you don't have P5 capacity, download our validated outputs from S3:

```bash
# Original reference clip
aws s3 cp s3://<DATASETS_BUCKET>/cosmos-samples/original_episode_000000.mp4 ./

# Generated: red block pick-and-place (specific prompt, seed 100) ← BEST RESULT
aws s3 cp s3://<DATASETS_BUCKET>/cosmos-samples/augmented_specific_prompt.mp4 ./

# Generated: blue block variant (seed 200)
aws s3 cp s3://<DATASETS_BUCKET>/cosmos-samples/augmented_blue_block.mp4 ./

# Generated: dim lighting variant (seed 300)
aws s3 cp s3://<DATASETS_BUCKET>/cosmos-samples/augmented_dim_lighting.mp4 ./
```

---

## How This Fits the Pipeline

Cosmos 3 generates **vision-only** synthetic demonstrations (no action labels). These are
useful for:

1. **Vision pre-training** — teach the VLA backbone what pick-and-place looks like under
   many conditions before fine-tuning on action-paired data (Lab 1)
2. **Dataset diversity** — expose the model to visual variations it'll encounter in deployment
3. **Policy evaluation** — generate rollouts to visually assess if a task was completed
4. **Scaling data** — go from 27 real demos to hundreds of synthetic ones for visual diversity

For **action-paired augmentation** (same video restyled with actions preserved), Cosmos
Transfer 2.5 (a different model) provides pixel-faithful restyling with edge/depth
control signals. See `docs/cosmos3-validated-runbook.md` for the comparison.

---

## Without Cosmos (Fallback)

Lab 4 works without this lab. Isaac Lab's built-in procedural domain randomization
provides position, lighting, and color variation for RL training without any external
model. Cosmos adds value when you need photorealistic visual diversity beyond what
procedural randomization provides.

---

## Production Path: EKS + Cosmos 3 Flywheel

For production-scale generation (thousands of videos, continuous flywheel loop), deploy
the vLLM-Omni server on **Amazon EKS** instead of bare EC2. The Kubernetes manifest
is at `kubernetes/generate-vllm-omni-super.yaml` (adapted from the validated
[awslabs/awsome-distributed-ai/cosmos3](https://github.com/awslabs/awsome-distributed-ai/tree/main/3.test_cases/pytorch/cosmos3)
reference, MIT-0).

---

## Full Reproduction Steps

The complete step-by-step runbook (every CLI command we ran to validate this lab) is in
[`docs/cosmos3-validated-runbook.md`](../docs/cosmos3-validated-runbook.md).

---

**Previous:** [← Lab 2: Isaac Sim Workstation](lab-2-isaac-workstation.md)
**Next:** [Lab 4: Cosmos Transfer →](lab-4-cosmos-transfer.md)
