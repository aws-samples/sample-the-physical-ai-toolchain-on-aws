# Cosmos 3 on EC2 — Deployment Guide

Deploy the Cosmos3-Super V2V generation server on a single EC2 instance (p5.48xlarge, 8x H100). This is the simplest path — one instance, SSM access, no cluster to manage.

> This guide defaults to Cosmos3-Super. To use the smaller Cosmos3-Nano instead (1 GPU, faster/cheaper, some quality tradeoff), see [Choosing Super vs Nano](README.md#choosing-super-vs-nano) in the README — set `COSMOS_MODEL="nano"` in both `setup-cosmos3-server.sh` and `generate-v2v.sh`, and use a single-GPU on-demand instance type (e.g. `g6e.4xlarge`) instead of p5.48xlarge — no Capacity Block needed. **Validated end-to-end**: apply `infra/ec2.tf` with `enable_ec2_server=true`, `server_instance_type=g6e.4xlarge`, and `capacity_reservation_id` left unset for an on-demand launch.

See [`README.md`](README.md) for the overview, the EC2-vs-EKS comparison, and validated proof that this pipeline generates world output conditioned on real UR3 robot data.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│  Your workstation / CI                                                    │
│  aws ssm send-command  ──────┐                                            │
└───────────────────────────────┼────────────────────────────────────────────┘
                                ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  EC2 p5.48xlarge (8x H100 80GB) — EC2 Capacity Block                     │
│                                                                            │
│   docker run vllm/vllm-omni:cosmos3                                      │
│     vllm serve nvidia/Cosmos3-Super --omni \                             │
│       --cfg-parallel-size 2 --ulysses-degree 4 \                         │
│       --use-hsdp --hsdp-shard-size 8                                     │
│                                                                            │
│   POST /v1/videos (async job) → poll → GET /v1/videos/{id}/content       │
│     -F input_reference=@your_ur3_clip.mp4   ← YOUR robot data            │
│     -F prompt="..."                          ← task description         │
│     -F seed=100                              ← trajectory variation      │
│         │                                                                 │
│         ▼                                                                 │
│   output.mp4 (189 frames, 1280x720, 24fps)                               │
└────────────────────────────┬───────────────────────────────────────────────┘
                              ▼
                    S3 (cosmos-samples/)
```

## Why EC2 (vs EKS)

- **Simpler.** One instance, SSM access, no cluster to manage.
- **Right-sized for experimentation.** A single p5.48xlarge gives you the full 8-GPU parallelism Cosmos3-Super needs (`cfg-parallel-size 2 × ulysses-degree 4`). You don't need multiple nodes to run one generation.
- **Matches how Capacity Blocks work.** P5 capacity is scarce and reserved in discrete blocks per instance — EC2 lets you launch directly into a block with no orchestration layer in between.
- **Move to EKS when:** you need multiple concurrent generation jobs (a "flywheel" continuously producing synthetic data), shared FSx storage across replicas, or integration with a broader Kubernetes-orchestrated pipeline. See the [EKS Deployment Guide](eks-deployment-guide.md).

---

## Prerequisites

| Requirement | Why | How to get it |
|-------------|-----|----------------|
| Foundation deployed | Cosmos IAM role, instance profile, ECR repos | `cd foundation/infra && terraform apply` |
| P5 Capacity Block (Super only) | P5 is rarely available on-demand. **Not needed for Nano** — any on-demand single-GPU type (e.g. `g6e.4xlarge`) works, see [Choosing Super vs Nano](README.md#choosing-super-vs-nano) | See [Step 1](#step-1-reserve-a-capacity-block) |
| HuggingFace token | Downloads Cosmos3-Super + guardrail weights | https://huggingface.co/settings/tokens |
| Accepted gated model licenses | vLLM-Omni downloads both at startup | Accept both below |
| A reference video | Your starting scene for V2V generation | Any wrist/scene camera MP4 (UR3 or other) |

**Accept BOTH gated licenses** on the HF account that owns your token (missing either → 403 → server crashes on startup):
- [`nvidia/Cosmos-Guardrail1`](https://huggingface.co/nvidia/Cosmos-Guardrail1) → "Expand to review and access"
- [`nvidia/Cosmos-1.0-Guardrail`](https://huggingface.co/nvidia/Cosmos-1.0-Guardrail) → "Expand to review and access"

Store your HF token in Secrets Manager:
```bash
aws secretsmanager create-secret --name physical-ai/hf-token \
  --secret-string "hf_YOUR_TOKEN_HERE" --region <REGION>
```

---

## Step 1: Reserve a Capacity Block

P5 instances are almost never available on-demand. Check offerings across regions/durations:

```bash
for REGION in us-east-1 us-east-2 us-west-2; do
  echo "=== $REGION ==="
  aws ec2 describe-capacity-block-offerings \
    --instance-type p5.48xlarge \
    --capacity-duration-hours 24 \
    --instance-count 1 \
    --region $REGION \
    --query 'CapacityBlockOfferings[*].{AZ:AvailabilityZone,Start:StartDate,End:EndDate,Hours:CapacityBlockDurationHours,Price:UpfrontFee}' \
    --output table
done
```

> **Note:** p5.48xlarge Capacity Blocks only support 24-hour minimum duration — shorter blocks return `InvalidParameterValue`.

Purchase the block once you find a suitable offering:

```bash
aws ec2 purchase-capacity-block \
  --capacity-block-offering-id <OFFERING_ID> \
  --instance-platform Linux/UNIX \
  --region <REGION> \
  --tag-specifications 'ResourceType=capacity-reservation,Tags=[{Key=Name,Value=cosmos3-p5}]'
```

Poll until `State` is `active`:

```bash
aws ec2 describe-capacity-reservations \
  --capacity-reservation-ids <CR_ID> --region <REGION> \
  --query 'CapacityReservations[0].{State:State,Start:StartDate,End:EndDate}'
```

---

## Step 2: Launch the Instance (Terraform)

The instance is defined in [`infra/ec2.tf`](infra/ec2.tf) and gated behind `enable_ec2_server` (default `false`) so `terraform apply` is always safe to run before you have an active Capacity Block. It reads the VPC, subnet, and Cosmos instance profile from Foundation via SSM — no hardcoded IDs.

Once your Capacity Block is `active` (Step 1), apply with the block's ID and AZ:

```bash
cd cosmos-on-aws/infra
terraform init
terraform apply \
  -var="aws_region=<REGION>" \
  -var="enable_ec2_server=true" \
  -var="capacity_reservation_id=<CR_ID>" \
  -var="availability_zone=<AZ>"
```

This creates:
1. A security group (outbound-only — SSM, no inbound needed)
2. The `p5.48xlarge` instance, launched into your Capacity Block via `capacity_reservation_specification` + `instance_market_options { market_type = "capacity-block" }` (both required for Capacity Block launches)
3. A 500 GB gp3 encrypted root volume (override with `-var="server_volume_size_gb=..."` if needed)

Grab the instance ID from the output:
```bash
terraform output server_instance_id
```

Wait for SSM to register (~1-3 min after the instance reaches `running`):
```bash
aws ssm describe-instance-information \
  --filters "Key=InstanceIds,Values=$(terraform output -raw server_instance_id)" \
  --region <REGION> --query 'InstanceInformationList[0].PingStatus'
```

**Validated result:** instance up with 8x H100 80GB confirmed via `nvidia-smi -L`.

> **Why Terraform doesn't manage the Capacity Block purchase itself:** a Capacity Block reservation transitions from `payment-pending` → `scheduled` → `active` asynchronously (minutes to days depending on availability), which doesn't fit a single `terraform apply` lifecycle. Purchase the block with the AWS CLI (Step 1), then point Terraform at its ID once active. `terraform destroy` removes the EC2 instance but the Capacity Block itself expires on its own schedule regardless — see [Step 5](#step-5-tear-down).

<details>
<summary>Legacy path: raw AWS CLI script (if you're not using Terraform)</summary>

[`launch-cosmos3.sh`](launch-cosmos3.sh) does the same thing via `aws ec2 run-instances` directly, with the AZ/subnet/CR ID hardcoded as script variables instead of Terraform inputs. Kept for reference — prefer the Terraform path above for anything beyond a quick one-off test, since it keeps the instance definition versioned alongside the rest of this component's infra.

**Edit the placeholders at the top of the script first** — `REGION`, `AVAILABILITY_ZONE`, `CAPACITY_RESERVATION_ID`, `SUBNET_ID`, `SECURITY_GROUP_ID`, `INSTANCE_PROFILE_NAME`, and `AMI_ID` are all account/environment-specific. `SUBNET_ID`, `SECURITY_GROUP_ID`, and `INSTANCE_PROFILE_NAME` come from the Foundation stack's Terraform outputs (`terraform -chdir=../foundation/infra output`); `CAPACITY_RESERVATION_ID` from Step 1; `AMI_ID` from your region's latest GPU-optimized AMI.

```bash
bash cosmos-on-aws/launch-cosmos3.sh
```

</details>

---

## Step 3: Start the Cosmos 3 Server

**Edit `REGION` at the top of [`setup-cosmos3-server.sh`](setup-cosmos3-server.sh)** to your own region before running — it's a placeholder, not a real value.

Run it on the instance via SSM (base64-encode it so multi-line content survives the SSM parameter):

```bash
INSTANCE_ID=<from Step 2>
REGION=<your region>
SCRIPT_B64=$(base64 -w0 cosmos-on-aws/setup-cosmos3-server.sh)

aws ssm send-command --instance-ids $INSTANCE_ID \
  --document-name AWS-RunShellScript \
  --parameters "{\"commands\":[\"echo $SCRIPT_B64 | base64 -d > /tmp/setup-cosmos3-server.sh\", \"chmod +x /tmp/setup-cosmos3-server.sh\", \"bash /tmp/setup-cosmos3-server.sh\"]}" \
  --timeout-seconds 900 --region $REGION
```

This script:
1. Verifies 8 GPUs
2. Pulls `vllm/vllm-omni:cosmos3` (~30 GB, 5-10 min)
3. Retrieves the HF token from Secrets Manager
4. Starts the server: `vllm serve nvidia/Cosmos3-Super --omni --cfg-parallel-size 2 --ulysses-degree 4 --use-hsdp --hsdp-shard-size 8`

**Model weight download is ~126 GB from HuggingFace** — takes 15-20 min on first start (cached on `/opt/hf-cache` for subsequent restarts on the same instance).

Check readiness:
```bash
aws ssm send-command --instance-ids $INSTANCE_ID \
  --document-name AWS-RunShellScript \
  --parameters '{"commands":["curl -s http://localhost:8000/v1/models"]}' \
  --region <REGION>
```
Ready when it returns `{"data":[{"id":"nvidia/Cosmos3-Super",...}]}`.

**Validated timing:** ~24 min total from container pull to `/v1/models` returning ready (includes an internal warmup generation run).

---

## Step 4: Generate a World from Your UR3 Data

**Edit `REGION` and `ACCOUNT_ID` at the top of [`generate-v2v.sh`](generate-v2v.sh)** to your own values before running (they're placeholders, not real values) — or just set `DATASETS_BUCKET` directly if you already have it from `terraform -chdir=../foundation/infra output -raw datasets_bucket_name`.

Run the script — it pulls a reference clip from your S3 dataset, submits it to the server, and uploads the result:

```bash
SCRIPT_B64=$(base64 -w0 cosmos-on-aws/generate-v2v.sh)

aws ssm send-command --instance-ids $INSTANCE_ID \
  --document-name AWS-RunShellScript \
  --parameters "{\"commands\":[\"echo $SCRIPT_B64 | base64 -d > /tmp/generate-v2v.sh\", \"chmod +x /tmp/generate-v2v.sh\", \"bash /tmp/generate-v2v.sh\"]}" \
  --timeout-seconds 900 --region $REGION
```

**What it does:**
1. Downloads a UR3 wrist-camera episode from `s3://<DATASETS_BUCKET>/groot-data/ur3/dataset/videos/.../episode_000000.mp4`
2. Copies it into the running container
3. Submits an async job to `/v1/videos` with the reference video + a task-specific prompt + generation parameters, polls `/v1/videos/{id}` until `completed`, then downloads from `/v1/videos/{id}/content`. (Not `/v1/videos/sync` — it has a hardcoded ~600s server-side abort that full-length generations, especially on Nano's single GPU, can exceed regardless of client timeout.)
4. Downloads the generated output and uploads both original + generated videos to `s3://<DATASETS_BUCKET>/cosmos-samples/`

**Required generation parameters** (from the vLLM-Omni Cosmos3-Super recipe):

| Parameter | Value | Why |
|-----------|-------|-----|
| `size` | `1280x720` | Output resolution |
| `num_frames` | `189` | ~8 sec at 24fps |
| `fps` | `24` | Output framerate |
| `num_inference_steps` | `35` | Diffusion steps |
| `guidance_scale` | `6.0` | Prompt adherence |
| `flow_shift` | `10.0` | Generation parameter |
| `extra_params` | `{"condition_frame_indexes_vision":[0,1],"condition_video_keep":"first"}` | V2V conditioning — anchors generation to your input's first frames |

**Validated result:** HTTP 200 (async job `completed`), 6.0 MB output video, generated in ~2 minutes with Super (warm model). Nano takes longer (~10-12 min, single GPU, no parallelism) — see [Choosing Super vs Nano](README.md#choosing-super-vs-nano) for the full comparison, also validated end-to-end.

**To use your own robot data:** swap the S3 source path in `generate-v2v.sh` for any wrist/scene-camera MP4 from your robot, and edit the prompt to match your task, target object, and camera framing. The pipeline is embodiment-agnostic — the reference video just needs to show the starting scene you want Cosmos 3 to continue from.

---

## Step 4b: Validate the Output in S3

Don't just trust the `HTTP:200` from the generation call — confirm the file actually landed in S3 and is a real, playable video before moving on.

**1. Confirm the object exists and check its size:**

```bash
aws s3 ls s3://<DATASETS_BUCKET>/cosmos-samples/ --region <REGION> --human-readable
```

You should see both the original reference clip and the generated output, e.g.:
```
2026-08-05 14:49:19  861.7 KiB original_episode_000000.mp4
2026-08-05 14:49:19    5.9 MiB augmented_episode_000000.mp4
```

A generated video should be **several MB** (189 frames at 1280x720). If it's only a few KB, the generation likely failed silently — check `docker logs cosmos3` on the instance for errors before continuing.

**2. Download both files locally to inspect:**

```bash
mkdir -p ./cosmos-output
aws s3 cp s3://<DATASETS_BUCKET>/cosmos-samples/original_episode_000000.mp4 ./cosmos-output/ --region <REGION>
aws s3 cp s3://<DATASETS_BUCKET>/cosmos-samples/augmented_episode_000000.mp4 ./cosmos-output/ --region <REGION>
ls -lh ./cosmos-output/
```

**3. Verify it's a valid video (not a truncated/corrupt file):**

```bash
# ffprobe reports duration, resolution, and codec — a corrupt file will error out here
ffprobe -v error -show_entries format=duration,size -show_entries stream=width,height,codec_name \
  ./cosmos-output/augmented_episode_000000.mp4
```

Expect `width=1280`, `height=720`, `duration≈7.9` (189 frames at 24fps), and a valid `codec_name` (e.g. `h264`). If `ffprobe` errors out or reports `0x0`, the download or generation was corrupted — re-run Step 4.

**4. Visually compare original vs. generated:**

Play both files side by side (VS Code's built-in preview, `vlc`, `mpv`, or any video player). You're checking:
- The generated video **starts from the same scene** as the original (same table, same blocks — Cosmos 3 conditions on your input's first frames)
- The action described in your prompt is **visibly happening** (the right block, moving toward the right target)
- No obvious artifacts: flickering, extra/missing objects, physically implausible motion

If the generated video doesn't match your prompt (wrong block color, no motion, hallucinated objects), revisit your prompt — see [Prompting for Your Own Robot/Task](#prompting-for-your-own-robottask) below. This is expected with vague prompts, not a pipeline bug.

---

## Prompting for Your Own Robot/Task

Cosmos 3 uses your input video for the **starting scene** and your prompt for the **action**. Get specific:

**Works well:**
```
A UR3 robot arm with a Robotiq gripper reaches down to a dark matte table, grasps a
small red wooden block, lifts it slowly, and places it onto a yellow sticky note
approximately 6 inches away. Top-down wrist camera view. Colorful wooden blocks are
scattered on the table. Smooth deliberate motion.
```
— specific robot + gripper, step-by-step action, concrete target, camera angle, motion style.

**Doesn't work well:**
```
A robot arm performing pick and place in a photorealistic industrial warehouse
```
— too vague, no concrete action sequence → model hallucinates unrelated content (moving objects, phantom hands).

Full prompt catalog with tested variations: [`docs/cosmos3-prompt-catalog.md`](../docs/cosmos3-prompt-catalog.md).

**Vary one element per generation to build a diverse dataset:**
- Same prompt, different `seed` → different trajectory/timing of the same action
- Change block color / lighting / approach angle / speed → visual diversity for the same task
- Batch several variations while your Capacity Block is active — block cost is fixed regardless of how many videos you generate in the window

---

## Step 5: Tear Down

```bash
cd cosmos-on-aws/infra
terraform destroy \
  -var="aws_region=<REGION>" \
  -var="enable_ec2_server=true" \
  -var="capacity_reservation_id=<CR_ID>" \
  -var="availability_zone=<AZ>"
```

(Or `aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region <REGION>` if you launched via the legacy script.)

The Capacity Block itself expires automatically at its end time — no separate cleanup needed. Instance charges stop on termination; the block's upfront cost is fixed regardless of usage.

---

## Cost

| Resource | Cost | Notes |
|----------|------|-------|
| Capacity Block (p5.48xlarge, 24 hr) | ~$997 (varies by offering) | Paid upfront for the full block window |
| Per generation | ~$0 marginal | Block cost is fixed — generate as many videos as fit in the window |
| S3 storage | ~$0.001/video | ~6-7 MB per generated clip |

**Example:** a 24-hr block generating a video every ~5 min = ~280 possible generations in the window — the more you batch, the lower your effective cost per synthetic demonstration.

---

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `403 Cannot access gated repo` | HF token lacks access to a guardrail model | Accept both licenses (see Prerequisites) |
| `InsufficientInstanceCapacity` on-demand launch | No P5 on-demand capacity | Use a Capacity Block instead |
| `market type (purchasing) option is not valid` | Missing `--instance-market-options` when launching into a Capacity Block | `launch-cosmos3.sh` already sets this — verify you're using it |
| Batch job/instance stuck waiting for capacity | EC2 G/VT-family vCPU quota exhausted by other running instances | Check `aws ec2 describe-instances` for idle GPU instances consuming quota; request a quota increase if needed |
| `condition_frame_indexes_vision outside latent video` | Wrong resolution/frame count | Use exactly `size=1280x720`, `num_frames=189`, `fps=24` |
| Download stalls on a weight shard | HuggingFace rate-limiting | `docker restart cosmos3` — cached shards persist in `/opt/hf-cache` |
| SSM not registering | Instance profile missing SSM policy, or no outbound internet | Verify `AmazonSSMManagedInstanceCore` attached; subnet has NAT/IGW |
| Generation request returns `HTTP:000` / curl times out, or server logs show `504 Gateway Timeout` after ~10 min | `/v1/videos/sync` has a hardcoded ~600s server-side abort, independent of your client's `--max-time` | Use the async job API instead (`POST /v1/videos` → poll `GET /v1/videos/{id}` → `GET /v1/videos/{id}/content`) — already what `generate-v2v.sh` and `cosmos3-job.yaml` do. Confirmed hitting this on Nano, whose full 189-frame/720p generation (~10-12 min on 1 GPU) exceeds the sync endpoint's limit |
| Container exits `255` within ~10 min of `docker run`, only ~12 log lines, no traceback | Transient failure during the first HuggingFace fetch. Not OOM and not disk — observed on a healthy `g6e.4xlarge` with 124 GB RAM and 375 GB free | `docker start cosmos3`. Weights already in `/opt/hf-cache` are reused, so the restart reaches `/v1/models` in ~10 min. Confirmed twice on first boot |
| `InsufficientInstanceCapacity` for `g6e.*` across every AZ | g6e (L40S) capacity in a region can be exhausted for all sizes at once, on spot *and* on-demand | Retry — it does free up (observed a ~30 min wait in us-west-2). Note `infra/ec2.tf` picks `subnets.ids[0]`, so pin `-var="availability_zone=..."` to steer to an AZ that has capacity instead of letting the provider retry one AZ for 30+ min |
| `torch.OutOfMemoryError` during VAE decode (Nano, single-GPU instance) | Default VAE decode path needs more VRAM than a `g6e.4xlarge`'s ~44GB usable | Add `--vae-use-tiling` to the `vllm serve` command — already set for `COSMOS_MODEL="nano"` in `setup-cosmos3-server.sh` and `cosmos3-job.yaml` |

---

## Validation Status

| Step | Status | Details |
|------|--------|---------|
| Capacity Block purchase + activation | ✅ Validated | us-east-2a, 24-hr block |
| Instance launch into block | ✅ Validated | p5.48xlarge, 8x H100 80GB confirmed |
| Container pull (`vllm/vllm-omni:cosmos3`) | ✅ Validated | ~30 GB, completed in setup script |
| Model weight download + server ready | ✅ Validated | ~126 GB, ~24 min total, `/v1/models` confirmed ready |
| V2V generation from real UR3 data | ✅ Validated | Input: 862 KB episode clip → Output: 6.0 MB, 189 frames, HTTP 200 |
| S3 upload of original + generated | ✅ Validated | `s3://physical-ai-dev-datasets-804152302157/cosmos-samples/{original,augmented}_episode_000000.mp4` |

---

## Files Used in This Guide

| File | Purpose |
|------|---------|
| [`infra/ec2.tf`](infra/ec2.tf) | Terraform: security group + p5.48xlarge launched into an active Capacity Block |
| [`launch-cosmos3.sh`](launch-cosmos3.sh) | Legacy: launch via raw AWS CLI instead of Terraform |
| [`setup-cosmos3-server.sh`](setup-cosmos3-server.sh) | Pull the container and start the Cosmos3-Super server |
| [`generate-v2v.sh`](generate-v2v.sh) | Generate a V2V synthetic demonstration from a reference video |

---

## Next Steps

- **EKS deployment** — for production-scale, multi-replica generation with shared FSx storage, see the [EKS Deployment Guide](eks-deployment-guide.md)
- **Action-paired augmentation** — for restyling existing video while preserving ground-truth actions, see Cosmos Transfer 2.5 (`containers/cosmos/`)
- **Feed generated videos into training** — use Cosmos 3 output for vision pre-training or dataset diversity ahead of [GR00T fine-tuning](../isaac-gr00t-on-aws/)
