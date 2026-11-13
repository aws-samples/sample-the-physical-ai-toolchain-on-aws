# Lab 4: Cosmos Transfer — Photorealistic Data Augmentation

> **Status:** Validated on p5.48xlarge (8x H100 80GB) using the NIM container.
> Also validated on a single **g6e.4xlarge (1x L40S 48GB)** using the source-built
> `physical-ai/cosmos-transfer` image instead of the NIM — no P5 and no Capacity
> Block needed. See [Single-GPU L40S path](#single-gpu-l40s-path-no-p5) below.
> Geometry preservation works. Color fidelity and framerate matching require tuning.
> See notes below for known issues and workarounds.

**Goal:** Use Cosmos Transfer 2.5 to restyle your existing training videos with photorealistic visual variations while preserving exact robot motion and geometry — making your action labels remain valid
**Time:** 1-2 hours
**Cost:** ~$37/hr via p5.48xlarge Capacity Block (H100 80GB required)

> **New to Transfer vs. Predict?** Lab 3 (Predict) generates *new* demonstrations from
> prompts. This lab (Transfer) restyles *existing* demonstrations — same motion, different
> visual appearance. Both use Cosmos, but for different purposes.

---

## What You're Building

You have 27 real demonstrations from Lab 1. The robot's motion and actions are perfect —
but the visual environment is always the same lab bench. When you deploy to a factory,
the policy fails because it's never seen scratched metal, dim lighting, or cluttered
backgrounds.

**Cosmos Transfer 2.5 solves this** by taking your existing wrist camera videos and
restyling them with different visual environments — while preserving the exact geometry,
motion, and pixel-level structure of every frame. Your action labels stay perfectly valid
because nothing moved — only the visual appearance changed.

**This is the canonical data augmentation approach** recommended by NVIDIA's Physical AI
Reference Architecture, the SO-101 Sim-to-Real tutorial, and the GR00T-Mimic Blueprint.

---

## How It Differs from Lab 3

| | Lab 3 (Predict/World Gen) | Lab 4 (Transfer) |
|---|---|---|
| Model | Cosmos 3 Super (64B) | Cosmos Transfer 2.5 (2B) |
| What it does | Generates new video from reference + prompt | Restyles existing video preserving structure |
| Actions preserved? | No — new trajectory, no action labels | **Yes — exact same motion, same labels** |
| Control signal | Text prompt only (loose) | Edge/depth/seg map (strict geometry constraint) |
| Input | Reference video + text prompt | Video + edge map + style prompt |
| Output | Novel demonstration (vision-only) | Restyled demonstration (action-paired) |
| Use case | Scale dataset with new trajectories | Scale dataset with visual diversity |
| GPU required | 8x H100 (640 GB) | 8x H100 80GB (p5.48xlarge) |
| Container | `vllm/vllm-omni:cosmos3` | `nvcr.io/nim/nvidia/cosmos-transfer2.5-2b` |
| Cost | ~$37/hr (p5 Capacity Block) | ~$37/hr (p5 Capacity Block) |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  Cosmos Transfer 2.5 — Pixel-Faithful Restyling                              │
│                                                                              │
│  ┌──────────────┐    ┌────────────────────┐    ┌───────────────┐            │
│  │ Lab 1 wrist  │    │  EC2 p5.48xlarge    │    │  Output       │            │
│  │ camera MP4   │───▶│  (Capacity Block)   │───▶│  Restyled MP4 │            │
│  │ + edge map   │    │                    │    │               │            │
│  │ + style      │    │  Cosmos Transfer   │    │  SAME motion  │            │
│  │   prompt     │    │  2.5 NIM           │    │  NEW visuals  │            │
│  └──────────────┘    │  (8x H100 80GB)    │    │  Actions stay │            │
│                      └────────────────────┘    │  valid!       │            │
│                                                └───────────────┘            │
└──────────────────────────────────────────────────────────────────────────────┘

Key: the edge map extracted from your input video tells the model WHERE everything is.
The prompt tells it HOW things should look. Geometry is locked — only appearance changes.
```

---

## Prerequisites

- **Lab 1 completed** — you have wrist camera MP4s in S3
- **NGC API key** in Secrets Manager (`physical-ai/ngc-api-key`) — Transfer NIM pulls from NGC
- **HuggingFace token** in Secrets Manager (`physical-ai/hf-token`) — for guardrail model
- **P5 GPU capacity** — p5.48xlarge (8x H100 80GB) via Capacity Block (see scanning script below)
- **HuggingFace licenses accepted:**
  - `nvidia/Cosmos-Guardrail1`
  - `nvidia/Cosmos-1.0-Guardrail`
  - `nvidia/Cosmos-Predict2.5-2B`
  - `nvidia/Cosmos-Transfer2.5-2B`

---

## Running This Lab

**One-command path (recommended):** The automation script handles everything — Capacity Block scan, purchase, instance launch, NIM pull, and server start:

```bash
bash scripts/cosmos-transfer-launch.sh          # full automated deploy
bash scripts/cosmos-transfer-launch.sh --dry-run  # preview without spending
```

**Step-by-step path:** Follow Steps 1-7 below if you want to understand each piece or customize the setup.

> **Bring your own data:** The example uses our UR3 pick-and-place episodes, but you can
> use any MP4 video (93-480 frames) from your own robot or simulation. The pipeline is
> the same regardless of robot or task.

> **Future improvement:** These steps will be integrated into a CDK stack for fully
> declarative infrastructure-as-code deployment.

---

## Step 1: Find and Purchase a Capacity Block

P5 instances are rarely available on-demand (at the time of this writing). Use a Capacity Block (reserved GPU allocation). Scan for availability across regions:

```bash
for REGION in us-east-1 us-east-2 us-west-2; do
  echo "=== $REGION ==="
  aws ec2 describe-capacity-block-offerings \
    --instance-type p5.48xlarge \
    --capacity-duration-hours 24 \
    --instance-count 1 \
    --region $REGION \
    --query 'CapacityBlockOfferings[*].{Hours:CapacityBlockDurationHours,Cost:UpfrontFee,AZ:AvailabilityZone,Start:StartDate}' \
    --output table
done
```

Purchase when you find one that fits your schedule:

```bash
aws ec2 purchase-capacity-block \
  --capacity-block-offering-id <OFFERING_ID> \
  --instance-platform Linux/UNIX \
  --region <REGION> \
  --tag-specifications 'ResourceType=capacity-reservation,Tags=[{Key=Name,Value=cosmos-transfer}]'
```

Wait for the block to go `active` (it activates at the scheduled StartDate):

```bash
aws ec2 describe-capacity-reservations \
  --capacity-reservation-ids <CR_ID> \
  --region <REGION> \
  --query 'CapacityReservations[0].State'
```

---

## Step 2: Launch the Instance

Once the block is active:

```bash
REGION="<your-region>"            # where the Capacity Block / instance is
SECRETS_REGION="<foundation-region>"  # where the physical-ai/* secrets were created (Lab 0)
CR_ID="<your-capacity-reservation-id>"
BLOCK_AZ="<AZ from block output>"

# Find a subnet in the block's AZ
SUBNET=$(aws ec2 describe-subnets --filters "Name=availability-zone,Values=$BLOCK_AZ" \
  --query 'Subnets[0].SubnetId' --output text --region $REGION)

# Get the Deep Learning AMI (NVIDIA drivers pre-installed)
AMI=$(aws ec2 describe-images --owners amazon \
  --filters "Name=name,Values=Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*" \
  --region $REGION --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId' --output text)

# Launch into the capacity block
aws ec2 run-instances \
  --image-id $AMI \
  --instance-type p5.48xlarge \
  --placement AvailabilityZone=$BLOCK_AZ \
  --security-group-ids <YOUR_SG_ID> \
  --iam-instance-profile Name=<YOUR_PROFILE> \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":500,"VolumeType":"gp3"}}]' \
  --instance-market-options '{"MarketType":"capacity-block"}' \
  --capacity-reservation-specification '{"CapacityReservationTarget":{"CapacityReservationId":"'$CR_ID'"}}' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=cosmos-transfer}]' \
  --region $REGION
```

> **Note:** The `--instance-market-options '{"MarketType":"capacity-block"}'` flag is
> required for Capacity Block instances. Without it you get "market type not valid."

---

## Step 3: Pull and Start the NIM

Wait for SSM to come online (~2 min after boot), then set up via SSM:

```bash
INSTANCE_ID=<from launch output>

# Verify GPUs
aws ssm send-command --instance-ids $INSTANCE_ID --document-name AWS-RunShellScript \
  --parameters '{"commands":["nvidia-smi -L | head -2"]}' --region $REGION

# Login to NGC
aws ssm send-command --instance-ids $INSTANCE_ID --document-name AWS-RunShellScript \
  --parameters '{"commands":["aws secretsmanager get-secret-value --secret-id physical-ai/ngc-api-key --region '"$SECRETS_REGION"' --query SecretString --output text | docker login nvcr.io --username \\$oauthtoken --password-stdin"]}' \
  --region $REGION

# Pull the Transfer NIM (~20 min, 56 GB)
aws ssm send-command --instance-ids $INSTANCE_ID --document-name AWS-RunShellScript \
  --parameters '{"commands":["docker pull nvcr.io/nim/nvidia/cosmos-transfer2.5-2b:latest"]}' \
  --timeout-seconds 1800 --region $REGION

# Start the NIM (auto-selects H100 profile)
aws ssm send-command --instance-ids $INSTANCE_ID --document-name AWS-RunShellScript \
  --parameters '{"commands":["NGC_KEY=$(aws secretsmanager get-secret-value --secret-id physical-ai/ngc-api-key --region '"$SECRETS_REGION"' --query SecretString --output text) && HF_TOKEN=$(aws secretsmanager get-secret-value --secret-id physical-ai/hf-token --region '"$SECRETS_REGION"' --query SecretString --output text) && docker run -d --name cosmos-transfer-nim --gpus all --ipc=host --shm-size=64g -p 8000:8000 -e NGC_API_KEY=$NGC_KEY -e HF_TOKEN=$HF_TOKEN nvcr.io/nim/nvidia/cosmos-transfer2.5-2b:latest"]}' \
  --region $REGION
```

Wait ~15 min for TRT engine download and model loading. Check readiness:

```bash
aws ssm send-command --instance-ids $INSTANCE_ID --document-name AWS-RunShellScript \
  --parameters '{"commands":["curl -s http://localhost:8000/v1/health/ready && echo READY || echo NOT_READY"]}' \
  --region $REGION
```

Server is ready when it returns HTTP 200.

---

## Step 4: Copy Your Training Video to the Instance

```bash
# Copy a wrist camera clip from your Lab 1 dataset (must be 93-480 frames)
aws ssm send-command --instance-ids $INSTANCE_ID --document-name AWS-RunShellScript \
  --parameters '{"commands":["aws s3 cp s3://<BUCKET>/groot-data/ur3/dataset/videos/chunk-000/observation.images.wrist/episode_000009.mp4 /tmp/input_video.mp4 --region '"$REGION"'"]}' \
  --region $REGION
```

> **Important:** Input must be **93-480 frames**. At 5fps, that's 18.6-96 seconds.
> If your clips are shorter than 93 frames, use a longer episode or pad with repeated frames.

---

## Step 5: Run Transfer Inference

Post your video to the NIM with an edge control signal and style prompt:

```bash
aws ssm send-command --instance-ids $INSTANCE_ID --document-name AWS-RunShellScript \
  --parameters '{"commands":["cat > /tmp/run_transfer.py << PYEOF\nimport base64, requests, json, subprocess\n\nwith open(\"/tmp/input_video.mp4\", \"rb\") as f:\n    video_b64 = base64.b64encode(f.read()).decode()\n\npayload = {\n    \"prompt\": \"Industrial factory with scratched metal table and fluorescent lighting\",\n    \"video\": video_b64,\n    \"edge\": {},\n    \"num_steps\": 35,\n    \"guidance\": 3\n}\n\nprint(f\"Posting {len(video_b64)//1024}KB...\")\nresp = requests.post(\"http://localhost:8000/v1/infer\", json=payload, timeout=900)\nprint(f\"Status: {resp.status_code}, Size: {len(resp.content)}\")\n\nif resp.status_code == 200:\n    data = resp.json()\n    video = base64.b64decode(data[\"b64_video\"])\n    with open(\"/tmp/transfer_raw.mp4\", \"wb\") as f:\n        f.write(video)\n    # Convert to H.264 at original fps for Mac/browser playback\n    subprocess.run([\"ffmpeg\", \"-y\", \"-i\", \"/tmp/transfer_raw.mp4\", \"-r\", \"5\", \"-c:v\", \"libx264\", \"-crf\", \"18\", \"/tmp/transfer_output.mp4\"], capture_output=True)\n    print(\"TRANSFER_SUCCESS\")\nelse:\n    print(resp.text[:500])\nPYEOF\npython3 /tmp/run_transfer.py"]}' \
  --timeout-seconds 900 --region $REGION
```

**Generation takes ~8-10 min** on H100 with 35 diffusion steps.

---

## Step 6: Download and Compare

Upload the result to S3, then download both original and restyled:

```bash
aws ssm send-command --instance-ids $INSTANCE_ID --document-name AWS-RunShellScript \
  --parameters '{"commands":["aws s3 cp /tmp/transfer_output.mp4 s3://<BUCKET>/cosmos-samples/transfer_result.mp4 --region '"$REGION"'"]}' \
  --region $REGION

# Download to your laptop
aws s3 cp s3://<BUCKET>/cosmos-samples/transfer_result.mp4 ./
```

Open both clips side by side — the output should show the same robot motion with a restyled environment.

> **Pre-generated samples:** If you don't have P5 capacity, see
> `training/data/cosmos-samples/` for before/after examples with prompts documented.

---

## Step 7: Terminate

```bash
aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region $REGION
```

The Capacity Block fee is already paid regardless.

---

## Single-GPU L40S Path (no P5)

The NIM auto-selects an H100 profile, so on L40S use the **source-built image**
(`physical-ai/cosmos-transfer`, built by `containers/cosmos/buildspec.yml`) and its
`examples/inference.py` entrypoint directly. Validated end-to-end on one
`g6e.4xlarge`: 93 frames at 640x480 took **~9.5 min** of diffusion at 35 steps and
peaked at **43.9 GB of the 46 GB** of L40S VRAM — it fits, but only with
`--offload-guardrail-models` to keep the guardrail models (Cosmos-Guardrail1 +
Cosmos-Reason1-7B) on the CPU. Guardrails stay enabled.

```bash
# Input must be 93-480 frames. Trim one chunk out of a longer episode:
ffmpeg -y -i episode_000009.mp4 -frames:v 93 -c:v libx264 -crf 18 in/ur3_input.mp4

# Parameter file. Note the control lives under a per-modality key ("edge", "depth",
# "seg" or "blur") and "name" is required — a flat "control" key is rejected.
cat > in/ur3_edge_spec.json <<'JSON'
{
  "name": "ur3_factory_edge",
  "prompt": "A UR3 robot arm with a Robotiq gripper performs a pick-and-place on a workbench inside a busy industrial factory. The table is scratched brushed metal under harsh overhead fluorescent lighting. Photorealistic, sharp focus. Top-down wrist camera view.",
  "video_path": "/workspace/in/ur3_input.mp4",
  "num_steps": 35,
  "guidance": 3,
  "seed": 100,
  "resolution": "720",
  "edge": {"control_weight": 1.0, "preset_edge_threshold": "medium"}
}
JSON

HF_TOKEN=$(aws secretsmanager get-secret-value --secret-id physical-ai/hf-token \
  --region $REGION --query SecretString --output text)
docker run --rm --gpus all --ipc=host --shm-size=32g \
  -e HF_TOKEN="$HF_TOKEN" -e HF_HOME=/workspace/hf-cache \
  -v /opt/hf-cache:/workspace/hf-cache \
  -v $PWD/in:/workspace/in -v $PWD/out:/workspace/out \
  <ACCOUNT>.dkr.ecr.$REGION.amazonaws.com/physical-ai/cosmos-transfer:latest \
  bash -c 'python /workspace/examples/inference.py \
    -i /workspace/in/ur3_edge_spec.json -o /workspace/out \
    --model edge --offload-guardrail-models'
```

Outputs land in `out/` as `<name>.mp4` plus `<name>_control_edge.mp4` (the Canny edge
map the model was conditioned on — useful for verifying geometry preservation).
`keep_input_resolution` defaults to `true`, so the result comes back at the input's
resolution and frame count, which is what keeps your action labels aligned.

---

## ✅ Lab 4 Checkpoint

- [ ] Capacity Block purchased and activated (p5.48xlarge)
- [ ] NIM pulled and server responding to `/v1/health/ready`
- [ ] Posted one clip (93+ frames) with edge control
- [ ] Output preserves robot motion (visual comparison confirms geometry)
- [ ] Converted output to H.264 at original fps for playback
- [ ] (Optional) Tried different prompts for varied environments
- [ ] (Optional) Merged restyled videos into augmented dataset
- [ ] Terminated instance

---

## Known Issues (for contributors)

- Colors shift toward the prompt's described environment (edge control preserves geometry but not color)
- `vis` control mode preserves colors better but produces artifacts on the FP8 "latency" profile
- The BF16 `throughput` NIM profile may produce better quality (not yet tested — restart container with `NIM_MODEL_PROFILE=throughput`)
- Lower `guidance` (1-2) may preserve more of the original appearance
- `depth` control mode may give better structure preservation than `edge`

---

## Cost Estimation

| Resource | Cost | Notes |
|----------|------|-------|
| p5.48xlarge (Capacity Block) | ~$37/hr (~$574 for 16hr block) | H100 80GB required |
| Generation time | ~8-10 min per 93-frame clip (35 steps) | Faster with fewer steps (lower quality) |
| NIM container pull | ~20 min first time | 56 GB image from NGC |

---

**Previous:** [← Lab 3: Cosmos World Generation](lab-3-cosmos-world-generation.md)
**Next:** [Lab 5: RL Policy Training with Isaac →](lab-5-rl-refinement-with-isaac.md)
