# Cosmos 3 on AWS

Generate synthetic robot demonstration videos with [NVIDIA Cosmos 3](https://www.nvidia.com/en-us/ai/cosmos/) (Cosmos3-Super, a World Foundation Model) — conditioned on your own robot data.

Two deployment paths are documented:

| Path | Guide | Best for |
|------|-------|----------|
| **EC2** | [EC2 Deployment Guide](ec2-deployment-guide.md) | Single-node experiments, workshops, one-off generation runs. **Validated end-to-end.** |
| **EKS** | [EKS Deployment Guide](eks-deployment-guide.md) | Production-scale generation, multi-replica flywheel. **Validated end-to-end**, including fully automatic GPU node scaling via Cluster Autoscaler (single-replica; multi-replica + shared FSx storage documented as future work). |

Both paths run the same server (`vllm/vllm-omni:cosmos3`) and default to `nvidia/Cosmos3-Super` on the same hardware (p5.48xlarge / p5en.48xlarge, 8x H100+) — validated end-to-end on both. A `COSMOS_MODEL` toggle in the scripts/manifest lets you switch to `nvidia/Cosmos3-Nano` instead, which needs only 1 GPU; see [Choosing Super vs Nano](#choosing-super-vs-nano) below. The difference between EC2 and EKS is orchestration: EC2 is a single instance you manage directly; EKS lets you run multiple generation jobs in parallel across a GPU node pool with shared storage (FSx).

---

## Can Cosmos 3 generate world output from UR3 data? — Yes, validated

Cosmos3-Super's **V2V (video-to-video)** mode takes a reference video as the starting condition and generates a new plausible continuation — this works with any input video, including your own UR3 teleoperation footage.

**Validated end-to-end on both EC2 and EKS:**
- **Input:** `episode_000000.mp4` — a real UR3 wrist-camera pick-and-place episode (862 KB, from the Lab 1 dataset)
- **Output:** A new 189-frame, 1280×720, 24fps synthetic video (~6.0 MB on EC2, ~6.1 MB on EKS) — the UR3 arm performing a novel pick-and-place trajectory, photorealistic quality
- **Generation time:** ~2-8 min on 8x H100 (first request includes JIT warmup)
- **Mechanism:** The first 1-2 frames of your input video anchor the scene (table, blocks, lighting); the text prompt + seed control what action gets generated

**Artifacts in S3 (for your own validation):**
```bash
DATASETS_BUCKET=$(terraform -chdir=../foundation/infra output -raw datasets_bucket_name)
aws s3 cp s3://${DATASETS_BUCKET}/cosmos-samples/original_episode_000000.mp4 ./
aws s3 cp s3://${DATASETS_BUCKET}/cosmos-samples/augmented_episode_000000.mp4 ./
```
Play both side by side — `original_episode_000000.mp4` (861.7 KiB, the real input) vs `augmented_episode_000000.mp4` (5.9 MiB, the Cosmos 3 output) — to see the generated trajectory against the source scene it was conditioned on.

This means: give it any wrist-camera clip from your robot, describe the task in the prompt, and it generates a new synthetic demonstration starting from your actual scene — not a generic stock scene.

**What this is / isn't good for:**
- ✅ Vision pre-training — teach a VLA backbone what your task looks like across many visual variations
- ✅ Dataset diversity — same task, different lighting/block-color/approach-angle/speed
- ✅ Policy evaluation — visually sanity-check whether a generated rollout looks physically plausible
- ❌ **No action labels** — output is video only (no joint positions/velocities). For action-paired augmentation (restyle existing video while preserving the original actions), use Cosmos Transfer 2.5 instead — different model, different workflow.

Full step-by-step generation, validation, and troubleshooting: **[→ EC2 Deployment Guide](ec2-deployment-guide.md)**.

---

## Choosing EC2 vs EKS

- **Start with EC2** if you're experimenting, running a workshop, or need to generate a batch of videos during a single Capacity Block window. Simpler — one instance, no cluster.
- **Move to EKS** when you need multiple concurrent generation jobs (a continuous "flywheel" producing synthetic data), automatic GPU node scaling (submit a Job, walk away — Cluster Autoscaler brings up a node and tears it down when idle, no manual `aws eks update-nodegroup-config` calls), or integration with a broader Kubernetes-orchestrated pipeline (e.g. [OSMO](../osmo-on-aws/)). Note EKS adds ~$0.14/hr of control-plane + system-node overhead beyond the shared GPU cost — worth it once you need concurrency or hands-off scaling, pure overhead for a single one-off generation.
- Multi-replica scaling and shared FSx storage across replicas are documented as future work in the EKS guide — the validated EKS path today runs one Cosmos3-Super replica, same as EC2, just orchestrated by Kubernetes instead of a raw instance.

---

## Choosing Super vs Nano

Cosmos 3 ships in three sizes — Super (64B), Nano (16B), and Edge (4B, not covered here; Edge doesn't support V2V). This repo's scripts and `cosmos3-job.yaml` default to **Super**, but include a `COSMOS_MODEL` toggle (`"super"` or `"nano"`) to switch. **Both validated end-to-end on EC2** (on-demand `g6e.4xlarge`, no Capacity Block needed for Nano).

| | **Cosmos3-Super** (default, validated) | **Cosmos3-Nano** (validated) |
|---|---|---|
| Parameters | 64B | 16B |
| GPUs needed | 8 (full p5.48xlarge) | 1 |
| Serve flags | `--cfg-parallel-size 2 --ulysses-degree 4 --use-hsdp --hsdp-shard-size 8` | `--vae-use-tiling` (required — see below) |
| Instance/node | p5.48xlarge (Capacity Block required — P5 is scarce) | Any single-GPU instance, e.g. `g6e.xlarge`/`g6e.4xlarge` (on-demand, no Capacity Block needed) |
| Model weight download | ~126 GB | ~33 GB |
| Generation time (189 frames/720p) | ~2-8 min (8-GPU parallelism) | ~10-12 min (single GPU, no parallelism) |
| Quality | Highest-fidelity generation, best for production synthetic data / teacher-model use | Good quality/speed balance, faster iteration |
| Cost | ~$37/hr (p5.48xlarge Capacity Block) | Roughly an order of magnitude less — single mid-tier GPU on-demand |

**Two things the Nano validation run surfaced, both already fixed in this repo's scripts:**
1. **`--vae-use-tiling` is required on single-GPU nodes.** Without it, VAE decode runs out of memory on a `g6e.4xlarge`'s ~44GB usable VRAM at the default 189-frame/720p size (confirmed: `torch.OutOfMemoryError` during decode). The flag cuts peak decode VRAM ~68% for ~13% extra latency — the tradeoff already documented in vLLM-Omni's own Cosmos3-Nano recipe.
2. **Use the async `/v1/videos` job API, not `/v1/videos/sync`.** `/v1/videos/sync` has a hardcoded ~600s server-side abort regardless of client `--max-time` — confirmed this killing full-length Nano generations (~10-12 min total) even after raising the client timeout. `generate-v2v.sh` and `cosmos3-job.yaml` both use `POST /v1/videos` → poll `GET /v1/videos/{id}` → `GET /v1/videos/{id}/content` for both models now, since it's the robust path either way (vLLM-Omni's docs recommend `/v1/videos/sync` only for quick benchmarks).

**Where to change it:**
- EC2 path: set `COSMOS_MODEL` at the top of [`setup-cosmos3-server.sh`](setup-cosmos3-server.sh) and [`generate-v2v.sh`](generate-v2v.sh) (must match in both). For an on-demand Nano instance via Terraform, apply `infra/ec2.tf` with `enable_ec2_server=true` and `server_instance_type=g6e.4xlarge`, leaving `capacity_reservation_id` unset.
- EKS path: set the `COSMOS_MODEL` env var in [`cosmos3-job.yaml`](cosmos3-job.yaml), and update `resources.requests`/`resources.limits` (`nvidia.com/gpu`) to `"1"` — Nano doesn't need the p5.48xlarge GPU node group or a Capacity Block at all, so you'd size a separate, much cheaper node group for it (see [eks-deployment-guide.md](eks-deployment-guide.md)). **Not yet runtime-validated on EKS** — the fix is applied to the manifest but hasn't been tested against a live single-GPU node group.

**Everything else in this repo (V2V mechanism, prompt structure, generation parameters, S3 upload flow) is identical between the two** — only the model ID, serve flags, and GPU count change.

---

## Files in This Directory

| File | Purpose |
|------|---------|
| [`ec2-deployment-guide.md`](ec2-deployment-guide.md) | Full EC2 deployment walkthrough — Capacity Block, launch, server setup, generation, S3 validation |
| [`eks-deployment-guide.md`](eks-deployment-guide.md) | Full EKS deployment walkthrough — cluster, GPU node group, IRSA, Job, generation, S3 validation |
| [`launch-cosmos3.sh`](launch-cosmos3.sh) | Launch p5.48xlarge into an active Capacity Block (EC2 path) |
| [`setup-cosmos3-server.sh`](setup-cosmos3-server.sh) | Pull the container and start the Cosmos3 server — Super or Nano via `COSMOS_MODEL` (EC2 path) |
| [`generate-v2v.sh`](generate-v2v.sh) | Generate a V2V synthetic demonstration from a reference video (EC2 path) |
| [`charts/cosmos3/`](charts/cosmos3/) | Helm chart for the reusable Cosmos server: one GPU Deployment, a private ClusterIP Service, fail-closed archive validation, and no ingress or credential resources |
| [`cosmos3-service.yaml`](cosmos3-service.yaml) | Legacy reusable-service manifest retained temporarily for migration and rollback |
| [`prepare-g1-service.py`](prepare-g1-service.py) | Legacy manifest renderer retained temporarily; new deployments should use the Helm chart |
| [`submit-g1-generation.py`](submit-g1-generation.py) | Submits one prompt/source request through a localhost port-forward, accepts local or S3-hosted paired edge-control videos, and writes uniquely named output to S3 |
| [`cosmos3-job.yaml`](cosmos3-job.yaml) | Legacy self-contained one-shot Kubernetes Job that starts a server, generates once, uploads, and exits |
| `infra/ec2.tf` | Terraform for the EC2 server instance |
| `infra/eks.tf` | Terraform for the dedicated EKS cluster + GPU node group |
| `infra/main.tf` | Terraform for the CodeBuild-based container build path (Cosmos 3 + Cosmos Transfer 2.5) |

---

## Agentic Orchestration with Strands Agents

[**strands-robots**](https://github.com/strands-labs/robots) ships a **Cosmos 3
trainer** (`strands_robots.training.cosmos3`) that drives the same
`cosmos-framework` SFT pipeline as a Python library. This lets a
[Strands Agent](https://strandsagents.com) orchestrate the data-generation stage —
scripting scene prompts, launching Cosmos runs, and curating the generated episodes
into a LeRobot v2 dataset for downstream GR00T fine-tuning. See
[strands-agents-on-aws](../strands-agents-on-aws/) for the orchestration layer.

---

## Next Steps

- **Action-paired augmentation** — for restyling existing video while preserving ground-truth actions, see Cosmos Transfer 2.5 (`containers/cosmos/`)
- **Feed generated videos into training** — use Cosmos 3 output for vision pre-training or dataset diversity ahead of [GR00T fine-tuning](../isaac-gr00t-on-aws/)
- **Scale EKS to multi-replica** — see [Future Work](eks-deployment-guide.md#future-work-multi-replica--shared-storage) in the EKS guide (FSx shared storage, multiple concurrent Jobs)
- **Full CLI runbook** — every command run during validation is in [`docs/cosmos3-validated-runbook.md`](../docs/cosmos3-validated-runbook.md)
