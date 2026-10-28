# Isaac Lab on AWS

Train reinforcement learning policies with [NVIDIA Isaac Lab](https://developer.nvidia.com/isaac/lab) on AWS GPU infrastructure. The validated G1 path uses the `3.0.0-beta2` container because its Isaac Sim renderer is compatible with the NVIDIA R595 driver on the current AWS Marketplace workstation. Isaac Lab runs thousands of parallel simulation environments on a single GPU, training robust robot policies through trial-and-error guided by reward signals.

---

## What You'll Build

```
┌──────────────────────────────────────────────────────────────┐
│  Foundation (shared base)                                     │
│  S3 buckets · ECR repos · IAM roles                           │
└───────────────────────────┬──────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│  Container Build (CodeBuild)                                  │
│  NGC base image (16 GB) + RL deps → ECR                       │
└───────────────────────────┬──────────────────────────────────┘
                            │
                ┌───────────┴───────────┐
                │                       │
                ▼                       ▼
┌──────────────────────┐  ┌──────────────────────────┐
│  Option A: SageMaker │  │  Option B: AWS Batch      │
│  Managed training    │  │  Self-managed compute     │
│  Pay per job         │  │  VPC + GPU fleet          │
│  Auto-teardown       │  │  Full control             │
└──────────────────────┘  └──────────────────────────┘
                │                       │
                └───────────┬───────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│  Checkpoints → S3                                             │
│  model_*.pt · tensorboard logs · policy.onnx                  │
└──────────────────────────────────────────────────────────────┘
```

Both options use the **same container image** and produce the same output — trained policy checkpoints in S3. The difference is how the GPU compute is provisioned and managed.

---

## Which Option Should I Use?

| | **SageMaker** | **AWS Batch** |
|---|---|---|
| **Best for** | Quick experiments, workshops, one-off runs | Repeated runs, custom networking, fleet control |
| **Setup effort** | Minimal — IAM role + ECR image | Moderate — VPC, compute env, job queue, launch template |
| **GPU provisioning** | Managed by SageMaker | You control instance type, AZs, scaling |
| **Cost model** | Pay per second of training | EC2 pricing (can keep instances warm) |
| **Teardown** | Automatic when job completes | Manual (scale to zero or destroy infra) |
| **Multi-node** | `--instance-count N` (managed networking) | Batch MNP with NCCL security group |
| **Scaling** | Change instance type per job | Adjust compute environment limits |
| **VPC required** | No | Yes (private subnets for compute) |
| **Checkpoint storage** | S3 (via SageMaker output path) | S3 (via boto3 upload in entrypoint) |
| **Monitoring** | SageMaker console + CloudWatch | Batch console + CloudWatch |

**Choose SageMaker when:**
- You want the fastest path to a trained policy
- You're running workshop labs or experimenting
- You don't need persistent GPU infrastructure
- You prefer zero infrastructure management

**Choose AWS Batch when:**
- You want full control over the compute fleet
- You need to keep GPU instances warm between runs
- You're running many training jobs back-to-back
- You need custom VPC or security group configuration
- You want to optimize for cost with EC2 pricing

---

## Shared Prerequisites

Both paths require:

1. **Foundation infrastructure deployed** — S3 buckets, ECR repos, IAM roles
2. **NGC API key in Secrets Manager** — for pulling the Isaac Lab base image from NVIDIA
3. **Container image built** — the Isaac Lab training container pushed to ECR

These steps are covered in both guides below.

---

## Deployment Guides

### [AWS Batch Guide →](batch-rl-training-guide.md)

Full Terraform-based deployment: Foundation → VPC → Batch compute → Container build → Job submission → S3 checkpoints.

### [SageMaker Guide →](sagemaker-rl-training-guide.md)

*(Coming soon)* — SageMaker-based deployment with `pai rl launch` CLI integration.

---

## Training Parameters

Both paths accept the same training configuration via environment variables:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TASK` | `Isaac-Velocity-Flat-G1-v0` | Isaac Lab task name |
| `NUM_ENVS` | `4096` | Parallel simulation environments per GPU |
| `MAX_ITERATIONS` | `100` | PPO training iterations |
| `PROC_PER_NODE` | `1` | GPUs per node (match your instance) |
| `FRAMEWORK` | `rsl_rl` | RL framework: `rsl_rl`, `skrl`, or `rl_games` |

---

## Available Tasks

| Task | Type | Description |
|------|------|-------------|
| `Isaac-Velocity-Flat-G1-v0` | Locomotion | Unitree G1 walking on flat terrain (validated on L40S) |
| `Isaac-Velocity-Flat-Anymal-D-v0` | Locomotion | Quadruped walking on flat terrain |
| `Isaac-Velocity-Rough-Anymal-D-v0` | Locomotion | Quadruped on rough terrain |
| `Isaac-Reach-Franka-v0` | Manipulation | Franka arm reaching |
| `PickAndPlaceUR3-v0` | Manipulation | UR3 pick-and-place (registered, not yet container-wired) |

---

## What Happens During Training

Isaac Lab runs PPO (Proximal Policy Optimization) with domain randomization:

1. **4096 robot copies** run simultaneously on one GPU
2. Each copy sees a **randomized scene** (varied positions, lighting, colors)
3. The policy must succeed across all variations to score well
4. PPO updates the neural network weights every iteration
5. Over 1000+ iterations, the policy learns robust strategies

Training output per iteration:
```
Learning iteration 50/100
Computation: 31367 steps/s (collection: 0.698s, learning 0.086s)
Mean reward: -0.30
Mean episode length: 73.70
```

A converged policy typically needs 1000-2000 iterations (20-60 min on a single GPU).

---

## Validate the Trained Policy with an Agent (Strands Agents)

[**strands-robots**](https://github.com/strands-labs/robots) provides the agentic
orchestration layer that runs an Isaac Lab policy in the loop. It exposes a full
**Isaac Sim backend** (`strands_robots.simulation.isaac`) alongside MuJoCo, plus RL
training env wrappers (`strands_robots.training.rl`), so the exported policy can be
regression-gated in sim and then supervised in natural language:

```python
from strands import Agent
from strands_robots import Robot

robot = Robot("anymal_d")                        # MuJoCo twin by default; Isaac Sim backend available
# Only use policies you trust — a checkpoint can execute arbitrary code on load.
robot.run_policy(policy_config={"pretrained_name_or_path": "s3://.../policy.pt"})
Agent(tools=[robot])("walk forward across the rough terrain")
```

The agent is the **feedback arrow** of the flywheel — it turns a trained `policy.pt`
into behavior, observes the outcome, and decides whether to refine the reward, add
domain randomization, or promote the checkpoint. See
[strands-agents-on-aws](../strands-agents-on-aws/) for the orchestration layer and
`pai agent sim` to run it from the CLI.
