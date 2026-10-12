# Strands Agents on AWS

The **Agentic AI Orchestration Layer** at the center of the Physical AI flywheel.

Where the other toolchain components each own one stage — [Cosmos](../cosmos-on-aws/)
generates data, [Isaac GR00T](../isaac-gr00t-on-aws/) and
[Isaac Lab](../isaac-lab-on-aws/) train policies, [Isaac Sim](../isaac-sim-on-aws/)
validates them — **[strands-robots](https://github.com/strands-labs/robots)** is the
natural-language control plane that ties them together and drives the robot at the
end of the loop.

```python
from strands import Agent
from strands_robots import Robot

robot = Robot("so100")                       # MuJoCo sim by default; mode="real" for hardware
Agent(tools=[robot])("pick up the red cube")  # natural language -> motor commands
```

One `Robot()` call returns a **MuJoCo simulation** (default — no GPU, no hardware)
or a **real robot** (`mode="real"`) behind the same interface, auto-joined to a
peer-to-peer mesh. A [Strands Agent](https://strandsagents.com) reasons over the
task in natural language and calls the robot's tools to act.

---

## Why this belongs in the toolchain

The [root README](../README.md) lists **Agentic Orchestration — Strands Agents SDK +
Amazon Bedrock AgentCore** as a pillar of the development flywheel. This component
provides that layer. It is the connective tissue between the compute-heavy stages:

| Flywheel stage | Toolchain component | How strands-robots connects |
|----------------|--------------------|------------------------------|
| Synthetic Data Generation | [cosmos-on-aws](../cosmos-on-aws/) | Agent scripts scene prompts and curates generated episodes |
| Model Training | [isaac-gr00t-on-aws](../isaac-gr00t-on-aws/), [isaac-lab-on-aws](../isaac-lab-on-aws/) | `train_policy` tool launches LeRobot / GR00T fine-tunes on demos the agent recorded |
| SIL Simulation | [isaac-sim-on-aws](../isaac-sim-on-aws/) | `Robot()` sim twin runs a candidate policy for regression gating before hardware |
| Sim-to-Real / HIL | *Edge (Jetson)* | Same policy interface runs on the physical arm; mesh coordinates a fleet |

The agent is the **feedback arrow** of the flywheel: it turns a trained checkpoint
(GR00T `model.tar.gz` in S3, or an Isaac Lab `policy.pt`) into closed-loop robot
behavior, observes the result, and decides what to generate or train next.

---

## What strands-robots gives an agent

- **Sim-first, safe by default.** `Robot("so100")` spins up a MuJoCo world. Real
  servos never move unless you explicitly pass `mode="real"`.
- **50+ robots, 8 categories.** Arms, humanoids, quadrupeds, hands, drones,
  bimanual rigs — resolved from a registry with auto-download of assets.
- **Any policy.** VLA models ([NVIDIA GR00T](https://developer.nvidia.com/isaac/gr00t),
  [LeRobot](https://github.com/huggingface/lerobot) ACT/Pi0/SmolVLA/Diffusion),
  plus classical planners and scripted controllers behind one interface — the same
  policies this toolchain trains.
- **Teleop + dataset recording.** Drive a real arm to collect demos as a
  LeRobot v2 dataset — the exact format [Lab 1](../workshop/lab-1-train-groot.md)
  fine-tunes GR00T on.
- **Mesh networking built in.** Every robot is a peer. `tell()` another robot what
  to do, or broadcast an E-STOP across a fleet; bridge to AWS IoT Core.
- **ROS 2 interop.** Observe and command any ROS 2 graph, or expose a running sim
  as a ROS node.

See the [strands-robots README](https://github.com/strands-labs/robots) for the full
capability surface.

---

## Two deployment shapes on AWS

**1. Laptop / workstation control plane (default).**
Run the agent locally against a MuJoCo sim or a robot on your network. This is the
fastest way to exercise the full loop with no cloud cost:

```bash
pip install 'strands-agents' 'strands-robots[sim-mujoco]'
pai agent sim --robot so101 --task "pick up the red cube"
```

**2. Amazon Bedrock AgentCore (managed, serverless).**
Deploy the same agent as a hosted runtime for fleet-scale orchestration — a durable
control plane that survives restarts and coordinates many robots. The agent reasons
with an [Amazon Bedrock](https://aws.amazon.com/bedrock/) model (Claude) and calls
robot tools over the mesh:

```
┌────────────────────────────────────────────────────────────┐
│  Amazon Bedrock AgentCore (managed runtime)                  │
│  Strands Agent + Bedrock model (Claude) + strands-robots     │
└───────────────────────────┬──────────────────────────────────┘
                            │  mesh (Zenoh) / AWS IoT Core
                ┌───────────┼───────────┐
                ▼           ▼           ▼
          ┌─────────┐ ┌─────────┐ ┌─────────┐
          │ Robot 1 │ │ Robot 2 │ │  Sim    │   ← MuJoCo twins + real arms
          │ (edge)  │ │ (edge)  │ │ (EC2)   │
          └─────────┘ └─────────┘ └─────────┘
```

Deployment to AgentCore is *Planned* in this component — the CLI (`pai agent`)
currently targets the laptop/workstation control plane, which is the documented,
validated path today. The sim path needs no AWS resources.

---

## Quickstart (sim, no hardware, no cloud)

```bash
# 1. Install the agent + MuJoCo sim extra
pip install 'strands-agents' 'strands-robots[sim-mujoco]'

# 2. Drive a simulated SO-101 arm with natural language
pai agent sim --robot so101 --task "pick up the red cube"

# 3. Inspect the robot registry (what embodiments are available)
pai agent info --robot so101
```

`pai agent sim` opens a MuJoCo world, hands the robot's tools to a Strands Agent,
and runs your natural-language task. Nothing physical moves; nothing is provisioned
on AWS.

---

## How it closes the loop with the rest of the toolchain

```
Lab 1 (GR00T)  ─ fine-tune ─▶  model.tar.gz in S3
                                     │
                          pai agent sim --policy s3://.../model.tar.gz
                                     │
                                     ▼
                        MuJoCo twin runs the policy   ◀── SIL gate (Isaac Sim)
                                     │  pass?
                                     ▼
                        pai agent sim --robot so101 --mode real   (HIL, opt-in)
                                     │
                          observe outcome ─▶ decide next data/train step
                                     │
                                     └────────▶  back to Cosmos / GR00T / Isaac Lab
```

The agent is what makes the flywheel *turn on its own*: it does not just run one
stage, it decides which stage to run next based on what it observed the robot do.

---

## Status

| Capability | Status |
|------------|--------|
| Natural-language sim control (`pai agent sim`) | Available |
| Robot registry inspection (`pai agent info`) | Available |
| LeRobot dataset recording (via strands-robots teleop) | Available (strands-robots) |
| Policy rollout — GR00T / LeRobot checkpoints | Available (strands-robots) |
| Bedrock AgentCore hosted runtime (`pai agent deploy`) | Planned |
| AWS IoT Core mesh bridge for fleets | Planned |

---

## License

Apache 2.0 — see [LICENSE](../LICENSE). `strands-robots` is Apache 2.0
([strands-labs/robots](https://github.com/strands-labs/robots)).
