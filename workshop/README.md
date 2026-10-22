# Workshop Introduction

Read this before starting the labs. It explains what Physical AI is, how robots learn, the key technologies you'll use, and what each lab teaches.

---

## What is Physical AI?

Physical AI is artificial intelligence that interacts with the real world. Unlike chatbots or image generators that produce text and pixels, Physical AI produces **motor commands** — signals that move robot arms, open grippers, and navigate through space.

A Physical AI system takes in camera images and joint sensor readings, reasons about what it sees, and outputs precise movements 50-200 times per second. Teaching a robot to pick up an object from a bin requires solving perception (where is it?), planning (how do I reach it?), and control (what exact motor commands get me there?) — all in real-time.

This toolkit provides the infrastructure and workflow to build these systems using AWS services and the NVIDIA robotics stack.

---

## How Robots Learn

Traditional robot programming is manual: engineers write explicit rules for every movement, every edge case, every variation. This breaks down in unstructured environments where objects can be anywhere and look different every time.

Modern Physical AI uses **learned policies** — neural networks trained from data that can generalize to new situations. There are two approaches:

### 1. Imitation Learning (Lab 1)

A human demonstrates the task using teleoperation (remote control). The robot records what it sees (camera) and what it does (motor commands). A foundation model called **GR00T** (Generalist Robot 00 Technology) is fine-tuned on these demonstrations to predict: *given what I see now, what should I do next?*

This gives you a working policy in hours from as few as 50 demonstrations. But it only works well in situations that look like the demos.

### 2. Reinforcement Learning (Lab 5)

The robot practices in simulation — millions of attempts with a reward signal ("+1 when the object is picked up, -0.1 for dropping it"). Through trial and error across thousands of randomized scenes, it discovers robust strategies that handle variations the demos never showed.

### Two approaches, two pipelines — choose per task

Imitation learning (GR00T) and reinforcement learning (Isaac Lab PPO) are **separate pipelines you choose between**, not stages you chain. You pick based on what you have: high-quality demonstrations, or a strong simulator with a definable reward.

```
Path A — Imitation:   Demos ──▶ GR00T fine-tune (SageMaker) ──┐
                                                              ├──▶ Deploy
Path B — RL:          Sim + reward ──▶ Isaac Lab PPO (GPU EC2)─┘
```

---

## What Each Lab Teaches

### [Lab 0: Prerequisites](lab-0-prerequisites.md)
Deploy AWS infrastructure with CDK. Creates S3 buckets, ECR repos, IAM roles, and triggers CodeBuild to build all container images. After this, everything else "just works."

### [Lab 1: Train from Demonstrations (GR00T)](lab-1-train-groot.md)
Fine-tune NVIDIA's **GR00T** foundation model (a 3B-parameter Vision-Language-Action model pre-trained on diverse robot data) on 27 real UR3 teleoperation episodes. You'll convert raw Zarr recordings to **LeRobot v2** format (HuggingFace's standard for robot training data — Parquet + MP4), upload to S3, and launch a **SageMaker** training pipeline (managed GPU training — provisions hardware, runs your container, uploads results, terminates with no idle cost). The result: a model that predicts robot motor commands from camera images. Two **optional bring-your-own-robot** sections bracket the lab for teams with a physical UR3 — record your own demonstrations by teleoperation (`pai groot record`) and run the trained policy on the arm (`pai groot control`), completing the full loop. No robot is required for the workshop.

### [Lab 2: Isaac Sim Workstation](lab-2-isaac-workstation.md)
Deploy a GPU-powered remote desktop running **Isaac Sim** (NVIDIA's physics-accurate 3D simulator — gravity, friction, collisions, cameras, like a video game engine for robots). Watch robots train in real-time, debug physics issues you can't see in logs, and iterate on RL environments visually before training at scale.

### [Lab 3: Cosmos World Generation (Predict)](lab-3-cosmos-world-generation.md)
Use **Cosmos 3 Super** (NVIDIA's 64B World Foundation Model) to generate entirely new synthetic pick-and-place demonstrations from text prompts. One reference video + varied prompts/seeds = dozens of novel training episodes. Scales your dataset without additional teleoperation.

### [Lab 4: Cosmos Transfer (Restyle)](lab-4-cosmos-transfer.md)
Use **Cosmos Transfer 2.5** to restyle your existing training videos — pixel-faithful restyling that preserves exact geometry and motion while changing visual appearance (factory lighting, worn surfaces, different materials). Preserves action labels while adding visual diversity for sim-to-real transfer.

### [Lab 5: RL Policy Training with Isaac Lab](lab-5-rl-refinement-with-isaac.md)
Train a robot policy from scratch with reinforcement learning using **Isaac Lab** (an RL training framework on top of Isaac Sim that runs 4096 parallel robot copies on one GPU). **PPO** (Proximal Policy Optimization — the standard RL algorithm for robotics) discovers robust strategies through trial-and-error guided by reward signals. Domain randomization makes the policy generalize to real hardware.

### Lab 6: OSMO Orchestration (Placeholder)
Chain multiple stages (generate → train → evaluate → deploy) into an automated production pipeline on EKS. The lab is not written yet; the OSMO deployment it would build on is in [osmo-on-aws](../osmo-on-aws/). Contributions welcome.

---

## Physical AI Terminology

New to robotics? Here's what the key terms mean.

### The Basics

| Term | Plain English |
|------|--------------|
| **Physical AI** | AI that moves things in the real world — robot arms picking objects, drones navigating, humanoids walking. Takes camera images in, produces motor commands out. |
| **Policy** | The trained "brain" of the robot. A neural network file (`.pt`) that takes sensor data and outputs actions. Roboticists say "policy" instead of "model" because it makes decisions. |
| **Teleoperation (Teleop)** | A human remotely controlling a robot to demonstrate a task. Recordings become training data. |
| **Episode** | One complete task demonstration from start to finish. A dataset contains 50-200 episodes. |
| **Embodiment** | The physical robot body. Policies are embodiment-specific — trained for one robot, won't work on another without retraining. |

### Training Approaches

| Term | Plain English |
|------|--------------|
| **Imitation Learning** | "Learn by watching." Show the robot 50 demos, it learns to copy. Fast but limited to what was demonstrated. |
| **Reinforcement Learning (RL)** | "Learn by practice." Millions of tries in simulation with a reward signal. Discovers strategies beyond what humans showed. |
| **Fine-tuning** | Train just the last few layers of a pre-trained model on your specific data. Cheap and effective. |
| **PPO** | Proximal Policy Optimization. The standard RL algorithm for robotics. What Isaac Lab uses. |
| **Domain Randomization** | Randomly vary everything during training (positions, lighting, colors). Forces the policy to generalize. |
| **Sim-to-Real Transfer** | The gap between simulation and reality. Domain randomization and Cosmos close this gap. |

### Models & Simulation

| Term | Plain English |
|------|--------------|
| **GR00T** | NVIDIA's robot foundation model. 3B parameters. Fine-tune with 50+ demonstrations. |
| **VLA (Vision-Language-Action)** | A model that sees images, understands language, and outputs motor commands. GR00T is a VLA. |
| **Isaac Sim** | NVIDIA's physics engine for robots. Simulates gravity, friction, collisions, cameras. |
| **Isaac Lab** | RL training framework on top of Isaac Sim. Creates thousands of parallel robot copies. |
| **Cosmos** | NVIDIA's World Foundation Model. Generates or restyles video for training data. |
| **Parallel Environments** | Running 4096 robot copies simultaneously on one GPU. Days of practice in seconds. |
| **Headless** | Running the simulator without graphics. Faster because the GPU focuses on computation. |

### Hardware & Data

| Term | Plain English |
|------|--------------|
| **UR3** | Universal Robots 6-joint arm. Most popular in research + light manufacturing. Our reference robot. |
| **Jetson** | NVIDIA's edge GPU board. Runs trained policies inside robots at 50-200 Hz. |
| **TensorRT** | NVIDIA's model compiler. Makes models 10-100x faster for real-time control. |
| **ROS 2** | Robot middleware. Pub/sub messaging between sensors, models, and actuators. |
| **LeRobot v2** | HuggingFace's standard format. Parquet (numbers) + MP4 (video). GR00T reads this. |
| **URDF** | Robot description file (XML). Geometry, joints, limits. Every simulator needs this. |
| **NICE DCV** | AWS remote desktop. Streams GPU graphics to your browser. How you see Isaac Sim. |

---

## Getting Started

→ **[Begin with Lab 0: Prerequisites](lab-0-prerequisites.md)**

For the complete glossary, see [docs/glossary.md](../docs/glossary.md).
