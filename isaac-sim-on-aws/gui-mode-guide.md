# Isaac Sim — GUI Mode Guide

Visual development and debugging with NVIDIA Isaac Sim via NICE DCV remote desktop.

---

## Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  Your Browser                                                    │
│  https://<IP>:8443 (NICE DCV)                                    │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  EC2 g6e.4xlarge (L40S GPU)                                      │
│  NVIDIA Isaac Sim pre-installed                                  │
│  Full 3D viewport, physics, sensors                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## Step 1: Deploy the Workstation

```bash
cd isaac-sim-on-aws/infra
terraform init
terraform apply -var="aws_region=us-east-2"
```

Note the instance IP from the Terraform output.

---

## Step 2: Connect via SSM and Set Password

```bash
# Connect via SSM (from AWS Console or CLI)
aws ssm start-session --target <INSTANCE_ID> --region us-east-2

# Set the DCV login password
sudo passwd ubuntu
```

---

## Step 3: Reboot (First Time Only)

The Isaac Sim AMI requires a reboot after first launch to initialize the GPU display properly:

```bash
sudo reboot
```

Wait 2 minutes for the instance to come back up.

---

## Step 4: Connect via DCV

1. Open `https://<INSTANCE_IP>:8443` in your browser
2. Accept the self-signed certificate warning
3. Login with username: `ubuntu`, password: (what you set in Step 2)
4. The Ubuntu desktop should render with GPU acceleration

---

## Step 5: Launch Isaac Sim

Open a terminal in the DCV desktop:

```bash
/opt/IsaacSim/isaac-sim.sh
```

> **First launch takes 3-5 minutes** (shader compilation and extension loading). Subsequent launches are faster.

You'll see the Isaac Sim editor with a 3D viewport, content browser, and property panels.

---

## Step 6: Load a Sample Robot Example

Once Isaac Sim is open, load the built-in Franka Pick and Place example:

1. In the menu bar, go to **Window → Examples → Robotics Examples**
2. A browser panel appears — click the **Robotics Examples** tab
3. Navigate to **Manipulation → Franka Pick and Place**
4. Click **Load** to load the example into the scene
5. Press **Play** (▶) in the toolbar to start the simulation

You'll see a Franka robot arm performing a pick-and-place task — reaching, grasping, lifting, and placing an object.

### Other examples to try

| Category | Example | What it shows |
|----------|---------|---------------|
| Manipulation | Franka Pick and Place | Robot arm grasping and placing objects |
| Manipulation | Follow Target | Robot end-effector following a moving target |
| Navigation | Carter Navigation | Autonomous mobile robot navigation |
| Quadruped | Anymal | Quadruped locomotion |

---

## Step 7: Run Isaac Lab in GUI Mode (Docker)

After training in [Isaac Lab](../isaac-lab-on-aws/) or [GR00T](../isaac-gr00t-on-aws/), you can visualize policies and run training with the GUI viewport inside a Docker container on the workstation.

> **Important:** GUI mode requires Isaac Lab `3.0.0-beta2` on the current workstation AMI. Legacy ECR images based on Isaac Lab 2.1.0 and the NGC 2.3.2 image crash in `librtx.scenedb.plugin.so` with driver 595.71.05. Use `nvcr.io/nvidia/isaac-lab:3.0.0-beta2` directly or rebuild `physical-ai/isaac-lab:3.0.0-beta2-g1` from the current Dockerfile.

### Pull the Isaac Lab 3.0.0-beta2 container

```bash
docker pull nvcr.io/nvidia/isaac-lab:3.0.0-beta2
```

### Launch the container with display passthrough

```bash
xhost +

docker run --shm-size=60g --entrypoint bash -it --gpus all \
  -e "ACCEPT_EULA=Y" \
  --rm --network=host \
  -e DISPLAY \
  -e "PRIVACY_CONSENT=Y" \
  -v ~/checkpoints:/workspace/checkpoints \
  nvcr.io/nvidia/isaac-lab:3.0.0-beta2
```

### Train with GUI visualization

Inside the container, run training with the GUI viewport visible:

```bash
./isaaclab.sh train --rl_library sb3 --task Isaac-Cartpole-v0 --num_envs 64 --viz kit
```

The Isaac Sim viewport opens showing the cartpole environments rendering in real time while training runs.

### Load and play a trained checkpoint with GUI

```bash
# Pull checkpoint from S3 (run on the host, not inside the container)
aws s3 cp s3://physical-ai-dev-checkpoints-<ACCOUNT_ID>/checkpoints/<JOB_ID>/ \
  ~/checkpoints/ --recursive --region us-east-2

# Inside the container — play the version-matched G1 checkpoint with Kit rendering
./isaaclab.sh play \
  --rl_library rsl_rl \
  --task Isaac-Velocity-Flat-G1-v0 \
  --checkpoint /workspace/checkpoints/model_99.pt \
  --num_envs 1 \
  --viz kit
```

### Container compatibility matrix

| Container | Headless Training | GUI Rendering | Notes |
|-----------|:-:|:-:|-------|
| `nvcr.io/nvidia/isaac-lab:3.0.0-beta2` | ✅ | ✅ | Validated on L40S with driver 595.71.05 |
| ECR `physical-ai/isaac-lab:3.0.0-beta2-g1` | ✅ | ✅ | Built from the same validated base image |
| Legacy ECR `physical-ai/isaac-lab:2.1.0-g1` | ✅ | ❌ Segfault | Keep only for reproducing legacy 2.1 checkpoints |
| `nvcr.io/nvidia/isaac-lab:2.3.2` | ✅ | ❌ Segfault | Same driver incompatibility |

---

## Start / Stop the Workstation

```bash
# Stop (saves money — EBS persists)
aws ec2 stop-instances --instance-ids <INSTANCE_ID> --region us-east-2

# Start
aws ec2 start-instances --instance-ids <INSTANCE_ID> --region us-east-2

# Get new IP after start (changes each time)
aws ec2 describe-instances --instance-ids <INSTANCE_ID> --region us-east-2 \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text
```

> ⚠️ The public IP changes on every stop/start. Re-fetch it each time.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Black screen on DCV | Reboot the instance: `sudo reboot` |
| DCV says "unknown username" | Use `ubuntu` (not `ec2-user`). Set password with `sudo passwd ubuntu` |
| Isaac Sim not found | Check `/opt/IsaacSim/isaac-sim.sh` |
| First launch very slow | Normal — shader compilation takes 3-5 min on first boot |
| `nvidia-smi` not working | Reboot — driver needs initialization after first launch |
| Volume too small error | AMI requires 512 GB minimum (set in Terraform variables) |
| Can't connect to DCV | Check security group allows port 8443 from your IP |

---

## Teardown

```bash
cd isaac-sim-on-aws/infra
terraform destroy -var="aws_region=us-east-2"
```
