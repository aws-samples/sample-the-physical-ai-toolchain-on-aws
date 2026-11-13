# Lab 2: Isaac Sim Development Workstation

**Goal:** Deploy a GPU-powered remote desktop for visual Isaac Lab environment development and debugging
**Time:** 30 min setup + ongoing development sessions
**Cost:** ~$3.00/hr + $40/month EBS when running (stop when not in use)

> **New to simulation?** *Isaac Sim* is NVIDIA's physics-accurate 3D simulator (gravity,
> friction, collisions, cameras — like a video game engine for robots). *Isaac Lab* is a
> training framework that runs *on top of* Isaac Sim — it creates thousands of parallel
> robot copies and trains policies via RL. See the [terminology guide](README.md#physical-ai-terminology)
> for more.

---

## 🏃 Quick Runbook (do this in order)

> Follow these steps top-to-bottom. Each step says **what to run** and **how you know it worked**. Full detail for each is in the numbered sections below.

| # | Action | Command (summary) | ✅ Success check |
|---|--------|-------------------|-----------------|
| 0 | Subscribe to the Isaac Sim AMI (once per account) | Marketplace console — see *Prerequisite* | Subscription shows **Active** |
| 1 | Set your region in `config.json` | `pai config set aws.region us-west-2` | matches the region you'll deploy in |
| 2 | Deploy the workstation | `pai deploy workstation` | `✅ CREATE_COMPLETE`, prints instance ID + IP |
| 3 | Get IP and password | `pai workstation ip` / `pai workstation password` | prints IP and sets password |
| 4 | Connect via browser | open `https://<IP>:8443` | DCV login → Ubuntu desktop renders |
| 5 | Launch Isaac Sim (visual) | `~/run-isaac-sim-gui.sh` | 3D viewport opens |
| 6a | Enter the training container | `~/run-isaac-lab.sh` | prompt changes to `/workspace/isaaclab#` (you're now inside the container) |
| 6b | Start visual RL training (run *inside* the container) | `./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Velocity-Flat-Anymal-D-v0` | render window + training logs steps/s |
| 7 | (Optional) Closed-loop policy eval | inside the same container: `eval_policy_server.py` + `eval_sim_client.py` (two shells) | writes `success_rate_pct` JSON (validated on L40S) |
| 8 | **Stop the instance** | `pai workstation stop` | state → `stopped` (billing halts) |

> **What is `~/run-isaac-lab.sh`?** It's a helper script the workstation set up for you. It starts the `isaac-lab` Docker container (the same image SageMaker trains in) and drops you into a shell *inside* it — your prompt becomes `/workspace/isaaclab#`. Every `./isaaclab.sh …` command in this lab is typed **inside that container**, not on the host. Type `exit` to leave the container.

**Before you start, confirm:**
- [ ] AWS credentials active for the **test account** (`aws sts get-caller-identity`)
- [ ] `config.json` `aws.region` is the region you intend (deploy reads it — Step 1)
- [ ] You completed the Marketplace **subscription** (Step 0 / *Prerequisite* below)
- [ ] Foundation stack already deployed (so the `isaac-lab` image is in your ECR) — needed only for Step 5

> 💸 **Cost reminder:** this instance bills ~$3.00/hr while running. Run Step 8 (`pai workstation stop`) the moment you walk away. Full cost breakdown is in the [main README](../README.md#cost-summary).

---

## Why You Need This

You can't see what's happening inside a headless SageMaker training job. When your RL reward isn't improving, you need to *watch* the robot:

- Is the gripper missing the object by 2mm?
- Is the object falling through the table (physics bug)?
- Is the camera seeing what you think it's seeing?
- Is domain randomization too aggressive (objects flying off screen)?

The Isaac Sim workstation gives you a full visual desktop with GPU rendering, connected to the same environments you'll train headlessly on SageMaker. **Develop visually here, train at scale there.**

**The workflow:**
1. Deploy workstation (one time, ~5 min)
2. Start instance when you need to develop
3. Connect via web browser (NICE DCV remote desktop)
4. Iterate on your RL environment visually — see the robot, tweak rewards, fix bugs
5. Once it works visually → launch headless training at scale on SageMaker (Lab 4)
6. Stop instance when done

> **Alternative:** NVIDIA's [Isaac Automator](https://github.com/isaac-sim/IsaacAutomator) deploys a
> standalone Isaac Sim box on AWS/Azure/GCP. Use it if that's all you need; use this lab for the
> AWS-native CDK integration (the workstation is one stack in a blueprint that also wires SageMaker,
> Batch MNP, and ECR builds).

---

## Prerequisite: Subscribe to the Isaac Sim Marketplace AMI (one time)

> Instance sizing, monthly cost estimates, and the architecture diagram for this workstation live in the [main README](../README.md#cost-summary). This lab stays focused on the hands-on steps.

This workstation boots the **NVIDIA Isaac Sim AMI** from AWS Marketplace, which ships with the NVIDIA driver, NICE DCV, Docker, and Isaac Sim **pre-installed**. You must subscribe to it **once per account** before deploying, or the EC2 launch fails with a subscription/opt-in error:

1. Open the listing: **https://aws.amazon.com/marketplace/pp/prodview-bl35herdyozhw**
2. Click **Continue to Subscribe** → **Accept Terms** (the AMI itself is free; you pay only for the EC2 instance + EBS).
3. Wait for the subscription to show **Active** (usually < 1 min).

> **Region note:** The region→AMI map lives in **`config.json`** (`workstation.amiMapping`). Marketplace AMI IDs change with each Isaac Sim release — if your region is missing or the deploy reports an invalid AMI, look up the current AMI ID for your region from the listing's **Launch** tab and update `config.json`.

---

## Step 1: Deploy the Workstation

All workstation settings (instance type, region→AMI map, EBS size, DCV port) now live in **`config.json`** at the repo root. The deploy reads them automatically and auto-detects your public IP for the security group.

```bash
# Deploy with auto IP detection and AZ retry on GPU capacity errors
pai deploy workstation

# To override the instance type
pai deploy workstation --instance-type g5.2xlarge

# To use a specific IP (e.g., if behind NAT or VPN)
pai deploy workstation --allowed-cidr "203.0.113.5/32"
```

The CLI auto-detects your public IP (via `ifconfig.me`) and will retry across Availability Zones if GPU capacity is exhausted in one AZ. If DCV won't connect after deployment, your browser may be behind a different IP — re-deploy with that IP in `--allowed-cidr`.

The command outputs:
- **WorkstationInstanceId** — the EC2 instance ID (use it for start/stop)
- **DCVWebURL** — how to connect via browser. The instance uses an **auto-assigned public IP** (not an Elastic IP), so this output gives you the command to fetch the current IP: `pai workstation ip`. ⚠️ The public IP **changes every time you stop/start** — re-fetch it after each start.
- **Cost** — ~$3.00/hr (on-demand g6e.4xlarge in us-west-2) + EBS

> ⚠️ **Note on the public IP:** because it's not an Elastic IP, your `allowedCidr` security-group rule is unaffected by stop/start (that's keyed to *your* IP), but the DCV URL you bookmark will change. Always run `pai workstation ip` after starting the instance.

<details>
<summary>Under the hood (raw commands)</summary>

The `pai deploy workstation` command wraps:

```bash
cd sample-the-physical-ai-toolchain-on-aws/cdk

# Deploy with the wrapper script
./deploy-workstation.sh

# Or use CDK directly
MY_IP=$(curl -s ifconfig.me)
npx cdk deploy PhysicalAi-dev-Workstation \
  --context mode=simple \
  --context workstation=true \
  --context allowedCidr="$MY_IP/32"
```

</details>

---

## Step 2: Connect via Browser

The workstation auto-configures everything on first boot (~15 min). Once ready:

```bash
# Get the current public IP
pai workstation ip

# Set the DCV password (if not already set)
pai workstation password

# Or print connection details
pai workstation connect
```

Then:

1. Open `https://<IP>:8443` in your browser (use IP from `pai workstation ip`)
2. Accept the self-signed certificate warning
3. Login: username `ubuntu`, password from `pai workstation password` (default: `pai-lab1`)
4. You'll see an Ubuntu desktop with GPU acceleration

> **Note:** DCV requires direct internet access (port 8443). If you're on a corporate VPN that blocks non-standard ports, disconnect VPN to access DCV.

> **Prefer a native client?** Instead of the browser you can use the **NICE DCV desktop client** ([download](https://www.amazondcv.com/)) — connect it to the same `<IP>:8443` with username `ubuntu` and your DCV password. The native client generally gives smoother rendering and better keyboard/mouse handling for 3D work than the browser.

<details>
<summary>Under the hood (raw commands)</summary>

```bash
# Get IP
INSTANCE_ID=<your-instance-id>
aws ec2 describe-instances --instance-ids $INSTANCE_ID \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text

# Set password via SSM
aws ssm send-command \
  --instance-ids $INSTANCE_ID \
  --document-name "AWS-RunShellScript" \
  --parameters 'commands=["echo ubuntu:YOUR_PASSWORD | sudo chpasswd"]'

# Connect via SSM (alternative)
aws ssm start-session --target $INSTANCE_ID
```

</details>

---

## Step 3: Launch Isaac Sim (Visual Path)

The workstation boots the **NVIDIA Isaac Sim Marketplace AMI**, so Isaac Sim is **pre-installed**. The AMI installs it to **`/opt/IsaacSim`** (also mirrored under `~/IsaacSim`), and the launcher is **not on your `PATH`**, so call it by absolute path from the DCV desktop terminal:

```bash
# Isaac Sim ships with the AMI at /opt/IsaacSim. Launch the GUI:
/opt/IsaacSim/isaac-sim.sh

# Convenience: the bootstrap also dropped ~/run-isaac-sim-gui.sh which calls the above.
~/run-isaac-sim-gui.sh
```

> **Note:** The install path (`/opt/IsaacSim`) is defined by the AMI and verified on the current Isaac Sim Marketplace release. If a future AMI moves it, find the launcher with `ls /opt/IsaacSim/isaac-sim.sh ~/IsaacSim/isaac-sim.sh`. For RL training, use the **`isaac-lab` container** (Step 4) — the same image SageMaker runs.

You'll see the full Isaac Sim visual editor — 3D viewport, content browser with robots and environments, scene tree.

---

## Step 4: Verify the Workstation Renders (visual smoke test)

The goal here is to confirm your GPU workstation actually renders and trains — *not* to
produce a useful policy. You'll run Isaac Lab's built-in **Anymal** task and watch it in
the DCV desktop; that proves the container + GPU + GUI passthrough all work. The real
training workflow (at scale, headless, on SageMaker/Batch) is **[Lab 4](lab-5-rl-refinement-with-isaac.md)**.

The toolchain code is already on the workstation — the deploy bundles your working tree as
an S3 asset and the bootstrap unzips it to `/home/ubuntu/aws-physical-ai-toolchain`:

```bash
cd ~/aws-physical-ai-toolchain
```

> (The public GitHub repo isn't released yet, so there's no `git clone` — once it's
> public you can pass `--context repoUrl=<url>` to clone instead.)

### One environment for everything: the `isaac-lab` container

All GPU work here — visual training, iteration, policy eval — runs **inside the `isaac-lab`
container**, the *same image* SageMaker and AWS Batch run in Lab 4 (Isaac Sim 4.5.0 + Isaac Lab
2.1.0). What you see render here is exactly what trains at scale. (The Marketplace AMI's native
Isaac Sim is only for the standalone GUI in Step 3; RL goes through the container.)

The bootstrap pre-pulls the image and drops a launcher that wires up GPU + GUI passthrough:

```bash
~/run-isaac-lab.sh
# prompt becomes /workspace/isaaclab# — you're inside the container
```

> ✅ **GUI passthrough validated** (g6e.4xlarge / L40S, driver 580): RTX/Vulkan renders to the DCV
> display and an Anymal task trains to completion. A non-fatal `Warp CUDA error: cuDeviceGetUuid`
> may print on boot — ignore it. **First** GUI boot is slow (~5 min: extension sync + shader
> compile); cached dirs make later boots fast.

Inside the container, run the built-in Anymal task and watch it render — a few hundred
iterations is plenty to confirm the workstation is healthy:

```bash
# Visual (render window opens on the DCV desktop):
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Flat-Anymal-D-v0 --num_envs 4096

# Headless (no GUI rendering — all physics still run, but faster because
# the GPU focuses on computation instead of rendering pixels):
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Flat-Anymal-D-v0 --num_envs 4096 --headless
```

Your repo working tree is mounted at `/workspace/toolchain` inside the container, so you
can edit files on the host (or in the DCV desktop's editor) and re-run immediately — no
rebuild. This is also where you'd iterate visually on a custom env before training it at
scale in Lab 4.

> 🛠️ **Want to write your own task?** `train.py`/`play.py` are Isaac Lab's bundled example
> scripts. To build a custom env, see NVIDIA's
> [Isaac Lab docs](https://isaac-sim.github.io/IsaacLab/main/source/overview/own-project/index.html)
> ([custom RL env tutorial](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/create_direct_rl_env.html)).
> This repo's UR3 env (`training/envs/pick_and_place_ur3.py`) is one example.

> 💾 **Where does the trained policy go?** The container runs with `--rm`, so checkpoints written
> to its default `logs/...` are **lost on exit.** To keep one, save it under the mounted repo
> (`/workspace/toolchain/...`, on the EBS volume) or copy it to S3. There's no automatic permanent
> storage here — fine for learning; production would checkpoint to a versioned S3 bucket.

**What to look for in the render window:**
- Robots tracking the commanded velocity (locomotion reward working)
- Stable gait, no flipping or limb clipping through the floor
- `steps/s` climbing in the terminal — the workstation is training

If you see the robots stepping and the logs ticking, your workstation is good to go.

> The toolchain's UR3 pick-and-place env (`PickAndPlaceUR3-v0`) is covered in **[Lab 4](lab-5-rl-refinement-with-isaac.md)**. It's not yet GPU-validated, so Lab 2 uses the proven Anymal task for this smoke test.

---

## Start/Stop the Workstation

**Stop when done (saves money):**
```bash
pai workstation stop
```

**Start when you need it again:**
```bash
pai workstation start

# Wait ~60s for boot, then get the new public IP and reconnect via DCV:
pai workstation ip
```

**Check current status:**
```bash
pai workstation status
```

> ⚠️ The instance has an **auto-assigned public IP, not an Elastic IP** — it changes on every stop/start. Re-run `pai workstation ip` after each start to get the current `https://<IP>:8443` URL.

Your work is preserved on the EBS volume — stopping only halts the compute charges.

<details>
<summary>Under the hood (raw commands)</summary>

```bash
INSTANCE_ID=<your-instance-id>

# Stop
aws ec2 stop-instances --instance-ids $INSTANCE_ID

# Start
aws ec2 start-instances --instance-ids $INSTANCE_ID

# Get IP
aws ec2 describe-instances --instance-ids $INSTANCE_ID \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text
```

</details>

---

## Running the Container Exactly as SageMaker Does (parity check)

Optional: run the **same image** through its `train` entrypoint non-interactively — exactly how
SageMaker/Batch invoke it in Lab 4 — to reproduce a job locally before scaling out. Needs Foundation
deployed so `physical-ai/isaac-lab:latest` is in your ECR (else `docker pull` returns `not found`).

```bash
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION=$(aws configure get region)
ECR_REGISTRY="$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com"
IMAGE="$ECR_REGISTRY/physical-ai/isaac-lab:latest"

aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$ECR_REGISTRY"
docker pull "$IMAGE"

# Run it like SageMaker would
docker run --gpus all \
  -v /tmp/test-output:/opt/ml/model \
  -v /tmp/test-config:/opt/ml/input/config \
  "$IMAGE" train
```

---

## Policy Evaluation (come back here after Lab 4)

Once you have a checkpoint from [Lab 4](lab-5-rl-refinement-with-isaac.md), you evaluate it **here** — export,
eval, and the live visual playback all run in this workstation's `isaac-lab` container on the L40S
GPU. The full walkthrough (fetch checkpoint → export → eval → **watch it run**) lives in
**[Lab 4 → Step 4](lab-5-rl-refinement-with-isaac.md#step-4-evaluate-the-trained-policy)**, next to the training
that produced it. Step 4c renders the trained policy live on this DCV desktop — the visual payoff of RL refinement.

---

## ✅ Lab 2 Checkpoint

You've completed Lab 2 if:
- [ ] Workstation is deployed and accessible via DCV
- [ ] You can launch Isaac Sim and see rendered environments
- [ ] You can run the training environment visually with a small number of envs
- [ ] You understand the start/stop workflow to manage costs
- [ ] You can pull and test training containers locally on the workstation
- [ ] (Optional) You ran the closed-loop policy evaluator (Lab 4 Step 4) here

---

## Tips for Cost Management

1. **Always stop when you walk away.** Set a calendar reminder or use AWS Instance Scheduler.
2. **Use Spot instances for non-critical work.** Modify the CDK stack to use Spot — saves ~70% but can be interrupted.
3. **Right-size the instance.** g6e.4xlarge (1× L40S, 48GB VRAM) is the recommended default for the Isaac Sim AMI and is sufficient for environment development. Change `workstation.instanceType` in `config.json` (or `INSTANCE_TYPE=…`) if you need more — only scale up if you run 4096+ envs visually.
4. **Delete when the project is done.** `cdk destroy PhysicalAi-dev-Workstation` removes everything.

---

**Previous:** [← Lab 1: Train from Demonstrations](lab-1-train-groot.md)
**Next:** [Lab 3: Cosmos World Generation →](lab-3-cosmos-world-generation.md)
