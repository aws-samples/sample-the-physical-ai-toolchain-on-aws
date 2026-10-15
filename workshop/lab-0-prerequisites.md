# Lab 0: Prerequisites

**Goal:** Verify your environment is ready to deploy the Physical AI Toolkit.
**Time:** 30 minutes
**Cost:** Free (no AWS resources created)

> **New to Physical AI?** This toolchain trains robot "brains" (called *policies*) that
> take camera images and produce motor commands. See the [terminology guide](README.md#physical-ai-terminology)
> for key concepts.

---

## What You're Setting Up

The Physical AI Toolkit is an end-to-end pipeline for training robot manipulation
policies on AWS. You'll deploy a "Foundation stack" — S3 buckets for data/models, ECR
repositories for containers, IAM roles for SageMaker, and CodeBuild projects that
automatically build GPU training containers. Once deployed, Labs 1-5 use this foundation
to train, refine, and deploy a pick-and-place robot policy.

---

## What You Need

| Requirement | Why | How to check |
|-------------|-----|--------------|
| AWS Account with GPU quota | Training runs on `ml.g5.12xlarge` | `aws service-quotas get-service-quota --service-code sagemaker --quota-code L-3B05A85A` |
| AWS CLI v2 configured | Deploy infrastructure | `aws sts get-caller-identity` |
| Node.js 18+ | CDK requires it | `node --version` |
| AWS CDK CLI | Deploy stacks | `npx cdk --version` |
| NVIDIA NGC API key | CodeBuild pulls Isaac Sim/Lab/Cosmos base images from NGC (NVIDIA's container registry) | Generate at https://ngc.nvidia.com/setup/api-key |
| Hugging Face token | Pull GR00T base model (during training) | Create at https://huggingface.co/settings/tokens |
| Git LFS | Lab 1 pulls the UR3 demo dataset (`training/data/*.zip`) from Git LFS | `git lfs version` (install: `brew install git-lfs` or `sudo apt-get install git-lfs`, then `git lfs install`) |

> **No Docker required.** All container images are built in **AWS CodeBuild** and
> pushed to ECR automatically when you deploy the Foundation stack. You never run
> `docker build` or pull multi-GB NVIDIA images locally — so a laptop with no GPU
> (including Apple Silicon) is perfectly fine. Docker is only needed if you opt
> into the local-build fast-path on a workstation (see Lab 2).

## Step-by-Step Setup

### 1. Clone the repo

```bash
git clone https://github.com/aws-samples/aws-physical-ai-toolchain.git
cd aws-physical-ai-toolchain
```

### 2. Install the `pai` CLI

The whole workshop is driven by the `pai` CLI (deploy infra, launch training,
manage the workstation). Install it once into a virtual environment — a venv
keeps its deps isolated and avoids the PEP 668 "externally-managed-environment"
error on system/Homebrew Python.

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e .                   # installs `pai` + all workshop dependencies
pai --version                      # confirm it's on PATH
```

Re-run `source .venv/bin/activate` in any new shell before using `pai`.

### 3. Install CDK dependencies

```bash
cd cdk
npm install
cd ..
```

### 4. Bootstrap CDK (once per account/region)

```bash
npx cdk bootstrap aws://<ACCOUNT_ID>/us-east-1
```

### 5. Set your region and environment variables

`config.json` is the single source of truth for the toolchain region — both `pai`
and CDK read it, so it can't drift from your shell. Set it once:

```bash
pai config set aws.region us-west-2    # or your target region

export CDK_DEFAULT_ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
export CDK_DEFAULT_REGION=us-east-1
export HF_TOKEN=hf_xxxx  # Your Hugging Face token
```

### 6. Store your NGC API key in Secrets Manager

CodeBuild uses this to pull NVIDIA base images (Isaac Sim/Lab, Cosmos) when it
builds the containers. One time per account:

```bash
aws secretsmanager create-secret --name physical-ai/ngc-api-key \
  --secret-string "YOUR_NGC_API_KEY" --region $CDK_DEFAULT_REGION
```

(The GR00T training image builds from a public base and needs no NGC key — so if
you only plan to do Lab 1, this step is optional.)

Optionally, store your Hugging Face token the same way. `pai groot launch` reads
`HF_TOKEN` from your shell first and falls back to this secret, so a new terminal
doesn't lose it. `read -rs` keeps the token out of your screen and shell history:

```bash
read -rs HF_TOKEN && aws secretsmanager create-secret --name physical-ai/hf-token \
  --secret-string "$HF_TOKEN" --region $CDK_DEFAULT_REGION && unset HF_TOKEN
```

### 7. Verify

```bash
pai doctor          # checks credentials, region, and (after deploy) the Foundation stack

cd cdk
npx cdk synth --context mode=simple 2>&1 | head -5
# Should print: Successfully synthesized to cdk.out
cd ..
```

### 8. Deploy the Foundation stack — builds start automatically

```bash
pai deploy foundation
# under the hood: npx cdk deploy PhysicalAi-dev-Foundation --context mode=simple
```

The stack itself deploys in ~3 minutes. As part of deploy, CDK uploads this repo
to S3 and **auto-triggers CodeBuild jobs** that build every container image and
push them to ECR — so you never build or pull large images locally. The builds
run in the background (groot ~10 min; isaac-lab/isaac-sim/cosmos can take up to an
hour). The stack's `BuildConsole` output links to the CodeBuild console to watch
progress. Builds re-run automatically on later `cdk deploy`s only when the source
changes.

---

## GPU Quota

SageMaker training (Path A) requires quota for `ml.g5.12xlarge` instances. Request it:

1. Go to [Service Quotas → SageMaker](https://console.aws.amazon.com/servicequotas/home/services/sagemaker/quotas)
2. Search for "ml.g5.12xlarge for training job usage"
3. Request increase to at least 1

**Quota approval typically takes 1-24 hours.** Request before the workshop.

---

## ✅ Ready Check

Run this to verify everything:

```bash
echo "=== Prerequisites Check ==="
echo "AWS CLI:     $(aws --version 2>&1 | head -1)"
echo "Account:     $(aws sts get-caller-identity --query Account --output text)"
echo "Region:      ${CDK_DEFAULT_REGION:-not set}"
echo "Node:        $(node --version)"
echo "CDK:         $(npx cdk --version 2>/dev/null || echo 'not installed')"
echo "pai CLI:     $(pai --version 2>/dev/null || echo 'not installed — run: pip install -e .')"
echo "HF_TOKEN:    ${HF_TOKEN:+set}${HF_TOKEN:-NOT SET}"
echo "NGC secret:  $(aws secretsmanager describe-secret --secret-id physical-ai/ngc-api-key --query Name --output text 2>/dev/null || echo 'not created (only needed for Isaac/Cosmos labs)')"
echo "==========================="
```

All items should show versions or "set". If anything says "not installed" or "NOT SET", fix it before proceeding to Lab 1. (Docker is **not** required — containers build in CodeBuild.)

---

**Next:** [Lab 1: Train Your First Robot Policy →](lab-1-train-groot.md)
