#!/usr/bin/env bash
# SPDX-License-Identifier: MIT-0
#
# Per-account build of the Isaac Lab and Isaac Lab-Arena images used by the workshop.
#
# Runs once at account provisioning (CodeBuild, privileged Docker) so attendees do not
# wait 2 to 3 hours for these builds during the lab. Each image is skipped when its tag
# is already in ECR, so a retry after a late failure only rebuilds what is missing.
#
#   3.0.0-beta2-g1             Isaac Lab training image        (Isaac Lab on AWS)
#   arena-base-0.2.1           Arena 0.2.1 recipe              (intermediate)
#   arena-base-0.2.1-torchfix  Arena with Isaac Sim's PyTorch  (Arena smoke test)
#   arena-gr00t-9468a3b        torchfix + GR00T policy servers (intermediate)
#   arena-g1-dex3              G1 Dex3 GR00T scoring image     (Arena GR00T evaluation)
#   arena-base-0.3.0           Arena 0.3.0, NVIDIA's recipe    (intermediate)
#   arena-g1-velocity          G1 velocity RL scoring image    (Arena RL evaluation)
#
# Environment:
#   ECR_REPO_URI        (required) <account>.dkr.ecr.<region>.amazonaws.com/physical-ai/isaac-lab
#   AWS_DEFAULT_REGION  (required)
#   NGC_SECRET_ID       (required) Secrets Manager id/ARN holding the NVIDIA NGC API key
#   TOOLKIT_DIR         Clone of this repository (default: the directory above this script)
#   FORCE_REBUILD       'true' rebuilds every image even when its tag exists
set -euo pipefail

: "${ECR_REPO_URI:?set ECR_REPO_URI}"
: "${AWS_DEFAULT_REGION:?set AWS_DEFAULT_REGION}"
: "${NGC_SECRET_ID:?set NGC_SECRET_ID}"
TOOLKIT_DIR="${TOOLKIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
FORCE_REBUILD="${FORCE_REBUILD:-false}"
REGISTRY="${ECR_REPO_URI%%/*}"
REPO_NAME="${ECR_REPO_URI#*/}"
WORK="${WORK:-/tmp/lab-images}"

# Pinned sources, identical to the lab pages.
ARENA_REPO="https://github.com/isaac-sim/IsaacLab-Arena.git"
ARENA_REF="release/0.2.1"
ARENA_COMMIT="8b4a3a47fc53de23e8205089d71109a2e2348acd"
CONNECTOR_COMMIT="9468a3bae1c1711b2d85d45ce222cfbc68dd832d"
GROOT_COMMIT="1a1837f20538b7d7e21f977a11a5aee14f99803c"
TOOLCHAIN_REPO="https://github.com/aws-samples/sample-the-physical-ai-toolchain-on-aws.git"
# Arena 0.3.0 adds experiment_runner and the RSL-RL policy used by the RL evaluation.
ARENA_RL_REF="release/0.3.0"
ARENA_RL_COMMIT="8737b4ceb25f99f81a81786b7fde73139b52f324"

log() { echo "[$(date -u +%H:%M:%S)] $*"; }
has_tag() {
  [ "$FORCE_REBUILD" != "true" ] && aws ecr describe-images --region "$AWS_DEFAULT_REGION" \
    --repository-name "$REPO_NAME" --image-ids imageTag="$1" >/dev/null 2>&1
}
push() { docker push "$ECR_REPO_URI:$1"; log "PUSHED $ECR_REPO_URI:$1"; }
# Drop local images once pushed; the Arena chain needs the disk.
prune() { docker image prune -af >/dev/null 2>&1 || true; df -h / | tail -1; }

mkdir -p "$WORK"
log "ECR login $REGISTRY"
aws ecr get-login-password --region "$AWS_DEFAULT_REGION" \
  | docker login --username AWS --password-stdin "$REGISTRY"
log "NGC login (key read from Secrets Manager, never printed)"
NGC_API_KEY=$(aws secretsmanager get-secret-value --region "$AWS_DEFAULT_REGION" \
  --secret-id "$NGC_SECRET_ID" --query SecretString --output text)
printf '%s' "$NGC_API_KEY" | docker login nvcr.io --username '$oauthtoken' --password-stdin
unset NGC_API_KEY

# 1. Isaac Lab training image ----------------------------------------------------------
if has_tag 3.0.0-beta2-g1; then
  log "SKIP 3.0.0-beta2-g1: already in ECR"
else
  log "BUILD 3.0.0-beta2-g1 from containers/isaac-lab/Dockerfile"
  (cd "$TOOLKIT_DIR" && docker build --platform linux/amd64 -t "$ECR_REPO_URI:3.0.0-beta2-g1" \
    -f containers/isaac-lab/Dockerfile .)
  docker tag "$ECR_REPO_URI:3.0.0-beta2-g1" "$ECR_REPO_URI:latest"
  push 3.0.0-beta2-g1
  push latest
  prune
fi

# 2. Arena base ------------------------------------------------------------------------
if has_tag arena-base-0.2.1-torchfix || has_tag arena-base-0.2.1; then
  log "SKIP arena-base-0.2.1: base or torchfix already in ECR"
else
  log "BUILD arena-base-0.2.1 from Arena $ARENA_REF @ $ARENA_COMMIT"
  git config --global url."https://github.com/".insteadOf "git@github.com:"
  rm -rf "$WORK/IsaacLab-Arena"
  git clone --branch "$ARENA_REF" --recurse-submodules "$ARENA_REPO" "$WORK/IsaacLab-Arena"
  git -C "$WORK/IsaacLab-Arena" checkout "$ARENA_COMMIT"
  git -C "$WORK/IsaacLab-Arena" submodule update --init --recursive
  (cd "$WORK/IsaacLab-Arena" && docker build --pull --build-arg WORKDIR=/workspace \
    --build-arg INSTALL_GROOT=true -t "$ECR_REPO_URI:arena-base-0.2.1" \
    --file docker/Dockerfile.isaaclab_arena .)
  push arena-base-0.2.1
  rm -rf "$WORK/IsaacLab-Arena"
fi

# 3. PyTorch fix and GR00T connector ---------------------------------------------------
# Rebuilding Arena 0.2.1 today replaces Isaac Sim's torch 2.10.0+cu128 with an
# incompatible newer build; this layer restores it.
if has_tag arena-base-0.2.1-torchfix && has_tag arena-gr00t-9468a3b; then
  log "SKIP arena-base-0.2.1-torchfix and arena-gr00t-9468a3b: already in ECR"
else
  log "BUILD arena-base-0.2.1-torchfix"
  mkdir -p "$WORK/fix"
  SP=/isaac-sim/kit/python/lib/python3.12/site-packages
  cat > "$WORK/fix/Dockerfile" <<EOF
FROM $ECR_REPO_URI:arena-base-0.2.1
USER root
RUN rm -rf $SP/torch $SP/torch-*.dist-info $SP/torchvision $SP/torchvision-*.dist-info $SP/torchgen $SP/functorch $SP/triton $SP/triton-*.dist-info
RUN /isaac-sim/python.sh -m pip install --no-deps --ignore-installed --index-url https://download.pytorch.org/whl/cu128 torch==2.10.0+cu128 torchvision==0.25.0+cu128 triton==3.6.0
RUN /isaac-sim/python.sh -c "import torch, torchvision; print('TORCH_OK', torch.__version__, torchvision.__version__, torch.version.cuda)"
EOF
  docker build -t "$ECR_REPO_URI:arena-base-0.2.1-torchfix" "$WORK/fix"
  push arena-base-0.2.1-torchfix
  log "BUILD arena-gr00t-9468a3b (connector source @ $CONNECTOR_COMMIT)"
  rm -rf "$WORK/connector"
  git clone --filter=blob:none "$TOOLCHAIN_REPO" "$WORK/connector"
  git -C "$WORK/connector" checkout "$CONNECTOR_COMMIT"
  (cd "$WORK/connector" && docker build --platform linux/amd64 \
    --build-arg ARENA_BASE_IMAGE="$ECR_REPO_URI:arena-base-0.2.1-torchfix" \
    -f containers/isaac-lab-arena/Dockerfile -t "$ECR_REPO_URI:arena-gr00t-9468a3b" .)
  push arena-gr00t-9468a3b
  rm -rf "$WORK/connector"
fi

# 4. G1 Dex3 GR00T scoring image -------------------------------------------------------
if has_tag arena-g1-dex3; then
  log "SKIP arena-g1-dex3: already in ECR"
else
  log "BUILD arena-g1-dex3 (Isaac-GR00T @ $GROOT_COMMIT)"
  ctx="$WORK/g1"
  rm -rf "$ctx" && mkdir -p "$ctx/files"
  cp "$TOOLKIT_DIR"/isaac-lab-arena-on-aws/g1-dex3/{arena_g1_bluebox_env.py,arena_g1_dex3_policy.py,arena_g1_eval.sh} "$ctx/files/"
  cp "$TOOLKIT_DIR/isaac-sim-on-aws/closedloop/policy_bridge.py" "$ctx/files/"
  cp "$TOOLKIT_DIR/containers/gr00t-training/g1_dex3_modality_config.py" "$ctx/files/"
  cat > "$ctx/Dockerfile" <<EOF
FROM $ECR_REPO_URI:arena-gr00t-9468a3b
USER root
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends unzip curl && rm -rf /var/lib/apt/lists/* \
 && curl -fsSL https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip -o /tmp/awscli.zip \
 && unzip -q /tmp/awscli.zip -d /tmp && /tmp/aws/install && rm -rf /tmp/aws /tmp/awscli.zip && aws --version
RUN GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/NVIDIA/Isaac-GR00T.git /workspace/gr00t-repo \
 && git -C /workspace/gr00t-repo checkout $GROOT_COMMIT \
 && test "\$(git -C /workspace/gr00t-repo rev-parse HEAD)" = "$GROOT_COMMIT" \
 && git lfs install --skip-repo \
 && git -C /workspace/gr00t-repo lfs pull --include "scripts/deployment/**/wheels/*.whl"
RUN cd /workspace/gr00t-repo && export UV_HTTP_RETRIES=10 UV_HTTP_TIMEOUT=120 \
 && (uv sync --no-dev --no-install-package deepspeed --python 3.12 || uv sync --no-dev --no-install-package deepspeed --python 3.12)
RUN /workspace/gr00t-repo/.venv/bin/python -c "import pyarrow; from gr00t.policy.gr00t_policy import Gr00tPolicy; print('BRIDGE_VENV_OK')"
COPY files/ /opt/g1dex3/
RUN mkdir -p /opt/ml/code && cp /opt/g1dex3/g1_dex3_modality_config.py /opt/ml/code/ \
 && SP=\$(/isaac-sim/python.sh -c "import site; print(site.getsitepackages()[0])") \
 && cp /opt/g1dex3/arena_g1_bluebox_env.py /opt/g1dex3/arena_g1_dex3_policy.py "\$SP/" \
 && chmod +x /opt/g1dex3/arena_g1_eval.sh \
 && /isaac-sim/python.sh -m py_compile "\$SP/arena_g1_bluebox_env.py" "\$SP/arena_g1_dex3_policy.py" \
 && grep -c "return Qwen3VLProcessor.from_pretrained(model_name, \*\*transformers_loading_kwargs)" /workspace/gr00t-repo/gr00t/model/gr00t_n1d7/processing_gr00t_n1d7.py
ENTRYPOINT ["/bin/bash", "/opt/g1dex3/arena_g1_eval.sh"]
CMD []
EOF
  docker build -t "$ECR_REPO_URI:arena-g1-dex3" "$ctx"
  push arena-g1-dex3
  rm -rf "$ctx"
fi

# 5. Arena 0.3.0 and the G1 velocity RL scoring image --------------------------------
if has_tag arena-g1-velocity; then
  log "SKIP arena-base-0.3.0 and arena-g1-velocity: already in ECR"
else
  if has_tag arena-base-0.3.0; then
    log "SKIP arena-base-0.3.0: already in ECR"
    docker pull --quiet "$ECR_REPO_URI:arena-base-0.3.0"
  else
    log "BUILD arena-base-0.3.0 from Arena $ARENA_RL_REF @ $ARENA_RL_COMMIT"
    git config --global url."https://github.com/".insteadOf "git@github.com:"
    rm -rf "$WORK/IsaacLab-Arena-0.3.0"
    git clone --branch "$ARENA_RL_REF" "$ARENA_REPO" "$WORK/IsaacLab-Arena-0.3.0"
    git -C "$WORK/IsaacLab-Arena-0.3.0" checkout "$ARENA_RL_COMMIT"
    git -C "$WORK/IsaacLab-Arena-0.3.0" submodule update --init --recursive --depth 1
    (cd "$WORK/IsaacLab-Arena-0.3.0" && docker build -t "$ECR_REPO_URI:arena-base-0.3.0" \
      --file docker/Dockerfile.isaaclab_arena .)
    push arena-base-0.3.0
    rm -rf "$WORK/IsaacLab-Arena-0.3.0"
  fi
  log "BUILD arena-g1-velocity from isaac-lab-arena-on-aws/g1-velocity"
  (cd "$TOOLKIT_DIR/isaac-lab-arena-on-aws/g1-velocity" && docker build \
    --build-arg BASE_IMAGE="$ECR_REPO_URI:arena-base-0.3.0" -t "$ECR_REPO_URI:arena-g1-velocity" .)
  push arena-g1-velocity
  prune
fi

log "DONE all Isaac Lab and Arena images are in $ECR_REPO_URI"
