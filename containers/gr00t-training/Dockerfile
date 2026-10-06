# GR00T N1.6 Fine-Tuning Container
#
# Based on the proven approach from:
# https://github.com/aws-samples/sample-embodied-ai-platform/blob/main/training/gr00t/N16/Dockerfile
#
# Fine-tunes NVIDIA GR00T N1.6-3B on customer data in LeRobot v2 format.
# Runs on SageMaker (ml.g5.12xlarge) or AWS Batch (g6e.4xlarge+).
#
# The base image (nvcr.io/nvidia/pytorch) is pullable without an NGC API key.

FROM nvcr.io/nvidia/pytorch:25.04-py3

ARG USE_STABLE=true
# https://github.com/NVIDIA/Isaac-GR00T/releases/tag/n1.6.1-release
ARG STABLE_COMMIT=5dc80c4afd726b34faad1d8f7e007a13b34e4c88

ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute

# System deps
RUN apt-get update && apt-get install -y \
    build-essential yasm cmake libtool git git-lfs pkg-config \
    libass-dev libfreetype6-dev libvorbis-dev \
    autoconf automake texinfo ffmpeg \
    wget curl ca-certificates unzip vim less htop \
    libegl1 \
    && rm -rf /var/lib/apt/lists/*

# AWS CLI v2
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
    unzip awscliv2.zip && ./aws/install && rm -rf awscliv2.zip aws

# UV package manager
RUN pip install uv

# Upstream dep fixes
RUN sed -i s/dill==0.3.9/dill==0.3.8/g /etc/pip/constraint.txt && \
    sed -i s/scipy==1.15.2/scipy==1.15.3/g /etc/pip/constraint.txt && \
    sed -i '/flash.attn/d' /etc/pip/constraint.txt

# N1.6 pip deps
RUN pip install imageio h5py boto3 transformers[torch] deepspeed timm peft \
    diffusers wandb tianshou dm_tree openai \
    albumentations==1.4.18 decord==0.6.0 torchcodec==0.4.0

# PyTorch3D from source
RUN cd /opt && git clone https://github.com/facebookresearch/pytorch3d.git && \
    cd pytorch3d && MAX_JOBS=2 pip install .

# Clone Isaac-GR00T (stable N1.6.1 release)
RUN git clone https://github.com/NVIDIA/Isaac-GR00T.git /workspace/gr00t-repo && \
    cd /workspace/gr00t-repo && \
    if [ "${USE_STABLE}" = "true" ]; then \
        echo "Using stable commit: ${STABLE_COMMIT}"; \
        git checkout ${STABLE_COMMIT}; \
    fi && \
    echo "GR00T version:" && git log -1 --format="%H %ai %s"

# Install gr00t package (retry uv sync in case of transient download failures)
RUN cd /workspace/gr00t-repo && \
    uv sync --no-install-project || uv sync --no-install-project && \
    uv pip install -e . --no-deps

# flash-attn (install into the gr00t venv created by uv sync)
RUN cd /workspace/gr00t-repo && \
    uv pip install flash-attn==2.7.4.post1 --no-build-isolation --python .venv/bin/python || \
    (sleep 10 && uv pip install flash-attn==2.7.4.post1 --no-build-isolation --python .venv/bin/python) || \
    (sleep 30 && uv pip install flash-attn==2.7.4.post1 --no-build-isolation --python .venv/bin/python)

# Activate the uv venv
ENV VIRTUAL_ENV="/workspace/gr00t-repo/.venv"
ENV PATH="/workspace/gr00t-repo/.venv/bin:${PATH}"

# EGL/Vulkan setup
RUN mkdir -p /usr/share/glvnd/egl_vendor.d && \
    echo '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}' \
    > /usr/share/glvnd/egl_vendor.d/10_nvidia.json
RUN mkdir -p /usr/share/vulkan/icd.d && \
    echo '{"file_format_version":"1.0.0","ICD":{"library_path":"libGLX_nvidia.so.0","api_version":"1.2.140"}}' \
    > /usr/share/vulkan/icd.d/nvidia_icd.json

# Additional utilities
RUN pip install notebook gpustat matplotlib pandas pyarrow "zarr>=2.16" opencv-python-headless

# Copy entrypoints
COPY train_entrypoint.py /opt/ml/code/train_entrypoint.py
COPY batch-train-entrypoint.sh /opt/ml/code/batch-train
RUN chmod +x /opt/ml/code/batch-train

# SageMaker compatibility
ENV PYTHONUNBUFFERED=1
ENV SAGEMAKER_PROGRAM=train_entrypoint.py
ENV PYTHONPATH="/workspace/gr00t-repo:/workspace"
ENV DATASET_LOCAL_DIR="/opt/ml/input/data/training"
ENV OUTPUT_DIR="/opt/ml/model"
ENV BASE_MODEL_PATH="nvidia/GR00T-N1.6-3B"
ENV GROOT_SDK_DIR="/workspace/gr00t-repo"

# Symlink so train_entrypoint.py finds SDK at the expected path
RUN ln -sf /workspace/gr00t-repo /opt/isaac-gr00t

RUN mkdir -p ${DATASET_LOCAL_DIR} ${OUTPUT_DIR}

WORKDIR /opt/ml/code

# Smart entrypoint: supports both SageMaker (default) and Batch (pass "batch-train" as arg)
# SageMaker: runs with no args -> executes train_entrypoint.py
# Batch: command=["batch-train"] -> executes batch-train-entrypoint.sh
COPY <<'EOF' /opt/ml/code/entrypoint.sh
#!/bin/bash
if [ "$1" = "batch-train" ]; then
    exec bash /opt/ml/code/batch-train
else
    exec python /opt/ml/code/train_entrypoint.py "$@"
fi
EOF
RUN chmod +x /opt/ml/code/entrypoint.sh

ENTRYPOINT ["/opt/ml/code/entrypoint.sh"]
