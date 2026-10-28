# Isaac Lab Training Container
#
# Extends the official NVIDIA Isaac Lab container with:
# - Additional RL and export dependencies
# - Custom UR3 pick-and-place environment
# - SageMaker and AWS Batch training entrypoints
# - S3 integration for checkpoints and models
#
# Build:
#   docker build -t isaac-lab-training:3.0.0-beta2-g1 .
#
# Run (headless training):
#   docker run --gpus all -v /data:/storage isaac-lab-training:3.0.0-beta2-g1 \
#     python /workspace/training/scripts/train.py --config /workspace/training/configs/ppo_pick_place.yaml

# Isaac Lab 3.0.0-beta2 is validated on the NVIDIA Isaac Sim Marketplace
# workstation with driver 595.71.05. Isaac Lab 2.1/Isaac Sim 4.5 crashes
# during renderer initialization on that driver branch.
FROM nvcr.io/nvidia/isaac-lab:3.0.0-beta2

# Isaac Lab 3 uses the Python 3.12 runtime bundled under _isaac_sim.
ARG ISAAC_LAB_PYTHON=/workspace/isaaclab/_isaac_sim/kit/python/bin/python3

# Install additional RL, export, and AWS dependencies into Isaac Lab's Python.
RUN ${ISAAC_LAB_PYTHON} -m pip install --no-cache-dir \
    rl_games==1.6.1 \
    tensorboard>=2.14 \
    boto3>=1.34 \
    pyyaml>=6.0 \
    onnx>=1.15 \
    onnxruntime-gpu>=1.17 \
    pyzmq>=25

# Install TensorRT Python bindings for the export step.
RUN ${ISAAC_LAB_PYTHON} -m pip install --no-cache-dir \
    tensorrt>=10.0

# Copy training entrypoints.
# NOTE: changing batch-train-entrypoint.sh changes the CodeBuild source-asset
# hash (ContainerBuild construct) so deploying Foundation retriggers the build.
COPY containers/isaac-lab/sm-train-entrypoint.sh /opt/ml/code/sm-train-entrypoint.sh
COPY containers/isaac-lab/batch-train-entrypoint.sh /opt/ml/code/batch-train
COPY containers/isaac-lab/train_entrypoint.py /opt/ml/code/train_entrypoint.py
RUN chmod +x /opt/ml/code/sm-train-entrypoint.sh /opt/ml/code/batch-train

# Copy training code.
COPY training/ /workspace/training/
COPY workflows/ /workspace/workflows/

# Copy custom UR3 environment. NOTE: not yet gym.register()-ed as an Isaac Lab
# task — see docs/ROADMAP.md (standalone RL feature). The validated workshop path
# uses Isaac Lab's built-in Isaac-Velocity-Flat-G1-v0 task.
COPY training/envs/ /workspace/training/envs/

WORKDIR /opt/ml/code

ENV ACCEPT_EULA=Y
ENV PRIVACY_CONSENT=Y
ENV OMNI_ENV_PRIVACY_CONSENT=Y

# Make Isaac Lab's Python available to entrypoints and S3 upload helpers.
RUN ln -sf ${ISAAC_LAB_PYTHON} /usr/local/bin/python3 \
    && ln -sf ${ISAAC_LAB_PYTHON} /usr/local/bin/python

# SageMaker invokes the container as: docker run <image> train.
RUN ln -sf /opt/ml/code/sm-train-entrypoint.sh /opt/ml/code/train \
    && chmod +x /opt/ml/code/train

ENV PATH="/workspace/isaaclab/_isaac_sim/kit/python/bin:/opt/ml/code:${PATH}"
ENV SAGEMAKER_PROGRAM=train

ENTRYPOINT []
CMD ["train"]
