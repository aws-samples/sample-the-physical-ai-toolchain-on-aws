# GR00T Inference / Serving Container
#
# Serves a fine-tuned GR00T N1.6 policy as a SageMaker real-time endpoint
# (BYOC: Flask /ping + /invocations on :8080). SageMaker runs it as
# `docker run <image> serve` — serve.py ignores the trailing arg and starts Flask.
#
# Builds in AWS CodeBuild → ECR (NEVER locally). Same DLC base + GR00T pin as the
# training image (containers/gr00t-training) so a checkpoint trained there loads
# here without SDK drift.
# DLC images are published in every region; pull from the build region so the
# CodeBuild role's region-scoped ECR grant covers it.
ARG DLC_REGION=us-west-2
FROM 763104351884.dkr.ecr.${DLC_REGION}.amazonaws.com/pytorch-inference:2.5.1-gpu-py311-cu124-ubuntu22.04-sagemaker

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends git-lfs ninja-build && \
    rm -rf /var/lib/apt/lists/* && git lfs install

# Isaac-GR00T pinned to the SAME ref the training image uses (n1.6-release), so
# serve loads checkpoints produced by training. Patch requires-python for the
# DLC's Python 3.11.
ARG GROOT_REF=n1.6-release
RUN git clone --depth 1 --branch ${GROOT_REF} https://github.com/NVIDIA/Isaac-GR00T.git /opt/isaac-gr00t && \
    cd /opt/isaac-gr00t && git lfs pull && \
    python -c "import pathlib; p=pathlib.Path('pyproject.toml'); p.write_text(p.read_text().replace('==3.10.*', '>=3.10'))" && \
    pip install --no-cache-dir -e . && \
    pip install --no-cache-dir transformers==4.51.3

# flash-attn (Eagle backbone). Prebuilt wheel matching torch2.7/cu12/cp311 — same
# approach as the training image. If it ends up broken, serve.py drops it at import.
RUN pip install --no-cache-dir \
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/flash_attn-2.8.3.post1+cu12torch2.7cxx11abiFALSE-cp311-cp311-linux_x86_64.whl" \
    || echo "flash-attn install failed; serve.py tolerates its absence"

# Flask + Pillow for the serving handler.
RUN pip install --no-cache-dir flask pillow

# Fast-fail canary: GR00T policy import + Eagle config present.
RUN python -c "from gr00t.policy.gr00t_policy import Gr00tPolicy; print('GR00T policy OK')" && \
    python -c "import torch; print(f'torch={torch.__version__}, cuda_build={torch.version.cuda}')"

COPY serve.py /opt/serve.py

ENV PYTHONUNBUFFERED=1
# Exec form so the process receives SIGTERM/SIGKILL from SageMaker. serve.py ignores
# the trailing `serve` arg SageMaker appends.
ENTRYPOINT ["python", "/opt/serve.py"]
