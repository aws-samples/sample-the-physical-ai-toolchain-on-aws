#!/bin/bash
# =============================================================================
# Run ON the Cosmos 3 p5.48xlarge instance (via SSM session)
#
# This script:
#   1. Verifies GPUs
#   2. Pulls the vLLM-Omni Cosmos3 container (~30GB)
#   3. Retrieves HF token from Secrets Manager
#   4. Starts the Cosmos3 server (Super or Nano, see COSMOS_MODEL below)
#
# Usage (from SSM session on the instance):
#   bash /tmp/setup-cosmos3-server.sh
# =============================================================================

set -e

# Region of the instance this runs on. Auto-detected from IMDSv2 so the script
# needs no editing; override by exporting REGION before running.
if [ -z "${REGION:-}" ]; then
  IMDS_TOKEN=$(curl -fsS --connect-timeout 1 -m 3 -X PUT "http://169.254.169.254/latest/api/token" \
    -H "X-aws-ec2-metadata-token-ttl-seconds: 300" 2>/dev/null || true)
  REGION=$(curl -fsS --connect-timeout 1 -m 3 -H "X-aws-ec2-metadata-token: $IMDS_TOKEN" \
    "http://169.254.169.254/latest/meta-data/placement/region" 2>/dev/null || true)
fi
if [ -z "$REGION" ]; then
  echo "ERROR: could not determine region from IMDS; export REGION=<region> and re-run" >&2
  exit 1
fi
echo "Region: $REGION"

# Model choice — "super" (64B, highest quality, needs all 8 GPUs) or "nano"
# (16B, needs only 1 GPU, faster/cheaper, some quality tradeoff). See
# cosmos-on-aws/README.md "Choosing Super vs Nano" for the full comparison.
COSMOS_MODEL="${COSMOS_MODEL:-super}"

case "$COSMOS_MODEL" in
  super)
    MODEL_ID="nvidia/Cosmos3-Super"
    SERVE_ARGS="--cfg-parallel-size 2 --ulysses-degree 4 --use-hsdp --hsdp-shard-size 8"
    GPUS_NEEDED=8
    ;;
  nano)
    MODEL_ID="nvidia/Cosmos3-Nano"
    # --vae-use-tiling is required on single-GPU instances like g6e.4xlarge
    # (~44GB usable VRAM) — without it, VAE decode OOMs at the default
    # 189-frame/720p generation size. Cuts peak decode VRAM ~68% for ~13%
    # extra latency. Validated: fixes CUDA OOM on L40S.
    SERVE_ARGS="--vae-use-tiling"
    GPUS_NEEDED=1
    ;;
  *)
    echo "ERROR: Unknown COSMOS_MODEL '$COSMOS_MODEL' (expected 'super' or 'nano')" >&2
    exit 1
    ;;
esac

echo "=== Step 1: Verify GPUs ==="
nvidia-smi -L
GPU_COUNT=$(nvidia-smi -L | wc -l)
echo "Found $GPU_COUNT GPUs ($COSMOS_MODEL needs $GPUS_NEEDED)"
if [ "$GPU_COUNT" -lt "$GPUS_NEEDED" ]; then
    echo "WARNING: $COSMOS_MODEL needs $GPUS_NEEDED GPUs, found $GPU_COUNT"
fi

echo ""
echo "=== Step 2: Pull vLLM-Omni Cosmos3 container ==="
echo "This is ~30GB, takes 5-10 min on high-bandwidth instance..."
docker pull vllm/vllm-omni:cosmos3

echo ""
echo "=== Step 3: Get HF token from Secrets Manager ==="
HF_TOKEN=$(aws secretsmanager get-secret-value \
  --secret-id physical-ai/hf-token \
  --region $REGION \
  --query 'SecretString' --output text)

if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: Could not retrieve HF token from Secrets Manager"
    echo "Create it with: aws secretsmanager create-secret --name physical-ai/hf-token --secret-string 'hf_...' --region $REGION"
    exit 1
fi
echo "HF token retrieved (${#HF_TOKEN} chars)"

echo ""
echo "=== Step 4: Start Cosmos3 vLLM server ($MODEL_ID) ==="
echo "Model download: Super ~128GB, Nano ~33GB from HuggingFace (first time)."
echo "Subsequent starts use cached weights."

docker run -d \
  --name cosmos3 \
  --gpus all \
  --ipc=host \
  --shm-size=64g \
  -p 8000:8000 \
  -e HF_TOKEN="$HF_TOKEN" \
  -e HF_HOME=/workspace/hf-cache \
  -v /opt/hf-cache:/workspace/hf-cache \
  vllm/vllm-omni:cosmos3 \
  bash -c "vllm serve $MODEL_ID --omni $SERVE_ARGS --init-timeout 2400 --stage-init-timeout 1800 --host 0.0.0.0 --port 8000"

echo ""
echo "=== Server starting (background) ==="
echo ""
echo "Monitor progress:"
echo "  docker logs -f cosmos3 2>&1 | grep -E '(download|ready|error|Worker)'"
echo ""
echo "Check if ready:"
echo "  curl -s http://localhost:8000/v1/models"
echo ""
echo "When ready, it returns: {\"data\":[{\"id\":\"$MODEL_ID\",...}]}"
