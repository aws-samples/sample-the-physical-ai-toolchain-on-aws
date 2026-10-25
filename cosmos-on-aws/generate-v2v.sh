#!/bin/bash
# =============================================================================
# Cosmos 3 — V2V Generation Script
#
# Run ON the Cosmos 3 instance after the server is ready.
# Generates a V2V (video-to-video) augmentation from a UR3 wrist camera clip.
#
# Usage:
#   bash /tmp/generate-v2v.sh
# =============================================================================

set -e

# Region and account are auto-detected (IMDSv2 + STS) so the script needs no
# editing. Override any of these by exporting them before running, e.g.
#   DATASETS_BUCKET=$(terraform -chdir=../foundation/infra output -raw datasets_bucket_name)
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
ACCOUNT_ID="${ACCOUNT_ID:-$(aws sts get-caller-identity --region "$REGION" --query Account --output text)}"
DATASETS_BUCKET="${DATASETS_BUCKET:-physical-ai-dev-datasets-${ACCOUNT_ID}}"
echo "Region: $REGION  Bucket: $DATASETS_BUCKET"

# Must match COSMOS_MODEL used in setup-cosmos3-server.sh ("super" or "nano").
COSMOS_MODEL="${COSMOS_MODEL:-super}"

case "$COSMOS_MODEL" in
  super) MODEL_ID="nvidia/Cosmos3-Super" ;;
  nano)  MODEL_ID="nvidia/Cosmos3-Nano" ;;
  *)
    echo "ERROR: Unknown COSMOS_MODEL '$COSMOS_MODEL' (expected 'super' or 'nano')" >&2
    exit 1
    ;;
esac

echo "=== Step 1: Verify server is ready ==="
MODELS=$(curl -s http://localhost:8000/v1/models 2>/dev/null)
if echo "$MODELS" | grep -q "$(basename "$MODEL_ID")"; then
    echo "✅ Server ready: $MODEL_ID"
else
    echo "❌ Server not ready or serving a different model. Check: docker logs cosmos3 2>&1 | tail -20"
    exit 1
fi

echo ""
echo "=== Step 2: Download input video from S3 ==="
aws s3 cp s3://${DATASETS_BUCKET}/groot-data/ur3/dataset/videos/chunk-000/observation.images.wrist/episode_000000.mp4 \
  /tmp/episode_000000.mp4 --region $REGION

echo "Input video:"
ls -lh /tmp/episode_000000.mp4

echo ""
echo "=== Step 3: Copy video into container ==="
docker cp /tmp/episode_000000.mp4 cosmos3:/tmp/episode_000000.mp4

echo ""
echo "=== Step 4: Generate V2V (Super: ~2-8 min, Nano: ~10-12 min) ==="
PROMPT="A UR3 robot arm with a Robotiq gripper reaches down to a dark matte table, grasps a small red wooden block, lifts it slowly, and places it onto a yellow sticky note approximately 6 inches away. Top-down wrist camera view. Colorful wooden blocks are scattered on the table. Smooth deliberate motion."

echo "Prompt: $PROMPT"
echo ""
echo "Generating 189 frames at 1280x720, 24fps..."
echo "Start time: $(date -u)"

# Uses the ASYNC job API (POST /v1/videos + poll + /content), not the
# /v1/videos/sync endpoint. /v1/videos/sync has a hardcoded server-side abort
# at ~600s regardless of client --max-time — confirmed hitting this on Nano
# (single GPU, no parallelism, full 189-frame/720p generation takes
# ~10-12 min). Super usually finishes well under 600s but async is used
# uniformly here since it's the robust path for either model — vLLM-Omni's
# own docs recommend /v1/videos/sync only for quick benchmarks.
CREATE_RESPONSE=$(docker exec cosmos3 curl -sS -X POST http://localhost:8000/v1/videos \
  -H "Accept: application/json" \
  -F "model=${MODEL_ID}" \
  -F "prompt=${PROMPT}" \
  -F "size=1280x720" \
  -F "num_frames=189" \
  -F "fps=24" \
  -F "num_inference_steps=35" \
  -F "guidance_scale=6.0" \
  -F "max_sequence_length=4096" \
  -F "flow_shift=10.0" \
  -F "extra_params={\"condition_frame_indexes_vision\":[0,1],\"condition_video_keep\":\"first\"}" \
  -F "seed=100" \
  -F "input_reference=@/tmp/episode_000000.mp4;type=video/mp4")
VIDEO_ID=$(echo "$CREATE_RESPONSE" | python3 -c "import json,sys; print(json.load(sys.stdin)['id'])")
echo "Job submitted: $VIDEO_ID"

echo "Polling for completion..."
for i in $(seq 1 90); do
  STATUS_JSON=$(docker exec cosmos3 curl -sS "http://localhost:8000/v1/videos/${VIDEO_ID}")
  STATUS=$(echo "$STATUS_JSON" | python3 -c "import json,sys; print(json.load(sys.stdin).get('status','unknown'))")
  echo "  [$i] status=$STATUS ($(date -u +%H:%M:%S))"
  if [ "$STATUS" = "completed" ]; then
    break
  fi
  if [ "$STATUS" = "failed" ] || [ "$STATUS" = "cancelled" ]; then
    echo "ERROR: job $STATUS: $STATUS_JSON" >&2
    exit 1
  fi
  sleep 10
done
if [ "$STATUS" != "completed" ]; then
  echo "ERROR: job did not complete in time (last status: $STATUS)" >&2
  exit 1
fi

echo "End time: $(date -u)"
echo ""

echo "=== Step 5: Verify output ==="
docker exec cosmos3 curl -sS -L "http://localhost:8000/v1/videos/${VIDEO_ID}/content" -o /tmp/output_v2v.mp4
docker cp cosmos3:/tmp/output_v2v.mp4 /tmp/output_v2v.mp4
ls -lh /tmp/output_v2v.mp4

echo ""
echo "=== Step 6: Upload to S3 ==="
aws s3 cp /tmp/output_v2v.mp4 \
  s3://${DATASETS_BUCKET}/cosmos-samples/augmented_episode_000000.mp4 \
  --region $REGION

aws s3 cp /tmp/episode_000000.mp4 \
  s3://${DATASETS_BUCKET}/cosmos-samples/original_episode_000000.mp4 \
  --region $REGION

echo ""
echo "============================================================"
echo "  ✅ V2V Generation Complete!"
echo ""
echo "  Original: s3://${DATASETS_BUCKET}/cosmos-samples/original_episode_000000.mp4"
echo "  Generated: s3://${DATASETS_BUCKET}/cosmos-samples/augmented_episode_000000.mp4"
echo ""
echo "  Download and compare:"
echo "    aws s3 cp s3://${DATASETS_BUCKET}/cosmos-samples/ ./cosmos-output/ --recursive --region $REGION"
echo "============================================================"
