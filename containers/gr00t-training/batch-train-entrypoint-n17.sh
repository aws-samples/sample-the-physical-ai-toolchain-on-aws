#!/bin/bash
# =============================================================================
# GR00T N1.7 AWS Batch Training Entrypoint
#
# Environment variables (set by job definition):
#   BASE_MODEL          - HuggingFace model name (default: nvidia/GR00T-N1.7-3B)
#   DATASET_S3_URI      - S3 URI to download dataset from
#   DATASET_PATH        - Local path to training data (default: /opt/ml/input/data/training)
#   MAX_STEPS           - Training steps (default: 10000)
#   BATCH_SIZE          - Global batch size (default: 12)
#   LEARNING_RATE       - Learning rate (default: 1e-4)
#   GRAD_ACCUM_STEPS    - Gradient accumulation steps (default: 4)
#   CHECKPOINT_BUCKET   - S3 bucket for checkpoint upload
#   CHECKPOINT_PREFIX   - S3 key prefix (default: checkpoints/groot-n17/<job_id>)
#   EMBODIMENT_CONFIG   - embodiment modality config to use: ur3 (default) | g1_dex3
#   MODALITY_CONFIG_PATH- explicit path to a modality config module (wins over
#                         EMBODIMENT_CONFIG; must exist inside the container)
#   HF_TOKEN            - HuggingFace token for model download (optional)
#   HF_TOKEN_SECRET_ID  - Secrets Manager secret id/ARN holding the token. Used
#                         only when HF_TOKEN is unset; resolved with the Batch
#                         instance role. Preferred over a plaintext HF_TOKEN in
#                         the job definition.
#
# NOTE: the base model nvidia/GR00T-N1.7-3B is NOT gated on HuggingFace
# (the hub API reports gated:false), so a missing/unresolvable token is a
# warning, not a failure. The token only matters for gated or private models.
# =============================================================================

set -e

# --- Configuration from environment ---
BASE_MODEL="${BASE_MODEL:-nvidia/GR00T-N1.7-3B}"
DATASET_PATH="${DATASET_PATH:-/opt/ml/input/data/training}"
DATASET_S3_URI="${DATASET_S3_URI:-}"
MAX_STEPS="${MAX_STEPS:-10000}"
BATCH_SIZE="${BATCH_SIZE:-12}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
OUTPUT_DIR="/opt/ml/model"
JOB_ID="${AWS_BATCH_JOB_ID:-local}"
EMBODIMENT_CONFIG="${EMBODIMENT_CONFIG:-ur3}"
MODALITY_CONFIG_PATH="${MODALITY_CONFIG_PATH:-}"
SECRET_ID="${HF_TOKEN_SECRET_ID:-}"

# --- Download dataset from S3 ---
if [ -n "$DATASET_S3_URI" ]; then
    echo "Downloading dataset from S3: $DATASET_S3_URI ..."
    mkdir -p "$DATASET_PATH"
    aws s3 sync "$DATASET_S3_URI" "$DATASET_PATH/"
    echo "Dataset ready at $DATASET_PATH"
    ls "$DATASET_PATH/meta/" 2>/dev/null || echo "WARNING: no meta/ directory found"
else
    echo "WARNING: DATASET_S3_URI not set. Training will fail unless data exists at $DATASET_PATH"
fi

# --- Count GPUs ---
NPROC=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "1")

# --- Print info ---
echo "============================================================"
echo "  GR00T N1.7 Fine-Tuning (AWS Batch)"
echo "  Base model:       $BASE_MODEL"
echo "  Dataset:          $DATASET_PATH"
echo "  Output:           $OUTPUT_DIR"
echo "  Max steps:        $MAX_STEPS"
echo "  Batch size:       $BATCH_SIZE"
echo "  Learning rate:    $LEARNING_RATE"
echo "  Grad accum:       $GRAD_ACCUM_STEPS"
echo "  GPUs:             $NPROC"
echo "  Job ID:           $JOB_ID"
echo "  Embodiment:       $EMBODIMENT_CONFIG"
echo "  Modality config:  ${MODALITY_CONFIG_PATH:-<from embodiment>}"
echo "============================================================"

mkdir -p "$OUTPUT_DIR"

# --- Resolve the HuggingFace token from Secrets Manager if not passed directly ---
# The value is assigned, never printed; only "resolved / not resolved" is logged.
if [ -z "${HF_TOKEN:-}" ] && [ -n "$SECRET_ID" ]; then
    echo "Resolving the HuggingFace token from Secrets Manager..."
    # `|| true`-style fallback: the base model is not gated, so a failed lookup
    # must not kill the job under `set -e`.
    HF_TOKEN=$(aws secretsmanager get-secret-value \
        --secret-id "$SECRET_ID" \
        --region "${AWS_DEFAULT_REGION:-us-east-2}" \
        --query SecretString --output text 2>/dev/null) || HF_TOKEN=""
    if [ -n "$HF_TOKEN" ]; then
        echo "  HuggingFace credential: resolved from Secrets Manager"
    else
        echo "  HuggingFace credential: NOT resolved (continuing; base model is not gated)"
    fi
fi

# --- Export HF token if set ---
if [ -n "${HF_TOKEN:-}" ]; then
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
    export HF_TOKEN="$HF_TOKEN"
fi

# --- Pass the embodiment selection through to the python entrypoint ---
# train_entrypoint.py reads these from the environment; a Batch job definition
# cannot override the Docker ENTRYPOINT, so this is the only seam available.
export EMBODIMENT_CONFIG
if [ -n "$MODALITY_CONFIG_PATH" ]; then
    export MODALITY_CONFIG_PATH
fi

# --- Set SageMaker-compatible env vars for the entrypoint ---
export SM_MODEL_DIR="$OUTPUT_DIR"
export SM_CHANNEL_TRAINING="$DATASET_PATH"
export SM_HP_base_model="$BASE_MODEL"
export SM_HP_max_steps="$MAX_STEPS"
export SM_HP_batch_size="$BATCH_SIZE"
export SM_HP_learning_rate="$LEARNING_RATE"
export SM_HP_gradient_accumulation_steps="$GRAD_ACCUM_STEPS"

echo "Launching GR00T N1.7 training..."
python3 /opt/ml/code/train_entrypoint.py

echo "Training complete."

# --- Upload checkpoints to S3 ---
if [ -n "${CHECKPOINT_BUCKET:-}" ]; then
    S3_PREFIX="${CHECKPOINT_PREFIX:-checkpoints/groot-n17/${JOB_ID}}"
    echo "Uploading checkpoints to s3://${CHECKPOINT_BUCKET}/${S3_PREFIX}/ ..."
    aws s3 sync "$OUTPUT_DIR" "s3://${CHECKPOINT_BUCKET}/${S3_PREFIX}/" --region "${AWS_DEFAULT_REGION:-us-east-2}"
    echo "Done. Checkpoints uploaded to s3://${CHECKPOINT_BUCKET}/${S3_PREFIX}/"
else
    echo "WARNING: CHECKPOINT_BUCKET not set. Checkpoints will be lost when container exits."
fi
