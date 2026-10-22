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
#   TUNE_VISUAL         - 0 (default, upstream behaviour) | 1. Off keeps the
#                         vision tower frozen so the job fits one 48 GB L40S;
#                         on adds ~1.15B trainable params and needs 4 GPUs.
#   HF_TOKEN            - HuggingFace token for model download. Needed (directly or
#                         via HF_TOKEN_SECRET_ID) unless the archive variables below
#                         are set: the Cosmos-Reason2-2B backbone is gated (see NOTE).
#   HF_TOKEN_SECRET_ID  - Secrets Manager secret id/ARN holding the token. Used
#                         only when local model archives are absent and HF_TOKEN is unset.
#   GROOT_BASE_MODEL_ARCHIVE_S3_URI - participant-S3 GR00T N1.7 tar archive
#   GROOT_BASE_MODEL_SHA256         - SHA-256 for that archive
#   COSMOS_REASON2_ARCHIVE_S3_URI   - participant-S3 Cosmos Reason2 tar archive
#   COSMOS_REASON2_SHA256           - SHA-256 for that archive
#
# Event accounts set all four archive variables. The entrypoint verifies and extracts
# both models, rewrites GR00T's nested backbone reference to the local Reason2 directory,
# and enables Hugging Face/Transformers offline mode.
#
# NOTE: nvidia/GR00T-N1.7-3B itself is not gated on HuggingFace (the hub API
# reports gated:false), but its VLM backbone nvidia/Cosmos-Reason2-2B IS a gated
# repo, and training downloads it. Without a token from an account that has been
# granted access to that repo, the backbone download fails with HTTP 401. A
# missing/unresolvable token is still only a warning here, not a failure, so the
# job stops later, at that download. The offline archive path needs no token.
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
TUNE_VISUAL="${TUNE_VISUAL:-0}"
SECRET_ID="${HF_TOKEN_SECRET_ID:-}"
GROOT_BASE_MODEL_ARCHIVE_S3_URI="${GROOT_BASE_MODEL_ARCHIVE_S3_URI:-}"
GROOT_BASE_MODEL_SHA256="${GROOT_BASE_MODEL_SHA256:-}"
COSMOS_REASON2_ARCHIVE_S3_URI="${COSMOS_REASON2_ARCHIVE_S3_URI:-}"
COSMOS_REASON2_SHA256="${COSMOS_REASON2_SHA256:-}"
LOCAL_MODEL_ROOT="/opt/ml/input/models"

# --- Prepare participant-S3 model archives for offline event training ---
materialize_model_archive() {  # label uri sha destination required-file
    label="$1"; uri="$2"; expected="$3"; destination="$4"; required="$5"
    [[ "$expected" =~ ^[0-9a-fA-F]{64}$ ]] \
      || { echo "ERROR: ${label} SHA-256 must be 64 hexadecimal characters" >&2; exit 1; }
    expected="$(printf '%s' "$expected" | tr '[:upper:]' '[:lower:]')"
    mkdir -p "$LOCAL_MODEL_ROOT"
    archive="${LOCAL_MODEL_ROOT}/$(basename "$destination").tar"
    next="${destination}.next"
    echo "Downloading ${label} from participant S3: ${uri}"
    aws s3 cp "$uri" "$archive" --region "${AWS_DEFAULT_REGION:-us-east-2}" --only-show-errors
    printf '%s  %s\n' "$expected" "$archive" | sha256sum -c -
    rm -rf "$next"
    mkdir -p "$next"
    tar -xf "$archive" -C "$next"
    [ -s "$next/$required" ] || { echo "ERROR: ${label} archive lacks ${required}" >&2; exit 1; }
    rm -rf "$destination"
    mv "$next" "$destination"
    rm -f "$archive"
    echo "  ${label}: ready at ${destination}"
}

ARCHIVE_SETTINGS_PRESENT="${GROOT_BASE_MODEL_ARCHIVE_S3_URI}${GROOT_BASE_MODEL_SHA256}${COSMOS_REASON2_ARCHIVE_S3_URI}${COSMOS_REASON2_SHA256}"
if [ -n "$ARCHIVE_SETTINGS_PRESENT" ]; then
    [ -n "$GROOT_BASE_MODEL_ARCHIVE_S3_URI" ] && [ -n "$GROOT_BASE_MODEL_SHA256" ] \
      && [ -n "$COSMOS_REASON2_ARCHIVE_S3_URI" ] && [ -n "$COSMOS_REASON2_SHA256" ] \
      || { echo "ERROR: all four GR00T/Reason2 archive settings are required together" >&2; exit 1; }

    GROOT_LOCAL_DIR="${LOCAL_MODEL_ROOT}/GR00T-N1.7-3B"
    REASON2_LOCAL_DIR="${LOCAL_MODEL_ROOT}/Cosmos-Reason2-2B"
    materialize_model_archive "GR00T N1.7" "$GROOT_BASE_MODEL_ARCHIVE_S3_URI" \
      "$GROOT_BASE_MODEL_SHA256" "$GROOT_LOCAL_DIR" "config.json"
    materialize_model_archive "Cosmos Reason2" "$COSMOS_REASON2_ARCHIVE_S3_URI" \
      "$COSMOS_REASON2_SHA256" "$REASON2_LOCAL_DIR" "model.safetensors"

    python3 - "$GROOT_LOCAL_DIR" "$REASON2_LOCAL_DIR" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
reason = sys.argv[2]
source = "nvidia/Cosmos-Reason2-2B"

def rewrite(value):
    if isinstance(value, dict):
        return {key: rewrite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [rewrite(item) for item in value]
    if isinstance(value, str):
        return value.replace(source, reason)
    return value

changed = 0
for path in root.rglob("*.json"):
    try:
        original = json.loads(path.read_text())
    except (UnicodeDecodeError, json.JSONDecodeError):
        continue
    updated = rewrite(original)
    if updated != original:
        path.write_text(json.dumps(updated, indent=2) + "\n")
        changed += 1
if changed == 0:
    raise SystemExit("No Cosmos Reason2 references were found in the GR00T snapshot")
print(f"Rewrote Cosmos Reason2 to local path in {changed} GR00T JSON files")
PY

    BASE_MODEL="$GROOT_LOCAL_DIR"
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    echo "Offline model preparation: PASS"
fi

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
echo "  Tune visual:      $TUNE_VISUAL"
echo "============================================================"

mkdir -p "$OUTPUT_DIR"

# --- Resolve the HuggingFace token from Secrets Manager if not passed directly ---
# The value is assigned, never printed; only "resolved / not resolved" is logged.
warn_gated_backbone() {
    echo "  WARNING: the GR00T N1.7 backbone nvidia/Cosmos-Reason2-2B is a gated HuggingFace"
    echo "           repo. Without a token its download will fail with HTTP 401 Unauthorized."
}
if [ -z "$ARCHIVE_SETTINGS_PRESENT" ] && [ -z "${HF_TOKEN:-}" ] && [ -n "$SECRET_ID" ]; then
    echo "Resolving the HuggingFace token from Secrets Manager..."
    # `|| true`-style fallback: a failed lookup must not kill the job under
    # `set -e`. It is reported below; the gated backbone download is where a
    # missing token actually fails.
    HF_TOKEN=$(aws secretsmanager get-secret-value \
        --secret-id "$SECRET_ID" \
        --region "${AWS_DEFAULT_REGION:-us-east-2}" \
        --query SecretString --output text 2>/dev/null) || HF_TOKEN=""
    if [ -n "$HF_TOKEN" ]; then
        echo "  HuggingFace credential: resolved from Secrets Manager"
    else
        echo "  HuggingFace credential: NOT resolved (continuing)"
        warn_gated_backbone
    fi
elif [ -z "$ARCHIVE_SETTINGS_PRESENT" ] && [ -z "${HF_TOKEN:-}" ]; then
    echo "  HuggingFace credential: none (HF_TOKEN and HF_TOKEN_SECRET_ID are both unset)"
    warn_gated_backbone
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
export TUNE_VISUAL
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

# Keep checkpoint metadata portable. Isaac Sim rewrites this canonical identifier to its
# participant-local Reason2 directory immediately before offline policy loading.
if [ -n "$ARCHIVE_SETTINGS_PRESENT" ]; then
    python3 - "$OUTPUT_DIR" "$REASON2_LOCAL_DIR" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
local_reason = sys.argv[2]
canonical = "nvidia/Cosmos-Reason2-2B"

def rewrite(value):
    if isinstance(value, dict):
        return {key: rewrite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [rewrite(item) for item in value]
    if isinstance(value, str):
        return value.replace(local_reason, canonical)
    return value

for path in root.rglob("*.json"):
    try:
        original = json.loads(path.read_text())
    except (UnicodeDecodeError, json.JSONDecodeError):
        continue
    updated = rewrite(original)
    if updated != original:
        path.write_text(json.dumps(updated, indent=2) + "\n")
PY
fi

# --- Upload checkpoints to S3 ---
if [ -n "${CHECKPOINT_BUCKET:-}" ]; then
    S3_PREFIX="${CHECKPOINT_PREFIX:-checkpoints/groot-n17/${JOB_ID}}"
    echo "Uploading checkpoints to s3://${CHECKPOINT_BUCKET}/${S3_PREFIX}/ ..."
    aws s3 sync "$OUTPUT_DIR" "s3://${CHECKPOINT_BUCKET}/${S3_PREFIX}/" --region "${AWS_DEFAULT_REGION:-us-east-2}"
    echo "Done. Checkpoints uploaded to s3://${CHECKPOINT_BUCKET}/${S3_PREFIX}/"
else
    echo "WARNING: CHECKPOINT_BUCKET not set. Checkpoints will be lost when container exits."
fi
