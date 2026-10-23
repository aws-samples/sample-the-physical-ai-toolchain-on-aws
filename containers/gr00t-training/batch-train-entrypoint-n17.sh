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
#   HF_TOKEN            - HuggingFace token for model download (optional)
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

    # GR00T uses config.model_name as a class discriminator, so it must remain the
    # canonical repository id even in offline mode. Seed the verified raw snapshot
    # into Hugging Face's cache instead of rewriting model_name to a filesystem path.
    REASON2_REVISION="9ce19a195e423419c349abfc86fd07178b230561"
    export HF_HOME="${LOCAL_MODEL_ROOT}/hf-cache"
    export HF_HUB_CACHE="${HF_HOME}/hub"
    REASON2_CACHE_DIR="${HF_HUB_CACHE}/models--nvidia--Cosmos-Reason2-2B"
    REASON2_SNAPSHOT_DIR="${REASON2_CACHE_DIR}/snapshots/${REASON2_REVISION}"
    mkdir -p "${REASON2_CACHE_DIR}/snapshots" "${REASON2_CACHE_DIR}/refs"
    rm -rf "$REASON2_SNAPSHOT_DIR"
    mv "$REASON2_LOCAL_DIR" "$REASON2_SNAPSHOT_DIR"
    # huggingface_hub treats every byte as part of the ref. Do not append a newline.
    printf '%s' "$REASON2_REVISION" > "${REASON2_CACHE_DIR}/refs/main"
    [ "$(wc -c < "${REASON2_CACHE_DIR}/refs/main")" -eq 40 ] \
      || { echo "ERROR: Cosmos Reason2 refs/main is not exactly 40 bytes" >&2; exit 1; }
    [ -s "$REASON2_SNAPSHOT_DIR/config.json" ] \
      && [ -s "$REASON2_SNAPSHOT_DIR/model.safetensors" ] \
      || { echo "ERROR: Cosmos Reason2 cache snapshot is incomplete" >&2; exit 1; }
    REASON2_LOCAL_DIR="$REASON2_SNAPSHOT_DIR"

    # Transformers' tokenizer performs a model_info() request when it receives a
    # repository id, even when every processor file is cached and offline mode is
    # enabled. GR00T still needs the canonical id for get_backbone_cls(), so patch
    # only its processor factory to pass the verified local snapshot directory.
    PROCESSOR_SOURCE="${GROOT_SDK_DIR:-/workspace/gr00t-repo}/gr00t/model/gr00t_n1d7/processing_gr00t_n1d7.py"
    python3 - "$PROCESSOR_SOURCE" "$REASON2_LOCAL_DIR" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
local_snapshot = sys.argv[2]
source = path.read_text()
needle = "return Qwen3VLProcessor.from_pretrained(model_name, **transformers_loading_kwargs)"
replacement = (
    "return Qwen3VLProcessor.from_pretrained("
    f"{local_snapshot!r}, **transformers_loading_kwargs)"
)
if source.count(needle) != 1:
    raise SystemExit(
        f"Expected exactly one GR00T processor load site, found {source.count(needle)}"
    )
path.write_text(source.replace(needle, replacement))
print(f"Redirected Qwen3-VL processor loading to {local_snapshot}")
PY

    python3 - "$GROOT_LOCAL_DIR" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
canonical = "nvidia/Cosmos-Reason2-2B"

# At least one GR00T config must retain the canonical model name. The pinned
# loader uses this value to select the Cosmos/Qwen3-VL backbone class before
# Hugging Face resolves the same id from the local cache.
for path in root.rglob("*.json"):
    try:
        value = json.loads(path.read_text())
    except (UnicodeDecodeError, json.JSONDecodeError):
        continue
    if canonical in json.dumps(value):
        print(f"Preserved canonical Cosmos Reason2 identity in {path}")
        break
else:
    raise SystemExit("GR00T snapshot has no canonical Cosmos Reason2 identity")
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
if [ -z "$ARCHIVE_SETTINGS_PRESENT" ] && [ -z "${HF_TOKEN:-}" ] && [ -n "$SECRET_ID" ]; then
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
