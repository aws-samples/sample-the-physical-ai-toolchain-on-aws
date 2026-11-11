#!/usr/bin/env bash
# Prepare the Isaac Sim closed-loop GR00T evaluation on the workshop workstation.
#
# Finds a trained GR00T checkpoint in the participant data bucket, downloads it with
# the Pick and Place dataset and the offline Cosmos Reason2 cache, validates them,
# and writes the values later lab steps use to ~/groot-eval/eval.env.
#
#   bash prepare_eval.sh                  newest checkpoint trained for 15 steps
#   bash prepare_eval.sh --steps 10000    newest checkpoint trained for 10,000 steps
#   bash prepare_eval.sh --job-id <ID>    one specific Batch job's checkpoint
#
# Every check prints one PASS or FAIL line. Reads from S3 only; it creates or changes
# no AWS resource.
set -uo pipefail

STACK_NAME="${STACK_NAME:-physical-ai-toolchain}"
AWS_REGION="${AWS_REGION:-us-east-2}"
EVAL_DIR="${EVAL_DIR:-$HOME/groot-eval}"
STEPS=15
JOB_ID=""
REASON2_REVISION="9ce19a195e423419c349abfc86fd07178b230561"

while [ $# -gt 0 ]; do
  case "$1" in
    --steps) STEPS="${2:-}"; shift 2 ;;
    --job-id) JOB_ID="${2:-}"; shift 2 ;;
    *) echo "usage: bash prepare_eval.sh [--steps N | --job-id ID]"; exit 2 ;;
  esac
done
case "$STEPS" in ''|*[!0-9]*) echo "FAIL - --steps must be a whole number"; exit 2 ;; esac

LOG="$EVAL_DIR/prepare_eval.log"
mkdir -p "$EVAL_DIR"/{ckpt,data,models,out,closedloop}
: > "$LOG"
cd "$EVAL_DIR" || exit 1

pass() { echo "PASS - $*"; }
fail() {
  echo "FAIL - $*"
  echo "------ last lines of $LOG ------"; tail -n 10 "$LOG"; echo "--------------------------------"
  exit 1
}
quiet() { echo "\$ $*" >> "$LOG"; "$@" >> "$LOG" 2>&1; }
stack_output() {
  aws cloudformation describe-stacks --stack-name "$STACK_NAME" --region "$AWS_REGION" \
    --query "Stacks[0].Outputs[?OutputKey=='$1'].OutputValue | [0]" --output text 2>> "$LOG"
}

echo "GR00T closed-loop evaluation setup (stack $STACK_NAME, Region $AWS_REGION)"
echo

# 1. Stack outputs ----------------------------------------------------------------
IMAGE_URI=$(stack_output Gr00tTrainingImageUri)
DATA_BUCKET=$(stack_output DataBucketName)
REASON2_ARCHIVE_S3_URI=$(stack_output CosmosReason2ArchiveS3Uri)
REASON2_SHA256=$(stack_output CosmosReason2ArchiveSha256)
for pair in "Gr00tTrainingImageUri=$IMAGE_URI" "DataBucketName=$DATA_BUCKET" \
  "CosmosReason2ArchiveS3Uri=$REASON2_ARCHIVE_S3_URI" "CosmosReason2ArchiveSha256=$REASON2_SHA256"; do
  case "${pair#*=}" in ''|None) fail "stack output ${pair%%=*} is missing" ;; esac
done
pass "stack outputs read (data bucket $DATA_BUCKET)"

# 2. Choose the checkpoint --------------------------------------------------------
read_metadata() {  # read_metadata JOB_ID -> prints max_steps, or nothing
  aws s3 cp "s3://$DATA_BUCKET/checkpoints/groot-n17/$1/training_metadata.json" - \
    --region "$AWS_REGION" 2>> "$LOG" \
    | python3 -c 'import json,sys; print(json.load(sys.stdin).get("max_steps",""))' 2>> "$LOG"
}
if [ -z "$JOB_ID" ]; then
  candidates=$(aws s3api list-objects-v2 --bucket "$DATA_BUCKET" --prefix checkpoints/groot-n17/ \
    --region "$AWS_REGION" \
    --query "reverse(sort_by(Contents[?ends_with(Key, '/training_metadata.json') && !contains(Key, '/checkpoint-')], &LastModified))[].Key" \
    --output text 2>> "$LOG") || fail "could not list checkpoints in s3://$DATA_BUCKET/checkpoints/groot-n17/"
  for key in $candidates; do
    [ "$key" = None ] && continue
    id=$(echo "$key" | cut -d/ -f3)
    if [ "$(read_metadata "$id")" = "$STEPS" ]; then JOB_ID="$id"; break; fi
  done
  [ -n "$JOB_ID" ] || fail "no checkpoint trained for $STEPS steps in s3://$DATA_BUCKET/checkpoints/groot-n17/"
fi
STEPS_FOUND=$(read_metadata "$JOB_ID")
[ -n "$STEPS_FOUND" ] || fail "no training_metadata.json for job $JOB_ID"
CHECKPOINT_PREFIX="checkpoints/groot-n17/$JOB_ID"
pass "checkpoint $JOB_ID ($STEPS_FOUND training steps)"

# 3. Download the checkpoint into an empty folder ---------------------------------
# Every GR00T run writes shards of identical size, and `aws s3 sync` skips same-sized
# files, so a sync over an older ckpt/ would keep the old weights.
rm -rf ckpt.next
quiet aws s3 cp "s3://$DATA_BUCKET/$CHECKPOINT_PREFIX/" ckpt.next/ \
  --recursive --exclude 'checkpoint-*/*' --region "$AWS_REGION" --only-show-errors \
  || fail "checkpoint download failed"
[ -s ckpt.next/training_metadata.json ] || fail "downloaded checkpoint has no training_metadata.json"
rm -rf ckpt && mv ckpt.next ckpt
pass "checkpoint downloaded ($(du -sh ckpt | cut -f1))"

# 4. Dataset ----------------------------------------------------------------------
quiet aws s3 sync "s3://$DATA_BUCKET/teleoperation-data/pick_place_data/" data/ \
  --region "$AWS_REGION" --only-show-errors || fail "dataset download failed"
pass "dataset downloaded ($(du -sh data | cut -f1))"

# 5. Offline Cosmos Reason2 cache -------------------------------------------------
REASON2_ARCHIVE="models/cosmos-reason2-2b.tar"
REASON2_CACHE="hf-cache/hub/models--nvidia--Cosmos-Reason2-2B"
REASON2_NEXT="${REASON2_CACHE}.next"
REASON2_SNAPSHOT="${REASON2_CACHE}/snapshots/${REASON2_REVISION}"
quiet aws s3 cp "$REASON2_ARCHIVE_S3_URI" "$REASON2_ARCHIVE" --region "$AWS_REGION" --only-show-errors \
  || fail "Reason2 archive download failed"
printf '%s  %s\n' "$REASON2_SHA256" "$REASON2_ARCHIVE" | quiet sha256sum -c - \
  || fail "Reason2 archive checksum mismatch"
rm -rf "$REASON2_NEXT"
mkdir -p "$REASON2_NEXT/snapshots/$REASON2_REVISION" "$REASON2_NEXT/refs"
quiet tar -xf "$REASON2_ARCHIVE" -C "$REASON2_NEXT/snapshots/$REASON2_REVISION" \
  || fail "Reason2 archive could not be unpacked"
printf '%s' "$REASON2_REVISION" > "$REASON2_NEXT/refs/main"
rm -rf "$REASON2_CACHE" && mv "$REASON2_NEXT" "$REASON2_CACHE"
ln -sfn "../$REASON2_SNAPSHOT" models/Cosmos-Reason2-2B
rm -f "$REASON2_ARCHIVE"
pass "Reason2 cache installed (checksum verified)"

# 6. Validate everything ----------------------------------------------------------
python3 - "$EVAL_DIR" "$REASON2_REVISION" <<'PY' || fail "validation failed; do not continue to policy loading"
import sys
from pathlib import Path

root, revision = Path(sys.argv[1]), sys.argv[2]
ckpt, data = root / "ckpt", root / "data"
cache = root / "hf-cache/hub/models--nvidia--Cosmos-Reason2-2B"
snapshot = cache / "snapshots" / revision
json_text = "\n".join(p.read_text(errors="ignore") for p in ckpt.rglob("*.json"))
checks = {
    "checkpoint config": (ckpt / "config.json").is_file(),
    "model shards": bool(list(ckpt.glob("*.safetensors")) + list(ckpt.glob("*.bin"))),
    "processor": (ckpt / "processor").is_dir() and any((ckpt / "processor").iterdir()),
    "training metadata": (ckpt / "training_metadata.json").is_file(),
    "dataset metadata": (data / "meta/info.json").is_file(),
    "episode actions": (data / "data/chunk-000/episode_000000.parquet").is_file(),
    "episode video": (data / "videos/chunk-000/observation.images.cam_high/episode_000000.mp4").is_file(),
    "Reason2 weights": (snapshot / "model.safetensors").is_file(),
    "40-byte main ref": (cache / "refs/main").read_bytes() == revision.encode(),
    "canonical Reason2 identity": "nvidia/Cosmos-Reason2-2B" in json_text,
}
for name, ok in checks.items():
    print(f"{'PASS' if ok else 'FAIL'} - {name}")
sys.exit(0 if all(checks.values()) else 1)
PY

# 7. GR00T image and robot start pose ---------------------------------------------
# Bootstrap normally pre-stages both (prestage_eval.sh). If it has not finished, pull
# the image and export the start pose here so the lab still works.
ECR_REGISTRY="${IMAGE_URI%%/*}"
ECR_REGION=$(echo "$ECR_REGISTRY" | cut -d. -f4)
aws ecr get-login-password --region "$ECR_REGION" \
  | docker login --username AWS --password-stdin "$ECR_REGISTRY" >> "$LOG" 2>&1 \
  || fail "ECR login for the GR00T image failed"
if docker image inspect "$IMAGE_URI" >/dev/null 2>&1; then
  pass "GR00T image already on the workstation"
else
  echo "Pulling the GR00T image (about 12 minutes on a cold workstation)..."
  quiet docker pull --quiet "$IMAGE_URI" || fail "GR00T image pull failed"
  pass "GR00T image pulled"
fi
if [ -s out/trace_ep0.json ]; then
  pass "robot start pose already exported (out/trace_ep0.json)"
else
  quiet docker run --rm -v "$EVAL_DIR:/eval" -v "$(cd "$(dirname "$0")" && pwd):/code:ro" \
    --entrypoint /workspace/gr00t-repo/.venv/bin/python \
    "$IMAGE_URI" /code/export_trace.py --episode 0 --no-policy \
    || fail "robot start pose export failed"
  [ -s out/trace_ep0.json ] || fail "out/trace_ep0.json was not written"
  pass "robot start pose exported (out/trace_ep0.json)"
fi

printf 'export AWS_REGION=%q\nexport IMAGE_URI=%q\nexport DATA_BUCKET=%q\nexport REASON2_ARCHIVE_S3_URI=%q\nexport REASON2_SHA256=%q\nexport JOB_ID=%q\nexport CHECKPOINT_PREFIX=%q\nexport CHECKPOINT_STEPS=%q\n' \
  "$AWS_REGION" "$IMAGE_URI" "$DATA_BUCKET" "$REASON2_ARCHIVE_S3_URI" "$REASON2_SHA256" \
  "$JOB_ID" "$CHECKPOINT_PREFIX" "$STEPS_FOUND" > "$EVAL_DIR/eval.env"

echo
echo "PASS - ready to evaluate the $STEPS_FOUND-step checkpoint. Values saved to $EVAL_DIR/eval.env"
