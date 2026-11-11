#!/usr/bin/env bash
# Pre-stage the Isaac Sim closed-loop GR00T evaluation during workstation bootstrap.
#
# Runs once in the background from the workstation user data, so attendees do not
# wait for it in the lab:
#   1. waits for the stack outputs (they appear when the stack reaches CREATE_COMPLETE)
#   2. pulls the GR00T image (~66 GiB), retrying until the image build has pushed it
#   3. syncs the Pick and Place dataset into ~/groot-eval/data
#   4. exports the robot's start pose to ~/groot-eval/out/trace_ep0.json
#
#   sudo bash prestage_eval.sh <stack-name> <region>
#
# Reads from S3 and ECR only; it creates or changes no AWS resource. Safe to re-run:
# each step is skipped when its result is already present.
set -uo pipefail

STACK_NAME="${1:-${STACK_NAME:-physical-ai-toolchain}}"
AWS_REGION="${2:-${AWS_REGION:-us-east-2}}"
EVAL_USER="${EVAL_USER:-ubuntu}"
EVAL_DIR="${EVAL_DIR:-/home/$EVAL_USER/groot-eval}"
CODE_DIR="$(cd "$(dirname "$0")" && pwd)"
TRACE="$EVAL_DIR/out/trace_ep0.json"
WAIT_OUTPUTS_MIN="${WAIT_OUTPUTS_MIN:-240}"   # stack creation ceiling
WAIT_IMAGE_MIN="${WAIT_IMAGE_MIN:-300}"       # GR00T image build ceiling

log() { echo "$(date -u +%FT%TZ) $*"; }
stack_output() {
  aws cloudformation describe-stacks --stack-name "$STACK_NAME" --region "$AWS_REGION" \
    --query "Stacks[0].Outputs[?OutputKey=='$1'].OutputValue | [0]" --output text 2>/dev/null
}
finish() {
  chown -R "$EVAL_USER:$EVAL_USER" "$EVAL_DIR" 2>/dev/null || true
  log "$1"
  exit "${2:-0}"
}

mkdir -p "$EVAL_DIR"/{data,out}
if [ -s "$TRACE" ]; then finish "SKIP - $TRACE already exists"; fi

# 1. Stack outputs ------------------------------------------------------------------
IMAGE_URI=""; DATA_BUCKET=""
for _ in $(seq 1 "$WAIT_OUTPUTS_MIN"); do
  IMAGE_URI=$(stack_output Gr00tTrainingImageUri)
  DATA_BUCKET=$(stack_output DataBucketName)
  case "$IMAGE_URI$DATA_BUCKET" in *None*|'') sleep 60 ;; *) break ;; esac
done
case "$IMAGE_URI" in ''|None) finish "FAIL - stack output Gr00tTrainingImageUri not available" 1 ;; esac
case "$DATA_BUCKET" in ''|None) finish "FAIL - stack output DataBucketName not available" 1 ;; esac
log "PASS - stack outputs read (image $IMAGE_URI)"

# 2. GR00T image --------------------------------------------------------------------
# The image build usually finishes after the stack, so retry until the tag exists.
ECR_REGISTRY="${IMAGE_URI%%/*}"
ECR_REGION=$(echo "$ECR_REGISTRY" | cut -d. -f4)
pulled=0
for _ in $(seq 1 $((WAIT_IMAGE_MIN / 5))); do
  if aws ecr get-login-password --region "$ECR_REGION" \
       | docker login --username AWS --password-stdin "$ECR_REGISTRY" >/dev/null 2>&1 \
     && docker pull --quiet "$IMAGE_URI" >/dev/null 2>&1; then
    pulled=1; break
  fi
  sleep 300
done
[ "$pulled" = 1 ] || finish "FAIL - could not pull $IMAGE_URI within $WAIT_IMAGE_MIN minutes" 1
log "PASS - GR00T image pulled"

# 3. Dataset ------------------------------------------------------------------------
aws s3 sync "s3://$DATA_BUCKET/teleoperation-data/pick_place_data/" "$EVAL_DIR/data/" \
  --region "$AWS_REGION" --only-show-errors \
  || finish "FAIL - dataset download from s3://$DATA_BUCKET/teleoperation-data/pick_place_data/" 1
log "PASS - dataset downloaded ($(du -sh "$EVAL_DIR/data" | cut -f1))"

# 4. Start pose trace ---------------------------------------------------------------
# --no-policy reads only the recorded episode, so no checkpoint or GPU is needed.
docker run --rm -v "$EVAL_DIR:/eval" -v "$CODE_DIR:/code:ro" \
  --entrypoint /workspace/gr00t-repo/.venv/bin/python \
  "$IMAGE_URI" /code/export_trace.py --episode 0 --no-policy \
  || finish "FAIL - export_trace.py failed" 1
[ -s "$TRACE" ] || finish "FAIL - $TRACE was not written" 1
finish "PASS - start pose exported to $TRACE"
