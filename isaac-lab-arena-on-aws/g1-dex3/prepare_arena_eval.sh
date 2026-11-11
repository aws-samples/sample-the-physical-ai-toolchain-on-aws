#!/usr/bin/env bash
# Prepare the Arena GR00T evaluation from the workshop workstation.
#
# Finds a trained g1_dex3 GR00T checkpoint in the participant data bucket, checks
# that the arena-g1-dex3 evaluation image exists, and prints the values the lab
# enters in the AWS Batch job definition.
#
#   bash prepare_arena_eval.sh                  newest checkpoint trained for 15 steps
#   bash prepare_arena_eval.sh --steps 2000     newest checkpoint trained for 2,000 steps
#   bash prepare_arena_eval.sh --job-id <ID>    one specific Batch job's checkpoint
#
# Every check prints one PASS or FAIL line. Read-only: it creates or changes no AWS
# resource.
set -uo pipefail

STACK_NAME="${STACK_NAME:-physical-ai-toolchain}"
AWS_REGION="${AWS_REGION:-us-east-2}"
STEPS=15
JOB_ID=""

while [ $# -gt 0 ]; do
  case "$1" in
    --steps) STEPS="${2:-}"; shift 2 ;;
    --job-id) JOB_ID="${2:-}"; shift 2 ;;
    *) echo "usage: bash prepare_arena_eval.sh [--steps N | --job-id ID]"; exit 2 ;;
  esac
done
case "$STEPS" in ''|*[!0-9]*) echo "FAIL - --steps must be a whole number"; exit 2 ;; esac

fail() { echo "FAIL - $*"; exit 1; }
stack_output() {
  aws cloudformation describe-stacks --stack-name "$STACK_NAME" --region "$AWS_REGION" \
    --query "Stacks[0].Outputs[?OutputKey=='$1'].OutputValue | [0]" --output text 2>/dev/null
}
metadata() {  # metadata JOB_ID FIELD -> prints the field from training_metadata.json
  aws s3 cp "s3://$DATA_BUCKET/checkpoints/groot-n17/$1/training_metadata.json" - \
    --region "$AWS_REGION" 2>/dev/null \
    | python3 -c "import json,sys; print(json.load(sys.stdin).get('$2',''))" 2>/dev/null
}

# 1. Stack outputs ------------------------------------------------------------------
DATA_BUCKET=$(stack_output DataBucketName)
REASON2_URI=$(stack_output CosmosReason2ArchiveS3Uri)
REASON2_SHA256=$(stack_output CosmosReason2ArchiveSha256)
for pair in "DataBucketName=$DATA_BUCKET" "CosmosReason2ArchiveS3Uri=$REASON2_URI" \
  "CosmosReason2ArchiveSha256=$REASON2_SHA256"; do
  case "${pair#*=}" in ''|None) fail "stack output ${pair%%=*} is missing" ;; esac
done
REASON2_KEY="${REASON2_URI#s3://"$DATA_BUCKET"/}"
[ "$REASON2_KEY" != "$REASON2_URI" ] || fail "Reason2 archive is not in the data bucket ($REASON2_URI)"
echo "PASS - stack outputs read (data bucket $DATA_BUCKET)"

# 2. Checkpoint ---------------------------------------------------------------------
if [ -z "$JOB_ID" ]; then
  candidates=$(aws s3api list-objects-v2 --bucket "$DATA_BUCKET" --prefix checkpoints/groot-n17/ \
    --region "$AWS_REGION" \
    --query "reverse(sort_by(Contents[?ends_with(Key, '/training_metadata.json') && !contains(Key, '/checkpoint-')], &LastModified))[].Key" \
    --output text 2>/dev/null) || fail "could not list s3://$DATA_BUCKET/checkpoints/groot-n17/"
  for key in $candidates; do
    [ "$key" = None ] && continue
    id=$(echo "$key" | cut -d/ -f3)
    if [ "$(metadata "$id" max_steps)" = "$STEPS" ]; then JOB_ID="$id"; break; fi
  done
  [ -n "$JOB_ID" ] || fail "no checkpoint trained for $STEPS steps; finish Isaac GR00T on AWS first"
fi
STEPS_FOUND=$(metadata "$JOB_ID" max_steps)
[ -n "$STEPS_FOUND" ] || fail "no training_metadata.json for job $JOB_ID"
[ "$(metadata "$JOB_ID" embodiment_config)" = g1_dex3 ] \
  || fail "checkpoint $JOB_ID was not trained with EMBODIMENT_CONFIG=g1_dex3"
CHECKPOINT_PREFIX="checkpoints/groot-n17/$JOB_ID"
echo "PASS - checkpoint $JOB_ID ($STEPS_FOUND training steps, g1_dex3)"

# 3. Evaluation image ---------------------------------------------------------------
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
IMAGE="$ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/physical-ai/isaac-lab:arena-g1-dex3"
aws ecr describe-images --region "$AWS_REGION" --repository-name physical-ai/isaac-lab \
  --image-ids imageTag=arena-g1-dex3 >/dev/null 2>&1 \
  || fail "image arena-g1-dex3 not found; check the physical-ai-toolchain-isaac-lab-image build"
echo "PASS - evaluation image arena-g1-dex3 exists"

echo
echo "Values for the arena-g1-dex3 job definition:"
echo "  Image:             $IMAGE"
echo "  DATA_BUCKET:       $DATA_BUCKET"
echo "  CHECKPOINT_PREFIX: $CHECKPOINT_PREFIX"
echo "  REASON2_KEY:       $REASON2_KEY"
echo "  REASON2_SHA256:    $REASON2_SHA256"
