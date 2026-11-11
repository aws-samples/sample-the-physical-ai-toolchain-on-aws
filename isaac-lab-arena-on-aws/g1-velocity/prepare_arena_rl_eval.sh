#!/usr/bin/env bash
# Prepare the Arena RL evaluation from the workshop workstation.
#
# Finds the newest Isaac Lab G1 velocity checkpoint (model_<N>.pt with params/agent.yaml)
# in the participant data bucket, checks that the arena-g1-velocity evaluation image
# exists, and prints the values the lab enters in the AWS Batch job definition.
#
#   bash prepare_arena_rl_eval.sh                 newest Isaac Lab training job
#   bash prepare_arena_rl_eval.sh --job-id <ID>   one specific training job
#
# Every check prints one PASS or FAIL line. Read-only: it creates or changes no AWS resource.
set -uo pipefail

STACK_NAME="${STACK_NAME:-physical-ai-toolchain}"
AWS_REGION="${AWS_REGION:-us-east-2}"
JOB_ID=""
while [ $# -gt 0 ]; do
  case "$1" in
    --job-id) JOB_ID="${2:-}"; shift 2 ;;
    *) echo "usage: bash prepare_arena_rl_eval.sh [--job-id ID]"; exit 2 ;;
  esac
done

fail() { echo "FAIL - $*"; exit 1; }
stack_output() {
  aws cloudformation describe-stacks --stack-name "$STACK_NAME" --region "$AWS_REGION" \
    --query "Stacks[0].Outputs[?OutputKey=='$1'].OutputValue | [0]" --output text 2>/dev/null
}

# 1. Data bucket ----------------------------------------------------------------------
DATA_BUCKET=$(stack_output DataBucketName)
case "$DATA_BUCKET" in ''|None) fail "stack output DataBucketName is missing" ;; esac
echo "PASS - stack outputs read (data bucket $DATA_BUCKET)"

# 2. Checkpoint -----------------------------------------------------------------------
# Isaac Lab writes checkpoints/<job ID>/rsl_rl/g1_flat/<run>/model_<N>.pt plus params/.
SEARCH="checkpoints/${JOB_ID:+$JOB_ID/}"
KEYS=$(aws s3api list-objects-v2 --bucket "$DATA_BUCKET" --prefix "$SEARCH" --region "$AWS_REGION" \
  --query "reverse(sort_by(Contents[?contains(Key, '/rsl_rl/g1_flat/') && ends_with(Key, '/params/agent.yaml')], &LastModified))[].Key" \
  --output text 2>/dev/null) || fail "could not list s3://$DATA_BUCKET/$SEARCH"
AGENT_KEY=$(echo "$KEYS" | tr '\t' '\n' | grep -v '^None$' | head -1)
[ -n "$AGENT_KEY" ] || fail "no Isaac Lab G1 checkpoint with params/agent.yaml under s3://$DATA_BUCKET/$SEARCH; finish Isaac Lab on AWS first"
TRAIN_JOB_ID=$(echo "$AGENT_KEY" | cut -d/ -f2)
RUN_PREFIX="${AGENT_KEY%/params/agent.yaml}"
MODEL=$(aws s3api list-objects-v2 --bucket "$DATA_BUCKET" --prefix "$RUN_PREFIX/model_" --region "$AWS_REGION" \
  --query 'Contents[].Key' --output text 2>/dev/null | tr '\t' '\n' | grep -E '/model_[0-9]+\.pt$' | sort -V | tail -1)
[ -n "$MODEL" ] || fail "no model_<N>.pt next to $AGENT_KEY"
echo "PASS - checkpoint ${MODEL##*/} from training job $TRAIN_JOB_ID (params/agent.yaml present)"
CHECKPOINT_PREFIX="checkpoints/$TRAIN_JOB_ID"

# 3. Evaluation image -----------------------------------------------------------------
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
IMAGE="$ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/physical-ai/isaac-lab:arena-g1-velocity"
aws ecr describe-images --region "$AWS_REGION" --repository-name physical-ai/isaac-lab \
  --image-ids imageTag=arena-g1-velocity >/dev/null 2>&1 \
  || fail "image arena-g1-velocity not found; check the physical-ai-toolchain-isaac-lab-image build"
echo "PASS - evaluation image arena-g1-velocity exists"

echo
echo "Values for the arena-g1-velocity job definition:"
echo "  Image:             $IMAGE"
echo "  DATA_BUCKET:       $DATA_BUCKET"
echo "  CHECKPOINT_PREFIX: $CHECKPOINT_PREFIX"
