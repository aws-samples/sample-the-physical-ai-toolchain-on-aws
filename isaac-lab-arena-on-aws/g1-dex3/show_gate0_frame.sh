#!/usr/bin/env bash
# Print the key lines of an arena-g1-dex3 gate 0 job and decode its first cam_high frame.
#
#   bash show_gate0_frame.sh <gate 0 Batch job ID>
#
# Writes ~/arena-g1/gate0.log and ~/arena-g1/cam_high_start.png. Read-only on AWS.
set -uo pipefail

JOB_ID="${1:-}"
AWS_REGION="${AWS_REGION:-us-east-2}"
OUT_DIR="${OUT_DIR:-$HOME/arena-g1}"
[ -n "$JOB_ID" ] || { echo "usage: bash show_gate0_frame.sh <gate 0 Batch job ID>"; exit 2; }

mkdir -p "$OUT_DIR"
STREAM=$(aws batch describe-jobs --region "$AWS_REGION" --jobs "$JOB_ID" \
  --query 'jobs[0].container.logStreamName' --output text)
case "$STREAM" in ''|None) echo "FAIL - job $JOB_ID has no log stream yet; wait until it is RUNNING"; exit 1 ;; esac

aws logs get-log-events --region "$AWS_REGION" --log-group-name /aws/batch/job \
  --log-stream-name "$STREAM" --start-from-head --query 'events[].message' --output text \
  | tr '\t' '\n' > "$OUT_DIR/gate0.log"
grep -E 'G1EVAL_CKPT|BRIDGE_READY|G1DEX3_ENV|G1DEX3_POLICY|Metrics|G1EVAL_EXIT' "$OUT_DIR/gate0.log"

if grep -m1 FRAME_PNG_B64 "$OUT_DIR/gate0.log" | sed 's/^.*FRAME_PNG_B64 //' \
     | base64 -d > "$OUT_DIR/cam_high_start.png" 2>/dev/null && [ -s "$OUT_DIR/cam_high_start.png" ]; then
  echo "PASS - frame written to $OUT_DIR/cam_high_start.png"
else
  echo "FAIL - no FRAME_PNG_B64 line in the log; was the job submitted with MODE=gate0?"
  exit 1
fi
