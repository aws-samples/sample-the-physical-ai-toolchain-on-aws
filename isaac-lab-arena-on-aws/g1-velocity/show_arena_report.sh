#!/usr/bin/env bash
# Download the Arena RL evaluation report of one Batch job and open it in Firefox.
#
#   bash show_arena_report.sh <arena-g1-velocity Batch job ID>
#
# Copies s3://<data bucket>/arena-results/<job ID>/ to ~/arena-rl/<job ID>/, prints the
# per-run summary and opens index.html. Read-only on AWS.
set -uo pipefail

JOB_ID="${1:-}"
STACK_NAME="${STACK_NAME:-physical-ai-toolchain}"
AWS_REGION="${AWS_REGION:-us-east-2}"
[ -n "$JOB_ID" ] || { echo "usage: bash show_arena_report.sh <Batch job ID>"; exit 2; }

DATA_BUCKET=$(aws cloudformation describe-stacks --stack-name "$STACK_NAME" --region "$AWS_REGION" \
  --query "Stacks[0].Outputs[?OutputKey=='DataBucketName'].OutputValue | [0]" --output text)
DEST="$HOME/arena-rl/$JOB_ID"
aws s3 sync "s3://$DATA_BUCKET/arena-results/$JOB_ID/" "$DEST/" --region "$AWS_REGION" --only-show-errors \
  || { echo "FAIL - could not download s3://$DATA_BUCKET/arena-results/$JOB_ID/"; exit 1; }
[ -s "$DEST/index.html" ] || { echo "FAIL - no report for job $JOB_ID; did the job succeed?"; exit 1; }

echo "PASS - report downloaded to $DEST"
echo
column -s, -t < "$DEST/summary.csv"
echo
# The workstation image has no web browser; install Firefox once (about a minute).
command -v firefox >/dev/null || sudo snap install firefox >/dev/null
if command -v firefox >/dev/null; then
  (DISPLAY="${DISPLAY:-:0}" firefox --new-window "file://$DEST/index.html" >/dev/null 2>&1 &)
  echo "Opened $DEST/index.html in Firefox"
else
  echo "Open file://$DEST/index.html in a browser on the DCV desktop"
fi
