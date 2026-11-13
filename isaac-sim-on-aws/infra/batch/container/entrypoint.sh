#!/bin/bash
set -e

# Validate required environment variables
if [ -z "${S3_BUCKET}" ]; then
    echo "ERROR: S3_BUCKET environment variable is not set"
    exit 1
fi

JOB_INDEX=${AWS_BATCH_JOB_ARRAY_INDEX:-0}
# Validate JOB_INDEX is a non-negative integer before it is used in S3 paths
# and the run ID, closing off shell-metacharacter / path-traversal injection.
if ! [[ "${JOB_INDEX}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: JOB_INDEX is not a valid integer: ${JOB_INDEX}"
    exit 1
fi
RUN_ID="run-$(date -u +%Y%m%d-%H%M%S)-${JOB_INDEX}"
START_TIME=$(date +%s)

echo "=== Isaac Sim Batch Job ==="
echo "Run ID: ${RUN_ID}"
echo "S3 Bucket: ${S3_BUCKET}"

# Report job started
aws cloudwatch put-metric-data \
    --namespace "IsaacSim/Jobs" \
    --metric-name "JobStarted" \
    --value 1 \
    --unit Count 2>/dev/null || true

mkdir -p /tmp/results/${RUN_ID}

# Download config (job-specific or default)
aws s3 cp "s3://${S3_BUCKET}/configs/config_${JOB_INDEX}.yaml" /tmp/config.yaml \
    2>/dev/null || \
    aws s3 cp "s3://${S3_BUCKET}/configs/default.yaml" /tmp/config.yaml

# Run simulation using Isaac Sim's Python launcher
# Environment and robot are loaded from NVIDIA's bundled assets (no S3 download needed)
/isaac-sim/python.sh /app/simulation/run_simulation.py \
    --config /tmp/config.yaml \
    --output-dir /tmp/results/${RUN_ID}

EXIT_CODE=$?

# Upload results
aws s3 cp /tmp/results/${RUN_ID}/ "s3://${S3_BUCKET}/results/${RUN_ID}/" --recursive

# Report job completed with duration
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

if [ $EXIT_CODE -eq 0 ]; then
    aws cloudwatch put-metric-data \
        --namespace "IsaacSim/Jobs" \
        --metric-name "JobSucceeded" \
        --value 1 \
        --unit Count 2>/dev/null || true
else
    aws cloudwatch put-metric-data \
        --namespace "IsaacSim/Jobs" \
        --metric-name "JobFailed" \
        --value 1 \
        --unit Count 2>/dev/null || true
fi

aws cloudwatch put-metric-data \
    --namespace "IsaacSim/Jobs" \
    --metric-name "JobDuration" \
    --value ${DURATION} \
    --unit Seconds 2>/dev/null || true

echo "=== Done: ${RUN_ID} (${DURATION}s) ==="
exit ${EXIT_CODE}
