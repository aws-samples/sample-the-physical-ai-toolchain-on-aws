#!/bin/bash
# =============================================================================
# Isaac Lab 3 AWS Batch Multi-Node Parallel Training Entrypoint
#
# Translates AWS Batch MNP environment variables to torchrun. The validated
# workshop path is one g6e.4xlarge, one L40S GPU, and the built-in G1 task.
#
# Environment variables (set by AWS Batch MNP):
#   AWS_BATCH_JOB_NUM_NODES                - Total number of nodes
#   AWS_BATCH_JOB_NODE_INDEX               - This node's index (0-based)
#   AWS_BATCH_JOB_MAIN_NODE_PRIVATE_IPV4_ADDRESS - Main node IP for rendezvous
#
# Environment variables (set by the job definition or node overrides):
#   TASK              - Isaac Lab task (default: Isaac-Velocity-Flat-G1-v0)
#   NUM_ENVS          - Parallel environments (default: 4096)
#   MAX_ITERATIONS    - Training iterations (default: 100)
#   PROC_PER_NODE     - Processes/GPUs per node (default: 1)
#   FRAMEWORK         - RL framework: skrl | rsl_rl | rl_games (default: rsl_rl)
#   CHECKPOINT_BUCKET - S3 bucket for checkpoints (default: none)
#   CHECKPOINT_PREFIX - S3 key prefix (default: checkpoints/<job_id>)
#
# Single-node, single-GPU G1 training is validated on g6e.4xlarge. Multi-node
# NCCL convergence remains unvalidated on hardware.
# =============================================================================

set -euo pipefail

ISAAC_LAB_ROOT="/workspace/isaaclab"
ISAAC_LAB_PYTHON="$ISAAC_LAB_ROOT/_isaac_sim/python.sh"

# --- Read AWS Batch MNP topology ---
NNODES=${AWS_BATCH_JOB_NUM_NODES:-1}
NODE_RANK=${AWS_BATCH_JOB_NODE_INDEX:-0}
MAIN_IP=${AWS_BATCH_JOB_MAIN_NODE_PRIVATE_IPV4_ADDRESS:-localhost}

# --- Read runtime configuration ---
NPROC=${PROC_PER_NODE:-1}
TASK="${TASK:-Isaac-Velocity-Flat-G1-v0}"
NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERATIONS="${MAX_ITERATIONS:-100}"
FRAMEWORK="${FRAMEWORK:-rsl_rl}"

# --- Print the effective request ---
echo "============================================================"
echo "  Isaac Lab 3 RL Training (AWS Batch)"
echo "  Main node IP:   $MAIN_IP"
echo "  Nodes:          $NNODES"
echo "  Node rank:      $NODE_RANK"
echo "  GPUs per node:  $NPROC"
echo "  Task:           $TASK"
echo "  Num envs:       $NUM_ENVS"
echo "  Max iterations: $MAX_ITERATIONS"
echo "  Framework:      $FRAMEWORK"
echo "============================================================"

export ACCEPT_EULA=Y
export PRIVACY_CONSENT=Y
export OMNI_ENV_PRIVACY_CONSENT=Y

# --- Determine the training script ---
case "$FRAMEWORK" in
    skrl)
        TRAIN_SCRIPT="$ISAAC_LAB_ROOT/scripts/reinforcement_learning/skrl/train.py"
        ;;
    rsl_rl)
        TRAIN_SCRIPT="$ISAAC_LAB_ROOT/scripts/reinforcement_learning/rsl_rl/train.py"
        ;;
    rl_games)
        TRAIN_SCRIPT="$ISAAC_LAB_ROOT/scripts/reinforcement_learning/rl_games/train.py"
        ;;
    *)
        echo "ERROR: Unknown framework: $FRAMEWORK"
        exit 1
        ;;
esac

# --- Persist checkpoints outside the container ---
JOB_ID=${AWS_BATCH_JOB_ID:-local}
export OUTPUT_DIR="/tmp/checkpoints/${JOB_ID}"
mkdir -p "$OUTPUT_DIR"

cd "$ISAAC_LAB_ROOT"
mkdir -p logs
FRAMEWORK_LOG_DIR="logs/${FRAMEWORK}"
if [ ! -L "$FRAMEWORK_LOG_DIR" ]; then
    mkdir -p "$OUTPUT_DIR/$FRAMEWORK"
    ln -sf "$OUTPUT_DIR/$FRAMEWORK" "$FRAMEWORK_LOG_DIR"
    echo "Linked $FRAMEWORK_LOG_DIR -> $OUTPUT_DIR/$FRAMEWORK"
fi

TRAIN_ARGS=(
    "$TRAIN_SCRIPT"
    "--task=$TASK"
    "--num_envs=$NUM_ENVS"
    "--max_iterations=$MAX_ITERATIONS"
    "--headless"
)

# Match the DCV-validated path for the workshop's single-GPU job. Keep torchrun
# only for explicitly configured multi-process or multi-node jobs.
if [ "$NNODES" -eq 1 ] && [ "$NPROC" -eq 1 ]; then
    echo "Launching validated single-GPU Isaac Lab 3 training..."
    "$ISAAC_LAB_ROOT/isaaclab.sh" -p "${TRAIN_ARGS[@]}"
else
    echo "Launching distributed training via torchrun ($NNODES node(s), $NPROC process(es) per node)..."
    TRAIN_ARGS+=("--distributed")
    "$ISAAC_LAB_PYTHON" -m torch.distributed.run \
        --nproc_per_node="$NPROC" \
        --nnodes="$NNODES" \
        --node_rank="$NODE_RANK" \
        --rdzv_id=isaaclab-batch-job \
        --rdzv_backend=c10d \
        --rdzv_endpoint="$MAIN_IP:29500" \
        "${TRAIN_ARGS[@]}"
fi

echo "Training complete. Checkpoints saved to $OUTPUT_DIR"

# --- Upload checkpoints to S3 (the image includes boto3; AWS CLI is not required) ---
if [ -n "${CHECKPOINT_BUCKET:-}" ]; then
    S3_PREFIX="${CHECKPOINT_PREFIX:-checkpoints/${JOB_ID}}"
    echo "Uploading checkpoints to s3://${CHECKPOINT_BUCKET}/${S3_PREFIX}/ ..."
    python3 -c "
import os
import pathlib
import boto3

bucket = os.environ['CHECKPOINT_BUCKET']
prefix = os.environ.get('CHECKPOINT_PREFIX', f'checkpoints/{os.environ.get(\"AWS_BATCH_JOB_ID\", \"local\")}')
s3 = boto3.client('s3')
uploaded = 0
seen_files = set()

search_dirs = [
    os.environ.get('OUTPUT_DIR', '/tmp/checkpoints'),
    '/workspace/isaaclab/logs',
    '/efs/models',
]

for search_dir in search_dirs:
    base = pathlib.Path(search_dir)
    if not base.exists():
        continue
    for file_path in base.rglob('*'):
        if not file_path.is_file():
            continue
        resolved = file_path.resolve()
        if resolved in seen_files:
            continue
        seen_files.add(resolved)
        key = f'{prefix}/{file_path.relative_to(base)}'
        print(f'  Uploading {file_path} -> s3://{bucket}/{key}')
        s3.upload_file(str(file_path), bucket, key)
        uploaded += 1

if uploaded == 0:
    raise RuntimeError('No Isaac Lab training artifacts were found to upload')

print(f'Done. Uploaded {uploaded} files to s3://{bucket}/{prefix}/')
"
else
    echo "WARNING: CHECKPOINT_BUCKET not set. Checkpoints will be lost when the container exits."
fi
