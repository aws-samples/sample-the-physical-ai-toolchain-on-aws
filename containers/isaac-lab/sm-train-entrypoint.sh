#!/bin/bash
# =============================================================================
# Isaac Lab SageMaker Training Entrypoint
#
# Based on the official AWS pattern from:
# https://github.com/awslabs/awsome-distributed-ai/tree/main/3.test_cases/pytorch/nvidia-isaac-lab
#
# Parses SageMaker's resource config for multi-node topology, then launches
# Isaac Lab RL training via torchrun.
#
# Environment variables (set by SageMaker or hyperparameters):
#   TASK              - Isaac Lab task name (default: Isaac-Velocity-Flat-G1-v0)
#   NUM_ENVS          - Parallel environments (default: 4096)
#   MAX_ITERATIONS    - Training iterations (default: 100)
#   FRAMEWORK         - RL framework: skrl | rsl_rl | rl_games (default: rsl_rl)
# =============================================================================

set -euo pipefail

ISAAC_LAB_ROOT="/workspace/isaaclab"
ISAAC_LAB_PYTHON="$ISAAC_LAB_ROOT/_isaac_sim/python.sh"

# --- Read SageMaker resource config for multi-node setup ---
RESOURCE_CONFIG="/opt/ml/input/config/resourceconfig.json"

if [ -f "$RESOURCE_CONFIG" ]; then
    CURRENT_HOST=$(python3 -c "import json; c=json.load(open('$RESOURCE_CONFIG')); print(c['current_host'])")
    ALL_HOSTS=$(python3 -c "import json; c=json.load(open('$RESOURCE_CONFIG')); print(','.join(c['hosts']))")
    MASTER_HOST=$(python3 -c "import json; c=json.load(open('$RESOURCE_CONFIG')); print(c['hosts'][0])")
    NNODES=$(python3 -c "import json; c=json.load(open('$RESOURCE_CONFIG')); print(len(c['hosts']))")
    NODE_RANK=$(python3 -c "import json; c=json.load(open('$RESOURCE_CONFIG')); print(c['hosts'].index(c['current_host']))")
else
    CURRENT_HOST="localhost"
    MASTER_HOST="localhost"
    NNODES=1
    NODE_RANK=0
fi

# --- Count GPUs ---
NPROC=$(nvidia-smi -L | wc -l)

# --- Read hyperparameters ---
HYPERPARAMS="/opt/ml/input/config/hyperparameters.json"
if [ -f "$HYPERPARAMS" ]; then
    TASK=$(python3 -c "import json; print(json.load(open('$HYPERPARAMS')).get('task', 'Isaac-Velocity-Flat-G1-v0'))")
    NUM_ENVS=$(python3 -c "import json; print(json.load(open('$HYPERPARAMS')).get('num_envs', '4096'))")
    MAX_ITERATIONS=$(python3 -c "import json; print(json.load(open('$HYPERPARAMS')).get('max_iterations', '100'))")
    FRAMEWORK=$(python3 -c "import json; print(json.load(open('$HYPERPARAMS')).get('framework', 'rsl_rl'))")
    MODE=$(python3 -c "import json; print(json.load(open('$HYPERPARAMS')).get('mode', 'train'))")
    VIDEO_LENGTH=$(python3 -c "import json; print(json.load(open('$HYPERPARAMS')).get('video_length', '400'))")
else
    TASK="${TASK:-Isaac-Velocity-Flat-G1-v0}"
    NUM_ENVS="${NUM_ENVS:-4096}"
    MAX_ITERATIONS="${MAX_ITERATIONS:-100}"
    FRAMEWORK="${FRAMEWORK:-rsl_rl}"
    MODE="${MODE:-train}"
    VIDEO_LENGTH="${VIDEO_LENGTH:-400}"
fi

# --- Print info ---
echo "============================================================"
echo "  Isaac Lab RL Training (SageMaker)"
echo "  Host:           $CURRENT_HOST"
echo "  Master:         $MASTER_HOST"
echo "  Nodes:          $NNODES"
echo "  Node rank:      $NODE_RANK"
echo "  GPUs per node:  $NPROC"
echo "  Task:           $TASK"
echo "  Num envs:       $NUM_ENVS"
echo "  Max iterations: $MAX_ITERATIONS"
echo "  Framework:      $FRAMEWORK"
echo "  Mode:           $MODE"
echo "============================================================"

# --- Set Isaac Sim environment ---
export ACCEPT_EULA=Y
export PRIVACY_CONSENT=Y
export OMNI_ENV_PRIVACY_CONSENT=Y

# =============================================================================
# PLAY / VIDEO MODE — load a trained checkpoint, roll it out headless, and
# record an MP4. SageMaker mounts the trained model at /opt/ml/input/data/model/
# (the model.tar.gz from a prior training job is auto-extracted there). The
# recorded video is copied to /opt/ml/model/ so SageMaker uploads it to S3.
# =============================================================================
if [ "$MODE" = "play" ] || [ "$MODE" = "video" ]; then
    echo "=== PLAY MODE: rendering a video of the trained policy ==="
    MODEL_IN="/opt/ml/input/data/model"

    # SageMaker delivers the input channel as-is — a plain S3 input does NOT
    # auto-extract, so the checkpoint arrives inside model.tar.gz. Extract any
    # tarballs first, then locate the checkpoint.
    for tgz in $(find "$MODEL_IN" -name '*.tar.gz' 2>/dev/null); do
        echo "Extracting $tgz ..."
        tar -xzf "$tgz" -C "$MODEL_IN"
    done

    # Find the checkpoint (rsl_rl saves model_<iter>.pt under logs/.../<run>/).
    CKPT=$(find "$MODEL_IN" -name 'model_*.pt' 2>/dev/null | sort -t_ -k2 -n | tail -1)
    if [ -z "$CKPT" ]; then
        echo "ERROR: no model_*.pt checkpoint found under $MODEL_IN"
        echo "Pass the prior training job's model.tar.gz as the 'model' input channel."
        ls -R "$MODEL_IN" 2>/dev/null | head -40
        exit 1
    fi
    echo "Using checkpoint: $CKPT"

    cd "$ISAAC_LAB_ROOT"
    PLAY_SCRIPT="$ISAAC_LAB_ROOT/scripts/reinforcement_learning/${FRAMEWORK}/play.py"

    # Single env for a clean, watchable video. play.py writes the MP4 to
    # <checkpoint_dir>/videos/play/, which we then copy to the SM output dir.
    "$ISAAC_LAB_PYTHON" "$PLAY_SCRIPT" \
        --task="$TASK" \
        --num_envs=1 \
        --checkpoint="$CKPT" \
        --video \
        --video_length="$VIDEO_LENGTH" \
        --headless

    echo "Collecting rendered video(s) → /opt/ml/model/ ..."
    mkdir -p /opt/ml/model/videos
    find "$(dirname "$CKPT")" -name '*.mp4' -exec cp -v {} /opt/ml/model/videos/ \;
    # Also surface any videos written under the repo logs dir (path varies by version).
    find /workspace/isaaclab/logs -name '*.mp4' -exec cp -v {} /opt/ml/model/videos/ \; 2>/dev/null || true
    echo "Done. Video(s) in /opt/ml/model/videos/ (uploaded to S3 by SageMaker)."
    ls -la /opt/ml/model/videos/ 2>/dev/null
    exit 0
fi

# --- Determine training script based on framework ---
case $FRAMEWORK in
    skrl)
        TRAIN_SCRIPT="scripts/reinforcement_learning/skrl/train.py"
        ;;
    rsl_rl)
        TRAIN_SCRIPT="scripts/reinforcement_learning/rsl_rl/train.py"
        ;;
    rl_games)
        TRAIN_SCRIPT="scripts/reinforcement_learning/rl_games/train.py"
        ;;
    *)
        echo "ERROR: Unknown framework: $FRAMEWORK"
        exit 1
        ;;
esac

# --- Launch training via torchrun ---
cd "$ISAAC_LAB_ROOT"

# rsl_rl/skrl/rl_games only enable multi-GPU gradient sync when --distributed is
# passed. Without it, all torchrun procs collide on cuda:0 with no NCCL init.
# Single-GPU single-node runs (the validated path) don't need it.
DIST_FLAG=""
if [ "$NNODES" -gt 1 ] || [ "$NPROC" -gt 1 ]; then
    DIST_FLAG="--distributed"
fi

"$ISAAC_LAB_PYTHON" -m torch.distributed.run \
    --nproc_per_node=$NPROC \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --rdzv_id=isaaclab-sm-job \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_HOST:29500 \
    $TRAIN_SCRIPT \
    --task=$TASK \
    --num_envs=$NUM_ENVS \
    --max_iterations=$MAX_ITERATIONS \
    --headless \
    $DIST_FLAG

# --- Copy artifacts to SageMaker output ---
echo "Copying training artifacts to /opt/ml/model/..."
if [ -d "logs" ]; then
    cp -r logs /opt/ml/model/
fi

# Save metadata
python3 -c "
import json
meta = {
    'task': '$TASK',
    'num_envs': int('$NUM_ENVS'),
    'max_iterations': int('$MAX_ITERATIONS'),
    'framework': '$FRAMEWORK',
    'nodes': int('$NNODES'),
    'gpus_per_node': int('$NPROC'),
}
json.dump(meta, open('/opt/ml/model/training_metadata.json', 'w'), indent=2)
"

echo "Done. Artifacts saved to /opt/ml/model/"
