#!/usr/bin/env bash
# AWS Batch job: score the fine-tuned GR00T g1_dex3 checkpoint in the Arena g1_blue_box task.
#
# Runs inside physical-ai/isaac-lab:arena-g1-dex3. Required environment:
#   DATA_BUCKET        workshop data bucket
#   CHECKPOINT_PREFIX  checkpoints/groot-n17/<training Batch job ID>
#   REASON2_KEY        S3 key of the verified Cosmos-Reason2-2B archive
#   REASON2_SHA256     its SHA-256
# Optional: EPISODES (10), SEED (7), SAMPLES (4), REGION (us-east-2), MODE (eval | gate0 | record)
#   gate0  = 1 episode, two camera frames in the log
#   record = 1 episode, one camera frame per policy query in the log (for a GIF)
set -uo pipefail
REGION="${REGION:-us-east-2}"
EPISODES="${EPISODES:-10}"
SEED="${SEED:-7}"
SAMPLES="${SAMPLES:-4}"
MODE="${MODE:-eval}"
W=/tmp/g1eval
REV=9ce19a195e423419c349abfc86fd07178b230561
SNAP="$W/hf-cache/hub/models--nvidia--Cosmos-Reason2-2B/snapshots/$REV"

stop() { echo "G1EVAL_FAIL $*"; exit 1; }
echo "G1EVAL_START $(date -u +%FT%TZ) mode=$MODE episodes=$EPISODES"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
: "${DATA_BUCKET:?}" "${CHECKPOINT_PREFIX:?}" "${REASON2_KEY:?}" "${REASON2_SHA256:?}"
mkdir -p "$W/ckpt" "$W/hf-cache/hub" && cd "$W" || stop "workdir"

# 1. Stage the checkpoint (final weights only), Reason2 and the episode-0 start pose.
aws s3 cp "s3://$DATA_BUCKET/$CHECKPOINT_PREFIX/" ckpt/ --recursive --exclude 'checkpoint-*/*' \
  --region "$REGION" --only-show-errors || stop "checkpoint download"
grep -q '"embodiment_config": "g1_dex3"' ckpt/training_metadata.json || stop "not a g1_dex3 checkpoint"
echo "G1EVAL_CKPT $(grep -o '"max_steps": [0-9]*' ckpt/training_metadata.json)"
aws s3 cp "s3://$DATA_BUCKET/$REASON2_KEY" reason2.tar --region "$REGION" --only-show-errors || stop "Reason2 download"
printf '%s  %s\n' "$REASON2_SHA256" reason2.tar | sha256sum -c - || stop "Reason2 checksum"
mkdir -p "$SNAP" "$W/hf-cache/hub/models--nvidia--Cosmos-Reason2-2B/refs"
tar -xf reason2.tar -C "$SNAP" && rm -f reason2.tar
printf '%s' "$REV" > "$W/hf-cache/hub/models--nvidia--Cosmos-Reason2-2B/refs/main"
test -s "$SNAP/model.safetensors" || stop "Reason2 snapshot"
aws s3 cp "s3://$DATA_BUCKET/teleoperation-data/pick_place_data/data/chunk-000/episode_000000.parquet" ep0.parquet \
  --region "$REGION" --only-show-errors || stop "episode 0 download"
/workspace/gr00t-repo/.venv/bin/python - <<'PY' || stop "start pose"
import json, pyarrow.parquet as pq
s = pq.read_table("ep0.parquet", columns=["observation.state"]).column(0)[0].as_py()
assert len(s) == 28, len(s)
json.dump([float(v) for v in s], open("start_pose.json", "w"))
print("G1EVAL_START_POSE", [round(v, 3) for v in s[:7]], "...")
PY

# 2. Start the GR00T bridge on 127.0.0.1:5568 (training-matched GR00T venv, offline).
HF_HOME="$W/hf-cache" HF_HUB_CACHE="$W/hf-cache/hub" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  PYTHONPATH=/workspace/gr00t-repo \
  /workspace/gr00t-repo/.venv/bin/python -u /opt/g1dex3/policy_bridge.py \
  --ckpt "$W/ckpt" --reason2-snapshot "$SNAP" --samples "$SAMPLES" > bridge.log 2>&1 &
BRIDGE_PID=$!
for _ in $(seq 1 120); do
  grep -q BRIDGE_READY bridge.log && break
  kill -0 "$BRIDGE_PID" 2>/dev/null || { tail -40 bridge.log; stop "bridge exited"; }
  sleep 5
done
grep -E 'PASS|BRIDGE_CONFIG|BRIDGE_READY' bridge.log
grep -q BRIDGE_READY bridge.log || { tail -40 bridge.log; stop "bridge not ready after 10 min"; }

# 3. Run Arena (same environment setup as the connector image's entrypoint).
set +u; [ -f /isaac-sim/setup_python_env.sh ] && source /isaac-sim/setup_python_env.sh; set -u
export CARB_APP_PATH="${CARB_APP_PATH:-/isaac-sim}" ISAAC_PATH="${ISAAC_PATH:-/isaac-sim}" EXP_PATH="${EXP_PATH:-/isaac-sim/apps}"
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=Y
[ -f /usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.0 ] && export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.0
ldconfig 2>/dev/null || true
[ "$MODE" = gate0 ] && EPISODES=1
RECORD_EVERY=0
[ "$MODE" = record ] && { EPISODES=1; RECORD_EVERY=1; }
cd /workspace || stop "arena workspace"
t0=$(date +%s)
G1DEX3_FRAME_DUMP=1 G1DEX3_RECORD_EVERY="$RECORD_EVERY" PYTHONUNBUFFERED=1 /isaac-sim/python.sh -u isaaclab_arena/evaluation/policy_runner.py \
  --policy_type arena_g1_dex3_policy.G1Dex3BridgePolicy --samples "$SAMPLES" \
  --num_episodes "$EPISODES" --num_envs 1 --seed "$SEED" --headless --enable_cameras \
  --external_environment_class_path arena_g1_bluebox_env:G1BlueBoxEnvironment \
  g1_blue_box --start_pose_json "$W/start_pose.json" > "$W/arena.log" 2>&1
rc=$?
grep -E 'G1DEX3_|FRAME_PNG_B64|Metrics|metrics|Simulation length|Episodes: 100%|Error|Traceback|rror:' "$W/arena.log" | tail -60
[ $rc -eq 0 ] || tail -60 "$W/arena.log"
grep -E 'latency|client' "$W/bridge.log" | tail -3
kill "$BRIDGE_PID" 2>/dev/null
echo "G1EVAL_EXIT rc=$rc seconds=$(( $(date +%s) - t0 ))"
exit $rc
