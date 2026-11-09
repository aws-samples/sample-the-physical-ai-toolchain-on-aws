#!/usr/bin/env python3
"""Export a full-episode joint trace for Isaac Sim replay.

Runs inside the gr00t-training:n17 container. For every frame of one episode it writes:

  recorded[i]   the recorded teleop `action` (28 absolute joint targets)
  state[i]      the recorded `observation.state` (28 joint positions)
  predicted[i]  the policy's action for frame i, taken from an action chunk

The policy is queried teacher-forced every --replan-every frames (default 8): at each
query frame it is given the REAL recorded state and the REAL camera frame, and the
returned 16-step chunk supplies the targets for that frame and the next N-1 frames.
This is how GR00T is meant to run; re-querying every frame and keeping only step 0
(--replan-every 1) re-samples the diffusion head each frame and jitters about 4x more
than the recorded actions.

Pass --no-policy to export only the recorded trace. That path needs no Hugging Face
token and no GPU, because the policy is never constructed.

Output: /eval/out/trace_ep<N>.json
"""
import argparse
import json
import os
import sys
import time
import traceback

try:
    import flash_attn  # noqa: F401
except Exception:  # noqa: BLE001
    for _k in list(sys.modules.keys()):
        if "flash_attn" in _k:
            del sys.modules[_k]

import numpy as np
import pandas as pd

DATA = "/eval/data"
CKPT = "/eval/ckpt"

GROUPS = ["left_arm", "right_arm", "left_hand", "right_hand"]
SLICES = {
    "left_arm": (0, 7),
    "right_arm": (7, 14),
    "left_hand": (14, 21),
    "right_hand": (21, 28),
}

# Isaac Sim / URDF joint names, in the dataset's 28-wide index order.
# Derived mechanically from the `kCamelCase` names in meta/info.json and verified
# against the G1 USD's `dof_names` (all 28 present, no misses).
SIM_JOINT_NAMES = [
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
    "left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint",
    "left_hand_middle_0_joint", "left_hand_middle_1_joint",
    "left_hand_index_0_joint", "left_hand_index_1_joint",
    "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
    "right_hand_index_0_joint", "right_hand_index_1_joint",
    "right_hand_middle_0_joint", "right_hand_middle_1_joint",
]


def frame_hwc(video_path, index):
    from gr00t.utils.video_utils import get_frames_by_indices

    arr = np.asarray(
        get_frames_by_indices(video_path, [int(index)], decoder_kwargs={"dimension_order": "NHWC"})
    )
    if arr.ndim == 4 and arr.shape[-1] != 3 and arr.shape[1] == 3:
        arr = np.transpose(arr, (0, 2, 3, 1))
    return np.ascontiguousarray(arr, dtype=np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--no-policy", action="store_true")
    ap.add_argument("--max-frames", type=int, default=0, help="0 = whole episode")
    ap.add_argument(
        "--replan-every",
        type=int,
        default=8,
        help="query the policy every N frames and play its action chunk in between "
        "(1 = re-query every frame and keep only step 0; max = action horizon 16)",
    )
    ap.add_argument(
        "--no-blend",
        action="store_true",
        help="do not cross-fade the previous chunk's unused tail into each new chunk",
    )
    ap.add_argument(
        "--samples",
        type=int,
        default=8,
        help="draw N diffusion samples per policy query and average them; reduces "
        "sampling noise (jitter) at N times the inference cost; default 8",
    )
    ap.add_argument(
        "--smooth-alpha",
        type=float,
        default=0.4,
        help="causal exponential smoothing of the predicted targets, "
        "y[t] = a*x[t] + (1-a)*y[t-1]; 1.0 = off, default 0.4, lower = smoother but laggier. "
        "Causal, so a real controller could apply the same filter",
    )
    ap.add_argument("--out", default="", help="output path; default /eval/out/trace_ep<N>.json")
    args, _ = ap.parse_known_args()
    args.blend = not args.no_blend
    if not 0.0 < args.smooth_alpha <= 1.0:
        raise ValueError(f"--smooth-alpha must be in (0, 1], got {args.smooth_alpha}")
    if not 1 <= args.replan_every <= 16:
        raise ValueError(f"--replan-every must be 1-16 (action horizon), got {args.replan_every}")
    if args.samples < 1:
        raise ValueError(f"--samples must be >= 1, got {args.samples}")

    ep = args.episode
    out_path = args.out or f"/eval/out/trace_ep{ep}.json"

    pq = os.path.join(DATA, f"data/chunk-000/episode_{ep:06d}.parquet")
    vid = os.path.join(DATA, f"videos/chunk-000/observation.images.cam_high/episode_{ep:06d}.mp4")
    df = pd.read_parquet(pq)
    n = len(df)
    if args.max_frames:
        n = min(n, args.max_frames)

    with open(os.path.join(DATA, "meta/info.json")) as fh:
        info = json.load(fh)
    fps = int(info["fps"])
    dataset_joint_names = list(info["features"]["observation.state"]["names"])

    tasks = {}
    with open(os.path.join(DATA, "meta/tasks.jsonl")) as fh:
        for line in fh:
            r = json.loads(line)
            tasks[int(r["task_index"])] = r["task"]

    recorded = np.stack([np.asarray(a, dtype=np.float32) for a in df["action"].to_numpy()[:n]])
    state = np.stack(
        [np.asarray(a, dtype=np.float32) for a in df["observation.state"].to_numpy()[:n]]
    )

    trace = {
        "episode": ep,
        "frames": int(n),
        "fps": fps,
        "task": tasks[int(df.iloc[0]["task_index"])],
        "dataset_joint_names": dataset_joint_names,
        "sim_joint_names": SIM_JOINT_NAMES,
        "recorded": recorded.tolist(),
        "state": state.tolist(),
        "predicted": None,
    }

    if not args.no_policy:
        try:
            import runpy

            runpy.run_path("/opt/ml/code/g1_dex3_modality_config.py")
        except Exception:  # noqa: BLE001
            pass

        import torch
        from gr00t.data.embodiment_tags import EmbodimentTag
        from gr00t.policy.gr00t_policy import Gr00tPolicy

        assert torch.cuda.is_available(), "no CUDA device visible"
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        policy = Gr00tPolicy(
            embodiment_tag=EmbodimentTag.NEW_EMBODIMENT, model_path=CKPT, device="cuda:0"
        )
        trace["load_seconds"] = round(time.time() - t0, 1)

        preds = []
        lat = []
        chunk = None
        blend_steps = 0
        for fi in range(n):
            k = fi % args.replan_every
            if k == 0:
                obs = {
                    "video": {"cam_high": frame_hwc(vid, fi)[None]},
                    "state": {
                        g: state[fi][a:b].reshape(1, 1, -1) for g, (a, b) in SLICES.items()
                    },
                    "language": {policy.language_key: [[trace["task"]]]},
                }
                t1 = time.time()
                draws = []
                for _ in range(args.samples):
                    action, _ = policy.get_action(obs)
                    # action[g] is (batch, horizon, dims); horizon step k targets frame fi+k.
                    draws.append(
                        np.concatenate([np.asarray(action[g])[0] for g in GROUPS], axis=-1)
                    )
                lat.append(time.time() - t1)
                new = np.mean(np.stack(draws), axis=0)
                if new.shape[0] < args.replan_every:
                    raise RuntimeError(
                        f"policy returned a {new.shape[0]}-step chunk, "
                        f"shorter than --replan-every {args.replan_every}"
                    )
                # Cross-fade from the unused tail of the previous chunk into the new one,
                # so chunk boundaries do not jump when the diffusion head re-samples.
                if args.blend and chunk is not None:
                    tail = chunk[args.replan_every:]
                    m = min(len(tail), args.replan_every, len(new))
                    if m:
                        w = (np.arange(1, m + 1, dtype=np.float32) / (m + 1))[:, None]
                        new = new.copy()
                        new[:m] = (1.0 - w) * tail[:m] + w * new[:m]
                        blend_steps = m
                chunk = new
            preds.append(chunk[k].astype(np.float32))
            if fi % 20 == 0:
                print(f"  frame {fi}/{n}", flush=True)

        pred = np.stack(preds)
        if args.smooth_alpha < 1.0:
            raw = pred
            pred = raw.copy()
            for t in range(1, len(pred)):
                pred[t] = args.smooth_alpha * raw[t] + (1.0 - args.smooth_alpha) * pred[t - 1]
            trace["mae_policy_unsmoothed"] = float(np.abs(raw - recorded).mean())
        trace["smooth_alpha"] = args.smooth_alpha
        trace["predicted"] = pred.tolist()
        trace["replan_every"] = args.replan_every
        trace["blend_steps"] = blend_steps
        trace["samples_per_query"] = args.samples
        trace["jitter_policy_hand"] = float(np.abs(np.diff(pred[:, 14:], axis=0)).mean())
        trace["jitter_recorded_hand"] = float(np.abs(np.diff(recorded[:, 14:], axis=0)).mean())
        trace["policy_queries"] = len(lat)
        trace["peak_vram_mib"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
        trace["latency_mean_s"] = round(float(np.mean(lat)), 3)
        trace["mae_policy_overall"] = float(np.abs(pred - recorded).mean())
        trace["mae_baseline_overall"] = float(np.abs(state - recorded).mean())
        trace["jitter_policy_arm"] = float(np.abs(np.diff(pred[:, :14], axis=0)).mean())
        trace["jitter_recorded_arm"] = float(np.abs(np.diff(recorded[:, :14], axis=0)).mean())
        trace["mae_per_group"] = {
            g: float(np.abs(pred[:, a:b] - recorded[:, a:b]).mean())
            for g, (a, b) in SLICES.items()
        }

    with open(out_path, "w") as fh:
        json.dump(trace, fh)

    summary = {
        k: v
        for k, v in trace.items()
        if k not in ("recorded", "state", "predicted", "dataset_joint_names", "sim_joint_names")
    }
    summary["has_predicted"] = trace["predicted"] is not None
    print(json.dumps(summary, indent=2), flush=True)
    print(f"WROTE {out_path}", flush=True)


if __name__ == "__main__":
    os.makedirs("/eval/out", exist_ok=True)
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        sys.exit(1)
