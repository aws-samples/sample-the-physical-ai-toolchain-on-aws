#!/usr/bin/env python3
"""Serve our fine-tuned GR00T N1.7 g1_dex3 checkpoint to a closed-loop Isaac Sim client.

Runs inside the gr00t-training:n17 container (the same one export_trace.py uses), so the
checkpoint loads exactly as it did for the offline trace. Isaac Sim's own Python cannot
import GR00T, so the two processes talk over a localhost socket using only the standard
library (multiprocessing.connection), with plain bytes and lists on the wire.

Requests (dicts):
  {"cmd": "config"}                                   -> modality keys and action horizon
  {"cmd": "act", "image": bytes, "shape": [H, W, 3],
   "state": [28 floats, dataset order], "task": str,
   "samples": optional int}                           -> {"action": horizon x 28 list}
  {"cmd": "close"}                                    -> ends this client connection

Listens on 127.0.0.1 only.

    docker run -d --name groot-bridge --gpus all --network host --shm-size=8g \
      -e HF_HOME=/eval/hf-cache -e HF_HUB_CACHE=/eval/hf-cache/hub \
      -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e HF_DATASETS_OFFLINE=1 \
      -v /home/ubuntu/groot-eval:/eval \
      --entrypoint /workspace/gr00t-repo/.venv/bin/python \
      <account>.dkr.ecr.us-east-2.amazonaws.com/physical-ai/gr00t-training:n17 \
      -u /eval/closedloop/policy_bridge.py
"""
import argparse
import json
import sys
import time
import traceback
from multiprocessing.connection import Listener

try:
    import flash_attn  # noqa: F401
except Exception:  # noqa: BLE001
    for _k in list(sys.modules.keys()):
        if "flash_attn" in _k:
            del sys.modules[_k]

import numpy as np

GROUPS = ["left_arm", "right_arm", "left_hand", "right_hand"]
SLICES = {"left_arm": (0, 7), "right_arm": (7, 14), "left_hand": (14, 21), "right_hand": (21, 28)}
AUTHKEY = b"groot-closedloop"


def describe(policy):
    """Best-effort read of the checkpoint's modality config."""
    out = {"language_key": getattr(policy, "language_key", None)}
    for attr in ("modality_configs", "modality_config", "get_modality_config"):
        val = getattr(policy, attr, None)
        if val is None:
            continue
        try:
            val = val() if callable(val) else val
            out["modality_config_repr"] = repr(val)[:3000]
            break
        except Exception as exc:  # noqa: BLE001
            out[f"{attr}_error"] = f"{type(exc).__name__}: {exc}"
    return out


PROCESSOR_SRC = "/workspace/gr00t-repo/gr00t/model/gr00t_n1d7/processing_gr00t_n1d7.py"
PROCESSOR_NEEDLE = "return Qwen3VLProcessor.from_pretrained(model_name, **transformers_loading_kwargs)"


def use_local_reason2(snapshot):
    """Point the Qwen3-VL processor load at the verified local snapshot (same edit as the
    evaluation page's step 4), so the bridge loads with Hugging Face offline mode on and
    needs no token. Edits only this container's copy of the file."""
    from pathlib import Path

    if not (Path(snapshot) / "config.json").is_file():
        raise SystemExit(f"Reason2 snapshot not found: {snapshot}")
    path = Path(PROCESSOR_SRC)
    source = path.read_text()
    replacement = f"return Qwen3VLProcessor.from_pretrained({snapshot!r}, **transformers_loading_kwargs)"
    if replacement in source:
        return
    if source.count(PROCESSOR_NEEDLE) != 1:
        raise SystemExit(f"Expected one Qwen3-VL processor load site, found {source.count(PROCESSOR_NEEDLE)}")
    path.write_text(source.replace(PROCESSOR_NEEDLE, replacement))
    print(f"PASS - Qwen3-VL processor uses {snapshot}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/eval/ckpt")
    ap.add_argument("--port", type=int, default=5568)
    ap.add_argument("--samples", type=int, default=1, help="diffusion samples averaged per query")
    ap.add_argument(
        "--reason2-snapshot",
        default="/eval/hf-cache/hub/models--nvidia--Cosmos-Reason2-2B/snapshots/9ce19a195e423419c349abfc86fd07178b230561",
        help="local Cosmos-Reason2-2B snapshot for the processor; empty string = load from the Hub",
    )
    args = ap.parse_args()
    if args.reason2_snapshot:
        use_local_reason2(args.reason2_snapshot)

    try:
        import runpy

        runpy.run_path("/opt/ml/code/g1_dex3_modality_config.py")
    except Exception:  # noqa: BLE001
        pass

    import torch
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    assert torch.cuda.is_available(), "no CUDA device visible"
    t0 = time.time()
    policy = Gr00tPolicy(embodiment_tag=EmbodimentTag.NEW_EMBODIMENT, model_path=args.ckpt, device="cuda:0")
    lang_key = policy.language_key

    def act(image, state, task, samples=None):
        obs = {
            "video": {"cam_high": image[None, None]},
            "state": {g: np.asarray(state[a:b], np.float32).reshape(1, 1, -1) for g, (a, b) in SLICES.items()},
            "language": {lang_key: [[task]]},
        }
        draws = []
        for _ in range(int(samples or args.samples)):
            action, _ = policy.get_action(obs)
            draws.append(np.concatenate([np.asarray(action[g])[0] for g in GROUPS], axis=-1))
        return np.mean(np.stack(draws), axis=0), {g: list(np.asarray(action[g]).shape) for g in GROUPS}

    # Self-test with a blank frame, so the startup log proves keys, dims and horizon.
    chunk, shapes = act(np.zeros((480, 640, 3), np.uint8), [0.0] * 28, "pick the blue box")
    info = describe(policy)
    info.update({
        "load_seconds": round(time.time() - t0, 1),
        "video_key": "cam_high",
        "state_groups": GROUPS,
        "action_shapes": shapes,
        "action_horizon": int(chunk.shape[0]),
        "action_dim": int(chunk.shape[1]),
    })
    print("BRIDGE_CONFIG " + json.dumps({k: v for k, v in info.items() if k != "modality_config_repr"}), flush=True)
    print("MODALITY " + str(info.get("modality_config_repr", "n/a"))[:1500], flush=True)

    listener = Listener(("127.0.0.1", args.port), authkey=AUTHKEY)
    print(f"BRIDGE_READY 127.0.0.1:{args.port}", flush=True)
    while True:
        conn = listener.accept()
        print("client connected", flush=True)
        try:
            while True:
                req = conn.recv()
                cmd = req.get("cmd")
                if cmd == "config":
                    conn.send(info)
                elif cmd == "act":
                    img = np.frombuffer(req["image"], np.uint8).reshape(req["shape"])
                    t1 = time.time()
                    chunk, _ = act(img, req["state"], req.get("task", "pick the blue box"), req.get("samples"))
                    conn.send({"action": chunk.astype(float).tolist(), "latency_s": round(time.time() - t1, 3)})
                elif cmd == "close":
                    break
                else:
                    conn.send({"error": f"unknown cmd {cmd!r}"})
        except EOFError:
            pass
        except Exception:  # noqa: BLE001
            traceback.print_exc()
        finally:
            conn.close()
            print("client disconnected", flush=True)


if __name__ == "__main__":
    main()
