"""Isaac Lab-Arena policy: the fine-tuned GR00T N1.7 `g1_dex3` checkpoint, served by policy_bridge.py.

Arena runs on Isaac Sim's Python, which cannot import GR00T, so the checkpoint is served by
assets/policy_bridge.py from the GR00T venv, on 127.0.0.1 (same protocol as the Isaac Sim lab).

Every control step this policy:
* sends the 640x480 `robot_head_cam_rgb` frame and the 28 arm and hand joints, in dataset order,
  when a new chunk is due;
* gets back 16 future targets recorded at 30 Hz, and samples them at Arena's control rate
  (50 Hz), so the motion keeps the speed it was recorded at;
* re-queries after `replan_every` dataset frames (8 of 16, like the Isaac Sim lab) and applies
  the same causal EMA smoothing;
* writes the 28 targets into the G1 whole-body action. All other joints hold their current
  angle (the WBC drives legs and waist), navigation is zero and base height is 0.75 m.

Load it in Arena with:  --policy_type arena_g1_dex3_policy.G1Dex3BridgePolicy
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import struct
import time
import zlib
from dataclasses import dataclass
from multiprocessing.connection import Client

import numpy as np
import torch

from isaaclab_arena.policy.policy_base import PolicyBase

from arena_g1_bluebox_env import DATASET_JOINTS

AUTHKEY = b"groot-closedloop"
DATA_HZ = 30.0
NUM_WBC_CMDS = 7  # navigate (3) + base height (1) + torso roll/pitch/yaw (3)


def _dump_png(rgb: np.ndarray) -> None:
    """Print one RGB frame as a base64 PNG line, so a headless job's camera view can be checked."""
    h, w = rgb.shape[:2]
    raw = b"".join(b"\x00" + np.ascontiguousarray(rgb[y, :, :3], np.uint8).tobytes() for y in range(h))

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")
    print("FRAME_PNG_B64 " + base64.b64encode(png).decode(), flush=True)


@dataclass
class G1Dex3BridgePolicyArgs:
    bridge_port: int = 5568
    samples: int = 4
    replan_every: int = 8
    smooth_alpha: float = 0.4
    base_height: float = 0.75
    camera_key: str = "robot_head_cam_rgb"

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> "G1Dex3BridgePolicyArgs":
        return cls(
            bridge_port=args.bridge_port,
            samples=args.samples,
            replan_every=args.replan_every,
            smooth_alpha=args.smooth_alpha,
            base_height=args.base_height,
            camera_key=args.camera_key,
        )


class G1Dex3BridgePolicy(PolicyBase):
    name = "g1_dex3_bridge"
    config_class = G1Dex3BridgePolicyArgs

    def __init__(self, config: G1Dex3BridgePolicyArgs):
        super().__init__(config)
        self.task_description = "pick the blue box"
        self.conn = Client(("127.0.0.1", config.bridge_port), authkey=AUTHKEY)
        self.conn.send({"cmd": "config"})
        info = self.conn.recv()
        print(f"G1DEX3_POLICY bridge config: horizon={info.get('action_horizon')} dim={info.get('action_dim')}", flush=True)
        self.horizon = int(info["action_horizon"])
        self._ready = False
        self.queries = 0
        self.latency_total = 0.0

    def _setup(self, env):
        names = list(env.unwrapped.scene["robot"].data.joint_names)
        missing = [j for j in DATASET_JOINTS if j not in names]
        assert not missing, f"G1 asset lacks dataset joints: {missing}"
        self.idx28 = torch.tensor([names.index(j) for j in DATASET_JOINTS], dtype=torch.long)
        self.num_joints = len(names)
        self.action_dim = int(env.action_space.shape[-1])
        assert self.action_dim == self.num_joints + NUM_WBC_CMDS, (self.action_dim, self.num_joints)
        self.num_envs = env.unwrapped.num_envs
        self.device = torch.device(env.unwrapped.device)
        self.step_dt = float(env.unwrapped.step_dt)
        self.chunks = np.zeros((self.num_envs, self.horizon, 28), np.float32)
        self.chunk_t = np.full(self.num_envs, np.inf)
        self.ema = [None] * self.num_envs
        print(f"G1DEX3_POLICY joints={self.num_joints} action_dim={self.action_dim} step_dt={self.step_dt:.4f}", flush=True)
        self._ready = True

    def set_task_description(self, task_description: str | None) -> str:
        if task_description:
            self.task_description = task_description
        return self.task_description

    def _diag(self, env, tag: str) -> None:
        """Print world poses of the camera, robot links and box, so the view can be checked numerically."""
        scene = env.unwrapped.scene
        try:
            import warp as wp

            def t(x):
                return wp.to_torch(x) if not isinstance(x, torch.Tensor) else x
        except Exception:  # noqa: BLE001
            def t(x):
                return x
        out = {}
        try:
            robot = scene["robot"]
            names = list(robot.data.body_names)
            pos = t(robot.data.body_pos_w)[0]
            for b in ("pelvis", "torso_link", "head_link", "right_hand_palm_link", "left_hand_palm_link"):
                if b in names:
                    out[b] = [round(float(v), 3) for v in pos[names.index(b)]]
        except Exception as exc:  # noqa: BLE001
            out["robot_err"] = repr(exc)[:200]
        try:
            out["box"] = [round(float(v), 3) for v in t(scene["blue_box"].data.root_pos_w)[0]]
        except Exception as exc:  # noqa: BLE001
            out["box_err"] = repr(exc)[:200]
        for key in ("robot_head_cam", "robot_head_cam_rgb"):
            try:
                cam = scene[key]
                out["cam_pos"] = [round(float(v), 3) for v in t(cam.data.pos_w)[0]]
                for q in ("quat_w_world", "quat_w_ros", "quat_w_opengl"):
                    if hasattr(cam.data, q):
                        out[q] = [round(float(v), 4) for v in t(getattr(cam.data, q))[0]]
                out["intrinsics"] = [round(float(v), 1) for v in t(cam.data.intrinsic_matrices)[0].flatten()[[0, 2, 4, 5]]]
                break
            except Exception as exc:  # noqa: BLE001
                out[f"{key}_err"] = repr(exc)[:120]
        print(f"G1DEX3_DIAG {tag} " + json.dumps(out), flush=True)

    def _query(self, image: np.ndarray, state: np.ndarray) -> np.ndarray:
        if self.queries == 0 and os.environ.get("G1DEX3_FRAME_DUMP"):
            _dump_png(image[::2, ::2])
        t0 = time.time()
        self.conn.send({
            "cmd": "act",
            "image": np.ascontiguousarray(image, np.uint8).tobytes(),
            "shape": list(image.shape),
            "state": state.astype(float).tolist(),
            "task": self.task_description,
            "samples": self.config.samples,
        })
        resp = self.conn.recv()
        if "error" in resp:
            raise RuntimeError(resp["error"])
        self.queries += 1
        self.latency_total += time.time() - t0
        return np.asarray(resp["action"], np.float32)

    def get_action(self, env, observation) -> torch.Tensor:
        if not self._ready:
            self._setup(env)
        joint_pos = observation["policy"]["robot_joint_pos"].detach().to("cpu").float()
        rgb = observation["camera_obs"][self.config.camera_key].detach().to("cpu").numpy()
        cfg = self.config
        out = torch.zeros((self.num_envs, self.action_dim), dtype=torch.float32)
        out[:, : self.num_joints] = joint_pos
        out[:, self.num_joints + 3] = cfg.base_height
        for i in range(self.num_envs):
            if self.chunk_t[i] * DATA_HZ >= cfg.replan_every:
                every = int(os.environ.get("G1DEX3_RECORD_EVERY", "0") or 0)
                if every > 0 and self.queries % every == 0 and self.queries not in (0, 12):
                    img_r = rgb[i][::2, ::2, :3]
                    _dump_png(img_r if img_r.dtype == np.uint8 else np.clip(img_r, 0, 255).astype(np.uint8))
                if self.queries in (0, 12):
                    self._diag(env, f"query{self.queries}")
                    if self.queries == 12 and os.environ.get("G1DEX3_FRAME_DUMP"):
                        _dump_png(rgb[i][::2, ::2, :3].astype(np.uint8) if rgb[i].dtype == np.uint8 else
                                  np.clip(rgb[i][::2, ::2, :3], 0, 255).astype(np.uint8))
                state = joint_pos[i, self.idx28].numpy()
                img = rgb[i]
                if img.dtype != np.uint8:
                    img = np.clip(img * (255.0 if img.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
                self.chunks[i] = self._query(img[..., :3], state)
                self.chunk_t[i] = 0.0
            f = min(self.chunk_t[i] * DATA_HZ, self.horizon - 1.0)
            k = int(np.floor(f))
            a = f - k
            target = self.chunks[i, k] * (1 - a) + self.chunks[i, min(k + 1, self.horizon - 1)] * a
            if self.ema[i] is None or cfg.smooth_alpha >= 1.0:
                self.ema[i] = target
            else:
                self.ema[i] = cfg.smooth_alpha * target + (1 - cfg.smooth_alpha) * self.ema[i]
            out[i, self.idx28] = torch.from_numpy(self.ema[i])
            self.chunk_t[i] += self.step_dt
        return out.to(self.device)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if not self._ready:
            return
        ids = range(self.num_envs) if env_ids is None else [int(i) for i in env_ids]
        for i in ids:
            self.chunk_t[i] = np.inf
            self.ema[i] = None
        if self.queries:
            print(f"G1DEX3_POLICY queries={self.queries} mean_latency_s={self.latency_total / self.queries:.3f}", flush=True)

    @staticmethod
    def add_args_to_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        g = parser.add_argument_group("G1 Dex3 GR00T bridge policy")
        g.add_argument("--bridge_port", type=int, default=5568)
        g.add_argument("--samples", type=int, default=4, help="diffusion samples averaged per query")
        g.add_argument("--replan_every", type=int, default=8, help="dataset frames executed before re-querying")
        g.add_argument("--smooth_alpha", type=float, default=0.4, help="causal EMA on targets; 1.0 = off")
        g.add_argument("--base_height", type=float, default=0.75)
        g.add_argument("--camera_key", type=str, default="robot_head_cam_rgb")
        return parser

    @staticmethod
    def from_args(args: argparse.Namespace) -> "G1Dex3BridgePolicy":
        return G1Dex3BridgePolicy(G1Dex3BridgePolicyArgs.from_cli_args(args))
