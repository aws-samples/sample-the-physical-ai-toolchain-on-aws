"""What the robot host needs from a real SO-101 follower, built directly on LeRobot 0.6.1.

Units are LeRobot's: joints in calibrated degrees, gripper 0..100. The pick skill's ACT checkpoint runs HERE, on the
robot host, at 30 Hz; the planner only names the skill.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
HZ = 30.0
CAL_DIR = Path.home() / ".cache/huggingface/lerobot/calibration/robots/so_follower"
POSES = json.loads((Path(__file__).with_name("poses.json")).read_text())


def limits_from_calibration(cal_id: str) -> dict:
    """Each joint's reachable range in LeRobot units, from the arm's own calibration file (DEGREES: raw counts about
    the calibrated midpoint, 360/4095 deg per count; gripper RANGE_0_100)."""
    cal = json.loads((CAL_DIR / f"{cal_id}.json").read_text())
    out = {}
    for n in NAMES:
        lo, hi = cal[n]["range_min"], cal[n]["range_max"]
        if n == "gripper":
            out[n] = (0.0, 100.0)
        else:
            half = (hi - lo) / 2 * 360 / 4095
            out[n] = (round(-half, 2), round(half, 2))
    return out


def make_follower(port: str, cal_id: str, cameras: dict, max_relative_target: float):
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
    from lerobot.robots.so_follower.so_follower import SO101Follower

    cams = {
        k: OpenCVCameraConfig(index_or_path=v["index"], width=v["w"], height=v["h"], fps=30) for k, v in cameras.items()
    }
    return SO101Follower(
        SO101FollowerConfig(
            port=port,
            id=cal_id,
            cameras=cams,
            max_relative_target=max_relative_target,
            disable_torque_on_disconnect=True,
        )
    )


class PickPolicy:
    """The ACT checkpoint, loaded once at host start (loading it per pick cost about 12 s), run the way
    lerobot-rollout runs it: saved pre/post processors, select_action every control step."""

    def __init__(self, repo: str, cams: list[str], device: str | None = None):
        import torch

        device = (
            device
            or os.environ.get("POLICY_DEVICE")
            or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        )
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.policies.factory import make_pre_post_processors

        self.torch, self.cams = torch, cams
        self.policy = ACTPolicy.from_pretrained(repo).to(device).eval()
        self.policy.config.device = device
        ov = {"device_processor": {"device": device}}
        self.pre, self.post = make_pre_post_processors(
            self.policy.config, pretrained_path=repo, preprocessor_overrides=ov, postprocessor_overrides=ov
        )

    def reset(self) -> None:
        self.policy.reset()

    def act(self, obs: dict, task: str) -> np.ndarray:
        t = self.torch
        batch = {"observation.state": t.tensor([[float(obs[f"{n}.pos"]) for n in NAMES]], dtype=t.float32)}
        for c in self.cams:
            batch[f"observation.images.{c}"] = (
                t.from_numpy(np.ascontiguousarray(obs[c])).permute(2, 0, 1).float()[None] / 255.0
            )
        batch["task"] = [task]
        with t.no_grad():
            return np.asarray(self.post(self.policy.select_action(self.pre(batch))).cpu()).reshape(-1)


class SO101Body:
    """robot_host.Device's body interface (limits, positions, command, hold, run_policy) for one SO-101."""

    def __init__(self, follower, arm: str, cal_id: str, policy: PickPolicy | None, task: str):
        self.robot, self.arm, self.policy, self.task = follower, arm, policy, task
        self.limits = limits_from_calibration(cal_id)
        self.rest = dict(zip(NAMES, POSES[arm]["rest"]))
        self.trace: list[tuple[float, float]] = []  # (pan, gripper) per step of the last pick, for its postcondition

    def positions(self) -> dict:
        pos = self.robot.bus.sync_read("Present_Position")
        return {n: round(float(pos[n]), 2) for n in NAMES}

    def command(self, targets: dict) -> None:
        self.robot.send_action({f"{k}.pos": float(v) for k, v in targets.items()})

    def hold(self) -> None:
        self.command(self.positions())

    def _check_range(self, targets: dict) -> str | None:
        """Rule 1 for the policy's own outputs: refused (the arm holds), never clamped, outside the range this arm's
        demonstrations commanded (+-5). Not the calibrated range: the leader commanded past it in ~half the recorded
        frames (the follower rests on its end stop) and the servos' position limits hold the calibrated range."""
        env = POSES[self.arm]["policy_envelope"]
        for j, v in targets.items():
            lo, hi = env[j]
            if not lo <= float(v) <= hi:
                return (
                    f"refused: the policy asked {j}={float(v):.1f} outside its demonstrated range [{lo}, {hi}]; holding"
                )
        return None

    def run_policy(self, spec: dict, seconds: float, why_stop) -> str | None:
        """Ramp to the trained rest pose (at most 3 s), then run the checkpoint for `seconds` at 30 Hz. Every step:
        the robot's rules (why_stop: e-stop, watchdog) and the range rule on the policy's own output."""
        self.trace = []
        start, ramp = self.positions(), int(3 * HZ)
        for i in range(1, ramp + 1):
            why = why_stop()
            if why:
                self.hold()
                return why
            u = i / ramp
            s = 3 * u * u - 2 * u * u * u
            self.command({j: start[j] + (self.rest[j] - start[j]) * s for j in NAMES})
            time.sleep(1 / HZ)
        self.policy.reset()
        t_end, nxt = time.monotonic() + float(seconds), time.monotonic()
        while time.monotonic() < t_end:
            why = why_stop()
            if why:
                self.hold()
                return why
            obs = self.robot.get_observation()
            self.trace.append((float(obs["shoulder_pan.pos"]), float(obs["gripper.pos"])))
            targets = dict(zip(NAMES, self.policy.act(obs, self.task).tolist()))
            bad = self._check_range(targets)
            if bad:
                self.hold()
                return bad
            self.command(targets)
            nxt += 1 / HZ
            time.sleep(max(0.0, nxt - time.monotonic()))
        return None

    def pick_check(self) -> dict:
        """HEURISTIC postcondition, from what the arm's own joints did (no camera looks at the bin): the gripper held
        something (reading 4..13, which only an object between the jaws produces; empty it closes to ~1) for at least
        0.5 s, then reopened (>15) with the pan within the release range this arm used in its demonstrations. It says
        the motion of a successful pick happened; it cannot see whether the part landed in the bin."""
        p = POSES[self.arm]
        lo, hi = p["hold_range"]
        open_above = p["open_above"]
        pan_lo, pan_hi = p["release_pan_p5_p95"]
        pan_lo, pan_hi = pan_lo - 8, pan_hi + 8
        best = run = 0
        end = None
        for i, (_, g) in enumerate(self.trace):
            run = run + 1 if lo <= g <= hi else 0
            if run > best:
                best, end = run, i
        held_s = best / HZ
        release = next(((pan, g) for pan, g in self.trace[end:] if g > open_above), None) if end is not None else None
        ok = held_s >= 0.5 and release is not None and pan_lo <= release[0] <= pan_hi
        return {
            "ok": bool(ok),
            "heuristic": "joint signature of a pick, not a camera on the bin",
            "held_s": round(held_s, 2),
            "release_pan": None if release is None else round(release[0], 1),
            "expected_release_pan": [pan_lo, pan_hi],
        }
