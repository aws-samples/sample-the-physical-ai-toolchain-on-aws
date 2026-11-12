"""PLUMBING TEST ONLY: an SO-101 follower with no hardware. Its motors and cameras replay one real recorded episode
from the
arm's own dataset (states and camera frames as recorded), so the ACT checkpoint sees real images; commands are
counted, not executed. The arm therefore does NOT follow the policy: this tests plumbing, rules and timing, not
pick success (say so wherever its numbers are quoted)."""

from __future__ import annotations

import os

import numpy as np

NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


class Replay:
    def __init__(self, ds_id: str, episode: int, cams: list[str]):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        root = os.path.expanduser(f"~/.cache/huggingface/lerobot/{ds_id}")
        ds = LeRobotDataset(ds_id, root=root, episodes=[episode])
        a, b = int(ds.meta.episodes["dataset_from_index"][episode]), int(ds.meta.episodes["dataset_to_index"][episode])
        self.states, self.images, self.k = [], {c: [] for c in cams}, 0
        start = int(
            os.environ.get("FAKE_START", 0)
        )  # a later start replays the post-pick rest: a pick that never happens
        for i in range(start, b - a):
            item = ds[i]
            self.states.append(item["observation.state"].numpy().astype(float))
            for c in cams:
                self.images[c].append(
                    (item[f"observation.images.{c}"].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
                )

    def state(self) -> dict:
        return dict(zip(NAMES, self.states[min(self.k, len(self.states) - 1)]))

    def advance(self) -> None:
        self.k = min(self.k + 1, len(self.states) - 1)


class FakeBus:
    def __init__(self, replay: Replay):
        self.replay, self.writes, self.reads, self.connected = replay, 0, 0, False

    def connect(self, handshake: bool = True) -> None:
        self.connected = True

    def sync_read(self, name: str, motors=None, **_):
        self.reads += 1
        return self.replay.state()

    def sync_write(self, *a, **k):
        self.writes += 1

    def write(self, *a, **k):
        self.writes += 1

    def disconnect(self, disable_torque: bool = True) -> None:
        self.connected = False


class FakeCamera:
    def __init__(self, replay: Replay, key: str):
        self.replay, self.key = replay, key

    def connect(self, warmup: bool = True) -> None:
        pass

    def read(self):
        return self.replay.images[self.key][self.replay.k]

    async_read = read

    def disconnect(self) -> None:
        pass


class ReplayFollower:
    def __init__(self, replay: Replay, cams: list[str]):
        self.replay, self.bus = replay, FakeBus(replay)
        self.cameras = {c: FakeCamera(replay, c) for c in cams}
        self.actions, self.is_connected = 0, False

    def connect(self, calibrate: bool = False) -> None:
        self.bus.connect()
        self.is_connected = True

    def get_observation(self) -> dict:
        obs = {f"{n}.pos": v for n, v in self.replay.state().items()}
        obs.update({c: cam.read() for c, cam in self.cameras.items()})
        self.replay.advance()
        return obs

    def send_action(self, action: dict) -> dict:
        self.actions += 1
        return action

    def disconnect(self) -> None:
        self.is_connected = False
