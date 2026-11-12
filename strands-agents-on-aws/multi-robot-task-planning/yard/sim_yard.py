"""The simulated yard: a Unitree G1 and three Pollen Robotics Microduck walkers in one MuJoCo world, each served as
a fleet robot over AWS IoT Core by the same robot host a real robot uses.

Zones: a dock (the yellow pad), a yard lane (the pale strip with a blue checkpoint post) and a yard bin (the green
walled area). Each robot's skills are its trained policies, each with a one-line description and a check that the
robot measures when the skill ends.
"""

from __future__ import annotations

import json
import subprocess
import threading
import time
import uuid

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from strands_robots import Robot, Simulation

from robot_host import Device, serve
from site_runner import Switchboard

BIN = (0.30, 1.0, -0.30, 0.40)  # x0, x1, y0, y1 of the yard bin
DOCK = (0.95, 1.85, -2.10, -1.10)  # the dock pad
LANE_Y, CHECK_X = 0.8, 0.33  # the yard lane, and how far along it the checkpoint is
ZONE = {"duck-1": "yard-bin", "duck-2": "yard-lane", "duck-3": "yard-lane", "g1-1": "dock"}
DUCK_SKILLS = {  # name: (trained policy, seconds, what it does)
    "walk": ("microduck?skill=velstand&vx=0.3", 4.0, "walk about 0.4 m along its lane, past the checkpoint"),
    "turn": ("microduck?skill=velstand&wz=1.2", 3.0, "turn on the spot"),
    "kick": ("microduck?skill=ball_kick_left", 2.0, "kick the part in front of its left foot into the yard bin"),
}
G1_SKILLS = {"walk": ("holosoma?vx=0.5", 4.0, "walk about 1.7 m onto the dock pad and stand guard there")}


def inside(p, box) -> bool:
    return box[0] <= p[0] <= box[1] and box[2] <= p[1] <= box[3]


class SimBody:
    """What the robot host needs from a body, for one simulated robot in the shared yard."""

    def __init__(self, yard: SimYard, did: str):
        self.yard, self.did, self.board = yard, did, yard.boards[did]
        controls = yard.robots[did].manifest.controls
        self.limits = {c.name.removesuffix(".pos"): tuple(round(float(v), 4) for v in c.range) for c in controls}

    def positions(self) -> dict:
        j = self.yard.robots[self.did].observe(images=False).joints
        return {n: round(float(p), 4) for n, p in zip(j.names, j.positions)}

    def hold(self) -> None:
        self.board.stop("hold")  # a biped holds by standing: its stand policy keeps it balanced

    def command(self, targets: dict) -> None:
        raise NotImplementedError("the yard robots move by skills only")

    def run_policy(self, spec, seconds: float, why_stop):
        self.board.start(spec, spec, seconds)
        while self.board.last.get("running"):
            why = why_stop()
            if why:
                self.hold()
                return why
            time.sleep(0.02)
        return None


class SimYard:
    """Builds the world, serves each robot over IoT Core, and records what happened."""

    def __init__(self, operator_public_key: str, tenant: str = "site"):
        self.tenant, self.log, self.offline, self.cut = tenant, [], set(), False
        self.bin = {"at": None, "t": None, "watch": None}
        self.world = w = Simulation("mujoco", render_size=(960, 540))
        with w.editing():
            self.robots = {d: Robot("microduck", id=d, sim=w) for d in ("duck-1", "duck-2", "duck-3")}
            self.robots["g1-1"] = Robot("g1", id="g1-1", sim=w)
            self._scenery()
        w.set_pose("duck-2", position=(0.0, LANE_Y, 0.12))
        w.set_pose("duck-3", position=(0.0, LANE_Y + 0.3, 0.12))
        w.set_pose("g1-1", position=(-0.45, -1.6, 0.79))
        stand = {d: "holosoma" if d == "g1-1" else "microduck?skill=velstand" for d in self.robots}
        self.boards = {d: Switchboard(r, stand[d]) for d, r in self.robots.items()}
        self.lane0 = {d: float(self.pos(d)[0]) for d in ("duck-2", "duck-3")}
        self.t0 = time.perf_counter()
        self.devices = {
            d: Device(SimBody(self, d), d, ZONE[d], self._skills(d), operator_public_key) for d in self.robots
        }
        self.stops = {d: threading.Event() for d in self.robots}
        for d, dev in self.devices.items():
            self._observe(d, dev)
            threading.Thread(target=serve, args=(dev, tenant, self.stops[d]), daemon=True).start()
        boards = dict(zip(self.robots.values(), self.boards.values()))
        threading.Thread(target=lambda: w.run(boards, seconds=900.0), daemon=True).start()

    # ---- the scene
    def _scenery(self) -> None:
        w = self.world

        def flat(name, x0, x1, y0, y1, color):
            w.add_object(
                name,
                shape="box",
                position=((x0 + x1) / 2, (y0 + y1) / 2, -0.0015),
                size=((x1 - x0) / 2, (y1 - y0) / 2, 0.002),
                color=color,
                free=False,
            )

        flat("yard_bin", *BIN, (0.2, 0.75, 0.3, 1))
        flat("dock_pad", *DOCK, (0.95, 0.8, 0.15, 1))
        flat("patrol_lane", 0.10, CHECK_X + 0.05, LANE_Y - 0.06, LANE_Y + 0.36, (0.55, 0.65, 0.8, 1))
        walls = (
            ("bin_back", (1.02, 0.05, 0.12), (0.02, 0.39, 0.12)),
            ("bin_left", (0.67, 0.42, 0.12), (0.37, 0.02, 0.12)),
            ("bin_right", (0.67, -0.32, 0.12), (0.37, 0.02, 0.12)),
        )
        for name, pos, size in walls:
            w.add_object(name, shape="box", position=pos, size=size, color=(0.85, 0.85, 0.85, 1), free=False)
        w.add_object(
            "checkpoint",
            shape="cylinder",
            position=(CHECK_X + 0.06, LANE_Y + 0.12, 0.06),
            size=(0.015, 0.06),
            color=(0.15, 0.35, 0.95, 1),
            free=False,
        )
        w.add_object(
            "part", shape="sphere", position=(0.06, 0.064, 0.03), size=0.03, mass=0.02, color=(0.95, 0.2, 0.2, 1)
        )
        w.light("sun", position=(1.0, -2.0, 4.0), directional=True)
        w.camera("overview", position=(0.7, 3.4, 2.3), target=(0.7, -0.5, 0.15))
        w.camera("walkers", position=(-0.75, 0.5, 0.95), target=(0.5, 0.45, 0.02))

    # ---- measurements
    def now(self) -> float:
        return round(time.perf_counter() - self.t0, 2)

    def pos(self, d: str):
        return self.robots[d].observe(images=False).pose.position

    def _watch_bin(self, seconds: float = 6.0) -> None:
        """Delivered means the part crosses into the bin (in this simulator a ball keeps rolling, so catch the
        moment)."""
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < seconds:
            p = self.world.pose("part").position
            if inside(p, BIN) and self.bin["at"] is None:
                self.bin["at"] = (round(time.perf_counter() - t0, 2), [round(float(v), 2) for v in p[:2]])
                self.bin["t"] = self.now()
            time.sleep(0.02)

    def _bin_check(self) -> dict:
        self.bin["watch"].join()
        e = self.bin["at"]
        return {"ok": e is not None, "in_bin_after_s": e[0] if e else None, "entered_at": e[1] if e else None}

    def _dock_check(self) -> dict:
        p = self.pos("g1-1")
        return {
            "ok": bool(inside(p, DOCK) and p[2] > 0.6),
            "at": [round(float(v), 2) for v in p[:2]],
            "z": round(float(p[2]), 2),
        }

    def _lane_check(self, d: str):
        def check():
            advanced = float(self.pos(d)[0] - self.lane0[d])
            return {"ok": advanced >= CHECK_X - 0.05, "advanced_m": round(advanced, 2)}

        return check

    def _skills(self, d: str) -> dict:
        if d == "g1-1":
            ((policy, secs, does),) = G1_SKILLS.values()
            return {"walk": {"policy": policy, "seconds": secs, "does": does, "check": self._dock_check}}
        checks = {"duck-1": {"kick": self._bin_check}}.get(d, {"walk": self._lane_check(d)})
        return {
            k: {"policy": p, "seconds": s, "does": a, **({"check": checks[k]} if k in checks else {})}
            for k, (p, s, a) in DUCK_SKILLS.items()
        }

    def _observe(self, d: str, dev: Device) -> None:
        """Log every dispatch, start the bin watcher when a kick starts, and, in a dropout run, cut one walker's
        network."""
        start = dev._start

        def logged(name, moves, by, policy=None):
            if d == "duck-1" and name == "kick":
                self.bin.update(at=None, t=None)
                self.bin["watch"] = threading.Thread(target=self._watch_bin, daemon=True)
                self.bin["watch"].start()
            out = start(name, moves, by, policy)
            self.log.append({"t": self.now(), "dispatched": name, "to": d})
            if self.cut and d in ("duck-2", "duck-3") and name == "walk" and not self.offline:
                threading.Timer(1.0, self._lose_network, args=(d,)).start()
            return out

        dev._start = logged

    def _lose_network(self, d: str) -> None:
        """This robot host loses its network: no description, no replies, no planner heartbeats."""
        self.log.append({"t": self.now(), "offline": d})
        self.offline.add(d)
        self.stops[d].set()

    def outcome(self) -> dict:
        end = {d: self.pos(d) for d in self.robots}
        g1 = end["g1-1"]
        ros = subprocess.run(
            [
                "docker",
                "exec",
                "ros-arm",
                "bash",
                "-lc",  # the ROS 2 arm's own /joint_states
                "source /opt/ros/jazzy/setup.bash && /venv/bin/python /ros2/ros_read.py",
            ],
            capture_output=True,
            text=True,
        ).stdout.strip()[-200:]
        return {
            "part_in_bin": self.bin["at"],
            "g1_on_dock": {
                "at": [round(float(v), 2) for v in g1[:2]],
                "z": round(float(g1[2]), 2),
                "ok": bool(inside(g1, DOCK) and g1[2] > 0.6),
            },
            "lane_advanced_m": {d: round(float(end[d][0] - self.lane0[d]), 2) for d in ("duck-2", "duck-3")},
            "stopped_by": {d: self.devices[d].status().get("stopped_by") for d in self.robots},
            "ros_arm_joints": ros,
        }

    def close(self) -> None:
        for e in self.stops.values():
            e.set()
        time.sleep(2.0)  # each robot clears its retained description as it leaves


# ---- running a goal through the planner on AgentCore Runtime, filmed in real time
FONT, SMALL = ImageFont.load_default(size=22), ImageFont.load_default(size=18)


def _label(img, text: str):
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im)
    dr.rectangle([0, 0, im.width, 30], fill=(20, 24, 32))
    dr.text((10, 5), text, fill=(240, 240, 240), font=SMALL)
    return np.asarray(im)


def _frame(yard: SimYard, done: float | None):
    st = {d: yard.devices[d].status() for d in yard.robots}
    tags = "   ".join(
        f"{d}: " + ("offline" if d in yard.offline else (st[d]["name"] if st[d].get("running") else "idle"))
        for d in yard.robots
    )
    row = np.hstack(
        [
            _label(yard.world.render("overview"), "yard overview   |   " + tags),
            _label(yard.world.render("walkers"), "close-up: walkers"),
        ]
    )
    e, busy = (yard.log[-1] if yard.log else None), any(s.get("running") for s in st.values())
    if done is not None and not busy:
        stage = f"planner finished at {done:.1f} s and reported back"
    elif e is None:
        stage = "planner: discovering robots over AWS IoT Core and planning"
    elif "offline" in e:
        stage = f"{e['offline']} lost its network connection"
    else:
        stage = f"planner dispatched '{e['dispatched']}' to {e['to']} (the robot runs the policy)"
    inbin = f"   |   part in the bin at t = {yard.bin['t']:.1f} s" if yard.bin["t"] is not None else ""
    ban = Image.new("RGB", (row.shape[1], 44), (12, 14, 20))  # 44 + 540: a height the video codec takes as is
    ImageDraw.Draw(ban).text(
        (12, 8),
        f"t = {yard.now():5.1f} s  |  Strands agent on Amazon Bedrock AgentCore Runtime  |  {stage}{inbin}",
        fill=(255, 210, 90),
        font=FONT,
    )
    return np.vstack([np.asarray(ban), row])


def run_goal(yard: SimYard, agentcore, planner_arn: str, goal: str, clip: str, cut: bool = False, min_devices: int = 5):
    """Send one goal to the planner, film the yard while it runs, and return (reply, measured outcome, clip path)."""
    yard.cut, yard.log, yard.offline, yard.t0 = cut, [], set(), time.perf_counter()
    frames, stop, done = [], threading.Event(), [None]

    def film():
        while not stop.is_set():
            frames.append((yard.now(), _frame(yard, done[0])))
            time.sleep(0.1)

    th = threading.Thread(target=film, daemon=True)
    th.start()
    r = agentcore.invoke_agent_runtime(
        agentRuntimeArn=planner_arn,
        runtimeSessionId=f"shift-{uuid.uuid4().hex}",
        payload=json.dumps({"prompt": goal, "min_devices": min_devices}).encode(),
    )
    out = json.loads(r["response"].read())
    done[0] = yard.now()
    t_end = time.perf_counter()
    while any(dev.status().get("running") for dev in yard.devices.values()) and time.perf_counter() - t_end < 10:
        time.sleep(0.2)
    time.sleep(1.5)
    stop.set()
    th.join()
    # rendering runs slower than real time: resample by timestamp so the clip plays in real time
    ts = [t for t, _ in frames]
    real = [frames[max(0, np.searchsorted(ts, k / 10, side="right") - 1)][1] for k in range(int(ts[-1] * 10) + 1)]
    imageio.mimsave(clip, real, fps=10, codec="libx264", quality=7, macro_block_size=8)
    return out, yard.outcome(), clip
