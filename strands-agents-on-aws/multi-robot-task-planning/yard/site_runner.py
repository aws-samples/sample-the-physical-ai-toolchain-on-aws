"""One stepping loop for a simulated site with several robots in ONE world.

Each robot that runs its own control loop in a shared world steps the shared physics, so every policy sees time
jump and the robots fall (RUNLOG T3). Calling world.run(..., steps=1) per tick does not work either: each call
starts a fresh run, so stateful policies would restart every step
and a walker falls. Here one world.run() covers the whole session, and each robot's policy is a Switchboard that
delegates
to whichever skill that robot is running now, or to its standing skill when idle.
"""

from __future__ import annotations

import threading
from typing import Any

from strands_robots import Policy
from strands_robots.policy.base import Policy as PolicyBase


class Switchboard(PolicyBase):
    """A robot's policy for the whole session: the current skill, else its stand skill."""

    units = "rad"

    def __init__(self, robot: Any, stand: str, hz: float = 50.0, envelopes: dict | None = None) -> None:
        self.robot, self.hz = robot, hz
        self.envelopes = envelopes or {}  # spec -> {"lo": [...], "hi": [...]}: each skill's own measured range + margin
        self._spec: str | None = None
        self._stand_spec = stand
        self._stand = Policy(stand)
        self._cur: Any = None
        self._name: str | None = None
        self._left = 0
        self._steps = 0
        self._lock = threading.Lock()
        self.last: dict[str, Any] = {"name": None, "running": False}
        # The run loop reads these hooks ONCE before step one, from this object: forward the stand skill's, or the
        # G1 runs with default PD gains instead of the ones holosoma was trained with, and falls.
        for hook in ("gains", "start_pose", "start_seconds", "hz"):
            if getattr(self._stand, hook, None) is not None:
                setattr(self, hook, getattr(self._stand, hook))
        self.realtime = True  # wall-clock pacing in sim, as on hardware, so dispatch timing is real

    def attach(self, robot: Any) -> None:
        for p in (self._stand,):
            if hasattr(p, "attach"):
                p.attach(robot)

    def start(self, name: str, spec: str, seconds: float) -> None:
        p = Policy(spec)
        if hasattr(p, "attach"):
            p.attach(self.robot)
        p.reset()
        with self._lock:
            self._cur, self._name, self._left, self._steps, self._spec = p, name, int(round(seconds * self.hz)), 0, spec
            self.last = {"name": name, "running": True, "steps": 0, "stopped_by": None, "fault": None}

    def stop(self, why: str = "stopped") -> None:
        with self._lock:
            if self._cur is not None:
                self.last = {
                    "name": self._name,
                    "running": False,
                    "steps": self._steps,
                    "stopped_by": why,
                    "fault": None,
                }
            self._cur = None
            self._stand.reset()

    def act(self, obs: Any, *, instruction: str | None = None) -> Any:
        with self._lock:
            cur = self._cur
            if cur is not None and self._left <= 0:
                self.last = {
                    "name": self._name,
                    "running": False,
                    "steps": self._steps,
                    "stopped_by": "seconds",
                    "fault": None,
                }
                self._cur, cur = None, None
                self._stand.reset()
            if cur is not None:
                self._left -= 1
                self._steps += 1
                self.last["steps"] = self._steps
        action = (cur or self._stand).act(obs, instruction=instruction)
        env = self.envelopes.get(self._spec) if cur is not None else None
        if env is not None and action is not None:
            import numpy as np

            a = np.asarray(action, float)
            bad = [
                (env["keys"][i], float(a[i]), env["lo"][i], env["hi"][i])
                for i in range(len(a))
                if not env["lo"][i] <= a[i] <= env["hi"][i]
            ]
            if bad:  # a policy commanding outside its own behaviour: stop the skill, stand, and say why
                k, v, lo, hi = bad[0]
                with self._lock:
                    self.last = {
                        "name": self._name,
                        "running": False,
                        "steps": self._steps,
                        "stopped_by": f"fault: {k}={v:.3f} outside the skill's envelope [{lo:.3f}, {hi:.3f}]",
                        "fault": True,
                    }
                    self._cur = None
                    self._stand.reset()
                return self._stand.act(obs, instruction=instruction)
        return action
