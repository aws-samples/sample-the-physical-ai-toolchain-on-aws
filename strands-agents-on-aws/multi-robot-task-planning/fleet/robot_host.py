"""One robot host: a ROS 2 (ros2_control) arm made a fleet device over AWS IoT Core. The safety rules live HERE.

  DEVICE=ros-arm ZONE=inspection-cell OPERATOR_KEY=<ed25519 public hex> python robot_host.py

Rules, each enforced on the robot, whatever the planner sends:
  1. a target outside a joint's URDF limits is refused, never clamped, and nothing moves;
  2. estop latches: every motion is refused until resumed;
  3. resume needs an operator's Ed25519 signature over this device and this stop's epoch (the planner holds no key);
  4. watchdog: if the commander of a running motion stops heartbeating, the arm holds where it is;
  5. motions are bounded (MAX_S) and run on the robot at 50 Hz, never one network round trip per tick.
"""

from __future__ import annotations

import os
import threading
import time
import uuid
import xml.etree.ElementTree as ET

MAX_S, HZ, QUIET_S = 30.0, 50.0, 1.5
POSE_TOL = 0.02  # rad: a scripted-move skill counts as done only if every joint ends this close to its last target


class Device:
    """The rules, independent of ROS and IoT so they can be tested alone. `arm` reads/writes joint positions."""

    def __init__(self, arm, name: str, zone: str, skills: dict, operator_key: str):
        self.arm, self.name, self.zone, self.skills = arm, name, zone, skills
        self.operator_key = bytes.fromhex(operator_key)
        self.estopped, self.epoch, self.task, self.seen = False, None, None, {}
        self.lock = threading.Lock()

    def describe(self) -> dict:
        return {
            "device": self.name,
            "zone": self.zone,
            "joints": self.arm.limits,
            "skills": {
                k: {
                    "seconds": v.get("seconds", sum(float(m[1]) for m in v.get("moves", []))),
                    "does": v.get("does", ""),
                }
                for k, v in self.skills.items()
            },
            "rpcs": ["describe", "observe", "act", "skill", "status", "estop", "resume"],
        }

    def observe(self) -> dict:
        return {"positions": self.arm.positions(), "estopped": self.estopped, "task": self.status()}

    def act(self, targets: dict, seconds: float, by: str) -> dict:
        return self._start("act", [(targets, seconds)], by)

    def skill(self, name: str, by: str) -> dict:
        if name not in self.skills:
            raise LookupError(f"no skill {name!r}; this robot offers {sorted(self.skills)}")
        spec = self.skills[name]
        if "policy" in spec:  # a trained policy that runs on this robot for a bounded time (rule 5)
            if not 0 < float(spec["seconds"]) <= MAX_S:
                raise ValueError(f"skill {name!r} runs {spec['seconds']} s; at most {MAX_S}")
            return self._start(name, [], by, policy=spec)
        out = self._start(name, spec["moves"], by)
        final = {}
        for targets, _seconds in spec["moves"]:  # the pose the skill should end in: each joint's last target
            final.update({j: float(v) for j, v in targets.items()})
        self.task["pose_target"] = final
        return out

    def _start(self, name: str, moves: list, by: str, policy: dict | None = None) -> dict:
        for targets, seconds in moves:  # rule 1 and 5: check every point before anything moves
            for j, v in targets.items():
                if j not in self.arm.limits:
                    raise LookupError(f"no joint {j!r}; joints are {sorted(self.arm.limits)}")
                lo, hi = self.arm.limits[j]
                if not lo <= float(v) <= hi:
                    raise ValueError(f"{j}={v} is outside its range [{lo}, {hi}]; refused, nothing moved")
            if not 0 < float(seconds) <= MAX_S:
                raise ValueError(f"seconds={seconds} must be in (0, {MAX_S}]")
        with self.lock:
            if self.estopped:  # rule 2
                raise PermissionError("e-stop is latched; an operator must resume this robot")
            if self.task and self.task["running"]:
                raise RuntimeError(f"busy with {self.task['name']!r}")
            self.task = {"name": name, "running": True, "by": by, "stopped_by": None, "id": uuid.uuid4().hex[:8]}
            self.seen[by] = time.monotonic()
        threading.Thread(target=self._motion, args=(self.task, moves, policy), daemon=True).start()
        return {"started": name, "id": self.task["id"]}

    def _motion(self, task: dict, moves: list, policy: dict | None = None) -> None:
        """The motion thread. A body that raises while it moves (a bus error, or a body that moves by skills only)
        must not leave the task "running" for good: hold if the body still can, and end the task as a fault."""
        try:
            self._run(task, moves, policy)
        except Exception as exc:
            try:
                self.arm.hold()
            except Exception:
                pass
            task["running"], task["stopped_by"] = False, f"fault: {type(exc).__name__}: {exc}"

    def _ok(self, task: dict) -> str | None:
        """Why the motion must stop now (rules 2 and 4), or None to carry on."""
        if self.estopped:
            return "estop"
        if time.monotonic() - self.seen.get(task["by"], 0) > QUIET_S:
            return "watchdog: commander went quiet, holding"
        return None

    def _run(self, task: dict, moves: list, policy: dict | None = None) -> None:
        if policy is not None:  # the policy loop runs on the robot; the robot checks the rules while it runs
            why = self.arm.run_policy(policy["policy"], float(policy["seconds"]), lambda: self._ok(task))
            return self._end(task, why or "done")
        for targets, seconds in moves:
            start, n = self.arm.positions(), max(1, int(float(seconds) * HZ))
            for i in range(1, n + 1):
                if self.estopped:
                    return self._end(task, "estop")
                if time.monotonic() - self.seen.get(task["by"], 0) > QUIET_S:  # rule 4
                    self.arm.hold()
                    return self._end(task, "watchdog: commander went quiet, holding")
                u = i / n
                s = 3 * u * u - 2 * u * u * u
                self.arm.command({j: start[j] + (float(v) - start[j]) * s for j, v in targets.items()})
                time.sleep(1.0 / HZ)
        self._end(task, "done")

    def _end(self, task: dict, why: str) -> None:
        check = self.skills.get(task["name"], {}).get("check")
        try:
            if check is not None and why == "done":  # the postcondition, measured once, when the skill ends
                task["check"] = check()
            elif why == "done" and task.get(
                "pose_target"
            ):  # a scripted-move skill: did the joints reach its final pose?
                time.sleep(0.3)  # let the controller settle before reading
                pos = self.arm.positions()
                err = max(abs(pos[j] - v) for j, v in task["pose_target"].items())
                task["check"] = {"ok": err <= POSE_TOL, "max_joint_error_rad": round(err, 4), "tolerance_rad": POSE_TOL}
        except Exception as exc:  # a check that cannot measure is a failed check, and the skill still ends
            task["check"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        task["running"], task["stopped_by"] = False, why

    def heartbeat(self, who: str) -> None:
        self.seen[who] = time.monotonic()

    def status(self) -> dict:
        """The current or last motion. Once a skill has ended, also its postcondition: finished is not done."""
        if not self.task:
            return {"name": None, "running": False}
        return dict(self.task)

    def estop(self, by: str) -> dict:
        with self.lock:  # rule 2: latch first, then hold
            if not self.estopped:
                self.estopped, self.epoch = True, uuid.uuid4().hex
        self.arm.hold()
        return {"estopped": True, "epoch": self.epoch, "by": by}

    def resume(self, operator: str, signature: str = "") -> dict:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

        if not self.estopped:
            return {"estopped": False}
        try:  # rule 3: a signature over this device and this stop, by the operator's key
            Ed25519PublicKey.from_public_bytes(self.operator_key).verify(
                bytes.fromhex(signature or ""), f"{self.name}:{self.epoch}".encode()
            )
        except (InvalidSignature, ValueError):
            raise PermissionError("resume needs the operator's signature for this stop; refused")
        self.estopped, self.epoch = False, None
        return {"estopped": False, "resumed_by": operator}


class RosArm:
    """ros2_control: limits from /robot_description, state from /joint_states, writes to a forward position
    controller."""

    def __init__(self, controller: str = "forward_position_controller"):
        import rclpy
        from rclpy.qos import DurabilityPolicy, QoSProfile
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Float64MultiArray, String

        rclpy.init()
        self.node, self.msg, self.state, urdf = rclpy.create_node("fleet_device"), Float64MultiArray, {}, []
        self.node.create_subscription(
            String,
            "/robot_description",
            lambda m: urdf.append(m.data),
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )
        self.node.create_subscription(
            JointState, "/joint_states", lambda m: self.state.update(zip(m.name, m.position)), 10
        )
        self.pub = self.node.create_publisher(Float64MultiArray, f"/{controller}/commands", 10)
        threading.Thread(target=rclpy.spin, args=(self.node,), daemon=True).start()
        while not (urdf and self.state):
            time.sleep(0.05)
        joints = ET.fromstring(urdf[0]).findall("joint")
        self.limits = {
            j.get("name"): (float(j.find("limit").get("lower")), float(j.find("limit").get("upper")))
            for j in joints
            if j.find("limit") is not None
        }
        self.order = [j.get("name") for j in joints if j.get("name") in self.limits]  # the controller's joint order
        self.target = self.positions()

    def positions(self) -> dict:
        return {j: round(float(self.state[j]), 4) for j in self.order}

    def command(self, targets: dict) -> None:
        self.target.update(targets)
        self.pub.publish(self.msg(data=[float(self.target[j]) for j in self.order]))

    def hold(self) -> None:
        self.target = self.positions()
        self.command({})


def announce(link, tenant: str, dev) -> tuple:
    """Publish the robot's description once, as a retained message: the planner reads it when it needs it, and IoT
    Core keeps it while the robot is connected (the Link's Will clears it if the connection drops). It is republished
    only when it changes (an e-stop or a resume) or after a reconnect. Returns the clean-shutdown clear."""
    topic = f"{tenant}/presence/{dev.name}"

    def publish() -> None:
        link.publish(
            topic,
            {
                "device": dev.name,
                "zone": dev.zone,
                "skills": sorted(dev.skills),
                "estopped": dev.estopped,
                "ts": time.time(),
            },
            retain=True,
        )

    publish()
    link.on_reconnect.append(publish)
    last = [dev.estopped]

    def watch(stop: threading.Event) -> None:  # a local check, no traffic unless the description changed
        while not stop.wait(0.5):
            if dev.estopped != last[0]:
                last[0] = dev.estopped
                publish()

    return publish, watch, lambda: link.publish(topic, None, retain=True)


def serve(dev: Device, tenant: str = "site", stop: threading.Event | None = None) -> None:
    """Answer the device's rpcs over IoT Core, with its description published once (retained), until `stop` is set."""
    from iotlink import Link

    link = Link(f"robot-{dev.name}", tenant, will=f"{tenant}/presence/{dev.name}")
    handlers = {
        "describe": lambda a, who: dev.describe(),
        "observe": lambda a, who: dev.observe(),
        "status": lambda a, who: dev.status(),
        "act": lambda a, who: dev.act(a["targets"], a["seconds"], who),
        "skill": lambda a, who: dev.skill(a["name"], who),
        "estop": lambda a, who: dev.estop(who),
        "resume": lambda a, who: dev.resume(a.get("operator", ""), a.get("signature", "")),
        "epoch": lambda a, who: {"estopped": dev.estopped, "epoch": dev.epoch},
    }  # read-only: what the operator signs

    def on_rpc(topic: str, req: dict) -> None:
        rpc, who = topic.rsplit("/", 1)[-1], req.get("from", "?")
        try:
            out = {"id": req["id"], "result": handlers[rpc](req.get("args") or {}, who)}
        except Exception as exc:  # every refusal goes back to the caller, by name
            out = {"id": req["id"], "error": f"{type(exc).__name__}: {exc}"}
        link.publish(f"{tenant}/reply/{who}", out)
        print(
            time.strftime("%H:%M:%S"),
            dev.name,
            rpc,
            "by",
            who,
            "->",
            "ok" if "result" in out else out["error"],
            flush=True,
        )

    link.subscribe(f"{tenant}/{dev.name}/rpc/+", on_rpc)
    link.subscribe(f"{tenant}/heartbeat/+", lambda t, b: dev.heartbeat(t.rsplit("/", 1)[-1]))
    stop = stop or threading.Event()
    _, watch, clear = announce(link, tenant, dev)
    try:
        watch(stop)
    finally:  # stop set, or Ctrl-C: IoT Core sends no Will on a clean disconnect, so the robot clears its description
        clear()
        time.sleep(0.2)
        link.close()


def main() -> None:
    import json

    skills = json.loads(os.environ.get("SKILLS", "{}"))
    dev = Device(RosArm(), os.environ["DEVICE"], os.environ.get("ZONE", "cell"), skills, os.environ["OPERATOR_KEY"])
    print("device", dev.name, "up", dev.describe(), flush=True)
    serve(dev, os.environ.get("TENANT", "site"))


if __name__ == "__main__":
    main()
