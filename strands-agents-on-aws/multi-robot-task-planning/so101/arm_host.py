"""Robot host for a real SO-101 follower: the five rules from fleet/robot_host.py, a LeRobot body, and one skill,
`pick`, which runs the arm's ACT checkpoint here on the robot host and waits for an operator's signed approval first.

  ARM=arm-1 python so101/arm_host.py --preflight    read-only: motors, calibration, cameras, pose vs rest; zero writes
  ARM=arm-1 python so101/arm_host.py                serve arm-1 over AWS IoT Core
  ARM=arm-1 python so101/arm_host.py --replay 95    no hardware: replay a recorded episode (plumbing test only)

Each arm's port, calibration id, cameras, zone and checkpoint come from so101/arms.json (copy arms.example.json).
Why the approval is asynchronous: a request over IoT Core times out in seconds and an operator takes longer. If the
call timed out while the operator was still deciding, the planner would think the pick failed while the arm then
moved. So the pick answers at once with state awaiting_approval, and the operator answers on a separate topic.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
FLEET = HERE.parent / "fleet"
sys.path.insert(0, str(FLEET))
from robot_host import Device, announce  # noqa: E402  (the five rules; the retained description)

APPROVAL_S = float(os.environ.get("APPROVAL_S", 180.0))  # how long a pick waits for the operator
TENANT = os.environ.get("TENANT", "site")
TASK = "Pick up the cube and place it in the box"  # the task string the policies were trained with


def config(arm: str) -> dict:
    path = HERE / "arms.json"
    if not path.exists():
        raise SystemExit(f"{path} not found: copy arms.example.json to arms.json and fill in your arms")
    arms = json.loads(path.read_text())
    if arm not in arms:
        raise SystemExit(f"{arm} is not in {path}; it lists {sorted(arms)}")
    return {"device": arm, **arms[arm]}


def operator_public_hex() -> str:
    return (FLEET / "operator_pub.hex").read_text().strip()


class PickDevice(Device):
    """Device + an approval step for skills marked approval=True. The planner's skill rpc returns at once; the
    operator answers on <tenant>/<device>/approval with a signature over "<device>:<request>:<decision>"."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.pending: dict | None = None

    def act(self, targets: dict, seconds: float, by: str) -> dict:
        """Refused: on a real arm every motion is a skill the operator approves. A raw joint move from the planner
        would go around that approval, and the planner never needs one here."""
        raise PermissionError(
            f"{self.name} takes no raw joint moves; it runs approved skills only ({sorted(self.skills)})"
        )

    def skill(self, name: str, by: str) -> dict:
        spec = self.skills.get(name)
        if spec is None or not spec.get("approval"):
            return super().skill(name, by)
        with self.lock:
            if self.estopped:
                raise PermissionError("e-stop is latched; an operator must resume this robot")
            if self.pending or (self.task and self.task.get("running")):
                raise RuntimeError("busy: a pick is running or awaiting approval")
            self.pending = {"id": uuid.uuid4().hex[:8], "name": name, "by": by, "asked": time.monotonic()}
            self.seen[by] = time.monotonic()
        req = self.pending
        print(
            f"\n>>> APPROVAL NEEDED on {self.name}: '{name}' for {spec['seconds']:.0f} s, asked by {by}.\n"
            f">>> answer:  python fleet/approve.py {self.name}   (or --deny)\n",
            flush=True,
        )
        timer = threading.Timer(APPROVAL_S, self._expire, args=(req["id"],))
        timer.daemon = True  # a pending approval must not keep a stopped host alive until it expires
        timer.start()
        return {"state": "awaiting_approval", "request": req["id"], "skill": name}

    def _expire(self, rid: str) -> None:
        with self.lock:
            if self.pending and self.pending["id"] == rid:
                self.task = {
                    "name": self.pending["name"],
                    "running": False,
                    "by": self.pending["by"],
                    "id": rid,
                    "stopped_by": f"no operator answer in {APPROVAL_S:.0f} s; not started",
                }
                self.pending = None

    def answer(self, rid: str, decision: str, operator: str, signature: str) -> str:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

        try:
            Ed25519PublicKey.from_public_bytes(self.operator_key).verify(
                bytes.fromhex(signature or ""), f"{self.name}:{rid}:{decision}".encode()
            )
        except (InvalidSignature, ValueError):
            return "ignored: bad or missing operator signature"
        with self.lock:
            req = self.pending
            if not req or req["id"] != rid:
                return "ignored: no such pending request (expired, answered, or e-stopped)"
            self.pending = None
        if decision != "approve":
            self.task = {
                "name": req["name"],
                "running": False,
                "by": req["by"],
                "id": rid,
                "stopped_by": f"declined by {operator}",
            }
            return "declined"
        self.seen[req["by"]] = time.monotonic()
        super().skill(req["name"], req["by"])  # starts now; every rule still applies
        self.task["approved_by"] = operator
        return "approved; started"

    def status(self) -> dict:
        p = self.pending
        if p:
            return {
                "name": p["name"],
                "running": False,
                "state": "awaiting_approval",
                "request": p["id"],
                "waiting_s": round(time.monotonic() - p["asked"], 1),
            }
        return super().status()

    def estop(self, by: str) -> dict:
        with self.lock:
            if self.pending:
                self.task = {
                    "name": self.pending["name"],
                    "running": False,
                    "by": self.pending["by"],
                    "id": self.pending["id"],
                    "stopped_by": "e-stop while awaiting approval; not started",
                }
                self.pending = None
        return super().estop(by)


def serve(dev: PickDevice, tenant: str = TENANT, stop: threading.Event | None = None) -> None:
    from iotlink import Link

    link = Link(f"robot-{dev.name}", tenant, will=f"{tenant}/presence/{dev.name}")
    handlers = {
        "describe": lambda a, w: dev.describe(),
        "observe": lambda a, w: dev.observe(),
        "status": lambda a, w: dev.status(),
        "act": lambda a, w: dev.act(a["targets"], a["seconds"], w),
        "skill": lambda a, w: dev.skill(a["name"], w),
        "estop": lambda a, w: dev.estop(w),
        "resume": lambda a, w: dev.resume(a.get("operator", ""), a.get("signature", "")),
        "epoch": lambda a, w: {"estopped": dev.estopped, "epoch": dev.epoch},
    }

    def on_rpc(topic: str, req: dict) -> None:
        rpc, who = topic.rsplit("/", 1)[-1], req.get("from", "?")
        try:
            out = {"id": req["id"], "result": handlers[rpc](req.get("args") or {}, who)}
        except Exception as exc:  # every refusal goes back to the caller, by name
            out = {"id": req["id"], "error": f"{type(exc).__name__}: {exc}"}
        link.publish(f"{tenant}/reply/{who}", out)
        print(time.strftime("%H:%M:%S"), rpc, "by", who, "->", "ok" if "result" in out else out["error"], flush=True)

    def on_approval(_t: str, b: dict) -> None:
        print(
            time.strftime("%H:%M:%S"),
            "approval",
            b.get("decision"),
            "by",
            b.get("operator"),
            "->",
            dev.answer(b.get("request", ""), b.get("decision", ""), b.get("operator", "?"), b.get("signature", "")),
            flush=True,
        )

    link.subscribe(f"{tenant}/{dev.name}/rpc/+", on_rpc)
    link.subscribe(f"{tenant}/{dev.name}/approval", on_approval)
    link.subscribe(f"{tenant}/heartbeat/+", lambda t, b: dev.heartbeat(t.rsplit("/", 1)[-1]))
    print("device", dev.name, "up", dev.describe(), flush=True)
    stop = stop or threading.Event()
    _, watch, clear = announce(link, tenant, dev)  # described once (retained); the Link's Will clears it on a drop
    try:
        watch(stop)
    finally:  # stop set, or Ctrl-C: clear the description, then close, so the host exits
        clear()
        time.sleep(0.2)
        link.close()


def build(arm: str, replay_episode: int | None, load_policy: bool = True):
    from body_lerobot import PickPolicy, SO101Body, make_follower

    c = config(arm)
    if replay_episode is None:
        follower = make_follower(c["port"], c["calibration_id"], c["cameras"], c["max_relative_target"])
    else:
        from replay_follower import ReplayFollower, Replay

        follower = ReplayFollower(Replay(c["dataset"], replay_episode, list(c["cameras"])), list(c["cameras"]))
    policy = PickPolicy(c["policy"], list(c["cameras"])) if load_policy else None
    return c, follower, SO101Body(follower, c["poses"], c["calibration_id"], policy, TASK)


def preflight(arm: str, replay_episode: int | None) -> int:
    """Read-only: open the bus with every write method replaced by a refusal, ping/read the motors through the
    arm's calibration, read each camera once, compare the pose with the trained rest pose. Zero writes."""
    import numpy as np

    c, follower, body = build(arm, replay_episode, load_policy=False)
    bus = follower.bus
    for m in (
        "write",
        "sync_write",
        "_write",
        "enable_torque",
        "disable_torque",
        "configure_motors",
        "write_calibration",
    ):
        if hasattr(bus, m):
            setattr(
                bus,
                m,
                lambda *a, _m=m, **k: (_ for _ in ()).throw(RuntimeError(f"preflight is read-only: {_m} refused")),
            )
    ok = True
    print(f"[{c['device']}] port {c.get('port')}  calibration {c['calibration_id']}  cameras {c['cameras']}")
    bus.connect(handshake=True)
    pos = body.positions()
    for j in pos:
        d = pos[j] - body.rest[j]
        flag = (
            "  <-- more than 15 off the trained rest pose: move the arm to its rest pose first"
            if abs(d) > 15 and j != "gripper"
            else ""
        )
        lo, hi = body.limits[j]
        print(f"  {j:14s} {pos[j]:8.2f}  rest {body.rest[j]:8.2f}  range [{lo}, {hi}]{flag}")
        ok &= not flag
    for k, cam in follower.cameras.items():
        cam.connect()
        img = cam.read()
        want = (c["cameras"][k]["h"], c["cameras"][k]["w"])
        good = tuple(np.asarray(img).shape[:2]) == want
        from PIL import Image

        Image.fromarray(np.asarray(img)).save(HERE / f"preflight_{c['device']}_{k}.jpg")
        print(
            f"  camera {k}: {np.asarray(img).shape} want {want} {'ok' if good else 'WRONG SIZE'}",
            f"-> preflight_{c['device']}_{k}.jpg",
        )
        ok &= good
        cam.disconnect()
    writes = getattr(bus, "writes", 0)
    bus.disconnect(disable_torque=False) if replay_episode is None else bus.disconnect()
    print(f"PREFLIGHT {'PASS' if ok else 'CHECK THE FLAGS ABOVE'}  (bus writes: {writes})")
    return 0 if ok else 1


def main() -> int:
    arm = os.environ.get("ARM", "arm-1")
    replay = int(sys.argv[sys.argv.index("--replay") + 1]) if "--replay" in sys.argv else None
    if "--preflight" in sys.argv:
        return preflight(arm, replay)
    c, follower, body = build(arm, replay)
    follower.connect(calibrate=False)
    skills = {
        "pick": {
            "does": "pick the part on this arm's spot and drop it into the shared bin",
            "policy": {"repo": c["policy"]},
            "seconds": 15.0,
            "approval": True,
            "check": body.pick_check,
        }
    }
    dev = PickDevice(body, c["device"], c["zone"], skills, operator_public_hex())
    try:
        serve(dev)
    finally:
        follower.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
