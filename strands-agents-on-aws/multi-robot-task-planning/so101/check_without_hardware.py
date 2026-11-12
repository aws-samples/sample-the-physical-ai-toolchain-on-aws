"""PLUMBING TEST, no hardware: drive an arm host that replays a recorded episode, over AWS IoT Core, the way the planner
and the operator would. Prints one JSON line per check. The arm does not move, so this says nothing about pick success.

  ARM=arm-2 APPROVAL_S=20 python so101/arm_host.py --replay 60     in one terminal
  ARM=arm-2 python so101/check_without_hardware.py                  in another
"""

import json, os, subprocess, sys, threading, time, uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "fleet"))
from cryptography.hazmat.primitives import serialization
from iotlink import Link

dev = os.environ.get("ARM", "arm-2")
TENANT = os.environ.get("TENANT", "site")
key = serialization.load_pem_private_key((HERE.parent / "fleet/operator_key.pem").read_bytes(), None)
me = Link(f"operator-test-{uuid.uuid4().hex[:8]}", TENANT)
beat = threading.Event()


def heartbeat():
    while not beat.is_set():
        me.publish(f"{TENANT}/heartbeat/{me.client_id}", {"ts": time.time()})
        time.sleep(0.5)


threading.Thread(target=heartbeat, daemon=True).start()


def out(**k):
    print(json.dumps(k, default=str), flush=True)


def call(rpc, **a):
    try:
        return me.call(dev, rpc, **a)
    except Exception as e:
        return {"error": str(e)[:200]}


def answer(rid, decision, sign=True):
    sig = key.sign(f"{dev}:{rid}:{decision}".encode()).hex() if sign else "00" * 64
    me.publish(
        f"{TENANT}/{dev}/approval",
        {"request": rid, "decision": decision, "operator": "test-operator", "signature": sig},
    )


def wait_done(limit=90):
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < limit:
        st = call("status")
        if not st.get("running") and st.get("state") != "awaiting_approval":
            return st, round(time.perf_counter() - t0, 1)
        time.sleep(0.5)
    return call("status"), limit


# 1. pick returns at once with awaiting_approval; status reports it
t0 = time.perf_counter()
r = call("skill", name="pick")
out(check="pick returns at once", reply=r, reply_s=round(time.perf_counter() - t0, 2))
out(check="status while waiting", status=call("status"))
out(check="second pick while waiting refused", reply=call("skill", name="pick"))
# 2. a forged answer is ignored
answer(r["request"], "approve", sign=False)
time.sleep(1.0)
out(check="unsigned approval ignored", status=call("status"))
# 3. signed approval starts it; the pick runs on the robot; postcondition at the end
t_ap = time.perf_counter()
answer(r["request"], "approve")
time.sleep(1.0)
out(check="after signed approval", status=call("status"))
st, s = wait_done()
out(check="pick finished", status=st, after_approval_s=round(time.perf_counter() - t_ap, 1))
# 4. deny path
r = call("skill", name="pick")
answer(r["request"], "deny")
time.sleep(1.0)
out(check="denied", status=call("status"))
# 5. no answer path (host started with a short APPROVAL_S)
r = call("skill", name="pick")
t_w = time.perf_counter()
st, s = wait_done(60)
out(check="no answer", status=st, waited_s=round(time.perf_counter() - t_w, 1))
# 6. e-stop mid-pick, refusals while latched, unsigned resume refused, signed resume works
r = call("skill", name="pick")
answer(r["request"], "approve")
time.sleep(6.0)
t_e = time.perf_counter()
e = call("estop")
out(check="estop mid-pick", reply=e, reply_s=round(time.perf_counter() - t_e, 2))
time.sleep(0.5)
out(check="status after estop", status=call("status"))
out(check="pick while latched", reply=call("skill", name="pick"))
out(check="unsigned resume", reply=call("resume", operator="planner", signature=""))
ep = call("epoch")
out(
    check="signed resume",
    reply=call("resume", operator="test-operator", signature=key.sign(f"{dev}:{ep['epoch']}".encode()).hex()),
)
# 7. a raw joint move from the planner is refused: this host runs approved skills only
out(check="act refused: approved skills only", reply=call("act", targets={"shoulder_pan": 10.0}, seconds=1.0))
# 8. watchdog: approve, then the commander goes quiet mid-pick
r = call("skill", name="pick")
answer(r["request"], "approve")
time.sleep(5.0)
beat.set()
time.sleep(4.0)
other = Link(f"operator-test-{uuid.uuid4().hex[:8]}", TENANT)
out(check="watchdog after commander quiet", status=other.call(dev, "status"))
other.close()
me.close()
