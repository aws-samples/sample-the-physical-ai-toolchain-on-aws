"""The operator's answer to a skill that needs approval (the SO-101 pick), sent out of band, not through the planner.

  python approve.py arm-1            show what arm-1 is waiting on, ask y/N, send the signed answer
  python approve.py arm-1 --deny     decline it

The answer is signed over "<device>:<request>:<decision>" with the operator's key. The planner's role is denied this
topic, and the robot ignores an answer without a valid signature.
"""

import os
import sys
import time
import uuid
from pathlib import Path

from cryptography.hazmat.primitives import serialization

from iotlink import Link

HERE = Path(__file__).resolve().parent
TENANT = os.environ.get("TENANT", "site")
OPERATOR = os.environ.get("OPERATOR", "operator")

dev = sys.argv[1]
key = serialization.load_pem_private_key((HERE / "operator_key.pem").read_bytes(), None)
link = Link(f"operator-{uuid.uuid4().hex[:8]}", TENANT)
try:
    st = link.call(dev, "status")
    if st.get("state") != "awaiting_approval":
        raise SystemExit(f"{dev} is not awaiting approval: {st}")
    if "--deny" in sys.argv:
        decision = "deny"
    else:
        ask = f"{dev} wants to run '{st['name']}' (waiting {st['waiting_s']} s). Approve? [y/N] "
        decision = "approve" if input(ask).strip().lower() == "y" else "deny"
    rid = st["request"]
    sig = key.sign(f"{dev}:{rid}:{decision}".encode()).hex()
    link.publish(
        f"{TENANT}/{dev}/approval", {"request": rid, "decision": decision, "operator": OPERATOR, "signature": sig}
    )
    time.sleep(0.8)
    print(decision, "sent for", dev, "| status now:", link.call(dev, "status"))
finally:
    link.close()
