"""The operator's signed resume after an e-stop. Only this releases a stopped robot: the planner has no resume tool,
and its role is denied the resume topic.

  python resume.py duck-3

It reads the current stop's epoch from the robot and signs "<device>:<epoch>", so a signature can't be replayed
against a later stop.
"""

import os
import sys
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
    e = link.call(dev, "epoch")
    if not e["estopped"]:
        raise SystemExit(f"{dev} is not stopped")
    print(link.call(dev, "resume", operator=OPERATOR, signature=key.sign(f"{dev}:{e['epoch']}".encode()).hex()))
finally:
    link.close()
