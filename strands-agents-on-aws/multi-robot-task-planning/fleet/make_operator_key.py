"""Create the operator's Ed25519 key once. The private key stays with the operator (never in the planner, never
committed);
robot hosts get only the public key (operator_pub.hex) and use it to check a signed resume."""

import os
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

here = os.path.dirname(os.path.abspath(__file__))
pem = os.path.join(here, "operator_key.pem")
if os.path.exists(pem):
    raise SystemExit(f"{pem} already exists; not overwriting it")
k = Ed25519PrivateKey.generate()
fd = os.open(pem, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(fd, "wb") as f:
    f.write(
        k.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
    )
pub = k.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw).hex()
open(os.path.join(here, "operator_pub.hex"), "w").write(pub)
print("wrote operator_key.pem (keep it private) and operator_pub.hex")
