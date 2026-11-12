"""Run inside a container: print one /joint_states message as JSON (an independent read, not through the robot host)."""

import json
import rclpy
from sensor_msgs.msg import JointState

rclpy.init()
node, got = rclpy.create_node("read_once"), []
node.create_subscription(JointState, "/joint_states", lambda m: got.append(m), 10)
while not got:
    rclpy.spin_once(node, timeout_sec=0.5)
print(json.dumps({n: round(float(p), 3) for n, p in zip(got[0].name, got[0].position)}))
