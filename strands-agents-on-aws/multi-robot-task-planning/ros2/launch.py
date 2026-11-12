"""Start the ROS 2 inspection arm (ros2_control, mock hardware) and its robot host in one container.

  docker build -t fleet-ros-arm:jazzy ros2/        once
  python ros2/launch.py                             start device ros-arm in zone inspection-cell

ROS_DOMAIN_ID (default 31) must not be used by another ROS 2 container on this Docker host: containers on one
Docker network share DDS discovery, and two controller managers on one domain make the spawner fail.
AWS credentials go into the container through the docker process's environment, never on its command line.
"""

import json
import os
import subprocess

import boto3

HERE = os.path.dirname(os.path.abspath(__file__))
FLEET = os.path.join(HERE, "..", "fleet")
DEVICE, ZONE, DOMAIN = "ros-arm", "inspection-cell", os.environ.get("ROS_DOMAIN_ID", "31")
REGION = os.environ.get("AWS_REGION", "us-west-2")
# Each skill is a bounded list of (joint targets, seconds) moves that runs ON the robot.
SKILLS = {
    "inspect": {
        "does": "turn the part under the arm through a slow inspection sweep",
        "seconds": 5.0,
        "moves": [[{"j1": 1.2, "j3": 0.5}, 2.0], [{"j4": 2.5}, 1.5], [{"j1": 0.0, "j3": 0.0, "j4": 0.0}, 1.5]],
    },
    "present": {
        "does": "hold the part up to the inspection camera",
        "seconds": 2.0,
        "moves": [[{"j2": 0.6, "j3": -0.6}, 2.0]],
    },
}

c = boto3.Session().get_credentials().get_frozen_credentials()
env = dict(
    os.environ,
    AWS_ACCESS_KEY_ID=c.access_key,
    AWS_SECRET_ACCESS_KEY=c.secret_key,
    AWS_SESSION_TOKEN=c.token or "",
    AWS_REGION=REGION,
    DEVICE=DEVICE,
    ZONE=ZONE,
    ROS_DOMAIN_ID=DOMAIN,
    SKILLS=json.dumps(SKILLS),
    IOT_ENDPOINT=os.environ.get("IOT_ENDPOINT")
    or boto3.client("iot", region_name=REGION).describe_endpoint(endpointType="iot:Data-ATS")["endpointAddress"],
    OPERATOR_KEY=open(os.path.join(FLEET, "operator_pub.hex")).read().strip(),
)
passthrough = []
for k in (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_REGION",
    "DEVICE",
    "ZONE",
    "ROS_DOMAIN_ID",
    "SKILLS",
    "OPERATOR_KEY",
    "IOT_ENDPOINT",
):
    passthrough += ["-e", k]
subprocess.run(["docker", "rm", "-f", DEVICE], capture_output=True)
subprocess.run(
    [
        "docker",
        "run",
        "-d",
        "--name",
        DEVICE,
        *passthrough,
        "-v",
        f"{os.path.abspath(FLEET)}:/fleet:ro",
        "-v",
        f"{HERE}:/ros2:ro",
        os.environ.get("ROS_ARM_IMAGE", "fleet-ros-arm:jazzy"),
        "bash",
        "-lc",
        "/ros2/start_arm.sh && source /opt/ros/jazzy/setup.bash && cd /fleet && /venv/bin/python -u robot_host.py",
    ],  # its log: docker logs ros-arm
    env=env,
    check=True,
    capture_output=True,
)
print("started", DEVICE, "in zone", ZONE)
