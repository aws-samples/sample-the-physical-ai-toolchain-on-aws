#!/usr/bin/env bash
# ros2_control mock arm: robot_state_publisher (latched /robot_description), controller_manager, two spawners.
# The URDF goes in through a params file: as a -p argument rcl's argument parser rejects the XML.
set -e
source /opt/ros/jazzy/setup.bash
python3 - <<'PY'
import yaml
urdf = open("/ros2/arm.urdf").read()
yaml.safe_dump({"robot_state_publisher": {"ros__parameters": {"robot_description": urdf}}}, open("/tmp/rsp.yaml", "w"))
PY
ros2 run robot_state_publisher robot_state_publisher --ros-args --params-file /tmp/rsp.yaml > /tmp/rsp.log 2>&1 &
ros2 run controller_manager ros2_control_node --ros-args --params-file /ros2/controllers.yaml > /tmp/cm.log 2>&1 &
sleep 3
timeout 60 ros2 run controller_manager spawner joint_state_broadcaster forward_position_controller > /tmp/spawn.log 2>&1
echo "arm up"
