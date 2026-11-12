FROM ros:jazzy
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends ros-jazzy-ros2-control ros-jazzy-ros2-controllers \
    ros-jazzy-robot-state-publisher python3-venv python3-pip libgl1 libegl1 && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv --system-site-packages /venv && /venv/bin/pip install -q "awsiotsdk>=1.31" "boto3>=1.43.17" "cryptography>=42"
COPY . /ros2/
WORKDIR /ros2
