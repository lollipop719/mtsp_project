## Entering Docker
newgrp docker
docker exec -it px4-stack bash

## Launching Gazebo and PX4
cd /workspace/PX4-Autopilot
make px4_sitl gz_x500

## Start bridge between ROS2 and PX4
MicroXRCEAgent udp4 -p 8888

## Checking the bridge
source /opt/ros/jazzy/setup.bash
source /opt/px4_ros_ws/install/setup.bash

ros2 topic list | grep /fmu

## QGround Control (outside docker)
flatpak run org.mavlink.qgroundcontrol

## Multi-drones
### Terminal 1
PX4_SYS_AUTOSTART=4001 PX4_SIM_MODEL=gz_x500 ./build/px4_sitl_default/bin/px4 -i 1
### Terminal 2
cd /workspace/PX4-Autopilot
PX4_GZ_STANDALONE=1 PX4_SYS_AUTOSTART=4001 PX4_GZ_MODEL_POSE="0,1" PX4_SIM_MODEL=gz_x500 ./build/px4_sitl_default/bin/px4 -i 2
### QGC and Agent
Open QGC, Agent (MicroXRCEAgent)
### Run python code
python3 two_drone_offboard.py