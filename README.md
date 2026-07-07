# Centralized Learned mTSP for Three PX4 Drones

A centralized learned planner for static multi-depot MinMax multiple Traveling Salesperson Problem (mTSP) with three drones.

## Pipeline

```text
Global 2D task coordinates
→ centralized attention policy
→ three ordered task routes
→ PX4 ROS 2 Offboard executor
→ Gazebo simulation

Scope
3 drones
15 static, globally known tasks
Fixed 2D planner workspace
Objective: minimize the maximum closed-route length across drones
Each drone returns to its original depot
Main components
env/: mTSP instance generation and route evaluation
baselines/: greedy and OR-Tools baselines
models/: centralized attention policy
training/: imitation learning and RL fine-tuning
evaluation/: benchmarking and learned-policy evaluation
ros2/: learned mission planning and PX4 Offboard execution
Best current result
Method	Mean makespan
OR-Tools reference	28.96 m
Greedy baseline	31.55 m
Learned policy after RL	29.26 m

The learned policy achieved a +1.24% mean makespan gap relative to the stored OR-Tools reference and outperformed the greedy baseline.

Environment

Development runs in Docker with:

Ubuntu 24.04
ROS 2 Jazzy
PX4 v1.17 SITL
Gazebo Harmonic
Micro XRCE-DDS Agent
PyTorch with CUDA

Generated datasets, model checkpoints, plots, and local virtual environments are intentionally excluded from Git.

