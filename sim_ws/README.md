# Autonomous Racing Simulation & Contingency Control

This repository is a specialized fork of the [f1tenth_gym_ros](https://github.com/f1tenth/f1tenth_gym_ros) environment. It extends the base simulator with a modular software stack designed for multi-agent racing, trajectory prediction, and future integration with Contingency Model Predictive Control (MPC).

## Project Overview
The goal of this project is to provide a robust framework for testing adversarial autonomous racing. Key additions include:
* **Custom Workspace (`sim_ws`):** A collection of ROS2 packages for agent behavior and environment perception.
* **Prediction Engine:** A dedicated module for tracking and forecasting opponent trajectories.
* **Containerized Workflow:** A `docker-compose` setup to ensure the simulation environment is easily replicable across different machines.
* **Pre-configured Visualization:** Custom RViz2 settings to visualize sensor data and planned paths immediately.

---

## Repository Structure
The primary logic resides in the `sim_ws` directory:

* **`sim_ws/src/prediction`**: Logic for tracking and forecasting the state of other agents.
* **`sim_ws/src/ego_agent`**: Core control and planning for the primary vehicle.
* **`sim_ws/src/opponent_agent`**: Behavior logic for adversarial/competitor vehicles (multiple modes)
* **`docker-compose.yml`**: Orchestrates the simulation, ROS bridge, and custom nodes.
* **`config/`**: Contains `.rviz` files with pre-configured displays for debugging.

---

## Replication

### Prerequisites
* [Docker](https://docs.docker.com/get-docker/)
* [Docker Compose]

### 1. Start the Environment
From your host machine (where you cloned the repo):
```bash
cd autonomous_racing_sim
docker-compose up
# In every new terminal
docker exec -it f1tenth_gym_ros-sim-1 /bin/bash
# Launch sim
source install/local_setup.bash
ros2 launch f1tenth_gym_ros gym_bridge_launch.py
# Launch package executable (ex: nice opp)
source install/local_setup.bash
ros2 run opponent_agent nice_opp

