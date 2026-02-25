# vlm-conformal

ROS 2 + PX4 workspace for comparing classical MPC and LLM-assisted local motion planning for drones using depth sensing.

This repository contains a PX4/Gazebo simulation workflow and a `llm_drone` package with:
- MPC local planners/controllers (classical baseline)
- LLM planners that output candidate waypoints from prompt-encoded sensor/state context
- Prompt recording utilities for cloud-LLM experiments
- Analysis/debug tools for comparing planner behavior

## Repository Layout

- `px4_ws/`: ROS 2 workspace (build/install/log + source packages)
- `px4_ws/src/llm_drone/`: main ROS 2 Python package
- `gazebo/`: simulation-related assets/configs (if used in your setup)
- `docs/`: notes and supporting documentation
- `scripts/`: helper scripts

## Key Components (`llm_drone`)

Main package path: `px4_ws/src/llm_drone/llm_drone`

- `mpc_local_planner.py`
  - Single-integrator style MPC local planner using depth-derived obstacle points and a FIFO local obstacle map.
  - Publishes trajectory setpoints and debug trajectory outputs.

- `mpc_vision_controller.py`
  - CVXPY-based vision MPC controller using point clouds.
  - Serves as a stronger optimization-based baseline.

- `llm_planner.py`
  - LLM-based planner (local Qwen via Ollama or OpenAI cloud API).
  - Builds an environment vector from odometry + depth, composes a prompt, requests an LLM response, parses JSON, and publishes the selected waypoint/setpoint.
  - Expected LLM output schema:
    - `waypoints` (candidate waypoint list)
    - `selected_waypoint_index`
    - `reasoning`

- `prompt_generator_samples.py`
  - ROS-backed prompt recorder for collecting timestamped prompt samples from live runs.
  - Useful for offline benchmarking of cloud LLMs without rerunning simulation.

- `performance_analyse.py`, `comparator.py`
  - Utilities for comparison and analysis of planner outputs.

- `dataset_generator.py`
  - Helpers for building datasets from simulation/planner data.

## Planner Data Flow (LLM Path)

1. Subscribe to vehicle odometry and depth camera.
2. Convert depth image -> obstacle points (camera -> body -> NED).
3. Maintain a FIFO local obstacle map consistent with MPC obstacle semantics.
4. Build a structured environment vector (position, velocity, goal, depth sector stats, nearest obstacle features).
5. Compose prompt (`system prompt` + deterministic text summary + numerical vector).
6. Query LLM backend (Qwen/Ollama or OpenAI).
7. Parse returned JSON and select the waypoint indicated by `selected_waypoint_index`.
8. Publish trajectory point / PX4 setpoint.

## ROS 2 Console Entrypoints

Defined in `px4_ws/src/llm_drone/setup.py`:

- `ros2 run llm_drone mpc`
- `ros2 run llm_drone mpc_local_planner`
- `ros2 run llm_drone mpc_sim`
- `ros2 run llm_drone mission_executor`
- `ros2 run llm_drone llm`
- `ros2 run llm_drone mpc_voxl`
- `ros2 run llm_drone llm_voxl`
- `ros2 run llm_drone performance_analyzer`
- `ros2 run llm_drone dataset_generator`
- `ros2 run llm_drone debug_pointcloud_obstacles`

## Prerequisites (Typical)

- Ubuntu 22.04
- ROS 2 Humble
- PX4 + Micro XRCE bridge setup
- Gazebo (matching your PX4 simulation stack)
- Python 3.10+
- Python packages used by the planners (examples):
  - `numpy`, `opencv-python`, `cv_bridge` (from ROS), `cvxpy`, `scipy`, `matplotlib`
  - `openai` (optional, only for cloud backend)

Note: `px4_msgs` must be available in the sourced ROS 2 environment for publishing PX4 `TrajectorySetpoint` messages. The code has limited fallbacks when `px4_msgs` is unavailable.

## Build

From the workspace root:

```bash
cd /home/prachit/Desktop/vlm-conformal/px4_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select llm_drone
source install/setup.bash
```

## Running (Typical Examples)

### MPC local planner

```bash
source /opt/ros/humble/setup.bash
cd /home/prachit/Desktop/vlm-conformal/px4_ws
source install/setup.bash
ros2 run llm_drone mpc_local_planner
```

### LLM planner (local Qwen via Ollama)

Make sure Ollama is running and a model is installed (for example `qwen2.5:3b`), then:

```bash
source /opt/ros/humble/setup.bash
cd /home/prachit/Desktop/vlm-conformal/px4_ws
source install/setup.bash
ros2 run llm_drone llm
```

Useful ROS params (examples):

```bash
ros2 run llm_drone llm --ros-args \
  -p llm_provider:=qwen \
  -p qwen_model:=qwen2.5:3b \
  -p goal_x:=35.0 -p goal_y:=3.0 -p goal_z:=2.5 \
  -p goal_frame:=ned
```

### LLM planner (OpenAI cloud backend)

Set `OPENAI_API_KEY` (or use the file-based key path expected by the code), then:

```bash
ros2 run llm_drone llm --ros-args \
  -p llm_provider:=openai \
  -p openai_model:=gpt-4o-mini
```

## Prompt Recording for Offline LLM Evaluation

To collect prompt samples from a live simulation run for later cloud-LLM testing:

```bash
source /opt/ros/humble/setup.bash
cd /home/prachit/Desktop/vlm-conformal/px4_ws
source install/setup.bash
python3 src/llm_drone/llm_drone/prompt_generator_samples.py
```

This records timestamped prompt text files containing:
- system prompt
- deterministic environment summary `T(v)`
- numerical environment vector `v`

These samples are useful for repeated sampling / self-refinement experiments without requiring a live simulator every time.

## Topics Used (Common)

The exact topic set varies by node, but commonly used topics include:

- Inputs
  - `/fmu/out/vehicle_odometry`
  - `/depth_camera` (depth image)
  - `/depth_camera/points` (point cloud, for `mpc_vision_controller.py`)

- Outputs
  - `/llm/trajectory`
  - `/mpc/trajectory`
  - `/fmu/in/trajectory_setpoint`
  - `/fmu/in/setpoint_velocity` (fallback in some environments)

## Notes for LLM Experiments

- `llm_planner.py` currently parses a JSON object from the model response and selects one waypoint using `selected_waypoint_index`.
- Prompt generation is grounded in sensor-derived obstacle features and a deterministic text translation of the environment vector.
- For reproducible evaluations, save the prompt, raw LLM response, and resulting waypoint selection for each trial.

## Development Notes

- The workspace may contain generated build artifacts (`build/`, `install/`, `log/`).
- Prompt sample directories under `generated_prompt_samples/` are generated data and can grow quickly.
- If you are comparing LLM vs MPC, align topic timestamps and goal/frame conventions (NED vs Gazebo/ENU) before interpreting errors.

## License

No top-level license file is currently declared in this repository. Add one before public redistribution if needed.
