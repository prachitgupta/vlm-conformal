# C++ MPC Migration Plan

## Goal

Keep the ROS topic interface identical while moving the hot path out of Python.
The preferred architecture is now:

- Subscribe: `/depth_camera`, `/fmu/out/vehicle_odometry`, `/fmu/out/vehicle_status`
- Publish: `/fmu/in/trajectory_setpoint`, `/fmu/in/offboard_control_mode`
- Preserve debug topics: `/mpc/trajectory`, `/mpc/trajectory_sequence`
- Keep `mpc_optimal_planner.py` as the only ROS planner node
- Move only the MPC solve path into a native C++ extension module

## Recommended Port Order

1. Keep the async solver orchestration in Python (`_mpc_step`, `_build_solver_request`, `_consume_solver_result`).
2. Port the numerical MPC core into the native module.
3. Port obstacle-map math only if profiling shows it is still hot.
4. Keep the full C++ ROS node only as an experimental scaffold, not the primary runtime path.

## Suggested Architecture

- `llm_drone/mpc_optimal_planner.py`
  - ROS publishers/subscribers/timers
  - PX4 setpoint republisher and offboard heartbeat
  - async request/result pipeline
  - calls into `llm_drone._mpc_native`
- `src/mpc_native.cpp`
  - solver request/response structs
  - native numerical solve path
  - releases the Python GIL while solving
- `src/mpc_optimal_planner_cpp.cpp`
  - optional full-node scaffold for future all-C++ migration

## First Functional Milestone

The first hybrid milestone is already the right shape:

- Python node builds the solve request
- native module solves the planar MPC subproblem
- Python node consumes the result and republishes to PX4

## Run Mode

Use the Python node and select the native backend explicitly:

- `ros2 run llm_drone mpc_optimal_planner --ros-args -p mpc_solver_backend:=cpp_osqp`

The legacy native subgradient path remains available as a fallback/debug backend:

- `ros2 run llm_drone mpc_optimal_planner --ros-args -p mpc_solver_backend:=cpp_subgradient`

Do not run both the Python planner and `mpc_optimal_planner_cpp` at the same time.
