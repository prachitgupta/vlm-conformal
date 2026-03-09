from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from ament_index_python.packages import get_package_share_directory

DEPTH_HFOV_DEG = 72.995
DEPTH_VFOV_DEG = 58.053
DEPTH_MIN_M = 0.2
DEPTH_MAX_M = 19.1
PROMPT_DEPTH_STATS_MAX_M = 20.0
OBS_FIFO_LEN = 200
MPC_DEPTH_SAMPLE_COUNT = 500

LLM_PLANNER_PROMPT_FILENAME = 'llm_planner_prompt.txt'
DATASET_PROMPT_FILENAME = 'llm_prompt.txt'
DEFAULT_SYSTEM_PROMPT = "You are a drone motion planning system. Generate safe trajectories."


def resolve_prompt_file(prompt_filename: str) -> Path:
    local_path = (Path(__file__).resolve().parent / f"../config/{prompt_filename}").resolve()
    if local_path.exists():
        return local_path

    share_dir = Path(get_package_share_directory('llm_drone'))
    return (share_dir / 'config' / prompt_filename).resolve()


def load_system_prompt(prompt_filename: str) -> str:
    return load_system_prompt_from_path(resolve_prompt_file(prompt_filename))


def load_system_prompt_from_path(path: Path | str) -> str:
    prompt_path = Path(path).expanduser().resolve()
    if not prompt_path.exists():
        return DEFAULT_SYSTEM_PROMPT
    return prompt_path.read_text()


def build_environment_vector(
    *,
    position_ned: np.ndarray,
    velocity_ned: np.ndarray,
    goal_ned: np.ndarray,
    depth_image: np.ndarray | None,
    latest_depth_obstacles_ned: np.ndarray,
    local_obstacle_snapshot: np.ndarray,
    depth_sample_count: int,
    waypoint_count: int,
) -> dict | None:
    if depth_image is None:
        return None

    pos = np.asarray(position_ned, dtype=float)
    vel = np.asarray(velocity_ned, dtype=float)
    goal = np.asarray(goal_ned, dtype=float)
    depth = np.asarray(depth_image, dtype=float)
    latest_obs = np.asarray(latest_depth_obstacles_ned, dtype=float)
    local_obs = np.asarray(local_obstacle_snapshot, dtype=float)

    valid_depth = depth[np.isfinite(depth)]
    valid_depth = valid_depth[(valid_depth > DEPTH_MIN_M) & (valid_depth < PROMPT_DEPTH_STATS_MAX_M)]

    depth_valid_fraction = float(valid_depth.size / depth.size) if depth.size > 0 else 0.0
    global_min = float(np.min(valid_depth)) if valid_depth.size > 0 else PROMPT_DEPTH_STATS_MAX_M
    global_mean = float(np.mean(valid_depth)) if valid_depth.size > 0 else PROMPT_DEPTH_STATS_MAX_M

    h, w = depth.shape
    sector_bounds = {
        "far_left": (0, int(0.2 * w)),
        "left": (int(0.2 * w), int(0.4 * w)),
        "center": (int(0.4 * w), int(0.6 * w)),
        "right": (int(0.6 * w), int(0.8 * w)),
        "far_right": (int(0.8 * w), w),
    }

    sector_min: dict[str, float] = {}
    sector_mean: dict[str, float] = {}
    for name, (start, end) in sector_bounds.items():
        patch = depth[:, start:end]
        patch_valid = patch[np.isfinite(patch)]
        patch_valid = patch_valid[(patch_valid > DEPTH_MIN_M) & (patch_valid < PROMPT_DEPTH_STATS_MAX_M)]
        sector_min[name] = float(np.min(patch_valid)) if patch_valid.size > 0 else PROMPT_DEPTH_STATS_MAX_M
        sector_mean[name] = float(np.mean(patch_valid)) if patch_valid.size > 0 else PROMPT_DEPTH_STATS_MAX_M

    nearest_sector = min(sector_min, key=sector_min.get)
    nearest_distance = float(sector_min[nearest_sector])
    if nearest_distance < 1.5:
        clearance_status = "blocked"
    elif nearest_distance < 3.0:
        clearance_status = "caution"
    else:
        clearance_status = "clear"

    goal_delta = goal - pos
    dist_to_goal = float(np.linalg.norm(goal_delta))
    speed = float(np.linalg.norm(vel))
    heading_to_goal_xy = float(np.arctan2(goal_delta[1], goal_delta[0]))

    if local_obs.ndim == 2 and local_obs.shape[0] > 0:
        rel = local_obs - pos[None, :]
        dists = np.linalg.norm(rel, axis=1)
        idx = int(np.argmin(dists))
        nearest_dist_ned = float(dists[idx])
        nearest_point_ned = [float(x) for x in local_obs[idx]]
        nearest_vec_ned = [float(x) for x in rel[idx]]
    else:
        nearest_dist_ned = PROMPT_DEPTH_STATS_MAX_M
        nearest_point_ned = [float(pos[0]), float(pos[1]), float(pos[2])]
        nearest_vec_ned = [PROMPT_DEPTH_STATS_MAX_M, 0.0, 0.0]

    latest_obs_count = int(latest_obs.shape[0]) if latest_obs.ndim == 2 else 0
    local_obs_count = int(local_obs.shape[0]) if local_obs.ndim == 2 else 0

    return {
        "current_position_ned_m": [float(x) for x in pos],
        "current_velocity_mps": [float(v) for v in vel],
        "goal_position_ned_m": [float(g) for g in goal],
        "goal_delta_m": [float(d) for d in goal_delta],
        "distance_to_goal_m": dist_to_goal,
        "speed_mps": speed,
        "heading_to_goal_xy_rad": heading_to_goal_xy,
        "obstacle_features_ned": {
            "pipeline": "mpc_consistent_depth_to_body_to_ned_fifo_local_map",
            "depth_camera_params": {
                "hfov_deg": DEPTH_HFOV_DEG,
                "vfov_deg": DEPTH_VFOV_DEG,
                "depth_min_m": DEPTH_MIN_M,
                "depth_max_m": DEPTH_MAX_M,
                "sample_count_per_frame_target": int(depth_sample_count),
            },
            "latest_depth_frame_obstacle_points_ned_count": latest_obs_count,
            "local_obstacle_fifo_count": local_obs_count,
            "nearest_obstacle_distance_ned_m": nearest_dist_ned,
            "nearest_obstacle_position_ned_m": nearest_point_ned,
            "nearest_obstacle_relative_ned_m": nearest_vec_ned,
            "mpc_nearest_obstacle_eq11_x_min_ned_m": nearest_point_ned,
        },
        "depth_features": {
            "valid_fraction": depth_valid_fraction,
            "global_min_m": global_min,
            "global_mean_m": global_mean,
            "sector_min_m": sector_min,
            "sector_mean_m": sector_mean,
            "nearest_obstacle_sector": nearest_sector,
            "nearest_obstacle_distance_m": nearest_distance,
            "clearance_status": clearance_status,
        },
        "waypoint_output_contract": {
            "frame": "ned",
            "count": int(waypoint_count),
            "planar_xy_only": True,
            "fixed_z_ned_m": float(goal[2]),
        },
    }


def translate_vector_to_nlp(vector: dict) -> str:
    if vector is None:
        return "Environment summary unavailable: no depth data."

    p = vector["current_position_ned_m"]
    vel = vector["current_velocity_mps"]
    g = vector["goal_position_ned_m"]
    d = vector["goal_delta_m"]
    obs_ned = vector["obstacle_features_ned"]
    depth = vector["depth_features"]
    output_contract = vector["waypoint_output_contract"]
    mins = depth["sector_min_m"]

    lateral_hint = "prefer center progress"
    side_clear = mins["right"] + mins["far_right"]
    side_left = mins["left"] + mins["far_left"]
    if side_clear > side_left + 0.8:
        lateral_hint = "right side is clearer"
    elif side_left > side_clear + 0.8:
        lateral_hint = "left side is clearer"

    return (
        "Environment section (T(v)):\n"
        f"- Current position NED is ({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f}) m.\n"
        f"- Current velocity is ({vel[0]:.2f}, {vel[1]:.2f}, {vel[2]:.2f}) m/s (speed {vector['speed_mps']:.2f} m/s).\n"
        f"- Goal position NED is ({g[0]:.2f}, {g[1]:.2f}, {g[2]:.2f}) m.\n"
        f"- Output contract: plan {output_contract['count']} waypoints in NED using x/y only; "
        f"every waypoint z must be fixed to goal z = {output_contract['fixed_z_ned_m']:.2f} m.\n"
        f"- Goal delta from current state is ({d[0]:.2f}, {d[1]:.2f}, {d[2]:.2f}) m with distance {vector['distance_to_goal_m']:.2f} m.\n"
        f"- MPC-consistent obstacle pipeline: {obs_ned['pipeline']}.\n"
        f"- Current depth frame obstacle points (NED): {obs_ned['latest_depth_frame_obstacle_points_ned_count']}; "
        f"FIFO local obstacle map size: {obs_ned['local_obstacle_fifo_count']} points. "
        f"Nearest obstacle in NED is at ({obs_ned['nearest_obstacle_position_ned_m'][0]:.2f}, "
        f"{obs_ned['nearest_obstacle_position_ned_m'][1]:.2f}, {obs_ned['nearest_obstacle_position_ned_m'][2]:.2f}) m, "
        f"relative vector ({obs_ned['nearest_obstacle_relative_ned_m'][0]:.2f}, "
        f"{obs_ned['nearest_obstacle_relative_ned_m'][1]:.2f}, {obs_ned['nearest_obstacle_relative_ned_m'][2]:.2f}) m, "
        f"distance {obs_ned['nearest_obstacle_distance_ned_m']:.2f} m.\n"
        f"- Depth validity fraction is {depth['valid_fraction']:.2f}. Global nearest obstacle distance is {depth['global_min_m']:.2f} m.\n"
        f"- Sector minimum distances [far_left, left, center, right, far_right] are "
        f"[{mins['far_left']:.2f}, {mins['left']:.2f}, {mins['center']:.2f}, {mins['right']:.2f}, {mins['far_right']:.2f}] m.\n"
        f"- Nearest obstacle sector is {depth['nearest_obstacle_sector']} at {depth['nearest_obstacle_distance_m']:.2f} m "
        f"and clearance status is {depth['clearance_status']}.\n"
        f"- Lateral planning hint: {lateral_hint}."
    )


def compose_final_prompt(env_vector: dict, env_text: str) -> str:
    waypoint_count = int(env_vector["waypoint_output_contract"]["count"])
    return f"""
Use the fixed planning policy exactly as defined in the system prompt.

Dynamic Environment Section (T(v)):
{env_text}

Numerical Environment Vector v:
{json.dumps(env_vector, indent=2)}

Sensor attachments:
1) Depth map visualization (red/yellow close, blue far)

Return only one JSON object with keys:
- waypoints
- selected_waypoint_index
- reasoning

All waypoint coordinates must be in NED.
Return exactly {waypoint_count} waypoints.
Only x and y should vary across the waypoint sequence.
For every waypoint, z must be exactly goal_position_ned_m[2].
""".strip() + "\n"
