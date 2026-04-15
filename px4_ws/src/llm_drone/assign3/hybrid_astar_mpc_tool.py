#!/usr/bin/env python3
"""Pre-defined planning tool for Assignment 3 (hybrid A* + global optimization)."""

from __future__ import annotations

import argparse
import heapq
import json
import math
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from llm_drone.llm.offline_ground_truth_support import (
    _solve_global_reference_path_xy,
    compute_world_bounds_xy,
    densify_polyline,
    resample_polyline_for_dynamics,
    segment_is_collision_free,
    simplify_polyline,
)

from problem_set import Waypoint, WaypointSolution, get_problem


HEADING_BINS = 16
STEP_M = 0.75
GOAL_TOLERANCE_M = 0.9
MAX_EXPANSIONS = 50000


@dataclass(order=True)
class _QueueItem:
    priority: float
    cost: float
    state_key: tuple[int, int, int]


def _wrap_heading_idx(index: int) -> int:
    return int(index) % HEADING_BINS


def _heading_to_angle(index: int) -> float:
    return (2.0 * math.pi * float(_wrap_heading_idx(index))) / float(HEADING_BINS)


def _angle_to_heading(angle_rad: float) -> int:
    wrapped = float(angle_rad) % (2.0 * math.pi)
    return _wrap_heading_idx(int(round((wrapped / (2.0 * math.pi)) * HEADING_BINS)))


def _state_key(point_xy: np.ndarray, heading_idx: int, resolution_m: float) -> tuple[int, int, int]:
    return (
        int(round(float(point_xy[0]) / float(resolution_m))),
        int(round(float(point_xy[1]) / float(resolution_m))),
        _wrap_heading_idx(heading_idx),
    )


def _heuristic(point_xy: np.ndarray, goal_xy: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(goal_xy, dtype=float) - np.asarray(point_xy, dtype=float)))


def _interpolate_along_path(path_xy: np.ndarray, distance_m: float) -> np.ndarray:
    if path_xy.shape[0] == 0:
        raise ValueError("Cannot sample an empty path")
    if path_xy.shape[0] == 1 or distance_m <= 0.0:
        return np.array(path_xy[0], dtype=float, copy=True)

    remaining = float(distance_m)
    point = np.array(path_xy[0], dtype=float, copy=True)
    for idx in range(path_xy.shape[0] - 1):
        nxt = np.array(path_xy[idx + 1], dtype=float, copy=True)
        segment = nxt - point
        seg_len = float(np.linalg.norm(segment))
        if seg_len < 1e-9:
            point = nxt
            continue
        if seg_len >= remaining:
            return point + (remaining / seg_len) * segment
        remaining -= seg_len
        point = nxt
    return np.array(path_xy[-1], dtype=float, copy=True)


def _sample_waypoints_from_path(path_ned_xy: np.ndarray, count: int = 5, spacing_m: float = 0.9) -> list[Waypoint]:
    samples: list[Waypoint] = []
    for idx in range(count):
        point = _interpolate_along_path(path_ned_xy, distance_m=float(idx + 1) * float(spacing_m))
        samples.append(Waypoint(x=float(point[0]), y=float(point[1])))
    return samples


def _hybrid_astar_path(problem_id: str) -> np.ndarray:
    problem = get_problem(problem_id)
    manifest = problem.manifest()
    start_xy = np.array([problem.current_position_ned[1], problem.current_position_ned[0]], dtype=float)
    goal_xy = np.array([problem.goal_position_ned[1], problem.goal_position_ned[0]], dtype=float)

    x_min, x_max, y_min, y_max = compute_world_bounds_xy(manifest, margin_m=problem.min_clearance_m + 1.0)
    resolution_m = min(STEP_M, 0.5)

    start_heading = _angle_to_heading(
        math.atan2(goal_xy[1] - start_xy[1], goal_xy[0] - start_xy[0])
    )
    start_key = _state_key(start_xy, start_heading, resolution_m)

    frontier: list[_QueueItem] = []
    heapq.heappush(frontier, _QueueItem(priority=_heuristic(start_xy, goal_xy), cost=0.0, state_key=start_key))
    parents: dict[tuple[int, int, int], tuple[int, int, int] | None] = {start_key: None}
    positions: dict[tuple[int, int, int], np.ndarray] = {start_key: start_xy}
    g_score: dict[tuple[int, int, int], float] = {start_key: 0.0}

    expansions = 0
    goal_key: tuple[int, int, int] | None = None
    steering_options = (-2, -1, 0, 1, 2)

    while frontier and expansions < MAX_EXPANSIONS:
        current = heapq.heappop(frontier)
        current_pos = positions[current.state_key]
        current_heading = current.state_key[2]
        expansions += 1

        if _heuristic(current_pos, goal_xy) <= GOAL_TOLERANCE_M:
            goal_key = current.state_key
            break

        for delta_heading in steering_options:
            next_heading = _wrap_heading_idx(current_heading + delta_heading)
            theta = _heading_to_angle(next_heading)
            step = np.array([math.cos(theta), math.sin(theta)], dtype=float) * STEP_M
            next_pos = current_pos + step

            if not (x_min <= float(next_pos[0]) <= x_max and y_min <= float(next_pos[1]) <= y_max):
                continue
            if not segment_is_collision_free(
                current_pos,
                next_pos,
                manifest.obstacles,
                inflation_m=max(0.0, problem.min_clearance_m - 0.05),
                sample_spacing_m=0.1,
            ):
                continue

            next_key = _state_key(next_pos, next_heading, resolution_m)
            step_cost = STEP_M + 0.12 * abs(delta_heading)
            tentative = g_score[current.state_key] + step_cost
            if tentative >= g_score.get(next_key, float("inf")):
                continue

            parents[next_key] = current.state_key
            positions[next_key] = next_pos
            g_score[next_key] = tentative
            heading_penalty = 0.04 * abs(delta_heading)
            priority = tentative + _heuristic(next_pos, goal_xy) + heading_penalty
            heapq.heappush(frontier, _QueueItem(priority=priority, cost=tentative, state_key=next_key))

    if goal_key is None:
        raise RuntimeError(f"Hybrid A* failed to find a path for {problem_id}")

    path: list[np.ndarray] = []
    node = goal_key
    while node is not None:
        path.append(np.array(positions[node], dtype=float, copy=True))
        node = parents[node]
    path.reverse()

    if _heuristic(path[-1], goal_xy) > 1e-3:
        path.append(np.array(goal_xy, dtype=float, copy=True))
    return np.vstack(path)


def plan_problem(problem_id: str) -> dict[str, Any]:
    problem = get_problem(problem_id)
    manifest = problem.manifest()

    hybrid_path_enu = _hybrid_astar_path(problem.problem_id)
    simplified_enu = simplify_polyline(
        hybrid_path_enu,
        manifest.obstacles,
        inflation_m=problem.min_clearance_m,
        sample_spacing_m=0.2,
    )
    dense_enu = resample_polyline_for_dynamics(
        simplified_enu,
        dt_s=0.5,
        route_step_m=0.6,
        max_speed_mps=problem.max_speed_mps,
    )
    dense_enu = densify_polyline(dense_enu, step_m=0.4)
    optimized_enu, optimization_metadata = _solve_global_reference_path_xy(
        dense_enu,
        manifest.obstacles,
        planner_clearance_m=problem.min_clearance_m,
        dt_s=0.5,
        max_speed_mps=problem.max_speed_mps,
        max_accel_mps2=problem.max_accel_mps2,
        grid_resolution_m=0.25,
    )
    optimized_ned = np.column_stack([optimized_enu[:, 1], optimized_enu[:, 0]])
    waypoints = _sample_waypoints_from_path(optimized_ned, count=5, spacing_m=0.9)

    solution = WaypointSolution(
        waypoints=waypoints,
        selected_waypoint_index=problem.selected_waypoint_index,
        reasoning=(
            f"Hybrid A* found a collision-free corridor and the optimizer smoothed it "
            f"with status {optimization_metadata.get('status', 'unknown')}."
        ),
    )
    return {
        "problem_id": problem.problem_id,
        "solution": solution,
        "solution_dict": solution.model_dump(),
        "tool_metadata": {
            "planner_method": "hybrid_astar_plus_global_qp",
            "hybrid_path_points": int(hybrid_path_enu.shape[0]),
            "optimized_path_points": int(optimized_ned.shape[0]),
            "optimization_metadata": optimization_metadata,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem-id", default="P1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = plan_problem(args.problem_id)
    serializable = dict(result)
    serializable["solution"] = result["solution"].model_dump()
    print(json.dumps(serializable, indent=2))


if __name__ == "__main__":
    main()
