#!/usr/bin/env python3
"""Automatic verifier for the Assignment 3 five-problem UAV planning suite."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from llm_drone.llm.offline_ground_truth_support import (
    min_signed_distance_to_obstacles,
    polyline_is_collision_free,
)

from problem_set import (
    Waypoint,
    WaypointSolution,
    all_problem_ids,
    get_problem,
)


DEFAULT_SAMPLE_SPACING_M = 0.2
_CURRENT_PROBLEM_ID = "P1"


def configure_problem(problem_id: str) -> None:
    global _CURRENT_PROBLEM_ID
    _CURRENT_PROBLEM_ID = str(problem_id).upper()


def _distance(point_a: tuple[float, float], point_b: tuple[float, float]) -> float:
    return math.hypot(float(point_a[0] - point_b[0]), float(point_a[1] - point_b[1]))


def _problem_manifest(problem_id: str):
    return get_problem(problem_id).manifest()


def _solution_points(solution: Any) -> np.ndarray:
    return np.array(
        [[float(waypoint.x), float(waypoint.y)] for waypoint in solution.waypoints],
        dtype=float,
    )


def _check_format(solution: Any) -> dict[str, Any]:
    violations: list[str] = []

    try:
        waypoints = list(solution.waypoints)
    except Exception as exc:
        return {"pass": False, "violations": [f"Missing waypoint data: {exc}"]}

    if len(waypoints) != 5:
        violations.append(f"Expected 5 waypoints, got {len(waypoints)}")

    for index, waypoint in enumerate(waypoints):
        for axis in ("x", "y"):
            value = getattr(waypoint, axis, None)
            if value is None or not math.isfinite(float(value)):
                violations.append(f"Waypoint {index} has invalid {axis} value: {value}")

    selected = getattr(solution, "selected_waypoint_index", None)
    if not isinstance(selected, int) or not (0 <= selected <= 4):
        violations.append(f"selected_waypoint_index must be an integer in [0, 4], got {selected}")

    reasoning = getattr(solution, "reasoning", "")
    if not isinstance(reasoning, str) or not reasoning.strip():
        violations.append("reasoning must be a non-empty string")

    return {"pass": not violations, "violations": violations}


def verify(solution: Any, problem_id: str | None = None) -> dict[str, Any]:
    problem = get_problem(problem_id or _CURRENT_PROBLEM_ID)
    manifest = _problem_manifest(problem.problem_id)

    fmt = _check_format(solution)
    if not fmt["pass"]:
        return {
            "pass": False,
            "details": {"format": fmt},
            "reason": "FORMAT FAIL: " + "; ".join(fmt["violations"]),
        }

    points = _solution_points(solution)
    current_xy = tuple(float(v) for v in problem.current_position_ned[:2])
    goal_xy = tuple(float(v) for v in problem.goal_position_ned[:2])
    selected_index = int(solution.selected_waypoint_index)

    segment_lengths = [_distance(current_xy, tuple(points[0]))]
    for index in range(1, points.shape[0]):
        segment_lengths.append(_distance(tuple(points[index - 1]), tuple(points[index])))
    speeds = segment_lengths
    accelerations = [
        abs(speeds[index] - speeds[index - 1])
        for index in range(1, len(speeds))
    ]

    kinematic_violations: list[str] = []
    if segment_lengths[0] > problem.max_first_leg_m:
        kinematic_violations.append(
            f"First leg {segment_lengths[0]:.2f} m exceeds {problem.max_first_leg_m:.2f} m"
        )

    for index, length in enumerate(segment_lengths):
        if length > problem.max_speed_mps:
            kinematic_violations.append(
                f"Segment {index} length {length:.2f} m exceeds speed proxy {problem.max_speed_mps:.2f} m"
            )
        if index > 0 and length < problem.min_step_m:
            kinematic_violations.append(
                f"Segment {index} length {length:.2f} m is shorter than {problem.min_step_m:.2f} m"
            )
        if index > 0 and length > problem.max_step_m:
            kinematic_violations.append(
                f"Segment {index} length {length:.2f} m is longer than {problem.max_step_m:.2f} m"
            )

    for index, accel in enumerate(accelerations, start=1):
        if accel > problem.max_accel_mps2:
            kinematic_violations.append(
                f"Segment {index} accel change {accel:.2f} m exceeds {problem.max_accel_mps2:.2f} m"
            )

    kinematics = {
        "pass": not kinematic_violations,
        "segment_lengths_m": [round(value, 3) for value in segment_lengths],
        "implied_speed_proxy_mps": [round(value, 3) for value in speeds],
        "implied_accel_proxy_mps2": [round(value, 3) for value in accelerations],
        "violations": kinematic_violations,
    }

    start_dist = _distance(current_xy, goal_xy)
    selected_dist = _distance(tuple(points[selected_index]), goal_xy)
    final_dist = _distance(tuple(points[-1]), goal_xy)
    goal_progress_violations: list[str] = []
    if start_dist - selected_dist < problem.min_selected_progress_m:
        goal_progress_violations.append(
            f"Selected waypoint only improves goal distance by {start_dist - selected_dist:.2f} m"
        )
    if start_dist - final_dist < problem.min_final_progress_m:
        goal_progress_violations.append(
            f"Final waypoint only improves goal distance by {start_dist - final_dist:.2f} m"
        )

    goal_progress = {
        "pass": not goal_progress_violations,
        "d_start_m": round(start_dist, 3),
        "d_selected_m": round(selected_dist, 3),
        "d_final_m": round(final_dist, 3),
        "progress_selected_m": round(start_dist - selected_dist, 3),
        "progress_final_m": round(start_dist - final_dist, 3),
        "violations": goal_progress_violations,
    }

    path_xy_enu = np.column_stack([points[:, 1], points[:, 0]])
    min_clearance = min_signed_distance_to_obstacles(path_xy_enu, manifest.obstacles)
    collision_free = polyline_is_collision_free(
        path_xy_enu,
        manifest.obstacles,
        inflation_m=problem.min_clearance_m,
        sample_spacing_m=DEFAULT_SAMPLE_SPACING_M,
    )
    safety_violations: list[str] = []
    if not collision_free:
        safety_violations.append("Polyline intersects inflated obstacle geometry")
    if min_clearance < problem.min_clearance_m:
        safety_violations.append(
            f"Minimum signed clearance {min_clearance:.2f} m is below {problem.min_clearance_m:.2f} m"
        )

    safety = {
        "pass": not safety_violations,
        "min_signed_clearance_m": round(float(min_clearance), 3),
        "required_clearance_m": round(problem.min_clearance_m, 3),
        "violations": safety_violations,
    }

    details = {
        "problem_id": problem.problem_id,
        "difficulty": problem.difficulty,
        "format": fmt,
        "kinematics": kinematics,
        "goal_progress": goal_progress,
        "safety": safety,
        "scenario": {
            "current_position_ned": list(problem.current_position_ned),
            "goal_position_ned": list(problem.goal_position_ned),
            "obstacle_count": len(problem.obstacles),
        },
    }

    failures: list[str] = []
    if not kinematics["pass"]:
        failures.append("KINEMATICS: " + "; ".join(kinematics["violations"]))
    if not goal_progress["pass"]:
        failures.append("GOAL_PROGRESS: " + "; ".join(goal_progress["violations"]))
    if not safety["pass"]:
        failures.append("SAFETY: " + "; ".join(safety["violations"]))

    return {
        "pass": not failures,
        "details": details,
        "reason": "All criteria passed." if not failures else " | ".join(failures),
    }


def _manual_pass_case(problem_id: str) -> WaypointSolution:
    cases: dict[str, list[tuple[float, float]]] = {
        "P1": [(0.8, 0.05), (1.7, 0.10), (2.6, 0.18), (3.5, 0.25), (4.4, 0.32)],
        "P2": [
            (0.7677425758709568, -0.4696502268112924),
            (1.5348346788044571, -0.9403603783790986),
            (2.296578584877827, -1.41966348208806),
            (3.0978228089887185, -1.8209798385919425),
            (3.984519252916508, -1.8157225600378732),
        ],
        "P3": [(0.7, 0.40), (1.4, 0.90), (2.1, 1.30), (2.9, 1.60), (3.8, 1.70)],
        "P4": [
            (0.6761315597043255, 0.5940085080051075),
            (1.3523305204359974, 1.187940291761906),
            (2.0273448194638934, 1.7832159969328145),
            (2.698239599144272, 2.383130323801928),
            (3.4164096427972304, 2.9228814585938196),
        ],
        "P5": [(0.7, -0.10), (1.4, -0.40), (2.2, -0.80), (3.1, -1.10), (4.0, -1.20)],
    }
    return WaypointSolution(
        waypoints=[Waypoint(x=x, y=y) for x, y in cases[problem_id]],
        selected_waypoint_index=0,
        reasoning=f"Hand-crafted pass case for {problem_id}.",
    )


def _manual_fail_case(problem_id: str) -> WaypointSolution:
    problem = get_problem(problem_id)
    obstacle = problem.obstacles[0]
    fail_points = [
        Waypoint(x=float(obstacle.center_ned[0]), y=float(obstacle.center_ned[1])),
        Waypoint(x=float(obstacle.center_ned[0]), y=float(obstacle.center_ned[1])),
        Waypoint(x=float(obstacle.center_ned[0]) - 0.1, y=float(obstacle.center_ned[1])),
        Waypoint(x=float(problem.current_position_ned[0]) - 0.4, y=float(problem.current_position_ned[1])),
        Waypoint(x=float(problem.current_position_ned[0]) - 0.8, y=float(problem.current_position_ned[1])),
    ]
    return WaypointSolution(
        waypoints=fail_points,
        selected_waypoint_index=0,
        reasoning=f"Hand-crafted fail case for {problem_id}.",
    )


def run_manual_tests() -> dict[str, Any]:
    results: dict[str, Any] = {}
    for problem_id in all_problem_ids():
        pass_case = _manual_pass_case(problem_id)
        fail_case = _manual_fail_case(problem_id)
        pass_result = verify(pass_case, problem_id=problem_id)
        fail_result = verify(fail_case, problem_id=problem_id)
        results[problem_id] = {
            "pass_case_solution": pass_case.model_dump(),
            "pass_case_result": pass_result,
            "fail_case_solution": fail_case.model_dump(),
            "fail_case_result": fail_result,
        }
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem-id", default="all")
    parser.add_argument("--manual-tests", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.manual_tests or args.problem_id == "all":
        print(json.dumps(run_manual_tests(), indent=2))
        return
    configure_problem(args.problem_id)
    sample = _manual_pass_case(args.problem_id.upper())
    print(json.dumps(verify(sample, problem_id=args.problem_id), indent=2))


if __name__ == "__main__":
    main()
