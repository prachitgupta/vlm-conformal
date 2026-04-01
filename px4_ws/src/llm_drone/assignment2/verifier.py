#!/usr/bin/env python3
"""Automatic verifier for assignment 2 waypoint outputs."""

from __future__ import annotations

import math
import re
from typing import Any


V_MAX_MPS = 5.0
A_MAX_MPS2 = 3.0
DT_S = 1.0
R_SAFE_M = 0.8
MAX_FIRST_LEG_M = 3.0

_SCENARIO = {
    "current_position_xy": (-0.14469, 0.30513),
    "goal_position_xy": (35.0, 3.0),
    "nearest_obstacle_xy": (-2.8593, 4.7420),
}


def _distance(point_a: tuple[float, float], point_b: tuple[float, float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(point_a, point_b)))


def _extract_triplet(pattern: str, text: str) -> tuple[float, float, float]:
    match = re.search(pattern, text, re.IGNORECASE)
    if not match:
        raise ValueError(f"Could not parse prompt field with pattern: {pattern}")
    return tuple(float(value.strip()) for value in match.groups())


def configure_scenario_from_prompt(prompt_text: str) -> None:
    # Parse the current environment prompt so verify(solution) can stay string-free.
    current = _extract_triplet(
        r"Current position NED is\s*\(([-0-9.eE+]+),\s*([-0-9.eE+]+),\s*([-0-9.eE+]+)\)",
        prompt_text,
    )
    goal = _extract_triplet(
        r"Goal position NED is\s*\(([-0-9.eE+]+),\s*([-0-9.eE+]+),\s*([-0-9.eE+]+)\)",
        prompt_text,
    )
    obstacle = _extract_triplet(
        r"Nearest depth obstacle in NED is at\s*\(([-0-9.eE+]+),\s*([-0-9.eE+]+),\s*([-0-9.eE+]+)\)",
        prompt_text,
    )
    _SCENARIO["current_position_xy"] = (current[0], current[1])
    _SCENARIO["goal_position_xy"] = (goal[0], goal[1])
    _SCENARIO["nearest_obstacle_xy"] = (obstacle[0], obstacle[1])


def _check_format(solution: Any) -> dict[str, Any]:
    # Expected input: a typed solution object with waypoints, selected_waypoint_index, and reasoning.
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


def verify(solution: Any) -> dict[str, Any]:
    # Input: a WaypointSolution-like object. Output: {"pass": bool, "details": dict, "reason": str}.
    fmt = _check_format(solution)
    if not fmt["pass"]:
        reason = "FORMAT FAIL: " + "; ".join(fmt["violations"])
        return {"pass": False, "details": {"format": fmt}, "reason": reason}

    current = _SCENARIO["current_position_xy"]
    goal = _SCENARIO["goal_position_xy"]
    obstacle = _SCENARIO["nearest_obstacle_xy"]

    points = [(float(wp.x), float(wp.y)) for wp in solution.waypoints]
    selected_index = int(solution.selected_waypoint_index)

    segment_lengths = [_distance(current, points[0])]
    for index in range(1, len(points)):
        segment_lengths.append(_distance(points[index - 1], points[index]))
    speeds = [segment / DT_S for segment in segment_lengths]
    accelerations = [
        abs(speeds[index] - speeds[index - 1]) / DT_S
        for index in range(1, len(speeds))
    ]

    kinematic_violations: list[str] = []
    if segment_lengths[0] > MAX_FIRST_LEG_M:
        kinematic_violations.append(
            f"First leg {segment_lengths[0]:.2f} m exceeds locality limit {MAX_FIRST_LEG_M:.2f} m"
        )
    for index, speed in enumerate(speeds):
        if speed > V_MAX_MPS:
            kinematic_violations.append(
                f"Segment {index}: speed {speed:.2f} m/s exceeds {V_MAX_MPS:.2f} m/s"
            )
    for index, accel in enumerate(accelerations, start=1):
        if accel > A_MAX_MPS2:
            kinematic_violations.append(
                f"Segment {index}: accel {accel:.2f} m/s^2 exceeds {A_MAX_MPS2:.2f} m/s^2"
            )
    kinematics = {
        "pass": not kinematic_violations,
        "segment_lengths_m": [round(value, 3) for value in segment_lengths],
        "implied_speeds_mps": [round(value, 3) for value in speeds],
        "implied_accels_mps2": [round(value, 3) for value in accelerations],
        "violations": kinematic_violations,
    }

    d0 = _distance(current, goal)
    selected_distance = _distance(points[selected_index], goal)
    final_distance = _distance(points[-1], goal)
    goal_violations: list[str] = []
    if selected_distance >= d0:
        goal_violations.append(
            f"Selected waypoint does not reduce goal distance: {selected_distance:.2f} m >= {d0:.2f} m"
        )
    if final_distance >= d0:
        goal_violations.append(
            f"Final waypoint does not reduce goal distance: {final_distance:.2f} m >= {d0:.2f} m"
        )
    goal_progress = {
        "pass": not goal_violations,
        "d0_m": round(d0, 3),
        "d_selected_m": round(selected_distance, 3),
        "d_final_m": round(final_distance, 3),
        "progress_selected_m": round(d0 - selected_distance, 3),
        "progress_final_m": round(d0 - final_distance, 3),
        "violations": goal_violations,
    }

    clearances = [_distance(point, obstacle) for point in points]
    safety = {
        "pass": min(clearances) >= R_SAFE_M,
        "min_clearance_m": round(min(clearances), 3),
        "per_waypoint_clearance_m": [round(value, 3) for value in clearances],
        "r_safe_m": R_SAFE_M,
        "violations": []
        if min(clearances) >= R_SAFE_M
        else [f"Nearest-obstacle proxy clearance {min(clearances):.2f} m is below {R_SAFE_M:.2f} m"],
    }

    details = {
        "format": fmt,
        "kinematics": kinematics,
        "goal_progress": goal_progress,
        "safety": safety,
        "scenario": {
            "current_position_xy": list(current),
            "goal_position_xy": list(goal),
            "nearest_obstacle_xy": list(obstacle),
            # Safety uses the nearest obstacle parsed from the prompt as a lightweight proxy.
            "obstacle_model": "nearest-obstacle proxy parsed from environment prompt",
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
