#!/usr/bin/env python3
"""Shared problem definitions and schemas for Assignment 3."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Any

from pydantic import BaseModel, Field, field_validator

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from llm_drone.llm.offline_ground_truth_support import ObstacleSpec, PoseENU, ScenarioManifest


DEFAULT_ALTITUDE_M = 2.5


class Waypoint(BaseModel):
    x: float = Field(..., description="North displacement in metres in the NED frame")
    y: float = Field(..., description="East displacement in metres in the NED frame")


class WaypointSolution(BaseModel):
    waypoints: list[Waypoint] = Field(
        ...,
        min_length=5,
        max_length=5,
        description="Exactly five ordered local x/y waypoint targets in NED metres",
    )
    selected_waypoint_index: int = Field(
        ...,
        ge=0,
        le=4,
        description="Index of the waypoint to execute next",
    )
    reasoning: str = Field(..., min_length=1, description="Short explanation of the chosen waypoint")

    @field_validator("waypoints")
    @classmethod
    def validate_waypoint_count(cls, value: list[Waypoint]) -> list[Waypoint]:
        if len(value) != 5:
            raise ValueError(f"Expected 5 waypoints, got {len(value)}")
        return value


@dataclass(frozen=True)
class Obstacle2D:
    name: str
    kind: str
    center_ned: tuple[float, float]
    radius_m: float | None = None
    size_ned_m: tuple[float, float] | None = None
    height_m: float = 3.0

    def prompt_line(self) -> str:
        if self.kind == "cylinder":
            return (
                f"- {self.name}: cylinder centered at NED ({self.center_ned[0]:.2f}, {self.center_ned[1]:.2f}) m "
                f"with radius {float(self.radius_m or 0.0):.2f} m."
            )
        if self.kind == "box":
            size = self.size_ned_m or (0.0, 0.0)
            return (
                f"- {self.name}: axis-aligned box centered at NED ({self.center_ned[0]:.2f}, {self.center_ned[1]:.2f}) m "
                f"with north/east footprint {size[0]:.2f} x {size[1]:.2f} m."
            )
        raise ValueError(f"Unsupported obstacle type: {self.kind!r}")

    def to_manifest_obstacle(self, altitude_m: float) -> ObstacleSpec:
        enu_x = float(self.center_ned[1])
        enu_y = float(self.center_ned[0])
        position = PoseENU(enu_x, enu_y, float(altitude_m), 0.0)
        if self.kind == "cylinder":
            if self.radius_m is None:
                raise ValueError(f"Cylinder obstacle missing radius: {self.name}")
            return ObstacleSpec(
                name=self.name,
                kind="cylinder",
                position=position,
                radius=float(self.radius_m),
                length=float(self.height_m),
            )
        if self.kind == "box":
            if self.size_ned_m is None:
                raise ValueError(f"Box obstacle missing size: {self.name}")
            north_len, east_len = self.size_ned_m
            # ENU boxes are expressed as (east, north, z).
            return ObstacleSpec(
                name=self.name,
                kind="box",
                position=position,
                size=(float(east_len), float(north_len), float(self.height_m)),
            )
        raise ValueError(f"Unsupported obstacle type: {self.kind!r}")


@dataclass(frozen=True)
class ProblemSpec:
    problem_id: str
    difficulty: str
    title: str
    statement: str
    deliverables: str
    difficulty_rationale: str
    current_position_ned: tuple[float, float, float]
    current_velocity_ned: tuple[float, float, float]
    goal_position_ned: tuple[float, float, float]
    obstacles: tuple[Obstacle2D, ...]
    expected_pattern: str
    max_speed_mps: float = 3.0
    max_accel_mps2: float = 2.5
    min_clearance_m: float = 0.8
    max_first_leg_m: float = 3.0
    min_step_m: float = 0.2
    max_step_m: float = 1.4
    min_selected_progress_m: float = 0.1
    min_final_progress_m: float = 1.0
    selected_waypoint_index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def altitude_m(self) -> float:
        return abs(float(self.goal_position_ned[2]))

    def manifest(self) -> ScenarioManifest:
        start = PoseENU(
            x=float(self.current_position_ned[1]),
            y=float(self.current_position_ned[0]),
            z=float(self.altitude_m),
            yaw=0.0,
        )
        goal = PoseENU(
            x=float(self.goal_position_ned[1]),
            y=float(self.goal_position_ned[0]),
            z=float(self.altitude_m),
            yaw=0.0,
        )
        return ScenarioManifest(
            scenario_id=self.problem_id.lower(),
            world_name=self.problem_id.lower(),
            start_pose_enu=start,
            goal_pose_enu=goal,
            fixed_altitude_m=float(self.altitude_m),
            obstacles=[obstacle.to_manifest_obstacle(self.altitude_m) for obstacle in self.obstacles],
            spawn_pose_enu=PoseENU(x=start.x, y=start.y, z=1.0, yaw=0.0),
            metadata={"assignment_problem_id": self.problem_id, **self.metadata},
        )


def _problem_prompt(problem: ProblemSpec) -> str:
    obstacle_lines = "\n".join(obstacle.prompt_line() for obstacle in problem.obstacles)
    return (
        f"Problem {problem.problem_id} ({problem.difficulty})\n"
        f"Title: {problem.title}\n\n"
        f"Problem statement:\n{problem.statement}\n\n"
        "Given state:\n"
        f"- Current position NED is ({problem.current_position_ned[0]:.2f}, {problem.current_position_ned[1]:.2f}, {problem.current_position_ned[2]:.2f}) m.\n"
        f"- Current velocity NED is ({problem.current_velocity_ned[0]:.2f}, {problem.current_velocity_ned[1]:.2f}, {problem.current_velocity_ned[2]:.2f}) m/s.\n"
        f"- Goal position NED is ({problem.goal_position_ned[0]:.2f}, {problem.goal_position_ned[1]:.2f}, {problem.goal_position_ned[2]:.2f}) m.\n"
        f"- Maintain at least {problem.min_clearance_m:.2f} m clearance from every listed obstacle.\n"
        f"- First waypoint must be within {problem.max_first_leg_m:.2f} m of the current position.\n"
        f"- Consecutive waypoint spacing should stay between {problem.min_step_m:.2f} m and {problem.max_step_m:.2f} m unless the last segment to the local horizon is shorter.\n"
        f"- Use {problem.selected_waypoint_index} as the selected waypoint index unless you have a stronger safety reason.\n\n"
        "Obstacle map:\n"
        f"{obstacle_lines}\n\n"
        f"Expected qualitative behavior:\n- {problem.expected_pattern}\n\n"
        f"Required deliverable:\n{problem.deliverables}\n"
    )


PROBLEMS: dict[str, ProblemSpec] = {
    "P1": ProblemSpec(
        problem_id="P1",
        difficulty="Easy",
        title="Clear Corridor Cruise",
        statement=(
            "Plan a smooth five-waypoint local trajectory for a quadrotor in an almost open corridor. "
            "The goal lies almost straight ahead and the nearest obstacles are well outside the direct path."
        ),
        deliverables=(
            "Return exactly five 2D NED waypoints, select the immediate next waypoint, and briefly explain "
            "why the trajectory is safe and goal-directed."
        ),
        difficulty_rationale=(
            "The direct corridor is unobstructed, so a competent LLM can usually succeed with a straight or "
            "nearly straight sequence."
        ),
        current_position_ned=(0.0, 0.0, -2.5),
        current_velocity_ned=(0.0, 0.0, 0.0),
        goal_position_ned=(8.0, 0.5, -2.5),
        obstacles=(
            Obstacle2D("north_tree", "cylinder", (4.3, 2.7), radius_m=0.7),
            Obstacle2D("south_crate", "box", (5.6, -2.8), size_ned_m=(1.2, 1.2)),
        ),
        expected_pattern="Track almost straight toward the goal with only a slight east bias.",
        min_final_progress_m=2.5,
    ),
    "P2": ProblemSpec(
        problem_id="P2",
        difficulty="Easy",
        title="Single-Obstacle Go-Around",
        statement=(
            "Plan through a corridor with one obstacle directly on the goal line. The vehicle must make a "
            "single lateral decision around the blocker and then recover the nominal heading."
        ),
        deliverables=(
            "Return a collision-free five-waypoint local plan that goes around the center obstacle while still "
            "reducing the goal distance at every useful step."
        ),
        difficulty_rationale=(
            "This problem only needs one clean side-step, so it is still fairly easy, but it is harder than P1 "
            "because a straight-line guess now fails."
        ),
        current_position_ned=(0.0, 0.0, -2.5),
        current_velocity_ned=(0.0, 0.0, 0.0),
        goal_position_ned=(10.0, 0.0, -2.5),
        obstacles=(
            Obstacle2D("center_pole", "cylinder", (3.6, 0.0), radius_m=1.0),
            Obstacle2D("north_guard", "cylinder", (6.0, 2.4), radius_m=0.7),
            Obstacle2D("south_guard", "cylinder", (6.1, -2.5), radius_m=0.7),
        ),
        expected_pattern="Make one deliberate lateral move around the blocker, then bend back toward the goal.",
        min_final_progress_m=2.2,
    ),
    "P3": ProblemSpec(
        problem_id="P3",
        difficulty="Hard",
        title="Alternating Slalom Corridor",
        statement=(
            "Plan a five-waypoint horizon through a staggered slalom of alternating obstacles inside a narrow lane. "
            "A naive single-turn plan is insufficient because the local path must set up at least two future turns."
        ),
        deliverables=(
            "Return a smooth five-waypoint sequence that threads the alternating obstacles while preserving dynamic "
            "feasibility and collision margin."
        ),
        difficulty_rationale=(
            "The planner must reason over multiple alternating turns instead of one local avoidance move, which tends "
            "to break shallow LLM reasoning."
        ),
        current_position_ned=(0.0, 0.0, -2.5),
        current_velocity_ned=(0.2, 0.0, 0.0),
        goal_position_ned=(14.0, 0.2, -2.5),
        obstacles=(
            Obstacle2D("left_wall", "box", (7.0, 4.3), size_ned_m=(18.0, 0.5)),
            Obstacle2D("right_wall", "box", (7.0, -4.3), size_ned_m=(18.0, 0.5)),
            Obstacle2D("slalom_1", "cylinder", (3.0, -1.1), radius_m=0.8),
            Obstacle2D("slalom_2", "cylinder", (5.4, 1.0), radius_m=0.8),
            Obstacle2D("slalom_3", "cylinder", (7.9, -0.9), radius_m=0.85),
            Obstacle2D("slalom_4", "cylinder", (10.4, 1.1), radius_m=0.8),
        ),
        expected_pattern="Commit early to a slalom line that alternates north/east offsets instead of reacting late.",
        min_final_progress_m=2.0,
        min_clearance_m=0.85,
    ),
    "P4": ProblemSpec(
        problem_id="P4",
        difficulty="Hard",
        title="Trap Exit Around a U-Shaped Blocker",
        statement=(
            "Plan out of a local trap created by a U-shaped obstacle cluster. The direct route toward the goal is "
            "blocked, and the vehicle must first bias upward to access the only viable channel."
        ),
        deliverables=(
            "Return five feasible local waypoints that begin escaping the trap while still making net progress toward "
            "the far goal."
        ),
        difficulty_rationale=(
            "The safe move initially looks less direct than the greedy move. Models that optimize only for immediate "
            "goal heading often drive into the blocked pocket."
        ),
        current_position_ned=(0.0, 0.0, -2.5),
        current_velocity_ned=(0.0, 0.0, 0.0),
        goal_position_ned=(13.0, 0.0, -2.5),
        obstacles=(
            Obstacle2D("center_wall", "box", (4.8, 0.0), size_ned_m=(1.4, 4.4)),
            Obstacle2D("upper_arm", "box", (7.0, 1.8), size_ned_m=(2.8, 0.9)),
            Obstacle2D("lower_arm", "box", (7.0, -1.8), size_ned_m=(2.8, 0.9)),
            Obstacle2D("lower_decoy", "cylinder", (8.8, -3.0), radius_m=0.8),
        ),
        expected_pattern="Climb toward positive east early, skirt around the upper arm, and avoid the tempting lower pocket.",
        min_final_progress_m=1.7,
        min_clearance_m=0.9,
    ),
    "P5": ProblemSpec(
        problem_id="P5",
        difficulty="Hard",
        title="Assignment 2 Long-Horizon Cluttered Corridor",
        statement=(
            "This is the Assignment 2 UAV local-waypoint task: generate five collision-safe local waypoints in the "
            "horizontal plane for a quadrotor using only the current state, goal, and a cluttered local obstacle map. "
            "The horizon is short relative to the total mission length, so the sequence must both remain safe now and "
            "set up future progress through dense clutter."
        ),
        deliverables=(
            "Return exactly five local x/y waypoints in NED, select the waypoint to execute next, and provide a brief "
            "reasoning string describing the chosen local route."
        ),
        difficulty_rationale=(
            "This is the original Assignment 2 formulation on a much longer, cluttered scene. It requires extended "
            "lookahead and a globally sensible local commitment, so the baseline model is expected to fail most often here."
        ),
        current_position_ned=(-0.14, 0.31, -2.5),
        current_velocity_ned=(0.0, 0.0, 0.0),
        goal_position_ned=(35.0, 3.0, -2.5),
        obstacles=(
            Obstacle2D("north_lane_wall", "box", (17.0, 6.0), size_ned_m=(38.0, 0.6)),
            Obstacle2D("south_lane_wall", "box", (17.0, -6.0), size_ned_m=(38.0, 0.6)),
            Obstacle2D("cluster_1", "cylinder", (4.2, 0.6), radius_m=0.8),
            Obstacle2D("cluster_2", "cylinder", (8.8, -1.2), radius_m=0.9),
            Obstacle2D("cluster_3", "cylinder", (12.2, 1.5), radius_m=0.8),
            Obstacle2D("cluster_4", "box", (16.2, 0.0), size_ned_m=(2.0, 2.8)),
            Obstacle2D("cluster_5", "cylinder", (20.4, -1.6), radius_m=0.9),
            Obstacle2D("cluster_6", "box", (24.2, 1.8), size_ned_m=(2.4, 1.2)),
            Obstacle2D("cluster_7", "cylinder", (28.1, 0.5), radius_m=0.85),
            Obstacle2D("cluster_8", "box", (31.2, 2.5), size_ned_m=(1.8, 1.0)),
        ),
        expected_pattern="Commit to a smooth long-corridor line that anticipates future blockers instead of only dodging the nearest one.",
        min_final_progress_m=2.0,
        min_clearance_m=0.9,
        metadata={"assignment2_final_problem": True},
    ),
}


def get_problem(problem_id: str) -> ProblemSpec:
    key = str(problem_id).upper()
    if key not in PROBLEMS:
        raise KeyError(f"Unknown problem id: {problem_id!r}")
    return PROBLEMS[key]


def all_problem_ids() -> list[str]:
    return list(PROBLEMS.keys())


def all_problems() -> list[ProblemSpec]:
    return [PROBLEMS[key] for key in all_problem_ids()]


def build_problem_prompt(problem_id: str) -> str:
    return _problem_prompt(get_problem(problem_id))


def straight_line_waypoints(problem_id: str, step_m: float = 0.9) -> WaypointSolution:
    problem = get_problem(problem_id)
    start = problem.current_position_ned[:2]
    goal = problem.goal_position_ned[:2]
    dx = goal[0] - start[0]
    dy = goal[1] - start[1]
    length = max(math.hypot(dx, dy), 1e-6)
    ux = dx / length
    uy = dy / length
    points = []
    current = (float(start[0]), float(start[1]))
    for _ in range(5):
        current = (current[0] + step_m * ux, current[1] + step_m * uy)
        points.append(Waypoint(x=current[0], y=current[1]))
    return WaypointSolution(
        waypoints=points,
        selected_waypoint_index=problem.selected_waypoint_index,
        reasoning="Greedy straight-line rollout toward the goal.",
    )
