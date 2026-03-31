#!/usr/bin/env python3
"""
Single-query LLM pipeline with instructor-enforced structured output.
Usage:  python pipeline.py
Requires: ANTHROPIC_API_KEY environment variable.
"""
import os, json
import anthropic
import instructor
from pydantic import BaseModel, Field, field_validator
from typing import List
from verifier import verify

# ── Pydantic schema ────────────────────────────────────────────────────────────

class Waypoint(BaseModel):
    x: float = Field(..., description="North (m) in NED frame")
    y: float = Field(..., description="East  (m) in NED frame")
    z: float = Field(..., description="Down  (m) in NED frame; negative = above ground")

class WaypointSolution(BaseModel):
    waypoints: List[Waypoint] = Field(
        ..., min_length=5, max_length=5,
        description="Exactly 5 collision-safe local waypoints in NED metres")
    selected_waypoint_index: int = Field(
        ..., ge=0, le=4,
        description="Index of the waypoint to execute next (0-indexed)")
    reasoning: str = Field(
        ..., max_length=200,
        description="Brief explanation (<= 35 words) of the chosen waypoint")

    @field_validator("waypoints")
    @classmethod
    def check_length(cls, v):
        if len(v) != 5:
            raise ValueError(f"Exactly 5 waypoints required, got {len(v)}")
        return v


# ── prompts ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are the motion planner for an autonomous quadrotor.

SECTION 1: CONTEXT AND INSTRUCTIONS
Task:
- Plan a short local trajectory of the next 5 waypoints (not just one waypoint).
- The 5-waypoint sequence should be consistent, smooth, collision-safe, and useful
  as a local trajectory label.
- Use all provided context, including odometry-derived state and obstacle/depth cues.

Frame and units:
- Coordinate frame is NED (North-East-Down), meters.
- x: North, y: East, z: Down.
- More negative z means higher altitude above ground.

Operational constraints:
- Hard safety: never place waypoints inside or too close to obstacles.
- Keep at least 1.5 m obstacle clearance when feasible.
- Keep waypoints dynamically feasible and smooth (no abrupt reversals or zig-zags).
- Prefer forward progress toward the goal.
- First waypoint must be local: at most 3.0 m from current position.
- Consecutive waypoint spacing should usually be 0.5 m to 3.0 m unless safety
  requires shorter moves.
- Respect altitude safety: keep z < 0 unless explicitly instructed otherwise.

Planning objective priority:
  1) Safety and feasibility
  2) Progress to goal
  3) Smoothness and efficiency
  4) Consistent 5-point local trajectory output

SECTION 2: FEW-SHOT DEMONSTRATIONS
Example A (clear corridor):
  current_position: [0.0, 0.0, -2.0], goal: [10.0, 0.0, -2.0], cue: clear ahead
  -> straight 5-point trajectory toward goal.

Example B (obstacle ahead, bypass right):
  current_position: [0.0, 0.0, -2.0], goal: [10.0, 0.0, -2.0],
  cue: near obstacle centered ahead, right side clearer
  -> steer right, then merge back toward goal line.

Example C (vertical avoidance):
  current_position: [0.0, 0.0, -2.0], goal: [8.0, 0.0, -2.0],
  cue: obstacle at current altitude directly ahead, vertical clearance available
  -> smooth climb in NED (more negative z) to clear obstacle, then level back.

SECTION 3: VERIFICATION LAYER (check before output)
1) Exactly 5 waypoints with numeric x, y, z.
2) selected_waypoint_index == 0.
3) First waypoint within 3.0 m of current position.
4) Smooth, feasible consecutive transitions (speed <= 5 m/s per 1 s interval).
5) Obstacle clearance >= 0.8 m maintained.
6) Trajectory progresses toward goal unless safety forces detour."""

USER_PROMPT = """Use the fixed planning policy exactly as defined in the system prompt.

Dynamic Environment Section (T(v)):
- Current position NED is (-0.14, 0.31, -2.43) m.
- Current velocity is (-0.01, -0.00, -0.00) m/s (speed 0.01 m/s).
- Goal position NED is (35.00, 3.00, 2.50) m.
- Goal delta from current state is (35.14, 2.69, 4.93) m with distance 35.59 m.
- Nearest obstacle in NED is at (-2.86, 4.74, -4.81) m, distance 5.72 m.
- Sector minimum distances [far_left, left, center, right, far_right]:
  [4.80, 4.81, 4.81, 4.81, 4.82] m.
- Nearest obstacle sector is far_left at 4.80 m; clearance status is CLEAR.
- Lateral planning hint: prefer center progress.

Numerical Environment Vector v:
{
  "current_position_ned_m":  [-0.14469, 0.30513, -2.42636],
  "current_velocity_mps":    [-0.00521, -0.00432, -0.00006],
  "goal_position_ned_m":     [35.0, 3.0, 2.5],
  "distance_to_goal_m":      35.59,
  "nearest_obstacle_distance_ned_m": 5.7227,
  "nearest_obstacle_position_ned_m": [-2.8593, 4.7420, -4.8127],
  "sector_min_m": {
    "far_left": 4.8048, "left": 4.8075, "center": 4.8101,
    "right":    4.8128, "far_right": 4.8155
  },
  "clearance_status": "clear"
}"""


# ── pipeline ──────────────────────────────────────────────────────────────────

def run_pipeline() -> dict:
    client = instructor.from_anthropic(
        anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    )
    solution: WaypointSolution = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": USER_PROMPT}],
        response_model=WaypointSolution,
    )
    result = verify(solution)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    run_pipeline()