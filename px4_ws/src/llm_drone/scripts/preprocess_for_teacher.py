#!/usr/bin/env python3
"""Add variant_X-style plan_trace fields to a CSV dataset without touching the source file."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_CSV = REPO_ROOT / "dataset" / "dataset_merged_without_reasoning.csv"
DEFAULT_OUTPUT_CSV = REPO_ROOT / "dataset" / "dataset_variant_x_preprocessed.csv"

SECTOR_RE = re.compile(r"Nearest obstacle sector is ([a-z_]+)", re.IGNORECASE)
CLEARANCE_RE = re.compile(r"clearance status is ([a-z_]+)", re.IGNORECASE)
HINT_RE = re.compile(r"Lateral planning hint: ([^.]+)", re.IGNORECASE)
GOAL_DELTA_RE = re.compile(
    r"Goal delta from current state is \(([-0-9.]+), ([-0-9.]+), ([-0-9.]+)\)",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", default=str(DEFAULT_INPUT_CSV))
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV))
    parser.add_argument(
        "--output-format",
        choices=("plan_trace", "reasoning"),
        default="plan_trace",
        help="Whether to write variant_X-style nested plan_trace or legacy top-level reasoning.",
    )
    return parser.parse_args()


def infer_plan_trace(prompt: str) -> tuple[str, str]:
    sector_match = SECTOR_RE.search(prompt or "")
    clearance_match = CLEARANCE_RE.search(prompt or "")
    hint_match = HINT_RE.search(prompt or "")
    goal_delta_match = GOAL_DELTA_RE.search(prompt or "")

    sector = sector_match.group(1).replace("_", " ") if sector_match else "unknown"
    clearance = clearance_match.group(1).replace("_", " ") if clearance_match else "unknown"
    hint = hint_match.group(1).strip() if hint_match else ""

    lateral_direction = "straight"
    if goal_delta_match:
        goal_dx = float(goal_delta_match.group(1))
        if goal_dx > 0.35:
            lateral_direction = "slight right"
        elif goal_dx < -0.35:
            lateral_direction = "slight left"

    if sector == "center" and clearance == "clear":
        observation = "Center corridor is clear, so moving straight toward the goal is safe."
        reasoning = "Straight path is clear."
    elif sector == "center":
        safer_side = "left" if "left side is clearer" in hint.lower() else "right" if "right side is clearer" in hint.lower() else lateral_direction
        observation = f"Center corridor is {clearance}, so a slight {safer_side} bias keeps safer clearance."
        reasoning = f"Bias {safer_side} from center."
    elif sector in {"left", "far left"}:
        observation = f"Left side is tighter with {clearance} clearance, so shifting right preserves safer spacing."
        reasoning = "Favor right side clearance."
    elif sector in {"right", "far right"}:
        observation = f"Right side is tighter with {clearance} clearance, so shifting left preserves safer spacing."
        reasoning = "Favor left side clearance."
    else:
        observation = f"Obstacle cue is {sector} with {clearance} clearance, so choose a smooth path that keeps progress safe."
        reasoning = f"Keep {lateral_direction} progress."

    return observation, reasoning


def main() -> int:
    args = parse_args()
    input_csv = Path(args.input_csv).expanduser().resolve()
    output_csv = Path(args.output_csv).expanduser().resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with input_csv.open(newline="", encoding="utf-8") as src, output_csv.open(
        "w", newline="", encoding="utf-8"
    ) as dst:
        reader = csv.DictReader(src)
        fieldnames = list(reader.fieldnames or [])
        extra_fields = ["observation", "reasoning", "plan_trace_json"]
        output_fields = fieldnames + [f for f in extra_fields if f not in fieldnames]
        writer = csv.DictWriter(dst, fieldnames=output_fields)
        writer.writeheader()

        for row in reader:
            observation, reasoning = infer_plan_trace((row.get("prompt") or "").strip())
            completion = json.loads(row["label_waypoints_json"])
            if args.output_format == "plan_trace":
                completion["plan_trace"] = {
                    "observation": observation,
                    "reasoning": reasoning,
                }
                completion.pop("reasoning", None)
            else:
                completion["reasoning"] = reasoning
                completion.pop("plan_trace", None)

            row["observation"] = observation
            row["reasoning"] = reasoning
            row["plan_trace_json"] = json.dumps(
                {"observation": observation, "reasoning": reasoning},
                separators=(",", ":"),
            )
            row["label_waypoints_json"] = json.dumps(completion, separators=(",", ":"))
            writer.writerow(row)

    print(f"Saved preprocessed dataset to {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
