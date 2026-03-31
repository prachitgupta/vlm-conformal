#!/usr/bin/env python3
"""Add simple variant_X-style observation and reasoning fields to a CSV dataset."""

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
GOAL_DELTA_RE = re.compile(
    r"Goal delta from current state is \(([-0-9.]+), ([-0-9.]+), ([-0-9.]+)\)",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", default=str(DEFAULT_INPUT_CSV))
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV))
    return parser.parse_args()


def infer_plan_trace(prompt: str) -> tuple[str, str]:
    sector_match = SECTOR_RE.search(prompt or "")
    clearance_match = CLEARANCE_RE.search(prompt or "")
    goal_delta_match = GOAL_DELTA_RE.search(prompt or "")

    sector = sector_match.group(1).replace("_", " ") if sector_match else "unknown"
    clearance = clearance_match.group(1).replace("_", " ") if clearance_match else "unknown"

    lateral_direction = "straight"
    if goal_delta_match:
        goal_dx = float(goal_delta_match.group(1))
        if goal_dx > 0.35:
            lateral_direction = "slight right"
        elif goal_dx < -0.35:
            lateral_direction = "slight left"

    if sector == "center":
        observation = f"Center path is {clearance}, so forward progress looks safe."
        reasoning = "Move straight toward goal."
    elif sector in {"left", "far left"}:
        observation = f"Left side looks tighter, so keep away and maintain safe clearance."
        reasoning = "Bias away from left."
    elif sector in {"right", "far right"}:
        observation = f"Right side looks tighter, so keep away and maintain safe clearance."
        reasoning = "Bias away from right."
    else:
        observation = f"Obstacle cue is {sector} with {clearance} clearance, so choose a smooth safe path."
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
            completion["reasoning"] = reasoning

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
