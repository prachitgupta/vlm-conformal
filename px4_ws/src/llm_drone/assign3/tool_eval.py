#!/usr/bin/env python3
"""Five-trial tool-augmented evaluation for Assignment 3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pipeline_common import (
    DEFAULT_MODEL,
    DEFAULT_PROBLEM_PROMPT_DIR,
    load_fixed_prompt,
    query_solution,
    serialize_trial,
)
from problem_set import all_problem_ids
from verifier import verify


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem-id", default="all")
    parser.add_argument("--mode", choices=("live", "mock"), default="live")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt-file", default="")
    parser.add_argument("--problem-prompt-dir", default=str(DEFAULT_PROBLEM_PROMPT_DIR))
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


def _problem_ids(problem_id: str) -> list[str]:
    return all_problem_ids() if str(problem_id).lower() == "all" else [str(problem_id).upper()]


def run_tool_eval(args: argparse.Namespace | None = None) -> dict[str, Any]:
    parsed = args or parse_args()
    system_prompt = load_fixed_prompt(parsed.prompt_file) if parsed.prompt_file else load_fixed_prompt()

    all_results: dict[str, Any] = {}
    for problem_id in _problem_ids(parsed.problem_id):
        trials: list[dict[str, Any]] = []
        for trial in range(1, parsed.trials + 1):
            solution, metadata = query_solution(
                problem_id=problem_id,
                mode=parsed.mode,
                system_prompt=system_prompt,
                model=parsed.model,
                temperature=parsed.temperature,
                max_tokens=parsed.max_tokens,
                trial_index=trial,
                tool_enabled=True,
                problem_prompt_dir=parsed.problem_prompt_dir,
            )
            verification = verify(solution, problem_id=problem_id)
            trials.append(
                serialize_trial(
                    problem_id=problem_id,
                    trial_index=trial,
                    solution=solution,
                    verification=verification,
                    metadata=metadata,
                )
            )
            print(f"{problem_id} tool trial {trial}: {'P' if verification['pass'] else 'F'} | {verification['reason']}")

        pass_count = sum(1 for trial in trials if trial["verification"]["pass"])
        all_results[problem_id] = {
            "trials": trials,
            "pass_count": pass_count,
            "trial_count": int(parsed.trials),
        }

    output = {
        "mode": parsed.mode,
        "model": parsed.model,
        "tool_enabled": True,
        "results": all_results,
    }

    if parsed.output_json:
        output_path = Path(parsed.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(output, indent=2))
    return output


if __name__ == "__main__":
    run_tool_eval()
