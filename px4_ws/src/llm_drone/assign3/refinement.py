#!/usr/bin/env python3
"""Self-refinement with tools for Assignment 3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pipeline_common import DEFAULT_MODEL, DEFAULT_PROBLEM_PROMPT_DIR, load_fixed_prompt, query_solution
from verifier import verify


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem-id", default="P5")
    parser.add_argument("--mode", choices=("live", "mock"), default="live")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt-file", default="")
    parser.add_argument("--problem-prompt-dir", default=str(DEFAULT_PROBLEM_PROMPT_DIR))
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument("--max-turns", type=int, default=3)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


def run_refinement(args: argparse.Namespace | None = None) -> dict[str, Any]:
    parsed = args or parse_args()
    system_prompt = load_fixed_prompt(parsed.prompt_file) if parsed.prompt_file else load_fixed_prompt()

    all_trials: list[dict[str, Any]] = []
    for trial in range(1, parsed.trials + 1):
        turns: list[dict[str, Any]] = []
        verifier_feedback: dict[str, Any] | None = None

        for turn in range(1, parsed.max_turns + 1):
            use_tool = True
            solution, metadata = query_solution(
                problem_id=parsed.problem_id,
                mode=parsed.mode,
                system_prompt=system_prompt,
                model=parsed.model,
                temperature=parsed.temperature,
                max_tokens=parsed.max_tokens,
                trial_index=trial,
                tool_enabled=use_tool,
                verifier_feedback=verifier_feedback,
                problem_prompt_dir=parsed.problem_prompt_dir,
            )
            verification = verify(solution, problem_id=parsed.problem_id)
            turns.append(
                {
                    "turn": turn,
                    "solution": solution.model_dump(),
                    "llm_response_text": solution.model_dump_json(indent=2),
                    "verification": verification,
                    "metadata": metadata,
                }
            )
            print(
                f"{parsed.problem_id} refinement trial {trial} turn {turn}: "
                f"{'P' if verification['pass'] else 'F'} | {verification['reason']}"
            )
            if verification["pass"]:
                break
            verifier_feedback = verification

        all_trials.append(
            {
                "trial": trial,
                "turns": turns,
                "final_pass": turns[-1]["verification"]["pass"],
            }
        )

    output = {
        "mode": parsed.mode,
        "model": parsed.model,
        "problem_id": parsed.problem_id,
        "trials": all_trials,
        "pass_count": sum(1 for trial in all_trials if trial["final_pass"]),
        "trial_count": int(parsed.trials),
    }

    if parsed.output_json:
        output_path = Path(parsed.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(output, indent=2))
    return output


if __name__ == "__main__":
    run_refinement()
