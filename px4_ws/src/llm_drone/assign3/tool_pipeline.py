#!/usr/bin/env python3
"""Tool-augmented single-run pipeline for Assignment 3 (Option A)."""

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
    tool_worked_example,
)
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
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


def run_tool_pipeline(args: argparse.Namespace | None = None) -> dict[str, Any]:
    parsed = args or parse_args()
    system_prompt = load_fixed_prompt(parsed.prompt_file) if parsed.prompt_file else load_fixed_prompt()

    solution, metadata = query_solution(
        problem_id=parsed.problem_id,
        mode=parsed.mode,
        system_prompt=system_prompt,
        model=parsed.model,
        temperature=parsed.temperature,
        max_tokens=parsed.max_tokens,
        trial_index=1,
        tool_enabled=True,
        problem_prompt_dir=parsed.problem_prompt_dir,
    )
    verification = verify(solution, problem_id=parsed.problem_id)
    output = {
        "mode": parsed.mode,
        "model": parsed.model,
        "tool_enabled": True,
        "result": serialize_trial(
            problem_id=parsed.problem_id,
            trial_index=1,
            solution=solution,
            verification=verification,
            metadata=metadata,
        ),
        "worked_example": tool_worked_example(parsed.problem_id),
    }

    if parsed.output_json:
        output_path = Path(parsed.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(output, indent=2))
    return output


if __name__ == "__main__":
    run_tool_pipeline()
