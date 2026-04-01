#!/usr/bin/env python3
"""Verifier-guided self-refinement for assignment 2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pipeline import (
    DEFAULT_MODEL,
    create_client,
    get_environment_prompt,
    load_fixed_prompt,
    parse_args as parse_pipeline_args,
    query_solution,
)
from verifier import configure_scenario_from_prompt, verify


def parse_args() -> argparse.Namespace:
    base = parse_pipeline_args()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-file", default=base.prompt_file)
    parser.add_argument("--dataset-csv", default=base.dataset_csv)
    parser.add_argument("--row-index", type=int, default=base.row_index)
    parser.add_argument("--env-prompt-file", default=base.env_prompt_file)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=base.temperature)
    parser.add_argument("--max-tokens", type=int, default=base.max_tokens)
    parser.add_argument("--max-turns", type=int, default=3)
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


def _refinement_feedback(verification: dict[str, Any]) -> str:
    details = verification.get("details", {})
    return (
        "Your previous solution failed the automatic verifier.\n\n"
        f"Summary reason:\n{verification['reason']}\n\n"
        "Detailed verifier output:\n"
        f"{json.dumps(details, indent=2)}\n\n"
        "Please return a corrected solution that satisfies all constraints while keeping the same schema."
    )


def run_self_refinement(args: argparse.Namespace | None = None) -> dict[str, Any]:
    parsed = args or parse_args()
    system_prompt = load_fixed_prompt(parsed.prompt_file)
    environment_prompt = get_environment_prompt(parsed)
    configure_scenario_from_prompt(environment_prompt)
    client = create_client()

    messages = [{"role": "user", "content": environment_prompt}]
    turns: list[dict[str, Any]] = []

    for turn_index in range(1, parsed.max_turns + 1):
        solution = query_solution(
            client,
            system_prompt=system_prompt,
            messages=messages,
            model=parsed.model,
            temperature=parsed.temperature,
            max_tokens=parsed.max_tokens,
        )
        verification = verify(solution)
        turn_output = {
            "turn": turn_index,
            "solution": solution.model_dump(),
            "llm_response_text": solution.model_dump_json(indent=2),
            "verification": verification,
        }
        turns.append(turn_output)
        print(f"Turn {turn_index}: {'Pass' if verification['pass'] else 'Fail'} | {verification['reason']}")

        messages.append({"role": "assistant", "content": solution.model_dump_json()})
        if verification["pass"]:
            break
        messages.append({"role": "user", "content": _refinement_feedback(verification)})

    output = {
        "model": parsed.model,
        "fixed_prompt_file": str(Path(parsed.prompt_file).expanduser().resolve()),
        "dataset_row_index": None if parsed.env_prompt_file else parsed.row_index,
        "environment_prompt": environment_prompt,
        "turns": turns,
        "final_pass": turns[-1]["verification"]["pass"] if turns else False,
    }

    if parsed.output_json:
        output_path = Path(parsed.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(output, indent=2))
    return output


if __name__ == "__main__":
    run_self_refinement()
