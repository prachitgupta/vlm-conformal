#!/usr/bin/env python3
"""Five independent structured-output trials for assignment 2."""

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
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


def run_repeated_sampling(args: argparse.Namespace | None = None) -> dict[str, Any]:
    parsed = args or parse_args()
    system_prompt = load_fixed_prompt(parsed.prompt_file)
    environment_prompt = get_environment_prompt(parsed)
    configure_scenario_from_prompt(environment_prompt)
    client = create_client()

    results: list[dict[str, Any]] = []
    for trial in range(1, parsed.trials + 1):
        solution = query_solution(
            client,
            system_prompt=system_prompt,
            messages=[{"role": "user", "content": environment_prompt}],
            model=parsed.model,
            temperature=parsed.temperature,
            max_tokens=parsed.max_tokens,
        )
        verification = verify(solution)
        results.append(
            {
                "trial": trial,
                "solution": solution.model_dump(),
                "llm_response_text": solution.model_dump_json(indent=2),
                "verification": verification,
            }
        )
        print(f"Trial {trial}: {'Pass' if verification['pass'] else 'Fail'} | {verification['reason']}")

    output = {
        "model": parsed.model,
        "fixed_prompt_file": str(Path(parsed.prompt_file).expanduser().resolve()),
        "dataset_row_index": None if parsed.env_prompt_file else parsed.row_index,
        "environment_prompt": environment_prompt,
        "results": results,
        "pass_count": sum(1 for result in results if result["verification"]["pass"]),
        "trial_count": parsed.trials,
    }

    if parsed.output_json:
        output_path = Path(parsed.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(output, indent=2))
    return output


if __name__ == "__main__":
    run_repeated_sampling()
