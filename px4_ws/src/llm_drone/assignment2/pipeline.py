#!/usr/bin/env python3
"""Single-query OpenAI pipeline for assignment 2."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any
import instructor
from openai import OpenAI
from pydantic import BaseModel, Field, field_validator

from verifier import configure_scenario_from_prompt, verify


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPT_FILE = Path(__file__).resolve().with_name("fixed_prompt.txt")
DEFAULT_DATASET_CSV = REPO_ROOT / "dataset" / "dataset_merged_without_reasoning.csv"
DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")


class Waypoint(BaseModel):
    x: float = Field(..., description="North displacement in metres in the NED frame")
    y: float = Field(..., description="East displacement in metres in the NED frame")


class WaypointSolution(BaseModel):
    waypoints: list[Waypoint] = Field(
        ...,
        min_length=5,
        max_length=5,
        description="Exactly five local x/y waypoint targets in NED metres",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-file", default=str(DEFAULT_PROMPT_FILE))
    parser.add_argument("--dataset-csv", default=str(DEFAULT_DATASET_CSV))
    parser.add_argument("--row-index", type=int, default=0)
    parser.add_argument("--env-prompt-file", default="")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=800)
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


def create_client() -> instructor.Instructor:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    return instructor.from_openai(OpenAI(api_key=api_key))


def load_fixed_prompt(prompt_file: str | Path) -> str:
    return Path(prompt_file).expanduser().resolve().read_text(encoding="utf-8").strip()


def load_environment_prompt(dataset_csv: str | Path, row_index: int) -> str:
    csv_path = Path(dataset_csv).expanduser().resolve()
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader):
            if index == row_index:
                prompt = (row.get("prompt") or "").strip()
                if not prompt:
                    raise ValueError(f"Dataset row {row_index} has an empty prompt")
                return prompt
    raise IndexError(f"Row index {row_index} is out of range for {csv_path}")


def get_environment_prompt(args: argparse.Namespace) -> str:
    if args.env_prompt_file:
        return Path(args.env_prompt_file).expanduser().resolve().read_text(encoding="utf-8").strip()
    return load_environment_prompt(args.dataset_csv, args.row_index)


def query_solution(
    client: instructor.Instructor,
    *,
    system_prompt: str,
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
) -> WaypointSolution:
    return client.chat.completions.create(
        model=model,
        response_model=WaypointSolution,
        messages=[{"role": "system", "content": system_prompt}, *messages],
        temperature=temperature,
        max_tokens=max_tokens,
    )


def build_run_output(
    *,
    model: str,
    prompt_file: str,
    row_index: int | None,
    environment_prompt: str,
    solution: WaypointSolution,
    verification: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model": model,
        "fixed_prompt_file": prompt_file,
        "dataset_row_index": row_index,
        "environment_prompt": environment_prompt,
        "llm_response_text": solution.model_dump_json(indent=2),
        "solution": solution.model_dump(),
        "verification": verification,
    }


def run_pipeline(args: argparse.Namespace | None = None) -> dict[str, Any]:
    parsed = args or parse_args()
    system_prompt = load_fixed_prompt(parsed.prompt_file)
    environment_prompt = get_environment_prompt(parsed)
    configure_scenario_from_prompt(environment_prompt)

    client = create_client()
    solution = query_solution(
        client,
        system_prompt=system_prompt,
        messages=[{"role": "user", "content": environment_prompt}],
        model=parsed.model,
        temperature=parsed.temperature,
        max_tokens=parsed.max_tokens,
    )
    verification = verify(solution)
    output = build_run_output(
        model=parsed.model,
        prompt_file=str(Path(parsed.prompt_file).expanduser().resolve()),
        row_index=None if parsed.env_prompt_file else parsed.row_index,
        environment_prompt=environment_prompt,
        solution=solution,
        verification=verification,
    )

    if parsed.output_json:
        output_path = Path(parsed.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(output, indent=2))
    return output


if __name__ == "__main__":
    run_pipeline()
