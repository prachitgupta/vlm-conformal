#!/usr/bin/env python3
"""Generate teacher-labeled data from an offline merged dataset."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import instructor
from openai import OpenAI
from pydantic import BaseModel


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_CSV = REPO_ROOT / "dataset" / "dataset_merged_without_reasoning.csv"
DEFAULT_PROMPT_FILE = REPO_ROOT / "config" / "variant_X.txt"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "dataset"
DEFAULT_OUTPUT_NAME = "teacher_dataset.csv"
DEFAULT_QWEN_MODEL = "Qwen/Qwen2.5-72B-Instruct"
DEFAULT_OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")


class Waypoint(BaseModel):
    x: float
    y: float


class PlanTrace(BaseModel):
    observation: str
    reasoning: str


class PlannerOutput(BaseModel):
    waypoints: list[Waypoint]
    plan_trace: PlanTrace


def build_train_completion(response: PlannerOutput) -> str:
    """Match the completion shape expected by scripts/train.py."""
    return json.dumps(
        {
            "waypoints": [waypoint.model_dump() for waypoint in response.waypoints],
            "reasoning": response.plan_trace.reasoning,
        },
        separators=(",", ":"),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", default=str(DEFAULT_INPUT_CSV))
    parser.add_argument("--prompt-file", default=str(DEFAULT_PROMPT_FILE))
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME, help="Output CSV filename to create inside dataset/")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--teacher", choices=("qwen", "chatgpt"), default="qwen")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="token-abc123")
    parser.add_argument("--model", default="")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument("--max-rows", type=int, default=0)
    return parser.parse_args()


def load_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def create_client(args: argparse.Namespace) -> tuple[instructor.Instructor, str]:
    if args.teacher == "chatgpt":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        model_name = args.model or DEFAULT_OPENAI_MODEL
        return instructor.from_openai(OpenAI(api_key=api_key)), model_name

    model_name = args.model or DEFAULT_QWEN_MODEL
    return instructor.from_openai(
        OpenAI(
            base_url=args.base_url,
            api_key=args.api_key,
        )
    ), model_name


def main() -> int:
    args = parse_args()
    input_csv = Path(args.input_csv).expanduser().resolve()
    prompt_file = Path(args.prompt_file).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / args.output_name

    system_prompt = prompt_file.read_text(encoding="utf-8")
    fieldnames, rows = load_rows(input_csv)
    rows = rows[args.start_row :]
    if args.max_rows > 0:
        rows = rows[: args.max_rows]

    client, model_name = create_client(args)

    extra_fields = [
        "teacher_model",
        "teacher_prompt_file",
        "teacher_response_json",
        "teacher_waypoints_json",
        "teacher_plan_trace_json",
    ]
    output_fields = list(fieldnames) + [f for f in extra_fields if f not in fieldnames]

    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields)
        writer.writeheader()

        for index, row in enumerate(rows, start=args.start_row):
            env_prompt = (row.get("prompt") or "").strip()
            if not env_prompt:
                raise ValueError(f"Row {index} is missing the prompt column")

            response = client.chat.completions.create(
                model=model_name,
                response_model=PlannerOutput,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": env_prompt},
                ],
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )

            row["teacher_model"] = model_name
            row["teacher_prompt_file"] = str(prompt_file)
            row["teacher_response_json"] = response.model_dump_json()
            # Keep the dataset directly compatible with scripts/train.py.
            row["label_waypoints_json"] = build_train_completion(response)
            row["teacher_waypoints_json"] = json.dumps(
                [waypoint.model_dump() for waypoint in response.waypoints],
                separators=(",", ":"),
            )
            row["teacher_plan_trace_json"] = json.dumps(
                response.plan_trace.model_dump(),
                separators=(",", ":"),
            )
            writer.writerow(row)

            if (index - args.start_row + 1) % 10 == 0:
                print(f"Wrote {index - args.start_row + 1} rows to {output_csv}")

    print(f"Saved teacher dataset to {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
