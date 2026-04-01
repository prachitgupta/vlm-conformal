#!/usr/bin/env python3
"""Query OpenAI ChatGPT models with prompt variants and dataset-selected environment prompts."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

from openai import OpenAI


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VARIANTS_DIR = REPO_ROOT / "config" / "variants"
DEFAULT_DATASET_CSV = REPO_ROOT / "dataset" / "dataset_merged_without_reasoning.csv"
DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query OpenAI ChatGPT with system prompt variants."
    )
    parser.add_argument("--variants-dir", type=Path, default=DEFAULT_VARIANTS_DIR)
    parser.add_argument("--variant", default="all", help="Variant filename to run, or 'all'.")
    parser.add_argument("--dataset-csv", type=Path, default=DEFAULT_DATASET_CSV)
    parser.add_argument("--row-index", type=int, default=0)
    parser.add_argument("--env-prompt-file", default="")
    parser.add_argument("--api-key", default=None, help="Defaults to OPENAI_API_KEY.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=256)
    return parser.parse_args()


def create_client(api_key: str | None) -> OpenAI:
    resolved_api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not resolved_api_key:
        raise RuntimeError("OPENAI_API_KEY is not set and no --api-key was provided")
    return OpenAI(api_key=resolved_api_key)


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


def get_variant_paths(variants_dir: Path, variant: str) -> list[Path]:
    if variant == "all":
        return sorted(variants_dir.glob("*.txt"))
    return [variants_dir / variant]


def build_messages(system_prompt: str, env_prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": env_prompt.strip()},
    ]


def query_chatgpt(
    client: OpenAI,
    *,
    model: str,
    system_prompt: str,
    env_prompt: str,
    temperature: float,
    max_tokens: int,
) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=build_messages(system_prompt, env_prompt),
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""


def run_queries(args: argparse.Namespace) -> None:
    variants = get_variant_paths(args.variants_dir, args.variant)
    if not variants:
        raise FileNotFoundError(f"No variant prompts found under: {args.variants_dir}")

    env_prompt = get_environment_prompt(args)
    client = create_client(args.api_key)

    for variant_path in variants:
        system_prompt = variant_path.read_text(encoding="utf-8")
        print(f"\n===== {variant_path.name} =====")
        response_text = query_chatgpt(
            client,
            model=args.model,
            system_prompt=system_prompt,
            env_prompt=env_prompt,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        print(response_text)


def main() -> None:
    run_queries(parse_args())


if __name__ == "__main__":
    main()
