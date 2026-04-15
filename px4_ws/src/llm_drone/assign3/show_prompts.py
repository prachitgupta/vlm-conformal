#!/usr/bin/env python3
"""Print the fixed prompt and one problem-specific prompt used by the live pipeline."""

from __future__ import annotations

import argparse
import json

from pipeline_common import load_fixed_prompt, load_problem_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem-id", default="P1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = {
        "problem_id": args.problem_id.upper(),
        "fixed_prompt": load_fixed_prompt(),
        "problem_prompt": load_problem_prompt(args.problem_id),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
