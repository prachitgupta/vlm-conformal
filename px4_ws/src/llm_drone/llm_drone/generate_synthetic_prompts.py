#!/usr/bin/env python3
"""
Generate synthetic refinement prompts from eval_and_execute memory.

Input memory is produced by eval_and_execute_llm_waypoints.py via eval_memory_path JSONL.
Each output prompt asks a cloud LLM to refine a previous response based on PASS/FAIL feedback.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_MEMORY_PATH = Path("/tmp/llm_eval_memory.jsonl")
DEFAULT_SYSTEM_PROMPT = (Path(__file__).resolve().parent / "../config/llm_prompt.txt").resolve()
DEFAULT_OUTPUT_DIR = (Path(__file__).resolve().parent / "../config/generated_synthetic_prompts").resolve()


def load_optional_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text()


def load_memory_entries(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def _json_dump(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=True)


def compose_refinement_prompt(system_prompt: str, mem: dict[str, Any]) -> str:
    result = mem.get("result", {}) if isinstance(mem.get("result"), dict) else {}
    passed = bool(result.get("passed", False))
    source_prompt = str(mem.get("source_prompt") or "").strip()
    feedback = str(mem.get("refinement_feedback") or "").strip()
    llm_response_text = str(mem.get("llm_response_text") or "").strip()
    llm_response_json = mem.get("llm_response_json")
    metrics = result.get("metrics", {})

    response_block = llm_response_text
    if not response_block and llm_response_json is not None:
        response_block = _json_dump(llm_response_json)

    status = "PASS" if passed else "FAIL"

    sections: list[str] = []
    if system_prompt.strip():
        sections.append("=== BASE SYSTEM PROMPT ===\n" + system_prompt.rstrip())
    if source_prompt:
        sections.append("=== ORIGINAL USER PROMPT ===\n" + source_prompt)
    sections.append(
        "=== EVALUATION MEMORY ===\n"
        f"timestamp_utc: {mem.get('timestamp_utc', 'unknown')}\n"
        f"pass_fail: {status}\n"
        f"failed_checks: format={not bool(result.get('format_pass', True))}, "
        f"kinematic={not bool(result.get('kinematic_pass', True))}, "
        f"progress={not bool(result.get('progress_pass', True))}, "
        f"safety={not bool(result.get('safety_pass', True))}\n"
        f"feedback:\n{feedback if feedback else '(no feedback text available)'}"
    )
    if response_block:
        sections.append("=== PREVIOUS LLM RESPONSE ===\n" + response_block)
    if metrics:
        sections.append("=== METRICS JSON ===\n" + _json_dump(metrics))

    sections.append(
        "=== REFINEMENT TASK ===\n"
        "Refine the waypoint response so it passes evaluation.\n"
        "Output exactly one JSON object with keys: waypoints, selected_waypoint_index, reasoning.\n"
        "Hard requirements:\n"
        "- waypoints must be exactly 5 items\n"
        "- each waypoint has numeric x, y, z in NED meters\n"
        "- selected_waypoint_index must be 0\n"
        "- prioritize obstacle safety, goal progress, and smoothness\n"
        "- no markdown, no extra keys, no extra text"
    )
    return "\n\n".join(sections).strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic refinement prompts from eval memory JSONL.")
    parser.add_argument("--memory-path", type=Path, default=DEFAULT_MEMORY_PATH, help="Path to eval memory JSONL")
    parser.add_argument(
        "--system-prompt-file",
        type=Path,
        default=DEFAULT_SYSTEM_PROMPT,
        help="Optional base system prompt text file",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory for .txt prompts")
    parser.add_argument("--take-last", type=int, default=1, help="How many latest memory rows to convert")
    args = parser.parse_args()

    rows = load_memory_entries(args.memory_path)
    if not rows:
        print(f"No memory rows found in {args.memory_path}")
        return

    count = max(1, int(args.take_last))
    selected = rows[-count:]
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    system_prompt = load_optional_text(args.system_prompt_file)

    written = 0
    for mem in selected:
        now = datetime.now(timezone.utc)
        stamp = now.strftime("%Y%m%dT%H%M%S_%fZ")
        out_path = out_dir / f"{stamp}_synthetic_refine_prompt.txt"
        out_path.write_text(compose_refinement_prompt(system_prompt=system_prompt, mem=mem))
        written += 1
        print(f"Wrote {out_path}")

    print(f"Generated {written} synthetic prompt(s) in {out_dir}")


if __name__ == "__main__":
    main()
