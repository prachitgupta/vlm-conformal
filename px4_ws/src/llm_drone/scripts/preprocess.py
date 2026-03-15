#!/usr/bin/env python3
"""Convert dataset.csv into a plain-text SFT corpus for non-reasoning Unsloth training."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

try:
    from llm_drone.llm.llm_prompt_common import (
        DATASET_PROMPT_FILENAME,
        load_system_prompt,
        serialize_prompt_bundle,
        split_prompt_bundle,
    )
except ImportError:
    DATASET_PROMPT_FILENAME = 'llm_prompt2d.txt'

    def load_system_prompt(prompt_filename: str) -> str:
        prompt_path = (Path(__file__).resolve().parents[1] / 'config' / prompt_filename).resolve()
        return prompt_path.read_text() if prompt_path.exists() else ''

    def serialize_prompt_bundle(system_prompt: str, user_prompt: str) -> str:
        return (
            '=== SYSTEM PROMPT ===\n'
            f'{str(system_prompt).strip()}\n\n'
            '=== USER PROMPT ===\n'
            f'{str(user_prompt).strip()}\n'
        )

    def split_prompt_bundle(prompt_text: str, fallback_system_prompt: str = '') -> tuple[str, str]:
        text = str(prompt_text or '').strip()
        system_marker = '=== SYSTEM PROMPT ===\n'
        user_marker = '\n\n=== USER PROMPT ===\n'
        if text.startswith(system_marker) and user_marker in text:
            system_text, _, user_text = text[len(system_marker):].partition(user_marker)
            return system_text.strip(), user_text.strip()
        return str(fallback_system_prompt or '').strip(), text


DEFAULT_INPUT = Path(__file__).resolve().parents[1] / 'dataset' / 'dataset.csv'
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / 'dataset' / 'dataset_unsloth.txt'
DEFAULT_RECORD_SEPARATOR = '\n\n<eos>\n\n'
DEFAULT_SYSTEM_PROMPT = load_system_prompt(DATASET_PROMPT_FILENAME)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Convert dataset.csv rows into prompt/response text records for Unsloth SFT.'
    )
    parser.add_argument('--input-csv', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--output-txt', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        '--record-separator',
        type=str,
        default=DEFAULT_RECORD_SEPARATOR,
        help='String inserted between training records in the output text file.',
    )
    parser.add_argument(
        '--include-reasoning',
        action='store_true',
        help='Keep the reasoning field in the assistant response instead of dropping it.',
    )
    parser.add_argument(
        '--include-evaluation',
        action='store_true',
        help='Keep the evaluation block from dataset_generator_sync labels in the assistant response.',
    )
    return parser.parse_args()


def normalize_response(raw_value: str) -> dict:
    parsed = json.loads(raw_value)
    if isinstance(parsed, dict) and 'waypoints' in parsed:
        waypoints = parsed['waypoints']
        reasoning = str(parsed.get('reasoning', '')).strip()
        evaluation = parsed.get('evaluation', None)
    elif isinstance(parsed, list):
        waypoints = parsed
        reasoning = ''
        evaluation = None
    else:
        raise ValueError('Unsupported label JSON format')

    normalized_waypoints = []
    for idx, waypoint in enumerate(waypoints):
        if isinstance(waypoint, dict):
            x = float(waypoint['x'])
            y = float(waypoint['y'])
        elif isinstance(waypoint, (list, tuple)) and len(waypoint) >= 2:
            x = float(waypoint[0])
            y = float(waypoint[1])
        else:
            raise ValueError(f'Unsupported waypoint format at index {idx}')
        normalized_waypoints.append({'x': x, 'y': y})

    return {
        'waypoints': normalized_waypoints,
        'reasoning': reasoning,
        'evaluation': evaluation,
    }


def build_record(
    prompt: str,
    response_obj: dict,
    include_reasoning: bool,
    include_evaluation: bool,
) -> str:
    response_payload = {
        'waypoints': response_obj['waypoints'],
    }
    if include_reasoning:
        response_payload['reasoning'] = str(response_obj.get('reasoning', '')).strip()
    if include_evaluation and response_obj.get('evaluation') is not None:
        response_payload['evaluation'] = response_obj['evaluation']

    return (
        '### Instruction:\n'
        f'{prompt.strip()}\n\n'
        '### Response:\n'
        f'{json.dumps(response_payload, separators=(",", ":"))}'
    )


def normalize_prompt(prompt: str) -> str:
    system_prompt, user_prompt = split_prompt_bundle(prompt, fallback_system_prompt=DEFAULT_SYSTEM_PROMPT)
    return serialize_prompt_bundle(system_prompt or DEFAULT_SYSTEM_PROMPT, user_prompt)


def main() -> int:
    args = parse_args()
    input_csv = args.input_csv.expanduser().resolve()
    output_txt = args.output_txt.expanduser().resolve()

    if not input_csv.exists():
        raise FileNotFoundError(f'Input CSV not found: {input_csv}')

    records: list[str] = []
    with input_csv.open('r', newline='') as fp:
        reader = csv.DictReader(fp)
        for row_idx, row in enumerate(reader, start=2):
            prompt = normalize_prompt((row.get('prompt') or '').strip())
            raw_response = (row.get('label_waypoints_json') or '').strip()
            if not prompt or not raw_response:
                continue
            try:
                response_obj = normalize_response(raw_response)
            except Exception as exc:
                raise ValueError(f'Failed to parse row {row_idx}: {exc}') from exc
            records.append(
                build_record(
                    prompt,
                    response_obj,
                    args.include_reasoning,
                    args.include_evaluation,
                )
            )

    output_txt.parent.mkdir(parents=True, exist_ok=True)
    output_txt.write_text(args.record_separator.join(records) + ('\n' if records else ''))
    print(f'Wrote {len(records)} records to {output_txt}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
