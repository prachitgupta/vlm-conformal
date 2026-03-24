#!/usr/bin/env python3
"""Query local vLLM Qwen server with prompt variants and print raw model responses."""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

DEFAULT_ENV_PROMPT = """Environment section (T(v)):
- Current position NED is (-0.02, 0.01, -2.16) m.
- Current velocity is (-0.00, -0.00, -0.00) m/s (speed 0.00 m/s).
- Goal position NED is (0.00, 29.00, -2.50) m.
- Output contract: predict exactly 5 waypoint x/y pairs in NED. The planner will set every waypoint z to goal z = -2.50 m before execution.
- Goal delta from current state is (0.02, 28.99, -0.34) m with distance 28.99 m.
- Obstacle pipeline: depth_camera_to_body_to_ned_current_frame.
- Current depth frame obstacle points projected into NED: 11. Nearest depth obstacle in NED is at (-0.36, 4.15, -2.13) m, relative vector (-0.35, 4.15, 0.03) m, distance 4.16 m.
- Depth validity fraction is 0.61. Global nearest obstacle distance is 4.02 m.
- Sector minimum distances [far_left, left, center, right, far_right] are [4.37, 4.37, 4.02, 4.38, 4.38] m.
- Nearest obstacle sector is center at 4.02 m and clearance status is clear.
- Lateral planning hint: prefer center progress."""


def build_messages(system_prompt: str, env_prompt: str) -> list[dict[str, str]]:
    # Mirror train.py inference structure exactly: system prompt stays in the
    # system turn, and the environment prompt is the lone user turn.
    return [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": env_prompt.strip()},
    ]


def query_vllm(url: str, api_key: str, model: str, system_prompt: str, env_prompt: str, temperature: float, max_tokens: int) -> str:
    payload = {
        "model": model,
        "messages": build_messages(system_prompt, env_prompt),
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }

    request = urllib.request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=60) as response:
        body = response.read().decode("utf-8")
    parsed = json.loads(body)
    return parsed["choices"][0]["message"]["content"]


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    default_variants_dir = repo_root / "config" / "variants"

    parser = argparse.ArgumentParser(description="Query local Qwen/vLLM with system prompt variants.")
    parser.add_argument("--variants-dir", type=Path, default=default_variants_dir)
    parser.add_argument("--variant", default="all", help="Variant filename to run, or 'all'.")
    parser.add_argument("--url", default="http://localhost:8000/v1/chat/completions")
    parser.add_argument("--api-key", default="token-abc123")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=400)
    args = parser.parse_args()

    variants = sorted(args.variants_dir.glob("*.txt"))
    if args.variant != "all":
        variants = [args.variants_dir / args.variant]

    if not variants:
        raise FileNotFoundError(f"No variant prompts found under: {args.variants_dir}")

    for variant_path in variants:
        system_prompt = variant_path.read_text()
        env_prompt = DEFAULT_ENV_PROMPT.strip()

        print(f"\n===== {variant_path.name} =====")
        response_text = query_vllm(
            url=args.url,
            api_key=args.api_key,
            model=args.model,
            system_prompt=system_prompt,
            env_prompt=env_prompt,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        print(response_text)


if __name__ == "__main__":
    main()
