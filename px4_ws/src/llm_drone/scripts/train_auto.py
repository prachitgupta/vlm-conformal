#!/usr/bin/env python3
"""Autoresearch-style sweep runner for llm_drone.

This wrapper uses the cloned third_party/autoresearch repo as the experiment
workspace and bookkeeping surface, but executes trial-local copies of this
project's scripts/train.py. The original scripts/train.py stays unchanged.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import random
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train.py"
ACTIVE_PROMPT_PATH = REPO_ROOT / "config" / "llm_prompt2d.txt"
STATIC_VARIANT_DIR = REPO_ROOT / "config" / "variants"
AUTORESEARCH_ROOT = REPO_ROOT / "third_party" / "autoresearch"
AUTORESEARCH_REMOTE = "https://github.com/karpathy/autoresearch.git"
VARIANT_DIR = AUTORESEARCH_ROOT / "llm_drone_prompt_variants"
RUN_ROOT = AUTORESEARCH_ROOT / "llm_drone_runs"
RESULTS_TSV = AUTORESEARCH_ROOT / "llm_drone_results.tsv"
TEMP_TRAIN_SCRIPT = REPO_ROOT / "scripts" / "_train_auto_trial.py"

JSON_RE = re.compile(r"JSON validity\s*:\s*([0-9.]+)%")
L2_RE = re.compile(r"Mean L2 error\s*:\s*(n/a|[0-9.]+)\s*m")

NO_REASONING_SECTION3 = """SECTION 3: EXPECTED OUTPUT
Return ONLY one valid JSON object. No markdown. No extra text.

Required schema (exact top-level keys):
{
  "waypoints": [
    {"x": <float>, "y": <float>},
    {"x": <float>, "y": <float>},
    {"x": <float>, "y": <float>},
    {"x": <float>, "y": <float>},
    {"x": <float>, "y": <float>}
  ]
}

Schema rules:
- "waypoints" must contain exactly 5 waypoint objects.
- Every waypoint must include numeric x and y.
- Waypoints must be ordered in execution order from immediate next step to farther local lookahead.
- Do not output z.
- Do not output a reasoning field.
- Do not add any extra top-level keys.
""".strip()

CONCISE_PROMPT = """You are the motion planner for an autonomous quadrotor.

Task:
- Predict the next 5 local waypoint x/y pairs.
- Use the provided odometry and obstacle/depth context.
- Plan only in the horizontal plane in NED meters.
- Do not output z.

Hard requirements:
- Safety first: avoid obstacles and preserve at least 1.5 m clearance when feasible.
- Maintain smooth, dynamically feasible motion.
- Ensure progress toward the goal.
- The first waypoint must be local: at most 3 m from current position.
- Consecutive waypoint spacing should usually stay between 0.5 m and 3.0 m unless safety requires shorter motion.

Priority order:
1) Safety and feasibility
2) Progress to goal
3) Smoothness and efficiency
4) Consistent 5-waypoint local plan

Output:
Return ONLY one valid JSON object with exactly these top-level keys:
{
  "waypoints": [
    {"x": <float>, "y": <float>},
    {"x": <float>, "y": <float>},
    {"x": <float>, "y": <float>},
    {"x": <float>, "y": <float>},
    {"x": <float>, "y": <float>}
  ],
  "reasoning": "Selected waypoint <int> moves the reference anchor from <float> m to <float> m from goal immediately and to <float> m by the final waypoint, so progress is <float> m selected / <float> m final and <passes|fails> with <monotonic|non-monotonic> goal-distance reduction. Kinematics is evaluated on 5 unique label waypoint(s), excluding consecutive duplicate terminal holds, and checks only that the trajectory stays within the configured bounds speed <= 15.00 m/s and acceleration <= 8.00 m/s^2, so kinematics <passes|fails>. <Goal is not reached within this label waypoint window.|Goal will be reached in <int> waypoint(s) within 0.35 m tolerance.|Goal has already been reached at the reference anchor within 0.35 m tolerance.> Safety <passes|fails>: <clearance sentence>."
}

Rules:
- Do not add markdown or extra text.
- Waypoints must be ordered from immediate next move to farther lookahead.
- Keep the reasoning sentence in the canonical order shown above.
""".strip()


CONCISE_NO_REASONING_PROMPT = """You are the motion planner for an autonomous quadrotor.

Task:
- Predict the next 5 local waypoint x/y pairs.
- Use the provided odometry and obstacle/depth context.
- Plan only in the horizontal plane in NED meters.
- Do not output z.

Hard requirements:
- Safety first: avoid obstacles and preserve at least 1.5 m clearance when feasible.
- Maintain smooth, dynamically feasible motion.
- Ensure progress toward the goal.
- The first waypoint must be local: at most 3 m from current position.
- Consecutive waypoint spacing should usually stay between 0.5 m and 3.0 m unless safety requires shorter motion.

Priority order:
1) Safety and feasibility
2) Progress to goal
3) Smoothness and efficiency
4) Consistent 5-waypoint local plan

Output:
Return ONLY one valid JSON object with exactly this top-level key:
{
  "waypoints": [
    {"x": <float>, "y": <float>},
    {"x": <float>, "y": <float>},
    {"x": <float>, "y": <float>},
    {"x": <float>, "y": <float>},
    {"x": <float>, "y": <float>}
  ]
}

Rules:
- Do not add markdown or extra text.
- Do not output a reasoning field.
- Waypoints must be ordered from immediate next move to farther lookahead.
""".strip()

MERGE_BLOCK = """    # 10. Optionally merge LoRA weights into base model
    # This creates a single standalone model (no adapter overhead at inference)
    log.info(f"Merging LoRA into base model → {merged_dir_path}")
    model.save_pretrained_merged(
        merged_dir_path,
        tokenizer,
        save_method="merged_16bit",
        # WHY MERGED: At inference time, a merged model has zero LoRA overhead.
        # The adapter math (W + BA) is pre-computed into W directly.
        # For a drone real-time system where latency matters, always use merged.
    )
    log.info("Done! Merged model ready for inference.")
    log.info(f"To run inference: load model from {merged_dir_path}")
"""

SKIP_MERGE_BLOCK = """    # 10. Skip merged export during autoresearch-style sweeps
    log.info("Skipping merged model export for this train_auto trial.")
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("random", "grid"), default="random")
    parser.add_argument("--max-trials", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--min-json-validity", type=float, default=95.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def ensure_layout() -> None:
    if not AUTORESEARCH_ROOT.exists():
        AUTORESEARCH_ROOT.parent.mkdir(parents=True, exist_ok=True)
        print(f"Cloning autoresearch into {AUTORESEARCH_ROOT} ...")
        subprocess.run(["git", "clone", AUTORESEARCH_REMOTE, str(AUTORESEARCH_ROOT)], check=True)
    if not AUTORESEARCH_ROOT.exists():
        raise FileNotFoundError(f"Expected autoresearch repo at {AUTORESEARCH_ROOT}")
    VARIANT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    if not RESULTS_TSV.exists():
        RESULTS_TSV.write_text(
            "trial_id\tprompt_variant\tlora_r\tlora_alpha\tlora_dropout\tlearning_rate\twarmup_ratio\tjson_validity_pct\tmean_l2_m\tstatus\tdescription\trun_log\toutput_dir\n"
        )


def split_prompt_sections(base_text: str) -> tuple[str, str, str]:
    section2 = "SECTION 2: FEW-SHOT DEMONSTRATIONS"
    section3 = "SECTION 3: EXPECTED OUTPUT"
    idx2 = base_text.index(section2)
    idx3 = base_text.index(section3)
    return base_text[:idx2].rstrip(), base_text[idx2:idx3].strip(), base_text[idx3:].strip()


def strip_reasoning_from_examples(section2_text: str) -> str:
    return re.sub(r',\s*\n\s*"reasoning":"[^"]*"', "", section2_text)


def keep_only_example_a(section2_text: str) -> str:
    marker = "Example B (obstacle on right, move left):"
    if marker not in section2_text:
        return section2_text.strip()
    return section2_text.split(marker, 1)[0].rstrip()


def create_prompt_variants() -> list[dict[str, str]]:
    static_variants = sorted(STATIC_VARIANT_DIR.glob("*.txt"))
    if static_variants:
        return [
            {"name": path.stem, "path": str(path), "text": path.read_text()}
            for path in static_variants
        ]

    base_text = ACTIVE_PROMPT_PATH.read_text()
    section1, section2, section3 = split_prompt_sections(base_text)
    section2_no_reasoning = strip_reasoning_from_examples(section2)
    section2_one_example_no_reasoning = keep_only_example_a(section2_no_reasoning)

    variants = [
        ("00_base_llm_prompt2d", base_text.strip()),
        ("01_no_reasoning", "\n\n".join([section1, section2_no_reasoning, NO_REASONING_SECTION3])),
        ("02_no_reasoning_one_example", "\n\n".join([section1, section2_one_example_no_reasoning, NO_REASONING_SECTION3])),
        (
            "03_no_reasoning_no_examples",
            "\n\n".join([
                section1,
                "SECTION 2: FEW-SHOT DEMONSTRATIONS\nExamples are intentionally omitted in this variant.",
                NO_REASONING_SECTION3,
            ]),
        ),
        (
            "04_no_examples_with_reasoning",
            "\n\n".join([
                section1,
                "SECTION 2: FEW-SHOT DEMONSTRATIONS\nExamples are intentionally omitted in this variant, but the reasoning contract is kept.",
                section3,
            ]),
        ),
        ("05_concise_with_reasoning", CONCISE_PROMPT),
        ("06_concise_no_reasoning", CONCISE_NO_REASONING_PROMPT),
    ]

    written: list[dict[str, str]] = []
    for name, text in variants:
        path = VARIANT_DIR / f"{name}.txt"
        path.write_text(text.strip() + "\n")
        written.append({"name": name, "path": str(path), "text": text.strip() + "\n"})
    return written


def build_trial_specs(variants: list[dict[str, str]], args: argparse.Namespace) -> list[dict]:
    lora_r_values = [16, 32, 48]
    alpha_multipliers = [1, 2]
    lora_dropout_values = [0.0, 0.05, 0.1]
    learning_rate_values = [5e-5, 1e-4, 2e-4]
    warmup_ratio_values = [0.03, 0.05, 0.08]

    specs: list[dict] = []
    for variant, rank, alpha_mul, dropout, lr, warmup in itertools.product(
        variants,
        lora_r_values,
        alpha_multipliers,
        lora_dropout_values,
        learning_rate_values,
        warmup_ratio_values,
    ):
        specs.append(
            {
                "prompt_variant": variant["name"],
                "prompt_path": variant["path"],
                "prompt_text": variant["text"],
                "lora_r": rank,
                "lora_alpha": rank * alpha_mul,
                "lora_dropout": dropout,
                "learning_rate": lr,
                "warmup_ratio": warmup,
            }
        )

    default_specs = [
        spec
        for spec in specs
        if spec["lora_r"] == 32
        and spec["lora_alpha"] == 64
        and spec["lora_dropout"] == 0.05
        and spec["learning_rate"] == 1e-4
        and spec["warmup_ratio"] == 0.05
    ]
    if not default_specs:
        raise RuntimeError("Failed to build default trial specs")

    baseline = default_specs[0]
    prompt_first = default_specs[1:]
    remaining = [spec for spec in specs if spec not in default_specs]

    chosen = [baseline] + prompt_first
    leftover = remaining
    if args.mode == "random":
        random.Random(args.seed).shuffle(leftover)
        budget = max(0, args.max_trials - len(chosen))
        chosen.extend(leftover[:budget])
    else:
        chosen.extend(leftover)
    return chosen


def build_trial_id(index: int, spec: dict) -> str:
    return (
        f"{index:03d}_{spec['prompt_variant']}"
        f"_r{spec['lora_r']}"
        f"_a{spec['lora_alpha']}"
        f"_d{str(spec['lora_dropout']).replace('.', 'p')}"
        f"_lr{str(spec['learning_rate']).replace('.', 'p').replace('-', 'm')}"
        f"_wu{str(spec['warmup_ratio']).replace('.', 'p')}"
    )


def build_trial_train_script(spec: dict, output_dir: Path) -> str:
    text = TRAIN_SCRIPT.read_text()
    replacements = {
        r"lora_r: int = 32": f"lora_r: int = {spec['lora_r']}",
        r"lora_alpha: int = 64": f"lora_alpha: int = {spec['lora_alpha']}",
        r"lora_dropout: float = 0.05": f"lora_dropout: float = {spec['lora_dropout']}",
        r"learning_rate: float = 1e-4": f"learning_rate: float = {spec['learning_rate']}",
        r"warmup_ratio: float = 0.05": f"warmup_ratio: float = {spec['warmup_ratio']}",
        r"output_dir: str = str\(DEFAULT_OUTPUT_DIR\)": f"output_dir: str = {str(output_dir)!r}",
    }
    for pattern, replacement in replacements.items():
        text, count = re.subn(pattern, replacement, text, count=1)
        if count != 1:
            raise RuntimeError(f"Failed to patch {pattern!r} in trial train.py template")
    if MERGE_BLOCK not in text:
        raise RuntimeError("Failed to locate merge block in scripts/train.py template")
    text = text.replace(MERGE_BLOCK, SKIP_MERGE_BLOCK, 1)
    return text


def parse_run_metrics(log_path: Path) -> tuple[float | None, float | None]:
    text = log_path.read_text(errors="replace")
    json_matches = JSON_RE.findall(text)
    l2_matches = L2_RE.findall(text)
    json_validity = float(json_matches[-1]) if json_matches else None
    if not l2_matches:
        mean_l2 = None
    else:
        last_l2 = l2_matches[-1]
        mean_l2 = None if last_l2 == "n/a" else float(last_l2)
    return json_validity, mean_l2


def append_result(row: dict[str, str]) -> None:
    with RESULTS_TSV.open("a", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                row["trial_id"],
                row["prompt_variant"],
                row["lora_r"],
                row["lora_alpha"],
                row["lora_dropout"],
                row["learning_rate"],
                row["warmup_ratio"],
                row["json_validity_pct"],
                row["mean_l2_m"],
                row["status"],
                row["description"],
                row["run_log"],
                row["output_dir"],
            ]
        )


def main() -> int:
    args = parse_args()
    ensure_layout()
    variants = create_prompt_variants()
    specs = build_trial_specs(variants, args)
    run_tag = datetime.now().strftime("llm_drone_%Y%m%d_%H%M%S")
    run_dir = RUN_ROOT / run_tag
    run_dir.mkdir(parents=True, exist_ok=True)

    static_variant_root = str(STATIC_VARIANT_DIR)
    using_static_variants = variants and all(spec["path"].startswith(static_variant_root) for spec in variants)
    action = "Loaded" if using_static_variants else "Created"
    source_dir = STATIC_VARIANT_DIR if using_static_variants else VARIANT_DIR
    print(f"{action} {len(variants)} prompt variants from {source_dir}")
    print(f"Planned {len(specs)} trial(s) in {args.mode} mode.")
    if args.dry_run:
        for index, spec in enumerate(specs, start=1):
            print(
                f"[{index}/{len(specs)}] {build_trial_id(index, spec)} :: "
                f"{spec['prompt_variant']} r={spec['lora_r']} alpha={spec['lora_alpha']} "
                f"dropout={spec['lora_dropout']} lr={spec['learning_rate']} warmup={spec['warmup_ratio']}"
            )
        return 0

    original_prompt_text = ACTIVE_PROMPT_PATH.read_text()
    best_mean_l2 = None
    best_json = None
    try:
        for index, spec in enumerate(specs, start=1):
            trial_id = build_trial_id(index, spec)
            output_dir = REPO_ROOT / "models" / "train_auto" / run_tag / trial_id
            output_dir.mkdir(parents=True, exist_ok=True)
            log_path = run_dir / f"{trial_id}.log"
            TEMP_TRAIN_SCRIPT.write_text(build_trial_train_script(spec, output_dir))
            ACTIVE_PROMPT_PATH.write_text(spec["prompt_text"])

            print(f"[{index}/{len(specs)}] {trial_id}")
            print(f"  prompt  : {spec['prompt_variant']}")
            print(f"  run log : {log_path}")

            with log_path.open("w") as log_handle:
                completed = subprocess.run(
                    [args.python, str(TEMP_TRAIN_SCRIPT)],
                    cwd=str(REPO_ROOT),
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    check=False,
                )

            json_validity, mean_l2 = parse_run_metrics(log_path)
            if completed.returncode != 0:
                status = "crash"
            elif json_validity is None or mean_l2 is None:
                status = "crash"
            elif json_validity < args.min_json_validity:
                status = "discard"
            elif best_mean_l2 is None or mean_l2 < best_mean_l2:
                status = "keep"
                best_mean_l2 = mean_l2
                best_json = json_validity
            else:
                status = "discard"

            description = (
                f"prompt={spec['prompt_variant']} r={spec['lora_r']} alpha={spec['lora_alpha']} "
                f"dropout={spec['lora_dropout']} lr={spec['learning_rate']} warmup={spec['warmup_ratio']}"
            )
            append_result(
                {
                    "trial_id": trial_id,
                    "prompt_variant": spec["prompt_variant"],
                    "lora_r": str(spec["lora_r"]),
                    "lora_alpha": str(spec["lora_alpha"]),
                    "lora_dropout": str(spec["lora_dropout"]),
                    "learning_rate": str(spec["learning_rate"]),
                    "warmup_ratio": str(spec["warmup_ratio"]),
                    "json_validity_pct": "n/a" if json_validity is None else f"{json_validity:.2f}",
                    "mean_l2_m": "n/a" if mean_l2 is None else f"{mean_l2:.6f}",
                    "status": status,
                    "description": description,
                    "run_log": str(log_path),
                    "output_dir": str(output_dir),
                }
            )
            if status == "keep":
                print(f"  result  : keep (json={json_validity:.2f}%, mean_l2={mean_l2:.6f} m)")
            elif status == "discard":
                print(f"  result  : discard (json={json_validity:.2f}%, mean_l2={mean_l2:.6f} m)")
            else:
                print("  result  : crash or missing metrics, see log")
    finally:
        ACTIVE_PROMPT_PATH.write_text(original_prompt_text)
        if TEMP_TRAIN_SCRIPT.exists():
            TEMP_TRAIN_SCRIPT.unlink()

    print(f"Results logged to {RESULTS_TSV}")
    if best_mean_l2 is not None and best_json is not None:
        print(f"Best observed keep metric: mean_l2={best_mean_l2:.6f} m with json={best_json:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
