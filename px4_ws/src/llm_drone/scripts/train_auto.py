#!/usr/bin/env python3
"""Search LoRA hyperparameters for one fixed dataset and one fixed prompt."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train.py"
FIXED_PROMPT_PATH = REPO_ROOT / "config" / "variant_X.txt"
FIXED_DATASET_PATH = REPO_ROOT / "dataset" / "dataset_variant_x_reasoning.csv"
AUTORESEARCH_ROOT = REPO_ROOT / "third_party" / "autoresearch"
AUTORESEARCH_REMOTE = "https://github.com/karpathy/autoresearch.git"
RUN_ROOT = AUTORESEARCH_ROOT / "llm_drone_runs"
RESULTS_TSV = AUTORESEARCH_ROOT / "llm_drone_results.tsv"
TEMP_TRAIN_SCRIPT = REPO_ROOT / "scripts" / "_train_auto_trial.py"
FINAL_MODEL_DIR = REPO_ROOT / "models" / "QWEN_STUDENT_VARIANT_X_REASONING"

EVAL_LOSS_RE = re.compile(r"(?:'eval_loss'|\"eval_loss\"|eval_loss)\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)")
SAFETY_PENALTY_RE = re.compile(r"Safety penalty\s*:\s*(n/a|[-+]?[0-9.]+)")
GOAL_PROGRESS_REWARD_RE = re.compile(r"Goal progress reward\s*:\s*(n/a|[-+]?[0-9.]+)")

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
    parser.add_argument("--data-path", default=str(FIXED_DATASET_PATH))
    parser.add_argument("--prompt-file", default=str(FIXED_PROMPT_PATH))
    parser.add_argument("--max-trials", type=int, default=24)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--rank-values", default="16,32,48")
    parser.add_argument("--rank-patience", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def ensure_layout() -> None:
    if not AUTORESEARCH_ROOT.exists():
        AUTORESEARCH_ROOT.parent.mkdir(parents=True, exist_ok=True)
        print(f"Cloning autoresearch into {AUTORESEARCH_ROOT} ...")
        subprocess.run(["git", "clone", AUTORESEARCH_REMOTE, str(AUTORESEARCH_ROOT)], check=True)
    if not AUTORESEARCH_ROOT.exists():
        raise FileNotFoundError(f"Expected autoresearch repo at {AUTORESEARCH_ROOT}")
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    if not RESULTS_TSV.exists():
        RESULTS_TSV.write_text(
            "trial_id\tstage\tprompt_file\ttarget_modules\tlora_r\tlora_alpha\tlora_dropout\tlearning_rate\twarmup_ratio\teval_loss\tsafety_penalty\tgoal_progress_reward\tstatus\tdescription\trun_log\toutput_dir\n"
        )


def parse_rank_values(raw: str) -> list[int]:
    values: list[int] = []
    for piece in raw.split(","):
        piece = piece.strip()
        if not piece:
            continue
        rank = int(piece)
        if rank <= 0:
            raise ValueError(f"Rank values must be positive, got {rank}")
        values.append(rank)
    unique_sorted = sorted(set(values))
    if not unique_sorted:
        raise ValueError("At least one rank value is required")
    return unique_sorted


def canonical_target_module_sets() -> list[tuple[str, list[str]]]:
    return [
        ("attention_plus_mlp", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
        ("all_linear_layers", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
    ]

def load_fixed_prompt(prompt_file: str) -> dict[str, str]:
    path = Path(prompt_file).expanduser()
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return {
        "prompt_name": path.stem,
        "prompt_path": str(path),
        "prompt_text": path.read_text(),
    }


def infer_prompt_schema(prompt_text: str) -> str:
    text = str(prompt_text or "")
    if "plan_trace" in text:
        return "plan_trace"
    if re.search(r'"\s*reasoning\s*"\s*:', text):
        return "reasoning"
    return "waypoints_only"


def dataset_completion_schema(data_path: Path) -> str:
    with data_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw = (row.get("label_waypoints_json") or row.get("completion") or "").strip()
            if not raw:
                continue
            payload = json.loads(raw)
            if isinstance(payload, dict) and isinstance(payload.get("plan_trace"), dict):
                return "plan_trace"
            if isinstance(payload, dict) and "reasoning" in payload:
                return "reasoning"
            return "waypoints_only"
    return "waypoints_only"


def ensure_dataset_matches_prompt(prompt: dict[str, str], dataset_path: Path) -> tuple[Path, str]:
    prompt_schema = infer_prompt_schema(prompt["prompt_text"])
    source_schema = dataset_completion_schema(dataset_path)
    if prompt_schema == source_schema:
        return dataset_path, source_schema
    if prompt_schema == "waypoints_only":
        return dataset_path, source_schema

    temp_dir = Path(tempfile.gettempdir()) / "llm_drone_prompt_adapted"
    temp_dir.mkdir(parents=True, exist_ok=True)
    suffix = "plan_trace" if prompt_schema == "plan_trace" else "reasoning"
    output_path = temp_dir / f"{dataset_path.stem}_{suffix}.csv"
    preprocess_script = REPO_ROOT / "scripts" / "preprocess_for_teacher.py"
    subprocess.run(
        [
            sys.executable,
            str(preprocess_script),
            "--input-csv",
            str(dataset_path),
            "--output-csv",
            str(output_path),
            "--output-format",
            "plan_trace" if prompt_schema == "plan_trace" else "reasoning",
        ],
        cwd=str(REPO_ROOT),
        check=True,
    )
    return output_path, prompt_schema


def default_trial_spec(prompt: dict[str, str]) -> dict:
    return {
        "prompt_name": prompt["prompt_name"],
        "prompt_path": prompt["prompt_path"],
        "prompt_text": prompt["prompt_text"],
        "data_path": "",
        "target_modules_name": "attention_plus_mlp",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "lora_r": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.05,
        "learning_rate": 1e-4,
        "warmup_ratio": 0.05,
    }


def build_trial_id(index: int, spec: dict) -> str:
    return (
        f"{index:03d}_{spec['stage']}_{spec['prompt_name']}"
        f"_{spec['target_modules_name']}"
        f"_r{spec['lora_r']}"
        f"_a{spec['lora_alpha']}"
        f"_d{str(spec['lora_dropout']).replace('.', 'p')}"
        f"_lr{str(spec['learning_rate']).replace('.', 'p').replace('-', 'm')}"
        f"_wu{str(spec['warmup_ratio']).replace('.', 'p')}"
    )


def build_trial_train_script(spec: dict, output_dir: Path, *, skip_merge: bool = True) -> str:
    text = TRAIN_SCRIPT.read_text()
    replacements = {
        r'data_path: str = str\(\(Path\(__file__\)\.resolve\(\)\.parents\[1\] / "dataset" / "offline_ground_truth_dataset\.csv"\)\)': f"data_path: str = {spec['data_path']!r}",
        r"lora_r: int = 32": f"lora_r: int = {spec['lora_r']}",
        r"lora_alpha: int = 64": f"lora_alpha: int = {spec['lora_alpha']}",
        r"lora_dropout: float = 0.05": f"lora_dropout: float = {spec['lora_dropout']}",
        r"learning_rate: float = 1e-4": f"learning_rate: float = {spec['learning_rate']}",
        r"warmup_ratio: float = 0.05": f"warmup_ratio: float = {spec['warmup_ratio']}",
        r"output_dir: str = str\(DEFAULT_OUTPUT_DIR\)": f"output_dir: str = {str(output_dir)!r}",
        r'DEFAULT_TRAIN_PROMPT_PATH = \(Path\(__file__\)\.resolve\(\)\.parents\[1\] / "config" / "variant_X\.txt"\)\.resolve\(\)': f"DEFAULT_TRAIN_PROMPT_PATH = Path({spec['prompt_path']!r}).resolve()",
    }
    for pattern, replacement in replacements.items():
        text, count = re.subn(pattern, replacement, text, count=1)
        if count != 1:
            raise RuntimeError(f"Failed to patch {pattern!r} in trial train.py template")
    target_modules_block = (
        'target_modules: list = field(default_factory=lambda: [\n'
        + "".join(f'        "{module}",' + "\n" for module in spec["target_modules"])
        + "    ])"
    )
    start_marker = 'target_modules: list = field(default_factory=lambda: ['
    start = text.find(start_marker)
    if start == -1:
        raise RuntimeError("Failed to locate target_modules block in trial train.py template")
    block_end = text.find("    ])", start)
    if block_end == -1:
        raise RuntimeError("Failed to find end of target_modules block in trial train.py template")
    text = text[:start] + target_modules_block + text[block_end + len("    ])"):]
    if skip_merge:
        if MERGE_BLOCK not in text:
            raise RuntimeError("Failed to locate merge block in scripts/train.py template")
        text = text.replace(MERGE_BLOCK, SKIP_MERGE_BLOCK, 1)
    return text


def parse_optional_float(matches: list[str]) -> float | None:
    if not matches:
        return None
    value = matches[-1]
    return None if value == "n/a" else float(value)


def parse_run_metrics(log_path: Path) -> tuple[float | None, float | None, float | None]:
    text = log_path.read_text(errors="replace")
    eval_loss_matches = EVAL_LOSS_RE.findall(text)
    safety_matches = SAFETY_PENALTY_RE.findall(text)
    goal_progress_matches = GOAL_PROGRESS_REWARD_RE.findall(text)
    eval_loss = float(eval_loss_matches[-1]) if eval_loss_matches else None
    safety_penalty = parse_optional_float(safety_matches)
    goal_progress_reward = parse_optional_float(goal_progress_matches)
    return eval_loss, safety_penalty, goal_progress_reward


def append_result(row: dict[str, str]) -> None:
    with RESULTS_TSV.open("a", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                row["trial_id"],
                row["stage"],
                row["prompt_file"],
                row["target_modules"],
                row["lora_r"],
                row["lora_alpha"],
                row["lora_dropout"],
                row["learning_rate"],
                row["warmup_ratio"],
                row["eval_loss"],
                row["safety_penalty"],
                row["goal_progress_reward"],
                row["status"],
                row["description"],
                row["run_log"],
                row["output_dir"],
            ]
        )


def run_single_trial(
    *,
    args: argparse.Namespace,
    spec: dict,
    index: int,
    total: int,
    run_tag: str,
    run_dir: Path,
) -> dict:
    trial_id = build_trial_id(index, spec)
    output_dir = REPO_ROOT / "models" / "train_auto" / run_tag / trial_id
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / f"{trial_id}.log"
    TEMP_TRAIN_SCRIPT.write_text(build_trial_train_script(spec, output_dir))

    print(f"[{index}/{total}] {trial_id}")
    print(f"  dataset : {spec['data_path']}")
    print(f"  stage   : {spec['stage']}")
    print(f"  prompt  : {spec['prompt_path']}")
    print(f"  layers  : {spec['target_modules_name']} -> {', '.join(spec['target_modules'])}")
    print(f"  run log : {log_path}")

    with log_path.open("w") as log_handle:
        completed = subprocess.run(
            [args.python, str(TEMP_TRAIN_SCRIPT)],
            cwd=str(REPO_ROOT),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=False,
        )

    eval_loss, safety_penalty, goal_progress_reward = parse_run_metrics(log_path)
    if completed.returncode != 0:
        status = "crash"
    elif eval_loss is None:
        status = "crash"
    else:
        status = "keep"

    description = (
        f"stage={spec['stage']} dataset={spec['data_path']} prompt={spec['prompt_path']} "
        f"target_modules={spec['target_modules_name']} "
        f"r={spec['lora_r']} alpha={spec['lora_alpha']} dropout={spec['lora_dropout']} "
        f"lr={spec['learning_rate']} warmup={spec['warmup_ratio']}"
    )
    append_result(
        {
            "trial_id": trial_id,
            "stage": spec["stage"],
            "prompt_file": spec["prompt_path"],
            "target_modules": ",".join(spec["target_modules"]),
            "lora_r": str(spec["lora_r"]),
            "lora_alpha": str(spec["lora_alpha"]),
            "lora_dropout": str(spec["lora_dropout"]),
            "learning_rate": str(spec["learning_rate"]),
            "warmup_ratio": str(spec["warmup_ratio"]),
            "eval_loss": "n/a" if eval_loss is None else f"{eval_loss:.6f}",
            "safety_penalty": "n/a" if safety_penalty is None else f"{safety_penalty:.6f}",
            "goal_progress_reward": "n/a" if goal_progress_reward is None else f"{goal_progress_reward:.6f}",
            "status": status,
            "description": description,
            "run_log": str(log_path),
            "output_dir": str(output_dir),
        }
    )
    if status == "keep":
        safety_text = "n/a" if safety_penalty is None else f"{safety_penalty:.6f}"
        progress_text = "n/a" if goal_progress_reward is None else f"{goal_progress_reward:.6f}"
        print(
            f"  result  : keep (eval_loss={eval_loss:.6f}, safety={safety_text}, "
            f"goal_progress={progress_text})"
        )
    else:
        print("  result  : crash or missing eval_loss")

    return {
        **spec,
        "trial_id": trial_id,
        "status": status,
        "eval_loss": eval_loss,
        "safety_penalty": safety_penalty,
        "goal_progress_reward": goal_progress_reward,
        "run_log": str(log_path),
        "output_dir": str(output_dir),
    }


def train_best_model(*, args: argparse.Namespace, spec: dict, run_tag: str, run_dir: Path) -> tuple[int, Path]:
    canonical_prompt = load_fixed_prompt(str(FIXED_PROMPT_PATH))
    canonical_dataset = FIXED_DATASET_PATH.resolve()
    canonical_dataset, canonical_schema = ensure_dataset_matches_prompt(canonical_prompt, canonical_dataset)
    final_spec = {
        **spec,
        "prompt_name": canonical_prompt["prompt_name"],
        "prompt_path": canonical_prompt["prompt_path"],
        "prompt_text": canonical_prompt["prompt_text"],
        "data_path": str(canonical_dataset),
    }
    final_output_dir = FINAL_MODEL_DIR
    final_output_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / f"{run_tag}_final_train.log"
    TEMP_TRAIN_SCRIPT.write_text(build_trial_train_script(final_spec, final_output_dir, skip_merge=False))

    print("Training final model with best hyperparameters...")
    print(f"  dataset : {final_spec['data_path']}")
    print(f"  prompt  : {final_spec['prompt_path']}")
    print(f"  schema  : {canonical_schema}")
    print(f"  layers  : {final_spec['target_modules_name']} -> {', '.join(final_spec['target_modules'])}")
    print(f"  output  : {final_output_dir}")
    print(f"  run log : {log_path}")

    with log_path.open("w") as log_handle:
        completed = subprocess.run(
            [args.python, str(TEMP_TRAIN_SCRIPT)],
            cwd=str(REPO_ROOT),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=False,
        )

    return completed.returncode, log_path


def score_for_selection(result: dict) -> tuple[float, float, float]:
    eval_loss = result["eval_loss"] if result["eval_loss"] is not None else float("inf")
    safety_penalty = result["safety_penalty"] if result["safety_penalty"] is not None else float("inf")
    goal_progress_reward = result["goal_progress_reward"] if result["goal_progress_reward"] is not None else float("-inf")
    return (eval_loss, safety_penalty, -goal_progress_reward)


def is_better(candidate: dict, incumbent: dict | None) -> bool:
    if candidate["status"] != "keep":
        return False
    if incumbent is None:
        return True
    return score_for_selection(candidate) < score_for_selection(incumbent)


def main() -> int:
    args = parse_args()
    ensure_layout()
    prompt = load_fixed_prompt(args.prompt_file)
    dataset_path = Path(args.data_path).expanduser()
    if not dataset_path.is_absolute():
        dataset_path = (REPO_ROOT / dataset_path).resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    dataset_path, dataset_schema = ensure_dataset_matches_prompt(prompt, dataset_path)
    rank_values = parse_rank_values(args.rank_values)
    target_module_sets = canonical_target_module_sets()
    total_planned = len(rank_values) + len(target_module_sets)
    trial_budget = min(args.max_trials, total_planned) if args.max_trials > 0 else total_planned
    run_tag = datetime.now().strftime("llm_drone_%Y%m%d_%H%M%S")
    run_dir = RUN_ROOT / run_tag
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using fixed dataset: {dataset_path}")
    print(f"Using fixed prompt: {prompt['prompt_path']}")
    print(f"Resolved prompt schema: {infer_prompt_schema(prompt['prompt_text'])}")
    print(f"Resolved dataset schema: {dataset_schema}")
    print(
        f"Planned up to {total_planned} LoRA trial(s): {len(rank_values)} rank trials, "
        f"then {len(target_module_sets)} target-module trials at the best rank. "
        f"Budget capped at {trial_budget}."
    )
    if args.dry_run:
        preview_index = 1
        base_spec = default_trial_spec(prompt)
        base_spec["data_path"] = str(dataset_path)
        for rank in rank_values:
            spec = {
                **base_spec,
                "stage": "rank_search",
                "lora_r": rank,
                "lora_alpha": rank * 2,
            }
            print(
                f"[{preview_index}/{total_planned}] {build_trial_id(preview_index, spec)} :: "
                f"{spec['prompt_path']} {spec['target_modules_name']} r={spec['lora_r']}"
            )
            preview_index += 1
        for layer_name, target_modules in target_module_sets:
            spec = {
                **base_spec,
                "stage": "target_module_search",
                "target_modules_name": layer_name,
                "target_modules": target_modules,
            }
            print(
                f"[{preview_index}/{total_planned}] {build_trial_id(preview_index, spec)} :: "
                f"{spec['prompt_path']} {spec['target_modules_name']} r={spec['lora_r']}"
            )
            preview_index += 1
        return 0

    best_result: dict | None = None
    final_train_log: Path | None = None
    trial_index = 1
    try:
        base_spec = default_trial_spec(prompt)
        base_spec["data_path"] = str(dataset_path)
        best_rank_result: dict | None = None
        no_improvement_count = 0

        # Stage 1: grow rank until validation loss stops improving.
        for rank in rank_values:
            if trial_index > trial_budget:
                print(f"Reached trial budget ({trial_budget}); stopping search.")
                break
            spec = {
                **base_spec,
                "stage": "rank_search",
                "lora_r": rank,
                "lora_alpha": rank * 2,
            }
            result = run_single_trial(
                args=args,
                spec=spec,
                index=trial_index,
                total=total_planned,
                run_tag=run_tag,
                run_dir=run_dir,
            )
            trial_index += 1

            if is_better(result, best_result):
                best_result = result

            if result["status"] != "keep":
                continue

            if best_rank_result is None:
                best_rank_result = result
                continue

            if is_better(result, best_rank_result):
                best_rank_result = result
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                print(
                    f"  rank stop check: no composite-metric improvement "
                    f"({no_improvement_count}/{args.rank_patience})"
                )
                if no_improvement_count >= args.rank_patience:
                    print(
                        f"  stopping rank growth at r={rank}; "
                        f"best rank is r={best_rank_result['lora_r']}"
                    )
                    break

        if best_rank_result is None:
            print("No successful LoRA trial met the selection criteria.")
        else:
            print(
                f"Starting target-module sweep at best rank r={best_rank_result['lora_r']}."
            )
            for layer_name, target_modules in target_module_sets:
                if trial_index > trial_budget:
                    print(f"Reached trial budget ({trial_budget}) during target-module sweep.")
                    break
                spec = {
                    **base_spec,
                    "stage": "target_module_search",
                    "lora_r": best_rank_result["lora_r"],
                    "lora_alpha": best_rank_result["lora_alpha"],
                    "target_modules_name": layer_name,
                    "target_modules": target_modules,
                }
                result = run_single_trial(
                    args=args,
                    spec=spec,
                    index=trial_index,
                    total=total_planned,
                    run_tag=run_tag,
                    run_dir=run_dir,
                )
                trial_index += 1
                if is_better(result, best_result):
                    best_result = result

        if best_result is not None and not args.dry_run:
            final_returncode, final_train_log = train_best_model(
                args=args,
                spec=best_result,
                run_tag=run_tag,
                run_dir=run_dir,
            )
            if final_returncode != 0:
                print(f"Final training failed. See log: {final_train_log}")
    finally:
        if TEMP_TRAIN_SCRIPT.exists():
            TEMP_TRAIN_SCRIPT.unlink()

    print(f"Results logged to {RESULTS_TSV}")
    if best_result is not None:
        safety_text = "n/a" if best_result["safety_penalty"] is None else f"{best_result['safety_penalty']:.6f}"
        progress_text = "n/a" if best_result["goal_progress_reward"] is None else f"{best_result['goal_progress_reward']:.6f}"
        print("Best hyperparameters:")
        print(f"  dataset        : {best_result['data_path']}")
        print(f"  prompt_file    : {best_result['prompt_path']}")
        print(f"  stage          : {best_result['stage']}")
        print(f"  target_modules : {best_result['target_modules_name']} -> {', '.join(best_result['target_modules'])}")
        print(f"  lora_r         : {best_result['lora_r']}")
        print(f"  lora_alpha     : {best_result['lora_alpha']}")
        print(f"  lora_dropout   : {best_result['lora_dropout']}")
        print(f"  learning_rate  : {best_result['learning_rate']}")
        print(f"  warmup_ratio   : {best_result['warmup_ratio']}")
        print(f"  eval_loss      : {best_result['eval_loss']:.6f}")
        print(f"  safety_penalty : {safety_text}")
        print(f"  goal_progress  : {progress_text}")
        print(f"  run_log        : {best_result['run_log']}")
        print(f"  output_dir     : {best_result['output_dir']}")
        if final_train_log is not None:
            print(f"Final model log   : {final_train_log}")
            print(f"Final model dir   : {FINAL_MODEL_DIR}")
    else:
        print("No successful trial met the selection criteria.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
