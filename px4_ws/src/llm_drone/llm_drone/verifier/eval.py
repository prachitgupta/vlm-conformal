#!/usr/bin/env python3
"""
Standalone LLM waypoint evaluator (no ROS dependencies).

This script evaluates a single LLM JSON response against:
- format/schema checks
- kinematic plausibility (speed/accel)
- goal progress
- optional obstacle safety

It writes:
- optional latex table + metrics logs
- memory JSONL row compatible with generate_synthetic_prompts.py
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_LLM_RESPONSE = """{"waypoints":[{"x":1.25,"y":0.41,"z":-2.43},{"x":2.65,"y":0.52,"z":-2.43},{"x":4.05,"y":0.63,"z":-2.43},{"x":5.45,"y":0.74,"z":-2.42},{"x":6.85,"y":0.84,"z":-2.42}],"selected_waypoint_index":0,"reasoning":"Clear center corridor with nearest obstacle off far left, so use a smooth straight 5-point trajectory toward the goal while maintaining safe altitude."}"""


@dataclass
class EvalResult:
    passed: bool
    format_pass: bool
    kinematic_pass: bool
    progress_pass: bool
    safety_pass: bool
    metrics: dict[str, Any]
    errors: list[str]
    warnings: list[str]
    parsed: dict[str, Any] | None = None


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def _seg_point_distance(a: np.ndarray, b: np.ndarray, p: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    p = np.asarray(p, dtype=float)
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom < 1e-12:
        return float(np.linalg.norm(p - a))
    t = float(np.dot(p - a, ab) / denom)
    t = max(0.0, min(1.0, t))
    proj = a + t * ab
    return float(np.linalg.norm(p - proj))


def parse_llm_response(text: str) -> dict[str, Any]:
    s = text.find("{")
    e = text.rfind("}") + 1
    if s < 0 or e <= s:
        raise ValueError("No JSON object found in response")
    data = json.loads(text[s:e])
    if "waypoints" not in data:
        raise ValueError("Missing 'waypoints'")

    waypoints_raw = data["waypoints"]
    if not isinstance(waypoints_raw, list) or len(waypoints_raw) != 5:
        raise ValueError("'waypoints' must be a list of exactly 5 items")

    selected_idx = int(data.get("selected_waypoint_index", 0))
    if selected_idx < 0 or selected_idx >= 5:
        raise ValueError("'selected_waypoint_index' out of range")

    waypoints = []
    for i, wp in enumerate(waypoints_raw):
        x = float(wp["x"])
        y = float(wp["y"])
        z = float(wp["z"])
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
            raise ValueError(f"Waypoint {i} contains non-finite values")
        waypoints.append(np.array([x, y, z], dtype=float))

    return {
        "waypoints": waypoints,
        "selected_waypoint_index": selected_idx,
        "reasoning": data.get("reasoning", ""),
        "raw": data,
    }


def evaluate_response(
    current_position_ned: np.ndarray,
    goal_ned: np.ndarray,
    llm_response_text: str,
    obstacle_points_ned: np.ndarray | None,
    dt: float,
    v_max: float,
    a_max: float,
    safety_radius: float,
) -> EvalResult:
    errors: list[str] = []
    warnings: list[str] = []
    try:
        parsed = parse_llm_response(llm_response_text)
    except Exception as e:
        return EvalResult(
            passed=False,
            format_pass=False,
            kinematic_pass=False,
            progress_pass=False,
            safety_pass=False,
            metrics={},
            errors=[f"parse_error: {e}"],
            warnings=[],
            parsed=None,
        )

    p0 = np.asarray(current_position_ned, dtype=float)
    goal = np.asarray(goal_ned, dtype=float)
    waypoints = parsed["waypoints"]
    idx = int(parsed["selected_waypoint_index"])
    p_sel = waypoints[idx]

    seq = [p0] + waypoints
    seg_lengths = [_dist(seq[i], seq[i + 1]) for i in range(len(seq) - 1)]
    speeds = [d / max(dt, 1e-6) for d in seg_lengths]
    accels = [abs(speeds[i + 1] - speeds[i]) / max(dt, 1e-6) for i in range(len(speeds) - 1)]
    kinematic_pass = (max(speeds) if speeds else 0.0) <= v_max and (max(accels) if accels else 0.0) <= a_max
    if not kinematic_pass:
        warnings.append(
            f"Kinematic violation: max_speed={max(speeds) if speeds else 0.0:.2f} m/s (limit {v_max:.2f}), "
            f"max_accel={max(accels) if accels else 0.0:.2f} m/s^2 (limit {a_max:.2f})"
        )

    d0 = _dist(p0, goal)
    d_sel = _dist(p_sel, goal)
    d5 = _dist(waypoints[-1], goal)
    progress_selected = d0 - d_sel
    progress_final = d0 - d5
    progress_pass = (progress_selected > 0.0) and (progress_final > 0.0)
    goal_dists = [_dist(wp, goal) for wp in waypoints]
    monotone_goal_progress = all(goal_dists[i] <= (d0 if i == 0 else goal_dists[i - 1]) + 1e-6 for i in range(len(goal_dists)))
    if not progress_pass:
        warnings.append(
            f"Goal progress violation: selected_progress={progress_selected:.2f} m, "
            f"final_progress={progress_final:.2f} m"
        )

    turn_sum = 0.0
    max_turn_deg = 0.0
    for i in range(1, len(seq) - 1):
        u = seq[i] - seq[i - 1]
        v = seq[i + 1] - seq[i]
        nu = float(np.linalg.norm(u))
        nv = float(np.linalg.norm(v))
        if nu > 1e-8 and nv > 1e-8:
            c = float(np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0))
            ang = math.acos(c)
            turn_sum += ang
            max_turn_deg = max(max_turn_deg, math.degrees(ang))

    # Smoothness proxy: mean second-difference magnitude across successive waypoints.
    curvature_terms = []
    for i in range(1, len(waypoints) - 1):
        curvature_terms.append(float(np.linalg.norm(waypoints[i + 1] - 2.0 * waypoints[i] + waypoints[i - 1])))
    smoothness_kappa_m = float(np.mean(curvature_terms)) if curvature_terms else 0.0

    min_clearance = None
    clearance_violations = 0
    safety_pass = True
    if obstacle_points_ned is not None and obstacle_points_ned.size > 0:
        min_clearance = float("inf")
        for i in range(len(seq) - 1):
            a = seq[i]
            b = seq[i + 1]
            for obs in obstacle_points_ned:
                d = _seg_point_distance(a, b, obs)
                if d < min_clearance:
                    min_clearance = d
                if d < safety_radius:
                    clearance_violations += 1
        safety_pass = bool(min_clearance >= safety_radius)
        if not safety_pass:
            warnings.append(
                f"Safety violation: min_clearance={min_clearance:.2f} m < safety_radius={safety_radius:.2f} m "
                f"(clearance_violations={clearance_violations})"
            )
    else:
        warnings.append("Safety check degraded: no obstacle list provided (set --obstacles-json for geometric safety)")

    metrics = {
        "selected_waypoint_index": idx,
        "distance_to_goal_before_m": d0,
        "distance_to_goal_selected_m": d_sel,
        "distance_to_goal_final_m": d5,
        "progress_selected_m": progress_selected,
        "progress_final_m": progress_final,
        "monotone_goal_progress": bool(monotone_goal_progress),
        "goal_distances_to_goal_m": goal_dists,
        "segment_lengths_m": seg_lengths,
        "speeds_mps": speeds,
        "accels_mps2": accels,
        "max_speed_mps": max(speeds) if speeds else 0.0,
        "max_accel_mps2": max(accels) if accels else 0.0,
        "min_clearance_m": min_clearance,
        "clearance_violations": clearance_violations,
        "clearance_is_proxy": bool(obstacle_points_ned is None or obstacle_points_ned.size == 0),
        "smoothness_kappa_m": smoothness_kappa_m,
        "turn_angle_sum_rad": turn_sum,
        "max_turn_deg": max_turn_deg,
        "reasoning": parsed.get("reasoning", ""),
    }
    passed = bool(kinematic_pass and progress_pass and safety_pass)
    return EvalResult(
        passed=passed,
        format_pass=True,
        kinematic_pass=bool(kinematic_pass),
        progress_pass=bool(progress_pass),
        safety_pass=bool(safety_pass),
        metrics=metrics,
        errors=errors,
        warnings=warnings,
        parsed=parsed,
    )


def build_latex_style_eval_table(res: EvalResult) -> str:
    m = res.metrics or {}
    overall = "PASS" if res.passed else "FAIL"
    min_clr = m.get("min_clearance_m", None)
    obs_val = "not provided" if min_clr is None else f"{float(min_clr):.2f} m"
    lines = [
        r"\begin{center}",
        r"\begin{tabular}{lll}",
        r"\toprule",
        r"\textbf{Metric} & \textbf{Value} & \textbf{Pass?} \\",
        r"\midrule",
        f"Format compliance & schema+JSON & \\textbf{{{'Pass' if res.format_pass else 'Fail'}}} \\\\",
        f"Kinematic plausibility & vmax={m.get('max_speed_mps', 0.0):.2f}, amax={m.get('max_accel_mps2', 0.0):.2f} & \\textbf{{{'Pass' if res.kinematic_pass else 'Fail'}}} \\\\",
        f"Goal progress (selected) & {m.get('progress_selected_m', 0.0):+.2f} m & \\textbf{{{'Pass' if m.get('progress_selected_m', 0.0) > 0 else 'Fail'}}} \\\\",
        f"Goal progress (final wp) & {m.get('progress_final_m', 0.0):+.2f} m & \\textbf{{{'Pass' if m.get('progress_final_m', 0.0) > 0 else 'Fail'}}} \\\\",
        f"Obstacle safety & min_clearance={obs_val} & \\textbf{{{'Pass' if res.safety_pass else 'Fail'}}} \\\\",
        r"\midrule",
        f"\\textbf{{Overall}} & --- & \\textbf{{{overall}}} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{center}",
    ]
    return "\n".join(lines)


def _sanitize_for_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, tuple):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _sanitize_for_json(obj.tolist())
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        return val if math.isfinite(val) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def build_refinement_feedback(res: EvalResult) -> str:
    checks = [
        ("format", bool(res.format_pass)),
        ("kinematic", bool(res.kinematic_pass)),
        ("goal_progress", bool(res.progress_pass)),
        ("safety", bool(res.safety_pass)),
    ]
    failed = [name for name, ok in checks if not ok]
    lines = [
        f"Evaluation result: {'PASS' if res.passed else 'FAIL'}",
        f"Failed checks: {', '.join(failed) if failed else 'none'}",
    ]
    if res.warnings:
        lines.append("Warnings:")
        lines.extend([f"- {w}" for w in res.warnings[:6]])
    if res.errors:
        lines.append("Errors:")
        lines.extend([f"- {e}" for e in res.errors[:6]])
    if res.metrics:
        lines.append(
            "Key metrics: "
            f"progress_selected_m={res.metrics.get('progress_selected_m')}, "
            f"progress_final_m={res.metrics.get('progress_final_m')}, "
            f"max_speed_mps={res.metrics.get('max_speed_mps')}, "
            f"max_accel_mps2={res.metrics.get('max_accel_mps2')}, "
            f"min_clearance_m={res.metrics.get('min_clearance_m')}"
        )
    lines.append(
        "Refinement instruction: regenerate exactly 5 waypoints with selected_waypoint_index=0, "
        "prioritize safety + goal progress + smoothness, and return strict JSON only."
    )
    return "\n".join(lines)


def _load_text_arg(value: str, value_file: str) -> str:
    if value_file.strip():
        return Path(value_file).expanduser().read_text()
    return value


def _load_obstacles(json_text: str, json_file: str) -> np.ndarray:
    if json_file.strip():
        raw = json.loads(Path(json_file).expanduser().read_text())
    elif json_text.strip():
        raw = json.loads(json_text)
    else:
        return np.zeros((0, 3), dtype=float)

    pts = np.asarray(raw, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("Obstacles must be Nx3 JSON array, e.g. [[x,y,z],[x,y,z]]")
    return pts


def save_eval_logs(
    out_dir: Path,
    res: EvalResult,
    latex_table: str,
    llm_response_text: str,
    current_position: np.ndarray,
    current_velocity: np.ndarray,
    goal_ned: np.ndarray,
    obstacle_count: int,
) -> tuple[Path, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y%m%dT%H%M%S_%fZ")
    tex_path = out_dir / f"{stamp}_eval_table.tex"
    json_path = out_dir / f"{stamp}_eval_metrics.json"
    response_path = out_dir / f"{stamp}_llm_response.json"

    tex_path.write_text(latex_table + "\n")
    payload = {
        "timestamp_utc": now.isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        "goal_ned_m": _sanitize_for_json(goal_ned),
        "current_position_ned_m": _sanitize_for_json(current_position),
        "current_velocity_mps": _sanitize_for_json(current_velocity),
        "obstacle_snapshot_count": int(obstacle_count),
        "result": {
            "passed": bool(res.passed),
            "format_pass": bool(res.format_pass),
            "kinematic_pass": bool(res.kinematic_pass),
            "progress_pass": bool(res.progress_pass),
            "safety_pass": bool(res.safety_pass),
            "errors": list(res.errors),
            "warnings": list(res.warnings),
            "metrics": _sanitize_for_json(res.metrics),
        },
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    if res.parsed is not None:
        response_path.write_text(json.dumps(_sanitize_for_json(res.parsed.get("raw", {})), indent=2) + "\n")
    else:
        response_path.write_text(json.dumps({"raw_text": llm_response_text}, indent=2) + "\n")
    return tex_path, json_path, response_path


def append_eval_memory(
    mem_path: Path,
    res: EvalResult,
    llm_response_text: str,
    latex_table: str,
    source_prompt: str,
    current_position: np.ndarray,
    current_velocity: np.ndarray,
    goal_ned: np.ndarray,
    obstacle_count: int,
) -> None:
    mem_path = mem_path.expanduser()
    mem_path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    entry = {
        "timestamp_utc": now.isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        "goal_ned_m": _sanitize_for_json(goal_ned),
        "current_position_ned_m": _sanitize_for_json(current_position),
        "current_velocity_mps": _sanitize_for_json(current_velocity),
        "obstacle_snapshot_count": int(obstacle_count),
        "source_prompt": source_prompt,
        "llm_response_text": llm_response_text,
        "llm_response_json": _sanitize_for_json(res.parsed.get("raw")) if res.parsed is not None else None,
        "verification_table_latex": latex_table,
        "result": {
            "passed": bool(res.passed),
            "format_pass": bool(res.format_pass),
            "kinematic_pass": bool(res.kinematic_pass),
            "progress_pass": bool(res.progress_pass),
            "safety_pass": bool(res.safety_pass),
            "errors": list(res.errors),
            "warnings": list(res.warnings),
            "metrics": _sanitize_for_json(res.metrics),
        },
        "refinement_feedback": build_refinement_feedback(res),
    }
    with mem_path.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description="Standalone evaluator for LLM waypoint JSON.")
    p.add_argument("--llm-response", type=str, default=DEFAULT_LLM_RESPONSE, help="Raw LLM JSON response text")
    p.add_argument("--llm-response-file", type=str, default="", help="Path to file containing LLM response text")
    p.add_argument("--source-prompt", type=str, default="", help="Original prompt text for memory row")
    p.add_argument("--source-prompt-file", type=str, default="", help="Path to original prompt file")
    p.add_argument("--current-x", type=float, default=0.0)
    p.add_argument("--current-y", type=float, default=0.0)
    p.add_argument("--current-z", type=float, default=-2.0)
    p.add_argument("--vel-x", type=float, default=0.0)
    p.add_argument("--vel-y", type=float, default=0.0)
    p.add_argument("--vel-z", type=float, default=0.0)
    p.add_argument("--goal-x", type=float, default=20.0)
    p.add_argument("--goal-y", type=float, default=0.0)
    p.add_argument("--goal-z", type=float, default=-5.0)
    p.add_argument("--eval-dt", type=float, default=1.0)
    p.add_argument("--eval-v-max", type=float, default=15.0)
    p.add_argument("--eval-a-max", type=float, default=8.0)
    p.add_argument("--eval-safety-radius", type=float, default=1.5)
    p.add_argument("--obstacles-json", type=str, default="", help="Optional obstacle list JSON, e.g. [[x,y,z],[x,y,z]]")
    p.add_argument("--obstacles-file", type=str, default="", help="Path to JSON file containing Nx3 obstacle points")
    p.add_argument("--save-eval-logs", action="store_true", help="Write latex+metrics logs")
    p.add_argument("--eval-log-dir", type=str, default="/tmp/llm_eval_logs")
    p.add_argument("--save-eval-memory", action="store_true", help="Append JSONL memory row")
    p.add_argument("--eval-memory-path", type=str, default="/tmp/llm_eval_memory.jsonl")
    p.add_argument("--print-latex-stdout", action="store_true", help="Print latex table to stdout")
    args = p.parse_args()

    llm_response_text = _load_text_arg(args.llm_response, args.llm_response_file)
    source_prompt = _load_text_arg(args.source_prompt, args.source_prompt_file)
    obstacles = _load_obstacles(args.obstacles_json, args.obstacles_file)

    current = np.array([args.current_x, args.current_y, args.current_z], dtype=float)
    velocity = np.array([args.vel_x, args.vel_y, args.vel_z], dtype=float)
    goal = np.array([args.goal_x, args.goal_y, args.goal_z], dtype=float)

    res = evaluate_response(
        current_position_ned=current,
        goal_ned=goal,
        llm_response_text=llm_response_text,
        obstacle_points_ned=obstacles,
        dt=args.eval_dt,
        v_max=args.eval_v_max,
        a_max=args.eval_a_max,
        safety_radius=args.eval_safety_radius,
    )

    print(f"EVAL PASS: {res.passed}")
    print(
        f"Checks: format={res.format_pass}, progress={res.progress_pass}, "
        f"kinematic={res.kinematic_pass}, safety={res.safety_pass}"
    )
    if res.errors:
        for err in res.errors:
            print(f"ERROR: {err}")
    if res.warnings:
        for warn in res.warnings:
            print(f"WARN: {warn}")

    latex_table = build_latex_style_eval_table(res)
    if args.print_latex_stdout:
        print("\n===== COPY-PASTE LATEX TABLE START =====")
        print(latex_table)
        print("===== COPY-PASTE LATEX TABLE END =====\n")
    print("Metrics:")
    print(json.dumps(_sanitize_for_json(res.metrics), indent=2))

    if args.save_eval_logs:
        tex_path, json_path, response_path = save_eval_logs(
            out_dir=Path(args.eval_log_dir).expanduser(),
            res=res,
            latex_table=latex_table,
            llm_response_text=llm_response_text,
            current_position=current,
            current_velocity=velocity,
            goal_ned=goal,
            obstacle_count=int(obstacles.shape[0]),
        )
        print(f"Saved evaluation table: {tex_path}")
        print(f"Saved evaluation metrics: {json_path}")
        print(f"Saved parsed LLM response: {response_path}")

    if args.save_eval_memory:
        mem_path = Path(args.eval_memory_path).expanduser()
        append_eval_memory(
            mem_path=mem_path,
            res=res,
            llm_response_text=llm_response_text,
            latex_table=latex_table,
            source_prompt=source_prompt,
            current_position=current,
            current_velocity=velocity,
            goal_ned=goal,
            obstacle_count=int(obstacles.shape[0]),
        )
        print(f"Appended evaluation memory: {mem_path}")


if __name__ == "__main__":
    main()
