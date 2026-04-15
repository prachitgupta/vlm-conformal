#!/usr/bin/env python3
"""Shared LLM/mocked pipeline utilities for Assignment 3."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import instructor
from openai import OpenAI

from hybrid_astar_mpc_tool import plan_problem
from problem_set import (
    WaypointSolution,
    build_problem_prompt,
    get_problem,
)
from verifier import _manual_fail_case, _manual_pass_case


DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_PROMPT_FILE = Path(__file__).resolve().with_name("fixed_prompt.txt")
DEFAULT_PROBLEM_PROMPT_DIR = Path(__file__).resolve().with_name("problem_prompts")


def load_fixed_prompt(prompt_file: str | Path = DEFAULT_PROMPT_FILE) -> str:
    return Path(prompt_file).expanduser().resolve().read_text(encoding="utf-8").strip()


def load_problem_prompt(
    problem_id: str,
    prompt_dir: str | Path = DEFAULT_PROBLEM_PROMPT_DIR,
) -> str:
    prompt_path = Path(prompt_dir).expanduser().resolve() / f"{str(problem_id).upper()}.txt"
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8").strip()
    return build_problem_prompt(problem_id)


def create_client() -> instructor.Instructor:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    return instructor.from_openai(OpenAI(api_key=api_key))


def _tool_context(problem_id: str) -> dict[str, Any]:
    result = plan_problem(problem_id)
    return {
        "problem_id": result["problem_id"],
        "solution": result["solution_dict"],
        "tool_metadata": result["tool_metadata"],
    }


def _mock_baseline_solution(problem_id: str, trial_index: int) -> WaypointSolution:
    pass_schedule = {
        "P1": {1, 2, 4, 5},
        "P2": {1, 3, 4},
        "P3": {2},
        "P4": {3},
        "P5": set(),
    }
    if int(trial_index) in pass_schedule[str(problem_id).upper()]:
        return _manual_pass_case(problem_id)
    return _manual_fail_case(problem_id)


def _mock_tool_solution(problem_id: str) -> WaypointSolution:
    return plan_problem(problem_id)["solution"]


def _live_messages(
    *,
    problem_prompt: str,
    tool_payload: dict[str, Any] | None = None,
    verifier_feedback: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    user_parts = [problem_prompt]
    if tool_payload is not None:
        user_parts.append(
            "Planning tool output (you may use it directly if it satisfies the task):\n"
            + json.dumps(tool_payload, indent=2)
        )
    if verifier_feedback is not None:
        user_parts.append(
            "Verifier feedback from the previous attempt:\n"
            + json.dumps(verifier_feedback, indent=2)
        )
    return [{"role": "user", "content": "\n\n".join(user_parts)}]


def query_solution(
    *,
    problem_id: str,
    mode: str,
    system_prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
    trial_index: int = 1,
    tool_enabled: bool = False,
    verifier_feedback: dict[str, Any] | None = None,
    problem_prompt_dir: str | Path = DEFAULT_PROBLEM_PROMPT_DIR,
) -> tuple[WaypointSolution, dict[str, Any]]:
    resolved_mode = str(mode).strip().lower()
    prompt_dir_path = Path(problem_prompt_dir).expanduser().resolve()
    problem_prompt_path = prompt_dir_path / f"{str(problem_id).upper()}.txt"
    problem_prompt_text = load_problem_prompt(problem_id, prompt_dir=prompt_dir_path)

    if resolved_mode == "mock":
        if tool_enabled:
            return _mock_tool_solution(problem_id), {
                "backend": "mock",
                "tool_used": True,
                "fixed_prompt_file": str(DEFAULT_PROMPT_FILE.resolve()),
                "problem_prompt_file": str(problem_prompt_path),
                "problem_prompt_text": problem_prompt_text,
            }
        if verifier_feedback is not None:
            return _mock_tool_solution(problem_id), {
                "backend": "mock",
                "tool_used": True,
                "refined": True,
                "fixed_prompt_file": str(DEFAULT_PROMPT_FILE.resolve()),
                "problem_prompt_file": str(problem_prompt_path),
                "problem_prompt_text": problem_prompt_text,
            }
        return _mock_baseline_solution(problem_id, trial_index), {
            "backend": "mock",
            "tool_used": False,
            "fixed_prompt_file": str(DEFAULT_PROMPT_FILE.resolve()),
            "problem_prompt_file": str(problem_prompt_path),
            "problem_prompt_text": problem_prompt_text,
        }

    if resolved_mode != "live":
        raise ValueError(f"Unsupported mode {mode!r}; expected 'live' or 'mock'")

    client = create_client()
    tool_payload = _tool_context(problem_id) if tool_enabled else None
    messages = _live_messages(
        problem_prompt=problem_prompt_text,
        tool_payload=tool_payload,
        verifier_feedback=verifier_feedback,
    )
    response = client.chat.completions.create(
        model=model,
        response_model=WaypointSolution,
        messages=[{"role": "system", "content": system_prompt}, *messages],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response, {
        "backend": "live",
        "tool_used": tool_enabled,
        "tool_payload": tool_payload,
        "fixed_prompt_file": str(DEFAULT_PROMPT_FILE.resolve()),
        "problem_prompt_file": str(problem_prompt_path),
        "problem_prompt_text": problem_prompt_text,
    }


def serialize_trial(
    *,
    problem_id: str,
    trial_index: int,
    solution: WaypointSolution,
    verification: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "problem_id": problem_id,
        "trial": int(trial_index),
        "solution": solution.model_dump(),
        "llm_response_text": solution.model_dump_json(indent=2),
        "verification": verification,
        "metadata": metadata,
    }


def tool_worked_example(problem_id: str = "P5") -> dict[str, Any]:
    tool_payload = _tool_context(problem_id)
    fixed_prompt = load_fixed_prompt()
    problem_prompt = load_problem_prompt(problem_id)
    return {
        "problem_id": problem_id,
        "fixed_prompt_excerpt": fixed_prompt[:1000],
        "problem_prompt_excerpt": problem_prompt[:1400],
        "tool_output": tool_payload,
        "final_answer": tool_payload["solution"],
    }


def problem_report_rows() -> list[dict[str, Any]]:
    rows = []
    for problem_id in ("P1", "P2", "P3", "P4", "P5"):
        problem = get_problem(problem_id)
        rows.append(
            {
                "problem_id": problem.problem_id,
                "difficulty": problem.difficulty,
                "title": problem.title,
                "statement": problem.statement,
                "deliverables": problem.deliverables,
                "rationale": problem.difficulty_rationale,
            }
        )
    return rows
