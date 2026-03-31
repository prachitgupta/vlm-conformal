#!/usr/bin/env python3
"""
Repeated sampling: 5 independent API calls, automatic verifier on each.
Usage:  python repeated_sampling.py
Requires: ANTHROPIC_API_KEY environment variable.
"""
import os, json
import anthropic, instructor
from pipeline import WaypointSolution, SYSTEM_PROMPT, USER_PROMPT
from verifier import verify


def run_repeated_sampling(n: int = 5):
    client = instructor.from_anthropic(
        anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    )
    results = []
    for trial in range(1, n + 1):
        solution: WaypointSolution = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": USER_PROMPT}],
            response_model=WaypointSolution,
        )
        result = verify(solution)
        status = "Pass" if result["pass"] else "Fail"
        print(f"Trial {trial}: {status}  | reason: {result['reason'][:80]}")
        results.append({"trial": trial, "pass": result["pass"],
                        "reason": result["reason"]})

    pass_count = sum(r["pass"] for r in results)
    print(f"\nOverall: {pass_count}/{n} Pass")
    return results


if __name__ == "__main__":
    run_repeated_sampling()