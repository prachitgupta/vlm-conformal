# Assignment 3 Setup

## Environment

- Python 3.10+ is used in this workspace.
- The Assignment 3 scripts are self-contained and do not require PX4, Gazebo, or ROS execution for the verifier/evaluation loop.
- The planning tool reuses the repo's offline planning utilities in `llm_drone.llm.offline_ground_truth_support`.

## Required Python Packages

Install these packages in your active environment:

```bash
python3 -m pip install --user openai instructor pydantic numpy cvxpy
```

Expected package floor:

- `openai >= 2.0.0`
- `instructor >= 1.4.0`
- `pydantic >= 2.7.0`
- `numpy >= 1.24.0`
- `cvxpy >= 1.4.0`

If `cvxpy` is unavailable, the planning tool still runs but falls back to the hybrid-A* initialization path without the global smoothing pass.

## API Key

Load the OpenAI key through the environment:

```bash
export OPENAI_API_KEY='your_key_here'
```

Then reload your shell if needed:

```bash
source ~/.bashrc
```

## Files

Main Assignment 3 files:

- `problem_set.py`
- `verifier.py`
- `baseline.py`
- `tool_pipeline.py`
- `tool_eval.py`
- `refinement.py`
- `hybrid_astar_mpc_tool.py`
- `fixed_prompt.txt`
- `problem_prompts/P1.txt` ... `problem_prompts/P5.txt`
- `show_prompts.py`
- `ECE498BH_HW3.tex`

## Running

From the `assign3/` directory:

Manual verifier tests:

```bash
python3 verifier.py --manual-tests
```

Baseline evaluation:

```bash
python3 baseline.py --mode live
python3 baseline.py --mode mock
```

Single tool-augmented run:

```bash
python3 tool_pipeline.py --problem-id P5 --mode live
```

Tool-augmented five-trial evaluation:

```bash
python3 tool_eval.py --mode live
python3 tool_eval.py --mode mock
```

Self-refinement with tools on the hardest problem:

```bash
python3 refinement.py --problem-id P5 --mode live
python3 refinement.py --problem-id P5 --mode mock
```

Inspect the exact prompt pair used for one problem:

```bash
python3 show_prompts.py --problem-id P3
```

Compile the report:

```bash
pdflatex ECE498BH_HW3.tex
```

## Notes

- `--mode live` uses the OpenAI API with `instructor`-enforced structured output.
- In `--mode live`, the model sees a fixed system prompt from `fixed_prompt.txt` plus a problem-specific user prompt from `problem_prompts/P1.txt` through `problem_prompts/P5.txt`.
- `--mode mock` does not query the LLM; it provides deterministic hand-crafted pass/fail outputs so the whole pipeline can still be exercised when API access is unavailable.
- No API keys appear in any submitted file.
