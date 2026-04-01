# Assignment 2 Setup

## Environment

- Python 3.10+ is used in this workspace.
- The report mentions Python 3.11+, but the current code also runs under Python 3.10 in this environment.
- No PX4, Gazebo, or external simulator is required for the verifier layer.

## Required Python Packages

Install these packages in your active environment:

```bash
python3 -m pip install --user openai instructor pydantic
```

Minimum package expectations from the report/code:

- `openai >= 2.0.0`
- `instructor >= 1.4.0`
- `pydantic >= 2.7.0`

## API Key

The code expects the OpenAI API key through an environment variable:

```bash
export OPENAI_API_KEY='your_key_here'
```

Then reload your shell if needed:

```bash
source ~/.bashrc
```

Check that it is set:

```bash
echo "$OPENAI_API_KEY"
```

## Files

Main files used in Assignment 2:

- `pipeline.py`
- `verifier.py`
- `repeat_sampling.py`
- `self_refinement.py`
- `fixed_prompt.txt`
- `assignment2.tex`

Input dataset used by the pipeline:

- `../dataset/dataset_merged_without_reasoning.csv`

## Running

From the `assignment2/` directory:

Single run on one dataset row:

```bash
python3 pipeline.py --row-index 0
```

Repeated sampling on one dataset row:

```bash
python3 repeat_sampling.py --row-index 0 --trials 5
```

Self-refinement on one dataset row:

```bash
python3 self_refinement.py --row-index 4 --max-turns 3
```

Write outputs to JSON files:

```bash
python3 pipeline.py --row-index 0 --output-json generated/pipeline_row0.json
python3 repeat_sampling.py --row-index 0 --output-json generated/repeat_row0.json
python3 self_refinement.py --row-index 4 --output-json generated/refine_row4.json
```

## Notes

- Different environment vectors are selected with `--row-index`.
- `repeat_sampling.py` uses the same fixed prompt and same environment prompt across all trials; only the API calls are repeated independently.
- `self_refinement.py` reuses the same environment prompt and feeds verifier feedback back into the model between turns.
- If the API returns `429 insufficient_quota`, billing/quota must be fixed in the OpenAI platform before the scripts can run successfully.
