# Prompt Variants for `llm_prompt2d.txt`

This directory contains 10 system-prompt variants for quick A/B testing:

1. `01_baseline_full.txt` — unchanged baseline copy of `llm_prompt2d.txt`.
2. `02_no_reasoning_keep_examples.txt` — reasoning removed, examples retained.
3. `03_no_reasoning_one_example.txt` — reasoning removed, only Example A retained.
4. `04_no_reasoning_no_examples.txt` — reasoning removed, all examples removed.
5. `05_reasoning_no_examples.txt` — reasoning retained, all examples removed.
6. `06_concise_with_reasoning.txt` — concise prompt with reasoning in output schema.
7. `07_concise_no_reasoning.txt` — concise prompt with waypoints-only schema.
8. `08_no_reasoning_center_bias.txt` — no reasoning plus extra center-progress hint.
9. `09_reasoning_two_examples.txt` — reasoning retained with two examples (A+B).
10. `10_minimal_waypoints_only.txt` — minimal contract-only waypoints prompt.
