# Plan for Task 633de9ba

**Task ID:** 633de9bae39e426eaa2674d31c062fc9
**Run ID:** task-20260205-633de9ba

## Steps

1. [x] Create `scripts/evaluate_linreg.py` — reads a metrics.jsonl via `--path` arg, prints `min_loss` and `last_loss`. Uses only stdlib (`json`, `argparse`, `sys`, `pathlib`). Follow existing code style from `analyze_metrics.py`.
2. [x] Update `README.md` — add evaluate usage example in the Quickstart section.
3. [x] Run `python3 scripts/run_tests.py` — verify all tests pass.
4. [x] Commit and push to branch, open PR.
