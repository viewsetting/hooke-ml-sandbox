# hooke-ml-sandbox

A tiny, self-contained "ML project" repo to test Project_Hooke end-to-end.

Goals:
- Fast to run on a laptop (incl. Apple Silicon)
- No heavy deps (pure Python)
- Has a training script + analysis script + tests
- Friendly to Hooke's workflow:
  - Hooke writes `.hooke/TASK.md` and `.hooke/PLAN.md` into run worktrees
  - Hooke proposes patches + runs commands + opens PRs

## Quickstart

Train a tiny linear regression model on synthetic data:

```bash
python3 scripts/train_linreg.py --steps 200 --lr 0.1 --seed 0 --out runs/demo.jsonl
python3 scripts/analyze_metrics.py --path runs/demo.jsonl
python3 scripts/evaluate_linreg.py --path runs/demo.jsonl
```

Run tests (uses built-in `unittest`):

```bash
python3 scripts/run_tests.py
```
