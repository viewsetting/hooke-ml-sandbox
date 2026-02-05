#!/usr/bin/env python3
"""Train a tiny linear regression model on synthetic data (pure Python)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _ensure_repo_root_on_path() -> None:
    # When running as `python scripts/train_linreg.py`, sys.path[0] is `scripts/`,
    # so ensure the repo root is importable.
    root = Path(__file__).resolve().parents[1]
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def main() -> int:
    _ensure_repo_root_on_path()

    from ml_sandbox.data import make_linreg_dataset
    from ml_sandbox.linreg import train
    from ml_sandbox.metrics import summarize_losses, write_jsonl

    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n", type=int, default=64, help="dataset size")
    p.add_argument("--out", type=str, default="runs/demo.jsonl")
    args = p.parse_args()

    ds = make_linreg_dataset(n=args.n, seed=args.seed)
    params, history = train(ds, steps=args.steps, lr=args.lr)

    out = Path(args.out)
    write_jsonl(out, history)
    summary = summarize_losses(out)

    print(f"wrote: {out}")
    print(f"final: loss={summary.last_loss:.6f}  a={params.a:.4f}  b={params.b:.4f}")
    print(f"best:  loss={summary.min_loss:.6f}  steps={summary.steps}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

