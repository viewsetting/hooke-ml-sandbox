#!/usr/bin/env python3
"""Simple analysis script for a JSONL metrics file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _ensure_repo_root_on_path() -> None:
    root = Path(__file__).resolve().parents[1]
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def main() -> int:
    _ensure_repo_root_on_path()

    from ml_sandbox.metrics import summarize_losses

    p = argparse.ArgumentParser()
    p.add_argument("--path", required=True, help="Path to metrics JSONL file")
    args = p.parse_args()

    path = Path(args.path)
    summary = summarize_losses(path)
    print(f"path: {path}")
    print(f"steps: {summary.steps}")
    print(f"min_loss: {summary.min_loss:.6f}")
    print(f"last_loss: {summary.last_loss:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

