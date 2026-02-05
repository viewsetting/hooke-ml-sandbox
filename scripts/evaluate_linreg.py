#!/usr/bin/env python3
"""Evaluate a linear regression run by reading its metrics JSONL file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description="Evaluate linear regression metrics")
    p.add_argument("--path", required=True, help="Path to metrics JSONL file")
    args = p.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"error: file not found: {path}", file=sys.stderr)
        return 1

    losses: list[float] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "loss" in row:
                losses.append(float(row["loss"]))

    if not losses:
        print(f"error: no loss values found in {path}", file=sys.stderr)
        return 1

    min_loss = min(losses)
    last_loss = losses[-1]
    print(f"min_loss: {min_loss:.6f}")
    print(f"last_loss: {last_loss:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
