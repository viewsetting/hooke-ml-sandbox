"""JSONL metrics utilities."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


@dataclass(frozen=True)
class MetricsSummary:
    min_loss: float
    last_loss: float
    steps: int


def summarize_losses(path: Path) -> MetricsSummary:
    losses: list[float] = []
    for row in iter_jsonl(path):
        if "loss" in row:
            losses.append(float(row["loss"]))
    if not losses:
        raise ValueError(f"no loss values found in {path}")
    return MetricsSummary(min_loss=min(losses), last_loss=losses[-1], steps=len(losses))

