"""Synthetic datasets for small ML demos (no external deps)."""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Dataset:
    xs: list[float]
    ys: list[float]


def make_linreg_dataset(
    *,
    n: int = 64,
    seed: int = 0,
    true_a: float = 2.0,
    true_b: float = -0.5,
    noise_std: float = 0.1,
) -> Dataset:
    """Generate y = a*x + b + noise."""
    rng = random.Random(seed)
    xs = [rng.uniform(-1.0, 1.0) for _ in range(n)]
    ys = [true_a * x + true_b + rng.gauss(0.0, noise_std) for x in xs]
    return Dataset(xs=xs, ys=ys)

