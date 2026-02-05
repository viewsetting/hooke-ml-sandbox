"""Tiny linear regression implementation (pure Python).

Model: y_hat = a*x + b
Loss:  MSE over the dataset
Optim: gradient descent
"""

from __future__ import annotations

from dataclasses import dataclass

from ml_sandbox.data import Dataset


@dataclass(frozen=True)
class LinRegParams:
    a: float
    b: float


def predict(params: LinRegParams, xs: list[float]) -> list[float]:
    return [params.a * x + params.b for x in xs]


def mse(preds: list[float], ys: list[float]) -> float:
    if len(preds) != len(ys):
        raise ValueError("preds and ys must have the same length")
    n = len(ys)
    if n == 0:
        raise ValueError("empty dataset")
    return sum((p - y) ** 2 for p, y in zip(preds, ys)) / n


def train(
    dataset: Dataset,
    *,
    steps: int = 200,
    lr: float = 0.1,
    init: LinRegParams | None = None,
) -> tuple[LinRegParams, list[dict[str, float]]]:
    """Train linear regression via gradient descent.

    Returns (params, metrics_history), where each metrics entry includes:
      - step
      - loss
      - a
      - b
    """
    if steps <= 0:
        raise ValueError("steps must be > 0")
    if lr <= 0:
        raise ValueError("lr must be > 0")

    a = init.a if init else 0.0
    b = init.b if init else 0.0
    xs = dataset.xs
    ys = dataset.ys
    n = len(xs)
    if n == 0:
        raise ValueError("empty dataset")

    history: list[dict[str, float]] = []

    for step in range(steps):
        preds = [a * x + b for x in xs]
        loss = sum((p - y) ** 2 for p, y in zip(preds, ys)) / n

        # d/d(a) MSE = 2/n * sum((pred - y) * x)
        # d/d(b) MSE = 2/n * sum((pred - y))
        da = (2.0 / n) * sum((p - y) * x for p, y, x in zip(preds, ys, xs))
        db = (2.0 / n) * sum((p - y) for p, y in zip(preds, ys))

        a -= lr * da
        b -= lr * db

        history.append(
            {
                "step": float(step),
                "loss": float(loss),
                "a": float(a),
                "b": float(b),
            }
        )

    return LinRegParams(a=a, b=b), history

