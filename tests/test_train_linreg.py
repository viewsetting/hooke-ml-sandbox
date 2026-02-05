from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


class TestTrainLinreg(unittest.TestCase):
    def test_train_script_writes_metrics_and_converges(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "metrics.jsonl"
            proc = subprocess.run(
                [
                    sys.executable,
                    "scripts/train_linreg.py",
                    "--steps",
                    "200",
                    "--lr",
                    "0.1",
                    "--seed",
                    "0",
                    "--n",
                    "64",
                    "--out",
                    str(out),
                ],
                cwd=_repo_root(),
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            self.assertTrue(out.exists())

            rows = _read_jsonl(out)
            self.assertEqual(len(rows), 200)
            losses = [float(r["loss"]) for r in rows]
            self.assertLess(min(losses), losses[0])
            # Should reliably converge on this synthetic dataset.
            self.assertLess(losses[-1], 0.2)

    def test_analyze_script_runs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "metrics.jsonl"
            out.write_text('{"loss": 1.0}\n{"loss": 0.5}\n', encoding="utf-8")

            proc = subprocess.run(
                [sys.executable, "scripts/analyze_metrics.py", "--path", str(out)],
                cwd=_repo_root(),
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            self.assertIn("min_loss", proc.stdout)


if __name__ == "__main__":
    unittest.main()
