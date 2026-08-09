"""Zero-cost baselines for the frozen test split (no LLM, no GPU, no API).

These establish the floor before any model money is spent:

1. `latex_parse_baseline`  — direct `sympy.parsing.latex.parse_latex` of the
   LaTeX, substituted at the test inputs, evaluated numerically. Measures how
   much of the task is *not* "just parse the LaTeX" (the competition's LaTeX
   is generated, non-standard, and frequently unparseable).
2. `trivial_floor`         — always answers 0. A pure chance/floor row: how
   often the correct answer is 0, and a strict lower bound any trained model
   must beat.

Both produce submission-format CSV + scored JSON identical to runner.py
outputs, so the README table is apples-to-apples.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import sympy as sp

from math2code.evaluation.eval import load_split
from math2code.evaluation.metrics import ScoreResult, bootstrap_ci, score_predictions
from math2code.schemas import MathCodePair

RESULTS_DIR = Path("results")


def latex_parse_baseline(pairs: list[MathCodePair]) -> list[list[str | None]]:
    """Try to parse the LaTeX directly with SymPy and evaluate at the inputs.

    Requires the optional `baseline` extra (antlr4-python3-runtime==4.11.1),
    which conflicts with hydra/omegaconf and so is not in [dev] or [train].
    """
    try:
        from sympy.parsing.latex import parse_latex

        parse_latex("x")  # probe: antlr4 runtime availability fails at call time
    except Exception as exc:
        raise RuntimeError(
            "parse_latex unavailable: install the optional baseline extra in a "
            "dedicated venv — `uv pip install -e '.[baseline]'` "
            "(antlr4 4.11.x conflicts with hydra/omegaconf 4.9.x, so it is not "
            "in the default extras)."
        ) from exc

    predictions: list[list[str | None]] = []
    for p in pairs:
        outs: list[str | None] = []
        try:
            expr = parse_latex(p.latex_expression)
        except Exception:
            expr = None
        for tc in p.test_cases:
            if expr is None:
                outs.append(None)
                continue
            try:
                subs = {sp.Symbol(k): v for k, v in tc.input.items()}
                val = complex(expr.subs(subs).evalf())
                if abs(val.imag) < 1e-9:
                    outs.append(str(val.real))
                else:
                    outs.append(f"{val.real}{val.imag:+}j")
            except Exception:
                outs.append(None)
        predictions.append(outs)
    return predictions


def trivial_floor(pairs: list[MathCodePair]) -> list[list[str | None]]:
    """Always answer 0.0 — the strict floor."""
    return [["0.0"] * len(p.test_cases) for p in pairs]


def run_baseline(
    pairs: list[MathCodePair],
    name: str,
    fn: Any,
    results_dir: Path = RESULTS_DIR,
) -> tuple[ScoreResult, Path]:
    t0 = time.time()
    predictions = fn(pairs)
    dt = time.time() - t0
    cases = [p.test_cases for p in pairs]  # type: ignore[arg-type]
    ids = [p.task_id for p in pairs]
    result = score_predictions(predictions, cases, ids)  # type: ignore[arg-type]

    results_dir.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    csv_path = results_dir / f"{name}_{stamp}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "outputs"])
        for tid, preds in zip(ids, predictions):
            writer.writerow([tid, json.dumps(preds)])

    passed = [all(d.passed for d in result.details if d.task_id == tid) for tid in ids]
    lo, hi = bootstrap_ci(passed)
    report = {
        "baseline": name,
        "timestamp": stamp,
        "n_problems": result.n_problems,
        "n_correct_problems": result.n_correct_problems,
        "per_problem_accuracy": result.per_problem_accuracy,
        "ci95": [lo, hi],
        "per_case_accuracy": result.per_case_accuracy,
        "wall_seconds": round(dt, 1),
        "predictions_file": str(csv_path),
    }
    (results_dir / f"{name}_{stamp}.json").write_text(json.dumps(report, indent=2))
    print(f"  [{name}] {result.summary().replace(chr(10), ' | ')}  ({dt:.1f}s)")
    print(f"  saved -> {csv_path}")
    return result, csv_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Zero-cost baselines on the test split")
    ap.add_argument("--split", default="data/split/test.json")
    args = ap.parse_args()

    pairs = load_split(args.split)
    print(f"test split: {len(pairs)} problems")
    for name, fn in (
        ("baseline_latex_parse", latex_parse_baseline),
        ("baseline_trivial_floor", trivial_floor),
    ):
        result, _ = run_baseline(pairs, name, fn)
        print(
            f"    {name}: {result.per_problem_accuracy:.4f} "
            f"({result.n_correct_problems}/{result.n_problems})"
        )


if __name__ == "__main__":
    main()
