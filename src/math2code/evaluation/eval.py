"""Evaluation harness.

The competition scoring metric: parse submitted outputs (incl. complex), compare
numerically with tolerance, report per-case + per-problem accuracy.

Subcommands:
    gold   -- sanity check: gold solutions from a scored split must score 1.0
    score  -- score a predictions CSV (id,outputs) against a scored split
    bench  -- run a model on the test set and score it (Weeks 3+; requires GPU/API)

Example:
    python -m math2code.evaluation.eval gold   --split data/split/test.json --n 50
    python -m math2code.evaluation.eval score  --split data/split/test.json --predictions results/foo.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

from math2code.evaluation.metrics import ScoreResult, score_predictions
from math2code.sandbox import SandboxPool
from math2code.schemas import MathCodePair


def load_split(path: str | Path) -> list[MathCodePair]:
    with open(path) as f:
        rows = json.load(f)
    if isinstance(rows, dict) and "items" in rows:  # our own manifest format
        rows = rows["items"]
    return [MathCodePair.model_validate(r) for r in rows]


def gold_check(
    pairs: list[MathCodePair], n: int | None, pool: SandboxPool
) -> ScoreResult:
    """Run gold solutions through the sandbox; must score 1.0 to trust the harness."""
    subset = pairs if n is None else pairs[:n]
    predictions: list[list[Any]] = []
    cases: list[list[Any]] = []
    ids: list[str] = []
    for p in subset:
        if not p.solution:
            raise ValueError(f"{p.task_id}: no gold solution")
        outputs, errors = pool.run_solution_on_cases(
            p.solution, [tc.input for tc in p.test_cases]
        )
        if errors:
            print(
                f"  [warn] {p.task_id}: execution errors -> {errors[0][:80]}",
                file=sys.stderr,
            )
        predictions.append(outputs)  # type: ignore[arg-type]
        cases.append(p.test_cases)  # type: ignore[arg-type]
        ids.append(p.task_id)
    return score_predictions(predictions, cases, ids)  # type: ignore[arg-type]


def score_predictions_csv(
    csv_path: str | Path, pairs: list[MathCodePair]
) -> ScoreResult:
    """Score a submission CSV (id,outputs) against a scored split."""
    by_id = {p.task_id: p for p in pairs}
    predictions: list[list[Any]] = []
    cases: list[list[Any]] = []
    ids: list[str] = []
    missing = 0
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            tid = row["id"].strip()
            pair = by_id.get(tid)
            if pair is None:
                missing += 1
                continue
            try:
                outputs = json.loads(row["outputs"])
            except (json.JSONDecodeError, KeyError):
                outputs = [None]
            if not isinstance(outputs, list):
                outputs = [outputs]
            predictions.append(outputs)
            cases.append(pair.test_cases)
            ids.append(tid)
    if missing:
        print(
            f"  [warn] {missing} prediction ids not in the scored split",
            file=sys.stderr,
        )
    return score_predictions(predictions, cases, ids)


def main() -> None:
    ap = argparse.ArgumentParser(description="Math2Code evaluation harness")
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gold")
    g.add_argument(
        "--split", required=True, help="scored split json (MathCodePair rows)"
    )
    g.add_argument("--n", type=int, default=None, help="limit rows")
    g.add_argument("--workers", type=int, default=4)
    g.add_argument(
        "--timeout",
        type=float,
        default=20.0,
        help="per-case sandbox timeout (eval, not RL: 20s)",
    )
    g.add_argument(
        "--memory",
        type=int,
        default=4096,
        help="per-worker memory cap in MB (eval: 4096; RL rollouts use ~512)",
    )

    s = sub.add_parser("score")
    s.add_argument("--split", required=True)
    s.add_argument("--predictions", required=True, help="submission CSV: id,outputs")

    b = sub.add_parser("bench")
    b.add_argument("--split", required=True)
    b.add_argument("--model", required=True, help="model id (HF) or api:<name>")
    b.add_argument("--out", default="results/latest.json")

    args = ap.parse_args()

    if args.cmd == "gold":
        pairs = load_split(args.split)
        with SandboxPool(
            n_workers=args.workers, timeout_s=args.timeout, memory_mb=args.memory
        ) as pool:
            result = gold_check(pairs, args.n, pool)
        print(result.summary())
        if result.per_problem_accuracy < 0.99:
            print("GOLD CHECK FAILED: harness or sandbox is broken.", file=sys.stderr)
            sys.exit(1)

    elif args.cmd == "score":
        pairs = load_split(args.split)
        result = score_predictions_csv(args.predictions, pairs)
        print(result.summary())
        Path("results").mkdir(exist_ok=True)
        out = Path(args.predictions).with_suffix(".scored.json")
        out.write_text(json.dumps(_result_to_dict(result), indent=2))
        print(f"saved -> {out}")

    elif args.cmd == "bench":
        from math2code.evaluation.runner import benchmark_model  # Week 3 module

        result = benchmark_model(args.split, args.model, args.out)
        print(result.summary())


def _result_to_dict(r: ScoreResult) -> dict:
    return {
        "n_cases": r.n_cases,
        "n_correct_cases": r.n_correct_cases,
        "n_problems": r.n_problems,
        "n_correct_problems": r.n_correct_problems,
        "per_case_accuracy": r.per_case_accuracy,
        "per_problem_accuracy": r.per_problem_accuracy,
    }


if __name__ == "__main__":
    main()
