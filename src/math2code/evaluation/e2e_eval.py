"""E2B-sandboxed final eval: re-execute saved model code in E2B sandboxes.

Purpose: the plan's final-eval protocol — verify the trained model's code
inside E2B's isolated cloud sandboxes (free hobby credits, up to 20
concurrent), independent of the local worker pool. Any disagreement with the
local pool is reported explicitly.

Usage (after a runner run with --save-code):

    E2B_API_KEY=... python -m math2code.evaluation.e2e_eval \\
        --split data/split/test.json --code results/code_xxx.csv [--concurrency 20]

The e2b_code_interpreter SDK is imported lazily: the module imports and
--help work without it.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import time
from pathlib import Path
from typing import Any

from math2code.evaluation.eval import load_split
from math2code.evaluation.metrics import bootstrap_ci, score_predictions
from math2code.schemas import MathCodePair

RESULTS_DIR = Path("results")


def _load_code(code_path: Path) -> dict[str, str]:
    code_by_id: dict[str, str] = {}
    with open(code_path, newline="") as f:
        for row in csv.DictReader(f):
            code_by_id[row["id"]] = row["code"]
    return code_by_id


async def _exec_one(sandbox: Any, code: str, inputs: list[dict]) -> list[str | None]:
    """Run code in an E2B sandbox, returning stdout-of-calculate per input."""
    outs: list[str | None] = []
    for inp in inputs:
        runner = (
            "import json\n"
            "def __go():\n"
            "    return calculate(**json.loads(__INPUT__))\n"
            "print(__go())\n"
        )
        full = f"{code}\n__INPUT__ = {json.dumps(json.dumps(inp))}\n{runner}"
        try:
            exec_res = await sandbox.run_code(full)
            if exec_res.error is not None:
                outs.append(None)
            else:
                outs.append(str(exec_res.text).strip())
        except Exception:
            outs.append(None)
    return outs


async def _re_execute(
    pairs: list[MathCodePair],
    code_by_id: dict[str, str],
    concurrency: int,
    sandbox_cls: Any | None = None,
) -> dict[str, list[str | None]]:
    if sandbox_cls is None:
        from e2b_code_interpreter import Sandbox  # lazy: SDK only needed to run

        sandbox_cls = Sandbox

    sem = asyncio.Semaphore(concurrency)

    async def one(pair: MathCodePair) -> tuple[str, list[str | None]]:
        code = code_by_id.get(pair.task_id, "")
        if not code:
            return pair.task_id, [None] * len(pair.test_cases)
        async with sem:
            async with sandbox_cls() as sbx:  # type: ignore[operator]
                return pair.task_id, await _exec_one(
                    sbx, code, [tc.input for tc in pair.test_cases]
                )

    results = await asyncio.gather(*(one(p) for p in pairs))
    return dict(results)


def main() -> None:
    ap = argparse.ArgumentParser(description="E2B-sandboxed final eval")
    ap.add_argument("--split", required=True)
    ap.add_argument("--code", required=True, help="id,code CSV from runner --save-code")
    ap.add_argument("--concurrency", type=int, default=20)
    ap.add_argument("--max-problems", type=int, default=None)
    args = ap.parse_args()

    if not os.environ.get("E2B_API_KEY"):
        raise SystemExit("E2B_API_KEY not set (free credits: e2b.dev/docs/hobby)")

    pairs = load_split(args.split)
    if args.max_problems:
        pairs = pairs[: args.max_problems]
    code_by_id = _load_code(Path(args.code))
    print(f"re-executing {len(pairs)} problems in E2B (concurrency {args.concurrency})")

    t0 = time.time()
    preds_by_id = asyncio.run(_re_execute(pairs, code_by_id, args.concurrency))
    dt = time.time() - t0

    predictions = [
        preds_by_id.get(p.task_id, [None] * len(p.test_cases)) for p in pairs
    ]
    ids = [p.task_id for p in pairs]
    result = score_predictions(predictions, [p.test_cases for p in pairs], ids)
    passed = [all(d.passed for d in result.details if d.task_id == tid) for tid in ids]
    lo, hi = bootstrap_ci(passed)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"e2b_final_{stamp}.json"
    report = {
        "protocol": "e2b-sandboxed re-execution",
        "timestamp": stamp,
        "code_file": str(Path(args.code)),
        "n_problems": result.n_problems,
        "n_correct_problems": result.n_correct_problems,
        "per_problem_accuracy": result.per_problem_accuracy,
        "ci95": [lo, hi],
        "per_case_accuracy": result.per_case_accuracy,
        "wall_seconds": round(dt, 1),
    }
    out.write_text(json.dumps(report, indent=2))
    print(
        f"  per-problem accuracy: {result.per_problem_accuracy:.4f} "
        f"({result.n_correct_problems}/{result.n_problems})  CI95 [{lo:.4f}, {hi:.4f}]"
    )
    print(f"  saved -> {out}")


if __name__ == "__main__":
    main()
