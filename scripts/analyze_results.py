"""Error analysis: score a predictions CSV and break down accuracy by group.

Reusable for every benchmark run (baselines, API models, GRPO checkpoints):

    python scripts/analyze_results.py --split data/split/test.json \\
        --predictions results/baseline_latex_parse_2026*.csv [--by equation_type]

Groups: equation_type | domain | complexity | output_type
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from math2code.evaluation.eval import load_split, score_predictions_csv  # noqa: E402
from math2code.schemas import MathCodePair  # noqa: E402


def analyze(
    pairs: list[MathCodePair],
    predictions_file: str,
    by: str,
) -> None:
    result = score_predictions_csv(predictions_file, pairs)
    by_id = {p.task_id: p for p in pairs}
    groups: dict[str, list[bool]] = defaultdict(list)
    for d in result.details:
        if d.task_id not in by_id:
            continue
        group = getattr(by_id[d.task_id], by)
        groups[str(group)].append(d.passed)

    print(f"\n== per-problem accuracy by {by} ==")
    print(f"{'group':<28} {'problems':>8} {'accuracy':>10}")
    for group, passed in sorted(groups.items(), key=lambda kv: -sum(kv[1])):
        n = len(passed)
        acc = sum(passed) / n
        print(f"{group:<28} {n:>8} {acc:>10.3f}  ({sum(passed)}/{n})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--predictions", required=True)
    ap.add_argument(
        "--by",
        default="equation_type",
        choices=["equation_type", "domain", "complexity", "output_type"],
    )
    args = ap.parse_args()
    pairs = load_split(args.split)
    analyze(pairs, args.predictions, args.by)


if __name__ == "__main__":
    main()
