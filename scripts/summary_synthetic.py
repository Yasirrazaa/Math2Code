#!/usr/bin/env python3
"""Distribution-shape report for the synthetic pool + training mixture.

Measurement-first evidence for the strategy's distribution-shape pass
(docs/DATA_STRATEGY.md §6-7): complexity histogram (normalized across the
string/int mismatch between raw split rows and synthetic rows), coefficient
kinds, cross-term share, slice composition, and a per-function vocabulary
scan. Prints a compact table; exit 0.

Usage:
  python scripts/summary_synthetic.py [--train data/split/train.jsonl]
                                      [--mixture data/synthetic/train_mixture_v1.jsonl]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

SYNTH_DIR = Path("data/synthetic")
FAMILY_FILES = [
    "calculus_indefinite_v1.jsonl",
    "calculus_definite_v1.jsonl",
    "calculus_variable_v1.jsonl",
    "functions_v1.jsonl",
    "ode_v1.jsonl",
    "multivariate_v1.jsonl",
    "sequences_v1.jsonl",
    "geometry_v1.jsonl",
]

_FUNCS = (
    "sin",
    "cos",
    "tan",
    "sec",
    "csc",
    "cot",
    "asin",
    "acos",
    "atan",
    "sinh",
    "cosh",
    "tanh",
    "log",
    "ln",
    "exp",
    "sqrt",
    "Abs",
    "floor",
    "ceiling",
    "factorial",
    "binomial",
    "Min",
    "Max",
    "Mod",
    "sign",
)


def _norm_complexity(v: object) -> int:
    """Raw split rows carry string complexity; synthetic rows carry int."""
    try:
        return int(str(v))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0


def _functions_in_latex(tex: str) -> set[str]:
    found: set[str] = set()
    for f in _FUNCS:
        if f"\\{f}" in tex or f"{f}(" in tex or f"\\operatorname{{{f}}}" in tex:
            found.add(f)
    if "\\pi" in tex:
        found.add("pi")
    return found


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", default="data/split/train.jsonl")
    ap.add_argument("--mixture", default="data/synthetic/train_mixture_v1.jsonl")
    args = ap.parse_args()

    def load(path: Path) -> list[dict]:
        if not path.exists():
            return []
        with open(path) as fh:
            return [json.loads(line) for line in fh if line.strip()]

    train = load(Path(args.train))
    mixture = load(Path(args.mixture))
    pool: list[dict] = []
    for name in FAMILY_FILES:
        pool += load(SYNTH_DIR / name)

    print(f"frozen train      : {len(train):6d} rows")
    print(
        f"synthetic pool    : {len(pool):6d} rows "
        f"({len({r['latex_expression'] for r in pool})} unique latex)"
    )
    print(
        f"training mixture  : {len(mixture):6d} rows "
        f"({100 * len(mixture) / max(len(train), 1):.1f}% of frozen-train size)"
    )

    if pool:
        slices = Counter(r.get("metadata", {}).get("slice", "other") for r in pool)
        print("\nsynthetic slices :", dict(slices))
        kinds = Counter(
            r.get("metadata", {}).get("coefficient_kind", "n/a") for r in pool
        )
        print("coefficient kinds:", dict(kinds))
        cross = sum(1 for r in pool if r.get("metadata", {}).get("cross_terms") is True)
        print(f"cross-term rows  : {cross} / {len(pool)}")

    print("\ncomplexity histogram (1-5):")
    print(f"  {'train':>22s}", _hist(train))
    if pool:
        print(f"  {'synthetic pool':>22s}", _hist(pool))
    if mixture:
        print(f"  {'mixture':>22s}", _hist(mixture))

    if pool:
        vocab: Counter[str] = Counter()
        for r in pool:
            vocab.update(_functions_in_latex(r.get("latex_expression", "")))
        print(f"\nfunction vocabulary (synthetic pool): {dict(sorted(vocab.items()))}")
    return 0


def _hist(rows: list[dict]) -> dict[int, int]:
    return dict(
        sorted(Counter(_norm_complexity(r.get("complexity")) for r in rows).items())
    )


if __name__ == "__main__":
    raise SystemExit(main())
