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
_FAMILY_FILES = [
    "calculus_indefinite_v1.jsonl",
    "calculus_definite_v1.jsonl",
    "calculus_variable_v1.jsonl",
    "derivative_v1.jsonl",
    "functions_v1.jsonl",
    "ode_v1.jsonl",
    "ode_c1_v1.jsonl",
    "summation_v1.jsonl",
    "limits_v1.jsonl",
    "series_v1.jsonl",
    "elementary_v1.jsonl",
    "complex_v1.jsonl",
    "polynomials_v1.jsonl",
    "matrix_v1.jsonl",
    "multivariate_v1.jsonl",
    "sequences_v1.jsonl",
    "geometry_v1.jsonl",
    "geometry_ext_v1.jsonl",
    "ntheory_ext_v1.jsonl",
    "combinatorics_v1.jsonl",
    "edge_v1.jsonl",
    "numtheory_v1.jsonl",
    "special_v1.jsonl",
    "stats_v1.jsonl",
    "sets_v1.jsonl",
    "solving_v1.jsonl",
]

# measured frozen-test notation surface (docs/DATA_STRATEGY.md §audit): the
# alignment score = fraction of test patterns also present in the pool.
_NOTATION_PATTERNS = [
    ("repr-integral", r"\\mathtt\{\\text\{Integral\("),
    ("repr-derivative", r"\\mathtt\{\\text\{Derivative\("),
    ("d/dx", r"\\frac\{d\}\{d"),
    ("sum", r"\\sum_"),
    ("prod", r"\\prod_"),
    ("lim", r"\\lim_"),
    ("int", r"\\int"),
    ("partial", r"\\partial"),
    ("det", r"\\det"),
    ("tr", r"\\operatorname\{tr\}"),
    ("coeff", r"\\operatorname\{coeff\}"),
    ("xkbracket", r"\\left\[x\^"),
    ("binom", r"\\binom"),
    ("gcd", r"gcd"),
    ("varphi", r"\\varphi"),
    ("sigma", r"\\sigma"),
    ("log", r"\\log"),
    ("exp-notation", r"e\^\{?\\-?"),
    ("mathbbE", r"\\mathbb\{E\}"),
    ("nabla", r"\\nabla"),
    ("matrix", r"begin\{matrix\}"),
    ("pmatrix", r"begin\{pmatrix\}"),
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
    for name in _FAMILY_FILES:
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

    # notation-alignment audit vs frozen test (measurement-first surface check)
    test_path = Path("data/split/test.json")
    if test_path.exists():
        test_rows: list[dict] = json.loads(test_path.read_text())
    if test_rows:
        import re

        def _count(rows: list[dict]) -> Counter[str]:
            c: Counter[str] = Counter()
            for r in rows:
                tex = r.get("latex_expression", "")
                for name, pat in _NOTATION_PATTERNS:
                    if re.search(pat, tex):
                        c[name] += 1
            return c

        test_nt = _count(test_rows)
        pool_nt = _count(pool)
        present = {k for k, v in test_nt.items() if v > 0}
        aligned = {k for k in present if pool_nt.get(k, 0) > 0}
        print(
            f"\nnotation alignment vs frozen test: {len(aligned)}/{len(present)} "
            f"patterns covered ({100 * len(aligned) / max(len(present), 1):.0f}%)"
        )
        print(f"  test patterns : {dict(sorted(test_nt.items()))}")
        print(
            f"  pool patterns : {dict(sorted({k: v for k, v in pool_nt.items() if v > 0}.items()))}"
        )
    return 0


def _hist(rows: list[dict]) -> dict[int, int]:
    return dict(
        sorted(Counter(_norm_complexity(r.get("complexity")) for r in rows).items())
    )


if __name__ == "__main__":
    raise SystemExit(main())
