#!/usr/bin/env python3
"""Build the training mixture: frozen competition train + verified synthetic.

Composition follows docs/DATA_STRATEGY.md §5-6 (defaults from the strategy):

  competition: 65%  (frozen `data/split/train.jsonl` — NEVER modified)
  synthetic:   35%  (verified `data/synthetic/*.jsonl` — oracle-accepted)

Synthetic slices (of the synthetic 35%): 30% calculus, 20% function
vocabulary, 15% multivariate, 10% ODE, 10% sequences/geometry/number-theory
(when available), 10% augmented/rational, 5% edge cases (when available).

Guarantees:
- deterministic (seed 42; input files are byte-identical artifacts)
- latex dedupe: no synthetic LaTeX collides with competition train, with
  other synthetic rows, or with the frozen test/val latex (contamination guard
  runs again here as a hard check)
- output is a SEPARATE artifact (data/synthetic/train_mixture.jsonl); the
  frozen split files and their SHA-256 manifest are untouched
- every emitted synthetic row carries verified ground truth (oracle-accepted)

Usage:
  python scripts/build_mixture.py [--synthetic data/synthetic/*.jsonl ...]
                                  [--out data/synthetic/train_mixture.jsonl]
                                  [--seed 42] [--size 22000]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

from math2code.data.synthesizer.core import int_seed
from math2code.schemas import MathCodePair, TestCase  # noqa: F401

# strategy composition (shares of the *synthetic* portion)
_SLICE_ORDER = [
    ("calculus", 0.30),
    ("vocab", 0.20),
    ("multivariate", 0.15),
    ("ode", 0.10),
    ("sequences", 0.05),
    ("geometry", 0.05),
    ("rational", 0.10),
    ("edge", 0.05),
]

_DEFAULT_SYNTH = [
    "data/synthetic/calculus_indefinite_v1.jsonl",
    "data/synthetic/calculus_definite_v1.jsonl",
    "data/synthetic/calculus_variable_v1.jsonl",
    "data/synthetic/functions_v1.jsonl",
    "data/synthetic/ode_v1.jsonl",
    "data/synthetic/multivariate_v1.jsonl",
    "data/synthetic/sequences_v1.jsonl",
    "data/synthetic/geometry_v1.jsonl",
    "data/synthetic/edge_v1.jsonl",
]


def load_jsonl(path: str) -> list[dict]:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--synthetic", nargs="*", default=_DEFAULT_SYNTH)
    ap.add_argument("--out", default="data/synthetic/train_mixture.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--size", type=int, default=22002)
    ap.add_argument("--train", default="data/split/train.jsonl")
    args = ap.parse_args()

    rng = random.Random(int_seed(f"mixture:{args.seed}"))

    # -- frozen competition train ------------------------------------------
    try:
        comp = load_jsonl(args.train)
    except FileNotFoundError:
        print(
            f"train split not found: {args.train} (run make splits first)",
            file=sys.stderr,
        )
        return 1
    comp_latex: set[str] = {r["latex_expression"] for r in comp}
    n_comp = int(round(args.size * 0.65))
    comp_sample = rng.sample(comp, min(n_comp, len(comp)))
    print(f"competition: {len(comp)} available -> sampled {len(comp_sample)} (65%)")

    # -- verified synthetic -------------------------------------------------
    synth: list[dict] = []
    seen: set[str] = set()
    for path in args.synthetic:
        try:
            rows = load_jsonl(path)
        except FileNotFoundError:
            print(f"  missing synthetic file (skipping): {path}")
            continue
        for r in rows:
            tex = r["latex_expression"]
            if tex in seen or tex in comp_latex:
                continue
            seen.add(tex)
            synth.append(r)
    print(f"synthetic: {len(synth)} oracle-verified rows loaded (latex-deduped)")

    # slice shares of the synthetic portion
    n_synth = args.size - len(comp_sample)
    by_slice: dict[str, list[dict]] = {}
    for r in synth:
        by_slice.setdefault(r.get("metadata", {}).get("slice", "other"), []).append(r)
    picked: list[dict] = []
    for sl, frac in _SLICE_ORDER:
        pool = by_slice.get(sl, [])
        take = min(len(pool), int(round(n_synth * frac)))
        if take:
            picked.extend(rng.sample(pool, take))
    # fill remainder from anything left (best-effort to hit target)
    leftover = [r for r in synth if r not in picked]
    picked.extend(rng.sample(leftover, min(len(leftover), n_synth - len(picked))))
    picked = picked[:n_synth]
    print(
        f"synthetic in mixture: {len(picked)} ({100 * len(picked) / args.size:.1f}%) "
        f"| slices: {dict(Counter(r.get('metadata', {}).get('slice', 'other') for r in picked))}"
    )

    # -- contamination hard check: none of our latex in frozen test/val -----
    frozen: set[str] = set()
    for name in ("test.json", "val.json"):
        path = Path("data/split") / name
        if path.exists():
            rows = json.loads(path.read_text())
            frozen.update(r.get("latex_expression", "") for r in rows)
    hits = [
        r["latex_expression"]
        for r in comp_sample + picked
        if r["latex_expression"] in frozen
    ]
    if hits:
        print(
            f"CONTAMINATION: {len(hits)} rows collide with frozen test/val!",
            file=sys.stderr,
        )
        return 2

    mixture = comp_sample + picked
    rng.shuffle(mixture)

    with open(args.out, "w") as fh:
        for r in mixture:
            fh.write(json.dumps(r, sort_keys=True) + "\n")
    print(f"saved -> {args.out} ({len(mixture)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
