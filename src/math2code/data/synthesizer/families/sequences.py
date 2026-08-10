"""Sequence families (docs/DATA_STRATEGY.md §3 P1; 10% of mix).

The competition train has zero sequence problems. Ground truth is the closed
form; inputs are integer indices (n >= 1), so the oracle exercises the
integer path end-to-end. Families: arithmetic, geometric, triangular,
square/cubic, and polynomial-in-n terms.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any

import sympy as sp

from math2code.data.synthesizer.core import SynthFamily, int_seed
from math2code.schemas import MathCodePair

_N = sp.Symbol("n")


def _intc(rng: random.Random, lo: int = 1, hi: int = 9) -> sp.Integer:
    return sp.Integer(int(rng.randint(lo, hi)))


_TEMPLATES: list[tuple[str, str, Callable[[random.Random], sp.Expr]]] = [
    (
        "arithmetic",
        "sequences",
        lambda r: _intc(r, 1, 6) + (_intc(r, 1, 9) * (_N - 1)),
    ),
    (
        "geometric",
        "sequences",
        lambda r: _intc(r, 1, 4) * _intc(r, 2, 6) ** (_N - 1),
    ),
    (
        "triangular",
        "sequences",
        lambda r: _N * (_N + 1) / 2 + _intc(r, 0, 3),
    ),
    (
        "square_terms",
        "sequences",
        lambda r: _intc(r, 1, 5) * _N**2 + _intc(r, 0, 5),
    ),
    (
        "cubic_terms",
        "sequences",
        lambda r: _N**3 - _intc(r, 0, 4) * _N,
    ),
    (
        "quadratic_terms",
        "sequences",
        lambda r: _intc(r, 1, 4) * _N**2 + _intc(r, 1, 6) * _N + _intc(r, 0, 4),
    ),
]


class SequenceFamily(SynthFamily):
    """Closed-form sequences evaluated at integer indices n >= 1."""

    domain = "Mathematics_General"

    def _gate(self, result_expr: sp.Expr) -> str:
        return "ground-truth AST is the truth (numeric oracle cross-check)"

    def generate(
        self, seed: int, prefix: str, count: int, **opts: Any
    ) -> list[MathCodePair]:
        n_variants = int(opts.get("n_variants", 2))
        rng = random.Random(int_seed(f"seq:{seed}"))
        out: list[MathCodePair] = []
        i = 0
        while len(out) < count * 2 and i < count * 60:
            i += 1
            name, etype, builder = _TEMPLATES[int(rng.randrange(len(_TEMPLATES)))]
            expr = builder(rng)
            rows = self._build_pair(
                f"{prefix}_{name}_{i}",
                expr,
                expr,
                [_N],
                int_seed(f"seq:{seed}:{i}"),
                n_variants=n_variants,
                meta={"vocab": name, "slice": "sequences"},
                sample_kwargs={"ints_only": True, "low": 1, "high": 10},
                equation_type=etype,
            )
            out.extend(rows)
        return out[: count * 2]
