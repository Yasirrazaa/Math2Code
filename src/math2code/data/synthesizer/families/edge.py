"""Edge-case / robustness family (docs/DATA_STRATEGY.md §3; 5% of mix).

Exercises the axes the competition train under-samples: extreme input
magnitudes (log-scale 1e-8..1e12, exercising the sampler's numeric-stability
guard), singularity-adjacent sampling (small pole_margin), domain-constrained
logs/radicals, deep nested compositions, and a notation trap (`sqrt(x**2)` —
its truth is `Abs(x)`, not `x`).

Every template remains oracle-verifiable: the ground truth is the AST; the
sampler's finiteness/realness gates reject the handful of inputs where the
truth is undefined.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any

import sympy as sp

from math2code.data.synthesizer.core import SynthFamily, int_seed
from math2code.schemas import MathCodePair

_X = sp.Symbol("x")


def _dec(rng: random.Random, lo: float = 0.5, hi: float = 4.0) -> sp.Float:
    return sp.Float(round(rng.uniform(lo, hi), 2))


def _intc(rng: random.Random, lo: int = 1, hi: int = 6) -> sp.Integer:
    return sp.Integer(int(rng.randint(lo, hi)))


_TEMPLATES: list[
    tuple[str, str, Callable[[random.Random], sp.Expr], dict[str, Any]]
] = [
    # extreme magnitudes: outputs span 1e-24 .. 1e36 before the stability guard
    (
        "logscale_poly",
        "polynomial",
        lambda r: _dec(r) * _X**3 + _dec(r) * _X**2 + _dec(r) * _X,
        {"log_scale": True, "low": -8, "high": 12},
    ),
    (
        "logscale_ratio",
        "rational",
        lambda r: _dec(r) * _X**3 / (_X**2 + _intc(r, 1, 4)),
        {"log_scale": True, "low": -6, "high": 10},
    ),
    # singularity-adjacent: x can land within 0.005 of the pole
    (
        "near_pole",
        "rational",
        lambda r: _dec(r) / (_X - sp.Float(0.99)),
        {"low": -2, "high": 2, "pole_margin": 0.005},
    ),
    # domain-constrained: realness gate keeps x > 0
    (
        "log_over_sqrt",
        "logrithmic",
        lambda r: _dec(r) * sp.log(_X) / sp.sqrt(_X) + _dec(r),
        {"low": 0.01, "high": 8},
    ),
    (
        "nested_log_sin",
        "trigonometric",
        lambda r: _dec(r) * sp.log(sp.sin(_X) + 1) + _dec(r),
        {"low": -3, "high": 3},
    ),
    # notation trap: sqrt(x**2) == Abs(x), not x
    (
        "sqrt_sq_trap",
        "radical",
        lambda r: sp.sqrt(_X**2) + _intc(r, 1, 5),
        {"low": -5, "high": 5},
    ),
    # deep nested composition
    (
        "deep_trig",
        "trigonometric",
        lambda r: sp.sin(sp.cos(_X)) + _dec(r) * sp.cos(sp.sin(_X)),
        {"low": -3, "high": 3},
    ),
    # sharp exponential decay (large rate)
    (
        "sharp_decay",
        "exponential",
        lambda r: _dec(r) * sp.exp(-_intc(r, 20, 60) * _X) + _dec(r),
        {"low": 0.0, "high": 1.5},
    ),
]


class EdgeCaseFamily(SynthFamily):
    """Robustness rows: magnitudes, near-singularity, domains, traps."""

    domain = "Mathematics_General"

    def _gate(self, result_expr: sp.Expr) -> str:
        return "ground-truth AST is the truth (numeric oracle cross-check)"

    def generate(
        self, seed: int, prefix: str, count: int, **opts: Any
    ) -> list[MathCodePair]:
        n_variants = int(opts.get("n_variants", 2))
        rng = random.Random(int_seed(f"edge:{seed}"))
        out: list[MathCodePair] = []
        i = 0
        while len(out) < count * 2 and i < count * 60:
            i += 1
            name, etype, builder, sopt = _TEMPLATES[int(rng.randrange(len(_TEMPLATES)))]
            expr = builder(rng)
            rows = self._build_pair(
                f"{prefix}_{name}_{i}",
                expr,
                expr,
                [_X],
                int_seed(f"edge:{seed}:{i}"),
                n_variants=n_variants,
                meta={"vocab": name, "slice": "edge"},
                sample_kwargs=sopt,
                equation_type=etype,
            )
            out.extend(rows)
        return out[: count * 2]
