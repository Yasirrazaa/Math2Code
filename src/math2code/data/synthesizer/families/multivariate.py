"""Multivariate coupling family (docs/DATA_STRATEGY.md §3 P1, 15% of mix).

Measured gap: 1,806 multivariable train rows have **0 cross-terms** (pure
sums of single-variable terms), and decimal coefficients appear in only 16.9%
of rows. This family generates cross-term-rich expressions in 2-3 variables —
including the public-probe `augmented_equation` shape (rational with an
exponential factor over a polynomial denominator, decimal coefficients,
`\\cdot` multiplication).

Sampling is domain-aware per template: `log(x*y + b)` needs `x*y + b > 0`,
differences of squares sample away from the diagonal (cancellation-heavy
points are numerically unstable), rationals rely on the finiteness gate.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any

import sympy as sp

from math2code.data.synthesizer.core import SynthFamily, int_seed
from math2code.schemas import MathCodePair

_X = sp.Symbol("x")
_Y = sp.Symbol("y")
_Z = sp.Symbol("z")


def _dec(rng: random.Random, lo: float = 1.0, hi: float = 9.0) -> sp.Float:
    """Small decimal coefficient (competition-style, e.g. 4.54832068863631)."""
    return sp.Float(round(rng.uniform(lo, hi), 3) * rng.choice([-1.0, 1.0]))


def _intc(rng: random.Random, lo: int = 1, hi: int = 7) -> sp.Integer:
    return sp.Integer(int(rng.randint(lo, hi)))


def _augmented(rng: random.Random) -> sp.Expr:
    """Public-probe shape: P(x,y)*exp(k*x)/Q(x,y) with decimal coefficients."""
    a, k, m = _dec(rng), _dec(rng), _intc(rng, 1, 4)
    p = (
        _dec(rng) * _X ** _intc(rng, 1, 4)
        + _dec(rng) * _X ** _intc(rng, 1, 3)
        + _dec(rng) * _Y ** _intc(rng, 1, 3)
        + _dec(rng)
    )
    q = (
        _dec(rng) * _Y ** _intc(rng, 1, 3)
        + _dec(rng) * _X ** _intc(rng, 1, 2)
        + _dec(rng)
    )
    return a * p * sp.exp(k * _X) / (q + m)


# (name, equation_type, builder(rng) -> expr, sampler_kwargs)
_TEMPLATES: list[
    tuple[str, str, Callable[[random.Random], sp.Expr], dict[str, Any]]
] = [
    # quadratic forms with cross terms; coefficients alternate int/decimal
    (
        "quad_xterm",
        "polynomial_multivariate",
        lambda r: (
            _dec(r, 1, 5) * _X**2
            + _dec(r, 1, 5) * _Y**2
            + _dec(r, 1, 9) * _X * _Y
            + _dec(r, 1, 9) * _X
            + _intc(r, 1, 5) * _Y
            + _intc(r, 1, 9)
        ),
        {"low": -3, "high": 3},
    ),
    (
        "diff_squares",
        "polynomial_multivariate",
        lambda r: _intc(r) * _X**2 - _intc(r) * _Y**2 + _intc(r, 1, 9),
        {"low": -4, "high": 4},  # numeric-stability guard drops the diagonal
    ),
    (
        "trig_cross",
        "trigonometric_multivariate",
        lambda r: (
            _dec(r) * _X * sp.sin(_Y) + _dec(r) * _Y * sp.cos(_X) + _intc(r, 1, 9)
        ),
        {},
    ),
    (
        "angle_diff",
        "trigonometric_multivariate",
        lambda r: sp.sin(_X) * sp.cos(_Y) - sp.cos(_X) * sp.sin(_Y) + _intc(r, 1, 5),
        {"low": -3, "high": 3},
    ),
    (
        "exp_cross",
        "exponential_multivariate",
        lambda r: _dec(r) * sp.exp(_dec(r) * _X - _dec(r) * _Y) + _dec(r) * _X * _Y,
        {"low": -2, "high": 2},
    ),
    (
        "log_prod",
        "logrithmic_multivariate",
        lambda r: _dec(r) * sp.log(_X * _Y + _intc(r, 1, 5)) + _dec(r),
        {"low": -3, "high": 3},
    ),
    (
        "augmented",
        "rational_multivariate",
        _augmented,
        {"low": -3, "high": 3, "pole_margin": 0.5},
    ),
    (
        "monomial_couple",
        "polynomial_multivariate",
        lambda r: (
            _dec(r) * _X ** _intc(r, 1, 3) * _Y ** _intc(r, 1, 3) + _intc(r, 1, 9)
        ),
        {"low": -3, "high": 3},
    ),
    (
        "triple",
        "polynomial_multivariate",
        lambda r: _dec(r) * _X * _Y * _Z + _dec(r) * sp.sin(_X) + _intc(r, 1, 9) * _Y,
        {"low": -3, "high": 3},
    ),
]


class MultivariateFamily(SynthFamily):
    """Cross-term multivariate rows; decimal and integer coefficient kinds."""

    domain = "Mathematics_General"

    def _gate(self, result_expr: sp.Expr) -> str:
        return "ground-truth AST is the truth (numeric oracle cross-check)"

    def generate(
        self, seed: int, prefix: str, count: int, **opts: Any
    ) -> list[MathCodePair]:
        n_variants = int(opts.get("n_variants", 2))
        rng = random.Random(int_seed(f"multivar:{seed}"))
        out: list[MathCodePair] = []
        i = 0
        while len(out) < count * 2 and i < count * 60:
            i += 1
            name, etype, builder, sopt = _TEMPLATES[int(rng.randrange(len(_TEMPLATES)))]
            expr = builder(rng)
            variables = sorted(expr.free_symbols, key=str)
            if not variables:
                continue
            n_vars = len(variables)
            cross = len({v for v in expr.free_symbols}) >= 2
            coeff_kind = (
                "decimal"
                if any(isinstance(a, sp.Float) for a in expr.atoms(sp.Float))
                else "integer"
            )
            rows = self._build_pair(
                f"{prefix}_{name}_{i}",
                expr,
                expr,
                variables,
                int_seed(f"multivar:{seed}:{i}"),
                n_variants=n_variants,
                meta={
                    "vocab": name,
                    "slice": "multivariate",
                    "n_vars": n_vars,
                    "cross_terms": bool(cross),
                    "coefficient_kind": coeff_kind,
                },
                sample_kwargs=sopt,
                equation_type=etype,
            )
            out.extend(rows)
        return out[: count * 2]
