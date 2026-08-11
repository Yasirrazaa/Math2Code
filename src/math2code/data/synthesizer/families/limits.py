"""Programmatic limits family (docs/SYNTHETIC_EXPANSION.md §2, Tier 2).

Generates `\\lim_{x \\to c} f(x)` problems whose ground truth is the *value* of
the limit — a finite constant (pattern B: constant-output rows with no input
variables). The model must parse the limit notation and emit the constant.

Sampled forms are restricted to removable-singularity and rational-ratio
shapes where SymPy resolves the limit to a finite real number; the gate
rejects unevaluated `Limit` objects, infinities, `nan`, and complex results.
Two-sided notation (`dir="+-"`) is used at finite points so the rendered
`\\lim_{x \\to c}` honestly matches the value (every generated shape has equal
left/right limits).
"""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any

import sympy as sp

from math2code.data.synthesizer.core import SynthFamily, int_seed
from math2code.schemas import MathCodePair

_X = sp.Symbol("x")

# Removable singularities at x -> 0 (k is a small integer coefficient).
_ZERO_FORMS: list[Callable[[int], sp.Expr]] = [
    lambda k: sp.sin(_X) / _X,  # -> 1
    lambda k: sp.sin(k * _X) / _X,  # -> k
    lambda k: sp.tan(_X) / _X,  # -> 1
    lambda k: (sp.exp(_X) - 1) / _X,  # -> 1
    lambda k: (sp.exp(k * _X) - 1) / _X,  # -> k
    lambda k: (1 - sp.cos(_X)) / _X**2,  # -> 1/2
    lambda k: (1 - sp.cos(k * _X)) / _X**2,  # -> k^2/2
    lambda k: (sp.sqrt(1 + _X) - 1) / _X,  # -> 1/2
    lambda k: (1 + _X) ** (1 / _X),  # -> e
]

# Removable singularities at x -> a (a != 0, n in 2..4).
_A_FORMS: list[Callable[[int, int], sp.Expr]] = [
    lambda a, n: (_X**n - a**n) / (_X - a),  # -> n a^(n-1)
    lambda a, n: sp.sin(_X - a) / (_X - a),  # -> 1
    lambda a, n: (_X - a) / sp.sin(_X - a),  # -> 1
]


def _sqrt_diff_form(a: int, n: int) -> sp.Expr:
    """(sqrt(x) - sqrt(a)) / (x - a) -> 1/(2 sqrt(a)); requires a > 0."""
    return (sp.sqrt(_X) - sp.sqrt(a)) / (_X - a)


def _int(rng: random.Random, lo: int, hi: int) -> int:
    return int(rng.randint(lo, hi))


class LimitsFamily(SynthFamily):
    """`\\lim_{x \\to c} f(x)` evaluated to a finite constant."""

    domain = "Mathematics_Calculus"
    equation_type = "limit"

    def _gate(self, result_expr: sp.Expr) -> str:
        return (
            "sp.limit(f, x, c) is a finite real constant "
            "(unevaluated Limit / oo / nan / complex rejected)"
        )

    @staticmethod
    def _valid(result: sp.Expr) -> bool:
        """Gate: finite real constant, numerically evaluable, bounded."""
        if result.has(sp.Limit) or result.has(sp.I) or not result.is_finite:
            return False
        try:
            val = float(sp.N(result))
        except Exception:
            return False
        return abs(val) < 1e12

    @staticmethod
    def _point_str(c: sp.Expr) -> str:
        if c == sp.oo:
            return "oo"
        if c == -sp.oo:
            return "-oo"
        return str(c)

    def _one_object(
        self, rng: random.Random
    ) -> tuple[sp.Expr, sp.Expr, sp.Expr] | None:
        """Return (point, function, limit value) or None when the gate fails."""
        group = _int(rng, 0, 2)
        try:
            if group == 0:  # removable singularities at 0
                c: sp.Expr = sp.Integer(0)
                f = _ZERO_FORMS[_int(rng, 0, len(_ZERO_FORMS) - 1)](_int(rng, 2, 5))
            elif group == 1:  # removable singularities at a != 0
                a = _int(rng, 1, 4) * (1 if _int(rng, 0, 1) == 0 else -1)
                n = _int(rng, 2, 4)
                if a > 0:
                    builders = _A_FORMS + [_sqrt_diff_form]
                else:  # sqrt form is undefined for negative a
                    builders = _A_FORMS
                f = builders[_int(rng, 0, len(builders) - 1)](a, n)
                c = sp.Integer(a)
            else:  # rational ratios at +/- infinity, (1 + k/x)^x -> e^k
                c, f = self._inf_form(rng)
            # two-sided at finite points (honest notation: every form has equal
            # left/right limits); plain limit at infinity (dir='+-' is invalid there)
            if c in (sp.oo, -sp.oo):
                result = sp.limit(f, _X, c)
            else:
                result = sp.limit(f, _X, c, dir="+-")
        except Exception:
            return None
        if not self._valid(result):
            return None
        return (c, f, result)

    @staticmethod
    def _inf_form(rng: random.Random) -> tuple[sp.Expr, sp.Expr]:
        """(point, function) pairs with finite limits at +/- infinity."""
        kind = _int(rng, 0, 2)
        if kind == 0:  # quadratic / quadratic -> a/d (a, d nonzero)
            a, d = _int(rng, 1, 7), _int(rng, 1, 7)
            b = _int(rng, -9, 9)
            c = _int(rng, -9, 9)
            e = _int(rng, -9, 9)
            f = _int(rng, -9, 9)
            return sp.oo, (a * _X**2 + b * _X + c) / (d * _X**2 + e * _X + f)
        if kind == 1:  # linear / linear -> a/c at oo or -oo
            a, c = _int(rng, 1, 7), _int(rng, 1, 7)
            b, d = _int(rng, -9, 9), _int(rng, -9, 9)
            point: sp.Expr = sp.oo if _int(rng, 0, 1) == 0 else -sp.oo
            return point, (a * _X + b) / (c * _X + d)
        k = _int(rng, 2, 5)
        return sp.oo, (1 + k / _X) ** _X  # -> e^k

    def generate(
        self, seed: int, prefix: str, count: int, **opts: Any
    ) -> list[MathCodePair]:
        """Deterministically produce `count` limit objects (constant rows)."""
        rng = random.Random(int_seed(f"limits:{seed}"))
        n_variants = int(opts.get("n_variants", 1))
        out: list[MathCodePair] = []
        i = 0
        while len(out) < count * n_variants and i < count * 60:
            i += 1
            obj = self._one_object(rng)
            if obj is None:
                continue
            c, f, result = obj
            # two-sided notation at finite points matches the (equal-sided) value
            if c in (sp.oo, -sp.oo):
                problem = sp.Limit(f, _X, c)
            else:
                problem = sp.Limit(f, _X, c, dir="+-")
            rows = self._build_pair(
                f"{prefix}_{i}",
                problem,
                result,
                [],  # constant-output rows: the value IS the answer
                int_seed(f"limits:{seed}:{i}"),
                n_variants=n_variants,
                meta={
                    "slice": "limits",
                    "point": self._point_str(c),
                    "vocab": "limit",
                },
            )
            out.extend(rows)
        return out[: count * n_variants]
