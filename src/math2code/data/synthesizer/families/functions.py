"""P0 function-vocabulary family (docs/DATA_STRATEGY.md §3).

Closes the measured gap: the 22,002-row competition train contains exactly
seven functions (log, tan, cos, sin, sec, csc, cot). This family programmatically
generates the inverse-trig, hyperbolic, abs/sign, floor/ceil, factorial,
binomial, log-base, min/max, modular, radical, exp-composite, and complex-slice
templates — each verified by the oracle and domain-aware sampling.

Every template returns `(expr, sampler_kwargs)` so the sampling domain is
derived from the actual coefficients (e.g. `asin(b*x)` needs `|x| <= 1/b`).
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


def _p(rng: random.Random) -> list[int]:
    """Small int params; avoid the trivial a=1,b=1 identity for trig transforms."""
    a = int(rng.randint(2, 7))
    b = int(rng.randint(1, 6))
    c = int(rng.randint(2, 9))
    return [a, b, c]


def _trig_domain(a: int, b: int, c: int) -> dict[str, Any]:
    """Domain that keeps `|b*x| <= 1` (real principal branch for asin/acos)."""
    del a, c
    return {"low": -0.95 / b, "high": 0.95 / b}


def _mk(
    rng: random.Random,
    fn: Callable[[int, int, int], sp.Expr],
    sopt: Callable[[int, int, int], dict[str, Any]] | None = None,
) -> tuple[sp.Expr, dict[str, Any]]:
    """Typed helper: draw params, build expr, derive the sampling domain."""
    a, b, c = _p(rng)
    return fn(a, b, c), (sopt(a, b, c) if sopt else {})


# Each template: (name, equation_type, builder(rng) -> (expr, sampler_kwargs))
_TEMPLATES: list[
    tuple[str, str, Callable[[random.Random], tuple[sp.Expr, dict[str, Any]]]]
] = [
    (
        "asin",
        "trigonometric",
        lambda r: _mk(r, lambda a, b, c: a * sp.asin(b * _X) + c, _trig_domain),
    ),
    (
        "acos",
        "trigonometric",
        lambda r: _mk(r, lambda a, b, c: a * sp.acos(b * _X) + c, _trig_domain),
    ),
    (
        "atan",
        "trigonometric",
        lambda r: _mk(r, lambda a, b, c: a * sp.atan(b * _X) + c),
    ),
    (
        "atan2",
        "trigonometric",
        lambda r: _mk(r, lambda a, b, c: a * sp.atan2(b * _X, c), _trig_domain),
    ),
    ("sinh", "hyperbolic", lambda r: _mk(r, lambda a, b, c: a * sp.sinh(b * _X) + c)),
    ("cosh", "hyperbolic", lambda r: _mk(r, lambda a, b, c: a * sp.cosh(b * _X) + c)),
    ("tanh", "hyperbolic", lambda r: _mk(r, lambda a, b, c: a * sp.tanh(b * _X) + c)),
    ("abs", "absolute", lambda r: _mk(r, lambda a, b, c: a * sp.Abs(b * _X) + c)),
    ("sign", "absolute", lambda r: _mk(r, lambda a, b, c: a * sp.sign(b * _X) + c)),
    (
        "floor",
        "floor_ceiling",
        lambda r: _mk(r, lambda a, b, c: a * sp.floor(b * _X) + c),
    ),
    (
        "ceiling",
        "floor_ceiling",
        lambda r: _mk(r, lambda a, b, c: a * sp.ceiling(b * _X) + c),
    ),
    (
        "factorial",
        "factorial",
        lambda r: _mk(
            r,
            lambda a, b, c: sp.factorial(_X) + a * _X + b,
            lambda a, b, c: {"ints_only": True, "low": 0, "high": 7},
        ),
    ),
    (
        "binomial",
        "factorial",
        lambda r: _mk(
            r,
            lambda a, b, c: a * sp.binomial(_X, 3) + b,
            lambda a, b, c: {"ints_only": True, "low": 3, "high": 9},
        ),
    ),
    (
        "log_base",
        "logrithmic",
        lambda r: _mk(
            r,
            lambda a, b, c: a * sp.log(_X, b),
            lambda a, b, c: {"low": 0.05, "high": 8},
        ),
    ),
    ("min", "min_max", lambda r: _mk(r, lambda a, b, c: sp.Min(a * _X, c))),
    ("max", "min_max", lambda r: _mk(r, lambda a, b, c: sp.Max(a * _X, c))),
    (
        "min2",
        "min_max",
        lambda r: _mk(r, lambda a, b, c: sp.Min(a * _X + c, b * _Y)),
    ),
    (
        "mod",
        "modular",
        lambda r: _mk(
            r,
            lambda a, b, c: sp.Mod(a * _X + c, b),
            lambda a, b, c: {"ints_only": True, "low": -20, "high": 20},
        ),
    ),
    (
        "cbrt",
        "radical",
        lambda r: _mk(
            r,
            lambda a, b, c: a * sp.root(_X, 3) + c,
            lambda a, b, c: {"low": 0.05, "high": 8},
        ),
    ),
    (
        "sqrt_quad",
        "radical",
        lambda r: _mk(
            r,
            lambda a, b, c: a * sp.sqrt(b * _X + c),
            lambda a, b, c: {"low": 0.05, "high": 8},
        ),
    ),
    (
        "exp_quad",
        "exponential",
        lambda r: _mk(
            r,
            lambda a, b, c: a * sp.exp(b * _X**2) + c,
            lambda a, b, c: {"low": -2, "high": 2},
        ),
    ),
    (
        "exp_lin",
        "exponential",
        lambda r: _mk(r, lambda a, b, c: a * sp.exp(b * _X) + c),
    ),
    # complex slice: real inputs, complex outputs ('re+imj' strings)
    (
        "sqrt_neg",
        "complex",
        lambda r: _mk(
            r,
            lambda a, b, c: a * sp.sqrt(-(b * _X) - c) + 1,
            lambda a, b, c: {"low": -8, "high": -0.1, "allow_complex": True},
        ),
    ),
    (
        "log_neg",
        "complex",
        lambda r: _mk(
            r,
            lambda a, b, c: a * sp.log(-(b * _X)) + c,
            lambda a, b, c: {"low": -8, "high": -0.1, "allow_complex": True},
        ),
    ),
    (
        "asin_ext",
        "complex",
        lambda r: _mk(
            r,
            lambda a, b, c: a * sp.asin(b * _X) + c,
            lambda a, b, c: {"low": -3, "high": 3, "allow_complex": True},
        ),
    ),
]

_GROUPS: dict[str, set[str]] = {
    "inverse_trig": {"asin", "acos", "atan", "atan2"},
    "hyperbolic": {"sinh", "cosh", "tanh"},
    "absolute": {"abs", "sign"},
    "floor_ceiling": {"floor", "ceiling"},
    "factorial": {"factorial", "binomial"},
    "logrithmic": {"log_base"},
    "min_max": {"min", "max", "min2"},
    "modular": {"mod"},
    "radical": {"cbrt", "sqrt_quad"},
    "exponential": {"exp_quad", "exp_lin"},
    "complex": {"sqrt_neg", "log_neg", "asin_ext"},
}


class FunctionVocabFamily(SynthFamily):
    """P0 vocabulary expansion: one row per template (variant rows included)."""

    domain = "Mathematics_General"

    def _gate(self, result_expr: sp.Expr) -> str:
        return "ground-truth AST is the truth (numeric oracle cross-check)"

    def generate(
        self, seed: int, prefix: str, count: int, **opts: Any
    ) -> list[MathCodePair]:
        kind = str(opts.get("kind", "all"))
        n_variants = int(opts.get("n_variants", 2))
        rng = random.Random(int_seed(f"functions:{seed}:{kind}"))
        wanted = _GROUPS.get(kind, set())

        def _match(name: str, etype: str) -> bool:
            return kind == "all" or kind == name or kind == etype or name in wanted

        pool = [t for t in _TEMPLATES if _match(t[0], t[1])]
        out: list[MathCodePair] = []
        i = 0
        while len(out) < count * 2 and i < count * 60 and pool:
            i += 1
            name, etype, builder = pool[int(rng.randrange(len(pool)))]
            expr, sopt = builder(rng)
            variables = [_X, _Y] if expr.free_symbols & {_Y} else [_X]
            rows = self._build_pair(
                f"{prefix}_{name}_{i}",
                expr,  # problem and result are the same expression here
                expr,
                variables,
                int_seed(f"functions:{seed}:{kind}:{i}"),
                n_variants=n_variants,
                meta={"vocab": name},
                sample_kwargs=sopt,
                equation_type=etype,
            )
            out.extend(rows)
        return out[: count * 2]
