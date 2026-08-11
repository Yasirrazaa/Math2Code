"""Taylor-series coefficient family (docs/SYNTHETIC_EXPANSION.md §Tier 2).

Constant-output rows: "coefficient of x^k in the Taylor expansion of f at 0"
rendered in operator notation and extraction-bracket notation via
`latex_override`. Ground truth is the EXACT Rational/Integer
`sp.series(f, x, 0, order).coeff(x, k)` — never float-drifted.

The query is a named-operator surface (``\\operatorname{{coeff}}_{{x^{{k}}}}`` /
``\\left[x^{{k}}\\right]``) that SymPy cannot render from a single AST, so this
family is the first consumer of the `latex_override` hook.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any

import sympy as sp

from math2code.data.synthesizer.core import SynthFamily, int_seed
from math2code.schemas import MathCodePair

_X = sp.Symbol("x")

# (vocab name, builder) — Taylor expansions around 0 with exact rational
# coefficients for arbitrary order.
_FUNCTIONS: list[tuple[str, Callable[[], sp.Expr]]] = [
    ("sin", lambda: sp.sin(_X)),
    ("cos", lambda: sp.cos(_X)),
    ("exp", lambda: sp.exp(_X)),
    ("sinh", lambda: sp.sinh(_X)),
    ("cosh", lambda: sp.cosh(_X)),
    ("log1p", lambda: sp.log(1 + _X)),
    ("atan", lambda: sp.atan(_X)),
    ("inv1m", lambda: 1 / (1 - _X)),
    ("inv1p", lambda: 1 / (1 + _X)),
]

_KS = (2, 3, 4, 5)  # requested coefficient powers
_POLY_MS = (-3, -2, 2, 3)  # (1+x)^m with small signed integer exponent


def _coeff_overrides(f: sp.Expr, k: int) -> list[str]:
    """Two LaTeX surfaces for the coefficient query (dedupe-safe)."""
    ftex = sp.latex(f)
    return [
        rf"\operatorname{{coeff}}_{{x^{{{k}}}}}\left({ftex}\right)",
        rf"\left[x^{{{k}}}\right]\left({ftex}\right)",
    ]


class SeriesCoefficientFamily(SynthFamily):
    """`coeff_{x^k}` of the Taylor expansion of f at 0 (exact rational)."""

    domain = "Mathematics_Calculus"
    equation_type = "series"

    def _gate(self, result_expr: sp.Expr) -> str:
        return (
            "coefficient == sp.series(f, x, 0, k+3).coeff(x, k) "
            "(exact rational, nonzero, finite)"
        )

    def generate(
        self, seed: int, prefix: str, count: int, **opts: Any
    ) -> list[MathCodePair]:
        rng = random.Random(int_seed(f"series_coeff:{seed}"))
        out: list[MathCodePair] = []
        i = 0
        while len(out) < count * 2 and i < count * 60:  # cap attempts for gates
            i += 1
            if rng.random() < 0.25:
                m = int(rng.choice(_POLY_MS))
                name = f"poly_{m}"
                f = (1 + _X) ** m
            else:
                name, builder = _FUNCTIONS[int(rng.randrange(len(_FUNCTIONS)))]
                f = builder()
            k = int(rng.choice(_KS))
            order = k + 3  # margin past the requested power
            coeff = sp.series(f, _X, 0, order).coeff(_X, k)
            # gate: exact, nonzero, finite — skip zero coefficients (parity
            # functions: even k for sin, odd k for cos) and any unevaluated AST
            if coeff == 0 or not getattr(coeff, "is_finite", False):
                continue
            if coeff.has(sp.O) or coeff.free_symbols:
                continue
            rows = self._build_pair(
                f"{prefix}_{name}_{i}",
                f,  # unused for rendering; latex_override wins
                coeff,
                [],  # constant-output rows (no input variables)
                int_seed(f"series_coeff:{seed}:{i}"),
                n_variants=1,  # variant list is provided via latex_override
                meta={
                    "slice": "series",
                    "vocab": name,
                    "coeff_k": k,
                    "coefficient_kind": "rational",
                },
                equation_type="series",
                latex_override=_coeff_overrides(f, k),
            )
            out.extend(rows)
        return out[: count * 2]