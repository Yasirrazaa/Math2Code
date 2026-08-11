"""Polynomial-invariant family (docs/SYNTHETIC_EXPANSION.md §Tier 2).

Exact scalar invariants of integer-coefficient polynomials, rendered as
named-operator queries via `latex_override` (SymPy cannot render these
surfaces from a single AST):

- Vieta: sum or product of the roots of P(x)
- coefficient of x^k in the expansion of P(x)
- discriminant of P(x)
- resultant of two polynomials P(x), Q(x)

Every ground truth is an exact `sp.Integer`/`sp.Rational` (never float).
Rows whose invariant vanishes (repeated roots, shared roots, zero
coefficient) are rejected: a zero answer carries no training signal and
would teach the model to pattern-match a degenerate case. All rows are
constant-output (no input variables): the model must parse the named
operator and emit the scalar value.

The committed `metadata` carries the problem polynomial as a sympy string
(`poly`, and `poly2` for resultants) so the mixture report and tests can
independently recompute every invariant — a second, symbolic proof layer.
"""

from __future__ import annotations

import random
from typing import Any

import sympy as sp

from math2code.data.synthesizer.core import SynthFamily, int_seed
from math2code.schemas import MathCodePair

_X = sp.Symbol("x")


def vieta_sum(poly: sp.Expr) -> sp.Expr:
    """Sum of roots of poly: -coeff_{x^{n-1}} / leading coefficient (Vieta)."""
    poly_ = sp.Poly(poly, _X)
    n = poly_.degree()
    return -poly_.coeff_monomial(_X ** (n - 1)) / poly_.LC()


def vieta_product(poly: sp.Expr) -> sp.Expr:
    """Product of roots of poly: (-1)^n * constant term / leading coefficient."""
    poly_ = sp.Poly(poly, _X)
    n = poly_.degree()
    return ((-1) ** n) * poly_.coeff_monomial(_X**0) / poly_.LC()


def coeff_of(poly: sp.Expr, k: int) -> sp.Expr:
    """Coefficient of x^k in poly (exact)."""
    return sp.Poly(poly, _X).coeff_monomial(_X**k)


def discriminant_of(poly: sp.Expr) -> sp.Expr:
    """Exact discriminant of poly."""
    return sp.discriminant(poly, _X)


def resultant_of(poly: sp.Expr, poly_q: sp.Expr) -> sp.Expr:
    """Exact resultant of poly and poly_q (zero iff they share a root)."""
    return sp.resultant(poly, poly_q, _X)


def _rand_int(rng: random.Random, lo: int, hi: int) -> int:
    return int(rng.randint(lo, hi))


class PolynomialInvariantsFamily(SynthFamily):
    """Exact scalar invariants of random integer polynomials (constant rows)."""

    domain = "Mathematics_General"
    equation_type = "polynomial"

    def _gate(self, result_expr: sp.Expr) -> str:
        return (
            "invariant recomputed from sp.Poly(poly)/sp.discriminant/"
            "sp.resultant matches exactly; vanishing invariants (0, "
            "non-finite) are rejected"
        )

    def generate(
        self, seed: int, prefix: str, count: int, **opts: Any
    ) -> list[MathCodePair]:
        rng = random.Random(int_seed(f"polynomial_invariants:{seed}"))
        out: list[MathCodePair] = []
        i = 0
        while len(out) < count and i < count * 80:  # cap attempts for gates
            i += 1
            kind = rng.choices(
                ("vieta", "coeff", "disc", "res"), weights=(0.35, 0.25, 0.20, 0.20)
            )[0]
            rows = self._one(kind, rng, prefix, i, seed)
            out.extend(rows)
        return out[:count]

    # -- per-kind builders ------------------------------------------------

    def _one(
        self,
        kind: str,
        rng: random.Random,
        prefix: str,
        i: int,
        base_seed: int,
    ) -> list[MathCodePair]:
        if kind == "vieta":
            return self._vieta(rng, prefix, i, base_seed)
        if kind == "coeff":
            return self._coeff(rng, prefix, i, base_seed)
        if kind == "disc":
            return self._disc(rng, prefix, i, base_seed)
        return self._res(rng, prefix, i, base_seed)

    def _vieta(
        self, rng: random.Random, prefix: str, i: int, base_seed: int
    ) -> list[MathCodePair]:
        degree = int(rng.choice([2, 3]))
        roots = rng.sample([k for k in range(-5, 6) if k != 0], degree)
        poly = sp.expand(sp.prod(_X - r for r in roots))
        ask_sum = rng.random() < 0.5
        value = vieta_sum(poly) if ask_sum else vieta_product(poly)
        if value == 0 or not getattr(value, "is_finite", False):  # degenerate
            return []
        # honesty gate: recompute from the actual roots and compare
        if set(sp.roots(poly).keys()) != set(roots):
            return []
        name = "rootsum" if ask_sum else "rootprod"
        tex = rf"\operatorname{{{name}}}\left({sp.latex(poly)}\right)"
        return self._build_pair(
            f"{prefix}_vieta_{i}",
            poly,  # unused for rendering; latex_override wins
            value,
            [],
            int_seed(f"polynomial_invariants:{base_seed}:{i}"),
            n_variants=1,
            meta={
                "slice": "polynomial",
                "kind": "vieta_sum" if ask_sum else "vieta_prod",
                "poly": sp.sstr(poly),
                "roots": sorted(roots),
                "degree": degree,
                "coefficient_kind": "integer",
            },
            equation_type="polynomial",
            latex_override=[tex],
        )

    def _coeff(
        self, rng: random.Random, prefix: str, i: int, base_seed: int
    ) -> list[MathCodePair]:
        a = _rand_int(rng, 1, 4) * rng.choice([-1, 1])
        n = int(rng.choice([2, 3, 4, 5]))
        k = _rand_int(rng, 1, n - 1)
        poly = sp.expand((_X + a) ** n)
        value = coeff_of(poly, k)
        if value == 0 or not getattr(value, "is_finite", False):
            return []
        tex = sp.latex(poly)
        overrides = [
            rf"\operatorname{{coeff}}_{{x^{{{k}}}}}\left({tex}\right)",
            rf"\left[x^{{{k}}}\right]\left({tex}\right)",
        ]
        return self._build_pair(
            f"{prefix}_coeff_{i}",
            poly,
            value,
            [],
            int_seed(f"polynomial_invariants:{base_seed}:{i}"),
            n_variants=1,  # variant list provided via latex_override
            meta={
                "slice": "polynomial",
                "kind": "coeff",
                "poly": sp.sstr(poly),
                "k": k,
                "degree": n,
                "coefficient_kind": "integer",
            },
            equation_type="polynomial",
            latex_override=overrides,
        )

    def _disc(
        self, rng: random.Random, prefix: str, i: int, base_seed: int
    ) -> list[MathCodePair]:
        if rng.random() < 0.5:
            a, b, c = _rand_int(rng, 1, 4), _rand_int(rng, -5, 5), _rand_int(rng, -5, 5)
            poly = a * _X**2 + b * _X + c
        else:  # cubic with three distinct integer roots -> nonzero discriminant
            roots = rng.sample([k for k in range(-4, 5) if k != 0], 3)
            poly = sp.expand(sp.prod(_X - r for r in roots))
        value = discriminant_of(poly)
        if value == 0 or not getattr(value, "is_finite", False):
            return []
        tex = rf"\operatorname{{disc}}\left({sp.latex(poly)}\right)"
        return self._build_pair(
            f"{prefix}_disc_{i}",
            poly,
            value,
            [],
            int_seed(f"polynomial_invariants:{base_seed}:{i}"),
            n_variants=1,
            meta={
                "slice": "polynomial",
                "kind": "disc",
                "poly": sp.sstr(poly),
                "degree": int(sp.degree(poly, _X)),
                "coefficient_kind": "integer",
            },
            equation_type="polynomial",
            latex_override=[tex],
        )

    def _res(
        self, rng: random.Random, prefix: str, i: int, base_seed: int
    ) -> list[MathCodePair]:
        a, c = _rand_int(rng, 1, 4), _rand_int(rng, 1, 4)
        b, d = _rand_int(rng, -5, 5), _rand_int(rng, -5, 5)
        poly = a * _X + b
        poly_q = c * _X + d
        value = resultant_of(poly, poly_q)
        if value == 0 or not getattr(value, "is_finite", False):
            return []  # shared root: no training signal in a zero resultant
        tex = rf"\operatorname{{res}}\left({sp.latex(poly)},\;{sp.latex(poly_q)}\right)"
        return self._build_pair(
            f"{prefix}_res_{i}",
            poly,
            value,
            [],
            int_seed(f"polynomial_invariants:{base_seed}:{i}"),
            n_variants=1,
            meta={
                "slice": "polynomial",
                "kind": "res",
                "poly": sp.sstr(poly),
                "poly2": sp.sstr(poly_q),
                "degree": 1,
                "coefficient_kind": "integer",
            },
            equation_type="polynomial",
            latex_override=[tex],
        )
