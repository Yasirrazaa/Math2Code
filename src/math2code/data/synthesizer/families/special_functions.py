"""Special-functions family (docs/SYNTHETIC_EXPANSION.md §Tier 3, GATED).

Gated portfolio family: `meta["gated"] = True` so the mixture builder excludes
these rows from the default RL mixture (`--include-gated-slices` opts in).

Constant-output rows (variables=[]) evaluating special functions at fixed
points — the model must parse the special-function notation and emit the
scalar value. Two ground-truth modes:

- **exact (pattern A)**: orthogonal polynomials (legendre / chebyshev T_n /
  hermite) at rational points -> exact Rational/Integer; gamma at integer and
  half-integer args -> factorial / `sqrt(pi)` forms; beta(a, b) rewritten via
  gamma -> exact Rational for every integer pair; zeta(2) -> `pi**2/6`.
- **evalf (pattern D)**: unevaluated calls (erf, besselj, zeta(3), airyai,
  airybi) whose value is `float(expr.evalf())` — solution == truth, both call
  `.evalf()`; mpmath-consistent within the oracle tolerance (1e-5).
"""

from __future__ import annotations

import random
from typing import Any

import sympy as sp

from math2code.data.synthesizer.core import SynthFamily, int_seed, roundtrips
from math2code.schemas import MathCodePair

_X_ORTHO = [
    sp.Integer(-1),
    sp.Rational(-1, 2),
    sp.Integer(0),
    sp.Rational(1, 2),
    sp.Integer(1),
    sp.Integer(2),
]
_X_CH = [sp.Integer(-1), sp.Integer(0), sp.Rational(1, 2), sp.Integer(1)]
_X_HER = [sp.Integer(0), sp.Integer(1), sp.Integer(2)]

# (sympy name, n range, evaluation points) — all results are exact Rational/Integer.
_ORTHO_SPECS = [
    ("legendre", (2, 3, 4, 5, 6), _X_ORTHO),
    ("chebyshevt", (2, 3, 4, 5, 6, 7), _X_CH),
    ("hermite", (2, 3, 4, 5, 6), _X_HER),
]

_GAMMA_HALF = [
    sp.Rational(1, 2),
    sp.Rational(3, 2),
    sp.Rational(5, 2),
    sp.Rational(7, 2),
]

# Unevaluated special-function calls -> pattern D (float(expr.evalf())).
_SPECIAL_D = [
    ("erf", (sp.Integer(1),)),
    ("erf", (sp.Rational(1, 2),)),
    ("besselj", (sp.Integer(0), sp.Integer(1))),
    ("besselj", (sp.Integer(1), sp.Integer(2))),
    ("zeta", (sp.Integer(3),)),
    ("airyai", (sp.Integer(1),)),
    ("airybi", (sp.Integer(0),)),
]

# Evaluates symbolically -> pattern A (exact pi-form).
_SPECIAL_A = [("zeta", (sp.Integer(2),))]


def _evalf_code(expr: sp.Expr) -> str:
    """Pattern D: solution == truth == float(expr.evalf())."""
    return (
        "import sympy as sp\n"
        f"_expr = sp.sympify({sp.sstr(expr)!r})\n"
        "def calculate():\n"
        "    return float(_expr.evalf())\n"
    )


def _kind_of(result: sp.Expr) -> str:
    """coefficient_kind metadata: integer | rational | float."""
    if isinstance(result, sp.Integer):
        return "integer"
    if isinstance(result, sp.Rational):
        return "rational"
    return "float"


class SpecialFunctionsFamily(SynthFamily):
    """Evaluate special functions at fixed points (exact or evalf ground truth)."""

    domain = "Mathematics_General"
    equation_type = "special_functions"

    def _gate(self, result_expr: sp.Expr) -> str:
        return (
            "exact closed form for orthogonal/gamma/beta/zeta(2) (roundtrips, "
            "finite); float(expr.evalf()) for erf/besselj/zeta(3)/airy "
            "(mpmath-consistent, oracle tolerance 1e-5)"
        )

    @staticmethod
    def _one_object(
        rng: random.Random,
    ) -> tuple[str, sp.Expr, sp.Expr] | None:
        """Return (name, problem_expr, result_expr) or None when the gate fails."""
        group = rng.randrange(5)
        try:
            if group == 0:  # orthogonal polynomials at rational points
                name, ns, xs = _ORTHO_SPECS[rng.randrange(len(_ORTHO_SPECS))]
                n = sp.Integer(int(rng.choice(ns)))
                x = xs[rng.randrange(len(xs))]
                problem = sp.Function(name)(n, x)
                result = getattr(sp, name)(n, x)
            elif group == 1:  # gamma at integer / half-integer args
                if rng.random() < 0.5:
                    n = sp.Integer(int(rng.randrange(3, 9)))
                    problem = sp.Function("gamma")(n)
                    result = sp.gamma(n)
                else:
                    h = _GAMMA_HALF[rng.randrange(len(_GAMMA_HALF))]
                    problem = sp.Function("gamma")(h)
                    result = sp.gamma(h)
                name = "gamma"
            elif group == 2:  # beta(a, b) -> rewrite via gamma (exact Rational)
                a = int(rng.randrange(2, 7))
                b = int(rng.randrange(2, 7))
                problem = sp.Function("beta")(sp.Integer(a), sp.Integer(b))
                result = sp.beta(a, b).rewrite(sp.gamma)
                name = "beta"
            elif group == 3:  # symbolically evaluating special function
                name, args = _SPECIAL_A[rng.randrange(len(_SPECIAL_A))]
                problem = sp.Function(name)(*args)
                result = getattr(sp, name)(*args)
            else:  # unevaluated special-function call (pattern D)
                name, args = _SPECIAL_D[rng.randrange(len(_SPECIAL_D))]
                problem = sp.Function(name)(*args)
                result = getattr(sp, name)(*args)
        except Exception:
            return None
        # finiteness + magnitude gate for exact rows (pattern A)
        if not result.has(sp.Function):
            try:
                val = float(sp.N(result))
            except Exception:
                return None
            if not result.is_finite or abs(val) > 1e12 or val == 0.0:
                return None
        return (name, problem, result)

    def generate(
        self, seed: int, prefix: str, count: int, **opts: Any
    ) -> list[MathCodePair]:
        """Deterministically produce `count` special-function objects."""
        rng = random.Random(int_seed(f"special_functions:{seed}"))
        out: list[MathCodePair] = []
        i = 0
        while len(out) < count and i < count * 60:
            i += 1
            obj = self._one_object(rng)
            if obj is None:
                continue
            name, problem, result = obj
            if not result.has(sp.Function):
                # pattern A: exact closed form must roundtrip through sympify
                if not roundtrips(result):
                    continue
                rows = self._build_pair(
                    f"{prefix}_{name}_{i}",
                    problem,
                    result,
                    [],  # constant-output rows
                    int_seed(f"special_functions:{seed}:{i}"),
                    n_variants=1,
                    meta={
                        "slice": "special",
                        "vocab": name,
                        "gated": True,
                        "coefficient_kind": _kind_of(result),
                        "mode": "exact",
                    },
                    equation_type="special_functions",
                )
            else:
                code = _evalf_code(result)
                rows = self._build_pair(
                    f"{prefix}_{name}_{i}",
                    problem,
                    result,
                    [],
                    int_seed(f"special_functions:{seed}:{i}"),
                    n_variants=1,
                    meta={
                        "slice": "special",
                        "vocab": name,
                        "gated": True,
                        "coefficient_kind": "float",
                        "mode": "evalf",
                    },
                    equation_type="special_functions",
                    custom_code=(code, code),
                )
            out.extend(rows)
        return out[:count]
