"""Differential-family variant: the integration constant is an input.

The frozen test's differential slice evaluates the GENERAL solution of an ODE
at an input pair (x, C1): rows like
`- (p(x)) y(x) + \frac{d}{d x} y(x) = 0` carry `inputs: {x: ..., C1: ...}`.
The model must solve the ODE symbolically and substitute the given constant —
unlike `ODEFamily`, no initial conditions are pinned, so the free constant(s)
stay in the ground truth and become input variables.

Gates (all enforced before a row can be emitted):

1. `checkodesol` — the general solution must satisfy the ODE identically.
2. closed-form gate — the solution RHS must be free of unevaluated
   `Integral` objects (a numeric oracle cannot score those honestly).
3. sympify round-trip + finiteness + pole avoidance (automatic in
   `_build_pair` / sampler).
"""

from __future__ import annotations

import random
from typing import Any

import sympy as sp

from math2code.data.synthesizer.core import SynthFamily, int_seed
from math2code.schemas import MathCodePair

_X = sp.Symbol("x")
_YF = sp.Function("y")(_X)
_C1 = sp.Symbol("C1")
_C2 = sp.Symbol("C2")


def _ints(rng: random.Random, n: int, lo: int = 1, hi: int = 6) -> list[int]:
    return [int(rng.randint(lo, hi)) for _ in range(n)]


class DifferentialC1Family(SynthFamily):
    """General-solution ODEs; ground truth y(x) with C1 (and C2) as inputs."""

    domain = "Mathematics_Calculus"
    equation_type = "differential"

    def _gate(self, result_expr: sp.Expr) -> str:
        return "checkodesol == True; rhs free of unevaluated Integral"

    @staticmethod
    def _make_ode(
        rng: random.Random,
    ) -> tuple[sp.Eq, list[sp.Symbol], dict[str, Any]]:
        """Random ODE -> (ode, input variables, sample_kwargs overrides)."""
        kind = int(rng.randrange(4))
        a, b, c = _ints(rng, 3)
        if kind == 0:  # homogeneous first-order: y' + p(x) y = 0
            pkind = int(rng.randrange(3))
            if pkind == 0:
                p = a * _X + b
            elif pkind == 1:
                p = sp.Rational(a, 1) / _X  # -> y = C1 x^{-a}
            else:
                p = a
            ode = sp.Eq(sp.Derivative(_YF, _X) + p * _YF, 0)
            return ode, [_X, _C1], {"low": -4.0, "high": 4.0}
        if kind == 1:  # non-homogeneous first-order: y' + a y = q(x)
            qkind = int(rng.randrange(4))
            if qkind == 0:
                q: sp.Expr = b
            elif qkind == 1:
                q = b * _X + c
            elif qkind == 2:
                q = b * sp.exp(c * _X)
            else:
                q = b * sp.sin(c * _X)
            ode = sp.Eq(sp.Derivative(_YF, _X) + a * _YF, q)
            sopt = {"low": -2.0, "high": 2.0} if qkind == 2 else {"low": -4.0, "high": 4.0}
            return ode, [_X, _C1], sopt
        if kind == 2:  # constant-coefficient second-order (homogeneous)
            ode = sp.Eq(
                sp.Derivative(_YF, (_X, 2)) + a * sp.Derivative(_YF, _X) + b * _YF,
                0,
            )
            return ode, [_X, _C1, _C2], {"low": -3.0, "high": 3.0}
        # constant-coefficient second-order with polynomial rhs
        q = c * _X + int(_ints(rng, 1)[0])
        ode = sp.Eq(
            sp.Derivative(_YF, (_X, 2)) + a * sp.Derivative(_YF, _X) + b * _YF, q
        )
        return ode, [_X, _C1, _C2], {"low": -3.0, "high": 3.0}

    def generate(
        self, seed: int, prefix: str, count: int, **opts: Any
    ) -> list[MathCodePair]:
        n_variants = int(opts.get("n_variants", 2))
        rng = random.Random(int_seed(f"ode_c1:{seed}"))
        out: list[MathCodePair] = []
        i = 0
        while len(out) < count * n_variants and i < count * 60:
            i += 1
            ode, variables, sopt = self._make_ode(rng)
            try:
                sol = sp.dsolve(ode, _YF)
            except Exception:
                continue
            if not isinstance(sol, sp.Eq):
                continue
            rhs = sp.simplify(sol.rhs)
            if rhs.has(sp.Integral):
                continue  # closed-form gate
            needed = {_C1, _C2} if len(variables) == 3 else {_C1}
            if not needed.issubset(rhs.free_symbols):
                continue  # constant vanished: not a general solution
            try:
                ok, _residual = sp.checkodesol(ode, sol)
            except Exception:
                ok = False
            if not ok:
                continue  # gate: solution must satisfy the ODE
            rows = self._build_pair(
                f"{prefix}_{i}",
                ode,
                rhs,
                variables,
                int_seed(f"ode_c1:{seed}:{i}"),
                n_variants=n_variants,
                meta={
                    "slice": "ode",
                    "vocab": "ode",
                    "n_constants": len(variables) - 1,
                    "ode": sp.sstr(ode),
                },
                sample_kwargs={"pole_margin": 0.3, **sopt},
            )
            out.extend(rows)
        return out[: count * n_variants]
