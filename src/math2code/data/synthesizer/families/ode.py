"""Programmatic ODE family (docs/DATA_STRATEGY.md §3 P1).

Competition differential rows are first-order linear `y' + p(x)y = 0` with a
narrow p(x) set. This family adds: non-homogeneous first-order, separable,
and constant-coefficient second-order ODEs — always WITH initial conditions so
the solution has no arbitrary constants (numeric evaluation is then well
defined, matching the competition contract of evaluating y(x) at inputs).

Anti-bug gates (both enforced before a row can be emitted):
1. `checkodesol` — substitute the solution back into the ODE; residual must be 0.
2. IC check — the solution must reproduce the initial condition(s) numerically.
"""

from __future__ import annotations

import random
from typing import Any

import sympy as sp

from math2code.data.synthesizer.core import SynthFamily, int_seed
from math2code.schemas import MathCodePair

_X = sp.Symbol("x")
_YF = sp.Function("y")(_X)


def _p(rng: random.Random) -> list[int]:
    return [int(rng.randint(1, 6)) for _ in range(3)]


class ODEFamily(SynthFamily):
    """ODE problems; ground truth is the (IC-pinned) solution y(x)."""

    domain = "Mathematics_Calculus"
    equation_type = "differential"

    def _gate(self, result_expr: sp.Expr) -> str:
        return "checkodesol == True and IC residual == 0"

    @staticmethod
    def _make_ode(rng: random.Random) -> tuple[sp.Eq, sp.Expr, sp.Expr]:
        """Return (ode, ic_value_y0, ic_value_deriv1_or_None)."""
        kind = int(rng.randrange(4))
        a, b, c = _p(rng)
        if kind == 0:  # homogeneous first-order: y' + p(x) y = 0
            p = sp.Rational(a) * _X + b
            ode = sp.Eq(sp.Derivative(_YF, _X) + p * _YF, 0)
            y0 = sp.Integer(int(rng.randint(1, 5)))
            return ode, y0, None
        if kind == 1:  # non-homogeneous first-order: y' + a y = q(x)
            q = sp.Rational(b) * _X + c
            ode = sp.Eq(sp.Derivative(_YF, _X) + a * _YF, q)
            y0 = sp.Integer(int(rng.randint(1, 5)))
            return ode, y0, None
        if kind == 2:  # separable: y' = k y (exponential decay family)
            k = -sp.Rational(a)  # negative -> decay; positive -> growth
            ode = sp.Eq(sp.Derivative(_YF, _X), k * _YF)
            y0 = sp.Integer(int(rng.randint(1, 5)))
            return ode, y0, None
        # constant-coefficient second-order: y'' + a y' + b y = 0
        ode = sp.Eq(
            sp.Derivative(_YF, (_X, 2)) + a * sp.Derivative(_YF, _X) + b * _YF, 0
        )
        y0 = sp.Integer(int(rng.randint(1, 5)))
        y1 = sp.Integer(int(rng.randint(0, 3)))
        return ode, y0, y1

    @staticmethod
    def _verify_ode(ode: sp.Eq, sol: sp.Eq, y0: sp.Expr, y1: sp.Expr | None) -> bool:
        """Gate: residual substitution (checkodesol) + IC reproduction."""
        try:
            ok, residual = sp.checkodesol(ode, sol)
        except Exception:
            return False
        if not ok:
            return False
        try:
            if abs(float(sp.N(sp.Abs(sol.rhs.subs(_X, 0) - y0)))) > 1e-6:
                return False
            if y1 is not None:
                d = sp.diff(sol.rhs, _X).subs(_X, 0)
                if abs(float(sp.N(sp.Abs(d - y1)))) > 1e-6:
                    return False
        except Exception:
            return False
        return True

    def generate(
        self, seed: int, prefix: str, count: int, **opts: Any
    ) -> list[MathCodePair]:
        n_variants = int(opts.get("n_variants", 2))
        rng = random.Random(int_seed(f"ode:{seed}"))
        out: list[MathCodePair] = []
        i = 0
        while len(out) < count * 2 and i < count * 60:
            i += 1
            ode, y0, y1 = self._make_ode(rng)
            try:
                ics = {_YF.subs(_X, 0): y0}
                if y1 is not None:
                    ics[sp.Derivative(_YF, _X).subs(_X, 0)] = y1
                sol = sp.dsolve(ode, _YF, ics=ics)
            except Exception:
                continue
            if not isinstance(sol, sp.Eq):
                continue  # dsolve returned a list / unevaluated form
            rhs = sp.simplify(sol.rhs)
            if not rhs.free_symbols or not self._verify_ode(ode, sol, y0, y1):
                continue  # gates failed
            rows = self._build_pair(
                f"{prefix}_{i}",
                ode,  # the problem is the ODE equation itself
                rhs,
                [_X],
                int_seed(f"ode:{seed}:{i}"),
                n_variants=n_variants,
                meta={"ic_y0": str(y0), "ic_y1": str(y1), "slice": "ode"},
                sample_kwargs={"low": -3.0, "high": 3.0, "pole_margin": 0.3},
            )
            out.extend(rows)
        return out[: count * 2]
