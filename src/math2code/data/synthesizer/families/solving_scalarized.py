"""Scalarized solving family (docs/SYNTHETIC_EXPANSION.md Tier 3, gated).

Constant-output rows: the query is an equation (or system) whose answer is a
single scalar — the unique root of a linear equation `a x + b = 0`, the
double root of a quadratic `(x - r)^2 = 0`, or the x-coordinate of a 2x2
linear system. All answers are exact Rationals/Integers; each emitted root is
proven at build time by substitution (residual == 0) and by `sp.solve`
agreement.

GATED: rows carry `metadata["gated"] = True` and are excluded from the
default RL mixture (opt-in via `--include-gated-slices`). The query surface
is the equation itself (competition style) — no natural-language words.
"""

from __future__ import annotations

import random
from typing import Any

import sympy as sp

from math2code.data.synthesizer.core import SynthFamily, int_seed
from math2code.schemas import MathCodePair

_X = sp.Symbol("x")
_Y = sp.Symbol("y")

_LINEAR_AS = [v for v in range(-6, 7) if v != 0]  # a in ±1..±6
_LINEAR_BS = [v for v in range(-12, 13) if v != 0]  # b in ±1..±12 (skip x=0)
_ROOTS = [v for v in range(-5, 6) if v != 0]  # double roots / system solutions


def _lhs(coeffs: list[int], vars_: list[str]) -> str:
    """Render `a1 v1 + a2 v2 + ...` with competition-style sign spacing."""
    parts: list[str] = []
    for c, v in zip(coeffs, vars_):
        if c == 0:
            continue
        if not v:  # constant term
            term = str(c)
        elif c == 1:
            term = v
        elif c == -1:
            term = f"-{v}"
        else:
            term = f"{c} {v}"
        if not parts:
            parts.append(term)
        elif term.startswith("-"):
            parts.append("- " + term[1:])
        else:
            parts.append("+ " + term)
    return " ".join(parts)


class SolvingScalarizedFamily(SynthFamily):
    """Solve-for-x queries whose answer is one exact scalar (gated)."""

    domain = "Mathematics_General"
    equation_type = "algebraic"

    def _gate(self, result_expr: sp.Expr) -> str:
        return (
            "substitution residual == 0: plugging the emitted root into the "
            "equation(s) gives exact zero; sp.solve agrees"
        )

    def _linear(
        self, rng: random.Random, task_id: str, i: int, seed: int
    ) -> list[MathCodePair]:
        a = int(rng.choice(_LINEAR_AS))
        b = int(rng.choice(_LINEAR_BS))
        root = sp.Rational(-b, a)
        eq = sp.Eq(a * _X + b, 0)
        sol = sp.solve(eq, _X)
        # gate: sp.solve agrees with the constructed root, and the root
        # annihilates the equation exactly
        if not sol or sp.simplify(sol[0] - root) != 0:
            return []
        if sp.simplify(eq.lhs.subs({_X: root}) - eq.rhs) != 0:
            return []
        lhs = _lhs([a, b], ["x", ""])
        return self._build_pair(
            task_id,
            eq,
            root,
            [],  # constant-output rows (no input variables)
            int_seed(f"solving_scalarized:{seed}:{i}"),
            n_variants=1,  # variant list is provided via latex_override
            meta={
                "slice": "algebraic",
                "kind": "linear",
                "gated": True,
                "eqs": [[sp.sstr(eq.lhs), sp.sstr(eq.rhs)]],
                "solution": {"x": sp.sstr(root)},
            },
            equation_type="algebraic",
            latex_override=[f"{lhs} = 0", rf"\left({lhs} = 0\right)"],
        )

    def _quadratic(
        self, rng: random.Random, task_id: str, i: int, seed: int
    ) -> list[MathCodePair]:
        r = int(rng.choice(_ROOTS))
        eq = sp.Eq((_X - r) ** 2, 0)  # discriminant zero -> unique double root
        sol = sp.solve(eq, _X)
        if not sol or sp.simplify(sol[0] - r) != 0:
            return []
        if sp.simplify(eq.lhs.subs({_X: r}) - eq.rhs) != 0:
            return []
        expanded = sp.expand((_X - r) ** 2)
        return self._build_pair(
            task_id,
            eq,
            sp.Integer(r),
            [],
            int_seed(f"solving_scalarized:{seed}:{i}"),
            n_variants=1,
            meta={
                "slice": "algebraic",
                "kind": "quadratic",
                "gated": True,
                "eqs": [
                    [sp.sstr((_X - r) ** 2), "0"],
                    [sp.sstr(expanded), "0"],
                ],
                "solution": {"x": sp.sstr(r)},
            },
            equation_type="algebraic",
            latex_override=[
                sp.latex(eq),  # factored: (x - 3)^{2} = 0
                sp.latex(sp.Eq(expanded, 0)),  # expanded: x^{2} - 6 x + 9 = 0
            ],
        )

    def _system(
        self, rng: random.Random, task_id: str, i: int, seed: int
    ) -> list[MathCodePair]:
        x0 = int(rng.choice(_ROOTS))
        y0 = int(rng.choice(_ROOTS))
        a, b, c, d = (int(rng.randint(-4, 4)) for _ in range(4))
        # gate: both equations genuinely involve a variable and the
        # coefficient matrix is non-singular (unique solution)
        if (a, b) == (0, 0) or (c, d) == (0, 0) or a * d - b * c == 0:
            return []
        e = a * x0 + b * y0  # rhs built from the chosen solution
        f = c * x0 + d * y0
        eq1 = sp.Eq(a * _X + b * _Y, e)
        eq2 = sp.Eq(c * _X + d * _Y, f)
        sol = sp.solve([eq1, eq2], [_X, _Y], dict=True)
        if (
            not sol
            or sp.simplify(sol[0][_X] - x0) != 0
            or sp.simplify(sol[0][_Y] - y0) != 0
        ):
            return []
        e1, e2 = sp.latex(eq1), sp.latex(eq2)
        return self._build_pair(
            task_id,
            eq1,
            sp.Integer(x0),  # the x-coordinate is the scalar answer
            [],
            int_seed(f"solving_scalarized:{seed}:{i}"),
            n_variants=1,
            meta={
                "slice": "algebraic",
                "kind": "system",
                "gated": True,
                "eqs": [
                    [sp.sstr(eq1.lhs), sp.sstr(eq1.rhs)],
                    [sp.sstr(eq2.lhs), sp.sstr(eq2.rhs)],
                ],
                "solution": {"x": sp.sstr(x0), "y": sp.sstr(y0)},
            },
            equation_type="algebraic",
            latex_override=[
                f"\\begin{{cases}}{e1} \\\\ {e2}\\end{{cases}}",
                f"{e1} \\; \\wedge \\; {e2}",
            ],
        )

    def generate(
        self, seed: int, prefix: str, count: int, **opts: Any
    ) -> list[MathCodePair]:
        rng = random.Random(int_seed(f"solving_scalarized:{seed}"))
        out: list[MathCodePair] = []
        i = 0
        while len(out) < count * 2 and i < count * 60:  # cap attempts for gates
            i += 1
            kind = rng.random()
            if kind < 0.4:
                rows = self._linear(rng, f"{prefix}_lin_{i}", i, seed)
            elif kind < 0.7:
                rows = self._quadratic(rng, f"{prefix}_quad_{i}", i, seed)
            else:
                rows = self._system(rng, f"{prefix}_sys_{i}", i, seed)
            out.extend(rows)
        return out[: count * 2]
