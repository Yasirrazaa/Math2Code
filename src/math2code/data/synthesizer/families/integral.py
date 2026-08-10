"""Programmatic integration family (indefinite, definite, variable-limit).

Ground truth is the antiderivative (C = 0 convention, matching the competition
contract: outputs are the antiderivative evaluated at the sampled inputs).
Anti-bug gate: `simplify(diff(result) - f) == 0` — the check numeric sampling
alone cannot provide (constant-of-integration errors pass numeric checks on
any finite grid).
"""

from __future__ import annotations

import random
from typing import Any

import sympy as sp

from math2code.data.synthesizer.core import SynthFamily, int_seed
from math2code.schemas import MathCodePair

_X = sp.Symbol("x")


def _random_integrand(rng: random.Random) -> sp.Expr:
    """Elementary integrand whose antiderivative is Piecewise-free and elementary."""
    a, b, c, d = (int(rng.randint(1, 9)) for _ in range(4))
    n = int(rng.choice([2, 3, 4]))
    kinds = [
        lambda: a * _X**n + b * _X + c,
        lambda: a * sp.sin(b * _X) + c,
        lambda: a * sp.cos(b * _X) + c,
        lambda: a * sp.exp(b * _X) + c,
        lambda: a / (b * _X + c) + d,
        lambda: a * _X / (_X**2 + 1) + b,
        lambda: a * _X * sp.exp(b * _X) + c,
        lambda: a * _X * sp.cos(b * _X) + c,
        lambda: sp.sin(_X) ** 2 + a,
        lambda: _X * sp.cos(_X**2) + a,
        lambda: a * sp.exp(-(b * _X)) + c,  # exponential decay slice
    ]
    return sp.expand(kinds[int(rng.randrange(len(kinds)))]())


class IntegralFamily(SynthFamily):
    """`\\int f(x) dx` problems (indefinite / definite / variable-limit)."""

    domain = "Mathematics_Calculus"
    equation_type = "integration"

    def _gate(self, result_expr: sp.Expr) -> str:
        return "simplify(diff(F) - f) == 0 (symbolic)"

    def _emit(
        self,
        prefix: str,
        seed: int,
        i: int,
        problem: sp.Expr,
        result: sp.Expr,
        f: sp.Expr,
        variables: list[sp.Symbol],
        n_variants: int,
        meta: dict[str, Any] | None = None,
    ) -> list[MathCodePair]:
        """Gate + emit one math object (indefinite/variable: differentiate-back;
        definite: numeric result is its own truth)."""
        if result.free_symbols and sp.simplify(sp.diff(result, _X) - f) != 0:
            return []  # anti-bug gate failed — never emit
        return self._build_pair(
            f"{prefix}_{i}",
            problem,
            result,
            variables,
            seed,
            n_variants=n_variants,
            meta=meta,
        )

    def generate(
        self, seed: int, prefix: str, count: int, **opts: Any
    ) -> list[MathCodePair]:
        kind = str(opts.get("kind", "indefinite"))
        n_variants = int(opts.get("n_variants", 2))
        rng = random.Random(int_seed(f"integral:{seed}:{kind}"))
        out: list[MathCodePair] = []
        i = 0
        while len(out) < count * 2 and i < count * 40:
            i += 1
            f = _random_integrand(rng)
            if kind == "definite":
                lo, hi = int(rng.randint(0, 3)), int(rng.randint(4, 8))
                result = sp.integrate(f, (_X, lo, hi))
                if result.free_symbols:
                    continue  # not a clean numeric definite integral
                out.extend(
                    self._emit(
                        f"{prefix}_def",
                        int_seed(f"integral:{seed}:def:{i}"),
                        i,
                        sp.Integral(f, (_X, lo, hi)),
                        result,
                        f,
                        [],
                        n_variants,
                        meta={
                            "kind": "definite",
                            "limits": [lo, hi],
                            "slice": "calculus",
                        },
                    )
                )
            elif kind == "variable":
                t = sp.Symbol("t")
                result = sp.integrate(f.subs(_X, t), (t, 0, _X))
                out.extend(
                    self._emit(
                        f"{prefix}_varlim",
                        int_seed(f"integral:{seed}:var:{i}"),
                        i,
                        sp.Integral(f.subs(_X, t), (t, 0, _X)),
                        result,
                        f,
                        [_X],
                        n_variants,
                        meta={"kind": "variable_limit", "slice": "calculus"},
                    )
                )
            else:  # indefinite
                result = sp.integrate(f, _X)
                if not result.free_symbols:
                    continue  # integrand was a total derivative: skip
                out.extend(
                    self._emit(
                        f"{prefix}_ind",
                        int_seed(f"integral:{seed}:ind:{i}"),
                        i,
                        sp.Integral(f, _X),
                        result,
                        f,
                        [_X],
                        n_variants,
                        meta={"kind": "indefinite", "slice": "calculus"},
                    )
                )
        return out[: count * 2]
