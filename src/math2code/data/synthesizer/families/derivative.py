"""Programmatic derivative family.

Builds an integrand AST, computes the derivative as ground truth, and asserts
the differentiate-back identity before emitting rows (gate: `diff(F) == f`,
trivially true by construction but enforced for honesty).
"""

from __future__ import annotations

import random
from typing import Any

import sympy as sp

from math2code.data.synthesizer.core import SynthFamily, int_seed
from math2code.schemas import MathCodePair

_X = sp.Symbol("x")


def _rand_ints(rng: random.Random, n: int, lo: int = 1, hi: int = 9) -> list[int]:
    return [int(rng.randint(lo, hi)) for _ in range(n)]


def _random_integrand(rng: random.Random, depth: int) -> sp.Expr:
    """A differentiable, elementary integrand in x with the given operator count."""
    (a, b, c, d) = _rand_ints(rng, 4)
    n = int(rng.choice([2, 3, 4]))
    kinds = [
        lambda: a * _X**n + b * _X + c,
        lambda: a * sp.sin(b * _X) + c * sp.cos(d * _X),
        lambda: a * sp.exp(b * _X) + c,
        lambda: a * sp.log(b * _X + c) + d,
        lambda: a * sp.atan(b * _X) + c,
        lambda: a * sp.sinh(b * _X) + c * sp.cosh(d * _X),
        lambda: a * _X**n * sp.sin(b * _X),
        lambda: sp.sin(a * _X**2),
        lambda: sp.exp(sp.sin(_X)) + a,
        lambda: sp.log(_X**2 + 1) * a,
        lambda: (a * _X + b) ** n + c,
        lambda: a / (b * _X + c) + d,
        lambda: a * _X * sp.exp(b * _X) + c,
    ]
    base = kinds[int(rng.randrange(len(kinds)))]()

    # compose a second operator for depth > 1 (chain-rule practice)
    if depth > 1 and rng.random() < 0.5:
        wraps = [
            lambda: base + a * sp.sin(_X),
            lambda: base * (a * _X + b),
            lambda: base / (a * _X + b),
            lambda: sp.exp(base),
            lambda: sp.sin(base),
        ]
        base = wraps[int(rng.randrange(len(wraps)))]()
    return sp.expand(base)  # canonical form (also exercise expansion)


class DerivativeFamily(SynthFamily):
    """`d/dx f(x)` problems; ground truth `result = diff(f, x)`."""

    domain = "Mathematics_Calculus"
    equation_type = "derivative"

    def _gate(self, result_expr: sp.Expr) -> str:
        return "result == diff(f) (symbolic); oracle numeric cross-check"

    def generate(
        self, seed: int, prefix: str, count: int, **opts: Any
    ) -> list[MathCodePair]:
        rng = random.Random(int_seed(f"derivative:{seed}"))
        out: list[MathCodePair] = []
        i = 0
        while len(out) < count * 2 and i < count * 40:  # cap attempts for gates
            i += 1
            integrand = _random_integrand(rng, depth=int(opts.get("depth", 1)))
            result = sp.diff(integrand, _X)
            if not result.free_symbols:  # constant derivative: trivial, skip
                continue
            # gate: result must be the derivative of the integrand (construction
            # identity; independent execution-level proof happens in the oracle)
            if sp.simplify(result - sp.diff(integrand, _X)) != 0:
                continue
            problem = sp.Derivative(integrand, _X)
            rows = self._build_pair(
                f"{prefix}_{i}",
                problem,
                result,
                [_X],
                int_seed(f"derivative:{seed}:{i}"),
                n_variants=int(opts.get("n_variants", 2)),
                meta={"slice": "calculus"},
            )
            out.extend(rows)
        return out[: count * 2]
