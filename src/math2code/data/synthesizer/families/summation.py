"""Closed-form finite summation/product family (sympy.concrete slice).

The frozen test contains 33 `\\sum` rows (e.g. `-8281 + \\sum_{x=1}^{3} 10^{x}`)
while the synthetic pool previously had zero — this family closes that gap.
It generates finitely-summable closed forms (power sums, geometric sums,
telescoping products, factorials) in two row kinds:

- **concrete** (constant-output rows): the upper bound is a literal integer,
  e.g. `\\sum_{k=1}^{6} k^{2}` -> 91, `\\prod_{k=1}^{5} \\frac{k}{k+1}` -> 1/6.
- **parameterized** (variables=[m]): the upper bound is an integer input
  variable, e.g. `\\sum_{k=1}^{m} 2^{k}` -> `2^{m+1} - 2` — exactly the
  competition's summation surface.

Gate: `Sum/Product(...).doit()` must return a closed form (never an
unevaluated Sum/Product object), the closed form is spot-checked against the
explicit term walk at fresh bound values, and only finite outputs are emitted.

Parameterized rows use ONLY closed forms that evaluate at every integer
`m` — the oracle jitters integer inputs by ±3, which can drift `m` to 0 or
negative. Factorials, `1/(m+1)`-style denominators, and negative bases are
therefore concrete-only (their closed forms crash or branch at `m <= 0`).
"""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any

import sympy as sp

from math2code.data.synthesizer.core import SynthFamily, int_seed
from math2code.schemas import MathCodePair

_K = sp.Symbol("k")  # bound index (plain symbol: sympify round-trip needs it)
_M = sp.Symbol("m")  # parameterized upper bound (input variable)

# (kind, term(k; r), start, max literal n, max parameter m, name)
# max_m == 0 -> concrete-only (see module docstring for why).
_TEMPLATES: list[tuple[str, Callable[[int], sp.Expr], int, int, int, str]] = [
    ("sum", lambda r: _K, 1, 15, 12, "sum_k"),
    ("sum", lambda r: _K**2, 1, 12, 12, "sum_k2"),
    ("sum", lambda r: _K**3, 1, 12, 12, "sum_k3"),
    ("sum", lambda r: 2 * _K - 1, 1, 15, 12, "sum_2k_1"),
    ("sum", lambda r: 2 * _K, 1, 15, 12, "sum_2k"),
    ("sum", lambda r: 2**_K, 1, 12, 12, "sum_2k_pow"),
    ("sum", lambda r: r**_K, 1, 12, 12, "sum_rk"),
    ("sum", lambda r: (-r) ** _K, 1, 12, 0, "sum_neg_rk"),
    ("sum", lambda r: _K * 2**_K, 1, 12, 12, "sum_k2k_pow"),
    ("prod", lambda r: _K, 1, 12, 0, "prod_k"),  # n! — factorial unsafe at m<=0
    ("prod", lambda r: 2 * _K, 1, 10, 0, "prod_2k"),
    ("prod", lambda r: 1 - 1 / _K**2, 2, 12, 0, "prod_1m1k2"),
    ("prod", lambda r: _K / (_K + 1), 1, 12, 0, "prod_k_k1"),
    ("prod", lambda r: 2**_K, 1, 10, 8, "prod_2k_pow"),
]

_RATIOS = (3, 4, 5, 8)


def _explicit(kind: str, term: sp.Expr, start: int, bound: int) -> sp.Expr:
    """Term-walk evaluation of the sum/product — the independent identity."""
    terms = [term.subs({_K: i}) for i in range(start, bound + 1)]
    if not terms:
        return sp.Integer(0) if kind == "sum" else sp.Integer(1)
    return sp.Add(*terms) if kind == "sum" else sp.Mul(*terms)


class SummationFamily(SynthFamily):
    """Finite sums/products with closed forms (concrete + m-parameterized)."""

    domain = "Mathematics_General"
    equation_type = "summation"

    def _gate(self, result_expr: sp.Expr) -> str:
        return "Sum/Product.doit() closed form == explicit term walk (spot-checked)"

    def generate(
        self, seed: int, prefix: str, count: int, **opts: Any
    ) -> list[MathCodePair]:
        rng = random.Random(int_seed(f"summation:{seed}"))
        kind: str = str(opts.get("kind", "mixed"))  # mixed|concrete|parameterized
        n_variants = int(opts.get("n_variants", 1))
        target = count * n_variants
        out: list[MathCodePair] = []
        i = 0
        while len(out) < target and i < count * 60:
            i += 1
            skind, term_fn, start, max_n, max_m, name = _TEMPLATES[
                int(rng.randrange(len(_TEMPLATES)))
            ]
            r = int(rng.choice(_RATIOS))
            term = term_fn(r)

            param = False
            if kind == "parameterized":
                if max_m == 0:
                    continue  # this template has no parameterized (jitter-safe) form
                param = True
            elif kind == "mixed":
                param = max_m > 0 and rng.random() < 0.5

            if param:
                bound = _M
                sum_obj: sp.Expr = (
                    sp.Sum(term, (_K, start, bound))
                    if skind == "sum"
                    else sp.Product(term, (_K, start, bound))
                )
                closed = sum_obj.doit()
                if closed.has(sp.Sum) or closed.has(sp.Product):
                    continue  # gate: unevaluated closed form
                # gate: closed form == explicit walk at fresh integer bounds
                ok = True
                for _ in range(2):
                    mv = int(rng.randint(start, 10))
                    direct = sp.simplify(
                        sum_obj.subs({_M: mv}).doit() - closed.subs({_M: mv})
                    )
                    if direct != 0:
                        ok = False
                        break
                if not ok:
                    continue
                variables = [_M]
                sample_kwargs: dict[str, Any] = {
                    "ints_only": True,
                    "low": 1,
                    "high": max_m,
                }
                meta: dict[str, Any] = {
                    "slice": "summation",
                    "sum_kind": "sum" if skind == "sum" else "product",
                    "param": True,
                }
            else:
                n = int(rng.randint(start, max_n))
                sum_obj = (
                    sp.Sum(term, (_K, start, n))
                    if skind == "sum"
                    else sp.Product(term, (_K, start, n))
                )
                closed = sum_obj.doit()
                if closed.has(sp.Sum) or closed.has(sp.Product):
                    continue
                if closed.is_finite is False:
                    continue
                # gate: closed form == explicit term walk over the literal bound
                if sp.simplify(closed - _explicit(skind, term, start, n)) != 0:
                    continue
                variables = []
                sample_kwargs = {}
                meta = {
                    "slice": "summation",
                    "sum_kind": "sum" if skind == "sum" else "product",
                    "param": False,
                }

            rows = self._build_pair(
                f"{prefix}_{name}_{i}",
                sum_obj,
                closed,
                variables,
                int_seed(f"summation:{seed}:{i}"),
                n_variants=n_variants,
                sample_kwargs=sample_kwargs,
                meta=meta,
            )
            out.extend(rows)
        return out[:target]
