"""Combinatorics families (docs/SYNTHETIC_EXPANSION.md §2 #15).

Exact-integer counting queries: Bell numbers, Catalan numbers, derangements
(subfactorial), Stirling numbers of the second kind, central binomial
coefficients, and the binomial-sum identity (sum_k binom(n, k) = 2^n).

Two row flavors:
- concrete rows: fixed n, output is the exact count as a constant;
- parameterized rows: input variable n, output is the exact count via the
  general operator (bell(n), catalan(n), binomial(n, 2)).

Verification is exact integer equality — the sandbox output must equal the
sympy count recomputed at the same argument; floats never enter the truth.
Int inputs stay ints through the oracle's jitter (`competition._jitter`
preserves int-ness), so the exact-integer guarantee holds end-to-end.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any

import sympy as sp
from sympy.functions.combinatorial.numbers import stirling

from math2code.data.synthesizer.core import SynthFamily, int_seed
from math2code.schemas import MathCodePair

_N = sp.Symbol("n")
_K = sp.Symbol("k", integer=True)

_MAX_VALUE = 10**9  # magnitude guard: counts stay float-safe, no AST blowup

# Builder -> (problem, result, variables, latex | None, vocab, custom_code | None)
_Builder = Callable[
    [random.Random],
    tuple[
        sp.Expr, sp.Expr, list[sp.Symbol], list[str] | None, str, tuple[str, str] | None
    ],
]


def _count_code(fn: str) -> str:
    """Exact-integer code: `int(sp.<fn>(int(n)))`.

    Parameterized rows use the truth_code contract because `_build_pair`
    computes committed expected outputs via `sp.Float` subs, which loses
    exactness for catalan/bell at larger n (e.g. catalan(13.0) ->
    742899.9999999999). Sandbox execution with int inputs stays exact.
    """
    return f"import sympy as sp\ndef calculate(n):\n    return int(sp.{fn}(int(n)))\n"


def _bell_concrete(
    rng: random.Random,
) -> tuple[
    sp.Expr, sp.Expr, list[sp.Symbol], list[str] | None, str, tuple[str, str] | None
]:
    n = int(rng.randint(3, 12))
    return (
        sp.Function("B")(n),
        sp.bell(n),
        [],
        [f"B_{{{n}}}", f"B{{\\left({n} \\right)}}"],
        "bell",
        None,
    )


def _catalan_concrete(
    rng: random.Random,
) -> tuple[
    sp.Expr, sp.Expr, list[sp.Symbol], list[str] | None, str, tuple[str, str] | None
]:
    n = int(rng.randint(2, 12))
    return (
        sp.Function("C")(n),
        sp.catalan(n),
        [],
        [f"C_{{{n}}}", f"C{{\\left({n} \\right)}}"],
        "catalan",
        None,
    )


def _derangement_concrete(
    rng: random.Random,
) -> tuple[
    sp.Expr, sp.Expr, list[sp.Symbol], list[str] | None, str, tuple[str, str] | None
]:
    n = int(rng.randint(2, 10))
    return (
        sp.Function("der")(n),
        sp.subfactorial(n),
        [],
        [f"!{n}", f"\\operatorname{{der}}\\left({n}\\right)"],
        "derangement",
        None,
    )


def _stirling2_concrete(
    rng: random.Random,
) -> tuple[
    sp.Expr, sp.Expr, list[sp.Symbol], list[str] | None, str, tuple[str, str] | None
]:
    n = int(rng.randint(4, 12))
    k = int(rng.randint(2, n - 1))  # 1 < k < n: non-degenerate partition count
    value = stirling(n, k, kind=2)
    return (
        sp.Function("S")(n, k),
        value,
        [],
        [f"S{{\\left({n},{k} \\right)}}", f"{{{n} \\brace {k}}}"],
        "stirling",
        None,
    )


def _central_binomial_concrete(
    rng: random.Random,
) -> tuple[
    sp.Expr, sp.Expr, list[sp.Symbol], list[str] | None, str, tuple[str, str] | None
]:
    n = int(rng.randint(2, 8))
    return (
        sp.Function("binomial")(2 * n, n),
        sp.binomial(2 * n, n),
        [],
        [f"\\binom{{{2 * n}}}{{{n}}}"],
        "binomial",
        None,
    )


def _binomial_sum_concrete(
    rng: random.Random,
) -> tuple[
    sp.Expr, sp.Expr, list[sp.Symbol], list[str] | None, str, tuple[str, str] | None
]:
    n = int(rng.randint(2, 10))
    problem = sp.Sum(sp.binomial(n, _K), (_K, 0, n))
    value = problem.doit()  # 2**n by the binomial theorem
    if not value == 2**n:
        # gate: identity must hold (sympy should always compute it)
        value = sp.Integer(2**n)
    return (
        problem,
        value,
        [],
        None,  # default render: \\sum_{{k=0}}^{{n}} {{\\binom{{n}}{{k}}}}
        "binomial_sum",
        None,
    )


def _bell_param(
    rng: random.Random,
) -> tuple[
    sp.Expr, sp.Expr, list[sp.Symbol], list[str] | None, str, tuple[str, str] | None
]:
    code = _count_code("bell")
    return (
        sp.Function("B")(_N),
        sp.bell(_N),
        [_N],
        [f"B_{{{_N}}}", f"B{{\\left({_N} \\right)}}"],
        "bell",
        (code, code),
    )


def _catalan_param(
    rng: random.Random,
) -> tuple[
    sp.Expr, sp.Expr, list[sp.Symbol], list[str] | None, str, tuple[str, str] | None
]:
    code = _count_code("catalan")
    return (
        sp.Function("C")(_N),
        sp.catalan(_N),
        [_N],
        [f"C_{{{_N}}}", f"C{{\\left({_N} \\right)}}"],
        "catalan",
        (code, code),
    )


def _binomial_param(
    rng: random.Random,
) -> tuple[
    sp.Expr, sp.Expr, list[sp.Symbol], list[str] | None, str, tuple[str, str] | None
]:
    code = (
        "import sympy as sp\n"
        "def calculate(n):\n"
        "    return int(sp.binomial(int(n), 2))\n"
    )
    return (
        sp.Function("binomial")(_N, 2),
        sp.binomial(_N, 2),
        [_N],
        [f"\\binom{{{_N}}}{{2}}", f"{{{_N} \\choose 2}}"],
        "binomial",
        (code, code),
    )


_BUILDERS: list[_Builder] = [
    _bell_concrete,
    _catalan_concrete,
    _derangement_concrete,
    _stirling2_concrete,
    _central_binomial_concrete,
    _binomial_sum_concrete,
    _bell_param,
    _catalan_param,
    _binomial_param,
]


class CombinatoricsFamily(SynthFamily):
    """Exact-integer combinatorics counts (Bell/Catalan/derangements/Stirling-2/binomial)."""

    domain = "Mathematics_General"
    equation_type = "combinatorics"

    def _gate(self, result_expr: sp.Expr) -> str:
        return (
            "exact integer count recomputed with sympy "
            "(bell/catalan/subfactorial/stirling/binomial); concrete rows via "
            "sympify constant, parameterized rows via truth_code executed in the "
            "sandbox (exact integer equality, no float path)"
        )

    def generate(
        self, seed: int, prefix: str, count: int, **opts: Any
    ) -> list[MathCodePair]:
        rng = random.Random(int_seed(f"combinatorics:{seed}"))
        out: list[MathCodePair] = []
        pool = opts.get("pool")  # shared SandboxPool -> fast truth_code evals
        i = 0
        while len(out) < count * 2 and i < count * 120:
            i += 1
            builder = _BUILDERS[int(rng.randrange(len(_BUILDERS)))]
            problem, result, variables, texs, vocab, custom_code = builder(rng)
            if variables:
                # parameterized (truth_code): ints_only sampler bounds n in 4..15
                # so oracle jitter (n +- 3) keeps n >= 1 where bell/catalan are
                # defined (bell(-1) raises); exactness via sandbox int execution
                sample_kwargs: dict[str, Any] | None = {
                    "ints_only": True,
                    "low": 4,
                    "high": 15,
                }
            else:
                # concrete: exact integer must be positive and float-safe
                if not result.is_integer or not result.is_finite:
                    continue
                if not (0 < int(result) <= _MAX_VALUE):
                    continue
                sample_kwargs = None
            rows = self._build_pair(
                f"{prefix}_{vocab}_{i}",
                problem,
                result,
                variables,
                int_seed(f"combinatorics:{seed}:{i}"),
                n_variants=2,
                meta={"vocab": vocab, "slice": "combinatorics"},
                sample_kwargs=sample_kwargs,
                latex_override=texs,
                custom_code=custom_code,
                pool=pool,
            )
            out.extend(rows)
        return out[: count * 2]
