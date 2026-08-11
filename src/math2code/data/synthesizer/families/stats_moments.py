"""Probability moments family (docs/SYNTHETIC_EXPANSION.md §Tier 3 gated).

Constant-output rows: E[X], Var[X], E[X^2] of a fixed distribution with
concrete parameters — exact closed forms (Rational/Integer), never
float-drifted. The distribution context is part of the query surface
(`X ~ Unif(a,b)` etc.), rendered via `latex_override` since SymPy cannot
produce `\mathbb{E}` notation from a single AST.

Epistemology (the family's `_gate`): every row's closed form is cross-checked
at build time against an INDEPENDENT computation — `sympy.stats.E`,
`variance`, or `E(X**2)` on the concrete distribution object; any mismatch
(or unevaluated/exception) skips the row. Gated rows are excluded from the
default RL mixture by the builder (`metadata["gated"] = True`).
"""

from __future__ import annotations

import random
from typing import Any

import sympy as sp

from math2code.data.synthesizer.core import SynthFamily, int_seed
from math2code.schemas import MathCodePair

_DISTS = ("uniform", "normal", "exp", "binomial", "poisson", "beta")
_MOMENTS = ("E", "Var", "E2")

_NORMAL_SIGMAS: tuple[sp.Expr, ...] = (
    sp.Rational(1, 2),
    sp.Integer(1),
    sp.Rational(3, 2),
    sp.Integer(2),
    sp.Integer(3),
    sp.Integer(4),
)


def _sample_params(rng: random.Random, dist: str) -> dict[str, sp.Expr]:
    """Concrete distribution parameters (all bounded — small exact values)."""
    if dist == "uniform":
        a = rng.randint(1, 9)
        return {"a": sp.Integer(a), "b": sp.Integer(rng.randint(a + 1, 10))}
    if dist == "normal":
        return {
            "mu": sp.Integer(rng.randint(-3, 3)),
            "sigma": _NORMAL_SIGMAS[rng.randrange(len(_NORMAL_SIGMAS))],
        }
    if dist == "exp":
        return {"lam": sp.Integer(rng.randint(1, 6))}
    if dist == "binomial":
        n = rng.randint(2, 10)
        d = rng.randint(2, 5)
        return {"n": sp.Integer(n), "p": sp.Rational(rng.randint(1, d - 1), d)}
    if dist == "poisson":
        return {"lam": sp.Integer(rng.randint(1, 8))}
    return {
        "alpha": sp.Integer(rng.randint(1, 5)),
        "beta": sp.Integer(rng.randint(1, 5)),
    }


def moment_value(dist: str, params: dict[str, sp.Expr], moment: str) -> sp.Expr:
    """Exact closed-form moment E[X], Var[X], or E[X^2] (Rational/Integer).

    Module-level so tests recompute committed outputs independently.
    """
    if dist == "uniform":
        a, b = params["a"], params["b"]
        e, var = (a + b) / 2, (b - a) ** 2 / 12
    elif dist == "normal":
        mu, s = params["mu"], params["sigma"]
        e, var = mu, s**2
    elif dist == "exp":
        lam = params["lam"]
        e, var = 1 / lam, 1 / lam**2
    elif dist == "binomial":
        n, p = params["n"], params["p"]
        e, var = n * p, n * p * (1 - p)
    elif dist == "poisson":
        lam = params["lam"]
        e, var = lam, lam
    else:  # beta
        a, b = params["alpha"], params["beta"]
        e = a / (a + b)
        var = a * b / ((a + b) ** 2 * (a + b + 1))
    if moment == "E":
        return sp.simplify(e)
    if moment == "Var":
        return sp.simplify(var)
    return sp.simplify(var + e**2)


def _dist_tex(dist: str, params: dict[str, sp.Expr]) -> str:
    """LaTeX for the distribution context, e.g. `\\mathrm{Unif}(2,6)`."""
    if dist == "uniform":
        return rf"\mathrm{{Unif}}\left({params['a']},{params['b']}\right)"
    if dist == "normal":
        return rf"\mathcal{{N}}\left({params['mu']},{sp.latex(params['sigma'])}\right)"
    if dist == "exp":
        return rf"\mathrm{{Exp}}\left({params['lam']}\right)"
    if dist == "binomial":
        return rf"\mathrm{{Bin}}\left({params['n']},{sp.latex(params['p'])}\right)"
    if dist == "poisson":
        return rf"\mathrm{{Pois}}\left({params['lam']}\right)"
    return rf"\mathrm{{Beta}}\left({params['alpha']},{params['beta']}\right)"


def _overrides(dist_tex: str, moment: str) -> list[str]:
    """Two natural LaTeX surfaces per moment kind (dedupe-safe, context inside)."""
    if moment == "E":
        return [
            rf"\mathbb{{E}}\left[X\right],\quad X \sim {dist_tex}",
            rf"\mathrm{{E}}\left[X\right],\; X \sim {dist_tex}",
        ]
    if moment == "Var":
        return [
            rf"\operatorname{{Var}}\left(X\right),\; X \sim {dist_tex}",
            rf"\operatorname{{Var}}\left[X\right],\; X \sim {dist_tex}",
        ]
    return [
        rf"\mathbb{{E}}\left[X^{{2}}\right],\; X \sim {dist_tex}",
        rf"\mathrm{{E}}\left[X^{{2}}\right],\; X \sim {dist_tex}",
    ]


def _crosscheck(
    dist: str, params: dict[str, sp.Expr], moment: str, value: sp.Expr
) -> bool:
    """Independent recomputation via sympy.stats on the concrete distribution."""
    from sympy import stats as sp_stats

    if dist == "uniform":
        rv = sp_stats.Uniform("X", params["a"], params["b"])
    elif dist == "normal":
        rv = sp_stats.Normal("X", params["mu"], params["sigma"])
    elif dist == "exp":
        rv = sp_stats.Exponential("X", params["lam"])
    elif dist == "binomial":
        rv = sp_stats.Binomial("X", params["n"], params["p"])
    elif dist == "poisson":
        rv = sp_stats.Poisson("X", params["lam"])
    else:
        rv = sp_stats.Beta("X", params["alpha"], params["beta"])
    try:
        if moment == "E":
            stat = sp_stats.E(rv)
        elif moment == "Var":
            stat = sp_stats.variance(rv)
        else:
            stat = sp_stats.E(rv**2)
        return bool(sp.simplify(stat - value) == 0)
    except Exception:
        return False


class StatsMomentsFamily(SynthFamily):
    """`E[X]` / `Var[X]` / `E[X^2]` for fixed distributions (exact, gated)."""

    domain = "Mathematics_General"
    equation_type = "probability"

    def _gate(self, result_expr: sp.Expr) -> str:
        return (
            "closed-form moment == sympy.stats E/variance/E(X**2) "
            "cross-check on the concrete distribution (build-time, exact)"
        )

    def generate(
        self, seed: int, prefix: str, count: int, **opts: Any
    ) -> list[MathCodePair]:
        rng = random.Random(int_seed(f"stats_moments:{seed}"))
        out: list[MathCodePair] = []
        i = 0
        while len(out) < count * 2 and i < count * 40:  # cap attempts for gates
            i += 1
            dist = str(rng.choice(_DISTS))
            params = _sample_params(rng, dist)
            moment = str(rng.choice(_MOMENTS))
            value = moment_value(dist, params, moment)
            # gate: exact, finite, symbol-free — skip anything suspicious
            if not getattr(value, "is_finite", False) or value.has(sp.Symbol):
                continue
            if not _crosscheck(dist, params, moment, value):
                continue
            rows = self._build_pair(
                f"{prefix}_{dist}_{i}",
                value,  # unused for rendering; latex_override wins
                value,
                [],  # constant-output rows (no input variables)
                int_seed(f"stats_moments:{seed}:{i}"),
                n_variants=1,  # variant list is provided via latex_override
                meta={
                    "slice": "stats",
                    "dist": dist,
                    "moment": moment,
                    "params": {k: sp.sstr(v) for k, v in params.items()},
                    "gated": True,
                },
                latex_override=_overrides(_dist_tex(dist, params), moment),
            )
            out.extend(rows)
        return out[: count * 2]
