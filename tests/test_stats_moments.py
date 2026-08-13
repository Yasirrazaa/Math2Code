"""Tests for the stats-moments family (gated portfolio breadth)."""

from __future__ import annotations

import math

import sympy as sp

from math2code.data.synthesizer.families.stats_moments import (
    StatsMomentsFamily,
    moment_value,
)
from math2code.data.synthesizer.verify import verify_generated
from math2code.evaluation.metrics import parse_number
from math2code.sandbox import SandboxPool
from math2code.schemas import MathCodePair

_FAMILY = StatsMomentsFamily()
_DISTS = {"uniform", "normal", "exp", "binomial", "poisson", "beta"}
_MOMENTS = {"E", "Var", "E2"}


def _rows(count: int = 5, seed: int = 42) -> list[MathCodePair]:
    return _FAMILY.generate(seed=seed, prefix="t", count=count)


def test_determinism() -> None:
    a = _rows(count=5, seed=42)
    b = _rows(count=5, seed=42)
    assert [(r.task_id, r.latex_expression) for r in a] == [
        (r.task_id, r.latex_expression) for r in b
    ]


def test_contract_shape() -> None:
    rows = _rows(count=5)
    assert rows
    for r in rows:
        assert isinstance(r, MathCodePair)
        assert r.latex_expression
        assert "X \\sim" in r.latex_expression  # distribution context present
        assert r.solution and "def calculate" in r.solution
        assert len(r.test_cases) == 5
        assert all(tc.input == {} for tc in r.test_cases)  # constant rows
        assert r.equation_type == "probability"
        assert r.domain == "Mathematics_General"
        assert r.synthetic is True
        assert r.output_type == "real"
        assert r.metadata["slice"] == "stats"
        assert r.metadata["gated"] is True
        assert r.metadata["dist"] in _DISTS
        assert r.metadata["moment"] in _MOMENTS
        for tc in r.test_cases:
            assert math.isfinite(parse_number(tc.output).real)
            assert abs(parse_number(tc.output).imag) < 1e-12


def test_committed_outputs_equal_closed_form() -> None:
    rows = _rows(count=8, seed=3)
    for r in rows:
        params = {k: sp.sympify(v) for k, v in r.metadata["params"].items()}
        expected = float(moment_value(r.metadata["dist"], params, r.metadata["moment"]))
        for tc in r.test_cases:
            assert abs(parse_number(tc.output).real - expected) < 1e-9, (
                r.task_id,
                r.metadata,
            )


def test_committed_outputs_match_sympy_stats() -> None:
    """Independent recomputation: closed forms agree with sympy.stats."""
    from sympy import stats as sp_stats

    for r in _rows(count=6, seed=11):
        p = {k: sp.sympify(v) for k, v in r.metadata["params"].items()}
        dist = r.metadata["dist"]
        moment = r.metadata["moment"]
        if dist == "uniform":
            rv = sp_stats.Uniform("X", p["a"], p["b"])
        elif dist == "normal":
            rv = sp_stats.Normal("X", p["mu"], p["sigma"])
        elif dist == "exp":
            rv = sp_stats.Exponential("X", p["lam"])
        elif dist == "binomial":
            rv = sp_stats.Binomial("X", p["n"], p["p"])
        elif dist == "poisson":
            rv = sp_stats.Poisson("X", p["lam"])
        else:
            rv = sp_stats.Beta("X", p["alpha"], p["beta"])
        stat = (
            sp_stats.E(rv)
            if moment == "E"
            else sp_stats.variance(rv)
            if moment == "Var"
            else sp_stats.E(rv**2)
        )
        expected = moment_value(dist, p, moment)
        assert sp.simplify(stat - expected) == 0, (r.task_id, stat, expected)


def test_oracle_verification() -> None:
    rows = _rows(count=3, seed=7)
    with SandboxPool(n_workers=2) as pool:
        outcome = verify_generated(rows, pool, n_points=20)
    assert not outcome.rejected, outcome.rejected[:3]
    assert len(outcome.kept) == len(rows)


def test_distribution_and_moment_coverage() -> None:
    rows = _rows(count=12, seed=5)
    dists = {r.metadata["dist"] for r in rows}
    moments = {r.metadata["moment"] for r in rows}
    assert len(dists) >= 4  # generation spreads across distributions
    assert len(moments) >= 2
    assert len({r.latex_expression for r in rows}) == len(rows)  # no dupes
