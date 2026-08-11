"""Tests for the special-functions family (gated portfolio, constant rows)."""

from __future__ import annotations

import math

import sympy as sp

from math2code.data.oracle import oracle_verify
from math2code.data.synthesizer.families.special_functions import (
    SpecialFunctionsFamily,
)
from math2code.evaluation.metrics import parse_number
from math2code.sandbox import SandboxPool, execute_code
from math2code.schemas import MathCodePair

_FAMILY = SpecialFunctionsFamily()
_NAME = "SpecialFunctionsFamily"


def _gen(count: int = 5, seed: int = 42) -> list[MathCodePair]:
    return _FAMILY.generate(seed, prefix="t", count=count)


def test_determinism() -> None:
    a = _gen(5, seed=42)
    b = _gen(5, seed=42)
    assert [(r.task_id, r.latex_expression, r.test_cases[0].output) for r in a] == [
        (r.task_id, r.latex_expression, r.test_cases[0].output) for r in b
    ]
    assert a


def test_contract_shape() -> None:
    rows = _gen(5)
    assert rows, "generate must produce rows"
    assert len(rows) == 5  # n_variants=1
    for r in rows:
        assert isinstance(r, MathCodePair)
        assert r.latex_expression
        assert r.solution and "def calculate" in r.solution
        assert len(r.test_cases) == 5
        assert all(tc.input == {} for tc in r.test_cases), "constant-row inputs"
        assert r.equation_type == "special_functions"
        assert r.domain == "Mathematics_General"
        assert r.synthetic is True
        assert r.metadata["slice"] == "special"
        assert r.metadata["gated"] is True
        assert r.output_type == "real"
        for tc in r.test_cases:
            assert tc.output is not None
            assert math.isfinite(float(tc.output))
            assert abs(complex(parse_number(str(tc.output))).imag) < 1e-12


def test_gated_flag_and_finiteness() -> None:
    rows = _gen(10)
    assert rows
    for r in rows:
        assert r.metadata["gated"] is True
        for tc in r.test_cases:
            assert math.isfinite(float(tc.output))


def test_both_modes_present() -> None:
    rows = _gen(12)
    modes = {r.metadata["mode"] for r in rows}
    assert "exact" in modes and "evalf" in modes, modes
    # exact rows carry sympy_exp; evalf rows carry truth_code
    assert any(r.sympy_exp for r in rows)
    assert any(r.truth_code for r in rows)


def test_committed_outputs_match_ground_truth() -> None:
    """Exact rows: output == sp.N(sympy_exp); evalf rows: output == truth_code."""
    rows = _gen(12)
    for r in rows:
        if r.metadata["mode"] == "exact":
            truth = sp.sympify(r.sympy_exp or "")
            assert truth.is_finite
            expected = float(sp.N(truth))
            assert abs(float(r.test_cases[0].output) - expected) < 1e-9, (
                r.task_id,
                r.sympy_exp,
                r.test_cases[0].output,
            )
        else:
            res = execute_code(r.truth_code or "", inputs={})
            assert res.ok, res.stderr
            assert abs(float(res.stdout) - float(r.test_cases[0].output)) < 1e-9, (
                r.task_id,
                res.stdout,
                r.test_cases[0].output,
            )


def test_known_exact_values() -> None:
    """Sanity anchors: gamma(5)=24, legendre(3,2)=17, zeta(2)=pi^2/6."""
    assert sp.gamma(5) == 24
    assert sp.legendre(3, 2) == 17
    assert sp.zeta(2) == sp.pi**2 / 6
    assert sp.beta(4, 5).rewrite(sp.gamma) == sp.Rational(1, 280)


def test_solution_matches_truth() -> None:
    """Executing each solution must reproduce the committed output."""
    for r in _gen(5):
        res = execute_code(r.solution or "", inputs={})
        assert res.ok, res.stderr
        assert abs(float(res.stdout) - float(r.test_cases[0].output)) < 1e-9


def test_oracle_verification() -> None:
    rows = _gen(3)
    assert rows
    with SandboxPool(n_workers=2, timeout_s=10, memory_mb=2048) as pool:
        for r in rows:
            ok, reasons = oracle_verify(r, r.solution or "", pool=pool)
            assert ok, f"{r.task_id}: {reasons}"


def test_family_metadata() -> None:
    assert _FAMILY.domain == "Mathematics_General"
    assert _FAMILY.equation_type == "special_functions"
    assert "evalf" in _FAMILY._gate(sp.erf(1))  # noqa: SLF001 - gate text audit
