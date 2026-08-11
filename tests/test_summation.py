"""Tests for the closed-form summation/product family (sympy.concrete)."""

from __future__ import annotations

import math

from math2code.data.oracle import oracle_verify
from math2code.data.synthesizer.families.summation import SummationFamily
from math2code.evaluation.metrics import parse_number
from math2code.sandbox import SandboxPool
from math2code.sandbox.base import execute_code
from math2code.schemas import MathCodePair


def _gen(kind: str = "mixed", count: int = 6) -> list[MathCodePair]:
    return SummationFamily().generate(seed=42, prefix="t", count=count, kind=kind)


def test_deterministic_generation() -> None:
    a = [(r.task_id, r.latex_expression) for r in _gen(count=8)]
    b = [(r.task_id, r.latex_expression) for r in _gen(count=8)]
    assert a == b
    assert a  # non-empty


def test_contract_shape_and_outputs() -> None:
    rows = _gen(kind="concrete", count=6)
    assert len(rows) == 6
    for r in rows:
        assert r.latex_expression
        assert "\\sum" in r.latex_expression or "\\prod" in r.latex_expression
        assert r.solution and "def calculate" in r.solution
        assert r.sympy_exp
        assert "Sum(" not in r.sympy_exp and "Product(" not in r.sympy_exp
        assert len(r.test_cases) == 5
        assert r.equation_type == "summation"
        assert r.domain == "Mathematics_General"
        assert r.synthetic is True
        assert r.metadata["slice"] == "summation"
        assert r.metadata["sum_kind"] in ("sum", "product")
        for tc in r.test_cases:
            assert tc.output is not None
            parsed = parse_number(tc.output)
            assert math.isfinite(parsed.real)
            assert parsed.imag == 0.0


def test_concrete_and_parameterized_shapes() -> None:
    conc = _gen(kind="concrete", count=8)
    assert conc
    for r in conc:
        assert r.metadata["param"] is False
        assert all(tc.input == {} for tc in r.test_cases)

    par = _gen(kind="parameterized", count=8)
    assert par
    for r in par:
        assert r.metadata["param"] is True
        assert "m" in r.latex_expression
        for tc in r.test_cases:
            assert set(tc.input) == {"m"}
            m = tc.input["m"]
            assert isinstance(m, int) and 1 <= m <= 12


def test_mixed_contains_both_row_kinds() -> None:
    rows = _gen(kind="mixed", count=12)
    kinds = {r.metadata["param"] for r in rows}
    assert kinds == {True, False}


def test_solution_matches_committed_outputs() -> None:
    """The solution code reproduces the committed outputs at the same inputs."""
    rows = _gen(kind="mixed", count=8)
    for r in rows:
        for tc in r.test_cases:
            res = execute_code(r.solution or "", inputs=tc.input)
            assert res.ok, res.stderr
            got = parse_number(res.stdout)
            exp = parse_number(tc.output)
            assert math.isclose(got.real, exp.real, rel_tol=1e-9, abs_tol=1e-9)
            assert math.isclose(got.imag, exp.imag, rel_tol=1e-9, abs_tol=1e-9)


def test_oracle_verification() -> None:
    """Full oracle (fresh jittered inputs, sandbox execution) accepts rows."""
    rows = _gen(kind="mixed", count=4)
    with SandboxPool(n_workers=2) as pool:
        for r in rows:
            ok, reasons = oracle_verify(r, r.solution or "", pool=pool)
            assert ok, reasons
