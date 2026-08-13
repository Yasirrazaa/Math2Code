"""Tests for the series-coefficient family (constant-output, exact rational)."""

from __future__ import annotations

import sympy as sp

from math2code.data.oracle import oracle_verify
from math2code.data.synthesizer.families.series_coeff import SeriesCoefficientFamily
from math2code.evaluation.metrics import parse_number
from math2code.sandbox import SandboxPool, execute_code
from math2code.schemas import MathCodePair

_FAMILY = SeriesCoefficientFamily()
_NAME = "SeriesCoefficientFamily"


def _gen(count: int = 5, seed: int = 42) -> list[MathCodePair]:
    return _FAMILY.generate(seed, prefix="t", count=count)


def test_determinism() -> None:
    a = _gen(5, seed=42)
    b = _gen(5, seed=42)
    assert [(r.task_id, r.latex_expression) for r in a] == [
        (r.task_id, r.latex_expression) for r in b
    ]
    assert a


def test_contract_shape() -> None:
    rows = _gen(5)
    assert rows, "generate must produce rows"
    assert len(rows) == 10  # 5 objects * 2 latex variants
    for r in rows:
        assert isinstance(r, MathCodePair)
        assert r.latex_expression
        assert r.solution and "def calculate" in r.solution
        assert len(r.test_cases) == 5
        assert all(tc.input == {} for tc in r.test_cases), "constant-row inputs"
        assert r.equation_type == "series"
        assert r.domain == "Mathematics_Calculus"
        assert r.synthetic is True
        assert r.metadata["slice"] == "series"
        assert r.output_type == "real"
        for tc in r.test_cases:
            assert tc.output is not None
            assert abs(complex(parse_number(str(tc.output))).imag) < 1e-12


def test_latex_surfaces() -> None:
    rows = _gen(5)
    for r in rows:
        assert (
            r"\operatorname{coeff}_{x^{" in r.latex_expression
            or r"\left[x^{" in r.latex_expression
        ), r.latex_expression
    surfaces = {r.latex_expression for r in rows}
    assert len(surfaces) == len(rows), "all rows must have distinct latex"


def test_gate_holds_exact_rational() -> None:
    """Ground truth (sympy_exp) must be an exact finite Rational/Integer."""
    rows = _gen(10)
    for r in rows:
        truth = sp.sympify(r.sympy_exp or "")
        assert truth.is_finite, r.sympy_exp
        assert isinstance(truth, (sp.Rational, sp.Integer)), type(truth)
        assert truth != 0
        # the committed output must equal the exact truth (float of Rational)
        for tc in r.test_cases:
            expected = float(sp.N(truth))
            assert abs(float(tc.output) - expected) < 1e-12  # type: ignore[arg-type]


def test_solution_matches_truth() -> None:
    """Executing each solution must reproduce the committed output."""
    for r in _gen(5):
        res = execute_code(r.solution or "", inputs={})
        assert res.ok, res.stderr
        assert abs(float(res.stdout) - float(r.test_cases[0].output)) < 1e-9  # type: ignore[arg-type]


def test_oracle_verification() -> None:
    rows = _gen(3)
    assert rows
    with SandboxPool(n_workers=2, timeout_s=10, memory_mb=2048) as pool:
        for r in rows:
            ok, reasons = oracle_verify(r, r.solution or "", pool=pool)
            assert ok, f"{r.task_id}: {reasons}"
