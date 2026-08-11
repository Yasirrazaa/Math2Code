"""Tests for the complex-output evaluation family (real inputs, 're+imj' truth)."""

from __future__ import annotations

import cmath

from math2code.data.synthesizer.families.complex_eval import (
    _MIN_IMAG,
    _VOCAB,
    ComplexEvalFamily,
)
from math2code.evaluation.metrics import parse_number
from math2code.sandbox import SandboxPool
from math2code.schemas import MathCodePair


def _rows(count: int = 6, seed: int = 42, n_variants: int = 2) -> list[MathCodePair]:
    return ComplexEvalFamily().generate(
        seed=seed, prefix="t", count=count, n_variants=n_variants
    )


def test_deterministic_generation() -> None:
    a = _rows(count=6, seed=42)
    b = _rows(count=6, seed=42)
    assert [(r.task_id, r.latex_expression) for r in a] == [
        (r.task_id, r.latex_expression) for r in b
    ]


def test_contract_shape_and_complex_outputs() -> None:
    rows = _rows(count=8)
    assert rows  # gate must not starve generation
    for r in rows:
        assert isinstance(r, MathCodePair)
        assert r.latex_expression  # non-empty
        assert r.solution and "def calculate" in r.solution
        assert len(r.test_cases) == 5
        assert r.equation_type == "complex"
        assert r.domain == "Mathematics_General"
        assert r.synthetic is True
        assert r.output_type == "complex"
        assert r.metadata["slice"] == "vocab"
        assert r.metadata["vocab"] in _VOCAB
        for tc in r.test_cases:
            # committed outputs must round-trip through the frozen metric and be
            # GENUINELY complex (the 're+imj' path only triggers then)
            val = parse_number(tc.output)
            assert abs(val.imag) > _MIN_IMAG, (r.task_id, tc.output)
            # inputs stay real (complex comes from the expression, not the input)
            for v in tc.input.values():
                assert not isinstance(v, complex) or abs(v.imag) <= 1e-12


def test_ground_truth_recomputed_matches_committed_outputs() -> None:
    import sympy as sp

    rows = _rows(count=6)
    for r in rows:
        expr = sp.sympify(r.sympy_exp)  # type: ignore[arg-type]
        for tc in r.test_cases:
            subs = {sp.Symbol(k): v for k, v in tc.input.items()}
            truth = complex(sp.N(expr.subs(subs)))
            got = parse_number(tc.output)
            assert cmath.isclose(got, truth, rel_tol=1e-9, abs_tol=1e-12), (
                r.task_id,
                tc.input,
                got,
                truth,
            )


def test_oracle_verification() -> None:
    rows = _rows(count=4, seed=7)
    assert rows
    with SandboxPool(n_workers=2, timeout_s=10, memory_mb=2048) as pool:
        from math2code.data.oracle import oracle_verify

        for r in rows[:2]:  # keep the suite fast
            ok, reasons = oracle_verify(r, r.solution, pool=pool)  # type: ignore[arg-type]
            assert ok, (r.task_id, reasons)


def test_vocabulary_coverage_and_finite_outputs() -> None:
    # a larger draw must exercise the full complex vocabulary (negative-imag
    # branches included: asin_shift, acos_shift, acosh_neg, tanh_imag)
    rows = _rows(count=60, seed=11, n_variants=1)
    seen = {r.metadata["vocab"] for r in rows}
    assert seen == set(_VOCAB), f"missing forms: {set(_VOCAB) - seen}"
    for r in rows:
        for tc in r.test_cases:
            val = parse_number(tc.output)
            assert cmath.isfinite(val.real) and cmath.isfinite(val.imag)


def test_all_rows_complex_regression() -> None:
    # regression: negative-imag outputs must round-trip parse_number (the
    # format_output '+-' bug made them unparseable -> rows were silently dropped)
    rows = _rows(count=20, seed=3, n_variants=1)
    for r in rows:
        for tc in r.test_cases:
            assert abs(parse_number(tc.output).imag) > _MIN_IMAG
