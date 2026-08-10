"""Code-first synthesizer tests: printer determinism, notation variants,
domain-aware sampling, family gates, and oracle acceptance of emitted rows.
"""

from __future__ import annotations

import json

import sympy as sp

from math2code.data.synthesizer import (
    DerivativeFamily,
    IntegralFamily,
    render_variants,
    sample_inputs,
)
from math2code.data.synthesizer.printer import VariantPrinter, int_seed
from math2code.sandbox import SandboxPool
from math2code.schemas import MathCodePair

_X = sp.Symbol("x")


# --------------------------------------------------------------------------
# printer: determinism + variants
# --------------------------------------------------------------------------


def test_render_variants_deterministic() -> None:
    expr = sp.exp(2 * _X + 1)
    a = render_variants(expr, seed=7, n_variants=3)
    b = render_variants(expr, seed=7, n_variants=3)
    assert a == b  # byte-identical regeneration


def test_render_variants_produce_distinct_forms() -> None:
    expr = sp.exp(2 * _X + 1)
    seen = set()
    for seed in range(10):
        seen.update(render_variants(expr, seed=seed, n_variants=2))
    assert any("e^{" in s for s in seen)  # e^{...} form
    assert any("\\exp" in s for s in seen)  # \exp(...) form


def test_printer_integer_guard_no_fake_fraction() -> None:
    # Integer subclasses Rational; the guard must keep integers plain
    import random

    expr = 3 * _X**2 + 1
    tex = VariantPrinter(random.Random(int_seed("t")), {}).doprint(expr)
    assert "\\frac{3}{1}" not in tex
    assert "3" in tex


def test_mul_dot_variant() -> None:
    import random

    expr = _X * sp.Symbol("y") * sp.sin(_X)
    tex = VariantPrinter(random.Random(int_seed("t2")), {"mul_symbol": "dot"}).doprint(
        expr
    )
    assert "\\cdot" in tex


def test_derivative_variants_cover_leibniz_and_lagrange() -> None:
    from sympy.printing.latex import LatexPrinter

    d = sp.Derivative(sp.sin(_X), _X)
    forms = {LatexPrinter().doprint(d)}
    for seed in range(20):
        forms.update(render_variants(d, seed=seed, n_variants=3))
    assert any("\\frac{d}{d" in s for s in forms)  # Leibniz
    assert any("f'" in s or "f^{(" in s for s in forms)  # Lagrange


# --------------------------------------------------------------------------
# sampler: domain awareness
# --------------------------------------------------------------------------


def test_sampler_avoids_poles() -> None:
    expr = 1 / (_X - 2)
    pts = sample_inputs(expr, [_X], n=40, seed=1)
    assert len(pts) == 40
    assert all(abs(p["x"] - 2.0) > 0.2 for p in pts)


def test_sampler_finite_for_log() -> None:
    expr = sp.log(_X)
    pts = sample_inputs(expr, [_X], n=30, seed=2)
    assert all(p["x"] > 0 for p in pts)


def test_sampler_ints_stay_ints() -> None:
    expr = _X**2 + 3 * _X
    pts = sample_inputs(expr, [_X], n=20, seed=3, ints_only=True)
    assert all(isinstance(p["x"], int) for p in pts)


# --------------------------------------------------------------------------
# families: gates + oracle acceptance
# --------------------------------------------------------------------------


def _dumps(rows: list[MathCodePair]) -> str:
    return json.dumps([r.model_dump() for r in rows], sort_keys=True)


def test_generation_is_byte_identical() -> None:
    a = DerivativeFamily().generate(seed=11, prefix="d", count=4)
    b = DerivativeFamily().generate(seed=11, prefix="d", count=4)
    assert _dumps(a) == _dumps(b)


def test_derivative_rows_have_variants_and_ground_truth() -> None:
    rows = DerivativeFamily().generate(seed=5, prefix="d", count=3)
    assert rows
    # each math object -> n_variants rows sharing a base id
    base_ids = {r.task_id.rsplit(":v", 1)[0] for r in rows}
    assert len(base_ids) == len(set(base_ids))
    for r in rows:
        assert r.solution and "calculate" in r.solution
        assert r.sympy_exp and sp.sympify(r.sympy_exp) is not None
        assert len(r.test_cases) == 5
        assert "diff(f)" in r.metadata["family_gate"]


def test_derivative_rows_pass_oracle() -> None:
    rows = DerivativeFamily().generate(seed=5, prefix="d", count=6)
    with SandboxPool(n_workers=2, timeout_s=10, memory_mb=2048) as pool:
        from math2code.data.oracle import oracle_verify

        for r in rows:
            ok, reasons = oracle_verify(r, r.solution or "", pool=pool)
            assert ok, f"{r.task_id}: {reasons}"


def test_integral_rows_pass_oracle_and_gate() -> None:
    rows = IntegralFamily().generate(seed=6, prefix="i", count=6, kind="indefinite")
    assert rows
    with SandboxPool(n_workers=2, timeout_s=10, memory_mb=2048) as pool:
        from math2code.data.oracle import oracle_verify

        for r in rows:
            # gate recorded + oracle passes
            assert "diff(F)" in r.metadata["family_gate"]
            ok, reasons = oracle_verify(r, r.solution or "", pool=pool)
            assert ok, f"{r.task_id}: {reasons}"


def test_definite_integral_rows_are_constant_output() -> None:
    rows = IntegralFamily().generate(seed=7, prefix="i", count=4, kind="definite")
    assert rows
    for r in rows:
        assert r.metadata["kind"] == "definite"
        assert all(tc.input == {} for tc in r.test_cases)
        assert all(isinstance(tc.output, (int, float)) for tc in r.test_cases)
        assert r.solution and ("float(" in r.solution or "complex(" in r.solution)
    with SandboxPool(n_workers=2, timeout_s=10, memory_mb=2048) as pool:
        from math2code.data.oracle import oracle_verify

        for r in rows:
            ok, reasons = oracle_verify(r, r.solution or "", pool=pool)
            assert ok, f"{r.task_id}: {reasons}"


def test_variable_limit_integral_rows() -> None:
    rows = IntegralFamily().generate(seed=8, prefix="i", count=4, kind="variable")
    assert rows
    for r in rows:
        assert r.metadata["kind"] == "variable_limit"
        assert "\\int" in r.latex_expression or "\\operatorname" in r.latex_expression
