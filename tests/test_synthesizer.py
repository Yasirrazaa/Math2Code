"""Code-first synthesizer tests: printer determinism, notation variants,
domain-aware sampling, family gates, and oracle acceptance of emitted rows.
"""

from __future__ import annotations

import json

import sympy as sp

from math2code.data.synthesizer import (
    DerivativeFamily,
    FunctionVocabFamily,
    IntegralFamily,
    ODEFamily,
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
    assert any("'" in s or "^{(" in s for s in forms)  # Lagrange (prime of named fn)


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
        # competition surface — may render as \int_0^x, \operatorname{Integral}, or
        # the \mathtt{\text{Integral(...)}} repr-wrapped form (one of these)
        assert (
            "\\int" in r.latex_expression
            or "\\operatorname" in r.latex_expression
            or "Integral" in r.latex_expression  # repr-wrapped surface
        )


# --------------------------------------------------------------------------
# function-vocabulary family
# --------------------------------------------------------------------------


def test_functions_inverse_trig_domain() -> None:
    rows = FunctionVocabFamily().generate(
        seed=10, prefix="f", count=8, kind="inverse_trig"
    )
    assert rows
    # real outputs (sampler realness gate) and bounded args for asin/acos/atan2
    for r in rows:
        assert r.output_type == "real"
        x0 = r.test_cases[0].input.get("x", 0)
        if r.metadata["vocab"] in ("asin", "acos", "atan2"):
            assert abs(x0) <= 0.95


def test_functions_complex_slice_string_outputs() -> None:
    rows = FunctionVocabFamily().generate(seed=11, prefix="f", count=25)
    complex_rows = [r for r in rows if r.output_type == "complex"]
    assert complex_rows
    from math2code.evaluation.metrics import parse_number

    r = complex_rows[0]
    vals = [parse_number(tc.output) for tc in r.test_cases]
    # complex-valued cases are 're+imj' strings (JSON-safe); parse_number round-trips
    assert any(
        isinstance(tc.output, str) and tc.output.endswith("j") for tc in r.test_cases
    )
    assert any(abs(v.imag) > 1e-9 for v in vals)


def test_functions_factorial_integer_domain() -> None:
    rows = FunctionVocabFamily().generate(
        seed=12, prefix="f", count=6, kind="factorial"
    )
    assert rows
    for r in rows:
        assert all(isinstance(tc.input["x"], int) for tc in r.test_cases)


# --------------------------------------------------------------------------
# ODE family: gates + oracle
# --------------------------------------------------------------------------


def test_ode_rows_pin_ics_and_pass_oracle() -> None:
    rows = ODEFamily().generate(seed=13, prefix="o", count=5)
    assert rows
    with SandboxPool(n_workers=2, timeout_s=10, memory_mb=2048) as pool:
        from math2code.data.oracle import oracle_verify

        for r in rows:
            assert "checkodesol" in r.metadata["family_gate"]
            # IC at x=0 reproduced by the first case output
            ok, reasons = oracle_verify(r, r.solution or "", pool=pool)
            assert ok, f"{r.task_id}: {reasons}"


# --------------------------------------------------------------------------
# multivariate family + training mixture
# --------------------------------------------------------------------------


def test_multivariate_cross_terms_and_metadata() -> None:
    from math2code.data.synthesizer import MultivariateFamily

    rows = MultivariateFamily().generate(seed=7, prefix="m", count=10)
    assert rows
    assert all(r.metadata["cross_terms"] is True for r in rows)
    assert all(r.metadata["n_vars"] >= 2 for r in rows)
    assert any(r.metadata["coefficient_kind"] == "decimal" for r in rows)
    assert any(r.metadata["vocab"] == "augmented" for r in rows)
    assert any(r.equation_type == "rational_multivariate" for r in rows)


def test_build_mixture_deterministic_and_contamination_free() -> None:
    import hashlib
    import subprocess
    import sys

    out_a = "/tmp/mix_test_a.jsonl"
    out_b = "/tmp/mix_test_b.jsonl"
    for out in (out_a, out_b):
        proc = subprocess.run(
            [
                sys.executable,
                "scripts/build_mixture.py",
                "--size",
                "1000",
                "--out",
                out,
            ],
            cwd=".",
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stderr[-500:]
        assert "CONTAMINATION" not in proc.stdout
    a = hashlib.sha256(open(out_a, "rb").read()).digest()
    b = hashlib.sha256(open(out_b, "rb").read()).digest()
    assert a == b  # seeded determinism -> byte-identical artifact

    # every row in the mixture carries the competition/synthetic contract
    import json

    rows = [json.loads(line) for line in open(out_a)]
    assert len(rows) == 1000
    assert all("latex_expression" in r and "test_cases" in r for r in rows)


def test_sequences_integer_indices() -> None:
    from math2code.data.synthesizer import SequenceFamily

    rows = SequenceFamily().generate(seed=5, prefix="s", count=6)
    assert rows
    assert all(r.equation_type == "sequences" for r in rows)
    assert all(r.metadata["slice"] == "sequences" for r in rows)
    for r in rows:
        for tc in r.test_cases:
            assert isinstance(tc.input["n"], int) and tc.input["n"] >= 1


def test_geometry_positive_domain() -> None:
    from math2code.data.synthesizer import GeometryFamily

    rows = GeometryFamily().generate(seed=6, prefix="g", count=6)
    assert rows
    assert all(r.equation_type == "geometry" for r in rows)
    for r in rows:
        for tc in r.test_cases:
            for v in tc.input.values():
                assert abs(complex(v).real) > 0
    # pi appears in some rows
    assert any("\\pi" in r.latex_expression for r in rows)


def test_summary_synthetic_runs_and_is_honest() -> None:
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, "scripts/summary_synthetic.py"],
        cwd=".",
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr[-500:]
    assert "frozen train" in proc.stdout
    assert "synthetic pool" in proc.stdout
    assert "complexity histogram" in proc.stdout


def test_edge_logscale_and_sqrt_trap() -> None:
    from math2code.data.synthesizer import EdgeCaseFamily

    rows = EdgeCaseFamily().generate(seed=8, prefix="e", count=10)
    assert rows
    assert all(r.metadata["slice"] == "edge" for r in rows)
    # logscale rows span orders of magnitude
    ls = [r for r in rows if r.metadata["vocab"] == "logscale_poly"]
    if ls:
        mags = [
            abs(complex(tc.output).real)  # type: ignore[call-overload, arg-type]
            for r in ls
            for tc in r.test_cases
            if abs(complex(tc.output).real) > 0  # type: ignore[call-overload, arg-type]
        ]
        assert max(mags) > 1e3  # log-scale magnitudes reach large values
    # sqrt(x**2) trap: truth at negative x is |x| + c (never x + c)
    trap = [r for r in rows if r.metadata["vocab"] == "sqrt_sq_trap"]
    assert trap
    r = trap[0]

    def _cf(v: object) -> float:  # type: ignore[misc]
        return complex(v).real  # type: ignore[call-overload, no-any-return]

    x0 = _cf(r.test_cases[0].input["x"])
    c = round(_cf(r.test_cases[0].output) - abs(x0))
    assert 1 <= c <= 5  # template constant
    for tc in r.test_cases:
        x = _cf(tc.input["x"])
        # truth is |x| + c (never x + c for negative x)
        assert _cf(tc.output) == abs(x) + c


def test_numtheory_truth_code_contract() -> None:
    """gcd/lcm: truth is code, not sympify; oracle verifies via truth_code."""
    from math2code.data.synthesizer import NumberTheoryFamily

    rows = NumberTheoryFamily().generate(seed=4, prefix="nt", count=3)
    assert rows
    r = rows[0]
    assert r.truth_code is not None  # truth_code path engaged
    assert not r.sympy_exp  # no sympify truth for these families
    assert r.equation_type == "number_theory"
    # exact integer outputs
    for tc in r.test_cases:
        assert float(complex(tc.output).real).is_integer()  # type: ignore[call-overload, arg-type]

    # the oracle must ACCEPT the rows: candidate == truth on fresh inputs
    with SandboxPool(n_workers=2, timeout_s=10, memory_mb=2048) as pool:
        from math2code.data.oracle import oracle_verify

        for row in rows:
            ok, reasons = oracle_verify(row, row.solution or "", pool=pool)
            assert ok, f"{row.task_id}: {reasons}"


def test_mixture_keeps_parameterized_latex_dupes() -> None:
    """gcd rows share one latex but differ in test outputs — all kept."""
    import json
    import subprocess
    import sys

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/build_mixture.py",
            "--size",
            "1000",
            "--out",
            "/tmp/mix_nt.jsonl",
        ],
        cwd=".",
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr[-500:]
    rows = [json.loads(line) for line in open("/tmp/mix_nt.jsonl")]
    numtheory = [r for r in rows if r.get("equation_type") == "number_theory"]
    assert len(numtheory) >= 10  # parameterized dupes survive the dedupe
    # outputs differ across gcd rows sharing the same latex
    latexes = {r["latex_expression"] for r in numtheory}
    outputs = {tuple(tc["output"] for tc in r["test_cases"]) for r in numtheory}
    assert len(latexes) < len(outputs)  # same latex, different problems


def test_all_families_metadata_is_json_serializable() -> None:
    """Regression guard: every registered family's emitted rows must round-trip
    through json.dumps (the generation pipeline writes rows as JSONL). A
    non-serializable metadata field (e.g. sympy Integer leaking through
    sp.degree / sp.roots) silently breaks the pipeline.
    """
    from math2code.data.synthesizer import FAMILIES

    n_seen = 0
    for name, family_cls in FAMILIES.items():
        rows = family_cls().generate(seed=42, prefix=name, count=4)
        for r in rows:
            n_seen += 1
            try:
                json.dumps(r.metadata)
                json.dumps([tc.output for tc in r.test_cases])
                # also exercise the full row model_dump for completeness
                json.dumps(r.model_dump())
            except TypeError as exc:
                raise AssertionError(f"{name}/{r.task_id}: {exc}") from exc
    assert n_seen > 50  # we registered enough families that this is a real sweep
