"""Tests for the extended number-theory family (σ/τ/φ/μ/π/modinv/powermod/p).

Generation is sandbox-heavy (~0.5 s per `execute_code` spawn), so rows are
generated ONCE in a module fixture and shared; operator-level correctness is
checked deterministically against the module-level code builders.
"""

from __future__ import annotations

import math

import pytest
import sympy as sp

from math2code.data.oracle import oracle_verify
from math2code.data.synthesizer.families.ntheory_ext import (
    NumberTheoryExtFamily,
    _modinv_code,
    _modinv_valid,
    _mu_code,
    _partition_code,
    _phi_code,
    _powermod_code,
    _primepi_code,
    _sigma_code,
    _tau_code,
)
from math2code.evaluation.metrics import parse_number
from math2code.sandbox import SandboxPool, execute_code
from math2code.schemas import MathCodePair
from math2code.schemas import TestCase as SchemaTestCase

_VOCAB = {
    "sigma",
    "tau",
    "varphi",
    "mu",
    "pi",
    "p",
    "modinv",
    "powermod",
}


@pytest.fixture(scope="module")
def rows() -> list[MathCodePair]:
    return NumberTheoryExtFamily().generate(seed=42, prefix="t", count=6)


def test_determinism() -> None:
    a = [
        (r.task_id, r.latex_expression)
        for r in NumberTheoryExtFamily().generate(seed=42, prefix="t", count=2)
    ]
    b = [
        (r.task_id, r.latex_expression)
        for r in NumberTheoryExtFamily().generate(seed=42, prefix="t", count=2)
    ]
    assert a == b
    assert len(a) == 2  # count * n_variants(1)
    assert len({tid for tid, _ in a}) == 2


def test_contract_shape(rows: list[MathCodePair]) -> None:
    assert rows
    for r in rows:
        assert isinstance(r, MathCodePair)
        assert r.latex_expression
        assert r.solution and "def calculate" in r.solution
        assert r.truth_code and "def calculate" in r.truth_code
        assert len(r.test_cases) == 5
        assert r.equation_type == "number_theory"
        assert r.domain == "Mathematics_General"
        assert r.synthetic is True
        assert r.metadata["slice"] == "numtheory"
        assert r.metadata["vocab"] in _VOCAB
        for tc in r.test_cases:
            parse_number(str(tc.output))  # must be parseable
        assert r.complexity in (1, 2, 3, 4, 5)


def test_all_operator_codes_correct() -> None:
    """Every operator's truth code must compute the canonical value (exact)."""
    cases = [
        (_sigma_code, {"n": 12}, 28.0),
        (_tau_code, {"n": 12}, 6.0),
        (_phi_code, {"n": 12}, 4.0),
        (_mu_code, {"n": 6}, 1.0),
        (_primepi_code, {"n": 100}, 25.0),
        (_partition_code, {"n": 10}, 42.0),
        (_modinv_code, {"a": 3, "m": 7}, 5.0),
        (_powermod_code, {"a": 3, "b": 5, "m": 7}, 5.0),
    ]
    for code_fn, inp, expected in cases:
        res = execute_code(code_fn(), inputs=inp)
        assert res.ok, code_fn.__name__
        assert abs(float(res.stdout) - expected) < 1e-9, code_fn.__name__


def test_committed_outputs_correct(rows: list[MathCodePair]) -> None:
    """Generated rows' committed outputs equal the canonical sympy values."""
    unary: dict[str, object] = {
        "sigma": sp.divisor_sigma,
        "tau": sp.divisor_count,
        "varphi": sp.totient,
        "mu": sp.mobius,
        "pi": sp.primepi,
        "p": sp.partition,
    }
    for r in rows:
        v = r.metadata["vocab"]
        for tc in r.test_cases:
            if v in unary:
                n = int(tc.input["n"])
                expected = float(unary[v](n))  # type: ignore[operator]
                assert abs(float(tc.output) - expected) < 1e-9, (r.task_id, v, n)
            elif v == "modinv":
                a, m = int(tc.input["a"]), int(tc.input["m"])
                assert math.gcd(a, m) == 1, (r.task_id, a, m)  # filter guarantees it
                assert abs(float(tc.output) - float(sp.mod_inverse(a, m))) < 1e-9
            elif v == "powermod":
                a, b, m = (
                    int(tc.input["a"]),
                    int(tc.input["b"]),
                    int(tc.input["m"]),
                )
                assert abs(float(tc.output) - float(pow(a, b, m))) < 1e-9
            else:  # pragma: no cover - vocab set guards this
                raise AssertionError(f"unknown vocab {v}")


def test_modinv_valid_filter() -> None:
    """The coprimality filter keeps only genuine inverse rows."""
    bad = MathCodePair(
        task_id="bad",
        latex_expression=r"\operatorname{modinv}{\left(a,m \right)}",
        test_cases=[
            SchemaTestCase(input={"a": 4, "m": 6}, output=0.0),  # gcd(4,6)=2 -> sentinel
            SchemaTestCase(input={"a": 3, "m": 7}, output=5.0),
        ],
    )
    good = MathCodePair(
        task_id="good",
        latex_expression=r"\operatorname{modinv}{\left(a,m \right)}",
        test_cases=[
            SchemaTestCase(input={"a": 3, "m": 7}, output=5.0),
            SchemaTestCase(input={"a": 2, "m": 5}, output=3.0),
        ],
    )
    assert not _modinv_valid(bad)
    assert _modinv_valid(good)


def test_truth_code_runs_and_matches_committed(rows: list[MathCodePair]) -> None:
    for r in rows[:3]:
        for tc in r.test_cases:
            res = execute_code(r.truth_code, inputs=tc.input)
            assert res.ok, (r.task_id, res.stderr)
            assert abs(float(res.stdout) - float(tc.output)) < 1e-9, r.task_id


def test_oracle_verify(rows: list[MathCodePair]) -> None:
    assert rows
    with SandboxPool(n_workers=2, timeout_s=10, memory_mb=2048) as pool:
        for r in rows[:2]:
            ok, reasons = oracle_verify(r, r.solution or "", pool=pool)
            assert ok, (r.task_id, reasons)
