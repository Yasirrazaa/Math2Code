"""Sandbox tests: safety analysis, execution, timeout/memory containment."""

from __future__ import annotations

import pytest

from math2code.sandbox import SafetyError, SandboxPool, analyze_code, execute_code
from math2code.sandbox.pool import _run_one


@pytest.mark.parametrize(
    "bad",
    [
        'import os\nos.system("rm -rf /")',
        "print(open('/etc/passwd').read())",
        'eval("1+1")',
        "from subprocess import run",
        "import socket",
        "exec('print(1)')",
        "from math2code.schemas import MathCodePair",  # arbitrary package import
    ],
)
def test_analyze_code_rejects_unsafe(bad: str) -> None:
    with pytest.raises(SafetyError):
        analyze_code(bad)


@pytest.mark.parametrize(
    "good",
    [
        "import numpy as np\ndef calculate(x): return np.sqrt(x)",
        "from sympy import symbols, exp\ndef calculate(x): return float(exp(x))",
        "def calculate(x, y): return x + y",
    ],
)
def test_analyze_code_accepts_safe(good: str) -> None:
    analyze_code(good)  # must not raise


def test_execute_code_valid() -> None:
    res = execute_code(
        "def calculate(x, y):\n    return x ** 2 + y", inputs={"x": 3, "y": 1}
    )
    assert res.ok
    assert res.stdout == "10.0"


def test_execute_code_complex_output() -> None:
    res = execute_code("def calculate(x):\n    return complex(1, x)", inputs={"x": 2})
    assert res.ok
    assert res.stdout == "1.0+2.0j"


def test_execute_code_suffix_matching() -> None:
    """Competition convention: function param x_val fed by input key x."""
    res = execute_code(
        "def calculate(x_val, y_val):\n    return x_val + y_val",
        inputs={"x": 1, "y": 2},
    )
    assert res.ok
    assert res.stdout == "3.0"


def test_execute_code_missing_input_fails() -> None:
    res = execute_code("def calculate(x, y):\n    return x + y", inputs={"x": 1})
    assert not res.ok
    assert "no value for parameter" in res.stderr


def test_execute_code_timeout() -> None:
    res = execute_code(
        "def calculate(x):\n    while True:\n        pass",
        inputs={"x": 1},
        timeout_s=1.0,
    )
    assert res.timed_out


def test_execute_code_memory_limit() -> None:
    res = execute_code(
        "def calculate(x):\n    big = [0.0] * 50_000_000\n    return x",
        inputs={"x": 1},
        memory_mb=64,
    )
    assert not res.ok


def test_sandbox_pool_roundtrip() -> None:
    with SandboxPool(n_workers=2) as pool:
        res = pool.execute("def calculate(x):\n    return x * 2", inputs={"x": 21})
        assert res.ok
        assert res.stdout == "42.0"


def test_sandbox_pool_safety() -> None:
    with SandboxPool(n_workers=2) as pool:
        res = pool.execute('import os\nos.system("id")', inputs={})
        assert not res.ok
        assert res.safety_error


def test_worker_run_one_direct() -> None:
    exit_code, stdout, stderr, timed_out = _run_one(
        ("def calculate(x):\n    return x + 1", {"x": 1}), 2.0
    )
    assert exit_code == 0 and stdout == "2.0" and not stderr and not timed_out


# ---------------------------------------------------------------------------
# run_many: concurrent batch execution (results aligned to inputs)
# ---------------------------------------------------------------------------


def test_run_many_aligns_with_execute() -> None:
    code = "def calculate(x):\n    return x * 2"
    inputs = [{"x": i} for i in range(8)]
    with SandboxPool(n_workers=2) as pool:
        many = pool.run_many([(code, inp) for inp in inputs])
        singles = [pool.execute(code, inputs=inp) for inp in inputs]
    assert len(many) == len(inputs)
    assert all(r.ok for r in many)
    assert [r.stdout for r in many] == [r.stdout for r in singles]  # order kept


def test_run_many_mixed_safety_and_errors() -> None:
    good = "def calculate(x):\n    return x + 1"
    bad = "import os\nos.system('id')"
    with SandboxPool(n_workers=2) as pool:
        results = pool.run_many([(good, {"x": 1}), (bad, {}), (good, {"x": 2})])
    assert results[0].ok and results[0].stdout == "2.0"
    assert not results[1].ok and results[1].safety_error
    assert results[2].ok and results[2].stdout == "3.0"


def test_run_many_timeout_bounded() -> None:
    hang = "def calculate(x):\n    while True:\n        pass"
    with SandboxPool(n_workers=2, timeout_s=1.0) as pool:
        results = pool.run_many([(hang, {"x": 1})])
    assert len(results) == 1
    assert results[0].timed_out
    assert not results[0].ok


def test_run_many_empty() -> None:
    with SandboxPool(n_workers=2) as pool:
        assert pool.run_many([]) == []


def test_run_solution_on_cases_concurrent() -> None:
    code = "def calculate(x):\n    return x ** 2"
    cases = [{"x": 1}, {"x": 2}, {"x": 3}]
    with SandboxPool(n_workers=3) as pool:
        outputs, errors = pool.run_solution_on_cases(code, cases)
    assert not errors
    assert outputs == ["1.0", "4.0", "9.0"]
