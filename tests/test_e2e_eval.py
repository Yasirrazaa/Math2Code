"""E2B eval tests: code-CSV loading + scoring path with a fake sandbox."""

from __future__ import annotations

import asyncio

from math2code.evaluation.e2e_eval import _exec_one, _load_code, _re_execute
from math2code.schemas import MathCodePair, TestCase


def test_load_code_csv(tmp_path) -> None:  # type: ignore[no-untyped-def]
    p = tmp_path / "code.csv"
    p.write_text('id,code\n"a","def calculate(x):\\n    return x + 1"\n"b",""\n')
    code_by_id = _load_code(p)
    assert "def calculate(x):" in code_by_id["a"]
    assert code_by_id["b"] == ""


class _FakeExec:
    def __init__(self, error=None, text=""):  # type: ignore[no-untyped-def]
        self.error = error
        self.text = text


class _FakeSandbox:
    async def __aenter__(self):  # type: ignore[no-untyped-def]
        return self

    async def __aexit__(self, *a):  # type: ignore[no-untyped-def]
        return False

    async def run_code(self, code: str):  # type: ignore[no-untyped-def]
        if "return x + 1" in code:
            return _FakeExec(text="3.0")
        return _FakeExec(error="boom")


def test_exec_one_runs_calculate() -> None:
    outs = asyncio.run(
        _exec_one(_FakeSandbox(), "def calculate(x):\n    return x + 1", [{"x": 2.0}])
    )
    assert outs == ["3.0"]


def test_exec_one_error_gives_none() -> None:
    outs = asyncio.run(
        _exec_one(_FakeSandbox(), "def bad():\n    return 1", [{"x": 1.0}])
    )
    assert outs == [None]


def test_re_execute_gathers_and_falls_back_to_none() -> None:
    pairs = [
        MathCodePair(
            task_id="a",
            latex_expression="x+1",
            test_cases=[TestCase(input={"x": 2.0}, output=3.0)],
        ),
        MathCodePair(
            task_id="b",
            latex_expression="x+1",
            test_cases=[TestCase(input={"x": 2.0}, output=3.0)],
        ),
    ]
    code_by_id = {"a": "def calculate(x):\n    return x + 1"}
    out = asyncio.run(
        _re_execute(pairs, code_by_id, concurrency=2, sandbox_cls=_FakeSandbox)
    )
    assert out["a"] == ["3.0"]
    assert out["b"] == [None]  # no code saved for b
