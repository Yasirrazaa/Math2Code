"""Prompt / extraction tests."""

from __future__ import annotations

from math2code.model.prompts import (
    build_prompt,
    extract_code,
    extract_final_answer,
    extract_variables,
)
from math2code.model.train import format_instruction


def test_build_prompt_contains_latex_and_function() -> None:
    prompt = build_prompt(r"\frac{x^{2} + 3y^{2}}{2x + 5y}", variables=["x", "y"])
    assert r"\frac{x^{2}" in prompt
    assert "def calculate(x, y)" in prompt


def test_extract_code_fence() -> None:
    text = "Here is the code:\n```python\nimport sympy as sp\ndef calculate(x):\n    return x\n```\nHope it helps!"
    assert extract_code(text) == "import sympy as sp\ndef calculate(x):\n    return x"


def test_extract_code_execute_tag() -> None:
    text = "<think>Let me solve.</think>\n<execute>\ndef calculate(x):\n    return x\n</execute>"
    assert extract_code(text) == "def calculate(x):\n    return x"


def test_extract_code_bare() -> None:
    text = "def calculate(x):\n    return x\n\nThis is the answer."
    assert "return x" in extract_code(text)  # type: ignore[operator]


def test_extract_code_none_when_absent() -> None:
    assert extract_code("I cannot solve this.") is None


def test_extract_final_answer() -> None:
    assert extract_final_answer("stuff <final_answer>42.0</final_answer>") == "42.0"
    assert extract_final_answer("no tags") is None


def test_format_instruction_sft() -> None:
    result = format_instruction(
        {
            "latex_expression": r"\alpha + \beta",
            "solution": "import sympy as sp\nx = 1",
        },
        format="sft",
    )
    assert r"\alpha + \beta" in result["text"]
    assert "### Code:" in result["text"]


def test_format_instruction_legacy_key() -> None:
    """The historical python_code key must still work."""
    result = format_instruction(
        {"latex_expression": "x+1", "python_code": "def f(): return 1"}
    )
    assert "def f(): return 1" in result["text"]


def test_format_instruction_tir() -> None:
    result = format_instruction(
        {"latex_expression": "x+1", "solution": "def f(): return 1"}, format="tir"
    )
    assert "<think>" in result["text"]
    assert "<execute>" in result["text"]
    assert "<final_answer>" in result["text"]


def test_extract_variables_filters_constants() -> None:
    vars_ = extract_variables(r"\frac{x^{2} + y}{e^{x}}")
    assert "x" in vars_
    assert "e" not in vars_
