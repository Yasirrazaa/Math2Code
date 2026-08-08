"""Prompt templates (single source of truth for train + eval + serve)."""

from __future__ import annotations

import re

# -- prompt used for zero-shot and fine-tuned inference -------------------------
ZERO_SHOT_PROMPT = """Convert the following LaTeX expression into a Python function named 'calculate' that takes the given parameters:
{latex}

Follow these guidelines strictly:
1. The code must contain a Python function definition using `def calculate({variables}):` for all equations including derivatives.
2. Import necessary libraries and functions at the beginning of the function. Prefer using `numpy` for mathematical operations and `sympy` for symbolic computations.
3. Use `sympy` exclusively to solve equations involving calculus operations like differentiation, integration, and logarithmic functions. Do not use `sp.lambdify`.
4. Ensure the function is callable with the provided parameters and returns the calculated result as `int`, `float`, or `complex number` with 'return'.

Now, apply the above instructions to convert the given LaTeX expression.

Python Code:
```python
"""

TIR_SYSTEM_PROMPT = (
    "You are a mathematical agent that converts LaTeX expressions into Python "
    "code using sympy. Think step by step, execute code, observe the output, and "
    "correct yourself if needed. End with the final answer in a <final_answer> tag."
)


def build_prompt(latex: str, variables: list[str] | None = None) -> str:
    """Build the inference prompt. Variables are optional (model may infer)."""
    var_str = ", ".join(variables) if variables else "the expression's variables"
    return ZERO_SHOT_PROMPT.format(latex=latex, variables=var_str)


# -- code extraction (used by eval, rewards, and serving) -----------------------
_FENCE_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)
_EXEC_RE = re.compile(r"<execute>\s*(.*?)\s*</execute>", re.DOTALL)


def extract_code(text: str) -> str | None:
    """Extract the first executable Python block from a model completion.

    Handles ```python fences, <execute> tags, and bare code starting with a
    `def`. Returns None when no plausible code is found.
    """
    if not text:
        return None
    m = _EXEC_RE.search(text)
    if m:
        return m.group(1).strip()
    m = _FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    # bare code: strip trailing prose after the final 'return' statement
    stripped = text.strip()
    if stripped.startswith("def ") or "def calculate" in stripped:
        lines = stripped.splitlines()
        out: list[str] = []
        for line in lines:
            out.append(line)
            if re.match(r"\s*return\s+", line):
                break
        return "\n".join(out)
    return None


def extract_final_answer(text: str) -> str | None:
    m = re.search(r"<final_answer>\s*(.*?)\s*</final_answer>", text, re.DOTALL)
    return m.group(1) if m else None


def extract_variables(latex: str) -> list[str]:
    """Best-effort variable extraction from a LaTeX expression.

    Prefer test-case input keys when available; this is only a fallback.
    Filters math constants (d=derivative operator, e=Euler, i=imaginary unit).
    """
    found = sorted(set(re.findall(r"\b([a-zA-Z])\b", latex.replace("\\", ""))))
    banned = {"d", "e", "i"}
    return [v for v in found if v not in banned] or ["x"]
