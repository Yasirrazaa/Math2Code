import pytest
from pydantic import ValidationError
from src.data.generate import MathCodePair

def test_math_code_pair_validation_success():
    pair = MathCodePair(
        latex_expression="\\frac{x}{y}",
        python_code="def evaluate(x, y): return x / y"
    )
    assert pair.latex_expression == "\\frac{x}{y}"
    assert pair.python_code == "def evaluate(x, y): return x / y"

def test_math_code_pair_missing_fields():
    with pytest.raises(ValidationError):
        MathCodePair(latex_expression="\\frac{x}{y}")
