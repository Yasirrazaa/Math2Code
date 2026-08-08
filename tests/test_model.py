from src.model.train import format_instruction

def test_format_instruction():
    sample = {
        "latex_expression": "\\alpha + \\beta",
        "python_code": "import sympy as sp\nalpha, beta = sp.symbols('alpha beta')\nexpr = alpha + beta"
    }
    
    result = format_instruction(sample)
    
    assert "Translate the following LaTeX expression to SymPy code:" in result["text"]
    assert "\\alpha + \\beta" in result["text"]
    assert "### Code:" in result["text"]
    assert "expr = alpha + beta" in result["text"]
