import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from src.serve.api import app

client = TestClient(app)

@patch("src.serve.api.client.completions.create")
@patch("src.serve.api.CodeInterpreter")
def test_generate_and_execute_success(mock_interpreter, mock_completions):
    # Mock vLLM response
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(text="```python\nprint('hello sympy')\n```")]
    mock_completions.return_value = mock_response

    # Mock E2B Sandbox execution
    mock_sandbox_instance = MagicMock()
    mock_interpreter.return_value.__enter__.return_value = mock_sandbox_instance
    
    mock_exec_result = MagicMock()
    mock_exec_result.error = None
    mock_log = MagicMock()
    mock_log.line = "hello sympy"
    mock_exec_result.logs.stdout = [mock_log]
    mock_sandbox_instance.notebook.exec_cell.return_value = mock_exec_result

    response = client.post(
        "/generate",
        json={"latex_expression": "\\frac{1}{2}"}
    )

    assert response.status_code == 200
    data = response.json()
    assert "print('hello sympy')" in data["python_code"]
    assert data["execution_result"] == "hello sympy"
    assert data["error"] is None

@patch("src.serve.api.client.completions.create")
def test_generate_vllm_error(mock_completions):
    # Mock vLLM throwing an error
    mock_completions.side_effect = Exception("vLLM is down")

    response = client.post(
        "/generate",
        json={"latex_expression": "\\frac{1}{2}"}
    )

    assert response.status_code == 502
    assert "Failed to communicate with vLLM" in response.json()["detail"]
