"""API tests (mocked vLLM + sandbox)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from math2code.serve.api import app

client = TestClient(app)


@patch("math2code.serve.api.get_sandbox")
@patch("math2code.serve.api.client")
def test_generate_and_execute_success(
    mock_client: MagicMock, mock_get_sandbox: MagicMock
) -> None:
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(text="```python\ndef calculate(x):\n    return x * 2\n```")
    ]
    mock_client.completions.create = AsyncMock(return_value=mock_response)

    sandbox = MagicMock()
    exec_result = MagicMock()
    exec_result.ok = True
    exec_result.stdout = "4.0"
    exec_result.stderr = ""
    exec_result.safety_error = None
    exec_result.timed_out = False
    exec_result.exit_code = 0
    sandbox.execute.return_value = exec_result
    mock_get_sandbox.return_value = sandbox

    response = client.post("/generate", json={"latex_expression": r"2 \cdot x"})
    assert response.status_code == 200
    data = response.json()
    assert "def calculate" in data["python_code"]
    assert data["execution_result"] == "4.0"
    assert data["error"] is None


@patch("math2code.serve.api.client")
def test_generate_vllm_error(mock_client: MagicMock) -> None:
    mock_client.completions.create = AsyncMock(side_effect=Exception("vLLM is down"))
    response = client.post("/generate", json={"latex_expression": r"x + 1"})
    assert response.status_code == 502
    assert "vLLM backend error" in response.json()["detail"]
