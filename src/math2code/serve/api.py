"""FastAPI inference server: vLLM proxy + sandboxed execution.

POST /generate  {latex_expression} -> {python_code, execution_result, error}
"""

from __future__ import annotations

import os

import uvicorn
from fastapi import FastAPI, HTTPException
from openai import AsyncOpenAI
from pydantic import BaseModel

from math2code.model.prompts import build_prompt, extract_code
from math2code.sandbox import SandboxPool

app = FastAPI(
    title="Math2Code API", description="LaTeX -> executable Python via vLLM + sandbox"
)

VLLM_API_BASE = os.environ.get("VLLM_API_BASE", "http://localhost:8080/v1")
VLLM_MODEL_NAME = os.environ.get("VLLM_MODEL_NAME", "AI-MO/NuminaMath-7B-TIR")

client = AsyncOpenAI(base_url=VLLM_API_BASE, api_key="vllm-proxy")
_sandbox: SandboxPool | None = None


class MathRequest(BaseModel):
    latex_expression: str


class MathResponse(BaseModel):
    python_code: str | None
    execution_result: str
    error: str | None = None


def get_sandbox() -> SandboxPool:
    global _sandbox
    if _sandbox is None:
        _sandbox = SandboxPool(n_workers=2)
    return _sandbox


@app.on_event("shutdown")
async def _shutdown() -> None:
    if _sandbox is not None:
        _sandbox.close()


@app.post("/generate", response_model=MathResponse)
async def generate_and_execute(req: MathRequest) -> MathResponse:
    prompt = build_prompt(req.latex_expression)

    # 1. Inference via vLLM (microservice)
    try:
        response = await client.completions.create(
            model=VLLM_MODEL_NAME, prompt=prompt, max_tokens=512, temperature=0.1
        )
        generated_text: str = response.choices[0].text
    except Exception as exc:
        raise HTTPException(
            status_code=502, detail=f"vLLM backend error: {exc}"
        ) from exc

    code = extract_code(generated_text)

    # 2. Execution in the local sandbox
    try:
        # derive inputs: if the model printed a value directly, still run the fn
        res = get_sandbox().execute(code or "", inputs={})
        if res.safety_error:
            return MathResponse(
                python_code=code,
                execution_result="",
                error=f"safety: {res.safety_error}",
            )
        result_text = (
            res.stdout
            if res.ok
            else res.stderr or ("timeout" if res.timed_out else f"exit {res.exit_code}")
        )
        return MathResponse(
            python_code=code,
            execution_result=result_text or "Executed successfully.",
            error=None if res.ok else result_text,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"sandbox error: {exc}") from exc


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
