import os

import uvicorn
from e2b_code_interpreter import CodeInterpreter
from fastapi import FastAPI, HTTPException
from openai import AsyncOpenAI
from pydantic import BaseModel

app = FastAPI(
    title="Math2Code API", description="FastAPI Proxy mapping requests to vLLM and E2B"
)

# Connect to the standalone vLLM server
VLLM_API_BASE = os.environ.get("VLLM_API_BASE", "http://localhost:8080/v1")
# We default to the fine-tuned adapter name if specified, otherwise the base model
VLLM_MODEL_NAME = os.environ.get("VLLM_MODEL_NAME", "AI-MO/NuminaMath-7B-TIR")

client = AsyncOpenAI(
    base_url=VLLM_API_BASE,
    api_key="vllm-proxy",  # vLLM doesn't require a real key by default
)


class MathRequest(BaseModel):
    latex_expression: str


class MathResponse(BaseModel):
    python_code: str
    execution_result: str
    error: str | None = None


@app.post("/generate", response_model=MathResponse)
async def generate_and_execute(req: MathRequest) -> MathResponse:
    prompt = f"Translate the following LaTeX expression to SymPy code:\n{req.latex_expression}\n\n### Code:\n"

    # 1. Inference via vLLM (Microservice)
    try:
        response = await client.completions.create(
            model=VLLM_MODEL_NAME, prompt=prompt, max_tokens=256, temperature=0.1
        )
        generated_text = response.choices[0].text
        cleaned_code = generated_text.strip().strip("`").replace("python\n", "")
    except Exception as e:
        raise HTTPException(
            status_code=502, detail=f"Failed to communicate with vLLM Backend: {str(e)}"
        )

    # 2. Execution via E2B Code Interpreter
    try:
        with CodeInterpreter() as sandbox:
            sandbox.notebook.exec_cell("!pip install sympy")
            exec_result = sandbox.notebook.exec_cell(cleaned_code)

            result_text = (
                "\n".join([log.line for log in exec_result.logs.stdout])
                if exec_result.logs.stdout
                else "Executed successfully."
            )
            error_text = exec_result.error.value if exec_result.error else None

        return MathResponse(
            python_code=cleaned_code, execution_result=result_text, error=error_text
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"E2B Sandbox Execution Error: {str(e)}"
        )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
