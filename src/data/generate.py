import asyncio
import json
import os

import instructor
from groq import AsyncGroq
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential


# Pydantic schema for structured output
class MathCodePair(BaseModel):
    latex_expression: str = Field(
        description="The mathematical expression in LaTeX format."
    )
    python_code: str = Field(
        description="The Python SymPy code that evaluates the LaTeX expression."
    )


class SyntheticDataBatch(BaseModel):
    items: list[MathCodePair]


# Initialize async Groq client with Instructor
# Note: Ensure GROQ_API_KEY is set in your environment variables.
client = instructor.from_groq(
    AsyncGroq(api_key=os.environ.get("GROQ_API_KEY", "dummy_key"))
)


@retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(5))
async def generate_math_batch(prompt: str) -> SyntheticDataBatch:
    """
    Generate a batch of synthetic LaTeX and Python code pairs.
    Uses tenacity for exponential backoff in case of rate limits.
    """
    response: SyntheticDataBatch = await client.chat.completions.create(
        model="llama3-70b-8192",
        response_model=SyntheticDataBatch,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert mathematician and Python programmer. "
                    "Generate diverse mathematical expressions in LaTeX and their "
                    "corresponding executable Python evaluation code using SymPy. "
                    "The Python code should define a function and evaluate the expression."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )
    return response


async def main() -> None:
    print("Generating synthetic data...")
    # Example prompts to cover different domains
    prompts = [
        "Generate 5 examples of algebraic equations.",
        "Generate 5 examples of calculus integration problems.",
        "Generate 5 examples of matrix operations.",
    ]

    tasks = [generate_math_batch(p) for p in prompts]
    results = await asyncio.gather(*tasks)

    all_data = []
    for batch in results:
        for item in batch.items:
            all_data.append(item.model_dump())

    # Save raw generated data
    os.makedirs("data/raw", exist_ok=True)
    with open("data/raw/synthetic_raw.json", "w") as f:
        json.dump(all_data, f, indent=2)

    print(
        f"Generated {len(all_data)} raw samples and saved to data/raw/synthetic_raw.json"
    )


if __name__ == "__main__":
    asyncio.run(main())
