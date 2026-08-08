"""Structured synthetic data generation (Groq + Instructor).

Generates (LaTeX, Python-code) pairs across math domains with pydantic-validated
structured outputs. This is the *secondary* data source: every generated pair
must pass the oracle (see `curate.py` / `data/oracle.py`) before entering the
training pool. The primary source is the competition `train.json`.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import instructor
from groq import AsyncGroq
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

from math2code.schemas import MathCodePair

DOMAIN_PROMPTS = [
    "Generate 5 examples of algebraic equations with multiple variables.",
    "Generate 5 examples of calculus integration problems (definite and indefinite).",
    "Generate 5 examples of matrix operations with sympy.",
    "Generate 5 examples of trigonometric identities with complex numbers.",
    "Generate 5 examples of differential equations (ordinary, first/second order).",
    "Generate 5 examples of limits and series (summation) expressions.",
]


class SyntheticDataBatch(BaseModel):
    items: list[MathCodePair]


client = instructor.from_groq(
    AsyncGroq(api_key=os.environ.get("GROQ_API_KEY", "dummy_key"))
)


@retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(5))
async def generate_math_batch(prompt: str) -> SyntheticDataBatch:
    """Generate a batch of LaTeX/code pairs; tenacity for rate limits."""
    response: SyntheticDataBatch = await client.chat.completions.create(
        model="llama3-70b-8192",
        response_model=SyntheticDataBatch,  # type: ignore[arg-type]  # instructor stub quirk
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert mathematician and Python programmer. "
                    "Generate diverse mathematical expressions in LaTeX and their "
                    "corresponding executable Python evaluation code using SymPy. "
                    "The Python code should define a function and evaluate the expression. "
                    "Each item needs: task_id (unique string), latex_expression, and "
                    "solution (plain runnable python, no markdown fences)."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )
    return response


async def main() -> None:
    print("Generating synthetic data...")
    tasks = [generate_math_batch(p) for p in DOMAIN_PROMPTS]
    results = await asyncio.gather(*tasks)

    all_data: list[dict[str, Any]] = []
    for batch in results:
        for i, item in enumerate(batch.items):
            if not item.task_id:
                item.task_id = f"syn_{len(all_data)}_{i}"
            all_data.append(item.model_dump())

    out_dir = Path("data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "synthetic_raw.json", "w") as f:
        json.dump(all_data, f, indent=2)
    print(f"Generated {len(all_data)} raw samples -> data/raw/synthetic_raw.json")


if __name__ == "__main__":
    asyncio.run(main())
