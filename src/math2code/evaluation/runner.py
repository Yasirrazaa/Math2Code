"""Benchmark runner: score a model on the frozen test split.

Backends (--model):
  hf:<model_id>    local transformers model (Week 6-8 trained models)
  api:<name>       openai-compatible API; credentials from env:
                     DEEPSEEK_API_KEY / DEEPSEEK_BASE_URL  -> deepseek-chat
                     OPENAI_API_KEY   / OPENAI_BASE_URL    -> gpt-4o-mini
                     API_KEY / API_BASE_URL / API_MODEL    -> generic

Pipeline per problem: build_prompt(latex) -> completion -> extract_code ->
sandbox execution on the 5 test cases -> outputs. Then the competition metric
+ bootstrap 95% CI. Also writes a submission-format CSV (id,outputs).

Fails cleanly when the backend is unavailable (no GPU / no API key).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from math2code.evaluation.eval import load_split
from math2code.evaluation.metrics import ScoreResult, bootstrap_ci, score_predictions
from math2code.model.prompts import build_prompt, extract_code
from math2code.sandbox import SandboxPool
from math2code.schemas import MathCodePair

RESULTS_DIR = Path("results")


class BackendError(RuntimeError):
    pass


class HFBackend:
    """Local transformers model (requires GPU for anything useful)."""

    def __init__(self, model_id: str, max_tokens: int = 1024) -> None:
        try:
            import torch  # noqa: F401
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise BackendError(
                "transformers/torch not installed; run `uv pip install -e '.[train]'`"
            ) from exc
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="auto", torch_dtype=torch.bfloat16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.max_tokens = max_tokens

    def complete(self, prompt: str) -> str:

        ids = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        out = self.model.generate(
            **ids,
            max_new_tokens=self.max_tokens,
            do_sample=False,  # greedy = pass@1
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        text = self.tokenizer.decode(
            out[0][ids["input_ids"].shape[1] :], skip_special_tokens=True
        )
        return str(text)


class APIBackend:
    """OpenAI-compatible API backend."""

    def __init__(self, name: str, max_tokens: int = 1024) -> None:
        env = os.environ
        key: str | None = None
        base: str | None = None
        model: str | None = None
        if name == "deepseek":
            key = env.get("DEEPSEEK_API_KEY")
            base = env.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
            model = "deepseek-chat"
        elif name == "openai":
            key = env.get("OPENAI_API_KEY")
            base = env.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
            model = "gpt-4o-mini"
        else:
            key = env.get("API_KEY")
            base = env.get("API_BASE_URL")
            model = env.get("API_MODEL")
        if not key or not base or not model:
            raise BackendError(
                f"api:{name} needs API_KEY/API_BASE_URL/API_MODEL (or DEEPSEEK_API_KEY / OPENAI_API_KEY)"
            )
        from openai import OpenAI

        self.client = OpenAI(base_url=base, api_key=key)
        self.model = model
        self.max_tokens = max_tokens

    def complete(self, prompt: str) -> str:
        resp = self.client.completions.create(
            model=self.model, prompt=prompt, max_tokens=self.max_tokens, temperature=0.0
        )
        return str(resp.choices[0].text)


def make_backend(spec: str) -> Any:
    if spec.startswith("hf:"):
        return HFBackend(spec[3:])
    if spec.startswith("api:"):
        return APIBackend(spec[4:])
    raise BackendError(f"unknown backend spec: {spec!r} (use hf:<id> or api:<name>)")


def _run_problem(
    pair: MathCodePair, backend: Any, pool: SandboxPool
) -> list[str | None]:
    """One problem: prompt -> generate -> extract -> execute -> outputs."""
    prompt = build_prompt(pair.latex_expression)
    try:
        raw = backend.complete(prompt)
    except Exception as exc:  # transient API/GPU failure -> count as all-fail
        print(f"  [warn] {pair.task_id}: generation failed ({exc})")
        return [None] * len(pair.test_cases)
    code = extract_code(raw)
    if code is None:
        return [None] * len(pair.test_cases)
    outputs, _ = pool.run_solution_on_cases(code, [tc.input for tc in pair.test_cases])
    return outputs


def benchmark_model(
    split: str,
    model: str,
    out: str,
    max_problems: int | None = None,
    backend: Any | None = None,
    results_dir: Path = RESULTS_DIR,
) -> ScoreResult:
    pairs = load_split(split)
    if max_problems:
        pairs = pairs[:max_problems]

    backend = backend or make_backend(model)

    with SandboxPool(n_workers=6, timeout_s=20, memory_mb=4096) as pool:
        t0 = time.time()
        predictions: list[list[Any]] = []
        ids: list[str] = []
        for i, p in enumerate(pairs):
            preds = _run_problem(p, backend, pool)
            predictions.append(preds)  # type: ignore[arg-type]
            ids.append(p.task_id)
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(pairs)} problems ({time.time() - t0:.0f}s)")
        cases = [p.test_cases for p in pairs]  # type: ignore[arg-type]
        result = score_predictions(predictions, cases, ids)

    # persist submission-format CSV
    results_dir.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    csv_path = results_dir / f"{model.replace(':', '_')}_{stamp}.csv"
    import csv

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "outputs"])
        for tid, preds in zip(ids, predictions):
            writer.writerow([tid, json.dumps(preds)])

    passed = [all(d.passed for d in result.details if d.task_id == tid) for tid in ids]
    lo, hi = bootstrap_ci(passed)
    report = {
        "model": model,
        "split": split,
        "timestamp": stamp,
        "n_problems": result.n_problems,
        "n_correct_problems": result.n_correct_problems,
        "per_problem_accuracy": result.per_problem_accuracy,
        "ci95": [lo, hi],
        "per_case_accuracy": result.per_case_accuracy,
        "predictions_file": str(csv_path),
    }
    (results_dir / f"{model.replace(':', '_')}_{stamp}.json").write_text(
        json.dumps(report, indent=2)
    )
    print(f"  saved -> {csv_path}")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark a model on the frozen split")
    ap.add_argument("--split", default="data/split/test.json")
    ap.add_argument("--model", required=True, help="hf:<id> or api:<name>")
    ap.add_argument("--out", default="results/latest.json")
    ap.add_argument("--max-problems", type=int, default=None)
    args = ap.parse_args()

    try:
        result = benchmark_model(args.split, args.model, args.out, args.max_problems)
    except BackendError as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(2)
    print(result.summary())


if __name__ == "__main__":
    main()
