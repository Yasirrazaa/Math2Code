"""Curation: validate generated samples through the sandbox + oracle.

Every candidate (LLM-generated or AST-generated) must execute in the sandbox
and, when expected outputs are available, match them numerically. Only samples
that pass are written to the processed pool.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from tqdm import tqdm

from math2code.evaluation.metrics import outputs_match
from math2code.sandbox import SandboxPool
from math2code.schemas import MathCodePair


def load_raw_data(filepath: str = "data/raw/synthetic_raw.json") -> list[MathCodePair]:
    path = Path(filepath)
    if not path.exists():
        print(f"Error: {filepath} not found.", file=sys.stderr)
        return []
    with open(path) as f:
        return [MathCodePair.model_validate(item) for item in json.load(f)]


def curate(
    data: list[MathCodePair], pool: SandboxPool
) -> tuple[list[MathCodePair], list[dict]]:
    """Keep samples whose solution runs without error on every test case and,
    when expected outputs exist, matches them within tolerance."""
    kept: list[MathCodePair] = []
    dropped: list[dict] = []
    for item in tqdm(data, desc="curating"):
        code = item.solution or ""
        if not code.strip():
            dropped.append({"task_id": item.task_id, "reason": "empty solution"})
            continue
        cases = item.test_cases
        if not cases:
            # no test cases -> at least require the code to compile & define a function
            res = pool.execute(code, inputs={})
            if res.ok:
                kept.append(item)
            else:
                dropped.append(
                    {
                        "task_id": item.task_id,
                        "reason": res.safety_error or res.stderr or "exec error",
                    }
                )
            continue
        outputs, errors = pool.run_solution_on_cases(code, [tc.input for tc in cases])
        if any(o is None for o in outputs):
            dropped.append({"task_id": item.task_id, "reason": "; ".join(errors)[:200]})
            continue
        if all(tc.output is None for tc in cases):
            kept.append(item)
            continue
        ok = all(
            tc.output is None or outputs_match(got, tc.output)
            for got, tc in zip(outputs, cases)
        )
        if ok:
            kept.append(item)
        else:
            dropped.append({"task_id": item.task_id, "reason": "numeric mismatch"})
    return kept, dropped


def main() -> None:
    data = load_raw_data()
    if not data:
        return
    print(f"Loaded {len(data)} raw samples.")
    with SandboxPool() as pool:
        kept, dropped = curate(data, pool)
    print(f"Curation complete. Kept {len(kept)} / {len(data)}.")

    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "synthetic_curated.json", "w") as f:
        json.dump([k.model_dump() for k in kept], f, indent=2)
    with open(out_dir / "curation_report.json", "w") as f:
        json.dump({"kept": len(kept), "dropped": dropped}, f, indent=2)
    print(f"Saved -> data/processed/synthetic_curated.json ({len(kept)} rows)")


if __name__ == "__main__":
    main()
