"""Publish the frozen split + curated data to the Hugging Face Hub.

Requires `HF_TOKEN` (write scope). Idempotent: creates the repo if missing,
then pushes the files (overwrite on change).

Run:  HF_TOKEN=hf_... python scripts/publish_hf.py [--repo Yasirrazaa/math2code-data]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

DEFAULT_REPO = "Yasirrazaa/math2code-data"
SPLIT_DIR = Path(__file__).resolve().parents[1] / "data" / "split"


def build_readme(repo: str) -> str:
    manifest = json.loads((SPLIT_DIR / "manifest.json").read_text())
    return f"""---
license: mit
task_categories:
  - text-generation
language:
  - en
tags:
  - math
  - latex
  - code-generation
  - sympy
---

# Math2Code — competition dataset (frozen split)

LaTeX expression -> executable Python (SymPy) code pairs from the bootcamp
Math2Code competition `train.json` (26,846 rows), deduplicated by LaTeX
expression (~4k duplicates removed) and split deterministically (seed {manifest["seed"]}).

## Files

| file        | rows    | content                                   |
|-------------|---------|-------------------------------------------|
| train.jsonl | {manifest["n_train"]} | SFT/GRPO training rows (JSON lines) |
| val.json    | {manifest["n_val"]}  | held-out validation (with expected outputs) |
| test.json   | {manifest["n_test"]} | frozen test set (with expected outputs) |
| manifest.json | -     | seed, counts, per-file SHA-256            |

Every row is a `MathCodePair`:
`task_id, latex_expression, sympy_exp, solution, domain, equation_type,
complexity, output_type, test_cases[{{input, output}}]`.

## Integrity

`test_ids.txt` lists the frozen test ids; check for contamination before
training. The manifest carries SHA-256 digests so the split can be audited.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=os.environ.get("MATH2CODE_HF_REPO", DEFAULT_REPO))
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        print(
            "HF_TOKEN not set; export it first (write scope).",
            file=__import__("sys").stderr,
        )
        raise SystemExit(1)

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    if not api.repo_exists(args.repo, repo_type="dataset"):
        print(f"creating dataset repo {args.repo}")
        api.create_repo(args.repo, repo_type="dataset", private=False)

    files = ["train.jsonl", "val.json", "test.json", "manifest.json", "test_ids.txt"]
    for name in files:
        p = SPLIT_DIR / name
        if not p.exists():
            print(
                f"  missing {p} — run scripts/make_splits.py first",
                file=__import__("sys").stderr,
            )
            continue
        api.upload_file(
            path_or_fileobj=str(p),
            path_in_repo=name,
            repo_id=args.repo,
            repo_type="dataset",
        )
        print(f"  uploaded {name} ({p.stat().st_size / 1e6:.1f} MB)")

    # README card last so it reflects the final manifest
    api.upload_file(
        path_or_fileobj=build_readme(args.repo).encode(),
        path_in_repo="README.md",
        repo_id=args.repo,
        repo_type="dataset",
    )
    print(f"done -> https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
