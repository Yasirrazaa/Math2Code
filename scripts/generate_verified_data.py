"""Generate + verify synthetic training rows (code-first synthesizer).

Usage:
    python scripts/generate_verified_data.py \
        --families derivative,integration \
        --kind indefinite --count 40 --seed 42 \
        --out data/synthetic/calculus_v1.jsonl

Pipeline: family generators (SymPy AST -> ground truth -> notation variants
-> domain-aware test cases) -> contamination guard (latex collision with the
frozen test/val splits) -> oracle verification (sandbox execution on fresh
inputs) -> accepted/rejected + manifest + report.

The frozen split files are NEVER touched; training consumes this file as part
of a mixture (see docs/DATA_STRATEGY.md §8.3).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from math2code.data.synthesizer import FAMILIES  # noqa: E402
from math2code.data.synthesizer.verify import (  # noqa: E402
    VerifyOutcome,
    verify_generated,
)
from math2code.sandbox import SandboxPool  # noqa: E402
from math2code.schemas import MathCodePair  # noqa: E402


def load_frozen_latex() -> set[str]:
    """LaTeX expressions of the frozen test + val splits (contamination guard)."""
    frozen: set[str] = set()
    for name in ("test.json", "val.json"):
        path = Path("data/split") / name
        if not path.exists():
            continue
        rows = json.loads(path.read_text())
        frozen.update(r.get("latex_expression", "") for r in rows)
    return frozen


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--families", default="derivative,integration")
    ap.add_argument(
        "--kind", default=None, help="family-specific (e.g. indefinite/definite)"
    )
    ap.add_argument("--count", type=int, default=40, help="math objects per family")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-variants", type=int, default=2)
    ap.add_argument("--out", default="data/synthetic/calculus_v1.jsonl")
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()

    fam_names = [f.strip() for f in args.families.split(",") if f.strip()]
    unknown = [f for f in fam_names if f not in FAMILIES]
    if unknown:
        print(
            f"unknown families: {unknown}; available: {sorted(FAMILIES)}",
            file=sys.stderr,
        )
        sys.exit(1)

    frozen = load_frozen_latex()
    opts: dict[str, object] = {}
    if args.kind:
        opts["kind"] = args.kind
    opts["n_variants"] = args.n_variants

    raw: list[MathCodePair] = []
    for fam in fam_names:
        rows = FAMILIES[fam]().generate(args.seed, prefix=fam, count=args.count, **opts)
        raw.extend(rows)
        print(f"{fam}: generated {len(rows)} variant rows")

    # contamination guard: latex collision with frozen test/val
    clean: list[MathCodePair] = []
    contaminated: list[dict] = []
    for r in raw:
        if r.latex_expression in frozen:
            contaminated.append(
                {"task_id": r.task_id, "reason": "latex collides with frozen split"}
            )
        else:
            clean.append(r)
    print(f"contamination guard: {len(contaminated)} dropped, {len(clean)} kept")

    outcome = VerifyOutcome()
    if args.no_verify:
        outcome = VerifyOutcome(kept=clean)
    else:
        with SandboxPool(n_workers=4, timeout_s=10, memory_mb=2048) as pool:
            outcome = verify_generated(clean, pool, n_points=20)
        print(f"oracle verify: kept {len(outcome.kept)} / {len(clean)}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in outcome.kept:
            f.write(json.dumps(r.model_dump()) + "\n")
    (out_path.with_suffix(".rejected.json")).write_text(
        json.dumps(outcome.rejected, indent=2)
    )
    report = {
        "seed": args.seed,
        "families": fam_names,
        "opts": opts,
        "generated": len(raw),
        "contamination_dropped": len(contaminated),
        "accepted": len(outcome.kept),
        "rejected": len(outcome.rejected),
        "reject_reasons": _tally(outcome.rejected),
        "sha256": sha256(out_path),
    }
    (out_path.with_suffix(".report.json")).write_text(json.dumps(report, indent=2))
    print(f"saved -> {out_path} ({len(outcome.kept)} rows)")
    print(f"report  -> {out_path.with_suffix('.report.json')}")


def _tally(rejected: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in rejected:
        key = r["reason"].split(":")[0].split(";")[0].strip()
        counts[key] = counts.get(key, 0) + 1
    return counts


if __name__ == "__main__":
    main()
