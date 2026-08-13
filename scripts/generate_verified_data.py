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

Parallelism (--workers N, default 1):
  - workers=1 is byte-identical to the previous single-process behavior: one
    SandboxPool is reused for BOTH generation (truth_code expected outputs in
    custom_code families) and oracle verification.
  - workers>1 shards generation across N processes. With multiple families,
    one worker runs each family with the same seed — output is byte-identical
    to workers=1. With a single family, the count is split into contiguous
    shards with per-shard derived seeds (equivalent to changing the seed;
    use workers=1 to reproduce the committed v1 pools exactly).
  - the oracle-verification pool always runs with N workers; verification is
    concurrent (see SandboxPool.run_many), so N also scales the accept step.

The frozen split files are NEVER touched; training consumes this file as part
of a mixture (see docs/DATA_STRATEGY.md §8.3).
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from math2code.data.synthesizer import FAMILIES  # noqa: E402
from math2code.data.synthesizer.core import int_seed  # noqa: E402
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


def _shard_ranges(count: int, workers: int) -> list[tuple[int, int]]:
    """Split `count` into contiguous chunks: (start_index, shard_size)."""
    base, extra = divmod(count, workers)
    out: list[tuple[int, int]] = []
    pos = 0
    for w in range(workers):
        size = base + (1 if w < extra else 0)
        if size:
            out.append((pos, size))
        pos += size
    return out


def _gen_one(payload: dict[str, Any]) -> tuple[str, list[MathCodePair]]:
    """Worker entrypoint for parallel generation: one (family, seed, count).

    A small private SandboxPool is created lazily here because truth_code
    families (gcd/lcm/ntheory/...) compute committed expected outputs by
    executing truth code — a real pool is ~10x cheaper per case than the
    standalone subprocess fallback. The pool cannot travel in `payload`
    (ProcessPoolExecutor pickling), so workers build their own.
    """
    fam_name = str(payload["family"])
    seed = int(payload["seed"])
    count = int(payload["count"])
    opts = payload["opts"]
    pool: SandboxPool | None = None
    if payload.get("use_pool"):
        pool = SandboxPool(n_workers=2, timeout_s=10, memory_mb=2048)
    try:
        rows = FAMILIES[fam_name]().generate(  # type: ignore[abstract]
            seed, prefix=fam_name, count=count, pool=pool, **opts
        )
    finally:
        if pool is not None:
            pool.close()
    return fam_name, rows


def _filter_and_verify(
    raw: list[MathCodePair],
    frozen: set[str],
    pool: SandboxPool | None,
    no_verify: bool,
) -> tuple[list[MathCodePair], VerifyOutcome]:
    """Contamination guard + oracle verification (shared by both modes)."""
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

    if no_verify:
        return clean, VerifyOutcome(kept=clean)
    assert pool is not None, "verify requires a sandbox pool"
    outcome = verify_generated(clean, pool, n_points=20)
    print(f"oracle verify: kept {len(outcome.kept)} / {len(clean)}")
    return clean, outcome


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--families", default="derivative,integration")
    ap.add_argument(
        "--kind", default=None, help="family-specific (e.g. indefinite/definite)"
    )
    ap.add_argument("--count", type=int, default=40, help="math objects per family")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-variants", type=int, default=2)
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "parallel generation processes (see module docstring for determinism "
            "semantics); the oracle-verification pool uses this many workers too"
        ),
    )
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

    workers = max(1, args.workers)
    frozen = load_frozen_latex()
    opts: dict[str, object] = {}
    if args.kind:
        opts["kind"] = args.kind
    opts["n_variants"] = args.n_variants

    raw: list[MathCodePair] = []
    outcome = VerifyOutcome()
    if workers == 1 or len(fam_names) == 1:
        # Sequential (byte-identical to historical behavior): one pool serves
        # generation (truth_code expected outputs) AND verification.
        pool_workers = 4 if workers == 1 else workers
        with SandboxPool(n_workers=pool_workers, timeout_s=10, memory_mb=2048) as pool:
            for fam in fam_names:
                rows = FAMILIES[fam]().generate(  # type: ignore[abstract]
                    args.seed, prefix=fam, count=args.count, pool=pool, **opts
                )
                raw.extend(rows)
                print(f"{fam}: generated {len(rows)} variant rows")
            clean, outcome = _filter_and_verify(raw, frozen, pool, args.no_verify)
    else:
        if len(fam_names) > 1:
            # one worker per family, same seed -> byte-identical to workers=1
            payloads = [
                {
                    "family": fam,
                    "seed": args.seed,
                    "count": args.count,
                    "opts": opts,
                    "use_pool": True,
                }
                for fam in fam_names
            ]
        else:
            # single family: count-shard with per-shard derived seeds
            fam = fam_names[0]
            payloads = [
                {
                    "family": fam,
                    "seed": int_seed(f"{args.seed}:shard:{w}"),
                    "count": size,
                    "opts": opts,
                    "use_pool": True,
                }
                for w, (_start, size) in enumerate(_shard_ranges(args.count, workers))
            ]
        with cf.ProcessPoolExecutor(max_workers=min(workers, len(payloads))) as ex:
            for fam, rows in ex.map(_gen_one, payloads):
                raw.extend(rows)
                print(f"{fam}: generated {len(rows)} variant rows")
        if args.no_verify:
            clean, outcome = _filter_and_verify(raw, frozen, None, True)
        else:
            with SandboxPool(n_workers=workers, timeout_s=10, memory_mb=2048) as pool:
                clean, outcome = _filter_and_verify(raw, frozen, pool, False)

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
        "workers": workers,
        "generated": len(raw),
        "contamination_dropped": len(raw) - len(clean),
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
