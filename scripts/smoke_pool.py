"""Sandbox throughput smoke: 10,000 sandboxed executions.

Gate: wall time must stay under `--target-sec` (default 300s = 5 min) with zero
execution failures or worker losses. Exits non-zero on violation.

Run:  python scripts/smoke_pool.py [--n 10000] [--workers N] [--target-sec 300]
"""

from __future__ import annotations

import argparse
import time

from math2code.sandbox import SandboxPool

CODE = """\
import math

def calculate(x, y):
    return math.sqrt(x * x + y * y)
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10_000, help="executions")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--target-sec", type=float, default=300.0)
    args = ap.parse_args()

    t0 = time.time()
    failures = 0
    with SandboxPool(n_workers=args.workers) as pool:
        for i in range(args.n):
            inputs = {"x": float(i % 97), "y": float(i % 89)}
            res = pool.execute(CODE, inputs=inputs)
            if not res.ok:
                failures += 1
                print(f"  failure {i}: {res.stderr or res.safety_error}")
                if failures >= 5:
                    print("aborting: too many failures")
                    break
    dt = time.time() - t0
    rate = args.n / dt
    print(
        f"executed {args.n} snippets in {dt:.1f}s -> {rate:.1f} exec/s, {failures} failures"
    )

    ok = failures == 0 and dt <= args.target_sec
    print(
        f"gate: {'PASS' if ok else 'FAIL'} (target {args.target_sec:.0f}s, actual {dt:.1f}s)"
    )
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
