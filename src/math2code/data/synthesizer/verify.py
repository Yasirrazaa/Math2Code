"""Accept/reject verification for generated rows.

Every code-first row must pass the oracle before entering the training pool:
syntax + sandbox execution on **fresh** jittered inputs vs the ground-truth
AST (`numeric_check` with `n` points). Family-specific symbolic gates were
already applied at build time (`family_gate` in metadata) — e.g.
differentiate-back for integrals — so this runner is the second, independent
layer (execution-level proof), exactly as the strategy's verification spine
requires.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from math2code.data.oracle import oracle_verify
from math2code.sandbox import SandboxPool
from math2code.schemas import MathCodePair


@dataclass
class VerifyOutcome:
    kept: list[MathCodePair] = field(default_factory=list)
    rejected: list[dict] = field(default_factory=list)


def verify_generated(
    rows: list[MathCodePair],
    pool: SandboxPool,
    n_points: int = 20,
) -> VerifyOutcome:
    """Run the full oracle (syntax + fresh-input sandbox execution) on each row.

    Rows without executable solutions are rejected up front. Rejection reasons
    are truncated for the report.
    """
    outcome = VerifyOutcome()
    for item in rows:
        code = item.solution or ""
        if not code.strip():
            outcome.rejected.append(
                {"task_id": item.task_id, "reason": "empty solution"}
            )
            continue
        ok, reasons = oracle_verify(item, code, pool=pool)
        if ok:
            outcome.kept.append(item)
        else:
            outcome.rejected.append(
                {"task_id": item.task_id, "reason": "; ".join(reasons)[:200]}
            )
    return outcome
