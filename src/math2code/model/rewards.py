"""Deterministic RLVR rewards (rule-based; no LLM judges).

All rewards are computed from observable facts: sandbox execution results,
the presence of tool-use tokens, string markers, and code complexity. There is
no learned/LLM reward anywhere.

Design (per PLAN.md §Rewards):
  R_terminal  +2.0   Monte Carlo oracle: final code correct on all (resampled) cases
  R_exec      +1.0 / -0.25  per <execute> block that ran cleanly vs errored
  R_tool      +0.1   at least one tool-use (execute) attempt
  R_meta      +0.2   traceback-style token present (model started self-correcting)
  R_complexity -0.01 * max(0, ops(generated) - ops(ground_truth))  on correct answers

Test cases are RESAMPLED per rollout (seeded by task_id + step) so the policy
cannot memorize the 5 fixed competition inputs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from math2code.data.competition import resample_test_cases
from math2code.evaluation.metrics import outputs_match
from math2code.model.prompts import extract_final_answer
from math2code.sandbox import SandboxPool
from math2code.schemas import MathCodePair

TERMINAL_REWARD = 2.0
EXEC_OK_REWARD = 1.0
EXEC_ERR_REWARD = -0.25
TOOL_ATTEMPT_REWARD = 0.1
META_TRACEBACK_REWARD = 0.2
COMPLEXITY_PENALTY_PER_OP = 0.01

_EXEC_RE = re.compile(r"<execute>\s*(.*?)\s*</execute>", re.DOTALL)
_TRACEBACK_HINTS = (
    "Traceback",
    "Error",
    "TypeError",
    "ValueError",
    "ZeroDivisionError",
)


@dataclass
class RewardBundle:
    terminal: float
    exec_: float
    tool: float
    meta: float
    complexity: float
    total: float

    def as_dict(self) -> dict[str, float]:
        return {
            "terminal": self.terminal,
            "exec": self.exec_,
            "tool": self.tool,
            "meta": self.meta,
            "complexity": self.complexity,
            "total": self.total,
        }


def _code_ops_estimate(code: str) -> int:
    """Rough operator-count proxy for R_complexity (symbolic ops per line)."""
    return sum(line.count(op) for op in "+-*/^" for line in code.splitlines())


def grade_trajectory(
    pair: MathCodePair,
    trajectory: str,
    pool: SandboxPool,
    cases: list[Any] | None = None,
    seed_offset: int = 0,
) -> RewardBundle:
    """Score one full trajectory (think/execute/observe/final_answer).

    `cases` defaults to freshly resampled test cases (per-rollout inputs).
    """
    tc = (
        cases if cases is not None else resample_test_cases(pair, n=5, seed=seed_offset)
    )
    blocks = [m.group(1).strip() for m in _EXEC_RE.finditer(trajectory)]
    final = extract_final_answer(trajectory)

    # tool-use present?
    tool = TOOL_ATTEMPT_REWARD if blocks else 0.0

    # meta: traceback tokens in the trajectory => model observed an error
    meta = (
        META_TRACEBACK_REWARD if any(h in trajectory for h in _TRACEBACK_HINTS) else 0.0
    )

    exec_reward = 0.0
    terminal = 0.0
    last_outputs: list[str | None] = []

    for code in blocks:
        outputs, errors = pool.run_solution_on_cases(code, [tc.input for tc in tc])
        if errors:
            exec_reward += EXEC_ERR_REWARD
            last_outputs = [None] * len(tc)
        else:
            exec_reward += EXEC_OK_REWARD
            last_outputs = outputs  # type: ignore[assignment]

    if blocks:
        correct = all(
            t.output is not None and outputs_match(got, t.output)
            for got, t in zip(last_outputs, tc)
        )
        if correct:
            terminal = TERMINAL_REWARD
            # complexity penalty only on correct answers
            if pair.sympy_exp:
                gt_ops = _code_ops_estimate(pair.solution or "")
                cand_ops = _code_ops_estimate(blocks[-1])
                penalty = min(
                    5.0, max(0, cand_ops - gt_ops) * COMPLEXITY_PENALTY_PER_OP
                )
                complexity = -penalty
            else:
                complexity = 0.0
        else:
            complexity = 0.0
    else:
        # no tool use at all: pure-thought trajectories earn nothing
        complexity = 0.0
        if final is not None:
            # direct numeric answer without execution: allow terminal credit only
            # if it matches (very rare; keeps the model from gaming the penalty)
            try:
                if all(
                    t.output is not None and outputs_match(final, t.output) for t in tc
                ):
                    terminal = TERMINAL_REWARD
            except ValueError:
                pass

    total = terminal + exec_reward + tool + meta + complexity
    return RewardBundle(
        terminal=terminal,
        exec_=exec_reward,
        tool=tool,
        meta=meta,
        complexity=complexity,
        total=total,
    )


def reward_func(
    prompts: list[str],
    completions: list[str],
    pool: SandboxPool,
    pairs: list[MathCodePair],
    seed_offset: int = 0,
) -> list[float]:
    """TRL-style reward function: (prompt, completion) -> score (per rollout)."""
    rewards: list[float] = []
    for prompt, completion, pair in zip(prompts, completions, pairs):
        bundle = grade_trajectory(pair, completion, pool, seed_offset=seed_offset)
        rewards.append(bundle.total)
    return rewards
