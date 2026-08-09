"""GRPO reward-contract tests: simulate TRL 1.9.x's exact reward invocation.

TRL calls sync reward funcs as
`fn(prompts=..., completions=..., completion_ids=..., **dataset_cols)` where
dataset columns arrive as lists. These tests exercise `make_reward_fn` with a
real SandboxPool — no GPU needed. Trajectories follow the TIR protocol the
SFT warmup teaches: `<think>` / `<execute>` / `<observation>` / `<final_answer>`.
"""

from __future__ import annotations

from math2code.model.grpo import make_reward_fn
from math2code.sandbox import SandboxPool
from math2code.schemas import MathCodePair, TestCase

GOLD_TRAJECTORY = """<think>Add one to x.</think>
<execute>
def calculate(x):
    return x + 1
</execute>
<observation>Executed successfully.</observation>
<final_answer>3.0</final_answer>"""

WRONG_TRAJECTORY = """<think>Return 999.</think>
<execute>
def calculate(x):
    return 999
</execute>
<observation>Executed successfully.</observation>
<final_answer>999</final_answer>"""


def _pair(tid: str = "a") -> MathCodePair:
    return MathCodePair(
        task_id=tid,
        latex_expression="x + 1",
        sympy_exp="x + 1",
        solution="def calculate(x):\n    return x + 1\n",
        test_cases=[
            TestCase(input={"x": 2.0}, output=3.0),
            TestCase(input={"x": 5.0}, output=6.0),
        ],
    )


def test_reward_fn_trl_call_contract() -> None:
    """The exact call shape TRL 1.9.x uses must return a float list."""
    pairs = [_pair("a"), _pair("b")]
    pair_by_id = {p.task_id: p for p in pairs}
    with SandboxPool(n_workers=2, timeout_s=10) as pool:
        reward_fn = make_reward_fn(pair_by_id, pool)
        rewards = reward_fn(
            prompts=["latex: x+1", "latex: x+1"],
            completions=[GOLD_TRAJECTORY, GOLD_TRAJECTORY],
            completion_ids=[[0, 1], [0, 1]],
            task_id=["a", "b"],
            trainer_state=None,  # extra kwargs must be tolerated
        )
    assert isinstance(rewards, list) and len(rewards) == 2
    assert all(isinstance(r, float) for r in rewards)
    # gold trajectory on resampled inputs must score strictly positive
    assert rewards[0] > 0.0


def test_reward_fn_unknown_task_id_scores_zero() -> None:
    pairs = [_pair("a")]
    with SandboxPool(n_workers=2, timeout_s=10) as pool:
        reward_fn = make_reward_fn({p.task_id: p for p in pairs}, pool)
        rewards = reward_fn(
            prompts=["latex: x+1"],
            completions=[GOLD_TRAJECTORY],
            completion_ids=[[0, 1]],
            task_id=["not-in-dataset"],
        )
    assert rewards == [0.0]


def test_reward_fn_garbage_completion_low() -> None:
    pairs = [_pair("a")]
    with SandboxPool(n_workers=2, timeout_s=10) as pool:
        reward_fn = make_reward_fn({p.task_id: p for p in pairs}, pool)
        good = reward_fn(prompts=["p"], completions=[GOLD_TRAJECTORY], task_id=["a"])[0]
        bad = reward_fn(prompts=["p"], completions=[WRONG_TRAJECTORY], task_id=["a"])[0]
    assert good > bad  # correct code out-scores wrong code
    # both used tool calls, but only the gold gets terminal credit
    assert good >= 2.0 and bad < 2.0
