"""Reward function tests: deterministic, sandbox-backed, anti-memorization."""

from __future__ import annotations

from math2code.model.rewards import (
    EXEC_ERR_REWARD,
    EXEC_OK_REWARD,
    TERMINAL_REWARD,
    TOOL_ATTEMPT_REWARD,
    grade_trajectory,
)
from math2code.sandbox import SandboxPool
from math2code.schemas import MathCodePair, TestCase


def _pair(coeff: int = 2) -> MathCodePair:
    return MathCodePair(
        task_id="rw1",
        latex_expression=f"{coeff} \\cdot x",
        sympy_exp=f"{coeff}*x",
        solution=f"def calculate(x):\n    return {coeff} * x",
        test_cases=[
            TestCase(input={"x": 1.0}, output=float(coeff)),
            TestCase(input={"x": 2.0}, output=float(2 * coeff)),
            TestCase(input={"x": 3.0}, output=float(3 * coeff)),
        ],
    )


def test_correct_trajectory_max_reward() -> None:
    pair = _pair(2)
    traj = (
        "<think>Use sympy.</think>\n"
        "<execute>\ndef calculate(x):\n    return 2 * x\n</execute>\n"
        "<observation>\nExecuted successfully.\n</observation>\n"
        "<final_answer>\nDone.\n</final_answer>"
    )
    with SandboxPool(n_workers=2) as pool:
        bundle = grade_trajectory(pair, traj, pool, cases=pair.test_cases)
    assert bundle.terminal == TERMINAL_REWARD
    assert bundle.exec_ == EXEC_OK_REWARD
    assert bundle.tool == TOOL_ATTEMPT_REWARD
    assert bundle.total > TERMINAL_REWARD  # dense reward on top of terminal


def test_wrong_code_no_terminal() -> None:
    pair = _pair(2)
    traj = "<execute>\ndef calculate(x):\n    return 999\n</execute>"
    with SandboxPool(n_workers=2) as pool:
        bundle = grade_trajectory(pair, traj, pool, cases=pair.test_cases)
    assert bundle.terminal == 0.0
    assert bundle.exec_ == EXEC_OK_REWARD  # code RAN (correctly), just wrong answer
    assert bundle.total < TERMINAL_REWARD


def test_crashing_code_negative_exec() -> None:
    pair = _pair(2)
    traj = "<execute>\ndef calculate(x):\n    raise ValueError('boom')\n</execute>"
    with SandboxPool(n_workers=2) as pool:
        bundle = grade_trajectory(pair, traj, pool, cases=pair.test_cases)
    assert bundle.exec_ == EXEC_ERR_REWARD
    assert bundle.terminal == 0.0


def test_no_tool_use_gets_nothing() -> None:
    pair = _pair(2)
    traj = "<think>I know the answer.</think>"
    with SandboxPool(n_workers=2) as pool:
        bundle = grade_trajectory(pair, traj, pool, cases=pair.test_cases)
    assert bundle.tool == 0.0
    assert bundle.exec_ == 0.0
    assert bundle.total == 0.0


def test_resampling_default_anti_memorization() -> None:
    """Default `cases=None` resamples inputs -> deterministic per seed."""
    pair = _pair(2)
    traj = "<execute>\ndef calculate(x):\n    return 2 * x\n</execute>"
    with SandboxPool(n_workers=2) as pool:
        a = grade_trajectory(pair, traj, pool, seed_offset=3)
        b = grade_trajectory(pair, traj, pool, seed_offset=3)
    assert a.total == b.total


def test_complexity_penalty_only_on_correct() -> None:
    pair = _pair(2)
    verbose = (
        "<execute>\ndef calculate(x):\n"
        "    y = x + x\n"
        "    z = y + 0\n"
        "    w = z * 1\n"
        "    return 2 * x + (0 * w)\n"
        "</execute>"
    )
    with SandboxPool(n_workers=2) as pool:
        bundle = grade_trajectory(pair, verbose, pool, cases=pair.test_cases)
    assert bundle.terminal == TERMINAL_REWARD
    assert bundle.complexity < 0.0
