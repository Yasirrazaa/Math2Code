"""GRPO training via TRL (rule-based rewards, sandboxed execution).

No custom rollout loop: we stand on TRL's `GRPOTrainer` with reward funcs
that execute the emitted code in the local sandbox against per-rollout
resampled inputs (see `model/rewards.py`). The `task_id` dataset column is
forwarded to the reward function by the trainer, so no prompt->id map is
needed.

Targets TRL 1.9.x (pinned in pyproject). Verified against 1.9.2:
GRPOConfig/GRPOTrainer kwargs and the reward-func kwargs contract
(dataset columns arrive as lists). The TRL `environment_factory` tool-calling
API was deliberately NOT used — base math models are not function-calling
models, and the sandbox already runs in the reward path.

Run (Week 5+; needs a GPU box):
    uv pip install -e ".[train]"
    python -m math2code.model.grpo

Heavy imports (torch/trl/vllm) are lazy so the package stays importable on
CPU machines and in CI.
"""

from __future__ import annotations

from typing import Any

from math2code.model.rewards import grade_trajectory
from math2code.sandbox import SandboxPool
from math2code.schemas import MathCodePair


def train() -> None:  # pragma: no cover - GPU stack
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base=None, config_path="../../../configs", config_name="grpo")
    def _run(cfg: DictConfig) -> None:
        _train_grpo(cfg)

    _run()


def make_reward_fn(pair_by_id: dict[str, MathCodePair], pool: SandboxPool) -> Any:
    """Reward func matching TRL 1.9.x's calling convention.

    TRL invokes sync reward funcs as
    ``fn(prompts=..., completions=..., completion_ids=..., **dataset_cols)``
    — dataset columns (here ``task_id``) arrive as lists via kwargs. This
    factory is module-level so the contract is unit-testable on CPU.
    """

    def reward_fn(
        prompts: list[str],
        completions: list[str],
        task_id: list[str],
        **kwargs: Any,
    ) -> list[float]:
        del kwargs  # completion_ids / trainer_state / log_metric etc.
        rewards: list[float] = []
        for completion, tid in zip(completions, task_id):
            pair = pair_by_id.get(tid)
            if pair is None:
                rewards.append(0.0)
                continue
            rewards.append(grade_trajectory(pair, completion, pool).total)
        return rewards

    return reward_fn


def _train_grpo(
    cfg: Any,
) -> None:  # pragma: no cover - GPU stack
    """Build the dataset, reward func, and launch TRL GRPO."""
    import json
    from pathlib import Path

    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    from math2code.model.prompts import build_prompt
    from math2code.sandbox import SandboxPool
    from math2code.schemas import MathCodePair

    # 1. prompt + task_id dataset (task_id is forwarded to reward kwargs)
    rows = json.loads(Path(cfg.data.train_file).read_text())
    pairs = [MathCodePair.model_validate(r) for r in rows[: cfg.data.max_prompts]]
    prompts = [build_prompt(p.latex_expression) for p in pairs]
    dataset = Dataset.from_dict(
        {"prompt": prompts, "task_id": [p.task_id for p in pairs]}
    )
    pair_by_id = {p.task_id: p for p in pairs}
    print(f"GRPO prompts: {len(dataset)} (model {cfg.model.base_model})")

    # 2. sandbox pool shared by the reward (2s / 512MB = RL profile)
    pool = SandboxPool(
        n_workers=cfg.environment.n_workers, timeout_s=cfg.environment.timeout_s
    )

    # 3. reward: grade each completion against resampled inputs (dense signal)
    reward_fn = make_reward_fn(pair_by_id, pool)

    # 4. TRL config: batch-scaled rewards, no vLLM on the burn-in run
    args = GRPOConfig(
        output_dir=cfg.training.output_dir,
        run_name=cfg.training.run_name,
        num_generations=cfg.training.num_generations,
        temperature=cfg.training.temperature,
        max_completion_length=cfg.training.max_completion_length,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        learning_rate=cfg.training.learning_rate,
        bf16=cfg.training.bf16,
        scale_rewards="batch",
        use_vllm=cfg.training.use_vllm,
        log_completions=True,
        logging_steps=cfg.training.logging_steps,
        save_steps=cfg.training.save_steps,
        save_total_limit=cfg.training.save_total_limit,
        report_to=list(cfg.training.report_to),
    )

    trainer = GRPOTrainer(
        model=cfg.model.base_model,
        args=args,
        train_dataset=dataset,
        reward_funcs=[reward_fn],
    )
    trainer.train()
    trainer.save_model(f"{cfg.training.output_dir}/final")
    pool.close()
    print(f"GRPO complete -> {cfg.training.output_dir}/final")


if __name__ == "__main__":
    train()
