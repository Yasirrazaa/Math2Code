"""GRPO training via TRL (rule-based rewards, sandboxed environment).

No custom rollout loop: we stand on TRL's `GRPOTrainer` with an
`environment_factory` whose `step(action)` executes the emitted code in the
local sandbox and returns an observation. Rewards come from `rewards.py`.

Run (Week 5+; needs a GPU box):
    uv pip install -e ".[train]"
    python -m math2code.model.grpo

Heavy imports (torch/trl/vllm) are lazy so the package stays importable on
CPU machines and in CI.
"""

from __future__ import annotations

from typing import Any


def train() -> None:  # pragma: no cover - GPU stack
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base=None, config_path="../../../configs", config_name="grpo")
    def _run(cfg: DictConfig) -> None:
        _train_grpo(cfg)

    _run()


def _train_grpo(
    cfg: Any,
) -> None:  # pragma: no cover - GPU stack
    """Build the dataset, environment factory, reward funcs, and launch TRL."""
    import json
    from pathlib import Path

    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    from math2code.model.prompts import build_prompt
    from math2code.model.rewards import grade_trajectory
    from math2code.sandbox import SandboxPool
    from math2code.schemas import MathCodePair

    # 1. prompt-only dataset (inputs/test-cases are injected by the env)
    rows = json.loads(Path(cfg.data.train_file).read_text())
    pairs = [MathCodePair.model_validate(r) for r in rows[: cfg.data.max_prompts]]
    prompts = [build_prompt(p.latex_expression) for p in pairs]
    dataset = Dataset.from_dict(
        {"prompt": prompts, "task_id": [p.task_id for p in pairs]}
    )
    prompt_to_task = dict(zip(prompts, [p.task_id for p in pairs]))
    print(f"GRPO prompts: {len(dataset)} (model {cfg.model.base_model})")

    # 2. sandbox pool shared by the environment (2s / 512MB = RL profile)
    pool = SandboxPool(
        n_workers=cfg.environment.n_workers, timeout_s=cfg.environment.timeout_s
    )

    # 3. environment factory: step(code) executes in the sandbox
    def environment_factory(question: str) -> Any:
        from trl.extras.environment import BaseEnvironment, Observation

        class MathEnv(BaseEnvironment):
            def __init__(self) -> None:
                super().__init__()
                self.turns = 0

            def reset(self) -> Observation:
                self.turns = 0
                return Observation(text="")

            def step(self, action: str) -> Observation:
                self.turns += 1
                # observation tokens come back as model text; we execute the code
                res = pool.execute(action, inputs={})
                if res.safety_error:
                    obs = f"Safety error: {res.safety_error}"
                elif res.timed_out:
                    obs = "Execution timed out."
                elif res.ok:
                    obs = res.stdout or "Executed successfully."
                else:
                    obs = res.stderr or "Execution failed."
                return Observation(text=f"\n{obs}\n")

        return MathEnv()

    # 4. reward: grade the whole trajectory against resampled cases
    pair_by_id = {p.task_id: p for p in pairs}

    def make_reward(seed_offset: int) -> Any:
        def reward_fn(
            prompts_batch: list[str], completions_batch: list[str]
        ) -> list[float]:
            rewards = []
            for prompt, completion in zip(prompts_batch, completions_batch):
                tid: str | None = prompt_to_task.get(prompt)
                pair = pair_by_id.get(tid)  # type: ignore[arg-type]
                if pair is None:
                    rewards.append(0.0)
                    continue
                bundle = grade_trajectory(
                    pair, completion, pool, seed_offset=seed_offset
                )
                rewards.append(bundle.total)
            return rewards

        return reward_fn

    # 5. TRL config: batch-scaled rewards, no vLLM on the burn-in run
    args = GRPOConfig(
        output_dir=cfg.training.output_dir,
        run_name=cfg.training.run_name,
        num_generations=cfg.training.num_generations,
        temperature=cfg.training.temperature,
        max_completion_length=cfg.training.max_completion_length,
        max_prompt_length=cfg.training.max_prompt_length,
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
        # observation tokens must be excluded from loss/logprob computation
        observation_token="<observation>",
        evaluation_strategy="no",
    )

    trainer = GRPOTrainer(
        model=cfg.model.base_model,
        args=args,
        train_dataset=dataset,
        reward_funcs=[make_reward(seed_offset=0)],
        environment_factory=environment_factory,
    )
    trainer.train()
    trainer.save_model(f"{cfg.training.output_dir}/final")
    pool.close()
    print(f"GRPO complete -> {cfg.training.output_dir}/final")


if __name__ == "__main__":
    train()
