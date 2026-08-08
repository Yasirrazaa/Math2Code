"""SFT warmup: teach the model the TIR format / LaTeX->code translation.

Heavy ML imports are lazy so that importing the package (and running tests/CI)
does not require torch. Run with:
    uv pip install -e ".[train]"
    python -m math2code.model.train
"""

from __future__ import annotations

from typing import Any


def format_instruction(sample: dict[str, Any], format: str = "sft") -> dict[str, Any]:
    """Format a dataset row into a training text.

    Compatible with both the canonical `solution` key (competition/train data)
    and the legacy `python_code` key. `format="tir"` wraps the trajectory in
    the TIR XML protocol used by GRPO.
    """
    solution = sample.get("solution") or sample.get("python_code") or ""
    latex = sample.get("latex_expression", "")
    if format == "tir":
        trajectory = (
            f"<think>\nTranslate the LaTeX expression to sympy code and evaluate it.\n"
            f"</think>\n<execute>\n{solution}\n</execute>\n"
            f"<observation>\nExecuted successfully.\n</observation>\n"
            f"<final_answer>\nDone.\n</final_answer>"
        )
        return {"text": f"<problem>\n{latex}\n</problem>\n{trajectory}"}
    return {
        "text": (
            f"Translate the following LaTeX expression to SymPy code:\n"
            f"{latex}\n\n### Code:\n{solution}"
        )
    }


def train() -> None:  # pragma: no cover - requires GPU stack
    """Entrypoint (Hydra config: configs/train.yaml)."""
    import hydra
    import torch
    from datasets import load_dataset
    from omegaconf import DictConfig
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from trl import SFTTrainer

    @hydra.main(version_base=None, config_path="../../../configs", config_name="train")
    def _run(cfg: DictConfig) -> None:
        # 1. Load + format dataset
        print(f"Loading dataset from {cfg.data.train_file}")
        dataset = load_dataset("json", data_files=cfg.data.train_file, split="train")
        dataset = dataset.map(
            lambda s: format_instruction(s, format=cfg.training.get("format", "sft"))
        )

        # 2. Model & tokenizer
        tokenizer = AutoTokenizer.from_pretrained(cfg.model.base_model)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model.base_model,
            device_map="auto",
            torch_dtype=torch.bfloat16 if cfg.training.bf16 else torch.float32,
            use_cache=False,
        )

        # 3. LoRA
        lora = LoraConfig(
            r=cfg.lora.r,
            lora_alpha=cfg.lora.lora_alpha,
            target_modules=list(cfg.lora.target_modules),
            lora_dropout=cfg.lora.lora_dropout,
            bias=cfg.lora.bias,
            task_type=cfg.lora.task_type,
        )

        # 4. Training args
        args = TrainingArguments(
            output_dir=cfg.training.output_dir,
            num_train_epochs=cfg.training.num_train_epochs,
            per_device_train_batch_size=cfg.training.per_device_train_batch_size,
            gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
            learning_rate=cfg.training.learning_rate,
            bf16=cfg.training.bf16,
            optim=cfg.training.optim,
            max_grad_norm=cfg.training.max_grad_norm,
            logging_steps=cfg.training.logging_steps,
            save_steps=cfg.training.save_steps,
            save_total_limit=cfg.training.save_total_limit,
            report_to=cfg.training.report_to,
            run_name=cfg.training.run_name,
        )

        # 5. SFT
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            peft_config=lora,
            dataset_text_field="text",
            max_seq_length=cfg.model.max_seq_length,
            tokenizer=tokenizer,
            args=args,
        )
        trainer.train()
        trainer.save_model(f"{cfg.training.output_dir}/final")
        print(f"Saved fine-tuned model to {cfg.training.output_dir}/final")

    _run()


if __name__ == "__main__":
    train()
