import hydra
import torch
from datasets import load_dataset
from omegaconf import DictConfig
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer


def format_instruction(sample: dict) -> dict:
    """
    Format the dataset row into a prompt instruction for the model.
    """
    prompt = f"Translate the following LaTeX expression to SymPy code:\n{sample['latex_expression']}\n\n### Code:\n{sample['python_code']}"
    return {"text": prompt}


@hydra.main(version_base=None, config_path="../../configs", config_name="train")
def train(cfg: DictConfig) -> None:
    # 1. Load Dataset
    print(f"Loading dataset from {cfg.data.train_file}")
    dataset = load_dataset("json", data_files=cfg.data.train_file, split="train")
    dataset = dataset.map(format_instruction)

    # 2. Setup Model & Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.base_model)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.base_model,
        device_map="auto",
        torch_dtype=torch.bfloat16 if cfg.training.bf16 else torch.float32,
    )

    # 3. Setup LoRA
    lora_config = LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.lora_alpha,
        target_modules=list(cfg.lora.target_modules),
        lora_dropout=cfg.lora.lora_dropout,
        bias=cfg.lora.bias,
        task_type=cfg.lora.task_type,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 4. Training Arguments (including WandB)
    training_args = TrainingArguments(
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
        run_name="math2code_lora_finetune",  # W&B Run Name
    )

    # 5. Initialize SFTTrainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=lora_config,
        dataset_text_field="text",
        max_seq_length=cfg.model.max_seq_length,
        tokenizer=tokenizer,
        args=training_args,
    )

    # 6. Train!
    print("Starting Training...")
    trainer.train()

    # 7. Save the model
    print(f"Saving fine-tuned model to {cfg.training.output_dir}/final")
    trainer.save_model(f"{cfg.training.output_dir}/final")


if __name__ == "__main__":
    train()
