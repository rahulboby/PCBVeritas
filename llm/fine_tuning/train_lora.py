"""
LoRA Fine-Tuning Script for Qwen2.5-1.5B-Instruct
====================================================
PURPOSE:
    Fine-tunes Qwen2.5-1.5B-Instruct on our synthetic PCB inspection
    dataset using LoRA (Low-Rank Adaptation). After fine-tuning, the
    model generates expert-quality PCB inspection reports.

WHAT IS LORA?
    LoRA is a parameter-efficient fine-tuning technique. Instead of
    updating all model weights (billions of parameters), LoRA adds
    small trainable "adapter" matrices to specific layers.

    For a weight matrix W (shape: d × k), LoRA adds:
    W' = W + BA
    Where:
    - B: shape (d × r) — small matrix, r << d
    - A: shape (r × k) — small matrix
    - r = rank (e.g., 16) — the only new parameters to train

    This means:
    - Original W: d × k parameters (FROZEN)
    - LoRA additions: d×r + r×k parameters (TRAINABLE)
    - With r=16, d=2048, k=2048: 2048²=4M frozen, 2×16×2048=65k trainable
    - We only train ~1% of total parameters!

WHY LORA FOR 6GB VRAM?
    Full fine-tuning of 1.5B params requires ~12GB VRAM minimum.
    LoRA + gradient checkpointing + fp16 reduces this to ~4-5GB.
    Perfect for our RTX 4050 (6GB).

TRAINING PIPELINE:
    1. Load Qwen2.5-1.5B-Instruct with fp16
    2. Apply LoRA adapters to attention and MLP layers
    3. Load synthetic PCB instruction dataset
    4. Format as chat template (system + user + assistant)
    5. Train with gradient accumulation (effective batch = 16)
    6. Save LoRA adapter weights

USAGE:
    python llm/fine_tuning/train_lora.py
    python llm/fine_tuning/train_lora.py --config configs/fine_tuning.yaml
    python llm/fine_tuning/train_lora.py --generate-data-only
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Optional
import yaml
import torch
import numpy as np
from loguru import logger
from rich.console import Console

# HuggingFace libraries
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    PeftModel,
)
from datasets import Dataset
import mlflow

console = Console()


def load_config(config_path: str) -> dict:
    """Load fine-tuning configuration."""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_tokenizer(model_name_or_path: str, local_dir: str) -> "AutoTokenizer":
    """
    Load Qwen tokenizer.

    The tokenizer converts text to token IDs that the model processes.
    Qwen uses a BPE (Byte Pair Encoding) tokenizer with 150k+ vocabulary.

    Args:
        model_name_or_path: HuggingFace model ID or local path.
        local_dir: Local cache directory.

    Returns:
        Loaded tokenizer.
    """
    local_path = Path(local_dir)

    if local_path.exists() and any(local_path.iterdir()):
        load_from = str(local_path)
        logger.info(f"Loading tokenizer from local: {local_path}")
    else:
        load_from = model_name_or_path
        logger.info(f"Loading tokenizer from HuggingFace: {model_name_or_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        load_from,
        cache_dir=str(local_path),
        trust_remote_code=True,
        padding_side="left",  # For causal LM batch inference
    )

    # Qwen uses a specific pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def load_base_model(
    model_name_or_path: str,
    local_dir: str,
    torch_dtype: str = "float16",
) -> "AutoModelForCausalLM":
    """
    Load Qwen2.5-1.5B-Instruct base model.

    Uses fp16 to halve memory usage (1.5B × 2 bytes ≈ 3GB VRAM).

    Args:
        model_name_or_path: HuggingFace model ID.
        local_dir: Local cache directory.
        torch_dtype: "float16" or "float32".

    Returns:
        Loaded model.
    """
    local_path = Path(local_dir)

    if local_path.exists() and any(local_path.iterdir()):
        load_from = str(local_path)
        logger.info(f"Loading model from local: {local_path}")
    else:
        load_from = model_name_or_path
        logger.info(f"Downloading model: {model_name_or_path}")
        logger.info("This will take a few minutes on first run...")

    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map.get(torch_dtype, torch.float16)

    model = AutoModelForCausalLM.from_pretrained(
        load_from,
        cache_dir=str(local_path),
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map="auto",  # Automatically places layers on GPU/CPU
    )

    # Save locally for future offline use
    if load_from != str(local_path):
        local_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(local_path))
        logger.info(f"Model saved locally: {local_path}")

    logger.info(
        f"Model loaded | "
        f"params={sum(p.numel() for p in model.parameters()) / 1e6:.0f}M | "
        f"dtype={dtype}"
    )

    return model


def apply_lora(model: "AutoModelForCausalLM", lora_config: dict) -> "PeftModel":
    """
    Apply LoRA adapters to the base model.

    LoRA modifies specific attention and MLP projection matrices.
    The original weights are frozen; only the small AB matrices are trained.

    Args:
        model: Base Qwen model.
        lora_config: LoRA configuration dict from YAML.

    Returns:
        Model with LoRA adapters applied.
    """
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_config.get("r", 16),
        lora_alpha=lora_config.get("lora_alpha", 32),
        target_modules=lora_config.get("target_modules", [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]),
        lora_dropout=lora_config.get("lora_dropout", 0.1),
        bias=lora_config.get("bias", "none"),
    )

    model = get_peft_model(model, peft_config)

    # Print trainable vs frozen parameter counts
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_pct = 100 * trainable_params / total_params

    logger.info(f"LoRA applied:")
    logger.info(f"  Total parameters:     {total_params:,}")
    logger.info(f"  Trainable (LoRA):     {trainable_params:,}")
    logger.info(f"  Trainable percentage: {trainable_pct:.2f}%")

    return model


def format_instruction_for_qwen(
    sample: dict,
    tokenizer: "AutoTokenizer",
    system_prompt: str,
    max_length: int = 1024,
) -> dict:
    """
    Format an instruction sample into Qwen's chat template.

    Qwen uses OpenAI-style chat format:
    <|im_start|>system
    You are a PCB inspection expert...
    <|im_end|>
    <|im_start|>user
    What causes missing holes?
    <|im_end|>
    <|im_start|>assistant
    Missing holes are caused by...
    <|im_end|>

    We tokenize the full sequence and mask the input portion
    (system + user) from the loss calculation — we only want
    the model to learn from the assistant's response.

    Args:
        sample: Dict with 'instruction', 'input', 'output'.
        tokenizer: Qwen tokenizer.
        system_prompt: System message for the model.
        max_length: Maximum sequence length.

    Returns:
        Dict with 'input_ids', 'attention_mask', 'labels'.
    """
    # Build messages list
    user_content = sample["instruction"]
    if sample.get("input"):
        user_content = f"{sample['instruction']}\n\nContext: {sample['input']}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": sample["output"]},
    ]

    # Apply Qwen's chat template
    # add_generation_prompt=False because we're including the response
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    # Tokenize
    tokenized = tokenizer(
        formatted,
        max_length=max_length,
        truncation=True,
        padding=False,
        return_tensors=None,
    )

    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]

    # Create labels: -100 for input (system + user), actual IDs for output
    # We find where the assistant response starts
    # The assistant response starts after "<|im_start|>assistant\n"
    assistant_marker = tokenizer.encode(
        "<|im_start|>assistant\n",
        add_special_tokens=False,
    )
    marker_len = len(assistant_marker)

    labels = [-100] * len(input_ids)  # Initialize all as ignored

    # Find assistant response start
    for i in range(len(input_ids) - marker_len):
        if input_ids[i:i+marker_len] == assistant_marker:
            # Labels start after the marker
            labels[i + marker_len:] = input_ids[i + marker_len:]
            break
    else:
        # Fallback: use last third as labels
        split = len(input_ids) * 2 // 3
        labels[split:] = input_ids[split:]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def load_instruction_dataset(
    data_path: str,
    tokenizer: "AutoTokenizer",
    system_prompt: str,
    max_length: int = 1024,
    train_split: float = 0.9,
    seed: int = 42,
) -> tuple[Dataset, Dataset]:
    """
    Load and tokenize the instruction dataset.

    Args:
        data_path: Path to pcb_instructions.json.
        tokenizer: Qwen tokenizer.
        system_prompt: System prompt text.
        max_length: Maximum sequence length.
        train_split: Train/val split ratio.
        seed: Random seed.

    Returns:
        Tuple of (train_dataset, eval_dataset).
    """
    data_file = Path(data_path)
    if not data_file.exists():
        raise FileNotFoundError(
            f"Instruction dataset not found: {data_file}\n"
            "Generate it first: python llm/fine_tuning/generate_dataset.py"
        )

    with open(data_file, encoding="utf-8") as f:
        raw_data = json.load(f)

    logger.info(f"Loaded {len(raw_data)} instruction samples from {data_file}")

    # Tokenize all samples
    tokenized_samples = []
    skipped = 0

    for sample in raw_data:
        try:
            tokenized = format_instruction_for_qwen(
                sample, tokenizer, system_prompt, max_length
            )
            # Skip samples that are too short (likely formatting errors)
            if len(tokenized["input_ids"]) < 20:
                skipped += 1
                continue
            tokenized_samples.append(tokenized)
        except Exception as e:
            logger.warning(f"Failed to tokenize sample: {e}")
            skipped += 1

    logger.info(f"Tokenized: {len(tokenized_samples)} | Skipped: {skipped}")

    # Create HuggingFace Dataset
    dataset = Dataset.from_list(tokenized_samples)

    # Split into train/val
    split = dataset.train_test_split(
        test_size=1 - train_split,
        seed=seed,
    )

    logger.info(f"Train: {len(split['train'])} | Val: {len(split['test'])}")
    return split["train"], split["test"]


def train_lora(config_path: str = "configs/fine_tuning.yaml") -> None:
    """
    Main LoRA fine-tuning function.

    Args:
        config_path: Path to fine-tuning configuration YAML.
    """
    config = load_config(config_path)

    console.rule("[bold blue]LoRA Fine-Tuning: Qwen2.5-1.5B-Instruct")

    # --- VRAM Check ---
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {vram_gb:.1f}GB")
        if vram_gb < 4:
            logger.warning("Less than 4GB VRAM. Training may fail. Reduce batch size.")
    else:
        logger.warning("No GPU. Training on CPU will be extremely slow.")

    # --- Load Tokenizer ---
    tokenizer = load_tokenizer(
        config["base_model"]["name"],
        config["base_model"]["local_dir"],
    )

    # --- Load Base Model ---
    model = load_base_model(
        config["base_model"]["name"],
        config["base_model"]["local_dir"],
        config["base_model"]["torch_dtype"],
    )

    # Enable gradient checkpointing (saves ~30% VRAM at slight speed cost)
    if config["training"].get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False  # Incompatible with gradient checkpointing

    # --- Apply LoRA ---
    model = apply_lora(model, config["lora"])

    # --- Load Dataset ---
    system_prompt = yaml.safe_load(
        open("configs/llm.yaml", encoding="utf-8")
    ).get("system_prompt", "You are a PCB inspection expert.")

    train_dataset, eval_dataset = load_instruction_dataset(
        data_path=config["dataset"]["path"],
        tokenizer=tokenizer,
        system_prompt=system_prompt,
        max_length=config["dataset"]["max_length"],
        train_split=config["dataset"]["train_split"],
        seed=config["dataset"]["seed"],
    )

    # Data collator handles padding within batches
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
        label_pad_token_id=-100,
    )

    # --- Training Arguments ---
    training_args = TrainingArguments(
        output_dir=config["training"]["output_dir"],
        num_train_epochs=config["training"]["num_train_epochs"],
        per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
        per_device_eval_batch_size=config["training"]["per_device_eval_batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        gradient_checkpointing=config["training"]["gradient_checkpointing"],
        warmup_steps=config["training"]["warmup_steps"],
        learning_rate=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        lr_scheduler_type=config["training"]["lr_scheduler_type"],
        save_steps=config["training"]["save_steps"],
        eval_steps=config["training"]["eval_steps"],
        logging_steps=config["training"]["logging_steps"],
        max_grad_norm=config["training"]["max_grad_norm"],
        fp16=config["training"]["fp16"],
        bf16=config["training"]["bf16"],
        dataloader_num_workers=config["training"]["dataloader_num_workers"],
        remove_unused_columns=False,
        group_by_length=config["training"]["group_by_length"],
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=["mlflow"] if config.get("mlflow", {}).get("enabled") else [],
        logging_dir="logs/fine_tuning",
        run_name="pcb_lora_qwen",
    )

    # --- MLflow Setup ---
    if config.get("mlflow", {}).get("enabled"):
        mlflow.set_experiment(config["mlflow"]["experiment_name"])

    # --- Trainer ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    # --- Train ---
    console.print(
        f"\n[yellow]Starting LoRA training...[/yellow]\n"
        f"  Epochs: {config['training']['num_train_epochs']}\n"
        f"  Train samples: {len(train_dataset)}\n"
        f"  Val samples: {len(eval_dataset)}\n"
        f"  Effective batch: "
        f"{config['training']['per_device_train_batch_size'] * config['training']['gradient_accumulation_steps']}"
    )

    trainer.train()

    # --- Save Final Model ---
    output_dir = Path(config["training"]["output_dir"])
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    logger.info(f"Fine-tuned model saved to: {output_dir}")
    console.print(f"\n[bold green]Training complete! Model saved to: {output_dir}[/bold green]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoRA Fine-Tune Qwen2.5 for PCB Inspection")
    parser.add_argument("--config", default="configs/fine_tuning.yaml")
    parser.add_argument(
        "--generate-data-only",
        action="store_true",
        help="Only generate synthetic dataset, don't train",
    )
    args = parser.parse_args()

    if args.generate_data_only:
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from llm.fine_tuning.generate_dataset import generate_dataset
        cfg = load_config(args.config)
        generate_dataset(
            output_path=cfg["generation"]["output_path"],
            n_samples=cfg["generation"]["synthetic_samples"],
        )
    else:
        train_lora(args.config)
