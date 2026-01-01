from __future__ import annotations

from typing import Any

from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

from numina_finetune.config import TrainingConfig


def build_trainer(
    *,
    config: TrainingConfig,
    model: Any,
    tokenizer: Any,
    train_dataset: Any,
    eval_dataset: Any | None = None,
) -> Trainer:
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        evaluation_strategy="steps" if eval_dataset is not None else "no",
        bf16=config.bf16,
        tf32=config.tf32,
        save_total_limit=2,
        report_to="none",
        seed=config.seed,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
