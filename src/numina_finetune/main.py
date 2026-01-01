from __future__ import annotations

from numina_finetune.config import TrainingConfig
from numina_finetune.data import build_message_dataset, tokenize_dataset, train_val_split
from numina_finetune.model import load_model_and_tokenizer
from numina_finetune.trainer import build_trainer


def main() -> None:
    config = TrainingConfig.from_args()

    dataset = build_message_dataset(
        dataset_name=config.dataset_name,
        dataset_config=config.dataset_config,
        split=config.dataset_split,
        system_prompt=config.system_prompt,
        question_field=config.question_field,
        answer_field=config.answer_field,
        tool_calls_field=config.tool_calls_field,
        tool_name_field=config.tool_name_field,
        tool_input_field=config.tool_input_field,
        tool_output_field=config.tool_output_field,
    )

    train_dataset, eval_dataset = train_val_split(dataset)

    model, tokenizer = load_model_and_tokenizer(
        config.model_name,
        gradient_checkpointing=config.gradient_checkpointing,
        bf16=config.bf16,
        tf32=config.tf32,
    )

    tokenized_train = tokenize_dataset(
        train_dataset,
        tokenizer=tokenizer,
        max_seq_length=config.max_seq_length,
    )
    tokenized_eval = tokenize_dataset(
        eval_dataset,
        tokenizer=tokenizer,
        max_seq_length=config.max_seq_length,
    )

    trainer = build_trainer(
        config=config,
        model=model,
        tokenizer=tokenizer,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
    )

    trainer.train()
    trainer.save_model(config.output_dir)


if __name__ == "__main__":
    main()
