from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Sequence


@dataclass
class TrainingConfig:
    model_name: str = "openai/oss-120b"
    dataset_name: str = "AI-MO/NuminaMath"
    dataset_config: str | None = None
    dataset_split: str = "train"
    output_dir: str = "outputs/oss-120b-numina"
    max_seq_length: int = 4096
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-5
    num_train_epochs: int = 1
    weight_decay: float = 0.1
    warmup_ratio: float = 0.03
    logging_steps: int = 10
    save_steps: int = 500
    seed: int = 42
    gradient_checkpointing: bool = True
    bf16: bool = True
    tf32: bool = True
    packing: bool = False
    system_prompt: str = (
        "You are a helpful math tutor. Provide clear, step-by-step reasoning and the final answer."
    )
    question_field: str | None = None
    answer_field: str | None = None
    tool_calls_field: str | None = None
    tool_name_field: str | None = None
    tool_input_field: str | None = None
    tool_output_field: str | None = None

    def to_argparse(self) -> argparse.Namespace:
        return argparse.Namespace(**self.__dict__)

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument("--model-name", default=TrainingConfig.model_name)
        parser.add_argument("--dataset-name", default=TrainingConfig.dataset_name)
        parser.add_argument("--dataset-config", default=TrainingConfig.dataset_config)
        parser.add_argument("--dataset-split", default=TrainingConfig.dataset_split)
        parser.add_argument("--output-dir", default=TrainingConfig.output_dir)
        parser.add_argument("--max-seq-length", type=int, default=TrainingConfig.max_seq_length)
        parser.add_argument(
            "--per-device-train-batch-size",
            type=int,
            default=TrainingConfig.per_device_train_batch_size,
        )
        parser.add_argument(
            "--gradient-accumulation-steps",
            type=int,
            default=TrainingConfig.gradient_accumulation_steps,
        )
        parser.add_argument("--learning-rate", type=float, default=TrainingConfig.learning_rate)
        parser.add_argument("--num-train-epochs", type=int, default=TrainingConfig.num_train_epochs)
        parser.add_argument("--weight-decay", type=float, default=TrainingConfig.weight_decay)
        parser.add_argument("--warmup-ratio", type=float, default=TrainingConfig.warmup_ratio)
        parser.add_argument("--logging-steps", type=int, default=TrainingConfig.logging_steps)
        parser.add_argument("--save-steps", type=int, default=TrainingConfig.save_steps)
        parser.add_argument("--seed", type=int, default=TrainingConfig.seed)
        parser.add_argument(
            "--gradient-checkpointing",
            action=argparse.BooleanOptionalAction,
            default=TrainingConfig.gradient_checkpointing,
        )
        parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
        parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
        parser.add_argument("--packing", action=argparse.BooleanOptionalAction, default=False)
        parser.add_argument("--system-prompt", default=TrainingConfig.system_prompt)
        parser.add_argument("--question-field", default=TrainingConfig.question_field)
        parser.add_argument("--answer-field", default=TrainingConfig.answer_field)
        parser.add_argument("--tool-calls-field", default=TrainingConfig.tool_calls_field)
        parser.add_argument("--tool-name-field", default=TrainingConfig.tool_name_field)
        parser.add_argument("--tool-input-field", default=TrainingConfig.tool_input_field)
        parser.add_argument("--tool-output-field", default=TrainingConfig.tool_output_field)
        return parser

    @classmethod
    def from_args(cls, args: Sequence[str] | None = None) -> "TrainingConfig":
        parser = argparse.ArgumentParser(description="Finetune OSS-120B on NuminaMath")
        cls.add_cli_args(parser)
        parsed = parser.parse_args(args=args)
        return cls(**vars(parsed))
