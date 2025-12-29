from __future__ import annotations

from collections.abc import Iterable
import json
from typing import Any, Dict, List, Tuple

from datasets import Dataset, load_dataset

Message = Dict[str, Any]


def _find_first_key(example: Dict[str, Any], candidates: Iterable[str]) -> str | None:
    for key in candidates:
        if key in example and example[key] not in (None, ""):
            return key
    return None


def _serialize_arguments(arguments: Any) -> str:
    if isinstance(arguments, str):
        return arguments
    return json.dumps(arguments, ensure_ascii=False)


def _normalize_tool_calls(tool_calls: Any) -> List[Dict[str, Any]]:
    if tool_calls is None:
        return []
    if isinstance(tool_calls, str):
        try:
            tool_calls = json.loads(tool_calls)
        except json.JSONDecodeError as exc:
            raise ValueError("tool_calls string must be valid JSON") from exc
    if isinstance(tool_calls, dict):
        tool_calls = [tool_calls]
    if not isinstance(tool_calls, list):
        raise ValueError("tool_calls must be a list of dicts compatible with OpenAI format")
    normalized = []
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            raise ValueError("Each tool call must be a dict")
        if tool_call.get("type") != "function":
            raise ValueError("tool_calls only supports type=function")
        function = tool_call.get("function") or {}
        if "name" not in function or "arguments" not in function:
            raise ValueError("tool_calls.function must include name and arguments")
        normalized.append(
            {
                "id": tool_call.get("id", "call_0"),
                "type": "function",
                "function": {
                    "name": function["name"],
                    "arguments": _serialize_arguments(function["arguments"]),
                },
            }
        )
    return normalized


def _infer_tool_fields(example: Dict[str, Any]) -> Tuple[str | None, str | None, str | None]:
    tool_name_field = _find_first_key(example, ("tool", "tool_name", "tool_type"))
    tool_input_field = _find_first_key(example, ("tool_input", "tool_arguments", "tool_args"))
    tool_output_field = _find_first_key(example, ("tool_output", "tool_result", "tool_response"))
    return tool_name_field, tool_input_field, tool_output_field


def _build_tool_calls_from_fields(
    example: Dict[str, Any],
    *,
    tool_name_field: str | None,
    tool_input_field: str | None,
) -> List[Dict[str, Any]]:
    if not tool_name_field or not tool_input_field:
        return []
    tool_name = str(example[tool_name_field]).strip()
    tool_input = example[tool_input_field]
    if not tool_name:
        return []
    return [
        {
            "id": "call_0",
            "type": "function",
            "function": {
                "name": tool_name,
                "arguments": _serialize_arguments(tool_input),
            },
        }
    ]


def format_example(
    example: Dict[str, Any],
    *,
    system_prompt: str,
    question_field: str | None = None,
    answer_field: str | None = None,
    tool_calls_field: str | None = None,
    tool_name_field: str | None = None,
    tool_input_field: str | None = None,
    tool_output_field: str | None = None,
) -> Dict[str, Any]:
    question_field = question_field or _find_first_key(
        example, ("problem", "question", "prompt", "query")
    )
    answer_field = answer_field or _find_first_key(example, ("solution", "answer", "response"))

    if not question_field or not answer_field:
        raise ValueError("Unable to infer question/answer fields in dataset example")

    messages: List[Message] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": str(example[question_field]).strip()},
    ]

    tool_calls: List[Dict[str, Any]] = []
    inferred_tool_name, inferred_tool_input, inferred_tool_output = _infer_tool_fields(example)
    tool_name_field = tool_name_field or inferred_tool_name
    tool_input_field = tool_input_field or inferred_tool_input
    tool_output_field = tool_output_field or inferred_tool_output

    if tool_calls_field and tool_calls_field in example:
        tool_calls = _normalize_tool_calls(example[tool_calls_field])
    else:
        tool_calls = _build_tool_calls_from_fields(
            example,
            tool_name_field=tool_name_field,
            tool_input_field=tool_input_field,
        )

    assistant_message: Message = {"role": "assistant", "content": str(example[answer_field]).strip()}

    if tool_output_field and tool_output_field in example and tool_calls:
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": tool_calls,
            }
        )
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_calls[0]["id"],
                "content": str(example[tool_output_field]).strip(),
            }
        )
    elif tool_calls:
        assistant_message["tool_calls"] = tool_calls

    messages.append(assistant_message)

    return {"messages": messages}


def build_message_dataset(
    *,
    dataset_name: str,
    dataset_config: str | None,
    split: str,
    system_prompt: str,
    question_field: str | None = None,
    answer_field: str | None = None,
    tool_calls_field: str | None = None,
    tool_name_field: str | None = None,
    tool_input_field: str | None = None,
    tool_output_field: str | None = None,
) -> Dataset:
    dataset = load_dataset(dataset_name, dataset_config, split=split)

    def _formatter(example: Dict[str, Any]) -> Dict[str, Any]:
        return format_example(
            example,
            system_prompt=system_prompt,
            question_field=question_field,
            answer_field=answer_field,
            tool_calls_field=tool_calls_field,
            tool_name_field=tool_name_field,
            tool_input_field=tool_input_field,
            tool_output_field=tool_output_field,
        )

    return dataset.map(_formatter, remove_columns=dataset.column_names)


def format_messages_to_text(messages: List[Message], tokenizer: Any) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    parts = []
    for message in messages:
        parts.append(f"{message['role']}: {message['content']}")
        if message.get("tool_calls"):
            parts.append(f"tool_calls: {message['tool_calls']}")
    return "\n".join(parts)


def tokenize_dataset(
    dataset: Dataset,
    *,
    tokenizer: Any,
    max_seq_length: int,
) -> Dataset:
    def _tokenize(batch: Dict[str, Any]) -> Dict[str, Any]:
        text = [format_messages_to_text(messages, tokenizer) for messages in batch["messages"]]
        return tokenizer(text, max_length=max_seq_length, truncation=True)

    return dataset.map(_tokenize, batched=True, remove_columns=dataset.column_names)


def train_val_split(dataset: Dataset, val_ratio: float = 0.02) -> Tuple[Dataset, Dataset]:
    split = dataset.train_test_split(test_size=val_ratio, seed=42)
    return split["train"], split["test"]
