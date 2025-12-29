from numina_finetune.data import format_example, format_messages_to_text


class DummyTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert tokenize is False
        assert add_generation_prompt is False
        return " | ".join(f"{m['role']}:{m['content']}" for m in messages)


def test_format_example_default_fields():
    example = {"problem": "2+2?", "solution": "4"}
    formatted = format_example(example, system_prompt="Solve.")
    assert formatted["messages"][0]["role"] == "system"
    assert formatted["messages"][1]["content"] == "2+2?"
    assert formatted["messages"][2]["content"] == "4"


def test_format_example_tool_calls():
    example = {
        "question": "Compute",
        "answer": "done",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "calc", "arguments": "{\"x\": 1}"},
            }
        ],
    }
    formatted = format_example(
        example,
        system_prompt="Solve.",
        question_field="question",
        answer_field="answer",
        tool_calls_field="tool_calls",
    )
    assert formatted["messages"][2]["tool_calls"][0]["function"]["name"] == "calc"


def test_format_messages_to_text_uses_chat_template():
    tokenizer = DummyTokenizer()
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
    ]
    text = format_messages_to_text(messages, tokenizer)
    assert text == "system:sys | user:hi"


def test_format_example_tir_tool_fields():
    example = {
        "problem": "Compute 2+2",
        "solution": "4",
        "tool": "calculator",
        "tool_input": {"expression": "2+2"},
        "tool_output": "4",
    }
    formatted = format_example(
        example,
        system_prompt="Solve.",
        tool_name_field="tool",
        tool_input_field="tool_input",
        tool_output_field="tool_output",
    )
    messages = formatted["messages"]
    assert messages[2]["role"] == "assistant"
    assert messages[2]["tool_calls"][0]["function"]["name"] == "calculator"
    assert messages[3]["role"] == "tool"
    assert messages[3]["content"] == "4"
    assert messages[4]["content"] == "4"
