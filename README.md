# AIMO NuminaMath Finetuning

This repo provides a modular training pipeline to finetune the **OpenAI OSS 120B** model on the
[NuminaMath collection](https://huggingface.co/collections/AI-MO/numinamath) datasets.
It uses Hugging Face `transformers`, `datasets`, and `accelerate` so it can scale to multi-GPU
setups and supports gradient checkpointing.

## Features

- **Modular code**: data formatting, model loading, and training are in separate modules.
- **OpenAI-compatible tool calls**: optional tool call field validation matches OpenAI's chat format.
- **Multi-GPU ready**: run with `accelerate launch`.
- **Gradient checkpointing**: enabled by default for memory efficiency.
- **Learning-rate sweep helper**: script for quick LR trials.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Optional (experiment tracking):

```bash
pip install -e ".[train]"
```

## Dataset selection

The NuminaMath collection contains multiple datasets and configurations. Provide the exact
Hugging Face dataset name/config you want to train on:

```bash
numina-finetune \
  --dataset-name AI-MO/NuminaMath \
  --dataset-config default
```

If the dataset fields are not `problem`/`solution`, use the CLI overrides:

```bash
numina-finetune \
  --dataset-name AI-MO/NuminaMath \
  --question-field question \
  --answer-field solution
```

### NuminaMath-TIR (tool-integrated reasoning)

`AI-MO/NuminaMath-TIR` can store tool calls and outputs in non-OpenAI formats. The formatter now
supports mapping tool metadata into OpenAI-compatible `tool_calls`:

```bash
numina-finetune \
  --dataset-name AI-MO/NuminaMath-TIR \
  --tool-name-field tool \
  --tool-input-field tool_input \
  --tool-output-field tool_output
```

If the dataset already includes `tool_calls` as JSON, set `--tool-calls-field tool_calls`.

## Training (single GPU)

```bash
numina-finetune \
  --model-name openai/oss-120b \
  --dataset-name AI-MO/NuminaMath \
  --output-dir outputs/oss-120b-numina \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --learning-rate 2e-5 \
  --num-train-epochs 1
```

## Training (multi-GPU)

```bash
accelerate config
accelerate launch \
  numina-finetune \
  --model-name openai/oss-120b \
  --dataset-name AI-MO/NuminaMath \
  --output-dir outputs/oss-120b-numina
```

## Gradient checkpointing

Enabled by default. To disable:

```bash
numina-finetune --no-gradient-checkpointing
```

## OpenAI-compatible tool calls

If your dataset includes tool calls, specify the field name when building your dataset
(see `numina_finetune.data.format_example`). Tool calls must match OpenAI's chat format:

```json
{
  "tool_calls": [
    {
      "id": "call_1",
      "type": "function",
      "function": {
        "name": "my_tool",
        "arguments": "{\"x\": 1}"
      }
    }
  ]
}
```

The formatter validates and normalizes these to keep compatibility.

## Learning rate sweeps

Quickly run a few initial trials using the helper script:

```bash
python scripts/lr_sweep.py --lrs 1e-5,2e-5,3e-5,5e-5
```

Common initial LRs for large-scale instruction tuning are typically in the `1e-5` to `5e-5` range.
You can adjust the sweep list based on your hardware and batch size.

## TPU instructions

1. Install TPU-enabled PyTorch/XLA (example for TPU v4/v5):
   ```bash
   pip install torch==2.2.2 torch_xla[tpu]==2.2.2 -f https://storage.googleapis.com/libtpu-releases/index.html
   ```
2. Configure `accelerate` for TPU:
   ```bash
   accelerate config
   ```
3. Launch training:
   ```bash
   accelerate launch --tpu \
     numina-finetune \
     --model-name openai/oss-120b \
     --dataset-name AI-MO/NuminaMath \
     --output-dir outputs/oss-120b-numina
   ```

Adjust `--per-device-train-batch-size` and `--gradient-accumulation-steps` as needed.

## Running tests

```bash
pytest
```
