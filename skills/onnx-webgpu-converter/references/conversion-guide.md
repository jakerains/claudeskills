# ONNX Conversion Deep Dive

## Table of Contents

- [Prerequisites Checklist](#prerequisites-checklist)
- [Method 1: optimum-cli (Recommended)](#method-1-optimum-cli-recommended)
- [Method 2: Python API](#method-2-python-api)
- [Method 3: torch.onnx (Low-Level)](#method-3-torchonnx-low-level)
- [Encoder-Decoder Models](#encoder-decoder-models)
- [Custom Architecture Models](#custom-architecture-models)
- [Quantization Deep Dive](#quantization-deep-dive)
- [Publishing to HuggingFace Hub](#publishing-to-huggingface-hub)
- [Common Errors and Solutions](#common-errors-and-solutions)

## Prerequisites Checklist

```bash
# Required
pip install "optimum[onnx]" onnxruntime torch transformers

# For GPU export
pip install onnxruntime-gpu

# For uploading to Hub
pip install huggingface-hub
huggingface-cli login

# Verify
python -c "import optimum; print(optimum.__version__)"
optimum-cli export onnx --help
```

**Python 3.8+ required. Recommended: Python 3.10+.**

## Method 1: optimum-cli (Recommended)

### Basic usage

```bash
# Auto-detect task from Hub model
optimum-cli export onnx --model distilbert-base-uncased-finetuned-sst-2-english ./distilbert_onnx/

# Explicit task for local models
optimum-cli export onnx --model ./local_model/ --task text-classification ./output_onnx/

# With optimization
optimum-cli export onnx --model gpt2 --optimize O2 ./gpt2_onnx/

# GPU export with fp16 optimization
optimum-cli export onnx --model gpt2 --device cuda --optimize O4 ./gpt2_fp16_onnx/
```

### Task detection

When exporting from a Hub model, the task is auto-detected. For local models, specify `--task`:

```python
# Check supported tasks for an architecture
from optimum.exporters.tasks import TasksManager
tasks = list(TasksManager.get_supported_tasks_for_model_type("distilbert", "onnx").keys())
print(tasks)
# ['default', 'fill-mask', 'text-classification', 'multiple-choice', 'token-classification', 'question-answering']
```

### Decoder models and KV cache

For text generation models, past key/value reuse is enabled by default. This corresponds to tasks ending in `-with-past`. To disable:

```bash
# With KV cache (default, recommended)
optimum-cli export onnx --model gpt2 ./gpt2_onnx/

# Without KV cache (not recommended for generation)
optimum-cli export onnx --model gpt2 --task text-generation ./gpt2_onnx/
```

## Method 2: Python API

```python
from optimum.onnxruntime import ORTModelForSequenceClassification

# Convert + load in one step
model = ORTModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english",
    export=True
)

# Save to disk
model.save_pretrained("./distilbert_onnx/")
```

### Available ORTModel classes

| Class | Task |
|-------|------|
| `ORTModelForSequenceClassification` | text-classification |
| `ORTModelForTokenClassification` | token-classification |
| `ORTModelForQuestionAnswering` | question-answering |
| `ORTModelForFeatureExtraction` | feature-extraction |
| `ORTModelForCausalLM` | text-generation |
| `ORTModelForSeq2SeqLM` | text2text-generation |
| `ORTModelForImageClassification` | image-classification |
| `ORTModelForSpeechSeq2Seq` | automatic-speech-recognition |

## Method 3: torch.onnx (Low-Level)

Use when optimum doesn't support the architecture:

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_id = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
dummy_input = tokenizer("Sample text", return_tensors="pt")

torch.onnx.export(
    model,
    tuple(dummy_input.values()),
    f="model.onnx",
    input_names=['input_ids', 'attention_mask'],
    output_names=['logits'],
    dynamic_axes={
        'input_ids': {0: 'batch_size', 1: 'sequence'},
        'attention_mask': {0: 'batch_size', 1: 'sequence'},
        'logits': {0: 'batch_size', 1: 'sequence'}
    },
    do_constant_folding=True,
    opset_version=14,
)
```

## Encoder-Decoder Models

Encoder-decoder models (T5, BART, Whisper) are split into multiple ONNX files:

```bash
optimum-cli export onnx --model t5-small ./t5_onnx/
# Produces:
#   encoder_model.onnx
#   decoder_model.onnx (merged with past by default)
```

### Quantizing Seq2Seq models

Each component must be quantized separately:

```python
from optimum.onnxruntime import ORTQuantizer, ORTModelForSeq2SeqLM
from optimum.onnxruntime.configuration import AutoQuantizationConfig

model = ORTModelForSeq2SeqLM.from_pretrained("optimum/t5-small")
model_dir = model.model_save_dir

quantizers = [
    ORTQuantizer.from_pretrained(model_dir, file_name="encoder_model.onnx"),
    ORTQuantizer.from_pretrained(model_dir, file_name="decoder_model.onnx"),
    ORTQuantizer.from_pretrained(model_dir, file_name="decoder_with_past_model.onnx"),
]

config = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
for q in quantizers:
    q.quantize(save_dir="./t5_quantized/", quantization_config=config)
```

## Custom Architecture Models

For models using `trust_remote_code=True`:

```bash
optimum-cli export onnx \
  --model THUDM/chatglm2-6b \
  --trust-remote-code \
  --task text-generation-with-past \
  ./chatglm_onnx/
```

If export fails, use the Python API with custom ONNX config. See:
https://huggingface.co/docs/optimum-onnx/onnx/usage_guides/export_a_model#custom-export-of-transformers-models

## Quantization Deep Dive

### Dynamic Quantization (simplest)

```python
from optimum.onnxruntime import ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig

quantizer = ORTQuantizer.from_pretrained("./model_onnx/")

# Choose target architecture
config = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
# Options: .arm64(), .avx2(), .avx512(), .avx512_vnni(), .tensorrt()

quantizer.quantize(save_dir="./quantized/", quantization_config=config)
```

### Static Quantization (better quality, requires calibration data)

```python
from functools import partial
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTQuantizer, ORTModelForSequenceClassification
from optimum.onnxruntime.configuration import AutoQuantizationConfig, AutoCalibrationConfig

model_id = "distilbert-base-uncased-finetuned-sst-2-english"
model = ORTModelForSequenceClassification.from_pretrained(model_id, export=True)
tokenizer = AutoTokenizer.from_pretrained(model_id)
quantizer = ORTQuantizer.from_pretrained(model)
config = AutoQuantizationConfig.arm64(is_static=True, per_channel=False)

def preprocess_fn(ex, tokenizer):
    return tokenizer(ex["sentence"])

calibration_dataset = quantizer.get_calibration_dataset(
    "glue", dataset_config_name="sst2",
    preprocess_function=partial(preprocess_fn, tokenizer=tokenizer),
    num_samples=50, dataset_split="train",
)

calibration_config = AutoCalibrationConfig.minmax(calibration_dataset)
ranges = quantizer.fit(
    dataset=calibration_dataset,
    calibration_config=calibration_config,
    operators_to_quantize=config.operators_to_quantize,
)

quantizer.quantize(
    save_dir="./static_quantized/",
    calibration_tensors_range=ranges,
    quantization_config=config,
)
```

### Recommendation for Transformers.js WebGPU

| Target | Recommended dtype | Rationale |
|--------|-------------------|-----------|
| WebGPU (desktop) | `fp16` or `q4f16` | Good GPU utilization, balanced |
| WebGPU (mobile) | `q4` | Minimize download + memory |
| WASM (CPU fallback) | `q8` | Best CPU performance |
| Maximum accuracy | `fp32` | No quantization loss |

## Publishing to HuggingFace Hub

### Create proper model card

Add `transformers.js` library tag and `onnx` tags for discoverability:

```yaml
---
library_name: transformers.js
tags:
  - onnx
  - webgpu
pipeline_tag: <your-task>
base_model: <original-model-id>
---
```

### Upload

```bash
huggingface-cli upload <username>/<model-name>-ONNX ./output_dir/
```

### Directory structure convention

Follow the `onnx-community` pattern:

```
model-repo/
├── onnx/
│   ├── model.onnx                    # fp32
│   ├── model_fp16.onnx               # fp16
│   ├── model_quantized.onnx          # int8
│   └── model_q4.onnx                 # int4
├── config.json                       # Model config
├── tokenizer.json                    # Tokenizer
├── tokenizer_config.json
├── special_tokens_map.json
└── README.md                         # Model card
```

## Common Errors and Solutions

### "Can't determine model task"
```
Error: Couldn't determine the model task automatically
```
**Fix:** Specify `--task` explicitly. Check supported tasks with:
```python
from optimum.exporters.tasks import TasksManager
print(TasksManager.get_supported_tasks_for_model_type("your-arch", "onnx"))
```

### "Model architecture not supported"
**Fix:** Check supported architectures list. For unsupported models, use `torch.onnx.export()` directly or create a custom ONNX config.

### "RuntimeError: ONNX export failed"
**Fixes:**
- Try a different opset: `--opset 14` or `--opset 17`
- Use `--no-post-process`
- Update optimum: `pip install -U optimum[onnx]`

### "Validation error: output mismatch"
**Fix:** Increase tolerance: `--atol 1e-3` or `--atol 1e-2`

### "Out of memory during export"
**Fixes:**
- Use `--device cpu`
- Reduce `--batch_size 1 --sequence_length 64`
- Close other applications

### "Model produces wrong outputs after conversion"
**Fixes:**
- Check the task is correct (especially `-with-past` suffix)
- Verify the model works in Python first
- Try without optimization (`--optimize` removed)
- Compare outputs with tolerance checking

### "WebGPU falls back to WASM in browser"
**Causes:**
- Browser doesn't support WebGPU
- Model has unsupported ONNX ops for WebGPU backend
**Fixes:**
- Use Chrome 113+ or Edge 113+
- Firefox: enable `dom.webgpu.enabled` flag
- Safari: enable WebGPU feature flag
- Try `fp16` or `fp32` dtype (some quantized ops may not have WebGPU kernels)
