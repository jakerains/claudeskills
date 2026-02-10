# Transformers.js WebGPU Usage Guide

## Table of Contents

- [Installation](#installation)
- [Pipeline API with WebGPU](#pipeline-api-with-webgpu)
- [Dtype Selection Guide](#dtype-selection-guide)
- [Per-Module Dtypes](#per-module-dtypes)
- [Task Examples](#task-examples)
- [Advanced Model Loading](#advanced-model-loading)
- [Browser Compatibility](#browser-compatibility)
- [Performance Tips](#performance-tips)

## Installation

```bash
npm install @huggingface/transformers
```

CDN (for quick prototyping):
```javascript
import { pipeline } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3";
```

## Pipeline API with WebGPU

```javascript
import { pipeline } from "@huggingface/transformers";

// WebGPU + quantization
const pipe = await pipeline("task-name", "model-id", {
  device: "webgpu",
  dtype: "q4",
});

const result = await pipe("Your input here");
```

### Device Options

| Device | Backend | When to Use |
|--------|---------|-------------|
| `"webgpu"` | GPU via WebGPU | Default for supported browsers, fastest |
| `"wasm"` | CPU via WebAssembly | Fallback, universal support |
| (omitted) | Auto-detect | Uses WASM by default |

### Dtype Options

| dtype | Description | Default For |
|-------|-------------|-------------|
| `"fp32"` | Full precision 32-bit | WebGPU (if no dtype specified) |
| `"fp16"` | Half precision 16-bit | Good WebGPU balance |
| `"q8"` | 8-bit quantized | WASM default |
| `"int8"` | Alias for q8 | — |
| `"uint8"` | Unsigned 8-bit | — |
| `"q4"` | 4-bit quantized | Smallest size |
| `"bnb4"` | BitsAndBytes 4-bit | — |
| `"q4f16"` | 4-bit weights, fp16 compute | WebGPU + small |

**Available dtypes vary by model.** Check the model's `onnx/` folder on the Hub.

## Per-Module Dtypes

Encoder-decoder models benefit from different quantization per component:

```javascript
import { Florence2ForConditionalGeneration } from "@huggingface/transformers";

const model = await Florence2ForConditionalGeneration.from_pretrained(
  "onnx-community/Florence-2-base-ft",
  {
    dtype: {
      embed_tokens: "fp16",
      vision_encoder: "fp16",
      encoder_model: "q4",
      decoder_model_merged: "q4",
    },
    device: "webgpu",
  }
);
```

**When to use per-module dtypes:**
- Whisper (encoder sensitive to quantization)
- Florence-2 (vision encoder needs higher precision)
- T5/BART family (encoder precision matters for quality)

## Task Examples

### Text Generation (LLM)

```javascript
import { pipeline } from "@huggingface/transformers";

const generator = await pipeline(
  "text-generation",
  "onnx-community/Qwen2.5-0.5B-Instruct",
  { dtype: "q4", device: "webgpu" }
);

const messages = [
  { role: "system", content: "You are a helpful assistant." },
  { role: "user", content: "Tell me a joke." },
];

const output = await generator(messages, { max_new_tokens: 128 });
console.log(output[0].generated_text.at(-1).content);
```

### Text Classification / Sentiment Analysis

```javascript
const classifier = await pipeline(
  "sentiment-analysis",
  "Xenova/distilbert-base-uncased-finetuned-sst-2-english",
  { device: "webgpu" }
);

const result = await classifier("I love this product!");
// [{ label: 'POSITIVE', score: 0.9998 }]
```

### Feature Extraction / Embeddings

```javascript
const extractor = await pipeline(
  "feature-extraction",
  "mixedbread-ai/mxbai-embed-xsmall-v1",
  { device: "webgpu" }
);

const embeddings = await extractor(
  ["Hello world!", "Another sentence."],
  { pooling: "mean", normalize: true }
);
console.log(embeddings.tolist());
```

### Speech Recognition (Whisper)

```javascript
const transcriber = await pipeline(
  "automatic-speech-recognition",
  "onnx-community/whisper-tiny.en",
  { device: "webgpu" }
);

const result = await transcriber("https://example.com/audio.wav");
console.log(result.text);
```

### Image Classification

```javascript
const classifier = await pipeline(
  "image-classification",
  "onnx-community/mobilenetv4_conv_small.e2400_r224_in1k",
  { device: "webgpu" }
);

const result = await classifier("https://example.com/image.jpg");
// [{ label: 'tiger', score: 0.61 }, ...]
```

### Object Detection

```javascript
const detector = await pipeline(
  "object-detection",
  "onnx-community/detr-resnet-50",
  { device: "webgpu" }
);

const result = await detector("https://example.com/image.jpg");
// [{ label: 'cat', score: 0.98, box: { xmin: 10, ymin: 20, xmax: 100, ymax: 200 } }]
```

### Zero-Shot Image Classification

```javascript
const classifier = await pipeline(
  "zero-shot-image-classification",
  "Xenova/clip-vit-base-patch32",
  { device: "webgpu" }
);

const result = await classifier("https://example.com/image.jpg", [
  "a photo of a cat",
  "a photo of a dog",
  "a photo of a bird",
]);
```

### Depth Estimation

```javascript
const estimator = await pipeline(
  "depth-estimation",
  "onnx-community/depth-anything-small",
  { device: "webgpu" }
);

const result = await estimator("https://example.com/image.jpg");
// result.depth is a Tensor with depth values
```

### Background Removal

```javascript
const remover = await pipeline(
  "background-removal",
  "briaai/RMBG-1.4",
  { device: "webgpu" }
);

const result = await remover("https://example.com/image.jpg");
// Returns image with transparent background
```

## Advanced Model Loading

### Custom model path

```javascript
import { env } from "@huggingface/transformers";

// Local models
env.localModelPath = "/path/to/models/";
env.allowRemoteModels = false;

// Custom WASM path
env.backends.onnx.wasm.wasmPaths = "/path/to/wasm/";
```

### Direct class instantiation

```javascript
import {
  AutoTokenizer,
  AutoModelForSequenceClassification,
} from "@huggingface/transformers";

const tokenizer = await AutoTokenizer.from_pretrained("model-id");
const model = await AutoModelForSequenceClassification.from_pretrained(
  "model-id",
  { device: "webgpu", dtype: "fp16" }
);

const inputs = tokenizer("Hello world!");
const { logits } = await model(inputs);
```

## Browser Compatibility

| Browser | WebGPU Status | Notes |
|---------|---------------|-------|
| Chrome 113+ | Supported | Default enabled |
| Edge 113+ | Supported | Default enabled |
| Firefox | Behind flag | Enable `dom.webgpu.enabled` in about:config |
| Safari 18+ | Behind flag | Enable WebGPU in Developer menu |
| Deno | Experimental | `--unstable-webgpu` flag |
| Node.js | Not supported | Use WASM backend |

**Global WebGPU support:** ~70% of users (as of 2025).

### Feature detection

```javascript
if (navigator.gpu) {
  // WebGPU available
  const pipe = await pipeline("task", "model", { device: "webgpu" });
} else {
  // Fallback to WASM
  const pipe = await pipeline("task", "model", { device: "wasm" });
}
```

## Performance Tips

1. **Use q4 for large models** — 4-bit quantization dramatically reduces download size and memory
2. **Use fp16 for WebGPU** — Half precision is the sweet spot for GPU inference quality/speed
3. **Use q8 for WASM** — 8-bit is the default and best performer on CPU
4. **Cache models** — Transformers.js caches models in browser storage automatically
5. **Warm up the pipeline** — First inference is slower due to shader compilation on WebGPU
6. **Use per-module dtypes** for encoder-decoder models to balance quality and size
7. **Prefer smaller models** — Models under 1B parameters work best in browser
8. **Check onnx-community** first — Pre-converted models are optimized and tested
