# Chapter 11: Model Serving & Inference

This chapter covers production model serving — frameworks, optimization techniques, scaling strategies, and networking patterns for deploying ML models as reliable, low-latency services.

## Serving Frameworks

### Comparison

| Framework | Protocol | Multi-Framework | Dynamic Batching | Best For |
|-----------|----------|-----------------|------------------|----------|
| **TF Serving** | gRPC, REST | TensorFlow only | Yes | TF models in production, Google ecosystem |
| **TorchServe** | gRPC, REST | PyTorch only | Yes | PyTorch models, simple setup, model versioning |
| **Triton Inference Server** | gRPC, REST | TF, PyTorch, ONNX, TensorRT, Python | Yes (advanced) | High-throughput multi-model serving, GPU sharing |
| **vLLM** | OpenAI-compatible API | PyTorch (LLMs) | Continuous batching | LLM serving with high throughput |

### TF Serving

```bash
# Serve a SavedModel
docker run -p 8501:8501 \
  --mount type=bind,source=/models/my_model,target=/models/my_model \
  -e MODEL_NAME=my_model \
  tensorflow/serving

# REST prediction
curl -d '{"instances": [[1.0, 2.0, 3.0]]}' \
  http://localhost:8501/v1/models/my_model:predict
```

Key features:
- Native SavedModel support with automatic batching
- Model versioning (serves latest version by default, can serve multiple)
- gRPC endpoint on port 8500, REST on 8501

### TorchServe

```bash
# Package model
torch-model-archiver --model-name resnet \
  --version 1.0 \
  --model-file model.py \
  --serialized-file resnet50.pt \
  --handler image_classifier

# Start server
torchserve --start --model-store model_store --models resnet=resnet.mar
```

Key features:
- `.mar` (Model Archive) packaging format
- Built-in handlers for common tasks (image classification, object detection, text)
- Metrics endpoint, model versioning, A/B testing support

### Triton Inference Server

```
# Model repository structure
models/
├── resnet50/
│   ├── config.pbtxt
│   └── 1/
│       └── model.onnx
└── bert/
    ├── config.pbtxt
    └── 1/
        └── model.pt
```

```bash
# Launch Triton
docker run --gpus=1 -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v /models:/models \
  nvcr.io/nvidia/tritonserver:latest \
  tritonserver --model-repository=/models
```

Key features:
- Concurrent model execution on shared GPUs
- Model ensembles (chain models in a DAG)
- Dynamic batching with configurable delay and max batch size
- Instance groups for running multiple model copies per GPU

### vLLM

```python
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-2-7b-chat-hf")
params = SamplingParams(temperature=0.8, max_tokens=256)
outputs = llm.generate(["Explain gradient descent"], params)
```

```bash
# OpenAI-compatible API server
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-2-7b-chat-hf \
  --port 8000
```

Key features:
- **PagedAttention** — manages KV cache like virtual memory pages, eliminates memory waste
- **Continuous batching** — adds new requests to running batch without waiting for all to finish
- Tensor parallelism for serving large models across multiple GPUs

## Model Optimization for Inference

### Quantization

Reduce numerical precision to decrease model size and increase inference speed.

| Approach | Precision | When to Use | Accuracy Impact |
|----------|-----------|-------------|-----------------|
| **Post-training quantization (PTQ)** | FP32 → INT8 | Quick wins, when small accuracy drop is acceptable | Slight degradation |
| **Quantization-aware training (QAT)** | FP32 → INT8 | When accuracy is critical | Minimal (model learns to compensate) |
| **FP16/BF16 inference** | FP32 → FP16/BF16 | Default first step, nearly free | Usually negligible |
| **GPTQ / AWQ** | 4-bit | LLM compression | Model-dependent |

```python
# PyTorch dynamic quantization (PTQ)
import torch

quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# TensorRT optimization (NVIDIA)
import torch_tensorrt
trt_model = torch_tensorrt.compile(model,
    inputs=[torch_tensorrt.Input(shape=[1, 3, 224, 224])],
    enabled_precisions={torch.half}
)
```

### Pruning

Remove unnecessary weights to reduce model size and computation.

**Structured pruning** — Remove entire channels, attention heads, or layers. Hardware-friendly (actual speedup without sparse tensor support).

**Unstructured pruning** — Zero out individual weights based on magnitude. Higher compression ratios but requires sparse computation support for actual speedup.

```python
import torch.nn.utils.prune as prune

# Unstructured: prune 30% of weights by magnitude
prune.l1_unstructured(layer, name='weight', amount=0.3)

# Structured: prune 20% of output channels
prune.ln_structured(layer, name='weight', amount=0.2, n=2, dim=0)
```

### Knowledge Distillation

Train a smaller "student" model to mimic a larger "teacher" model.

```
Teacher (large, accurate) ──→ Soft labels (probability distributions)
                                        │
Student (small, fast) ◄─────────────────┘
  trains on: α × hard_label_loss + (1-α) × distillation_loss
```

- **Temperature scaling** — Soften teacher outputs to reveal more information about class relationships
- Trade training cost for inference speed — student can be 3-10x smaller
- Common in production: distill BERT-large → BERT-small, or GPT-4 → smaller model

### ONNX Export

Convert models to a framework-agnostic format for cross-platform optimization.

```python
import torch.onnx

torch.onnx.export(model, dummy_input, "model.onnx",
    input_names=['input'], output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
```

Benefits:
- Run PyTorch models in TF Serving or Triton without TensorFlow dependency
- ONNX Runtime applies graph-level optimizations (operator fusion, constant folding)
- Hardware-specific backends (TensorRT, OpenVINO, CoreML)

## Inference Patterns

### Batch vs Real-Time vs Near-Real-Time

| Pattern | Latency | Throughput | Cost | Use Cases |
|---------|---------|------------|------|-----------|
| **Real-time** | < 100ms | Lower | Higher (always-on) | User-facing APIs, chatbots, search ranking |
| **Near-real-time** | Seconds | Medium | Medium | Streaming features, fraud detection |
| **Batch** | Minutes–hours | Highest | Lowest | Recommendations precompute, reports, embeddings |

### Dynamic Batching

Accumulate incoming requests into batches before processing to maximize GPU utilization.

```
Request 1 ──┐
Request 2 ──┤  Batch window    ┌─────────┐    ┌──────────┐
Request 3 ──┼─ (e.g., 10ms) ──►  Batch   ├───►│  GPU     │
Request 4 ──┤                  │ [1,2,3,4]│    │ Process  │
            └──────────────────└─────────┘    └──────────┘
```

Configuration trade-offs:
- **Larger max batch size** → Higher throughput, higher latency
- **Longer batch window** → More requests batched, higher latency
- **Preferred batch size** → Process immediately when reached, even before timeout

Triton config example:
```
dynamic_batching {
  preferred_batch_size: [4, 8]
  max_queue_delay_microseconds: 100
}
```

### Cold Start and Autoscaling

| Strategy | Trade-off |
|----------|-----------|
| Minimum replicas (keep-warm) | Higher baseline cost, no cold start |
| Model caching on disk | Faster load than downloading, still needs GPU transfer |
| Pre-loaded model pools | Memory cost for idle models, instant switching |
| Queue-depth-based scaling | Scale on actual demand, not just CPU/memory |
| Predictive autoscaling | Scale up before expected traffic spikes |

**Serverless ML trade-offs:**
- Pros: pay per request, auto-scales to zero
- Cons: cold start latency (seconds for large models), limited GPU options
- Best for: infrequent, bursty workloads where cost matters more than latency

## Networking for Model Serving

### HTTP/REST vs gRPC

| Aspect | HTTP/REST | gRPC |
|--------|-----------|------|
| Serialization | JSON (text) | Protobuf (binary) |
| Latency | Higher (~1.5-3x gRPC) | Lower |
| Payload size | Larger | ~30% smaller |
| Streaming | Limited (SSE, WebSocket) | Native bidirectional |
| Browser support | Full | Limited (grpc-web) |
| Debugging | Easy (curl, Postman) | Harder (need grpcurl) |
| Type safety | Weak (JSON schema optional) | Strong (protobuf definitions) |

**Use REST when:** public APIs, browser clients, debugging/prototyping, simple request/response.

**Use gRPC when:** internal microservices, low-latency model serving, streaming inference, high throughput.

### API Gateway Patterns

**Load balancing across model replicas:**
- Round-robin for homogeneous replicas
- Least-connections for variable-latency models
- Weighted routing for A/B testing different model versions

**A/B routing for models:**
```
Client → API Gateway → 90% → Model v1 (production)
                     → 10% → Model v2 (experiment)
```

**Rate limiting:** Protect GPU-backed endpoints from overload. Per-client limits, global limits, burst allowances.

**API versioning strategies:**
- URL path versioning: `/v1/predict`, `/v2/predict`
- Header versioning: `X-Model-Version: 2`
- Model version in request body (flexible, harder to route)

**Health checks and readiness probes:**
```yaml
# Kubernetes readiness probe for model server
readinessProbe:
  httpGet:
    path: /v2/health/ready   # Triton readiness endpoint
    port: 8000
  initialDelaySeconds: 30     # Time for model loading
  periodSeconds: 10
```
