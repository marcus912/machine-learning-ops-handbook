# GPU/TPU Debugging Handbook

A beginner-friendly guide to understanding, debugging, and optimizing GPU and TPU workloads for machine learning. Written for engineers preparing to work with accelerated computing in production environments.

## Table of Contents

| Chapter | Title | Description |
|---------|-------|-------------|
| [01](01-gpu-fundamentals.md) | GPU Fundamentals | GPU architecture, CUDA programming model, memory hierarchy, and how GPUs differ from CPUs |
| [02](02-nvidia-tools.md) | NVIDIA Debugging Tools | nvidia-smi, CUDA-GDB, NCCL debugging, and essential command-line tools |
| [03](03-tpu-architecture.md) | TPU Architecture | TPU vs GPU, Matrix Multiply Unit (MXU), XLA compilation, and TPU generations |
| [04](04-common-errors.md) | Common Errors & Fixes | OOM errors, CUDA errors, XLA failures, distributed training errors with solutions |
| [05](05-distributed-training.md) | Distributed Training | Data/model/pipeline parallelism, DDP, FSDP, multi-GPU and multi-node training |
| [06](06-profiling-tools.md) | Profiling Tools | PyTorch Profiler, TensorBoard, NVIDIA Nsight Systems, and performance analysis |
| [07](07-gcp-gpu-tpu.md) | GCP GPU/TPU Infrastructure | Compute Engine GPU VMs, Cloud TPU, GKE with GPUs, Deep Learning VMs, and multislice training |
| [08](08-memory-optimization.md) | Memory Optimization | Gradient checkpointing, mixed precision, DeepSpeed ZeRO, and memory-efficient training |
| [09](09-best-practices.md) | Best Practices | Debugging workflows, framework-specific tips, and a structured learning path |
| [10](10-gcp-vertex-ai-ecosystem.md) | GCP Vertex AI Ecosystem | Vertex AI Workbench, Colab Enterprise, TensorBoard, model serving, HP tuning, and AI Hypercomputer |
| [11](11-model-serving-and-inference.md) | Model Serving & Inference | Serving frameworks (TF Serving, Triton, vLLM), quantization, pruning, batching strategies, and API patterns |

### Related Sections

Additional ML topics have been organized into separate folders:

- **[ML Fundamentals](../ml-fundamentals/)** - Optimizers, regularization, transfer learning, data pipelines, feature stores
- **[LLM & GenAI](../llm-and-genai/)** - Transformer architecture, attention, tokenization, RAG, vector search
- **[MLOps](../mlops/)** - CI/CD for ML, model monitoring, A/B testing, case study framework

## Recommended Reading Order

**If you're completely new to GPU/ML:**
1. Start with [Chapter 01](01-gpu-fundamentals.md) to understand GPU hardware
2. Read [Chapter 04](04-common-errors.md) to recognize common problems early
3. Move to [Chapter 02](02-nvidia-tools.md) for hands-on debugging tools
4. Continue sequentially from Chapter 03 onward

**If you already know GPU basics:**
1. Skim [Chapter 01](01-gpu-fundamentals.md) for any gaps
2. Jump to [Chapter 04](04-common-errors.md) and [Chapter 05](05-distributed-training.md)
3. Read [Chapter 08](08-memory-optimization.md) for optimization techniques
4. Fill in remaining chapters as needed

**If you're preparing for a Google interview:**
1. Focus on Chapters 03, 04, 05, 07 (TPU knowledge, errors, distributed, GCP infra)
2. Read Chapter 10 for Vertex AI ecosystem and managed ML services
3. Review Chapter 08 for optimization strategies
4. Read Chapter 11 for model serving and inference optimization
5. See [ML Fundamentals](../ml-fundamentals/), [LLM & GenAI](../llm-and-genai/), and [MLOps](../mlops/) folders for additional topics
6. Use Chapter 09 as a checklist

## 11-Week Learning Path

This learning path assumes ~5 hours per week of study and hands-on practice.

| Week | Focus | Chapters | Activities |
|------|-------|----------|------------|
| 1 | GPU Basics | Ch 01 | Run `nvidia-smi`, understand GPU specs |
| 2 | NVIDIA Tools | Ch 02 | Practice CUDA-GDB, monitor GPU utilization |
| 3 | TPU Concepts | Ch 03 | Read Google TPU docs, compare TPU vs GPU |
| 4 | Error Recognition | Ch 04 (first half) | Deliberately trigger OOM, fix it |
| 5 | Error Resolution | Ch 04 (second half) | Practice debugging XLA and distributed errors |
| 6 | Distributed Training | Ch 05 | Run DDP training on multi-GPU setup |
| 7 | Profiling | Ch 06 | Profile a real training loop, find bottlenecks |
| 8 | GCP Infrastructure | Ch 07 | Set up GPU VM, Cloud TPU, GKE GPU node pool |
| 9 | Vertex AI Ecosystem | Ch 10 | Explore Workbench notebooks, set up TensorBoard, deploy endpoint |
| 10 | Memory Optimization | Ch 08 | Implement mixed precision, gradient checkpointing |
| 11 | Review & Practice | Ch 09 + All | End-to-end debugging exercise, review best practices |

## Prerequisites

- Basic Python programming
- Familiarity with PyTorch or TensorFlow (beginner level is fine)
- Access to a machine with an NVIDIA GPU (or Google Colab free tier)
- Basic Linux command-line knowledge
