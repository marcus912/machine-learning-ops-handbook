# Chapter 3: TPU Architecture

## What is a TPU?

A Tensor Processing Unit (TPU) is Google's custom-designed application-specific integrated circuit (ASIC) for accelerating machine learning workloads. Unlike NVIDIA GPUs, which are general-purpose parallel processors adapted for ML, TPUs are purpose-built from the ground up for matrix operations.

TPUs are only available through Google Cloud Platform (GCP). You cannot buy a TPU — they're a cloud service.

## TPU vs GPU

| Aspect | NVIDIA GPU (A100) | Google TPU v4 |
|--------|-------------------|---------------|
| **Design philosophy** | General-purpose parallel processor | Purpose-built for ML |
| **Core unit** | CUDA Cores + Tensor Cores | Matrix Multiply Unit (MXU) |
| **Memory** | 80 GB HBM2e | 32 GB HBM |
| **Memory bandwidth** | 2,039 GB/s | 1,200 GB/s |
| **Peak TFLOPS (BF16)** | 312 | 275 |
| **Interconnect** | NVLink (900 GB/s for H100) | ICI (Inter-Chip Interconnect) |
| **Programming model** | CUDA (imperative) | XLA (graph-compiled) |
| **Availability** | Any cloud + on-premises | Google Cloud only |
| **Frameworks** | PyTorch, TensorFlow, JAX, etc. | TensorFlow, JAX, PyTorch/XLA |
| **Price model** | Per-hour instances | Per-hour, preemptible, on-demand |

### When to Use TPUs

**TPUs excel at:**
- Large-scale training (especially Transformer models)
- Workloads dominated by dense matrix multiplications
- TensorFlow and JAX workflows
- Training on very large batch sizes
- Google Cloud-native deployments

**GPUs are better for:**
- Dynamic/irregular computation graphs
- Custom CUDA kernels
- Small-scale experiments
- Inference with variable batch sizes
- Workloads with lots of branching logic
- Multi-cloud or on-premises deployments

## TPU Hardware Architecture

### Matrix Multiply Unit (MXU)

The MXU is the heart of a TPU. It's a systolic array — a grid of multiply-accumulate units that data flows through in a wave pattern.

```
Systolic Array (simplified 4x4, actual MXU is 128x128):

Weight matrix flows down ↓
                col0  col1  col2  col3
               ┌─────┬─────┬─────┬─────┐
Input flows →  │ MAC │ MAC │ MAC │ MAC │ → Partial sums
               ├─────┼─────┼─────┼─────┤    accumulate
               │ MAC │ MAC │ MAC │ MAC │ →   as data
               ├─────┼─────┼─────┼─────┤    flows
               │ MAC │ MAC │ MAC │ MAC │ →   through
               ├─────┼─────┼─────┼─────┤
               │ MAC │ MAC │ MAC │ MAC │ → Output
               └─────┴─────┴─────┴─────┘

MAC = Multiply-Accumulate unit
Each MAC: output = input × weight + accumulated_sum
```

Key properties of the MXU:
- **128 × 128** multiply-accumulate units in TPU v2/v3/v4
- Processes matrix multiplications in a **systolic** pattern (data flows through in waves)
- Operates on **BF16** (bfloat16) inputs with **FP32** accumulation
- Extremely efficient for large, regular matrix operations
- Less efficient for small matrices or irregular shapes

### TPU Chip Components

```
TPU Chip
├── MXU (Matrix Multiply Unit)
│   └── 128×128 systolic array
├── VPU (Vector Processing Unit)
│   └── Handles element-wise ops: activations, normalization, softmax
├── SFU (Scalar/Special Function Unit)
│   └── Control flow, addressing
├── HBM (High Bandwidth Memory)
│   └── 32 GB (v4), stores weights and activations
├── CMEM (Common Memory)
│   └── On-chip scratchpad memory
└── ICI (Inter-Chip Interconnect)
    └── High-speed link to other TPU chips
```

### TPU Generations

| Generation | Year | TFLOPS (BF16) | HBM | ICI Topology | Key Feature |
|------------|------|---------------|-----|-------------|-------------|
| TPU v2 | 2017 | 46 per chip | 8 GB | 2D torus | First cloud TPU |
| TPU v3 | 2018 | 123 per chip | 16 GB | 2D torus | Liquid cooling, 2x perf |
| TPU v4 | 2021 | 275 per chip | 32 GB | 3D torus | 4096-chip pods |
| TPU v5e | 2023 | 197 per chip | 16 GB | 2D torus | Cost-optimized for inference |
| TPU v5p | 2023 | 459 per chip | 95 GB | 3D torus | Training-optimized, 8960 chips/pod |
| TPU v6e (Trillium) | 2024 | 918 per chip | 32 GB | 3D torus | 4.7x improvement over v5e |

### TPU Pods

TPUs are designed to scale by connecting many chips via the Inter-Chip Interconnect (ICI).

```
TPU Pod (e.g., TPU v4 pod):
┌─────┐  ICI  ┌─────┐  ICI  ┌─────┐  ICI  ┌─────┐
│ TPU │───────│ TPU │───────│ TPU │───────│ TPU │
│ v4  │       │ v4  │       │ v4  │       │ v4  │
└──┬──┘       └──┬──┘       └──┬──┘       └──┬──┘
   │ICI          │ICI          │ICI          │ICI
┌──┴──┐       ┌──┴──┐       ┌──┴──┐       ┌──┴──┐
│ TPU │───────│ TPU │───────│ TPU │───────│ TPU │
│ v4  │       │ v4  │       │ v4  │       │ v4  │
└─────┘       └─────┘       └─────┘       └─────┘
... (up to 4096 chips in a v4 pod)
```

A **TPU slice** is a subset of a pod that you allocate. Common sizes:
- `v4-8` = 1 host, 4 chips (8 TensorCores)
- `v4-32` = 4 hosts, 16 chips
- `v4-128` = 16 hosts, 64 chips
- Up to `v4-8192` = full pod

## XLA (Accelerated Linear Algebra)

XLA is the compiler that makes TPUs work. It's also increasingly used with GPUs. Understanding XLA is critical for debugging TPU programs.

### How XLA Works

```
Python Code (PyTorch/JAX/TensorFlow)
         │
         ▼
    HLO (High-Level Operations) IR
         │
         ▼
    XLA Compiler
    ├── Graph optimizations (op fusion, constant folding)
    ├── Memory planning (buffer allocation)
    ├── Layout assignment (data format for hardware)
    └── Code generation (TPU/GPU machine code)
         │
         ▼
    Optimized executable for target hardware
```

### XLA Compilation: Tracing vs Eager

**Eager mode** (default PyTorch, TensorFlow eager):
- Operations execute immediately, one at a time
- Easy to debug (print statements work, standard Python debugger works)
- No whole-graph optimization

**XLA/Traced mode** (TPUs, `torch.compile`, `jax.jit`, `tf.function`):
- The framework traces your Python code to capture the computation graph
- XLA compiles the entire graph at once
- Enables cross-operation optimizations (kernel fusion, memory planning)
- Harder to debug (execution is deferred)

```python
# JAX example: jit compilation
import jax
import jax.numpy as jnp

@jax.jit
def matrix_multiply(a, b):
    return jnp.dot(a, b)

# First call: XLA compiles the function (slow)
result = matrix_multiply(jnp.ones((1024, 1024)), jnp.ones((1024, 1024)))

# Subsequent calls: uses cached compiled version (fast)
result = matrix_multiply(jnp.ones((1024, 1024)), jnp.ones((1024, 1024)))
```

### Why XLA Compilation Matters for Debugging

1. **Recompilation:** XLA recompiles when input shapes change. Dynamic shapes cause repeated compilation, destroying performance.
2. **Error messages:** Errors reference XLA/HLO operations, not your Python code. You need to map back.
3. **Numerical differences:** XLA may reorder operations or fuse kernels, causing slight numerical differences vs eager mode.
4. **Graph breaks:** Operations that can't be compiled (like data-dependent shapes) force "graph breaks" that reduce optimization.

### Inspecting XLA Compilation

```python
# JAX: print HLO
from jax import make_jaxpr

def my_fn(x):
    return jax.nn.relu(x @ x.T + 1.0)

print(make_jaxpr(my_fn)(jnp.ones((4, 4))))
# Outputs the JAX expression (jaxpr) showing the computation graph
```

```python
# PyTorch/XLA: enable debug logging
import torch_xla.debug.metrics as met

# After running some operations:
print(met.metrics_report())
# Shows compilation count, execution time, data transfer metrics
```

```bash
# TensorFlow: dump XLA HLO
export XLA_FLAGS="--xla_dump_to=/tmp/xla_dump --xla_dump_hlo_as_text"
python train.py
# Inspect /tmp/xla_dump/ for .txt files showing HLO graphs
```

## BFloat16 (BF16)

TPUs natively use BF16, which is different from the FP16 used by NVIDIA GPUs.

```
FP32:  [1 sign] [8 exponent] [23 mantissa]  — Full precision
FP16:  [1 sign] [5 exponent] [10 mantissa]  — Narrow range, more precision
BF16:  [1 sign] [8 exponent] [ 7 mantissa]  — Same range as FP32, less precision
```

| Format | Range | Precision | Use Case |
|--------|-------|-----------|----------|
| FP32 | ±3.4 × 10³⁸ | ~7 decimal digits | Reference precision |
| FP16 | ±65,504 | ~3 decimal digits | NVIDIA mixed precision |
| BF16 | ±3.4 × 10³⁸ | ~2 decimal digits | TPU native, increasingly used on GPUs |

**Why BF16 matters:**
- Same **exponent range** as FP32, so you rarely get overflow/underflow
- Less precision than FP16 but more numerically stable for training
- No loss scaling needed (unlike FP16 mixed precision)
- All TPU generations, NVIDIA Ampere+, and recent AMD GPUs support BF16

## PyTorch/XLA

PyTorch/XLA allows PyTorch code to run on TPUs.

### Basic TPU Usage with PyTorch/XLA

```python
import torch
import torch_xla
import torch_xla.core.xla_model as xm

# Get TPU device
device = xm.xla_device()

# Move model and data to TPU
model = MyModel().to(device)
data = torch.randn(32, 784).to(device)

# Forward pass (traced, not executed yet)
output = model(data)
loss = loss_fn(output, target.to(device))

# Backward pass
loss.backward()

# Optimizer step
optimizer.step()

# CRITICAL: Mark step to execute accumulated operations
xm.mark_step()
```

### Key Differences from CUDA PyTorch

| CUDA PyTorch | PyTorch/XLA (TPU) |
|-------------|-------------------|
| Operations execute eagerly | Operations are traced and compiled |
| `tensor.cuda()` | `tensor.to(xm.xla_device())` |
| Errors appear immediately | Errors may appear at `xm.mark_step()` |
| Dynamic shapes are fine | Dynamic shapes cause recompilation |
| `print(tensor)` works normally | `print(tensor)` triggers sync + transfer |
| `.item()` is cheap | `.item()` forces device sync (slow) |

### Common PyTorch/XLA Pitfalls

```python
# BAD: Dynamic tensor shapes cause recompilation
for i, batch in enumerate(dataloader):
    if len(batch) != expected_batch_size:
        # Different batch size → XLA recompiles!
        pass

# GOOD: Pad or drop last batch to maintain consistent shapes
dataloader = DataLoader(dataset, batch_size=32, drop_last=True)

# BAD: Accessing tensor values forces device sync
loss_value = loss.item()  # Transfers data from TPU to CPU, blocks execution
print(f"Loss: {loss_value}")

# GOOD: Print only periodically, or use xm.add_step_closure
xm.add_step_closure(lambda: print(f"Loss: {loss.item()}"))

# BAD: Python control flow dependent on tensor values
if output.sum() > threshold:  # Forces sync to evaluate condition
    do_something()

# GOOD: Keep control flow independent of tensor values
# Or use XLA-compatible control flow
```

## TPU Debugging Tools

### TPU Profiler

```python
# TensorFlow TPU profiling
import tensorflow as tf

tf.profiler.experimental.start('logdir')
# ... training code ...
tf.profiler.experimental.stop()

# Then view in TensorBoard:
# tensorboard --logdir=logdir
```

```python
# JAX TPU profiling
import jax

jax.profiler.start_trace("/tmp/jax_trace")
# ... computation ...
jax.profiler.stop_trace()

# View in TensorBoard or Perfetto
```

### TPU Debugging Environment Variables

```bash
# XLA debug flags
export XLA_FLAGS="--xla_dump_to=/tmp/xla_dump"  # Dump compilation artifacts
export XLA_FLAGS="--xla_force_host_platform_device_count=8"  # Simulate TPU on CPU

# PyTorch/XLA debugging
export PT_XLA_DEBUG=1              # Enable debug mode
export XLA_IR_DEBUG=1              # Print XLA IR
export XLA_HLO_DEBUG=1             # Print HLO
export XLA_SAVE_TENSORS_FILE=/tmp/tensors.txt  # Save tensor info

# TPU runtime flags
export TPU_CHIPS_PER_HOST_BOUNDS=2,2,1  # TPU topology specification
export TPU_HOST_BOUNDS=1,1,1
```

## Key Takeaways

1. **TPUs are purpose-built for ML** with systolic array MXUs optimized for matrix multiplications — not general-purpose processors like GPUs.
2. **XLA compilation** is the fundamental difference in programming model: code is traced and compiled, not executed eagerly. This enables optimization but makes debugging harder.
3. **BF16** is the native TPU data type. It has the same range as FP32 but less precision, making it more numerically stable than FP16 for training.
4. **Dynamic shapes are the enemy** on TPUs. Each new shape triggers XLA recompilation, which can make training extremely slow.
5. **TPU Pods** scale to thousands of chips connected by high-bandwidth ICI, enabling training runs that would require complex networking on GPU clusters.
6. **Debugging on TPUs** requires different tools and techniques than GPUs — you work with XLA dumps, profiler traces, and PyTorch/XLA metrics rather than nvidia-smi and CUDA-GDB.
