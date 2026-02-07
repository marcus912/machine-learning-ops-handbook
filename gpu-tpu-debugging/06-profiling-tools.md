# Chapter 6: Profiling Tools

## Why Profile?

Profiling answers the question: **where is time (or memory) being spent?**

Without profiling, debugging performance is guesswork. Common misconceptions:
- "The GPU is the bottleneck" — often it's data loading or CPU preprocessing
- "The model is too large" — it might be memory fragmentation, not model size
- "Training is slow because of the network" — could be a single slow operation

Profiling gives you facts instead of guesses.

## PyTorch Profiler

The built-in PyTorch profiler traces CPU and GPU operations with minimal setup.

### Basic Usage

```python
import torch
from torch.profiler import profile, record_function, ProfilerActivity

model = MyModel().cuda()
data = torch.randn(32, 3, 224, 224).cuda()

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    with record_function("forward_pass"):
        output = model(data)
    with record_function("backward_pass"):
        output.sum().backward()

# Print summary sorted by CUDA time
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
```

**Example output:**
```
Name                       CPU total   CUDA total   # Calls
-------------------------  ----------  -----------  --------
forward_pass               45.2ms      38.1ms       1
  aten::conv2d             12.3ms      18.5ms       5
  aten::batch_norm         4.1ms       6.2ms        5
  aten::relu               1.2ms       2.1ms        5
  aten::linear             3.4ms       8.3ms        2
backward_pass              62.1ms      55.3ms       1
  aten::conv2d_backward    22.1ms      28.4ms       5
  ...
```

### Profiling Training Loops

```python
from torch.profiler import profile, schedule, tensorboard_trace_handler

# Profile a training loop with warm-up
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(
        wait=1,    # Skip first batch (warm-up)
        warmup=1,  # Profiler warm-up (included but results discarded)
        active=3,  # Profile these batches
        repeat=1,  # Number of cycles
    ),
    on_trace_ready=tensorboard_trace_handler("./profiler_logs"),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    for step, (data, target) in enumerate(dataloader):
        if step >= 1 + 1 + 3:  # wait + warmup + active
            break

        data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        prof.step()  # Signal profiler to advance to next step

# View in TensorBoard:
# tensorboard --logdir=./profiler_logs
```

### Reading Profiler Output

Key columns to understand:

| Column | Meaning | What to Look For |
|--------|---------|-----------------|
| **CPU total** | Total CPU time including child ops | High CPU time with low CUDA time = CPU bottleneck |
| **CUDA total** | Total GPU time | Largest values are optimization targets |
| **Self CPU** | CPU time excluding children | Identifies the actual expensive operations |
| **Self CUDA** | GPU time excluding children | The real GPU cost of each op |
| **# Calls** | Number of invocations | Unexpected high count = bug or unnecessary recomputation |
| **CPU Mem** | CPU memory allocated | Memory leaks show increasing allocation |
| **CUDA Mem** | GPU memory allocated | Track memory growth |

### Memory Profiling

```python
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    profile_memory=True,
) as prof:
    model = MyModel().cuda()
    data = torch.randn(32, 784).cuda()
    output = model(data)
    loss = output.sum()
    loss.backward()

# Sort by GPU memory usage
print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=15))
```

### Custom Annotations

```python
from torch.profiler import record_function

class MyModel(torch.nn.Module):
    def forward(self, x):
        with record_function("encoder"):
            x = self.encoder(x)

        with record_function("attention"):
            x = self.attention(x)

        with record_function("decoder"):
            x = self.decoder(x)

        return x

# These labels appear in profiler output and TensorBoard trace viewer
```

## TensorBoard for Profiling

TensorBoard provides visual profiling analysis for both PyTorch and TensorFlow.

### PyTorch + TensorBoard

```bash
# Install if needed
pip install tensorboard torch-tb-profiler

# After running profiler with tensorboard_trace_handler:
tensorboard --logdir=./profiler_logs --port=6006
```

**TensorBoard Profiler Views:**

1. **Overview** — Summary of step time breakdown (kernel execution, communication, idle time)
2. **Operator View** — Table of operations sorted by time (like the text output but interactive)
3. **GPU Kernel View** — Individual CUDA kernels with launch parameters
4. **Trace View** — Timeline showing CPU and GPU operations (most useful for finding gaps)
5. **Memory View** — Memory allocation over time

### TensorFlow + TensorBoard

```python
import tensorflow as tf

# TensorFlow profiling
tf.profiler.experimental.start('logdir')

# Training steps to profile
for step in range(100):
    train_step(data, labels)

tf.profiler.experimental.stop()
```

```bash
tensorboard --logdir=logdir
```

### Reading the Trace View

The trace view shows a timeline of operations:

```
Timeline (simplified):
CPU Thread 0: [DataLoad][  H2D  ][  idle  ][DataLoad][  H2D  ][  idle  ]
GPU Stream 0:           [conv2d ][  relu  ]          [conv2d ][  relu  ]
GPU Stream 1:                    [bn_fwd ]                    [bn_fwd ]

H2D = Host-to-Device data transfer
```

**What to look for:**
- **Gaps between GPU operations** = GPU is idle (likely waiting for CPU or data transfer)
- **Long H2D transfers** = Data transfer bottleneck (increase `num_workers` in DataLoader)
- **CPU operations overlapping GPU idle** = CPU preprocessing is the bottleneck
- **One very long GPU kernel** = That specific operation dominates training time

## NVIDIA Nsight Systems

Nsight Systems provides system-wide profiling including CPU, GPU, network, and OS-level events.

### Basic Usage

```bash
# Profile a Python training script
nsys profile --trace=cuda,nvtx,osrt \
    --output=profile_report \
    --force-overwrite true \
    python train.py

# Generate summary statistics
nsys stats profile_report.nsys-rep

# Open in Nsight Systems GUI for visual analysis
nsys-ui profile_report.nsys-rep
```

### Common Profiling Scenarios

```bash
# Profile only specific time range (seconds)
nsys profile --trace=cuda,nvtx \
    --delay=10 --duration=30 \
    python train.py

# Capture CUDA API calls and kernel launches
nsys profile --trace=cuda,cudnn,cublas \
    --cuda-memory-usage=true \
    python train.py

# Profile distributed training (one profile per rank)
nsys profile --trace=cuda,nvtx \
    --output=profile_rank_%q{RANK} \
    torchrun --nproc_per_node=4 train.py
```

### Adding NVTX Annotations

NVTX (NVIDIA Tools Extension) lets you annotate your code for Nsight:

```python
import torch

# PyTorch automatically adds NVTX ranges for autograd operations
# You can add custom ranges:

import nvtx  # pip install nvtx

@nvtx.annotate("training_step", color="blue")
def training_step(model, data, target):
    with nvtx.annotate("forward", color="green"):
        output = model(data)
        loss = criterion(output, target)

    with nvtx.annotate("backward", color="red"):
        loss.backward()

    with nvtx.annotate("optimizer", color="yellow"):
        optimizer.step()
        optimizer.zero_grad()

    return loss

# Or use torch's built-in NVTX
with torch.cuda.nvtx.range("my_operation"):
    result = complex_operation()
```

### Nsight Systems vs PyTorch Profiler

| Feature | PyTorch Profiler | Nsight Systems |
|---------|-----------------|----------------|
| Ease of use | Easy (Python API) | Moderate (CLI + GUI) |
| Scope | PyTorch operations | System-wide (CPU, GPU, OS, network) |
| Distributed | Per-process | Can correlate across processes |
| Overhead | Low-moderate | Very low |
| Output | Text + TensorBoard | nsys-rep (visual GUI) |
| Best for | PyTorch-specific optimization | System-level bottlenecks, multi-GPU |

## NVIDIA Nsight Compute

Nsight Compute profiles individual CUDA kernels in detail. Use this when you've identified a slow kernel and need to understand why.

```bash
# Profile all kernels
ncu --target-processes all python train.py

# Profile a specific kernel by name
ncu --kernel-name "volta_fp16_s884gemm" python train.py

# Generate roofline analysis
ncu --set full --target-processes all python train.py
```

**When to use Nsight Compute:**
- You've already identified a slow CUDA kernel using Nsight Systems or PyTorch Profiler
- You need to understand if a kernel is compute-bound or memory-bound
- You're optimizing a custom CUDA kernel

## Identifying Common Bottlenecks

### Bottleneck 1: Data Loading

**Symptoms:**
- GPU utilization is low (< 50%)
- Profiler shows gaps between GPU operations
- CPU is at 100% during data loading

**Diagnosis:**
```python
import time

# Simple data loading timing
start = time.time()
for i, (data, target) in enumerate(dataloader):
    if i >= 10:
        break
    data_time = time.time() - start

    # GPU computation
    gpu_start = time.time()
    output = model(data.cuda())
    loss = criterion(output, target.cuda())
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    torch.cuda.synchronize()  # Wait for GPU to finish
    gpu_time = time.time() - gpu_start

    print(f"Step {i}: data={data_time:.3f}s, gpu={gpu_time:.3f}s")
    start = time.time()

# If data_time >> gpu_time, data loading is the bottleneck
```

**Fixes:**
```python
dataloader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=8,         # Increase workers (start with 4× num GPUs)
    pin_memory=True,       # Pin memory for faster CPU→GPU transfer
    prefetch_factor=2,     # Pre-fetch 2 batches per worker
    persistent_workers=True,  # Keep workers alive between epochs
)
```

### Bottleneck 2: CPU-GPU Transfer

**Symptoms:**
- Profiler shows long `cudaMemcpy` or `aten::to` operations
- Data is being moved between CPU and GPU frequently

**Fixes:**
```python
# BAD: Moving data per-operation
for batch in dataloader:
    data = batch['input'].cuda()         # Transfer
    mask = batch['mask'].cuda()          # Transfer
    labels = batch['labels'].cuda()      # Transfer
    extra = process_on_cpu(data.cpu())   # Transfer back!
    result = model(data, mask, extra.cuda())  # Transfer again!

# GOOD: Move data once, keep on GPU
for batch in dataloader:
    batch = {k: v.cuda() for k, v in batch.items()}  # One transfer
    result = model(**batch)  # Everything stays on GPU
```

### Bottleneck 3: Small Operations

**Symptoms:**
- Profiler shows many tiny CUDA kernels
- High kernel launch overhead relative to kernel execution time

**Fixes:**
```python
# Use torch.compile to fuse operations (PyTorch 2.0+)
model = torch.compile(model)

# Or use scripting for specific functions
@torch.jit.script
def fused_activation(x):
    return torch.relu(x) * torch.sigmoid(x)
```

### Bottleneck 4: Synchronization Points

**Symptoms:**
- GPU has idle periods
- `.item()`, `print(tensor)`, or `tensor.cpu()` appear in hot path

**Fixes:**
```python
# BAD: Synchronizing every step
for step, (data, target) in enumerate(dataloader):
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    print(f"Loss: {loss.item()}")  # Forces GPU sync every step!

# GOOD: Log periodically
for step, (data, target) in enumerate(dataloader):
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    if step % 100 == 0:
        print(f"Loss: {loss.item()}")  # Sync only every 100 steps
```

## Profiling Checklist

When investigating a performance issue, follow this order:

1. **Check GPU utilization** with `nvidia-smi -l 1`
   - Low utilization → data loading or CPU bottleneck
   - High utilization but slow → kernel efficiency issue

2. **Run PyTorch Profiler** with TensorBoard output
   - Identify the top operations by time
   - Check the trace view for gaps

3. **Check data loading speed** independently
   - Time the dataloader without GPU computation
   - Increase `num_workers`, enable `pin_memory`

4. **Look for unnecessary synchronization**
   - Search for `.item()`, `.cpu()`, `print(tensor)` in training loop
   - Move logging outside the critical path

5. **Profile communication** (distributed training)
   - Check NCCL timing in Nsight Systems
   - Look for stragglers (one slow rank)

6. **Deep-dive with Nsight** if needed
   - Nsight Systems for system-level view
   - Nsight Compute for individual kernel analysis

## Key Takeaways

1. **Profile before optimizing.** Intuition about bottlenecks is frequently wrong.
2. **PyTorch Profiler** is the easiest starting point — use it with TensorBoard for visual analysis.
3. **GPU utilization < 80%** usually means the bottleneck is outside the GPU (data loading, CPU preprocessing, or synchronization).
4. **Data loading** is the most common bottleneck. Increase `num_workers`, enable `pin_memory`, use `persistent_workers`.
5. **Avoid synchronization** in the training loop — `.item()`, `.cpu()`, and `print(tensor)` all force the GPU to sync.
6. **Use Nsight Systems** when you need to see the full picture across CPU, GPU, network, and multiple processes.
