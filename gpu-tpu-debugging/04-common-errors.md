# Chapter 4: Common Errors & Fixes

This chapter catalogs the most common errors you'll encounter when working with GPUs and TPUs for machine learning, organized by category with causes and solutions.

## CUDA Out of Memory (OOM)

The most frequent GPU error in deep learning.

### The Error

```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
(GPU 0; 79.35 GiB total capacity; 76.28 GiB already allocated;
1.45 GiB free; 77.50 GiB reserved in total by PyTorch)
```

### Understanding the Error Message

| Field | Meaning |
|-------|---------|
| `Tried to allocate` | Size of the tensor that failed to allocate |
| `total capacity` | Total GPU VRAM |
| `already allocated` | Memory actively used by PyTorch tensors |
| `free` | Free memory on the GPU |
| `reserved in total by PyTorch` | Memory in PyTorch's caching allocator |

**Important:** `reserved - allocated` = memory PyTorch has cached but isn't actively using. This isn't wasted — it's pre-allocated for efficiency.

### Causes and Solutions

**Cause 1: Batch size too large**
```python
# Reduce batch size
# Before: batch_size = 64
batch_size = 32  # or 16, 8

# Better: use gradient accumulation to maintain effective batch size
effective_batch_size = 64
micro_batch_size = 16
accumulation_steps = effective_batch_size // micro_batch_size  # 4

optimizer.zero_grad()
for i, (data, target) in enumerate(dataloader):
    output = model(data)
    loss = criterion(output, target)
    loss = loss / accumulation_steps  # Scale loss
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**Cause 2: Accumulating gradients unintentionally**
```python
# BAD: tensors on GPU accumulate across iterations
losses = []
for data, target in dataloader:
    output = model(data)
    loss = criterion(output, target)
    losses.append(loss)  # Keeps entire computation graph in memory!

# GOOD: detach from computation graph
losses = []
for data, target in dataloader:
    output = model(data)
    loss = criterion(output, target)
    losses.append(loss.item())  # .item() returns a Python float
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

**Cause 3: Not using `torch.no_grad()` during inference**
```python
# BAD: stores activations for backward pass during inference
output = model(test_data)

# GOOD: no computation graph needed for inference
with torch.no_grad():
    output = model(test_data)

# Also good: use inference_mode for even more optimization
with torch.inference_mode():
    output = model(test_data)
```

**Cause 4: Memory fragmentation**
```python
# Solution 1: Configure PyTorch allocator
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Solution 2: Clear cache periodically
torch.cuda.empty_cache()  # Returns cached memory to CUDA (doesn't free tensors)

# Solution 3: Use memory snapshots to find fragmentation
torch.cuda.memory._record_memory_history()
# ... run code that causes OOM ...
torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
# Visualize at: https://pytorch.org/memory_viz
```

**Cause 5: Model too large for single GPU**
```python
# Solution 1: Use gradient checkpointing (see Chapter 8)
from torch.utils.checkpoint import checkpoint

# Solution 2: Use model parallelism (see Chapter 5)
# Solution 3: Use DeepSpeed ZeRO (see Chapter 8)
# Solution 4: Use mixed precision training (see Chapter 8)
```

### Diagnosing OOM: Step-by-Step

```python
import torch

def print_gpu_memory(tag=""):
    """Print current GPU memory usage."""
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    print(f"[{tag}] Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

# Track memory at each step
print_gpu_memory("Start")

model = MyModel().cuda()
print_gpu_memory("After model load")

data = next(iter(dataloader)).cuda()
print_gpu_memory("After data load")

output = model(data)
print_gpu_memory("After forward pass")  # Activations stored here

loss = criterion(output, target)
loss.backward()
print_gpu_memory("After backward pass")  # Gradients stored here

optimizer.step()
print_gpu_memory("After optimizer step")  # Optimizer states stored here

optimizer.zero_grad()
torch.cuda.empty_cache()
print_gpu_memory("After cleanup")
```

## CUDA Runtime Errors

### CUDA Error: Device-Side Assert Triggered

```
RuntimeError: CUDA error: device-side assert triggered
```

**Cause:** An assertion failed inside a CUDA kernel. Common triggers:
- Index out of bounds (e.g., label index >= number of classes)
- NaN in loss computation
- Invalid input to a CUDA operation

**Debugging:**
```bash
# Step 1: Enable synchronous execution for accurate stack trace
export CUDA_LAUNCH_BLOCKING=1

# Step 2: Enable device-side assertions for detailed message
export TORCH_USE_CUDA_DSA=1

# Step 3: Run again
python train.py
```

**Common specific cause — label out of range:**
```python
# If using CrossEntropyLoss with 10 classes:
# Labels must be in range [0, 9]
print(f"Label min: {labels.min()}, max: {labels.max()}")
print(f"Num classes: {model.num_classes}")
# Fix: ensure labels are in valid range
assert labels.max() < model.num_classes, f"Label {labels.max()} >= num_classes {model.num_classes}"
```

### CUDA Error: An Illegal Memory Access

```
RuntimeError: CUDA error: an illegal memory access was encountered
```

**Cause:** A CUDA kernel accessed memory outside allocated bounds.

**Debugging steps:**
1. Set `CUDA_LAUNCH_BLOCKING=1` for accurate error location
2. Check for indexing errors in custom CUDA kernels
3. Run with `compute-sanitizer` for detailed memory error info:
   ```bash
   compute-sanitizer --tool memcheck python train.py
   ```
4. Check for tensor shape mismatches

### CUDA Error: CUBLAS_STATUS_ALLOC_FAILED

```
RuntimeError: CUDA error: CUBLAS_STATUS_ALLOC_FAILED when calling cublasCreate(handle)
```

**Cause:** cuBLAS couldn't allocate workspace memory.

**Solutions:**
```python
# Reduce batch size or model size to free memory
# Or clear cache before the operation
torch.cuda.empty_cache()
```

### CUDA Error: NCCL Error / Timeout

```
RuntimeError: NCCL error in: /pytorch/torch/lib/c10d/ProcessGroupNCCL.cpp:1234
torch.distributed.DistributedBackendError: NCCL communicator was aborted
```

**See the NCCL section below for distributed training errors.**

## XLA Compilation Errors (TPU)

### Recompilation Storm

**Symptom:** Training is extremely slow, with long pauses between steps.

```python
# PyTorch/XLA: check compilation metrics
import torch_xla.debug.metrics as met
print(met.metrics_report())

# Look for high "CompileTime" and frequent "ExecuteTime" entries
# High "CompilationCacheHit" / "CompilationCacheMiss" ratio indicates recompilation
```

**Cause:** Dynamic tensor shapes trigger XLA recompilation.

**Common triggers:**
```python
# BAD: Variable-length sequences without padding
for batch in dataloader:
    # Each batch may have different sequence lengths
    # → different shapes → recompilation each step
    output = model(batch)

# GOOD: Pad to fixed length
from torch.nn.utils.rnn import pad_sequence
padded_batch = pad_sequence(sequences, batch_first=True, padding_value=0)

# BAD: Last batch is smaller
dataloader = DataLoader(dataset, batch_size=32)  # Last batch might be < 32

# GOOD: Drop the last incomplete batch
dataloader = DataLoader(dataset, batch_size=32, drop_last=True)

# BAD: Data-dependent shapes
mask = tensor > threshold
filtered = tensor[mask]  # Shape depends on data values!

# GOOD: Use where() to maintain shape
filtered = torch.where(tensor > threshold, tensor, torch.zeros_like(tensor))
```

### XLA Type Errors

```
InvalidArgumentError: Shapes must be equal rank, but are 2 and 3
```

**Cause:** Shape mismatch in XLA graph. Unlike eager mode where shapes are checked per-operation, XLA catches these at compile time.

**Debugging:**
```bash
# Dump XLA HLO for inspection
export XLA_FLAGS="--xla_dump_to=/tmp/xla_dump --xla_dump_hlo_as_text"
python train.py

# Look at the .txt files in /tmp/xla_dump/
# Find the operation with mismatched shapes
```

### XLA Slow Operations Fallback

Some operations aren't efficiently supported on TPU and fall back to CPU:

```python
# PyTorch/XLA: check for CPU fallbacks
import torch_xla.debug.metrics as met
# Look for "TransferFromServerTime" — indicates data moving from TPU to CPU
# Any operation that triggers a transfer is a performance bottleneck
```

**Operations that commonly fall back:**
- Scatter/gather with dynamic indices
- Non-contiguous tensor operations
- Complex indexing patterns
- Operations on non-standard dtypes
- Sorting and topk with large dimensions

## Distributed Training Errors

### NCCL Timeout

```
RuntimeError: NCCL communicator was aborted on rank 2.
Original reason for the abort was: Watchdog caught collective operation timeout.
```

**Causes and solutions:**

| Cause | Solution |
|-------|----------|
| One rank crashed | Check all rank logs for the root error |
| Network issue | Check `NCCL_SOCKET_IFNAME`, test connectivity |
| Uneven workload | Ensure all ranks process same number of batches |
| Deadlock from different code paths | Ensure all ranks execute same collective operations |
| Slow rank (straggler) | Increase timeout: `dist.init_process_group(timeout=timedelta(seconds=1800))` |

**Debugging steps:**
```bash
# Step 1: Enable NCCL debug logging
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# Step 2: Check that all ranks are reaching the same collective
# Add logging before and after each collective operation
# Rank 0: "Entering all_reduce at step 42"
# Rank 1: "Entering all_reduce at step 42"
# Rank 2: ← missing? This rank is the problem

# Step 3: Check for asymmetric operations
# Common bug: only rank 0 does something (e.g., logging that triggers a sync)
```

### Address Already in Use

```
RuntimeError: Address already in use
# or
OSError: [Errno 98] Address already in use
```

**Cause:** Previous training run didn't clean up, or another process is using the port.

```bash
# Find and kill the process using the port
lsof -i :29500   # Default PyTorch distributed port
kill -9 <PID>

# Or use a different port
export MASTER_PORT=29501
```

### NCCL Unhandled System Error

```
NCCL WARN Connect to <IP>:<PORT> failed: No route to host
```

**Causes:**
- Firewall blocking inter-node communication
- Wrong network interface selected
- Nodes on different subnets

```bash
# Specify the correct network interface
export NCCL_SOCKET_IFNAME=eth0  # Or the interface connected to your training network

# Test connectivity between nodes
ping <other_node_ip>
nc -zv <other_node_ip> 29500

# Check firewall rules
sudo iptables -L -n | grep 29500
```

### Rank Mismatch / World Size Error

```
RuntimeError: The number of world size (4) is not equal to the expected (8)
```

**Cause:** Mismatch between expected and actual number of processes.

```bash
# Ensure all nodes are launching the correct number of processes
# For 2 nodes × 4 GPUs each:
# Node 0:
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 \
         --master_addr=node0 --master_port=29500 train.py

# Node 1:
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 \
         --master_addr=node0 --master_port=29500 train.py
```

## Numerical Errors

### NaN Loss

```python
# Loss becomes NaN during training

# Step 1: Find where NaN first appears
torch.autograd.set_detect_anomaly(True)  # Slow but pinpoints the operation

# Step 2: Common causes and fixes

# Cause: Learning rate too high
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)  # Reduce LR

# Cause: Division by zero
# BAD:
normalized = x / x.sum(dim=-1, keepdim=True)
# GOOD:
normalized = x / (x.sum(dim=-1, keepdim=True) + 1e-8)

# Cause: Log of zero
# BAD:
log_probs = torch.log(probs)
# GOOD:
log_probs = torch.log(probs + 1e-8)
# Or better:
log_probs = torch.log_softmax(logits, dim=-1)  # Numerically stable

# Cause: Mixed precision overflow (FP16)
# BAD:
scaler = torch.cuda.amp.GradScaler()  # With default settings
# GOOD: Check if loss scale is going to zero
# If loss scale keeps decreasing, there may be genuine numerical issues
```

### NaN/Inf in Gradients

```python
# Check for NaN/Inf gradients
for name, param in model.named_parameters():
    if param.grad is not None:
        if torch.isnan(param.grad).any():
            print(f"NaN gradient in {name}")
        if torch.isinf(param.grad).any():
            print(f"Inf gradient in {name}")

# Solution: Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
# or
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
```

### Loss Not Decreasing

Not an error per se, but a common debugging scenario:

```python
# Debugging checklist for non-converging training:

# 1. Verify data is correct
for batch in dataloader:
    data, labels = batch
    print(f"Data shape: {data.shape}, Labels shape: {labels.shape}")
    print(f"Data range: [{data.min():.3f}, {data.max():.3f}]")
    print(f"Label distribution: {labels.unique(return_counts=True)}")
    break

# 2. Verify model output range matches loss function
output = model(data)
print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
# CrossEntropyLoss expects raw logits, not probabilities!

# 3. Overfit on a single batch first
single_batch = next(iter(dataloader))
for epoch in range(100):
    output = model(single_batch[0].cuda())
    loss = criterion(output, single_batch[1].cuda())
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(f"Epoch {epoch}: loss = {loss.item():.4f}")
# If it can't overfit a single batch, there's a bug in model/loss/data

# 4. Check if gradients are flowing
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad mean={param.grad.mean():.6f}, std={param.grad.std():.6f}")
    else:
        print(f"{name}: NO GRADIENT")  # Layer is not connected or frozen
```

## Error Quick-Reference Table

| Error | Most Likely Cause | First Step |
|-------|------------------|------------|
| `CUDA out of memory` | Batch size too large or memory leak | Reduce batch size, check for graph accumulation |
| `device-side assert triggered` | Index out of bounds (often labels) | `CUDA_LAUNCH_BLOCKING=1`, check label range |
| `illegal memory access` | Buffer overflow in CUDA kernel | `compute-sanitizer --tool memcheck` |
| `NCCL timeout` | One rank failed or deadlocked | Check all rank logs, `NCCL_DEBUG=INFO` |
| `Address already in use` | Previous run didn't clean up | `lsof -i :<port>`, kill stale process |
| `Expected all tensors on same device` | Model/data on different GPUs | Check `.device` of all tensors |
| `NaN loss` | Numerical instability | `torch.autograd.set_detect_anomaly(True)` |
| XLA recompilation (slow TPU) | Dynamic tensor shapes | `drop_last=True`, pad sequences |
| `CUBLAS_STATUS_ALLOC_FAILED` | Not enough GPU memory for workspace | `torch.cuda.empty_cache()`, reduce batch |
| `No route to host` (NCCL) | Network/firewall issue | Check `NCCL_SOCKET_IFNAME`, test connectivity |

## Key Takeaways

1. **OOM is the most common error.** Start with batch size reduction, then gradient accumulation, then move to advanced techniques (Chapter 8).
2. **`CUDA_LAUNCH_BLOCKING=1`** should be your first debugging step for any CUDA error — it gives you the correct stack trace.
3. **Distributed training errors** are usually caused by one rank failing while others wait. Always check logs from all ranks.
4. **NaN loss** has a systematic debugging approach: detect anomaly, check data, verify output ranges, and try overfitting a single batch.
5. **XLA errors on TPU** are almost always related to dynamic shapes. The fix is to make all tensor shapes static and predictable.
