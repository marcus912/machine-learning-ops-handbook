# Chapter 2: NVIDIA Debugging Tools

## nvidia-smi (System Management Interface)

`nvidia-smi` is the most important tool for GPU monitoring and debugging. It comes with every NVIDIA driver installation.

### Basic Usage

```bash
# One-shot status of all GPUs
nvidia-smi
```

**Example output:**
```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.129.03             Driver Version: 535.129.03   CUDA Version: 12.2       |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|=========================================+========================+======================|
|   0  NVIDIA A100-SXM4-80GB         On  | 00000000:07:00.0  Off |                    0 |
| N/A   34C    P0              63W / 400W |   45231MiB /  81920MiB |     78%      Default |
|-----------------------------------------+------------------------+----------------------+
```

### Reading nvidia-smi Output

| Field | What It Means | What to Look For |
|-------|--------------|-----------------|
| **Driver Version** | NVIDIA driver version | Must be compatible with your CUDA version |
| **CUDA Version** | Maximum CUDA version supported by driver | Does not mean CUDA toolkit is installed |
| **Persistence-M** | Persistence mode (On/Off) | Should be "On" in production for faster GPU initialization |
| **Temp** | GPU temperature | Warning above 80°C, throttling above 83-90°C |
| **Perf** | Performance state (P0-P12) | P0 = max performance, P8 = idle |
| **Pwr:Usage/Cap** | Power draw vs limit | Near cap = GPU working hard, throttling possible |
| **Memory-Usage** | Used/Total VRAM | Primary OOM indicator |
| **GPU-Util** | GPU compute utilization % | Low % during training = bottleneck elsewhere |
| **ECC** | Error-correcting code errors | Non-zero uncorrectable = hardware issue |

### Continuous Monitoring

```bash
# Refresh every 1 second (like top for GPUs)
nvidia-smi -l 1

# Monitor specific metrics in CSV format
nvidia-smi --query-gpu=timestamp,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw --format=csv -l 1

# Monitor specific GPU (useful in multi-GPU systems)
nvidia-smi -i 0 -l 1

# Watch just memory and utilization (compact)
watch -n 1 nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
```

### Process-Level GPU Usage

```bash
# Show which processes are using GPUs
nvidia-smi pmon -i 0 -s um -d 1

# Fields: GPU index, PID, Type (C=compute, G=graphics),
#         SM%, Mem%, Encoder%, Decoder%, FB Usage, Command

# Alternative: show compute processes
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

### Debugging Scenarios with nvidia-smi

**Scenario 1: GPU utilization is low during training**
```bash
# Check utilization over time
nvidia-smi dmon -i 0 -s pucvmet -d 1
# Columns: power, SM util, memory util, encoder, decoder,
#          clock speed, memory clock, temperature

# If GPU-Util is < 50% during training, the bottleneck is likely:
# - Data loading (CPU-bound preprocessing)
# - CPU-GPU data transfer
# - Small batch size (not enough work to saturate GPU)
```

**Scenario 2: Checking for thermal throttling**
```bash
# Monitor temperature and clock speeds
nvidia-smi --query-gpu=temperature.gpu,clocks.current.sm,clocks.max.sm,power.draw,power.limit --format=csv -l 1

# If current clock < max clock AND temp > 80°C → thermal throttling
```

**Scenario 3: Multi-GPU memory imbalance**
```bash
# Quick check all GPUs
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv

# If GPU 0 has much more memory used than others:
# - Likely the default device is collecting tensors
# - Check for model outputs being gathered to GPU 0
# - Verify DataParallel vs DistributedDataParallel usage
```

## CUDA-GDB

CUDA-GDB extends GDB to debug CUDA kernels. While you rarely debug raw CUDA kernels in ML workflows, it's useful for debugging custom CUDA extensions or PyTorch C++/CUDA operations.

### Basic Usage

```bash
# Launch program under CUDA-GDB
cuda-gdb --args python train.py

# Or attach to a running process
cuda-gdb -p <PID>
```

### Key Commands

```
# Standard GDB commands work for host code
(cuda-gdb) break main
(cuda-gdb) run
(cuda-gdb) next
(cuda-gdb) print variable

# CUDA-specific commands
(cuda-gdb) info cuda threads          # List all CUDA threads
(cuda-gdb) info cuda kernels          # List active kernels
(cuda-gdb) cuda thread (0,0,0)        # Switch to specific thread
(cuda-gdb) cuda block (1,0,0)         # Switch to specific block
(cuda-gdb) info cuda lanes            # Show lanes in current warp

# Set breakpoint in CUDA kernel
(cuda-gdb) break my_kernel
(cuda-gdb) break my_kernel.cu:42

# Inspect GPU memory
(cuda-gdb) print *(@global float *)0x... # Read global memory
```

### Practical: Debugging a Custom CUDA Extension

If you've written a custom CUDA kernel for PyTorch:

```bash
# Enable CUDA error checking (makes debugging easier, slower execution)
export CUDA_LAUNCH_BLOCKING=1

# Run with CUDA-GDB
cuda-gdb --args python -c "
import torch
from my_custom_op import custom_kernel
x = torch.randn(100, device='cuda')
result = custom_kernel(x)
"
```

**Tip:** `CUDA_LAUNCH_BLOCKING=1` forces synchronous CUDA execution. Without it, errors may appear to come from a later operation because CUDA kernels are asynchronous by default.

## CUDA Error Checking

### CUDA_LAUNCH_BLOCKING

The most important debugging environment variable:

```bash
# Force synchronous CUDA execution
export CUDA_LAUNCH_BLOCKING=1
python train.py
```

Why this helps:
- CUDA kernels normally execute **asynchronously** — the CPU queues work and moves on
- When an error occurs, the Python stack trace points to wherever the CPU was when the error was **detected**, not where it **occurred**
- `CUDA_LAUNCH_BLOCKING=1` forces the CPU to wait after each kernel, so errors point to the correct line

**Example without CUDA_LAUNCH_BLOCKING:**
```python
# Error appears here...
output = model(input)          # ← The actual bug might be here
loss = criterion(output, target)
loss.backward()
optimizer.step()               # ← ...but the error is reported here
```

**Example with CUDA_LAUNCH_BLOCKING=1:**
```python
output = model(input)          # ← Error correctly reported here
```

### TORCH_USE_CUDA_DSA (Device-Side Assertions)

```bash
# Enable device-side assertions (PyTorch 2.0+)
export TORCH_USE_CUDA_DSA=1
export CUDA_LAUNCH_BLOCKING=1  # Required alongside DSA
python train.py
```

This gives more detailed error messages for index-out-of-bounds errors and assertion failures within CUDA kernels.

## NCCL Debugging

NCCL (NVIDIA Collective Communications Library) handles GPU-to-GPU communication in distributed training. When multi-GPU training hangs or crashes, NCCL debugging is essential.

### NCCL Environment Variables

```bash
# Enable NCCL debug logging
export NCCL_DEBUG=INFO        # Options: VERSION, WARN, INFO, TRACE
export NCCL_DEBUG_SUBSYS=ALL  # Or: INIT, COLL, P2P, SHM, NET, GRAPH, TUNING

# Force specific communication protocol
export NCCL_P2P_DISABLE=1     # Disable peer-to-peer (NVLink), fall back to shared memory
export NCCL_SHM_DISABLE=1     # Disable shared memory transport
export NCCL_IB_DISABLE=1      # Disable InfiniBand (for multi-node)

# Set NCCL timeout (seconds)
export NCCL_TIMEOUT=1800      # Default varies; increase for large models

# Network interface selection (multi-node)
export NCCL_SOCKET_IFNAME=eth0  # Specify which network interface to use
```

### Diagnosing Common NCCL Issues

**Hang during distributed training:**
```bash
# Step 1: Enable verbose logging
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# Step 2: Check if all ranks are reaching the collective operation
# Look for log lines like:
# "NCCL INFO Launch mode Parallel"
# If some ranks show this and others don't, there's a synchronization issue

# Step 3: Check NCCL topology detection
# Look for: "NCCL INFO Trees [0] ..."
# This shows the communication topology NCCL selected
```

**NCCL version mismatch:**
```python
# Check NCCL version in PyTorch
import torch
print(torch.cuda.nccl.version())  # Should match across all nodes
```

**Network issues in multi-node training:**
```bash
# Test NCCL communication between nodes
# On node 0:
NCCL_DEBUG=INFO python -c "
import torch
import torch.distributed as dist
dist.init_process_group('nccl', rank=0, world_size=2)
tensor = torch.ones(1).cuda()
dist.all_reduce(tensor)
print(f'Result: {tensor}')
"
```

## nvidia-smi Advanced Features

### Multi-Instance GPU (MIG) — A100/H100

MIG partitions a single GPU into isolated instances, each with dedicated compute, memory, and cache.

```bash
# Check if MIG is enabled
nvidia-smi -i 0 --query-gpu=mig.mode.current --format=csv

# Enable MIG mode (requires root, GPU must be idle)
sudo nvidia-smi -i 0 -mig 1

# List available MIG profiles
nvidia-smi mig -lgip

# Create a MIG instance (e.g., 3g.40gb = 3 compute slices, 40GB memory)
sudo nvidia-smi mig -cgi 9,9 -i 0    # Create two 3g.40gb instances
sudo nvidia-smi mig -cci -i 0         # Create compute instances

# List MIG instances
nvidia-smi mig -lgi
nvidia-smi mig -lci

# Use specific MIG instance
export CUDA_VISIBLE_DEVICES=MIG-<UUID>
```

### GPU Reset

```bash
# Reset a hung GPU (use with caution)
sudo nvidia-smi -r -i 0

# If GPU is truly stuck, try:
sudo nvidia-smi --gpu-reset -i 0

# Check for GPU errors (Xid errors in dmesg)
sudo dmesg | grep -i "nvrm\|nvidia\|xid"
```

### Xid Errors

Xid errors are NVIDIA GPU error codes found in system logs:

```bash
# Check for Xid errors
sudo dmesg | grep -i xid
```

| Xid Code | Meaning | Action |
|----------|---------|--------|
| 13 | Graphics Engine Exception | Application bug or driver issue |
| 31 | GPU memory page fault | Likely a software bug (bad memory access) |
| 43 | GPU stopped processing | Possible hardware issue or driver bug |
| 45 | Preemptive cleanup | GPU reset due to application error |
| 48 | Double Bit ECC Error | Hardware failure — replace GPU |
| 63 | ECC page retirement | Row remapping exhausted — replace GPU |
| 64 | ECC page retirement (pending) | ECC errors accumulating — monitor closely |
| 79 | GPU fallen off the bus | Hardware/power issue — check seating/PSU |
| 94 | Contained ECC error | Correctable — monitor frequency |
| 95 | Uncontained ECC error | Replace GPU |

## CUDA Toolkit Utilities

### cuda-memcheck / compute-sanitizer

```bash
# Memory error detection (CUDA 11.6+: use compute-sanitizer)
compute-sanitizer --tool memcheck python train.py

# Race condition detection
compute-sanitizer --tool racecheck python train.py

# Memory initialization check
compute-sanitizer --tool initcheck python train.py

# Synchronization check
compute-sanitizer --tool synccheck python train.py
```

### nvcc (NVIDIA CUDA Compiler)

```bash
# Check CUDA toolkit version
nvcc --version

# Compile CUDA code
nvcc -o output kernel.cu

# Compile with debug info (for CUDA-GDB)
nvcc -g -G -o output kernel.cu
```

## Environment Variables Reference

| Variable | Values | Purpose |
|----------|--------|---------|
| `CUDA_VISIBLE_DEVICES` | `0,1,2` or `MIG-UUID` | Control which GPUs are visible to the application |
| `CUDA_LAUNCH_BLOCKING` | `0` or `1` | Force synchronous CUDA execution for debugging |
| `TORCH_USE_CUDA_DSA` | `0` or `1` | Enable device-side assertions in PyTorch |
| `CUDA_DEVICE_ORDER` | `PCI_BUS_ID` | Match GPU indices to physical order |
| `NCCL_DEBUG` | `VERSION/WARN/INFO/TRACE` | NCCL debug log verbosity |
| `NCCL_P2P_DISABLE` | `0` or `1` | Disable NVLink peer-to-peer |
| `NCCL_IB_DISABLE` | `0` or `1` | Disable InfiniBand |
| `NCCL_SOCKET_IFNAME` | `eth0`, `ens3`, etc. | Select network interface for NCCL |
| `NCCL_TIMEOUT` | seconds | Timeout for NCCL collective operations |
| `PYTORCH_CUDA_ALLOC_CONF` | see below | Configure PyTorch CUDA memory allocator |

### PYTORCH_CUDA_ALLOC_CONF Options

```bash
# Expand CUDA memory pool to reduce fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Set max split size to reduce fragmentation
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Combine options
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

# Enable garbage collection threshold
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6
```

## Key Takeaways

1. **`nvidia-smi`** is your first stop for any GPU issue — check memory, utilization, temperature, and running processes.
2. **`CUDA_LAUNCH_BLOCKING=1`** is the single most useful debugging variable — it makes CUDA errors point to the correct line.
3. **NCCL debugging** via `NCCL_DEBUG=INFO` is essential for diagnosing distributed training hangs and failures.
4. **Xid errors** in `dmesg` help distinguish hardware failures from software bugs.
5. **`PYTORCH_CUDA_ALLOC_CONF`** can help resolve memory fragmentation issues without code changes.
6. **`CUDA_VISIBLE_DEVICES`** controls GPU visibility and is the simplest way to select specific GPUs.
