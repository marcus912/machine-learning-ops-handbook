# Chapter 1: GPU Fundamentals

## Why GPUs for Machine Learning?

CPUs are designed for general-purpose computing: they excel at complex logic, branching, and sequential tasks. A modern CPU might have 8-64 cores, each capable of handling sophisticated operations independently.

GPUs take a fundamentally different approach. They have thousands of smaller, simpler cores designed to execute the same operation on many data elements simultaneously. This is called **SIMT** (Single Instruction, Multiple Threads).

Machine learning workloads—especially training neural networks—consist largely of matrix multiplications and element-wise operations on large tensors. These operations are highly parallelizable, making GPUs 10-100x faster than CPUs for ML tasks.

### CPU vs GPU Architecture

```
CPU (few powerful cores):          GPU (many simple cores):
┌─────────┐ ┌─────────┐           ┌──┐┌──┐┌──┐┌──┐┌──┐┌──┐┌──┐┌──┐
│  Core 0  │ │  Core 1  │           │SM││SM││SM││SM││SM││SM││SM││SM│
│ (complex)│ │ (complex)│           └──┘└──┘└──┘└──┘└──┘└──┘└──┘└──┘
└─────────┘ └─────────┘           ┌──┐┌──┐┌──┐┌──┐┌──┐┌──┐┌──┐┌──┐
┌─────────┐ ┌─────────┐           │SM││SM││SM││SM││SM││SM││SM││SM│
│  Core 2  │ │  Core 3  │           └──┘└──┘└──┘└──┘└──┘└──┘└──┘└──┘
│ (complex)│ │ (complex)│           ... (80+ SMs, each with 128 CUDA cores)
└─────────┘ └─────────┘
```

**Key differences:**

| Aspect | CPU | GPU |
|--------|-----|-----|
| Core count | 8-64 | 1,000-16,000+ CUDA cores |
| Core complexity | High (out-of-order execution, branch prediction) | Low (simple ALU) |
| Clock speed | 3-5 GHz | 1-2 GHz |
| Memory bandwidth | ~50-100 GB/s | ~900-3,350 GB/s |
| Best for | Sequential logic, branching | Parallel numeric computation |

## NVIDIA GPU Architecture

### Streaming Multiprocessors (SMs)

The GPU is organized into **Streaming Multiprocessors (SMs)**. Each SM contains:

- **CUDA Cores** — Execute integer and floating-point arithmetic
- **Tensor Cores** — Specialized for matrix multiply-accumulate operations (critical for deep learning)
- **Registers** — Fast per-thread storage
- **Shared Memory / L1 Cache** — Fast memory shared within a thread block
- **Warp Schedulers** — Schedule groups of 32 threads (a "warp") for execution

```
Streaming Multiprocessor (SM)
├── Warp Scheduler (schedules 32-thread warps)
├── CUDA Cores (64-128 per SM)
├── Tensor Cores (4-8 per SM)
├── Register File (65,536 registers)
├── Shared Memory / L1 Cache (up to 164–228 KB combined, depending on generation)
└── Special Function Units (SFUs) for transcendentals
```

### Tensor Cores

Tensor Cores are purpose-built hardware for the operation `D = A × B + C` where A, B, C, D are small matrix fragments. They perform this fused multiply-add at much higher throughput than CUDA Cores — processing multiple elements per cycle.

Why this matters for ML:
- Neural network forward passes are sequences of matrix multiplications
- Tensor Cores deliver 2-8x speedup over CUDA Cores for matrix math
- They support reduced precision formats (FP16, BF16, TF32, INT8) for even higher throughput

### NVIDIA GPU Generations for ML

| Generation | Architecture | Key GPUs | Tensor Cores | Notable Features |
|------------|-------------|----------|-------------|-----------------|
| Volta (2017) | V100 | V100 (16/32GB) | 1st gen | First Tensor Cores, NVLink |
| Ampere (2020) | A100 | A100 (40/80GB) | 3rd gen | TF32, structural sparsity, MIG |
| Hopper (2022) | H100 | H100 (80GB) | 4th gen | FP8, Transformer Engine, NVLink 4.0 |
| Blackwell (2024) | B200 | B200 (192GB) | 5th gen | FP4, 2nd gen Transformer Engine |

## The CUDA Programming Model

CUDA (Compute Unified Device Architecture) is NVIDIA's parallel computing platform. Understanding the CUDA programming model helps you reason about performance and debug issues, even if you never write raw CUDA code.

### Key Concepts

**Host and Device:**
- **Host** = CPU and its memory (system RAM)
- **Device** = GPU and its memory (VRAM/HBM)
- Data must be explicitly transferred between host and device

**Kernels:**
A kernel is a function that runs on the GPU. When launched, it executes across many threads in parallel.

**Threads, Blocks, and Grids:**
```
Grid (all thread blocks for a kernel launch)
├── Block (0,0)         Block (1,0)         Block (2,0)
│   ├── Thread (0,0)    ├── Thread (0,0)    ├── Thread (0,0)
│   ├── Thread (1,0)    ├── Thread (1,0)    ├── Thread (1,0)
│   ├── Thread (0,1)    ├── Thread (0,1)    ├── Thread (0,1)
│   └── ...             └── ...             └── ...
├── Block (0,1)         Block (1,1)         Block (2,1)
└── ...                 ...                 ...
```

- **Thread** — Smallest unit of execution. Each runs the kernel code.
- **Block** — A group of threads (up to 1024) that can cooperate via shared memory and synchronize.
- **Grid** — All blocks for a single kernel launch. Blocks within a grid execute independently.
- **Warp** — Hardware scheduling unit. 32 threads execute in lockstep (SIMT).

### How PyTorch Uses CUDA

When you write PyTorch code, the framework handles CUDA details for you:

```python
import torch

# Move tensor to GPU (host → device transfer)
x = torch.randn(1000, 1000).cuda()  # or .to('cuda')

# Operations on GPU tensors automatically use CUDA kernels
y = torch.matmul(x, x.T)  # Launches a CUDA matrix multiplication kernel

# Move result back to CPU (device → host transfer)
result = y.cpu().numpy()
```

Behind the scenes, `torch.matmul` on a CUDA tensor:
1. Selects an optimized CUDA kernel (often from cuBLAS for matrix operations)
2. Launches the kernel with an appropriate grid/block configuration
3. The GPU executes the operation across thousands of threads
4. The result stays in GPU memory until you explicitly move it

### Device Management in PyTorch

```python
import torch

# Check if CUDA is available
print(torch.cuda.is_available())          # True/False
print(torch.cuda.device_count())          # Number of GPUs
print(torch.cuda.get_device_name(0))      # e.g., "NVIDIA A100-SXM4-80GB"
print(torch.cuda.current_device())        # Currently selected device index

# Set the default GPU
torch.cuda.set_device(1)                  # Use GPU 1

# Context manager for device
with torch.cuda.device(2):
    # Operations here use GPU 2
    tensor = torch.randn(100, 100, device='cuda')

# Explicit device placement
model = MyModel().to('cuda:0')            # Place model on GPU 0
data = data.to('cuda:0')                  # Place data on same GPU
```

**Common mistake:** Putting the model and data on different devices.

```python
# This will ERROR:
model = MyModel().to('cuda:0')
data = data.to('cuda:1')
output = model(data)  # RuntimeError: Expected all tensors to be on the same device
```

## GPU Memory Hierarchy

Understanding GPU memory is critical for debugging out-of-memory errors and optimizing performance.

```
┌─────────────────────────────────────────────────┐
│                 Off-chip (slow)                   │
│  ┌─────────────────────────────────────────────┐ │
│  │          Global Memory (HBM/VRAM)           │ │
│  │     40-192 GB, ~2-3 TB/s bandwidth          │ │
│  │     Accessible by all threads                │ │
│  └─────────────────────────────────────────────┘ │
│                        ↕                          │
│  ┌─────────────────────────────────────────────┐ │
│  │            L2 Cache (on-chip)               │ │
│  │     40-60 MB, shared across all SMs         │ │
│  └─────────────────────────────────────────────┘ │
│                        ↕                          │
│  ┌────────────────────────────┐                  │
│  │  Per-SM (fast, on-chip)    │                  │
│  │  ┌──────────────────────┐  │                  │
│  │  │ Shared Memory/L1     │  │                  │
│  │  │ Up to 228 KB per SM  │  │                  │
│  │  │ Very high bandwidth  │  │                  │
│  │  └──────────────────────┘  │                  │
│  │  ┌──────────────────────┐  │                  │
│  │  │ Register File        │  │                  │
│  │  │ 256 KB per SM        │  │                  │
│  │  │ Fastest access       │  │                  │
│  │  └──────────────────────┘  │                  │
│  └────────────────────────────┘                  │
└─────────────────────────────────────────────────┘
```

### Memory Types Explained

| Memory | Scope | Speed | Size | Use Case |
|--------|-------|-------|------|----------|
| Registers | Per thread | Fastest | 256 KB/SM | Local variables |
| Shared Memory | Per block | Very fast | Up to 228 KB/SM | Thread cooperation within a block |
| L2 Cache | Global | Fast | 40-60 MB | Automatic caching of global memory |
| Global Memory (HBM) | All threads | Slowest on GPU (~3 TB/s) | 40-192 GB | Model weights, activations, gradients |
| Host Memory (RAM) | CPU | Slow from GPU (~32 GB/s PCIe) | 64-2048 GB | Dataset storage, preprocessing |

### What Lives in GPU Global Memory During Training

When you train a neural network, GPU memory holds:

1. **Model parameters (weights)** — The learnable weights of your model
2. **Gradients** — Same size as parameters (one gradient per parameter)
3. **Optimizer states** — Adam stores 2 additional copies (momentum + variance) per parameter
4. **Activations** — Intermediate outputs saved for backpropagation. This is often the largest consumer.
5. **Temporary buffers** — Workspace for operations like convolutions and matrix multiplications

**Rule of thumb for Adam optimizer:**
- FP32 training: ~16 bytes per parameter (4 param + 4 grad + 4 momentum + 4 variance)
- A 7B parameter model needs ~112 GB just for parameters + optimizer states
- Plus activations, which scale with batch size and sequence length

```python
# Check GPU memory usage in PyTorch
import torch

print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Reserved:  {torch.cuda.memory_reserved() / 1e9:.2f} GB")
print(f"Max allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

# Get detailed memory summary
print(torch.cuda.memory_summary())
```

**Allocated vs Reserved:**
- **Allocated** = Memory actively used by tensors
- **Reserved** = Memory in PyTorch's caching allocator (pre-allocated for efficiency). Always ≥ allocated.
- The gap between reserved and allocated is memory PyTorch has reserved but isn't currently using. It's available for new PyTorch allocations without requesting more from CUDA.

## NVLink and GPU Interconnects

When using multiple GPUs, the interconnect between them matters enormously for distributed training.

### Interconnect Types

| Interconnect | Bandwidth | Use Case |
|-------------|-----------|----------|
| PCIe Gen4 x16 | 32 GB/s per direction | Standard GPU connection |
| PCIe Gen5 x16 | 64 GB/s per direction | Newer systems |
| NVLink 3.0 (A100) | 600 GB/s total | Multi-GPU within a node |
| NVLink 4.0 (H100) | 900 GB/s total | Multi-GPU within a node |
| NVSwitch | Full bisection bandwidth | Connects all GPUs in a node |
| InfiniBand HDR | 200 Gb/s | Multi-node communication |
| InfiniBand NDR | 400 Gb/s | Multi-node communication |

### Checking GPU Topology

```bash
# Show NVLink topology
nvidia-smi topo -m

# Example output (8x A100 with NVSwitch):
#         GPU0  GPU1  GPU2  GPU3  GPU4  GPU5  GPU6  GPU7
# GPU0     X    NV12  NV12  NV12  NV12  NV12  NV12  NV12
# GPU1    NV12   X    NV12  NV12  NV12  NV12  NV12  NV12
# ...
# NV12 = Connected via 12 NVLink connections
# PIX  = Connected via PCIe (same switch)
# PHB  = Connected via PCIe (through host bridge)
# SYS  = Connected across NUMA nodes
```

**Why this matters:** If GPUs communicate via PCIe instead of NVLink, gradient synchronization in distributed training can become a bottleneck, dramatically slowing training.

## Key Takeaways

1. **GPUs excel at ML** because neural network operations (matrix multiplications) are massively parallel.
2. **Tensor Cores** provide hardware acceleration specifically for matrix multiply-add operations used in deep learning.
3. **The CUDA model** organizes computation into threads, blocks, and grids. PyTorch handles this for you but understanding it helps debug performance.
4. **GPU memory is the primary constraint** for training. Model parameters, gradients, optimizer states, and activations all compete for limited VRAM.
5. **GPU interconnects** (NVLink vs PCIe) significantly impact multi-GPU training performance.
6. **Memory hierarchy matters** — data in shared memory or registers is orders of magnitude faster to access than global memory (HBM).
