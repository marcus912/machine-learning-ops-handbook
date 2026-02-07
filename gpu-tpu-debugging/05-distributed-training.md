# Chapter 5: Distributed Training

## Why Distributed Training?

As models grow larger, a single GPU is insufficient for training:
- A 7B parameter model needs ~112 GB just for parameters + Adam optimizer states in FP32
- Large batch training requires more memory for activations
- Training on a single GPU would take weeks or months for large models

Distributed training spreads the workload across multiple GPUs or multiple machines.

## Parallelism Strategies

There are three fundamental parallelism strategies, and modern systems often combine them.

### Data Parallelism

The most common and simplest strategy. Each GPU holds a **complete copy of the model** and processes different data.

```
┌─────────────────────────────────────────────────────────┐
│                     Data Parallelism                      │
│                                                           │
│  GPU 0              GPU 1              GPU 2              │
│  ┌──────────┐      ┌──────────┐      ┌──────────┐       │
│  │Full Model│      │Full Model│      │Full Model│       │
│  └────┬─────┘      └────┬─────┘      └────┬─────┘       │
│       │                  │                  │              │
│  Batch 0            Batch 1            Batch 2            │
│       │                  │                  │              │
│       ▼                  ▼                  ▼              │
│    Forward            Forward            Forward           │
│    Backward           Backward           Backward          │
│       │                  │                  │              │
│       └──────── All-Reduce Gradients ───────┘              │
│                          │                                 │
│                    Update Weights                          │
│                  (same on all GPUs)                        │
└─────────────────────────────────────────────────────────┘
```

**How it works:**
1. Each GPU gets a copy of the model
2. Training data is split across GPUs (each gets a different mini-batch)
3. Each GPU computes forward + backward pass independently
4. Gradients are **all-reduced** (averaged) across all GPUs
5. Each GPU updates its model weights identically

**When to use:** When the model fits in a single GPU's memory but you want to train faster with larger effective batch sizes.

### Model Parallelism (Tensor Parallelism)

Split individual layers across GPUs.

```
┌─────────────────────────────────────────────────┐
│              Tensor Parallelism                   │
│                                                   │
│  Linear layer: Y = XW + b                        │
│  W is (4096 × 4096), split across 2 GPUs:       │
│                                                   │
│  GPU 0                    GPU 1                   │
│  ┌──────────────┐        ┌──────────────┐        │
│  │ W[:, :2048]  │        │ W[:, 2048:]  │        │
│  │ (4096×2048)  │        │ (4096×2048)  │        │
│  └──────┬───────┘        └──────┬───────┘        │
│         │                       │                 │
│    Y_0 = X @ W_0          Y_1 = X @ W_1         │
│         │                       │                 │
│         └───── All-Gather ──────┘                 │
│                    │                              │
│               Y = [Y_0, Y_1]                     │
└─────────────────────────────────────────────────┘
```

**When to use:** When a single layer is too large for one GPU's memory (common in very large models).

### Pipeline Parallelism

Split model layers across GPUs sequentially.

```
┌───────────────────────────────────────────────────────────┐
│                  Pipeline Parallelism                       │
│                                                             │
│  GPU 0          GPU 1          GPU 2          GPU 3        │
│  ┌──────┐      ┌──────┐      ┌──────┐      ┌──────┐      │
│  │Layers│      │Layers│      │Layers│      │Layers│      │
│  │ 0-5  │─────▶│ 6-11 │─────▶│12-17 │─────▶│18-23 │      │
│  └──────┘      └──────┘      └──────┘      └──────┘      │
│                                                             │
│  Micro-batch scheduling (GPipe style):                     │
│                                                             │
│  Time →  t0   t1   t2   t3   t4   t5   t6   t7            │
│  GPU 0: [F0] [F1] [F2] [F3] [B3] [B2] [B1] [B0]          │
│  GPU 1:      [F0] [F1] [F2] [F3] [B3] [B2] [B1]          │
│  GPU 2:           [F0] [F1] [F2] [F3] [B3] [B2]          │
│  GPU 3:                [F0] [F1] [F2] [F3] [B3]          │
│                                                             │
│  F = Forward micro-batch, B = Backward micro-batch         │
│  "Bubble" = idle time when GPUs are waiting                │
└───────────────────────────────────────────────────────────┘
```

**When to use:** When the model is too large for one GPU, especially for very deep models.

**Pipeline bubble:** GPUs sit idle while waiting for their micro-batch. This is the main inefficiency. Solutions: more micro-batches per batch, interleaved scheduling.

### Combining Strategies (3D Parallelism)

Large-scale training (e.g., training GPT-3, LLaMA) often uses all three:

```
3D Parallelism Example (64 GPUs):
- 4-way tensor parallelism (TP)     → each layer split across 4 GPUs
- 4-way pipeline parallelism (PP)   → model split into 4 stages
- 4-way data parallelism (DP)       → 4 replicas of the full pipeline

Total: 4 × 4 × 4 = 64 GPUs
```

## PyTorch Distributed Data Parallel (DDP)

DDP is the standard approach for multi-GPU training in PyTorch. It's more efficient than `DataParallel` (DP).

### DataParallel vs DistributedDataParallel

| Feature | `DataParallel` (DP) | `DistributedDataParallel` (DDP) |
|---------|--------------------|---------------------------------|
| Simplicity | One line of code | Requires `init_process_group` |
| Processes | Single process, multi-thread | Multi-process (one per GPU) |
| Communication | Gather to GPU 0, then scatter | All-reduce (balanced) |
| GPU memory | GPU 0 uses more memory | Balanced across GPUs |
| Speed | Slower (GIL bottleneck) | Faster |
| Multi-node | No | Yes |
| **Recommendation** | **Avoid** | **Always prefer** |

### DDP Implementation

```python
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

def setup(rank, world_size):
    """Initialize the distributed environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size):
    setup(rank, world_size)

    # Create model and move to GPU
    model = MyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    # DistributedSampler ensures each GPU gets different data
    dataset = MyDataset()
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)

    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=1e-3)

    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)  # Important: shuffle differently each epoch

        for data, target in dataloader:
            data, target = data.to(rank), target.to(rank)

            optimizer.zero_grad()
            output = ddp_model(data)
            loss = criterion(output, target)
            loss.backward()  # Gradients are all-reduced automatically
            optimizer.step()

    cleanup()

# Launch training
if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size,), nprocs=world_size)
```

### Using torchrun (Recommended)

Instead of `mp.spawn`, use `torchrun` for better fault tolerance:

```python
# train.py
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def main():
    # torchrun sets these environment variables automatically
    dist.init_process_group("nccl")
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)

    model = MyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    # ... training loop ...

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
```

```bash
# Launch on a single node with 4 GPUs
torchrun --nproc_per_node=4 train.py

# Launch on 2 nodes with 4 GPUs each
# Node 0:
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 \
         --master_addr=node0_ip --master_port=29500 train.py
# Node 1:
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 \
         --master_addr=node0_ip --master_port=29500 train.py
```

### Common DDP Pitfalls

**Pitfall 1: Forgetting `sampler.set_epoch(epoch)`**
```python
# Without this, data order is the same every epoch
for epoch in range(num_epochs):
    sampler.set_epoch(epoch)  # Must call this!
```

**Pitfall 2: Saving/loading checkpoints incorrectly**
```python
# SAVE: Only save on rank 0 to avoid corruption
if rank == 0:
    torch.save(ddp_model.module.state_dict(), "checkpoint.pt")
dist.barrier()  # Wait for rank 0 to finish saving

# LOAD: Load on all ranks
map_location = f"cuda:{rank}"
state_dict = torch.load("checkpoint.pt", map_location=map_location)
ddp_model.module.load_state_dict(state_dict)
```

**Pitfall 3: Operations only on rank 0 that cause hanging**
```python
# BAD: Only rank 0 logs, but logging triggers a sync
if rank == 0:
    wandb.log({"loss": loss.item()})  # .item() triggers GPU sync only on rank 0
    # Other ranks are waiting at the next all-reduce...

# GOOD: All ranks compute loss.item(), only rank 0 logs
loss_val = loss.item()  # All ranks do this
if rank == 0:
    wandb.log({"loss": loss_val})
```

**Pitfall 4: Unused parameters causing hanging**
```python
# If some parameters aren't used in every forward pass,
# DDP will hang waiting for their gradients.

# Solution: Tell DDP to expect unused parameters
ddp_model = DDP(model, device_ids=[rank], find_unused_parameters=True)
# Note: This adds overhead. Better to restructure the model if possible.
```

## Fully Sharded Data Parallel (FSDP)

FSDP is PyTorch's implementation of the ZeRO algorithm. It shards model parameters, gradients, and optimizer states across GPUs, dramatically reducing per-GPU memory.

### How FSDP Differs from DDP

```
DDP: Each GPU holds full model copy
┌──────────┐  ┌──────────┐  ┌──────────┐
│ Params   │  │ Params   │  │ Params   │
│ Grads    │  │ Grads    │  │ Grads    │
│ Opt State│  │ Opt State│  │ Opt State│
│ (all)    │  │ (all)    │  │ (all)    │
└──────────┘  └──────────┘  └──────────┘
   GPU 0         GPU 1         GPU 2

FSDP: Each GPU holds a shard
┌──────────┐  ┌──────────┐  ┌──────────┐
│ Params/3 │  │ Params/3 │  │ Params/3 │
│ Grads/3  │  │ Grads/3  │  │ Grads/3  │
│ OptSt/3  │  │ OptSt/3  │  │ OptSt/3  │
└──────────┘  └──────────┘  └──────────┘
   GPU 0         GPU 1         GPU 2

When a layer needs computation:
1. All-gather full parameters for that layer
2. Compute forward/backward
3. Reduce-scatter gradients
4. Discard non-local parameters
```

### FSDP Implementation

```python
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy

# Define mixed precision policy
mixed_precision_policy = MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.bfloat16,
    buffer_dtype=torch.bfloat16,
)

# Wrap model with FSDP
model = MyLargeModel().to(rank)
fsdp_model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,  # ZeRO-3 equivalent
    mixed_precision=mixed_precision_policy,
    device_id=rank,
)

# Training loop is the same as DDP
optimizer = torch.optim.AdamW(fsdp_model.parameters(), lr=1e-4)

for data, target in dataloader:
    optimizer.zero_grad()
    output = fsdp_model(data.to(rank))
    loss = criterion(output, target.to(rank))
    loss.backward()
    optimizer.step()
```

### FSDP Sharding Strategies

| Strategy | Memory Savings | Communication Cost | Equivalent |
|----------|---------------|-------------------|------------|
| `FULL_SHARD` | Maximum | Highest | ZeRO Stage 3 |
| `SHARD_GRAD_OP` | Medium | Medium | ZeRO Stage 2 |
| `NO_SHARD` | None (same as DDP) | Lowest | ZeRO Stage 0 / DDP |
| `HYBRID_SHARD` | Shard within node, replicate across | Balanced | — |

### FSDP Checkpoint Saving

```python
from torch.distributed.fsdp import FullStateDictConfig, StateDictType

# Save full model checkpoint (gathers to rank 0)
save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
with FSDP.state_dict_type(fsdp_model, StateDictType.FULL_STATE_DICT, save_policy):
    state_dict = fsdp_model.state_dict()
    if rank == 0:
        torch.save(state_dict, "checkpoint.pt")
```

## Multi-Node Training

### Architecture

```
┌──────────────────────┐         ┌──────────────────────┐
│       Node 0          │  Network │       Node 1          │
│  ┌────┐ ┌────┐       │ (IB/Eth)│  ┌────┐ ┌────┐       │
│  │GPU0│ │GPU1│       │◄───────▶│  │GPU4│ │GPU5│       │
│  └────┘ └────┘       │         │  └────┘ └────┘       │
│  ┌────┐ ┌────┐       │         │  ┌────┐ ┌────┐       │
│  │GPU2│ │GPU3│       │         │  │GPU6│ │GPU7│       │
│  └────┘ └────┘       │         │  └────┘ └────┘       │
│                       │         │                       │
│  NVLink/NVSwitch      │         │  NVLink/NVSwitch      │
│  (intra-node: fast)   │         │  (intra-node: fast)   │
└──────────────────────┘         └──────────────────────┘
        ▲                                  ▲
        └──── InfiniBand (inter-node) ─────┘
```

### Key Considerations

**Network bandwidth:**
- Intra-node (NVLink): 600-900 GB/s
- Inter-node (InfiniBand): 25-50 GB/s
- Inter-node (Ethernet): 1-12.5 GB/s

**Rule of thumb:** Minimize inter-node communication. Use tensor parallelism within a node (requires high bandwidth) and data/pipeline parallelism across nodes.

### Debugging Multi-Node Issues

```bash
# Test connectivity between nodes
# On each node:
python -c "
import socket
hostname = socket.gethostname()
ip = socket.gethostbyname(hostname)
print(f'Node: {hostname}, IP: {ip}')
"

# Test NCCL communication
NCCL_DEBUG=INFO torchrun \
    --nproc_per_node=4 --nnodes=2 --node_rank=0 \
    --master_addr=node0_ip --master_port=29500 \
    -c "
import torch.distributed as dist
dist.init_process_group('nccl')
rank = dist.get_rank()
tensor = torch.ones(1).cuda()
dist.all_reduce(tensor)
print(f'Rank {rank}: tensor = {tensor.item()}')
dist.destroy_process_group()
"
```

### Common Multi-Node Issues

| Issue | Symptom | Fix |
|-------|---------|-----|
| DNS resolution | Timeout on init | Use IP addresses instead of hostnames |
| Firewall | Connection refused | Open ports 29500-29600 (or your configured range) |
| Wrong interface | Timeout or slow | Set `NCCL_SOCKET_IFNAME` to the right NIC |
| Clock skew | Intermittent failures | Sync NTP across nodes |
| Shared filesystem | Checkpoint corruption | Use `dist.barrier()` around save/load |
| Different NCCL versions | Mysterious hangs | Ensure identical software stacks on all nodes |

## Collective Operations Reference

Understanding collective operations helps debug distributed training:

| Operation | Description | Use In Training |
|-----------|-------------|----------------|
| **All-Reduce** | Reduce + broadcast result to all ranks | Gradient synchronization (DDP) |
| **All-Gather** | Gather data from all ranks to all ranks | Parameter gathering (FSDP) |
| **Reduce-Scatter** | Reduce + scatter results across ranks | Gradient sharding (FSDP) |
| **Broadcast** | Send data from one rank to all | Distributing model weights at init |
| **Reduce** | Aggregate data to one rank | Collecting metrics to rank 0 |
| **Scatter** | Split data from one rank to all | Distributing data |
| **Barrier** | Synchronize all ranks | Checkpoint save/load coordination |

```python
import torch.distributed as dist

# All-reduce: sum gradients across all ranks
tensor = torch.tensor([1.0]).cuda()
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
# tensor is now world_size * 1.0 on every rank

# Broadcast: send from rank 0 to all
if rank == 0:
    tensor = torch.tensor([42.0]).cuda()
else:
    tensor = torch.zeros(1).cuda()
dist.broadcast(tensor, src=0)
# tensor is now 42.0 on every rank

# Barrier: wait for all ranks
dist.barrier()
# All ranks reach here before any proceeds
```

## Key Takeaways

1. **Data parallelism** is the simplest and most common strategy. Use DDP, not DataParallel.
2. **FSDP** (ZeRO-3) dramatically reduces per-GPU memory by sharding parameters, gradients, and optimizer states. Use it when the model is too large for DDP.
3. **torchrun** is the recommended way to launch distributed training — it handles environment setup and provides elastic training support.
4. **Multi-node training** requires careful attention to network configuration, especially NCCL interface selection and firewall rules.
5. **Common DDP bugs** include forgetting `sampler.set_epoch()`, saving checkpoints incorrectly, and operations only on rank 0 that cause hangs.
6. **Minimize inter-node communication** by using tensor parallelism within nodes (NVLink) and data/pipeline parallelism across nodes (InfiniBand).
