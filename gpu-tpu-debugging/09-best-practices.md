# Chapter 9: Best Practices

## Debugging Workflow

When something goes wrong with GPU/TPU training, follow this systematic approach instead of making random changes.

### Step 1: Identify the Problem Category

```
What's happening?
├── Training crashes immediately
│   ├── CUDA OOM → Chapter 4, Chapter 8
│   ├── CUDA runtime error → Chapter 4 (set CUDA_LAUNCH_BLOCKING=1)
│   ├── Import error / driver mismatch → Check driver/CUDA/PyTorch compatibility
│   ├── XLA compilation error → Chapter 3 (check shapes)
│   └── Vertex AI job stuck/failing → Check quota, container, permissions (Ch 7, 10)
│
├── Training crashes after some time
│   ├── OOM (memory leak) → Profile memory over time
│   ├── NaN loss → Chapter 4 (numerical issues)
│   ├── NCCL timeout → Chapter 4, Chapter 5 (check all ranks)
│   └── Hardware error → Check dmesg for Xid errors (Chapter 2)
│
├── Training is slow
│   ├── Low GPU utilization → Chapter 6 (data loading bottleneck)
│   ├── High GPU utilization but slow → Profile kernels (Chapter 6)
│   ├── Distributed training slow → Check interconnect (Chapter 1, 5)
│   └── XLA recompilation → Chapter 3 (dynamic shapes)
│
└── Training doesn't converge
    ├── Loss not decreasing → Chapter 4 (debugging checklist)
    ├── Loss decreasing but accuracy poor → Data or evaluation issue
    └── Metrics differ from baseline → Check reproducibility settings
```

### Step 2: Gather Information

Before changing anything, collect diagnostic information:

```bash
# Hardware info
nvidia-smi                              # GPU status, memory, utilization
nvidia-smi topo -m                      # GPU interconnect topology
nvidia-smi -q -d ECC                    # ECC error count

# Software versions
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"
nvidia-smi --query-gpu=driver_version --format=csv,noheader
nvcc --version 2>/dev/null

# System info
free -h                                 # System RAM
nproc                                   # CPU cores
lscpu | grep "Model name"              # CPU model
sudo dmesg | grep -i "nvrm\|xid"      # GPU errors in kernel log
```

### Step 3: Reproduce with Minimal Example

Strip your training code down to the minimum that reproduces the issue:

```python
# Minimal reproduction template
import torch
import torch.nn as nn

# Minimal model
model = nn.Linear(1024, 1024).cuda()

# Minimal data
x = torch.randn(32, 1024).cuda()
target = torch.randn(32, 1024).cuda()

# Minimal training loop
optimizer = torch.optim.Adam(model.parameters())
for step in range(100):
    output = model(x)
    loss = nn.MSELoss()(output, target)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if step % 10 == 0:
        print(f"Step {step}: loss={loss.item():.4f}, "
              f"mem={torch.cuda.memory_allocated()/1e9:.2f}GB")
```

If the minimal example works, gradually add back components until you find the culprit.

### Step 4: Apply Targeted Fix

Based on your diagnosis, apply the specific fix from the relevant chapter. Don't make multiple changes at once — change one thing, test, and verify.

## Environment Setup Best Practices

### Docker for Reproducibility

```dockerfile
# Dockerfile for GPU training
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy training code
COPY . /app
WORKDIR /app

# Set environment variables for debugging
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID
ENV PYTHONUNBUFFERED=1

CMD ["python", "train.py"]
```

```bash
# Run with GPU access
docker run --gpus all -v /data:/data my-training-image
```

### Version Compatibility Matrix

Always verify compatibility between:

```
NVIDIA Driver ≥ CUDA Toolkit ≥ PyTorch CUDA version
```

| PyTorch | CUDA Toolkit | Minimum Driver |
|---------|-------------|----------------|
| 2.1.x | 11.8 or 12.1 | 520.61+ or 530.30+ |
| 2.2.x | 11.8 or 12.1 | 520.61+ or 530.30+ |
| 2.3.x | 11.8 or 12.1 | 520.61+ or 530.30+ |
| 2.4.x | 11.8 or 12.4 | 520.61+ or 550.54+ |

```python
# Check compatibility in Python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version (PyTorch): {torch.version.cuda}")
print(f"cuDNN version: {torch.backends.cudnn.version()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
```

### Reproducibility Settings

```python
import torch
import random
import numpy as np

def set_seed(seed=42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic algorithms (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

set_seed(42)

# Note: some operations don't have deterministic implementations.
# You'll get an error if one is encountered with use_deterministic_algorithms(True).
# Set this environment variable for deterministic scatter/gather:
# export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

## Framework-Specific Tips

### PyTorch

```python
# 1. Use torch.compile for automatic optimization (PyTorch 2.0+)
model = torch.compile(model)

# 2. Pin memory in DataLoader for faster CPU→GPU transfer
dataloader = DataLoader(dataset, pin_memory=True, num_workers=4)

# 3. Use channels_last memory format for CNNs
model = model.to(memory_format=torch.channels_last)
data = data.to(memory_format=torch.channels_last)

# 4. Disable gradient computation during validation
@torch.no_grad()
def validate(model, val_loader):
    model.eval()
    for data, target in val_loader:
        output = model(data.cuda())
        # ... compute metrics ...
    model.train()

# 5. Use BF16 on Ampere+ GPUs
with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    output = model(data)
```

### TensorFlow

```python
# 1. Use tf.function for graph compilation
@tf.function
def train_step(data, labels):
    with tf.GradientTape() as tape:
        predictions = model(data, training=True)
        loss = loss_fn(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# 2. Mixed precision
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# 3. XLA compilation
tf.config.optimizer.set_jit(True)

# 4. Memory growth (prevent TF from grabbing all GPU memory)
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
```

### JAX

```python
import jax
import jax.numpy as jnp

# 1. Always use jit for performance
@jax.jit
def train_step(state, batch):
    def loss_fn(params):
        logits = model.apply(params, batch['input'])
        return cross_entropy(logits, batch['label'])

    grads = jax.grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state

# 2. Use pmap for multi-device data parallelism
@jax.pmap
def parallel_train_step(state, batch):
    # Automatically replicated across all devices
    return train_step(state, batch)

# 3. Avoid Python control flow dependent on values
# BAD: forces sync
if jnp.sum(loss) > threshold:
    ...

# GOOD: use jax.lax.cond
result = jax.lax.cond(loss > threshold, true_fn, false_fn, operand)

# 4. Pre-compile to avoid first-step latency
compiled_step = jax.jit(train_step).lower(state, sample_batch).compile()
```

## Checkpointing Best Practices

### Save Frequently, Save Efficiently

```python
import torch
import os

def save_checkpoint(model, optimizer, epoch, step, path="checkpoints"):
    os.makedirs(path, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
    }
    filepath = os.path.join(path, f"checkpoint_epoch{epoch}_step{step}.pt")
    torch.save(checkpoint, filepath)

    # Keep only last N checkpoints to save disk space
    checkpoints = sorted(
        [f for f in os.listdir(path) if f.startswith("checkpoint_")],
        key=lambda f: os.path.getmtime(os.path.join(path, f)),
    )
    for old in checkpoints[:-3]:  # Keep last 3
        os.remove(os.path.join(path, old))

def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path, map_location="cuda")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["epoch"], checkpoint["step"]
```

### Distributed Checkpointing

```python
import torch.distributed as dist

def save_distributed_checkpoint(model, optimizer, epoch, step, rank):
    """Save checkpoint from rank 0 only."""
    if rank == 0:
        save_checkpoint(model.module, optimizer, epoch, step)
    dist.barrier()  # All ranks wait for rank 0

def load_distributed_checkpoint(model, optimizer, path, rank):
    """Load checkpoint on all ranks."""
    map_location = f"cuda:{rank}"
    checkpoint = torch.load(path, map_location=map_location)
    model.module.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["epoch"], checkpoint["step"]
```

## Monitoring During Training

### Essential Metrics to Track

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def log_training_step(step, loss, lr, gpu_mem, grad_norm):
    logger.info(
        f"step={step} | loss={loss:.4f} | lr={lr:.2e} | "
        f"gpu_mem={gpu_mem:.1f}GB | grad_norm={grad_norm:.4f}"
    )

# In training loop:
for step, (data, target) in enumerate(dataloader):
    # ... training step ...

    if step % log_interval == 0:
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))
        gpu_mem = torch.cuda.memory_allocated() / 1e9
        lr = optimizer.param_groups[0]['lr']
        log_training_step(step, loss.item(), lr, gpu_mem, grad_norm)
```

### Warning Signs to Watch For

| Metric | Warning Sign | Possible Issue |
|--------|-------------|---------------|
| Loss | Sudden spike or NaN | Learning rate too high, data issue |
| GPU memory | Steadily increasing | Memory leak (graph accumulation) |
| GPU utilization | Dropping below 50% | Data loading bottleneck |
| Gradient norm | Exploding (> 100) | Need gradient clipping |
| Gradient norm | Vanishing (< 1e-7) | Vanishing gradients, dead layers |
| Learning rate | Not following schedule | Scheduler not stepping correctly |
| Step time | Increasing over time | Memory pressure causing swapping |

## Pre-Flight Checklist

Run through this before starting any significant training job:

```
Hardware & Environment:
[ ] nvidia-smi shows expected GPUs
[ ] CUDA, PyTorch, and driver versions are compatible
[ ] Sufficient GPU memory for model + batch size
[ ] Network connectivity between nodes (distributed training)
[ ] NVLink/ICI topology is correct

Code:
[ ] Model compiles and runs for 1 step on CPU
[ ] Model runs for 10 steps on GPU with target batch size
[ ] Mixed precision enabled (BF16 preferred)
[ ] Gradient checkpointing enabled if needed
[ ] DataLoader has num_workers > 0 and pin_memory=True
[ ] Checkpointing saves and loads correctly
[ ] Reproducibility seeds are set

Distributed Training:
[ ] DistributedSampler with set_epoch() is used
[ ] Only rank 0 saves checkpoints
[ ] All ranks reach the same collective operations
[ ] NCCL_DEBUG=INFO for first run to verify communication

Monitoring:
[ ] Loss, learning rate, GPU memory logged every N steps
[ ] Gradient norms monitored
[ ] Checkpoint saved at regular intervals
[ ] Preemption handler installed (if using spot/preemptible)
```

## Common Mistakes Summary

| Mistake | Consequence | Fix |
|---------|------------|-----|
| Not using `torch.no_grad()` for eval | OOM during validation | Wrap eval in `torch.no_grad()` or `inference_mode()` |
| Accumulating tensors in a list | Memory leak | Use `.item()` or `.detach()` |
| `DataParallel` instead of `DDP` | Slow, unbalanced memory | Switch to `DistributedDataParallel` |
| Missing `sampler.set_epoch()` | Same data order every epoch | Call `sampler.set_epoch(epoch)` |
| FP16 without loss scaling | NaN loss | Use `GradScaler` or switch to BF16 |
| Dynamic shapes on TPU | Constant recompilation | Pad to fixed sizes, `drop_last=True` |
| Only rank 0 computes `.item()` | Deadlock in DDP | All ranks compute, only rank 0 logs |
| No gradient clipping | Exploding gradients, NaN | `clip_grad_norm_(params, max_norm=1.0)` |
| Saving all checkpoints | Disk full | Keep only last N checkpoints |
| Not testing locally first | Expensive cloud failures | Test 10 steps locally before scaling |

## Structured Learning Path

If you're starting from zero with GPU/TPU debugging, here's a structured approach:

### Phase 1: Foundations (Chapters 1-2)
**Goal:** Understand GPU hardware and basic tools.
- Read Chapter 1 to understand GPU architecture
- Practice `nvidia-smi` commands from Chapter 2
- Run a simple PyTorch training script on GPU
- Deliberately cause an OOM error, then fix it

### Phase 2: Error Handling (Chapters 3-4)
**Goal:** Recognize and fix common errors.
- Read Chapter 3 to understand TPU differences
- Work through every error in Chapter 4 — try to reproduce each one
- Practice the debugging workflow (CUDA_LAUNCH_BLOCKING, anomaly detection)
- Build a personal runbook of errors you've seen

### Phase 3: Scaling (Chapters 5-6)
**Goal:** Run distributed training and identify bottlenecks.
- Set up DDP training on 2+ GPUs (Chapter 5)
- Profile your training loop with PyTorch Profiler (Chapter 6)
- Identify and fix your main bottleneck
- Try FSDP for a model that doesn't fit in DDP

### Phase 4: Optimization & GCP Platform (Chapters 7-8, 10)
**Goal:** Optimize memory and cost, use managed services.
- Set up a GCP training job (Chapter 7)
- Explore Vertex AI Workbench, TensorBoard, and model serving (Chapter 10)
- Apply mixed precision, gradient checkpointing, and DeepSpeed (Chapter 8)
- Measure the memory savings from each technique
- Try QLoRA fine-tuning of a 7B model on a single GPU

### Phase 5: Mastery (Chapter 9 + Practice)
**Goal:** Handle any GPU/TPU issue with confidence.
- Practice the debugging workflow on unfamiliar errors
- Set up end-to-end monitoring for a training job
- Run through the pre-flight checklist for a real project
- Teach someone else — explaining cements understanding

## Key Takeaways

1. **Follow a systematic debugging workflow** — don't change things randomly. Identify the category, gather information, reproduce minimally, then fix.
2. **Environment reproducibility** matters — use Docker, pin versions, and set seeds.
3. **Monitor training metrics** actively — catching problems early saves GPU hours and money.
4. **The pre-flight checklist** catches most issues before they waste compute time.
5. **Common mistakes have common fixes** — reference the summary table when you encounter a familiar pattern.
6. **Learning is progressive** — master single-GPU training before distributed, master debugging before optimization.
