# Chapter 8: Memory Optimization

## Why Memory Optimization Matters

GPU memory is the primary constraint in training large models. A 7B parameter model with Adam optimizer in FP32 needs:

```
Model parameters:     7B × 4 bytes  =  28 GB
Gradients:            7B × 4 bytes  =  28 GB
Adam momentum:        7B × 4 bytes  =  28 GB
Adam variance:        7B × 4 bytes  =  28 GB
                                     --------
Total (params+opt):                   112 GB

Plus activations (varies with batch size and sequence length):
- Batch size 1, seq len 2048: ~20-40 GB
- Batch size 8, seq len 2048: ~160-320 GB
```

A single A100 (80GB) can't even hold the parameters + optimizer states. Memory optimization techniques are essential.

## Mixed Precision Training

Train with lower-precision numbers (FP16 or BF16) while maintaining FP32 accuracy.

### How Mixed Precision Works

```
Standard FP32 Training:
Params (FP32) → Forward (FP32) → Loss (FP32) → Backward (FP32) → Update (FP32)
Memory: 4 bytes per parameter for every tensor

Mixed Precision Training:
Params (FP32 master copy) → Cast to FP16/BF16 → Forward (FP16) → Loss (FP32)
→ Scale loss → Backward (FP16) → Unscale gradients → Update FP32 master copy
```

**Memory savings:** Activations and gradients are stored in FP16/BF16 (2 bytes instead of 4), roughly halving their memory footprint. The FP32 master weights are kept for numerical stability.

### PyTorch Automatic Mixed Precision (AMP)

```python
import torch

model = MyModel().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scaler = torch.amp.GradScaler()  # Handles loss scaling for FP16 stability

for data, target in dataloader:
    data, target = data.cuda(), target.cuda()

    optimizer.zero_grad()

    # Forward pass in FP16
    with torch.autocast(device_type='cuda'):
        output = model(data)
        loss = criterion(output, target)

    # Backward pass with scaled loss
    scaler.scale(loss).backward()

    # Unscale gradients and step
    scaler.step(optimizer)
    scaler.update()
```

### BF16 vs FP16

| Feature | FP16 | BF16 |
|---------|------|------|
| Exponent bits | 5 | 8 |
| Mantissa bits | 10 | 7 |
| Range | ±65,504 | ±3.4 × 10³⁸ |
| Precision | Higher | Lower |
| Loss scaling needed | Yes | No |
| Hardware support | All NVIDIA GPUs since Pascal | Ampere+ (A100, H100), all TPUs |

**Recommendation:** Use BF16 when hardware supports it (A100+, TPU). It eliminates the need for loss scaling and has fewer numerical issues.

```python
# BF16 with PyTorch — no GradScaler needed
with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    output = model(data)
    loss = criterion(output, target)

loss.backward()
optimizer.step()
```

### Common Mixed Precision Issues

**Issue: Loss goes to NaN with FP16**
```python
# Cause: FP16 overflow (values > 65,504)
# Solution 1: Use BF16 instead
with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    ...

# Solution 2: Check and fix loss scale
# If scaler._scale keeps decreasing to very small values:
print(f"Loss scale: {scaler.get_scale()}")
# This means there are frequent overflows — check for numerical instability
```

**Issue: Model accuracy degrades with mixed precision**
```python
# Some operations need FP32 precision. PyTorch AMP handles most cases,
# but custom operations may need explicit casting:
class MyModule(nn.Module):
    def forward(self, x):
        # Force FP32 for numerically sensitive operation
        with torch.autocast(device_type='cuda', enabled=False):
            x = x.float()  # Cast to FP32
            x = my_sensitive_operation(x)
        return x  # Will be cast back to FP16/BF16 by autocast
```

## Gradient Checkpointing

Trade compute for memory by not storing all activations during the forward pass. Instead, recompute them during the backward pass.

### How It Works

```
Standard training:
Forward:  Layer1 → save act1 → Layer2 → save act2 → Layer3 → save act3 → Loss
Backward: Use act3 → grad3 → Use act2 → grad2 → Use act1 → grad1
Memory: All activations stored simultaneously

Gradient checkpointing:
Forward:  Layer1 → save act1 → Layer2 → (discard act2) → Layer3 → (discard act3) → Loss
Backward: Recompute act3 → grad3 → Recompute act2 → grad2 → Use act1 → grad1
Memory: Only checkpointed activations stored (e.g., every other layer)
```

**Trade-off:** ~33% more compute (recomputation) for ~60-70% less activation memory.

### Implementation in PyTorch

```python
from torch.utils.checkpoint import checkpoint

class TransformerBlock(nn.Module):
    def forward(self, x):
        x = self.attention(x)
        x = self.feed_forward(x)
        return x

class MyTransformer(nn.Module):
    def __init__(self, num_layers=24):
        super().__init__()
        self.layers = nn.ModuleList([TransformerBlock() for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            # Checkpoint each transformer block
            x = checkpoint(layer, x, use_reentrant=False)
        return x
```

### HuggingFace Gradient Checkpointing

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
model.gradient_checkpointing_enable()  # One line to enable

# Now activations are recomputed during backward pass
# Memory savings depend on model architecture
```

### Selective Checkpointing

You don't need to checkpoint every layer. Checkpoint the most memory-hungry ones:

```python
class MyModel(nn.Module):
    def forward(self, x):
        # Don't checkpoint cheap operations
        x = self.embedding(x)

        # Checkpoint expensive transformer layers
        for layer in self.transformer_layers:
            x = checkpoint(layer, x, use_reentrant=False)

        # Don't checkpoint the final head
        x = self.output_head(x)
        return x
```

## DeepSpeed ZeRO

DeepSpeed's ZeRO (Zero Redundancy Optimizer) partitions model states across GPUs to eliminate memory redundancy.

### ZeRO Stages

```
Stage 0 (Baseline DDP):
GPU 0: [Full Params] [Full Grads] [Full Opt State]
GPU 1: [Full Params] [Full Grads] [Full Opt State]
GPU 2: [Full Params] [Full Grads] [Full Opt State]
Memory per GPU: P + G + O (where P=params, G=grads, O=optimizer state)

Stage 1 (Partition optimizer states):
GPU 0: [Full Params] [Full Grads] [Opt State / 3]
GPU 1: [Full Params] [Full Grads] [Opt State / 3]
GPU 2: [Full Params] [Full Grads] [Opt State / 3]
Memory per GPU: P + G + O/N

Stage 2 (+ Partition gradients):
GPU 0: [Full Params] [Grads / 3] [Opt State / 3]
GPU 1: [Full Params] [Grads / 3] [Opt State / 3]
GPU 2: [Full Params] [Grads / 3] [Opt State / 3]
Memory per GPU: P + G/N + O/N

Stage 3 (+ Partition parameters):
GPU 0: [Params / 3] [Grads / 3] [Opt State / 3]
GPU 1: [Params / 3] [Grads / 3] [Opt State / 3]
GPU 2: [Params / 3] [Grads / 3] [Opt State / 3]
Memory per GPU: (P + G + O) / N
```

### Memory Savings Example (7B Model, 3 GPUs)

| Component | DDP | ZeRO-1 | ZeRO-2 | ZeRO-3 |
|-----------|-----|--------|--------|--------|
| Parameters | 28 GB | 28 GB | 28 GB | 9.3 GB |
| Gradients | 28 GB | 28 GB | 9.3 GB | 9.3 GB |
| Optimizer (Adam) | 56 GB | 18.7 GB | 18.7 GB | 18.7 GB |
| **Total per GPU** | **112 GB** | **74.7 GB** | **56 GB** | **37.3 GB** |

### DeepSpeed Configuration

```json
{
    "train_batch_size": 32,
    "gradient_accumulation_steps": 4,
    "fp16": {
        "enabled": true,
        "loss_scale": 0,
        "initial_scale_power": 16
    },
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": true,
        "allgather_bucket_size": 2e8,
        "reduce_scatter": true,
        "reduce_bucket_size": 2e8,
        "overlap_comm": true,
        "contiguous_gradients": true
    },
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 1e-4,
            "weight_decay": 0.01
        }
    }
}
```

### Using DeepSpeed with HuggingFace

```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    deepspeed="ds_config.json",  # Path to DeepSpeed config
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()
```

```bash
# Launch with DeepSpeed
deepspeed --num_gpus=4 train.py --deepspeed ds_config.json
```

### ZeRO-Offload (CPU/NVMe Offloading)

When even ZeRO-3 isn't enough, offload to CPU RAM or NVMe storage:

```json
{
    "zero_optimization": {
        "stage": 3,
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        }
    }
}
```

**Trade-off:** Offloading to CPU/NVMe is slower due to PCIe bandwidth limitations. Use only when GPU memory is truly insufficient.

## Gradient Accumulation

Simulate larger batch sizes without more memory by accumulating gradients over multiple micro-batches.

```python
accumulation_steps = 8
micro_batch_size = 4
# Effective batch size = 8 × 4 = 32

optimizer.zero_grad()
for i, (data, target) in enumerate(dataloader):
    output = model(data.cuda())
    loss = criterion(output, target.cuda())
    loss = loss / accumulation_steps  # Normalize loss
    loss.backward()  # Accumulates gradients

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**Why this saves memory:** Each micro-batch uses less activation memory than one large batch. Gradients accumulate in-place (same memory as one micro-batch).

## Model Quantization

Reduce model size by using lower-precision integers for weights.

### Quantization Levels

| Precision | Bytes per param | 7B Model Size | Quality Impact |
|-----------|----------------|---------------|----------------|
| FP32 | 4 | 28 GB | Reference |
| FP16/BF16 | 2 | 14 GB | Negligible |
| INT8 | 1 | 7 GB | Minimal |
| INT4 | 0.5 | 3.5 GB | Noticeable for small models |

### Quantization with bitsandbytes

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# 8-bit quantization
model_8bit = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=BitsAndBytesConfig(load_in_8bit=True),
    device_map="auto",
)
# Uses ~7 GB instead of 28 GB (FP32) or 14 GB (FP16)

# 4-bit quantization (QLoRA-compatible)
model_4bit = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    ),
    device_map="auto",
)
# Uses ~3.5 GB
```

### QLoRA: Quantized Fine-Tuning

Fine-tune a quantized model by training small adapter layers:

```python
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Prepare quantized model for training
model = prepare_model_for_kbit_training(model_4bit)

# Add LoRA adapters
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: 4,194,304 || all params: 3,740,000,000 || trainable%: 0.112%
```

## Memory Optimization Decision Tree

```
Can model fit on a single GPU with FP32?
├── Yes → Train normally (consider mixed precision for speed)
├── No → Try mixed precision (BF16/FP16)
│   ├── Fits now → Done
│   └── Still OOM → Try gradient checkpointing
│       ├── Fits now → Done
│       └── Still OOM → Reduce batch size + gradient accumulation
│           ├── Fits now → Done
│           └── Still OOM (even batch_size=1) → Model is too large for 1 GPU
│               ├── Multiple GPUs available → Use FSDP/ZeRO
│               │   ├── ZeRO Stage 1 (optimizer partitioned)
│               │   ├── ZeRO Stage 2 (+ gradients partitioned)
│               │   └── ZeRO Stage 3 (+ parameters partitioned)
│               └── Single GPU only → Quantize (8-bit or 4-bit) + QLoRA
│                   ├── Fits now → Fine-tune with LoRA adapters
│                   └── Still OOM → CPU offloading (ZeRO-Offload)
```

## Memory Debugging Tools

### PyTorch Memory Snapshot

```python
import torch

# Record memory history
torch.cuda.memory._record_memory_history(max_entries=100000)

# Run your training code
train_one_epoch()

# Save snapshot
torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")

# Disable recording
torch.cuda.memory._record_memory_history(enabled=None)

# Visualize: upload to https://pytorch.org/memory_viz
```

### Tracking Memory Per Layer

```python
import torch

def memory_hook(name):
    def hook(module, input, output):
        allocated = torch.cuda.memory_allocated() / 1e9
        print(f"{name}: {allocated:.2f} GB allocated")
    return hook

# Register hooks on all layers
for name, module in model.named_modules():
    module.register_forward_hook(memory_hook(name))

# Run forward pass to see memory growth per layer
with torch.no_grad():
    output = model(sample_input.cuda())
```

## Key Takeaways

1. **Mixed precision** (BF16 preferred) is the first optimization to apply — minimal code changes, significant memory and speed improvements.
2. **Gradient checkpointing** trades ~33% more compute for ~60-70% less activation memory. Enable it with one line in HuggingFace models.
3. **Gradient accumulation** lets you simulate large batches without the memory cost. No accuracy impact.
4. **DeepSpeed ZeRO** stages progressively shard more state across GPUs. Start with Stage 2 and move to Stage 3 if needed.
5. **Quantization + LoRA (QLoRA)** enables fine-tuning 7B+ models on a single consumer GPU.
6. **Follow the decision tree:** mixed precision → checkpointing → smaller batches → FSDP/ZeRO → quantization → offloading.
