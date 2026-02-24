# Chapter 12: ML Fundamentals Review

A concise reference for core machine learning concepts — optimizers, learning rate schedules, regularization, and the deep learning vs traditional ML decision framework. Intended as a quick refresher for interviews and design discussions.

## Optimizers

### Comparison

| Optimizer | Update Rule (Intuition) | Key Property | Best For |
|-----------|------------------------|--------------|----------|
| **SGD** | Step proportional to gradient | Simple, well-understood convergence theory | Large-scale training when tuning budget is available |
| **SGD + Momentum** | Weighted moving average of gradients | Accelerates convergence, dampens oscillation | Most DL tasks with careful tuning |
| **Adam** | Per-parameter adaptive LR using 1st and 2nd moment estimates | Works well out of the box | Default choice for most tasks |
| **AdaGrad** | Adapts LR based on cumulative gradient history | Larger updates for infrequent features | Sparse data (NLP, recommender systems) |
| **RMSProp** | Like AdaGrad but with exponential decay of gradient history | Fixes AdaGrad's diminishing LR | Non-stationary objectives, RNNs |
| **AdamW** | Adam with decoupled weight decay | Better generalization than Adam with L2 | Transformer fine-tuning, most modern DL |

### How Adam Works

```
m_t = β₁ * m_{t-1} + (1 - β₁) * g_t          # First moment (mean of gradients)
v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²          # Second moment (variance of gradients)
m̂_t = m_t / (1 - β₁^t)                         # Bias correction
v̂_t = v_t / (1 - β₂^t)                         # Bias correction
θ_t = θ_{t-1} - lr * m̂_t / (√v̂_t + ε)         # Parameter update
```

Default hyperparameters: `β₁ = 0.9, β₂ = 0.999, ε = 1e-8`

### AdamW vs Adam + L2

```
Adam + L2:    g_t = ∇loss + λθ     (weight decay coupled with gradient)
AdamW:        θ_t -= lr * λ * θ_t  (weight decay applied directly to params)
```

AdamW decouples weight decay from the adaptive learning rate, which leads to better generalization. Use AdamW for transformer training.

## Learning Rate Scheduling

### Schedule Types

| Schedule | Shape | How It Works | When to Use |
|----------|-------|--------------|-------------|
| **Constant** | Flat | LR never changes | Baseline, quick experiments |
| **Warm-up** | Ramp up | Start low, linearly increase to target | Large batch training, transformers |
| **Step decay** | Staircase down | Reduce LR by factor at fixed epochs | CNNs, when you know good milestones |
| **Cosine annealing** | Smooth cosine curve | LR decays following cosine from max to min | Transformer pre-training, general DL |
| **Reduce on plateau** | Adaptive drops | Lower LR when val loss stalls for N epochs | When optimal schedule is unknown |
| **Cyclical** | Oscillating | LR oscillates between min and max bounds | Escaping local minima, exploration |
| **1cycle** | Rise then fall | Warm up to max, then anneal to near-zero | Fast convergence (Leslie Smith) |

### Warm-up + Cosine Decay (Common Transformer Schedule)

```python
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

warmup = LinearLR(optimizer, start_factor=0.1, total_iters=1000)
cosine = CosineAnnealingLR(optimizer, T_max=50000, eta_min=1e-6)
scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[1000])
```

### Why Warm-up Matters

- Early gradients are noisy (random weights) — large LR causes instability
- Warm-up lets Adam/AdamW accumulate reliable moment estimates before taking large steps
- Especially important for: large batch sizes, transformers, distributed training

## Regularization Techniques

### Overview

| Technique | Mechanism | Effect | When to Use |
|-----------|-----------|--------|-------------|
| **L1 (Lasso)** | Adds `λ * |w|` to loss | Encourages sparsity (zeros out weights) | Feature selection, sparse models |
| **L2 (Ridge)** | Adds `λ * w²` to loss | Penalizes large weights uniformly | General regularization, prevent any weight from dominating |
| **Dropout** | Randomly zero out neurons (p=0.1-0.5) | Ensemble effect, prevents co-adaptation | FC layers, transformers (attention dropout) |
| **Early stopping** | Stop when val loss increases for N epochs | Prevents overfitting to training data | Always use as a safety net |
| **Data augmentation** | Apply transforms to training samples | Increases effective dataset size | Images (flip, rotate, crop), text (back-translation) |
| **Batch normalization** | Normalize layer inputs to zero mean, unit variance | Stabilizes training, implicit regularization | CNNs (after conv layers), less common in transformers |
| **Layer normalization** | Normalize across features (not batch) | Stable with variable batch sizes | Transformers, RNNs |
| **Weight decay** | Shrink weights each step: `w *= (1 - λ*lr)` | Same as L2 for SGD, different for Adam | Use AdamW for proper weight decay with Adam |

### Dropout in Practice

```python
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(768, 256)
        self.dropout = nn.Dropout(p=0.1)  # 10% dropout rate
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)  # Only active during training
        return self.fc2(x)
```

**Important:** Dropout is automatically disabled during `model.eval()`. Always call `model.eval()` before inference.

## Deep Learning vs Traditional ML

### Decision Framework

```
                    ┌─ Small dataset (< 10K)?
                    │   └─ Yes → Traditional ML (risk of DL overfitting)
                    │
                    ├─ Unstructured data (images, text, audio)?
                    │   └─ Yes → Deep Learning
                    │
                    ├─ Tabular / structured data?
                    │   └─ Yes → Tree-based models (XGBoost, LightGBM)
Start ──────────────┤
                    ├─ Interpretability required?
                    │   └─ Yes → Traditional ML (linear models, decision trees)
                    │
                    ├─ Strict latency constraints (< 1ms)?
                    │   └─ Yes → Traditional ML (simpler models are faster)
                    │
                    └─ Large dataset + complex patterns?
                        └─ Yes → Deep Learning
```

### Detailed Comparison

| Aspect | Traditional ML | Deep Learning |
|--------|---------------|---------------|
| **Data requirements** | 100s–10Ks of samples | 10Ks–millions of samples |
| **Feature engineering** | Manual, domain expertise needed | Automatic feature extraction |
| **Interpretability** | High (coefficients, feature importance) | Low (black box, needs SHAP/LIME) |
| **Compute** | CPU sufficient | GPU/TPU required |
| **Training time** | Seconds–minutes | Hours–days–weeks |
| **Unstructured data** | Poor (needs manual features) | Excellent (images, text, audio) |
| **Tabular data** | Excellent (XGBoost, LightGBM) | Good but often overkill |
| **Inference latency** | Very low (< 1ms) | Higher (1-100ms+ depending on model) |

### When Deep Learning Wins
- Image recognition, computer vision
- Natural language processing (transformers)
- Speech recognition and generation
- Complex pattern recognition with large datasets
- Generative tasks (text, images, code)

### When Traditional ML Wins
- Small datasets (overfitting risk with DL)
- Tabular/structured data (gradient boosting often beats neural nets)
- Need for interpretability (healthcare, finance, regulatory)
- Limited compute resources
- Real-time inference with strict latency requirements (< 1ms)

## Transfer Learning

### Workflow

```
1. Pre-trained model (ImageNet, BERT, GPT)
   └─ Trained on large general dataset
        │
2. Freeze early layers
   └─ Generic features (edges, shapes / word embeddings, syntax)
        │
3. Fine-tune later layers
   └─ Task-specific features (your domain)
        │
4. Train on your dataset
   └─ Much less data needed (100s-1000s vs millions)
```

### Strategies

| Strategy | What to Do | When to Use |
|----------|-----------|-------------|
| **Feature extraction** | Freeze all pre-trained layers, only train new head | Very small dataset, similar domain |
| **Fine-tune top layers** | Freeze early layers, unfreeze and train later layers | Moderate dataset, somewhat similar domain |
| **Full fine-tuning** | Unfreeze all layers, train with low LR | Larger dataset, different domain |

### Example: Fine-tuning a Pre-trained Model

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=2
)

# Freeze base model, only train classifier head
for param in model.bert.parameters():
    param.requires_grad = False

# Later: unfreeze and fine-tune with lower LR
for param in model.bert.encoder.layer[-2:].parameters():  # Last 2 layers
    param.requires_grad = True
```

### Key Considerations
- **Learning rate:** Use a lower LR for pre-trained layers (1e-5) vs new layers (1e-3)
- **Domain similarity:** The more different your domain, the more layers to fine-tune
- **Dataset size:** Smaller dataset → freeze more layers to prevent overfitting

## Online vs Batch vs Mini-Batch Training

| Approach | Update Frequency | Memory | Convergence | Use Case |
|----------|-----------------|--------|-------------|----------|
| **Batch (full)** | After entire dataset | High (all data in memory) | Stable, slow | Small datasets, convex problems |
| **Online (SGD)** | After each sample | Low | Noisy, can escape local minima | Streaming data, infinite data |
| **Mini-batch** | After N samples (32-512) | Moderate | Good balance of stability and speed | Standard for deep learning |

**Mini-batch is the standard** for deep learning because:
- GPU parallelism requires batches (single samples underutilize GPU)
- Gradient estimates are noisy enough to regularize but stable enough to converge
- Batch size is a hyperparameter: larger → faster training, potentially worse generalization
