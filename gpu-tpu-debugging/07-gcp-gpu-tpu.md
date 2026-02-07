# Chapter 7: GCP GPU/TPU Infrastructure

## Overview

Google Cloud Platform provides GPU and TPU infrastructure for ML workloads through several services. Understanding the options helps you choose the right approach for your use case.

```
GCP ML Compute Options:
┌─────────────────────────────────────────────────────────┐
│                     Vertex AI                            │
│  (Managed ML platform — training, serving, pipelines)   │
│  ┌─────────────────┐  ┌─────────────────────────────┐  │
│  │ Custom Training  │  │ AutoML / Model Garden        │  │
│  │ (your code)      │  │ (pre-built models)           │  │
│  └─────────────────┘  └─────────────────────────────┘  │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │   GKE       │  │  Compute     │  │  Cloud TPU    │  │
│  │  (K8s +     │  │  Engine      │  │  (Managed     │  │
│  │   GPUs)     │  │  (GPU VMs)   │  │   TPU VMs)    │  │
│  └─────────────┘  └──────────────┘  └───────────────┘  │
│                                                          │
│                 Infrastructure Layer                     │
└─────────────────────────────────────────────────────────┘
```

## GCP GPU Options

### Available GPU Types

| GPU | Memory | Use Case | Machine Type Prefix |
|-----|--------|----------|-------------------|
| NVIDIA T4 | 16 GB | Inference, light training | `n1-standard-*` + T4 |
| NVIDIA L4 | 24 GB | Inference, fine-tuning | `g2-standard-*` |
| NVIDIA V100 | 16 GB | Training | `n1-standard-*` + V100 |
| NVIDIA A100 (40GB) | 40 GB | Large-scale training | `a2-highgpu-*` |
| NVIDIA A100 (80GB) | 80 GB | Large model training | `a2-ultragpu-*` |
| NVIDIA H100 (80GB) | 80 GB | LLM training | `a3-highgpu-*` |

### Creating a GPU VM

```bash
# Create a VM with 4x A100 GPUs
gcloud compute instances create gpu-training-vm \
    --zone=us-central1-a \
    --machine-type=a2-highgpu-4g \
    --accelerator=type=nvidia-tesla-a100,count=4 \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=200GB \
    --maintenance-policy=TERMINATE \
    --metadata="install-nvidia-driver=True"

# SSH into the VM
gcloud compute ssh gpu-training-vm --zone=us-central1-a

# Verify GPUs are available
nvidia-smi
```

### GPU Quotas

GPU quotas are the most common blocker on GCP. You need to request quota increases before using GPUs.

```bash
# Check your current GPU quotas
gcloud compute regions describe us-central1 \
    --format="table(quotas.filter(metric='NVIDIA_A100_GPUS'))"

# Request a quota increase through the console:
# Console → IAM & Admin → Quotas → Filter by GPU type → Edit Quotas
```

**Tips for quota requests:**
- Request for specific zones (us-central1-a, us-east1-b, etc.)
- Start small (4-8 GPUs), then increase after initial approval
- Preemptible/spot quotas are separate and often easier to get
- A100 and H100 quotas are limited; apply early

## Cloud TPU

### TPU Types and Pricing

| TPU Type | Cores | HBM per chip | Topology | Best For |
|----------|-------|-------------|----------|----------|
| v2-8 | 8 | 8 GB | Single host | Small experiments |
| v3-8 | 8 | 16 GB | Single host | Medium training |
| v4-8 | 8 | 32 GB | Single host | Large training |
| v5e-4 | 4 | 16 GB | Single host | Cost-efficient inference |
| v5p-8 | 8 | 95 GB | Single host | Large model training |
| v4-32 to v4-8192 | 32-8192 | 32 GB/chip | Multi-host pod | Distributed training |

### Creating a TPU VM

```bash
# Create a TPU v4-8 VM
gcloud compute tpus tpu-vm create my-tpu \
    --zone=us-central2-b \
    --accelerator-type=v4-8 \
    --version=tpu-vm-tf-2.14.0  # Or tpu-vm-pt-2.1 for PyTorch

# SSH into the TPU VM
gcloud compute tpus tpu-vm ssh my-tpu --zone=us-central2-b

# For multi-host TPU pods, SSH into specific worker
gcloud compute tpus tpu-vm ssh my-tpu --zone=us-central2-b --worker=0
```

### TPU VM vs TPU Node (Legacy)

| Feature | TPU VM (current) | TPU Node (legacy) |
|---------|-----------------|-------------------|
| Access | Direct SSH to TPU host | Separate GCE VM + TPU over gRPC |
| Performance | Better (no network hop) | Slower (data transfers over network) |
| Debugging | Standard Linux tools | Limited (must debug through GCE VM) |
| **Recommendation** | **Use this** | Avoid for new projects |

### Running JAX on TPU

```python
# JAX automatically detects TPU
import jax
print(jax.devices())  # Should show TPU devices

# Example: simple training step on TPU
import jax.numpy as jnp
from jax import grad, jit

@jit
def loss_fn(params, x, y):
    pred = jnp.dot(x, params)
    return jnp.mean((pred - y) ** 2)

@jit
def train_step(params, x, y, lr=0.01):
    grads = grad(loss_fn)(params, x, y)
    return params - lr * grads

# Initialize on TPU
params = jnp.ones((784, 10))
x = jnp.ones((32, 784))
y = jnp.ones((32, 10))

# Train
for step in range(1000):
    params = train_step(params, x, y)
```

### Running PyTorch/XLA on TPU

```python
import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp

def train_fn(index, flags):
    device = xm.xla_device()

    model = MyModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Use ParallelLoader for efficient data loading on TPU
    train_loader = DataLoader(dataset, batch_size=32)
    para_loader = pl.ParallelLoader(train_loader, [device])

    for epoch in range(num_epochs):
        for data, target in para_loader.per_device_loader(device):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            xm.optimizer_step(optimizer)  # Includes mark_step()

# Launch on all TPU cores
xmp.spawn(train_fn, args=(flags,))
```

## Vertex AI

Vertex AI is Google's managed ML platform. It abstracts infrastructure management so you can focus on ML code.

### Custom Training Jobs

```bash
# Submit a custom training job with GPUs
gcloud ai custom-jobs create \
    --region=us-central1 \
    --display-name=my-training-job \
    --worker-pool-spec=machine-type=a2-highgpu-4g,replica-count=1,accelerator-type=NVIDIA_TESLA_A100,accelerator-count=4,container-image-uri=us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-13:latest \
    --command='python train.py' \
    --args='--epochs=10,--batch-size=64'
```

### Custom Training with Python SDK

```python
from google.cloud import aiplatform

aiplatform.init(project="my-project", location="us-central1")

# Define custom training job
job = aiplatform.CustomTrainingJob(
    display_name="my-training-job",
    script_path="train.py",
    container_uri="us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-13:latest",
    requirements=["transformers", "datasets"],
)

# Run with GPU
model = job.run(
    replica_count=1,
    machine_type="a2-highgpu-4g",
    accelerator_type="NVIDIA_TESLA_A100",
    accelerator_count=4,
)
```

### Vertex AI Training with TPU

```python
from google.cloud import aiplatform

job = aiplatform.CustomTrainingJob(
    display_name="tpu-training-job",
    script_path="train.py",
    container_uri="us-docker.pkg.dev/vertex-ai/training/tf-tpu.2-12:latest",
)

# Run on TPU
model = job.run(
    replica_count=1,
    machine_type="cloud-tpu",
    accelerator_type="TPU_V4_POD",
    accelerator_count=8,  # v4-8
)
```

### Debugging Vertex AI Jobs

```bash
# View job logs
gcloud ai custom-jobs describe JOB_ID --region=us-central1

# Stream logs
gcloud ai custom-jobs stream-logs JOB_ID --region=us-central1

# List recent jobs
gcloud ai custom-jobs list --region=us-central1 --limit=10

# Cancel a running job
gcloud ai custom-jobs cancel JOB_ID --region=us-central1
```

**Common Vertex AI issues:**

| Issue | Cause | Solution |
|-------|-------|----------|
| Job stuck in PENDING | Insufficient quota | Check GPU/TPU quota in the region |
| Container fails to start | Missing dependencies | Test container locally first |
| OOM during training | Batch size too large for GPU | Reduce batch size or use larger GPU |
| Job timeout | Default timeout exceeded | Set `--timeout` flag |
| Permission denied | Service account lacks permissions | Grant `roles/ml.developer` and storage access |

## GKE with GPUs

Google Kubernetes Engine can run GPU workloads using node pools with GPUs attached.

### Setting Up a GPU Node Pool

```bash
# Create a GKE cluster
gcloud container clusters create ml-cluster \
    --zone=us-central1-a \
    --num-nodes=1 \
    --machine-type=n1-standard-4

# Add a GPU node pool
gcloud container node-pools create gpu-pool \
    --cluster=ml-cluster \
    --zone=us-central1-a \
    --machine-type=a2-highgpu-4g \
    --accelerator=type=nvidia-tesla-a100,count=4 \
    --num-nodes=2 \
    --enable-autoscaling --min-nodes=0 --max-nodes=4

# Install NVIDIA GPU drivers (DaemonSet)
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded-latest.yaml
```

### GPU Pod Specification

```yaml
# gpu-training-pod.yaml
apiVersion: v1
kind: Pod
metadata:
  name: gpu-training
spec:
  containers:
  - name: training
    image: pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
    command: ["python", "train.py"]
    resources:
      limits:
        nvidia.com/gpu: 4    # Request 4 GPUs
      requests:
        cpu: "16"
        memory: "64Gi"
    volumeMounts:
    - name: training-data
      mountPath: /data
  volumes:
  - name: training-data
    persistentVolumeClaim:
      claimName: training-pvc
  nodeSelector:
    cloud.google.com/gke-accelerator: nvidia-tesla-a100
  tolerations:
  - key: nvidia.com/gpu
    operator: Exists
    effect: NoSchedule
```

### Multi-Node Training on GKE

For distributed training across multiple nodes, use Kubernetes Jobs or training operators:

```yaml
# pytorch-distributed-job.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: distributed-training
spec:
  parallelism: 2         # 2 nodes
  completions: 2
  template:
    spec:
      containers:
      - name: training
        image: my-training-image:latest
        command:
        - torchrun
        - --nproc_per_node=4
        - --nnodes=2
        - --rdzv_backend=c10d
        - --rdzv_endpoint=$(MASTER_ADDR):29500
        - train.py
        resources:
          limits:
            nvidia.com/gpu: 4
        env:
        - name: MASTER_ADDR
          value: "distributed-training-0"  # First pod's hostname
        - name: NCCL_DEBUG
          value: "INFO"
      restartPolicy: Never
```

### Debugging GPU Pods on GKE

```bash
# Check pod status
kubectl get pods -o wide

# Check GPU availability on nodes
kubectl describe nodes | grep -A 5 "nvidia.com/gpu"

# View pod logs
kubectl logs gpu-training -f

# Exec into running pod for debugging
kubectl exec -it gpu-training -- bash
nvidia-smi  # Verify GPUs are visible inside container

# Check events for scheduling issues
kubectl describe pod gpu-training
# Look for events like:
# "Insufficient nvidia.com/gpu" — not enough GPUs available
# "FailedScheduling" — no node matches requirements
```

## Cost Optimization

### Preemptible / Spot VMs

Spot VMs are 60-91% cheaper but can be preempted with 30 seconds notice.

```bash
# Create a spot GPU VM
gcloud compute instances create spot-gpu-vm \
    --zone=us-central1-a \
    --machine-type=a2-highgpu-4g \
    --accelerator=type=nvidia-tesla-a100,count=4 \
    --provisioning-model=SPOT \
    --instance-termination-action=STOP \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release
```

**Making training preemption-safe:**
```python
# Save checkpoints frequently
import signal
import sys

def handle_preemption(signum, frame):
    """Save checkpoint when preempted."""
    print("Preemption signal received, saving checkpoint...")
    save_checkpoint(model, optimizer, epoch, step)
    sys.exit(0)

# GCP sends SIGTERM 30 seconds before preemption
signal.signal(signal.SIGTERM, handle_preemption)

# Also save checkpoints periodically during training
for step, (data, target) in enumerate(dataloader):
    # ... training step ...
    if step % checkpoint_interval == 0:
        save_checkpoint(model, optimizer, epoch, step)
```

### Reserved Instances (CUDs)

Committed Use Discounts provide 1-year or 3-year discounts:
- 1-year commitment: ~37% discount
- 3-year commitment: ~55% discount

Use these for sustained, predictable GPU workloads.

### Cost Monitoring

```bash
# View estimated costs
gcloud billing budgets list

# Set up a budget alert
gcloud billing budgets create \
    --billing-account=BILLING_ACCOUNT_ID \
    --display-name="GPU Budget" \
    --budget-amount=1000USD \
    --threshold-rule=percent=0.5 \
    --threshold-rule=percent=0.9 \
    --threshold-rule=percent=1.0
```

## Key Takeaways

1. **Choose the right abstraction level:** Vertex AI for managed training, GKE for containerized workloads, Compute Engine for full control.
2. **GPU quotas** are the most common blocker — request quota increases early and for specific zones.
3. **TPU VMs** (not TPU Nodes) are the current recommended way to use Cloud TPU.
4. **Spot/preemptible VMs** save 60-91% on GPU costs — make your training preemption-safe with frequent checkpointing.
5. **GKE GPU node pools** with autoscaling let you scale GPU resources up and down with demand.
6. **Always test locally first** (or on a small VM) before submitting large training jobs to avoid expensive failures.
