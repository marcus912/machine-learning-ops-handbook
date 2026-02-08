# Chapter 10: GCP Vertex AI Ecosystem

## Overview

Chapter 7 covered GCP's core infrastructure (Compute Engine, Cloud TPU, GKE). This chapter covers the managed services layer — the Vertex AI ecosystem that sits on top of that infrastructure.

```
Vertex AI Ecosystem:
┌──────────────────────────────────────────────────────────────┐
│                    Development Layer                          │
│  ┌──────────────────────┐  ┌──────────────────────────────┐ │
│  │ Vertex AI Workbench   │  │ Colab Enterprise              │ │
│  │ (managed JupyterLab)  │  │ (collaborative notebooks)     │ │
│  └──────────────────────┘  └──────────────────────────────┘ │
├──────────────────────────────────────────────────────────────┤
│                   Training & Tuning Layer                     │
│  ┌──────────────────────┐  ┌──────────────────────────────┐ │
│  │ Custom Training       │  │ Hyperparameter Tuning         │ │
│  │ (Ch 7)                │  │ (automated search)            │ │
│  └──────────────────────┘  └──────────────────────────────┘ │
├──────────────────────────────────────────────────────────────┤
│                   Serving & Monitoring Layer                  │
│  ┌──────────────────────┐  ┌──────────────────────────────┐ │
│  │ Predictions /         │  │ Vertex AI TensorBoard         │ │
│  │ Endpoints (GPU/TPU)   │  │ (managed profiling)           │ │
│  └──────────────────────┘  └──────────────────────────────┘ │
├──────────────────────────────────────────────────────────────┤
│              AI Hypercomputer (infrastructure)                │
│      TPU v5p/v6e  •  A3 Ultra (H200)  •  A4 (B200)          │
│      Pathways  •  XLA  •  Dynamic Workload Scheduler         │
└──────────────────────────────────────────────────────────────┘
```

## Vertex AI Workbench

Vertex AI Workbench provides managed JupyterLab instances with GPU and TPU support, pre-installed ML frameworks, and direct integration with GCP services (BigQuery, Cloud Storage, Vertex AI training).

### Instance Types

| Type | Description | Best For |
|------|-------------|----------|
| **Instances** (current) | Fully managed JupyterLab on Compute Engine VMs | Production development, GPU-heavy work |
| **Managed Notebooks** (legacy) | Google-managed runtime | Being deprecated in favor of Instances |
| **User-Managed Notebooks** (legacy) | Full VM control with JupyterLab | Being deprecated in favor of Instances |

### Creating a Workbench Instance with GPU

```bash
# Create a Workbench instance with T4 GPU
gcloud workbench instances create my-notebook \
    --location=us-central1-a \
    --machine-type=n1-standard-8 \
    --accelerator-type=NVIDIA_TESLA_T4 \
    --accelerator-core-count=1 \
    --install-gpu-driver

# Create with A100 for large model work
gcloud workbench instances create my-gpu-notebook \
    --location=us-central1-a \
    --machine-type=a2-highgpu-1g \
    --accelerator-type=NVIDIA_TESLA_A100 \
    --accelerator-core-count=1 \
    --install-gpu-driver \
    --boot-disk-size=200

# List instances
gcloud workbench instances list --location=us-central1-a

# Open in browser
gcloud workbench instances describe my-notebook \
    --location=us-central1-a \
    --format="value(gceSetup.proxyUri)"
```

### Using GPU Inside Workbench

Once connected to JupyterLab, GPUs are available immediately:

```python
# In a notebook cell
import torch
print(f"GPU available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
```

```python
# Run nvidia-smi from notebook
!nvidia-smi
```

### Workbench Integration with GCP Services

```python
# Read data directly from BigQuery
from google.cloud import bigquery
client = bigquery.Client()
df = client.query("SELECT * FROM my_dataset.my_table LIMIT 1000").to_dataframe()

# Read data from Cloud Storage
import gcsfs
fs = gcsfs.GCSFileSystem()
with fs.open("gs://my-bucket/data/train.csv") as f:
    df = pd.read_csv(f)

# Submit a Vertex AI training job from notebook
from google.cloud import aiplatform
aiplatform.init(project="my-project", location="us-central1")

job = aiplatform.CustomTrainingJob(
    display_name="training-from-notebook",
    script_path="train.py",
    container_uri="us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-1:latest",
)
job.run(machine_type="a2-highgpu-4g", accelerator_type="NVIDIA_TESLA_A100", accelerator_count=4)
```

### Common Workbench Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Kernel dies during training | OOM — model too large for GPU | Reduce batch size, or use a larger GPU instance |
| GPU not detected in notebook | Driver not installed | Recreate with `--install-gpu-driver`, or run `sudo /opt/deeplearning/install-driver.sh` |
| Slow notebook startup | Large boot disk or custom container | Use default container image, increase disk IOPS |
| Idle shutdown | Auto-shutdown after inactivity | Configure `--idle-shutdown-timeout` (default 180 min) |
| Permission denied to GCS/BQ | Service account lacks roles | Grant `roles/storage.objectViewer` and `roles/bigquery.user` |

## Colab Enterprise

Colab Enterprise is Google's managed notebook service for teams, integrated with GCP projects and VPC. It combines the familiar Colab interface with enterprise security and GPU/TPU access.

### Colab Enterprise vs Vertex AI Workbench

| Feature | Colab Enterprise | Vertex AI Workbench |
|---------|-----------------|-------------------|
| **Interface** | Colab-style (familiar) | Full JupyterLab |
| **Collaboration** | Real-time multi-user editing | Single user per instance |
| **GPU support** | Via runtime templates | Direct GPU attachment |
| **Persistence** | Notebooks stored in GCS | Persistent VM with local disk |
| **Customization** | Runtime templates | Full VM control |
| **Cost model** | Pay for runtime (auto-stops) | Pay for VM (manual stop) |
| **Best for** | Team collaboration, exploration | Solo development, heavy GPU work |

### Setting Up Colab Enterprise with GPUs

```bash
# Colab Enterprise is enabled per-region in your GCP project
# Configure through Console: Vertex AI → Colab Enterprise

# Create a runtime template with GPU
gcloud colab runtime-templates create gpu-template \
    --region=us-central1 \
    --machine-type=n1-standard-8 \
    --accelerator-type=NVIDIA_TESLA_T4 \
    --accelerator-count=1

# List runtime templates
gcloud colab runtime-templates list --region=us-central1
```

**Using Colab Enterprise:**
1. Navigate to Vertex AI → Colab Enterprise in the Cloud Console
2. Create or open a notebook
3. Select a runtime template (CPU-only or GPU-enabled)
4. The runtime starts with pre-installed ML frameworks
5. Access GCS and BigQuery directly from notebook cells

### Debugging Colab Enterprise

| Issue | Cause | Solution |
|-------|-------|----------|
| "No runtimes available" | Quota or capacity limit | Check GPU quota for the region, try different zone |
| Runtime disconnects | Idle timeout or preemption | Save work frequently, increase idle timeout in template |
| Package not found | Not in default runtime | `!pip install package` or create custom runtime template |
| Slow data loading | Reading from remote GCS | Use `gcsfs` with caching, or copy data to runtime local disk |

## Vertex AI TensorBoard

Vertex AI TensorBoard is a managed version of TensorBoard integrated with Vertex AI training jobs. It provides experiment tracking, metric visualization, and profiling without managing your own TensorBoard server.

### Setting Up Managed TensorBoard

```python
from google.cloud import aiplatform

aiplatform.init(project="my-project", location="us-central1")

# Create a TensorBoard instance
tb = aiplatform.Tensorboard.create(
    display_name="my-experiment-tracker",
    description="Training experiments for my model",
)
print(f"TensorBoard resource: {tb.resource_name}")
```

### Integrating with Training Jobs

```python
from google.cloud import aiplatform

# Create training job with TensorBoard integration
job = aiplatform.CustomTrainingJob(
    display_name="tracked-training",
    script_path="train.py",
    container_uri="us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-1:latest",
)

# Run with TensorBoard — logs are automatically uploaded
job.run(
    machine_type="a2-highgpu-4g",
    accelerator_type="NVIDIA_TESLA_A100",
    accelerator_count=4,
    tensorboard="projects/my-project/locations/us-central1/tensorboards/TB_ID",
    service_account="my-sa@my-project.iam.gserviceaccount.com",
)
```

### Writing Logs for Vertex AI TensorBoard

```python
# In your training script — use standard TensorBoard writers
from torch.utils.tensorboard import SummaryWriter

# Vertex AI expects logs in a specific env var path
import os
log_dir = os.environ.get("AIP_TENSORBOARD_LOG_DIR", "./logs")
writer = SummaryWriter(log_dir=log_dir)

for step in range(num_steps):
    # ... training step ...
    writer.add_scalar("loss/train", loss.item(), step)
    writer.add_scalar("metrics/gpu_memory_gb",
                      torch.cuda.memory_allocated() / 1e9, step)
    writer.add_scalar("metrics/learning_rate",
                      optimizer.param_groups[0]["lr"], step)

writer.close()
```

### GPU/TPU Profiling via TensorBoard

Vertex AI TensorBoard includes profiling views for GPU/TPU workloads (see Chapter 6 for profiling fundamentals):

- **Overview page** — Step time breakdown: kernel execution, communication, idle
- **Trace viewer** — Timeline of CPU and GPU/TPU operations (find gaps and bottlenecks)
- **GPU kernel stats** — Individual kernel execution times
- **Memory profile** — GPU memory usage over time
- **TPU compatibility** — XLA operation-level profiling for TPU workloads

```bash
# View TensorBoard in browser
gcloud ai tensorboards open TB_ID --region=us-central1

# Or get the URL
gcloud ai tensorboards describe TB_ID --region=us-central1 \
    --format="value(blobStoragePathPrefix)"
```

### Common TensorBoard Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Logs not appearing | Wrong log directory | Use `AIP_TENSORBOARD_LOG_DIR` env var in training script |
| Missing profiling data | Profiler not enabled in training | Add PyTorch/TF profiler code (see Chapter 6) |
| Stale data | Upload delay | Logs upload periodically; wait or check `SummaryWriter.flush()` |
| Permission denied | Service account issue | Grant `roles/aiplatform.user` to the training SA |

## Vertex AI Predictions / Endpoints

Vertex AI Endpoints serve trained models for inference, with GPU acceleration and autoscaling.

### Online vs Batch Predictions

| Feature | Online Predictions | Batch Predictions |
|---------|-------------------|-------------------|
| **Latency** | Low (real-time) | High (minutes to hours) |
| **Requires endpoint** | Yes (deployed model) | No (runs as a job) |
| **Scaling** | Autoscales with traffic | Scales with data volume |
| **Use case** | API serving, real-time apps | Bulk scoring, offline processing |
| **GPU support** | Yes | Yes |

### Deploying a Model with GPU Serving

```python
from google.cloud import aiplatform

aiplatform.init(project="my-project", location="us-central1")

# Upload model to Vertex AI Model Registry
model = aiplatform.Model.upload(
    display_name="my-model",
    artifact_uri="gs://my-bucket/model/",
    serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/pytorch-gpu.2-1:latest",
)

# Create an endpoint
endpoint = aiplatform.Endpoint.create(display_name="my-endpoint")

# Deploy model to endpoint with GPU
model.deploy(
    endpoint=endpoint,
    machine_type="n1-standard-8",
    accelerator_type="NVIDIA_TESLA_T4",
    accelerator_count=1,
    min_replica_count=1,
    max_replica_count=5,        # Autoscaling up to 5 replicas
    traffic_percentage=100,
)
```

### Getting Predictions

```python
# Online prediction
instances = [{"input": [1.0, 2.0, 3.0]}, {"input": [4.0, 5.0, 6.0]}]
predictions = endpoint.predict(instances=instances)
print(predictions.predictions)

# Batch prediction (no endpoint needed)
batch_job = model.batch_predict(
    job_display_name="batch-scoring",
    gcs_source="gs://my-bucket/input/data.jsonl",
    gcs_destination_prefix="gs://my-bucket/output/",
    machine_type="n1-standard-8",
    accelerator_type="NVIDIA_TESLA_T4",
    accelerator_count=1,
)
```

### Debugging Serving Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| High latency (p99 > 1s) | Cold start or model too large | Set `min_replica_count >= 1` to keep warm instances |
| OOM during inference | Batch too large or model exceeds GPU memory | Reduce request batch size, use larger GPU |
| 503 errors | All replicas busy | Increase `max_replica_count`, check request rate |
| Model loading timeout | Large model artifact | Use a larger machine type, optimize model serialization |
| Prediction mismatch | Preprocessing difference | Ensure serving preprocessing matches training |

```bash
# Debug endpoint issues
gcloud ai endpoints describe ENDPOINT_ID --region=us-central1

# Check deployed model status
gcloud ai endpoints list --region=us-central1

# View serving logs
gcloud logging read 'resource.type="aiplatform.googleapis.com/Endpoint"' \
    --limit=50 --format=json
```

## Vertex AI Hyperparameter Tuning

Vertex AI Hyperparameter Tuning automates the search for optimal training parameters (learning rate, batch size, architecture choices) across multiple GPU/TPU trials.

### How It Works

```
Hyperparameter Tuning Flow:
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Define search   │────▶│  Run parallel    │────▶│  Select best    │
│  space & metric  │     │  training trials │     │  parameters     │
└─────────────────┘     │  (each on GPU)   │     └─────────────────┘
                         └─────────────────┘
                         Trial 1: lr=1e-3, bs=32
                         Trial 2: lr=1e-4, bs=64
                         Trial 3: lr=5e-4, bs=32
                         ...
```

**Search algorithms:**

| Algorithm | Description | Best For |
|-----------|-------------|----------|
| Bayesian (default) | Uses prior results to guide search | Most cases — efficient exploration |
| Grid search | Exhaustive search over all combinations | Small, discrete parameter spaces |
| Random search | Random sampling from parameter space | Large spaces, quick initial exploration |

### Setting Up a Tuning Job

```python
from google.cloud import aiplatform

aiplatform.init(project="my-project", location="us-central1")

# Define the tuning job
hp_job = aiplatform.HyperparameterTuningJob(
    display_name="tune-my-model",
    custom_job=aiplatform.CustomJob(
        display_name="trial",
        worker_pool_specs=[{
            "machine_spec": {
                "machine_type": "a2-highgpu-1g",
                "accelerator_type": "NVIDIA_TESLA_A100",
                "accelerator_count": 1,
            },
            "replica_count": 1,
            "container_spec": {
                "image_uri": "us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-1:latest",
                "command": ["python", "train.py"],
            },
        }],
    ),
    metric_spec={"val_accuracy": "maximize"},
    parameter_spec={
        "learning_rate": aiplatform.hyperparameter_tuning.DoubleParameterSpec(
            min=1e-5, max=1e-2, scale="log",
        ),
        "batch_size": aiplatform.hyperparameter_tuning.DiscreteParameterSpec(
            values=[16, 32, 64, 128], scale="linear",
        ),
        "dropout": aiplatform.hyperparameter_tuning.DoubleParameterSpec(
            min=0.0, max=0.5, scale="linear",
        ),
    },
    max_trial_count=20,
    parallel_trial_count=4,      # Run 4 trials in parallel (4 GPUs)
    search_algorithm="random",   # Or None for Bayesian (default)
)

hp_job.run()

# Get best trial
best = hp_job.trials[0]  # Sorted by objective
print(f"Best params: {best.parameters}")
print(f"Best accuracy: {best.final_measurement.metrics[0].value}")
```

### Reading HP Parameters in Training Script

```python
# train.py — Vertex AI passes HP values as command-line args
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--dropout", type=float, default=0.1)
args = parser.parse_args()

# Use args.learning_rate, args.batch_size, args.dropout in training

# Report metric back to Vertex AI HP tuning service
import hypertune
hpt = hypertune.HyperTune()
hpt.report_hyperparameter_tuning_metric(
    hyperparameter_metric_tag='val_accuracy',
    metric_value=final_accuracy,
    global_step=epoch)
```

### Debugging Tuning Jobs

```bash
# Check tuning job status
gcloud ai hp-tuning-jobs describe JOB_ID --region=us-central1

# List trials and their metrics
gcloud ai hp-tuning-jobs list --region=us-central1

# Stream logs for a specific trial
gcloud ai custom-jobs stream-logs TRIAL_JOB_ID --region=us-central1
```

**Common issues:**
- **All trials fail:** Check that the training script handles HP args correctly and reports the metric
- **Quota exhaustion:** `parallel_trial_count` GPUs run simultaneously — ensure enough quota
- **Poor results:** Widen the search space, increase `max_trial_count`, or switch to Bayesian search

## AI Hypercomputer

AI Hypercomputer is Google's integrated infrastructure architecture for large-scale AI training. It's not a single product — it's the combination of hardware, software, and orchestration optimized to work together.

### Key Components

```
AI Hypercomputer Stack:
┌─────────────────────────────────────────────────────────┐
│  ML Frameworks: JAX, PyTorch, TensorFlow                 │
├─────────────────────────────────────────────────────────┤
│  Compiler & Runtime: XLA, Pathways, GSPMD               │
├─────────────────────────────────────────────────────────┤
│  Orchestration: GKE, Dynamic Workload Scheduler (DWS)   │
├─────────────────────────────────────────────────────────┤
│  Storage: Hyperdisk ML, GCS FUSE, Parallelstore         │
├─────────────────────────────────────────────────────────┤
│  Hardware: TPU v5p/v6e, A3 Ultra (H200), A4 (B200)     │
│  Networking: ICI, NVLink, InfiniBand, Jupiter fabric    │
└─────────────────────────────────────────────────────────┘
```

| Component | Role |
|-----------|------|
| **TPU v5p / v6e** | Google's latest TPU accelerators for training and inference |
| **A3 Ultra / A4** | GPU instances with H200 and B200 for GPU-based workloads |
| **Pathways** | Google's distributed ML runtime for multi-host orchestration |
| **XLA / GSPMD** | Compiler-level optimization and automatic parallelism |
| **Dynamic Workload Scheduler** | Queues and schedules GPU/TPU jobs based on capacity |
| **Hyperdisk ML** | High-throughput storage optimized for checkpoint loading |
| **GCS FUSE** | Mount Cloud Storage as a local filesystem for training data |
| **Parallelstore** | High-performance parallel filesystem for multi-node I/O |

### When AI Hypercomputer Matters

| Scale | Approach | Infrastructure |
|-------|----------|---------------|
| Single GPU | Vertex AI or Compute Engine | A2/A3/G2 instance |
| Multi-GPU (1 node) | DDP/FSDP on a single VM | A2/A3 with NVLink |
| Multi-node (< 100 GPUs) | GKE + DDP/FSDP | A3 node pool + InfiniBand |
| Large-scale (100+ accelerators) | **AI Hypercomputer** | TPU pods or A3/A4 clusters with DWS |

**For an interview:** AI Hypercomputer represents Google's strategy of vertically integrating hardware, compilers, and orchestration for ML workloads. The key differentiator vs. competitors is the co-design of TPU hardware with XLA compiler and Pathways runtime, enabling higher utilization at scale.

### Dynamic Workload Scheduler (DWS)

DWS manages GPU/TPU allocation for training workloads on GKE:

```bash
# DWS is configured through GKE — create a cluster with DWS enabled
gcloud container clusters create ml-cluster \
    --zone=us-central1-a \
    --enable-dynamic-workload-scheduler

# Submit a training job that DWS will schedule
# DWS handles: capacity reservation, preemption recovery, gang scheduling
```

**Key DWS concepts:**
- **Gang scheduling** — All pods for a distributed training job start together or not at all (prevents partial allocation deadlocks)
- **Capacity reservation** — Reserve GPU/TPU capacity for predictable scheduling
- **Flex-start** — Lower-priority jobs that run when capacity is available (similar to spot but managed by DWS)

## Key Takeaways

1. **Vertex AI Workbench** provides managed JupyterLab with GPU/TPU — use it for interactive development with direct access to BigQuery, GCS, and Vertex AI training.
2. **Colab Enterprise** adds real-time collaboration on top of GCP-managed notebooks — use it for team exploration, Workbench for heavy individual GPU work.
3. **Vertex AI TensorBoard** eliminates TensorBoard infrastructure management — integrate it with training jobs via the `AIP_TENSORBOARD_LOG_DIR` env var for automatic log upload.
4. **Vertex AI Endpoints** serve models with GPU acceleration and autoscaling — set `min_replica_count >= 1` to avoid cold start latency in production.
5. **Hyperparameter Tuning** automates the HP search loop — bound `parallel_trial_count` by your GPU quota to avoid resource exhaustion.
6. **AI Hypercomputer** is Google's integrated ML infrastructure stack — understand it conceptually as the co-design of TPU/GPU hardware, XLA compiler, and DWS orchestration for large-scale training.
