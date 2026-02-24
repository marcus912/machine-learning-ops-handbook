# Chapter 15: MLOps Practices

This chapter covers the operational side of ML systems — CI/CD for models, monitoring in production, deployment strategies, containerization, microservices patterns, and a structured framework for ML system design.

## CI/CD for ML

ML systems have two CI/CD dimensions: the code and the model.

### Code CI/CD vs Model CI/CD

| Aspect | Code CI/CD | Model CI/CD |
|--------|-----------|-------------|
| **Trigger** | Code commit | New training data, schedule, or performance degradation |
| **Tests** | Unit tests, integration tests, linting | Model performance tests, bias audits, data validation |
| **Artifacts** | Docker images, packages | Model files, metrics reports, data snapshots |
| **Validation** | Tests pass/fail | Metrics above threshold, no regression |
| **Deployment** | Rolling update, blue/green | Shadow mode → canary → full rollout |

### ML Pipeline CI/CD

```
Code Change          Data Change            Schedule
     │                    │                     │
     ▼                    ▼                     ▼
┌─────────┐      ┌──────────────┐      ┌──────────────┐
│ Lint +   │      │ Data         │      │ Retrain      │
│ Unit Test│      │ Validation   │      │ Pipeline     │
└────┬─────┘      └──────┬───────┘      └──────┬───────┘
     │                    │                     │
     ▼                    ▼                     ▼
┌─────────┐      ┌──────────────┐      ┌──────────────┐
│ Build    │      │ Feature      │      │ Evaluate     │
│ Pipeline │      │ Engineering  │      │ Metrics      │
└────┬─────┘      └──────┬───────┘      └──────┬───────┘
     │                    │                     │
     └────────────────────┼─────────────────────┘
                          ▼
                   ┌──────────────┐
                   │ Model        │
                   │ Validation   │
                   │ Gate         │
                   └──────┬───────┘
                          ▼
                   ┌──────────────┐
                   │ Registry +   │
                   │ Deploy       │
                   └──────────────┘
```

### Infrastructure as Code for ML

| Tool | Scope | Key Features |
|------|-------|-------------|
| **Terraform** | Multi-cloud infrastructure | Declarative, state management, large provider ecosystem |
| **Pulumi** | Multi-cloud infrastructure | Real programming languages (Python, Go, TS), better for complex logic |
| **Helm** | Kubernetes applications | Chart-based packaging, templating, release management |
| **KServe** | Model serving on k8s | Serverless inference, autoscaling, canary rollouts |

```hcl
# Terraform: Vertex AI training pipeline
resource "google_vertex_ai_training_pipeline" "training" {
  display_name = "model-training-v2"
  region       = "us-central1"

  training_task_definition = "gs://google-cloud-aiplatform/schema/trainingjob/definition/custom_task_1.0.0.yaml"
  training_task_inputs     = jsonencode({
    workerPoolSpecs = [{
      machineSpec = {
        machineType      = "n1-standard-8"
        acceleratorType  = "NVIDIA_TESLA_V100"
        acceleratorCount = 1
      }
      replicaCount = 1
      containerSpec = {
        imageUri = "gcr.io/my-project/training:latest"
      }
    }]
  })
}
```

## Model Monitoring

### Types of Drift

| Drift Type | What Changes | Detection Method | Impact |
|------------|-------------|------------------|--------|
| **Data drift** | Input feature distributions shift | PSI, KS test, JS divergence | Model receives unexpected inputs |
| **Concept drift** | Relationship between features and target changes | Performance metric degradation | Model predictions become wrong |
| **Feature drift** | Individual feature distributions shift | Per-feature monitoring, anomaly detection | Subset of predictions affected |
| **Label drift** | Target distribution changes | Monitor prediction distribution | May need to rebalance training |

### Monitoring Architecture

```
Production Traffic
       │
       ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ Log Requests │────►│ Feature Store │────►│ Drift        │
│ + Predictions│     │ (comparison)  │     │ Detection    │
└──────────────┘     └──────────────┘     └──────┬───────┘
                                                  │
       ┌──────────────────────────────────────────┤
       │                                          │
       ▼                                          ▼
┌──────────────┐                          ┌──────────────┐
│ Performance  │                          │ Alert +      │
│ Tracking     │                          │ Retrain      │
│ (accuracy,   │                          │ Trigger      │
│  latency)    │                          └──────────────┘
└──────────────┘
```

### Performance Tracking

Monitor these metrics continuously:

| Category | Metrics | Alert Threshold |
|----------|---------|----------------|
| **Model quality** | Accuracy, F1, AUC (when labels available) | > 5% degradation from baseline |
| **Prediction distribution** | Mean, std, percentiles of predictions | Significant shift from training distribution |
| **Latency** | p50, p95, p99 inference latency | p99 > SLO threshold |
| **Throughput** | Requests per second, GPU utilization | Below expected capacity |
| **Errors** | Error rate, timeout rate | > 1% error rate |

### Feedback Loops and Retraining

```
Initial Model → Predictions → User Actions → Labels
                                                │
                                    ┌───────────▼───────────┐
                                    │ Has enough new labels? │
                                    │ Drift detected?        │
                                    │ Scheduled retrain?     │
                                    └───────────┬───────────┘
                                                │ Yes
                                                ▼
                                         Retrain Pipeline
                                                │
                                                ▼
                                    Evaluate → Gate → Deploy
```

**Retraining triggers:**
- **Scheduled** — Fixed cadence (daily, weekly)
- **Performance-based** — When monitored metrics drop below threshold
- **Drift-based** — When data drift exceeds threshold (PSI > 0.2)
- **Volume-based** — When enough new labeled data accumulates

## A/B Testing and Canary Deployments

### Deployment Strategies for Models

| Strategy | How It Works | Risk | Use Case |
|----------|-------------|------|----------|
| **Shadow mode** | New model runs alongside, predictions logged but not served | Zero risk | Initial validation |
| **Canary** | Route small % of traffic to new model | Low risk | Gradual rollout |
| **A/B test** | Split traffic between models, measure metrics | Medium risk | Comparing model versions |
| **Blue/green** | Two full environments, switch traffic atomically | Low risk (easy rollback) | Stable releases |
| **Multi-armed bandit** | Dynamically adjust traffic based on performance | Low risk | Continuous optimization |

### A/B Testing for Models

```
                    ┌─── 90% ──→ Model A (control)   ──→ Measure: CTR, latency
User Traffic ──────┤
                    └─── 10% ──→ Model B (experiment) ──→ Measure: CTR, latency
                                                              │
                                                    Statistical significance test
                                                              │
                                                    If B > A with p < 0.05 → ship B
```

Key considerations:
- **Metric selection:** Primary metric (e.g., click-through rate) + guardrail metrics (latency, error rate)
- **Sample size:** Calculate required sample size before starting (power analysis)
- **Duration:** Run long enough to capture weekly/seasonal patterns
- **Randomization:** Hash user ID to ensure consistent experience per user

## Model Versioning and Registry

### Model Registry

Centralized store for model artifacts, metadata, and lifecycle state.

```
Model Registry
├── recommendation-model
│   ├── v1.0 (archived)    - metrics: AUC=0.82, latency=12ms
│   ├── v2.0 (production)  - metrics: AUC=0.87, latency=15ms
│   └── v3.0 (staging)     - metrics: AUC=0.89, latency=14ms
└── fraud-detector
    ├── v1.0 (production)
    └── v2.0 (staging)
```

| Tool | Type | Key Features |
|------|------|-------------|
| **MLflow Model Registry** | Open source | Stage transitions, annotations, CI/CD integration |
| **Vertex AI Model Registry** | Google managed | Integrated with Vertex endpoints, auto-versioning |
| **Weights & Biases** | SaaS | Experiment tracking + model registry + visualization |
| **DVC** | Open source | Git-based model versioning, lightweight |

## Containerization for ML

### Docker for ML Workloads

```dockerfile
# Multi-stage build for training image
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04 AS builder
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04
COPY --from=builder /usr/local/lib/python3/dist-packages /usr/local/lib/python3/dist-packages
COPY train.py /app/
WORKDIR /app
CMD ["python", "train.py"]
```

```dockerfile
# Lean serving image
FROM python:3.11-slim
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY model/ /app/model/
COPY serve.py /app/
EXPOSE 8080
CMD ["python", "/app/serve.py"]
```

Best practices:
- Separate training and serving images (training has dev tools, serving is lean)
- Pin all dependency versions for reproducibility
- Use multi-stage builds to reduce image size
- NVIDIA base images for GPU support (`nvidia/cuda:*`)

### Kubernetes for ML

```yaml
# GPU-enabled training job
apiVersion: batch/v1
kind: Job
metadata:
  name: model-training
spec:
  template:
    spec:
      containers:
      - name: trainer
        image: gcr.io/my-project/trainer:v2
        resources:
          limits:
            nvidia.com/gpu: 2           # Request 2 GPUs
            memory: "64Gi"
          requests:
            nvidia.com/gpu: 2
            memory: "32Gi"
      nodeSelector:
        cloud.google.com/gke-accelerator: nvidia-tesla-a100
      restartPolicy: Never
```

GPU scheduling considerations:
- GPUs are non-sharable by default in k8s (one pod per GPU)
- Use NVIDIA GPU Operator or device plugins for GPU discovery
- Node selectors or taints/tolerations to target GPU nodes
- Consider time-slicing or MIG (Multi-Instance GPU) for sharing A100s

## Microservices Architecture for ML

### Service Decomposition

```
┌──────────────────────────────────────────────────────────┐
│                     API Gateway                           │
│              (auth, rate limit, routing)                  │
└───────┬──────────┬──────────┬──────────┬────────────────┘
        │          │          │          │
        ▼          ▼          ▼          ▼
   ┌────────┐ ┌────────┐ ┌────────┐ ┌──────────┐
   │Feature │ │Inference│ │ Post-  │ │Monitoring│
   │Service │ │Service  │ │Process │ │Service   │
   │        │ │(GPU)    │ │        │ │          │
   └───┬────┘ └───┬────┘ └───┬────┘ └──────────┘
       │          │          │
       ▼          ▼          ▼
   Feature     Model      Results
   Store       Store      Store
```

### Service Mesh

Service mesh (Istio, Linkerd) provides:
- **Traffic management** — A/B routing, canary deployments, circuit breaking
- **Observability** — Distributed tracing, metrics, access logs
- **Security** — mTLS between services, RBAC policies
- **Resilience** — Retries, timeouts, rate limiting

### Async Processing

```
Sync (real-time):
Client → API → Inference → Response
                           (wait for GPU)

Async (queue-based):
Client → API → Pub/Sub/Kafka → Worker (GPU) → Result Store
         │                                          │
         └──── Return job ID ◄──── Poll/webhook ────┘
```

**Use async when:**
- Inference takes > 1 second (large models, batch processing)
- Traffic is bursty and you want to smooth GPU load
- Client can tolerate delayed results
- Processing multiple items in batch is more efficient

Message queue options:
| Queue | Best For |
|-------|----------|
| **Cloud Pub/Sub** | GCP native, serverless, at-least-once |
| **Kafka** | High throughput, ordering guarantees, replay |
| **Cloud Tasks** | Task-level control, rate limiting, scheduling |
| **Celery + Redis** | Python-native, simple setup |

## ML Case Study Framework

A structured approach for answering ML system design questions in interviews.

### The 7-Step Framework

#### Step 1: Clarify the Problem

- What is the business objective?
- Who are the users? What are their needs?
- What does success look like from a business perspective?
- What are the constraints (latency, cost, privacy, fairness)?

#### Step 2: Define Metrics

| Metric Type | Examples | Purpose |
|-------------|----------|---------|
| **Offline metrics** | Precision, recall, F1, AUC, NDCG | Evaluate model quality before deployment |
| **Online metrics** | CTR, conversion rate, engagement time | Measure real-world impact |
| **Guardrail metrics** | Latency, error rate, fairness metrics | Ensure no degradation in key areas |

**Key insight:** Offline metrics are necessary but not sufficient. A model can have great AUC but hurt user experience.

#### Step 3: Data Strategy

- What data sources are available?
- How are labels obtained? (human annotation, implicit signals, weak supervision)
- Data quality and bias considerations
- Feature engineering approach
- Training/validation/test split strategy

#### Step 4: Model Selection

```
Start simple → Establish baseline → Increase complexity if needed

Heuristic baseline → Logistic regression → Gradient boosting → Neural network → LLM
```

Justify complexity increases with clear evidence that simpler models are insufficient.

Trade-offs to discuss:
- Accuracy vs latency vs cost
- Interpretability vs performance
- Development time vs model quality

#### Step 5: Training Approach

- Online vs batch training
- Distributed training needs (data size, model size)
- Hyperparameter tuning strategy
- Regularization approach
- Evaluation: cross-validation, hold-out set, temporal split (for time-series)

#### Step 6: Serving & Deployment

- Batch vs real-time vs near-real-time inference
- Model optimization (quantization, pruning, distillation)
- Rollout strategy: shadow mode → canary → A/B test → full deployment
- Autoscaling and infrastructure requirements
- Fallback strategy if model fails

#### Step 7: Monitoring & Iteration

- Data drift and concept drift detection
- Performance tracking with feedback loops
- Retraining triggers and cadence
- A/B testing framework for model iterations
- Incident response plan

### Example: Design a Content Recommendation System

```
1. Clarify: Recommend articles to users on a news platform.
   Goal: increase user engagement. Constraint: < 100ms latency.

2. Metrics:
   - Offline: precision@K, NDCG, recall@K
   - Online: CTR, time-on-page, return rate
   - Guardrails: diversity, freshness, latency p99

3. Data: User click history, article content, user profiles.
   Labels: implicit (clicks = positive, skips = negative).
   Cold-start: use content-based for new users/articles.

4. Model:
   - Baseline: popularity-based ranking
   - Candidate generation: two-tower model (user embedding + item embedding)
   - Ranking: gradient-boosted trees or neural ranker on top-K candidates

5. Training: Daily batch retraining on latest interaction data.
   Features from feature store (user history, article features).

6. Serving: Two-stage - candidate retrieval (ANN search, ~10ms)
   + ranking (~20ms). Cache popular recommendations.
   Canary deploy with 5% traffic.

7. Monitoring: Track CTR daily, monitor feature drift,
   retrain if CTR drops > 5%. A/B test new models.
```
