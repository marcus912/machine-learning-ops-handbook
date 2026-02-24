# Chapter 13: Data Pipelines & Feature Engineering

This chapter covers the data side of ML systems — ingestion pipelines, quality validation, versioning, feature stores, and strategies for handling skewed and imbalanced datasets.

## Data Quality Validation

Production ML systems fail silently when data quality degrades. Validate data at every stage of the pipeline.

### Validation Layers

```
Raw Data → Schema Validation → Distribution Checks → Anomaly Detection → Feature Pipeline
              │                      │                      │
              │                      │                      └─ Alert on unexpected
              │                      │                         nulls, outliers,
              │                      │                         volume changes
              │                      └─ Compare against
              │                         training baselines
              └─ Column types,
                 required fields,
                 value ranges
```

### Schema Checks

Validate that incoming data matches expected structure before processing.

```python
# Example: Great Expectations schema validation (GX 1.x)
import great_expectations as gx

suite = gx.ExpectationSuite(name="data_quality")
suite.add_expectation(gx.expectations.ExpectColumnToExist(column="user_id"))
suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="user_id"))
suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(
    column="age", min_value=0, max_value=150
))
suite.add_expectation(gx.expectations.ExpectColumnValuesToBeInSet(
    column="country", value_set=["US", "UK", "CA", "DE", "FR"]
))
```

Key checks:
- Column types and names match expected schema
- Required fields are non-null
- Value ranges are within expected bounds
- Cardinality of categorical features is reasonable
- No unexpected new categories

### Distribution Checks

Detect when incoming data differs from the training data distribution.

| Metric | What It Measures | Threshold Guidance |
|--------|-----------------|-------------------|
| **PSI (Population Stability Index)** | Overall distribution shift | < 0.1 stable, 0.1-0.2 moderate, > 0.2 significant |
| **KS test (Kolmogorov-Smirnov)** | Max difference between CDFs | p-value < 0.05 indicates significant shift |
| **Jensen-Shannon divergence** | Symmetric divergence between distributions | > 0.1 worth investigating |
| **Chi-squared test** | Categorical distribution differences | p-value < 0.05 indicates significant shift |

```python
from scipy import stats

# KS test: compare training vs production distributions
ks_stat, p_value = stats.ks_2samp(train_feature, prod_feature)
if p_value < 0.05:
    alert(f"Distribution shift detected: KS={ks_stat:.3f}, p={p_value:.4f}")
```

### Anomaly Detection in Data Pipelines

Monitor for sudden changes that indicate upstream issues:
- **Volume anomalies** — row count drops/spikes beyond normal variance
- **Null rate spikes** — sudden increase in missing values
- **Outlier frequency** — more values outside expected bounds
- **Timeliness** — data arriving late or not at all

### Data Lineage

Track the full transformation path from source to feature.

```
Source Table → ETL Job (v2.1) → Feature Table → Training Dataset (exp-042)
                                                        │
                                                  Model v3.2
```

Benefits:
- Debug model issues by tracing back to data source
- Audit trail for regulatory compliance
- Impact analysis when upstream data changes
- Reproducibility of experiments

Tools: Apache Atlas, Google Data Catalog, Amundsen, dbt (for transformation lineage)

## Data Versioning

### Tools Comparison

| Tool | Approach | Best For |
|------|----------|----------|
| **DVC** | Git-like CLI, tracks large files via remote storage (S3, GCS) | ML teams using Git, dataset/model versioning |
| **Delta Lake** | ACID transactions on data lakes, time travel | Data engineering teams, Spark ecosystem |
| **LakeFS** | Git-like branching for data lakes | Data lake versioning at scale |
| **ML Metadata (MLMD)** | Track artifacts, executions, and contexts | Experiment tracking, pipeline metadata |

### DVC Workflow

```bash
# Initialize DVC in a Git repo
dvc init

# Track a large dataset
dvc add data/training_set.parquet
git add data/training_set.parquet.dvc data/.gitignore
git commit -m "Add training dataset v1"

# Push data to remote storage
dvc remote add -d myremote gs://my-bucket/dvc-store
dvc push

# Switch to a different dataset version
git checkout v2-dataset
dvc checkout
```

Key principles:
- Git tracks `.dvc` files (small pointers), DVC tracks actual data in remote storage
- Every Git commit corresponds to a specific dataset version
- Enables reproducible experiments by tying code + data versions together

### Delta Lake Time Travel

```sql
-- Query data as it existed at a specific version
SELECT * FROM my_table VERSION AS OF 42;

-- Query data as of a timestamp
SELECT * FROM my_table TIMESTAMP AS OF '2025-01-15';

-- Restore a table to a previous version
RESTORE TABLE my_table TO VERSION AS OF 42;
```

## Streaming vs Batch Ingestion

### Architecture Comparison

```
Batch Pipeline:
Source → Extract → Load to Storage → Transform → Feature Table
         (scheduled: hourly/daily)

Streaming Pipeline:
Source → Message Queue → Stream Processor → Feature Table
         (Pub/Sub/Kafka)  (Dataflow/Flink)   (continuous)
```

### Detailed Comparison

| Aspect | Batch | Streaming |
|--------|-------|-----------|
| **Latency** | Minutes to hours | Milliseconds to seconds |
| **Complexity** | Lower (simpler error handling, replay) | Higher (ordering, exactly-once, backpressure) |
| **Cost** | Lower (compute used periodically) | Higher (always-on infrastructure) |
| **Data consistency** | Full dataset available for joins/aggregations | Windowed aggregations, eventual consistency |
| **Error recovery** | Re-run failed job | Checkpointing, dead-letter queues |
| **Tools** | Spark, BigQuery, Airflow, dbt | Kafka, Pub/Sub, Dataflow, Flink, Spark Streaming |

### When to Use Each

**Batch:** Periodic model retraining, daily feature computation, historical aggregations, reporting.

**Streaming:** Real-time features (fraud detection, recommendations), online feature updates, event-driven ML pipelines.

**Lambda architecture:** Run both — batch for accurate historical features, streaming for real-time features. More complex but covers both needs.

**Kappa architecture:** Streaming-only — treat everything as a stream, replay from message queue for reprocessing. Simpler but requires robust streaming infrastructure.

## Handling Data Skew & Imbalanced Datasets

### Techniques

| Technique | Category | How It Works | Considerations |
|-----------|----------|-------------|----------------|
| **SMOTE** | Oversampling | Generate synthetic minority samples by interpolating between neighbors | Can create noisy samples near class boundaries |
| **Random oversampling** | Oversampling | Duplicate minority samples | Risk of overfitting to duplicated samples |
| **Random undersampling** | Undersampling | Remove majority samples | Loses potentially useful data |
| **Class weighting** | Loss modification | Increase loss for minority class misclassification | No data manipulation needed; adjust `class_weight` in loss |
| **Focal loss** | Loss modification | Down-weight easy examples, focus on hard ones | Good for extreme imbalance (object detection) |
| **Stratified sampling** | Splitting | Ensure train/val/test preserve class distribution | Always use for imbalanced data splits |

### Class Weighting Example

```python
import torch
import torch.nn as nn

# Inverse frequency weighting
class_counts = [10000, 500, 200]  # samples per class
weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
weights = weights / weights.sum()  # normalize

criterion = nn.CrossEntropyLoss(weight=weights.cuda())
```

### Evaluation for Imbalanced Data

**Don't use accuracy** — 99% accuracy means nothing if 99% of data is one class.

| Metric | What It Captures | When to Use |
|--------|-----------------|-------------|
| **Precision** | Of predicted positives, how many are correct? | When false positives are costly |
| **Recall** | Of actual positives, how many were found? | When false negatives are costly (fraud, disease) |
| **F1 score** | Harmonic mean of precision and recall | General imbalanced classification |
| **PR-AUC** | Area under precision-recall curve | Comprehensive view for imbalanced data |
| **ROC-AUC** | Separability across all thresholds | Can be misleading for extreme imbalance |

**Use precision-recall curves and F1** over accuracy for imbalanced datasets.

## Feature Stores

### Architecture

```
                 ┌──────────────────────────────┐
                 │        Feature Store          │
                 │                               │
Training ◄──────│  Offline Store                │
(batch)         │  (BigQuery, Hive, S3)         │
                 │  - Historical feature values   │
                 │  - Full dataset for training    │
                 │                               │
Serving ◄───────│  Online Store                 │
(real-time)     │  (Redis, Bigtable, DynamoDB)  │
                 │  - Latest feature values       │
                 │  - Low-latency lookups (< 10ms)│
                 │                               │
                 │  Feature Registry             │
                 │  - Feature definitions         │
                 │  - Metadata, ownership, docs   │
                 └──────────────────────────────┘
```

### Training-Serving Skew Prevention

Training-serving skew occurs when features are computed differently at training time vs serving time.

| Skew Type | Cause | Prevention |
|-----------|-------|-----------|
| **Feature computation skew** | Different code paths for training vs serving | Single feature definition, shared computation |
| **Data distribution skew** | Training data doesn't reflect production | Monitor feature distributions, retrain regularly |
| **Time-travel skew** | Using future data during training | Point-in-time feature retrieval |

Feature stores solve this by:
- Single feature definition used for both offline (training) and online (serving)
- Point-in-time correct joins for training data (no data leakage)
- Automated materialization from offline to online store

### Feature Store Tools

| Tool | Type | Key Features |
|------|------|-------------|
| **Feast** | Open source | Python SDK, works with any storage backend, Kubernetes-native |
| **Vertex AI Feature Store** | Google managed | Integrated with Vertex AI, BigQuery offline store, Bigtable online store |
| **Tecton** | Managed SaaS | Real-time feature computation, streaming support |
| **Amazon SageMaker Feature Store** | AWS managed | Integrated with SageMaker, S3 offline, DynamoDB online |

### Feast Example

```python
from feast import FeatureStore

store = FeatureStore(repo_path="feature_repo/")

# Get training data (offline)
training_df = store.get_historical_features(
    entity_df=entity_df,  # entities + timestamps
    features=["user_features:age", "user_features:total_purchases"],
).to_df()

# Get serving features (online)
feature_vector = store.get_online_features(
    features=["user_features:age", "user_features:total_purchases"],
    entity_rows=[{"user_id": 12345}],
).to_dict()
```
