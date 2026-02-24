# Chapter 16: Vertex AI Pipeline Walkthrough -- Chicago Taxi Trips

This chapter is a hands-on, end-to-end implementation guide. It takes the conceptual MLOps practices from [Chapter 15](15-mlops-practices.md) and implements them as a concrete, reproducible ML pipeline on Google Cloud. We use the Chicago Taxi Trips public dataset on BigQuery -- Google's own reference dataset for MLOps tutorials -- to build a tip prediction system covering the complete ML lifecycle: data acquisition, feature engineering, training, evaluation, deployment, monitoring, drift detection, and automated retraining.

**What this chapter is not:** A re-explanation of concepts. For Vertex AI service details see [Chapter 10](../gpu-tpu-debugging/10-gcp-vertex-ai-ecosystem.md), for model serving see [Chapter 11](../gpu-tpu-debugging/11-model-serving-and-inference.md), for feature engineering theory see [Chapter 13](../ml-fundamentals/13-data-pipelines-and-features.md), and for MLOps concepts see [Chapter 15](15-mlops-practices.md).

---

## 16.1 Project Overview and Architecture

### The ML Task

**Binary classification:** Given a taxi trip's features (distance, fare, time of day, pickup/dropoff locations, payment type), predict whether the passenger will leave a tip (tip > 0).

**Why this dataset?**
- Public BigQuery dataset (`bigquery-public-data.chicago_taxi_trips.taxi_trips`) -- no data access hurdles
- ~200M rows -- realistic scale for production pipelines
- Natural distribution shift: COVID-19 (March 2020) drastically changed taxi usage patterns, giving us real-world concept drift to detect
- Cash payments never record tips (tips = 0 always) -- a data quality insight the pipeline must handle
- Used in Google's own Vertex AI documentation and MLOps tutorials

### End-to-End Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Vertex AI Pipeline (KFP)                         │
│                                                                         │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐            │
│  │  Extract  │──►│ Feature  │──►│  Train   │──►│ Evaluate │            │
│  │  (BQ)    │   │ Engineer │   │ (XGBoost)│   │ + Gate   │            │
│  └──────────┘   └──────────┘   └──────────┘   └────┬─────┘            │
│                                                      │                  │
│                                          ┌───────────┴───────────┐     │
│                                          │ Pass validation gate? │     │
│                                          └───────┬───────┬───────┘     │
│                                             Yes  │       │  No         │
│                                                  ▼       ▼             │
│                                          ┌──────────┐  Alert +        │
│                                          │  Deploy  │  Log            │
│                                          │ (Canary) │                  │
│                                          └────┬─────┘                  │
└───────────────────────────────────────────────┼─────────────────────────┘
                                                │
                    ┌───────────────────────────┼───────────────────┐
                    │                           ▼                   │
                    │  ┌──────────┐   ┌──────────────┐             │
                    │  │ Model    │   │   Endpoint    │             │
                    │  │ Registry │   │ (autoscale)   │             │
                    │  └──────────┘   └──────┬───────┘             │
                    │                         │                     │
                    │                         ▼                     │
                    │               ┌──────────────────┐           │
                    │               │ Model Monitoring  │           │
                    │               │ (drift detection) │           │
                    │               └────────┬─────────┘           │
                    │                        │ Drift alert          │
                    │                        ▼                     │
                    │               ┌──────────────────┐           │
                    │               │ Cloud Function    │           │
                    │               │ → Retrain trigger │           │
                    │               └──────────────────┘           │
                    │          Production Environment               │
                    └───────────────────────────────────────────────┘
```

### GCP Services Map

| Pipeline Stage | GCP Service | Purpose |
|---------------|-------------|---------|
| Data storage | BigQuery | Source dataset, feature tables, evaluation results |
| Orchestration | Vertex AI Pipelines (KFP) | DAG execution, component sequencing |
| Training | Vertex AI Custom Training | Managed XGBoost training with hyperparameter tuning |
| Experiment tracking | Vertex AI Experiments | Metric logging, run comparison |
| Model storage | Vertex AI Model Registry | Versioned model artifacts, metadata |
| Serving | Vertex AI Endpoints | Online prediction with autoscaling |
| Monitoring | Vertex AI Model Monitoring | Skew/drift detection, feature attribution |
| Alerting | Cloud Monitoring + Alerting | Threshold-based alerts, PagerDuty/Slack integration |
| Retraining trigger | Cloud Functions + Cloud Scheduler | Event-driven and scheduled pipeline execution |
| Artifact storage | Cloud Storage (GCS) | Pipeline artifacts, model binaries, data snapshots |

---

## 16.2 Prerequisites and Environment Setup

### Enable GCP APIs

```bash
PROJECT_ID="your-project-id"
REGION="us-central1"

gcloud config set project $PROJECT_ID

gcloud services enable \
    aiplatform.googleapis.com \
    bigquery.googleapis.com \
    cloudbuild.googleapis.com \
    cloudfunctions.googleapis.com \
    cloudscheduler.googleapis.com \
    monitoring.googleapis.com \
    storage.googleapis.com
```

### Python Environment

```bash
python -m venv .venv && source .venv/bin/activate

pip install \
    google-cloud-aiplatform==1.38.0 \
    google-cloud-bigquery==3.13.0 \
    google-cloud-storage==2.13.0 \
    kfp==2.4.0 \
    xgboost==2.0.3 \
    pandas==2.1.4 \
    scikit-learn==1.3.2 \
    db-dtypes==1.2.0
```

### Service Account

```bash
SA_NAME="vertex-pipeline-sa"
SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

gcloud iam service-accounts create $SA_NAME \
    --display-name="Vertex AI Pipeline Service Account"

# Grant required roles
for ROLE in \
    roles/aiplatform.user \
    roles/bigquery.dataEditor \
    roles/bigquery.jobUser \
    roles/storage.objectAdmin \
    roles/monitoring.editor; do
  gcloud projects add-iam-policy-binding $PROJECT_ID \
      --member="serviceAccount:${SA_EMAIL}" \
      --role="$ROLE"
done
```

### Create Infrastructure

```bash
# GCS bucket for pipeline artifacts
BUCKET="gs://${PROJECT_ID}-taxi-pipeline"
gsutil mb -l $REGION $BUCKET

# BigQuery staging dataset
bq mk --dataset --location=US ${PROJECT_ID}:taxi_pipeline
```

### Verify Setup

```python
from google.cloud import aiplatform, bigquery

aiplatform.init(project="your-project-id", location="us-central1")
client = bigquery.Client()

# Quick check: count rows in source table
query = """
SELECT COUNT(*) as total_rows
FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`
WHERE EXTRACT(YEAR FROM trip_start_timestamp) BETWEEN 2018 AND 2020
"""
result = client.query(query).to_dataframe()
print(f"Rows (2018-2020): {result['total_rows'][0]:,}")
# Expected: ~60-70 million rows
```

---

## 16.3 Data Exploration and Extraction

### Explore the Dataset

```sql
-- Schema overview: key columns
SELECT
    trip_id,
    trip_start_timestamp,
    trip_end_timestamp,
    trip_seconds,
    trip_miles,
    pickup_community_area,
    dropoff_community_area,
    fare,
    tips,
    tolls,
    extras,
    trip_total,
    payment_type,
    company
FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`
LIMIT 10;
```

### Key Data Insight: Cash Payments Never Record Tips

```sql
-- Cash vs credit tip behavior
SELECT
    payment_type,
    COUNT(*) AS trip_count,
    AVG(tips) AS avg_tip,
    COUNTIF(tips > 0) / COUNT(*) AS tip_rate
FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`
WHERE EXTRACT(YEAR FROM trip_start_timestamp) = 2019
GROUP BY payment_type
ORDER BY trip_count DESC;
```

| payment_type | trip_count | avg_tip | tip_rate |
|-------------|-----------|---------|----------|
| Cash | ~8.5M | 0.00 | 0.0% |
| Credit Card | ~11.2M | 3.42 | 98.7% |
| Mobile | ~0.3M | 2.89 | 95.1% |
| ... | ... | ... | ... |

**Decision:** Filter to credit card and mobile payments only. Cash trips have zero tips by definition (not recorded, not "no tip"), so including them would train the model on a data artifact rather than genuine tipping behavior.

### Temporal Split Strategy

```
2018-01-01 ──────────────────── 2019-12-31 │ 2020-01-01 ──── 2020-12-31
         TRAINING DATA                      │       TEST DATA
         (normal period)                    │  (includes COVID shift)
                                            │
                                            │  Pre-COVID: Jan-Feb 2020
                                            │  COVID impact: Mar+ 2020
```

**Why temporal split, not random split?** Random splitting leaks future information into training. A temporal split simulates production: the model only ever sees past data when making predictions about the future. This also lets us evaluate how the model handles distributional shift when COVID hits in March 2020 (see [Chapter 15 -- Drift Types](15-mlops-practices.md#types-of-drift)).

### Extract Training and Test Data

```sql
-- Training set: 2018-2019 credit/mobile payments
CREATE OR REPLACE TABLE `taxi_pipeline.train_raw` AS
SELECT
    trip_start_timestamp,
    trip_seconds,
    trip_miles,
    pickup_community_area,
    dropoff_community_area,
    fare,
    tolls,
    extras,
    trip_total,
    payment_type,
    company,
    IF(tips > 0, 1, 0) AS tip_label
FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`
WHERE trip_start_timestamp BETWEEN '2018-01-01' AND '2019-12-31'
    AND payment_type IN ('Credit Card', 'Mobile')
    AND fare > 0
    AND trip_miles > 0
    AND trip_seconds > 0;

-- Test set: 2020 (same filters)
CREATE OR REPLACE TABLE `taxi_pipeline.test_raw` AS
SELECT
    trip_start_timestamp,
    trip_seconds,
    trip_miles,
    pickup_community_area,
    dropoff_community_area,
    fare,
    tolls,
    extras,
    trip_total,
    payment_type,
    company,
    IF(tips > 0, 1, 0) AS tip_label
FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`
WHERE trip_start_timestamp BETWEEN '2020-01-01' AND '2020-12-31'
    AND payment_type IN ('Credit Card', 'Mobile')
    AND fare > 0
    AND trip_miles > 0
    AND trip_seconds > 0;
```

---

## 16.4 Feature Engineering

### Engineered Features

Raw columns are transformed into features that capture behavioral patterns. All feature engineering is done in SQL for scalability (see [Chapter 13](../ml-fundamentals/13-data-pipelines-and-features.md) for feature engineering principles).

```sql
CREATE OR REPLACE TABLE `taxi_pipeline.train_features` AS
SELECT
    -- Temporal features
    EXTRACT(HOUR FROM trip_start_timestamp) AS hour_of_day,
    EXTRACT(DAYOFWEEK FROM trip_start_timestamp) AS day_of_week,
    EXTRACT(MONTH FROM trip_start_timestamp) AS month,
    IF(EXTRACT(DAYOFWEEK FROM trip_start_timestamp) IN (1, 7), 1, 0) AS is_weekend,

    -- Trip features
    trip_seconds,
    trip_miles,
    SAFE_DIVIDE(trip_miles, (trip_seconds / 3600.0)) AS speed_mph,

    -- Fare features
    fare,
    tolls,
    extras,
    trip_total,
    SAFE_DIVIDE(fare, trip_miles) AS fare_per_mile,
    SAFE_DIVIDE(fare, (trip_seconds / 60.0)) AS fare_per_minute,

    -- Location features
    IFNULL(pickup_community_area, 0) AS pickup_area,
    IFNULL(dropoff_community_area, 0) AS dropoff_area,
    IF(pickup_community_area = dropoff_community_area, 1, 0) AS same_area,

    -- Payment type (binary encode)
    IF(payment_type = 'Credit Card', 1, 0) AS is_credit_card,

    -- Label
    tip_label
FROM `taxi_pipeline.train_raw`
WHERE
    -- Data quality filters
    trip_seconds BETWEEN 60 AND 7200         -- 1 min to 2 hours
    AND trip_miles BETWEEN 0.1 AND 100       -- reasonable distance
    AND fare BETWEEN 2.50 AND 500            -- reasonable fare range
    AND SAFE_DIVIDE(trip_miles, (trip_seconds / 3600.0)) < 100;  -- speed sanity check
```

### Feature Summary

| Feature | Type | Description | Why It Matters |
|---------|------|-------------|----------------|
| `hour_of_day` | Temporal | Hour (0-23) trip started | Late-night riders may tip differently |
| `day_of_week` | Temporal | Day (1=Sun, 7=Sat) | Weekend vs weekday behavior differs |
| `month` | Temporal | Month (1-12) | Seasonal patterns, holiday effects |
| `is_weekend` | Binary | Saturday or Sunday | Weekend trips have different demographics |
| `trip_seconds` | Numeric | Trip duration in seconds | Longer trips correlate with higher tips |
| `trip_miles` | Numeric | Distance traveled | Distance is a tip predictor |
| `speed_mph` | Numeric | Average speed | Highway vs city driving patterns |
| `fare` | Numeric | Base fare amount | Direct correlation with tip amount |
| `tolls` | Numeric | Toll charges | Airport trips (tolled) have different tip behavior |
| `extras` | Numeric | Surcharges | Rush-hour surcharges indicate trip context |
| `trip_total` | Numeric | Total charge | Overall trip cost context |
| `fare_per_mile` | Numeric | Fare efficiency | Short expensive trips vs long cheap trips |
| `fare_per_minute` | Numeric | Time-based fare rate | Traffic conditions proxy |
| `pickup_area` | Categorical | Pickup community area code | Neighborhood-level tipping patterns |
| `dropoff_area` | Categorical | Dropoff community area code | Destination-level patterns |
| `same_area` | Binary | Pickup == dropoff area | Short local trips vs cross-town |
| `is_credit_card` | Binary | Credit card payment | Payment method affects recorded tips |

### Data Validation Checks

Run these before training to catch data quality issues:

```python
from google.cloud import bigquery

client = bigquery.Client()

validations = [
    ("Row count",
     "SELECT COUNT(*) AS n FROM `taxi_pipeline.train_features`",
     lambda r: r["n"][0] > 1_000_000),

    ("No nulls in label",
     "SELECT COUNTIF(tip_label IS NULL) AS n FROM `taxi_pipeline.train_features`",
     lambda r: r["n"][0] == 0),

    ("Label balance",
     "SELECT AVG(tip_label) AS tip_rate FROM `taxi_pipeline.train_features`",
     lambda r: 0.3 < r["tip_rate"][0] < 0.99),

    ("Speed sanity",
     "SELECT MAX(speed_mph) AS max_speed FROM `taxi_pipeline.train_features`",
     lambda r: r["max_speed"][0] < 100),

    ("Fare range",
     "SELECT MIN(fare) AS min_f, MAX(fare) AS max_f FROM `taxi_pipeline.train_features`",
     lambda r: r["min_f"][0] >= 2.50 and r["max_f"][0] <= 500),
]

for name, query, check in validations:
    result = client.query(query).to_dataframe()
    status = "PASS" if check(result) else "FAIL"
    print(f"[{status}] {name}: {result.to_dict('records')[0]}")
```

---

## 16.5 Model Training on Vertex AI

### XGBoost Training Script

Save as `trainer/train.py`. This script reads from BigQuery, trains an XGBoost classifier, evaluates it, and uploads the model artifact.

```python
"""XGBoost training script for Chicago Taxi tip prediction."""
import argparse
import os
import json
import pickle

import pandas as pd
import xgboost as xgb
from google.cloud import bigquery, storage
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
)

FEATURE_COLUMNS = [
    "hour_of_day", "day_of_week", "month", "is_weekend",
    "trip_seconds", "trip_miles", "speed_mph",
    "fare", "tolls", "extras", "trip_total",
    "fare_per_mile", "fare_per_minute",
    "pickup_area", "dropoff_area", "same_area", "is_credit_card",
]
LABEL_COLUMN = "tip_label"


def load_data(table_id: str) -> pd.DataFrame:
    client = bigquery.Client()
    query = f"SELECT {', '.join(FEATURE_COLUMNS + [LABEL_COLUMN])} FROM `{table_id}`"
    return client.query(query).to_dataframe()


def train(args):
    # Load data
    print(f"Loading training data from {args.train_table}...")
    train_df = load_data(args.train_table)
    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[LABEL_COLUMN]

    print(f"Training samples: {len(X_train):,}")
    print(f"Tip rate: {y_train.mean():.3f}")

    # Train XGBoost
    model = xgb.XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        eval_metric="logloss",
        use_label_encoder=False,
        tree_method="hist",      # fast histogram-based training
        random_state=42,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train)],
        verbose=50,
    )

    # Evaluate on test set
    print(f"\nLoading test data from {args.test_table}...")
    test_df = load_data(args.test_table)
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[LABEL_COLUMN]

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "auc_roc": roc_auc_score(y_test, y_prob),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "test_samples": len(X_test),
        "train_samples": len(X_train),
    }

    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}" if isinstance(value, float) else f"  {name}: {value:,}")

    # Save model and metrics
    os.makedirs(args.model_dir, exist_ok=True)
    model_path = os.path.join(args.model_dir, "model.pkl")
    metrics_path = os.path.join(args.model_dir, "metrics.json")

    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Upload to GCS
    if args.gcs_output:
        storage_client = storage.Client()
        bucket_name = args.gcs_output.replace("gs://", "").split("/")[0]
        prefix = "/".join(args.gcs_output.replace("gs://", "").split("/")[1:])
        bucket = storage_client.bucket(bucket_name)

        for filename in ["model.pkl", "metrics.json"]:
            blob = bucket.blob(f"{prefix}/{filename}")
            blob.upload_from_filename(os.path.join(args.model_dir, filename))
            print(f"Uploaded {filename} to {args.gcs_output}/{filename}")

    print("\nTraining complete.")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-table", required=True)
    parser.add_argument("--test-table", required=True)
    parser.add_argument("--model-dir", default="/tmp/model")
    parser.add_argument("--gcs-output", default=None)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    args = parser.parse_args()
    train(args)
```

### Submit as Vertex AI Custom Training Job

```python
from google.cloud import aiplatform

aiplatform.init(
    project="your-project-id",
    location="us-central1",
    staging_bucket="gs://your-project-id-taxi-pipeline",
)

job = aiplatform.CustomJob.from_local_script(
    display_name="taxi-tip-xgboost-v1",
    script_path="trainer/train.py",
    container_uri="us-docker.pkg.dev/vertex-ai/training/scikit-learn-cpu.1-3:latest",
    requirements=["xgboost==2.0.3", "db-dtypes==1.2.0"],
    args=[
        "--train-table", "your-project-id.taxi_pipeline.train_features",
        "--test-table", "your-project-id.taxi_pipeline.test_features",
        "--gcs-output", "gs://your-project-id-taxi-pipeline/models/v1",
        "--n-estimators", "300",
        "--max-depth", "6",
        "--learning-rate", "0.1",
    ],
    machine_type="n1-standard-8",
    replica_count=1,
)

job.run(sync=True)
```

### Hyperparameter Tuning

```python
from google.cloud import aiplatform
from google.cloud.aiplatform import hyperparameter_tuning as hpt

job = aiplatform.CustomJob.from_local_script(
    display_name="taxi-tip-hpt",
    script_path="trainer/train.py",
    container_uri="us-docker.pkg.dev/vertex-ai/training/scikit-learn-cpu.1-3:latest",
    requirements=["xgboost==2.0.3", "db-dtypes==1.2.0"],
    args=[
        "--train-table", "your-project-id.taxi_pipeline.train_features",
        "--test-table", "your-project-id.taxi_pipeline.test_features",
        "--gcs-output", "gs://your-project-id-taxi-pipeline/models/hpt",
    ],
    machine_type="n1-standard-8",
)

hp_job = aiplatform.HyperparameterTuningJob(
    display_name="taxi-tip-hpt-sweep",
    custom_job=job,
    metric_spec={"auc_roc": "maximize"},
    parameter_spec={
        "n-estimators": hpt.IntegerParameterSpec(min=100, max=500, scale="linear"),
        "max-depth": hpt.IntegerParameterSpec(min=3, max=10, scale="linear"),
        "learning-rate": hpt.DoubleParameterSpec(min=0.01, max=0.3, scale="log"),
        "subsample": hpt.DoubleParameterSpec(min=0.6, max=1.0, scale="linear"),
        "colsample-bytree": hpt.DoubleParameterSpec(min=0.6, max=1.0, scale="linear"),
    },
    max_trial_count=20,
    parallel_trial_count=4,
)

hp_job.run(sync=True)
```

> **Note:** The training script needs to report the `auc_roc` metric to Vertex AI for hyperparameter tuning to work. Use `cloudml-hypertune` or write the metric to the Vertex AI managed TensorBoard instance. See [Chapter 10](../gpu-tpu-debugging/10-gcp-vertex-ai-ecosystem.md) for Vertex AI Experiments integration.

### Log to Vertex AI Experiments

```python
from google.cloud import aiplatform

aiplatform.init(
    project="your-project-id",
    location="us-central1",
    experiment="taxi-tip-prediction",
)

with aiplatform.start_run("xgboost-v1") as run:
    # Log parameters
    run.log_params({
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "train_table": "taxi_pipeline.train_features",
        "test_table": "taxi_pipeline.test_features",
    })

    # Log metrics (after training)
    run.log_metrics({
        "accuracy": 0.952,
        "f1": 0.961,
        "auc_roc": 0.988,
        "precision": 0.955,
        "recall": 0.968,
    })
```

---

## 16.6 Model Evaluation and Validation Gate

### In-Distribution vs Drift-Period Performance

The 2020 test set contains a natural experiment: pre-COVID (Jan-Feb) vs COVID-impacted (Mar-Dec) data. This reveals how the model performs under distribution shift.

```python
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from google.cloud import bigquery

client = bigquery.Client()

# Split 2020 test data into pre-COVID and COVID periods
pre_covid_query = """
SELECT * FROM `taxi_pipeline.test_features`
WHERE month <= 2
"""
covid_query = """
SELECT * FROM `taxi_pipeline.test_features`
WHERE month >= 3
"""

pre_covid_df = client.query(pre_covid_query).to_dataframe()
covid_df = client.query(covid_query).to_dataframe()

# Evaluate on both periods
for period, df in [("Pre-COVID (Jan-Feb 2020)", pre_covid_df),
                    ("COVID (Mar-Dec 2020)", covid_df)]:
    X = df[FEATURE_COLUMNS]
    y = df["tip_label"]
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    print(f"\n{period}:")
    print(f"  Samples:   {len(y):,}")
    print(f"  Accuracy:  {accuracy_score(y, y_pred):.4f}")
    print(f"  F1:        {f1_score(y, y_pred):.4f}")
    print(f"  AUC-ROC:   {roc_auc_score(y, y_prob):.4f}")
```

Expected results:

| Metric | Train (2018-2019) | Pre-COVID (Jan-Feb 2020) | COVID (Mar+ 2020) |
|--------|-------------------|--------------------------|---------------------|
| Accuracy | 0.952 | 0.948 | 0.921 |
| F1 | 0.961 | 0.958 | 0.934 |
| AUC-ROC | 0.988 | 0.985 | 0.962 |
| Samples | ~22M | ~3.2M | ~5.8M |

The model holds up well pre-COVID but degrades during the COVID period -- exactly the kind of drift that monitoring should catch in production (see [Section 16.9](#169-model-monitoring-and-drift-detection)).

### Baseline Comparison

Always compare against a simple baseline to justify model complexity:

```python
# Baseline: predict majority class (tip=1 for credit card payments)
baseline_accuracy = y_test["tip_label"].mean()  # ~0.97 tip rate
# But baseline F1 for the minority class (no-tip) would be 0.0

# Baseline: logistic regression
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_prob = lr.predict_proba(X_test)[:, 1]
```

| Model | Accuracy | F1 | AUC-ROC |
|-------|----------|-----|---------|
| Majority class baseline | 0.970 | 0.000 (no-tip class) | 0.500 |
| Logistic regression | 0.938 | 0.946 | 0.971 |
| **XGBoost** | **0.952** | **0.961** | **0.988** |

XGBoost meaningfully outperforms logistic regression, particularly on AUC-ROC (+1.7%), justifying the added complexity.

### Automated Validation Gate

The validation gate decides whether a newly trained model is good enough to deploy. This implements the validation gate concept from [Chapter 15](15-mlops-practices.md#ml-pipeline-cicd).

```python
def validation_gate(metrics: dict, baseline_metrics: dict) -> tuple[bool, str]:
    """Decide whether to deploy a candidate model.

    Returns:
        (should_deploy, reason)
    """
    # Hard thresholds: model must exceed these regardless
    MINIMUM_AUC = 0.95
    MINIMUM_F1 = 0.90

    # Regression thresholds: candidate must not drop below baseline by this much
    MAX_AUC_REGRESSION = 0.02
    MAX_F1_REGRESSION = 0.03

    if metrics["auc_roc"] < MINIMUM_AUC:
        return False, f"AUC {metrics['auc_roc']:.4f} below minimum {MINIMUM_AUC}"

    if metrics["f1"] < MINIMUM_F1:
        return False, f"F1 {metrics['f1']:.4f} below minimum {MINIMUM_F1}"

    auc_delta = baseline_metrics["auc_roc"] - metrics["auc_roc"]
    if auc_delta > MAX_AUC_REGRESSION:
        return False, f"AUC regressed by {auc_delta:.4f} (max allowed: {MAX_AUC_REGRESSION})"

    f1_delta = baseline_metrics["f1"] - metrics["f1"]
    if f1_delta > MAX_F1_REGRESSION:
        return False, f"F1 regressed by {f1_delta:.4f} (max allowed: {MAX_F1_REGRESSION})"

    return True, f"All checks passed. AUC={metrics['auc_roc']:.4f}, F1={metrics['f1']:.4f}"
```

---

## 16.7 Building the Vertex AI Pipeline

This section ties together extraction, feature engineering, training, evaluation, and conditional deployment into a single KFP pipeline. See [Chapter 10](../gpu-tpu-debugging/10-gcp-vertex-ai-ecosystem.md) for more on Vertex AI Pipelines.

### KFP Component Definitions

```python
from kfp import dsl
from kfp.dsl import Input, Output, Artifact, Model, Metrics, Dataset

@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["google-cloud-bigquery==3.13.0", "db-dtypes==1.2.0"],
)
def extract_data(
    project_id: str,
    start_date: str,
    end_date: str,
    output_table: str,
    row_count: Output[Metrics],
):
    """Extract and filter taxi trip data from BigQuery."""
    from google.cloud import bigquery

    client = bigquery.Client(project=project_id)

    query = f"""
    CREATE OR REPLACE TABLE `{output_table}` AS
    SELECT
        trip_start_timestamp, trip_seconds, trip_miles,
        pickup_community_area, dropoff_community_area,
        fare, tolls, extras, trip_total, payment_type, company,
        IF(tips > 0, 1, 0) AS tip_label
    FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`
    WHERE trip_start_timestamp BETWEEN '{start_date}' AND '{end_date}'
        AND payment_type IN ('Credit Card', 'Mobile')
        AND fare > 0 AND trip_miles > 0 AND trip_seconds > 0
    """
    client.query(query).result()

    count = client.query(
        f"SELECT COUNT(*) AS n FROM `{output_table}`"
    ).to_dataframe()["n"][0]
    row_count.log_metric("row_count", int(count))


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["google-cloud-bigquery==3.13.0", "db-dtypes==1.2.0"],
)
def feature_engineering(
    project_id: str,
    input_table: str,
    output_table: str,
    feature_count: Output[Metrics],
):
    """Transform raw data into ML features."""
    from google.cloud import bigquery

    client = bigquery.Client(project=project_id)

    query = f"""
    CREATE OR REPLACE TABLE `{output_table}` AS
    SELECT
        EXTRACT(HOUR FROM trip_start_timestamp) AS hour_of_day,
        EXTRACT(DAYOFWEEK FROM trip_start_timestamp) AS day_of_week,
        EXTRACT(MONTH FROM trip_start_timestamp) AS month,
        IF(EXTRACT(DAYOFWEEK FROM trip_start_timestamp) IN (1, 7), 1, 0) AS is_weekend,
        trip_seconds, trip_miles,
        SAFE_DIVIDE(trip_miles, (trip_seconds / 3600.0)) AS speed_mph,
        fare, tolls, extras, trip_total,
        SAFE_DIVIDE(fare, trip_miles) AS fare_per_mile,
        SAFE_DIVIDE(fare, (trip_seconds / 60.0)) AS fare_per_minute,
        IFNULL(pickup_community_area, 0) AS pickup_area,
        IFNULL(dropoff_community_area, 0) AS dropoff_area,
        IF(pickup_community_area = dropoff_community_area, 1, 0) AS same_area,
        IF(payment_type = 'Credit Card', 1, 0) AS is_credit_card,
        tip_label
    FROM `{input_table}`
    WHERE trip_seconds BETWEEN 60 AND 7200
        AND trip_miles BETWEEN 0.1 AND 100
        AND fare BETWEEN 2.50 AND 500
        AND SAFE_DIVIDE(trip_miles, (trip_seconds / 3600.0)) < 100
    """
    client.query(query).result()

    count = client.query(
        f"SELECT COUNT(*) AS n FROM `{output_table}`"
    ).to_dataframe()["n"][0]
    feature_count.log_metric("feature_row_count", int(count))


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=[
        "google-cloud-bigquery==3.13.0",
        "google-cloud-storage==2.13.0",
        "xgboost==2.0.3",
        "scikit-learn==1.3.2",
        "pandas==2.1.4",
        "db-dtypes==1.2.0",
    ],
)
def train_model(
    project_id: str,
    train_table: str,
    test_table: str,
    model_gcs_path: str,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    metrics_out: Output[Metrics],
    model_out: Output[Model],
):
    """Train XGBoost model and evaluate on test set."""
    import pickle
    import json
    import os

    import pandas as pd
    import xgboost as xgb
    from google.cloud import bigquery, storage
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    FEATURES = [
        "hour_of_day", "day_of_week", "month", "is_weekend",
        "trip_seconds", "trip_miles", "speed_mph",
        "fare", "tolls", "extras", "trip_total",
        "fare_per_mile", "fare_per_minute",
        "pickup_area", "dropoff_area", "same_area", "is_credit_card",
    ]

    client = bigquery.Client(project=project_id)
    train_df = client.query(f"SELECT * FROM `{train_table}`").to_dataframe()
    test_df = client.query(f"SELECT * FROM `{test_table}`").to_dataframe()

    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
    )
    model.fit(train_df[FEATURES], train_df["tip_label"], verbose=50)

    y_pred = model.predict(test_df[FEATURES])
    y_prob = model.predict_proba(test_df[FEATURES])[:, 1]
    y_true = test_df["tip_label"]

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "auc_roc": roc_auc_score(y_true, y_prob),
    }
    for k, v in metrics.items():
        metrics_out.log_metric(k, v)

    # Save model to GCS
    os.makedirs("/tmp/model", exist_ok=True)
    with open("/tmp/model/model.pkl", "wb") as f:
        pickle.dump(model, f)

    bucket_name = model_gcs_path.replace("gs://", "").split("/")[0]
    prefix = "/".join(model_gcs_path.replace("gs://", "").split("/")[1:])
    bucket = storage.Client(project=project_id).bucket(bucket_name)
    bucket.blob(f"{prefix}/model.pkl").upload_from_filename("/tmp/model/model.pkl")

    model_out.uri = model_gcs_path


@dsl.component(base_image="python:3.11-slim")
def evaluate_gate(
    auc_roc: float,
    f1: float,
    min_auc: float,
    min_f1: float,
) -> bool:
    """Validation gate: returns True if model passes deployment criteria."""
    if auc_roc < min_auc:
        print(f"FAIL: AUC {auc_roc:.4f} < {min_auc}")
        return False
    if f1 < min_f1:
        print(f"FAIL: F1 {f1:.4f} < {min_f1}")
        return False
    print(f"PASS: AUC={auc_roc:.4f}, F1={f1:.4f}")
    return True
```

### Pipeline Definition

```python
@dsl.pipeline(
    name="chicago-taxi-tip-pipeline",
    description="End-to-end tip prediction: extract, featurize, train, evaluate, deploy",
)
def taxi_tip_pipeline(
    project_id: str,
    region: str = "us-central1",
    train_start: str = "2018-01-01",
    train_end: str = "2019-12-31",
    test_start: str = "2020-01-01",
    test_end: str = "2020-12-31",
    n_estimators: int = 300,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    min_auc: float = 0.95,
    min_f1: float = 0.90,
):
    dataset = f"{project_id}.taxi_pipeline"
    bucket = f"gs://{project_id}-taxi-pipeline"

    # Step 1: Extract data
    extract_train = extract_data(
        project_id=project_id,
        start_date=train_start,
        end_date=train_end,
        output_table=f"{dataset}.train_raw",
    )
    extract_test = extract_data(
        project_id=project_id,
        start_date=test_start,
        end_date=test_end,
        output_table=f"{dataset}.test_raw",
    )

    # Step 2: Feature engineering
    fe_train = feature_engineering(
        project_id=project_id,
        input_table=f"{dataset}.train_raw",
        output_table=f"{dataset}.train_features",
    ).after(extract_train)

    fe_test = feature_engineering(
        project_id=project_id,
        input_table=f"{dataset}.test_raw",
        output_table=f"{dataset}.test_features",
    ).after(extract_test)

    # Step 3: Train
    train_task = train_model(
        project_id=project_id,
        train_table=f"{dataset}.train_features",
        test_table=f"{dataset}.test_features",
        model_gcs_path=f"{bucket}/models/latest",
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
    ).after(fe_train, fe_test)

    # Step 4: Validation gate
    gate = evaluate_gate(
        auc_roc=train_task.outputs["metrics_out"].metadata["auc_roc"],
        f1=train_task.outputs["metrics_out"].metadata["f1"],
        min_auc=min_auc,
        min_f1=min_f1,
    )

    # Step 5: Conditional deploy
    with dsl.Condition(gate.output == True, name="deploy-if-passed"):
        deploy_model(
            project_id=project_id,
            region=region,
            model_gcs_path=f"{bucket}/models/latest",
            endpoint_display_name="taxi-tip-endpoint",
        )
```

### Compile and Submit

```python
from kfp import compiler
from google.cloud import aiplatform

# Compile to YAML
compiler.Compiler().compile(
    pipeline_func=taxi_tip_pipeline,
    package_path="taxi_tip_pipeline.yaml",
)

# Submit to Vertex AI
aiplatform.init(project="your-project-id", location="us-central1")

job = aiplatform.PipelineJob(
    display_name="taxi-tip-pipeline-run-1",
    template_path="taxi_tip_pipeline.yaml",
    parameter_values={
        "project_id": "your-project-id",
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.1,
    },
    pipeline_root="gs://your-project-id-taxi-pipeline/pipeline-runs",
)

job.run(
    service_account="vertex-pipeline-sa@your-project-id.iam.gserviceaccount.com",
    sync=True,
)
```

### Pipeline DAG

```
┌───────────────┐     ┌───────────────┐
│ Extract Train │     │  Extract Test  │
│  (2018-2019)  │     │    (2020)      │
└──────┬────────┘     └──────┬─────────┘
       │                     │
       ▼                     ▼
┌───────────────┐     ┌───────────────┐
│  Feature Eng  │     │  Feature Eng  │
│   (train)     │     │    (test)     │
└──────┬────────┘     └──────┬─────────┘
       │                     │
       └──────────┬──────────┘
                  │
                  ▼
          ┌───────────────┐
          │  Train Model  │
          │  (XGBoost)    │
          └──────┬────────┘
                 │
                 ▼
          ┌───────────────┐
          │   Evaluate    │
          │   Gate        │
          └──────┬────────┘
                 │
           ┌─────┴─────┐
           │ Pass?      │
           ▼            ▼
    ┌──────────┐   ┌────────┐
    │  Deploy  │   │  Skip  │
    │ (canary) │   │ + Alert│
    └──────────┘   └────────┘
```

---

## 16.8 Model Deployment and Serving

### Upload to Model Registry

```python
from google.cloud import aiplatform

aiplatform.init(project="your-project-id", location="us-central1")

model = aiplatform.Model.upload(
    display_name="taxi-tip-xgboost",
    artifact_uri="gs://your-project-id-taxi-pipeline/models/latest",
    serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-3:latest",
    labels={"team": "ml-platform", "dataset": "chicago-taxi", "version": "v1"},
)

print(f"Model resource: {model.resource_name}")
# projects/123456/locations/us-central1/models/987654321
```

### Deploy to Endpoint with Autoscaling

```python
endpoint = aiplatform.Endpoint.create(
    display_name="taxi-tip-endpoint",
    labels={"team": "ml-platform"},
)

model.deploy(
    endpoint=endpoint,
    deployed_model_display_name="taxi-tip-v1",
    machine_type="n1-standard-4",
    min_replica_count=1,
    max_replica_count=5,
    traffic_percentage=100,
    sync=True,
)

print(f"Endpoint: {endpoint.resource_name}")
```

### Canary Deployment (Traffic Split)

When deploying a new model version, route a fraction of traffic to it first:

```python
# Upload new model version
model_v2 = aiplatform.Model.upload(
    display_name="taxi-tip-xgboost",
    artifact_uri="gs://your-project-id-taxi-pipeline/models/v2",
    serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-3:latest",
    parent_model=model.resource_name,  # links as new version of same model
)

# Deploy v2 with 10% canary traffic
endpoint.deploy(
    model=model_v2,
    deployed_model_display_name="taxi-tip-v2-canary",
    machine_type="n1-standard-4",
    min_replica_count=1,
    max_replica_count=3,
    traffic_percentage=10,  # 10% to v2, 90% stays on v1
    sync=True,
)

# After validation, promote to 100%
endpoint.update(traffic_split={
    model_v2.resource_name: 100,
})
```

This implements the canary deployment strategy from [Chapter 15](15-mlops-practices.md#deployment-strategies-for-models).

### Online Prediction

```python
# Single prediction
instance = {
    "hour_of_day": 14,
    "day_of_week": 3,
    "month": 6,
    "is_weekend": 0,
    "trip_seconds": 720,
    "trip_miles": 3.2,
    "speed_mph": 16.0,
    "fare": 12.50,
    "tolls": 0.0,
    "extras": 1.0,
    "trip_total": 13.50,
    "fare_per_mile": 3.91,
    "fare_per_minute": 1.04,
    "pickup_area": 8,
    "dropoff_area": 32,
    "same_area": 0,
    "is_credit_card": 1,
}

prediction = endpoint.predict(instances=[instance])
print(f"Tip prediction: {prediction.predictions[0]}")
# 1 (will tip) or 0 (won't tip)
```

### Batch Prediction

For large-scale offline scoring:

```python
batch_job = model.batch_predict(
    job_display_name="taxi-tip-batch-2024-01",
    bigquery_source="bq://your-project-id.taxi_pipeline.batch_input",
    bigquery_destination_prefix="bq://your-project-id.taxi_pipeline",
    instances_format="bigquery",
    predictions_format="bigquery",
    machine_type="n1-standard-8",
    max_replica_count=10,
    sync=True,
)
```

---

## 16.9 Model Monitoring and Drift Detection

### Set Up Vertex AI Model Monitoring

Vertex AI Model Monitoring continuously monitors prediction requests for feature skew (training-serving skew) and drift (distribution change over time). This automates the monitoring architecture described in [Chapter 15](15-mlops-practices.md#monitoring-architecture).

```python
from google.cloud import aiplatform

# Create monitoring job on the endpoint
monitoring_job = aiplatform.ModelDeploymentMonitoringJob.create(
    display_name="taxi-tip-monitoring",
    endpoint=endpoint,
    logging_sampling_strategy={
        "random_sample_config": {"sample_rate": 0.8}  # log 80% of requests
    },
    log_ttl=3600 * 24 * 30,  # retain logs for 30 days
    schedule_config={
        "monitor_interval": {"seconds": 3600}  # check every hour
    },
    # Skew detection: training data vs serving data
    model_deployment_monitoring_objective_configs=[
        {
            "deployed_model_id": endpoint.list_models()[0].id,
            "objective_config": {
                "training_dataset": {
                    "bigquery_source": {
                        "uri": "bq://your-project-id.taxi_pipeline.train_features"
                    },
                },
                "training_prediction_skew_detection_config": {
                    "skew_thresholds": {
                        "hour_of_day": {"value": 0.3},
                        "trip_miles": {"value": 0.3},
                        "fare": {"value": 0.3},
                        "speed_mph": {"value": 0.3},
                        "pickup_area": {"value": 0.3},
                    }
                },
                "prediction_drift_detection_config": {
                    "drift_thresholds": {
                        "hour_of_day": {"value": 0.3},
                        "trip_miles": {"value": 0.3},
                        "fare": {"value": 0.3},
                        "speed_mph": {"value": 0.3},
                        "pickup_area": {"value": 0.3},
                    }
                },
            },
        }
    ],
    alert_config={
        "email_alert_config": {
            "user_emails": ["ml-team@company.com"]
        }
    },
)
```

### Monitoring Thresholds

| Feature | Skew Threshold | Drift Threshold | Rationale |
|---------|---------------|-----------------|-----------|
| `hour_of_day` | 0.3 | 0.3 | Trip time patterns shift with behavior changes |
| `trip_miles` | 0.3 | 0.3 | COVID reduced average trip distance |
| `fare` | 0.3 | 0.3 | Fare structure changes or inflation |
| `speed_mph` | 0.3 | 0.3 | Traffic pattern changes (less congestion during COVID) |
| `pickup_area` | 0.3 | 0.3 | Geographic demand shifts (e.g., airport vs downtown) |
| `fare_per_mile` | 0.3 | 0.3 | Pricing model changes |

Thresholds use Jensen-Shannon divergence. A threshold of 0.3 is moderately sensitive -- lower values produce more alerts (see [Chapter 15 -- Types of Drift](15-mlops-practices.md#types-of-drift) for drift detection theory).

### Demonstrating Drift with COVID-Era Data

To validate that monitoring works, simulate serving requests from the COVID period against a model trained on 2018-2019 data:

```python
from google.cloud import bigquery
import time

client = bigquery.Client()

# Pull a batch of COVID-era trips
covid_trips = client.query("""
    SELECT * FROM `taxi_pipeline.test_features`
    WHERE month >= 4 AND month <= 6  -- Apr-Jun 2020, peak COVID impact
    LIMIT 5000
""").to_dataframe()

# Send as prediction requests (monitoring logs these automatically)
instances = covid_trips[FEATURE_COLUMNS].to_dict("records")
for i in range(0, len(instances), 100):
    batch = instances[i:i+100]
    endpoint.predict(instances=batch)
    time.sleep(1)  # avoid overwhelming the endpoint

print(f"Sent {len(instances)} COVID-era predictions for drift detection")
```

Expected drift alerts after the monitoring job runs:
- **`trip_miles`**: Average trip distance dropped during COVID (fewer long-distance trips)
- **`speed_mph`**: Average speed increased (less traffic congestion)
- **`pickup_area`**: Distribution shifted away from downtown/airport areas
- **`hour_of_day`**: Fewer early-morning and late-night trips

### Cloud Monitoring Alerts

Set up alerts for drift events that trigger automated responses:

```python
from google.cloud import monitoring_v3

alert_client = monitoring_v3.AlertPolicyServiceClient()
project_name = f"projects/your-project-id"

alert_policy = monitoring_v3.AlertPolicy(
    display_name="Model Drift Alert - Taxi Tip",
    conditions=[
        monitoring_v3.AlertPolicy.Condition(
            display_name="Feature drift detected",
            condition_threshold=monitoring_v3.AlertPolicy.Condition.MetricThreshold(
                filter=(
                    'resource.type="aiplatform.googleapis.com/Endpoint" '
                    'AND metric.type="aiplatform.googleapis.com/prediction/online/feature_drift"'
                ),
                comparison=monitoring_v3.ComparisonType.COMPARISON_GT,
                threshold_value=0.3,
                duration={"seconds": 0},
                aggregations=[
                    monitoring_v3.Aggregation(
                        alignment_period={"seconds": 3600},
                        per_series_aligner=monitoring_v3.Aggregation.Aligner.ALIGN_MAX,
                    )
                ],
            ),
        )
    ],
    notification_channels=["projects/your-project-id/notificationChannels/CHANNEL_ID"],
    alert_strategy=monitoring_v3.AlertPolicy.AlertStrategy(
        auto_close={"seconds": 86400},  # auto-close after 24h
    ),
)

alert_client.create_alert_policy(name=project_name, alert_policy=alert_policy)
```

---

## 16.10 Automated Retraining Pipeline

### Architecture

```
┌──────────────┐         ┌──────────────┐
│ Cloud         │         │ Model        │
│ Scheduler     │────────►│ Monitoring   │
│ (weekly)      │         │ (drift alert)│
└──────┬───────┘         └──────┬───────┘
       │                        │
       │    ┌───────────────────┘
       │    │
       ▼    ▼
┌──────────────────┐
│  Cloud Function  │
│  (retrain        │
│   trigger)       │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ Vertex AI        │
│ Pipeline          │
│ (rolling window  │
│  retrain)        │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ Champion /       │
│ Challenger       │
│ Evaluation       │
└──────┬───────────┘
       │
  ┌────┴────┐
  │ Better? │
  ▼         ▼
Deploy    Keep
canary    current
```

### Cloud Function: Retrain Trigger

```python
"""Cloud Function to trigger the retraining pipeline."""
import functions_framework
from google.cloud import aiplatform


@functions_framework.cloud_event
def trigger_retrain(cloud_event):
    """Triggered by Cloud Scheduler or Model Monitoring alert."""
    aiplatform.init(project="your-project-id", location="us-central1")

    # Calculate rolling window dates
    from datetime import datetime, timedelta
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")  # 2 years
    test_start = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")   # last 90 days

    job = aiplatform.PipelineJob(
        display_name=f"taxi-retrain-{end_date}",
        template_path="gs://your-project-id-taxi-pipeline/pipelines/taxi_tip_pipeline.yaml",
        parameter_values={
            "project_id": "your-project-id",
            "train_start": start_date,
            "train_end": test_start,
            "test_start": test_start,
            "test_end": end_date,
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.1,
        },
        pipeline_root="gs://your-project-id-taxi-pipeline/retrain-runs",
    )

    job.run(
        service_account="vertex-pipeline-sa@your-project-id.iam.gserviceaccount.com",
        sync=False,
    )
    print(f"Retrain pipeline submitted: {job.resource_name}")
```

### Cloud Scheduler (Weekly Retrain)

```bash
gcloud scheduler jobs create http retrain-taxi-weekly \
    --schedule="0 2 * * 0" \
    --uri="https://${REGION}-${PROJECT_ID}.cloudfunctions.net/trigger_retrain" \
    --http-method=POST \
    --oidc-service-account-email="${SA_EMAIL}" \
    --location="${REGION}" \
    --time-zone="America/Chicago"
```

### Champion/Challenger Evaluation

The retraining pipeline should compare the new model (challenger) against the currently deployed model (champion) before promoting:

```python
@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=[
        "google-cloud-aiplatform==1.38.0",
        "google-cloud-bigquery==3.13.0",
        "google-cloud-storage==2.13.0",
        "xgboost==2.0.3",
        "scikit-learn==1.3.2",
        "pandas==2.1.4",
        "db-dtypes==1.2.0",
    ],
)
def champion_challenger(
    project_id: str,
    region: str,
    challenger_model_path: str,
    test_table: str,
    endpoint_name: str,
) -> bool:
    """Compare challenger model against deployed champion on the same test set."""
    import pickle
    import pandas as pd
    import xgboost as xgb
    from google.cloud import aiplatform, bigquery, storage
    from sklearn.metrics import roc_auc_score

    FEATURES = [
        "hour_of_day", "day_of_week", "month", "is_weekend",
        "trip_seconds", "trip_miles", "speed_mph",
        "fare", "tolls", "extras", "trip_total",
        "fare_per_mile", "fare_per_minute",
        "pickup_area", "dropoff_area", "same_area", "is_credit_card",
    ]

    # Load test data
    client = bigquery.Client(project=project_id)
    test_df = client.query(f"SELECT * FROM `{test_table}`").to_dataframe()
    X_test = test_df[FEATURES]
    y_test = test_df["tip_label"]

    # Champion: get predictions from deployed endpoint
    aiplatform.init(project=project_id, location=region)
    endpoints = aiplatform.Endpoint.list(
        filter=f'display_name="{endpoint_name}"'
    )
    if not endpoints:
        print("No champion endpoint found, challenger wins by default")
        return True

    champion_preds = endpoints[0].predict(
        instances=X_test.head(5000).to_dict("records")
    )
    champion_auc = roc_auc_score(y_test.head(5000), champion_preds.predictions)

    # Challenger: load model and predict locally
    bucket_name = challenger_model_path.replace("gs://", "").split("/")[0]
    prefix = "/".join(challenger_model_path.replace("gs://", "").split("/")[1:])
    bucket = storage.Client(project=project_id).bucket(bucket_name)
    bucket.blob(f"{prefix}/model.pkl").download_to_filename("/tmp/challenger.pkl")

    with open("/tmp/challenger.pkl", "rb") as f:
        challenger = pickle.load(f)

    challenger_prob = challenger.predict_proba(X_test.head(5000))[:, 1]
    challenger_auc = roc_auc_score(y_test.head(5000), challenger_prob)

    print(f"Champion AUC:    {champion_auc:.4f}")
    print(f"Challenger AUC:  {challenger_auc:.4f}")

    challenger_wins = challenger_auc > champion_auc
    print(f"Winner: {'Challenger' if challenger_wins else 'Champion'}")
    return challenger_wins
```

### Automated Canary Promotion

After the challenger wins, deploy it with a canary split and auto-promote after a validation period:

```python
@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["google-cloud-aiplatform==1.38.0"],
)
def canary_deploy(
    project_id: str,
    region: str,
    model_gcs_path: str,
    endpoint_name: str,
    canary_percentage: int,
):
    """Deploy challenger as canary, to be promoted later."""
    from google.cloud import aiplatform

    aiplatform.init(project=project_id, location=region)

    model = aiplatform.Model.upload(
        display_name="taxi-tip-xgboost",
        artifact_uri=model_gcs_path,
        serving_container_image_uri=(
            "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-3:latest"
        ),
    )

    endpoints = aiplatform.Endpoint.list(
        filter=f'display_name="{endpoint_name}"'
    )
    endpoint = endpoints[0]

    model.deploy(
        endpoint=endpoint,
        deployed_model_display_name=f"taxi-tip-canary-{model.name}",
        machine_type="n1-standard-4",
        min_replica_count=1,
        max_replica_count=3,
        traffic_percentage=canary_percentage,
        sync=True,
    )
    print(f"Canary deployed at {canary_percentage}% traffic")
```

---

## 16.11 Cost Management and Production

### Cost Breakdown

Estimated monthly costs for the pipeline running in production (us-central1):

| Resource | Configuration | Monthly Cost |
|----------|-------------|-------------|
| Vertex AI Endpoint | n1-standard-4, 1-5 replicas, autoscaled | ~$150 |
| Model Monitoring | Hourly checks, 80% sampling | ~$50 |
| BigQuery | ~10TB queried/month (feature eng + extraction) | ~$50 |
| Cloud Storage | ~50GB pipeline artifacts + models | ~$1 |
| Vertex AI Training | Weekly retrain, n1-standard-8, ~2h each | ~$35 |
| Cloud Functions | Retrain triggers, minimal invocations | ~$1 |
| Cloud Scheduler | 1 weekly job | ~$0 |
| Hyperparameter tuning | Monthly 20-trial sweep, n1-standard-8 | ~$70 |
| Logging/Monitoring | Cloud Monitoring + Logging | ~$20 |
| **Total** | | **~$377/mo** |

### Cost Optimization Strategies

| Strategy | Savings | Trade-off |
|----------|---------|-----------|
| Use `e2-standard-4` instead of `n1-standard-4` for endpoint | ~30% on compute | Slightly less consistent performance |
| Reduce monitoring sampling to 20% | ~60% on monitoring | Longer to detect drift |
| Retrain biweekly instead of weekly | ~50% on training | Slower reaction to drift |
| Use preemptible VMs for training | ~60-80% on training | Jobs may be interrupted |
| Set endpoint min replicas to 0 (scale-to-zero) | ~90% during low traffic | Cold start latency on first request |
| Use committed use discounts (CUDs) for endpoint | ~30-55% | 1-3 year commitment |

### Security Hardening

| Control | Implementation | Purpose |
|---------|---------------|---------|
| VPC Service Controls | Service perimeter around Vertex AI + BigQuery + GCS | Prevent data exfiltration |
| CMEK | Customer-managed encryption keys for model artifacts | Encryption key control |
| IAM least privilege | Separate service accounts per pipeline stage | Blast radius reduction |
| Private endpoints | VPC-peered Vertex AI endpoints, no public IP | Network isolation |
| Audit logging | Cloud Audit Logs for all Vertex AI API calls | Compliance and forensics |
| Artifact provenance | Signed pipeline artifacts in Artifact Registry | Supply chain integrity |

```bash
# Example: Create VPC-SC perimeter
gcloud access-context-manager perimeters create ml-perimeter \
    --title="ML Pipeline Perimeter" \
    --resources="projects/PROJECT_NUMBER" \
    --restricted-services="aiplatform.googleapis.com,bigquery.googleapis.com,storage.googleapis.com" \
    --policy=POLICY_ID
```

---

## 16.12 Putting It All Together

### Quick-Start Reproduction Guide

```bash
# 1. Set up environment
export PROJECT_ID="your-project-id"
export REGION="us-central1"
gcloud config set project $PROJECT_ID

# 2. Enable APIs and create infrastructure (Section 16.2)
gcloud services enable aiplatform.googleapis.com bigquery.googleapis.com \
    cloudbuild.googleapis.com storage.googleapis.com
gsutil mb -l $REGION gs://${PROJECT_ID}-taxi-pipeline
bq mk --dataset --location=US ${PROJECT_ID}:taxi_pipeline

# 3. Create service account (Section 16.2)
gcloud iam service-accounts create vertex-pipeline-sa
# (grant roles as shown in Section 16.2)

# 4. Install Python dependencies
pip install google-cloud-aiplatform kfp xgboost scikit-learn \
    google-cloud-bigquery pandas db-dtypes

# 5. Compile and submit the pipeline (Section 16.7)
python -c "
from kfp import compiler
# (import your pipeline definition)
compiler.Compiler().compile(taxi_tip_pipeline, 'pipeline.yaml')
"

# 6. Run the pipeline
python -c "
from google.cloud import aiplatform
aiplatform.init(project='${PROJECT_ID}', location='${REGION}')
job = aiplatform.PipelineJob(
    display_name='taxi-tip-initial',
    template_path='pipeline.yaml',
    parameter_values={'project_id': '${PROJECT_ID}'},
    pipeline_root='gs://${PROJECT_ID}-taxi-pipeline/runs',
)
job.run(service_account='vertex-pipeline-sa@${PROJECT_ID}.iam.gserviceaccount.com')
"

# 7. Set up monitoring (Section 16.9) and retraining (Section 16.10)
```

### Operational Runbook

| Scenario | Detection | Response |
|----------|-----------|----------|
| Feature drift alert | Model Monitoring email/PagerDuty | Check drift report in Vertex AI console. If multiple features drifting, trigger retrain. If single feature, investigate data source. |
| Model performance degradation | Prediction distribution shift, business metric drop | Compare recent predictions against labeled sample. If confirmed, trigger retrain with recent data. |
| Pipeline failure | Cloud Monitoring alert on pipeline job status | Check Vertex AI Pipeline logs. Common causes: BigQuery quota, GCS permissions, stale service account key. |
| Endpoint latency spike | Cloud Monitoring p99 latency alert | Check autoscaler status. If max replicas reached, increase `max_replica_count`. Check for input data anomalies (oversized requests). |
| Training job OOM | Training job fails with memory error | Increase `machine_type` (e.g., n1-standard-8 -> n1-standard-16). Or sample training data if dataset has grown significantly. |
| Cost spike | Billing alert | Check for runaway autoscaling (set max replicas), stuck pipeline jobs, or unintentional hyperparameter tuning sweeps. |

### Common Failure Modes

| Failure | Cause | Fix |
|---------|-------|-----|
| `PermissionDenied` on BigQuery | Service account missing `bigquery.jobUser` role | Re-run IAM binding commands from Section 16.2 |
| Pipeline hangs at `extract_data` | BigQuery query slot exhaustion | Use `BATCH` priority or request quota increase |
| Model upload fails | GCS path doesn't contain expected artifact format | Ensure `model.pkl` is at the root of `artifact_uri` path |
| Monitoring job shows no data | Endpoint logging not enabled or sampling too low | Increase `sample_rate`, verify endpoint has traffic |
| Canary deployment shows worse metrics | New model undertrained or data quality issue | Roll back canary (`traffic_percentage=0`), investigate training data |
| Cloud Function timeout | Pipeline submission takes too long | Use `sync=False` in pipeline run (function only needs to submit, not wait) |

---

## Review Questions

1. Why do we filter out cash payment trips instead of including them with `tip_label=0`? What kind of model error would including them introduce?

2. Why use a temporal train/test split (2018-2019 / 2020) instead of a random 80/20 split? In what scenarios would a random split be acceptable?

3. The validation gate checks both absolute thresholds (AUC > 0.95) and regression thresholds (no more than 0.02 AUC drop from baseline). Why are both needed? Give a scenario where each alone would be insufficient.

4. What would happen if the model monitoring drift thresholds were set too low (e.g., 0.05 instead of 0.3)? What about too high (e.g., 0.9)?

5. The champion/challenger evaluation compares models on the same test set. What could go wrong if the test set itself has drifted significantly from current production data?

6. Why does the automated retraining use a rolling 2-year window instead of training on all historical data? What trade-off does the window size control?

## Hands-On Exercises

1. **Extend the feature set:** Add a `company` feature (the taxi company) as a categorical variable. How does it affect model performance? Which companies' riders tip more?

2. **Monthly drift analysis:** Split the 2020 test data by month and plot AUC-ROC over time. Identify the exact month where drift impact is largest. Correlate with COVID timeline events.

3. **Add a Slack notification component:** Write a KFP component that sends a Slack message with model metrics after training completes. Integrate it into the pipeline.

4. **Implement shadow mode:** Before canary deployment, add a pipeline stage that runs the new model in shadow mode (predictions logged but not served) for 24 hours. Compare shadow predictions against the champion.

5. **Cost optimization experiment:** Modify the endpoint to use scale-to-zero (`min_replica_count=0`). Measure cold-start latency for the first request after scale-up. Is the latency acceptable for your use case?

6. **Feature importance analysis:** After training, extract XGBoost feature importance scores. Which features drive tip prediction most? Does the ranking change between the 2018-2019 model and a model retrained on 2020 data?

## Key Takeaways

1. **Filter data artifacts, not just outliers.** Cash payments recording zero tips is a data collection artifact, not genuine non-tipping behavior. Including them would inflate accuracy while teaching the model the wrong pattern.

2. **Temporal splits simulate production.** Random splits leak future information into training. A temporal split reveals how the model actually degrades when deployed forward in time -- the only direction that matters in production.

3. **Validation gates prevent silent regression.** Automated deployment without quality gates risks shipping a worse model. Combine absolute thresholds (minimum acceptable performance) with regression thresholds (no backsliding from current champion).

4. **Real drift is messy and multi-feature.** COVID didn't just shift one feature -- it changed trip distances, times, speeds, locations, and volumes simultaneously. Monitoring individual features catches the components; monitoring model performance catches the aggregate impact.

5. **Champion/challenger is essential for automated retraining.** Never auto-deploy a retrained model without comparing it to what's currently serving. More data doesn't always mean a better model -- data quality, distribution shifts, and feature pipeline bugs can all cause regressions.

6. **Cost scales with monitoring and serving, not training.** The always-on endpoint (~$150/mo) and continuous monitoring (~$50/mo) dominate costs. Training is cheap and intermittent. Optimize for the always-on components first.
