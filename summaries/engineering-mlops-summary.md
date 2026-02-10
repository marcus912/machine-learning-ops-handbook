# Engineering MLOps - Comprehensive Summary

**Author:** Emmanuel Raj (Senior ML Engineer at TietoEvry, Member of European AI Alliance)
**Publisher:** Packt Publishing, April 2021
**Pages:** 370 (13 Chapters)
**Focus:** Building, deploying, and managing production-ready ML lifecycles at scale

---

## Core Definition

> "Machine learning is not just code; it is code plus data."

**MLOps = Machine Learning + DevOps + Data Engineering**

MLOps bridges data and code together over time, solving challenges like slow/brittle deployments, lack of reproducibility, and poor traceability.

---

## The MLOps Workflow Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    MLOps Pipeline                       │
│         BUILD  →  DEPLOY  →  MONITOR                    │
├─────────────────────────────────────────────────────────┤
│  Drivers: Data, Code, Artifacts, Middleware, Infra      │
└─────────────────────────────────────────────────────────┘
```

---

## Section 1: Framework for Building Machine Learning Models

### Chapter 1: Fundamentals of an MLOps Workflow

**Evolution of Software Development:**
1. **Waterfall Method** (~1995): Linear, non-iterative, requirements must be fixed upfront
2. **Agile Method**: Iterative, user-centric, enables co-creation with customers
3. **DevOps Method**: Extends Agile with CI/CD, enables shipping software in minutes
4. **MLOps Method**: Addresses unique ML challenges (data + code evolution)

**Key Insight:** Moore's Law (2x compute every 18 months) has been outpaced. Post-2012, AI compute doubles every 3.4 months with 35x growth per 18 months.

**ML Adoption Drivers:**
- $70B+ global private AI investment (2019)
- 300% growth in peer-reviewed AI papers (1998-2018)
- 58% of large companies adopted AI in at least one function (2019)

### Chapter 2: Characterizing Your Machine Learning Problem

**Types of ML Models:**
| Type | Description | Examples |
|------|-------------|----------|
| **Learning Models** | Supervised, unsupervised, reinforcement | Classification, clustering, RL agents |
| **Hybrid Models** | Combine multiple approaches | Ensemble methods |
| **Statistical Models** | Traditional statistical methods | Time series, regression |
| **HITL Models** | Human-in-the-Loop systems | Active learning, annotation |

**MLOps Scale Patterns:**
- **Small Data Ops**: Single team, limited data
- **Big Data Ops**: Large datasets, distributed processing
- **Hybrid MLOps**: Mix of batch and real-time
- **Large-scale MLOps**: Enterprise-wide, multiple teams

**Implementation Roadmap:**
1. **Phase 1 - ML Development**: Problem definition, data collection, model training
2. **Phase 2 - Transition to Operations**: Packaging, API development, CI/CD setup
3. **Phase 3 - Operations**: Production deployment, monitoring, governance

### Chapter 3: Code Meets Data

**Tools Setup:**
- **MLflow**: Experiment tracking, model versioning
- **Azure Machine Learning**: Cloud ML platform
- **Azure DevOps**: CI/CD pipelines
- **JupyterHub**: Development environment

**10 Principles of Source Code Management for ML:**
1. Version control everything (code, data, models)
2. Modular and reusable code
3. Separate configuration from code
4. Use environment management
5. Document dependencies
6. Implement logging
7. Write testable code
8. Use consistent coding standards
9. Automate repetitive tasks
10. Enable reproducibility

**Data Quality Assessment:**
- Calibrating missing data
- Label encoding
- Feature engineering (e.g., Future_weather_condition)
- Data correlations and filtering
- Time series analysis
- Data registration and versioning

**Feature Store:** Centralized repository for storing and serving ML features to eliminate training-serving skew.

### Chapter 4: Machine Learning Pipelines

**Pipeline Components:**
```
Data Ingestion → Feature Engineering → Model Training → Model Testing → Model Packaging → Model Registration
```

**Data Ingestion:**
- Extract, Transform, Load (ETL) operations
- Split and version data (e.g., 80/20 train/test split)
- Enables end-to-end traceability

**Model Training with Hyperparameter Optimization:**
- **Grid Search**: Exhaustive search over parameter combinations
- **Random Search**: Random sampling of parameter space

**Example Models Trained:**
1. **Support Vector Machine (SVM)**: Classification using hyperplanes, known for accuracy with less compute
2. **Random Forest**: Ensemble of decision trees, robust and interpretable

**Model Packaging with ONNX:**
```python
from skl2onnx import convert_sklearn
initial_type = [('float_input', FloatTensorType([None, 6]))]
onx = convert_sklearn(model, initial_types=initial_type)
with open("model.onnx", "wb") as f:
    f.write(onx.SerializeToString())
```

**Benefits of ONNX:**
- Framework agnostic (sklearn, TensorFlow, PyTorch interoperability)
- Portable across platforms
- Standard for model serialization

### Chapter 5: Model Evaluation and Packaging

**Evaluation Metrics:**
| Metric | Formula | Use Case |
|--------|---------|----------|
| **Accuracy** | Correct / Total | Overall correctness |
| **Precision** | TP / (TP + FP) | When false positives are costly |
| **Recall** | TP / (TP + FN) | When false negatives are costly |
| **F-Score** | 2 * (P * R) / (P + R) | Balance precision and recall |

**Production Testing Methods:**
- **Batch Testing**: Evaluate on held-out datasets
- **A/B Testing**: Compare models on live traffic
- **Shadow/Stage Testing**: Run new model alongside production without affecting users
- **Testing in CI/CD**: Automated testing in pipelines

---

## Section 2: Deploying Machine Learning Models at Scale

### Chapter 6: Key Principles for Deploying Your ML System

**Research vs Production:**
| Aspect | Research | Production |
|--------|----------|------------|
| Data | Static datasets | Dynamic, streaming |
| Fairness | Optional | Required |
| Interpretability | Nice to have | Often mandated |
| Performance | Accuracy focused | Latency, reliability critical |
| Priority | Experimentation | Stability |

**Deployment Targets:**
- **Azure Container Instances (ACI)**: For testing and dev environments
- **Azure Kubernetes Service (AKS)**: For production with auto-scaling
- **Edge Devices**: For low-latency, offline inference

**Deployment Code Example (ACI):**
```python
from azureml.core.webservice import AciWebservice
deployment_config = AciWebservice.deploy_configuration(
    cpu_cores=1, memory_gb=1,
    auth_enabled=False, ssl_enabled=False,
    app_insights_enabled=True
)
```

### Chapter 7: Building Robust CI/CD Pipelines

**Three Pillars:**
1. **Continuous Integration (CI)**: Automated testing of code changes
2. **Continuous Delivery (CD)**: Automated preparation for deployment
3. **Continuous Deployment**: Automated deployment to production

**Pipeline Execution Triggers:**
| Trigger Type | Description | Example |
|-------------|-------------|---------|
| **Git Triggers** | Code commit, PR | Retrain on develop branch changes |
| **Artifactory Triggers** | New model registered | Deploy new model version |
| **Schedule Triggers** | Time-based | Daily retraining at 12:00 |
| **API Triggers** | External events | Admin comment triggers retrain |
| **Docker Hub Triggers** | New image pushed | Deploy updated container |

**Azure DevOps Pipeline Setup:**
1. Create service principal for authentication
2. Install Azure ML extension
3. Configure build and release pipelines
4. Set up test environment (ACI)
5. Configure Git triggers for develop branch
6. Set up production environment (AKS)

### Chapter 8: APIs and Microservices

**Why Microservices for ML?**
- Independent scaling of components
- Team autonomy (separate ownership)
- Fault isolation (one crash doesn't bring down system)
- Easier maintenance and updates

**Monolith vs Microservices Evolution:**
1. **PoC Stage (Monolith)**: Quick validation, all-in-one app
2. **Production Stage (Microservices)**: Separated components for reliability

**REST API HTTP Methods:**
| Method | CRUD | ML Use Case |
|--------|------|-------------|
| GET | Read | Retrieve predictions, model info |
| POST | Create | Submit data for inference |
| PUT | Update | Update model configuration |
| DELETE | Delete | Remove deployed model |

**FastAPI Implementation:**
```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class WeatherVariables(BaseModel):
    temp_c: float
    humidity: float
    wind_speed_kmph: float
    # ... other features

@app.post('/predict')
def predict_weather(data: WeatherVariables):
    # Scale input data
    scaled_data = scaler.transform(data_array)
    # Run inference
    prediction = sess.run([label_name], {input_name: scaled_data})
    return {"prediction": prediction}
```

**Docker Containerization:**
```dockerfile
FROM python:3.8-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "weather_api:app", "--host", "0.0.0.0", "--port", "80"]
```

### Chapter 9: Testing and Securing Your ML Solution

**Testing Types:**
| Type | Purpose | When |
|------|---------|------|
| **Data Testing** | Schema validation, quality checks | Before training |
| **Model Testing** | Performance metrics, bias detection | After training |
| **Pre-training Tests** | Data pipeline validation | Pipeline setup |
| **Post-training Tests** | Model performance validation | Before deployment |
| **Integration Tests** | End-to-end system testing | Before production |

**Security Considerations:**
- Authentication/authorization for APIs
- SSL/TLS encryption
- Input validation and sanitization
- Model access control
- Data privacy compliance (GDPR, etc.)

### Chapter 10: Essentials of Production Release

**Production Infrastructure Setup:**
- Kubernetes for orchestration
- Auto-scaling based on load
- Load balancing for high availability
- Health checks and readiness probes

**Release Strategies:**
- **Blue-Green**: Two identical environments, instant switch
- **Canary**: Gradual rollout to small percentage
- **Rolling**: Incremental replacement of instances

---

## Section 3: Monitoring Machine Learning Models in Production

### Chapter 11: Key Principles for Monitoring Your ML System

**Explainable Monitoring = Model Transparency + Explainability**

**Three Questions to Answer:**
1. How are model decisions reached?
2. Are model decisions fair and unbiased?
3. Is the model compliant with regulations?

**Explainable AI Methods:**
- **SHAP (SHapley Additive exPlanations)**: Feature importance based on game theory
- **LIME (Local Interpretable Model-agnostic Explanations)**: Local explanations
- **Feature Importance**: Global feature rankings

### Chapter 12: Model Serving and Monitoring

**Three Types of Drift:**
| Type | Description | Detection |
|------|-------------|-----------|
| **Data Drift** | Input data distribution changes | Statistical tests on features |
| **Feature Drift** | Individual feature distributions change | Per-feature monitoring |
| **Model/Concept Drift** | Relationship between inputs and outputs changes | Prediction quality degradation |

**Drift Detection Actions:**
- If drift exceeds threshold (e.g., 70%), trigger:
  - Email notification to admin
  - Automatic model retraining
  - Deploy backup model

**Application Performance Monitoring:**
- Azure Application Insights integration
- Track failed requests
- Monitor server response time
- Analyze request throughput
- Check service availability

### Chapter 13: Governing the ML System for Continual Learning

**Governance Framework:**
- Model versioning and lineage tracking
- Audit trails for all operations
- Access control and permissions
- Compliance documentation
- Model retirement policies

**Continual Learning Triggers:**
- Performance degradation detection
- Significant data distribution changes
- Scheduled interval retraining
- New labeled data availability

**CI/CD for Continual Learning:**
1. Monitor detects drift or degradation
2. Trigger retraining pipeline
3. Evaluate new model against baseline
4. A/B test or canary deploy
5. Gradual rollout if metrics improve
6. Full deployment or rollback

---

## Key Tools & Technologies

| Category | Tools |
|----------|-------|
| **ML Frameworks** | scikit-learn, TensorFlow, PyTorch |
| **Experiment Tracking** | MLflow, Azure ML Experiments |
| **Model Serving** | FastAPI, Flask, Azure ML Endpoints |
| **Model Format** | ONNX (Open Neural Network Exchange) |
| **Containerization** | Docker, Kubernetes |
| **CI/CD** | Azure DevOps, GitHub Actions |
| **Monitoring** | Azure Monitor, Application Insights |
| **Data Versioning** | Azure ML Datasets, DVC |

---

## MLOps Maturity Levels

| Level | Name | Characteristics |
|-------|------|-----------------|
| 0 | Manual | Experimental, infrequent releases, minimal monitoring |
| 1 | Pipeline Automation | Automated retraining, data/model validation, feature stores |
| 2 | CI/CD Automation | Full automation, source control, model registries, comprehensive monitoring |

---

## Best Practices Summary

1. **Version Everything**: Data, code, models, and configurations
2. **Automate Pipelines**: CI/CD for training and deployment
3. **Monitor Continuously**: Track drift, performance, and business metrics
4. **Ensure Reproducibility**: Same inputs produce same outputs
5. **Document Thoroughly**: Model cards, data lineage, decision logs
6. **Test Rigorously**: Unit, integration, and production testing
7. **Plan for Failure**: Rollback strategies, fault tolerance
8. **Think Security First**: Authentication, encryption, access control
9. **Enable Explainability**: Build trust and meet compliance
10. **Iterate Continuously**: MLOps is a cycle, not a one-time setup

---

## Target Audience

- Data Scientists transitioning to production
- ML Engineers building scalable systems
- Software Developers integrating ML
- DevOps Professionals supporting ML teams

*Best suited for intermediate ML practitioners seeking production implementation guidance using Azure cloud services.*
