# Blockchain Tethered AI - Comprehensive Summary

**Authors:** Karen Kilroy, Lynn Riley, Deepak Bhatta
**Publisher:** O'Reilly Media, 2023
**Pages:** 307
**Focus:** Using enterprise blockchain to create trackable, traceable, auditable, and reversible AI/ML systems

---

## Core Definition

> "AI's ability to change itself through program synthesis could take the technology beyond human control... This hands-on book describes how to build simple blockchain controls for verifying, tracking, tracing, auditing, and even reversing AI."

---

## Chapter 1: Why Build a Blockchain Truth Machine for AI?

### AI Trust Deficit - Key Concerns

| Concern | Description |
|---------|-------------|
| **Opaque Box Algorithms** | Cannot explain how decisions are made |
| **Genetic Algorithms** | Self-modifying code with unpredictable evolution |
| **Data Quality Issues** | Outliers, edge cases, biased training data |
| **Program Synthesis** | AI writing its own code |
| **Model/Data Drift** | Performance degradation over time |
| **Adversarial Attacks** | Malicious manipulation of inputs/models |
| **Technological Singularity** | AI surpassing human control |

### Blockchain as an AI Tether

**Enterprise Blockchain Features:**
- **Distributed Ledger**: Replicated across network nodes
- **Immutability**: Tamper-evident record keeping
- **Smart Contracts**: Automated governance enforcement
- **Cryptographic Verification**: SHA-256 hashing for integrity
- **Identity Management**: Verifiable participant identities

### Four Blockchain Controls Framework

1. **Pre-establishing Identity and Workflow**
2. **Distributing Tamper-Evident Verification**
3. **Governing, Instructing, and Inhibiting Intelligent Agents**
4. **Showing Authenticity Through User-Viewable Provenance**

---

## Chapter 2: Blockchain Controls for AI

### Control 1: Identity and Workflow

**Identity Management:**
- Digital certificates for people and systems
- Integration with enterprise directories (LDAP)
- Membership Service Providers (MSPs)
- Role-based access control

**Workflow Criteria:**
- Predefined approval chains
- Multi-party consensus requirements
- Automated state transitions

### Control 2: Tamper-Evident Verification

**Crypto Anchors:**
- SHA-256 hashing of datasets, models, and pipelines
- Hash storage on blockchain (not raw data)
- Verification before deployment

**AI Hack Detection:**
- Data poisoning detection
- Model tampering alerts
- Pipeline integrity verification

**Federated Learning Integration:**
- Distributed model training provenance
- Cross-organization collaboration tracking

### Control 3: On-Chain Governance

**Governance Group:**
- Stakeholder voting mechanisms
- Approval workflows
- Purpose and domain constraints

**Compliant Intelligent Agents:**
- Ethics-aware shutdown capabilities
- Boundary enforcement
- Drift monitoring alerts

### Control 4: User-Viewable Provenance

**Audit Trail Contents:**
- Who created/modified the model
- What data was used
- When changes occurred
- Why decisions were made
- Full chain of custody

---

## Chapter 3: User Interfaces

### Design Principles
- **Design Thinking**: User-centered interface development
- **Transparency**: Clear provenance visualization
- **Accessibility**: Web, mobile, API, and email interfaces

### BTA User Interface Components
| Component | Purpose |
|-----------|---------|
| **Project Dashboard** | Overview of all AI projects |
| **Version Management** | Track model iterations |
| **Review Workflow** | Approval process interface |
| **Audit View** | Blockchain transaction history |
| **Verification Panel** | Hash comparison tools |

### Security Layers
- AI Security (model integrity)
- Database Security (storage protection)
- Blockchain Security (cryptographic verification)
- Additional Security (MFA, encryption)

---

## Chapter 4: Planning Your BTA

### BTA Architecture Overview
```
[AI Engineer] → [MLOps Engineer] → [Stakeholder]
     ↓               ↓                  ↓
 [Develop]       [Review]           [Approve]
     ↓               ↓                  ↓
     └───────── [Blockchain Ledger] ─────────┘
                     ↓
              [Oracle Cloud Bucket]
```

### User Roles and Permissions

| Role | Responsibilities |
|------|-----------------|
| **Super Admin** | System configuration, subscription approval |
| **Organization Admin** | Node/channel setup, user management |
| **AI Engineer** | Model development, experiment submission |
| **MLOps Engineer** | Model review, deployment, monitoring |
| **Stakeholder** | Purpose definition, final approval/decline |

### AI Factsheet Components
- Model name and version
- Purpose and intended domain
- Training/test data sources
- Hyperparameters
- Performance metrics
- Key contacts and ownership

### Blockchain Touchpoints
1. Project creation
2. Version submission
3. Data/model hash recording
4. Review status changes
5. Approval/decline decisions
6. Purpose documentation
7. Production deployment

---

## Chapter 5: Running Your Model

### Oracle Cloud Setup
1. Create compartment (BTA-staffings)
2. Create buckets for each staffing role
3. Configure pre-authenticated requests
4. Set up groups and policies
5. Generate secret keys

### Model Training Workflow
```bash
# Environment setup
pip install torch pytorch-lightning boto3

# Configure cloud storage
export OCI_BUCKET_NAME=ai-engineer-staffing
export OCI_NAMESPACE=your-namespace

# Train model
python train.py
```

### Key Hyperparameters
| Parameter | Description |
|-----------|-------------|
| **Learning Rate** | Controls gradient descent step size |
| **Epochs** | Number of training iterations |
| **Batch Size** | Samples per gradient update |
| **Hidden Layers** | Network architecture depth |

### Performance Metrics
- **Accuracy**: Correct predictions / total predictions
- **Loss**: Error measurement during training
- **Precision**: True positives / predicted positives
- **Recall**: True positives / actual positives
- **F1 Score**: Harmonic mean of precision and recall

---

## Chapter 6: Instantiating Your Blockchain

### Hyperledger Fabric 2.0 Setup

**Required Nodes:**
- Ordering Service (consensus)
- Peer Nodes (ledger maintenance)
- Certificate Authority (identity management)

**Installation Steps:**
```bash
# Install prerequisites
npm install -g @nestjs/cli

# Clone and configure Fabric
git clone hyperledger/fabric-samples
cd fabric-samples/test-network
./network.sh up createChannel

# Deploy chaincode
./network.sh deployCC -ccn basic -ccp ../asset-transfer-basic
```

### Channel Configuration
- **Global Channel**: Cross-organization communication
- **Private Channels**: Organization-specific data
- **Anchor Peers**: Channel gateway nodes

### Chaincode Components (Smart Contracts)

| Chaincode | Purpose |
|-----------|---------|
| **project** | Project metadata management |
| **model-version** | Version tracking |
| **model-review** | Review workflow |
| **model-artifact** | Asset hash storage |
| **model-experiment** | Experiment logging |

### Blockchain Connector Architecture
```
[BTA Frontend] ←→ [Blockchain Connector] ←→ [Hyperledger Fabric]
                         ↓
                  [Oracle Connector]
                         ↓
                  [OCI Object Storage]
```

---

## Chapter 7: Preparing Your BTA

### BTA Installation
```bash
# Backend setup
git clone bta-backend
cd bta-backend
npm install
cp .env.example .env
# Configure environment variables
npm run start:dev

# Frontend setup
git clone bta-frontend
cd bta-frontend
npm install
ng serve
```

### User Provisioning Workflow
1. Super Admin creates subscription
2. Organization Admin activates
3. Configure nodes and channels
4. Create staffings (AI Engineer, MLOps, Stakeholder)
5. Add users to staffings
6. Generate blockchain keys

### Organization Structure
```
Organization Unit
├── ai-engineer-staffing
│   └── ai-engineer-user
├── mlops-engineer-staffing
│   └── mlops-engineer-user
└── stakeholder-staffing
    └── stakeholder-user
```

---

## Chapter 8: Using Your BTA

### Model Lifecycle Status Flow

```
Draft → Pending → Review Passed/Failed → Deployed → QA → Production → Monitoring → Complete/Decline
```

### Recording AI Touchpoints

**Data Elements Recorded:**
| Element | Blockchain Storage |
|---------|-------------------|
| Training Data | Hash of dataset |
| Test Data | Hash of dataset |
| Model Binary | Hash of .pkl file |
| Hyperparameters | JSON in log hash |
| Performance Metrics | JSON in log hash |
| Purpose Statement | Direct storage |

### Hash Verification Process
```python
# Example: Verify model integrity
submitted_hash = get_hash_from_blockchain(model_id)
current_hash = compute_sha256(model_file)
if submitted_hash == current_hash:
    print("Model verified - no tampering")
else:
    print("WARNING: Model has been modified!")
```

### Reversing Tethered AI
When issues are detected:
1. Check training data hashes
2. Verify algorithm integrity
3. Review experiment logs
4. Rollback to verified version
5. Retrain with corrected data

---

## Key Tools & Technologies

| Category | Tools |
|----------|-------|
| **Blockchain** | Hyperledger Fabric 2.0 |
| **Cloud Storage** | Oracle Cloud Infrastructure (OCI) |
| **Backend** | Node.js, NestJS |
| **Frontend** | Angular |
| **ML Framework** | PyTorch, PyTorch Lightning |
| **MLOps Integration** | DVC, MLflow, DagsHub |
| **Identity** | Membership Service Providers (MSP) |
| **Hashing** | SHA-256 |

---

## Best Practices Summary

1. **Define Purpose Early**: Document intended domain and limitations before development
2. **Hash Everything Critical**: Training data, test data, models, logs, artifacts
3. **Separate Duties**: AI Engineer develops, MLOps reviews, Stakeholder approves
4. **Version Systematically**: Each experiment gets a unique version with full provenance
5. **Store Hashes Not Data**: Blockchain for verification, cloud storage for large files
6. **Automate Verification**: Pre-deployment hash checks should be mandatory
7. **Enable Rollback**: Maintain ability to revert to any previous verified state
8. **Governance First**: Establish approval workflows before model deployment

---

## Target Audience

- **System Architects**: Designing AI governance frameworks
- **Software Engineers**: Implementing BTA systems
- **MLOps Engineers**: Integrating blockchain with ML pipelines
- **AI Governance Officers**: Establishing AI accountability
- **Compliance Teams**: Audit and regulatory requirements
- **CISOs/Security**: AI security and integrity verification

---

## Key Takeaways for ML Engineers

1. **Provenance is Critical**: Every model change should be traceable
2. **Hashing Enables Verification**: SHA-256 proves data/model integrity
3. **Blockchain ≠ Storage**: Store hashes on-chain, data in cloud buckets
4. **Roles Enforce Accountability**: Clear separation of development, review, approval
5. **Smart Contracts Automate Governance**: Rules enforced by code, not trust
6. **Reversibility is Possible**: Full audit trail enables rollback to any verified state
7. **Integration with MLOps**: Works alongside DVC, MLflow, DagsHub
8. **NIST AI RMF Alignment**: Supports emerging AI governance standards
