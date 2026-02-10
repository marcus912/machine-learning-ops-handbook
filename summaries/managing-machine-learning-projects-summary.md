# Managing Machine Learning Projects - Comprehensive Summary

**Author:** Simon Thompson
**Publisher:** Manning Publications, 2023
**Pages:** 273
**Focus:** End-to-end ML project management from design through deployment and production maintenance

---

## Core Definition

> "Delivering machine learning projects is hard; let's do it better."

This book provides a structured, sprint-based methodology for managing ML projects that bridges the gap between data science experimentation and production-ready systems.

---

## Book Structure Overview

The book follows a sprint-based project lifecycle:

| Phase | Focus Areas |
|-------|-------------|
| **Pre-Project** | Understand, Estimate, Plan |
| **Sprint 0** | Onboard, Organize, Set-up |
| **Sprint 1** | Infrastructure, Analysis, Data |
| **Sprint 2** | Modelling, Evaluation |
| **Sprint 3** | Integration, Production |
| **In-life (Sprint Ω)** | Governance, Management |

---

## Section 1: Pre-Project Phase (Chapters 2-3)

### Chapter 2: From Opportunity to Requirements

**Key Activities:**
- Set up **project management infrastructure** from day one
- Create and maintain a **risk register** - document unknowns, track mitigation
- Define **funding model** and business requirements
- Assess **data availability** and quality
- Address **security, privacy, and ethical considerations**
- Plan **development and production architecture**

**Risk Management Approach:**
- Turn risks into questions to be explored
- Expose uncertainty before establishing business value
- Weekly risk review meetings with stakeholders

### Chapter 3: From Requirements to Proposal

**Building Project Estimates:**
- **Time and effort estimates** - realistic scoping
- **Team design for ML projects** - required roles and skills
- **Project risks** - technical, data, organizational
- Pre-sales/pre-project checklist validation

**Team Composition Considerations:**
- Data Scientists, ML Engineers, Data Engineers
- DevOps/MLOps specialists
- Domain experts / SMEs
- Project management

---

## Section 2: Sprint 0 - Getting Started (Chapter 4)

### Key Deliverables:
1. **Finalize team design and resourcing**
2. **Establish way of working**
   - Process and structure
   - Heartbeat and communication plan
   - Tooling selection
   - Standards and practices
   - Documentation requirements

3. **Infrastructure plan**
   - System access
   - Technical infrastructure evaluation

4. **The Data Story** - Critical documentation including:
   - Data collection motivation
   - Data collection mechanism
   - Data lineage
   - Event tracking

5. **Privacy, security, and ethics plan**
6. **Project roadmap**

---

## Section 3: Sprint 1 - Diving into the Problem (Chapters 5-6)

### Chapter 5: Understanding the Data

**Data Survey Process:**
- **Numerical data** - distributions, ranges, outliers
- **Categorical data** - cardinality, missing values
- **Unstructured data** - text, images, special handling

**Building Data Pipelines:**
- Address **data fusion challenges**
- Avoid **pipeline jungles** (overly complex interconnections)
- Implement **data testing** at each stage

**Model Repository Setup:**
- Feature tracking
- Foundational models
- Training regime documentation
- Version control for models

### Chapter 6: EDA, Ethics, and Baseline Evaluations

**Exploratory Data Analysis (EDA) Objectives:**
- Summarizing and describing data
- Plots and visualizations
- Unstructured data analysis

**Ethics Checkpoint:**
- Formal review point before modeling begins
- Bias assessment
- Fairness considerations

**Baseline Models:**
- Establish performance benchmarks
- Simple models first for comparison

---

## Section 4: Sprint 2 - Modelling (Chapters 7-8)

### Chapter 7: Making Useful Models with ML

**Feature Engineering:**
- Domain-driven feature creation
- Feature selection and validation
- Documentation of all features

**Model Design Considerations:**
- Match model complexity to requirements
- Consider interpretability needs
- Plan for production constraints

### Chapter 8: Testing and Selection

**Quantitative Selection Methods:**
- **Precision and recall** over simple accuracy
- **F1 score** for balanced assessment
- **Multi-criteria decision making (MCDM)**
- **Pareto optimality** - models that excel on at least one dimension
- **Ranking aggregation** - combining multiple test results

**Qualitative Selection Criteria:**
| Criterion | Description |
|-----------|-------------|
| **Model Security** | Resistance to adversarial attacks, manipulation |
| **Privacy** | Prevention of information leakage, PII exposure |
| **Fairness** | Absence of harmful biases from training data |
| **Interpretability** | Ability to explain model decisions to humans |

**Occam's Razor Principle:**
> "When in doubt, simple and sweet wins out."

Prefer lower parameter counts and simpler architectures when performance is equivalent.

**Sprint 2 Checklist Items:**
- S2.1: Feature engineering implemented and documented
- S2.2: Model designs documented
- S2.3: Models developed with full reproducibility
- S2.4: Performance properly assessed and recorded
- S2.5: Model issues discovered are documented
- S2.6: Test environment commissioned
- S2.7: Appropriate tests designed
- S2.8: Test data gathered
- S2.9: Model selection documented

---

## Section 5: Sprint 3 - System Building and Production (Chapter 9)

### Integration Activities:
- **ML System Integration** with existing infrastructure
- **Productionization** of selected models
- **Model serving** architecture
- **Testing** in production-like environments

### Production Requirements:
- Logging and monitoring setup
- Performance baselines
- Rollback procedures
- API design and documentation

---

## Section 6: Post-Project / Sprint Ω (Chapter 10)

### Governance and Management

**Model Drift Monitoring:**
- Track prediction accuracy over time
- Detect data distribution changes
- Automated alerts for performance degradation

**Team Post-Project Review:**
- Structured feedback collection
- Lessons learned documentation
- Process improvement identification

**Review Template Questions:**
- Was project purpose and direction clear?
- Were resources sufficient?
- Was timeline realistic?
- How well did the team work together?

**Continuous Improvement:**
- Document what worked well
- Identify areas for improvement
- Update organizational playbooks

---

## Key Tools & Technologies

| Category | Tools/Approaches |
|----------|------------------|
| **Project Management** | Risk registers, sprint backlogs, checklists |
| **Data Pipeline** | Data warehouses, Python scripting, SQL |
| **Model Training** | BERT, autoencoders, time series models |
| **Testing** | Cross-validation, A/B testing, multi-armed bandits |
| **Documentation** | Model repository, version control, metadata tracking |
| **Monitoring** | Performance dashboards, drift detection |

---

## Best Practices Summary

1. **Risk-First Planning**: Establish a risk register from day one and manage risks throughout the project lifecycle

2. **Data Story Documentation**: Document data lineage, collection mechanisms, and motivations before modeling

3. **Ethics Checkpoints**: Build formal ethics reviews into the process, not as an afterthought

4. **Baseline First**: Always establish simple baseline models before complex approaches

5. **Multi-Criteria Selection**: Never rely on a single metric for model selection; use MCDM or Pareto approaches

6. **Reproducibility**: Maintain complete documentation and versioning for all models and experiments

7. **Test Environment Parity**: Test environment should mirror production security and privacy requirements

8. **Iterative Validation**: Expect to revisit earlier sprints; the process is iterative, not linear

9. **Team Communication**: Establish "heartbeat" meetings and clear communication channels

10. **Post-Project Reviews**: Conduct structured retrospectives to improve future projects

---

## Case Study: The Bike Shop

The book uses a running case study throughout - "The Bike Shop" - demonstrating:

- **Customer churn prediction** system
- **Demand forecasting** system
- Integration of **news sentiment** and **economic data**
- Multi-region model deployment
- Feature engineering for time series (EMA, SMA, Bollinger Bands, RSI, etc.)
- Model selection across different market sizes

Key lesson: News sentiment models performed better in large countries with well-funded media; this insight led to selecting different models per country based on media environment.

---

## Target Audience

- **ML/AI Project Managers** - Primary audience
- **Data Science Team Leads** - Sprint planning and execution
- **ML Engineers** - Production deployment considerations
- **Technical Program Managers** - Stakeholder management
- **Data Scientists** - Understanding project lifecycle beyond modeling
- **Engineering Managers** - Resource planning and team design
