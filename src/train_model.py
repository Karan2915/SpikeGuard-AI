# 🛡️ SpikeGuard

### Merchant-Side Order Spike Risk Detection using Machine Learning

SpikeGuard is a machine-learning based risk assessment system designed to help merchants identify whether a sudden increase in order activity is likely to be an **organic sales spike** or a potentially **anomalous order pattern associated with return/refund abuse**.

The project combines:

- Cost-sensitive machine learning
- Feature engineering
- XGBoost classification
- Business-cost-based threshold optimization
- SHAP explainability
- Held-out test evaluation
- Interactive Streamlit dashboard
- Manual risk assessment
- Model deployment

---

## 🚀 Live Demo

**Streamlit Application:**

https://spikeguard-ai.streamlit.app

The deployed application allows users to:

- Browse spikes from the held-out test dataset
- Filter flagged and non-flagged spikes
- Inspect individual predictions
- Enter custom spike characteristics
- Receive a risk score
- View the model's decision threshold
- See whether a spike is flagged
- View SHAP-based feature explanations

---

# 📌 Problem Statement

Sudden increases in order volume are not necessarily fraudulent.

A merchant may experience a legitimate spike because of:

- A sale
- A product launch
- Seasonal demand
- Marketing campaigns
- Influencer promotions
- Festival periods
- Other genuine business events

However, abnormal spikes can also be associated with coordinated abuse.

For example, a suspicious spike may involve:

- An unusually large increase in order volume
- A high proportion of first-time buyers
- Unusual geographic distribution
- High return or cancellation rates
- Unusual refund behavior
- Highly variable delivery patterns
- Very fast refund claims

A simple rule such as:

> "If orders increase by more than X%, flag the merchant"

would generate many false positives.

SpikeGuard instead combines multiple behavioral signals and produces a probability-based risk score.

---

# 🎯 Project Objective

The objective of SpikeGuard is to build a prototype merchant-side risk engine that answers:

> **"Does this order spike look more like a normal sales event or an anomalous pattern requiring investigation?"**

The system produces:

1. A risk probability
2. A binary decision
3. An explanation of the factors influencing the prediction

---

# 🧠 How SpikeGuard Works

The overall pipeline is:

```text
Merchant / Test Data
        │
        ▼
Feature Engineering
        │
        ▼
Data Preparation
        │
        ▼
Cost-Sensitive XGBoost Model
        │
        ▼
Validation Predictions
        │
        ▼
Cost-Based Threshold Optimization
        │
        ▼
Locked Decision Threshold
        │
        ▼
Held-Out Test Evaluation
        │
        ▼
Risk Score + Decision
        │
        ▼
SHAP Explanation
        │
        ▼
Streamlit Dashboard
