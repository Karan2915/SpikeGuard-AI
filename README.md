# 🛡️ SpikeGuard AI

### E-commerce Spike Anomaly Detection & Explainability System

SpikeGuard AI is a machine-learning based system designed to identify suspicious
transaction-volume spikes in e-commerce merchant activity.

The system classifies a spike as either:

- 🟢 **Organic** — a legitimate sales surge
- 🔴 **Anomalous** — a suspicious spike pattern that may resemble organized
  return/refund abuse

Unlike a simple binary classifier, SpikeGuard also provides a **risk score,
cost-aware decision threshold, and SHAP-based feature explanation** for each
prediction.

---

## 🚀 Project Overview

Sudden increases in order volume are not always genuine sales events.

A merchant may experience a legitimate promotion-driven spike, but suspicious
activity can also produce unusual combinations of:

- Order volume
- Volume growth relative to baseline
- New buyer concentration
- Geographic distribution
- Delivery behavior
- Return/cancellation behavior
- Refund-claim timing

SpikeGuard uses these behavioral signals to estimate the probability that a
spike is anomalous.

The project focuses on **decision support and explainability**, rather than
claiming that model predictions are causal proof of fraud.

---

## 🏗️ System Architecture

```text
                    Synthetic Spike Data
                           │
                           ▼
                  Feature Generation
                           │
                           ▼
                    Train / Test Split
                           │
                           ▼
                 Cost-Sensitive XGBoost
                           │
                           ▼
                  Validation Threshold
                       Selection
                           │
                           ▼
                 Locked Threshold = 0.06
                           │
                           ▼
                  Held-Out Test Set
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
       Risk Prediction              SHAP Analysis
             │                           │
             ▼                           ▼
     Flagged / Organic          Feature Attribution
             │
             └─────────────┬─────────────┘
                           ▼
                  Streamlit Dashboard
