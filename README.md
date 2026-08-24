# 🛡️ SpikeGuard AI

## Cost-Aware E-Commerce Spike Anomaly Detection with Explainable AI

SpikeGuard AI is an explainable machine-learning system designed to detect
suspicious transaction-volume spikes in e-commerce merchant activity.

The system analyzes behavioral characteristics of an unusual increase in
orders and estimates whether the spike is more consistent with:

- 🟢 **Organic activity** — a legitimate sales surge
- 🔴 **Anomalous activity** — a suspicious behavioral pattern resembling
  organized return/refund abuse

SpikeGuard is designed as a **decision-support system** rather than a system
that automatically declares a merchant, customer, or transaction fraudulent.

The project combines:

- Synthetic behavioral data generation
- Feature engineering
- Cost-sensitive machine learning
- XGBoost classification
- Validation-based threshold optimization
- Cost-sensitive decision making
- Held-out test evaluation
- SHAP-based explainability
- Interactive Streamlit dashboard

The central idea behind the project is:

> Detect suspicious spikes while explicitly considering the business cost of
> false positives and false negatives.

---

# 📌 Table of Contents

- [1. Project Overview](#1-project-overview)
- [2. Problem Statement](#2-problem-statement)
- [3. Motivation](#3-motivation)
- [4. Project Objective](#4-project-objective)
- [5. Key Features](#5-key-features)
- [6. How the System Works](#6-how-the-system-works)
- [7. System Architecture](#7-system-architecture)
- [8. End-to-End Pipeline](#8-end-to-end-pipeline)
- [9. Dataset](#9-dataset)
- [10. Synthetic Data Generation](#10-synthetic-data-generation)
- [11. Feature Engineering](#11-feature-engineering)
- [12. Feature Description](#12-feature-description)
- [13. Data Splitting Strategy](#13-data-splitting-strategy)
- [14. Class Distribution](#14-class-distribution)
- [15. Class Imbalance Handling](#15-class-imbalance-handling)
- [16. Machine Learning Model](#16-machine-learning-model)
- [17. Why XGBoost](#17-why-xgboost)
- [18. Cost-Sensitive Learning](#18-cost-sensitive-learning)
- [19. Decision Threshold](#19-decision-threshold)
- [20. Threshold Optimization](#20-threshold-optimization)
- [21. Cost Function](#21-cost-function)
- [22. Threshold Locking](#22-threshold-locking)
- [23. Final Model Evaluation](#23-final-model-evaluation)
- [24. Confusion Matrix](#24-confusion-matrix)
- [25. Evaluation Metrics](#25-evaluation-metrics)
- [26. SHAP Explainability](#26-shap-explainability)
- [27. Streamlit Dashboard](#27-streamlit-dashboard)
- [28. Dashboard — Test Set Explorer](#28-dashboard--test-set-explorer)
- [29. Dashboard — Manual Spike Checker](#29-dashboard--manual-spike-checker)
- [30. Dashboard — About Section](#30-dashboard--about-section)
- [31. Example Organic Spike](#31-example-organic-spike)
- [32. Example Anomalous Spike](#32-example-anomalous-spike)
- [33. Project Structure](#33-project-structure)
- [34. File Responsibilities](#34-file-responsibilities)
- [35. Installation](#35-installation)
- [36. Virtual Environment](#36-virtual-environment)
- [37. Requirements](#37-requirements)
- [38. Running the Project](#38-running-the-project)
- [39. Generating Data](#39-generating-data)
- [40. Training the Model](#40-training-the-model)
- [41. Running Explainability](#41-running-explainability)
- [42. Running the Dashboard](#42-running-the-dashboard)
- [43. Model Artifacts](#43-model-artifacts)
- [44. Configuration](#44-configuration)
- [45. Reproducibility](#45-reproducibility)
- [46. Design Decisions](#46-design-decisions)
- [47. Why Accuracy Alone Is Not Enough](#47-why-accuracy-alone-is-not-enough)
- [48. Why Recall Matters](#48-why-recall-matters)
- [49. Why PR-AUC Is Included](#49-why-pr-auc-is-included)
- [50. Why the Threshold Is Low](#50-why-the-threshold-is-low)
- [51. Interpretation of Predictions](#51-interpretation-of-predictions)
- [52. Explainability vs Causality](#52-explainability-vs-causality)
- [53. Limitations](#53-limitations)
- [54. Responsible Use](#54-responsible-use)
- [55. Security and Privacy Considerations](#55-security-and-privacy-considerations)
- [56. Future Scope](#56-future-scope)
- [57. Potential Production Architecture](#57-potential-production-architecture)
- [58. Possible Improvements](#58-possible-improvements)
- [59. Technology Stack](#59-technology-stack)
- [60. Summary](#60-summary)
- [61. Author](#61-author)
- [62. License](#62-license)

---

# 1. Project Overview

SpikeGuard AI is a machine-learning prototype for identifying unusual
e-commerce transaction-volume spikes.

In an e-commerce environment, a sudden increase in order volume can have
multiple explanations.

For example:

A merchant may launch a successful promotional campaign, resulting in a
large but legitimate increase in orders.

On the other hand, suspicious activity may also produce a sudden increase in
orders followed by unusual delivery, return, cancellation, or refund behavior.

Therefore, simply detecting an increase in order volume is not sufficient.

SpikeGuard instead examines a combination of behavioral signals.

The model produces a probability-like risk score representing the estimated
likelihood of the anomalous class.

The risk score is then compared against a decision threshold.

If:

    risk_score >= threshold

the spike is flagged as anomalous.

Otherwise:

    risk_score < threshold

the spike is considered not flagged / organic.

The threshold is not arbitrarily set to 0.50.

Instead, it is selected using a cost-based optimization process on the
validation dataset.

---

# 2. Problem Statement

E-commerce platforms need to distinguish between legitimate sales spikes and
potentially suspicious spikes.

A legitimate spike can occur because of:

- Promotional campaigns
- Discounts
- Seasonal events
- Product launches
- Marketing campaigns
- Social media exposure
- Festival sales
- Influencer campaigns
- Other legitimate demand changes

However, suspicious activity can also generate unusual transaction patterns.

Examples of suspicious behavioral signals may include:

- Extremely rapid order growth
- High concentration of new buyers
- Unusual geographic distribution
- High return or cancellation activity
- Unusual delivery-time behavior
- Unusual refund timing
- Combinations of several abnormal behaviors

The challenge is therefore:

> How can a machine-learning system identify suspicious transaction spikes
> without incorrectly flagging too many legitimate business events?

This is a classification problem with an important business-cost component.

---

# 3. Motivation

A standard binary classifier often uses a threshold of:

    0.50

However, the default threshold does not necessarily represent the optimal
business decision.

Consider two types of errors:

### False Positive

A legitimate sales spike is classified as anomalous.

Potential consequences:

- Merchant investigation
- Operational overhead
- Delayed business activity
- Customer experience impact
- Loss of trust

### False Negative

An anomalous spike is classified as organic.

Potential consequences:

- Suspicious activity is missed
- Potential financial loss
- Additional fraudulent orders
- Additional return/refund exposure
- Increased investigation difficulty later

These errors can have different costs.

SpikeGuard therefore treats threshold selection as a **cost optimization
problem** rather than simply selecting the threshold that gives the highest
accuracy.

---

# 4. Project Objective

The primary objectives of SpikeGuard are:

1. Generate behavioral transaction-spike data.
2. Represent both organic and anomalous patterns.
3. Train a machine-learning classifier.
4. Handle class imbalance.
5. Optimize the classification threshold using a business-cost function.
6. Lock the threshold using validation data.
7. Evaluate the final system on unseen held-out test data.
8. Provide interpretable predictions.
9. Provide an interactive interface for merchants or analysts.
10. Explain why a spike was flagged.

---

# 5. Key Features

## 5.1 Spike Classification

The model classifies spikes into:

```text
Organic
Anomalous