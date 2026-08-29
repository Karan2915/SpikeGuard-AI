"""
SpikeGuard early-warning model.

The main model (train_model.py) uses return_rate, refund_claim_speed_days,
mean_delivery_days, etc. -- these are LAGGING signals: they only exist
after an order has been delivered and, in the anomalous case, already
refunded. By the time that model can score a spike with full confidence,
the money for those orders is already gone.

This script trains a second model using ONLY features available at
order-placement time, before any delivery or refund has happened:
    n_orders, duration_days, ramp_rate_vs_baseline,
    pct_new_buyers, n_distinct_states, geo_entropy

This "Early Warning Score" trades accuracy for timing -- it works with
strictly less information than the full model, so it will generally show
lower precision and/or lower overall separability (PR-AUC). Exactly how
much worse depends on how separable the leading signals are in your
specific data; the comparison printed below is measured on your actual
dataset, not assumed. That gap IS the honest cost of acting sooner: a
merchant using only the Early Warning score accepts more false alarms
(or a coarser risk read) in exchange for being able to flag a pattern
before any refund has been paid out, rather than only after.

In the real Meesho case, the fraud ran for 7 months and 2,500 orders
before being caught. An early-warning score firing on the leading signals
alone -- rapid ramp, narrow buyer clustering, high new-buyer share -- could
flag that seller relationship for review within days of the pattern
starting, long before the full 7-month loss accumulates.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.metrics import (
    precision_recall_curve, confusion_matrix, classification_report, average_precision_score
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "Datasets"
SRC_DIR = PROJECT_ROOT / "src"

# same cost assumptions as the main model, for a like-for-like comparison
FALSE_POSITIVE_COST = 150
FALSE_NEGATIVE_COST = 64000

EARLY_FEATURE_COLS = [
    "n_orders", "duration_days", "ramp_rate_vs_baseline",
    "pct_new_buyers", "n_distinct_states", "geo_entropy",
]

FULL_FEATURE_COLS = EARLY_FEATURE_COLS + [
    "mean_delivery_days", "std_delivery_days",
    "return_rate", "refund_claim_speed_days", "refund_claim_speed_std",
]


def load_data():
    train = pd.read_csv(DATA_DIR / "spikeguard_train.csv")
    test = pd.read_csv(DATA_DIR / "spikeguard_test.csv")
    y_train = (train["spike_type"] == "anomalous").astype(int)
    y_test = (test["spike_type"] == "anomalous").astype(int)
    return train, test, y_train, y_test


def train_and_eval(X_train, y_train, X_test, y_test, label):
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / n_pos

    model = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.08,
        scale_pos_weight=scale_pos_weight, eval_metric="aucpr", random_state=42,
    )
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)[:, 1]
    pr_auc = average_precision_score(y_test, y_proba)

    # cost-optimal threshold, same logic as the main model
    best_t, best_cost = 0.5, float("inf")
    for t in np.arange(0.05, 0.95, 0.01):
        preds = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
        cost = fp * FALSE_POSITIVE_COST + fn * FALSE_NEGATIVE_COST
        if cost < best_cost:
            best_cost, best_t = cost, t

    preds_best = (y_proba >= best_t).astype(int)
    print(f"\n{'='*60}\n{label}\n{'='*60}")
    print(f"PR-AUC: {pr_auc:.4f}")
    print(f"Cost-optimal threshold: {best_t:.2f}, total cost: Rs {best_cost:,.0f}")
    print(classification_report(y_test, preds_best, target_names=["organic", "anomalous"]))

    return model, best_t, pr_auc, best_cost


def main():
    train, test, y_train, y_test = load_data()

    # train both models on the SAME data split, differing only in which
    # columns they're allowed to see -- this isolates the effect of
    # early vs. full information cleanly
    early_model, early_t, early_auc, early_cost = train_and_eval(
        train[EARLY_FEATURE_COLS], y_train, test[EARLY_FEATURE_COLS], y_test,
        "EARLY WARNING MODEL (leading features only -- available at order-placement time)"
    )
    full_model, full_t, full_auc, full_cost = train_and_eval(
        train[FULL_FEATURE_COLS], y_train, test[FULL_FEATURE_COLS], y_test,
        "CONFIRMED RISK MODEL (full features -- available after delivery/return cycle)"
    )

    print(f"\n{'='*60}\nCOMPARISON\n{'='*60}")
    print(f"Early Warning  -- PR-AUC: {early_auc:.4f}, cost: Rs {early_cost:,.0f}")
    print(f"Confirmed Risk -- PR-AUC: {full_auc:.4f}, cost: Rs {full_cost:,.0f}")
    print(f"\nThe gap between these two numbers IS the honest cost of acting early:")
    print(f"less accuracy, in exchange for being able to act before losses accumulate,")
    print(f"instead of only after the refund cycle has already completed.")

    joblib.dump(early_model, SRC_DIR / "spikeguard_early_model.joblib")
    joblib.dump(
        {"threshold": early_t, "feature_cols": EARLY_FEATURE_COLS, "pr_auc": early_auc},
        SRC_DIR / "spikeguard_early_config.joblib",
    )
    print("\nSaved: spikeguard_early_model.joblib, spikeguard_early_config.joblib")


if __name__ == "__main__":
    main()