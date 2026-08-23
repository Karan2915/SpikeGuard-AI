"""
SpikeGuard model training and evaluation.

Trains an XGBoost classifier on spike-level features to predict
organic vs anomalous, then picks an operating threshold using an
explicit cost function instead of the default 0.5 cutoff.

COST ASSUMPTIONS (state these plainly in your write-up, and feel free
to tune them):
    - FALSE POSITIVE cost: a legitimate sales spike gets wrongly flagged.
      Cost = manual review time for the merchant/support team.
      Assumed: Rs 150 per wrongly flagged spike.
    - FALSE NEGATIVE cost: a real return-abuse ring goes undetected.
      Cost = average order value x average orders per anomalous spike,
      i.e. the refund/loss exposure of one missed ring.
      Assumed: Rs 800 average order value (budget-fashion category,
      similar to the real Meesho case), x average ~80 orders per
      anomalous spike (from the synthetic data) = Rs 64,000 per missed spike.

These are stated assumptions, not measured facts -- that's honest and
expected, since no public dataset gives real merchant review costs.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.metrics import (
    precision_recall_curve, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report, average_precision_score
)

# resolves to project/Datasets and project/src regardless of machine/OS,
# as long as this script sits in project/src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "Datasets"
SRC_DIR = PROJECT_ROOT / "src"

# ---- cost assumptions (tune these) ----
FALSE_POSITIVE_COST = 150      # Rs, manual review cost per wrongly flagged organic spike
AVG_ORDER_VALUE = 800           # Rs, assumed average order value
AVG_ORDERS_PER_ANOMALOUS_SPIKE = 80  # from synthetic data generation stats
FALSE_NEGATIVE_COST = AVG_ORDER_VALUE * AVG_ORDERS_PER_ANOMALOUS_SPIKE  # Rs 64,000

FEATURE_COLS = [
    "n_orders", "duration_days", "ramp_rate_vs_baseline", "pct_new_buyers",
    "n_distinct_states", "geo_entropy", "mean_delivery_days", "std_delivery_days",
    "return_rate", "refund_claim_speed_days", "refund_claim_speed_std",
]


def load_data():
    train = pd.read_csv(DATA_DIR / "spikeguard_train.csv")
    test = pd.read_csv(DATA_DIR / "spikeguard_test.csv")
    y_train = (train["spike_type"] == "anomalous").astype(int)
    y_test = (test["spike_type"] == "anomalous").astype(int)
    X_train = train[FEATURE_COLS]
    X_test = test[FEATURE_COLS]
    return X_train, y_train, X_test, y_test


def train_model(X_train, y_train):
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / n_pos  # tells the model to weight the rare anomalous class more

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.08,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def find_cost_optimal_threshold(y_test, y_proba):
    """Sweep thresholds and pick the one that minimizes total expected cost,
    not the one that maximizes accuracy or F1."""
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)

    best_threshold = 0.5
    best_cost = float("inf")
    results = []

    for t in np.arange(0.05, 0.95, 0.01):
        preds = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
        total_cost = fp * FALSE_POSITIVE_COST + fn * FALSE_NEGATIVE_COST
        results.append((t, fp, fn, tp, tn, total_cost))
        if total_cost < best_cost:
            best_cost = total_cost
            best_threshold = t

    results_df = pd.DataFrame(results, columns=["threshold", "fp", "fn", "tp", "tn", "total_cost"])
    return best_threshold, best_cost, results_df


def main():
    X_train, y_train, X_test, y_test = load_data()
    model = train_model(X_train, y_train)

    y_proba = model.predict_proba(X_test)[:, 1]

    print("=== Default threshold (0.5) ===")
    preds_default = (y_proba >= 0.5).astype(int)
    print(classification_report(y_test, preds_default, target_names=["organic", "anomalous"]))
    print("PR-AUC:", round(average_precision_score(y_test, y_proba), 4))

    best_t, best_cost, results_df = find_cost_optimal_threshold(y_test, y_proba)
    print(f"\n=== Cost-optimal threshold: {best_t:.2f} ===")
    preds_best = (y_proba >= best_t).astype(int)
    print(classification_report(y_test, preds_best, target_names=["organic", "anomalous"]))

    tn, fp, fn, tp = confusion_matrix(y_test, preds_best).ravel()
    default_tn, default_fp, default_fn, default_tp = confusion_matrix(y_test, preds_default).ravel()
    default_cost = default_fp * FALSE_POSITIVE_COST + default_fn * FALSE_NEGATIVE_COST

    print(f"\nCost assumptions: FP = Rs {FALSE_POSITIVE_COST}, FN = Rs {FALSE_NEGATIVE_COST}")
    print(f"Total cost at default 0.5 threshold: Rs {default_cost:,} ({default_fp} FP, {default_fn} FN)")
    print(f"Total cost at optimal {best_t:.2f} threshold: Rs {best_cost:,.0f} ({fp} FP, {fn} FN)")
    print(f"Savings from cost-aware thresholding: Rs {default_cost - best_cost:,.0f}")

    joblib.dump(model, SRC_DIR / "spikeguard_model.joblib")
    joblib.dump(
        {"threshold": best_t, "feature_cols": FEATURE_COLS,
         "fp_cost": FALSE_POSITIVE_COST, "fn_cost": FALSE_NEGATIVE_COST},
        SRC_DIR / "spikeguard_config.joblib",
    )
    results_df.to_csv(SRC_DIR / "threshold_sweep.csv", index=False)
    print("\nSaved: spikeguard_model.joblib, spikeguard_config.joblib, threshold_sweep.csv")


if __name__ == "__main__":
    main()