"""
SpikeGuard AI - Model Training and Evaluation

Pipeline:

    Train data
        ↓
    Train / Validation split
        ↓
    Train XGBoost
        ↓
    Select cost-aware threshold on validation set
        ↓
    Lock threshold
        ↓
    Evaluate once on untouched test set
        ↓
    Save model + configuration

The test set is never used for threshold selection.

COST ASSUMPTIONS
----------------
FALSE POSITIVE:
A legitimate sales spike is incorrectly flagged.
Assumed cost = Rs 150 per wrongly flagged spike.

FALSE NEGATIVE:
An anomalous spike is missed.
Assumed cost = Rs 64,000 per missed anomalous spike.

These are prototype assumptions, not measured Razorpay costs.
"""

import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    average_precision_score,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "Datasets"
SRC_DIR = PROJECT_ROOT / "src"


# ============================================================
# COST ASSUMPTIONS
# ============================================================

FALSE_POSITIVE_COST = 150

AVG_ORDER_VALUE = 800

AVG_ORDERS_PER_ANOMALOUS_SPIKE = 80

FALSE_NEGATIVE_COST = (
    AVG_ORDER_VALUE * AVG_ORDERS_PER_ANOMALOUS_SPIKE
)


# ============================================================
# FEATURES
# ============================================================

FEATURE_COLS = [
    "n_orders",
    "duration_days",
    "ramp_rate_vs_baseline",
    "pct_new_buyers",
    "n_distinct_states",
    "geo_entropy",
    "mean_delivery_days",
    "std_delivery_days",
    "return_rate",
    "refund_claim_speed_days",
    "refund_claim_speed_std",
]


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    train = pd.read_csv(
        DATA_DIR / "spikeguard_train.csv"
    )

    test = pd.read_csv(
        DATA_DIR / "spikeguard_test.csv"
    )

    y_train = (
        train["spike_type"] == "anomalous"
    ).astype(int)

    y_test = (
        test["spike_type"] == "anomalous"
    ).astype(int)

    X_train = train[FEATURE_COLS]

    X_test = test[FEATURE_COLS]

    return X_train, y_train, X_test, y_test


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

def create_validation_split(X_train, y_train):

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.20,
        stratify=y_train,
        random_state=42,
    )

    return X_tr, X_val, y_tr, y_val


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(X_train, y_train):

    n_pos = y_train.sum()

    n_neg = len(y_train) - n_pos

    scale_pos_weight = n_neg / n_pos

    print("\nClass distribution used for training:")
    print(f"Organic   : {n_neg}")
    print(f"Anomalous : {n_pos}")

    print(
        f"Scale positive weight: "
        f"{scale_pos_weight:.3f}"
    )

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.08,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=42,
    )

    model.fit(
        X_train,
        y_train
    )

    return model


# ============================================================
# COST FUNCTION
# ============================================================

def calculate_cost(y_true, y_pred):

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    total_cost = (
        fp * FALSE_POSITIVE_COST
        +
        fn * FALSE_NEGATIVE_COST
    )

    return (
        total_cost,
        tn,
        fp,
        fn,
        tp
    )


# ============================================================
# FIND COST-OPTIMAL THRESHOLD
# ============================================================

def find_cost_optimal_threshold(y_validation, validation_probabilities):
    """
    Select a threshold using validation data only.

    Business rule:
    1. Precision must be at least 0.70.
    2. Among thresholds satisfying that constraint,
       choose the one with the lowest expected cost.

    This prevents the model from achieving low cost simply by
    flagging almost everything as anomalous.
    """

    MIN_PRECISION = 0.70

    best_threshold = 0.50
    best_cost = float("inf")
    results = []

    for threshold in np.arange(0.01, 0.96, 0.01):

        predictions = (
            validation_probabilities >= threshold
        ).astype(int)

        cost, tn, fp, fn, tp = calculate_cost(
            y_validation,
            predictions
        )

        precision = (
            tp / (tp + fp)
            if (tp + fp) > 0
            else 0
        )

        recall = (
            tp / (tp + fn)
            if (tp + fn) > 0
            else 0
        )

        results.append(
            (
                threshold,
                precision,
                recall,
                fp,
                fn,
                tp,
                tn,
                cost
            )
        )

        # Only consider thresholds that meet the
        # minimum precision requirement
        if precision >= MIN_PRECISION and cost < best_cost:
            best_cost = cost
            best_threshold = threshold

    results_df = pd.DataFrame(
        results,
        columns=[
            "threshold",
            "precision",
            "recall",
            "fp",
            "fn",
            "tp",
            "tn",
            "total_cost"
        ]
    )

    return best_threshold, best_cost, results_df


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    y_test,
    test_probabilities,
    threshold
):

    predictions = (
        test_probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    pr_auc = average_precision_score(
        y_test,
        test_probabilities
    )

    cost, tn, fp, fn, tp = calculate_cost(
        y_test,
        predictions
    )

    false_positive_rate = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0
    )

    false_negative_rate = (
        fn / (fn + tp)
        if (fn + tp) > 0
        else 0
    )

    print("\n")
    print("=" * 60)
    print("SPIKEGUARD FINAL TEST EVALUATION")
    print("=" * 60)

    print(
        f"\nDecision Threshold : {threshold:.2f}"
    )

    print(
        f"\nPrecision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1 Score  : {f1:.4f}"
    )

    print(
        f"PR-AUC    : {pr_auc:.4f}"
    )

    print("\nConfusion Matrix:")

    print(
        f"TN = {tn}"
    )

    print(
        f"FP = {fp}"
    )

    print(
        f"FN = {fn}"
    )

    print(
        f"TP = {tp}"
    )

    print(
        f"\nFalse Positive Rate : "
        f"{false_positive_rate:.4f}"
    )

    print(
        f"False Negative Rate : "
        f"{false_negative_rate:.4f}"
    )

    print(
        f"\nExpected Test Cost : "
        f"Rs {cost:,.0f}"
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "organic",
                "anomalous"
            ],
            zero_division=0
        )
    )

    print("=" * 60)

    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": pr_auc,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
        "test_cost": cost,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("\nLoading SpikeGuard dataset...")

    X_train_full, y_train_full, X_test, y_test = (
        load_data()
    )

    print(
        f"Training dataset: "
        f"{len(X_train_full)} spikes"
    )

    print(
        f"Test dataset: "
        f"{len(X_test)} spikes"
    )


    # --------------------------------------------------------
    # TRAIN / VALIDATION SPLIT
    # --------------------------------------------------------

    (
        X_train,
        X_validation,
        y_train,
        y_validation
    ) = create_validation_split(
        X_train_full,
        y_train_full
    )

    print(
        f"\nActual training samples: "
        f"{len(X_train)}"
    )

    print(
        f"Validation samples: "
        f"{len(X_validation)}"
    )

    print(
        f"Final held-out test samples: "
        f"{len(X_test)}"
    )


    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model = train_model(
        X_train,
        y_train
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    validation_probabilities = (
        model.predict_proba(
            X_validation
        )[:, 1]
    )


    # --------------------------------------------------------
    # THRESHOLD SELECTION
    # --------------------------------------------------------

    (
        best_threshold,
        validation_cost,
        threshold_results
    ) = find_cost_optimal_threshold(
        y_validation,
        validation_probabilities
    )

    print("\n")
    print("=" * 60)
    print("VALIDATION THRESHOLD SELECTION")
    print("=" * 60)

    print(
        f"\nCost-optimal threshold: "
        f"{best_threshold:.2f}"
    )

    print(
        f"Validation cost: "
        f"Rs {validation_cost:,.0f}"
    )

    print(
        "\nThe threshold is now LOCKED."
    )

    print(
        "The test set will not be used to "
        "change it."
    )


    # --------------------------------------------------------
    # FINAL TEST EVALUATION
    # --------------------------------------------------------

    test_probabilities = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    metrics = evaluate_model(
        y_test,
        test_probabilities,
        best_threshold
    )


    # --------------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------------

    model_path = (
        SRC_DIR /
        "spikeguard_model.joblib"
    )

    config_path = (
        SRC_DIR /
        "spikeguard_config.joblib"
    )

    threshold_path = (
        SRC_DIR /
        "threshold_sweep.csv"
    )


    joblib.dump(
        model,
        model_path
    )


    joblib.dump(
        {
            "threshold": best_threshold,
            "feature_cols": FEATURE_COLS,
            "fp_cost": FALSE_POSITIVE_COST,
            "fn_cost": FALSE_NEGATIVE_COST,
            "metrics": metrics,
        },
        config_path
    )


    threshold_results.to_csv(
        threshold_path,
        index=False
    )


    print("\nSaved files:")

    print(
        f"✓ {model_path}"
    )

    print(
        f"✓ {config_path}"
    )

    print(
        f"✓ {threshold_path}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()