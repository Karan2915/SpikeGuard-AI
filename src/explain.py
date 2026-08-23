"""
SpikeGuard explanation layer.

Loads the trained model and generates a per-spike explanation for why it
was flagged anomalous (or not) -- top contributing features and their
direction of effect, using SHAP values.

This is the "root cause" feature from the project pitch, done honestly:
it's feature attribution on the ONE model actually trained and evaluated,
not a separate causal system.
"""

import pandas as pd
import numpy as np
import joblib
import shap
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "Datasets"
SRC_DIR = PROJECT_ROOT / "src"

# human-readable labels for each feature, used when building explanation text
FEATURE_LABELS = {
    "n_orders": "total orders in the spike",
    "duration_days": "spike duration",
    "ramp_rate_vs_baseline": "volume ramp vs. normal baseline",
    "pct_new_buyers": "share of first-time buyer accounts",
    "n_distinct_states": "number of distinct buyer states",
    "geo_entropy": "geographic spread of buyers",
    "mean_delivery_days": "average delivery time",
    "std_delivery_days": "variation in delivery time",
    "return_rate": "return/cancellation rate",
    "refund_claim_speed_days": "average time to claim a refund after delivery",
    "refund_claim_speed_std": "variation in refund claim speed",
}


def load_model_and_data():
    model = joblib.load(SRC_DIR / "spikeguard_model.joblib")
    config = joblib.load(SRC_DIR / "spikeguard_config.joblib")
    test = pd.read_csv(DATA_DIR / "spikeguard_test.csv")
    return model, config, test


def explain_spike(model, explainer, row, feature_cols, threshold, top_n=4):
    """Return a plain-language explanation for one spike's prediction."""
    X_row = row[feature_cols].values.reshape(1, -1)
    proba = model.predict_proba(X_row)[0, 1]
    flagged = proba >= threshold

    shap_values = explainer(X_row)
    contributions = shap_values.values[0]  # one SHAP value per feature for this row

    # rank features by how much they pushed the prediction toward "anomalous"
    order = np.argsort(-np.abs(contributions))[:top_n]

    reasons = []
    for idx in order:
        feat = feature_cols[idx]
        val = row[feat]
        contrib = contributions[idx]
        direction = "raised" if contrib > 0 else "lowered"
        reasons.append(
            f"  - {FEATURE_LABELS.get(feat, feat)} = {val} ({direction} risk score, impact {contrib:+.3f})"
        )

    verdict = "FLAGGED as anomalous" if flagged else "not flagged (organic)"
    header = f"Spike risk score: {proba:.3f} (threshold {threshold:.2f}) -> {verdict}"
    return header + "\n" + "\n".join(reasons)


def main():
    model, config, test = load_model_and_data()
    feature_cols = config["feature_cols"]
    threshold = config["threshold"]

    explainer = shap.TreeExplainer(model)

    # show explanations for the top 5 highest-risk spikes in the test set,
    # so you can see what a real flagged case looks like
    proba_all = model.predict_proba(test[feature_cols])[:, 1]
    test = test.copy()
    test["risk_score"] = proba_all
    top_flagged = test.sort_values("risk_score", ascending=False).head(5)

    print(f"=== Top 5 highest-risk spikes (threshold = {threshold:.2f}) ===\n")
    for i, (_, row) in enumerate(top_flagged.iterrows(), 1):
        actual = row["spike_type"]
        print(f"--- Spike #{i} (actual label: {actual}) ---")
        print(explain_spike(model, explainer, row, feature_cols, threshold))
        print()

    # also show one example of a spike near the decision boundary --
    # the genuinely ambiguous cases are the most interesting to demo
    test["dist_from_threshold"] = (test["risk_score"] - threshold).abs()
    borderline = test.sort_values("dist_from_threshold").head(2)
    print(f"=== Borderline spikes (closest to the {threshold:.2f} threshold) ===\n")
    for i, (_, row) in enumerate(borderline.iterrows(), 1):
        actual = row["spike_type"]
        print(f"--- Borderline spike #{i} (actual label: {actual}) ---")
        print(explain_spike(model, explainer, row, feature_cols, threshold))
        print()


if __name__ == "__main__":
    main()
