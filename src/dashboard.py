"""
SpikeGuard merchant dashboard.

Loads the already-trained model (spikeguard_model.joblib) -- no training
happens here. Lets you either:
  (a) pick a spike from the test set to inspect, or
  (b) enter your own spike's numbers manually
and see the risk score, flag decision, and SHAP-based explanation.

Run with:  streamlit run dashboard.py
"""

import pandas as pd
import numpy as np
import joblib
import shap
import streamlit as st
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "Datasets"
SRC_DIR = PROJECT_ROOT / "src"

FEATURE_LABELS = {
    "n_orders": "Total orders in the spike",
    "duration_days": "Spike duration (days)",
    "ramp_rate_vs_baseline": "Volume ramp vs. normal baseline",
    "pct_new_buyers": "Share of first-time buyer accounts",
    "n_distinct_states": "Number of distinct buyer states",
    "geo_entropy": "Geographic spread of buyers",
    "mean_delivery_days": "Average delivery time (days)",
    "std_delivery_days": "Variation in delivery time",
    "return_rate": "Return / cancellation rate",
    "refund_claim_speed_days": "Avg. time to claim refund after delivery (days)",
    "refund_claim_speed_std": "Variation in refund claim speed",
}

# sensible input ranges for the manual-entry sliders, spanning both classes
FEATURE_RANGES = {
    "n_orders": (5, 200, 20),
    "duration_days": (1, 7, 3),
    "ramp_rate_vs_baseline": (1.0, 70.0, 4.0),
    "pct_new_buyers": (0.0, 1.0, 0.45),
    "n_distinct_states": (1, 20, 8),
    "geo_entropy": (0.01, 3.0, 1.5),
    "mean_delivery_days": (0.1, 15.0, 10.0),
    "std_delivery_days": (0.05, 10.0, 6.0),
    "return_rate": (0.0, 1.0, 0.05),
    "refund_claim_speed_days": (0.05, 16.0, 9.0),
    "refund_claim_speed_std": (0.02, 6.0, 3.5),
}


@st.cache_resource
def load_model():
    model = joblib.load(SRC_DIR / "spikeguard_model.joblib")
    config = joblib.load(SRC_DIR / "spikeguard_config.joblib")
    return model, config


@st.cache_data
def load_test_data():
    return pd.read_csv(DATA_DIR / "spikeguard_test.csv")


def explain_row(model, explainer, feature_cols, X_row):
    proba = model.predict_proba(X_row)[0, 1]
    shap_values = explainer(X_row)
    contributions = shap_values.values[0]
    order = np.argsort(-np.abs(contributions))
    rows = []
    for idx in order:
        feat = feature_cols[idx]
        rows.append({
            "Feature": FEATURE_LABELS.get(feat, feat),
            "Value": round(float(X_row[0, idx]), 3),
            "Impact on risk score": round(float(contributions[idx]), 3),
        })
    return proba, pd.DataFrame(rows)


def main():
    st.set_page_config(page_title="SpikeGuard", page_icon="🛡️", layout="wide")
    st.title("🛡️ SpikeGuard")
    st.caption("Merchant-side spike classifier: tells you whether a sudden order spike looks like a real sale or an organized return-abuse ring.")

    model, config = load_model()
    feature_cols = config["feature_cols"]
    threshold = config["threshold"]

    tab1, tab2, tab3 = st.tabs(["📊 Test set explorer", "✍️ Check a spike manually", "ℹ️ About"])

    explainer = shap.TreeExplainer(model)

    with tab1:
        st.subheader("Browse spikes from the held-out test set")
        test = load_test_data()
        proba_all = model.predict_proba(test[feature_cols])[:, 1]
        test = test.copy()
        test["risk_score"] = proba_all
        test["flagged"] = test["risk_score"] >= threshold

        col_filter, col_count = st.columns([2, 1])
        with col_filter:
            show = st.selectbox("Show", ["All", "Flagged only", "Not flagged only"])
        if show == "Flagged only":
            display_df = test[test["flagged"]]
        elif show == "Not flagged only":
            display_df = test[~test["flagged"]]
        else:
            display_df = test
        with col_count:
            st.metric("Spikes shown", len(display_df))

        display_df = display_df.sort_values("risk_score", ascending=False).reset_index(drop=True)
        st.dataframe(
            display_df[["spike_type", "risk_score", "flagged"] + feature_cols].round(3),
            use_container_width=True,
            height=300,
        )

        st.divider()
        idx = st.number_input("Row index to explain (from table above)", min_value=0,
                               max_value=max(len(display_df) - 1, 0), value=0)
        if len(display_df) > 0:
            row = display_df.iloc[idx]
            X_row = row[feature_cols].values.reshape(1, -1).astype(float)
            proba, explanation_df = explain_row(model, explainer, feature_cols, X_row)

            verdict = "🚩 FLAGGED as anomalous" if proba >= threshold else "✅ Not flagged (organic)"
            c1, c2, c3 = st.columns(3)
            c1.metric("Risk score", f"{proba:.3f}")
            c2.metric("Threshold", f"{threshold:.2f}")
            c3.metric("Actual label", row["spike_type"])
            st.markdown(f"### {verdict}")

            st.bar_chart(explanation_df.set_index("Feature")["Impact on risk score"])
            st.dataframe(explanation_df, use_container_width=True)

    with tab2:
        st.subheader("Enter a spike's numbers to check it")
        st.caption("Simulates what a merchant would see for one of their own order spikes.")

        cols = st.columns(2)
        manual_values = {}
        for i, feat in enumerate(feature_cols):
            lo, hi, default = FEATURE_RANGES[feat]
            with cols[i % 2]:
                if isinstance(lo, int) and isinstance(hi, int):
                    manual_values[feat] = st.slider(FEATURE_LABELS.get(feat, feat), lo, hi, default)
                else:
                    manual_values[feat] = st.slider(FEATURE_LABELS.get(feat, feat), float(lo), float(hi), float(default))

        if st.button("Check this spike", type="primary"):
            X_row = np.array([[manual_values[f] for f in feature_cols]], dtype=float)
            proba, explanation_df = explain_row(model, explainer, feature_cols, X_row)

            verdict = "🚩 FLAGGED as anomalous" if proba >= threshold else "✅ Not flagged (organic)"
            c1, c2 = st.columns(2)
            c1.metric("Risk score", f"{proba:.3f}")
            c2.metric("Threshold", f"{threshold:.2f}")
            st.markdown(f"### {verdict}")

            st.bar_chart(explanation_df.set_index("Feature")["Impact on risk score"])
            st.dataframe(explanation_df, use_container_width=True)

    with tab3:
        st.markdown("""
        **SpikeGuard** classifies a merchant's transaction-volume spike as either:
        - **Organic** — a genuine sales surge
        - **Anomalous** — a pattern resembling organized return-abuse, modeled on a real
          2026 Meesho fraud case (2,500 fraudulent orders, near-100% fake return/refund cycle)

        The organic baseline is calibrated from real Olist e-commerce statistics
        (order volume, delivery timing, buyer geographic spread). The anomalous
        pattern is grounded in the documented Meesho case and Razorpay's published
        account-freeze risk factors, since no public dataset contains labeled
        Indian merchant fraud data.

        The model is a cost-sensitive XGBoost classifier. The decision threshold
        was chosen to minimize total expected cost (false positive cost: wrongly
        flagging a real sale vs. false negative cost: missing a fraud ring), not
        just to maximize accuracy.

        Explanations use SHAP values on the trained model — this shows which
        features actually drove each individual prediction.
        """)


if __name__ == "__main__":
    main()
