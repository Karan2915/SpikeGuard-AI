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
from sklearn.metrics import average_precision_score

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


@st.cache_resource
def load_early_model():
    """Returns (None, None) if the early model hasn't been trained yet --
    lets the dashboard degrade gracefully instead of crashing."""
    early_path = SRC_DIR / "spikeguard_early_model.joblib"
    config_path = SRC_DIR / "spikeguard_early_config.joblib"
    if not early_path.exists() or not config_path.exists():
        return None, None
    return joblib.load(early_path), joblib.load(config_path)


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


def plain_english_summary(explanation_df, flagged):
    """Turn the top 2-3 SHAP contributors into one auto-generated sentence,
    so the explanation reads like a finding, not a table of numbers."""
    top = explanation_df.reindex(explanation_df["Impact on risk score"].abs().sort_values(ascending=False).index)
    top = top.head(3)
    driving = top[top["Impact on risk score"] > 0]["Feature"].tolist()
    restraining = top[top["Impact on risk score"] < 0]["Feature"].tolist()

    def lower_first(items):
        return [s[0].lower() + s[1:] if s else s for s in items]

    driving = lower_first(driving)
    restraining = lower_first(restraining)

    if flagged:
        if driving:
            text = f"Flagged mainly due to **{', '.join(driving)}** — a pattern consistent with organized return abuse."
        else:
            text = "Flagged, though no single feature dominated — the combination of signals crossed the risk threshold."
    else:
        if restraining:
            text = f"Not flagged — **{', '.join(restraining)}** kept the risk score below the threshold, despite some elevated signals."
        else:
            text = "Not flagged — this spike's pattern is consistent with a genuine sales surge."
    return text


def style_flagged_rows(df):
    def highlight(row):
        color = "background-color: rgba(255, 80, 80, 0.18)" if row["flagged"] else "background-color: rgba(60, 200, 120, 0.10)"
        return [color] * len(row)
    return df.style.apply(highlight, axis=1)


def render_metric_cards(test_df, fn_cost):
    total = len(test_df)
    flagged_count = int(test_df["flagged"].sum())
    flagged_pct = (flagged_count / total * 100) if total else 0
    exposure_flagged = flagged_count * fn_cost
    pr_auc = average_precision_score(
        (test_df["spike_type"] == "anomalous").astype(int), test_df["risk_score"]
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Spikes analyzed", f"{total:,}")
    c2.metric("Flagged as anomalous", f"{flagged_count:,}", f"{flagged_pct:.1f}% of total")
    c3.metric("Est. exposure flagged", f"₹{exposure_flagged:,.0f}",
              help="Flagged spikes × assumed loss per undetected ring (₹64,000). "
                   "This is total exposure surfaced for review, not confirmed prevented loss — "
                   "some flagged spikes will be false positives, and ground truth isn't known at prediction time.")
    c4.metric("Model PR-AUC", f"{pr_auc:.3f}", help="Precision-recall AUC, computed live on this held-out test set.")


def assign_risk_tier(risk_score, threshold):
    """Four-tier risk banding instead of a flat flagged/not-flagged split.
    Bands are set relative to the model's own threshold, not fixed numbers,
    so they stay sensible if the threshold is ever retuned."""
    if risk_score < threshold * 0.5:
        return "Low"
    elif risk_score < threshold:
        return "Medium"
    elif risk_score < threshold + (1 - threshold) * 0.5:
        return "High"
    else:
        return "Critical"


TIER_ORDER = ["Low", "Medium", "High", "Critical"]
TIER_ACTIONS = {
    "Low": "No action needed — pattern consistent with normal sales activity.",
    "Medium": "Worth a light check if this seller has had prior flags, but not urgent on its own.",
    "High": "Recommend manual review — several signals point toward organized abuse.",
    "Critical": "Recommend prioritized review — this pattern closely matches known fraud-ring behavior.",
}


def compute_global_importance(model, explainer, X, feature_cols):
    """Mean absolute SHAP value per feature across a whole dataset --
    answers 'what matters most overall', not just for one row."""
    shap_values = explainer(X)
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    imp_df = pd.DataFrame({
        "Feature": [FEATURE_LABELS.get(f, f) for f in feature_cols],
        "Mean |impact| on risk score": mean_abs,
    }).sort_values("Mean |impact| on risk score", ascending=False)
    return imp_df


def main():
    st.set_page_config(page_title="SpikeGuard", page_icon="🛡️", layout="wide")
    st.title("🛡️ SpikeGuard")
    st.caption("Merchant-side spike classifier: tells you whether a sudden order spike looks like a real sale or an organized return-abuse ring.")

    model, config = load_model()
    feature_cols = config["feature_cols"]
    threshold = config["threshold"]
    fn_cost = config.get("fn_cost", 64000)
    early_model, early_config = load_early_model()

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["📊 Test set explorer", "⏱️ Early vs Confirmed", "📈 Insights",
         "✍️ Check a spike manually", "📁 Upload your data", "ℹ️ About"]
    )

    explainer = shap.TreeExplainer(model)

    with tab1:
        test = load_test_data()
        proba_all = model.predict_proba(test[feature_cols])[:, 1]
        test = test.copy()
        test["risk_score"] = proba_all
        test["flagged"] = test["risk_score"] >= threshold

        render_metric_cards(test, fn_cost)
        st.divider()

        st.subheader("Browse spikes from the held-out test set")
        show = st.selectbox("Show", ["All", "Flagged only", "Not flagged only"])
        if show == "Flagged only":
            display_df = test[test["flagged"]]
        elif show == "Not flagged only":
            display_df = test[~test["flagged"]]
        else:
            display_df = test

        display_df = display_df.sort_values("risk_score", ascending=False).reset_index(drop=True)
        table_cols = ["spike_type", "risk_score", "flagged"] + feature_cols
        st.dataframe(
            style_flagged_rows(display_df[table_cols].round(3)),
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
            flagged = proba >= threshold

            verdict = "🚩 FLAGGED as anomalous" if flagged else "✅ Not flagged (organic)"
            c1, c2, c3 = st.columns(3)
            c1.metric("Risk score", f"{proba:.3f}")
            c2.metric("Threshold", f"{threshold:.2f}")
            c3.metric("Actual label", row["spike_type"])
            st.markdown(f"### {verdict}")
            st.info(plain_english_summary(explanation_df, flagged))

            st.bar_chart(explanation_df.set_index("Feature")["Impact on risk score"])
            st.dataframe(explanation_df, use_container_width=True)

    with tab2:
        st.subheader("Early Warning vs. Confirmed Risk")
        if early_model is None:
            st.warning(
                "Early-warning model not found. Run `python train_early_model.py` in "
                "src/ and redeploy to enable this tab."
            )
        else:
            early_feature_cols = early_config["feature_cols"]
            early_threshold = early_config["threshold"]

            st.info(
                "**Two models, two moments in a spike's lifecycle.** The Confirmed Risk model "
                "(used elsewhere in this app) uses delivery and return/refund timing — signals "
                "that only exist after an order has been delivered and, in fraud cases, already "
                "refunded. The Early Warning model below uses only signals available the moment "
                "orders start coming in: ramp rate, buyer clustering, new-buyer share. It can "
                "flag a suspicious pattern *before* any money is lost — at the honest cost of "
                "more false alarms, since it has less information to work with."
            )

            test = load_test_data()
            test = test.copy()
            test["early_score"] = early_model.predict_proba(test[early_feature_cols])[:, 1]
            test["confirmed_score"] = model.predict_proba(test[feature_cols])[:, 1]
            test["early_flagged"] = test["early_score"] >= early_threshold
            test["confirmed_flagged"] = test["confirmed_score"] >= threshold

            c1, c2, c3 = st.columns(3)
            c1.metric("Early Warning PR-AUC", f"{early_config.get('pr_auc', 0):.3f}",
                      help="Measured on the same held-out test set as the Confirmed Risk model.")
            early_precision = (
                (test["early_flagged"] & (test["spike_type"] == "anomalous")).sum()
                / max(test["early_flagged"].sum(), 1)
            )
            c2.metric("Early Warning precision", f"{early_precision:.0%}",
                      help="Of spikes the Early model flags, this share are truly anomalous. "
                           "Lower than the Confirmed model by design — it's trading precision for timing.")
            both_flagged = (test["early_flagged"] & test["confirmed_flagged"]).sum()
            c3.metric("Flagged by both models", f"{both_flagged}",
                      help="Spikes both models agree on — the highest-confidence cases.")

            st.divider()
            st.caption(
                "Recommended action: use Early Warning to hold payouts / flag for lightweight "
                "review as orders come in. Use Confirmed Risk, once delivery/return data exists, "
                "to justify stronger action (account freeze, escalation)."
            )

            compare_cols = ["spike_type", "early_score", "early_flagged", "confirmed_score", "confirmed_flagged"]
            st.dataframe(
                test[compare_cols].sort_values("early_score", ascending=False).round(3),
                use_container_width=True, height=300,
            )

    with tab3:
        st.subheader("Global feature importance")
        st.caption("Which signals matter most across the whole test set, not just one spike.")

        test_for_importance = load_test_data()
        X_full = test_for_importance[feature_cols]
        global_imp = compute_global_importance(model, explainer, X_full, feature_cols)
        st.bar_chart(global_imp.set_index("Feature")["Mean |impact| on risk score"])

        st.divider()
        st.subheader("Risk tier breakdown")
        test_tiers = test_for_importance.copy()
        test_tiers["risk_score"] = model.predict_proba(test_tiers[feature_cols])[:, 1]
        test_tiers["tier"] = test_tiers["risk_score"].apply(lambda s: assign_risk_tier(s, threshold))

        tier_counts = test_tiers["tier"].value_counts().reindex(TIER_ORDER, fill_value=0)
        st.bar_chart(tier_counts)

        tier_cols = st.columns(4)
        for i, tier in enumerate(TIER_ORDER):
            with tier_cols[i]:
                st.metric(tier, int(tier_counts[tier]))
                st.caption(TIER_ACTIONS[tier])

        if "start_day" in test_for_importance.columns:
            st.divider()
            st.subheader("Spikes over time")
            st.caption("When spikes occurred during the 180-day simulated window (day of simulation, not a real calendar).")
            timeline_df = test_tiers[["start_day", "risk_score", "spike_type"]].sort_values("start_day")
            st.scatter_chart(timeline_df, x="start_day", y="risk_score", color="spike_type")
        else:
            st.caption(
                "Timeline view needs a `start_day` column, which isn't in the current dataset. "
                "Regenerate with the updated generate_spikes.py to enable this."
            )

    with tab4:
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
            flagged = proba >= threshold

            verdict = "🚩 FLAGGED as anomalous" if flagged else "✅ Not flagged (organic)"
            c1, c2 = st.columns(2)
            c1.metric("Risk score", f"{proba:.3f}")
            c2.metric("Threshold", f"{threshold:.2f}")
            st.markdown(f"### {verdict}")
            st.info(plain_english_summary(explanation_df, flagged))

            st.bar_chart(explanation_df.set_index("Feature")["Impact on risk score"])
            st.dataframe(explanation_df, use_container_width=True)

    with tab5:
        st.subheader("Upload your own order-spike data")
        st.caption(
            "Upload a CSV with one row per spike. Required columns: "
            + ", ".join(f"`{c}`" for c in feature_cols)
        )

        with st.expander("What should my CSV look like?"):
            st.write(
                "Each row represents one detected order-volume spike for a seller, "
                "already summarized into these numbers (not raw per-order data):"
            )
            sample = pd.DataFrame([{c: FEATURE_RANGES[c][2] for c in feature_cols}])
            st.dataframe(sample, use_container_width=True)
            csv_template = sample.to_csv(index=False).encode("utf-8")
            st.download_button("Download a template CSV", csv_template, "spikeguard_template.csv", "text/csv")

        uploaded = st.file_uploader("Choose a CSV file", type=["csv"])

        if uploaded is not None:
            try:
                user_df = pd.read_csv(uploaded)
            except Exception as e:
                st.error(f"Couldn't read that file as a CSV: {e}")
                user_df = None

            if user_df is not None:
                missing = [c for c in feature_cols if c not in user_df.columns]
                if missing:
                    st.error(
                        "This file is missing required column(s): "
                        + ", ".join(f"`{c}`" for c in missing)
                        + ". Download the template above to see the expected format."
                    )
                else:
                    extra = [c for c in user_df.columns if c not in feature_cols]
                    if extra:
                        st.info(f"Ignoring extra column(s) not used by the model: {', '.join(extra)}")

                    numeric_df = user_df[feature_cols].apply(pd.to_numeric, errors="coerce")
                    bad_rows = numeric_df[numeric_df.isna().any(axis=1)]
                    if len(bad_rows) > 0:
                        st.warning(
                            f"{len(bad_rows)} row(s) have non-numeric or missing values in required "
                            f"columns and will be skipped."
                        )
                    valid_mask = numeric_df.notna().all(axis=1)
                    clean_df = user_df[valid_mask].copy()
                    clean_numeric = numeric_df[valid_mask]

                    if len(clean_df) == 0:
                        st.error("No valid rows to score after checking for missing/non-numeric values.")
                    else:
                        proba_upload = model.predict_proba(clean_numeric[feature_cols])[:, 1]
                        clean_df["risk_score"] = proba_upload
                        clean_df["flagged"] = clean_df["risk_score"] >= threshold

                        st.success(f"Scored {len(clean_df)} spike(s).")
                        render_metric_cards(
                            clean_df.assign(spike_type=np.where(clean_df["flagged"], "anomalous", "organic")),
                            fn_cost,
                        )
                        st.divider()

                        result_cols = ["risk_score", "flagged"] + feature_cols
                        st.dataframe(
                            style_flagged_rows(clean_df[result_cols].round(3)),
                            use_container_width=True,
                            height=300,
                        )

                        result_csv = clean_df.to_csv(index=False).encode("utf-8")
                        st.download_button("Download scored results", result_csv, "spikeguard_results.csv", "text/csv")

                        st.divider()
                        up_idx = st.number_input(
                            "Row index to explain (from table above)", min_value=0,
                            max_value=max(len(clean_df) - 1, 0), value=0, key="upload_idx",
                        )
                        row = clean_df.iloc[up_idx]
                        X_row = clean_numeric.iloc[[up_idx]].values.astype(float)
                        proba, explanation_df = explain_row(model, explainer, feature_cols, X_row)
                        flagged = proba >= threshold

                        verdict = "🚩 FLAGGED as anomalous" if flagged else "✅ Not flagged (organic)"
                        c1, c2 = st.columns(2)
                        c1.metric("Risk score", f"{proba:.3f}")
                        c2.metric("Threshold", f"{threshold:.2f}")
                        st.markdown(f"### {verdict}")
                        st.info(plain_english_summary(explanation_df, flagged))
                        st.bar_chart(explanation_df.set_index("Feature")["Impact on risk score"])
                        st.dataframe(explanation_df, use_container_width=True)

    with tab6:
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

        **Why two models?** The main ("Confirmed Risk") model uses delivery and
        return/refund timing — but those signals only exist after an order has
        already been delivered and, in fraud cases, already refunded. By then
        the money is gone. The "Early Warning" model uses only signals available
        at order-placement time (ramp rate, buyer clustering, new-buyer share),
        so it can flag a suspicious pattern before losses accumulate — at the
        honest cost of more false alarms. See the "Early vs Confirmed" tab.
        """)


if __name__ == "__main__":
    main()