# SpikeGuard
**AI Risk Manager track — fraud-spike detector**

## Problem
Indian merchants lose money not just to individual fraudulent transactions, but to
*organized abuse rings* that show up as sudden, unusual spikes in order volume.
A real example: a Meesho seller relationship was exploited by a gang that placed
~2,500 fraudulent orders over 7 months using fabricated buyer profiles and fake
addresses, falsely claimed refunds on every "undelivered" parcel, and cost ₹5.5
crore before an internal audit caught it. Separately, Razorpay's own published
guidance names sudden volume spikes, geographic drift, and patterns inconsistent
with a merchant's declared profile as triggers for account freezes and reserves —
meaning undetected fraud doesn't just cost the merchant money, it can get their
account shut down.

**SpikeGuard classifies a merchant's transaction-volume spike as either:**
- **Organic** — a genuine sales surge
- **Anomalous** — a pattern resembling organized return-abuse

This is a spike-level classifier, not a transaction-level one — it looks at the
*shape* of a spike (ramp rate, buyer diversity, delivery/return timing) rather
than flagging individual orders, which matches how the real Meesho case actually
played out.

## Data
No public dataset contains labeled Indian merchant fraud data, so SpikeGuard uses
synthetic spike-level data, calibrated against two real sources rather than
invented:

- **Organic class** — parameters calibrated from real statistics computed on the
  Olist Brazilian e-commerce dataset: normal daily order volume per seller
  (mean ~1.3, spikes up to ~4-5x), delivery timing (median ~10 days), buyer
  geographic spread (median 4 distinct states), and baseline cancel/return rate
  (~1.2%).
- **Anomalous class** — parameters grounded in the documented Meesho case (near-
  instant fake delivery, near-100% return/refund cycle, narrow buyer/address
  clustering) and Razorpay's published freeze-risk factors.

1,412 labeled spikes were generated, with deliberate feature-level noise and
overlap between classes (plus ~3% label noise) so the classification problem is
genuinely hard, not trivially separable — a first version of the generator
produced 100% accuracy, which was a red flag rather than a good result, and was
corrected before training.

## Model
XGBoost classifier, trained on 1,059 spikes, tested on 353 held-out spikes the
model never saw during training or threshold selection.

| Metric | Organic | Anomalous |
|---|---|---|
| Precision | 0.98 | 0.98–0.99 |
| Recall | 0.99–1.00 | 0.94 |

**PR-AUC: 0.9505**

## Cost-sensitive threshold
Rather than using the default 0.5 cutoff, the operating threshold was chosen to
minimize total expected cost:
- **False positive cost (₹150):** manual review time when a real sale gets
  wrongly flagged.
- **False negative cost (₹64,000):** estimated refund/loss exposure of one
  missed abuse ring (assumed ₹800 average order value × ~80 orders per
  anomalous spike, the average size observed in the synthetic data). These are
  stated assumptions, not measured facts — no public dataset gives real
  merchant review costs.

At the chosen threshold (~0.56): 1 false positive, 6 false negatives, total
cost ₹384,150. A full threshold sweep showed those 6 missed cases stay missed
even at very low thresholds (down to 0.05) — they are genuinely
hard-to-detect rings deliberately designed to mimic organic behavior closely.
Lowering the threshold further only adds false-positive cost without
recovering them. This is a stated limitation, not a hidden one.

## Explanation layer
Each prediction is paired with a SHAP-based explanation showing which features
drove the risk score and in which direction — this is feature attribution on
the one model actually trained and evaluated, not a separate causal system.
Two examples from testing:

- A genuine organic spike scored 0.551, just under the 0.56 threshold —
  correctly left unflagged, though its longer delivery time had pushed the
  score up; low order count and refund-timing variation pulled it back down.
- A genuine anomalous spike scored 0.630 and was correctly flagged — even
  though its volume ramp and new-buyer share were less extreme than typical
  rings, uniform refund-claim timing was enough to catch it.

## Product
A Streamlit dashboard lets a user browse held-out spikes with their risk scores
and explanations, or manually enter a spike's numbers to check it live. It
loads the already-trained model — no training happens in the app.

## Defense-only by design
SpikeGuard only scores and explains; it never blocks, auto-refunds, or takes
action on a transaction. It is a decision-support tool, not an autonomous system.

## Limitations
- Anomalous-class data is synthetic, not sourced from real labeled fraud
  (standard practice in this space — no public Indian merchant fraud logs
  exist — but stated plainly rather than hidden).
- ~6% of anomalous spikes (the most sophisticated rings) are not caught by
  the current feature set at any threshold.
- Cost assumptions (₹150 / ₹64,000) are illustrative and should be replaced
  with a merchant's real numbers in production.
