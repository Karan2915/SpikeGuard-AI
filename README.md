# SpikeGuard
**AI Risk Manager track — fraud-spike detector**

🔗 **Live demo:** [spikeguard-ai.streamlit.app](https://spikeguard-ai.streamlit.app)

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
  Olist Brazilian e-commerce dataset: normal daily order volume per seller,
  delivery timing, buyer geographic spread, and baseline cancel/return rate.
  Five distinct organic event types are simulated (regular promotion, flash sale,
  festival demand, product launch, viral promotion), each with its own realistic
  volume, buyer-mix, and timing profile — a real sales spike is not one uniform
  shape, and treating it as one would make the classifier brittle.
- **Anomalous class** — parameters grounded in the documented Meesho case and
  Razorpay's published freeze-risk factors, simulated across three severity tiers
  (strong, moderate, and subtle rings), so the model has to learn a genuine
  pattern rather than a single obvious signature.

1,737 training and 580 held-out test spikes were generated, with deliberate
feature-level noise, overlap between classes, and a small amount of label
ambiguity, so the classification problem is genuinely hard rather than trivially
separable — an earlier, simpler version of the generator produced suspiciously
perfect (100%) accuracy, which was identified as a red flag and corrected before
training.

## Model
XGBoost classifier, trained on the training split, evaluated on the 580 held-out
test spikes the model never saw during training or threshold selection.

| Metric | Anomalous (default 0.5 threshold) |
|---|---|
| Precision | 0.97 |
| Recall | 0.91 |

**PR-AUC: 0.9573**

## Cost-sensitive threshold
Rather than using the default 0.5 cutoff, the operating threshold was chosen to
minimize total expected cost:
- **False positive cost (₹150):** manual review time when a real sale gets
  wrongly flagged.
- **False negative cost (₹64,000):** estimated refund/loss exposure of one
  missed abuse ring (assumed ₹800 average order value × ~80 orders per
  anomalous spike). These are stated assumptions, not measured facts — no
  public dataset gives real merchant review costs.

Because the false-negative cost so heavily outweighs the false-positive cost,
the cost-optimal threshold (0.06) sits far below the default. At this
threshold: 77 false positives, 12 false negatives, total cost ₹779,550 — a
**₹245,200 saving** versus the default threshold's ₹1,024,750 (5 false
positives, 16 false negatives). This is an honest trade-off, not a free
improvement: the cost-optimal model flags many more organic spikes for review
in exchange for catching more real fraud, because in this cost model a missed
ring is over 400x more expensive than an unnecessary review.

## Early Warning model — and an honest limitation of the confirmed model
The confirmed-risk model above uses `return_rate`, `refund_claim_speed_days`,
and delivery timing — but these are **lagging signals**: they only exist after
an order has been delivered and, in the fraud case, already refunded. By the
time the confirmed model can score a spike with full information, the money
for those specific orders is already gone.

To address this honestly, a second model — **Early Warning** — is trained
using only signals available at order-placement time, before any delivery or
refund has happened: order ramp rate, buyer clustering, new-buyer share,
geographic spread. It cannot see delivery or return data at all.

| | Early Warning (leading features only) | Confirmed Risk (full features) |
|---|---|---|
| PR-AUC | 0.8528 | 0.9573 |
| Precision (anomalous, at its own cost-optimal threshold) | 0.50 | 0.68 |
| Recall (anomalous, at its own cost-optimal threshold) | 0.98 | 0.93 |

The Early Warning model is measurably less accurate — it has strictly less
information to work with, and that shows up clearly in both PR-AUC and
precision. That is the honest, stated cost of acting sooner. Its value isn't
raw accuracy: it's *timing*. It can flag a suspicious seller relationship for
review while orders are still being placed, rather than only after the full
return/refund cycle has completed.

This directly addresses the real-world timing problem: no model — this one
included — can catch the very first order in a new fraud ring before it plays
out; there's no way to know a brand-new order is fraudulent with zero history.
What this two-stage design actually prevents is the *remainder* of an ongoing
ring. In the real Meesho case, the fraud ran for 7 months and 2,500 orders
before an internal audit caught it. An Early Warning score, firing on
leading signals alone within days of a pattern starting, could flag that
seller relationship for review long before order #2,500 — turning a 7-month,
₹5.5 crore loss into a much smaller one.

**Recommended two-tier action:**
- **Early Warning fires →** hold payout / flag for lightweight manual review.
  Low-stakes action, appropriate given the lower precision.
- **Confirmed Risk fires →** justifies stronger action (account freeze,
  escalation), given its higher precision.

## Explanation layer
Each prediction is paired with a SHAP-based explanation showing which features
drove the risk score and in which direction — this is feature attribution on
the models actually trained and evaluated, not a separate causal system. The
dashboard also surfaces **global feature importance** (which signals matter
most across the whole test set, not just one spike) and **risk tiers**
(Low / Medium / High / Critical, each with a recommended action), rather than
only a binary flagged/not-flagged split.

## Product
A Streamlit dashboard provides:
- **Test set explorer** — browse held-out spikes with risk scores, color-coded
  by flag status, with summary metrics (spikes analyzed, flagged count,
  estimated exposure flagged, live-computed PR-AUC).
- **Early vs Confirmed** — compare both models side by side on the same
  spikes, with the two-tier action recommendation above.
- **Insights** — global feature importance, risk-tier breakdown, and a spike
  timeline view.
- **Check a spike manually** — interactive sliders to test any combination of
  values and see the live risk score and explanation.
- **Upload your data** — a merchant can upload their own CSV of spike-level
  data (with a downloadable template and column validation) and get every
  row scored, explained, and downloadable as results.

All views load the already-trained models — no training happens in the app.

## Defense-only by design
SpikeGuard only scores and explains; it never blocks, auto-refunds, or takes
action on a transaction. It is a decision-support tool, not an autonomous system.

## Limitations
- Anomalous-class data is synthetic, not sourced from real labeled fraud
  (standard practice in this space — no public Indian merchant fraud logs
  exist — but stated plainly rather than hidden).
- The Early Warning model's much lower precision (0.50) means, in production,
  its output should inform a lightweight review step, not an automated
  penalty — this is reflected in the recommended two-tier action above.
- Cost assumptions (₹150 / ₹64,000) are illustrative and should be replaced
  with a merchant's real numbers in production.
