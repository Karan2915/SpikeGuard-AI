"""
SpikeGuard synthetic data generator.

Simulates merchant order streams over time, injects two kinds of volume
spikes, and outputs a spike-level labeled dataset ready for training a
classifier (organic vs anomalous).

Baseline ("organic") parameters are calibrated from real Olist e-commerce
statistics (computed separately from the actual Kaggle dataset):
    - normal daily orders per active seller: mean ~1.3, most sellers 1-2/day
    - organic spike peak: up to ~4-5x baseline
    - delivery time: median ~10 days, mean ~12, std ~9.5
    - buyer geographic spread: median 4 distinct states, up to 27 for big sellers
    - normal cancel/unavailable rate: ~1.2%

Anomalous parameters are calibrated from the real Meesho return-abuse case
(2,500 fraudulent orders over 7 months, ~250 SIM cards, ~50 bank accounts,
near-100% "undelivered -> returned -> refunded" cycle rate) and Razorpay's
published account-freeze risk factors (sudden volume spikes, geographic
drift, pattern inconsistent with declared profile).

This is a SIMULATION. No real merchant or buyer data is used or represented.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass

RNG = np.random.default_rng(42)

N_SELLERS = 3000          # number of simulated merchants
N_DAYS = 180              # simulation horizon per seller
ORGANIC_SPIKE_RATE = 0.35   # probability a seller gets an organic spike at some point
ANOMALOUS_SPIKE_RATE = 0.12 # probability a seller gets an anomalous spike at some point

N_STATES = 27  # roughly matches distinct Indian states/UTs count, used as a generic geo-cluster proxy


@dataclass
class Spike:
    seller_id: int
    spike_type: str            # "organic" or "anomalous"
    start_day: int
    duration_days: int
    baseline_daily_orders: float
    peak_daily_orders: float
    n_orders: int
    pct_new_buyers: float
    n_distinct_states: int
    mean_delivery_days: float
    std_delivery_days: float
    return_rate: float
    refund_claim_speed_days: float
    refund_claim_speed_std: float


def simulate_organic_spike(seller_id, start_day):
    """A real sales surge: gradual ramp, diverse buyers, normal delivery/return behavior."""
    baseline = RNG.uniform(0.8, 2.0)
    peak_multiplier = RNG.uniform(2.5, 5.0)          # matches Olist's observed organic peak range
    duration = int(RNG.integers(2, 6))                # sales events run a few days
    n_orders = int(peak_multiplier * baseline * duration * RNG.uniform(0.8, 1.2))
    n_orders = max(n_orders, 8)

    # occasional "hard" organic case: a flash sale that ramps unusually fast
    # and pulls in more first-time buyers than typical -- looks closer to
    # anomalous on ramp/new-buyer features alone, forcing the model to rely
    # on the full feature set rather than any single signal
    is_hard_case = RNG.random() < 0.18
    if is_hard_case:
        peak_multiplier = RNG.uniform(6, 11)
        duration = int(RNG.integers(1, 3))
        n_orders = int(peak_multiplier * baseline * duration * RNG.uniform(0.8, 1.2))
        n_orders = max(n_orders, 8)
        pct_new_buyers = RNG.uniform(0.55, 0.78)
        n_distinct_states = int(RNG.integers(3, 7))
        mean_delivery_days = max(RNG.uniform(3, 7), 1.5)
        return_rate = RNG.uniform(0.15, 0.32)
        refund_claim_speed_days = RNG.uniform(2, 5)
    else:
        pct_new_buyers = RNG.uniform(0.3, 0.6)
        n_distinct_states = int(RNG.integers(4, 15))
        mean_delivery_days = max(RNG.normal(11, 3), 1.5)
        return_rate = RNG.uniform(0.01, 0.09)
        refund_claim_speed_days = RNG.uniform(5, 15)

    return Spike(
        seller_id=seller_id,
        spike_type="organic",
        start_day=start_day,
        duration_days=duration,
        baseline_daily_orders=baseline,
        peak_daily_orders=baseline * peak_multiplier,
        n_orders=n_orders,
        pct_new_buyers=pct_new_buyers,
        n_distinct_states=n_distinct_states,
        mean_delivery_days=mean_delivery_days,
        std_delivery_days=RNG.uniform(5, 10) if not is_hard_case else RNG.uniform(1.5, 4),
        return_rate=return_rate,
        refund_claim_speed_days=refund_claim_speed_days,
        refund_claim_speed_std=RNG.uniform(2, 6) if not is_hard_case else RNG.uniform(0.5, 2),
    )


def simulate_anomalous_spike(seller_id, start_day):
    """A return-abuse ring: sudden ramp, buyer/address clustering, near-instant fake returns."""
    baseline = RNG.uniform(0.8, 2.0)
    peak_multiplier = RNG.uniform(15, 60)              # sudden, extreme jump (Meesho: 2500 orders in months from near-zero)
    duration = int(RNG.integers(1, 3))                 # sharper, more compressed than organic
    n_orders = int(peak_multiplier * baseline * duration * RNG.uniform(0.9, 1.1))
    n_orders = max(n_orders, 15)

    # occasional "hard" anomalous case: a smaller, less extreme ring that
    # partially mimics organic behavior (wider geo spread, slower fake
    # delivery/return cycle) -- a less sloppy fraud ring, closer to the
    # detection boundary. Ranges here deliberately overlap the organic
    # hard-case ranges above, so no single feature perfectly separates
    # the classes -- the model has to combine several signals.
    is_hard_case = RNG.random() < 0.2
    if is_hard_case:
        peak_multiplier = RNG.uniform(5, 12)
        pct_new_buyers = RNG.uniform(0.6, 0.85)
        n_distinct_states = int(RNG.integers(2, 6))
        mean_delivery_days = RNG.uniform(1.5, 5)
        return_rate = RNG.uniform(0.2, 0.45)
        refund_claim_speed_days = RNG.uniform(1.5, 4)
    else:
        pct_new_buyers = RNG.uniform(0.85, 1.0)
        n_distinct_states = int(RNG.integers(1, 3))
        mean_delivery_days = RNG.uniform(0.2, 1.5)
        return_rate = RNG.uniform(0.75, 1.0)
        refund_claim_speed_days = RNG.uniform(0.1, 1.0)

    return Spike(
        seller_id=seller_id,
        spike_type="anomalous",
        start_day=start_day,
        duration_days=duration,
        baseline_daily_orders=baseline,
        peak_daily_orders=baseline * peak_multiplier,
        n_orders=n_orders,
        pct_new_buyers=pct_new_buyers,
        n_distinct_states=n_distinct_states,
        mean_delivery_days=mean_delivery_days,
        std_delivery_days=RNG.uniform(0.1, 1.0),
        return_rate=return_rate,
        refund_claim_speed_days=refund_claim_speed_days,
        refund_claim_speed_std=RNG.uniform(0.05, 0.5),
    )


def spike_to_features(spike: Spike) -> dict:
    """Convert a simulated spike into the feature row a classifier will see.

    Adds realistic per-spike noise so features aren't perfectly separable —
    a real classifier has to learn a boundary, not just read off the label.
    """
    ramp_rate = spike.peak_daily_orders / max(spike.baseline_daily_orders, 0.1)
    ramp_rate *= RNG.uniform(0.85, 1.15)

    # normalized geographic entropy proxy: distinct states relative to a log of order count
    geo_entropy = spike.n_distinct_states / np.log2(spike.n_orders + 2)
    geo_entropy *= RNG.uniform(0.85, 1.15)

    return {
        "seller_id": spike.seller_id,
        "spike_type": spike.spike_type,   # label
        "n_orders": max(int(spike.n_orders + RNG.normal(0, spike.n_orders * 0.08)), 1),
        "duration_days": spike.duration_days,
        "ramp_rate_vs_baseline": round(max(ramp_rate, 0.1), 3),
        "pct_new_buyers": round(np.clip(spike.pct_new_buyers + RNG.normal(0, 0.06), 0, 1), 3),
        "n_distinct_states": max(int(spike.n_distinct_states + RNG.normal(0, 0.8)), 1),
        "geo_entropy": round(max(geo_entropy, 0.01), 3),
        "mean_delivery_days": round(max(spike.mean_delivery_days + RNG.normal(0, 1.2), 0.05), 3),
        "std_delivery_days": round(max(spike.std_delivery_days + RNG.normal(0, 0.3), 0.05), 3),
        "return_rate": round(np.clip(spike.return_rate + RNG.normal(0, 0.06), 0, 1), 3),
        "refund_claim_speed_days": round(max(spike.refund_claim_speed_days + RNG.normal(0, 0.8), 0.05), 3),
        "refund_claim_speed_std": round(max(spike.refund_claim_speed_std, 0.02), 3),
    }


def generate_dataset(n_sellers=N_SELLERS, n_days=N_DAYS):
    rows = []
    for seller_id in range(n_sellers):
        # each seller may get 0, 1, or 2 spikes across the simulation window
        if RNG.random() < ORGANIC_SPIKE_RATE:
            start_day = int(RNG.integers(0, n_days - 10))
            rows.append(spike_to_features(simulate_organic_spike(seller_id, start_day)))
        if RNG.random() < ANOMALOUS_SPIKE_RATE:
            start_day = int(RNG.integers(0, n_days - 10))
            rows.append(spike_to_features(simulate_anomalous_spike(seller_id, start_day)))

    df = pd.DataFrame(rows)
    df = add_overlap_and_label_noise(df)
    return df


# typical central values per class, used as blend targets below (rough
# midpoints of each class's normal range, not tied to any single spike)
BLEND_TARGETS = {
    "organic":   {"ramp_rate_vs_baseline": 4.0, "pct_new_buyers": 0.45, "n_distinct_states": 8,
                  "mean_delivery_days": 10.5, "return_rate": 0.05, "refund_claim_speed_days": 9.0},
    "anomalous": {"ramp_rate_vs_baseline": 30.0, "pct_new_buyers": 0.88, "n_distinct_states": 2,
                  "mean_delivery_days": 1.2, "return_rate": 0.8, "refund_claim_speed_days": 0.8},
}
BLENDABLE_FEATURES = list(BLEND_TARGETS["organic"].keys())


def add_overlap_and_label_noise(df):
    """Make the two classes genuinely overlap in feature space, not just on
    paper. Without this, correlated 'hard case' clusters still form two
    separable groups even though each individual feature's range overlaps --
    real classifiers get near-perfect scores on data like that, which is a
    red flag rather than a good result. Two adjustments:

    1. For a random subset of spikes, independently blend a random handful
       of features partway toward the OPPOSITE class's typical values. Doing
       this per-feature and independently (rather than as one correlated
       cluster) breaks the multivariate separability.
    2. Flip a small fraction of labels outright, simulating real-world
       annotation ambiguity (cases where even a human reviewer would
       disagree on how to classify the spike).
    """
    df = df.copy()
    n = len(df)

    # 1. independent per-feature blending toward the opposite class
    blend_mask = RNG.random(n) < 0.35
    for idx in df.index[blend_mask]:
        row_type = df.at[idx, "spike_type"]
        opposite_type = "anomalous" if row_type == "organic" else "organic"
        n_features_to_blend = RNG.integers(2, 5)
        chosen = RNG.choice(BLENDABLE_FEATURES, size=n_features_to_blend, replace=False)
        for feat in chosen:
            w = RNG.uniform(0.25, 0.55)
            target = BLEND_TARGETS[opposite_type][feat]
            current = df.at[idx, feat]
            blended = current * (1 - w) + target * w
            # keep values in sane ranges
            if feat in ("pct_new_buyers", "return_rate"):
                blended = float(np.clip(blended, 0, 1))
            elif feat == "n_distinct_states":
                blended = max(int(round(blended)), 1)
            else:
                blended = max(blended, 0.05)
            df.at[idx, feat] = round(blended, 3) if isinstance(blended, float) else blended

    # 2. small amount of outright label noise (ambiguous ground truth)
    flip_mask = RNG.random(n) < 0.03
    df.loc[flip_mask, "spike_type"] = df.loc[flip_mask, "spike_type"].map(
        {"organic": "anomalous", "anomalous": "organic"}
    )

    return df


if __name__ == "__main__":
    df = generate_dataset()
    out_path = "/home/claude/spikeguard/spikeguard_spikes.csv"
    df.to_csv(out_path, index=False)

    print(f"Generated {len(df)} labeled spikes -> {out_path}")
    print()
    print("Label distribution:")
    print(df["spike_type"].value_counts())
    print(f"Anomalous share: {(df['spike_type']=='anomalous').mean():.1%}")
    print()
    print("Feature summary by class:")
    print(df.groupby("spike_type").mean(numeric_only=True).round(2).T)

    # stratified train/test split, saved separately so the modeling step
    # never touches held-out rows during feature/threshold tuning
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(
        df, test_size=0.25, stratify=df["spike_type"], random_state=42
    )
    train_df.to_csv("/home/claude/spikeguard/spikeguard_train.csv", index=False)
    test_df.to_csv("/home/claude/spikeguard/spikeguard_test.csv", index=False)
    print(f"\nTrain: {len(train_df)} rows -> spikeguard_train.csv")
    print(f"Test:  {len(test_df)} rows -> spikeguard_test.csv")
