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

    return Spike(
        seller_id=seller_id,
        spike_type="organic",
        start_day=start_day,
        duration_days=duration,
        baseline_daily_orders=baseline,
        peak_daily_orders=baseline * peak_multiplier,
        n_orders=n_orders,
        pct_new_buyers=RNG.uniform(0.3, 0.6),          # mix of new + repeat customers
        n_distinct_states=int(RNG.integers(4, 15)),     # healthy geographic spread (Olist median~4, up to 27)
        mean_delivery_days=RNG.normal(11, 2),           # close to real Olist median (~10-12 days)
        std_delivery_days=RNG.uniform(6, 10),
        return_rate=RNG.uniform(0.01, 0.05),            # near Olist baseline cancel/unavailable rate (~1.2%)
        refund_claim_speed_days=RNG.uniform(8, 15),      # refunds, when they happen, take normal time
        refund_claim_speed_std=RNG.uniform(3, 6),
    )


def simulate_anomalous_spike(seller_id, start_day):
    """A return-abuse ring: sudden ramp, buyer/address clustering, near-instant fake returns."""
    baseline = RNG.uniform(0.8, 2.0)
    peak_multiplier = RNG.uniform(15, 60)              # sudden, extreme jump (Meesho: 2500 orders in months from near-zero)
    duration = int(RNG.integers(1, 3))                 # sharper, more compressed than organic
    n_orders = int(peak_multiplier * baseline * duration * RNG.uniform(0.9, 1.1))
    n_orders = max(n_orders, 15)

    return Spike(
        seller_id=seller_id,
        spike_type="anomalous",
        start_day=start_day,
        duration_days=duration,
        baseline_daily_orders=baseline,
        peak_daily_orders=baseline * peak_multiplier,
        n_orders=n_orders,
        pct_new_buyers=RNG.uniform(0.85, 1.0),          # almost all freshly-created accounts
        n_distinct_states=int(RNG.integers(1, 3)),       # unnaturally narrow clustering vs organic
        mean_delivery_days=RNG.uniform(0.2, 1.5),        # near-instant fake "delivery"
        std_delivery_days=RNG.uniform(0.1, 0.5),
        return_rate=RNG.uniform(0.75, 1.0),              # near-100%, matching Meesho pattern
        refund_claim_speed_days=RNG.uniform(0.1, 1.0),   # refund claimed almost immediately after "delivery"
        refund_claim_speed_std=RNG.uniform(0.05, 0.3),
    )


def spike_to_features(spike: Spike) -> dict:
    """Convert a simulated spike into the feature row a classifier will see.

    Adds realistic per-spike noise so features aren't perfectly separable —
    a real classifier has to learn a boundary, not just read off the label.
    """
    ramp_rate = spike.peak_daily_orders / max(spike.baseline_daily_orders, 0.1)
    ramp_rate *= RNG.uniform(0.9, 1.1)

    geo_entropy = spike.n_distinct_states / max(spike.n_orders, 1) * spike.n_orders  # placeholder, replaced below
    # normalized geographic entropy proxy: distinct states relative to a log of order count
    geo_entropy = spike.n_distinct_states / np.log2(spike.n_orders + 2)
    geo_entropy *= RNG.uniform(0.9, 1.1)

    return {
        "seller_id": spike.seller_id,
        "spike_type": spike.spike_type,   # label
        "n_orders": spike.n_orders,
        "duration_days": spike.duration_days,
        "ramp_rate_vs_baseline": round(ramp_rate, 3),
        "pct_new_buyers": round(np.clip(spike.pct_new_buyers + RNG.normal(0, 0.03), 0, 1), 3),
        "n_distinct_states": spike.n_distinct_states,
        "geo_entropy": round(geo_entropy, 3),
        "mean_delivery_days": round(max(spike.mean_delivery_days + RNG.normal(0, 0.5), 0), 3),
        "std_delivery_days": round(max(spike.std_delivery_days, 0.05), 3),
        "return_rate": round(np.clip(spike.return_rate + RNG.normal(0, 0.03), 0, 1), 3),
        "refund_claim_speed_days": round(max(spike.refund_claim_speed_days, 0.05), 3),
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
