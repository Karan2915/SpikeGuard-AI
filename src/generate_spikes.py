"""
SpikeGuard synthetic data generator - V2.

Purpose
-------
Generate a realistic synthetic merchant-risk dataset for SpikeGuard.

SpikeGuard answers one question:

    "Is a sudden transaction-volume spike more consistent with
     an organic sales event or an anomalous return/refund pattern?"

The dataset contains two classes:

    organic
        Legitimate business events such as flash sales, promotions,
        product launches and festival demand.

    anomalous
        Suspicious compressed transaction spikes showing combinations
        of unusual volume acceleration, buyer concentration,
        geographic concentration and abnormal return/refund behavior.

IMPORTANT
---------
This is SYNTHETIC data.

No real merchant, buyer, payment or transaction records are used.
Publicly reported patterns are used only as inspiration for simulation
ranges. The generated records do not represent any real merchant.

The generator deliberately creates overlap between the two classes.
A legitimate merchant can look suspicious on some dimensions, while
an anomalous spike can look relatively normal on others.

This prevents the classifier from learning a trivial rule such as:

    "high order volume = fraud"

Instead, the model must combine multiple behavioral signals.
"""

import numpy as np
import pandas as pd

from dataclasses import dataclass
from pathlib import Path
from sklearn.model_selection import train_test_split


# ============================================================
# PATHS
# ============================================================

# Project structure:
#
# SpikeGuard_AI/
# ├── Datasets/
# └── src/
#     └── generate_spikes.py

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "Datasets"

DATA_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# RANDOMNESS
# ============================================================

RNG = np.random.default_rng(42)


# ============================================================
# SIMULATION SETTINGS
# ============================================================

N_SELLERS = 4000
N_DAYS = 180

# Probability that a seller experiences each type of spike.
ORGANIC_SPIKE_RATE = 0.40
ANOMALOUS_SPIKE_RATE = 0.16

N_STATES = 27


# ============================================================
# SPIKE DATA STRUCTURE
# ============================================================

@dataclass
class Spike:

    seller_id: int

    spike_type: str

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


# ============================================================
# HELPER
# ============================================================

def bounded_normal(mean, std, low, high):
    """
    Draw a normally distributed value and constrain it
    to a realistic range.
    """
    value = RNG.normal(mean, std)
    return float(np.clip(value, low, high))


# ============================================================
# ORGANIC SPIKES
# ============================================================

def simulate_organic_spike(seller_id, start_day):
    """
    Simulate a legitimate sales spike.

    Examples:
        - flash sale
        - festival demand
        - influencer promotion
        - new product launch
        - seasonal demand

    Organic spikes can still be large and fast.

    This is important because a real risk system should NOT
    simply assume that every large volume increase is suspicious.
    """

    baseline = RNG.uniform(1.0, 3.0)

    event_type = RNG.choice(
        [
            "normal_promotion",
            "flash_sale",
            "festival",
            "product_launch",
            "viral_promotion",
        ],
        p=[0.28, 0.20, 0.22, 0.15, 0.15],
    )

    # --------------------------------------------------------
    # Organic volume behavior
    # --------------------------------------------------------

    if event_type == "normal_promotion":

        peak_multiplier = RNG.uniform(2.5, 6.0)
        duration = int(RNG.integers(2, 7))

        pct_new_buyers = RNG.uniform(0.25, 0.65)
        n_states = int(RNG.integers(4, 16))

        delivery_days = bounded_normal(
            10.5, 3.0, 3.0, 18.0
        )

        return_rate = RNG.uniform(0.01, 0.12)

        refund_speed = RNG.uniform(5, 15)

    elif event_type == "flash_sale":

        peak_multiplier = RNG.uniform(5.0, 12.0)
        duration = int(RNG.integers(1, 3))

        pct_new_buyers = RNG.uniform(0.45, 0.82)
        n_states = int(RNG.integers(3, 12))

        delivery_days = bounded_normal(
            8.0, 2.5, 2.5, 15.0
        )

        return_rate = RNG.uniform(0.04, 0.22)

        refund_speed = RNG.uniform(4, 12)

    elif event_type == "festival":

        peak_multiplier = RNG.uniform(4.0, 10.0)
        duration = int(RNG.integers(2, 6))

        pct_new_buyers = RNG.uniform(0.40, 0.75)
        n_states = int(RNG.integers(6, 20))

        delivery_days = bounded_normal(
            11.0, 3.0, 4.0, 20.0
        )

        return_rate = RNG.uniform(0.03, 0.18)

        refund_speed = RNG.uniform(5, 14)

    elif event_type == "product_launch":

        peak_multiplier = RNG.uniform(4.0, 9.0)
        duration = int(RNG.integers(2, 5))

        pct_new_buyers = RNG.uniform(0.50, 0.85)
        n_states = int(RNG.integers(4, 15))

        delivery_days = bounded_normal(
            9.0, 2.5, 3.0, 17.0
        )

        return_rate = RNG.uniform(0.02, 0.15)

        refund_speed = RNG.uniform(5, 13)

    else:  # viral promotion

        peak_multiplier = RNG.uniform(6.0, 15.0)
        duration = int(RNG.integers(1, 4))

        pct_new_buyers = RNG.uniform(0.55, 0.90)
        n_states = int(RNG.integers(5, 18))

        delivery_days = bounded_normal(
            7.5, 2.5, 2.0, 15.0
        )

        return_rate = RNG.uniform(0.05, 0.25)

        refund_speed = RNG.uniform(3, 12)

    # --------------------------------------------------------
    # Orders
    # --------------------------------------------------------

    n_orders = int(
        peak_multiplier
        * baseline
        * duration
        * RNG.uniform(0.85, 1.15)
    )

    n_orders = max(n_orders, 8)

    # --------------------------------------------------------
    # Organic "difficult" cases
    # --------------------------------------------------------
    #
    # Some legitimate spikes intentionally look suspicious.
    # This is important for reducing false positives.

    if RNG.random() < 0.22:

        peak_multiplier = RNG.uniform(7.0, 18.0)

        duration = int(RNG.integers(1, 4))

        pct_new_buyers = RNG.uniform(0.65, 0.92)

        n_states = int(RNG.integers(3, 10))

        delivery_days = bounded_normal(
            5.5, 2.0, 2.0, 11.0
        )

        return_rate = RNG.uniform(0.10, 0.35)

        refund_speed = RNG.uniform(2.5, 9)

        n_orders = int(
            peak_multiplier
            * baseline
            * duration
            * RNG.uniform(0.85, 1.15)
        )

        n_orders = max(n_orders, 10)

    return Spike(
        seller_id=seller_id,
        spike_type="organic",

        start_day=start_day,
        duration_days=duration,

        baseline_daily_orders=baseline,
        peak_daily_orders=baseline * peak_multiplier,

        n_orders=n_orders,

        pct_new_buyers=pct_new_buyers,

        n_distinct_states=n_states,

        mean_delivery_days=delivery_days,

        std_delivery_days=RNG.uniform(
            2.5, 8.5
        ),

        return_rate=return_rate,

        refund_claim_speed_days=refund_speed,

        refund_claim_speed_std=RNG.uniform(
            1.0, 5.0
        ),
    )


# ============================================================
# ANOMALOUS SPIKES
# ============================================================

def simulate_anomalous_spike(seller_id, start_day):
    """
    Simulate a suspicious transaction spike.

    The anomaly is NOT defined by volume alone.

    Instead it can combine:

        - sudden acceleration
        - unusually high first-time buyers
        - geographic concentration
        - unusually fast delivery/return cycle
        - high return rate
        - rapid refund claims

    Some anomalous cases are intentionally subtle.
    """

    baseline = RNG.uniform(1.0, 3.0)

    anomaly_type = RNG.choice(
        [
            "strong_ring",
            "moderate_ring",
            "subtle_ring",
        ],
        p=[0.45, 0.35, 0.20],
    )

    # --------------------------------------------------------
    # STRONG ANOMALOUS CASE
    # --------------------------------------------------------

    if anomaly_type == "strong_ring":

        peak_multiplier = RNG.uniform(12.0, 35.0)

        duration = int(RNG.integers(1, 3))

        pct_new_buyers = RNG.uniform(0.78, 0.99)

        n_states = int(RNG.integers(1, 5))

        delivery_days = RNG.uniform(0.5, 3.0)

        return_rate = RNG.uniform(0.65, 0.98)

        refund_speed = RNG.uniform(0.2, 2.0)

        delivery_std = RNG.uniform(0.1, 1.5)

        refund_std = RNG.uniform(0.05, 0.7)

    # --------------------------------------------------------
    # MODERATE ANOMALOUS CASE
    # --------------------------------------------------------

    elif anomaly_type == "moderate_ring":

        peak_multiplier = RNG.uniform(7.0, 22.0)

        duration = int(RNG.integers(1, 4))

        pct_new_buyers = RNG.uniform(0.65, 0.93)

        n_states = int(RNG.integers(2, 8))

        delivery_days = RNG.uniform(1.5, 6.0)

        return_rate = RNG.uniform(0.35, 0.85)

        refund_speed = RNG.uniform(0.8, 4.0)

        delivery_std = RNG.uniform(0.3, 3.0)

        refund_std = RNG.uniform(0.1, 1.5)

    # --------------------------------------------------------
    # SUBTLE ANOMALOUS CASE
    # --------------------------------------------------------

    else:

        peak_multiplier = RNG.uniform(5.0, 14.0)

        duration = int(RNG.integers(2, 5))

        pct_new_buyers = RNG.uniform(0.55, 0.85)

        n_states = int(RNG.integers(3, 10))

        delivery_days = bounded_normal(
            5.0, 2.0, 1.0, 10.0
        )

        return_rate = RNG.uniform(0.20, 0.60)

        refund_speed = RNG.uniform(2.0, 7.0)

        delivery_std = RNG.uniform(1.0, 4.5)

        refund_std = RNG.uniform(0.5, 3.0)

    # --------------------------------------------------------
    # Orders
    # --------------------------------------------------------

    n_orders = int(
        peak_multiplier
        * baseline
        * duration
        * RNG.uniform(0.85, 1.15)
    )

    n_orders = max(n_orders, 12)

    return Spike(
        seller_id=seller_id,
        spike_type="anomalous",

        start_day=start_day,
        duration_days=duration,

        baseline_daily_orders=baseline,
        peak_daily_orders=baseline * peak_multiplier,

        n_orders=n_orders,

        pct_new_buyers=pct_new_buyers,

        n_distinct_states=n_states,

        mean_delivery_days=delivery_days,

        std_delivery_days=delivery_std,

        return_rate=return_rate,

        refund_claim_speed_days=refund_speed,

        refund_claim_speed_std=refund_std,
    )


# ============================================================
# CONVERT SPIKE TO MODEL FEATURES
# ============================================================

def spike_to_features(spike: Spike) -> dict:
    """
    Convert a simulated spike into the exact feature representation
    used by the machine-learning model.

    Noise is deliberately added so the classes do not become
    perfectly separable.
    """

    # --------------------------------------------------------
    # Volume acceleration
    # --------------------------------------------------------

    ramp_rate = (
        spike.peak_daily_orders
        / max(spike.baseline_daily_orders, 0.1)
    )

    ramp_rate *= RNG.uniform(0.88, 1.12)

    # --------------------------------------------------------
    # Geographic entropy proxy
    # --------------------------------------------------------

    geo_entropy = (
        spike.n_distinct_states
        / np.log2(spike.n_orders + 2)
    )

    geo_entropy *= RNG.uniform(0.88, 1.12)

    # --------------------------------------------------------
    # Feature noise
    # --------------------------------------------------------

    n_orders = int(
        spike.n_orders
        + RNG.normal(
            0,
            max(spike.n_orders * 0.08, 1)
        )
    )

    pct_new_buyers = (
        spike.pct_new_buyers
        + RNG.normal(0, 0.045)
    )

    n_states = int(
        spike.n_distinct_states
        + RNG.normal(0, 0.7)
    )

    mean_delivery = (
        spike.mean_delivery_days
        + RNG.normal(0, 0.8)
    )

    std_delivery = (
        spike.std_delivery_days
        + RNG.normal(0, 0.25)
    )

    return_rate = (
        spike.return_rate
        + RNG.normal(0, 0.045)
    )

    refund_speed = (
        spike.refund_claim_speed_days
        + RNG.normal(0, 0.55)
    )

    # --------------------------------------------------------
    # Final row
    # --------------------------------------------------------

    return {

        "seller_id":
            spike.seller_id,

        "spike_type":
            spike.spike_type,

        "start_day":
            spike.start_day,

        "n_orders":
            max(n_orders, 1),

        "duration_days":
            spike.duration_days,

        "ramp_rate_vs_baseline":
            round(max(ramp_rate, 0.1), 3),

        "pct_new_buyers":
            round(
                float(np.clip(
                    pct_new_buyers,
                    0,
                    1
                )),
                3
            ),

        "n_distinct_states":
            max(n_states, 1),

        "geo_entropy":
            round(
                max(geo_entropy, 0.01),
                3
            ),

        "mean_delivery_days":
            round(
                max(mean_delivery, 0.05),
                3
            ),

        "std_delivery_days":
            round(
                max(std_delivery, 0.05),
                3
            ),

        "return_rate":
            round(
                float(np.clip(
                    return_rate,
                    0,
                    1
                )),
                3
            ),

        "refund_claim_speed_days":
            round(
                max(refund_speed, 0.05),
                3
            ),

        "refund_claim_speed_std":
            round(
                max(
                    spike.refund_claim_speed_std
                    + RNG.normal(0, 0.15),
                    0.02
                ),
                3
            ),
    }


# ============================================================
# CLASS OVERLAP
# ============================================================

BLEND_TARGETS = {

    "organic": {

        "ramp_rate_vs_baseline": 6.0,

        "pct_new_buyers": 0.55,

        "n_distinct_states": 9,

        "mean_delivery_days": 9.0,

        "return_rate": 0.10,

        "refund_claim_speed_days": 8.0,
    },

    "anomalous": {

        "ramp_rate_vs_baseline": 15.0,

        "pct_new_buyers": 0.78,

        "n_distinct_states": 4,

        "mean_delivery_days": 4.0,

        "return_rate": 0.55,

        "refund_claim_speed_days": 3.0,
    },
}


BLENDABLE_FEATURES = list(
    BLEND_TARGETS["organic"].keys()
)


def add_overlap_and_label_noise(df):
    """
    Make the classification problem harder and more realistic.

    Two mechanisms are used:

    1. Feature overlap:
       Some individual features are shifted toward the opposite class.

    2. Label ambiguity:
       A small fraction of labels are flipped to represent
       borderline cases where the true classification is uncertain.

    This means there is no perfect deterministic rule separating
    organic and anomalous spikes.
    """

    df = df.copy()

    n = len(df)

    # --------------------------------------------------------
    # Feature overlap
    # --------------------------------------------------------

    blend_mask = RNG.random(n) < 0.40

    for idx in df.index[blend_mask]:

        row_type = df.at[idx, "spike_type"]

        opposite_type = (
            "anomalous"
            if row_type == "organic"
            else "organic"
        )

        # Blend between 2 and 4 features.

        n_features = int(
            RNG.integers(2, 5)
        )

        chosen_features = RNG.choice(
            BLENDABLE_FEATURES,
            size=n_features,
            replace=False,
        )

        for feature in chosen_features:

            weight = RNG.uniform(
                0.20,
                0.50
            )

            target = BLEND_TARGETS[
                opposite_type
            ][feature]

            current = df.at[
                idx,
                feature
            ]

            blended = (
                current * (1 - weight)
                + target * weight
            )

            # Keep values realistic.

            if feature in (
                "pct_new_buyers",
                "return_rate",
            ):

                blended = float(
                    np.clip(
                        blended,
                        0,
                        1
                    )
                )

            elif feature == "n_distinct_states":

                blended = max(
                    int(round(blended)),
                    1
                )

            else:

                blended = max(
                    float(blended),
                    0.05
                )

            df.at[
                idx,
                feature
            ] = round(
                blended,
                3
            )

    # --------------------------------------------------------
    # Label ambiguity
    # --------------------------------------------------------

    flip_mask = RNG.random(n) < 0.025

    df.loc[
        flip_mask,
        "spike_type"
    ] = (
        df.loc[
            flip_mask,
            "spike_type"
        ].map(
            {
                "organic": "anomalous",
                "anomalous": "organic",
            }
        )
    )

    return df


# ============================================================
# DATASET GENERATION
# ============================================================

def generate_dataset(
    n_sellers=N_SELLERS,
    n_days=N_DAYS
):
    """
    Generate the complete synthetic dataset.
    """

    rows = []

    for seller_id in range(
        n_sellers
    ):

        # ----------------------------------------------------
        # Organic spike
        # ----------------------------------------------------

        if RNG.random() < ORGANIC_SPIKE_RATE:

            start_day = int(
                RNG.integers(
                    0,
                    max(n_days - 10, 1)
                )
            )

            spike = simulate_organic_spike(
                seller_id,
                start_day
            )

            rows.append(
                spike_to_features(spike)
            )

        # ----------------------------------------------------
        # Anomalous spike
        # ----------------------------------------------------

        if RNG.random() < ANOMALOUS_SPIKE_RATE:

            start_day = int(
                RNG.integers(
                    0,
                    max(n_days - 10, 1)
                )
            )

            spike = simulate_anomalous_spike(
                seller_id,
                start_day
            )

            rows.append(
                spike_to_features(spike)
            )

    df = pd.DataFrame(rows)

    # Add realistic overlap and ambiguity.

    df = add_overlap_and_label_noise(df)

    return df


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("SPIKEGUARD V2 - SYNTHETIC DATA GENERATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Generate
    # --------------------------------------------------------

    df = generate_dataset()

    full_path = (
        DATA_DIR
        / "spikeguard_spikes.csv"
    )

    df.to_csv(
        full_path,
        index=False
    )

    # --------------------------------------------------------
    # Basic statistics
    # --------------------------------------------------------

    print()
    print(
        f"Generated {len(df)} spike records."
    )

    print(
        f"Saved to: {full_path}"
    )

    print()
    print("=" * 70)
    print("CLASS DISTRIBUTION")
    print("=" * 70)

    print(
        df["spike_type"].value_counts()
    )

    anomalous_share = (
        df["spike_type"]
        .eq("anomalous")
        .mean()
    )

    print()
    print(
        f"Anomalous share: "
        f"{anomalous_share:.1%}"
    )

    # --------------------------------------------------------
    # Feature statistics
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FEATURE MEANS BY CLASS")
    print("=" * 70)

    summary = (
        df.groupby("spike_type")
        .mean(
            numeric_only=True
        )
        .round(2)
        .T
    )

    print(summary)

    # --------------------------------------------------------
    # Train / test split
    # --------------------------------------------------------

    train_df, test_df = train_test_split(

        df,

        test_size=0.25,

        stratify=df["spike_type"],

        random_state=42,
    )

    train_path = (
        DATA_DIR
        / "spikeguard_train.csv"
    )

    test_path = (
        DATA_DIR
        / "spikeguard_test.csv"
    )

    train_df.to_csv(
        train_path,
        index=False
    )

    test_df.to_csv(
        test_path,
        index=False
    )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("DATASET SPLIT")
    print("=" * 70)

    print(
        f"Train: {len(train_df)} rows"
    )

    print(
        f"Test : {len(test_df)} rows"
    )

    print()
    print(
        f"Train file: {train_path}"
    )

    print(
        f"Test file : {test_path}"
    )

    print()
    print("=" * 70)
    print("TRAIN CLASS DISTRIBUTION")
    print("=" * 70)

    print(
        train_df["spike_type"]
        .value_counts()
    )

    print()
    print("=" * 70)
    print("TEST CLASS DISTRIBUTION")
    print("=" * 70)

    print(
        test_df["spike_type"]
        .value_counts()
    )

    print()
    print("=" * 70)
    print("GENERATION COMPLETE")
    print("=" * 70)