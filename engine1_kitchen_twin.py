"""
Engine 1: The Kitchen Twin (Discrete Event Simulation)
======================================================
Maps the physical kitchen pipeline for each order.
Groups similar items (batching), calculates queue wait times
based on hardware occupancy (oven/tandoor/fryer/steamer/grill capacity).

Input : Active Cart Items + Hardware Constraints + Batching Logic
Output: Structural Minimum Time (seconds) per order
"""

import pandas as pd
import numpy as np
import random

# ── Reproducibility ──────────────────────────────────────────────────────────
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# ── Load Datasets ────────────────────────────────────────────────────────────
orders_df = pd.read_csv("delhi_ncr_simpy_20k_orders.csv")
kitchens_df = pd.read_csv("delhi_ncr_static_kitchens_v2.csv")

print(f"Loaded {len(orders_df)} orders and {len(kitchens_df)} kitchens.")

# ── Prep-type definitions with base cook-time ranges (seconds) ───────────────
PREP_TYPES = {
    "Tandoor":  {"time_range": (300, 600),  "unit_col": "tandoor_units"},
    "Deep Fry": {"time_range": (180, 420),  "unit_col": "deep_fry_units"},
    "Steamer":  {"time_range": (240, 480),  "unit_col": "steamer_units"},
    "Grill":    {"time_range": (240, 500),  "unit_col": "grill_units"},
    "Assembly": {"time_range": (60, 180),   "unit_col": "assembly_units"},
}

PREP_NAMES = list(PREP_TYPES.keys())

# ── Assign each order to a random merchant ───────────────────────────────────
merchant_ids = kitchens_df["merchant_id"].values
orders_df["merchant_id"] = np.random.choice(merchant_ids, size=len(orders_df))

# Merge kitchen hardware info onto orders
orders_df = orders_df.merge(kitchens_df, on="merchant_id", how="left")


# ── Batching logic ───────────────────────────────────────────────────────────
def batch_items(cart_size):
    """
    Randomly assign cart items to prep types, then group (batch) identical
    prep types together  —  e.g. if an order has 3 fries they share one fryer
    basket instead of 3 sequential uses.
    """
    items = [random.choice(PREP_NAMES) for _ in range(cart_size)]
    # Group by prep type with counts
    batches = {}
    for item in items:
        batches[item] = batches.get(item, 0) + 1
    return batches


# ── Simulate kitchen twin for every order ────────────────────────────────────
def compute_structural_minimum(row):
    """
    For one order, simulate the physical kitchen pipeline:
      1. Batch identical prep types
      2. For each batch: cook_time / hardware_units  (parallel capacity)
      3. Add a small queue-wait penalty proportional to global order volume
      4. Scale by historical_speed_index (slower kitchens > 1.0)
      5. Result = Structural Minimum Time (the physics-limited floor)
    """
    cart_items = int(row["cart_items"])
    batches = batch_items(cart_items)

    total_time = 0.0

    for prep_type, count in batches.items():
        cfg = PREP_TYPES[prep_type]
        lo, hi = cfg["time_range"]

        # Base cook time for the batch (one cycle covers the whole batch)
        base_cook = np.random.randint(lo, hi)

        # Hardware parallelism: more units → faster throughput
        units = int(row[cfg["unit_col"]])

        # Effective cook time  =  base / units  (batched items share capacity)
        # If batch count > units, we need ceil(count/units) cycles
        cycles = int(np.ceil(count / units))
        effective_cook = base_cook * cycles

        # Queue-wait penalty: approximate contention from concurrent orders
        # Using a lightweight heuristic instead of full SimPy here
        # (the original DES already produced the raw structural_time_sec)
        queue_penalty = np.random.uniform(5, 30) * (cart_items - 1)

        total_time += effective_cook + queue_penalty

    # Scale by kitchen speed index (>1 = historically slower kitchen)
    speed_idx = float(row["historical_speed_index"])
    total_time *= speed_idx

    return round(total_time, 2)


orders_df["structural_minimum_time_sec"] = orders_df.apply(
    compute_structural_minimum, axis=1
)

# ── Output ───────────────────────────────────────────────────────────────────
output_cols = [
    "order_id",
    "merchant_id",
    "zone",
    "cart_items",
    "structural_minimum_time_sec",
    "historical_speed_index",
]

output_df = orders_df[output_cols].copy()
output_df.to_csv("engine1_output.csv", index=False)

print(f"\n✅  Engine 1 complete — {len(output_df)} rows written to engine1_output.csv")
print(output_df.describe())
