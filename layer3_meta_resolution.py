"""
Layer 3: The Meta-Resolution & Truth Layer (The Brain)
=====================================================
Acts as the Supreme Court, weighing model predictions against
volatile human inputs.

Step A: Anomaly Detection (Lie-Detector)
Step B: Non-Linear Aggregation (Meta-Model)
Step C: Margin of Error + Dynamic Buffer

Input : Engine 1 output + Engine 2 output + Engine 3 output + Static kitchen data
Output: Final Predicted KPT with trust scores, ceiling/floor, and buffer
"""

import pandas as pd
import numpy as np
import random

# ── Reproducibility ──────────────────────────────────────────────────────────
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# ── Load All Engine Outputs ──────────────────────────────────────────────────
engine1_df = pd.read_csv("engine1_output.csv")
engine2_df = pd.read_csv("engine2_output.csv")
engine3_df = pd.read_csv("engine3_output.csv")
kitchens_df = pd.read_csv("delhi_ncr_static_kitchens_v2.csv")
orders_df = pd.read_csv("delhi_ncr_simpy_20k_orders.csv")

print(f"Loaded Engine outputs: E1={len(engine1_df)}, E2={len(engine2_df)}, E3={len(engine3_df)}")

# ── Merge all engine outputs on order_id ─────────────────────────────────────
merged = engine1_df[["order_id", "merchant_id", "zone",
                      "structural_minimum_time_sec", "historical_speed_index"]].copy()

merged = merged.merge(
    engine2_df[["order_id", "has_customization", "complexity_delta_sec"]],
    on="order_id", how="left"
)
merged = merged.merge(
    engine3_df[["order_id", "weather", "traffic_index", "rider_density",
                "hour_of_day", "ambient_friction_multiplier"]],
    on="order_id", how="left"
)

# Fill any NaN from merges
merged["complexity_delta_sec"] = merged["complexity_delta_sec"].fillna(0)
merged["ambient_friction_multiplier"] = merged["ambient_friction_multiplier"].fillna(1.0)

print(f"Merged dataset: {len(merged)} orders")

# ═════════════════════════════════════════════════════════════════════════════
# STEP A: Anomaly Detection — The Lie Detector
# ═════════════════════════════════════════════════════════════════════════════
# Simulates "Merchant Marked as Ready" timestamps.
# If a merchant clicks "Ready" in a time that is physically impossible
# (less than the structural minimum), assign a low trust weight.

def simulate_merchant_ready_and_trust(row):
    """
    Simulate the merchant's "Marked as Ready" click.
    Some merchants are honest, some click prematurely.
    """
    structural_min = row["structural_minimum_time_sec"]

    # Simulate merchant behavior:
    # 70% honest (mark ready at or after structural min)
    # 20% slightly early (within 80-100% of structural min)
    # 10% suspicious (mark ready at 40-70% of structural min — lying)
    behavior = np.random.choice(
        ["honest", "slightly_early", "suspicious"],
        p=[0.70, 0.20, 0.10]
    )

    if behavior == "honest":
        # Marked ready at structural_min + some realistic buffer
        marked_ready_sec = structural_min + np.random.uniform(0, 60)
        trust_weight = 1.0
    elif behavior == "slightly_early":
        # Marked ready slightly before food is actually done
        marked_ready_sec = structural_min * np.random.uniform(0.80, 0.99)
        trust_weight = 0.75
    else:
        # Suspicious: impossible timing
        marked_ready_sec = structural_min * np.random.uniform(0.40, 0.70)
        trust_weight = 0.30  # Low trust for this session

    # Rule: If marked_ready < structural_minimum * 0.75, it's physically impossible
    if marked_ready_sec < structural_min * 0.75:
        trust_weight = min(trust_weight, 0.25)
        is_anomaly = True
    else:
        is_anomaly = False

    return pd.Series({
        "merchant_marked_ready_sec": round(marked_ready_sec, 2),
        "trust_weight": round(trust_weight, 2),
        "is_anomaly": is_anomaly,
    })


print("\n── Step A: Running Anomaly Detection (Lie Detector) ──")
anomaly_results = merged.apply(simulate_merchant_ready_and_trust, axis=1)
merged = pd.concat([merged, anomaly_results], axis=1)

anomaly_count = merged["is_anomaly"].sum()
print(f"   Anomalies detected: {anomaly_count} ({anomaly_count/len(merged)*100:.1f}%)")
print(f"   Mean trust weight: {merged['trust_weight'].mean():.3f}")


# ═════════════════════════════════════════════════════════════════════════════
# STEP B: Non-Linear Aggregation — Meta-Model
# ═════════════════════════════════════════════════════════════════════════════
# Combines Engine outputs using a weighted ensemble.
# Trust weight influences how much we rely on merchant signals vs model.

def compute_meta_kpt(row):
    """
    Lightweight meta-model: weighted combination of engine outputs.
    """
    structural = row["structural_minimum_time_sec"]
    customization = row["complexity_delta_sec"]
    friction = row["ambient_friction_multiplier"]
    trust = row["trust_weight"]

    # Base KPT from engines
    # Tightening the basic friction
    base_kpt = (structural + customization) * np.sqrt(friction)

    # Trust adjustment: if merchant is untrustworthy, add a tiny penalty
    trust_penalty = (1.0 - trust) * structural * 0.08  # slight increase to 8%

    raw_kpt = base_kpt + trust_penalty

    # ── Historical accuracy bounds (ceiling & floor) ──
    speed_idx = row["historical_speed_index"]

    # Floor: optimistic estimate
    kpt_floor = raw_kpt * np.interp(speed_idx, [0.7, 1.0, 1.3], [0.90, 0.95, 0.98])

    # Ceiling: pessimistic estimate
    kpt_ceiling = raw_kpt * np.interp(speed_idx, [0.7, 1.0, 1.3], [1.02, 1.05, 1.10])

    return pd.Series({
        "raw_kpt_sec": round(raw_kpt, 2),
        "kpt_floor_sec": round(kpt_floor, 2),
        "kpt_ceiling_sec": round(kpt_ceiling, 2),
    })


print("\n── Step B: Running Non-Linear Aggregation ──")
meta_results = merged.apply(compute_meta_kpt, axis=1)
merged = pd.concat([merged, meta_results], axis=1)

print(f"   Mean raw KPT: {merged['raw_kpt_sec'].mean():.1f} sec "
      f"({merged['raw_kpt_sec'].mean()/60:.1f} min)")
print(f"   KPT range: [{merged['kpt_floor_sec'].mean():.1f}, "
      f"{merged['kpt_ceiling_sec'].mean():.1f}] sec")


# ═════════════════════════════════════════════════════════════════════════════
# STEP C: Margin of Error — Dynamic Buffer
# ═════════════════════════════════════════════════════════════════════════════

def compute_dynamic_buffer(row):
    """
    Dynamic buffer based on predictors. Keep it very tight
    to match the ~80% precision target on buckets.
    """
    speed_idx = row["historical_speed_index"]
    hour = row["hour_of_day"]
    weather = row["weather"]

    predictability = abs(speed_idx - 1.0)

    # Base buffer 
    base_buffer = np.interp(predictability, [0.0, 0.10, 0.20, 0.30],
                                            [0, 10, 20, 35])

    peak_hours = {12, 13, 14, 19, 20, 21, 22}
    peak_amplifier = 1.15 if hour in peak_hours else 1.0

    weather_amp = {
        "Clear": 1.0,
        "Heatwave": 1.05,
        "Smog": 1.02,
        "Fog": 1.05,
        "Rain": 1.10,
    }.get(weather, 1.0)

    buffer_sec = base_buffer * peak_amplifier * weather_amp

    # Clamp tightly
    buffer_sec = max(0, min(50, buffer_sec))

    return round(buffer_sec, 2)


print("\n── Step C: Computing Dynamic Buffers ──")
merged["buffer_sec"] = merged.apply(compute_dynamic_buffer, axis=1)

print(f"   Mean buffer: {merged['buffer_sec'].mean():.1f} sec "
      f"({merged['buffer_sec'].mean()/60:.1f} min)")
print(f"   Buffer range: [{merged['buffer_sec'].min():.0f}, "
      f"{merged['buffer_sec'].max():.0f}] sec")


# ═════════════════════════════════════════════════════════════════════════════
# FINAL: Compute the Final Predicted KPT
# ═════════════════════════════════════════════════════════════════════════════

merged["final_predicted_kpt_sec"] = (merged["raw_kpt_sec"] + merged["buffer_sec"]).round(2)
merged["final_predicted_kpt_min"] = (merged["final_predicted_kpt_sec"] / 60).round(2)

# ── Output ───────────────────────────────────────────────────────────────────
output_cols = [
    "order_id",
    "merchant_id",
    "zone",
    "weather",
    "traffic_index",
    "rider_density",
    "hour_of_day",
    # Engine outputs
    "structural_minimum_time_sec",
    "complexity_delta_sec",
    "ambient_friction_multiplier",
    # Anomaly detection
    "merchant_marked_ready_sec",
    "trust_weight",
    "is_anomaly",
    # Meta-model
    "raw_kpt_sec",
    "kpt_floor_sec",
    "kpt_ceiling_sec",
    # Buffer
    "buffer_sec",
    # Final output
    "final_predicted_kpt_sec",
    "final_predicted_kpt_min",
]

output_df = merged[output_cols].copy()
output_df.to_csv("layer3_final_output.csv", index=False)

# ── Summary Statistics ───────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  LAYER 3 — META-RESOLUTION COMPLETE")
print(f"{'='*65}")
print(f"  Total orders processed:   {len(output_df)}")
print(f"  Anomalous merchants:      {output_df['is_anomaly'].sum()} "
      f"({output_df['is_anomaly'].mean()*100:.1f}%)")
print(f"  Mean trust weight:        {output_df['trust_weight'].mean():.3f}")
print(f"")
print(f"  ┌─ Final Predicted KPT ────────────────────────────┐")
print(f"  │  Mean:   {output_df['final_predicted_kpt_min'].mean():>7.2f} min "
      f"({output_df['final_predicted_kpt_sec'].mean():>8.1f} sec) │")
print(f"  │  Median: {output_df['final_predicted_kpt_min'].median():>7.2f} min "
      f"({output_df['final_predicted_kpt_sec'].median():>8.1f} sec) │")
print(f"  │  Min:    {output_df['final_predicted_kpt_min'].min():>7.2f} min "
      f"({output_df['final_predicted_kpt_sec'].min():>8.1f} sec) │")
print(f"  │  Max:    {output_df['final_predicted_kpt_min'].max():>7.2f} min "
      f"({output_df['final_predicted_kpt_sec'].max():>8.1f} sec) │")
print(f"  └──────────────────────────────────────────────────┘")
print(f"\n  KPT Bounds [Floor → Ceiling]:")
print(f"  Mean: [{output_df['kpt_floor_sec'].mean():.0f}, "
      f"{output_df['kpt_ceiling_sec'].mean():.0f}] sec")
print(f"\n  Buffer distribution:")
print(f"  {output_df['buffer_sec'].describe().to_string()}")
print(f"\n✅  Final output saved → layer3_final_output.csv")
