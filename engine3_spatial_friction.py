"""
Engine 3: The Spatial Friction Tracker (GNN / Temporal Modeler)
==============================================================
Models compounding local delays from live neighborhood variables
specific to Delhi NCR: weather, traffic, rider density, time-of-day,
and zone-level congestion patterns.

Input : Live neighborhood variables (weather, traffic_index, rider_density)
Output: Ambient Friction Multiplier per order
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
print(f"Loaded {len(orders_df)} orders for spatial friction analysis.")

# ── Assign merchants (same seed as Engine 1 for consistency) ─────────────────
np.random.seed(RANDOM_SEED)
merchant_ids = kitchens_df["merchant_id"].values
orders_df["merchant_id"] = np.random.choice(merchant_ids, size=len(orders_df))
orders_df = orders_df.merge(kitchens_df[["merchant_id", "zone"]], on="merchant_id", how="left")

# ─────────────────────────────────────────────────────────────────────────────
# Delhi NCR Weather Impact Model
# ─────────────────────────────────────────────────────────────────────────────
# Weather conditions and their friction multipliers on kitchen/delivery ops
# Delhi-specific: Smog (winter), Heatwave (summer), Fog (Dec-Jan mornings)

WEATHER_FRICTION = {
    "Clear":    {"base": 1.00, "variance": 0.02},
    "Heatwave": {"base": 1.08, "variance": 0.04},   # staff fatigue, AC load
    "Rain":     {"base": 1.18, "variance": 0.06},   # waterlogging in many areas
    "Smog":     {"base": 1.10, "variance": 0.05},   # reduced visibility, slow riders
    "Fog":      {"base": 1.15, "variance": 0.05},   # morning fog in NCR winter
}

# ─────────────────────────────────────────────────────────────────────────────
# Delhi NCR Zone Congestion Profiles
# ─────────────────────────────────────────────────────────────────────────────
# Baseline congestion multipliers per zone (1.0 = neutral)
# Higher = more congested area with parking issues, narrow lanes, etc.

ZONE_CONGESTION = {
    "Connaught Place":   1.20,   # Dense CBD, limited parking
    "Gurgaon CyberHub":  1.15,   # Corporate hub, peak-hour nightmare
    "Noida Sector 18":   1.10,   # Mall area, parking delays
    "Saket":             1.08,   # Select City area, moderate
    "Dwarka":            1.05,   # Residential, decent roads
    "Lajpat Nagar":      1.18,   # Market area, narrow lanes
    "Rohini":            1.06,   # Suburban, decent infra
    "Faridabad":         1.04,   # Outskirts, less congestion
}

# ─────────────────────────────────────────────────────────────────────────────
# Rider Density Impact
# ─────────────────────────────────────────────────────────────────────────────
RIDER_DENSITY_IMPACT = {
    "High":   0.95,   # Many riders → faster pickup, slight positive
    "Medium": 1.00,   # Neutral
    "Low":    1.12,   # Few riders → longer wait, compounding delays
}

# ─────────────────────────────────────────────────────────────────────────────
# Time-of-Day Temporal Model (Delhi NCR patterns)
# ─────────────────────────────────────────────────────────────────────────────
def get_hour_of_day(order_index, total_orders):
    """
    Simulate hour-of-day distribution across orders.
    Orders are distributed across a realistic Delhi NCR ordering pattern:
    - Lunch peak: 12:00-14:00
    - Dinner peak: 19:00-22:00
    - Late night: 22:00-01:00 (still active in NCR)
    """
    # Weighted hour distribution
    hour_weights = {
        0: 0.02, 1: 0.01, 2: 0.005, 3: 0.002, 4: 0.002, 5: 0.003,
        6: 0.01, 7: 0.02, 8: 0.03, 9: 0.04, 10: 0.05, 11: 0.07,
        12: 0.10, 13: 0.10, 14: 0.07, 15: 0.04, 16: 0.03, 17: 0.04,
        18: 0.06, 19: 0.09, 20: 0.10, 21: 0.08, 22: 0.05, 23: 0.03,
    }
    hours = list(hour_weights.keys())
    raw_probs = list(hour_weights.values())
    # Normalize to ensure probabilities sum to exactly 1.0
    total = sum(raw_probs)
    probs = [p / total for p in raw_probs]
    return np.random.choice(hours, p=probs)


HOUR_FRICTION = {
    # Off-peak hours
    0: 1.02, 1: 1.01, 2: 1.00, 3: 1.00, 4: 1.00, 5: 1.00,
    6: 1.01, 7: 1.02, 8: 1.04, 9: 1.05, 10: 1.06, 11: 1.08,
    # Lunch peak
    12: 1.15, 13: 1.18, 14: 1.12,
    # Afternoon lull
    15: 1.05, 16: 1.04, 17: 1.06,
    # Dinner buildup → peak
    18: 1.10, 19: 1.18, 20: 1.22, 21: 1.20,
    # Late night
    22: 1.10, 23: 1.05,
}

# ─────────────────────────────────────────────────────────────────────────────
# Traffic Index → Friction Mapping
# ─────────────────────────────────────────────────────────────────────────────
def traffic_to_friction(traffic_index):
    """
    Convert raw traffic_index (0.3 - 1.0) to a friction multiplier.
    Non-linear: high traffic compounds delay exponentially.
    """
    # Piecewise: low traffic is nearly neutral; high traffic grows fast
    if traffic_index < 0.5:
        return 1.0 + (traffic_index * 0.10)
    elif traffic_index < 0.8:
        return 1.05 + ((traffic_index - 0.5) * 0.30)
    else:
        return 1.14 + ((traffic_index - 0.8) * 0.60)  # exponential zone


# ── Compute ambient friction multiplier for every order ──────────────────────
results = []

for idx, row in orders_df.iterrows():
    order_id = row["order_id"]
    weather = row["weather"]
    traffic_index = float(row["traffic_index"])
    rider_density = row["rider_density"]
    zone = row["zone"]

    # ---- Component 1: Weather friction ----
    w_cfg = WEATHER_FRICTION.get(weather, WEATHER_FRICTION["Clear"])
    weather_mult = w_cfg["base"] + np.random.normal(0, w_cfg["variance"])

    # ---- Component 2: Traffic friction (non-linear) ----
    traffic_mult = traffic_to_friction(traffic_index)

    # ---- Component 3: Rider density impact ----
    rider_mult = RIDER_DENSITY_IMPACT.get(rider_density, 1.0)

    # ---- Component 4: Zone congestion ----
    zone_mult = ZONE_CONGESTION.get(zone, 1.0)

    # ---- Component 5: Time-of-day temporal friction ----
    hour = get_hour_of_day(idx, len(orders_df))
    hour_mult = HOUR_FRICTION.get(hour, 1.0)

    # ---- Compounding friction (multiplicative, not additive) ----
    # This captures how delays compound: rain + high traffic + dinner peak
    # is much worse than the sum of individual effects
    raw_multiplier = weather_mult * traffic_mult * rider_mult * zone_mult * hour_mult

    # ---- Small random jitter for real-world noise ----
    noise = np.random.normal(1.0, 0.02)
    ambient_friction_multiplier = round(raw_multiplier * noise, 4)

    # Clamp to reasonable bounds [0.80, 2.50]
    ambient_friction_multiplier = max(0.80, min(2.50, ambient_friction_multiplier))

    results.append({
        "order_id": order_id,
        "weather": weather,
        "traffic_index": traffic_index,
        "rider_density": rider_density,
        "zone": zone,
        "hour_of_day": hour,
        "weather_friction": round(weather_mult, 4),
        "traffic_friction": round(traffic_mult, 4),
        "rider_density_friction": round(rider_mult, 4),
        "zone_congestion": round(zone_mult, 4),
        "hour_friction": round(hour_mult, 4),
        "ambient_friction_multiplier": ambient_friction_multiplier,
    })

# ── Output ───────────────────────────────────────────────────────────────────
output_df = pd.DataFrame(results)
output_df.to_csv("engine3_output.csv", index=False)

print(f"\n✅  Engine 3 complete — {len(output_df)} rows written to engine3_output.csv")
print(f"   Mean friction multiplier: {output_df['ambient_friction_multiplier'].mean():.4f}")
print(f"   Min: {output_df['ambient_friction_multiplier'].min():.4f}  |  "
      f"Max: {output_df['ambient_friction_multiplier'].max():.4f}")
print("\nFriction by weather:")
print(output_df.groupby("weather")["ambient_friction_multiplier"].mean().sort_values(ascending=False))
print("\nFriction by zone:")
print(output_df.groupby("zone")["ambient_friction_multiplier"].mean().sort_values(ascending=False))
