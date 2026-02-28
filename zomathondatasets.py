import pandas as pd
import numpy as np
import random

np.random.seed(42)

# -----------------------------
# STATIC KITCHEN DATA
# -----------------------------

zones = [
    "Connaught Place",
    "Gurgaon CyberHub",
    "Noida Sector 18",
    "Saket",
    "Dwarka",
    "Lajpat Nagar",
    "Rohini",
    "Faridabad"
]

prep_types = ["Tandoor", "Deep Fry", "Steamer", "Grill", "Assembly"]

static_data = []

for i in range(1, 100001):
    static_data.append({
        "merchant_id": f"NCR_M{i:03}",
        "zone": random.choice(zones),
        "tandoor_units": np.random.randint(1, 4),
        "deep_fry_units": np.random.randint(1, 4),
        "steamer_units": np.random.randint(1, 3),
        "grill_units": np.random.randint(1, 3),
        "assembly_units": np.random.randint(2, 6),
        "historical_speed_index": round(np.random.uniform(0.7, 1.3), 2)
    })

static_df = pd.DataFrame(static_data)
static_df.to_csv("delhi_ncr_static_kitchens_v2.csv", index=False)

# -----------------------------
# WEATHER PROBABILITIES
# -----------------------------

weather_probs = {
    "Clear": 0.45,
    "Heatwave": 0.15,
    "Smog": 0.15,
    "Fog": 0.10,
    "Rain": 0.15
}

weather_choices = list(weather_probs.keys())
weather_weights = list(weather_probs.values())

# -----------------------------
# BASE PREP TIMES (seconds)
# -----------------------------

base_prep_time = {
    "Tandoor": (300, 600),
    "Deep Fry": (180, 420),
    "Steamer": (240, 480),
    "Grill": (240, 500),
    "Assembly": (60, 180)
}

# -----------------------------
# SIMULATE 100K ORDERS
# -----------------------------

orders = []

for order_id in range(1, 100001):

    kitchen = static_df.sample(1).iloc[0]
    weather = np.random.choice(weather_choices, p=weather_weights)
    
    rider_density = random.choice(["High", "Medium", "Low"])
    traffic_index = round(np.random.uniform(0.3, 1.0), 2)
    live_volume = np.random.randint(1, 25)

    # Select random food types
    item_count = np.random.randint(1, 5)
    selected_preps = random.choices(prep_types, k=item_count)

    # ----------------------
    # Engine 1: Kitchen Twin
    # ----------------------
    structural_time = 0

    for prep in selected_preps:
        min_t, max_t = base_prep_time[prep]
        prep_time = np.random.randint(min_t, max_t)

        capacity = kitchen[f"{prep.lower().replace(' ','_')}_units"]
        queue_delay = live_volume * 8

        structural_time += (prep_time / capacity) + queue_delay

    # ----------------------
    # Engine 2: Customization
    # ----------------------
    customization = np.random.choice([0, 1], p=[0.7, 0.3])
    customization_delta = np.random.randint(10, 90) if customization else 0

    # ----------------------
    # Engine 3: Friction
    # ----------------------

    weather_impact = {
        "Clear": 0.00,
        "Heatwave": 0.07,
        "Smog": 0.05,
        "Fog": 0.12,
        "Rain": 0.15
    }[weather]

    rider_impact = {
        "High": -0.03,
        "Medium": 0.05,
        "Low": 0.18
    }[rider_density]

    ambient_multiplier = 1 + weather_impact + rider_impact + (traffic_index * 0.15)

    raw_kpt = (structural_time + customization_delta) * ambient_multiplier

    # ----------------------
    # Dynamic Buffer
    # ----------------------

    reliability_buffer = np.interp(
        kitchen["historical_speed_index"],
        [0.7, 1.3],
        [420, 120]
    )

    final_predicted_kpt = raw_kpt + reliability_buffer

    # ----------------------
    # Rider Arrival Simulation
    # ----------------------

    base_travel_time = np.random.randint(300, 900)

    travel_time = base_travel_time * (1 + weather_impact + traffic_index)

    rider_arrival_time = travel_time

    # ----------------------
    # Ground Truth Calculation
    # ----------------------

    food_ready_time = structural_time + customization_delta

    if rider_arrival_time > food_ready_time:
        inferred_true_prep_time = food_ready_time
    else:
        inferred_true_prep_time = rider_arrival_time

    orders.append({
        "order_id": order_id,
        "merchant_id": kitchen["merchant_id"],
        "zone": kitchen["zone"],
        "weather": weather,
        "rider_density": rider_density,
        "traffic_index": traffic_index,
        "live_order_volume": live_volume,
        "prep_types": ",".join(selected_preps),
        "structural_time_sec": round(structural_time, 2),
        "customization_delta_sec": customization_delta,
        "ambient_multiplier": round(ambient_multiplier, 3),
        "final_predicted_kpt_sec": round(final_predicted_kpt, 2),
        "rider_arrival_time_sec": round(rider_arrival_time, 2),
        "inferred_true_prep_time_sec": round(inferred_true_prep_time, 2)
    })

orders_df = pd.DataFrame(orders)
orders_df.to_csv("delhi_ncr_orders_100k_simulated.csv", index=False)

print("100K Delhi NCR simulation dataset created successfully.")