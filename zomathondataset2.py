import simpy
import pandas as pd
import numpy as np
import random

RANDOM_SEED = 42
NUM_ORDERS = 20000
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# ----------------------------------
# Delhi NCR Weather Distribution
# ----------------------------------

weather_choices = ["Clear", "Heatwave", "Rain", "Smog", "Fog"]
weather_probs = [0.45, 0.15, 0.12, 0.18, 0.10]

# ----------------------------------
# Preparation Types
# ----------------------------------

prep_types = {
    "Tandoor": (300, 600),
    "Deep Fry": (180, 420),
    "Steamer": (240, 480),
    "Stir Fry": (120, 300),
    "Stewing": (600, 1200),
    "Grilling": (300, 540)
}

# ----------------------------------
# Kitchen Environment
# ----------------------------------

class Kitchen:
    def __init__(self, env):
        self.env = env
        self.tandoor = simpy.Resource(env, capacity=3)
        self.fryer = simpy.Resource(env, capacity=2)
        self.steamer = simpy.Resource(env, capacity=2)
        self.stove = simpy.Resource(env, capacity=4)

# ----------------------------------
# Order Process
# ----------------------------------

def order_process(env, order_id, kitchen, results):

    arrival_time = env.now
    
    weather = np.random.choice(weather_choices, p=weather_probs)
    traffic_index = np.random.uniform(0.3, 1.0)
    rider_density = np.random.choice(["High","Medium","Low"], p=[0.4,0.35,0.25])
    
    cart_items = np.random.randint(1,5)
    special_instructions = random.choice([0,1])
    
    start_cook = None
    finish_cook = None
    
    for _ in range(cart_items):
        prep = random.choice(list(prep_types.keys()))
        process_time = np.random.randint(*prep_types[prep])
        
        if prep == "Tandoor":
            resource = kitchen.tandoor
        elif prep == "Deep Fry":
            resource = kitchen.fryer
        elif prep == "Steamer":
            resource = kitchen.steamer
        else:
            resource = kitchen.stove
        
        with resource.request() as req:
            yield req
            
            if start_cook is None:
                start_cook = env.now
            
            yield env.timeout(process_time)
    
    finish_cook = env.now
    
    structural_time = finish_cook - arrival_time
    
    customization_delta = np.random.randint(10,70) if special_instructions else 0
    yield env.timeout(customization_delta)
    
    prep_complete_time = env.now
    
    # Rider Dispatch Simulation
    weather_impact = {
        "Clear":0.0,"Heatwave":0.07,"Rain":0.12,"Smog":0.05,"Fog":0.15
    }[weather]
    
    rider_impact = {
        "High":-0.03,"Medium":0.05,"Low":0.18
    }[rider_density]
    
    travel_delay = (traffic_index * 600) + (weather_impact * 300)
    
    rider_arrival = prep_complete_time + travel_delay
    
    dwell_time = max(0, rider_arrival - prep_complete_time)
    
    true_prep_time = prep_complete_time - arrival_time
    
    results.append({
        "order_id": order_id,
        "arrival_time": arrival_time,
        "weather": weather,
        "traffic_index": round(traffic_index,2),
        "rider_density": rider_density,
        "cart_items": cart_items,
        "structural_time_sec": round(structural_time,2),
        "customization_delta_sec": customization_delta,
        "prep_complete_time": round(prep_complete_time,2),
        "rider_arrival_time": round(rider_arrival,2),
        "dwell_time_sec": round(dwell_time,2),
        "inferred_true_prep_time_sec": round(true_prep_time,2)
    })

# ----------------------------------
# Simulation Runner
# ----------------------------------

def run_simulation():
    env = simpy.Environment()
    kitchen = Kitchen(env)
    results = []
    
    for i in range(NUM_ORDERS):
        env.process(order_process(env, i, kitchen, results))
        env.timeout(np.random.exponential(30))  # order inter-arrival
    
    env.run()
    
    df = pd.DataFrame(results)
    df.to_csv("delhi_ncr_simpy_20k_orders.csv", index=False)
    print("Simulation complete. 20K DES dataset saved.")

run_simulation()