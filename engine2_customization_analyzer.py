"""
Engine 2: The Customization Analyzer (Lightweight NLP / Rules)
==============================================================
Activates only when a customer typed special instructions.
Generates synthetic instruction text and applies keyword-based
penalty rules to compute an Order-Specific Complexity Delta.

Input : Customer Special Instructions + Baseline Item Complexity
Output: complexity_delta_sec per order
"""

import pandas as pd
import numpy as np
import random
import re

# ── Reproducibility ──────────────────────────────────────────────────────────
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# ── Load Orders ──────────────────────────────────────────────────────────────
orders_df = pd.read_csv("delhi_ncr_simpy_20k_orders.csv")
print(f"Loaded {len(orders_df)} orders for customization analysis.")

# ─────────────────────────────────────────────────────────────────────────────
# Instruction Templates — Delhi NCR food-specific
# ─────────────────────────────────────────────────────────────────────────────
INSTRUCTION_POOL = [
    # Spice modifications
    "Make it extra spicy",
    "No spice at all please",
    "Medium spice level",
    "Add extra green chilli",
    "Jain preparation no onion no garlic",
    # Packaging
    "Separate packaging for each item",
    "Pack roti and curry separately",
    "Need extra tissue and cutlery",
    "Double wrap the paratha",
    # Additions
    "Extra cheese on top",
    "Add extra butter on naan",
    "Extra raita on the side",
    "Add paneer tikka as extra",
    "Extra chutney packets please",
    # Cooking style
    "Make the tikka well done",
    "Less oil in preparation",
    "Steam instead of fry",
    "Charred tandoori finish",
    "Do not overcook the biryani rice",
    # Dietary
    "No MSG please",
    "Use only mustard oil",
    "Gluten free preparation",
    "Sugar free dessert",
    # Complex / multi-step
    "Separate dal and rice, extra papad, no pickle",
    "Half portion butter chicken, double naan, raita side",
    "Make everything mild spice, pack sauces separate",
]

# ─────────────────────────────────────────────────────────────────────────────
# Keyword → Penalty Rules (seconds added to prep time)
# ─────────────────────────────────────────────────────────────────────────────
KEYWORD_PENALTIES = {
    # Spice / flavour changes require re-seasoning
    "spicy":          15,
    "spice":          12,
    "chilli":         10,
    "no spice":       8,
    "jain":           25,   # Jain prep = separate utensils
    "no onion":       20,
    "no garlic":      15,
    # Packaging overhead
    "separate":       20,
    "double wrap":    15,
    "extra tissue":   5,
    "cutlery":        5,
    # Additions = extra assembly time
    "extra cheese":   18,
    "extra butter":   10,
    "extra raita":    12,
    "extra chutney":  8,
    "extra papad":    10,
    "paneer tikka":   30,   # cooking a whole new item
    # Cooking style modifications
    "well done":      25,
    "less oil":       10,
    "steam instead":  20,
    "charred":        15,
    "do not overcook":12,
    # Dietary restrictions (requires separate prep area)
    "no msg":         8,
    "mustard oil":    10,
    "gluten free":    20,
    "sugar free":     15,
    # Multi-part instructions (complex parsing)
    "half portion":   12,
    "double naan":    15,
    "mild spice":     10,
    "pack sauces":    12,
}


def generate_instructions(has_customization):
    """Generate 1-3 synthetic instruction strings for customized orders."""
    if not has_customization:
        return ""
    n = random.choices([1, 2, 3], weights=[0.55, 0.30, 0.15])[0]
    return " | ".join(random.sample(INSTRUCTION_POOL, min(n, len(INSTRUCTION_POOL))))


def compute_nlp_penalty(instruction_text):
    """
    Lightweight keyword-match NLP:
    Scan instruction text against known penalty keywords.
    Returns total penalty in seconds.
    """
    if not instruction_text or instruction_text.strip() == "":
        return 0

    text_lower = instruction_text.lower()
    total_penalty = 0

    for keyword, penalty in KEYWORD_PENALTIES.items():
        if keyword in text_lower:
            total_penalty += penalty

    return total_penalty


def compute_packaging_overhead(cart_items, instruction_text):
    """
    Additional time for multi-item orders with packaging instructions.
    Each extra item beyond 1 adds overhead if separate packaging requested.
    """
    if not instruction_text:
        return 0

    text_lower = instruction_text.lower()
    if "separate" in text_lower or "pack" in text_lower:
        return max(0, (cart_items - 1)) * 8  # 8 sec per extra item
    return 0


# ── Process each order ──────────────────────────────────────────────────────
results = []

for _, row in orders_df.iterrows():
    order_id = row["order_id"]
    cart_items = int(row["cart_items"])
    original_delta = float(row["customization_delta_sec"])

    # Determine if this order has customization
    has_customization = original_delta > 0

    # Generate synthetic instruction text
    instruction_text = generate_instructions(has_customization)

    # Compute NLP-based penalty
    nlp_penalty = compute_nlp_penalty(instruction_text)

    # Packaging overhead
    packaging_overhead = compute_packaging_overhead(cart_items, instruction_text)

    # Base complexity from item count (more items = more coordination)
    item_complexity_base = cart_items * 5  # 5 sec per item baseline

    # Final complexity delta
    if has_customization:
        complexity_delta = nlp_penalty + packaging_overhead + item_complexity_base
    else:
        complexity_delta = 0  # Engine only activates when user typed text

    results.append({
        "order_id": order_id,
        "has_customization": has_customization,
        "special_instructions": instruction_text,
        "nlp_penalty_sec": nlp_penalty,
        "packaging_overhead_sec": packaging_overhead,
        "item_complexity_base_sec": item_complexity_base,
        "complexity_delta_sec": complexity_delta,
    })

# ── Output ───────────────────────────────────────────────────────────────────
output_df = pd.DataFrame(results)
output_df.to_csv("engine2_output.csv", index=False)

print(f"\n✅  Engine 2 complete — {len(output_df)} rows written to engine2_output.csv")
print(f"   Orders with customization: {output_df['has_customization'].sum()}")
print(f"   Avg complexity delta (customized): "
      f"{output_df.loc[output_df['has_customization'], 'complexity_delta_sec'].mean():.1f} sec")
print(output_df.describe())
