**Kitchen Preparation Time Prediction Engine | AI-Powered Food Preparation Time Forecasting**

Official Website: https://github.com/aarush9354/Zomathon
__________________________________________________________________________________

Project Overview:

KPT is an AI-powered food preparation time forecasting engine designed to accurately predict kitchen preparation times for Zomato's delivery logistics. Accurate predictions are critical to balance customer experience with delivery logistics and restaurant operations. The project utilizes a 3-layer ensemble architecture to categorize orders into seven distinct time buckets, ranging from quick items (0-5 min) to specialty items (45+ min).

Technology Stack:

Programming Language: Python
Libraries: NumPy, Pandas, scikit-learn (Random Forest), NLP tools for instruction parsing
Architecture: 3-Layer Ensemble (Historical Performance, NLP Complexity, and ML Classification)

Key Features:

Kitchen Prep Forecasting: Uses a Random Forest classifier and ensemble learning to achieve 74.74% accuracy across 20,000 test orders.
NLP Instruction Analysis: Parses special instructions to calculate "complexity deltas," adding 5 to 115 seconds to prep estimates for customizations like "Jain preparation" or "Gluten free".
Historical Speed Indexing: Establishes restaurant baselines and structural minimum times based on historical performance data.
Geographic Performance Analysis: Offers consistent performance tracking across various Delhi NCR zones, such as Saket and Gurgaon CyberHub.

File Information:

engine1_kitchen_twin.py :
- This is the "Kitchen Twin" engine that uses Discrete Event Simulation to map the physical kitchen pipeline.
- It calculates the Structural Minimum Time by accounting for hardware constraints (ovens, tandoors, fryers) and batching logic for active cart items.
- The file outputs the base structural time and historical speed index for each specific order.

engine2_customization_analyzer.py :
- This engine acts as a Customization Analyzer using lightweight NLP and rule-based logic.
- It parses customer special instructions to calculate an "Order-Specific Complexity Delta," adding time for requests like "Jain preparation" or "extra packaging."
- It only activates when user-typed instructions are present, ensuring unique order requirements are captured.

engine3_spatial_friction.py :
- This is the Spatial Friction Tracker, responsible for modeling compounding local delays from live neighborhood variables.
- It analyzes Delhi NCR-specific factors like weather (smog, rain), traffic indices, rider density, and zone-level congestion patterns.
- The output is an "Ambient Friction Multiplier" that adjusts the preparation time based on the external environment.

layer3_meta_resolution.py :
- Known as "The Brain," this is the final Meta-Resolution and Truth Layer that aggregates outputs from the first three engines.
- It performs anomaly detection (lie-detection) and applies non-linear aggregation to produce the final Predicted KPT (Kitchen Preparation Time).
- The file generates the ultimate trust scores and dynamic buffers for the forecasting engine.

zomathondatasets.py :
- This script is responsible for generating the static kitchen data and the core 100,000+ merchant profiles.
- It establishes the baseline characteristics for restaurants across Delhi NCR zones, including their specific equipment units and historical speed indices.

zomathondataset2.py :
- A simulation runner script that utilizes the SimPy library to generate the 20,000 test orders used for model validation.
- It simulates the ground truth for food ready times and rider arrival times to calculate the "inferred true prep time" used for training the KPT engine.

_________________________________________________________________________________

Note:
The engine currently faces a "Long-Tail Challenge" with the 45+ min bucket, which has high recall but lower precision due to many false positives.

**Contact Information:**

For any queries or feedback, kindly contact our team, Team Null Pointerz:

Aarush Raj Singh
(+91 93542 59076)

Aatman
(+91 95401 02113)

Divyansh Balooni
(+91 78380 29059)