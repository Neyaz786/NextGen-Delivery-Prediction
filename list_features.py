"""
List all features used in the pipeline.
"""

import pandas as pd
from pathlib import Path

# Define paths
project_root = Path(__file__).parent
processed_dir = project_root / "processed"
data_dir = project_root / "data"

print("=" * 80)
print("FEATURES USED IN THE PIPELINE")
print("=" * 80)

# Check if processed data exists
processed_path = processed_dir / "merged_processed.csv"
if processed_path.exists():
    df = pd.read_csv(processed_path)
    print(f"\nLoading from: {processed_path}")
else:
    # Load and prepare from original data
    from src.data_prep import read_data_files, merge_data, drop_leakage_columns, create_target_variables
    from src.features import prepare_features
    
    orders_df, delivery_df, routes_df = read_data_files(data_dir)
    merged_df = merge_data(orders_df, delivery_df, routes_df)
    cleaned_df = drop_leakage_columns(merged_df)
    final_df = create_target_variables(cleaned_df)
    df, numeric_features, ordinal_features, nominal_features = prepare_features(final_df)
    print(f"\nPrepared from original data")

# Exclude targets
target_cols = ["delay_days", "is_delayed", "Order_ID"]
feature_cols = [col for col in df.columns if col not in target_cols]

print(f"\nTotal Features: {len(feature_cols)}")
print(f"\n" + "=" * 80)

# Categorize features
ordinal_features = ["Priority", "Weather_Impact"]
ordinal_features = [f for f in ordinal_features if f in feature_cols]

numeric_features = []
for col in feature_cols:
    if col not in ordinal_features and df[col].dtype in ['int64', 'float64']:
        numeric_features.append(col)

nominal_features = []
for col in feature_cols:
    if col not in ordinal_features and col not in numeric_features:
        nominal_features.append(col)

print("\n1. NUMERIC FEATURES:")
print("-" * 80)
for i, feat in enumerate(numeric_features, 1):
    print(f"   {i:2d}. {feat:30s} (dtype: {df[feat].dtype})")

print(f"\n2. ORDINAL FEATURES:")
print("-" * 80)
for i, feat in enumerate(ordinal_features, 1):
    print(f"   {i:2d}. {feat:30s} (encoded)")

print(f"\n3. NOMINAL/CATEGORICAL FEATURES:")
print("-" * 80)
for i, feat in enumerate(nominal_features, 1):
    print(f"   {i:2d}. {feat:30s} (one-hot encoded)")

print(f"\n" + "=" * 80)
print("\nPOTENTIAL DATA LEAKAGE CHECK:")
print("-" * 80)

# Check for problematic columns
problematic = []
if "Customer_Rating" in feature_cols:
    problematic.append("Customer_Rating - Post-delivery rating, highly correlated with delays")
if "Actual_Delivery_Days" in feature_cols:
    problematic.append("Actual_Delivery_Days - This is the target, should not be a feature!")
if "Promised_Delivery_Days" in feature_cols:
    problematic.append("Promised_Delivery_Days - Used to calculate target, may cause leakage")

if problematic:
    print("WARNING: Potential data leakage detected!")
    for issue in problematic:
        print(f"   - {issue}")
else:
    print("✓ No obvious data leakage detected in feature list")

print(f"\n" + "=" * 80)
print(f"\nDerived Features Created:")
print("-" * 80)
if "value_per_km" in feature_cols:
    print("   - value_per_km (Order_Value_INR / Distance_KM)")
if "fuel_efficiency" in feature_cols:
    print("   - fuel_efficiency (Distance_KM / Fuel_Consumption_L)")

