import os
from src.features import extract_load_features
from src.ingestion import load_raw_data

# Load the raw data generated earlier
raw_df = load_raw_data("data/raw/cer_raw_data.csv")

# Extract features per meter
feature_df = extract_load_features(raw_df)

# Ensure the 'data/processed' directory exists
os.makedirs("data/processed", exist_ok=True)

# Save processed features
feature_df.to_csv("data/processed/cer_features.csv", index=False)

# View results
print("\nExtracted Features Sample:")
print(
    feature_df[
        [
            "meter_id",
            "mean_kwh",
            "load_factor",
            "night_ratio",
            "eve_peak_ratio",
        ]
    ].head()
)
from src.features import extract_load_features
from src.ingestion import load_raw_data

# Load the raw data you generated earlier
raw_df = load_raw_data("data/raw/cer_raw_data.csv")

# Extract features per meter
feature_df = extract_load_features(raw_df)

# Save processed features to data/processed/
feature_df.to_csv("data/processed/cer_features.csv", index=False)

# View results
print("\nExtracted Features Sample:")
print(
    feature_df[
        [
            "meter_id",
            "mean_kwh",
            "load_factor",
            "night_ratio",
            "eve_peak_ratio",
        ]
    ].head()
)
