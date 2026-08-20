import os

from src.features import extract_load_features
from src.ingestion import load_raw_data

raw_df = load_raw_data("data/raw/cer_raw_data.csv")
feature_df = extract_load_features(raw_df)

os.makedirs("data/processed", exist_ok=True)
feature_df.to_csv("data/processed/cer_features.csv", index=False)

print("\nExtracted Features Sample:")
print(feature_df[["meter_id", "mean_kwh", "load_factor", "night_ratio", "eve_peak_ratio"]].head())
