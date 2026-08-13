import os
import numpy as np
import pandas as pd


def generate_raw_cer_dataset(
    output_filepath: str = "data/raw/cer_raw_data.csv",
    num_meters: int = 50,
    days: int = 14,
    seed: int = 42,
) -> pd.DataFrame:
    """Generates synthetic smart meter data matching the exact schema and half-hourly

    interval structure of the Irish CER Smart Metering trial.
    """
    np.random.seed(seed)
    n_intervals = days * 48  # 48 half-hour intervals per day
    timestamps = pd.date_range(
        start="2026-01-01", periods=n_intervals, freq="30min"
    )

    records = []

    for meter_id in range(1000, 1000 + num_meters):
        # 0 = Standard Residential, 1 = Night/EV Heavy, 2 = Commercial
        profile_type = np.random.choice([0, 1, 2], p=[0.60, 0.25, 0.15])

        for ts in timestamps:
            hour = ts.hour + ts.minute / 60.0
            day_of_week = ts.dayofweek

          # Add noise and ensure non-negative consumption
            kwh = max(0.01, kwh + np.random.normal(0, 0.05))

            records.append(
                {
                    "meter_id": meter_id,
                    "timestamp": ts,
                    "kwh": round(kwh, 4),
                }
            )

    df_raw = pd.DataFrame(records)

    # Ensure output directory exists before saving
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    df_raw.to_csv(output_filepath, index=False)
    print(f"Raw dataset created successfully: {output_filepath}")
    return df_raw


def load_raw_data(filepath: str) -> pd.DataFrame:
    """Reads a raw CER CSV dataset and converts timestamp columns to datetime objects."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"File not found at {filepath}. Run generate_raw_cer_dataset() first!"
        )

    print(f"Reading raw data from: {filepath}...")
    df = pd.read_csv(filepath)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df
