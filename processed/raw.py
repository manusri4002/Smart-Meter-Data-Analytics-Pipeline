import os
import numpy as np
import pandas as pd
    records = []

    for meter_id in range(1000, 1000 + num_meters):
        # 0 = Standard Residential, 1 = Night/EV Heavy, 2 = Commercial
        profile_type = np.random.choice([0, 1, 2], p=[0.60, 0.25, 0.15])

        for ts in timestamps:
            hour = ts.hour + ts.minute / 60.0
            day_of_week = ts.dayofweek

            # Generate realistic load patterns based on profile archetype
            if profile_type == 0:  # Residential (Morning & Evening peaks)
                m_peak = np.exp(-((hour - 8) ** 2) / 4)
                e_peak = np.exp(-((hour - 19) ** 2) / 8)
                kwh = 0.2 + 0.8 * m_peak + 1.2 * e_peak
            elif profile_type == 1:  # Night/EV Heavy (Overnight peak)
                n_peak = np.exp(-((hour - 2) ** 2) / 4)
                e_peak = np.exp(-((hour - 19) ** 2) / 8)
                kwh = 0.15 + 1.8 * n_peak + 0.7 * e_peak
            else:  # Commercial (Weekday daytime heavy)
                if day_of_week < 5 and 8 <= hour <= 18:
                    kwh = 2.5 + np.random.normal(0, 0.2)
                else:
                    kwh = 0.3

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
