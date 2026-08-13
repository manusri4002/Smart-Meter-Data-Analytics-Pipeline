import os
import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest

def detect_meter_anomalies(
    segmented_df: pd.DataFrame,
    contamination: float = 0.05,
    model_dir: str = "models",
    random_state: int = 42,
) -> tuple[pd.DataFrame, IsolationForest]:
  
    df = segmented_df.copy()
    feature_cols = [
        "mean_kwh",
        "load_factor",
        "night_ratio",
        "morn_peak_ratio",
        "eve_peak_ratio",
        "half_period_ratio",
        "daily_consumption_std",
    ]

    X = df[feature_cols]

    # Initialize and fit Isolation Forest
    iso_model = IsolationForest(
        contamination=contamination, random_state=random_state
    )

    #Indicates an anomaly, 1 indicates normal
    df["anomaly_score"] = iso_model.fit_predict(X)
    df["is_anomaly"] = df["anomaly_score"] == -1

    anom_count = df["is_anomaly"].sum()
    print("Anomaly Detection Complete.")
    print(
        f"Flagged {anom_count} out of {len(df)} meters as potential anomalies/faults."
    )

    #Save model artifact
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(iso_model, os.path.join(model_dir, "iso_forest.joblib"))
    print(f" Anomaly model saved to: '{model_dir}/iso_forest.joblib'")

    return df, iso_model
