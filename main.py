import argparse
import os

from src.anomaly import detect_meter_anomalies
from src.features import extract_load_features
from src.ingestion import (
    DEFAULT_TAMPERED_METER_IDS,
    generate_raw_cer_dataset,
    load_raw_data,
)
from src.segmentation import train_customer_clusters


def run_pipeline(
    raw_data_path: str = "data/raw/cer_raw_data.csv",
    processed_output_path: str = "data/processed/cer_final_analytics.csv",
    num_meters: int = 50,
    days: int = 14,
    force_regenerate: bool = False,
    contamination: float = 0.05,
    inject_tamper_events: bool = False,
):
    """Executes the full Smart Meter Data Analytics Pipeline end-to-end."""
    print("=" * 60)
    print("STARTING SMART METER ANALYTICS PIPELINE")
    print("=" * 60)

    # 1. Ingestion / Data Generation
    regenerate_needed = force_regenerate or inject_tamper_events or not os.path.exists(raw_data_path)
    df_raw = None

    if not regenerate_needed:
        existing = load_raw_data(raw_data_path)
        actual_num_meters = existing["meter_id"].nunique()
        actual_days = (existing["timestamp"].max() - existing["timestamp"].min()).days + 1

        if actual_num_meters != num_meters or actual_days != days:
            print(
                f"\n[STEP 1/4]Existing raw data at '{raw_data_path}' has "
                f"{actual_num_meters} meters / {actual_days} days, but this run "
                f"requested {num_meters} meters / {days} days. Regenerating to match."
            )
            regenerate_needed = True
        else:
            print(
                f"\n[STEP 1/4] Reusing existing raw data at '{raw_data_path}' "
                f"({actual_num_meters} meters, {actual_days} days) - matches requested parameters."
            )
            df_raw = existing

    if regenerate_needed:
        print(f"\n[STEP 1/4] Generating raw CER dataset ({num_meters} meters, {days} days)"
              f"{', WITH injected tamper events' if inject_tamper_events else ''}...")
        df_raw = generate_raw_cer_dataset(
            output_filepath=raw_data_path, num_meters=num_meters, days=days,
            inject_tamper_events=inject_tamper_events,
        )

    # 2. Feature Engineering
    print("\n[STEP 2/4] Extracting load profile features...")
    df_features = extract_load_features(df_raw)

    # 3. Segmentation (Clustering)
    print("\n[STEP 3/4] Training customer clustering model (K-Means)...")
    df_segmented, _, _ = train_customer_clusters(df_features, n_clusters=3)

    # 4. Anomaly Detection
    print(
        "\n[STEP 4/4] Detecting non-technical losses & faults (Isolation Forest)..."
    )
    df_final, _ = detect_meter_anomalies(df_segmented, contamination=contamination)

    # Save final results
    out_dir = os.path.dirname(processed_output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df_final.to_csv(processed_output_path, index=False)

    print("\n" + "=" * 60)
    print(f"PIPELINE COMPLETE! Final output saved to: '{processed_output_path}'")
    print("=" * 60)

    if inject_tamper_events:
        tampered_ids = DEFAULT_TAMPERED_METER_IDS
        flagged_tampered = df_final[df_final["meter_id"].isin(tampered_ids)]["is_anomaly"].tolist()
        print(f"\n DEMO CHECK: tampered meters {tampered_ids} -> "
              f"flagged as anomalies: {flagged_tampered}")

    return df_final


def _parse_args():
    parser = argparse.ArgumentParser(description="Run the Smart Meter Analytics Pipeline.")
    parser.add_argument("--num-meters", type=int, default=50, help="Number of synthetic meters to generate.")
    parser.add_argument("--days", type=int, default=14, help="Number of days of data to generate.")
    parser.add_argument("--contamination", type=float, default=0.05, help="Assumed anomaly rate for Isolation Forest.")
    parser.add_argument("--force-regenerate", action="store_true", help="Always regenerate raw data, even if it already matches.")
    parser.add_argument(
        "--inject-tamper-events", action="store_true",
        help="Inject a known ground-truth tamper event (meters 1000, 1001 drop to near-zero halfway "
             "through the window) to demo/validate the anomaly detector's temporal features.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_pipeline(
        num_meters=args.num_meters,
        days=args.days,
        contamination=args.contamination,
        force_regenerate=args.force_regenerate,
        inject_tamper_events=args.inject_tamper_events,
    )

