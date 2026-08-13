import os
import glob
from datetime import date, datetime, timedelta
import numpy as np
import pandas as pd
def generate_raw_cer_dataset(
    output_filepath: str = "data/raw/cer_raw_data.csv",
    num_meters: int = 50,
    days: int = 14,
    seed: int = 42,
    inject_tamper_events: bool = False,
    n_tamper_events: int = 2,
) -> pd.DataFrame:
    
    np.random.seed(seed)
    n_intervals = days * 48  # 48 half-hour intervals per day
    timestamps = pd.date_range(
        start="2026-01-01", periods=n_intervals, freq="30min"
    )

    tampered_meter_ids = set(range(1000, 1000 + n_tamper_events)) if inject_tamper_events else set()
    midpoint_ts = timestamps[len(timestamps) // 2]

    records = []

    for meter_id in range(1000, 1000 + num_meters):
        # 0 = Standard Residential, 1 = Night/EV Heavy, 2 = Commercial
        profile_type = np.random.choice([0, 1, 2], p=[0.60, 0.25, 0.15])

        for ts in timestamps:
            hour = ts.hour + ts.minute / 60.0
            day_of_week = ts.dayofweek

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

            kwh = max(0.01, kwh + np.random.normal(0, 0.05))

            if meter_id in tampered_meter_ids and ts >= midpoint_ts:
                kwh = max(0.01, 0.02 + np.random.normal(0, 0.01))

            records.append({"meter_id": meter_id, "timestamp": ts, "kwh": round(kwh, 4)})

    df_raw = pd.DataFrame(records)

    if inject_tamper_events:
        print(f" Injected simulated tamper events (mid-window drop to near-zero) into meter IDs: {sorted(tampered_meter_ids)}")

    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    df_raw.to_csv(output_filepath, index=False)
    print(f"Raw dataset created successfully: {output_filepath}")
    return df_raw


def load_raw_data(filepath: str) -> pd.DataFrame:
    """Reads a raw CER CSV dataset (canonical meter_id/timestamp/kwh schema)
    and converts timestamp columns to datetime objects."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"File not found at {filepath}. Run generate_raw_cer_dataset() first!"
        )

    print(f"📖 Reading raw data from: {filepath}...")
    df = pd.read_csv(filepath)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


CER_TRIAL_EPOCH_DATE = date(2009, 1, 1)  # VERIFY against real codebook
INTERVALS_PER_DAY = 48


    if day_code < 1:
        raise ValueError(f"Invalid CER timecode {timecode}: day code {day_code} must be >= 1")
    if not (1 <= halfhour_code <= INTERVALS_PER_DAY):
        raise ValueError(
            f"Invalid CER timecode {timecode}: halfhour code {halfhour_code} "
            f"must be between 1 and {INTERVALS_PER_DAY}"
        )

    interval_date = epoch_date + timedelta(days=day_code - 1)
    interval_time = timedelta(minutes=30 * (halfhour_code - 1))
    return datetime.combine(interval_date, datetime.min.time()) + interval_time


def load_cer_meter_reads(
    filepaths,
    epoch_date: date = CER_TRIAL_EPOCH_DATE,
    delimiter: str = None,
) -> pd.DataFrame:
    
    if isinstance(filepaths, str):
        filepaths = sorted(glob.glob(filepaths))
    if not filepaths:
        raise FileNotFoundError("No CER meter-read files matched the given path/pattern.")

    frames = []
    for fp in filepaths:
        print(f"Reading real CER meter-read file: {fp}...")
        df = pd.read_csv(
            fp, header=None, names=["meter_id", "timecode", "kwh"],
            sep=delimiter, engine="python" if delimiter is None else "c",
        )
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    print(f"   Decoding {len(combined):,} timecodes...")
    combined["timestamp"] = combined["timecode"].apply(lambda tc: decode_cer_timecode(tc, epoch_date))
    combined = combined[["meter_id", "timestamp", "kwh"]]
    return combined


def load_cer_allocations(filepath: str) -> pd.DataFrame:
    """
    Loads the CER allocation file mapping each meter_id to a Residential/
    SME segment (and tariff/stimulus group, if present). Column names in
    the real file aren't confirmed without the actual codebook, so this
    tries several common naming conventions and fails with the ACTUAL
    columns found if none match, rather than silently misreading the file.
    """
    df = pd.read_csv(filepath)

    id_col = next((c for c in df.columns if c.strip().lower() in ("id", "meter_id", "meterid")), None)
    segment_col = next(
        (c for c in df.columns if "code" in c.strip().lower() or "allocation" in c.strip().lower()),
        None,
    )

    # 1. Drop exact duplicate (meter_id, timestamp, kwh) rows outright.
    df = df.drop_duplicates()

    # 2. For remaining (meter_id, timestamp) duplicates with DIFFERENT kwh
    # values (ambiguous which reading is correct), keep the first and
    # report how many were affected rather than silently averaging two
    # potentially-erroneous readings.
    dup_mask = df.duplicated(subset=["meter_id", "timestamp"], keep="first")
    n_ambiguous_dupes = int(dup_mask.sum())
    if n_ambiguous_dupes > 0:
        print(f" Dropped {n_ambiguous_dupes} conflicting duplicate (meter_id, timestamp) reads (kept first occurrence).")
    df = df[~dup_mask]

    # 3. Physically-impossible negative consumption -> clip to 0, report count.
    n_negative = int((df["kwh"] < 0).sum())
    if n_negative > 0:
        print(f" Clipped {n_negative} negative kwh readings to 0 (data errors, not real consumption).")
        df.loc[df["kwh"] < 0, "kwh"] = 0.0

    # 4. Completeness check per meter: drop meters with too many missing
    # intervals to trust their aggregated features.
    span_days = (df["timestamp"].max() - df["timestamp"].min()).days + 1
    expected_readings = span_days * expected_intervals_per_day
    counts = df.groupby("meter_id").size()
    completeness = counts / expected_readings
    incomplete_meters = completeness[completeness < min_completeness].index.tolist()

    if incomplete_meters:
        print(
            f"Dropping {len(incomplete_meters)} meter(s) below {min_completeness:.0%} "
            f"data completeness (missing too many intervals to trust): {incomplete_meters[:10]}"
            + (" ..." if len(incomplete_meters) > 10 else "")
        )
        df = df[~df["meter_id"].isin(incomplete_meters)]

    n_after = len(df)
    print(f"✅ Data quality pass complete: {n_before:,} -> {n_after:,} rows "
          f"({n_before - n_after:,} removed), {len(incomplete_meters)} meter(s) dropped entirely.")
    return df.reset_index(drop=True)


def load_real_cer_dataset(
    meter_read_filepaths,
    allocation_filepath: str = None,
    epoch_date: date = CER_TRIAL_EPOCH_DATE,
    min_completeness: float = 0.95,
    output_filepath: str = None,
) -> pd.DataFrame:
  
  
