import os
import glob
from datetime import date, datetime, timedelta
import numpy as np
import pandas as pd

TAMPER_METER_ID_START = 1000
DEFAULT_N_TAMPER_EVENTS = 2
DEFAULT_TAMPERED_METER_IDS = list(
    range(TAMPER_METER_ID_START, TAMPER_METER_ID_START + DEFAULT_N_TAMPER_EVENTS)
)


def generate_raw_cer_dataset(
    output_filepath: str = "data/raw/cer_raw_data.csv",
    num_meters: int = 50,
    days: int = 14,
    seed: int = 42,
    inject_tamper_events: bool = False,
    n_tamper_events: int = DEFAULT_N_TAMPER_EVENTS,
) -> pd.DataFrame:
    """Generates synthetic smart meter data matching the exact schema and half-hourly

    interval structure of the Irish CER Smart Metering trial.

    inject_tamper_events: if True, the first `n_tamper_events` generated
    meter IDs have their consumption drop to near-zero for the SECOND HALF
    of the observation window, simulating a bypass/tamper event partway
    through - a known ground truth for validating that the anomaly
    detection pipeline's temporal features (half_period_ratio,
    daily_consumption_std in features.py) actually catch a mid-window
    behavior change.
    """
    np.random.seed(seed)
    n_intervals = days * 48  # 48 half-hour intervals per day
    timestamps = pd.date_range(
        start="2026-01-01", periods=n_intervals, freq="30min"
    )

    tampered_meter_ids = (
        set(range(TAMPER_METER_ID_START, TAMPER_METER_ID_START + n_tamper_events))
        if inject_tamper_events
        else set()
    )
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
        print(f"Injected simulated tamper events (mid-window drop to near-zero) into meter IDs: {sorted(tampered_meter_ids)}")

    out_dir = os.path.dirname(output_filepath)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
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

    print(f"Reading raw data from: {filepath}...")
    df = pd.read_csv(filepath)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

CER_TRIAL_EPOCH_DATE = date(2009, 1, 1)  # VERIFY against real codebook
INTERVALS_PER_DAY = 48


def decode_cer_timecode(timecode: int, epoch_date: date = CER_TRIAL_EPOCH_DATE) -> datetime:
    """
    Decodes the CER trial's 5-digit day+halfhour timecode into a real
    datetime. Format: first 3 digits = day code (day 1 = epoch_date,
    counting sequentially), last 2 digits = half-hour code (1-48, where
    1 = 00:00-00:30, 48 = 23:30-00:00).

    Example: timecode 19501 -> day code 195, halfhour code 01
             -> epoch_date + 194 days, at 00:00.
    """
    timecode = int(timecode)
    day_code = timecode // 100
    halfhour_code = timecode % 100

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
    """
    Loads one or more raw CER meter-read files (unlabeled, positional
    columns: meter_id, timecode, kwh) and decodes them into the canonical
    meter_id/timestamp/kwh schema. Accepts a glob pattern or an explicit
    list of file paths, since the real dataset ships split across
    multiple files.
    """
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

    df = pd.read_csv(filepath)

    id_col = next((c for c in df.columns if c.strip().lower() in ("id", "meter_id", "meterid")), None)
    segment_col = next(
        (c for c in df.columns if "code" in c.strip().lower() or "allocation" in c.strip().lower()),
        None,
    )

    if id_col is None or segment_col is None:
        raise ValueError(
            f"Could not identify meter ID / segment columns in allocation file '{filepath}'. "
            f"Actual columns found: {list(df.columns)}. Update load_cer_allocations() to match."
        )

    result = df[[id_col, segment_col]].rename(columns={id_col: "meter_id", segment_col: "segment_code"})
    print(f"Loaded allocation data for {len(result)} meters from '{filepath}' (columns: {id_col}, {segment_col}).")
    return result


def clean_cer_data(
    df: pd.DataFrame,
    expected_intervals_per_day: int = INTERVALS_PER_DAY,
    min_completeness: float = 0.95,
) -> pd.DataFrame:
   
    df = df.copy()
    n_before = len(df)

    # 1. Drop exact duplicate (meter_id, timestamp, kwh) rows outright.
    df = df.drop_duplicates()

    # 2. For remaining (meter_id, timestamp) duplicates with DIFFERENT kwh
    # values (ambiguous which reading is correct), keep the first and
    # report how many were affected rather than silently averaging two
    # potentially-erroneous readings.
    dup_mask = df.duplicated(subset=["meter_id", "timestamp"], keep="first")
    n_ambiguous_dupes = int(dup_mask.sum())
    if n_ambiguous_dupes > 0:
        print(f"Dropped {n_ambiguous_dupes} conflicting duplicate (meter_id, timestamp) reads (kept first occurrence).")
    df = df[~dup_mask]

    # 3. Physically-impossible negative consumption -> clip to 0, report count.
    n_negative = int((df["kwh"] < 0).sum())
    if n_negative > 0:
        print(f"Clipped {n_negative} negative kwh readings to 0 (data errors, not real consumption).")
        df.loc[df["kwh"] < 0, "kwh"] = 0.0

    # 4. Completeness check, computed PER METER (not globally) - each meter
    # is judged against its own observed date span, since different meters
    # can have different install/removal dates in real trial data. 
    per_meter_span_days = df.groupby("meter_id")["timestamp"].agg(
        lambda s: (s.max() - s.min()).days + 1
    )
    expected_per_meter = per_meter_span_days * expected_intervals_per_day
    counts = df.groupby("meter_id").size()
    completeness = counts / expected_per_meter
    incomplete_meters = completeness[completeness < min_completeness].index.tolist()

    if incomplete_meters:
        print(
            f"Dropping {len(incomplete_meters)} meter(s) below {min_completeness:.0%} "
            f"data completeness (missing too many intervals to trust): {incomplete_meters[:10]}"
            + (" ..." if len(incomplete_meters) > 10 else "")
        )
        df = df[~df["meter_id"].isin(incomplete_meters)]

    n_after = len(df)
    print(f"Data quality pass complete: {n_before:,} -> {n_after:,} rows "
          f"({n_before - n_after:,} removed), {len(incomplete_meters)} meter(s) dropped entirely.")
    return df.reset_index(drop=True)


def load_real_cer_dataset(
    meter_read_filepaths,
    allocation_filepath: str = None,
    epoch_date: date = CER_TRIAL_EPOCH_DATE,
    min_completeness: float = 0.95,
    output_filepath: str = None,
) -> pd.DataFrame:
    
    df = load_cer_meter_reads(meter_read_filepaths, epoch_date=epoch_date)
    df = clean_cer_data(df, min_completeness=min_completeness)

    if allocation_filepath:
        allocations = load_cer_allocations(allocation_filepath)
        before = len(df)
        df = df.merge(allocations, on="meter_id", how="left")
        unmatched = df["segment_code"].isna().sum()
        if unmatched > 0:
            print(f" {unmatched} rows have no matching allocation record (meter_id not found in allocation file).")

    if output_filepath:
        out_dir = os.path.dirname(output_filepath)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        df.to_csv(output_filepath, index=False)
        print(f"Real CER dataset processed and saved: {output_filepath}")

    return df
