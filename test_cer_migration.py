import os
from datetime import date, datetime
import pandas as pd
from src.ingestion import (
    decode_cer_timecode,
    load_cer_meter_reads,
    load_cer_allocations,
    clean_cer_data,
    load_real_cer_dataset,
    CER_TRIAL_EPOCH_DATE,
)

def test_decode_cer_timecode_hand_verified():
    result = decode_cer_timecode(101, epoch_date=date(2009, 1, 1))
    expected = datetime(2009, 1, 1, 0, 0, 0)
    assert result == expected, f"Expected {expected}, got {result}"
    print(f"Timecode 00101 -> {result} (matches hand-verified {expected})")

def test_decode_cer_timecode_midday():

    result = decode_cer_timecode(19525, epoch_date=date(2009, 1, 1))
    expected_date = date(2009, 1, 1)
    from datetime import timedelta
    expected = datetime.combine(expected_date + timedelta(days=194), datetime.min.time()) + timedelta(hours=12)
    assert result == expected, f"Expected {expected}, got {result}"
    print(f"Timecode 19525 -> {result} (matches hand-computed {expected})")

def test_decode_cer_timecode_last_interval_of_day():
   
    result = decode_cer_timecode(148, epoch_date=date(2009, 1, 1))
    expected = datetime(2009, 1, 1, 23, 30, 0)
    assert result == expected, f"Expected {expected}, got {result}"
    print(f"Timecode 00148 -> {result} (matches hand-verified {expected}, last interval of day 1)")

def test_decode_cer_timecode_rejects_invalid_halfhour():
  
    try:
        decode_cer_timecode(100)  # halfhour code 0
        assert False, "Expected ValueError for halfhour code 0"
    except ValueError:
        print("Correctly rejected halfhour code 0")

    try:
        decode_cer_timecode(149)  # halfhour code 49
        assert False, "Expected ValueError for halfhour code 49"
    except ValueError:
        print("Correctly rejected halfhour code 49")

def _build_synthetic_fixture(tmp_path: str, n_meters: int = 3, n_days: int = 3):
    
    os.makedirs(tmp_path, exist_ok=True)
    filepath = os.path.join(tmp_path, "File1.txt")
    lines = []
    for meter_id in range(2000, 2000 + n_meters):
        for day_code in range(1, n_days + 1):
            for halfhour_code in range(1, 49):
                timecode = day_code * 100 + halfhour_code
                kwh = 0.5  # constant for simplicity
                lines.append(f"{meter_id} {timecode} {kwh}")

    # Inject one duplicate, one negative value, and one meter with missing data
    lines.append(f"2000 101 0.5")  # exact duplicate of an existing row
    lines[10] = lines[10].rsplit(" ", 1)[0] + " -0.3"  # corrupt one reading to negative

    with open(filepath, "w") as f:
        f.write("\n".join(lines))

    return filepath

def test_end_to_end_synthetic_fixture():
    
    tmp_dir = "/tmp/cer_fixture_test"
    filepath = _build_synthetic_fixture(tmp_dir, n_meters=3, n_days=3)

    df = load_cer_meter_reads(filepath, epoch_date=date(2009, 1, 1))
    assert set(df.columns) == {"meter_id", "timestamp", "kwh"}
    assert df["meter_id"].nunique() == 3
    print(f"Loaded {len(df)} rows across {df['meter_id'].nunique()} meters, canonical schema confirmed.")

    n_before_negative = (df["kwh"] < 0).sum()
    assert n_before_negative >= 1, "Fixture should contain at least one negative value before cleaning"

    cleaned = clean_cer_data(df, min_completeness=0.5)
    assert (cleaned["kwh"] < 0).sum() == 0, "Negative values should be clipped to 0 after cleaning"
    assert len(cleaned) < len(df), "Cleaning should have removed at least the duplicate row"
    print(f"Cleaning pass: {len(df)} -> {len(cleaned)} rows, all negatives clipped.")

if __name__ == "__main__":
    test_decode_cer_timecode_hand_verified()
    test_decode_cer_timecode_midday()
    test_decode_cer_timecode_last_interval_of_day()
    test_decode_cer_timecode_rejects_invalid_halfhour()
    test_end_to_end_synthetic_fixture()
    print("\nAll CER migration scaffolding tests passed.")
