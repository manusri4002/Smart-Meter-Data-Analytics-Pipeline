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

def test_end_to_end_synthetic_fixture():
    
    tmp_dir = "/tmp/cer_fixture_test"
    filepath = _build_synthetic_fixture(tmp_dir, n_meters=3, n_days=3)

if __name__ == "__main__":
    test_decode_cer_timecode_hand_verified()
    test_decode_cer_timecode_midday()
    test_decode_cer_timecode_last_interval_of_day()
    test_decode_cer_timecode_rejects_invalid_halfhour()
    test_end_to_end_synthetic_fixture()
    print("\n All CER migration scaffolding tests passed.")
