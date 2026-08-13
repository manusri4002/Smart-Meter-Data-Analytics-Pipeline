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

