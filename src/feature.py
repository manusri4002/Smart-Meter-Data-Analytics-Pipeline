import pandas as pd


def extract_load_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    # Ensure timestamp is datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    # Extract time components
    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["date"] = df["timestamp"].dt.date
    # Define Time-of-Use (TOU) windows (Irish grid standard intervals)
    df["is_night"] = df["hour"].isin([23, 0, 1, 2, 3, 4, 5, 6])
    df["is_morn_peak"] = df["hour"].isin([7, 8, 9, 10])
    df["is_eve_peak"] = df["hour"].isin([17, 18, 19, 20, 21])
    df["is_weekend"] = df["dayofweek"] >= 5

    features = []
    # Group by individual meter
    for meter_id, group in df.groupby("meter_id"):
        tot_kwh = group["kwh"].sum()
        mean_kwh = group["kwh"].mean()
        max_kwh = group["kwh"].max()
        std_kwh = group["kwh"].std()

        # Load Factor = Average Load / Peak Load (Key grid efficiency metric)
        load_factor = (mean_kwh / max_kwh) if max_kwh > 0 else 0.0

        # Consumption Ratios across Time-of-Use windows
        night_kwh = group[group["is_night"]]["kwh"].sum()
        morn_kwh = group[group["is_morn_peak"]]["kwh"].sum()
        eve_kwh = group[group["is_eve_peak"]]["kwh"].sum()
        night_ratio = (night_kwh / tot_kwh) if tot_kwh > 0 else 0.0
        morn_peak_ratio = (morn_kwh / tot_kwh) if tot_kwh > 0 else 0.0
        eve_peak_ratio = (eve_kwh / tot_kwh) if tot_kwh > 0 else 0.0

        # Weekend vs Weekday load ratio.
        # Guards against wknd_kwh being NaN (a meter whose date range
        # happens to contain no weekend days at all - e.g. a very short
        # observation window) in addition to guarding against wkdy_kwh
        # being 0/NaN.
        wknd_kwh = group[group["is_weekend"]]["kwh"].mean()
        wkdy_kwh = group[~group["is_weekend"]]["kwh"].mean()
        if pd.notna(wkdy_kwh) and wkdy_kwh > 0 and pd.notna(wknd_kwh):
            weekend_ratio = wknd_kwh / wkdy_kwh
        else:
            weekend_ratio = 1.0

        sorted_dates = sorted(group["date"].unique())
        midpoint = len(sorted_dates) // 2
        first_half_dates = set(sorted_dates[:midpoint]) if midpoint > 0 else set()
        second_half_dates = set(sorted_dates[midpoint:])
        first_half_mean = group[group["date"].isin(first_half_dates)]["kwh"].mean() if first_half_dates else float("nan")
        second_half_mean = group[group["date"].isin(second_half_dates)]["kwh"].mean() if second_half_dates else float("nan")
        if pd.notna(first_half_mean) and first_half_mean > 0 and pd.notna(second_half_mean):
            half_period_ratio = second_half_mean / first_half_mean
        else:
            half_period_ratio = 1.0

        daily_totals = group.groupby("date")["kwh"].sum()
        daily_consumption_std = daily_totals.std() if len(daily_totals) > 1 else 0.0

        features.append(
            {
                "meter_id": meter_id,
                "tot_kwh": round(tot_kwh, 2),
                "mean_kwh": round(mean_kwh, 4),
                "max_kwh": round(max_kwh, 4),
                "std_kwh": round(std_kwh, 4),
                "load_factor": round(load_factor, 4),
                "night_ratio": round(night_ratio, 4),
                "morn_peak_ratio": round(morn_peak_ratio, 4),
                "eve_peak_ratio": round(eve_peak_ratio, 4),
                "weekend_ratio": round(float(weekend_ratio), 4),
                "half_period_ratio": round(float(half_period_ratio), 4),
                "daily_consumption_std": round(float(daily_consumption_std), 4),
            }
        )

    df_features = pd.DataFrame(features)
    print(
        f"Extracted load profile features for {len(df_features)} unique meters."
    )
    return df_features
