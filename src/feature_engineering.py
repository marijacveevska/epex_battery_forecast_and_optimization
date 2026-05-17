import numpy as np
import pandas as pd
from pathlib import Path
from workalendar.europe import Netherlands as NLCalendar

PROCESSED = Path(__file__).parent.parent / "Data" / "processed"
FEATURES = Path(__file__).parent.parent / "Data" / "features"


def add_price_lags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    price = df["price_eur_kwh"]
    df["price_lag_1h"] = price.shift(1)
    df["price_lag_24h"] = price.shift(24)
    df["price_lag_48h"] = price.shift(48)
    df["price_lag_168h"] = price.shift(168)
    return df


def add_rolling_stats(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["price_rolling_mean_24h"] = df["price_eur_kwh"].rolling(24, min_periods=1).mean()
    df["price_rolling_std_24h"] = df["price_eur_kwh"].rolling(24, min_periods=1).std()
    df["load_rolling_mean_24h"] = df["actual_load_mw"].rolling(24, min_periods=1).mean()
    df["load_rolling_std_24h"] = df["actual_load_mw"].rolling(24, min_periods=1).std()
    df["wind_rolling_mean_24h"] = df["wind_offshore"].rolling(24, min_periods=1).mean()
    df["wind_rolling_std_24h"] = df["wind_offshore"].rolling(24, min_periods=1).std()
    return df


def add_residual_load(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["residual_load"] = (
        df["actual_load_mw"]
        - df["wind_offshore"]
        - df["wind_onshore"]
        - df["solar"]
    )
    return df


def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    # Use UTC hour/dow/month; all timestamps are stored in UTC per project convention
    df = df.copy()
    hour = df.index.hour
    dow = df.index.dayofweek
    month = df.index.month
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    df["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)
    return df


def add_holiday_flag(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cal = NLCalendar()
    years = sorted(df.index.year.unique())
    holiday_dates = set()
    for year in years:
        for date, name in cal.holidays(year):
            holiday_dates.add(date)
    print(f"  Dutch holidays across {years[0]}–{years[-1]}: {len(holiday_dates)} dates")
    holiday_ts = pd.DatetimeIndex(
        pd.to_datetime(list(holiday_dates))
    ).tz_localize("UTC")
    df["is_holiday"] = df.index.normalize().isin(holiday_ts).astype(int)
    print(f"  Holiday hours flagged: {df['is_holiday'].sum():,}")
    return df


def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    # hydro_run_of_river is all zeros for Netherlands
    df = df.drop(columns=["hydro_run_of_river"])
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    sep = "=" * 60
    print(f"Input  : {df.shape[0]:,} rows × {df.shape[1]} columns")

    steps = [
        ("drop_unused_columns", drop_unused_columns),
        ("add_price_lags", add_price_lags),
        ("add_rolling_stats", add_rolling_stats),
        ("add_residual_load", add_residual_load),
        ("add_cyclical_features", add_cyclical_features),
        ("add_holiday_flag", add_holiday_flag),
    ]

    for name, fn in steps:
        print(f"\n[{name}]")
        df = fn(df)
        nan_count = int(df.isna().sum().sum())
        print(f"  Shape  : {df.shape[0]:,} rows × {df.shape[1]} columns")
        print(f"  NaN    : {nan_count:,} cells total")
        print(f"  Columns: {list(df.columns)}")

    print(f"\n{sep}")
    print("NaN report — by column")
    print(sep)
    nan_counts = df.isna().sum()
    total_nan = int(nan_counts.sum())
    if total_nan == 0:
        print("  No NaN values")
    else:
        print(f"  Total NaN cells: {total_nan:,}")
        print(nan_counts[nan_counts > 0].to_string())

    print(f"\n{sep}")
    print("Train / test split")
    print(sep)
    train = df[df.index.year <= 2024]
    test = df[df.index.year == 2025]

    train_nan = int(train.isna().sum().sum())
    test_nan = int(test.isna().sum().sum())
    print(f"  Train (2021-2024): {train.shape[0]:,} rows | {train.index.min()} → {train.index.max()}")
    print(f"  Test  (2025)     : {test.shape[0]:,} rows  | {test.index.min()} → {test.index.max()}")
    print(f"  Train NaN cells  : {train_nan:,}")
    print(f"  Test  NaN cells  : {test_nan:,}")

    FEATURES.mkdir(parents=True, exist_ok=True)
    train_path = FEATURES / "features_train.parquet"
    test_path = FEATURES / "features_test.parquet"
    train.to_parquet(train_path)
    test.to_parquet(test_path)
    print(f"\n  Saved → {train_path}")
    print(f"  Saved → {test_path}")

    return df


if __name__ == "__main__":
    sep = "=" * 60
    print(sep)
    print("Feature engineering pipeline")
    print(sep)

    merged_path = PROCESSED / "merged.parquet"
    print(f"Loading {merged_path}")
    df_raw = pd.read_parquet(merged_path)
    print(f"Loaded : {df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns")
    print(f"Range  : {df_raw.index.min()} → {df_raw.index.max()}")
    print()

    result = build_features(df_raw)

    train = result[result.index.year <= 2024]
    test = result[result.index.year == 2025]

    print(f"\n{sep}")
    print("Final summary")
    print(sep)
    print(f"\nFull feature matrix : {result.shape[0]:,} rows × {result.shape[1]} columns")
    print(f"\nAll columns ({result.shape[1]}):")
    for i, col in enumerate(result.columns, 1):
        print(f"  {i:2}. {col}")

    print(f"\nTrain set: {train.shape[0]:,} rows × {train.shape[1]} columns")
    print(f"  {train.index.min()} → {train.index.max()}")
    print(f"\nTest  set: {test.shape[0]:,} rows × {test.shape[1]} columns")
    print(f"  {test.index.min()} → {test.index.max()}")

    new_cols = [c for c in result.columns if c not in df_raw.columns]
    print(f"\nDescriptive stats — engineered features ({len(new_cols)} columns):")
    print(result[new_cols].describe().to_string())
