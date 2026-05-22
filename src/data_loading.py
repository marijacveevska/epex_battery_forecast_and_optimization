from pathlib import Path

import pandas as pd

RAW = Path(__file__).parent.parent / "Data" / "raw"
PROCESSED = Path(__file__).parent.parent / "Data" / "processed"

_GENMIX_RENAME = {
    "Biomass": "biomass",
    "Fossil Gas": "fossil_gas",
    "Fossil Hard coal": "fossil_hard_coal",
    "Hydro Run-of-river and poundage": "hydro_run_of_river",
    "Nuclear": "nuclear",
    "Other": "other",
    "Solar": "solar",
    "Waste": "waste",
    "Wind Offshore": "wind_offshore",
    "Wind Onshore": "wind_onshore",
}


def load_prices() -> pd.Series:
    price_dir = RAW / "manual" / "day_ahead_prices"
    files = sorted(price_dir.glob("Netherlands_*.csv"))
    print(f"[prices] Found {len(files)} files: {[f.name for f in files]}")

    frames = []
    for f in files:
        df_year = pd.read_csv(f)
        print(f"  {f.name}: {len(df_year):,} rows")
        frames.append(df_year)

    df = pd.concat(frames, ignore_index=True)
    print(f"[prices] After concat: {len(df):,} rows")

    df["Datetime (UTC)"] = pd.to_datetime(df["Datetime (UTC)"], utc=True)
    df = df.set_index("Datetime (UTC)")
    df = df.drop(columns=["Country", "ISO3 Code", "Datetime (Local)"])
    s = df["Price (EUR/kWhe)"].rename("price_eur_kwh")

    dupes = s.index.duplicated().sum()
    missing = s.isna().sum()
    print(f"[prices] {len(s):,} rows | {s.index.min()} → {s.index.max()}")
    print(f"[prices] Duplicates: {dupes} | Missing: {missing}")
    return s


def load_weather() -> pd.DataFrame:
    path = RAW / "manual" / "meteo" / "open-meteo-2021-2025.csv"
    print(f"[weather] Loading {path.name}")

    df = pd.read_csv(path, skiprows=3)
    print(f"[weather] Raw rows: {len(df):,} columns: {list(df.columns)}")

    df["time"] = pd.to_datetime(df["time"]).dt.tz_localize("UTC")
    df = df.set_index("time")

    df.columns = [
        "temperature_2m",
        "wind_speed_10m_kmh",
        "wind_speed_100m_kmh",
        "cloud_cover_pct",
        "shortwave_radiation_wm2",
        "direct_radiation_wm2",
    ]

    missing = df.isna().sum()
    print(f"[weather] {len(df):,} rows | {df.index.min()} → {df.index.max()}")
    print(f"[weather] Missing values:\n{missing.to_string()}")
    return df


def _load_entso_e_15min(path: Path, col_name: str) -> pd.Series:
    df = pd.read_csv(path, index_col=0)
    print(f"  Raw rows: {len(df):,} | index sample: {df.index[0]!r}")

    df.index = pd.to_datetime(df.index, utc=True)
    print(f"  After UTC parse: {df.index.min()} → {df.index.max()}")

    s = df.iloc[:, 0]
    missing_before = s.isna().sum()
    print(f"  Missing before resample: {missing_before}")

    s = s.resample("1h").mean()
    missing_after = s.isna().sum()
    print(f"  After hourly resample: {len(s):,} rows | Missing: {missing_after}")

    return s.rename(col_name)


def load_actual_load() -> pd.Series:
    print("[actual_load] Loading nl_actual_load.csv")
    s = _load_entso_e_15min(RAW / "entso_e" / "nl_actual_load.csv", "actual_load_mw")
    print(f"[actual_load] {len(s):,} rows | {s.index.min()} → {s.index.max()}")
    return s


def load_load_forecast() -> pd.Series:
    print("[load_forecast] Loading nl_load_forecast.csv")
    s = _load_entso_e_15min(RAW / "entso_e" / "nl_load_forecast.csv", "load_forecast_mw")
    print(f"[load_forecast] {len(s):,} rows | {s.index.min()} → {s.index.max()}")
    return s


def load_generation_mix() -> pd.DataFrame:
    path = RAW / "entso_e" / "nl_generation_mix.csv"
    print(f"[generation_mix] Loading {path.name}")

    df = pd.read_csv(path, header=[0, 1], index_col=0)
    print(f"[generation_mix] Raw rows: {len(df):,} | {len(df.columns)} columns (multi-level)")
    print(f"  Fuel types: {list(df.columns.get_level_values(0).unique())}")

    df.index = pd.to_datetime(df.index, utc=True)
    print(f"[generation_mix] After UTC parse: {df.index.min()} → {df.index.max()}")

    agg = df.xs("Actual Aggregated", axis=1, level=1)
    print(f"[generation_mix] After keeping Actual Aggregated: {len(agg.columns)} columns")

    unmapped = [c for c in agg.columns if c not in _GENMIX_RENAME]
    if unmapped:
        print(f"  WARNING: unmapped fuel types dropped: {unmapped}")
    agg = agg.rename(columns=_GENMIX_RENAME)

    agg = agg.resample("1h").mean()
    missing = agg.isna().sum()
    print(f"[generation_mix] After hourly resample: {len(agg):,} rows")
    print(f"[generation_mix] Missing values per column:\n{missing.to_string()}")
    return agg


def load_consumption() -> pd.Series:
    path = RAW / "manual" / "building_consumption_2025.csv"
    print(f"[consumption] Loading {path.name}")

    df = pd.read_csv(path, header=None, names=["timestamp", "consumption_kwh"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%m/%d/%y %H:%M")
    df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    df = df.set_index("timestamp")

    s = df["consumption_kwh"].resample("1h").sum()
    s.name = "consumption_kwh"

    print(f"[consumption] {len(s):,} rows | {s.index.min()} → {s.index.max()}")
    print(f"[consumption] Min: {s.min():.1f} kWh | Max: {s.max():.1f} kWh | Mean: {s.mean():.1f} kWh")
    return s


def merge_all() -> pd.DataFrame:
    sep = "=" * 60
    print(sep)
    print("Loading all datasets")
    print(sep)

    prices = load_prices()
    print()
    weather = load_weather()
    print()
    actual_load = load_actual_load()
    print()
    load_forecast = load_load_forecast()
    print()
    generation_mix = load_generation_mix()

    print()
    print(sep)
    print("Merging on UTC hourly index (inner join)")
    print(sep)

    merged = (
        prices.to_frame()
        .join(weather, how="inner")
        .join(actual_load, how="inner")
        .join(load_forecast, how="inner")
        .join(generation_mix, how="inner")
    )

    print(f"Shape      : {merged.shape}")
    print(f"Date range : {merged.index.min()} → {merged.index.max()}")
    print(f"Columns    : {list(merged.columns)}")

    missing = merged.isna().sum()
    total_missing = int(missing.sum())
    if total_missing == 0:
        print("Missing values: none")
    else:
        print(f"Missing values (total {total_missing}):")
        print(missing[missing > 0].to_string())

    PROCESSED.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED / "merged.parquet"
    merged.to_parquet(out_path)
    print(f"Saved → {out_path}")

    return merged


if __name__ == "__main__":
    df = merge_all()
    print()
    print("=" * 60)
    print("Final summary")
    print("=" * 60)
    print(df.describe().to_string())
