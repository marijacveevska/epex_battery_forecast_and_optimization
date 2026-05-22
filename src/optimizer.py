"""MPC battery optimizer and 2025 backtest."""
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import lightgbm as lgb
import numpy as np
import pandas as pd
import pulp
import sys

sys.path.insert(0, str(Path(__file__).parent))
from data_loading import load_consumption

# ── Battery and grid constants ─────────────────────────────────────────────
CAPACITY_KWH = 1000.0
MAX_POWER_KW = 500.0
SOC_MIN_KWH = 100.0          # 10 % of capacity
SOC_MAX_KWH = 900.0          # 90 % of capacity
CHARGE_EFF = 0.9487          # sqrt(0.90) round-trip efficiency
DISCHARGE_EFF = 0.9487
GRID_IMPORT_LIMIT_KWH = 350.0
DEGRADATION_COST_EUR_KWH = 0.0001

RESULTS_DIR = Path(__file__).parent.parent / "Optimization" / "results"
PLOTS_DIR = Path(__file__).parent.parent / "Analytics" / "plots"


# ── LP dispatch solver ─────────────────────────────────────────────────────
def solve_dispatch(
    price_forecast_24h: np.ndarray,
    consumption_24h: np.ndarray,
    current_soc_kwh: float,
) -> dict:
    """
    Solve a 24-step LP to find the optimal battery dispatch.

    Variables (all kWh per hour, AC-side):
      charge[t]    energy consumed from grid for charging     [0, MAX_POWER_KW]
      discharge[t] energy made available by battery           [0, MAX_POWER_KW]
      grid_imp[t]  total energy imported from grid            [0, GRID_IMPORT_LIMIT]
      grid_exp[t]  total energy exported to grid              [0, inf)
      soc_next[t]  SOC at end of hour t                      [SOC_MIN, SOC_MAX]

    Energy balance: grid_imp[t] + discharge[t] = consumption[t] + charge[t] + grid_exp[t]
    SOC update    : soc[t+1] = soc[t] + charge[t]*CHARGE_EFF - discharge[t]/DISCHARGE_EFF

    Objective: maximise (grid_exp - grid_imp)*price - degradation*(charge + discharge)
    Equivalent to maximising savings vs paying market price for all consumption.
    """
    H = len(price_forecast_24h)

    prob = pulp.LpProblem("dispatch", pulp.LpMaximize)

    charge = [pulp.LpVariable(f"c{t}", 0, MAX_POWER_KW) for t in range(H)]
    discharge = [pulp.LpVariable(f"d{t}", 0, MAX_POWER_KW) for t in range(H)]
    grid_imp = [pulp.LpVariable(f"m{t}", 0, GRID_IMPORT_LIMIT_KWH) for t in range(H)]
    grid_exp = [pulp.LpVariable(f"e{t}", 0) for t in range(H)]
    soc_next = [pulp.LpVariable(f"s{t}", SOC_MIN_KWH, SOC_MAX_KWH) for t in range(H)]

    soc_prev = current_soc_kwh  # float at t=0, LpVariable for t>0

    for t in range(H):
        prob += grid_imp[t] + discharge[t] == float(consumption_24h[t]) + charge[t] + grid_exp[t]
        prob += soc_next[t] == soc_prev + charge[t] * CHARGE_EFF - discharge[t] / DISCHARGE_EFF
        soc_prev = soc_next[t]

    p = [float(x) for x in price_forecast_24h]
    prob += pulp.lpSum(
        (grid_exp[t] - grid_imp[t]) * p[t]
        - DEGRADATION_COST_EUR_KWH * (charge[t] + discharge[t])
        for t in range(H)
    )

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    def _v(var):
        v = pulp.value(var)
        return float(v) if v is not None else 0.0

    c = [max(0.0, _v(charge[t])) for t in range(H)]
    d = [max(0.0, _v(discharge[t])) for t in range(H)]
    m = [max(0.0, _v(grid_imp[t])) for t in range(H)]
    e = [max(0.0, _v(grid_exp[t])) for t in range(H)]
    s = [current_soc_kwh] + [
        float(np.clip(_v(soc_next[t]), SOC_MIN_KWH, SOC_MAX_KWH)) for t in range(H)
    ]

    return {
        "charge_kwh": c,
        "discharge_kwh": d,
        "grid_import_kwh": m,
        "grid_export_kwh": e,
        "soc_kwh": s,
        "expected_revenue_eur": sum((e[t] - m[t]) * p[t] for t in range(H)),
    }


# ── Naive rule-based strategy ──────────────────────────────────────────────
def _naive_step(soc: float, consumption: float, hour: int):
    """Charge 02-05 UTC, discharge 17-20 UTC, otherwise do nothing."""
    if hour in {2, 3, 4, 5}:
        headroom = max(0.0, GRID_IMPORT_LIMIT_KWH - consumption)
        charge = min(MAX_POWER_KW, (SOC_MAX_KWH - soc) / CHARGE_EFF, headroom)
        charge = max(0.0, charge)
        discharge = 0.0
    elif hour in {17, 18, 19, 20}:
        discharge = min(MAX_POWER_KW, (soc - SOC_MIN_KWH) * DISCHARGE_EFF)
        discharge = max(0.0, discharge)
        charge = 0.0
    else:
        charge = 0.0
        discharge = 0.0

    net = consumption + charge - discharge
    grid_import = max(0.0, min(GRID_IMPORT_LIMIT_KWH, net))
    grid_export = max(0.0, -net)
    new_soc = float(np.clip(
        soc + charge * CHARGE_EFF - discharge / DISCHARGE_EFF,
        SOC_MIN_KWH, SOC_MAX_KWH,
    ))
    return charge, discharge, grid_import, grid_export, new_soc


# ── Backtest ───────────────────────────────────────────────────────────────
def run_backtest(
    forecast_df: pd.DataFrame,
    consumption_series: pd.Series,
    initial_soc_kwh: float = 500.0,
) -> pd.DataFrame:
    """
    Roll the MPC one hour at a time across all 2025 hours.

    forecast_df must have columns: actual_price, forecast_price.
    consumption_series must be hourly kWh indexed by UTC datetime.

    Revenue = savings vs no-battery baseline (paying market price for all consumption):
        revenue[t] = (discharge[t] - charge[t]) * actual_price[t]
    """
    datetimes = forecast_df.index
    n = len(datetimes)

    records = []
    soc_m = initial_soc_kwh
    soc_n = initial_soc_kwh
    soc_p = initial_soc_kwh

    t0 = time.time()
    print(f"Starting backtest: {n} hours | 2 LP solves/hour (model + perfect foresight)")

    for i, dt in enumerate(datetimes):
        if i % 500 == 0 and i > 0:
            elapsed = time.time() - t0
            eta = elapsed / i * (n - i)
            print(f"  Hour {i:5d}/{n} | {dt.date()} | elapsed {elapsed:.0f}s | ETA {eta:.0f}s")

        actual_price = float(forecast_df.loc[dt, "actual_price"])
        consumption = float(consumption_series.loc[dt])

        # Rolling 24-h look-ahead (shorter at end of year)
        horizon = min(24, n - i)
        idx = datetimes[i: i + horizon]
        p_fc = forecast_df.loc[idx, "forecast_price"].values.astype(float)
        p_act = forecast_df.loc[idx, "actual_price"].values.astype(float)
        cons_24h = consumption_series.reindex(idx).fillna(consumption).values.astype(float)

        # ── Model MPC (forecasted prices) ─────────────────────────────────
        res_m = solve_dispatch(p_fc, cons_24h, soc_m)
        c_m = res_m["charge_kwh"][0]
        d_m = res_m["discharge_kwh"][0]
        imp_m = res_m["grid_import_kwh"][0]
        exp_m = res_m["grid_export_kwh"][0]
        soc_m = float(np.clip(
            soc_m + c_m * CHARGE_EFF - d_m / DISCHARGE_EFF,
            SOC_MIN_KWH, SOC_MAX_KWH,
        ))
        rev_m = (d_m - c_m) * actual_price

        # ── Naive ─────────────────────────────────────────────────────────
        c_n, d_n, imp_n, exp_n, soc_n = _naive_step(soc_n, consumption, dt.hour)
        rev_n = (d_n - c_n) * actual_price

        # ── Perfect foresight MPC (actual prices) ─────────────────────────
        res_p = solve_dispatch(p_act, cons_24h, soc_p)
        c_p = res_p["charge_kwh"][0]
        d_p = res_p["discharge_kwh"][0]
        imp_p = res_p["grid_import_kwh"][0]
        exp_p = res_p["grid_export_kwh"][0]
        soc_p = float(np.clip(
            soc_p + c_p * CHARGE_EFF - d_p / DISCHARGE_EFF,
            SOC_MIN_KWH, SOC_MAX_KWH,
        ))
        rev_p = (d_p - c_p) * actual_price

        records.append({
            "datetime": dt,
            "actual_price": actual_price,
            "forecast_price": float(forecast_df.loc[dt, "forecast_price"]),
            "consumption_kwh": consumption,
            "soc_model": soc_m,
            "charge_model": c_m,
            "discharge_model": d_m,
            "grid_import_model": imp_m,
            "grid_export_model": exp_m,
            "revenue_model": rev_m,
            "soc_naive": soc_n,
            "charge_naive": c_n,
            "discharge_naive": d_n,
            "grid_import_naive": imp_n,
            "grid_export_naive": exp_n,
            "revenue_naive": rev_n,
            "soc_perfect": soc_p,
            "charge_perfect": c_p,
            "discharge_perfect": d_p,
            "grid_import_perfect": imp_p,
            "grid_export_perfect": exp_p,
            "revenue_perfect": rev_p,
        })

    elapsed = time.time() - t0
    print(f"Backtest complete: {n} hours in {elapsed:.1f}s ({elapsed/n*1000:.1f} ms/hour)")
    return pd.DataFrame(records).set_index("datetime")


# ── Plots ──────────────────────────────────────────────────────────────────
def plot_results(results_df: pd.DataFrame) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    def _save(fig, name):
        for d in [RESULTS_DIR, PLOTS_DIR]:
            fig.savefig(d / name, dpi=150, bbox_inches="tight")
        plt.close(fig)

    # 1. SOC evolution
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(results_df.index, results_df["soc_model"], lw=0.7, color="steelblue", label="Model MPC")
    ax.plot(results_df.index, results_df["soc_naive"], lw=0.7, color="darkorange", label="Naive")
    ax.plot(results_df.index, results_df["soc_perfect"], lw=0.7, color="seagreen", label="Perfect Foresight")
    ax.axhline(SOC_MIN_KWH, color="red", ls="--", lw=0.8, label=f"SOC min ({SOC_MIN_KWH:.0f} kWh)")
    ax.axhline(SOC_MAX_KWH, color="purple", ls="--", lw=0.8, label=f"SOC max ({SOC_MAX_KWH:.0f} kWh)")
    ax.set_xlabel("Date")
    ax.set_ylabel("State of Charge (kWh)")
    ax.set_title("Battery State of Charge — 2025")
    ax.legend(loc="upper right", fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.tight_layout()
    _save(fig, "soc_evolution_2025.png")
    print("Saved: soc_evolution_2025.png")

    # 2. Cumulative revenue
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(results_df.index, results_df["revenue_model"].cumsum(), color="steelblue", label="Model MPC")
    ax.plot(results_df.index, results_df["revenue_naive"].cumsum(), color="darkorange", label="Naive")
    ax.plot(results_df.index, results_df["revenue_perfect"].cumsum(), color="seagreen", label="Perfect Foresight")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Savings (EUR)")
    ax.set_title("Cumulative Battery Savings vs No-Battery Baseline — 2025")
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.tight_layout()
    _save(fig, "cumulative_revenue_2025.png")
    print("Saved: cumulative_revenue_2025.png")

    # 3. Monthly revenue bar chart
    monthly = results_df[["revenue_model", "revenue_naive", "revenue_perfect"]].resample("ME").sum()
    x = np.arange(len(monthly))
    w = 0.25
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - w, monthly["revenue_model"], w, color="steelblue", label="Model MPC")
    ax.bar(x, monthly["revenue_naive"], w, color="darkorange", label="Naive")
    ax.bar(x + w, monthly["revenue_perfect"], w, color="seagreen", label="Perfect Foresight")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([d.strftime("%b") for d in monthly.index])
    ax.set_xlabel("Month")
    ax.set_ylabel("Savings (EUR)")
    ax.set_title("Monthly Battery Savings by Strategy — 2025")
    ax.legend(fontsize=9)
    plt.tight_layout()
    _save(fig, "monthly_revenue_2025.png")
    print("Saved: monthly_revenue_2025.png")

    # 4. Sample week Jan 1-7
    w_start = pd.Timestamp("2025-01-01", tz="UTC")
    w_end = pd.Timestamp("2025-01-07 23:00", tz="UTC")
    week = results_df.loc[w_start:w_end]

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    axes[0].plot(week.index, week["actual_price"], color="steelblue", label="Actual")
    axes[0].plot(week.index, week["forecast_price"], color="darkorange", ls="--", lw=0.9, label="Forecast")
    axes[0].set_ylabel("EUR/kWh")
    axes[0].set_title("Day-Ahead Price")
    axes[0].legend(loc="upper right", fontsize=8)

    axes[1].fill_between(week.index, 0, week["consumption_kwh"], color="slategray", alpha=0.7)
    axes[1].set_ylabel("kWh/h")
    axes[1].set_title("Building Consumption")

    axes[2].plot(week.index, week["soc_model"], color="steelblue", label="Model MPC")
    axes[2].plot(week.index, week["soc_naive"], color="darkorange", ls="--", lw=0.9, label="Naive")
    axes[2].plot(week.index, week["soc_perfect"], color="seagreen", ls=":", lw=0.9, label="Perfect")
    axes[2].axhline(SOC_MIN_KWH, color="red", ls="--", lw=0.6)
    axes[2].axhline(SOC_MAX_KWH, color="purple", ls="--", lw=0.6)
    axes[2].set_ylabel("kWh")
    axes[2].set_title("State of Charge")
    axes[2].legend(loc="upper right", fontsize=8)

    bar_w = pd.Timedelta(hours=1) * 0.85
    axes[3].bar(week.index, week["charge_model"], width=bar_w, align="edge",
                color="royalblue", label="Charge")
    axes[3].bar(week.index, -week["discharge_model"], width=bar_w, align="edge",
                color="tomato", label="Discharge")
    axes[3].axhline(0, color="black", lw=0.5)
    axes[3].set_ylabel("kWh/h")
    axes[3].set_title("Model MPC Dispatch  (Charge +, Discharge −)")
    axes[3].legend(loc="upper right", fontsize=8)

    axes[3].xaxis.set_major_formatter(mdates.DateFormatter("%a %d"))
    axes[3].xaxis.set_major_locator(mdates.DayLocator())
    plt.tight_layout()
    _save(fig, "sample_week_jan2025.png")
    print("Saved: sample_week_jan2025.png")


# ── Summary table ──────────────────────────────────────────────────────────
def print_summary(results_df: pd.DataFrame) -> None:
    sep = "=" * 65

    total_m = results_df["revenue_model"].sum()
    total_n = results_df["revenue_naive"].sum()
    total_p = results_df["revenue_perfect"].sum()

    cycles_m = results_df["discharge_model"].sum() / CAPACITY_KWH
    cycles_n = results_df["discharge_naive"].sum() / CAPACITY_KWH
    cycles_p = results_df["discharge_perfect"].sum() / CAPACITY_KWH

    n_days = len(results_df) / 24

    peak_import = results_df["grid_import_model"].max()
    near_limit = int((results_df["grid_import_model"] > GRID_IMPORT_LIMIT_KWH * 0.9).sum())
    near_pct = near_limit / len(results_df) * 100

    pct_m = total_m / total_p * 100 if total_p != 0 else float("nan")
    pct_n = total_n / total_p * 100 if total_p != 0 else float("nan")

    print()
    print(sep)
    print("BACKTEST SUMMARY — 2025")
    print("Revenue = savings vs paying market price for all consumption from grid")
    print(sep)
    print(f"\n{'Metric':<32} {'Model MPC':>12} {'Naive':>10} {'Perfect':>10}")
    print("-" * 65)
    print(f"{'Total savings (EUR)':<32} {total_m:>12.2f} {total_n:>10.2f} {total_p:>10.2f}")
    print(f"{'Avg daily savings (EUR/day)':<32} {total_m/n_days:>12.2f} {total_n/n_days:>10.2f} {total_p/n_days:>10.2f}")
    print(f"{'Full cycles (discharge/cap)':<32} {cycles_m:>12.1f} {cycles_n:>10.1f} {cycles_p:>10.1f}")
    print(f"{'% of perfect foresight captured':<32} {pct_m:>11.1f}% {pct_n:>9.1f}%    100.0%")
    print("-" * 65)
    print(f"\n{'Peak grid import (kWh/h)':<42} {peak_import:>6.1f}  (limit: {GRID_IMPORT_LIMIT_KWH:.0f})")
    print(f"{'Hours >90% of grid limit':<42} {near_limit:>6d}  ({near_pct:.1f}% of hours)")
    print(sep)


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 65)
    print("EPEX Battery MPC Backtest — 2025")
    print("=" * 65)

    # Find latest lgbm_point model
    model_dir = Path(__file__).parent.parent / "ML_model" / "models"
    model_files = sorted(model_dir.glob("lgbm_point*.txt"))
    if not model_files:
        raise FileNotFoundError(f"No lgbm_point*.txt found in {model_dir}")
    model_path = model_files[-1]
    print(f"\nLoading model: {model_path.name}")
    model = lgb.Booster(model_file=str(model_path))

    # Load 2025 test features and generate price forecasts
    feat_path = Path(__file__).parent.parent / "Data" / "features" / "features_test.parquet"
    print(f"Loading features: {feat_path.name}")
    feat_df = pd.read_parquet(feat_path)
    feat_cols = [c for c in feat_df.columns if c != "price_eur_kwh"]
    forecasts = model.predict(feat_df[feat_cols])
    forecast_df = pd.DataFrame(
        {"actual_price": feat_df["price_eur_kwh"].values, "forecast_price": forecasts},
        index=feat_df.index,
    )
    print(f"Forecast: {len(forecast_df):,} rows | {forecast_df.index.min()} → {forecast_df.index.max()}")

    # Load and align building consumption
    print()
    consumption = load_consumption()
    consumption = consumption.reindex(forecast_df.index)
    n_missing = int(consumption.isna().sum())
    if n_missing:
        print(f"WARNING: {n_missing} missing consumption hours — forward-filling")
        consumption = consumption.ffill()
    print(f"Consumption aligned: {len(consumption):,} rows | {n_missing} hours filled")

    # Run backtest
    print()
    results = run_backtest(forecast_df, consumption)

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = RESULTS_DIR / "backtest_2025.csv"
    results.to_csv(out_csv)
    print(f"\nResults saved → {out_csv}  ({len(results):,} rows, {len(results.columns)} columns)")

    # Generate plots
    print("\nGenerating plots...")
    plot_results(results)

    # Print summary
    print_summary(results)
