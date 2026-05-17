# Energy Price Forecasting & Battery Optimization — Netherlands

A machine learning and optimization project that predicts Dutch day-ahead electricity prices and uses those forecasts to simulate the optimal charge/discharge schedule of a home battery storage system. Built as a fully reproducible historical backtest on real market data from 2021 to 2025.

---

## What this project does

Trains a price forecasting model on Dutch electricity market data from 2021 to 2024, evaluates it on 2025 data it has never seen, and runs a battery optimizer that uses those forecasts to decide when to charge and discharge — hour by hour — to maximize revenue. Results are benchmarked against a naive time-of-day strategy and a perfect foresight upper bound.

---

## Data sources

All data covers the Netherlands, 2021–2025, hourly resolution.

Day-ahead prices — EPEX SPOT NL auction clearing prices downloaded from the ENTSO-E Transparency Platform. One price per hour for each of the 24 hours of the next day, in EUR/kWh.

Actual load and load forecast — Dutch electricity consumption (actual and day-ahead forecast) from ENTSO-E, resampled from 15-minute to hourly resolution.

Generation mix — Hourly generation per fuel type for the Netherlands from ENTSO-E: wind offshore, wind onshore, solar, fossil gas, nuclear, biomass, hard coal, waste, and other. Used to compute residual load.

Weather — Hourly wind speed at 10m and 100m, solar irradiance, temperature, and cloud cover for De Bilt (52.1N, 5.18E) from Open-Meteo historical API.

---

## Pipeline

1. Data loading — src/data_loading.py

Loads and cleans all five raw data sources, aligns them on a UTC hourly timestamp index, and merges into a single parquet file at Data/processed/merged.parquet. Handles timezone conversion from CET to UTC, 15-minute to hourly resampling, and multi-level column flattening for the generation mix.

2. Feature engineering — src/feature_engineering.py

Builds the full feature matrix from the merged dataset. Features include price lags at 1h, 24h, 48h, and 168h, rolling 24h mean and standard deviation for price, load, and wind, residual load computed as total load minus wind offshore, wind onshore, and solar, cyclical time encodings for hour of day, day of week, and month as sine and cosine pairs, and a Dutch public holiday flag via workalendar. Saves features_train.parquet covering 2021 to 2024 and features_test.parquet covering 2025, which is held out during all training decisions.

3. Price forecasting model — src/model.py

Trains three LightGBM models on the 2021 to 2024 training set using walk-forward cross-validation: a point forecast model with RMSE objective, a lower bound model with quantile regression at alpha 0.10, and an upper bound model with quantile regression at alpha 0.90. Evaluated on the held-out 2025 test set. Metrics reported: MAE, RMSE, MAPE, and directional accuracy.

4. Battery optimizer — src/optimizer.py

Model Predictive Control using a linear program built with PuLP. At each hour the optimizer receives the 24-hour ahead price forecast and the current battery state of charge, solves the optimal charge and discharge schedule for the next 24 hours, applies only the first decision, and advances to the next hour. Battery parameters: 1000 kWh capacity, 500 kW maximum charge and discharge power, SOC bounds of 10 to 90 %, 90 % round-trip efficiency, and a degradation cost penalty to prevent over-cycling.

5. Backtest — full simulation over 2025

Three-way benchmark comparison: naive strategy charging from 02:00 to 05:00 and discharging from 17:00 to 20:00, model-driven MPC using LightGBM price forecasts, and perfect foresight MPC using actual realized 2025 prices as the theoretical upper bound.

---

## Results

To be filled in after backtest is complete.

Key metrics reported: total revenue in EUR, number of full cycles completed, average daily revenue, forecast MAE and RMSE on the 2025 test set, and the percentage of perfect foresight revenue captured by the model-driven strategy.

---

## Project structure

epex_battery_forecast_and_optimization/
├── Data/
│   ├── raw/
│   │   ├── entso_e/          <- ENTSO-E API downloads
│   │   └── manual/           <- manually downloaded files
│   ├── processed/            <- merged.parquet
│   └── features/             <- features_train.parquet, features_test.parquet
├── ML_model/
│   ├── notebooks/            <- experimentation notebooks
│   ├── models/               <- saved LightGBM model files
│   └── evaluation/           <- forecast_vs_actual_2025.csv, metrics
├── Optimization/
│   └── results/              <- SOC trace, dispatch decisions, P&L
├── Analytics/
│   ├── notebooks/            <- EDA notebooks
│   ├── plots/                <- saved chart images
│   └── reports/              <- summary reports
├── src/
│   ├── data_loading.py
│   ├── feature_engineering.py
│   ├── model.py
│   └── optimizer.py
├── main.py
├── CLAUDE.md
├── README.md
├── requirements.txt
└── .gitignore

---

## Tech stack

Python 3.13, LightGBM, PuLP, pandas, numpy, scikit-learn, matplotlib, plotly, entsoe-py, workalendar, pyarrow.

---

## Background

Built to understand how Dutch electricity markets work, how price forecasting models are constructed and validated on real energy data, and how battery storage systems are optimized in practice using model predictive control. The Netherlands sits within the Central Western Europe price coupling zone alongside Germany, Belgium, and France, making it an interesting market for price forecasting due to strong cross-border dynamics and renewable energy variability.