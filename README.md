# Energy Price Forecasting & Battery Optimization — Netherlands
A machine learning and optimization project that predicts Dutch day-ahead electricity prices and uses those forecasts to optimize the charge/discharge schedule of a simulated battery storage system. Built in two parts: a fully reproducible historical backtest (Part 1) and a live real-time dispatch system (Part 2).

### What this project does
Part 1 trains a price forecasting model on Dutch electricity market data from 2022 to 2024, tests it on 2025 data it has never seen, and then runs a battery optimizer that uses those forecasts to decide when to charge and discharge — hour by hour — to maximize revenue. The results are benchmarked against a naive time-of-day strategy and a perfect foresight upper bound.\\

Part 2 takes the same pipeline and makes it live: data is fetched automatically, the model retrains daily, and the optimizer dispatches the battery every hour based on the latest forecast.

## Project structure
### Part 1 — Batch pipeline

Data collection: ENTSO-E Transparency Platform exports (day-ahead prices, load forecast and actual, generation mix by fuel type) and Open-Meteo historical weather (wind speed, solar irradiance, temperature) for the Netherlands, 2022–2024. \\

Feature engineering: hourly price lags (1h, 24h, 48h, 168h), rolling means and standard deviations, residual load (total load minus wind minus solar), cyclical time encodings (hour, weekday, month as sine/cosine pairs), Dutch public holiday flags, and cross-border flow features (NL-DE, NL-BE).\\

Model: LightGBM with quantile regression to produce point forecasts plus p10/p90 confidence intervals. Trained with walk-forward cross-validation to prevent data leakage.\\

Optimization: Model Predictive Control using a linear program (PuLP/cvxpy). At each hour the optimizer receives the 24-hour ahead price forecast and current battery state, solves the optimal charge/discharge schedule, applies the first decision, and advances. \\

Backtest: full simulation over 2025 with three-way comparison — naive strategy (charge overnight, discharge evening peak) vs model-driven MPC vs perfect foresight MPC.

### Part 2 — Live system

Hourly data ingestion via ENTSO-E API and Open-Meteo API, stored in a local DuckDB database.
Daily model refit triggered automatically at 13:30 CET after new day-ahead prices are published by ENTSO-E.
Hourly optimizer dispatch: reads latest forecast and current SOC from the database, solves the LP, logs the decision and resulting revenue.
Streamlit dashboard showing live SOC, today's forecast vs actual prices, cumulative revenue, and strategy comparison.
Daily email digest with performance summary.

