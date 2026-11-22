# StockLab — Naïve vs ML Stock Forecasting Dashboard

A complete Python project with a **Streamlit dashboard**, a real **naïve persistence baseline**, a **Random Forest** forecasting model, and the existing **PyTorch Transformer** (optional). It builds on the Stock-Forecasting repository while preserving its original notebook and inference API.

> **Research and educational use only.** Market data may be incomplete, delayed, or incorrect; future prices cannot be predicted reliably. This project is not trading advice.

## Features

- **Naïve (last-close) model:** repeats the most recent observed close at every horizon; requires no training.
- **Random Forest ML model:** learns a direct multi-step forecast from OHLCV and rolling features; fast on CPU.
- **Transformer ML option:** original Transformer encoder and direct multi-horizon head, with early stopping.
- **Fairer evaluation:** train 70%, validation 10%, test 20%, in timestamp order. Scalers fit on training only, and the target scaler also scales the Close input feature. All models are evaluated on identical test windows.
- **Dashboard:** Yahoo Finance or uploaded CSV, plus an offline synthetic demonstration; interactive historical + future charts, next-session test chart, overall metrics and per-horizon errors, forecast and backtest CSV downloads, and metrics JSON download.
- **Exchange calendar:** official exchange sessions for live/uploaded data (`pandas-market-calendars`); offline demo uses weekdays only (explicitly labeled).
- **Tools:** CLI, companion notebook, tests, pre-existing FastAPI endpoint, Docker and CI.

## Quick start — Windows / VS Code

1. Extract the ZIP to a folder; **do not run the Python scripts while still inside the ZIP**.
2. In VS Code, **File → Open Folder**, select the extracted `Stock-Forecasting-Naive-ML-Dashboard` directory.
3. Open a terminal in that root directory:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Visit **http://localhost:8501**. You can also double-click `launch_dashboard.bat` after dependencies are installed.

If `py -3.12` is not available, use a supported Python 3.11/3.12 interpreter instead (for example `python -m venv .venv`). Ensure your VS Code kernel/interpreter points to this virtual environment.

## Quick start — macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Dashboard use

1. Set **Data source** to `Yahoo Finance`, `Demo (offline)`, or `Upload CSV`.
2. Select the ticker (e.g. `AAPL` or `RELIANCE.NS`), historical period and **exchange calendar** (`NYSE`, `NASDAQ`, `NSE`, `LSE`). The ticker is **not** automatically mapped to an exchange.
3. Select **Random Forest** (recommended for fast CPU demonstration) or **Transformer**.
4. Choose lookback, horizon and training hyperparameters. Click **Run forecast comparison**.
5. Compare Naïve vs ML error, inspect day-by-day MAE, examine the history/test plots, and download results.

**No internet?** Select `Demo (offline)`; this creates synthetic prices, not a real market series. It does not need live Yahoo Finance access or an exchange-holiday lookup. However you must install Python dependencies at least once.

### Uploaded CSV format

```csv
Date,Open,High,Low,Close,Volume
2025-01-02,100,104,99,102,123456
2025-01-03,102,105,100,103,134567
```

Column names are case-insensitive. Provide enough daily observations for training, validation and testing (for default settings, at least several hundred rows are recommended). Do not pass intraday rows with a daily horizon; the UI assumes **one observation per trading session**. Invalid, duplicate, and missing data are rejected.

## CLI: benchmark without the dashboard

```bash
# Offline synthetic demo — produces forecast, backtest, metrics files in outputs/
python -m ml.run_comparison --demo --ml-model random_forest

# Real market, Random Forest
python -m ml.run_comparison --ticker AAPL --period 5y --exchange NYSE --ml-model random_forest

# Real market, PyTorch Transformer
python -m ml.run_comparison --ticker RELIANCE.NS --period 5y --exchange NSE --ml-model transformer --epochs 15
```

Results in `outputs/` are excluded from git. `run_comparison` produces evaluation reports, not a loadable Random Forest model artifact. The older `python -m ml.run_training --ticker AAPL` is still available if you want to train and **save a Transformer checkpoint** for the FastAPI service.

## Notebooks

- `ml/compare_models.ipynb` — **new**, offline-first comparison notebook for both models and interactive Plotly charts.
- `ml/notebook.ipynb` — original improved Transformer training walkthrough, with project-root detection to avoid `ModuleNotFoundError: No module named 'ml'`.

Start Jupyter from the extracted project directory, and use the Python environment with all project dependencies installed.

## Existing FastAPI endpoint

```bash
python -m ml.run_training --ticker AAPL --exchange NYSE
python -m uvicorn api.main:app --reload
```

- Health: http://localhost:8000/health
- Swagger: http://localhost:8000/docs
- Existing Transformer inference (needs saved checkpoint): `GET /forecast/AAPL`
- Naïve baseline inference (no checkpoint needed): `GET /naive/AAPL?exchange=NYSE&horizon=7`

The Streamlit dashboard independently runs the benchmark and does **not** need the API to start. The API provides a fit-free naïve endpoint and a Transformer checkpoint endpoint; it does not serve the Random Forest benchmark. Do not load `.joblib` files from untrusted sources.

## Docker (optional)

```bash
docker compose up --build
```

- Dashboard: http://localhost:8501
- API: http://localhost:8000/docs (requires saved Transformer artifacts to predict)

Use local development only; secure/authenticate both services before any public hosting.

## Project structure

```text
app.py                          Interactive Streamlit dashboard
.streamlit/config.toml           Dashboard theme
ml/src/naive.py                  Naïve persistence forecast class
ml/src/comparison.py             Both models, fair holdout evaluation, future forecast
ml/src/sample_data.py            Offline fake OHLCV + uploaded CSV parser
ml/src/transformer.py            PyTorch Transformer model
ml/src/preprocessing.py          OHLCV features, chronological splits, scalers
ml/src/evaluate.py               Overall and per-horizon errors
ml/src/forecast.py               Holiday calendars and Transformer model artifacts
ml/run_comparison.py             CLI for naïve vs ML comparison
ml/run_training.py               Legacy Transformer training and checkpoint saving
ml/compare_models.ipynb          New notebook with both models
ml/notebook.ipynb                Existing repaired Transformer notebook
api/main.py                      Naïve + saved Transformer inference API
tests/                           Unit, integration, and API tests
Dockerfile.dashboard             Streamlit container
Dockerfile                       API container
docker-compose.yml               API + dashboard
requirements.txt                 Dashboard + ML + API packages
```

## Tests

```bash
python -m pip install -r requirements-dev.txt
python -m compileall -q app.py ml api tests
python -m pytest -q
```

## Forecasting caveats

- The **naïve model may outperform ML**, especially on price series whose next-session change is hard to forecast. This is a valid result, not a UI bug.
- For the ML models, training is on the oldest 70% of the history, validation on the next 10%, and test on the newest 20%; the dashboard does **not** retrain after testing using all recent observations. Thus its forward forecast uses a model that may be stale.
- Across multi-session horizons, predictions overlap in calendar time. Per-horizon metrics clarify performance but do **not** make all test windows statistically independent.
- Test metrics are **single chronological holdout** scores, not walk-forward backtests. For investment research, add multiple walk-forward test periods, transaction costs, leakage audits, calibration, and realistic baselines.
- Forecasts are **point estimates**. They are not prediction intervals or guarantees. Calendar schedules and symbol/currency selection should be checked independently.
- All market data is for educational use, and exchange data is obtained from Yahoo Finance; availability varies by symbol.
