"""Read-only inference API. Start: uvicorn api.main:app --host 0.0.0.0 --port 8000."""
from __future__ import annotations
import os
import re
from pathlib import Path
from fastapi import FastAPI, HTTPException, Query
from ml.src.data_fetch import get_data
from ml.src.preprocessing import add_features
from ml.src.forecast import load_artifacts, forecast_next_sessions, future_sessions
from ml.src.naive import NaiveLastClose

app = FastAPI(title="Stock Forecasting API", version="2.0.0")
ARTIFACT_DIR = Path(os.environ.get("ARTIFACT_DIR", "outputs"))
TICKER_PATTERN = r"^[A-Za-z0-9.^_=-]{1,25}$"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/forecast/{ticker}")
def forecast(ticker: str, period: str = Query(default="5y", min_length=1, max_length=16),
             interval: str = Query(default="1d", pattern=r"^1d$")):
    if not re.fullmatch(TICKER_PATTERN, ticker):
        raise HTTPException(422, "Invalid ticker format")
    try:
        model, xs, ys, config = load_artifacts(ARTIFACT_DIR, ticker)
    except FileNotFoundError:
        raise HTTPException(404, "No trained model available for this ticker")
    except (ValueError, KeyError) as exc:
        raise HTTPException(500, f"Model artifacts are incompatible: {exc}")
    try:
        raw = get_data(ticker, period, interval)
        features = add_features(raw)
        params = config["model_config"]
        result = forecast_next_sessions(model, features, xs, ys,
              config["feature_names"], lookback=params["lookback"],
              horizon=params["horizon"], exchange=config["exchange"])
    except (ValueError, RuntimeError, KeyError) as exc:
        raise HTTPException(422, str(exc))
    return {"ticker": ticker, "exchange": config["exchange"],
            "forecast": [{"date": row.Forecast_Date.date().isoformat(),
                          "predicted_close": float(row.Predicted_Close)}
                         for row in result.itertuples(index=False)],
            "disclaimer": "Experimental model; not investment advice. Predictions may be inaccurate."}


@app.get("/naive/{ticker}")
def naive_forecast(ticker: str, period: str = Query(default="5y", min_length=1, max_length=16),
                   exchange: str = Query(default="NYSE", min_length=1, max_length=12),
                   horizon: int = Query(default=7, ge=1, le=60)):
    """Fit-free persistence forecast: every session equals the last close."""
    if not re.fullmatch(TICKER_PATTERN, ticker):
        raise HTTPException(422, "Invalid ticker format")
    try:
        raw = get_data(ticker, period, "1d")
        close = float(raw["Close"].iloc[-1])
        dates = future_sessions(raw.index[-1], horizon, exchange)
        values = NaiveLastClose(horizon, 0).predict_prices(close)
    except (ValueError, RuntimeError, KeyError) as exc:
        raise HTTPException(422, str(exc))
    return {"ticker": ticker, "model": "naive_last_close", "exchange": exchange,
            "last_observed_close": close,
            "forecast": [{"date": date.date().isoformat(), "predicted_close": float(value)}
                         for date, value in zip(dates, values)],
            "disclaimer": "Experimental baseline; not investment advice."}
