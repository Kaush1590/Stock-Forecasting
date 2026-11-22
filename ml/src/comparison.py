"""Leakage-aware evaluation and forward forecasts: naïve vs ML.

ML choices: Random Forest (fast, CPU-friendly) or the original PyTorch Transformer.
Evaluation is on a held-out chronological test range, not on the latest dates
used to build the forward forecast. Overlapping forecast windows remain correlated.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor

from .naive import NaiveLastClose
from .preprocessing import add_features, split, preprocessing, transform_input
from .evaluate import inverse_prices, evaluate_predictions
from .forecast import future_sessions
from .train import train_model, set_seed
from .transformer import StockTransformer

MODEL_CHOICES = {"random_forest": "Random Forest", "transformer": "Transformer"}


@dataclass
class ComparisonResult:
    ticker: str
    model_key: str
    model_name: str
    metrics: dict
    forecast: pd.DataFrame
    backtest: pd.DataFrame
    historical: pd.DataFrame
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    counts: dict
    history: dict = field(default_factory=dict)


def run_comparison(raw_df: pd.DataFrame, *, ticker: str = "AAPL",
                   exchange: str = "NYSE", lookback: int = 30, horizon: int = 7,
                   ml_model: str = "random_forest", epochs: int = 12,
                   seed: int = 42, device: str | None = None,
                   batch_size: int = 64, learning_rate: float = 5e-4,
                   n_estimators: int = 100,
                   calendar_mode: str = "exchange") -> ComparisonResult:
    """Train selected ML model, test against naïve, and forecast future sessions.

    Each test target begins within the held-out test segment; history for its
    input may come from earlier splits. Scalers fit only the training segment.
    """
    if calendar_mode not in {"exchange", "weekdays"}:
        raise ValueError("calendar_mode must be exchange or weekdays")
    if ml_model not in MODEL_CHOICES:
        raise ValueError(f"ml_model must be one of {list(MODEL_CHOICES)}")
    if not (1 <= lookback <= 512 and 1 <= horizon <= 60):
        raise ValueError("lookback must be 1..512 and horizon 1..60")
    if n_estimators < 1 or epochs < 1:
        raise ValueError("n_estimators and epochs must be positive")
    set_seed(seed)
    data = add_features(raw_df)
    train, val, test = split(data, train_size=.70, val_size=.10)
    (Xtr, ytr), (Xva, yva), (Xte, yte), x_scaler, y_scaler, features = preprocessing(
        train, val, test, lookback, horizon)
    close_idx = features.index("Close")
    naive = NaiveLastClose(horizon, close_idx)
    naive_scaled = naive.predict(Xte)
    history = {}

    if ml_model == "random_forest":
        # RandomForestRegressor supports multioutput regression out of the box.
        ml = RandomForestRegressor(
            n_estimators=n_estimators, max_depth=10, min_samples_leaf=3,
            max_features=.8, random_state=seed, n_jobs=-1,
        )
        ml.fit(Xtr.reshape(len(Xtr), -1), ytr)
        ml_scaled = ml.predict(Xte.reshape(len(Xte), -1))
        latest = transform_input(data, x_scaler, features, y_scaler)[-lookback:]
        next_scaled = ml.predict(latest.reshape(1, -1))
    else:
        model_config = dict(n_features=len(features), lookback=lookback,
                            horizon=horizon, target_idx=close_idx, d_model=32,
                            nhead=4, num_layers=2, dim_feedforward=64,
                            dropout=.1, input_dropout=.1, noise_std=.02)
        ml = StockTransformer(**model_config)
        selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ml, history = train_model(
            ml, Xtr, ytr, Xva, yva, epochs=epochs, batch_size=batch_size,
            learning_rate=learning_rate, patience=min(5, epochs),
            device=selected_device, seed=seed)
        ml.eval()
        latest = transform_input(data, x_scaler, features, y_scaler)[-lookback:]
        with torch.no_grad():
            ml_scaled = ml(torch.as_tensor(Xte, dtype=torch.float32, device=selected_device)).cpu().numpy()
            next_scaled = ml(torch.as_tensor(latest[None], dtype=torch.float32, device=selected_device)).cpu().numpy()

    actual = inverse_prices(yte, y_scaler)
    naive_prices = inverse_prices(naive_scaled, y_scaler)
    ml_prices = inverse_prices(np.asarray(ml_scaled), y_scaler)
    metrics = {
        "naive": evaluate_predictions(actual, naive_prices),
        "ml": evaluate_predictions(actual, ml_prices),
    }

    dates = (future_sessions(data.index[-1], horizon, exchange)
             if calendar_mode == "exchange" else
             pd.bdate_range(pd.Timestamp(data.index[-1]) + pd.Timedelta(days=1), periods=horizon))
    forward_ml = inverse_prices(np.asarray(next_scaled).reshape(1, horizon), y_scaler)[0]
    forward_naive = naive.predict_prices(float(data["Close"].iloc[-1]))
    forecast = pd.DataFrame({
        "Forecast_Date": dates,
        "Naive_Close": forward_naive,
        "ML_Close": forward_ml,
    })

    # Each row represents the one-day-ahead prediction at a unique test origin;
    # unlike flattening all horizons, there are no duplicate dates here.
    first_test = len(train) + len(val)
    target_dates = data.index[first_test:len(data) - horizon + 1]
    if len(target_dates) != len(actual):
        raise RuntimeError("Backtest target date alignment failed")
    backtest = pd.DataFrame({
        "Date": target_dates,
        "Actual_Close": actual[:, 0],
        "Naive_Close": naive_prices[:, 0],
        "ML_Close": ml_prices[:, 0],
    })
    return ComparisonResult(
        ticker=ticker.upper(), model_key=ml_model, model_name=MODEL_CHOICES[ml_model],
        metrics=metrics, forecast=forecast, backtest=backtest,
        historical=data[["Open", "High", "Low", "Close", "Volume"]].copy(),
        test_start=pd.Timestamp(test.index[0]), test_end=pd.Timestamp(test.index[-1]),
        counts={"train": len(train), "validation": len(val), "test": len(test),
                "test_windows": len(Xte)}, history=history,
    )
