"""CLI training pipeline: python -m ml.run_training --ticker AAPL."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .src.data_fetch import get_data
from .src.preprocessing import add_features, split, preprocessing
from .src.transformer import StockTransformer
from .src.train import train_model, set_seed
from .src.evaluate import evaluate_model
from .src.forecast import forecast_next_sessions, save_artifacts


def run(ticker, period="5y", interval="1d", exchange="NYSE", lookback=30,
        horizon=7, epochs=30, batch_size=64, learning_rate=5e-4, seed=42,
        output_dir="outputs", device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)
    raw = get_data(ticker, period, interval)
    feature_df = add_features(raw)
    train, val, test = split(feature_df)
    (X_train, y_train), (X_val, y_val), (X_test, y_test), xs, ys, features = preprocessing(
        train, val, test, lookback, horizon)
    model_config = {"n_features": len(features), "lookback": lookback,
                    "horizon": horizon, "target_idx": features.index("Close"),
                    "d_model": 64, "nhead": 4, "num_layers": 3,
                    "dim_feedforward": 128, "dropout": .1}
    model = StockTransformer(**model_config)
    model, history = train_model(model, X_train, y_train, X_val, y_val,
                                epochs=epochs, batch_size=batch_size,
                                learning_rate=learning_rate, device=device, seed=seed,
                                patience=5)
    metrics, actual, predicted = evaluate_model(model, X_test, y_test, ys,
                                                 device=device, target_idx=features.index("Close"))
    forecast = forecast_next_sessions(model, feature_df, xs, ys, features,
                                     lookback, horizon, device, exchange)
    out = Path(output_dir)
    save_artifacts(out, ticker, model, xs, ys, features, model_config,
                   {"ticker": ticker, "period": period, "interval": interval,
                    "epochs": epochs, "batch_size": batch_size,
                    "learning_rate": learning_rate, "seed": seed}, exchange)
    from .src.forecast import artifact_slug
    slug = artifact_slug(ticker)
    forecast.to_csv(out / f"{slug}_forecast.csv", index=False)
    raw.to_csv(out / f"{slug}_historical.csv")
    (out / f"{slug}_metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / f"{slug}_history.json").write_text(json.dumps(history, indent=2))
    print(json.dumps(metrics, indent=2))
    return metrics, forecast


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train stock price forecasting model")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--period", default="5y")
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--exchange", default="NYSE", help="pandas_market_calendars market name")
    parser.add_argument("--lookback", type=int, default=30)
    parser.add_argument("--horizon", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()
    run(args.ticker, args.period, args.interval, args.exchange, args.lookback,
        args.horizon, args.epochs, args.batch_size, args.learning_rate,
        args.seed, args.output_dir)
