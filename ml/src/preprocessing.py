"""Leakage-aware chronological preprocessing for direct multi-step forecasts."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    needed = {"Open", "High", "Low", "Close", "Volume"}
    if not needed.issubset(df.columns):
        raise ValueError(f"Missing OHLCV columns: {sorted(needed-set(df.columns))}")
    if (df[["Open", "High", "Low", "Close"]] <= 0).any().any() or (df["Volume"] < 0).any():
        raise ValueError("OHLC prices must be positive and volume nonnegative")
    out = df.copy().sort_index()
    out["Return"] = np.log(out["Close"]).diff()
    out["HL_range"] = (out["High"] - out["Low"]) / out["Close"]
    out["OC_change"] = (out["Close"] - out["Open"]) / out["Open"]
    out["MA5_ratio"] = out["Close"].rolling(5).mean() / out["Close"] - 1
    out["MA20_ratio"] = out["Close"].rolling(20).mean() / out["Close"] - 1
    out["Volatility_10"] = out["Return"].rolling(10).std()
    out["Vol_change"] = np.log1p(out["Volume"]).diff()
    return out.replace([np.inf, -np.inf], np.nan).dropna()


def split(df: pd.DataFrame, train_size: float = .7, val_size: float = .1):
    if not (0 < train_size < 1 and 0 < val_size < 1 and train_size + val_size < 1):
        raise ValueError("Expected 0 < train_size, val_size and train_size + val_size < 1")
    n = len(df)
    a, b = int(n * train_size), int(n * (train_size + val_size))
    if a == 0 or b == a or b == n:
        raise ValueError("Too few rows to form nonempty train, validation and test splits")
    return df.iloc[:a].copy(), df.iloc[a:b].copy(), df.iloc[b:].copy()


def fit_scaler(train_df: pd.DataFrame):
    if train_df.empty or "Close" not in train_df:
        raise ValueError("Training data must contain Close prices")
    names = list(train_df.columns)
    # Standardize non-target predictors, but use the SAME target scaler for the
    # Close input feature and for labels. This gives the linear skip a meaningful baseline.
    x_scaler = StandardScaler().fit(train_df[names])
    y_scaler = MinMaxScaler().fit(train_df[["Close"]])
    return x_scaler, y_scaler, names

# Backward compatible alias for the original project's misspelling.
fit_scalar = fit_scaler


def transform_input(df: pd.DataFrame, x_scaler, feature_names, y_scaler=None):
    missing = set(feature_names) - set(df.columns)
    if missing:
        raise ValueError(f"Input missing features: {sorted(missing)}")
    X = x_scaler.transform(df[feature_names]).astype(np.float32)
    if y_scaler is not None and "Close" in feature_names:
        X[:, feature_names.index("Close")] = y_scaler.transform(df[["Close"]]).ravel()
    return X


def make_sequences(features, close_values, lookback: int = 60, horizon: int = 7):
    features, close_values = np.asarray(features), np.asarray(close_values)
    if lookback < 1 or horizon < 1:
        raise ValueError("lookback and horizon must be positive")
    if len(features) != len(close_values):
        raise ValueError("Features and targets length mismatch")
    if len(features) < lookback + horizon:
        raise ValueError("Insufficient rows: need at least lookback + horizon")
    return windows_for_targets(lookback, len(features), features, close_values, lookback, horizon)


def windows_for_targets(start, stop, X, y, lookback, horizon):
    if lookback < 1 or horizon < 1 or start < lookback or stop > len(X) or len(X) != len(y):
        raise ValueError("Invalid window boundaries or input lengths")
    if stop - start < horizon:
        raise ValueError("Target segment needs at least horizon rows")
    xs, ys = [], []
    for target_start in range(start, stop - horizon + 1):
        xs.append(X[target_start-lookback:target_start])
        ys.append(y[target_start:target_start+horizon])
    return np.asarray(xs, np.float32), np.asarray(ys, np.float32)


def preprocessing(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                  lookback: int = 60, horizon: int = 7):
    if lookback < 1 or horizon < 1:
        raise ValueError("lookback and horizon must be positive")
    if len(train_df) < lookback + horizon or len(val_df) < horizon or len(test_df) < horizon:
        raise ValueError("Need train >= lookback+horizon, val >= horizon, test >= horizon")
    if list(train_df.columns) != list(val_df.columns) or list(train_df.columns) != list(test_df.columns):
        raise ValueError("Split feature columns must match")
    if not (train_df.index.max() < val_df.index.min() < test_df.index.min()):
        raise ValueError("Splits must be in strictly increasing chronological order")
    x_scaler, y_scaler, names = fit_scaler(train_df)
    full = pd.concat([train_df, val_df, test_df])
    X = transform_input(full, x_scaler, names, y_scaler)
    y = y_scaler.transform(full[["Close"]]).astype(np.float32).ravel()
    a, b = len(train_df), len(train_df) + len(val_df)
    # Each set owns only targets in its date range; earlier features are
    # available as historical context, never later information.
    train = windows_for_targets(lookback, a, X, y, lookback, horizon)
    val = windows_for_targets(a, b, X, y, lookback, horizon)
    test = windows_for_targets(b, len(full), X, y, lookback, horizon)
    return train, val, test, x_scaler, y_scaler, names
