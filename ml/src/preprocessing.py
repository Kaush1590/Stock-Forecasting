import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def add_features(df):
    out = df.copy()
    out["Return"] = np.log(out["Close"]).diff()
    out["HL_range"] = (out["High"] - out["Low"]) / out["Close"]
    out["OC_change"] = (out["Close"] - out["Open"]) / out["Open"]
    out["MA5_ratio"] = out["Close"].rolling(5).mean() / out["Close"] - 1
    out["MA20_ratio"] = out["Close"].rolling(20).mean() / out["Close"] - 1
    out["Volatility_10"] = out["Return"].rolling(10).std()
    out["Vol_change"] = np.log1p(out["Volume"]).diff()
    return out.replace([np.inf, -np.inf], np.nan).dropna()

def split(df: pd.DataFrame, train_size=0.7, val_size=0.1):
    n = len(df)
    train_len = int(n * train_size)
    val_len = int(n * (train_size + val_size))

    return df.iloc[:train_len], df.iloc[train_len: val_len], df.iloc[val_len:]

def fit_scalar(train_df):
    feature_names = list(train_df.columns)
    x_scalar = StandardScaler()
    y_scalar = MinMaxScaler()

    x_scalar.fit(train_df[feature_names])
    y_scalar.fit(train_df[["Close"]])
    return x_scalar, y_scalar, feature_names

def transform_input(df: pd.DataFrame, x_scalar, feature_names):
    return x_scalar.transform(df[feature_names]).astype(np.float32)

def make_sequences(features, close_values, lookback=60, horizon=7):
    X, y = [], []
    for end in range(lookback, len(features) - horizon + 1):
        X.append(features[end - lookback: end])
        y.append(close_values[end: end + horizon].reshape(-1))

    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)

def windows_for_targets(start, stop, X, y, lookback, horizon):
    Xs, ys = [], []
    for target_start in range(start, stop - horizon + 1):
        Xs.append(X[target_start-lookback:target_start])
        ys.append(y[target_start:target_start+horizon])
    return np.asarray(Xs, np.float32), np.asarray(ys, np.float32)

def preprocessing(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, lookback=60, horizon=7):
    x_scaler, y_scaler, feature_names = fit_scalar(train_df)
    full = pd.concat([train_df, val_df, test_df])

    X = transform_input(full, x_scaler, feature_names)
    y = y_scaler.transform(full[["Close"]]).astype(np.float32).ravel()

    a, b = len(train_df), len(train_df) + len(val_df)
    X_train, y_train = make_sequences(X[:a], y[:a], lookback, horizon)
    X_val, y_val = make_sequences(X[:b], y[:b], lookback, horizon)
    X_test, y_test = make_sequences(X, y, lookback, horizon)
    X_val, y_val = X_val[max(0, len(X_train)-lookback):], y_val[max(0, len(X_train)-lookback):]

    X_val, y_val = windows_for_targets(a, b, X, y, lookback, horizon)
    X_test, y_test = windows_for_targets(b, len(full), X, y, lookback, horizon)
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), x_scaler, y_scaler, feature_names