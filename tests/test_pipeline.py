import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.preprocessing import MinMaxScaler
from ml.src.preprocessing import add_features, split, preprocessing, make_sequences, windows_for_targets
from ml.src.transformer import StockTransformer, PositionalEncoding
from ml.src.evaluate import evaluate_predictions, naive_last_close_baseline, inverse_prices
from ml.src.forecast import artifact_slug, save_artifacts, load_artifacts
from ml.src.train import train_model


def data(rows=250):
    idx = pd.bdate_range("2023-01-02", periods=rows)
    close = 100 + np.arange(rows, dtype=float) * .3
    return pd.DataFrame({"Open": close-.2, "High": close+1, "Low": close-1,
                         "Close": close, "Volume": np.full(rows, 10000)}, index=idx)


def test_feature_engineering():
    d = add_features(data())
    assert len(d) == 231 and d.isna().sum().sum() == 0
    assert "MA20_ratio" in d.columns


def test_split_invalid():
    with pytest.raises(ValueError):
        split(data(), .9, .2)


def test_train_only_scaling_and_boundaries():
    feat = add_features(data())
    tr, va, te = split(feat)
    (tx, ty), (vx, vy), (sx, sy), xs, ys, names = preprocessing(tr, va, te, lookback=10, horizon=3)
    assert tx.shape[1:] == (10, len(names))
    assert ty.shape[1] == vy.shape[1] == sy.shape[1] == 3
    assert len(vx) == len(va)-2 and len(sx) == len(te)-2
    assert np.isclose(tx[0, -1, names.index("Close")], ys.transform(tr[["Close"]]).ravel()[9])
    # Validation labels start at the first validation timestamp; prior history is context only.
    assert np.isclose(vy[0, 0], ys.transform(va[["Close"]]).ravel()[0])
    assert np.isclose(sy[0, 0], ys.transform(te[["Close"]]).ravel()[0])


def test_too_short_sequences():
    with pytest.raises(ValueError, match="Insufficient"):
        make_sequences(np.zeros((5, 3)), np.zeros(5), 4, 3)
    with pytest.raises(ValueError):
        windows_for_targets(3, 5, np.zeros((5, 3)), np.zeros(5), 4, 2)


def test_model_shape_and_invalid_inputs():
    model = StockTransformer(5, lookback=10, horizon=3, target_idx=3,
                             d_model=8, nhead=2, num_layers=1, dim_feedforward=16,
                             dropout=0, input_dropout=0, noise_std=0)
    model.eval()
    assert model(torch.zeros((2, 10, 5))).shape == (2, 3)
    with pytest.raises(ValueError):
        model(torch.zeros((2, 9, 5)))
    with pytest.raises(ValueError):
        StockTransformer(5, d_model=7, nhead=2)


def test_evaluation_and_baseline():
    a = np.array([[10., 11.], [12., 13.]])
    metrics = evaluate_predictions(a, a+1)
    assert metrics["MAE"] == 1
    assert len(metrics["per_horizon"]) == 2
    scale = MinMaxScaler().fit(np.array([[10.], [20.]]))
    X = np.zeros((2, 3, 5), dtype=np.float32)
    X[:, -1, 3] = [.2, .8]
    assert np.allclose(naive_last_close_baseline(X, scale, 3), [12, 18])
    assert inverse_prices(np.array([[0., 1.]]), scale).shape == (1, 2)


def test_train_validates_empty():
    model = StockTransformer(5, lookback=10, horizon=3)
    with pytest.raises(ValueError):
        train_model(model, np.empty((0, 10, 5)), np.empty((0, 3)),
                    np.empty((0, 10, 5)), np.empty((0, 3)))


def test_artifact_roundtrip(tmp_path):
    feat = add_features(data())
    tr, va, te = split(feat)
    _, _, _, xs, ys, names = preprocessing(tr, va, te, 10, 3)
    config = dict(n_features=len(names), lookback=10, horizon=3, target_idx=names.index("Close"),
                  d_model=8, nhead=2, num_layers=1, dim_feedforward=16, dropout=0)
    model = StockTransformer(**config)
    save_artifacts(tmp_path, "TEST", model, xs, ys, names, config, {"seed": 42})
    restored, rx, ry, metadata = load_artifacts(tmp_path, "TEST")
    assert metadata["model_config"]["lookback"] == 10
    assert restored(torch.zeros((1, 10, len(names)))).shape == (1, 3)
    assert artifact_slug("^GSPC") == "GSPC"


def test_one_epoch_training_synthetic():
    model = StockTransformer(5, lookback=6, horizon=2, target_idx=3,
                             d_model=8, nhead=2, num_layers=1, dim_feedforward=16,
                             dropout=0, input_dropout=0, noise_std=0)
    X = np.random.default_rng(42).normal(size=(12, 6, 5)).astype(np.float32)
    y = np.random.default_rng(43).normal(size=(12, 2)).astype(np.float32)
    trained, history = train_model(model, X[:8], y[:8], X[8:], y[8:],
                                    epochs=1, batch_size=4, device="cpu")
    assert len(history["val_loss"]) == 1
    assert np.isfinite(history["val_loss"][0])
