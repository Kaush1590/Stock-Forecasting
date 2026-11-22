"""Offline tests for baseline, ML model comparison and CSV ingestion."""
import numpy as np
import pandas as pd
import pytest

from ml.src.naive import NaiveLastClose
from ml.src.sample_data import make_demo_data, parse_upload
from ml.src.comparison import run_comparison


def test_naive_predicts_last_close_across_all_horizons():
    x=np.array([[[2, 5],[3, 8]], [[2, 9],[4, 6]]], dtype=np.float32)
    model=NaiveLastClose(horizon=3, target_idx=1)
    assert model.predict(x).tolist()==[[8,8,8],[6,6,6]]
    assert model.predict_prices(42.5).tolist()==[42.5]*3


def test_naive_rejects_invalid_input():
    with pytest.raises(ValueError):
        NaiveLastClose(0,0)
    with pytest.raises(ValueError):
        NaiveLastClose(3,1).predict(np.array([1,2,3]))
    with pytest.raises(ValueError):
        NaiveLastClose(3,0).predict_prices(float('nan'))


def test_synthetic_data_valid_ohlcv():
    df=make_demo_data(rows=260)
    assert len(df)==260
    assert (df.Low<=df.Close).all()
    assert (df.High>=df.Close).all()
    assert df.index.is_monotonic_increasing


def test_parse_csv_case_insensitive_columns_and_dates():
    csv=b"date,OPEN,high,low,close,volume\n2025-02-01,10,11,9,10.5,100\n2025-02-02,12,13,11,12.5,200\n"
    df=parse_upload(csv)
    assert list(df.columns)==["Open","High","Low","Close","Volume"]
    assert df.Close.iloc[0]==10.5
    assert isinstance(df.index, pd.DatetimeIndex)


def test_parse_csv_rejects_missing_columns():
    with pytest.raises(ValueError, match="missing columns"):
        parse_upload(b"Date,Close\n2025-01-01,100\n")


def test_rf_comparison_end_to_end_offline(monkeypatch):
    import ml.src.comparison as comp
    monkeypatch.setattr(comp, "future_sessions", lambda last,h,exchange: pd.bdate_range(
        pd.Timestamp(last)+pd.Timedelta(days=1), periods=h))
    df=make_demo_data(rows=280, seed=123)
    out=run_comparison(df, ticker="DEMO", exchange="NYSE", lookback=12,
                       horizon=3, ml_model="random_forest", n_estimators=10)
    assert len(out.forecast)==3
    assert len(out.backtest)==out.counts["test_windows"]
    assert len(out.metrics["ml"]["per_horizon"])==3
    assert out.metrics["naive"]["MAE"]>=0
    assert out.metrics["ml"]["MAE"]>=0
    assert np.all(np.isfinite(out.forecast["ML_Close"]))
    assert np.allclose(out.forecast["Naive_Close"], df.Close.iloc[-1])
    # The day-1 plotted backtest error is the day_1 metric.
    err=np.abs(out.backtest.Actual_Close - out.backtest.ML_Close).mean()
    assert err == pytest.approx(out.metrics["ml"]["per_horizon"]["day_1"]["MAE"])
    assert pd.Timestamp(out.backtest.Date.iloc[0]) >= out.test_start


def test_invalid_model_choice():
    with pytest.raises(ValueError, match="ml_model"):
        run_comparison(make_demo_data(200), ml_model="unknown")
