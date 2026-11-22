from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_missing_artifact():
    result = client.get("/forecast/NOT_TRAINED")
    assert result.status_code == 404


def test_invalid_ticker():
    result = client.get("/forecast/invalid!ticker")
    assert result.status_code == 422


def test_naive_rejects_invalid_ticker():
    assert client.get("/naive/bad!ticker").status_code == 422

def test_naive_endpoint_without_external_network(monkeypatch):
    import pandas as pd
    import api.main as api_main
    import numpy as np
    rows = 100
    dates = pd.bdate_range("2025-01-01", periods=rows)
    data = pd.DataFrame({"Close": np.arange(rows, dtype=float) + 101}, index=dates)
    monkeypatch.setattr(api_main, "get_data", lambda ticker, period, interval: data)
    monkeypatch.setattr(api_main, "future_sessions",
                        lambda last, horizon, exchange: pd.bdate_range(
                            pd.Timestamp(last) + pd.Timedelta(days=1), periods=horizon))
    result = client.get("/naive/TEST?horizon=3")
    assert result.status_code == 200
    body=result.json()
    assert len(body["forecast"]) == 3
    assert [e["predicted_close"] for e in body["forecast"]] == [200.0]*3
