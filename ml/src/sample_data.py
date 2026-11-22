"""Deterministic artificial OHLCV data for offline demonstrations and tests."""
from __future__ import annotations
import numpy as np
import pandas as pd


def make_demo_data(rows: int = 900, seed: int = 23) -> pd.DataFrame:
    """Synthetic prices — NOT live/historical financial data."""
    if rows < 80:
        raise ValueError("Demo requires at least 80 observations")
    rng = np.random.default_rng(seed)
    index = pd.bdate_range(end=pd.Timestamp.now().normalize(), periods=rows)
    shocks = rng.normal(0.00028, 0.012, size=rows)
    cycle = 0.002 * np.sin(np.arange(rows) / 24)
    close = 105 * np.exp(np.cumsum(shocks + cycle))
    open_price = close * np.exp(rng.normal(0, .004, rows))
    high = np.maximum(close, open_price) * (1 + rng.uniform(.002, .015, rows))
    low = np.minimum(close, open_price) * (1 - rng.uniform(.002, .015, rows))
    volume = rng.integers(500_000, 6_000_000, size=rows)
    return pd.DataFrame({"Open": open_price, "High": high, "Low": low,
                         "Close": close, "Volume": volume}, index=index)


def parse_upload(uploaded_bytes: bytes) -> pd.DataFrame:
    """Parse an OHLCV CSV with a Date/Datetime/Timestamp column."""
    from io import BytesIO
    df = pd.read_csv(BytesIO(uploaded_bytes), encoding="utf-8-sig")
    df.columns = [str(c).strip() for c in df.columns]
    normalized = {c.lower(): c for c in df.columns}
    date_col = next((normalized[k] for k in ("date", "datetime", "timestamp") if k in normalized), None)
    if date_col is None:
        raise ValueError("CSV must include a Date, Datetime or Timestamp column")
    names = ["Open", "High", "Low", "Close", "Volume"]
    missing = [n for n in names if n.lower() not in normalized]
    if missing:
        raise ValueError(f"CSV missing columns: {', '.join(missing)}")
    date = pd.to_datetime(df[date_col], errors="coerce", utc=True)
    if date.isna().any():
        raise ValueError("CSV has invalid dates")
    # Strip timezone only after conversion, keeping all dates in one consistent zone.
    date = date.dt.tz_convert(None)
    out = pd.DataFrame({name: pd.to_numeric(df[normalized[name.lower()]], errors="coerce")
                        for name in names})
    out.index = pd.DatetimeIndex(date, name="Date")
    if out.isna().any().any():
        raise ValueError("CSV has missing or non-numeric OHLCV values")
    if out.index.duplicated().any():
        raise ValueError("CSV contains duplicate timestamps")
    return out.sort_index()
