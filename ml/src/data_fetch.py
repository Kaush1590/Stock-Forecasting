"""Market data acquisition and validation."""
from __future__ import annotations
import pandas as pd

PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def get_data(ticker: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    if not isinstance(ticker, str) or not ticker.strip():
        raise ValueError("ticker must be a nonempty string")
    if not isinstance(period, str) or not period.strip():
        raise ValueError("period must be a nonempty string")
    if not isinstance(interval, str) or not interval.strip():
        raise ValueError("interval must be a nonempty string")
    try:
        import yfinance as yf
        df = yf.download(tickers=ticker.strip(), period=period, interval=interval,
                         auto_adjust=True, progress=False, multi_level_index=False)
    except Exception as exc:
        raise RuntimeError(f"Market data download failed for {ticker!r}") from exc
    if df is None or df.empty:
        raise ValueError(f"No market data returned for ticker={ticker!r}, period={period!r}, interval={interval!r}")
    missing = set(PRICE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Market data missing required columns: {sorted(missing)}")
    df = df[PRICE_COLUMNS].copy().sort_index()
    df = df[~df.index.duplicated(keep="last")]
    if df.isna().any().any():
        raise ValueError("Downloaded market data contains missing OHLCV observations")
    if (df[["Open", "High", "Low", "Close"]] <= 0).any().any() or (df["Volume"] < 0).any():
        raise ValueError("Downloaded market data contains invalid prices or volume")
    return df
