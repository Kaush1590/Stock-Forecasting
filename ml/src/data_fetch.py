import os
from dotenv import load_dotenv
import yfinance as yf
import pandas as pd

load_dotenv()

def get_data(ticker, period, interval):
    df = yf.download(
        tickers=ticker,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
        multi_level_index=False
    )

    if df.empty:
        raise ValueError(f"Failed to fetch data with variables {ticker} | {period} | {interval}. Please set a proper variable and try again")

    cols = ["Open", "High", "Low", "Close", "Volume"]
    df = df[cols].copy()
    return df