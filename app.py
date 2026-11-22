"""Streamlit UI for naïve vs machine-learning stock-price forecasting.

Run from project root: python -m streamlit run app.py
"""
from __future__ import annotations
import html
import json
import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from ml.src.comparison import MODEL_CHOICES, run_comparison
from ml.src.data_fetch import get_data
from ml.src.sample_data import make_demo_data, parse_upload

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
st.set_page_config(page_title="StockLab | Forecast Studio", page_icon="📈",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
  .stApp {background: #0B1220; color: #F1F5F9;}
  section[data-testid="stSidebar"] {background:#0F1B2D; border-right:1px solid #25344A;}
  .block-container {padding-top:1.7rem; max-width:1400px;}
  h1,h2,h3 {letter-spacing:-.035em;}
  div[data-testid="stMetric"] {background:#111F33; padding:14px 18px;
      border:1px solid #293D53; border-radius:15px;}
  div[data-testid="stMetricValue"] {font-size:1.45rem;}
  .hero {border:1px solid #294460; border-radius:22px; padding:28px 32px;
       background:linear-gradient(115deg,#18304F 0%,#112034 52%,#102B2B 100%);
       margin-bottom:1.2rem;}
  .hero h1 {margin:0; color:#F7FCFF; font-size:2.35rem;}
  .hero p {color:#A8BED4; margin:.55rem 0 0 0;}
  .eyebrow {color:#63E6BE; letter-spacing:.19em; font-weight:750;
            font-size:.74rem; margin-bottom:.45rem;}
  .muted {color:#9CAFC3; font-size:.9rem;}
  div.stButton>button[kind="primary"], div.stFormSubmitButton>button[kind="primary"]
       {border:0; background:linear-gradient(120deg,#17A78F,#268AC4); color:white;}
  .note {background:#101D30; border:1px solid #2B3F53; border-radius:12px;
         padding:13px 16px; color:#AFC3D5; margin:.8rem 0;}
</style>
""", unsafe_allow_html=True)

st.markdown("""<div class="hero">
<div class="eyebrow">STOCKLAB / FORECAST STUDIO</div>
<h1 align="center">Stock Forecasting Dashboard</h1>
</div>""", unsafe_allow_html=True)

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_prices(ticker: str, period: str):
    return get_data(ticker, period, "1d")


with st.sidebar:
    st.markdown("### ⚙️ Experiment settings")
    st.caption("All prices use the market data's original currency.")
    data_source = st.selectbox("Data source", ["Yahoo Finance", "Demo (offline)", "Upload CSV"],
                               index=0)
    model_name = st.selectbox("ML model", list(MODEL_CHOICES.values()), index=0,
                              help="Random Forest trains quickly on CPU; Transformer uses PyTorch.")
    model_key = next(key for key, name in MODEL_CHOICES.items() if name == model_name)
    with st.form("forecast_settings"):
        ticker = st.text_input("Stock symbol", value=os.getenv("STOCK", "AAPL"),
                               help="Examples: AAPL, MSFT, RELIANCE.NS").strip().upper()
        period = st.selectbox("Price history", ["2y", "5y", "10y", "max"], index=1)
        upload = st.file_uploader("OHLCV file", type=["csv"],
                                  help="CSV columns: Date, Open, High, Low, Close, Volume") if data_source == "Upload CSV" else None
        exchange = st.selectbox("Trading calendar", ["NYSE", "NASDAQ", "NSE", "LSE"], index=0,
                                help="Select the exchange for future trading sessions.")
        lookback = st.slider("Historical lookback (sessions)", min_value=10, max_value=90, value=30, step=5)
        horizon = st.slider("Forecast horizon (sessions)", min_value=1, max_value=14, value=7)
        if model_key == "transformer":
            epochs = st.slider("Maximum training epochs", min_value=3, max_value=60, value=12, step=3)
            n_estimators = 100
        else:
            n_estimators = st.slider("Random Forest trees", min_value=25, max_value=200, value=100, step=25)
            epochs = 12
        run_pressed = st.form_submit_button("Run forecast comparison", type="primary", use_container_width=True)
    st.markdown("""<div class="note">💡 <b>Naïve</b> repeats the last observed closing price.
    <b>ML</b> learns from historical features and predicts several sessions at once.</div>""", unsafe_allow_html=True)

if run_pressed:
    if not ticker:
        st.error("Enter a stock symbol before running.")
    elif data_source == "Upload CSV" and upload is None:
        st.error("Upload a CSV file containing Date, Open, High, Low, Close and Volume.")
    else:
        try:
            with st.spinner("Loading data and evaluating forecasting models…"):
                if data_source == "Demo (offline)":
                    raw = make_demo_data()
                elif data_source == "Upload CSV":
                    raw = parse_upload(upload.getvalue())
                else:
                    raw = fetch_prices(ticker, period)
                result = run_comparison(raw, ticker=ticker, exchange=exchange,
                    lookback=lookback, horizon=horizon, ml_model=model_key,
                    epochs=epochs, n_estimators=n_estimators,
                    calendar_mode="weekdays" if data_source == "Demo (offline)" else "exchange")
                st.session_state["comparison"] = result
                st.session_state["data_source_label"] = data_source
        except (ValueError, RuntimeError, KeyError, ImportError, OSError) as exc:
            st.error(f"Couldn't complete the forecast: {exc}")
            st.info("For an offline demo choose **Demo (offline)**. For live symbols check your internet connection, the ticker, and installed dependencies.")

result = st.session_state.get("comparison")
if result is None:
    st.info("Choose a ticker and model on the left, then click **Run forecast comparison**. For a no-internet trial, select **Demo (offline)**.")
    st.markdown("#### What you will see")
    c1,c2,c3=st.columns(3)
    c1.markdown("**1 · Forecast**  \nNext trading-session price paths for both models")
    c2.markdown("**2 · Backtest**  \nOut-of-sample test accuracy, side by side")
    c3.markdown("**3 · Export**  \nDownload the prediction tables and metrics")
    st.stop()

source = st.session_state.get("data_source_label", "Unknown")
if source == "Demo (offline)":
    st.warning("**Demo mode:** These OHLCV prices are synthetic. They do not represent a real security or actual market performance.")
else:
    st.caption(f"Data source: {source} · Exchange calendar selected at execution time")

last_close = float(result.historical["Close"].iloc[-1])
naive_mae = float(result.metrics["naive"]["MAE"])
ml_mae = float(result.metrics["ml"]["MAE"])
relative_improvement = 100 * (naive_mae - ml_mae) / naive_mae if naive_mae > 0 else float("nan")

st.markdown(f"### {html.escape(result.ticker)} · {html.escape(result.model_name)} vs naïve")
m1,m2,m3,m4=st.columns(4)
m1.metric("Latest observed close", f"{last_close:,.2f}")
m2.metric("Naïve test MAE", f"{naive_mae:,.2f}")
m3.metric(f"{result.model_name} test MAE", f"{ml_mae:,.2f}")
m4.metric("ML MAE improvement", f"{relative_improvement:+.1f}%" if pd.notna(relative_improvement) else "N/A",
          help="Positive means lower average test error than naïve. Negative means the naïve model performed better.")

forecast_tab, backtest_tab, diagnostics_tab, data_tab = st.tabs(
    ["📈 Future forecast", "🎯 Backtest", "🔬 Diagnostics", "⬇️ Data & exports"])

PLOT = dict(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#CBD5E1"),
            margin=dict(l=8, r=8, t=35, b=5), hovermode="x unified")

with forecast_tab:
    st.markdown("#### Forecast for upcoming exchange sessions")
    fig=go.Figure()
    hist = result.historical.tail(105)
    fc = result.forecast
    fig.add_trace(go.Scatter(x=hist.index, y=hist["Close"], name="Observed close",
                             line=dict(color="#A9BAC9", width=2.4)))
    # Link the two forecast trajectories visually to the last observed close.
    joined_dates=[hist.index[-1], *fc["Forecast_Date"].tolist()]
    fig.add_trace(go.Scatter(x=joined_dates, y=[last_close, *fc["Naive_Close"].tolist()],
                             name="Naïve baseline", line=dict(color="#F5BC68", width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=joined_dates, y=[last_close, *fc["ML_Close"].tolist()],
                             name=result.model_name, line=dict(color="#53D9B8", width=3)))
    fig.add_vline(x=hist.index[-1].timestamp()*1000, line_dash="dot", line_color="#668198")
    fig.update_layout(**PLOT, height=430, yaxis_title="Price (source currency)",
                      xaxis_title="Trading date", legend=dict(orientation="h", y=1.11, x=0))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(("Demo uses weekdays (not an exchange holiday calendar). " if source == "Demo (offline)" else
               "Future dates follow the selected exchange's session calendar. ") +
               "Predictions are point estimates, not confidence intervals.")
    table = fc.rename(columns={"Forecast_Date":"Trading date",
                               "Naive_Close":"Naïve forecast", "ML_Close":f"{result.model_name} forecast"})
    st.dataframe(table.style.format({"Naïve forecast":"{:,.2f}", f"{result.model_name} forecast":"{:,.2f}"}),
                 hide_index=True, use_container_width=True)

with backtest_tab:
    st.markdown("#### Day-ahead predictions on the held-out test set")
    bt = result.backtest
    fig=go.Figure()
    for col, title, color, dash in [
        ("Actual_Close", "Actual", "#B9CCD8", "solid"),
        ("Naive_Close", "Naïve", "#F5BC68", "dash"),
        ("ML_Close", result.model_name, "#53D9B8", "solid")]:
        fig.add_trace(go.Scatter(x=bt["Date"], y=bt[col], name=title,
                                 line=dict(color=color, width=2, dash=dash)))
    fig.update_layout(**PLOT, height=410, xaxis_title="Test target date",
                      yaxis_title="Price (source currency)", legend=dict(orientation="h", y=1.11))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Test dates: {result.test_start.date()} through {result.test_end.date()}. "
               "Only horizon +1 is graphed here; all horizon errors appear below.")
    rows=[]
    for name,key in [("Naïve", "naive"), (result.model_name, "ml")]:
        summary=result.metrics[key]
        rows.append({"Model":name, "MAE":summary["MAE"], "RMSE":summary["RMSE"],
                     "MAPE (%)":summary["MAPE_percent"]})
    st.dataframe(pd.DataFrame(rows).style.format({"MAE":"{:.3f}","RMSE":"{:.3f}","MAPE (%)":"{:.2f}"}),
                 hide_index=True, use_container_width=True)

with diagnostics_tab:
    st.markdown("#### Error by forecast horizon")
    records=[]
    for key,name in [("naive","Naïve"),("ml",result.model_name)]:
        for step,values in result.metrics[key]["per_horizon"].items():
            records.append({"Horizon":int(step.split("_")[1]), "Model":name,
                            "MAE":values["MAE"], "RMSE":values["RMSE"]})
    diagnostics=pd.DataFrame(records)
    fig=go.Figure()
    for name,color in [("Naïve","#F5BC68"),(result.model_name,"#53D9B8")]:
        records_for_model=diagnostics[diagnostics["Model"]==name]
        fig.add_trace(go.Scatter(x=records_for_model["Horizon"], y=records_for_model["MAE"],
                                 mode="lines+markers", name=name, line=dict(color=color, width=3)))
    fig.update_layout(**PLOT, height=330, xaxis_title="Trading sessions ahead", yaxis_title="MAE")
    st.plotly_chart(fig, use_container_width=True)
    c1,c2,c3=st.columns(3)
    c1.metric("Training rows", result.counts["train"])
    c2.metric("Validation rows", result.counts["validation"])
    c3.metric("Test windows", result.counts["test_windows"])
    if result.history.get("train_loss"):
        st.markdown("#### Transformer learning curve · Huber loss")
        learning=pd.DataFrame(result.history)
        learning.index=learning.index+1
        st.line_chart(learning, x_label="Epoch", y_label="Huber loss")
    st.info("The ML model is fitted on the earliest 70% of data (the next 10% is reserved for validation). "
            "The latest 20% is held out for testing. These forecasts are not walk-forward-retrained. "
            "Overlapping multi-horizon windows are correlated, so aggregate errors are not independent observations.")

with data_tab:
    st.markdown("#### Download your experiment")
    c1,c2,c3=st.columns(3)
    slug = "".join(c if c.isalnum() else "_" for c in result.ticker)
    c1.download_button("Download future forecast CSV", result.forecast.to_csv(index=False).encode(),
                       file_name=f"{slug}_future_forecast.csv", mime="text/csv", use_container_width=True)
    c2.download_button("Download backtest CSV", result.backtest.to_csv(index=False).encode(),
                       file_name=f"{slug}_backtest.csv", mime="text/csv", use_container_width=True)
    c3.download_button("Download metrics JSON", json.dumps(result.metrics, indent=2).encode(),
                       file_name=f"{slug}_metrics.json", mime="application/json", use_container_width=True)
    st.markdown("#### Observed OHLCV data")
    st.dataframe(result.historical.tail(150).sort_index(ascending=False), use_container_width=True)
    st.download_button("Download OHLCV CSV", result.historical.to_csv(index_label="Date").encode(),
                       file_name=f"{slug}_historical.csv", mime="text/csv")
