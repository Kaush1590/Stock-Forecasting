"""Compare persistence baseline with Random Forest or PyTorch Transformer.

Examples:
  python -m ml.run_comparison --demo --ml-model random_forest
  python -m ml.run_comparison --ticker AAPL --ml-model transformer --epochs 15
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from .src.comparison import run_comparison
from .src.data_fetch import get_data
from .src.sample_data import make_demo_data
from .src.forecast import artifact_slug


def main(argv=None):
    parser=argparse.ArgumentParser(description="Naïve vs ML stock forecasting benchmark")
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--period", default="5y")
    parser.add_argument("--exchange", default="NYSE")
    parser.add_argument("--ml-model", choices=["random_forest", "transformer"], default="random_forest")
    parser.add_argument("--lookback", type=int, default=30)
    parser.add_argument("--horizon", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--trees", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--demo", action="store_true", help="Use synthetic demo OHLCV data (offline)")
    args=parser.parse_args(argv)
    raw=make_demo_data() if args.demo else get_data(args.ticker, args.period)
    report=run_comparison(raw, ticker="DEMO" if args.demo else args.ticker,
                          exchange=args.exchange, lookback=args.lookback,
                          horizon=args.horizon, ml_model=args.ml_model,
                          epochs=args.epochs, n_estimators=args.trees, seed=args.seed,
                          calendar_mode="weekdays" if args.demo else "exchange")
    out=Path(args.output_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    slug=artifact_slug(report.ticker)
    report.forecast.to_csv(out / f"{slug}_{args.ml_model}_forecast.csv", index=False)
    report.backtest.to_csv(out / f"{slug}_{args.ml_model}_backtest.csv", index=False)
    (out / f"{slug}_{args.ml_model}_comparison.json").write_text(
         json.dumps({"data_source":"synthetic demo" if args.demo else "Yahoo Finance",
                     "ticker":report.ticker,"ml_model":args.ml_model,
                     "train_validation_test_rows":report.counts,
                     "metrics":report.metrics}, indent=2), encoding="utf-8")
    print(f"Saved prediction tables and metrics to {out.resolve()}")
    print(json.dumps({"naive_MAE":report.metrics["naive"]["MAE"],
                      "ml_MAE":report.metrics["ml"]["MAE"]}, indent=2))
    return report


if __name__ == "__main__":
    main()
