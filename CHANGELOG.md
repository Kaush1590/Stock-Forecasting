# Changes compared with Kaush1590/Stock-Forecasting

This is an independently assembled, refactored source distribution based on the
public code reviewed on 2026-11-22. Git clone/download of the original repo was
not available in the build environment, so original notebook outputs and git
commit history are not preserved. The notebook was rewritten to invoke the
new supported training pipeline. Original project attribution and MIT license
are retained.

## Implementation
- Revised data ingestion and error checks.
- Kept technical features, added rigorous split/window validation.
- Matched Close input scale and target price scale.
- Kept Transformer architecture; added input shape validation.
- Added seeding, early stopping, and hyperparameter validation.
- Added per-horizon error metrics and constant-last-close benchmark.
- Added exchange-aware business session dates (calendar package required).
- Saved model/configuration/scalers with typed local artifact loading.
- Added training CLI, FastAPI read-only forecast endpoint and health check.
- Added unit tests, CI, Docker support, documentation and example env config.

## Not included / needs external validation
- Live Yahoo market-data fetch and a full training run on real prices were not
  tested offline; retraining is mandatory before inference.
- No production authentication, public API rate-limiting, distributed caching,
  frontend, trained model weights, or deployed service is bundled.
- No statistical interval calibration or multi-fold walk-forward backtesting.

## Notebook import fix

- The notebook now discovers its repository root when Jupyter uses `ml/` as its working directory.
- `.env` and output artifact paths are resolved relative to the repository root.
- Simplified the saved training history path to use the imported `artifact_slug` function.

## Dashboard and two-model comparison (2026-11-22)

- Added an explicit `NaiveLastClose` predictor and joint holdout benchmark with **Random Forest** or **Transformer**.
- Added Streamlit price/forecast/backtest interface with source selection, CSV upload, test diagnostics and download buttons.
- Added fully offline synthetic demo and notebook `ml/compare_models.ipynb`.
- Added `python -m ml.run_comparison` reporting CLI, Docker dashboard target and optional local launcher scripts.
- Added offline tests for naïve forecasting, CSV parsing, and full Random Forest evaluation.
- Original Transformer checkpoint API and notebook remain available.
