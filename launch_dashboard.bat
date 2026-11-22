@echo off
cd /d "%~dp0"
python -m streamlit run app.py
if errorlevel 1 (
  echo.
  echo Startup failed. Install dependencies first:
  echo python -m pip install -r requirements.txt
  pause
)
