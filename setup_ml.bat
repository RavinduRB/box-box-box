@echo off
echo ==================================
echo F1 Race ML Pipeline - Setup Script
echo ==================================
echo.

REM Check Python version
echo Checking Python...
python --version
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python not found. Please install Python 3.7+
    pause
    exit /b 1
)

echo.
echo Installing dependencies...
echo This may take a few minutes...
pip install -r requirements_ml.txt

if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo ==================================
echo Setup Complete!
echo ==================================
echo.
echo Next steps:
echo 1. Run: jupyter notebook f1_race_ml_notebook.ipynb
echo 2. In Jupyter: Kernel menu ^> Restart ^& Run All
echo 3. Wait 5-10 minutes for models to train
echo 4. Review charts and metrics
echo.
echo Generated files:
echo - model_random_forest.pkl
echo - model_lstm.h5
echo - scaler.pkl
echo - feature_columns.pkl
echo.
pause
