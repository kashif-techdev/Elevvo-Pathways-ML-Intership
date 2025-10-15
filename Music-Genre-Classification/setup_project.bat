@echo off
echo ========================================
echo Music Genre Classification Setup
echo ========================================
echo.

echo [STEP 1] Installing Python...
echo Please choose one of the following options:
echo.
echo Option 1: Download from python.org
echo   - Go to https://www.python.org/downloads/
echo   - Download Python 3.9 or 3.10
echo   - Install with "Add Python to PATH" checked
echo.
echo Option 2: Use Windows Store
echo   - Open Microsoft Store
echo   - Search for "Python 3.9" or "Python 3.10"
echo   - Install it
echo.
echo Option 3: Use winget (if available)
echo   - Run: winget install Python.Python.3.11
echo.

pause

echo [STEP 2] Verifying Python installation...
python --version
if %errorlevel% neq 0 (
    echo ERROR: Python not found! Please install Python first.
    pause
    exit /b 1
)

echo [STEP 3] Installing project dependencies...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install dependencies!
    pause
    exit /b 1
)

echo [STEP 4] Setting up dataset...
python scripts/download_dataset.py

echo [STEP 5] Running complete training pipeline...
python scripts/train_all_models.py

echo [STEP 6] Launching demo app...
echo Opening Streamlit app in your browser...
streamlit run app.py

pause
