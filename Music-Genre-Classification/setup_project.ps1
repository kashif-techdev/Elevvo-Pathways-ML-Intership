# Music Genre Classification Setup Script
# PowerShell script to set up the project

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Music Genre Classification Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Test if Python is available
Write-Host "`n[STEP 1] Testing Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Python found: $pythonVersion" -ForegroundColor Green
    } else {
        Write-Host "❌ Python not found!" -ForegroundColor Red
        Write-Host "Please install Python from https://www.python.org/downloads/" -ForegroundColor Yellow
        Write-Host "Make sure to check 'Add Python to PATH' during installation" -ForegroundColor Yellow
        exit 1
    }
} catch {
    Write-Host "❌ Python not found!" -ForegroundColor Red
    Write-Host "Please install Python from https://www.python.org/downloads/" -ForegroundColor Yellow
    exit 1
}

# Test if pip is available
Write-Host "`n[STEP 2] Testing pip..." -ForegroundColor Yellow
try {
    $pipVersion = pip --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ pip found: $pipVersion" -ForegroundColor Green
    } else {
        Write-Host "❌ pip not found!" -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "❌ pip not found!" -ForegroundColor Red
    exit 1
}

# Install dependencies
Write-Host "`n[STEP 3] Installing dependencies..." -ForegroundColor Yellow
try {
    pip install -r requirements.txt
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Dependencies installed successfully!" -ForegroundColor Green
    } else {
        Write-Host "❌ Failed to install dependencies!" -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "❌ Failed to install dependencies!" -ForegroundColor Red
    exit 1
}

# Test Python script
Write-Host "`n[STEP 4] Testing Python script..." -ForegroundColor Yellow
try {
    python test_python.py
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Python script test passed!" -ForegroundColor Green
    } else {
        Write-Host "❌ Python script test failed!" -ForegroundColor Red
    }
} catch {
    Write-Host "❌ Python script test failed!" -ForegroundColor Red
}

Write-Host "`n🎉 Setup complete! Next steps:" -ForegroundColor Green
Write-Host "1. python scripts/download_dataset.py" -ForegroundColor Cyan
Write-Host "2. python scripts/train_all_models.py" -ForegroundColor Cyan
Write-Host "3. streamlit run app.py" -ForegroundColor Cyan

Write-Host "`nPress any key to continue..." -ForegroundColor Yellow
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
