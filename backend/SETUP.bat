@echo off
echo ========================================
echo TDR Backend Setup Script
echo ========================================
echo.

echo Checking Python version...
python --version
echo.

echo If you see Python 3.14, close this window and open a NEW terminal!
echo.
pause

echo Removing old virtual environment...
if exist venv (
    rmdir /s /q venv
    echo Old venv removed.
) else (
    echo No old venv found.
)
echo.

echo Creating new virtual environment with Python 3.11...
python -m venv venv
echo Virtual environment created!
echo.

echo Activating virtual environment...
call venv\Scripts\activate
echo.

echo Installing PyTorch (CPU version)...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
echo.

echo Installing other requirements...
pip install -r requirements.txt
echo.

echo ========================================
echo Setup Complete! ✓
echo ========================================
echo.
echo To start the backend server, run:
echo   cd backend
echo   venv\Scripts\activate
echo   python main.py
echo.
pause
