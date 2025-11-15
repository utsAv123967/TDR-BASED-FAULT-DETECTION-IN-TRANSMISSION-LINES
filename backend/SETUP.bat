@echo off
echo ========================================
echo TDR Backend Setup Script
echo ========================================
echo.

echo Checking Python version...
python --version
echo.

echo Creating virtual environment...
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
echo Next steps:
echo 1. Place your image_fault_classifier.pth in the models\ folder
echo 2. Run START.bat to start the server
echo.
pause
