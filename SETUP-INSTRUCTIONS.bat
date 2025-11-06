@echo off
echo ===============================================
echo  TDR System - Complete Rebuild Done!
echo ===============================================
echo.
echo NEXT STEPS:
echo.
echo 1. Add Your Models to: backend\models\
echo    - image_fault_classifier.pth
echo    - hybrid_distance_regressor.pth
echo    - meta_ct.joblib
echo.
echo 2. Setup Backend (First time only):
echo    cd backend
echo    python -m venv venv
echo    venv\Scripts\activate
echo    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
echo    pip install -r requirements.txt
echo.
echo 3. Run Backend:
echo    cd backend
echo    venv\Scripts\activate
echo    python main.py
echo.
echo 4. Run Frontend (new terminal):
echo    npm run dev
echo.
echo 5. Open: http://localhost:3001
echo.
echo ===============================================
pause
