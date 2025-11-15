@echo off
echo ============================================================
echo TDR FAULT CLASSIFICATION SYSTEM - STARTUP SCRIPT
echo ============================================================
echo.

echo Starting Backend Server...
echo.
start "TDR Backend" cmd /k "cd Backend && .\venv\Scripts\uvicorn.exe main:app --host 0.0.0.0 --port 8000"

timeout /t 5 /nobreak >nul

echo Starting Frontend Server...
echo.
start "TDR Frontend" cmd /k "npm run dev"

echo.
echo ============================================================
echo Both servers are starting in separate windows!
echo.
echo Backend:  http://localhost:8000
echo Frontend: http://localhost:3001
echo.
echo Wait a few seconds for both servers to fully start.
echo Then open your browser to http://localhost:3001
echo ============================================================
pause
