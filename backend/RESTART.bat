@echo off
echo Restarting Backend Server...
echo.
echo Press Ctrl+C in the backend window to stop the old server first.
echo Then run this command:
echo.
echo .\venv\Scripts\uvicorn.exe main:app --host 0.0.0.0 --port 8000
echo.
pause
