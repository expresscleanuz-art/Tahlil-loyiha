@echo off
title Start AI Web App

echo ==============================================
echo   Starting Backend (FastAPI)...
echo ==============================================
start cmd /k "pip install -r requirements.txt && uvicorn backend.main:app --reload --port 8000"

echo.
echo ==============================================
echo   Starting Frontend (React/Vite)...
echo ==============================================
cd frontend
start cmd /k "npm run dev"

echo.
echo Servers are starting! 
echo Frontend: http://localhost:5173
echo Backend API: http://localhost:8000
pause
