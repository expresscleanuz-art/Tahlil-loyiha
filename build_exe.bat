@echo off
title Kiber AI - EXE Build Script
echo ==========================================================
echo        Kiber AI Desktop (.EXE) Yaratish Dasturi
echo ==========================================================

echo.
echo [1/3] Frontend fayllari build qilinmoqda (React Vite)...
cd frontend
call npm run build
if %errorlevel% neq 0 (
    echo.
    echo [XATO] Frontend build jarayonida xatolik yuz berdi!
    cd ..
    pause
    exit /b %errorlevel%
)
cd ..

echo.
echo [2/3] Kerakli kutubxonalar tekshirilmoqda...
pip install "numpy<2" pyinstaller pywebview
pip install -r requirements.txt

echo.
echo [3/3] PyInstaller orqali .EXE dasturi yig'ilmoqda...
pyinstaller --noconfirm --onedir --windowed --name "Kiber_AI" ^
  --add-data "frontend/dist;frontend/dist" ^
  --add-data "backend;backend" ^
  app_desktop.py

echo.
echo config.json fayli dastur papkasiga nusxalanmoqda...
copy config.json dist\Kiber_AI\config.json

echo.
echo ==========================================================
echo   TABRIKLAYMIZ! EXE dastur muvaffaqiyatli yaratildi!
echo.
echo   Joylashgan joyi: 
echo   dist\Kiber_AI\Kiber_AI.exe
echo   dist\Kiber_AI\config.json (Dastur nomini o'zgartirish fayli)
echo ==========================================================
pause
