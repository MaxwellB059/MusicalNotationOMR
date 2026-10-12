@echo off
REM Secundo one-click launcher (Windows). Double-click to start backend + frontend.
cd /d "%~dp0"
title Secundo Launcher

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python not found. Install Python 3.10+ from python.org, tick "Add to PATH", then retry.
    pause
    exit /b 1
)
where npm >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Node.js/npm not found. Install Node from nodejs.org, then retry.
    pause
    exit /b 1
)

REM ---- Python setup (first run only). The marker file means a half-finished install gets redone. ----
if not exist ".venv\Scripts\python.exe" (
    echo [setup] Creating Python virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Could not create the virtual environment.
        pause
        exit /b 1
    )
)
if not exist ".venv\.deps_installed" (
    echo [setup] Installing Python packages ^(first run only, TensorFlow is large - a few minutes^)...
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] pip install failed. Fix the error above and run this again.
        pause
        exit /b 1
    )
    echo ok> ".venv\.deps_installed"
)

REM ---- Frontend setup (first run only) ----
if not exist "node_modules\.bin\vite.cmd" (
    echo [setup] Installing frontend packages ^(first run only^)...
    call npm install
    if errorlevel 1 (
        echo [ERROR] npm install failed. Fix the error above and run this again.
        pause
        exit /b 1
    )
)

REM ---- Backend port check: the frontend always calls http://127.0.0.1:8000 ----
netstat -ano | findstr /R /C:":8000 .*LISTENING" >nul
if not errorlevel 1 (
    echo [ERROR] Port 8000 is already in use, so the Secundo backend cannot start.
    echo         Close whatever is using it ^(an old Secundo Backend window?^) and run this again.
    pause
    exit /b 1
)

echo [start] Backend  -^> http://127.0.0.1:8000  ^(takes ~15s to load the model^)
start "Secundo Backend" cmd /k ".venv\Scripts\python.exe -m uvicorn omr_server:app --host 127.0.0.1 --port 8000"

echo [start] Frontend -^> http://localhost:8080/Secundo-v-1.0/  ^(next free port if 8080 is busy; your browser opens automatically^)
start "Secundo Frontend" cmd /k "npm run dev -- --open"

echo.
echo Secundo is starting. Wait for "Application startup complete" in the Backend window before converting.
echo To stop: close the "Secundo Backend" and "Secundo Frontend" windows.
timeout /t 8 >nul
