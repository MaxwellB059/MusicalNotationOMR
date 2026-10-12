#!/bin/bash
# Secundo one-click launcher (macOS/Linux). Double-click in Finder to start backend + frontend.
# First time only: run `chmod +x launch.command` in this folder.
cd "$(dirname "$0")" || exit 1

command -v python3 >/dev/null || { echo "[ERROR] Python 3 not found."; read -rp "Press Enter..."; exit 1; }
command -v npm >/dev/null || { echo "[ERROR] Node.js/npm not found."; read -rp "Press Enter..."; exit 1; }

if [ ! -x ".venv/bin/python" ]; then
    echo "[setup] Creating Python virtual environment..."
    python3 -m venv .venv || exit 1
    echo "[setup] Installing Python packages (first run only, TensorFlow is large)..."
    .venv/bin/python -m pip install --upgrade pip
    .venv/bin/python -m pip install -r requirements.txt || exit 1
fi

if [ ! -d "node_modules" ]; then
    echo "[setup] Installing frontend packages (first run only)..."
    npm install || exit 1
fi

trap 'kill 0' EXIT INT TERM
echo "[start] Backend  -> http://127.0.0.1:8000"
.venv/bin/python -m uvicorn omr_server:app --host 127.0.0.1 --port 8000 &
echo "[start] Frontend -> http://localhost:8080/Secundo-v-1.0/ (next free port if 8080 is busy)"
npm run dev -- --open &   # --open: Vite opens the browser at whatever port it actually got

echo "Secundo is running. Press Ctrl+C or close this window to stop."
wait
