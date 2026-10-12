# Secundo-v-1.0

Secundo is an Optical Music Recognition (OMR) system that converts images of sheet music into MusicXML, a standard format supported by notation software such as MuseScore and Finale. The goal is to remove manual transcription by allowing users to upload scanned or photographed sheet music and receive an editable digital score.

The project is built in Python and combines classical image processing with a TensorFlow/Keras CNN to recognize musical symbols. A preprocessing pipeline corrects skew, normalizes layout, and prepares note images for classification. Recognized symbols are then programmatically assembled into a valid MusicXML file.

Secundo includes a lightweight backend service and web interface that together provide an end-to-end pipeline from image upload to downloadable MusicXML output.

Key technologies: Python, TensorFlow/Keras, OpenCV, CNNs, FastAPI, MusicXML, React/Vite


## Quick Start (one-click launch)

Prerequisites: [Python 3.10+](https://www.python.org/downloads/) and [Node.js](https://nodejs.org/).

- **Windows:** double-click `launch.bat`
- **macOS/Linux:** double-click `launch.command` (first time only, run `chmod +x launch.command`)

The first run creates a `.venv`, installs the Python and npm packages (TensorFlow is large, so this takes a few minutes), then starts both servers and opens the app in your browser. If port 8080 is already in use, Vite moves to the next free port and opens the browser there automatically. Later runs start in seconds.

| Service | URL |
|---|---|
| Web app (Vite) | http://localhost:8080/Secundo-v-1.0/ |
| OMR API (FastAPI) | http://127.0.0.1:8000/api/omr |

To stop, close the "Secundo Backend" and "Secundo Frontend" windows (macOS: Ctrl+C in the terminal window).

### Manual launch
```bash
pip install -r requirements.txt
uvicorn omr_server:app --port 8000   # terminal 1
npm install && npm run dev           # terminal 2
```
