# omr_server.py
import json
from pathlib import Path

import numpy as np
import cv2
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

# Preprocessing + MusicXML building are shared with single_note_cnn.py (training)
from omr_common import (IMG_SIZE, preprocess_bgr_to_tensor, build_single_note_musicxml,
                        apply_windows_keras_fix)

# --- Load model + labels once at startup ---

MODEL_PATH = Path("runs/seven_notes/best.keras")  # adjust if needed
LABELS_PATH = Path("runs/seven_notes/labels.json")

apply_windows_keras_fix()  # see omr_common.py: Keras 2.15 .keras files break on Windows
print("Loading model...")
model = tf.keras.models.load_model(str(MODEL_PATH))
with open(LABELS_PATH, "r") as f:
    labels = json.load(f)

# Two label formats exist:
#   legacy / single-head:  {"classes": [...pitches]}                       -> pitch only
#   current / multi-head:  {"pitch_classes": [...], "duration_classes": [...]}
#                          (what single_note_cnn.py train writes)
if "pitch_classes" in labels:
    pitch_classes = labels["pitch_classes"]
    duration_classes = labels.get("duration_classes")
else:
    pitch_classes = labels["classes"]
    duration_classes = None
print("Pitch classes:", pitch_classes)
print("Duration classes:", duration_classes or "(none; defaulting to whole note)")

DEFAULT_DURATION = "whole"

app = FastAPI()

# Allow the frontend from any localhost port: Vite falls back to 8081, 8082... when 8080 is busy.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def predict_pitch_and_duration(x):
    """Run the model on a (1, H, W, 1) batch; return (pitch_name, duration_type)."""
    out = model.predict(x, verbose=0)

    if duration_classes is None:
        # single-head model: output is one softmax array
        pitch_probs = out[0]
        return pitch_classes[int(np.argmax(pitch_probs))], DEFAULT_DURATION

    # multi-head model: dict keyed by output name (or a list in pitch, duration order)
    if isinstance(out, dict):
        pitch_probs, dur_probs = out["pitch"][0], out["duration"][0]
    else:
        pitch_probs, dur_probs = out[0][0], out[1][0]
    return (
        pitch_classes[int(np.argmax(pitch_probs))],
        duration_classes[int(np.argmax(dur_probs))],
    )


@app.post("/api/omr", response_class=PlainTextResponse)
async def omr_endpoint(file: UploadFile = File(...)):
    """
    Accept an uploaded measure image, classify the note, and return MusicXML text.
    """
    contents = await file.read()
    file_bytes = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    x = preprocess_bgr_to_tensor(img, target=IMG_SIZE)
    x = np.expand_dims(x, axis=0)

    pitch_name, duration_type = predict_pitch_and_duration(x)
    xml_str = build_single_note_musicxml(pitch_name, duration_type=duration_type)

    # Frontend can both preview this text and offer it as a .musicxml download
    return xml_str
