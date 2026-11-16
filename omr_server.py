# omr_server.py
"""Entry point: `uvicorn omr_server:app`. Loads the current model and builds the app from omr.api.

The HTTP layer, CORS, health check, error handling and optional MongoDB logging live in omr/api.py
and omr/db.py so they can be tested without TensorFlow.
"""
import json
import os
from pathlib import Path

import numpy as np
import tensorflow as tf

# Preprocessing + MusicXML building are shared with single_note_cnn.py (training)
from omr_common import (IMG_SIZE, preprocess_bgr_to_tensor, build_single_note_musicxml,
                        apply_windows_keras_fix)
from omr.api import create_app
from omr.db import get_db

# --- Load model + labels once at startup ---

MODEL_PATH = Path(os.environ.get("OMR_MODEL_PATH", "runs/seven_notes/best.keras"))
LABELS_PATH = Path(os.environ.get("OMR_LABELS_PATH", "runs/seven_notes/labels.json"))

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


def recognize_single_note(img_bgr):
    """Original single-note pipeline (kept as the last-resort fallback)."""
    x = preprocess_bgr_to_tensor(img_bgr, target=IMG_SIZE)
    x = np.expand_dims(x, axis=0)
    pitch_name, duration_type = predict_pitch_and_duration(x)
    return build_single_note_musicxml(pitch_name, duration_type=duration_type)


def recognize(img_bgr):
    """Trained line reader (CRNN) when its model is present; otherwise the original single-note model.
    The geometry pipeline (omr.recognize) is available but only reads clean renders, so it is not the default."""
    from omr import recognize as R
    if R.crnn_available():
        return R.recognize(img_bgr)
    return recognize_single_note(img_bgr)


# MONGODB_URI unset -> stateless (db is None). Set ALLOWED_ORIGINS for the deployed frontend URL.
app = create_app(recognize, db=get_db())
