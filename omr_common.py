# omr_common.py
"""Code shared by the training script (single_note_cnn.py) and the API server (omr_server.py).

Keeping preprocessing in ONE place matters: the model is trained on exactly what
preprocess_bgr_to_tensor() produces, so the server must feed it the same thing.
"""
import math

import cv2
import numpy as np
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom import minidom

IMG_SIZE = 64  # square input to the CNN


def apply_windows_keras_fix():
    """Make Keras 2.15 `.keras` files load/save on Windows.

    Keras 2.15 builds weight paths inside the .keras archive with os.path.join, which on
    Windows gives a backslash-separated path (layers + backslash + conv2d). HDF5 only understands '/', so every layer loads with
    0 variables ("Layer 'conv2d' expected 2 variables, but received 0"). Normalising the
    separator fixes loading, and also makes models trained on Windows load on Mac/Linux.
    Safe no-op on other platforms and on Keras 3.
    """
    import os
    if os.sep != "\\":
        return
    try:
        from keras.src.saving import saving_lib
    except ImportError:
        return
    store = getattr(saving_lib, "H5IOStore", None)
    if store is None or getattr(store, "_slash_fix", False):
        return
    for name in ("get", "make"):
        orig = getattr(store, name)
        setattr(store, name, (lambda f: lambda self, path: f(self, path.replace("\\", "/")))(orig))
    store._slash_fix = True

def preprocess_bgr_to_tensor(img_bgr, target=IMG_SIZE):
    """Same as in single_note_cnn.py."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    m = cv2.moments(gray)
    if abs(m["mu02"]) > 1e-3:
        skew = m["mu11"] / m["mu02"]
        angle = math.degrees(math.atan(skew))
    else:
        angle = 0.0
    if abs(angle) > 0.3 and abs(angle) < 10:
        (h, w) = gray.shape
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        gray = cv2.warpAffine(
            gray, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )

    bw = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 10
    )

    ys, xs = np.where(bw > 0)
    if len(xs) < 10 or len(ys) < 10:
        roi = bw
    else:
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        roi = bw[y0:y1 + 1, x0:x1 + 1]

    h, w = roi.shape[:2]
    pad = abs(h - w)
    if h > w:
        left = pad // 2
        right = pad - left
        roi = cv2.copyMakeBorder(
            roi, 0, 0, left, right, cv2.BORDER_CONSTANT, value=0
        )
    elif w > h:
        top = pad // 2
        bottom = pad - top
        roi = cv2.copyMakeBorder(
            roi, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=0
        )

    roi = cv2.resize(roi, (target, target), interpolation=cv2.INTER_AREA)
    roi = roi.astype(np.float32) / 255.0
    roi = np.expand_dims(roi, axis=-1)
    return roi

def build_single_note_musicxml(
    pitch_name,
    divisions=8,
    duration_type="whole",
    beats=4,
    beat_type=4,
    clef_sign="G",
    clef_line=2,
    key_fifths=0,
):
    """Return a MusicXML string for a single-note, single-measure score."""
    DUR = {"whole": 4, "half": 2, "quarter": 1, "eighth": 0.5, "16th": 0.25}
    dur_divs = int(DUR[duration_type] * divisions)

    def prettify(elem):
        rough = tostring(elem, encoding="utf-8")
        return minidom.parseString(rough).toprettyxml(indent="  ")

    score = Element("score-partwise", version="3.1")
    part_list = SubElement(score, "part-list")
    sp = SubElement(part_list, "score-part", id="P1")
    SubElement(sp, "part-name").text = "OMR"

    part = SubElement(score, "part", id="P1")
    measure = SubElement(part, "measure", number="1")
    attr = SubElement(measure, "attributes")
    SubElement(attr, "divisions").text = str(divisions)
    ks = SubElement(attr, "key")
    SubElement(ks, "fifths").text = str(key_fifths)
    ts = SubElement(attr, "time")
    SubElement(ts, "beats").text = str(beats)
    SubElement(ts, "beat-type").text = str(beat_type)
    clef = SubElement(attr, "clef")
    SubElement(clef, "sign").text = clef_sign
    SubElement(clef, "line").text = str(clef_line)

    note = SubElement(measure, "note")
    p = SubElement(note, "pitch")
    step, octave = pitch_name[:-1], int(pitch_name[-1])
    SubElement(p, "step").text = step
    SubElement(p, "octave").text = str(octave)
    SubElement(note, "duration").text = str(dur_divs)
    SubElement(note, "voice").text = "1"
    SubElement(note, "type").text = duration_type
    SubElement(note, "staff").text = "1"

    return prettify(score)
