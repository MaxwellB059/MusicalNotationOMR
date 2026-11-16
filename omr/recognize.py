"""One entry point for 'image in, MusicXML out' (PLAN.md S7/S8).

Order of preference:
  1. the trained CRNN (runs/crnn/best.onnx via onnxruntime, else best.keras via TensorFlow) -> reads real images
  2. the geometry pipeline (omr.assemble) with the interim classifier -> clean renders only
  3. the original single-note model, handled by omr_server.recognize (kept for backward compatibility)

Every path returns MusicXML text; problems are written into a `needs-review` comment, never silently guessed.
"""
import json
import os
from pathlib import Path

import cv2
import numpy as np

from . import crnn_data as D, tokens

CRNN_DIR = Path(os.environ.get("OMR_CRNN_DIR", Path(__file__).resolve().parent.parent / "runs" / "crnn"))


class CrnnReader:
    """Loads the CRNN once and turns a BGR image into tokens."""

    def __init__(self, directory=CRNN_DIR):
        self.vocab = json.load(open(Path(directory) / "vocab.json"))
        onnx, keras = Path(directory) / "best.onnx", Path(directory) / "best.keras"
        if onnx.exists():
            import onnxruntime as ort
            self.sess = ort.InferenceSession(str(onnx), providers=["CPUExecutionProvider"])
            self.run = lambda x: self.sess.run(None, {self.sess.get_inputs()[0].name: x})[0]
        elif keras.exists():
            from .omr_compat import apply_keras_fix
            apply_keras_fix()
            import tensorflow as tf
            model = tf.keras.models.load_model(str(keras), compile=False)
            self.run = lambda x: model.predict(x, verbose=0)
        else:
            raise FileNotFoundError(f"no CRNN model in {directory}")

    def tokens(self, img_bgr):
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
        h, w = gray.shape
        gray = cv2.resize(gray, (max(int(w * D.HEIGHT / h), D.WIDTH_STRIDE), D.HEIGHT), interpolation=cv2.INTER_AREA)
        x, widths = D.pad_batch([(255 - gray).astype(np.float32) / 255.0])
        probs = self.run(x)[0]
        return D.greedy_decode(probs[: max(widths[0] // D.WIDTH_STRIDE, 1)].argmax(-1), self.vocab)


_reader = None


def crnn_available(directory=CRNN_DIR):
    return (Path(directory) / "vocab.json").exists() and any((Path(directory) / n).exists()
                                                            for n in ("best.onnx", "best.keras"))


def recognize(img_bgr, reader=None):
    """Return MusicXML text for one line of music in `img_bgr`."""
    global _reader
    if reader is None and crnn_available():
        _reader = _reader or CrnnReader()
        reader = _reader
    if reader is not None:
        return tokens.tokens_to_musicxml(reader.tokens(img_bgr))
    from . import assemble
    return assemble.recognize_line(img_bgr)["xml"]
