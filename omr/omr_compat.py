"""Thin wrapper so the training code can reuse the Windows Keras-2.15 loader fix in omr_common.py."""
import sys
from pathlib import Path


def apply_keras_fix():
    root = str(Path(__file__).resolve().parent.parent)
    if root not in sys.path:
        sys.path.insert(0, root)
    from omr_common import apply_windows_keras_fix
    apply_windows_keras_fix()
