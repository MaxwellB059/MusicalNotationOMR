"""Export a trained Keras symbol CNN to ONNX so production serves it with onnxruntime (PLAN.md S8a).

    python tools/export_onnx.py --model runs/symbols/best.keras --out runs/symbols/best.onnx

Needs `pip install tf2onnx onnxruntime` in the TensorFlow env. Also verifies parity: ONNX and Keras
outputs must agree within --tol on random inputs, otherwise the script exits non-zero.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tol", type=float, default=1e-4)
    a = ap.parse_args()

    from omr.omr_compat import apply_keras_fix
    apply_keras_fix()
    import tensorflow as tf
    import tf2onnx
    import onnxruntime as ort

    model = tf.keras.models.load_model(a.model)
    spec = (tf.TensorSpec((None,) + tuple(model.input_shape[1:]), tf.float32, name="input"),)
    tf2onnx.convert.from_keras(model, input_signature=spec, output_path=a.out)

    x = np.random.default_rng(0).random((16,) + tuple(model.input_shape[1:])).astype(np.float32)
    want = model.predict(x, verbose=0)
    sess = ort.InferenceSession(a.out, providers=["CPUExecutionProvider"])
    got = sess.run(None, {sess.get_inputs()[0].name: x})[0]
    err = float(np.abs(want - got).max())
    print(f"max abs diff Keras vs ONNX: {err:.2e}")
    if err > a.tol:
        sys.exit(f"parity FAILED ({err:.2e} > {a.tol})")
    print("parity OK ->", a.out)


if __name__ == "__main__":
    main()
