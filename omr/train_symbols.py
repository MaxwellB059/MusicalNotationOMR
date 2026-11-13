"""Train + evaluate the symbol CNN (PLAN.md S4). Runs in the TensorFlow 2.15 env.

    python -m omr.train_symbols --config configs/symbols.yaml            # real run (GPU machine)
    python -m omr.train_symbols --config configs/symbols.yaml --smoke    # ~2 min CPU pipeline check

Outputs in out_dir: best.keras, labels.json, metrics.json (gate table), confusion.csv.
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np

GATES = {"per_class_min": 0.98, "perturbed_overall": 0.95, "unknown_recall": 0.95}


def load_config(path):
    import yaml
    return yaml.safe_load(open(path, encoding="utf-8"))


def build_model(n_classes, img=64):
    import tensorflow as tf
    L = tf.keras.layers
    m = tf.keras.Sequential([
        L.Input((img, img, 1)),
        L.Conv2D(32, 3, padding="same", activation="relu"), L.BatchNormalization(), L.MaxPool2D(),
        L.Conv2D(64, 3, padding="same", activation="relu"), L.BatchNormalization(), L.MaxPool2D(),
        L.Conv2D(128, 3, padding="same", activation="relu"), L.BatchNormalization(), L.MaxPool2D(),
        L.Conv2D(128, 3, padding="same", activation="relu"), L.MaxPool2D(),
        L.Flatten(), L.Dropout(0.4), L.Dense(256, activation="relu"), L.Dense(n_classes, activation="softmax"),
    ])
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
    return m


def perturb(x, rng):
    """The 'perturbed images' protocol from the mid-sem evidence: small rotation, scale, shift, blur."""
    out = np.empty_like(x)
    for i in range(len(x)):
        im = x[i, ..., 0]
        h, w = im.shape
        m = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-4, 4), 1 + rng.uniform(-0.08, 0.08))
        m[:, 2] += (rng.uniform(-2, 2), rng.uniform(-2, 2))
        im = cv2.warpAffine(im, m, (w, h))
        if rng.random() < 0.5:
            im = cv2.GaussianBlur(im, (0, 0), rng.uniform(0.3, 1.0))
        out[i, ..., 0] = im
    return out


def evaluate(model, x, y, classes, seed=0):
    rng = random.Random(seed)
    pred = model.predict(x, verbose=0).argmax(1)
    pert = model.predict(perturb(x, rng), verbose=0).argmax(1)
    n = len(classes)
    conf = np.zeros((n, n), int)
    for t, p in zip(y, pred):
        conf[t, p] += 1
    per_class = {classes[c]: float(conf[c, c] / conf[c].sum()) for c in range(n) if conf[c].sum() > 0}
    unk = classes.index("unknown")
    metrics = {
        "n_test": int(len(y)),
        "overall": float((pred == y).mean()),
        "perturbed_overall": float((pert == y).mean()),
        "per_class": per_class,
        "per_class_min": float(min(per_class.values())),
        "unknown_recall": per_class.get("unknown", 0.0),
        "classes_missing_from_test": [classes[c] for c in range(n) if conf[c].sum() == 0],
    }
    metrics["gates"] = {k: {"threshold": v, "value": metrics[k], "pass": metrics[k] >= v}
                        for k, v in GATES.items()}
    return metrics, conf


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/symbols.yaml")
    ap.add_argument("--smoke", action="store_true", help="tiny run to verify the pipeline")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    if a.smoke:
        cfg.update(cfg.get("smoke", {}))
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"])

    from . import symbols
    from .omr_compat import apply_keras_fix
    apply_keras_fix()
    import tensorflow as tf
    tf.random.set_seed(cfg["seed"])

    out = Path(cfg["out_dir"] + ("_smoke" if a.smoke else ""))
    out.mkdir(parents=True, exist_ok=True)
    data_path = Path(cfg["data_path"] + (".smoke.npz" if a.smoke else ""))
    if not data_path.exists():
        print("generating dataset ->", data_path, flush=True)
        print(symbols.build_dataset(cfg["pages"], cfg["seed"], data_path, aug_per_crop=cfg["aug_per_crop"]))
    d = np.load(data_path)
    classes = [str(c) for c in d["classes"]]
    xtr, ytr, xva, yva, xte, yte = (d["x_train"], d["y_train"], d["x_val"], d["y_val"], d["x_test"], d["y_test"])

    counts = np.bincount(ytr, minlength=len(classes)).astype(float)
    weights = {i: float(counts.sum() / (len(classes) * max(c, 1))) for i, c in enumerate(counts)}
    model = build_model(len(classes))
    ckpt = out / "best.keras"
    if a.resume and ckpt.exists():
        model = tf.keras.models.load_model(str(ckpt))
    cbs = [tf.keras.callbacks.ModelCheckpoint(str(ckpt), monitor="val_accuracy", save_best_only=True),
           tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=cfg["patience"],
                                            restore_best_weights=True)]
    model.fit(xtr, ytr, validation_data=(xva, yva), epochs=cfg["epochs"], batch_size=cfg["batch_size"],
              class_weight=weights, callbacks=cbs, verbose=2)
    if not ckpt.exists():
        model.save(str(ckpt))
    json.dump({"classes": classes, "img_size": 64}, open(out / "labels.json", "w"), indent=2)
    metrics, conf = evaluate(model, xte, yte, classes, cfg["seed"])
    json.dump(metrics, open(out / "metrics.json", "w"), indent=2)
    np.savetxt(out / "confusion.csv", conf, fmt="%d", delimiter=",", header=",".join(classes))
    print(json.dumps({k: metrics[k] for k in ("overall", "perturbed_overall", "per_class_min", "unknown_recall")}))
    for k, g in metrics["gates"].items():
        print(f"GATE {k}: {g['value']:.3f} (>= {g['threshold']}) {'PASS' if g['pass'] else 'FAIL'}")
    return metrics


if __name__ == "__main__":
    main()
