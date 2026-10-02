import os, sys, json, math, glob, argparse, random
from pathlib import Path

import numpy as np
import cv2
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import tensorflow as tf

from omr_common import (IMG_SIZE, preprocess_bgr_to_tensor, build_single_note_musicxml,
                        apply_windows_keras_fix)

# -----------------------------
# Config defaults
# -----------------------------
apply_windows_keras_fix()  # see omr_common.py
SEED     = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

# Preprocessing lives in omr_common.py (shared with omr_server.py)

def load_image_as_tensor(path, target=IMG_SIZE):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return preprocess_bgr_to_tensor(img, target=target)

# -----------------------------
# Dataset: pitch + duration
# -----------------------------
def list_dataset_pitch_duration(root_dir):
    """
    Expected layout:

      root_dir/
        PITCH_1/
          DURATION_1/*.png
          DURATION_2/*.png
        PITCH_2/
          DURATION_1/*.png
          DURATION_2/*.png
        ...

    Returns:
      paths, pitch_labels, duration_labels, pitch_classes, duration_classes
    """
    root = Path(root_dir)
    pitch_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    if not pitch_dirs:
        raise RuntimeError(f"No pitch folders found in {root_dir}")

    pitch_classes = [d.name for d in pitch_dirs]

    # assume all pitches share the same duration folder names
    sample_dur_root = pitch_dirs[0]
    duration_dirs = sorted([d for d in sample_dur_root.iterdir() if d.is_dir()])
    if not duration_dirs:
        raise RuntimeError(f"No duration folders found inside {sample_dur_root}")
    duration_classes = [d.name for d in duration_dirs]

    paths = []
    pitch_labels = []
    duration_labels = []

    for p_idx, p_dir in enumerate(pitch_dirs):
        for d_idx, d_name in enumerate(duration_classes):
            d_dir = p_dir / d_name
            if not d_dir.is_dir():
                # skip if this particular combination doesn't exist
                continue
            for img_path in glob.glob(str(d_dir / "*")):
                if img_path.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
                    paths.append(img_path)
                    pitch_labels.append(p_idx)
                    duration_labels.append(d_idx)

    if not paths:
        raise RuntimeError(f"No images found in nested pitch/duration folders under {root_dir}")

    return paths, pitch_labels, duration_labels, pitch_classes, duration_classes


def make_tf_dataset_multi(paths, pitch_labels, duration_labels,
                          batch_size=64, shuffle=True, augment=True):
    """
    Build a tf.data.Dataset that yields (image, {'pitch': y_pitch, 'duration': y_dur})
    """
    x  = np.array(paths)
    yp = np.array(pitch_labels, dtype=np.int32)
    yd = np.array(duration_labels, dtype=np.int32)

    def _load_img(path):
        path = path.numpy().decode("utf-8")
        arr = load_image_as_tensor(path)  # (H,W,1)
        return arr

    def _tf_parse(path, y_pitch, y_dur):
        arr = tf.py_function(_load_img, [path], tf.float32)
        arr.set_shape((IMG_SIZE, IMG_SIZE, 1))
        return arr, {"pitch": y_pitch, "duration": y_dur}

    ds = tf.data.Dataset.from_tensor_slices((x, yp, yd))
    ds = ds.map(_tf_parse, num_parallel_calls=tf.data.AUTOTUNE)

    if augment:
        def _aug(img, labels):
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_brightness(img, 0.15)
            img = tf.image.random_contrast(img, 0.8, 1.2)
            return img, labels
        ds = ds.map(_aug, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(4096, seed=SEED)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

# -----------------------------
# Model (compact CNN) - multi-head
# -----------------------------
def build_cnn_multi(n_pitch_classes, n_duration_classes):
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 1))

    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool2D()(x)

    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool2D()(x)

    x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool2D()(x)

    x = tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool2D()(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)

    # two heads
    pitch_out    = tf.keras.layers.Dense(n_pitch_classes,    activation="softmax", name="pitch")(x)
    duration_out = tf.keras.layers.Dense(n_duration_classes, activation="softmax", name="duration")(x)

    model = tf.keras.Model(inputs=inputs, outputs={"pitch": pitch_out, "duration": duration_out},
                           name="SingleNoteOMR_CNN_MultiHead")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss={
            "pitch": "sparse_categorical_crossentropy",
            "duration": "sparse_categorical_crossentropy",
        },
        metrics={
            "pitch": "accuracy",
            "duration": "accuracy",
        },
    )
    return model

# -----------------------------
# MusicXML (single note helper)
# -----------------------------
def write_single_note_musicxml(pitch_name, out_path="single_note.xml", **kwargs):
    """Write build_single_note_musicxml() output (from omr_common) to a file."""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(build_single_note_musicxml(pitch_name, **kwargs))
    return out_path

# -----------------------------
# Train / Evaluate (multi-head)
# -----------------------------
def train(args):
    # load dataset for pitch + duration
    train_paths, train_pitch_labels, train_dur_labels, pitch_classes, duration_classes = \
        list_dataset_pitch_duration(args.train_dir)
    val_paths, val_pitch_labels, val_dur_labels, pitch_classes_val, duration_classes_val = \
        list_dataset_pitch_duration(args.val_dir)

    if pitch_classes != pitch_classes_val or duration_classes != duration_classes_val:
        raise RuntimeError(
            "Class folders in train and val do not match.\n"
            f"train pitch classes: {pitch_classes}\n"
            f"val pitch classes:   {pitch_classes_val}\n"
            f"train duration classes: {duration_classes}\n"
            f"val duration classes:   {duration_classes_val}"
        )

    print(f"Pitch classes ({len(pitch_classes)}): {pitch_classes}")
    print(f"Duration classes ({len(duration_classes)}): {duration_classes}")
    bs = args.batch_size

    ds_train = make_tf_dataset_multi(train_paths, train_pitch_labels, train_dur_labels,
                                     batch_size=bs, shuffle=True, augment=True)
    ds_val   = make_tf_dataset_multi(val_paths,   val_pitch_labels,   val_dur_labels,
                                     batch_size=bs, shuffle=False, augment=False)

    model = build_cnn_multi(
        n_pitch_classes=len(pitch_classes),
        n_duration_classes=len(duration_classes),
    )
    model.summary()

    ckpt_path = Path(args.out_dir) / "best.keras"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(ckpt_path),
            monitor="val_pitch_accuracy",   # main metric to watch
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_pitch_accuracy",
            patience=8,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            verbose=1,
        ),
    ]

    history = model.fit(ds_train, validation_data=ds_val,
                        epochs=args.epochs, callbacks=callbacks)

    # Evaluate and print sklearn report for pitch and duration separately
    y_true_pitch, y_pred_pitch = [], []
    y_true_dur,   y_pred_dur   = [], []

    for xb, yb in ds_val:
        probs = model.predict(xb, verbose=0)
        pitch_probs    = probs["pitch"]
        duration_probs = probs["duration"]

        y_pred_pitch.extend(np.argmax(pitch_probs, axis=1).tolist())
        y_pred_dur.extend(np.argmax(duration_probs, axis=1).tolist())

        y_true_pitch.extend(yb["pitch"].numpy().tolist())
        y_true_dur.extend(yb["duration"].numpy().tolist())

    print("\n=== Pitch classification report ===")
    print(classification_report(y_true_pitch, y_pred_pitch,
                                target_names=pitch_classes, digits=4))

    print("\n=== Duration classification report ===")
    print(classification_report(y_true_dur, y_pred_dur,
                                target_names=duration_classes, digits=4))

    cm_pitch = confusion_matrix(y_true_pitch, y_pred_pitch)
    cm_dur   = confusion_matrix(y_true_dur,   y_pred_dur)

    print("Pitch confusion matrix:\n", cm_pitch)
    print("Duration confusion matrix:\n", cm_dur)

    # Save label map
    label_map_path = Path(args.out_dir) / "labels.json"
    with open(label_map_path, "w") as f:
        json.dump(
            {
                "pitch_classes": pitch_classes,
                "duration_classes": duration_classes,
            },
            f,
            indent=2,
        )
    print(f"Saved best model to: {ckpt_path}")
    print(f"Saved labels to:     {label_map_path}")

# -----------------------------
# Predict (multi-head)
# -----------------------------
def predict_image(args):
    # Load model + labels
    model = tf.keras.models.load_model(args.model_path)

    with open(args.labels_path, "r") as f:
        label_info = json.load(f)
        pitch_classes    = label_info["pitch_classes"]
        duration_classes = label_info["duration_classes"]

    # Preprocess
    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(args.image)
    x = preprocess_bgr_to_tensor(img, target=IMG_SIZE)
    x = np.expand_dims(x, axis=0)  # batch

    # Predict
    probs = model.predict(x, verbose=0)
    pitch_probs    = probs["pitch"][0]
    duration_probs = probs["duration"][0]

    pitch_idx    = int(np.argmax(pitch_probs))
    duration_idx = int(np.argmax(duration_probs))

    pred_pitch    = pitch_classes[pitch_idx]
    pred_duration = duration_classes[duration_idx]

    conf_pitch    = float(pitch_probs[pitch_idx])
    conf_duration = float(duration_probs[duration_idx])

    print(f"Prediction (pitch):    {pred_pitch}  (conf {conf_pitch:.3f})")
    print(f"Prediction (duration): {pred_duration}  (conf {conf_duration:.3f})")

    # Optional: write a one-note MusicXML with that pitch + duration
    if args.write_musicxml:
        out_xml = Path(args.out_xml or "predicted_note.xml")
        path = write_single_note_musicxml(
            pred_pitch,
            out_path=str(out_xml),
            duration_type=pred_duration,
        )
        print(f"Wrote MusicXML: {path}")

# -----------------------------
# CLI
# -----------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Single-note OMR with TensorFlow (CNN), multi-head (pitch + duration).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train", help="Train a CNN on single-note images (pitch + duration).")
    tr.add_argument("--train_dir", required=True, help="Path to train/PITCH/DURATION/*.png")
    tr.add_argument("--val_dir",   required=True, help="Path to val/PITCH/DURATION/*.png")
    tr.add_argument("--epochs", type=int, default=30)
    tr.add_argument("--batch_size", type=int, default=64)
    tr.add_argument("--out_dir", default="omr_runs/singlenote_multi")
    tr.set_defaults(func=train)

    pr = sub.add_parser("predict", help="Predict a single image and (optionally) emit MusicXML.")
    pr.add_argument("--image", required=True)
    pr.add_argument("--model_path", required=True)
    pr.add_argument("--labels_path", required=True)
    pr.add_argument("--write_musicxml", action="store_true")
    pr.add_argument("--out_xml", default=None)
    pr.set_defaults(func=predict_image)

    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    args.func(args)