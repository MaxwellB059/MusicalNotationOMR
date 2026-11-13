"""Interim symbol classifier: a random forest on hand-made shape features (PLAN.md S4 stopgap).

The CNN in omr.train_symbols is the intended classifier. Until it is trained, the assembly step
(S7) needs *some* classifier, and this one trains in under a minute on CPU from the same
labelled symbol crops. It is also a useful baseline for the CNN: if the CNN can't beat shape
features on the same split, something is wrong.

Features are measured in staff spaces (d) so they are independent of zoom.
"""
import random
import tempfile
from pathlib import Path

import cv2
import numpy as np

from . import symbols

MODEL_PATH = Path(__file__).resolve().parent.parent / "runs" / "symbols_tree" / "tree.joblib"


def features(mask, d=20.0):
    m = (mask > 0).astype(np.uint8)
    h, w = m.shape
    area = float(m.sum())
    ys, xs = np.nonzero(m)
    cy = ys.mean() / h if area else 0.5
    cx = xs.mean() / w if area else 0.5
    cnts, hier = cv2.findContours(m * 255, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    holes = [cv2.contourArea(c) / (d * d) for i, c in enumerate(cnts) if hier is not None and hier[0][i][3] != -1]
    col_runs = []
    for j in range(w):
        col = m[:, j]
        best = cur = 0
        for v in col:
            cur = cur + 1 if v else 0
            best = max(best, cur)
        col_runs.append(best)
    col_runs = np.array(col_runs)
    stem_cols = float((col_runs >= 0.6 * h).sum()) / max(d, 1)
    rows = m.sum(axis=1).astype(float)
    cols = m.sum(axis=0).astype(float)
    rp = np.array([b.sum() for b in np.array_split(rows, 8)]) / max(area, 1)
    cp = np.array([b.sum() for b in np.array_split(cols, 8)]) / max(area, 1)
    return np.array([w / d, h / d, area / (w * h), area / (d * d), len(holes), max(holes) if holes else 0.0,
                     sum(holes), stem_cols, col_runs.max() / h, cy, cx, w / max(h, 1), *rp, *cp], np.float32)


def collect(pages, seed, workdir):
    """Render labelled pages -> (X, y) with extra 'unknown' junk samples."""
    rng = random.Random(seed)
    X, y = [], []
    for i in range(pages):
        for label, mask in symbols.render_page(seed * 100000 + i, workdir):
            X.append(features(mask)); y.append(label)
            if rng.random() < 0.15:
                X.append(features(symbols.half_cut(mask, rng))); y.append("unknown")
        for _ in range(3):
            X.append(features(symbols.junk_crop(rng))); y.append("unknown")
    return np.stack(X), np.array(y)


def train(pages=150, seed=7, out=MODEL_PATH):
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report
    with tempfile.TemporaryDirectory() as wd:
        X, y = collect(pages, seed, wd)
        Xv, yv = collect(max(pages // 5, 10), seed + 1, wd)        # different pages: no leak
    clf = RandomForestClassifier(n_estimators=40, min_samples_leaf=2, max_depth=18, class_weight="balanced", random_state=seed, n_jobs=-1)
    clf.fit(X, y)
    report = classification_report(yv, clf.predict(Xv), zero_division=0, output_dict=True)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, out, compress=9)
    return report


_clf = None


def classify(mask):
    """Return a class name from symbols.CLASSES for one binary crop (ink > 0)."""
    global _clf
    if _clf is None:
        import joblib
        _clf = joblib.load(MODEL_PATH)
    return str(_clf.predict([features(mask)])[0])


if __name__ == "__main__":
    import json
    rep = train()
    keep = {k: round(v["f1-score"], 3) for k, v in rep.items() if isinstance(v, dict) and k not in ("macro avg", "weighted avg")}
    print(json.dumps({"accuracy": round(rep["accuracy"], 3), "f1": keep}, indent=1))
