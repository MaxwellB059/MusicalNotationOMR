"""Score the pipeline on real (Camera-)PrIMuS images: clean print vs camera-distorted.

    python tools/eval_primus.py --root ~/datasets/camera_primus/Corpus --n 40 --out benchmarks/primus.json

Samples are drawn with a fixed seed from the corpus. Samples with tokens we can't score (tuplets,
grace notes, fermatas) are skipped, and the skip count is reported.
"""
import argparse
import json
import random
import sys
import traceback
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from omr import assemble, primus, recognize as R, score as sc  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path.home() / "datasets" / "camera_primus" / "Corpus"))
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--out", default="benchmarks/primus.json")
    ap.add_argument("--pipeline", choices=["auto", "geometry"], default="auto",
                    help="auto = trained CRNN if runs/crnn exists, else geometry; geometry forces the hand-built pipeline")
    a = ap.parse_args()

    dirs = sorted(p for p in Path(a.root).iterdir() if p.is_dir())
    random.Random(a.seed).shuffle(dirs)
    res = {"clean": [], "camera": []}
    errors = {"clean": [], "camera": []}
    skipped, used = 0, 0
    for d in dirs:
        if used >= a.n:
            break
        sem = next(d.glob("*.semantic"), None)
        if sem is None:
            continue
        truth = primus.parse_semantic(sem.read_text(encoding="utf-8"))
        if truth["unsupported"] or not truth["events"]:
            skipped += 1
            continue
        used += 1
        imgs = {"clean": next(d.glob("*.png"), None), "camera": next(d.glob("*_distorted.jpg"), None)}
        zero = {k: 0.0 for k in sc.score(truth, truth)}
        for kind, path in imgs.items():
            if path is None:
                continue
            try:
                img = cv2.imread(str(path))
                xml = assemble.recognize_line(img)["xml"] if a.pipeline == "geometry" else R.recognize(img)
                res[kind].append(sc.score(sc.extract(xml), truth))
            except Exception as exc:
                errors[kind].append(f"{d.name}: {type(exc).__name__}: {exc}")
                res[kind].append(dict(zero))
        print(used, d.name, flush=True)
    summary = {k: sc.aggregate(v) for k, v in res.items() if v}
    out = {"n": used, "skipped_unsupported": skipped, "aggregate": summary,
           "errors": {k: v[:15] for k, v in errors.items()}, "error_counts": {k: len(v) for k, v in errors.items()}}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    keys = list(next(iter(summary.values())).keys())
    print("| metric | " + " | ".join(summary) + " |\n|---|" + "---|" * len(summary))
    for k in keys:
        print(f"| {k} | " + " | ".join(f"{summary[m][k]:.3f}" for m in summary) + " |")
    print("errors:", out["error_counts"], "skipped:", skipped)


if __name__ == "__main__":
    main()
