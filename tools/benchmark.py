"""Benchmark Secundo against homr on the SAME generated lines, scored by omr.score (PLAN.md S0/G0).

    python tools/benchmark.py --n 30 --data data/synth --out benchmarks        # ours + homr
    python tools/benchmark.py --n 30 --no-homr                                 # ours only

homr is AGPL, so it runs as a separate program (HOMR_BIN, default ~/venvs/homr/Scripts/homr.exe),
never imported. Its outputs are cached in <out>/homr/ so reruns are free; delete a file to redo it.
Uses the TEST split of the score-level split, so none of these lines were used to fit anything.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from omr import score as sc, split  # noqa: E402


def run_homr(png, out_xml, homr_bin):
    if Path(out_xml).exists():
        return True
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copy(png, Path(tmp) / "in.png")
        subprocess.run([homr_bin, "in.png"], cwd=tmp, capture_output=True, text=True, timeout=300)
        got = Path(tmp) / "in.musicxml"
        if not got.exists():
            return False
        Path(out_xml).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(got, out_xml)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--data", default="data/synth")
    ap.add_argument("--out", default="benchmarks")
    ap.add_argument("--no-homr", action="store_true")
    ap.add_argument("--no-ours", action="store_true")
    a = ap.parse_args()
    homr_bin = os.environ.get("HOMR_BIN") or str(Path.home() / "venvs" / "homr" / "Scripts" / "homr.exe")

    stems = sorted(p.stem for p in Path(a.data).glob("score_*.png"))
    test = split.split_scores(stems, seed=1)["test"][: a.n]
    zero = None
    results = {"stems": test, "ours": [], "homr": []}
    if not a.no_ours:
        from omr import assemble
    for stem in test:
        truth = Path(a.data) / f"{stem}.musicxml"
        png = Path(a.data) / f"{stem}.png"
        t = sc.extract(truth)
        zero = {k: 0.0 for k in sc.score(t, t)}
        if not a.no_ours:
            try:
                xml = assemble.recognize_line(cv2.imread(str(png)))["xml"]
                results["ours"].append(sc.score(sc.extract(xml), t))
            except Exception:
                results["ours"].append(dict(zero))
        if not a.no_homr:
            out_xml = Path(a.out) / "homr" / f"{stem}.musicxml"
            try:
                ok = run_homr(png, out_xml, homr_bin)
                results["homr"].append(sc.score(sc.extract(out_xml), t) if ok else dict(zero))
            except Exception:
                results["homr"].append(dict(zero))
        print(stem, "done", flush=True)

    agg = {k: sc.aggregate(v) for k, v in results.items() if k != "stems" and v}
    Path(a.out).mkdir(exist_ok=True)
    json.dump({"n": len(test), "stems": test, "aggregate": agg}, open(Path(a.out) / "results.json", "w"), indent=1)
    keys = list(next(iter(agg.values())).keys())
    lines = ["| metric | " + " | ".join(agg) + " |", "|---|" + "---|" * len(agg)]
    lines += [f"| {k} | " + " | ".join(f"{agg[m][k]:.3f}" for m in agg) + " |" for k in keys]
    table = "\n".join(lines)
    (Path(a.out) / "results.md").write_text(f"Benchmark on {len(test)} held-out generated lines\n\n{table}\n", encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()
