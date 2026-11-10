"""Synthetic one-line score generator (PLAN.md S0).

Each score is generated from a seed: a list of events (notes/rests) -> LilyPond source
(rendered to PNG) and MusicXML (the answer key). Splitting is done by score, never by symbol.
"""
import argparse
import json
import os
import random
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from music21 import clef as m21clef, key as m21key, meter, note, stream

LILYPOND = os.environ.get("LILYPOND_BIN") or shutil.which("lilypond") or str(
    Path.home() / "tools" / "lilypond-2.24.4" / "bin" / "lilypond.exe")

CLEFS = {"treble": ("treble", m21clef.TrebleClef, (62, 84)),
         "bass": ("bass", m21clef.BassClef, (40, 62)),
         "alto": ("alto", m21clef.AltoClef, (53, 74))}
DURS = {"whole": 4.0, "half": 2.0, "quarter": 1.0, "eighth": 0.5, "16th": 0.25}
LY_DUR = {"whole": "1", "half": "2", "quarter": "4", "eighth": "8", "16th": "16"}
TIMES = [(4, 4), (3, 4), (2, 4), (6, 8)]
STEPS = "cdefgab"


def _midi_to_ly(n):
    """Return LilyPond pitch text (sharps only) for a music21 Note."""
    p = n.pitch
    name = p.step.lower() + ("is" * (1 if p.accidental and p.accidental.name == "sharp" else 0)) \
        + ("es" * (1 if p.accidental and p.accidental.name == "flat" else 0))
    return name + ("'" * (p.octave - 3) if p.octave >= 3 else "," * (3 - p.octave))


def make_score(seed, measures=2, clef="treble"):
    """Return (events, meta). events: list of dicts {type, pitch, dur}. Deterministic in seed."""
    rng = random.Random(seed)
    fifths = rng.randint(-3, 3)
    beats, beat_type = rng.choice(TIMES)
    lo, hi = CLEFS[clef][2]
    qlen = beats * 4.0 / beat_type
    events = []
    for _ in range(measures):
        left = qlen
        while left > 1e-9:
            options = [d for d, v in DURS.items() if v <= left + 1e-9]
            d = rng.choice(options)
            if rng.random() < 0.15:
                events.append({"type": "rest", "dur": d})
            else:
                n = note.Note()
                n.pitch.midi = rng.randint(lo, hi)
                events.append({"type": "note", "pitch": n.pitch.nameWithOctave.replace("-", "b"), "dur": d})
            left -= DURS[d]
    meta = {"seed": seed, "clef": clef, "fifths": fifths, "beats": beats, "beat_type": beat_type,
            "measures": measures}
    return events, meta


def to_musicxml(events, meta, path):
    s = stream.Score()
    p = stream.Part()
    p.append(CLEFS[meta["clef"]][1]())
    p.append(m21key.KeySignature(meta["fifths"]))
    p.append(meter.TimeSignature(f"{meta['beats']}/{meta['beat_type']}"))
    for e in events:
        if e["type"] == "rest":
            r = note.Rest()
            r.duration.type = e["dur"]
            p.append(r)
        else:
            n = note.Note(e["pitch"])
            n.duration.type = e["dur"]
            p.append(n)
    p.makeMeasures(inPlace=True)
    s.append(p)
    s.write("musicxml", fp=str(path))


def to_lilypond(events, meta):
    cl = CLEFS[meta["clef"]][0]
    modes = {0: "c", 1: "g", 2: "d", 3: "a", -1: "f", -2: "bes", -3: "ees"}
    tonic = modes[meta["fifths"]]
    body = []
    for e in events:
        d = LY_DUR[e["dur"]]
        if e["type"] == "rest":
            body.append("r" + d)
        else:
            n = note.Note(e["pitch"])
            body.append(_midi_to_ly(n) + d)
    bs = chr(92)  # backslash, built explicitly so the template can't be mangled by escapes
    header = (f'{bs}version "2.24.4"\n'
              f'{bs}paper {{ indent = 0{bs}mm line-width = 160{bs}mm tagline = ##f '
              f'page-breaking = #ly:one-line-breaking }}\n')
    staff = (f'{bs}score {{ {bs}new Staff {{ {bs}clef {cl} {bs}key {tonic} {bs}major '
             f'{bs}time {meta["beats"]}/{meta["beat_type"]} {" ".join(body)} {bs}bar "|." }} '
             f'{bs}layout {{}} }}\n')
    return header + staff


def render_png(ly_text, out_stem, dpi=150):
    out_stem = Path(out_stem)
    # Render in a private temp dir, then copy results in: cloud-synced folders (Dropbox) lock
    # files mid-write, which broke in-place rename. LilyPond also exits non-zero on some
    # Windows setups even when the PNG is fine, so success = "PNG exists", not exit code.
    with tempfile.TemporaryDirectory() as tmp:
        ly = Path(tmp) / "s.ly"
        ly.write_text(ly_text, encoding="utf-8")
        proc = subprocess.run([LILYPOND, "--png", "-dresolution=%d" % dpi, "-dno-point-and-click",
                               "-dcrop", "-o", str(Path(tmp) / "s"), str(ly)],
                              capture_output=True, text=True)
        produced = Path(tmp) / "s.cropped.png"
        if not produced.exists():
            produced = Path(tmp) / "s.png"
        if not produced.exists():
            raise RuntimeError("lilypond failed: " + proc.stderr[-800:])
        png = out_stem.with_suffix(".png")
        _copy_retry(produced, png)
        _copy_retry(ly, out_stem.with_suffix(".ly"))
    return png


def _copy_retry(src, dst, tries=20):
    for i in range(tries):
        try:
            shutil.copyfile(src, dst)
            return
        except PermissionError:
            if i == tries - 1:
                raise
            time.sleep(0.25)


def generate(n, seed, out_dir, measures=2):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    items = []
    for i in range(n):
        sid = seed * 100000 + i
        clef = list(CLEFS)[rng.randrange(3)]
        events, meta = make_score(sid, measures, clef)
        stem = out / f"score_{sid}"
        to_musicxml(events, meta, stem.with_suffix(".musicxml"))
        render_png(to_lilypond(events, meta), stem)
        (stem.with_suffix(".json")).write_text(json.dumps({"meta": meta, "events": events}))
        items.append(stem.name)
    return items


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/synth")
    ap.add_argument("--measures", type=int, default=2)
    a = ap.parse_args()
    print(len(generate(a.n, a.seed, a.out, a.measures)), "scores written to", a.out)
