"""End-to-end geometric pitch on REAL LilyPond renders, scored against the MusicXML answer key.

This is the slides' next question: can staff geometry + rules recover pitch on images the CNN
never saw? Accidentals and rhythm are out of scope here (S4/S7); we compare diatonic letter+octave.
"""
import pathlib

import cv2
import pytest
from music21 import note as m21note

from omr import noteheads, preprocess, semantics, staff, synth

HAVE_LILY = pathlib.Path(synth.LILYPOND).exists()
pytestmark = pytest.mark.skipif(not HAVE_LILY, reason="LilyPond not installed")


def run_one(path, truth_events, clef):
    img = cv2.imread(str(path))
    bw, _ = preprocess.preprocess(img)
    bw, _ = staff.normalize_scale(bw)
    st = staff.detect_staves(bw)
    if len(st) != 1:
        return None
    heads = noteheads.find_noteheads(bw, st[0])
    truth = [e for e in truth_events if e["type"] == "note"]
    got = [semantics.position_to_letter_octave(h["position"], clef) for h in heads]
    want = []
    for e in truth:
        p = m21note.Note(e["pitch"]).pitch
        want.append((p.step, p.octave))
    return got, want


@pytest.mark.xfail(reason="G3 NOT MET YET: first-pass notehead detector matches note counts on ~10/25 "
                   "lines (clef curls, accidentals, beams, hollow heads confuse it). Track progress here; "
                   "remove xfail when the gate passes.", strict=False)
def test_pitch_geometry_on_generated_lines(tmp_path):
    import json
    items = synth.generate(25, seed=21, out_dir=tmp_path)
    exact, total, count_ok = 0, 0, 0
    for stem in items:
        meta = json.loads((tmp_path / (stem + ".json")).read_text())
        r = run_one(tmp_path / (stem + ".png"), meta["events"], meta["meta"]["clef"])
        assert r is not None, f"{stem}: staff not found"
        got, want = r
        count_ok += len(got) == len(want)
        if len(got) == len(want):
            total += len(want)
            exact += sum(1 for g, w in zip(got, want) if g == w)
    print(f"count matched on {count_ok}/{len(items)} lines; pitch {exact}/{total}")
    assert count_ok / len(items) >= 0.8, "noteheads not found reliably"
    assert exact / max(total, 1) >= 0.95, f"geometric pitch only {exact}/{total}"
