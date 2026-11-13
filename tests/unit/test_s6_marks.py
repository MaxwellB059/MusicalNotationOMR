"""S6 classical-fallback tests for hairpins and ties, on drawn shapes and real LilyPond renders.

These are the baseline numbers the YOLO detector must beat (gate D in PLAN.md). Homr found
0/4 hairpins and 0/2 ties on our pages.
"""
import pathlib
import tempfile

import cv2
import numpy as np
import pytest

from omr import marks, noteheads, preprocess, staff, synth

HAVE_LILY = pathlib.Path(synth.LILYPOND).exists()
BS = chr(92)


def fake_staff(d=20, top=100):
    return staff.Staff(lines=[top + i * d for i in range(5)], space=float(d), thickness=3.0, x0=0, x1=900)


def drawn_hairpin(opening_right=True, d=20):
    img = np.zeros((300, 900), np.uint8)
    x0, x1, y = 200, 200 + 6 * d, 230
    if opening_right:
        cv2.line(img, (x0, y), (x1, y - d // 2), 255, 3)
        cv2.line(img, (x0, y), (x1, y + d // 2), 255, 3)
    else:
        cv2.line(img, (x0, y - d // 2), (x1, y), 255, 3)
        cv2.line(img, (x0, y + d // 2), (x1, y), 255, 3)
    return img


def test_drawn_crescendo_and_decrescendo_detected_with_direction():
    st = fake_staff()
    c = marks.find_hairpins(drawn_hairpin(True), st)
    dcr = marks.find_hairpins(drawn_hairpin(False), st)
    assert [h["kind"] for h in c] == ["crescendo"]
    assert [h["kind"] for h in dcr] == ["decrescendo"]


def test_no_hairpin_in_blank_beam_or_slab():
    st = fake_staff()
    assert marks.find_hairpins(np.zeros((300, 900), np.uint8), st) == []
    beam = np.zeros((300, 900), np.uint8)
    cv2.rectangle(beam, (200, 220), (320, 232), 255, -1)       # solid slab like a beam
    assert marks.find_hairpins(beam, st) == []


def _render(ly_body, tmp):
    src = (f'{BS}version "2.24.4"\n{BS}paper {{ indent = 0{BS}mm line-width = 200{BS}mm tagline = ##f '
           f'page-breaking = #ly:one-line-breaking }}\n{BS}score {{ {BS}new Staff {{ {BS}omit Staff.TimeSignature '
           f'{ly_body} }} {BS}layout {{}} }}\n')
    d = pathlib.Path(tmp)
    d.mkdir(parents=True, exist_ok=True)
    return synth.render_png(src, d / "m")


@pytest.mark.skipif(not HAVE_LILY, reason="LilyPond not installed")
def test_real_lilypond_hairpins():
    hits = {"crescendo": 0, "decrescendo": 0}
    with tempfile.TemporaryDirectory() as tmp:
        for kind, mark in (("crescendo", BS + "<"), ("decrescendo", BS + ">")):
            for n, pitch in enumerate(("c'", "e'", "g'")):
                png = _render(f"{pitch}1{mark} {pitch}1{BS}!", pathlib.Path(tmp) / f"{kind}{n}")
                bw, _ = preprocess.preprocess(cv2.imread(str(png)))
                st = staff.detect_staves(bw)[0]
                cleaned = staff.remove_staff_lines(bw, [st])
                found = [h["kind"] for h in marks.find_hairpins(cleaned, st)]
                hits[kind] += found == [kind]
    print("hairpin hits", hits)
    assert hits["crescendo"] >= 2 and hits["decrescendo"] >= 2     # of 3 each; homr: 0


@pytest.mark.skipif(not HAVE_LILY, reason="LilyPond not installed")
def test_real_lilypond_ties():
    ok = 0
    total = 4
    with tempfile.TemporaryDirectory() as tmp:
        for n, pitch in enumerate(("c'", "e'", "g'", "b'")):
            png = _render(f"{pitch}2~ {pitch}2 {pitch}1", pathlib.Path(tmp) / f"t{n}")
            bw, _ = preprocess.preprocess(cv2.imread(str(png)))
            st = staff.detect_staves(bw)[0]
            heads = noteheads.find_noteheads(bw, st)
            cleaned = staff.remove_staff_lines(bw, [st])
            ties = marks.find_ties(cleaned, st, heads)
            ok += len(ties) >= 1
    print("tie hits", ok, "/", total)
    assert ok >= 2, f"ties found on only {ok}/{total}"                # homr: 0/2
