"""S7 tests: MusicXML builder (exact), then end-to-end line scoring with the interim classifier."""
import json
import pathlib

import cv2
import pytest

from omr import musicxml, score as sc, synth

HAVE_LILY = pathlib.Path(synth.LILYPOND).exists()
TREE = pathlib.Path(__file__).resolve().parents[2] / "runs" / "symbols_tree" / "tree.joblib"


def test_builder_roundtrips_through_music21_exactly(tmp_path):
    measures = [[{"type": "note", "pitch": "C#4", "dur": "quarter"}, {"type": "rest", "dur": "quarter"},
                 {"type": "note", "pitch": "Bb3", "dur": "half"}],
                [{"type": "note", "pitch": "G5", "dur": "whole"}]]
    p = tmp_path / "a.musicxml"
    p.write_text(musicxml.build_musicxml(measures, clef="bass", fifths=-2, beats=4, beat_type=4))
    t = sc.extract(p)
    assert t["clef"] == "BassClef" and t["fifths"] == -2 and t["time"] == (4, 4)
    assert t["events"] == [("note", 61, 1.0), ("rest", None, 1.0), ("note", 58, 2.0), ("note", 79, 4.0)]


def test_builder_marks_ties_articulations_dynamics_hairpins(tmp_path):
    ev = [{"type": "note", "pitch": "C4", "dur": "half", "tie_start": True, "dynamic": "mf",
           "articulations": ["accent"], "hairpin": "crescendo"},
          {"type": "note", "pitch": "C4", "dur": "half", "tie_stop": True, "hairpin": "stop"}]
    xml = musicxml.build_musicxml([ev])
    for needle in ('<tie type="start"', '<tie type="stop"', "<accent", "<mf", 'wedge type="crescendo"',
                   'wedge type="stop"'):
        assert needle in xml, needle
    p = tmp_path / "m.musicxml"
    p.write_text(xml)
    assert len(sc.extract(p)["events"]) == 2          # music21 still parses it


def test_measure_arithmetic_and_flags_in_output():
    ev = [{"type": "note", "pitch": "C4", "dur": "quarter"}] * 3
    assert musicxml.measure_units(ev) == 12 and musicxml.expected_units(4, 4) == 16
    xml = musicxml.build_musicxml([ev], flags=["measure 1 short"])
    assert "needs-review" in xml


@pytest.mark.skipif(not (HAVE_LILY and TREE.exists()), reason="needs LilyPond and the interim classifier")
def test_end_to_end_line_scoring(tmp_path):
    """Measured, not gated yet: records the first honest end-to-end numbers (PLAN.md G7)."""
    from omr import assemble
    items = synth.generate(20, seed=31, out_dir=tmp_path)
    rows, errors = [], 0
    for stem in items:
        try:
            out = assemble.recognize_line(cv2.imread(str(tmp_path / (stem + ".png"))))
            (tmp_path / (stem + ".pred.musicxml")).write_text(out["xml"])
            rows.append(sc.score_files(tmp_path / (stem + ".pred.musicxml"), tmp_path / (stem + ".musicxml")))
        except Exception as exc:                       # a crash counts as a zero line, never hidden
            errors += 1
            rows.append({k: 0.0 for k in sc.score(sc.extract(tmp_path / (stem + ".musicxml")),
                                                   sc.extract(tmp_path / (stem + ".musicxml")))})
    agg = sc.aggregate(rows)
    print("END-TO-END", json.dumps({k: round(v, 3) for k, v in agg.items()}), "crashes:", errors)
    assert errors <= len(items) // 2, "pipeline crashes on more than half of the lines"
    assert agg["note_f1"] > 0.1                        # sanity floor only; real gate G7 set by the team
