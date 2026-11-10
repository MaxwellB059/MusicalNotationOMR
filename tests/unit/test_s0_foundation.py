"""S0 gate tests: generator determinism, answer-key validity, scorer correctness, leak check."""
import json
import shutil

import pytest

from omr import score as sc
from omr import split, synth

HAVE_LILY = bool(shutil.which("lilypond")) or __import__("pathlib").Path(synth.LILYPOND).exists()


def test_make_score_deterministic():
    a = synth.make_score(42, 2, "treble")
    b = synth.make_score(42, 2, "treble")
    assert a == b
    assert synth.make_score(43, 2, "treble") != a


@pytest.mark.parametrize("clef", ["treble", "bass", "alto"])
def test_measures_add_up_to_time_signature(clef):
    for seed in range(30):
        events, meta = synth.make_score(seed, 2, clef)
        total = sum(synth.DURS[e["dur"]] for e in events)
        assert total == pytest.approx(2 * meta["beats"] * 4.0 / meta["beat_type"])


def test_pitches_stay_in_clef_range():
    for clef, (_, _, (lo, hi)) in synth.CLEFS.items():
        from music21 import note
        for seed in range(30):
            events, _ = synth.make_score(seed, 2, clef)
            for e in events:
                if e["type"] == "note":
                    assert lo <= note.Note(e["pitch"]).pitch.midi <= hi


def test_musicxml_roundtrips_and_scorer_is_perfect_on_truth(tmp_path):
    events, meta = synth.make_score(7, 2, "bass")
    p = tmp_path / "s.musicxml"
    synth.to_musicxml(events, meta, p)
    truth = sc.extract(p)
    assert len(truth["events"]) == len(events)
    assert truth["clef"] == "BassClef"
    assert truth["fifths"] == meta["fifths"]
    assert truth["time"] == (meta["beats"], meta["beat_type"])
    m = sc.score(truth, truth)
    assert all(v == 1.0 for v in m.values())


def _mk(events, clef="TrebleClef", fifths=0, time=(4, 4)):
    return {"events": events, "clef": clef, "fifths": fifths, "time": time}


def test_scorer_wrong_pitch():
    t = _mk([("note", 60, 1.0), ("note", 62, 1.0)])
    p = _mk([("note", 60, 1.0), ("note", 64, 1.0)])
    m = sc.score(p, t)
    assert m["pitch_acc"] == 0.5 and m["duration_acc"] == 1.0 and m["exact"] == 0.0


def test_scorer_dropped_and_extra_notes():
    t = _mk([("note", 60, 1.0), ("note", 62, 1.0), ("note", 64, 1.0)])
    dropped = _mk([("note", 60, 1.0), ("note", 64, 1.0)])
    m = sc.score(dropped, t)
    assert m["note_precision"] == 1.0 and m["note_recall"] == pytest.approx(2 / 3)
    extra = _mk(t["events"] + [("note", 65, 1.0)])
    m = sc.score(extra, t)
    assert m["note_recall"] == 1.0 and m["note_precision"] == pytest.approx(3 / 4)


def test_scorer_header_and_rest():
    t = _mk([("rest", None, 4.0)], clef="BassClef", fifths=2, time=(3, 4))
    p = _mk([("note", 60, 4.0)], clef="TrebleClef", fifths=2, time=(4, 4))
    m = sc.score(p, t)
    assert m["clef_ok"] == 0.0 and m["key_ok"] == 1.0 and m["time_ok"] == 0.0
    assert m["rest_recall"] == 0.0


def test_split_is_by_score_and_deterministic():
    stems = [f"s{i}" for i in range(100)]
    a = split.split_scores(stems, seed=1)
    assert a == split.split_scores(stems, seed=1)
    all_ = a["train"] + a["val"] + a["test"]
    assert sorted(all_) == sorted(stems) and len(set(all_)) == 100


def test_leak_check_catches_duplicate_images(tmp_path):
    (tmp_path / "a.png").write_bytes(b"same")
    (tmp_path / "b.png").write_bytes(b"same")
    (tmp_path / "c.png").write_bytes(b"different")
    with pytest.raises(AssertionError):
        split.assert_no_leak({"train": ["a"], "val": ["b"]}, tmp_path)
    assert split.assert_no_leak({"train": ["a"], "val": ["c"]}, tmp_path)


@pytest.mark.skipif(not HAVE_LILY, reason="LilyPond not installed")
def test_render_end_to_end_small(tmp_path):
    items = synth.generate(3, seed=5, out_dir=tmp_path)
    assert len(items) == 3
    import cv2
    for stem in items:
        img = cv2.imread(str(tmp_path / (stem + ".png")), 0)
        assert img is not None and img.shape[0] > 30 and img.min() < 128
        assert sc.extract(tmp_path / (stem + ".musicxml"))["events"]
        meta = json.loads((tmp_path / (stem + ".json")).read_text())["meta"]
        assert meta["clef"] in synth.CLEFS
    # different scores must render to different images (no accidental duplicates)
    splits = {"train": items[:2], "val": items[2:]}
    assert split.assert_no_leak(splits, tmp_path)
