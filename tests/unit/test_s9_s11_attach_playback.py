"""S9 (attach rules) and S11 (playback) tests."""
from omr import attach, musicxml, playback


def ev(t, x, dur="quarter", pitch="C4"):
    e = {"type": t, "dur": dur, "x": x}
    if t == "note":
        e["pitch"] = pitch
    return e


def measures():
    return [[ev("note", 100, pitch="C4"), ev("note", 200, pitch="C4"), ev("rest", 300), ev("note", 400, pitch="D4")],
            [ev("note", 500, pitch="E4"), ev("note", 600, pitch="E4")]]


D = 20


def test_tie_links_adjacent_same_pitch_notes_only():
    m = measures()
    assert attach.attach_ties(m, [{"x0": 105, "x1": 198}], D) == 1
    assert m[0][0]["tie_start"] and m[0][1]["tie_stop"]
    m2 = measures()
    assert attach.attach_ties(m2, [{"x0": 205, "x1": 405}], D) == 0          # C4 -> D4: not a tie
    assert attach.attach_ties(m2, [{"x0": 105, "x1": 405}], D) == 0          # not adjacent
    assert attach.attach_ties(m2, [{"x0": 505, "x1": 595}], D) == 1          # across the measure only if adjacent notes


def test_hairpin_spans_first_to_last_event_in_range():
    m = measures()
    assert attach.attach_hairpins(m, [{"kind": "crescendo", "x0": 150, "x1": 420}], D) == 1
    assert m[0][1]["hairpin"] == "crescendo" and m[0][3]["hairpin"] == "stop"
    assert "hairpin" not in m[0][0]


def test_dynamic_attaches_to_note_at_or_after_its_x():
    m = measures()
    assert attach.attach_dynamics(m, [{"label": "mf", "x": 195}], D) == 1
    assert m[0][1]["dynamic"] == "mf"
    assert attach.attach_dynamics(measures(), [{"label": "f", "x": 5000}], D) == 0   # nothing near: no guess


def test_articulations_go_to_nearest_note_not_rests():
    m = measures()
    assert attach.attach_articulations(m, [{"kind": "staccato", "x": 305}], D) == 0   # only a rest nearby
    assert attach.attach_articulations(m, [{"kind": "accent", "x": 402}], D) == 1
    assert m[0][3]["articulations"] == ["accent"]


def test_attached_marks_survive_into_musicxml():
    m = measures()
    attach.attach_ties(m, [{"x0": 105, "x1": 198}], D)
    attach.attach_dynamics(m, [{"label": "p", "x": 100}], D)
    xml = musicxml.build_musicxml(m)
    assert '<tie type="start"' in xml and "<p/>" in xml


def test_playback_midi_numbers_are_exact(tmp_path):
    xml = musicxml.build_musicxml([[{"type": "note", "pitch": "C#4", "dur": "quarter"},
                                    {"type": "note", "pitch": "Db4", "dur": "quarter"},
                                    {"type": "rest", "dur": "half"}]])
    got = playback.notes_from_musicxml(xml)
    assert got == [(61, 1.0), (61, 1.0), (None, 2.0)]                   # C#4 == Db4 == 61
    assert sum(q for _, q in got) == 4.0                               # duration equals the measure
    out = playback.musicxml_to_midi(xml, tmp_path / "a.mid")
    assert out.exists() and out.stat().st_size > 20
