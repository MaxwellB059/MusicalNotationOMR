"""PrIMuS semantic parser: exact on hand-written examples, round-trips through the scorer."""
from omr import primus, score as sc


def test_parse_basic_line():
    t = primus.parse_semantic("clef-G2\tkeySignature-EbM\ttimeSignature-3/4\tnote-Bb4_quarter\trest-eighth\t"
                              "barline\tnote-C#5_half.\tnote-D4_sixteenth")
    assert t["clef"] == "TrebleClef" and t["fifths"] == -3 and t["time"] == (3, 4)
    assert t["events"] == [("note", 70, 1.0), ("rest", None, 0.5), ("note", 73, 3.0), ("note", 62, 0.25)]
    assert t["unsupported"] == 0


def test_keys_and_clefs_and_common_time():
    assert primus.parse_semantic("keySignature-DM")["fifths"] == 2
    assert primus.parse_semantic("keySignature-Am")["fifths"] == 0
    assert primus.parse_semantic("keySignature-F#m")["fifths"] == 3
    assert primus.parse_semantic("clef-F4")["clef"] == "BassClef"
    assert primus.parse_semantic("clef-C3")["clef"] == "AltoClef"
    assert primus.parse_semantic("timeSignature-C")["time"] == (4, 4)


def test_unsupported_tokens_are_counted_not_guessed():
    t = primus.parse_semantic("note-C4_quarter_fermata gracenote-D4_eighth multirest-12 note-C4_quarter")
    assert t["unsupported"] >= 2


def test_parsed_truth_scores_perfectly_against_itself():
    t = primus.parse_semantic("clef-G2 keySignature-CM timeSignature-4/4 note-C4_quarter rest-quarter note-E4_half")
    assert all(v == 1.0 for v in sc.score(t, t).values())
