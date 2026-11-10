"""S3 gate: geometric pitch is exact for every clef, key, ledger position and accidental.

Cross-checked against music21 (an independent implementation) rather than against our own tables.
"""
import numpy as np
import pytest
from music21 import clef as m21clef, key as m21key, pitch as m21pitch

from omr import semantics as sem
from omr.staff import Staff

CLEF_CLASSES = {"treble": m21clef.TrebleClef, "bass": m21clef.BassClef, "alto": m21clef.AltoClef,
                "tenor": m21clef.TenorClef}


@pytest.mark.parametrize("name", CLEF_CLASSES)
def test_bottom_line_matches_music21(name):
    letter, octave = sem.CLEF_BOTTOM_LINE[name]
    assert CLEF_CLASSES[name]().lowestLine == sem.diatonic_index(letter, octave) + 1


@pytest.mark.parametrize("clef", CLEF_CLASSES)
def test_every_position_including_ledger_lines(clef):
    m21_base = CLEF_CLASSES[clef]().lowestLine           # diatonic number of the bottom line
    for pos in range(-10, 19):                           # ~10 ledger positions each way
        letter, octave = sem.position_to_letter_octave(pos, clef)
        p = m21pitch.Pitch()
        p.diatonicNoteNum = m21_base + pos
        assert (letter, octave) == (p.step, p.octave), (clef, pos)


@pytest.mark.parametrize("fifths", range(-7, 8))
def test_key_signature_matches_music21(fifths):
    ks = m21key.KeySignature(fifths)
    mine = sem.key_alterations(fifths)
    for letter in sem.LETTERS:
        theirs = ks.accidentalByStep(letter)
        assert mine.get(letter, 0) == (theirs.alter if theirs else 0), (fifths, letter)


def test_midi_numbers_match_music21():
    for letter in sem.LETTERS:
        for octave in range(0, 8):
            for alter in (-1, 0, 1):
                name = sem.pitch_name(letter, octave, alter)
                assert sem.to_midi(letter, octave, alter) == m21pitch.Pitch(
                    name.replace("b", "-") if alter == -1 else name).midi


def test_known_treble_examples_from_proposal():
    # PROPOSAL_staff_position_pitch.md: -1 = D4, 1 = F4, 2 = G4, 3 = A4, 4 = B4, 10 = A5, 12 = C6
    st = sem.MeasureState(0, "treble")
    for pos, want in {-1: "D4", 1: "F4", 2: "G4", 3: "A4", 4: "B4", 10: "A5", 12: "C6"}.items():
        assert st.resolve(pos)[0] == want


def test_the_seven_original_cnn_pitches_are_covered_and_unseen_ones_too():
    st = sem.MeasureState(0, "treble")
    seen = {"A4", "A5", "B4", "C6", "D4", "F4", "G4"}
    produced = {st.resolve(p)[0] for p in range(-1, 13)}
    assert seen <= produced and len(produced - seen) >= 6   # works for pitches the CNN never saw


def test_key_signature_applies_and_c_sharp_is_d_flat_in_midi():
    st = sem.MeasureState(2, "treble")          # D major: F# and C#
    assert st.resolve(1)[0] == "F#4" and st.resolve(5)[0] == "C#5"
    assert sem.to_midi("C", 4, 1) == sem.to_midi("D", 4, -1) == 61


def test_accidentals_carry_within_measure_and_reset_at_barline():
    st = sem.MeasureState(0, "treble")
    assert st.resolve(1)[0] == "F4"
    assert st.resolve(1, "sharp")[0] == "F#4"
    assert st.resolve(1)[0] == "F#4"            # carried
    assert st.resolve(8)[0] == "F5"             # other octave unaffected
    assert st.resolve(1, "natural")[0] == "F4"
    st.new_measure()
    st = sem.MeasureState(1, "treble")           # G major: F# by key
    st.resolve(1, "natural")
    assert st.resolve(1)[0] == "F4"
    st.new_measure()
    assert st.resolve(1)[0] == "F#4"            # back to key signature after barline


def test_y_to_position_roundtrip_with_noise():
    staff = Staff(lines=[100 + 12 * i for i in range(5)], space=12.0, thickness=2.0)
    rng = np.random.default_rng(0)
    for pos in range(-8, 17):
        y = staff.position_to_y(pos)
        assert sem.y_to_position(y, staff) == pos
        assert sem.y_to_position(y + rng.uniform(-1.5, 1.5), staff) == pos   # < quarter space
    assert sem.y_to_position(staff.bottom + 12 * 3, staff) == -6            # 3 spaces below = 6 steps
