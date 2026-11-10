"""Stage S3: pitch from staff position (PLAN.md): pitch = lookup(position, clef) + key signature.

Staff position is an integer: 0 = bottom line, 1 = first space, 2 = second line, ... 8 = top line;
negative values and values above 8 are below/above the staff (ledger-line territory).
No training data is needed for any pitch or clef: another clef is just another table entry.
"""
LETTERS = "CDEFGAB"
SEMITONE = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
SHARP_ORDER = "FCGDAEB"
FLAT_ORDER = "BEADGCF"

# (letter, octave) of the BOTTOM staff line for each clef
CLEF_BOTTOM_LINE = {
    "treble": ("E", 4),
    "bass": ("G", 2),
    "alto": ("F", 3),   # alto clef: middle line is C4, so the bottom line is F3
    "tenor": ("D", 3),
}


def diatonic_index(letter, octave):
    return octave * 7 + LETTERS.index(letter)


def position_to_letter_octave(position, clef="treble"):
    letter, octave = CLEF_BOTTOM_LINE[clef]
    idx = diatonic_index(letter, octave) + int(position)
    return LETTERS[idx % 7], idx // 7


def key_alterations(fifths):
    """Map letter -> alteration (+1/-1) for a key signature given as number of sharps (+) or flats (-)."""
    if not -7 <= fifths <= 7:
        raise ValueError(f"fifths out of range: {fifths}")
    if fifths >= 0:
        return {l: 1 for l in SHARP_ORDER[:fifths]}
    return {l: -1 for l in FLAT_ORDER[:-fifths]}


def to_midi(letter, octave, alter=0):
    return 12 * (octave + 1) + SEMITONE[letter] + alter


def pitch_name(letter, octave, alter=0):
    acc = {1: "#", -1: "b", 0: ""}[alter]
    return f"{letter}{acc}{octave}"


class MeasureState:
    """Carries accidentals through a measure: an accidental on a pitch applies to later notes of the
    same letter and octave until the barline. Call `new_measure()` at each barline."""

    def __init__(self, fifths=0, clef="treble"):
        self.clef = clef
        self.key = key_alterations(fifths)
        self.carried = {}

    def new_measure(self):
        self.carried = {}

    def resolve(self, position, accidental=None):
        """accidental in {None, 'sharp', 'flat', 'natural'}. Returns (name, midi)."""
        letter, octave = position_to_letter_octave(position, self.clef)
        if accidental is not None:
            alter = {"sharp": 1, "flat": -1, "natural": 0}[accidental]
            self.carried[(letter, octave)] = alter
        else:
            alter = self.carried.get((letter, octave), self.key.get(letter, 0))
        return pitch_name(letter, octave, alter), to_midi(letter, octave, alter)


def y_to_position(y, staff):
    """Nearest staff position for a notehead centre at pixel row y (works above/below the staff)."""
    return int(round((staff.bottom - y) / (staff.space / 2.0)))
