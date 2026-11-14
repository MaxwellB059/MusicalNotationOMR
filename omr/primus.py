"""Read PrIMuS / Camera-PrIMuS ground truth (the 'semantic' encoding) into our scorer's format.

Semantic encoding is one line of tab-separated tokens, e.g.
    clef-G2  keySignature-EbM  timeSignature-3/4  note-Bb4_quarter  rest-eighth  barline  note-C#5_half.
We turn it into the same dict that omr.score.extract() returns, so a prediction can be scored
against real, human-curated music with no MusicXML round trip.

Camera-PrIMuS images are the same incipits as PrIMuS, printed then distorted like a camera photo,
so they are the closest thing to 'real images' we can score automatically.
"""
import re
from pathlib import Path

LETTER = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
DUR_QL = {"whole": 4.0, "half": 2.0, "quarter": 1.0, "eighth": 0.5, "sixteenth": 0.25,
          "thirty_second": 0.125, "sixty_fourth": 0.0625, "double_whole": 8.0}
DUR_OURS = {"sixteenth": "16th"}                      # our naming for 16ths
MAJOR_FIFTHS = {"C": 0, "G": 1, "D": 2, "A": 3, "E": 4, "B": 5, "F#": 6, "C#": 7,
                "F": -1, "Bb": -2, "Eb": -3, "Ab": -4, "Db": -5, "Gb": -6, "Cb": -7}
MINOR_FIFTHS = {"A": 0, "E": 1, "B": 2, "F#": 3, "C#": 4, "G#": 5, "D#": 6, "A#": 7,
                "D": -1, "G": -2, "C": -3, "F": -4, "Bb": -5, "Eb": -6, "Ab": -7}
CLEF_CLASS = {"G2": "TrebleClef", "F4": "BassClef", "C3": "AltoClef", "C4": "TenorClef", "C1": "SopranoClef",
              "C2": "MezzoSopranoClef", "C5": "BaritoneClef", "G1": "FrenchViolinClef"}


def midi(pitch):
    m = re.match(r"([A-G])([#b]*)(\d)", pitch)
    if not m:
        return None
    alter = m.group(2).count("#") - m.group(2).count("b")
    return 12 * (int(m.group(3)) + 1) + LETTER[m.group(1)] + alter


def ql(dur_token):
    """'quarter.' -> 1.5, 'half..' -> 3.5, 'quarter' -> 1.0 ; returns None for tuplets/unknown."""
    dots = len(dur_token) - len(dur_token.rstrip("."))
    base = DUR_QL.get(dur_token.rstrip("."))
    if base is None:
        return None
    return base * (2 - 0.5 ** dots)


def fifths(token):
    tonic = token.split("-", 1)[1]
    if tonic.endswith("m"):
        return MINOR_FIFTHS.get(tonic[:-1])
    return MAJOR_FIFTHS.get(tonic[:-1] if tonic.endswith("M") else tonic)


def parse_semantic(text):
    """Return {'events': [(kind, midi|None, quarterLength)], 'clef', 'fifths', 'time', 'multirest': bool}."""
    out = {"events": [], "clef": None, "fifths": None, "time": None, "unsupported": 0}
    for tok in text.strip().split():
        if tok.startswith("clef-"):
            out["clef"] = CLEF_CLASS.get(tok[5:])
        elif tok.startswith("keySignature-"):
            out["fifths"] = fifths(tok)
        elif tok.startswith("timeSignature-"):
            ts = tok.split("-", 1)[1]
            if ts == "C":
                out["time"] = (4, 4)
            elif ts == "C/":
                out["time"] = (2, 2)
            elif "/" in ts:
                n, d = ts.split("/")
                out["time"] = (int(n), int(d))
        elif tok.startswith("note-"):
            pitch, _, dur = tok[5:].partition("_")
            q = ql(dur)
            if q is None or midi(pitch) is None:
                out["unsupported"] += 1
            else:
                out["events"].append(("note", midi(pitch), q))
        elif tok.startswith("rest-"):
            q = ql(tok[5:])
            if q is None:
                out["unsupported"] += 1
            else:
                out["events"].append(("rest", None, q))
        elif tok.startswith("gracenote-") or tok.startswith("multirest-"):
            out["unsupported"] += 1
    return out


def load_sample(sample_dir):
    """sample_dir holds <id>.semantic and one or more PNGs. Returns (truth_dict, png_paths)."""
    d = Path(sample_dir)
    sem = next(d.glob("*.semantic"))
    pngs = sorted(d.glob("*.png"))
    return parse_semantic(sem.read_text(encoding="utf-8")), pngs
