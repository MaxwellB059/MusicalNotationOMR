"""CRNN tokens -> MusicXML (no TensorFlow needed, so the API and tests can use it)."""
from . import musicxml, primus

CLEF = {"G2": "treble", "F4": "bass", "C3": "alto", "C4": "tenor"}
DUR_NAME = {"whole": "whole", "half": "half", "quarter": "quarter", "eighth": "eighth", "sixteenth": "16th",
            "thirty_second": "32nd"}


def tokens_to_measures(tokens):
    """Return (measures, clef, fifths, (beats, beat_type), flags). Never raises on a bad sequence:
    malformed fragments are dropped and reported in flags (needs-review), not guessed."""
    flags, clef, fifths, time = [], "treble", 0, None
    measures, cur = [], []
    last_note, pending_tie = None, False
    i = 0
    while i < len(tokens):
        t = tokens[i]
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        if t.startswith("clef-"):
            code = t[5:]
            if code in CLEF:
                clef = CLEF[code]
            else:
                flags.append(f"clef {code} not supported, assuming treble")
            i += 1
        elif t.startswith("keySignature-"):
            f = primus.fifths(t)
            if f is None:
                flags.append(f"unreadable key {t}")
            else:
                fifths = f
            i += 1
        elif t.startswith("timeSignature-"):
            ts = t.split("-", 1)[1]
            time = (4, 4) if ts == "C" else (2, 2) if ts == "C/" else tuple(int(v) for v in ts.split("/"))
            i += 1
        elif t == "tie":
            pending_tie = True                       # applies from the previous note to the next one
            i += 1
        elif t == "barline":
            if cur:
                measures.append(cur)
                cur = []
            i += 1
        elif (t.startswith("n:") or t == "r") and nxt.startswith("d:"):
            dur = nxt[2:]
            dotted = dur.endswith(".")
            name = DUR_NAME.get(dur.rstrip("."))
            if name is None:
                flags.append(f"unsupported duration {dur}")
            else:
                ev = {"type": "rest" if t == "r" else "note", "dur": name}
                if t != "r":
                    ev["pitch"] = t[2:]
                if dotted:
                    ev["dots"] = 1
                if t != "r":
                    if pending_tie and last_note is not None and last_note["pitch"] == ev["pitch"]:
                        last_note["tie_start"] = True
                        ev["tie_stop"] = True
                    elif pending_tie:
                        flags.append("tie between different pitches ignored")
                    pending_tie = False
                    last_note = ev
                cur.append(ev)
            i += 2
        else:
            flags.append(f"dropped stray token {t}")
            i += 1
    if cur:
        measures.append(cur)
    if time is None:
        time = (4, 4)
        flags.append("no time signature read, assuming 4/4")
    return measures, clef, fifths, time, flags


def tokens_to_musicxml(tokens):
    measures, clef, fifths, (beats, beat_type), flags = tokens_to_measures(tokens)
    if not measures:
        raise ValueError("no notes or rests in the predicted sequence")
    return musicxml.build_musicxml(measures, clef, fifths, beats, beat_type, flags=flags)
