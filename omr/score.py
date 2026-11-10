"""Scorer: compare predicted MusicXML to ground-truth MusicXML (PLAN.md S0).

Reports per-feature numbers the same way the homr baseline table in A2_REPORT.md does:
pitch, duration, rests, clef, key, time, plus note-level precision/recall/F1.
"""
import argparse
import json
from difflib import SequenceMatcher
from pathlib import Path

from music21 import clef as m21clef, converter, key as m21key, meter, note


def extract(path_or_text):
    """Return {'events': [(kind, midi_or_None, quarterLength)], 'clef', 'fifths', 'time'}."""
    src = str(path_or_text)
    if src.lstrip().startswith("<"):   # raw MusicXML text (e.g. API response)
        s = converter.parse(src, format="musicxml")
    else:                              # file path
        s = converter.parse(src)
    flat = s.flatten()
    events = []
    for el in flat.notesAndRests:
        if isinstance(el, note.Rest):
            events.append(("rest", None, float(el.quarterLength)))
        elif isinstance(el, note.Note):
            events.append(("note", el.pitch.midi, float(el.quarterLength)))
    clefs = list(flat.getElementsByClass(m21clef.Clef))
    keys = list(flat.getElementsByClass(m21key.KeySignature))
    times = list(flat.getElementsByClass(meter.TimeSignature))
    return {
        "events": events,
        "clef": type(clefs[0]).__name__ if clefs else None,
        "fifths": keys[0].sharps if keys else None,
        "time": (times[0].numerator, times[0].denominator) if times else None,
    }


def _prf(tp, n_pred, n_true):
    p = tp / n_pred if n_pred else (1.0 if n_true == 0 else 0.0)
    r = tp / n_true if n_true else (1.0 if n_pred == 0 else 0.0)
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def score(pred, truth):
    """Compare two extract() dicts. Returns a flat metrics dict (all values 0..1)."""
    pe, te = pred["events"], truth["events"]
    sm = SequenceMatcher(a=pe, b=te, autojunk=False)
    matched = sum(b.size for b in sm.get_matching_blocks())
    p, r, f = _prf(matched, len(pe), len(te))
    # position-aligned accuracies over the overlapping prefix length
    n = min(len(pe), len(te))
    denom = max(len(te), 1)
    pitch_ok = sum(1 for i in range(n) if pe[i][0] == te[i][0] and pe[i][1] == te[i][1])
    dur_ok = sum(1 for i in range(n) if abs(pe[i][2] - te[i][2]) < 1e-6)
    rest_true = [i for i, e in enumerate(te) if e[0] == "rest"]
    rest_ok = sum(1 for i in rest_true if i < len(pe) and pe[i][0] == "rest")
    return {
        "note_precision": p, "note_recall": r, "note_f1": f,
        "pitch_acc": pitch_ok / denom,
        "duration_acc": dur_ok / denom,
        "rest_recall": rest_ok / len(rest_true) if rest_true else 1.0,
        "clef_ok": float(pred["clef"] == truth["clef"]),
        "key_ok": float(pred["fifths"] == truth["fifths"]),
        "time_ok": float(pred["time"] == truth["time"]),
        "exact": float(pe == te and pred["clef"] == truth["clef"]
                       and pred["fifths"] == truth["fifths"] and pred["time"] == truth["time"]),
    }


def score_files(pred_path, truth_path):
    return score(extract(pred_path), extract(truth_path))


def aggregate(rows):
    keys = rows[0].keys() if rows else []
    return {k: sum(r[k] for r in rows) / len(rows) for k in keys}


def format_table(agg):
    lines = ["| metric | value |", "|---|---|"]
    lines += [f"| {k} | {v:.3f} |" for k, v in agg.items()]
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Score predicted MusicXML dir against truth dir (matching stems).")
    ap.add_argument("--pred", required=True)
    ap.add_argument("--truth", required=True)
    a = ap.parse_args()
    rows = []
    for t in sorted(Path(a.truth).glob("*.musicxml")):
        p = Path(a.pred) / t.name
        rows.append(score_files(p, t) if p.exists() else
                    {k: 0.0 for k in score(extract(t), extract(t))})
    print(format_table(aggregate(rows)))
