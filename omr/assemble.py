"""Stage S7: assemble one line of music from an image into MusicXML (PLAN.md).

preprocess -> staff -> components -> classify each symbol -> measures at barlines ->
pitch from notehead position + clef + key -> MusicXML, with needs-review flags instead of silent
guesses. The classifier is injected: the CNN (omr.train_symbols) when trained, otherwise the
interim shape-feature forest (omr.classify_tree).
"""
import cv2
import numpy as np

from . import musicxml, noteheads, preprocess, semantics, staff, symbols


def default_classifier():
    from . import classify_tree
    return classify_tree.classify


def _heads_in(comp, heads, d):
    x0, y0, w, h = comp[0], comp[1], comp[2], comp[3]
    return [hd for hd in heads if x0 - 0.3 * d <= hd["x"] <= x0 + w + 0.3 * d
            and y0 - 0.3 * d <= hd["y"] <= y0 + h + 0.3 * d]


def recognize_line(img_bgr, classify=None):
    """Return {xml, measures, clef, fifths, beats, beat_type, flags}."""
    classify = classify or default_classifier()
    flags = []
    bw, info = preprocess.preprocess(img_bgr)
    sts = staff.detect_staves(bw)
    if not sts:
        raise ValueError("No staff found in image")
    if len(sts) > 1:
        flags.append(f"{len(sts)} staves found; only the first was read (page mode is S10)")
    st = sts[0]
    d = st.space
    comps = symbols.components(bw, st)
    heads = noteheads.find_noteheads(bw, st)

    # clef: the leftmost component
    clef = "treble"
    if comps and comps[0][3] >= 3 * d:
        lab = classify(comps[0][4])
        clef = lab.replace("clef_", "") if lab.startswith("clef_") else "treble"
        if not lab.startswith("clef_"):
            flags.append(f"first symbol classified {lab}, assuming treble clef")
        comps = comps[1:]
    else:
        flags.append("no clef found, assuming treble")

    labelled = [(c, "barline" if symbols.is_barline(c, d) else classify(c[4])) for c in comps]
    first_music = next((i for i, (c, l) in enumerate(labelled) if l.startswith(("note_", "rest_"))), None)
    if first_music is None:
        raise ValueError("No notes or rests found")
    x_first = labelled[first_music][0][0]
    header = [(c, l) for c, l in labelled if c[0] < x_first - 2.5 * d]
    body = [(c, l) for c, l in labelled if c[0] >= x_first - 2.5 * d]
    sharps = sum(1 for _, l in header if l == "sharp")
    flats = sum(1 for _, l in header if l == "flat")
    fifths = sharps if sharps >= flats else -flats

    state = semantics.MeasureState(fifths, clef)
    measures, cur, pending_acc = [], [], None
    used = set()
    for c, lab in body:
        if lab == "barline":
            if cur:
                measures.append(cur)
                cur = []
            state.new_measure()
            pending_acc = None
        elif lab in ("sharp", "flat", "natural"):
            pending_acc = lab
        elif lab == "beamed_group":
            hs = sorted([h for h in _heads_in(c, heads, d) if id(h) not in used], key=lambda h: h["x"])
            if len(hs) < 2:
                flags.append(f"beamed group at x={c[0]} has {len(hs)} noteheads; skipped")
                continue
            # beams: count ink bands in the column band between the first two stems
            x0 = int(hs[0]["x"] - c[0] + 0.7 * d)
            x1 = int(hs[1]["x"] - c[0] - 0.7 * d)
            col = c[4][:, max(x0, 0):max(x1, x0 + 1)]
            profile = (col.sum(axis=1) > 0.6 * 255 * max(col.shape[1], 1)).astype(int)
            bands = int(((np.diff(np.concatenate([[0], profile])) == 1)).sum()) if profile.any() else 1
            dur = "16th" if bands >= 2 else "eighth"
            flags.append(f"beamed group at x={c[0]}: {len(hs)} notes read as {dur} ({bands} beam band(s))")
            for h in hs:
                used.add(id(h))
                name, _ = state.resolve(h["position"], pending_acc)
                pending_acc = None
                cur.append({"type": "note", "pitch": name, "dur": dur})
        elif lab.startswith("rest_"):
            cur.append({"type": "rest", "dur": lab[5:]})
        elif lab.startswith("note_"):
            hs = [h for h in _heads_in(c, heads, d) if id(h) not in used]
            if not hs:
                flags.append(f"{lab} at x={c[0]} has no notehead; skipped")
                continue
            for h in hs:
                used.add(id(h))
                name, _ = state.resolve(h["position"], pending_acc)
                pending_acc = None
                cur.append({"type": "note", "pitch": name, "dur": lab[5:]})
    if cur:
        measures.append(cur)

    # time signature is read from measure content for now (digits aren't classified yet)
    first = measures[0] if measures else []
    units = musicxml.measure_units(first)
    beats, beat_type = (max(units // 4, 1), 4)
    flags.append("time signature inferred from first measure length (digit reader pending)")
    for i, m in enumerate(measures, 1):
        if musicxml.measure_units(m) != musicxml.expected_units(beats, beat_type):
            flags.append(f"measure {i} lasts {musicxml.measure_units(m)}/16 but {beats}/{beat_type} expects "
                         f"{musicxml.expected_units(beats, beat_type)}/16")
    xml = musicxml.build_musicxml(measures, clef, fifths, beats, beat_type, flags=flags)
    return {"xml": xml, "measures": measures, "clef": clef, "fifths": fifths, "beats": beats,
            "beat_type": beat_type, "flags": flags}
