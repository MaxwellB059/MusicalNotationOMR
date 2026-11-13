"""Symbol dataset + crop utilities for the symbol CNN (PLAN.md S4).

The CNN answers "what IS this symbol", never "what pitch". Training crops come from rendered
LilyPond pages in which every measure holds one known symbol, so each connected component's label
is known by construction (no hand labelling). Pages are split by seed, never by crop.

Classes (v1): 5 note types, 5 rest types, sharp/flat/natural, 3 clefs, barline, and `unknown`
(junk, text, half-cut symbols) so the model can refuse instead of guessing.
"""
import argparse
import hashlib
import random
from pathlib import Path

import cv2
import numpy as np

from . import preprocess, staff, synth

NOTE_TYPES = ["whole", "half", "quarter", "eighth", "16th"]
CLASSES = ([f"note_{d}" for d in NOTE_TYPES] + [f"rest_{d}" for d in NOTE_TYPES] +
           ["sharp", "flat", "natural", "clef_treble", "clef_bass", "clef_alto", "barline", "unknown"])
IMG = 64
BS = chr(92)  # backslash, built explicitly so templates can't be mangled by escapes
UNITS = {"whole": 16, "half": 8, "quarter": 4, "eighth": 2, "16th": 1}   # in 16th notes


def _ly_pitch(midi_name):
    from music21 import note
    return synth._midi_to_ly(note.Note(midi_name))


def _fill(units_used):
    rest = 16 - units_used
    return f" s16*{rest}" if rest > 0 else ""


def make_cells(rng, clef):
    """Return list of (ly_text_for_one_4/4_measure, expected_component_labels_left_to_right)."""
    from music21 import note
    lo, hi = synth.CLEFS[clef][2]

    def rand_pitch(acc=None):
        n = note.Note()
        n.pitch.midi = rng.randint(lo + 2, hi - 2)
        name = n.pitch.name.replace("#", "").replace("-", "")   # natural letter only
        n2 = note.Note(f"{name}{n.pitch.octave}")
        return n2, synth._midi_to_ly(n2)

    cells = []
    for d in NOTE_TYPES:                                    # single notes
        _, ly = rand_pitch()
        cells.append((f"{ly}{synth.LY_DUR[d]}{_fill(UNITS[d])}", [f"note_{d}"]))
    for d in NOTE_TYPES:                                    # single rests
        cells.append((f"r{synth.LY_DUR[d]}{_fill(UNITS[d])}", [f"rest_{d}"]))
    n, ly = rand_pitch()
    base = ly[0]
    tail = ly[1:]
    cells.append((f"{base}is{tail}4{_fill(4)}", ["sharp", "note_quarter"]))
    cells.append((f"{base}es{tail}4{_fill(4)}", ["flat", "note_quarter"]))
    cells.append((f"{base}is{tail}4 {base}{tail}4 {_fill(8)}".replace("  ", " "),
                  ["sharp", "note_quarter", "natural", "note_quarter"]))
    return cells


def page_ly(cells, clef):
    cl = synth.CLEFS[clef][0]
    body = " ".join("{ " + c[0] + " } " + BS + 'bar "|"' for c in cells)
    return (f'{BS}version "2.24.4"\n'
            f'{BS}paper {{ indent = 0{BS}mm line-width = 400{BS}mm tagline = ##f '
            f'page-breaking = #ly:one-line-breaking }}\n'
            f'{BS}score {{ {BS}new Staff {{ {BS}clef {cl} {BS}omit Staff.TimeSignature '
            f'{BS}time 4/4 {body} {BS}bar "" }} {BS}layout {{}} }}\n')


def components(bw, st):
    """Connected components of the staff-line-free page, left to right: list of (x, y, w, h, mask)."""
    cleaned = staff.remove_staff_lines(bw, [st])
    n, lab, stats, _ = cv2.connectedComponentsWithStats(cleaned)
    d = st.space
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 0.12 * d * d:
            continue
        out.append((x, y, w, h, (lab[y:y + h, x:x + w] == i).astype(np.uint8) * 255))
    return merge_fragments(sorted(out, key=lambda c: c[0]), d)


def merge_fragments(comps, d, grow=0.12):
    """Staff-line removal can cut a symbol in pieces (hollow heads, rest bars, flat bowls).
    Re-join components whose boxes touch (within `grow` staff spaces). Barlines never merge."""
    g = grow * d
    n = len(comps)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def touches(a, b):
        # pieces stacked in the same columns (hollow head halves, rest bars): allow a small vertical gap
        xo = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
        yo = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
        if xo >= 0.4 * min(a[2], b[2]) and yo >= -g:
            return True
        # side-by-side pieces must really touch (a flat's bowl vs its stem); a gap means two symbols
        return xo >= -0.1 * d and yo >= 0.3 * min(a[3], b[3])

    for i in range(n):
        for j in range(i + 1, n):
            if is_barline(comps[i], d) or is_barline(comps[j], d):
                continue
            if touches(comps[i], comps[j]):
                parent[find(i)] = find(j)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(comps[i])
    merged = []
    for grp in groups.values():
        x0 = min(c[0] for c in grp); y0 = min(c[1] for c in grp)
        x1 = max(c[0] + c[2] for c in grp); y1 = max(c[1] + c[3] for c in grp)
        mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
        for c in grp:
            mask[c[1] - y0:c[1] - y0 + c[3], c[0] - x0:c[0] - x0 + c[2]] |= c[4]
        merged.append((x0, y0, x1 - x0, y1 - y0, mask))
    return sorted(merged, key=lambda c: c[0])


def is_barline(c, d):
    return c[2] <= 0.45 * d and c[3] >= 3.6 * d


def to_tensor(mask):
    """Pad a binary crop to square (keeping aspect ratio), resize to IMG x IMG, scale to 0..1."""
    h, w = mask.shape[:2]
    s = max(h, w) + 6
    canvas = np.zeros((s, s), np.uint8)
    canvas[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = mask
    return cv2.resize(canvas, (IMG, IMG), interpolation=cv2.INTER_AREA).astype(np.float32)[..., None] / 255.0


def junk_crop(rng, kind=None):
    """Synthetic 'not a symbol' crops: noise blobs, stray lines, text, digits."""
    s = 80
    img = np.zeros((s, s), np.uint8)
    kind = kind or rng.choice(["blob", "lines", "text", "scatter"])
    if kind == "blob":
        cv2.ellipse(img, (s // 2, s // 2), (rng.randint(4, 30), rng.randint(2, 20)), rng.randint(0, 180), 0, 360, 255,
                    rng.choice([-1, 2, 4]))
    elif kind == "lines":
        for _ in range(rng.randint(1, 4)):
            cv2.line(img, (rng.randint(0, s), rng.randint(0, s)), (rng.randint(0, s), rng.randint(0, s)), 255,
                     rng.randint(1, 4))
    elif kind == "text":
        cv2.putText(img, rng.choice(list("abcdefghkmnprstuvwxyz0123456789")), (rng.randint(5, 25), rng.randint(40, 70)),
                    cv2.FONT_HERSHEY_SIMPLEX, rng.uniform(0.8, 2.2), 255, rng.randint(1, 4))
    else:
        for _ in range(rng.randint(3, 15)):
            cv2.circle(img, (rng.randint(0, s), rng.randint(0, s)), rng.randint(1, 4), 255, -1)
    ys, xs = np.where(img > 0)
    if len(xs) == 0:
        return junk_crop(rng, "blob")
    return img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def half_cut(mask, rng):
    """Cut a real symbol in half (partial symbols must be rejected as unknown)."""
    h, w = mask.shape
    if rng.random() < 0.5 and w > 6:
        c = rng.randint(w // 3, 2 * w // 3)
        m = mask[:, :c] if rng.random() < 0.5 else mask[:, c:]
    else:
        c = rng.randint(h // 3, 2 * h // 3) if h > 6 else h // 2
        m = mask[:c] if rng.random() < 0.5 else mask[c:]
    return m if m.size and m.any() else mask


def augment_mask(mask, rng, strength=1.0):
    """Small random rotation/scale/blur/noise on a binary crop (training-time augmentation)."""
    h, w = mask.shape
    pad = max(h, w) // 3 + 4
    canvas = np.zeros((h + 2 * pad, w + 2 * pad), np.uint8)
    canvas[pad:pad + h, pad:pad + w] = mask
    ang = rng.uniform(-6, 6) * strength
    sc = 1 + rng.uniform(-0.12, 0.12) * strength
    m = cv2.getRotationMatrix2D((canvas.shape[1] / 2, canvas.shape[0] / 2), ang, sc)
    out = cv2.warpAffine(canvas, m, (canvas.shape[1], canvas.shape[0]), flags=cv2.INTER_LINEAR)
    if rng.random() < 0.6 * strength:
        out = cv2.GaussianBlur(out, (0, 0), rng.uniform(0.4, 1.4))
    out = (out > 110).astype(np.uint8) * 255
    ys, xs = np.where(out > 0)
    return out[ys.min():ys.max() + 1, xs.min():xs.max() + 1] if len(xs) else mask


def render_page(page_seed, workdir):
    """Render one labelled page. Returns list of (label, mask) crops (anomalous pages are skipped)."""
    rng = random.Random(page_seed)
    clef = list(synth.CLEFS)[rng.randrange(3)]
    cells = make_cells(rng, clef)
    stem = Path(workdir) / f"sym_{page_seed}"
    png = synth.render_png(page_ly(cells, clef), stem)
    bw, _ = preprocess.preprocess(cv2.imread(str(png)))
    sts = staff.detect_staves(bw)
    if len(sts) != 1:
        return []
    comps = components(bw, sts[0])
    d = sts[0].space
    comps = comps[1:]                                   # first component is the clef
    out, i = [], 0
    for ly, labels in cells:
        # components of this cell are everything up to (and consuming) the next barline
        got = []
        while i < len(comps) and not is_barline(comps[i], d):
            got.append(comps[i])
            i += 1
        bar = comps[i] if i < len(comps) else None
        i += 1
        if len(got) != len(labels) or bar is None:
            continue                                    # segmentation mismatch: skip, never mislabel
        out += [(lab, c[4]) for lab, c in zip(labels, got)]
        out.append(("barline", bar[4]))
    # clef crops come from the page header
    full = [c for c in components(bw, sts[0])]
    if full and full[0][3] >= 3 * d:
        out.append((f"clef_{clef}", full[0][4]))
    return out


def build_dataset(pages, seed, out_path, aug_per_crop=3, junk_ratio=0.12, workdir=None):
    """Render `pages` pages and write an .npz with train/val/test arrays split BY PAGE."""
    import tempfile
    rng = random.Random(seed)
    tmp = tempfile.TemporaryDirectory()
    wd = workdir or tmp.name
    page_seeds = [seed * 100000 + i for i in range(pages)]
    rng.shuffle(page_seeds)
    n_val = max(1, pages // 10)
    splits = {"val": page_seeds[:n_val], "test": page_seeds[n_val:2 * n_val], "train": page_seeds[2 * n_val:]}
    data = {}
    for name, seeds in splits.items():
        xs, ys, hashes = [], [], set()
        for ps in seeds:
            for label, mask in render_page(ps, wd):
                k = 2 if name != "train" else aug_per_crop
                for j in range(k):
                    m = mask if j == 0 else augment_mask(mask, rng)
                    xs.append(to_tensor(m))
                    ys.append(CLASSES.index(label))
                if rng.random() < junk_ratio:
                    jm = junk_crop(rng) if rng.random() < 0.6 else half_cut(mask, rng)
                    xs.append(to_tensor(jm))
                    ys.append(CLASSES.index("unknown"))
        for _ in range(max(4, len(xs) // 25)):          # extra pure-junk samples
            xs.append(to_tensor(junk_crop(rng)))
            ys.append(CLASSES.index("unknown"))
        data[name] = (np.stack(xs).astype(np.float32), np.array(ys, np.int64))
    # Leak control (the 265/273 failure): fixed-appearance symbols (rests, barlines, clefs) render
    # identically on every page, so identical crops WILL recur across pages. Evaluation must not
    # reward memorising them, so any val/test crop identical to a train crop is dropped, and the
    # count is recorded. Val/test also get an augmented copy of each crop so they stay informative.
    hashset = lambda a: {hashlib.sha1(x.tobytes()).hexdigest() for x in a}
    train_h = hashset(data["train"][0])
    dropped = {}
    for other in ("val", "test"):
        x, y = data[other]
        keep = [i for i in range(len(x)) if hashlib.sha1(x[i].tobytes()).hexdigest() not in train_h]
        dropped[other] = len(x) - len(keep)
        data[other] = (x[keep], y[keep])
        assert not (train_h & hashset(data[other][0])), "leak survived dedupe"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, classes=np.array(CLASSES),
                        **{f"x_{k}": v[0] for k, v in data.items()}, **{f"y_{k}": v[1] for k, v in data.items()})
    return {**{k: len(v[1]) for k, v in data.items()}, "dropped_as_duplicates": dropped}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="data/symbols.npz")
    ap.add_argument("--aug", type=int, default=3)
    a = ap.parse_args()
    print(build_dataset(a.pages, a.seed, a.out, a.aug))
