"""Tokens, vocabulary, dataset split and image loading for the real-image CRNN (no TensorFlow needed).

Camera-PrIMuS gives one 'semantic' token sequence per incipit. We factor note tokens so the
vocabulary stays small and every symbol is seen often:
    note-Bb4_quarter  ->  n:Bb4  d:quarter        rest-eighth  ->  r  d:eighth
Other tokens (clef-G2, keySignature-EbM, timeSignature-3/4, barline) are kept as they are.
Samples containing tokens we can't write as MusicXML (tuplets, grace notes, 32nd notes, ...) are
dropped from training, and the drop rate is reported.
"""
import hashlib
import random
from pathlib import Path

import cv2
import numpy as np

from . import primus

HEIGHT = 64
WIDTH_STRIDE = 4            # the network shrinks width by this factor (see omr.crnn)
SUPPORTED_DURS = {"whole", "half", "quarter", "eighth", "sixteenth", "thirty_second"}
# Keras ctc_batch_cost reserves the LAST class as the CTC blank, so the vocab ends with "<blank>".


def tokenize(semantic_text):
    """Return a list of model tokens, or None if the sample contains unsupported symbols."""
    out = []
    for tok in semantic_text.strip().split():
        if tok.startswith("note-"):
            pitch, _, dur = tok[5:].partition("_")
            base = dur.rstrip(".")
            if base not in SUPPORTED_DURS or len(dur) - len(base) > 1 or primus.midi(pitch) is None:
                return None
            out += [f"n:{pitch}", f"d:{base}" + ("." if dur.endswith(".") else "")]
        elif tok.startswith("rest-"):
            dur = tok[5:]
            base = dur.rstrip(".")
            if base not in SUPPORTED_DURS or len(dur) - len(base) > 1:
                return None
            out += ["r", f"d:{base}" + ("." if dur.endswith(".") else "")]
        elif tok.startswith(("clef-", "keySignature-", "timeSignature-")) or tok in ("barline", "tie"):
            out.append(tok)
        else:
            return None                                  # tie, fermata, gracenote, multirest, ...
    return out


def detokenize(tokens):
    """Inverse of tokenize(): model tokens -> PrIMuS semantic text (used to score via omr.primus)."""
    out, i = [], 0
    while i < len(tokens):
        t = tokens[i]
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        if t.startswith("n:") and nxt.startswith("d:"):
            out.append(f"note-{t[2:]}_{nxt[2:]}"); i += 2
        elif t == "r" and nxt.startswith("d:"):
            out.append(f"rest-{nxt[2:]}"); i += 2
        elif t.startswith(("clef-", "keySignature-", "timeSignature-")) or t in ("barline", "tie"):
            out.append(t); i += 1
        else:
            i += 1                                       # malformed fragment: dropped, never guessed
    return "\t".join(out)


def piece_id(sample_name):
    """'000051650-1_1_1' -> '000051650': incipits of one piece must stay in one split."""
    return sample_name.split("-")[0]


def split_of(sample_name, val=0.05, test=0.05, seed=1):
    h = int(hashlib.sha1(f"{seed}:{piece_id(sample_name)}".encode()).hexdigest(), 16) % 10000 / 10000.0
    return "test" if h < test else "val" if h < test + val else "train"


def list_samples(root, limit=None, seed=1):
    """[(name, semantic_tokens, clean_png, camera_jpg)] with supported tokens only; also drop stats."""
    items, dropped = [], 0
    for d in sorted(Path(root).iterdir()):
        if not d.is_dir():
            continue
        sem = next(d.glob("*.semantic"), None)
        if sem is None:
            continue
        toks = tokenize(sem.read_text(encoding="utf-8"))
        if not toks:
            dropped += 1
            continue
        items.append((d.name, toks, next(d.glob("*.png"), None), next(d.glob("*_distorted.jpg"), None)))
        if limit and len(items) >= limit:
            break
    return items, dropped


def build_vocab(items):
    toks = sorted({t for _, ts, _, _ in items for t in ts})
    return toks + ["<blank>"]


def load_image(path, augment=False, rng=None):
    """Grayscale, height normalised to HEIGHT, ink bright (1.0) on dark, float32 (H, W)."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    img = cv2.resize(img, (max(int(w * HEIGHT / h), WIDTH_STRIDE), HEIGHT), interpolation=cv2.INTER_AREA)
    img = 255 - img
    if augment and rng is not None:
        m = cv2.getRotationMatrix2D((img.shape[1] / 2, HEIGHT / 2), rng.uniform(-1.5, 1.5), rng.uniform(0.95, 1.05))
        img = cv2.warpAffine(img, m, (img.shape[1], HEIGHT))
        if rng.random() < 0.5:
            img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.3, 1.0))
        img = np.clip(img.astype(np.float32) * rng.uniform(0.8, 1.2) + rng.uniform(-10, 10), 0, 255)
    x = img.astype(np.float32) / 255.0
    return x


def pad_batch(images):
    """Stack variable-width images into (B, H, W, 1), W a multiple of WIDTH_STRIDE; returns also widths."""
    widths = [im.shape[1] for im in images]
    W = int(np.ceil(max(widths) / WIDTH_STRIDE) * WIDTH_STRIDE)
    out = np.zeros((len(images), HEIGHT, W, 1), np.float32)
    for i, im in enumerate(images):
        out[i, :, :im.shape[1], 0] = im
    return out, widths


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def greedy_decode(logits_argmax, vocab):
    """Collapse repeats and drop blanks from per-timestep argmax ids."""
    blank = len(vocab) - 1
    out, prev = [], blank
    for k in logits_argmax:
        if k != prev and k != blank:
            out.append(vocab[k])
        prev = k
    return out
