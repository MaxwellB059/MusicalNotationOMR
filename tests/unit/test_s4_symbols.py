"""S4 tests: symbol dataset construction, leak control, and (where TensorFlow exists) a smoke train."""
import importlib.util
import pathlib
import random
import subprocess
import sys

import numpy as np
import pytest

from omr import symbols, synth

HAVE_LILY = pathlib.Path(synth.LILYPOND).exists()
HAVE_TF = importlib.util.find_spec("tensorflow") is not None
ROOT = pathlib.Path(__file__).resolve().parents[2]


def test_class_list_is_stable_and_has_unknown():
    assert symbols.CLASSES[-1] == "unknown" and len(set(symbols.CLASSES)) == len(symbols.CLASSES)
    assert "clef_alto" in symbols.CLASSES and "note_16th" in symbols.CLASSES


def test_to_tensor_shape_range_and_aspect_preserved():
    mask = np.zeros((40, 10), np.uint8)
    mask[:, :] = 255
    t = symbols.to_tensor(mask)
    assert t.shape == (symbols.IMG, symbols.IMG, 1) and 0.0 <= t.min() and t.max() <= 1.0
    cols = np.where(t[..., 0].max(axis=0) > 0.5)[0]
    rows = np.where(t[..., 0].max(axis=1) > 0.5)[0]
    assert (rows.max() - rows.min()) > 2.5 * (cols.max() - cols.min())   # tall stays tall


def test_junk_and_half_cut_are_nonempty_and_deterministic():
    a = symbols.junk_crop(random.Random(1))
    b = symbols.junk_crop(random.Random(1))
    assert a.size and np.array_equal(a, b)
    m = np.zeros((30, 20), np.uint8)
    m[5:25, 5:15] = 255
    assert symbols.half_cut(m, random.Random(2)).any()


def test_merge_fragments_rejoins_stacked_pieces_but_not_barlines_or_gapped_symbols():
    d = 20
    top = (10, 10, 15, 12, np.full((12, 15), 255, np.uint8))
    bottom = (10, 22, 15, 12, np.full((12, 15), 255, np.uint8))        # touching, same columns -> one head
    far = (60, 10, 15, 12, np.full((12, 15), 255, np.uint8))           # separate symbol
    bar = (30, 0, 4, 90, np.full((90, 4), 255, np.uint8))              # barline
    merged = symbols.merge_fragments(sorted([top, bottom, far, bar], key=lambda c: c[0]), d)
    assert len(merged) == 3
    assert any(c[3] == 24 for c in merged)                              # the rejoined head


@pytest.mark.skipif(not HAVE_LILY, reason="LilyPond not installed")
def test_render_page_labels_match_construction(tmp_path):
    seen = {}
    for seed in range(1, 6):
        for label, mask in symbols.render_page(seed, tmp_path):
            seen.setdefault(label, 0)
            seen[label] += 1
            assert mask.any()
    for must in ("barline", "note_quarter", "rest_quarter", "note_eighth"):
        assert seen.get(must, 0) >= 3, seen
    assert not (set(seen) - set(symbols.CLASSES))


@pytest.mark.skipif(not HAVE_LILY, reason="LilyPond not installed")
def test_dataset_split_by_page_and_no_train_val_duplicates(tmp_path):
    out = tmp_path / "s.npz"
    info = symbols.build_dataset(pages=12, seed=3, out_path=out, aug_per_crop=2)
    d = np.load(out)
    import hashlib
    h = lambda a: {hashlib.sha1(x.tobytes()).hexdigest() for x in a}
    assert not (h(d["x_train"]) & h(d["x_val"])) and not (h(d["x_train"]) & h(d["x_test"]))
    assert info["train"] > info["val"] and "dropped_as_duplicates" in info


@pytest.mark.skipif(not (HAVE_LILY and HAVE_TF), reason="needs LilyPond and TensorFlow")
def test_smoke_training_runs_end_to_end():
    r = subprocess.run([sys.executable, "-m", "omr.train_symbols", "--config", "configs/symbols.yaml", "--smoke"],
                       cwd=ROOT, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-1500:]
    assert "GATE per_class_min" in r.stdout
    out = ROOT / "runs" / "symbols_smoke"
    assert (out / "best.keras").exists() and (out / "metrics.json").exists()
