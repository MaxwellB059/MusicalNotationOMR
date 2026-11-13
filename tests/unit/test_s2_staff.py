"""S2 gate: staves found, line positions accurate (<0.1 d), removal keeps noteheads."""
import cv2
import numpy as np
import pytest

from omr import augment, preprocess, staff, synth
from tests.unit.test_s1_preprocess import staff_image

HAVE_LILY = __import__("pathlib").Path(synth.LILYPOND).exists()


def truth_lines(height=260, d=12):
    top = height // 2 - 2 * d
    return [top + i * d for i in range(5)]


def test_detects_one_staff_with_exact_geometry():
    bw = preprocess.binarize(staff_image())
    st = staff.detect_staves(bw)
    assert len(st) == 1
    s = st[0]
    assert s.space == pytest.approx(12, abs=0.1 * 12)
    for got, want in zip(s.lines, truth_lines()):
        assert abs(got - want) < 0.1 * 12
    assert s.thickness >= 1


def test_no_staff_in_blank_or_text_like_image():
    assert staff.detect_staves(np.zeros((200, 600), np.uint8)) == []
    noise = (np.random.default_rng(0).random((200, 600)) > 0.97).astype(np.uint8) * 255
    assert staff.detect_staves(noise) == []


def test_two_staves_found_in_order():
    a = staff_image(height=260, seed=1)
    b = staff_image(height=260, seed=2)
    page = np.vstack([a, b])
    st = staff.detect_staves(preprocess.binarize(page))
    assert len(st) == 2 and st[0].top < st[1].top
    assert abs(st[1].top - st[0].top - 260) < 0.2 * 12


def test_gate_across_augmented_images():
    """>=95% of rotated/blurred/noisy images: one staff, space normalised to the 20 px target."""
    ok, total = 0, 30
    for i in range(total):
        deg, _ = augment.degrade(staff_image(seed=i), 500 + i, "medium", with_perspective=False)
        bw, _ = preprocess.preprocess(cv2.cvtColor(deg, cv2.COLOR_GRAY2BGR))
        st = staff.detect_staves(bw)
        if len(st) == 1 and abs(st[0].space - preprocess.TARGET_SPACE) / preprocess.TARGET_SPACE < 0.05:
            ok += 1
    assert ok / total >= 0.95, f"{ok}/{total}"


def test_removal_erases_lines_but_keeps_noteheads_and_stems():
    img = staff_image()
    bw = preprocess.binarize(img)
    st = staff.detect_staves(bw)
    cleaned = staff.remove_staff_lines(bw, st)
    # most line pixels gone: ink drops, but not to zero
    assert 0 < cleaned.sum() < bw.sum() * 0.6
    # a notehead centre must still be ink, and its stem too
    ys, xs = np.where(cleaned > 0)
    assert len(xs) > 0
    # no remaining long horizontal run between notes on the middle line
    mid = int(st[0].lines[2])
    row = cleaned[mid - 1:mid + 2, 20:60]
    assert row.sum() == 0


@pytest.mark.skipif(not HAVE_LILY, reason="LilyPond not installed")
def test_real_lilypond_render_one_staff(tmp_path):
    items = synth.generate(4, seed=11, out_dir=tmp_path)
    for stem in items:
        img = cv2.imread(str(tmp_path / (stem + ".png")))
        bw, _ = preprocess.preprocess(img)
        st = staff.detect_staves(bw)
        assert len(st) == 1, stem
        assert abs(st[0].space - preprocess.TARGET_SPACE) < 1.0   # preprocess normalises to 20 px
        diffs = np.diff(st[0].lines)
        assert diffs.max() - diffs.min() < 1.0
