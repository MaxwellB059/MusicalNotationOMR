"""S1 gate: deskew recovers applied rotation to <=0.5 deg on >=95% of degraded images."""
import cv2
import numpy as np
import pytest

from omr import augment, preprocess


def staff_image(width=900, height=260, d=12, seed=0):
    """White page with a 5-line staff plus note-like blobs and stems, drawn deterministically."""
    rng = np.random.default_rng(seed)
    img = np.full((height, width), 255, np.uint8)
    top = height // 2 - 2 * d
    for i in range(5):
        cv2.line(img, (20, top + i * d), (width - 20, top + i * d), 0, 2)
    for x in range(80, width - 60, 70):
        y = top + int(rng.integers(-1, 6)) * d // 2
        cv2.ellipse(img, (x, y), (int(d * 0.65), int(d * 0.45)), -20, 0, 360, 0, -1)
        cv2.line(img, (x + int(d * 0.6), y), (x + int(d * 0.6), y - 3 * d), 0, 2)
    return img


def test_decode_errors_are_clear():
    with pytest.raises(preprocess.ImageError):
        preprocess.load_image(b"")
    with pytest.raises(preprocess.ImageError):
        preprocess.load_image(b"not an image at all")


def test_load_png_roundtrip():
    ok, buf = cv2.imencode(".png", staff_image())
    assert preprocess.load_image(buf.tobytes()).shape[:2] == (260, 900)


def test_zero_skew_stays_near_zero():
    bw = preprocess.binarize(staff_image())
    assert abs(preprocess.estimate_skew(bw)) <= 0.3


def _recovery(level, with_perspective, tol, total=40):
    ok = 0
    for i in range(total):
        degraded, p = augment.degrade(staff_image(seed=i), seed=1000 + i, level=level,
                                      with_perspective=with_perspective)
        angle = preprocess.estimate_skew(preprocess.binarize(degraded))
        # rotate(img, +r) is undone by rotating by -r; estimate_skew returns the correction angle
        if abs(angle + p["rotation_deg"]) <= tol:
            ok += 1
    return ok / total


def test_skew_recovery_gate():
    """Gate (PLAN.md S1): within 0.5 deg on >=95% of rotated+blurred+noisy+JPEG images."""
    for level in ("mild", "medium", "hard"):
        assert _recovery(level, False, 0.5) >= 0.95, level


def test_skew_with_perspective_is_within_one_degree():
    """Perspective adds unlabelled tilt, so the honest tolerance there is looser (1 deg)."""
    assert _recovery("medium", True, 1.0) >= 0.95


def test_preprocess_output_is_binary_and_cropped():
    degraded, _ = augment.degrade(staff_image(), seed=3, level="mild")
    bw, info = preprocess.preprocess(cv2.cvtColor(degraded, cv2.COLOR_GRAY2BGR))
    assert set(np.unique(bw)) <= {0, 255}
    assert bw.shape[0] < degraded.shape[0] * 1.5 and (bw > 0).any()
    assert "skew_deg" in info


def test_augment_deterministic_and_bounded():
    base = staff_image()
    a, pa = augment.degrade(base, 5, "hard")
    b, pb = augment.degrade(base, 5, "hard")
    assert np.array_equal(a, b) and pa == pb
    assert abs(pa["rotation_deg"]) <= augment.LEVELS["hard"]["rot"]
    c, _ = augment.degrade(base, 6, "hard")
    assert not np.array_equal(a, c) or a.shape != c.shape
