"""Photo-degradation augmenter: turns clean renders into 'simulated phone photos' (PLAN.md S0).

Every function is deterministic given a numpy Generator, so test sets are reproducible.
`degrade` returns the image plus the parameters used, so tests can check recovery
(e.g. that deskew undoes the rotation we applied).
"""
import cv2
import numpy as np

LEVELS = {
    # max rotation deg, blur sigma, noise std, jpeg quality floor, perspective frac, lighting strength
    "mild":   dict(rot=3,  blur=0.8, noise=4,  jpeg=70, persp=0.01, light=0.15),
    "medium": dict(rot=8,  blur=1.5, noise=8,  jpeg=50, persp=0.025, light=0.3),
    "hard":   dict(rot=15, blur=2.2, noise=14, jpeg=35, persp=0.045, light=0.5),
}


def rotate(img, deg, fill=255):
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(img, m, (nw, nh), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(fill,) * img.shape[2]
                          if img.ndim == 3 else fill)


def perspective(img, frac, rng, fill=255):
    h, w = img.shape[:2]
    j = lambda: rng.uniform(-frac, frac)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[j() * w, j() * h], [w + j() * w, j() * h],
                      [w + j() * w, h + j() * h], [j() * w, h + j() * h]])
    m = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, m, (w, h), borderMode=cv2.BORDER_CONSTANT,
                               borderValue=(fill,) * img.shape[2] if img.ndim == 3 else fill)


def blur(img, sigma):
    return cv2.GaussianBlur(img, (0, 0), sigma) if sigma > 0 else img


def add_noise(img, std, rng):
    n = rng.normal(0, std, img.shape)
    return np.clip(img.astype(np.float32) + n, 0, 255).astype(np.uint8)


def lighting(img, strength, rng):
    """Multiply by a smooth brightness gradient (uneven phone lighting / shadow)."""
    h, w = img.shape[:2]
    ang = rng.uniform(0, 2 * np.pi)
    xs, ys = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
    g = 1 - strength * 0.5 * (1 + np.cos(ang) * xs + np.sin(ang) * ys) / 2
    g = g if img.ndim == 2 else g[..., None]
    return np.clip(img.astype(np.float32) * g, 0, 255).astype(np.uint8)


def jpeg(img, quality):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    return cv2.imdecode(buf, cv2.IMREAD_UNCHANGED if img.ndim == 3 else cv2.IMREAD_GRAYSCALE)


def degrade(img, seed, level="medium", with_perspective=True):
    """Return (degraded_image, params). `params['rotation_deg']` is the applied rotation.

    Perspective warp adds extra, unlabelled tilt (up to ~1 deg), so tests that check skew
    recovery against `rotation_deg` alone should pass with_perspective=False.
    """
    cfg = LEVELS[level]
    rng = np.random.default_rng(seed)
    rot = float(rng.uniform(-cfg["rot"], cfg["rot"]))
    out = rotate(img, rot)
    persp_rng = np.random.default_rng(seed + 7919)   # keep other draws identical either way
    if with_perspective:
        out = perspective(out, cfg["persp"], persp_rng)
    out = lighting(out, cfg["light"], rng)
    out = blur(out, float(rng.uniform(0.3, cfg["blur"])))
    out = add_noise(out, float(rng.uniform(1, cfg["noise"])), rng)
    q = int(rng.integers(cfg["jpeg"], 95))
    out = jpeg(out, q)
    return out, {"rotation_deg": rot, "jpeg_quality": q, "level": level, "seed": seed}
