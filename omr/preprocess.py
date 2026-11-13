"""Stage S1: image intake -> straight, binarized, margin-cropped image (PLAN.md).

Staff detection (S2) assumes horizontal staff lines, so skew estimation matters most here.
Skew is estimated by a projection-profile sweep: the correct angle makes staff lines line up
in rows, which maximises the variance of the row sums.
"""
import cv2
import numpy as np


class ImageError(ValueError):
    """Raised with a user-readable message when an upload can't be used."""


def load_image(data):
    """Decode bytes (PNG/JPG; PDF if PyMuPDF is installed) into a BGR array."""
    if not data:
        raise ImageError("Empty file")
    if data[:4] == b"%PDF":
        try:
            import fitz  # PyMuPDF, optional
        except ImportError:
            raise ImageError("PDF support needs PyMuPDF (pip install pymupdf)")
        doc = fitz.open(stream=data, filetype="pdf")
        pix = doc[0].get_pixmap(dpi=200)
        arr = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR if pix.n == 3 else cv2.COLOR_RGBA2BGR)
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ImageError("Could not decode image (expected PNG, JPG or PDF)")
    return img


def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def binarize(gray):
    """Ink = 255, paper = 0. Adaptive threshold copes with uneven phone lighting."""
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    bw = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 41, 12)
    return bw


def estimate_skew(bw, max_deg=20.0, coarse=1.0, fine=0.1):
    """Angle in degrees to rotate by to make staff lines horizontal (positive = counter-clockwise)."""
    h, w = bw.shape
    scale = 800.0 / max(h, w)
    small = cv2.resize(bw, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else bw
    sh, sw = small.shape
    pad = int(np.hypot(sh, sw)) // 2

    def score(angle):
        m = cv2.getRotationMatrix2D((sw / 2, sh / 2), angle, 1.0)
        r = cv2.warpAffine(small, m, (sw, sh), flags=cv2.INTER_LINEAR)
        rows = r.sum(axis=1, dtype=np.float64)
        return float(np.var(np.diff(rows)))

    best = max(np.arange(-max_deg, max_deg + coarse, coarse), key=score)
    best = max(np.arange(best - coarse, best + coarse + fine, fine), key=score)
    return float(best)


def rotate_keep(img, angle, fill=0):
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(img, m, (nw, nh), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=fill)


def crop_margins(bw, pad=12):
    ys, xs = np.where(bw > 0)
    if len(xs) < 10:
        return bw
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    return bw[max(y0 - pad, 0):y1 + pad + 1, max(x0 - pad, 0):x1 + pad + 1]


TARGET_SPACE = 20.0   # every downstream stage sees staff spaces of ~20 px


def preprocess(img_bgr, target_space=TARGET_SPACE):
    """Full S1: returns (bw_straight_scaled_cropped, info).

    Order matters: rotate and rescale the GRAYSCALE image, binarize last. Binarizing a small
    image and then upscaling the binary one mangles noteheads.
    info: skew_deg (correction applied), scale (resize factor), staff_space (px before scaling).
    """
    gray = to_gray(img_bgr)
    angle = estimate_skew(binarize(gray))
    gray = rotate_keep(gray, angle, fill=int(np.median(gray)))
    scale = 1.0
    from .staff import detect_staves   # local import: staff.py imports nothing from here
    st = detect_staves(binarize(gray))
    if st:
        scale = target_space / st[0].space
        if abs(scale - 1.0) > 0.05:
            gray = cv2.resize(gray, None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
        else:
            scale = 1.0
    d = target_space if st else 12.0
    block = int(d * 6) | 1
    bw = cv2.adaptiveThreshold(cv2.GaussianBlur(gray, (3, 3), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, block, 10)
    bw = crop_margins(bw)
    return bw, {"skew_deg": angle, "scale": scale, "staff_space": st[0].space if st else None}
