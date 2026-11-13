"""Notehead detection on a staff-line-free image, then geometric pitch (PLAN.md S3 end-to-end).

Approach: after staff removal, open with a disc roughly one staff space wide. Stems and barlines
(thin) vanish, filled noteheads survive. Hollow noteheads (half/whole) are first filled by
closing their interior. Each surviving blob's centre gives (x, y); y -> staff position -> pitch.
Rests, clefs and accidentals are not noteheads: they are rejected by size/shape checks here and
named properly by the symbol CNN in S4.
"""
import cv2
import numpy as np

HOLE_MIN, HOLE_MAX, DISC_R = 0.04, 1.6, 0.46
SIZE = (0.7, 2.2, 0.6, 1.8)   # min/max blob width and height, in staff spaces   # tuned against synthetic lines (tests/unit)

from . import semantics
from .staff import remove_staff_lines


def _fill_hollow(bw, d):
    """Fill the interior of hollow noteheads (half/whole notes) so every head is a solid blob."""
    k = max(int(round(d * 0.9)), 3) | 1
    closed = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    # keep closing only where it fills holes (avoid merging neighbouring stems into slabs)
    contours, hier = cv2.findContours(bw, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    filled = bw.copy()
    if hier is None:
        return filled
    for i, c in enumerate(contours):
        if hier[0][i][3] != -1:                       # an inner contour = a hole
            area = cv2.contourArea(c)
            if HOLE_MIN * d * d < area < HOLE_MAX * d * d:   # hole is notehead-sized, not a clef curl
                cv2.drawContours(filled, [c], -1, 255, -1)
    return filled


def find_noteheads(bw, staff, min_cover=0.55):
    """Return list of dicts {x, y, position} sorted left to right, for one staff.

    `bw` is the full binary page (ink=255) with staff lines still present or already removed.
    """
    d = staff.space
    cleaned = remove_staff_lines(bw, [staff])
    solid = _fill_hollow(cleaned, d)
    r = max(int(round(d * DISC_R)), 2)
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    heads = cv2.morphologyEx(solid, cv2.MORPH_OPEN, disc)
    # restrict to the staff band plus room for ledger lines
    lo, hi = int(staff.top - 5 * d), int(staff.bottom + 5 * d)
    band = np.zeros_like(heads)
    band[max(lo, 0):hi] = heads[max(lo, 0):hi]
    n, _, stats, cent = cv2.connectedComponentsWithStats(band)
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if not (SIZE[0] * d <= w <= SIZE[1] * d and SIZE[2] * d <= h <= SIZE[3] * d):
            continue                                     # too small/large: not a lone notehead
        if area < min_cover * (np.pi / 4) * w * h:       # must be roughly elliptical
            continue
        cx, cy = cent[i]
        out.append({"x": float(cx), "y": float(cy),
                    "position": semantics.y_to_position(cy, staff)})
    return sorted(out, key=lambda n: n["x"])
