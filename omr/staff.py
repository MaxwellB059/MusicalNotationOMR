"""Stage S2: staff-line detection, staff-space measurement and staff-line removal (PLAN.md).

Input is a straightened binary image (ink = 255) from omr.preprocess. Everything downstream is
measured in units of the staff space `d` (distance between adjacent lines) so it works at any zoom.
"""
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class Staff:
    lines: list            # y centre of each of the 5 lines, top to bottom (float, pixels)
    space: float           # mean distance between adjacent lines (the staff space d)
    thickness: float       # line thickness in pixels
    x0: int = 0            # horizontal extent of the staff
    x1: int = 0
    extra: dict = field(default_factory=dict)

    @property
    def top(self):
        return self.lines[0]

    @property
    def bottom(self):
        return self.lines[-1]

    def position_to_y(self, pos):
        """Staff position -> y pixel. pos 0 = bottom line, 1 = first space, 2 = second line, ..."""
        return self.bottom - pos * self.space / 2.0


def _row_profile(bw):
    """Fraction of the image width covered by ink in each row, using the longest horizontal run."""
    h, w = bw.shape
    # keep only long horizontal strokes so note blobs and text don't create false rows
    k = max(w // 20, 15)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1))
    horiz = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
    return horiz.sum(axis=1, dtype=np.float64) / 255.0 / w, horiz


def _line_centres(profile, min_cover):
    rows = np.where(profile >= min_cover)[0]
    if len(rows) == 0:
        return [], []
    groups, cur = [], [rows[0]]
    for r in rows[1:]:
        if r - cur[-1] <= 1:
            cur.append(r)
        else:
            groups.append(cur)
            cur = [r]
    groups.append(cur)
    centres = [float(np.average(g, weights=profile[g])) for g in groups]
    thick = [len(g) for g in groups]
    return centres, thick


def detect_staves(bw, min_cover_frac=0.35):
    """Return a list of Staff, top to bottom. Empty list if no staff found."""
    profile, horiz = _row_profile(bw)
    if profile.max() <= 0:
        return []
    centres, thick = _line_centres(profile, max(profile.max() * min_cover_frac, 0.02))
    staves, i = [], 0
    while i + 4 < len(centres):
        window = centres[i:i + 5]
        gaps = np.diff(window)
        if gaps.min() > 0 and (gaps.max() - gaps.min()) <= max(0.25 * gaps.mean(), 2.0):
            cols = np.where(horiz[int(window[0]):int(window[-1]) + 1].sum(axis=0) > 0)[0]
            staves.append(Staff(lines=window, space=float(gaps.mean()),
                                thickness=float(np.mean(thick[i:i + 5])),
                                x0=int(cols.min()) if len(cols) else 0,
                                x1=int(cols.max()) if len(cols) else bw.shape[1] - 1))
            i += 5
        else:
            i += 1
    return staves


def remove_staff_lines(bw, staves, margin=1):
    """Erase staff-line pixels but keep ink that continues above or below the line (stems, heads).

    A pixel on a line is erased only if the pixel `t` rows above AND below is empty (so the stroke
    is just the thin line). Returns a new image; the input is untouched.
    """
    out = bw.copy()
    h = bw.shape[0]
    for st in staves:
        t = int(round(st.thickness)) + margin
        for y in st.lines:
            lo, hi = int(round(y - st.thickness / 2)) - margin, int(round(y + st.thickness / 2)) + margin
            for row in range(max(lo, 0), min(hi, h - 1) + 1):
                above = bw[max(row - t - 1, 0)]
                below = bw[min(row + t + 1, h - 1)]
                erase = (above == 0) & (below == 0)
                out[row, erase] = 0
    return out


def crop_staff(img, staff, pad_spaces=4):
    """Crop one staff with `pad_spaces` staff spaces of room above and below (for ledger lines)."""
    pad = int(round(pad_spaces * staff.space))
    return img[max(int(staff.top) - pad, 0):int(staff.bottom) + pad + 1]


def normalize_scale(bw, target_space=20.0):
    """Resize so the staff space is ~target_space px (all geometry is then zoom-independent).

    Returns (resized_bw, scale). Uses the first detected staff; unchanged (scale 1.0) if none.
    """
    st = detect_staves(bw)
    if not st:
        return bw, 1.0
    scale = target_space / st[0].space
    if abs(scale - 1.0) < 0.05:
        return bw, 1.0
    h, w = bw.shape
    out = cv2.resize(bw, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_CUBIC)
    return ((out > 127).astype(np.uint8) * 255), scale
