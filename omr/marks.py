"""Stage S6 (classical fallback): find marks that homr misses, using only geometry (PLAN.md).

  hairpins   two long, nearly straight strokes that meet at a point and open to the other side
  ties       a thin arc joining two noteheads of the same pitch (shallow curve, ends at heads)
  accents/staccato  small marks near a notehead on the side opposite the stem

This is the BASELINE the YOLO detector must beat (gate D). It runs on the staff-line-free page from
omr.staff.remove_staff_lines and uses the staff space `d` for every size, so it works at any zoom.
Dynamics (p, mf, f ...) need a glyph classifier and are handled in S6/YOLO, not here.
"""
import cv2
import numpy as np


def _components(bw, min_area):
    n, lab, stats, cent = cv2.connectedComponentsWithStats(bw)
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area >= min_area:
            out.append({"x": x, "y": y, "w": w, "h": h, "area": area,
                        "mask": (lab[y:y + h, x:x + w] == i).astype(np.uint8) * 255})
    return out


def find_hairpins(cleaned, staff, below_only=False):
    """Return [{kind: 'crescendo'|'decrescendo', x0, x1, y}] for hairpins near the staff.

    A hairpin component is wide (>= 3d), short (<= 2d), has little ink for its box (two thin lines),
    and its vertical thickness grows (crescendo opens to the right) or shrinks along x.
    """
    d = staff.space
    found = []
    for c in _components(cleaned, min_area=int(0.4 * d * d)):
        if not (c["w"] >= 3.0 * d and 0.4 * d <= c["h"] <= 2.2 * d):
            continue
        if c["area"] > 0.45 * c["w"] * c["h"]:
            continue                                    # too solid: a beam or slur, not two thin lines
        mask = c["mask"] > 0
        cols = mask.sum(axis=0)
        span = np.array([np.ptp(np.where(mask[:, j])[0]) + 1 if mask[:, j].any() else 0 for j in range(c["w"])])
        k = max(c["w"] // 4, 1)
        left, right = span[:k].mean(), span[-k:].mean()
        if abs(right - left) < 0.5 * d:
            continue
        # the two strokes must be fairly straight: span should vary roughly linearly with x
        xs = np.where(span > 0)[0]
        if len(xs) < 4:
            continue
        slope, icpt = np.polyfit(xs, span[xs], 1)
        resid = np.abs(span[xs] - (slope * xs + icpt)).mean()
        if resid > 0.25 * d:
            continue
        found.append({"kind": "crescendo" if right > left else "decrescendo",
                      "x0": int(c["x"]), "x1": int(c["x"] + c["w"]),
                      "y": float(c["y"] + c["h"] / 2)})
    return sorted(found, key=lambda h: h["x0"])


def find_ties(cleaned, staff, heads):
    """Return [{x0, x1, from_idx, to_idx}] for thin arcs linking two successive noteheads.

    `heads` is the left-to-right notehead list from omr.noteheads (needs x, y, position).
    A tie is a thin arc whose ends land within ~1d of two adjacent heads of equal position.
    """
    d = staff.space
    ties = []
    for c in _components(cleaned, min_area=int(0.3 * d * d)):
        if not (1.2 * d <= c["w"] <= 8 * d and c["h"] <= 1.3 * d):
            continue
        if c["area"] > 0.5 * c["w"] * c["h"]:
            continue
        x0, x1 = c["x"], c["x"] + c["w"]
        left = [i for i, h in enumerate(heads) if abs(h["x"] - x0) <= 1.2 * d]
        right = [i for i, h in enumerate(heads) if abs(h["x"] - x1) <= 1.2 * d]
        for i in left:
            for j in right:
                if j == i + 1 and heads[i]["position"] == heads[j]["position"] \
                        and abs(c["y"] - heads[i]["y"]) <= 2.0 * d:
                    ties.append({"x0": int(x0), "x1": int(x1), "from_idx": i, "to_idx": j})
    return ties
