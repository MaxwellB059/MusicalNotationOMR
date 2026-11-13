"""Stage S9: attach detected marks to the notes they belong to (PLAN.md).

Pure rules on x positions, independent of how the marks were detected (classical fallback or
YOLO). Events are the dicts produced by omr.assemble, each carrying an `x` pixel position.
  tie        links two successive notes of the same pitch -> tie_start / tie_stop
  hairpin    spans from the first note at/after its left end to the last note at/before its right end
  dynamic    applies from the nearest event at or after its x (a dynamic sits under its note)
  accent / staccato  belong to the nearest note by x
"""


def _flat(measures):
    return [e for m in measures for e in m]


def _nearest(events, x, only_notes=False, after=False, tol=None):
    pool = [e for e in events if (e["type"] == "note" or not only_notes) and "x" in e]
    if after:
        pool = [e for e in pool if e["x"] >= x - (tol or 0)]
    if not pool:
        return None
    best = min(pool, key=lambda e: abs(e["x"] - x))
    return best if tol is None or abs(best["x"] - x) <= tol else (best if after else None)


def attach_ties(measures, ties, d):
    """ties: [{x0, x1}] in pixels. Marks the two nearest notes if they share a pitch."""
    notes = [e for e in _flat(measures) if e["type"] == "note"]
    n_done = 0
    for t in ties:
        a, b = _nearest(notes, t["x0"], True, tol=1.5 * d), _nearest(notes, t["x1"], True, tol=1.5 * d)
        if a is None or b is None or a is b or a["pitch"] != b["pitch"]:
            continue
        if notes.index(b) != notes.index(a) + 1:
            continue
        a["tie_start"] = True
        b["tie_stop"] = True
        n_done += 1
    return n_done


def attach_hairpins(measures, hairpins, d):
    n_done = 0
    events = _flat(measures)
    for h in hairpins:
        inside = [e for e in events if "x" in e and h["x0"] - d <= e["x"] <= h["x1"] + d]
        if not inside:
            continue
        inside[0]["hairpin"] = h["kind"]
        if len(inside) > 1:
            inside[-1]["hairpin"] = "stop"
        else:
            inside[0]["hairpin"] = h["kind"]
        n_done += 1
    return n_done


def attach_dynamics(measures, dynamics, d):
    """dynamics: [{label: 'p'|'mp'|'mf'|'f'|'ff'|..., x}]"""
    n_done = 0
    events = _flat(measures)
    for dyn in dynamics:
        e = _nearest(events, dyn["x"], after=True, tol=1.0 * d)
        if e is not None:
            e["dynamic"] = dyn["label"]
            n_done += 1
    return n_done


def attach_articulations(measures, marks, d):
    """marks: [{kind: 'accent'|'staccato', x}] -> nearest note within one staff space + head width."""
    n_done = 0
    notes = [e for e in _flat(measures) if e["type"] == "note"]
    for m in marks:
        e = _nearest(notes, m["x"], True, tol=1.2 * d)
        if e is not None:
            e.setdefault("articulations", []).append(m["kind"])
            n_done += 1
    return n_done
