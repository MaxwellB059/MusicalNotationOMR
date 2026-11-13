"""MusicXML builder for whole lines (replaces the single-note builder; PLAN.md S7).

Takes measures of events instead of one note, writes the REAL clef, key and time signature, and
supports ties, articulations, dynamics and hairpins so S9 can attach marks without new plumbing.
"""
from xml.dom import minidom
from xml.etree.ElementTree import Element, SubElement, tostring

DIVISIONS = 4  # per quarter note, so a 16th = 1
DUR = {"whole": 16, "half": 8, "quarter": 4, "eighth": 2, "16th": 1}
CLEF_XML = {"treble": ("G", 2), "bass": ("F", 4), "alto": ("C", 3), "tenor": ("C", 4)}


def parse_pitch(name):
    """'C#4' -> ('C', 1, 4); 'Bb3' -> ('B', -1, 3)."""
    step = name[0]
    octave = int(name[-1])
    acc = name[1:-1]
    return step, {"": 0, "#": 1, "b": -1, "-": -1}[acc], octave


def measure_units(events):
    return sum(DUR[e["dur"]] * (1.5 if e.get("dots") else 1) for e in events)


def expected_units(beats, beat_type):
    return int(beats * 16 / beat_type)


def build_musicxml(measures, clef="treble", fifths=0, beats=4, beat_type=4, flags=None):
    """measures: list of lists of event dicts {type:'note'|'rest', pitch?, dur, tie_start?, tie_stop?,
    articulations?, dynamic?, hairpin?: 'crescendo'|'decrescendo'|'stop'}. Returns a MusicXML string."""
    score = Element("score-partwise", version="3.1")
    pl = SubElement(score, "part-list")
    sp = SubElement(pl, "score-part", id="P1")
    SubElement(sp, "part-name").text = "OMR"
    part = SubElement(score, "part", id="P1")
    for mi, events in enumerate(measures, 1):
        meas = SubElement(part, "measure", number=str(mi))
        if mi == 1:
            at = SubElement(meas, "attributes")
            SubElement(at, "divisions").text = str(DIVISIONS)
            k = SubElement(at, "key")
            SubElement(k, "fifths").text = str(fifths)
            t = SubElement(at, "time")
            SubElement(t, "beats").text = str(beats)
            SubElement(t, "beat-type").text = str(beat_type)
            c = SubElement(at, "clef")
            sign, line = CLEF_XML[clef]
            SubElement(c, "sign").text = sign
            SubElement(c, "line").text = str(line)
        for e in events:
            if e.get("dynamic") or e.get("hairpin"):
                d = SubElement(SubElement(meas, "direction", placement="below"), "direction-type")
                if e.get("dynamic"):
                    SubElement(SubElement(d, "dynamics"), e["dynamic"])
                if e.get("hairpin"):
                    SubElement(d, "wedge", type={"crescendo": "crescendo", "decrescendo": "diminuendo",
                                                 "stop": "stop"}[e["hairpin"]])
            n = SubElement(meas, "note")
            if e["type"] == "rest":
                SubElement(n, "rest")
            else:
                step, alter, octave = parse_pitch(e["pitch"])
                p = SubElement(n, "pitch")
                SubElement(p, "step").text = step
                if alter:
                    SubElement(p, "alter").text = str(alter)
                SubElement(p, "octave").text = str(octave)
            units = DUR[e["dur"]] * (1.5 if e.get("dots") else 1)
            SubElement(n, "duration").text = str(int(units))
            if e.get("tie_stop"):
                SubElement(n, "tie", type="stop")
            if e.get("tie_start"):
                SubElement(n, "tie", type="start")
            SubElement(n, "type").text = e["dur"]
            if e.get("dots"):
                SubElement(n, "dot")
            if e.get("articulations"):
                nt = SubElement(n, "notations")
                art = SubElement(nt, "articulations")
                for a in e["articulations"]:
                    SubElement(art, a)
    if flags:
        score.insert(0, __import__("xml.etree.ElementTree", fromlist=["Comment"]).Comment(
            " needs-review: " + "; ".join(flags) + " "))
    return minidom.parseString(tostring(score, encoding="utf-8")).toprettyxml(indent="  ")
