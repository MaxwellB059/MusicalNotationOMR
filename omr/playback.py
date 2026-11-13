"""Stage S11 (stretch): MusicXML -> MIDI, with exact note numbers (C#4 = Db4 = 61).

The browser plays the MIDI with a soundfont (Tone.js / html-midi-player); the server's job is the
exact, deterministic pitch and duration list, which is what the embedding experiment showed we
cannot get from similarity search.
"""
from music21 import converter, note, tempo


def notes_from_musicxml(xml_text_or_path):
    """Return [(midi_number or None for rests, quarter_length)] in time order."""
    src = str(xml_text_or_path)
    s = converter.parse(src, format="musicxml") if src.lstrip().startswith("<") else converter.parse(src)
    out = []
    for el in s.flatten().notesAndRests:
        out.append((None, float(el.quarterLength)) if isinstance(el, note.Rest)
                   else (el.pitch.midi, float(el.quarterLength)))
    return out


def musicxml_to_midi(xml_text_or_path, midi_path, bpm=100):
    src = str(xml_text_or_path)
    s = converter.parse(src, format="musicxml") if src.lstrip().startswith("<") else converter.parse(src)
    s.insert(0, tempo.MetronomeMark(number=bpm))
    s.write("midi", fp=str(midi_path))
    return midi_path
