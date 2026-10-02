# Secundo Roadmap: from one note to a full line of music

**Goal:** upload an image of printed sheet music and get back MusicXML that matches it, with nothing hardcoded.

## Where we are (verified Oct 2026)

- Works for one cropped whole note on a treble staff, one of 7 pitches: A4 A5 B4 C6 D4 F4 G4.
- Hardcoded in the output: whole note, 4/4, treble clef, C major.
- The CNN learns pitch directly from images, so every new pitch (and every new clef) needs its own labelled data.
- Training data is unbalanced: almost all whole notes, half notes only for A5.
- No staff detection, symbol segmentation, or multi-note assembly in the repo.

## Design principles

1. **Classify what a symbol IS, compute where it SITS.**
   The CNN identifies the symbol (duration, rest, accidental). Pitch comes
    from geometry: the notehead's
   vertical position relative to the five staff lines, mapped through the clef. This works for any pitch
   without per-pitch training data.
2. **Everything relative to the staff space** (distance between two staff lines), so it works at any zoom.
3. **Generated, labelled data.** Render music with music21 + MuseScore; the source MusicXML is the answer key.
4. **Every step has a score.** No stage is "done" until a metric on the generated test set says so.

## Supported symbol list (agree on this first, treat everything else as out of scope)

Proposed v1: treble clef (bass clef next), whole/half/quarter/eighth/16th notes, matching rests,
sharp/flat/natural, augmentation dot, barlines, key signatures (sharps/flats), time signatures.

Out of scope for v1: beams spanning groups, ties and slurs, chords, tuplets, ornaments, lyrics, multiple voices.

## Stages

### 1. Data generator + scorer  (everything else depends on this)
- Script that builds random one-line scores with music21, exports MusicXML, renders to PNG (MuseScore CLI or LilyPond).
- Save both the image and the ground-truth XML; also export per-symbol crops with labels.
- `score_line.py`: compares predicted MusicXML to ground truth, reports note-level pitch / duration / overall accuracy.
- Keep the A4 single-note image as a regression test.
- **Done when:** 1,000+ rendered lines with answer keys, scorer runs on them, existing single-note behaviour reproduced.

### 2. Symbol classifier
- Replace the pitch-based CNN with a symbol-type CNN: duration, rests, accidentals, dots.
- Train on generated crops; balance classes; add augmentation (scale, blur, noise, slight rotation).
- Add an "unknown / not a symbol" class so it stops answering confidently on things it has never seen.
- **Done when:** >= 98% per-class accuracy on held-out generated crops, and unknown-class rejects clef/text/junk.

### 3. Staff detection + geometric pitch
- Deskew (reuse `Deskewing/deskew_any_angle.py`), binarize, row-projection to find five evenly spaced peaks.
- Measure staff space and line thickness; return line positions.
- Pitch = notehead centre relative to the lines (+ ledger lines), mapped by the clef.
- **Done when:** pitch is correct on 99%+ of generated notes across the full staff range, including ledger lines.

### 4. Line header: clef, key signature, time signature
- Detect and classify the symbols at the start of the line (clef first, then key, then time).
- Key signature changes which pitches get sharps/flats; time signature drives measure length.
- Replace the hardcoded header in `build_single_note_musicxml` with the detected values.
- **Done when:** header read correctly on 95%+ of generated lines.

### 5. Full line assembly
- Remove staff lines, find connected components, split into clef / header / barlines / symbols (sorted left to right).
- Run each symbol through stages 2-3, group into measures at barlines.
- Generalise the MusicXML builder to take a list of notes/rests per measure.
- Add a line mode to `/api/omr`; keep single-note behaviour working. Add a debug overlay on the site (boxes + labels).
- **Done when:** measured on the generated test set (target to be set by the team after the first run).

### 6. Full pages
- Find multiple staves, group into systems, handle reading order, page-level deskew. Reuses stages 2-5.

## Cross-cutting

- Keep `omr_common.py` as the single home for preprocessing and XML code.
- Regression: the single-note before/after comparison and the A4 image must keep passing until replaced.
- Real-world images (scans, photos) come after clean synthetic ones work; expect a gap and plan for it.
- Pin versions in `requirements.txt` (TensorFlow 2.15 / Keras 2.15) unless models are re-saved.

## Open questions for the team

1. Does anyone already have staff/line detection code that isn't pushed yet? Where is it?
2. Final v1 symbol list (above).
3. Keep a CNN for pitch as a cross-check, or go geometry-only?
4. Which renderer: MuseScore CLI or LilyPond (licensing, install burden on Windows/Mac)?
5. What is the target accuracy for a "working" line, and on what kind of input (rendered, scanned, photographed)?
6. Who owns which stage?
