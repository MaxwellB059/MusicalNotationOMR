# Secundo: Full Build Plan (slides scope), parallel-agent ready

> **For agents:** this file is the source of truth. Pick a track from "Critical path & parallel tracks", work in your own git worktree/branch, touch only files your track owns, and do not start a step until its upstream gate has passed. Each step lists its test suite; run it and report the gate numbers before merging. Stop and ask the human for anything under "Decisions needed". Training runs follow `TRAINING_RUNBOOK.md`.


## Context
Secundo turns a photo of printed music into MusicXML (opens/plays in MuseScore). Today: one cropped whole note -> one of 7 pitches (`omr_server.py`, `single_note_cnn.py`, `omr_common.py`); clef/key/time/duration hardcoded in `build_single_note_musicxml`. No staff detection, segmentation, assembly, scorer, or tests. The 99% val accuracy is invalid (265/273 val images duplicate train).
Mid-sem slides + A2 report promise: every clef/time sig/note/rest, **plus ties, hairpins, accents, dynamics**, playback as stretch. Design from slides: *geometry finds the staff, learned models name symbols, rules compute pitch/duration, homr is the benchmark.* This plan makes that real, with a test gate after every large step.

Decisions already made by user: LilyPond renders synthetic data (automation); MuseScore is the end-user target and final "opens correctly" check. Follow slides exactly (A2 report's YOLO/DeepScoresV2 detector is the symbol detector for marks); feasibility below.

## Scope tiers (what "done" means)
- **T1 Notes:** clefs (treble, bass, alto), key sigs, time sigs, whole..16th notes + matching rests, accidentals, dots, barlines, one staff.
- **T2 Marks (slide promise):** accents/staccato, ties, dynamics (p..ff), hairpins (cresc/decresc).
- **T3 Page:** multi-staff, systems, grand staff, real phone photos.
- **T4 Stretch:** playback (MusicXML->MIDI->audio), review/edit UI.
Out of scope: chords/beams-over-groups/tuplets/lyrics/voices -> T3+ only if time (beams needed for 8th/16th: handled via stem+beam analysis in S5).

## Stack
- Python 3.10, FastAPI/uvicorn (existing), OpenCV + NumPy (geometry), TensorFlow 2.15/Keras (existing symbol CNN; Windows-CPU only, pinned), **music21** (generation + MusicXML parse/compare), **LilyPond** (render), **pytest** (+ httpx TestClient, already in requirements), **PyTorch + Ultralytics YOLO** in a **separate env** (marks detector; TF 2.15 and torch must not share the venv), React/Vite frontend (existing `src/components`), Vitest for frontend, GitHub Actions CI (CPU, small fixtures only).
- Package layout (Jira OMR-3): `omr/{preprocess,staff,symbols,marks,semantics,assemble,score}.py`, `omr_server.py` thin, `tests/{unit,integration,regression}`, `data/synth/` (git-ignored, regenerated), `tests/fixtures/` (tiny, committed), `benchmarks/` (homr runs).
- Keep `omr_common.py` as the single preprocessing/XML home until moved into `omr/` (re-export for compat).

## What YOLO means (for the team)
YOLO = "You Only Look Once": a real-time **object detector**. Given a whole image it outputs, in one pass, many boxes each with a class and confidence ("dynamic-mf at x,y,w,h"). Our CNN only classifies an already-cropped single symbol; a detector *finds and classifies* symbols anywhere on a page, which is why it suits dynamics/hairpins/ties (page positions, not sequence slots - the thing homr fails at). Version: Ultralytics YOLOv8/11 (easy training API) vs the Faster R-CNN baseline in the DeepScoresV2 paper.

## Feasibility estimate: detector for marks (Track D)
| Factor | Finding | Verdict |
|---|---|---|
| Compute | No NVIDIA GPU on this machine; TF 2.15 has no native Windows GPU | Train on Colab/Kaggle/campus GPU, not locally. Inference on CPU is OK (seconds/page) |
| Disk | Local C: has ~13 GB free (98% full) | DeepScoresV2 *dense* subset (~1.7k pages, 135 classes) is feasible; full ~255k-page set is not. Store data on external/cloud, verify sizes before download |
| Data | Only DeepScoresV2 labels cover dynamics/hairpins/ties; paper baseline 0.70 mAP overall, ties 0.698, hairpin/dynamics classes rare | Medium risk: rare classes will be weak; mitigate with synthetic LilyPond pages containing marks (we control the labels) |
| Domain gap | DeepScores pages are synthetic; phone photos differ | Augmentation (blur/rotate/perspective/JPEG) + T3 photo eval; expect a gap |
| Effort | Env + data conversion to YOLO format ~2-3 days; first model ~1 day of GPU; iterate 1-2 weeks | **Feasible for a semester if scoped to ~8 mark classes**; **not** feasible for all 135 classes |
| Fallback | Classical rules for marks: dynamics = text-like glyph clusters below staff -> small CNN; hairpin = two converging line segments (Hough); tie = arc between two noteheads | Cheaper, good on clean renders; use if detector misses gate |
**Recommendation:** train YOLO on a mark-only class list (dyn p/mp/mf/f/ff, cresc/decresc hairpin, tie, accent, staccato) using synthetic LilyPond + DeepScoresV2 dense. Gate it with a go/no-go checkpoint (D-gate) vs the classical fallback. Core notes pipeline (T1) does NOT depend on YOLO: it uses staff geometry + the existing-style CNN, per the slides.

## Critical path & parallel tracks
Everything blocks on S0. After S0, tracks run in parallel; merge points are the numbered gates. Each agent owns disjoint files (interfaces below) to avoid conflicts; work in separate git worktrees.

```
S0 Foundation+Data+Scorer ──► (gate G0)
   ├─ Track A: S1 Preprocess ─► S2 Staff detect/removal ─► S3 Geometric pitch ──┐
   ├─ Track B: S4 Symbol CNN (needs only S0 crops)  ───────────────────────────┤
   ├─ Track C: S5 Header (clef/key/time) (needs S2 + S4 interface) ────────────┤
   ├─ Track D: S6 Marks detector/fallback (needs S0 data; independent of A-C)──┤
   └─ Track E: S8 API/Frontend skeleton against mocked pipeline ───────────────┤
                                                                   S7 Assembly (needs S3,S4,S5) ─► G7
                                  G7 + S6 ─► S9 Marks attach ─► S10 Pages/photos ─► S11 Playback ─► S12 Hardening
```
**Critical path (longest dependency chain):** S0 -> S1 -> S2 -> S3 -> S7 -> S9 -> S10. Track B/C/D finish in parallel off it; D is the longest-running (GPU), so start it right after G0.

## Steps, with test suites and gates
Rule: a step is merged only when its suite is green in CI and its gate metric is met on the **held-out, source-split** set (never random crop split; generate train/val/test by *score*, not by symbol, to avoid the 265/273 leak).

**S0 Foundation + data + scorer (blocks all).**
Build: install LilyPond+music21; `omr/synth.py` random score generator (seeded, config for clef/key/time/rhythm/marks) -> MusicXML + LilyPond -> PNG + per-symbol boxes/labels (from LilyPond SVG/`-dpixmap` or LilyPond point-and-click positions); photo-degradation augmenter (rotate, perspective, blur, noise, JPEG, lighting); `omr/score.py` comparing predicted vs truth MusicXML (per-feature: pitch, duration, rest, clef, key, time, accent, tie, dynamic, hairpin; precision/recall/F1 + note-sequence edit distance; same table format as the homr baseline); dedupe/split-by-score utility; package restructure; pytest+CI; keep A4 single-note image as regression fixture.
Tests: generator determinism (same seed -> same output); every generated XML validates and round-trips through music21; scorer unit tests with hand-made pairs (identical=1.0, one wrong pitch, dropped tie, extra note); split test asserts zero image-hash overlap between train/val/test; augmenter bounds.
Gate G0: 1,000+ lines with answer keys; scorer = 1.0 on truth-vs-truth; leak check passes; existing single-note regression passes; homr run on 13 baseline pages reproduced through the scorer (benchmark column).

**S1 Preprocess (Track A).** Reuse `Deskewing/deskew_any_angle.py`, `preprocess_bgr_to_tensor`; add PDF/PNG/JPG intake, scale normalise, binarize, margin crop.
Tests: synthetic skew +-15 deg recovered to <=0.5 deg; output binary/size invariants; PDF multi-page; corrupt file -> clear error. Gate: >=95% of augmented test images deskewed within 0.5 deg.

**S2 Staff detection + removal (Track A).** Row-projection peaks -> 5 lines, staff-space `d`, line thickness; barline/system grouping later; staff removal preserving stems/noteheads (run-length/morphology); per-staff crops.
Tests: line y-positions vs LilyPond ground truth (tol 0.1d); lines-per-staff count; removal doesn't delete notehead pixels (pixel-overlap vs truth mask); rotated/blurred variants. Gate: staves found 99% clean, >=95% on augmented; staff-space error <5%.

**S3 Geometric pitch (Track A) - answers the slides' "next question".** notehead centre (CC after removal, or ellipse fit) -> staff position (ledger lines included) -> `pitch = lookup(pos, clef) + key`; accidentals carried within a measure.
Tests: lookup tables exhaustive (treble/bass/alto, positions -6..+14) vs music21; property test: any generated pitch round-trips; ledger line cases; sharps/flats/naturals; **never-seen pitches** (hold out pitches that the old CNN trained on). Compare vs homr on same pages. Gate G3 (slides decision point): >=99% pitch on clean generated notes across full range; >=95% on augmented; decision recorded: geometry Defer -> Adopt/Reject.

**S4 Symbol CNN (Track B).** Retarget `single_note_cnn.py` to symbol type (note head type, rests, accidentals, dots, clef types, time-sig digits, barline) + **"unknown/not-a-symbol"** class; train on S0 crops with augmentation, balanced classes; keep TF/Keras so the server loads it as today; fix Keras Windows loader via `apply_windows_keras_fix`.
Tests: per-class confusion matrix on held-out *score-split* set; unknown-class rejects clefs/text/junk blobs from segmentation; perturbation robustness (reuse 90.5% perturbed protocol, now on clean split); label.json schema test; model loads in server. Gate: >=98% per class (core), >=95% under augmentation, junk-reject >=95%. Also tests CNN-on-clefs (slides' pending test) vs the 4/10 LLM result.

**S5 Header reader (Track C).** Clef (CNN), key signature (count/type sharps/flats near clef -> fifths), time signature (digits -> beats/beat-type); replaces hardcoded values in XML builder.
Tests: header read exact-match on generated lines (all clefs x 15 keys x common time sigs); change-of-key mid-line (stretch); tampered header -> flagged not guessed. Gate: >=95% exact header; failures flagged "needs review" not silently wrong.

**S6 Marks (Track D, independent, GPU).** Mark classes: accent, staccato, tie, dynamics, hairpins. Step 1: classical fallback (Hough hairpins, arc detect for ties, dynamic-glyph CNN) - cheap baseline. Step 2: YOLO (Ultralytics, separate env) on synthetic LilyPond marks pages + DeepScoresV2 dense; export ONNX for CPU inference.
Tests: mAP@0.5 per class on held-out score-split; per-class recall vs homr's 0/10, 0/4, 0/2, 5/12; false-positive rate on mark-free pages; inference-time budget (<10 s/page CPU); ONNX output parity with torch. Gate D: pick YOLO if beats fallback by a margin on dynamics+hairpins+ties and recall >=0.8 on each; else ship fallback and record it.

**S7 Assembly (needs S3, S4, S5).** Segment (CC left-to-right), group into measures by barlines, durations from head type + stem/flag/beam/dots, rests, rule check "measure sums to time signature" (flags errors; Jira OMR-24); generalised MusicXML builder (replace single-note builder; `<attributes>` per line, multiple notes/rests/measures, ties/articulations/dynamics/direction elements).
Tests: scorer end-to-end on generated lines (pitch/duration/overall); measure-sum validator unit tests; MusicXML schema validation (XSD) + music21 parse; round-trip truth XML -> render -> pipeline -> compare; regression: A4 single image still returns A4. Gate G7: target set by team after first run (proposed: pitch >=95%, rhythm >=90%, note-level F1 >=0.9 on clean lines; compare to homr 1.00 and report gap honestly).

**S8 API + frontend (Track E, mockable from day one).** `/api/omr` stays backward compatible (single image -> XML); add line/page mode, structured errors, job ID + status endpoint for long pages (Jira OMR-30), debug overlay (boxes+labels), score preview, download. Frontend in `src/components/` (existing `FileUpload`, `ImagePreview`, `ConversionActions`).
Tests: FastAPI TestClient (valid/invalid/oversize/non-image, backward-compat shape); contract test with mocked pipeline so Track E is independent; Vitest component tests; Playwright smoke (upload -> preview -> download). Gate: contract test green with mock, then with real pipeline.

**S9 Attach marks to notes.** Rules: tie links two same-pitch notes; hairpin spans a note range; dynamic applies from its x onward; accent/staccato to nearest notehead. Tests: attachment unit tests on synthetic layouts; scorer per-feature F1 for ties/dynamics/hairpins/accents vs homr. Gate: beats homr on each mark feature (homr = 0), overall pitch/rhythm regression <=1 pt.

**S10 Pages + photos (T3).** Multi-staff, systems/braces/grand staff, reading order, page-level deskew, real phone photos (collect 20-30 real photos, with ground-truth XML).
Tests: staff count/system grouping on generated full pages; per-page scorer; real-photo eval reported separately (expect gap). Gate: staff/system count 99% clean; photo metrics reported, with target agreed by team.

**S11 Playback (stretch).** MusicXML -> MIDI (music21) -> in-browser audio (Tone.js/soundfont). Tests: MIDI note numbers/durations vs truth (C#4 = Db4 = 61); duration total equals measure sums.

**S12 Hardening.** Error-analysis report grouped by stage (Jira OMR-33), CI regression suite, README/arch docs, demo.

## Final test strategy (after build)
1. **Unit** per module (above). 2. **Integration** per step interface. 3. **End-to-end scoring** on three sets: (a) generated clean (b) generated+degraded "simulated photos" (c) real pages/photos with hand/MuseScore-verified ground truth. 4. **Benchmark vs homr** (and oemer/Audiveris if run) on the same pages, same per-feature table as `A2_REPORT.md`. 5. **Leak control:** split by score, hash-dedupe check in CI. 6. **Ablations** to prove each design choice: geometry vs CNN pitch; YOLO vs classical marks; with/without measure-sum validator. 7. **Robustness sweep:** rotation/scale/blur/JPEG/lighting curves per metric. 8. **Human check:** MusicXML opens in MuseScore, sound check on 10 pages; 3-person blind review of failures. 9. **Silent-failure audit:** every stage emits confidence/"needs review"; test that bad inputs produce flags, not wrong output (lesson from dropped half notes / 4-4 read as 5-4). 10. **Report** using same hypothesis/evidence/decision format as the dossier so results feed the final presentation.

## Training handoff (user trains on a teammate's GPU machine via a remote Claude Code instance)
We do NOT train here. We deliver everything ready-to-run, committed and pushed so the remote machine only does `git fetch` and points its Claude Code at `TRAINING_RUNBOOK.md`:
- `TRAINING_RUNBOOK.md`: step-by-step instructions written for an autonomous Claude Code agent: env setup (separate TF env and torch/Ultralytics env, pinned `requirements-train-*.txt`), GPU sanity check, data generation command (`python -m omr.synth --n 5000 --seed 1 --out data/synth`), DeepScoresV2-dense download + convert script, train commands (symbol CNN, YOLO marks), eval commands that print gate tables, expected outputs/paths, pass/fail thresholds, what to commit back (`runs/*/best.*`, `labels.json`, metrics JSON, confusion matrices; large files via release/LFS/Drive), and stop-and-ask conditions.
- Scripts are config-driven (`configs/*.yaml`), deterministic (seeded), resumable, with a `--smoke` flag (tiny run, 2 min, CPU) that CI and I run here to prove the pipeline works before the GPU run.
- Everything verified on CPU here with `--smoke`; real training is the only thing left undone.

## Decisions needed from you (will ask, not guess)
- Real-photo ground truth: who photographs/annotates pages (T3).
- Compute for YOLO: Colab/Kaggle/campus GPU access and where to store DeepScoresV2 (C: has ~13 GB free).
- Team targets for G7/S10 after the first scorer run.
- Whether chords/beam groups/multi-voice are in or out of final scope.
- Stage owners per teammate (Anagha / Elijah / Max).

## Verification of this plan's execution
Run order for each step: `pytest tests/unit/<step>` -> `pytest tests/integration` -> `python -m omr.score --set <name>` prints the gate table; CI blocks merge if gate fails. End-to-end smoke: `launch.bat`, upload a generated line, confirm MusicXML opens in MuseScore.
