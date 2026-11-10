# TRAINING_RUNBOOK (for a Claude Code agent on a GPU machine)

Goal: train the Secundo models and report gate metrics. Read `PLAN.md` first (steps S4 symbol CNN, S6 marks detector). Work autonomously; stop and ask the human only under "Stop and ask".

> Status: **template.** Commands marked `(pending)` depend on code that the build agents add (`omr/synth.py`, `omr/score.py`, `configs/`, `--smoke`). Do not run a pending command until its file exists; check `git log` / `ls omr configs` first.

## 0. Preconditions
```bash
git fetch && git checkout elijah/roadmap-plan && git pull
nvidia-smi                      # must list a GPU; if not, stop and ask
python --version                # 3.10 expected
```
Disk: need ~30 GB free (DeepScoresV2 dense + synthetic data). Check with `df -h`.

## 1. Environments (two, never merged)
```bash
# A. symbol CNN (TensorFlow 2.15 / Keras 2.15; repo is pinned to this)
python -m venv .venv-tf && source .venv-tf/bin/activate   # Windows: .venv-tf\Scripts\activate
pip install -r requirements.txt music21 pytest              # (pending) requirements-train-tf.txt
# NOTE: TF 2.15 has no native Windows GPU. On Windows use WSL2 or Linux, else it trains on CPU.

# B. marks detector (PyTorch + Ultralytics)
python -m venv .venv-yolo && source .venv-yolo/bin/activate
pip install torch torchvision ultralytics onnx onnxruntime  # (pending) requirements-train-yolo.txt
python -c "import torch; print(torch.cuda.is_available())"  # must print True
```
LilyPond is required for data generation: install it (`apt install lilypond` / choco / zip) and confirm `lilypond --version`.

## 2. Smoke test everything first (CPU, ~2 min)  (pending)
```bash
pytest tests/unit -q
python -m omr.synth --n 20 --seed 0 --out data/smoke
python -m omr.train_symbols --config configs/symbols.yaml --smoke
python -m omr.train_marks   --config configs/marks.yaml   --smoke
```
All must pass before spending GPU time.

## 3. Data  (pending)
```bash
python -m omr.synth --n 5000 --seed 1 --out data/synth      # split is BY SCORE; script asserts zero image-hash overlap
python tools/fetch_deepscores_dense.py --out data/deepscores # then tools/deepscores_to_yolo.py ; check sizes before download
```

## 4. Train  (pending)
```bash
# .venv-tf
python -m omr.train_symbols --config configs/symbols.yaml     # -> runs/symbols/best.keras + labels.json + metrics.json
# .venv-yolo
python -m omr.train_marks --config configs/marks.yaml         # -> runs/marks/best.pt, best.onnx, metrics.json
```
Training is resumable (`--resume`). Seeds are fixed in the configs.

## 5. Evaluate against gates
```bash
python -m omr.score --set synth_test --model runs/symbols      # prints gate table (PLAN.md S4)
python -m omr.eval_marks --weights runs/marks/best.onnx        # per-class recall/mAP vs homr (0/10 dyn, 0/4 hairpin, 0/2 tie)
```
Pass thresholds: symbol CNN >=98% per core class, >=95% under augmentation, junk-reject >=95%. Marks: pick YOLO only if recall >=0.8 on dynamics, hairpins and ties and it beats the classical fallback; otherwise report that and keep the fallback.

## 6. Return results
Commit only small artifacts: `runs/*/labels.json`, `metrics.json`, confusion matrices, `best.keras` / `best.onnx` if < 50 MB (otherwise upload to the team Drive and put the link in `runs/README.md`). Push to a new branch `train/<date>`; never to `main`.

## Stop and ask
GPU missing or unused; disk < 30 GB; LilyPond cannot be installed; a gate is missed by > 5 points after one retune; any step needs credentials, paid services or a download > 20 GB.
