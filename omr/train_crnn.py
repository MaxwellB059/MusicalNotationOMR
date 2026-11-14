"""Train + evaluate the real-image CRNN on Camera-PrIMuS. See TRAINING_RUNBOOK.md.

    python -m omr.train_crnn --config configs/crnn.yaml [--smoke] [--resume]

Writes <out_dir>/best.keras (inference model), vocab.json, metrics.json. Split is by PIECE so
different incipits of one piece never straddle train and test.
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np

from . import crnn_data as D


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/crnn.yaml")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args(argv)
    import yaml
    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    if a.smoke:
        cfg.update(cfg.get("smoke", {}))
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"])

    from .omr_compat import apply_keras_fix
    apply_keras_fix()
    import tensorflow as tf
    from . import crnn
    tf.random.set_seed(cfg["seed"])

    root = Path(cfg["data_root"]).expanduser()
    items, dropped = D.list_samples(root, limit=cfg.get("limit"))
    print(f"{len(items)} usable samples; dropped {dropped} with unsupported symbols", flush=True)
    vocab = D.build_vocab(items)
    max_len = max(len(it[1]) for it in items)
    split = {"train": [], "val": [], "test": []}
    for it in items:
        split[D.split_of(it[0], seed=cfg["seed"])].append(it)
    if a.smoke:                                          # tiny but non-empty val/test
        split["val"] = split["val"] or split["train"][:4]
        split["test"] = split["test"] or split["train"][:4]
    print({k: len(v) for k, v in split.items()}, "vocab", len(vocab), "max tokens", max_len, flush=True)

    out = Path(cfg["out_dir"] + ("_smoke" if a.smoke else ""))
    out.mkdir(parents=True, exist_ok=True)
    json.dump(vocab, open(out / "vocab.json", "w"))
    train_m, infer = crnn.build_models(len(vocab), max_len)
    ckpt = out / "best.keras"
    if a.resume and ckpt.exists():
        infer.load_weights(str(ckpt))                    # weights are shared with the training model
    tr = crnn.as_sequence(crnn.Batches(split["train"], vocab, cfg["batch_size"], True, max_len, cfg["seed"]))
    va = crnn.as_sequence(crnn.Batches(split["val"], vocab, cfg["batch_size"], False, max_len, cfg["seed"]))
    best = {"v": 1e9}

    class Save(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            v = (logs or {}).get("val_loss", 1e9)
            if v < best["v"]:
                best["v"] = v
                infer.save(str(ckpt))

    train_m.fit(tr, validation_data=va, epochs=cfg["epochs"], verbose=2,
                callbacks=[Save(), tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=cfg["patience"],
                                                                     restore_best_weights=True)])
    infer.save(str(ckpt)) if not ckpt.exists() else None
    metrics = crnn.evaluate(infer, vocab, split["test"], limit=cfg.get("eval_limit"))
    metrics["gates"] = {"camera_ter_max": {"threshold": 0.15, "value": metrics["camera"]["ter"],
                                           "pass": metrics["camera"]["ter"] <= 0.15},
                        "camera_seq_acc_min": {"threshold": 0.5, "value": metrics["camera"]["seq_acc"],
                                               "pass": metrics["camera"]["seq_acc"] >= 0.5}}
    json.dump(metrics, open(out / "metrics.json", "w"), indent=2)
    print(json.dumps({k: metrics[k] for k in ("camera", "clean", "fields")}))
    for k, g in metrics["gates"].items():
        print(f"GATE {k}: {g['value']:.3f} ({g['threshold']}) {'PASS' if g['pass'] else 'FAIL'}")


if __name__ == "__main__":
    main()
