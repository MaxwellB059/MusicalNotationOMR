"""CRNN + CTC model and training for real (camera) images of one line of music. TensorFlow 2.15.

    python -m omr.train_crnn --config configs/crnn.yaml            # GPU machine
    python -m omr.train_crnn --config configs/crnn.yaml --smoke    # CPU pipeline check

Reads a whole line image and emits tokens (clef, key, time, notes, rests, barlines) with no
segmentation, which is why it copes with the dots, hollow heads and camera blur that break the
hand-built pipeline. Tokens go through omr.tokens_to_musicxml to become MusicXML.
"""
import numpy as np

from . import crnn_data as D


def build_models(vocab_size, max_label_len):
    import tensorflow as tf
    L = tf.keras.layers
    K = tf.keras.backend

    image = L.Input((D.HEIGHT, None, 1), name="image")
    x = image
    for filters, pool in [(32, (2, 2)), (64, (2, 2)), (128, (2, 1)), (128, (2, 1))]:
        x = L.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = L.BatchNormalization()(x)
        x = L.MaxPool2D(pool)(x)                      # width shrinks by 4 overall (WIDTH_STRIDE)
    # (B, H/16, W/4, C) -> (B, W/4, H/16*C)
    x = L.Permute((2, 1, 3))(x)
    x = L.Reshape((-1, (D.HEIGHT // 16) * 128))(x)
    x = L.Dense(256, activation="relu")(x)
    x = L.Dropout(0.25)(x)
    x = L.Bidirectional(L.LSTM(192, return_sequences=True, dropout=0.2))(x)
    x = L.Bidirectional(L.LSTM(192, return_sequences=True, dropout=0.2))(x)
    probs = L.Dense(vocab_size, activation="softmax", name="probs")(x)
    infer = tf.keras.Model(image, probs, name="crnn_infer")

    labels = L.Input((max_label_len,), name="labels", dtype="float32")
    in_len = L.Input((1,), name="input_len", dtype="int64")
    lab_len = L.Input((1,), name="label_len", dtype="int64")

    class CTC(L.Layer):
        def call(self, y_true, y_pred, il, ll):
            loss = K.ctc_batch_cost(y_true, y_pred, il, ll)
            self.add_loss(tf.reduce_mean(loss))
            return y_pred

    out = CTC(name="ctc")(labels, probs, in_len, lab_len)
    train = tf.keras.Model([image, labels, in_len, lab_len], out, name="crnn_train")
    train.compile(optimizer=tf.keras.optimizers.Adam(1e-3))
    return train, infer


class Batches:
    """Keras Sequence: variable-width images + padded token ids + CTC lengths."""

    def __init__(self, items, vocab, batch_size, train, max_len, seed=0):
        self.items, self.vocab, self.bs, self.train, self.max_len = items, vocab, batch_size, train, max_len
        self.index = {t: i for i, t in enumerate(vocab)}
        self.rng = np.random.default_rng(seed)
        self.pyrng = __import__("random").Random(seed)
        # one entry per (sample, image) so camera and clean copies are both trained on
        self.entries = [(it, p) for it in items for p in (it[2], it[3]) if p is not None]
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.entries) / self.bs))

    def on_epoch_end(self):
        if self.train:
            self.pyrng.shuffle(self.entries)

    def __getitem__(self, i):
        chunk = self.entries[i * self.bs:(i + 1) * self.bs]
        imgs = [D.load_image(p, augment=self.train, rng=self.pyrng) for _, p in chunk]
        x, widths = D.pad_batch(imgs)
        labels = np.zeros((len(chunk), self.max_len), np.float32)
        ll = np.zeros((len(chunk), 1), np.int64)
        for k, (it, _) in enumerate(chunk):
            ids = [self.index[t] for t in it[1]][: self.max_len]
            labels[k, :len(ids)] = ids
            ll[k, 0] = len(ids)
        il = np.array([[w // D.WIDTH_STRIDE] for w in widths], np.int64)
        return {"image": x, "labels": labels, "input_len": il, "label_len": ll}, np.zeros((len(chunk),), np.float32)


def as_sequence(batches):
    """Wrap Batches as a tf.keras.utils.Sequence (class must be created after TF is imported)."""
    import tensorflow as tf

    class Seq(tf.keras.utils.Sequence):
        def __len__(self):
            return len(batches)

        def __getitem__(self, i):
            return batches[i]

        def on_epoch_end(self):
            batches.on_epoch_end()

    return Seq()


def predict_tokens(infer, vocab, images):
    x, widths = D.pad_batch(images)
    probs = infer.predict(x, verbose=0)
    out = []
    for k, w in enumerate(widths):
        T = max(w // D.WIDTH_STRIDE, 1)
        out.append(D.greedy_decode(probs[k, :T].argmax(-1), vocab))
    return out


def evaluate(infer, vocab, items, limit=None):
    """Token error rate, whole-sequence accuracy and per-field accuracy on (name, tokens, png, jpg) items."""
    n = tot = seq = 0
    per = {"clef": [0, 0], "key": [0, 0], "time": [0, 0]}
    for kind in (3, 2):                                  # camera first, then clean
        pairs = [(it[1], it[kind]) for it in items if it[kind] is not None][: limit or None]
        for i in range(0, len(pairs), 16):
            chunk = pairs[i:i + 16]
            preds = predict_tokens(infer, vocab, [D.load_image(p) for _, p in chunk])
            for (truth, _), pred in zip(chunk, preds):
                n += D.edit_distance(pred, truth)
                tot += len(truth)
                seq += pred == truth
                for field, pref in (("clef", "clef-"), ("key", "keySignature-"), ("time", "timeSignature-")):
                    t = [x for x in truth if x.startswith(pref)]
                    p = [x for x in pred if x.startswith(pref)]
                    per[field][0] += t == p
                    per[field][1] += 1
        if kind == 3:
            cam = {"ter": n / max(tot, 1), "seq_acc": seq / max(len(pairs), 1)}
            n = tot = seq = 0
    clean = {"ter": n / max(tot, 1), "seq_acc": seq / max(len(pairs), 1)}
    return {"camera": cam, "clean": clean, "fields": {k: v[0] / max(v[1], 1) for k, v in per.items()}}
