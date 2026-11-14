"""CRNN data layer + tokens -> MusicXML: exact round trips (no TensorFlow), plus a TF smoke train."""
import importlib.util
import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest

from omr import crnn_data as D, primus, score as sc, tokens

HAVE_TF = importlib.util.find_spec("tensorflow") is not None
ROOT = pathlib.Path(__file__).resolve().parents[2]
CORPUS = pathlib.Path.home() / "datasets" / "camera_primus" / "Corpus"

SAMPLE = ("clef-G2\tkeySignature-EbM\ttimeSignature-3/4\tnote-Bb4_quarter\trest-eighth\tnote-C5_eighth\tbarline\t"
          "note-C5_half.\tbarline\tnote-D4_thirty_second\tnote-D4_sixteenth\ttie\tnote-D4_eighth")


def test_tokenize_detokenize_roundtrip():
    toks = D.tokenize(SAMPLE)
    assert toks[:5] == ["clef-G2", "keySignature-EbM", "timeSignature-3/4", "n:Bb4", "d:quarter"]
    assert D.detokenize(toks) == SAMPLE.replace("\t", "\t")


def test_unsupported_samples_are_dropped_not_guessed():
    for bad in ("note-C4_quarter gracenote-D4_eighth", "multirest-8", "note-C4_quarter_fermata", "note-C4_eighth.."):
        assert D.tokenize(bad) is None


def test_tokens_to_musicxml_matches_truth_exactly(tmp_path):
    xml = tokens.tokens_to_musicxml(D.tokenize(SAMPLE))
    p = tmp_path / "a.musicxml"
    p.write_text(xml)
    got = sc.extract(p)
    truth = primus.parse_semantic(SAMPLE)
    assert got["clef"] == truth["clef"] and got["fifths"] == truth["fifths"] and got["time"] == truth["time"]
    assert got["events"] == truth["events"]
    assert '<tie type="start"' in xml and '<tie type="stop"' in xml and "<dot" in xml


def test_bad_sequences_are_flagged_not_crashing():
    measures, clef, fifths, time, flags = tokens.tokens_to_measures(["n:C4", "barline", "d:half", "r"])
    assert any("stray" in f for f in flags) and any("4/4" in f for f in flags)
    with pytest.raises(ValueError):
        tokens.tokens_to_musicxml(["clef-G2"])


def test_greedy_ctc_decode_collapses_repeats_and_blanks():
    vocab = ["a", "b", "<blank>"]
    assert D.greedy_decode([0, 0, 2, 0, 1, 1, 2, 2, 1], vocab) == ["a", "a", "b", "b"]


def test_edit_distance_and_split_by_piece():
    assert D.edit_distance(list("kitten"), list("sitting")) == 3
    assert D.split_of("000051650-1_1_1") == D.split_of("000051650-1_1_2")      # same piece, same split
    splits = {D.split_of(f"{i:09d}-1_1_1") for i in range(2000)}
    assert splits == {"train", "val", "test"}


def test_pad_batch_width_multiple_of_stride():
    x, w = D.pad_batch([np.ones((64, 10), np.float32), np.ones((64, 17), np.float32)])
    assert x.shape == (2, 64, 20, 1) and w == [10, 17] and x[0, :, 10:, 0].sum() == 0


@pytest.mark.skipif(not (HAVE_TF and CORPUS.exists()), reason="needs TensorFlow and the Camera-PrIMuS download")
def test_smoke_training_on_real_samples():
    r = subprocess.run([sys.executable, "-m", "omr.train_crnn", "--config", "configs/crnn.yaml", "--smoke"],
                       cwd=ROOT, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-1500:]
    assert "GATE camera_ter_max" in r.stdout
    assert (ROOT / "runs" / "crnn_smoke" / "best.keras").exists()
