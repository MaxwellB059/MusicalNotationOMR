"""recognize(): uses the CRNN when present, falls back to geometry otherwise, never crashes silently."""
import numpy as np
import pytest

from omr import crnn_data as D, recognize as R, score as sc


class FakeReader:
    """Stands in for a trained CRNN: returns fixed tokens, ignores pixels."""

    def tokens(self, img):
        return D.tokenize("clef-F4 keySignature-GM timeSignature-2/4 note-F#3_quarter rest-quarter barline "
                          "note-G3_half")


def test_crnn_path_produces_exact_musicxml(tmp_path):
    xml = R.recognize(np.zeros((60, 400, 3), np.uint8), reader=FakeReader())
    p = tmp_path / "a.musicxml"
    p.write_text(xml)
    t = sc.extract(p)
    assert t["clef"] == "BassClef" and t["fifths"] == 1 and t["time"] == (2, 4)
    assert t["events"] == [("note", 54, 1.0), ("rest", None, 1.0), ("note", 55, 2.0)]


def test_empty_prediction_raises_clear_error():
    class Empty:
        def tokens(self, img):
            return ["clef-G2"]
    with pytest.raises(ValueError):
        R.recognize(np.zeros((60, 400, 3), np.uint8), reader=Empty())


def test_crnn_available_false_when_model_missing(tmp_path):
    assert not R.crnn_available(tmp_path)
    (tmp_path / "vocab.json").write_text("[]")
    assert not R.crnn_available(tmp_path)            # vocab alone is not a model
    (tmp_path / "best.onnx").write_bytes(b"x")
    assert R.crnn_available(tmp_path)
