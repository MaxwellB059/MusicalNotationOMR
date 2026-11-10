"""Split by SCORE (never by symbol/crop) and assert no image duplicates cross splits.

The mid-sem failure was 265 of 273 validation images being copies of training images.
This module makes that impossible to repeat silently.
"""
import hashlib
import random
from pathlib import Path


def split_scores(stems, seed=0, fractions=(0.8, 0.1, 0.1)):
    """Deterministically split a list of score stems into train/val/test."""
    stems = sorted(stems)
    random.Random(seed).shuffle(stems)
    n = len(stems)
    a = int(n * fractions[0])
    b = a + int(n * fractions[1])
    return {"train": stems[:a], "val": stems[a:b], "test": stems[b:]}


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def assert_no_leak(splits, directory, suffix=".png"):
    """Raise AssertionError if any image content appears in more than one split."""
    seen = {}
    for name, stems in splits.items():
        for stem in stems:
            h = file_hash(Path(directory) / (stem + suffix))
            if h in seen and seen[h][0] != name:
                raise AssertionError(f"image {stem} ({name}) duplicates {seen[h][1]} ({seen[h][0]})")
            seen.setdefault(h, (name, stem))
    return True
