"""Behaviour a user would rely on, and the mistakes they would actually make."""
import numpy as np
import pytest

import zquant


def corpus(n, d, seed=0, offset=0.0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, d)).astype(np.float32) + offset
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_version():
    assert zquant.version()


def test_self_retrieval():
    x = corpus(2000, 64)
    with zquant.Index(dim=64, bits=5) as ix:
        ix.add(x)
        assert len(ix) == 2000
        ids, scores = ix.search(x[:50], k=10)
        assert ids.shape == (50, 10) and scores.shape == (50, 10)
        # A vector should retrieve itself; quantization may reorder near-ties but not
        # push a vector out of its own top-10.
        assert sum(i in ids[i] for i in range(50)) >= 48
        # Scores are descending within each row.
        assert (np.diff(scores, axis=1) <= 1e-5).all()


def test_recall_against_exact():
    x, q = corpus(5000, 128, seed=1), corpus(200, 128, seed=2)
    exact = np.argsort(-(q @ x.T), axis=1)[:, :10]
    with zquant.Index(dim=128, bits=5) as ix:
        ix.add(x)
        ids, _ = ix.search(q, k=10)
    recall = np.mean([len(set(a) & set(b)) / 10 for a, b in zip(ids, exact)])
    assert recall > 0.85, f"recall {recall:.3f} too low"


def test_calibration_helps_on_offset_data():
    """The centroid rule, as a test: calibration pays when the corpus sits off-origin."""
    x, q = corpus(5000, 128, seed=3, offset=0.85), corpus(200, 128, seed=4, offset=0.85)
    exact = np.argsort(-(q @ x.T), axis=1)[:, :10]

    def recall(calibrate):
        with zquant.Index(dim=128, bits=3) as ix:
            if calibrate:
                ix.calibrate(x[:2000])
            ix.add(x)
            ids, _ = ix.search(q, k=10)
        return np.mean([len(set(a) & set(b)) / 10 for a, b in zip(ids, exact)])

    assert recall(True) > recall(False) + 0.03


def test_threads_agree_with_single():
    x, q = corpus(3000, 64, seed=5), corpus(64, 64, seed=6)
    with zquant.Index(dim=64, bits=5) as ix:
        ix.add(x)
        a, _ = ix.search(q, k=10, threads=1)
        b, _ = ix.search(q, k=10, threads=4)
    assert (a == b).all()


def test_chunks_longer_than_searcher_capacity():
    """More queries than one native call accepts must still come back correct."""
    x = corpus(1000, 32, seed=7)
    q = corpus(500, 32, seed=8)
    with zquant.Index(dim=32, bits=5) as ix:
        ix.add(x)
        ids, _ = ix.search(q, k=5)
        one = np.vstack([ix.search(q[i:i + 1], k=5)[0] for i in range(len(q))])
    assert ids.shape == (500, 5)
    assert (ids == one).all()


def test_fewer_vectors_than_k():
    x = corpus(3, 16, seed=9)
    with zquant.Index(dim=16, bits=5) as ix:
        ix.add(x)
        ids, scores = ix.search(x[:1], k=10)
    assert (ids[0, 3:] == np.iinfo(np.uint32).max).all()
    assert np.isneginf(scores[0, 3:]).all()


def test_errors_are_python_exceptions():
    x = corpus(100, 16)
    with zquant.Index(dim=16, bits=5) as ix:
        with pytest.raises(ValueError, match="dim"):
            ix.add(np.zeros((5, 17), dtype=np.float32))
        with pytest.raises(zquant.ZquantError):
            ix.search(x[:1], k=10)          # empty index
        ix.add(x)
        with pytest.raises(zquant.ZquantError, match="state"):
            ix.calibrate(x)                  # calibrate after add
        with pytest.raises(ValueError):
            ix.search(x[:1], k=0)
    with pytest.raises(ValueError, match="metric"):
        zquant.Index(dim=8, metric="nope")


def test_accepts_non_float32_input():
    x = np.random.default_rng(0).standard_normal((100, 16))   # float64
    with zquant.Index(dim=16, bits=5) as ix:
        ix.add(x)
        assert len(ix) == 100
        ids, _ = ix.search(list(x[0]), k=3)   # a plain Python list, 1-D
        assert ids.shape == (1, 3)
