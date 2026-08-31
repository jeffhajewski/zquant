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


def test_codec_round_trip():
    x = corpus(500, 64, seed=20)
    with zquant.Codec(dim=64, bits=5) as c:
        codes, norms = c.encode(x)
        assert codes.shape == (500, c.code_bytes) and codes.dtype == np.uint8
        assert norms.shape == (500,) and norms.dtype == np.float32
        back = c.decode(codes, norms)
    assert back.shape == x.shape
    rel = np.linalg.norm(x - back) / np.linalg.norm(x)
    assert rel < 0.1, f"reconstruction error {rel:.3f}"


def test_codec_bits_control_size_and_error():
    """More bits must cost more storage and buy less error, monotonically."""
    x = corpus(400, 64, seed=21)
    sizes, errs = [], []
    for bits in (2, 3, 4, 5):
        with zquant.Codec(dim=64, bits=bits) as c:
            back = c.decode(*c.encode(x))
            sizes.append(c.code_bytes)
            errs.append(float(np.linalg.norm(x - back) / np.linalg.norm(x)))
    assert sizes == sorted(sizes) and len(set(sizes)) == 4, sizes
    assert errs == sorted(errs, reverse=True), errs


def test_codec_beats_int4_on_reconstruction():
    """The KV-cache claim, as a test: at matched storage, beat per-row int4.

    int4 with a per-row scale is what KV quantization usually means in practice. At
    d=64 it costs 36 B/vector (32 packed + a 4-byte scale), which is what Codec(bits=4)
    costs (32 packed + a 4-byte norm).
    """
    x = corpus(1000, 64, seed=22)

    amax = np.abs(x).max(axis=1, keepdims=True)
    scale = np.where(amax > 0, amax / 7.0, 1.0)          # int4 symmetric: 7 levels
    int4 = np.round(x / scale) * scale
    int4_err = np.linalg.norm(x - int4) / np.linalg.norm(x)

    with zquant.Codec(dim=64, bits=4) as c:
        assert c.code_bytes == 32, c.code_bytes        # matched storage with int4
        back = c.decode(*c.encode(x))
    zq_err = np.linalg.norm(x - back) / np.linalg.norm(x)

    assert zq_err < int4_err, f"zquant {zq_err:.4f} not better than int4 {int4_err:.4f}"


def test_codec_errors_bad_code_width():
    x = corpus(50, 32, seed=23)
    with zquant.Codec(dim=32, bits=5) as c:
        codes, norms = c.encode(x)
        with pytest.raises(ValueError, match="codes must be"):
            c.decode(codes[:, :-1], norms)


def test_codec_errors_bad_norm_count():
    x = corpus(50, 32, seed=23)
    with zquant.Codec(dim=32, bits=5) as c:
        codes, norms = c.encode(x)
        with pytest.raises(ValueError, match="norms must be"):
            c.decode(codes, norms[:-1])


def test_codec_errors_bad_dim():
    with zquant.Codec(dim=32, bits=5) as c:
        with pytest.raises(ValueError, match="dim"):
            c.encode(np.zeros((4, 33), dtype=np.float32))
