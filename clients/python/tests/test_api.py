"""Parallel add, saving and loading, reranking, scoring chosen ids, and automatic
calibration."""
import os

import numpy as np
import pytest

import zquant


def unit_rows(n, d, seed=0, offset=0.0):
    x = np.random.default_rng(seed).standard_normal((n, d)).astype(np.float32) + offset
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def recall(ids, exact):
    k = exact.shape[1]
    return np.mean([len(set(a) & set(b)) / k for a, b in zip(ids, exact)])


# ── add(threads=) ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("threads", [0, 2, 4, 7])
def test_parallel_add_builds_the_same_index(threads):
    x = unit_rows(3000, 48, seed=1)
    with zquant.Index(dim=48, bits=4) as one, zquant.Index(dim=48, bits=4) as many:
        one.add(x)
        many.add(x, threads=threads)
        assert len(many) == 3000
        assert many.to_bytes() == one.to_bytes()


def test_parallel_add_appends_after_existing_vectors():
    x = unit_rows(2000, 32, seed=2)
    with zquant.Index(dim=32) as a, zquant.Index(dim=32) as b:
        a.add(x)
        b.add(x[:700])
        b.add(x[700:], threads=3)
        assert a.to_bytes() == b.to_bytes()


# ── to_bytes / from_bytes / save / load ─────────────────────────────────────────────


@pytest.mark.parametrize("metric, compact, calibrate", [
    ("ip", True, False), ("cosine", True, True), ("l2", False, False),
])
def test_round_trip_searches_and_grows_identically(metric, compact, calibrate):
    x, q = unit_rows(1500, 40, seed=3, offset=0.4), unit_rows(30, 40, seed=4)
    more = unit_rows(200, 40, seed=5)
    with zquant.Index(dim=40, bits=5, metric=metric, compact=compact) as ix:
        if calibrate:
            ix.calibrate(x[:500])
        ix.add(x)
        data = ix.to_bytes()
        assert isinstance(data, bytes)
        with zquant.Index.from_bytes(data) as back:
            assert back.dim == 40 and len(back) == 1500
            assert back.bytes_per_vector == ix.bytes_per_vector
            assert back.to_bytes() == data
            for got, want in zip(back.search(q, k=10), ix.search(q, k=10)):
                assert np.array_equal(got, want)
            # A loaded index accepts new vectors exactly as the saved one does.
            ix.add(more)
            back.add(more)
            assert back.to_bytes() == ix.to_bytes()


@pytest.mark.parametrize("wrap", [bytes, bytearray, memoryview,
                                  lambda b: np.frombuffer(b, dtype=np.uint8)])
def test_from_bytes_takes_any_bytes_like(wrap):
    with zquant.Index(dim=16) as ix:
        ix.add(unit_rows(50, 16))
        data = ix.to_bytes()
    with zquant.Index.from_bytes(wrap(data)) as back:
        assert len(back) == 50


def test_save_and_load(tmp_path):
    x, q = unit_rows(800, 24, seed=6), unit_rows(10, 24, seed=7)
    path = tmp_path / "corpus.zq"
    with zquant.Index(dim=24) as ix:
        ix.add(x)
        ix.save(path)
        ix.save(str(path))                       # str paths too, and overwriting
        want = ix.search(q, k=5)
    assert sorted(os.listdir(tmp_path)) == ["corpus.zq"]      # no temp file left behind
    for p in (path, str(path)):
        with zquant.Index.load(p) as back:
            got = back.search(q, k=5)
            assert np.array_equal(got[0], want[0]) and np.array_equal(got[1], want[1])


def test_empty_index_round_trips():
    with zquant.Index(dim=8, bits=3) as ix:
        with zquant.Index.from_bytes(ix.to_bytes()) as back:
            assert len(back) == 0 and back.dim == 8


def test_damaged_data_is_a_format_error(tmp_path):
    with zquant.Index(dim=16) as ix:
        ix.add(unit_rows(40, 16))
        data = ix.to_bytes()
    flipped = bytearray(data)
    flipped[len(flipped) // 2] ^= 0x01
    for bad in (b"", b"not an index at all", data[:-1], data[: len(data) // 2],
                bytes(flipped)):
        with pytest.raises(zquant.FormatError) as e:
            zquant.Index.from_bytes(bad)
        assert isinstance(e.value, ValueError)
    junk = tmp_path / "junk.zq"
    junk.write_bytes(os.urandom(4096))
    with pytest.raises(zquant.FormatError):
        zquant.Index.load(junk)
    with pytest.raises(FileNotFoundError):
        zquant.Index.load(tmp_path / "missing.zq")


# ── search_rerank ──────────────────────────────────────────────────────────────────


def test_rerank_recovers_recall_with_exact_scores():
    x, q = unit_rows(5000, 64, seed=8), unit_rows(100, 64, seed=9)
    exact_scores = q @ x.T
    exact = np.argsort(-exact_scores, axis=1)[:, :10]
    with zquant.Index(dim=64, bits=3) as ix:
        ix.add(x)
        plain, _ = ix.search(q, k=10)
        ids, scores = ix.search_rerank(q, x, k=10, candidates=80)
    assert ids.shape == scores.shape == (100, 10)
    assert recall(ids, exact) > recall(plain, exact)
    assert recall(ids, exact) > 0.95
    # Scores are exact inner products against the originals, best first.
    np.testing.assert_allclose(scores, np.take_along_axis(exact_scores, ids.astype(np.int64),
                                                          axis=1), rtol=1e-5, atol=1e-6)
    assert (np.diff(scores, axis=1) <= 0).all()


def test_rerank_uses_the_index_metric():
    x, q = unit_rows(500, 16, seed=10, offset=0.3) * 3, unit_rows(5, 16, seed=11)
    with zquant.Index(dim=16, metric="l2") as ix:
        ix.add(x)
        ids, scores = ix.search_rerank(q, x, k=4, candidates=40)
    d2 = ((q[:, None, :] - x[ids.astype(np.int64)]) ** 2).sum(axis=2)
    np.testing.assert_allclose(scores, -d2, rtol=1e-4, atol=1e-5)


def test_rerank_threads_and_long_batches_agree():
    x, q = unit_rows(1000, 32, seed=12), unit_rows(300, 32, seed=13)
    with zquant.Index(dim=32) as ix:
        ix.add(x)
        a = ix.search_rerank(q, x, k=5)
        b = ix.search_rerank(q, x, k=5, threads=4)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_rerank_pads_like_search():
    x = unit_rows(3, 8, seed=14)
    with zquant.Index(dim=8) as ix:
        ix.add(x)
        ids, scores = ix.search_rerank(x[:1], x, k=5, candidates=5)
    assert (ids[0, 3:] == np.iinfo(np.uint32).max).all()
    assert np.isneginf(scores[0, 3:]).all()
    assert ids[0, 0] == 0


def test_rerank_argument_errors():
    x = unit_rows(100, 8, seed=15)
    with zquant.Index(dim=8) as ix:
        ix.add(x)
        with pytest.raises(ValueError, match="candidates"):
            ix.search_rerank(x[:1], x, k=10, candidates=5)
        with pytest.raises(ValueError, match="originals has 99 rows"):
            ix.search_rerank(x[:1], x[:99], k=3)
        with pytest.raises(ValueError, match="dim"):
            ix.search_rerank(x[:1], np.zeros((100, 9)), k=3)


# ── score_ids ──────────────────────────────────────────────────────────────────────


def test_score_ids_matches_what_search_reports():
    x, q = unit_rows(2000, 32, seed=16), unit_rows(5, 32, seed=17)
    with zquant.Index(dim=32, bits=4) as ix:
        ix.add(x)
        ids, scores = ix.search(q, k=20)
        for i in range(len(q)):
            got = ix.score_ids(q[i], ids[i])
            assert got.dtype == np.float32 and got.shape == (20,)
            np.testing.assert_allclose(got, scores[i], rtol=1e-5, atol=1e-6)
        # Any order, repeats, any integer type or a plain list.
        chosen = [17, 3, 3, 1999, 0]
        got = ix.score_ids(q[0], chosen)
        np.testing.assert_array_equal(got, ix.score_ids(q[0], np.array(chosen, np.int64)))
        assert got[1] == got[2]


def test_score_ids_errors():
    x = unit_rows(10, 8, seed=18)
    with zquant.Index(dim=8) as ix:
        ix.add(x)
        assert ix.score_ids(x[0], []).shape == (0,)
        for bad in ([10], [-1], [0, 11]):
            with pytest.raises(IndexError, match="out of range"):
                ix.score_ids(x[0], bad)
        with pytest.raises(TypeError, match="integers"):
            ix.score_ids(x[0], [0.5])
        with pytest.raises(ValueError, match="one query"):
            ix.score_ids(x[:2], [0])
        q = x[0].copy()
        q[1] = np.nan
        with pytest.raises(zquant.InvalidArgumentError, match="NaN"):
            ix.score_ids(q, [0])


# ── calibrate(auto=True) ───────────────────────────────────────────────────────────


def centroid_norm(x):
    u = x / np.linalg.norm(x, axis=1, keepdims=True)
    return float(np.linalg.norm(u.mean(axis=0)))


def test_auto_calibration_applies_on_off_origin_data():
    x = unit_rows(3000, 64, seed=19, offset=0.85)
    with zquant.Index(dim=64, bits=3) as ix:
        got = ix.calibrate(x[:2000], auto=True)
        assert isinstance(got, zquant.Calibration)
        assert got.applied is True
        assert got.centroid_norm == pytest.approx(centroid_norm(x[:2000]), rel=1e-3)
        assert got.centroid_norm >= 0.3


def test_auto_calibration_skips_centred_data():
    x = unit_rows(3000, 64, seed=20)
    with zquant.Index(dim=64, bits=3) as ix, zquant.Index(dim=64, bits=3) as plain:
        got = ix.calibrate(x[:2000], auto=True)
        assert got.applied is False and got.centroid_norm < 0.3
        # Skipped means skipped: the index is the same as one never calibrated.
        ix.add(x)
        plain.add(x)
        assert ix.to_bytes() == plain.to_bytes()


def test_auto_calibration_fails_like_calibrate():
    x = unit_rows(100, 16, seed=21)
    with zquant.Index(dim=16) as ix:
        ix.add(x)
        with pytest.raises(zquant.StateError):
            ix.calibrate(x, auto=True)
        with pytest.raises(zquant.StateError):
            ix.calibrate(x)


def test_manual_calibration_returns_none():
    with zquant.Index(dim=16) as ix:
        assert ix.calibrate(unit_rows(100, 16, offset=1.0)) is None
