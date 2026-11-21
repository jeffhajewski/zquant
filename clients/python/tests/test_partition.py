"""Partitioned indexes: training, probing, saving, and what a partitioned index refuses."""
import numpy as np
import pytest

import zquant


def clustered(n, d, clusters=16, seed=0):
    rng = np.random.default_rng(seed)
    centres = rng.standard_normal((clusters, d)).astype(np.float32)
    x = centres[rng.integers(0, clusters, n)] + 0.5 * rng.standard_normal((n, d)).astype(np.float32)
    x += 0.4
    return x * rng.uniform(0.5, 2.5, (n, 1)).astype(np.float32)


def exact_top(x, q, k):
    return np.argsort(-(q @ x.T), axis=1)[:, :k]


def recall(ids, exact):
    return np.mean([len(set(a) & set(b)) / exact.shape[1] for a, b in zip(ids, exact)])


def test_partitioning_lifts_recall_and_probes_trade_it_for_speed():
    x, q = clustered(6000, 64, seed=1), clustered(60, 64, seed=2)
    exact = exact_top(x, q, 10)
    with zquant.Index(dim=64, bits=2) as flat, zquant.Index(dim=64, bits=2) as ivf:
        flat.calibrate(x)
        flat.add(x)
        ivf.partition(x, lists=64, threads=4)
        assert ivf.lists == 64 and flat.lists == 0
        ivf.add(x, threads=4)
        assert len(ivf) == 6000
        assert ivf.bytes_per_vector == flat.bytes_per_vector + 6
        flat_recall = recall(flat.search(q, k=10)[0], exact)
        by_probes = [recall(ivf.search(q, k=10, probes=p)[0], exact) for p in (1, 8, 64)]
        assert by_probes[0] < by_probes[1]
        assert by_probes[2] > flat_recall + 0.05


def test_probes_property_and_argument():
    x = clustered(3000, 32, seed=3)
    with zquant.Index(dim=32, bits=3) as ix:
        ix.partition(x, lists=32)
        ix.add(x)
        assert ix.probes is None
        default = ix.search(x[:20], k=5)
        assert np.array_equal(default[0], ix.search(x[:20], k=5, probes=4)[0])
        ix.probes = 32
        everything = ix.search(x[:20], k=5)
        assert np.array_equal(everything[0], ix.search(x[:20], k=5, probes=32)[0])
        # A per-call value does not stick.
        ix.search(x[:20], k=5, probes=1)
        assert np.array_equal(everything[0], ix.search(x[:20], k=5)[0])
        with pytest.raises(ValueError):
            ix.probes = 0


def test_threads_and_batches_agree():
    x, q = clustered(4000, 48, seed=4), clustered(100, 48, seed=5)
    with zquant.Index(dim=48, bits=4) as ix:
        ix.partition(x, lists=40)
        ix.add(x, threads=3)
        one = ix.search(q, k=10, probes=6)
        many = ix.search(q, k=10, probes=6, threads=4)
        assert np.array_equal(one[0], many[0])
        assert np.array_equal(one[1], many[1])
        single = [ix.search(row, k=10, probes=6) for row in q[:5]]
        for i, (ids, scores) in enumerate(single):
            assert np.array_equal(ids[0], one[0][i])
            assert np.array_equal(scores[0], one[1][i])


def test_short_lists_pad_and_rerank_skips_the_padding():
    x = clustered(200, 16, seed=6)
    with zquant.Index(dim=16, bits=3) as ix:
        ix.partition(x, lists=8)
        ix.add(x[:6])
        ids, scores = ix.search(x[:1], k=10, probes=8)
        assert set(ids[0][:6]) == set(range(6))
        assert (ids[0][6:] == 2**32 - 1).all() and np.isneginf(scores[0][6:]).all()
        ids, scores = ix.search_rerank(x[:1], x[:6], k=8, candidates=10, probes=8)
        assert set(ids[0][:6]) == set(range(6))
        assert np.allclose(scores[0][:6], np.sort(x[:6] @ x[0])[::-1], rtol=1e-5)
        assert (ids[0][6:] == 2**32 - 1).all()


def test_what_a_partitioned_index_refuses():
    x = clustered(500, 16, seed=7)
    with zquant.Index(dim=16, bits=3) as ix:
        with pytest.raises(zquant.StateError):
            ix.search(x[:1], k=1, probes=2)  # empty, and not partitioned
        ix.partition(x, lists=10)
        with pytest.raises(zquant.StateError):
            ix.partition(x, lists=10)
        with pytest.raises(zquant.StateError):
            ix.calibrate(x)
    with zquant.Index(dim=16, bits=3) as flat:
        flat.add(x)
        with pytest.raises(zquant.StateError):
            flat.search(x[:1], k=1, probes=2)
        with pytest.raises(zquant.StateError):
            flat.partition(x, lists=4)
    with zquant.Index(dim=16, bits=3) as cal:
        cal.calibrate(x)
        with pytest.raises(zquant.StateError):
            cal.partition(x, lists=4)
    with zquant.Index(dim=16, bits=3) as few:
        with pytest.raises(zquant.InvalidArgumentError):
            few.partition(x[:3], lists=4)
        with pytest.raises(ValueError):
            few.partition(x, lists=0)


def test_a_saved_partitioned_index_loads_partitioned(tmp_path):
    x = clustered(3000, 32, seed=11)
    q = clustered(20, 32, seed=12)
    with zquant.Index(dim=32, bits=3, metric="cosine") as ix:
        ix.partition(x, lists=24, threads=2)
        ix.add(x[:2000])
        ix.save(tmp_path / "ivf.zq")
        with zquant.Index.load(tmp_path / "ivf.zq") as back:
            assert back.lists == 24
            assert len(back) == 2000
            assert back.bytes_per_vector == ix.bytes_per_vector
            assert (back.bits, back.metric) == (3, "cosine")
            for probes in (1, 6, 24):
                want = ix.search(q, k=10, probes=probes)
                got = back.search(q, k=10, probes=probes)
                np.testing.assert_array_equal(want[0], got[0])
                np.testing.assert_array_equal(want[1], got[1])
            # Both keep accepting vectors, into the same lists.
            ix.add(x[2000:])
            back.add(x[2000:], threads=3)
            assert back.to_bytes() == ix.to_bytes()


def test_score_ids_on_a_partitioned_index_matches_search():
    x = clustered(2000, 24, seed=13)
    q = clustered(5, 24, seed=14)
    with zquant.Index(dim=24, bits=4) as ix:
        ix.partition(x, lists=16)
        ix.add(x)
        ids, scores = ix.search(q, k=15, probes=3)
        for i in range(len(q)):
            np.testing.assert_array_equal(ix.score_ids(q[i], ids[i]), scores[i])
        # Ids in lists the search never probed are scored too.
        everything = ix.score_ids(q[0], np.arange(len(x)))
        assert everything.shape == (len(x),) and np.isfinite(everything).all()
        with pytest.raises(IndexError):
            ix.score_ids(q[0], [len(x)])
