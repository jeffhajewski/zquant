"""Sharing one Index or Codec between Python threads.

ctypes drops the GIL for every native call, so these threads really are inside the
library at the same time. Before the searcher pool, two threads searching one Index
shared one native searcher and got each other's results back, with no error.
"""
import threading

import numpy as np
import pytest

import zquant

N_THREADS = 8


def unit_rows(n, d, seed):
    x = np.random.default_rng(seed).standard_normal((n, d)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def spawn(n, target):
    """Start target(i) on n threads; return a function that joins them and re-raises
    the first exception any of them hit."""
    errors = []

    def wrap(i):
        try:
            target(i)
        except BaseException as e:      # noqa: BLE001 - re-raised by join
            errors.append(e)

    threads = [threading.Thread(target=wrap, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()

    def join():
        for t in threads:
            t.join(timeout=300)
            assert not t.is_alive(), "thread did not finish"
        if errors:
            raise errors[0]
    return join


def run_threads(n, target):
    """Run target(i) on n threads released together, and wait for them."""
    barrier = threading.Barrier(n)

    def go(i):
        barrier.wait()
        target(i)
    spawn(n, go)()


CONFIGS = [(10, 1), (3, 1), (10, 4)]       # (k, threads)


def test_concurrent_searches_equal_a_single_threaded_run():
    x = unit_rows(4000, 64, seed=1)
    q = unit_rows(300, 64, seed=2)
    with zquant.Index(dim=64, bits=4) as ix:
        ix.add(x)
        want = {c: ix.search(q, k=c[0], threads=c[1]) for c in CONFIGS}

        def worker(i):
            rng = np.random.default_rng(100 + i)
            for _ in range(30):
                k, t = CONFIGS[rng.integers(len(CONFIGS))]
                lo = int(rng.integers(0, len(q) - 1))
                hi = int(rng.integers(lo + 1, len(q) + 1))   # sometimes > one batch
                ids, scores = ix.search(q[lo:hi], k=k, threads=t)
                assert np.array_equal(ids, want[(k, t)][0][lo:hi]), (i, k, t, lo, hi)
                assert np.array_equal(scores, want[(k, t)][1][lo:hi]), (i, k, t, lo, hi)

        run_threads(N_THREADS, worker)


def test_concurrent_searches_get_their_own_searchers():
    """Searches that overlap each check out a different native searcher, which is what
    lets them run in parallel instead of queueing on one."""
    x = unit_rows(4000, 64, seed=3)
    q = unit_rows(64, 64, seed=4)
    with zquant.Index(dim=64) as ix:
        ix.add(x)
        run_threads(N_THREADS, lambda i: [ix.search(q, k=10) for _ in range(20)])
        # Searchers are only created when none is idle, so more than one means
        # searches were inside the library at the same time.
        assert 1 < len(ix._pool) <= N_THREADS

        with ix._rw.read():
            a = ix._pool.checkout(ix._handle, 5, 1)
            b = ix._pool.checkout(ix._handle, 5, 1)
            assert a != b
            ix._pool.checkin(5, 1, a)
            ix._pool.checkin(5, 1, b)


def check_result(ids, scores, max_count, k):
    """A result that is internally consistent, whatever the count was at the time."""
    found = ids != np.iinfo(np.uint32).max
    assert (ids[found] < max_count).all()
    assert np.isfinite(scores[found]).all()
    assert np.isneginf(scores[~found]).all()
    # Padding only ever trails the real results.
    assert (found[:, :-1] | ~found[:, 1:]).all()
    assert (np.diff(np.where(found, scores, -np.inf), axis=1) <= 1e-5).all()


def test_add_and_calibrate_concurrent_with_searches():
    d = 32
    x = unit_rows(3000, d, seed=5) + 0.5            # off-origin, so calibration applies
    q = unit_rows(40, d, seed=6)
    batches = np.array_split(x, 15)
    stop = threading.Event()
    searched = [0] * N_THREADS

    ix = zquant.Index(dim=d, bits=4)

    def searcher(i):
        while not stop.is_set():
            try:
                ids, scores = ix.search(q, k=7, threads=1 + i % 2)
            except zquant.StateError as e:
                assert "empty" in str(e)
                continue
            check_result(ids, scores, len(x), 7)
            searched[i] += 1

    join = spawn(N_THREADS, searcher)
    try:
        ix.calibrate(x[:500])
        for b in batches:
            ix.add(b)
    finally:
        stop.set()
        join()
    assert sum(searched) > 0

    # Nothing was lost or scrambled: the same calls made with nobody else around give
    # the same index, result for result.
    with zquant.Index(dim=d, bits=4) as ref:
        ref.calibrate(x[:500])
        for b in batches:
            ref.add(b)
        assert len(ix) == len(ref) == len(x)
        a, b = ix.search(q, k=10), ref.search(q, k=10)
        assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    ix.close()


@pytest.mark.parametrize("round_", range(5))
def test_close_concurrent_with_searches(round_):
    """Closing while searches run waits for them, and every later call raises
    StateError instead of touching freed memory."""
    x = unit_rows(2000, 32, seed=7 + round_)
    q = unit_rows(16, 32, seed=8)
    ix = zquant.Index(dim=32)
    ix.add(x)
    started = threading.Barrier(N_THREADS + 1)
    outcomes = [None] * N_THREADS

    def searcher(i):
        started.wait()
        n = 0
        while True:
            try:
                ids, scores = ix.search(q, k=5, threads=1 + i % 3)
            except zquant.StateError as e:
                assert "closed" in str(e)
                outcomes[i] = n
                return
            check_result(ids, scores, len(x), 5)
            n += 1

    join = spawn(N_THREADS, searcher)
    started.wait()
    ix.close()
    join()
    assert all(o is not None for o in outcomes)

    assert ix.closed
    for call in (lambda: ix.search(q), lambda: ix.add(x[:1]), lambda: len(ix),
                 lambda: ix.calibrate(x)):
        with pytest.raises(zquant.StateError, match="closed"):
            call()
    ix.close()                              # twice is harmless
    assert "closed" in repr(ix)


def test_shared_codec_is_safe():
    x = unit_rows(500, 64, seed=9)
    with zquant.Codec(dim=64, bits=4) as c:
        want_codes, want_norms = c.encode(x)
        want_back = c.decode(want_codes, want_norms)

        def worker(i):
            for _ in range(10):
                codes, norms = c.encode(x)
                assert np.array_equal(codes, want_codes)
                assert np.array_equal(norms, want_norms)
                assert np.array_equal(c.decode(codes, norms), want_back)

        run_threads(N_THREADS, worker)


def test_closed_codec_raises():
    c = zquant.Codec(dim=8)
    c.close()
    with pytest.raises(zquant.StateError, match="closed"):
        c.encode(np.zeros((1, 8)))
    c.close()
