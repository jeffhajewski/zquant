"""zquant — TurboQuant vector quantization.

    import numpy as np, zquant

    index = zquant.Index(dim=256, bits=5)
    index.calibrate(sample)          # optional; see `calibrate`
    index.add(vectors)
    ids, scores = index.search(queries, k=10)

Vectors are float32, row-major, one per row. Anything array-like that numpy can view
without copying is accepted; anything else is converted, which costs a copy.
"""
from __future__ import annotations

import ctypes
import os
import threading
from pathlib import Path
from typing import Literal, NamedTuple, Optional, Union

import numpy as np

from . import _lib
from ._sync import RWLock, SearcherPool
from ._errors import (
    ZQ_ERR_FORMAT,
    ZQ_ERR_STATE,
    ZQ_ERR_UNSUPPORTED,
    ZQ_OK,
    CpuNotSupportedError,
    FormatError,
    InvalidArgumentError,
    LibraryNotFoundError,
    OutOfMemoryError,
    StateError,
    UnsupportedError,
    ZquantError,
    error_for,
)

__all__ = [
    "Index",
    "Codec",
    "ZquantError",
    "OutOfMemoryError",
    "InvalidArgumentError",
    "StateError",
    "UnsupportedError",
    "CpuNotSupportedError",
    "FormatError",
    "LibraryNotFoundError",
    "BuildInfo",
    "Calibration",
    "build_info",
    "version",
    "MAX_DIM",
    "MAX_THREADS",
]

_LIB, _LIB_PATH, _LIB_PORTABLE = _lib.load()

#: Largest dimension the library accepts (ZQ_MAX_DIM).
MAX_DIM = 65536
#: Largest worker count for search or add (ZQ_MAX_THREADS).
MAX_THREADS = 1024

_METRICS = {"ip": 0, "inner_product": 0, "cosine": 1, "l2": 2}
_METRIC_NAMES = {0: "ip", 1: "cosine", 2: "l2"}
_U64_MAX = (1 << 64) - 1


def _status_text(status: int) -> str:
    return _LIB.zq_status_string(status).decode()


def _check(status: int, context: str) -> None:
    if status != ZQ_OK:
        raise error_for(status, context, _status_text)


def version() -> str:
    return _LIB.zq_version().decode()


class BuildInfo(NamedTuple):
    """Which native library was loaded, and what it was built for."""

    version: str
    #: The CPU model the library was compiled for, e.g. "x86_64_v2" or "apple_m1".
    cpu: str
    #: Which scan kernels it uses; "scalar" in here means the slow portable fallback.
    kernels: str
    #: The file that was loaded.
    library: Path
    #: True if this CPU could not run the release build and the portable one was loaded.
    portable: bool


def build_info() -> BuildInfo:
    """Describe the loaded library. Handy in a bug report, or to check you are not
    running the scalar fallback by accident."""
    return BuildInfo(
        version=version(),
        cpu=_LIB.zq_build_cpu().decode(),
        kernels=_LIB.zq_build_kernels().decode(),
        library=_LIB_PATH,
        portable=_LIB_PORTABLE,
    )


class Calibration(NamedTuple):
    """What `Index.calibrate(sample, auto=True)` measured and did."""

    #: ||mean(x / ||x||)|| over the sample: how far the data's directions sit from the
    #: origin. Calibration paid 9 to 22 points at 0.65 and above, and about one at 0.12;
    #: `auto` applies it from 0.3.
    centroid_norm: float
    #: Whether the index was calibrated.
    applied: bool


# ── Argument checks ─────────────────────────────────────────────────────────────────
#
# The library rejects all of these too, but only with "invalid argument". Checking here
# first says which argument and what range it should be in.


def _int_arg(value, name: str, lo: int, hi: int) -> int:
    # bool is an int subclass; Index(dim=True) is a mistake, not a 1-dimensional index.
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    v = int(value)
    if not lo <= v <= hi:
        raise ValueError(f"{name} must be in {lo}..{hi}, got {v}")
    return v


def _dim_arg(dim) -> int:
    return _int_arg(dim, "dim", 1, MAX_DIM)


def _seed_arg(seed) -> int:
    # ctypes would silently keep the low 64 bits of anything larger, or wrap a negative.
    return _int_arg(seed, "seed", 0, _U64_MAX)


def _threads_arg(threads) -> int:
    return _int_arg(threads, "threads", 0, MAX_THREADS)


def _as_rows(x, dim: int, name: str) -> np.ndarray:
    """Any array-like of real numbers as C-contiguous float32 rows of `dim` columns."""
    a = np.asarray(x)
    if a.dtype.kind not in "fiub":
        # Complex would lose its imaginary part with only a warning; strings and objects
        # would fail later with a less useful message.
        raise TypeError(f"{name} must hold real numbers, got dtype {a.dtype}")
    a = np.ascontiguousarray(a, dtype=np.float32)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if a.ndim != 2:
        raise ValueError(f"{name} must be 1-D or 2-D, got {a.ndim}-D")
    if a.shape[1] != dim:
        raise ValueError(f"{name} has dim {a.shape[1]}, expected {dim}")
    return a


def _check_rows(status: int, a: np.ndarray, context: str, name: str) -> None:
    """`_check`, explaining the usual cause of an invalid argument when it is that."""
    if status != ZQ_OK and not np.isfinite(a).all():
        raise InvalidArgumentError(
            f"{context}: {name} contain NaN or infinity, which the library refuses", status
        )
    _check(status, context)


def _ptr(a: np.ndarray, ctype):
    return a.ctypes.data_as(ctypes.POINTER(ctype))


# How many queries one native search call takes per worker. Larger batches amortize the
# corpus scan over more queries; 32 is well inside the library's limit of 128.
_SEARCH_BATCH = 32


def _create_searcher(index: int, k: int, threads: int) -> int:
    handle = ctypes.c_void_p()
    _check(_LIB.zq_searcher_create(index, _SEARCH_BATCH, k, threads, ctypes.byref(handle)),
           "searcher_create")
    return handle.value


def _free_searcher(handle: int) -> None:
    _LIB.zq_searcher_free(handle)


class Index:
    """A quantized index: flat (exhaustive) unless `partition` splits it into lists.

    Every vector is scanned for every query — there is no graph or partitioning — so
    recall is bounded only by quantization, and cost grows linearly with the corpus.

    `bits` is the total budget per coordinate, 2 to 6. Storage is about `dim*(bits-1)/8`
    bytes per vector plus a few for scalars; `bytes_per_vector` reports the exact figure.

    `compact=False` stores dequantized int8 instead, roughly doubling memory in exchange
    for some throughput. On a 100k x 256 corpus that trade was worth about 20%.

    An Index can be shared between threads. Searches run in parallel, each on its own
    native searcher drawn from a pool. `add`, `calibrate` and `close` wait for running
    searches to finish and hold off new ones until they are done.
    """

    def __init__(
        self,
        dim: int,
        bits: int = 5,
        metric: Literal["ip", "inner_product", "cosine", "l2"] = "ip",
        seed: int = 0x5EED,
        compact: bool = True,
    ):
        dim = _dim_arg(dim)
        bits = _int_arg(bits, "bits", 2, 6)
        seed = _seed_arg(seed)
        if metric not in _METRICS:
            raise ValueError(f"metric must be one of {sorted(_METRICS)}, got {metric!r}")
        cfg = _lib.Config(dim=dim, bits=bits, metric=_METRICS[metric], seed=seed,
                          compact=1 if compact else 0)
        handle = ctypes.c_void_p()
        _check(_LIB.zq_index_create(ctypes.byref(cfg), ctypes.byref(handle)), "index_create")
        self._adopt(handle.value)

    def _adopt(self, handle: int) -> None:
        self._handle = handle
        self._dim = int(_LIB.zq_index_dim(handle))
        # Read back rather than remembered from the constructor, so a loaded index
        # reports what it was saved with. An index saved through the Zig API can carry
        # options the C configuration has no field for; it works, but cannot say.
        cfg = _lib.Config()
        if _LIB.zq_index_config(handle, ctypes.byref(cfg)) == ZQ_OK:
            self._config = (int(cfg.bits), _METRIC_NAMES[cfg.metric], int(cfg.seed), bool(cfg.compact))
        else:
            self._config = None
        self._rw = RWLock()
        self._pool = SearcherPool(_create_searcher, _free_searcher)
        self._probes: Optional[int] = None

    def _live(self) -> int:
        """The native handle. Call with the lock held, read or write."""
        if not self._handle:
            raise StateError("the index is closed", ZQ_ERR_STATE)
        return self._handle

    def __del__(self):
        # No lock: nothing else can hold a reference to an object being finalized.
        # Guarded because at interpreter exit the module globals may already be gone.
        try:
            pool, handle = self.__dict__.get("_pool"), self.__dict__.get("_handle")
            if pool is not None:
                pool.free_all()
            if handle:
                self._handle = None
                _LIB.zq_index_free(handle)
        except Exception:
            pass

    def close(self) -> None:
        """Free the native index and its searchers. Waits for running searches; later
        calls raise `StateError`. Closing twice is harmless."""
        with self._rw.write():
            self._pool.free_all()
            if self._handle:
                handle, self._handle = self._handle, None
                _LIB.zq_index_free(handle)

    @property
    def closed(self) -> bool:
        return not self._handle

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    @property
    def dim(self) -> int:
        return self._dim

    def _setting(self, i: int):
        if self._config is None:
            raise UnsupportedError(
                "this index was saved through the Zig API with options the C "
                "configuration cannot describe", ZQ_ERR_UNSUPPORTED)
        return self._config[i]

    @property
    def bits(self) -> int:
        """Bits per coordinate, as passed to the constructor or saved in the file."""
        return self._setting(0)

    @property
    def metric(self) -> str:
        """`"ip"`, `"cosine"` or `"l2"`."""
        return self._setting(1)

    @property
    def seed(self) -> int:
        return self._setting(2)

    @property
    def compact(self) -> bool:
        return self._setting(3)

    @property
    def calibrated(self) -> bool:
        """Whether `calibrate` has been applied, here or before the index was saved."""
        with self._rw.read():
            return bool(_LIB.zq_index_calibrated(self._live()))

    def __len__(self) -> int:
        with self._rw.read():
            return _LIB.zq_index_count(self._live())

    @property
    def bytes_per_vector(self) -> int:
        with self._rw.read():
            return _LIB.zq_index_bytes_per_vector(self._live())

    def calibrate(self, sample, auto: bool = False) -> Optional[Calibration]:
        """Fit the corpus centre from a sample. Must precede `add`.

        Each vector is then stored relative to the centre, and the centre's share of
        every score is computed exactly, for 2 more bytes per vector. Worth it when the
        corpus sits away from the origin: compute
        `np.linalg.norm(np.mean(x / np.linalg.norm(x, axis=1, keepdims=True), axis=0))`.
        At 0.65 and above it gained 9 to 22 points of recall@10 at bits 2 and 3; at 0.12,
        about a point. A few hundred rows are enough.

        `auto=True` lets the library make that call: it measures the sample's centroid
        norm and calibrates only if it is at least 0.3. It returns a `Calibration`
        saying what it measured and whether it calibrated; it fails exactly as a plain
        calibration would (after `add`, say), whatever it measured. It is the sensible
        choice when you do not know your data. Without `auto` this returns None.
        """
        a = _as_rows(sample, self._dim, "sample")
        with self._rw.write():
            h = self._live()
            if not auto:
                _check_rows(_LIB.zq_index_calibrate(h, _ptr(a, ctypes.c_float), a.shape[0]),
                            a, "calibrate", "sample")
                return None
            norm = ctypes.c_float()
            applied = ctypes.c_int()
            _check_rows(
                _LIB.zq_index_calibrate_auto(h, _ptr(a, ctypes.c_float), a.shape[0],
                                             ctypes.byref(norm), ctypes.byref(applied)),
                a, "calibrate", "sample")
        return Calibration(centroid_norm=float(norm.value), applied=bool(applied.value))

    def partition(self, sample, lists: int, threads: int = 1) -> None:
        """Split the index into `lists` lists by k-means over a sample. Must precede `add`.

        Each vector added afterwards goes to the list whose centroid is nearest its
        direction and is stored relative to that centroid. A search then scans only the
        `probes` lists nearest each query, an eighth of them unless you say otherwise
        (`probes` on `search`, or the `probes` property). Storing vectors against nearby
        centroids also makes their estimates more accurate, so recall rises even with
        every list scanned.

        Around `sqrt(n)` lists suits a corpus of n vectors. The sample needs at least
        `lists` rows; past 256 per list the rest are not used. Partitioning takes the
        place of `calibrate`, and an index can be partitioned only once.

        A partitioned index cannot yet be saved, and `score_ids` is not available on it.
        """
        lists = _int_arg(lists, "lists", 1, 1 << 16)
        threads = _threads_arg(threads)
        a = _as_rows(sample, self._dim, "sample")
        with self._rw.write():
            h = self._live()
            _check_rows(_LIB.zq_index_partition(h, _ptr(a, ctypes.c_float), a.shape[0], lists,
                                                threads), a, "partition", "sample")
            # Searchers made before partitioning have no lists to probe.
            self._pool.free_all()

    @property
    def lists(self) -> int:
        """Lists in a partitioned index, or 0 for a flat one."""
        with self._rw.read():
            return int(_LIB.zq_index_lists(self._live()))

    @property
    def probes(self) -> Optional[int]:
        """Lists each query scans on a partitioned index, when `search` is not given
        `probes`. None means an eighth of the lists."""
        return self._probes

    @probes.setter
    def probes(self, value: Optional[int]) -> None:
        self._probes = None if value is None else _int_arg(value, "probes", 1, 1 << 16)

    def _apply_probes(self, h: int, searcher: int, probes: Optional[int]) -> None:
        """Set the searcher's probes for this call. Call with the read lock held."""
        lists = int(_LIB.zq_index_lists(h))
        if probes is not None:
            probes = _int_arg(probes, "probes", 1, 1 << 16)
            if lists == 0:
                raise StateError("probes applies only to a partitioned index", ZQ_ERR_STATE)
        if lists == 0:
            return
        if probes is None:
            probes = self._probes if self._probes is not None else max(1, lists // 8)
        _check(_LIB.zq_searcher_set_probes(searcher, probes), "search")

    def add(self, vectors, threads: int = 1) -> None:
        """Append vectors; the first gets id `len(index)` as it was before the call.

        `threads > 1` encodes on that many threads. The index comes out byte for byte
        the same either way. Below a few hundred vectors per thread the library stays on
        the calling thread, since starting threads costs more than the encoding.

        A batch with any NaN or infinity in it is refused whole; nothing is added.
        """
        threads = _threads_arg(threads)
        a = _as_rows(vectors, self._dim, "vectors")
        with self._rw.write():
            h = self._live()
            if threads <= 1:
                status = _LIB.zq_index_add(h, _ptr(a, ctypes.c_float), a.shape[0])
            else:
                status = _LIB.zq_index_add_parallel(h, _ptr(a, ctypes.c_float), a.shape[0],
                                                    threads)
            _check_rows(status, a, "add", "vectors")

    def search(self, queries, k: int = 10, threads: int = 1, probes: Optional[int] = None):
        """Return `(ids, scores)`, each shape `(n_queries, k)`, best first.

        Where the index holds fewer than `k` vectors the tail is `2**32-1` / `-inf`, as
        it is on a partitioned index where the lists scanned hold fewer than `k`.
        `probes` sets how many lists each query scans on a partitioned index, for this
        call only.
        `threads > 1` splits the batch across workers, which only pays for batches at
        least that large. Separate Python threads calling `search` at once also run in
        parallel, each on its own searcher.
        """
        k = _int_arg(k, "k", 1, 1 << 32)
        threads = _threads_arg(threads)
        q = _as_rows(queries, self._dim, "queries")
        nq = q.shape[0]
        ids = np.empty((nq, k), dtype=np.uint32)
        scores = np.empty((nq, k), dtype=np.float32)

        with self._rw.read():
            h = self._live()
            if _LIB.zq_index_count(h) == 0:
                raise StateError("search on an empty index", ZQ_ERR_STATE)
            with self._pool.borrow(h, k, threads) as searcher:
                self._apply_probes(h, searcher, probes)
                capacity = _LIB.zq_searcher_capacity(searcher)
                # The native call takes at most `capacity` queries, so long inputs are
                # chunked here rather than making that the caller's problem.
                for start in range(0, nq, capacity):
                    n = min(capacity, nq - start)
                    chunk = q[start:start + n]
                    _check_rows(
                        _LIB.zq_search(
                            h,
                            searcher,
                            _ptr(chunk, ctypes.c_float),
                            n,
                            _ptr(ids[start:start + n], ctypes.c_uint32),
                            _ptr(scores[start:start + n], ctypes.c_float),
                        ),
                        chunk,
                        "search",
                        "queries",
                    )
        return ids, scores

    def search_rerank(self, queries, originals, k: int = 10, candidates: Optional[int] = None,
                      threads: int = 1, probes: Optional[int] = None):
        """Search, then rescore the candidates exactly against the original vectors.

        Fetches `candidates` results per query from the quantized index (default
        `4 * k`), scores each against its original vector with the index's metric, and
        returns the best `k` as `(ids, scores)`, padded like `search`. The scores are
        exact, not estimates. Over-fetching 4-10x and reranking recovers near-exact
        top-k.

        `originals` holds the vectors you added, row i being the one that got id i, and
        must cover every id in the index. Keeping them is the price: full precision
        costs `4 * dim` bytes per vector, against the index's `bytes_per_vector`.
        `probes` is as for `search`.
        Pass float32 C-contiguous rows to avoid a copy on every call.
        """
        k = _int_arg(k, "k", 1, 1 << 32)
        candidates = 4 * k if candidates is None else _int_arg(candidates, "candidates",
                                                               1, 1 << 32)
        if candidates < k:
            raise ValueError(f"candidates ({candidates}) must be at least k ({k})")
        threads = _threads_arg(threads)
        q = _as_rows(queries, self._dim, "queries")
        orig = _as_rows(originals, self._dim, "originals")
        nq = q.shape[0]
        ids = np.empty((nq, k), dtype=np.uint32)
        scores = np.empty((nq, k), dtype=np.float32)

        with self._rw.read():
            h = self._live()
            count = _LIB.zq_index_count(h)
            if count == 0:
                raise StateError("search on an empty index", ZQ_ERR_STATE)
            if orig.shape[0] < count:
                raise ValueError(f"originals has {orig.shape[0]} rows, but the index holds "
                                 f"{count} vectors and every one needs its original")
            with self._pool.borrow(h, candidates, threads) as searcher:
                self._apply_probes(h, searcher, probes)
                capacity = _LIB.zq_searcher_capacity(searcher)
                for start in range(0, nq, capacity):
                    n = min(capacity, nq - start)
                    chunk = q[start:start + n]
                    status = _LIB.zq_search_rerank(
                        h, searcher, _ptr(chunk, ctypes.c_float), n,
                        _ptr(orig, ctypes.c_float), orig.shape[0], k,
                        _ptr(ids[start:start + n], ctypes.c_uint32),
                        _ptr(scores[start:start + n], ctypes.c_float),
                    )
                    _check_rows(status, chunk, "search_rerank", "queries")
        return ids, scores

    def score_ids(self, query, ids) -> np.ndarray:
        """Score chosen stored vectors against one query.

        Returns float32 scores, `scores[j]` being the estimate `search` would report for
        `ids[j]`. For ranking a candidate set picked some other way - a metadata filter,
        another index - without scanning the whole corpus. An id outside
        `0..len(index)-1` raises IndexError.
        """
        q = _as_rows(query, self._dim, "query")
        if q.shape[0] != 1:
            raise ValueError(f"score_ids takes one query, got {q.shape[0]}")
        raw = np.asarray(ids)
        if raw.size and raw.dtype.kind not in "iu":
            raise TypeError(f"ids must be integers, got dtype {raw.dtype}")
        if raw.ndim > 1:
            raise ValueError(f"ids must be 1-D, got shape {raw.shape}")
        raw = raw.reshape(-1)
        out = np.empty(raw.shape[0], dtype=np.float32)

        with self._rw.read():
            h = self._live()
            count = _LIB.zq_index_count(h)
            if raw.size and (raw.min() < 0 or raw.max() >= count):
                bad = raw[(raw < 0) | (raw >= count)][0]
                raise IndexError(f"id {bad} is out of range for an index of {count} vectors")
            id_arr = np.ascontiguousarray(raw, dtype=np.uint32)
            with self._pool.borrow(h, 1, 1) as searcher:
                status = _LIB.zq_score_ids(h, searcher, _ptr(q, ctypes.c_float),
                                           _ptr(id_arr, ctypes.c_uint32), id_arr.shape[0],
                                           _ptr(out, ctypes.c_float))
                _check_rows(status, q, "score_ids", "query")
        return out

    # ── Saving and loading ─────────────────────────────────────────────────────────
    #
    # The format holds the configuration, any calibration and the codes. It is
    # versioned, checksummed and the same on every platform, so a file saved on one
    # machine loads on another. Loading checks every field and the checksum, so
    # anything that is not an intact index raises FormatError rather than crashing.

    def _serialize(self) -> np.ndarray:
        with self._rw.read():
            h = self._live()
            size = _LIB.zq_index_serialized_size(h)
            buf = np.empty(size, dtype=np.uint8)
            written = ctypes.c_size_t()
            _check(_LIB.zq_index_serialize(h, buf.ctypes.data, size, ctypes.byref(written)),
                   "serialize")
        return buf[:written.value]

    def to_bytes(self) -> bytes:
        """The index as bytes, for `Index.from_bytes`."""
        return self._serialize().tobytes()

    def save(self, path: Union[str, os.PathLike]) -> None:
        """Write the index to a file, for `Index.load`.

        Written to a temporary file beside `path` and renamed into place, so a crash
        part way through never leaves a truncated index under the real name.
        """
        data = self._serialize()
        path = Path(path)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        try:
            with open(tmp, "xb") as f:
                f.write(memoryview(data))
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    @classmethod
    def from_bytes(cls, data) -> "Index":
        """Rebuild an index from `to_bytes` output (bytes, bytearray, memoryview or a
        uint8 array). The data is copied, so it can be discarded afterwards."""
        buf = np.frombuffer(data, dtype=np.uint8)
        if buf.size == 0:
            raise FormatError("from_bytes: no data", ZQ_ERR_FORMAT)
        handle = ctypes.c_void_p()
        _check(_LIB.zq_index_deserialize(buf.ctypes.data, buf.shape[0], ctypes.byref(handle)),
               "from_bytes")
        index = cls.__new__(cls)
        index._adopt(handle.value)
        return index

    @classmethod
    def load(cls, path: Union[str, os.PathLike]) -> "Index":
        """Read an index written by `save`."""
        return cls.from_bytes(np.fromfile(os.fspath(path), dtype=np.uint8))

    def __repr__(self) -> str:
        with self._rw.read():
            if not self._handle:
                return f"<zquant.Index dim={self._dim} closed>"
            n = _LIB.zq_index_count(self._handle)
            bpv = _LIB.zq_index_bytes_per_vector(self._handle)
        settings = "" if self._config is None else f" bits={self._config[0]} metric={self._config[1]}"
        return f"<zquant.Index dim={self._dim}{settings} vectors={n} {bpv}B/vector>"


class Codec:
    """Encode and decode vectors without an index.

    `Index` answers "which stored vectors best match this query". A KV cache asks
    something else: store these vectors compactly and give them back. Attention needs
    *every* score rather than the top few, and needs them accurate in absolute terms,
    so the index's estimator — whose per-vector correction is fitted to preserve
    *ranking* — is the wrong tool there and measures considerably worse than plain
    reconstruction.

        codec = zquant.Codec(dim=64, bits=5)
        codes, norms = codec.encode(keys)        # keep these two
        keys_back = codec.decode(codes, norms)

    Codes are bit-packed, so `code_bytes` is the storage actually consumed; norms cost
    four bytes per vector on top. `bits` is the codebook width directly, 1 to 6, where
    `Index` reserves one of its bits for a residual sketch, so `Codec(bits=4)` packs as
    many bits per coordinate as `Index(bits=5)`.

    Is it a good KV-cache quantizer? Not the best one measured. At matched memory,
    KIVI-style 4-bit quantization (keys per channel, values per token) gave lower
    perplexity than `bits=4` on every model tried, and on a model whose keys carry a
    large per-channel offset (Qwen2.5) zquant's keys failed outright; there is no
    centring option to correct for that yet. On the other models it does beat plain
    int4, and it reconstructs values as well as anything measured: the loss is in keys.
    docs/kv-cache.md in the repository has the measurements and their caveats.

    A Codec owns native scratch space that two calls cannot share, so its calls take a
    lock: sharing one between threads is safe, but they take turns. To encode on
    several threads at once, give each its own Codec.
    """

    def __init__(self, dim: int, bits: int = 5, seed: int = 0x5EED):
        dim = _dim_arg(dim)
        bits = _int_arg(bits, "bits", 1, 6)
        seed = _seed_arg(seed)
        cfg = _lib.CodecConfig(dim=dim, bits=bits, seed=seed)
        self._lock = threading.Lock()
        handle = ctypes.c_void_p()
        _check(_LIB.zq_codec_create(ctypes.byref(cfg), ctypes.byref(handle)), "codec_create")
        self._handle = handle.value
        self._dim = dim
        self._code_bytes = _LIB.zq_codec_code_bytes(handle)
        if self._code_bytes <= 0:
            self.close()
            raise UnsupportedError(
                f"codec reported {self._code_bytes} code bytes for dim={dim} bits={bits}")

    def __del__(self):
        try:
            handle = self.__dict__.get("_handle")
            if handle:
                self._handle = None
                _LIB.zq_codec_free(handle)
        except Exception:
            pass

    def close(self) -> None:
        """Free the native codec. Later calls raise `StateError`."""
        with self._lock:
            if self._handle:
                handle, self._handle = self._handle, None
                _LIB.zq_codec_free(handle)

    def _live(self) -> int:
        if not self._handle:
            raise StateError("the codec is closed", ZQ_ERR_STATE)
        return self._handle

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def code_bytes(self) -> int:
        """Packed bytes of code per vector, excluding the four-byte norm."""
        return self._code_bytes

    def encode(self, vectors):
        """Return `(codes, norms)`: uint8 `(n, code_bytes)` and float32 `(n,)`.

        Both are needed to decode. Keeping them separate rather than packing the norm
        into the code lets a caller store the codes contiguously, which is what makes
        the decode a single pass.
        """
        a = _as_rows(vectors, self._dim, "vectors")
        n = a.shape[0]
        codes = np.empty((n, self._code_bytes), dtype=np.uint8)
        norms = np.empty(n, dtype=np.float32)
        with self._lock:
            _check_rows(
                _LIB.zq_codec_encode(
                    self._live(),
                    _ptr(a, ctypes.c_float),
                    n,
                    _ptr(codes, ctypes.c_uint8),
                    _ptr(norms, ctypes.c_float),
                ),
                a,
                "encode",
                "vectors",
            )
        return codes, norms

    def decode(self, codes, norms):
        """Reconstruct `(n, dim)` float32 from what `encode` returned."""
        c = np.ascontiguousarray(codes, dtype=np.uint8)
        nm = np.ascontiguousarray(norms, dtype=np.float32)
        if c.ndim != 2 or c.shape[1] != self._code_bytes:
            raise ValueError(f"codes must be (n, {self._code_bytes}), got {c.shape}")
        if nm.shape != (c.shape[0],):
            raise ValueError(f"norms must be ({c.shape[0]},), got {nm.shape}")
        out = np.empty((c.shape[0], self._dim), dtype=np.float32)
        with self._lock:
            _check(
                _LIB.zq_codec_decode(
                    self._live(),
                    _ptr(c, ctypes.c_uint8),
                    _ptr(nm, ctypes.c_float),
                    c.shape[0],
                    _ptr(out, ctypes.c_float),
                ),
                "decode",
            )
        return out

    def __repr__(self) -> str:
        return f"<zquant.Codec dim={self._dim} {self._code_bytes}B/vector>"
