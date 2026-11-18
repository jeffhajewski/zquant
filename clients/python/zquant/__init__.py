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
import threading
from pathlib import Path
from typing import Literal, NamedTuple

import numpy as np

from . import _lib
from ._sync import RWLock, SearcherPool
from ._errors import (
    ZQ_ERR_STATE,
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
    """A flat (exhaustive) quantized index.

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
        self._rw = RWLock()
        self._pool = SearcherPool(_create_searcher, _free_searcher)

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

    def __len__(self) -> int:
        with self._rw.read():
            return _LIB.zq_index_count(self._live())

    @property
    def bytes_per_vector(self) -> int:
        with self._rw.read():
            return _LIB.zq_index_bytes_per_vector(self._live())

    def calibrate(self, sample) -> None:
        """Fit per-coordinate shift and scale from a sample. Must precede `add`.

        Worth enabling when the corpus centroid sits away from the origin — compute
        `np.linalg.norm(np.mean(x / np.linalg.norm(x, axis=1, keepdims=True), axis=0))`
        and expect a real gain above about 0.3 and none below. It is not free: on
        low-rank zero-mean data it has cost several points of recall. A few thousand rows
        are enough; more did not measurably help.
        """
        a = _as_rows(sample, self._dim, "sample")
        with self._rw.write():
            h = self._live()
            _check_rows(_LIB.zq_index_calibrate(h, _ptr(a, ctypes.c_float), a.shape[0]),
                        a, "calibrate", "sample")

    def add(self, vectors) -> None:
        a = _as_rows(vectors, self._dim, "vectors")
        with self._rw.write():
            h = self._live()
            _check_rows(_LIB.zq_index_add(h, _ptr(a, ctypes.c_float), a.shape[0]),
                        a, "add", "vectors")

    def search(self, queries, k: int = 10, threads: int = 1):
        """Return `(ids, scores)`, each shape `(n_queries, k)`, best first.

        Where the index holds fewer than `k` vectors the tail is `2**32-1` / `-inf`.
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

    def __repr__(self) -> str:
        with self._rw.read():
            if not self._handle:
                return f"<zquant.Index dim={self._dim} closed>"
            n = _LIB.zq_index_count(self._handle)
            bpv = _LIB.zq_index_bytes_per_vector(self._handle)
        return f"<zquant.Index dim={self._dim} vectors={n} {bpv}B/vector>"


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
    four bytes per vector on top. Measured on real attention tensors at `bits=5`, this
    gives 4x lower attention-output error than per-row int4 at the same memory.

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
