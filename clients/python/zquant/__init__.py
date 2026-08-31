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
from typing import Literal

import numpy as np

from . import _lib

__all__ = ["Index", "Codec", "ZquantError", "version"]

_LIB = _lib.bind(_lib.load())

_METRICS = {"ip": 0, "inner_product": 0, "cosine": 1, "l2": 2}

OK = 0


class ZquantError(RuntimeError):
    """A native call failed. The message is the library's own status string."""

    def __init__(self, status: int, context: str):
        self.status = status
        text = _LIB.zq_status_string(status).decode()
        super().__init__(f"{context}: {text} ({status})")


def _check(status: int, context: str) -> None:
    if status != OK:
        raise ZquantError(status, context)


def version() -> str:
    return _LIB.zq_version().decode()


def _as_rows(x, dim: int, name: str) -> np.ndarray:
    a = np.ascontiguousarray(x, dtype=np.float32)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if a.ndim != 2:
        raise ValueError(f"{name} must be 1-D or 2-D, got {a.ndim}-D")
    if a.shape[1] != dim:
        raise ValueError(f"{name} has dim {a.shape[1]}, index expects {dim}")
    return a


def _ptr(a: np.ndarray, ctype):
    return a.ctypes.data_as(ctypes.POINTER(ctype))


class Index:
    """A flat (exhaustive) quantized index.

    Every vector is scanned for every query — there is no graph or partitioning — so
    recall is bounded only by quantization, and cost grows linearly with the corpus.

    `bits` is the total budget per coordinate, 2 to 6. Storage is about `dim*(bits-1)/8`
    bytes per vector plus a few for scalars; `bytes_per_vector` reports the exact figure.

    `compact=False` stores dequantized int8 instead, roughly doubling memory in exchange
    for some throughput. On a 100k x 256 corpus that trade was worth about 20%.
    """

    def __init__(
        self,
        dim: int,
        bits: int = 5,
        metric: Literal["ip", "inner_product", "cosine", "l2"] = "ip",
        seed: int = 0x5EED,
        compact: bool = True,
    ):
        if metric not in _METRICS:
            raise ValueError(f"metric must be one of {sorted(_METRICS)}, got {metric!r}")
        cfg = _lib.Config(dim=dim, bits=bits, metric=_METRICS[metric], seed=seed,
                          compact=1 if compact else 0)
        handle = ctypes.c_void_p()
        _check(_LIB.zq_index_create(ctypes.byref(cfg), ctypes.byref(handle)), "index_create")
        self._handle = handle
        self._dim = dim
        self._searchers: dict[tuple[int, int], ctypes.c_void_p] = {}

    def __del__(self):
        self.close()

    def close(self) -> None:
        for s in getattr(self, "_searchers", {}).values():
            _LIB.zq_searcher_free(s)
        self._searchers = {}
        h = getattr(self, "_handle", None)
        if h:
            _LIB.zq_index_free(h)
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    @property
    def dim(self) -> int:
        return self._dim

    def __len__(self) -> int:
        return _LIB.zq_index_count(self._handle)

    @property
    def bytes_per_vector(self) -> int:
        return _LIB.zq_index_bytes_per_vector(self._handle)

    def calibrate(self, sample) -> None:
        """Fit per-coordinate shift and scale from a sample. Must precede `add`.

        Worth enabling when the corpus centroid sits away from the origin — compute
        `np.linalg.norm(np.mean(x / np.linalg.norm(x, axis=1, keepdims=True), axis=0))`
        and expect a real gain above about 0.3 and none below. It is not free: on
        low-rank zero-mean data it has cost several points of recall. A few thousand rows
        are enough; more did not measurably help.
        """
        a = _as_rows(sample, self._dim, "sample")
        _check(_LIB.zq_index_calibrate(self._handle, _ptr(a, ctypes.c_float), a.shape[0]),
               "calibrate")

    def add(self, vectors) -> None:
        a = _as_rows(vectors, self._dim, "vectors")
        _check(_LIB.zq_index_add(self._handle, _ptr(a, ctypes.c_float), a.shape[0]), "add")

    def _searcher(self, k: int, threads: int):
        key = (k, threads)
        if key not in self._searchers:
            handle = ctypes.c_void_p()
            batch = 32
            _check(
                _LIB.zq_searcher_create(self._handle, batch, k, threads, ctypes.byref(handle)),
                "searcher_create",
            )
            self._searchers[key] = handle
        return self._searchers[key]

    def search(self, queries, k: int = 10, threads: int = 1):
        """Return `(ids, scores)`, each shape `(n_queries, k)`, best first.

        Where the index holds fewer than `k` vectors the tail is `2**32-1` / `-inf`.
        `threads > 1` splits the batch across workers, which only pays for batches at
        least that large.
        """
        if k < 1:
            raise ValueError("k must be at least 1")
        if len(self) == 0:
            raise ZquantError(-3, "search on an empty index")
        q = _as_rows(queries, self._dim, "queries")
        nq = q.shape[0]

        searcher = self._searcher(k, threads)
        capacity = _LIB.zq_searcher_capacity(searcher)

        ids = np.empty((nq, k), dtype=np.uint32)
        scores = np.empty((nq, k), dtype=np.float32)
        # The native call takes at most `capacity` queries, so long inputs are chunked
        # here rather than making that the caller's problem.
        for start in range(0, nq, capacity):
            n = min(capacity, nq - start)
            _check(
                _LIB.zq_search(
                    self._handle,
                    searcher,
                    _ptr(q[start:start + n], ctypes.c_float),
                    n,
                    _ptr(ids[start:start + n], ctypes.c_uint32),
                    _ptr(scores[start:start + n], ctypes.c_float),
                ),
                "search",
            )
        return ids, scores

    def __repr__(self) -> str:
        return (f"<zquant.Index dim={self._dim} vectors={len(self)} "
                f"{self.bytes_per_vector}B/vector>")


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

    A Codec owns scratch space and is not safe for concurrent use; make one per thread.
    """

    def __init__(self, dim: int, bits: int = 5, seed: int = 0x5EED):
        cfg = _lib.CodecConfig(dim=dim, bits=bits, seed=seed)
        handle = ctypes.c_void_p()
        _check(_LIB.zq_codec_create(ctypes.byref(cfg), ctypes.byref(handle)), "codec_create")
        self._handle = handle
        self._dim = dim
        self._code_bytes = _LIB.zq_codec_code_bytes(handle)

    def __del__(self):
        self.close()

    def close(self) -> None:
        h = getattr(self, "_handle", None)
        if h:
            _LIB.zq_codec_free(h)
            self._handle = None

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
        _check(
            _LIB.zq_codec_encode(
                self._handle,
                _ptr(a, ctypes.c_float),
                n,
                codes.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                _ptr(norms, ctypes.c_float),
            ),
            "encode",
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
        _check(
            _LIB.zq_codec_decode(
                self._handle,
                c.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                _ptr(nm, ctypes.c_float),
                c.shape[0],
                _ptr(out, ctypes.c_float),
            ),
            "decode",
        )
        return out

    def __repr__(self) -> str:
        return f"<zquant.Codec dim={self._dim} {self._code_bytes}B/vector>"
