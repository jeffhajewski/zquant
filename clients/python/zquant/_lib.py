"""Locating and declaring the native library.

ctypes rather than a compiled extension: the binding then needs no compiler at install
time and no build step during development, and the ABI it depends on is the same one the
JavaScript and Go clients use, so a mistake here surfaces in all three rather than hiding
in one.
"""
from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

_NAMES = {
    "darwin": "libzquant.dylib",
    "win32": "zquant.dll",
}
_LIBNAME = _NAMES.get(sys.platform, "libzquant.so")


def _candidates() -> list[Path]:
    """Search order: explicit override, packaged copy, then a development build tree."""
    out = []
    env = os.environ.get("ZQUANT_LIBRARY")
    if env:
        out.append(Path(env))
    here = Path(__file__).resolve().parent
    out.append(here / _LIBNAME)
    # Development: clients/python/zquant -> repo root -> zig-out/lib
    out.append(here.parents[2] / "zig-out" / "lib" / _LIBNAME)
    return out


def load() -> ctypes.CDLL:
    tried = []
    for path in _candidates():
        if path.exists():
            return ctypes.CDLL(str(path))
        tried.append(str(path))
    raise OSError(
        f"could not find {_LIBNAME}. Build it with `zig build lib` at the repository "
        f"root, or set ZQUANT_LIBRARY to its path.\nLooked in:\n  " + "\n  ".join(tried)
    )


class CodecConfig(ctypes.Structure):
    _fields_ = [
        ("dim", ctypes.c_uint32),
        ("bits", ctypes.c_uint8),
        ("seed", ctypes.c_uint64),
    ]


class Config(ctypes.Structure):
    _fields_ = [
        ("dim", ctypes.c_uint32),
        ("bits", ctypes.c_uint8),
        ("metric", ctypes.c_int),
        ("seed", ctypes.c_uint64),
        ("compact", ctypes.c_int),
    ]


def bind(lib: ctypes.CDLL) -> ctypes.CDLL:
    f32p = ctypes.POINTER(ctypes.c_float)
    u32p = ctypes.POINTER(ctypes.c_uint32)
    vp = ctypes.c_void_p

    lib.zq_version.restype = ctypes.c_char_p
    lib.zq_status_string.argtypes = [ctypes.c_int]
    lib.zq_status_string.restype = ctypes.c_char_p

    lib.zq_index_create.argtypes = [ctypes.POINTER(Config), ctypes.POINTER(vp)]
    lib.zq_index_create.restype = ctypes.c_int
    lib.zq_index_free.argtypes = [vp]
    lib.zq_index_calibrate.argtypes = [vp, f32p, ctypes.c_size_t]
    lib.zq_index_calibrate.restype = ctypes.c_int
    lib.zq_index_add.argtypes = [vp, f32p, ctypes.c_size_t]
    lib.zq_index_add.restype = ctypes.c_int
    lib.zq_index_count.argtypes = [vp]
    lib.zq_index_count.restype = ctypes.c_size_t
    lib.zq_index_bytes_per_vector.argtypes = [vp]
    lib.zq_index_bytes_per_vector.restype = ctypes.c_size_t
    lib.zq_index_dim.argtypes = [vp]
    lib.zq_index_dim.restype = ctypes.c_uint32

    lib.zq_searcher_create.argtypes = [vp, ctypes.c_size_t, ctypes.c_size_t,
                                       ctypes.c_size_t, ctypes.POINTER(vp)]
    lib.zq_searcher_create.restype = ctypes.c_int
    lib.zq_searcher_free.argtypes = [vp]
    lib.zq_searcher_capacity.argtypes = [vp]
    lib.zq_searcher_capacity.restype = ctypes.c_size_t

    lib.zq_search.argtypes = [vp, vp, f32p, ctypes.c_size_t, u32p, f32p]
    lib.zq_search.restype = ctypes.c_int

    u8p = ctypes.POINTER(ctypes.c_uint8)
    lib.zq_codec_create.argtypes = [ctypes.POINTER(CodecConfig), ctypes.POINTER(vp)]
    lib.zq_codec_create.restype = ctypes.c_int
    lib.zq_codec_free.argtypes = [vp]
    lib.zq_codec_code_bytes.argtypes = [vp]
    lib.zq_codec_code_bytes.restype = ctypes.c_size_t
    lib.zq_codec_dim.argtypes = [vp]
    lib.zq_codec_dim.restype = ctypes.c_uint32
    lib.zq_codec_encode.argtypes = [vp, f32p, ctypes.c_size_t, u8p, f32p]
    lib.zq_codec_encode.restype = ctypes.c_int
    lib.zq_codec_decode.argtypes = [vp, u8p, f32p, ctypes.c_size_t, f32p]
    lib.zq_codec_decode.restype = ctypes.c_int
    return lib
