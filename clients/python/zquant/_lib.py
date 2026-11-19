"""Locating, checking and declaring the native library.

ctypes rather than a compiled extension: the binding then needs no compiler at install
time and no build step during development, and the ABI it depends on is the same one the
JavaScript and Go clients use, so a mistake here surfaces in all three rather than hiding
in one.

Loading happens in two steps. First find a library file: an explicit path in
``ZQUANT_LIB``, then a copy bundled in this package (what a wheel ships), then the
repository's ``zig-out/lib`` (development). Then ask it whether this CPU can run it.
Release builds use SSSE3 on x86_64 and the dot-product extension on arm64, and on a CPU
without them the first search would die of an illegal instruction rather than raise. So
``zq_cpu_supported`` - compiled for the bare baseline, safe anywhere - is called before
anything else, and if it says no, a portable build sitting in the same directory is used
instead.
"""
from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path
from typing import Callable, List, Tuple

from ._errors import ZQ_ERR_CPU, CpuNotSupportedError, LibraryNotFoundError

if sys.platform == "darwin":
    LIBNAME, PORTABLE_NAME = "libzquant.dylib", "libzquant-portable.dylib"
elif sys.platform == "win32":
    LIBNAME, PORTABLE_NAME = "zquant.dll", "zquant-portable.dll"
else:
    LIBNAME, PORTABLE_NAME = "libzquant.so", "libzquant-portable.so"

#: Environment variable naming the library file to load, overriding the search.
ENV_VAR = "ZQUANT_LIB"
#: The name this binding used to read. Still honoured, after ZQUANT_LIB.
LEGACY_ENV_VAR = "ZQUANT_LIBRARY"

_PACKAGE_DIR = Path(__file__).resolve().parent


def candidates() -> List[Path]:
    """Where to look, in order.

    An explicit path from the environment is returned alone: if someone points the
    binding at a file, silently loading a different one when theirs is missing would
    hide the mistake.
    """
    for var in (ENV_VAR, LEGACY_ENV_VAR):
        explicit = os.environ.get(var)
        if explicit:
            return [Path(explicit)]
    # Development: clients/python/zquant -> clients/python -> clients -> repo root.
    repo_build = _PACKAGE_DIR.parents[2] / "zig-out" / "lib" / LIBNAME
    return [_PACKAGE_DIR / LIBNAME, repo_build]


def find() -> Path:
    paths = candidates()
    for path in paths:
        if path.is_file():
            return path
    explicit = os.environ.get(ENV_VAR) or os.environ.get(LEGACY_ENV_VAR)
    if explicit:
        raise LibraryNotFoundError(
            f"{ENV_VAR} points at {explicit}, which does not exist or is not a file."
        )
    raise LibraryNotFoundError(
        f"could not find {LIBNAME}. Build it with `zig build lib` at the repository "
        f"root, install a wheel that bundles it, or set {ENV_VAR} to its path.\n"
        "Looked in:\n  " + "\n  ".join(str(p) for p in paths)
    )


def _open(path: Path) -> ctypes.CDLL:
    lib = ctypes.CDLL(str(path))
    # Declared here, ahead of everything else, because they are what decides whether
    # anything else in this library is safe to call.
    lib.zq_cpu_supported.argtypes = []
    lib.zq_cpu_supported.restype = ctypes.c_int
    lib.zq_build_cpu.argtypes = []
    lib.zq_build_cpu.restype = ctypes.c_char_p
    return lib


def select(path: Path, opener: Callable[[Path], ctypes.CDLL] = _open) -> Tuple[ctypes.CDLL, Path, bool]:
    """Open `path`, or the portable build beside it if this CPU cannot run `path`.

    Returns the library, the file it came from, and whether that file is the portable
    build. `opener` exists so the fallback can be tested on a CPU that has every
    instruction the release build wants.
    """
    lib = opener(path)
    if lib.zq_cpu_supported():
        return lib, path, False
    wanted = (lib.zq_build_cpu() or b"?").decode(errors="replace")
    portable = path.with_name(PORTABLE_NAME)
    if portable.is_file():
        plib = opener(portable)
        if plib.zq_cpu_supported():
            return plib, portable, True
        raise CpuNotSupportedError(
            f"{portable} also reports that this CPU cannot run it "
            f"(built for {(plib.zq_build_cpu() or b'?').decode(errors='replace')}).",
            ZQ_ERR_CPU,
        )
    raise CpuNotSupportedError(
        f"{path} was built for CPU '{wanted}', and this CPU lacks instructions it "
        f"needs (SSSE3 on x86_64, the dot-product extension on arm64). Put the "
        f"portable build beside it as {portable.name}, or point {ENV_VAR} at a "
        f"portable build.",
        ZQ_ERR_CPU,
    )


def load() -> Tuple[ctypes.CDLL, Path, bool]:
    lib, path, portable = select(find())
    return bind(lib), path, portable


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
    u8p = ctypes.POINTER(ctypes.c_uint8)
    intp = ctypes.POINTER(ctypes.c_int)
    sizep = ctypes.POINTER(ctypes.c_size_t)
    size = ctypes.c_size_t
    vp = ctypes.c_void_p

    def decl(name: str, args: list, res=ctypes.c_int) -> None:
        fn = getattr(lib, name)
        fn.argtypes = args
        fn.restype = res

    decl("zq_version", [], ctypes.c_char_p)
    decl("zq_status_string", [ctypes.c_int], ctypes.c_char_p)
    decl("zq_build_kernels", [], ctypes.c_char_p)

    decl("zq_index_create", [ctypes.POINTER(Config), ctypes.POINTER(vp)])
    decl("zq_index_free", [vp], None)
    decl("zq_index_calibrate", [vp, f32p, size])
    decl("zq_index_calibrate_auto", [vp, f32p, size, f32p, intp])
    decl("zq_index_add", [vp, f32p, size])
    decl("zq_index_add_parallel", [vp, f32p, size, size])
    decl("zq_index_serialized_size", [vp], size)
    decl("zq_index_serialize", [vp, vp, size, sizep])
    decl("zq_index_deserialize", [vp, size, ctypes.POINTER(vp)])
    decl("zq_index_count", [vp], size)
    decl("zq_index_bytes_per_vector", [vp], size)
    decl("zq_index_dim", [vp], ctypes.c_uint32)
    decl("zq_index_config", [vp, ctypes.POINTER(Config)])
    decl("zq_index_calibrated", [vp], ctypes.c_int)
    decl("zq_index_partition", [vp, f32p, size, ctypes.c_uint32, size])
    decl("zq_index_lists", [vp], ctypes.c_uint32)

    decl("zq_searcher_create", [vp, size, size, size, ctypes.POINTER(vp)])
    decl("zq_searcher_free", [vp], None)
    decl("zq_searcher_capacity", [vp], size)
    decl("zq_searcher_set_probes", [vp, size])
    decl("zq_search", [vp, vp, f32p, size, u32p, f32p])
    decl("zq_search_rerank", [vp, vp, f32p, size, f32p, size, size, u32p, f32p])
    decl("zq_score_ids", [vp, vp, f32p, u32p, size, f32p])

    decl("zq_codec_create", [ctypes.POINTER(CodecConfig), ctypes.POINTER(vp)])
    decl("zq_codec_free", [vp], None)
    decl("zq_codec_code_bytes", [vp], size)
    decl("zq_codec_dim", [vp], ctypes.c_uint32)
    decl("zq_codec_encode", [vp, f32p, size, u8p, f32p])
    decl("zq_codec_decode", [vp, u8p, f32p, size, f32p])
    return lib

