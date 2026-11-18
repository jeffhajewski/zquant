"""Exceptions, and the mapping from the library's status codes to them.

Every failure the library reports is a `ZquantError`, so one `except` catches them all.
Each status also has its own subclass, and the ones that mean "you passed something
wrong" also derive from the built-in a Python caller would reach for first: a bad
argument is a `ValueError`, running out of memory is a `MemoryError`.

This module does not touch the native library, so the loader can raise these before
there is a library to ask for its status strings.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Type

ZQ_OK = 0
ZQ_ERR_ALLOC = -1
ZQ_ERR_INVALID = -2
ZQ_ERR_STATE = -3
ZQ_ERR_UNSUPPORTED = -4
ZQ_ERR_CPU = -5
ZQ_ERR_FORMAT = -6


class ZquantError(RuntimeError):
    """A zquant call failed. `status` is the library's status code, where there is one."""

    def __init__(self, message: str, status: Optional[int] = None):
        super().__init__(message)
        self.status = status


class OutOfMemoryError(ZquantError, MemoryError):
    """The library could not allocate (ZQ_ERR_ALLOC)."""


class InvalidArgumentError(ZquantError, ValueError):
    """The library rejected an argument (ZQ_ERR_INVALID): a NaN or infinity in the
    input, a size it cannot hold, an id past the end of the index, and so on."""


class StateError(ZquantError):
    """The call is not valid in the object's current state (ZQ_ERR_STATE): calibrating
    an index that already holds vectors, searching an empty one, using a closed one."""


class UnsupportedError(ZquantError):
    """The library does not support this configuration (ZQ_ERR_UNSUPPORTED)."""


class CpuNotSupportedError(ZquantError):
    """This build of the library needs instructions this CPU lacks (ZQ_ERR_CPU)."""


class FormatError(ZquantError, ValueError):
    """Bytes given to `Index.from_bytes` or `Index.load` are not an intact index from a
    compatible version (ZQ_ERR_FORMAT)."""


class LibraryNotFoundError(OSError):
    """No native library was found where the binding looks for one."""


_BY_STATUS: Dict[int, Type[ZquantError]] = {
    ZQ_ERR_ALLOC: OutOfMemoryError,
    ZQ_ERR_INVALID: InvalidArgumentError,
    ZQ_ERR_STATE: StateError,
    ZQ_ERR_UNSUPPORTED: UnsupportedError,
    ZQ_ERR_CPU: CpuNotSupportedError,
    ZQ_ERR_FORMAT: FormatError,
}


def error_for(status: int, context: str, describe: Callable[[int], str]) -> ZquantError:
    """The exception for a failed call. `describe` turns a status into the library's own
    text for it, and `context` names what was being attempted."""
    cls = _BY_STATUS.get(status, ZquantError)
    return cls(f"{context}: {describe(status)} ({status})", status)
