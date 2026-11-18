"""Locking for an Index shared between Python threads.

ctypes releases the GIL for the length of every native call, so two Python threads
really are inside the library at once. The library's rules are:

* A searcher holds per-query scratch and must not be used by two calls at once.
  Distinct searchers on one index may search concurrently.
* A call that modifies the index (add, calibrate) must not overlap any other call on
  that index, and freeing it obviously must not either.

So an Index holds a readers-writer lock - searches are readers, add, calibrate and
close are writers - and a pool of searchers from which each search checks out its own.
"""
from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Callable, Dict, Iterator, List, Tuple


class RWLock:
    """Many readers or one writer.

    Writer-preferring: once a writer is waiting, new readers queue behind it, so a
    steady stream of searches cannot starve an add or a close forever. Not reentrant -
    a thread holding it must not acquire it again.
    """

    def __init__(self) -> None:
        self._cond = threading.Condition(threading.Lock())
        self._readers = 0
        self._writer = False
        self._writers_waiting = 0

    @contextmanager
    def read(self) -> Iterator[None]:
        with self._cond:
            while self._writer or self._writers_waiting:
                self._cond.wait()
            self._readers += 1
        try:
            yield
        finally:
            with self._cond:
                self._readers -= 1
                if self._readers == 0:
                    self._cond.notify_all()

    @contextmanager
    def write(self) -> Iterator[None]:
        with self._cond:
            self._writers_waiting += 1
            try:
                while self._writer or self._readers:
                    self._cond.wait()
            finally:
                self._writers_waiting -= 1
            self._writer = True
        try:
            yield
        finally:
            with self._cond:
                self._writer = False
                self._cond.notify_all()


Key = Tuple[int, int]


class SearcherPool:
    """Idle searchers, by (k, threads), each in use by at most one call at a time.

    The pool grows to the largest number of searches that have run at once for a
    configuration and keeps them, since creating a searcher allocates its scratch.
    Callers must hold the index's read lock across `checkout`, the search and `checkin`,
    and the write lock across `free_all`.
    """

    def __init__(self, create: Callable[[int, int, int], int], free: Callable[[int], None]):
        # Plain functions of the index handle rather than bound methods of the Index, so
        # the pool holds no reference back to it and an Index is freed as soon as the
        # last reference goes, not whenever the cycle collector runs.
        self._create = create
        self._free = free
        self._lock = threading.Lock()
        self._idle: Dict[Key, List[int]] = {}
        self._count = 0

    def checkout(self, index: int, k: int, threads: int) -> int:
        key = (k, threads)
        with self._lock:
            idle = self._idle.get(key)
            if idle:
                return idle.pop()
        # Created outside the pool's lock, so one thread allocating does not hold up
        # others that only need an idle searcher.
        handle = self._create(index, k, threads)
        with self._lock:
            self._count += 1
        return handle

    def checkin(self, k: int, threads: int, handle: int) -> None:
        with self._lock:
            self._idle.setdefault((k, threads), []).append(handle)

    @contextmanager
    def borrow(self, index: int, k: int, threads: int) -> Iterator[int]:
        handle = self.checkout(index, k, threads)
        try:
            yield handle
        finally:
            self.checkin(k, threads, handle)

    def __len__(self) -> int:
        """Searchers created and not yet freed."""
        with self._lock:
            return self._count

    def free_all(self) -> None:
        with self._lock:
            idle, self._idle, self._count = self._idle, {}, 0
        for handles in idle.values():
            for h in handles:
                self._free(h)
