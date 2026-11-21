"""Replay the cross-language conformance fixture through the public API.

tests/golden/conformance.json is checked against the library itself by the Zig suite,
so a hash that differs here is a bug in this binding: a field passed wrongly, a seed
rounded on its way to C, a buffer read at the wrong length.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import zquant

FIXTURE = Path(__file__).resolve().parents[3] / "tests" / "golden" / "conformance.json"
GOLDEN = json.loads(FIXTURE.read_text())


def fixture_rows(rows, dim, salt):
    """x[r][j] = ((r*7919 + j*104729 + salt) mod 2001 - 1000) / 1024, as float32.

    Integers over a power of two, so the float32 values are exact in every language.
    """
    r = np.arange(rows, dtype=np.int64)[:, None]
    j = np.arange(dim, dtype=np.int64)[None, :]
    return (((r * 7919 + j * 104729 + salt) % 2001 - 1000) / 1024).astype(np.float32)


def seed_of(case):
    # Hex strings, because some exceed 2^53 and would not survive as JSON numbers.
    # Python ints are exact, and the binding refuses anything outside uint64.
    return int(case["seed"], 16)


def test_fixture_formula_matches_a_hand_computed_value():
    x = fixture_rows(3, 5, salt=7)
    assert x[2, 4] == np.float32(((2 * 7919 + 4 * 104729 + 7) % 2001 - 1000) / 1024)


def test_large_seeds_are_exact():
    big = [c for c in GOLDEN["index"] + GOLDEN["codec"] if seed_of(c) > 2 ** 53]
    assert big, "the fixture is meant to include seeds above 2^53"
    assert seed_of(big[0]) == 0xFFFFFFFFFFFFFFC5
    assert float(seed_of(big[0])) != seed_of(big[0])   # a double would have rounded it


@pytest.mark.parametrize("case", GOLDEN["index"],
                         ids=lambda c: f"d{c['dim']}-b{c['bits']}-{c['metric']}"
                                       + (f"-l{c['lists']}" if c.get("lists") else ""))
def test_index_case(case):
    x = fixture_rows(case["rows"], case["dim"], case["salt"])
    with zquant.Index(dim=case["dim"], bits=case["bits"], metric=case["metric"],
                      seed=seed_of(case), compact=case["compact"]) as ix:
        if case["calibrate"]:
            ix.calibrate(x)
        if case.get("lists"):
            ix.partition(x, lists=case["lists"])
        ix.add(x)
        assert hashlib.sha256(ix.to_bytes()).hexdigest() == case["sha256"]


@pytest.mark.parametrize("case", GOLDEN["codec"], ids=lambda c: f"d{c['dim']}-b{c['bits']}")
def test_codec_case(case):
    x = fixture_rows(case["rows"], case["dim"], case["salt"])
    with zquant.Codec(dim=case["dim"], bits=case["bits"], seed=seed_of(case)) as c:
        codes, norms = c.encode(x)
    h = hashlib.sha256(codes.tobytes() + norms.astype("<f4").tobytes())
    assert h.hexdigest() == case["sha256"]
