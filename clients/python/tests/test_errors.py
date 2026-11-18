"""Bad arguments are caught up front with a message that names them, and every status
the library can return arrives as its own exception."""
import numpy as np
import pytest

import zquant
from zquant import _errors


def unit_rows(n, d, seed=0):
    x = np.random.default_rng(seed).standard_normal((n, d)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


@pytest.mark.parametrize("dim", [0, -1, zquant.MAX_DIM + 1])
def test_dim_out_of_range(dim):
    with pytest.raises(ValueError, match="dim must be in 1..65536"):
        zquant.Index(dim=dim)
    with pytest.raises(ValueError, match="dim must be in 1..65536"):
        zquant.Codec(dim=dim)


@pytest.mark.parametrize("bad", ["64", 64.0, True, None])
def test_dim_must_be_an_integer(bad):
    with pytest.raises(TypeError, match="dim must be an integer"):
        zquant.Index(dim=bad)


def test_numpy_integers_are_integers():
    with zquant.Index(dim=np.int64(8), bits=np.uint8(4)) as ix:
        assert ix.dim == 8


@pytest.mark.parametrize("bits", [0, 1, 7])
def test_index_bits_are_2_to_6(bits):
    with pytest.raises(ValueError, match="bits must be in 2..6"):
        zquant.Index(dim=8, bits=bits)


@pytest.mark.parametrize("bits", [0, 7])
def test_codec_bits_are_1_to_6(bits):
    with pytest.raises(ValueError, match="bits must be in 1..6"):
        zquant.Codec(dim=8, bits=bits)


def test_codec_accepts_one_bit():
    with zquant.Codec(dim=8, bits=1) as c:
        assert c.code_bytes == 1


@pytest.mark.parametrize("seed", [-1, 1 << 64])
def test_seed_must_fit_in_64_bits(seed):
    with pytest.raises(ValueError, match="seed"):
        zquant.Index(dim=8, seed=seed)
    with pytest.raises(ValueError, match="seed"):
        zquant.Codec(dim=8, seed=seed)


def test_largest_seed_is_accepted():
    with zquant.Index(dim=8, seed=(1 << 64) - 1):
        pass


def test_threads_limit():
    with zquant.Index(dim=8) as ix:
        ix.add(unit_rows(10, 8))
        with pytest.raises(ValueError, match="threads must be in 0..1024"):
            ix.search(unit_rows(1, 8), k=1, threads=zquant.MAX_THREADS + 1)


def test_nan_in_vectors_is_refused_whole():
    x = unit_rows(10, 8)
    x[7, 3] = np.nan
    with zquant.Index(dim=8) as ix:
        with pytest.raises(zquant.InvalidArgumentError, match="NaN or infinity") as e:
            ix.add(x)
        assert isinstance(e.value, ValueError) and e.value.status == -2
        assert len(ix) == 0


def test_infinity_in_a_query_is_refused():
    with zquant.Index(dim=8) as ix:
        ix.add(unit_rows(10, 8))
        q = unit_rows(1, 8)
        q[0, 0] = np.inf
        with pytest.raises(zquant.InvalidArgumentError, match="NaN or infinity"):
            ix.search(q, k=3)


def test_nan_in_codec_input_is_refused():
    x = unit_rows(4, 8)
    x[0, 0] = np.nan
    with zquant.Codec(dim=8) as c:
        with pytest.raises(zquant.InvalidArgumentError, match="NaN or infinity"):
            c.encode(x)


@pytest.mark.parametrize("bad", [np.ones((2, 8), dtype=np.complex64),
                                 np.array([["a"] * 8]), np.array([[object()] * 8])])
def test_non_real_input_is_a_type_error(bad):
    with zquant.Index(dim=8) as ix:
        with pytest.raises(TypeError, match="real numbers"):
            ix.add(bad)


def test_states_are_state_errors():
    x = unit_rows(10, 8)
    with zquant.Index(dim=8) as ix:
        with pytest.raises(zquant.StateError, match="empty"):
            ix.search(x[:1], k=1)
        ix.add(x)
        with pytest.raises(zquant.StateError, match="state"):
            ix.calibrate(x)


@pytest.mark.parametrize("status, cls, builtin", [
    (-1, zquant.OutOfMemoryError, MemoryError),
    (-2, zquant.InvalidArgumentError, ValueError),
    (-3, zquant.StateError, RuntimeError),
    (-4, zquant.UnsupportedError, RuntimeError),
    (-5, zquant.CpuNotSupportedError, RuntimeError),
    (-6, zquant.FormatError, ValueError),
    (-99, zquant.ZquantError, RuntimeError),
])
def test_every_status_has_an_exception(status, cls, builtin):
    err = _errors.error_for(status, "op", zquant._status_text)
    assert type(err) is cls
    assert isinstance(err, zquant.ZquantError) and isinstance(err, builtin)
    assert err.status == status
    assert str(err).startswith("op: ") and f"({status})" in str(err)


def test_status_text_comes_from_the_library():
    assert "CPU" in str(_errors.error_for(-5, "op", zquant._status_text))
    assert "incompatible version" in str(_errors.error_for(-6, "op", zquant._status_text))


def test_inputs_are_converted_not_rejected():
    """float64, float16, ints, Fortran order and strided views all reach the library
    as the same C-contiguous float32 rows."""
    base = unit_rows(200, 16, seed=3)
    q = unit_rows(5, 16, seed=4)
    with zquant.Index(dim=16) as ix:
        ix.add(base)
        want_ids, want_scores = ix.search(q, k=5)
        wide = np.zeros((5, 32), dtype=np.float32)
        wide[:, ::2] = q
        for variant in (q.astype(np.float64), np.asfortranarray(q), wide[:, ::2],
                        q.tolist()):
            ids, scores = ix.search(variant, k=5)
            assert (ids == want_ids).all()
            assert np.array_equal(scores, want_scores)
        ix.add(np.arange(16 * 3).reshape(3, 16))               # integers
        ix.add(base[:2].astype(np.float16))
        assert len(ix) == 205
