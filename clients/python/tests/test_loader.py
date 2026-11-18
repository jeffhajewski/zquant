"""Finding the native library, and the CPU check that decides which build to use."""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import zquant
from zquant import _lib

PYTHON_ROOT = Path(__file__).resolve().parents[1]


def test_build_info_describes_the_loaded_library():
    info = zquant.build_info()
    assert info.version == zquant.version()
    assert info.library.is_file()
    assert info.library.name in (_lib.LIBNAME, _lib.PORTABLE_NAME)
    assert info.cpu
    assert info.kernels.startswith("zquant-kernels ")
    # A development build targets this machine, so it never needs the fallback.
    assert info.portable is (info.library.name == _lib.PORTABLE_NAME)


def test_search_order_without_override(monkeypatch):
    monkeypatch.delenv(_lib.ENV_VAR, raising=False)
    monkeypatch.delenv(_lib.LEGACY_ENV_VAR, raising=False)
    pkg = Path(zquant.__file__).resolve().parent
    found = _lib.candidates()
    assert found[0] == pkg / _lib.LIBNAME
    assert found[1].parts[-3:] == ("zig-out", "lib", _lib.LIBNAME)


def test_override_is_the_only_candidate(monkeypatch, tmp_path):
    target = tmp_path / "custom.lib"
    monkeypatch.setenv(_lib.ENV_VAR, str(target))
    assert _lib.candidates() == [target]
    with pytest.raises(zquant.LibraryNotFoundError, match=_lib.ENV_VAR):
        _lib.find()     # an explicit path that is wrong must not fall through


class FakeLib:
    """Stands in for a loaded library, so the fallback can be tested on a CPU that has
    every instruction the release build wants."""

    def __init__(self, path, supported, cpu):
        self.path, self._supported, self._cpu = path, supported, cpu

    def zq_cpu_supported(self):
        return 1 if self._supported else 0

    def zq_build_cpu(self):
        return self._cpu.encode()


def fake_opener(supported_names):
    def opener(path):
        return FakeLib(path, path.name in supported_names, "x86_64_v2")
    return opener


def touch(path):
    path.write_bytes(b"")
    return path


def test_supported_cpu_uses_the_release_build(tmp_path):
    main = touch(tmp_path / _lib.LIBNAME)
    touch(tmp_path / _lib.PORTABLE_NAME)
    lib, path, portable = _lib.select(main, fake_opener({_lib.LIBNAME, _lib.PORTABLE_NAME}))
    assert path == main and not portable and lib.path == main


def test_unsupported_cpu_falls_back_to_portable(tmp_path):
    main = touch(tmp_path / _lib.LIBNAME)
    port = touch(tmp_path / _lib.PORTABLE_NAME)
    lib, path, portable = _lib.select(main, fake_opener({_lib.PORTABLE_NAME}))
    assert path == port and portable and lib.path == port


def test_unsupported_cpu_without_portable_names_the_required_cpu(tmp_path):
    main = touch(tmp_path / _lib.LIBNAME)
    with pytest.raises(zquant.CpuNotSupportedError, match="x86_64_v2") as e:
        _lib.select(main, fake_opener(set()))
    assert _lib.PORTABLE_NAME in str(e.value)
    assert e.value.status == -5


def test_portable_that_also_fails_is_an_error(tmp_path):
    main = touch(tmp_path / _lib.LIBNAME)
    touch(tmp_path / _lib.PORTABLE_NAME)
    with pytest.raises(zquant.CpuNotSupportedError, match="also"):
        _lib.select(main, fake_opener(set()))


def run_python(code, env_extra, cwd):
    env = dict(os.environ)
    env.pop(_lib.ENV_VAR, None)
    env.pop(_lib.LEGACY_ENV_VAR, None)
    env.update(env_extra)
    env["PYTHONPATH"] = str(PYTHON_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run([sys.executable, "-c", code], env=env, cwd=str(cwd),
                          capture_output=True, text=True)


def test_env_var_loads_that_exact_file(tmp_path):
    copy = tmp_path / "elsewhere" / _lib.LIBNAME
    copy.parent.mkdir()
    shutil.copy2(zquant.build_info().library, copy)
    r = run_python("import zquant; print(zquant.build_info().library)",
                   {_lib.ENV_VAR: str(copy)}, tmp_path)
    assert r.returncode == 0, r.stderr
    assert Path(r.stdout.strip()).resolve() == copy.resolve()


def test_env_var_pointing_nowhere_fails_on_import(tmp_path):
    r = run_python("import zquant", {_lib.ENV_VAR: str(tmp_path / "missing")}, tmp_path)
    assert r.returncode != 0
    assert _lib.ENV_VAR in r.stderr and "missing" in r.stderr
