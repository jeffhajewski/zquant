"""Wheel tagging. Everything else about the package is in pyproject.toml.

The binding is pure Python over ctypes, so with no native library in the package a wheel
is `py3-none-any` and finds the library through ZQUANT_LIB or a development build. With
one copied into zquant/ (scripts/build_wheel.sh does that) the wheel only works on the
platform the library was built for, so it is tagged for that platform. The tag is
`py3-none-<platform>` rather than CPython-specific: ctypes has no Python ABI to match.
"""
import os
import platform
import sys
from pathlib import Path

from setuptools import Distribution, setup

try:
    from setuptools.command.bdist_wheel import bdist_wheel
except ImportError:  # setuptools < 70.1 kept it in the wheel package
    from wheel.bdist_wheel import bdist_wheel

HERE = Path(__file__).resolve().parent
NATIVE = ("libzquant.so", "libzquant.dylib", "zquant.dll",
          "libzquant-portable.so", "libzquant-portable.dylib", "zquant-portable.dll")


def bundles_native_library() -> bool:
    return any((HERE / "zquant" / name).is_file() for name in NATIVE)


class NativeDistribution(Distribution):
    # Saying the package has extension modules is what makes setuptools build a
    # platform wheel and put the package at the wheel's root (platlib) rather than
    # under .data/purelib.
    def has_ext_modules(self):
        return bundles_native_library()


class platform_wheel(bdist_wheel):
    def get_tag(self):
        python, abi, plat = super().get_tag()
        if self.root_is_pure:
            return python, abi, plat
        # A Python built as universal2 reports that as its platform, but the library
        # beside it is built for one architecture. Claim only that one, unless the
        # caller chose a platform explicitly with --plat-name.
        explicit = getattr(self, "plat_name_supplied", False)
        if sys.platform == "darwin" and plat.endswith("universal2") and not explicit:
            plat = plat[: -len("universal2")] + platform.machine()
        # bdist_wheel will not tag below the macOS the *interpreter* was built for, which
        # is right for a compiled extension and wrong here: the only native code is our
        # library, built for MACOSX_DEPLOYMENT_TARGET by scripts/build_wheel.sh. Without
        # this, a wheel built with Homebrew's Python on macOS 26 claims to need 26.
        target = os.environ.get("MACOSX_DEPLOYMENT_TARGET")
        if sys.platform == "darwin" and target and not explicit:
            major, _, minor = target.partition(".")
            minor = minor.split(".")[0] if minor else "0"
            if int(major) >= 11:
                minor = "0"
            plat = "macosx_{}_{}_{}".format(major, minor, platform.machine())
        return "py3", "none", plat


setup(distclass=NativeDistribution, cmdclass={"bdist_wheel": platform_wheel})
