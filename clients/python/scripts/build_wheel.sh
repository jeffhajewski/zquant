#!/usr/bin/env bash
# Build a wheel of the Python binding with the native library inside it.
#
#   clients/python/scripts/build_wheel.sh [extra pip wheel arguments]
#
# Builds the library the way a release does, for the CPU the release targets on this
# architecture (SSSE3 on x86_64, dotprod on arm64) rather than for this exact machine,
# copies it into the package, and runs `pip wheel`. Where that CPU is above the
# architecture baseline it also builds a portable baseline library and bundles it as
# libzquant-portable.*, which the binding falls back to on a CPU without those
# instructions. The copies are removed again afterwards, so a development checkout keeps
# loading zig-out/lib.
#
# Environment:
#   PYTHON           interpreter to run pip with (default: python3)
#   WHEEL_DIR        where the wheel goes (default: clients/python/dist)
#   ZIG_CPU          override the -Dcpu the main library is built for
#   ZQUANT_PORTABLE  0 to skip the portable library, 1 to build it even where the
#                    release CPU is already the baseline
#   MACOSX_DEPLOYMENT_TARGET
#                    oldest macOS the wheel supports (default 13.0). A native build
#                    stamps the library with this machine's macOS version, and
#                    bdist_wheel tags the wheel from the library, so without this a
#                    wheel built on macOS 26 refuses to install anywhere older.
#
# Without network access, pass --no-build-isolation (setuptools and wheel must then be
# installed in PYTHON's environment already).
set -euo pipefail

pkg_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo_root="$(cd "$pkg_root/../.." && pwd)"
python="${PYTHON:-python3}"
wheel_dir="${WHEEL_DIR:-$pkg_root/dist}"

case "$(uname -s)" in
  Darwin)               lib=libzquant.dylib; portable=libzquant-portable.dylib ;;
  MINGW*|MSYS*|CYGWIN*) lib=zquant.dll;      portable=zquant-portable.dll ;;
  *)                    lib=libzquant.so;    portable=libzquant-portable.so ;;
esac

# The CPUs release.yml builds for. macOS's baselines already have the instructions.
case "$(uname -s)/$(uname -m)" in
  Darwin/*)                release_cpu=baseline ;;
  */x86_64|*/amd64)        release_cpu=x86_64_v2 ;;
  */aarch64|*/arm64)       release_cpu=generic+dotprod ;;
  *)                       release_cpu=baseline ;;
esac
cpu="${ZIG_CPU:-$release_cpu}"

# An explicit target only where the OS version needs pinning; elsewhere the host's.
target_args=()
if [ "$(uname -s)" = Darwin ]; then
  export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-13.0}"
  case "$(uname -m)" in arm64) zarch=aarch64 ;; *) zarch="$(uname -m)" ;; esac
  target_args=(-Dtarget="$zarch-macos.$MACOSX_DEPLOYMENT_TARGET")
fi
if [ "${ZQUANT_PORTABLE:-}" = 1 ] || { [ "${ZQUANT_PORTABLE:-}" != 0 ] && [ "$cpu" != baseline ]; }; then
  want_portable=1
else
  want_portable=0
fi

work="$(mktemp -d)"
cleanup() {
  rm -f "$pkg_root/zquant/$lib" "$pkg_root/zquant/$portable"
  # setuptools stages package files in build/; a library left there would end up in
  # the next wheel even after the copy above is gone.
  rm -rf "$pkg_root/build" "$work"
}
trap cleanup EXIT
rm -rf "$pkg_root/build"

# Built into a scratch prefix so the developer's zig-out/ is left as it was.
built_lib() {
  local prefix="$1" name="$2"
  for dir in lib bin; do
    if [ -f "$prefix/$dir/$name" ]; then echo "$prefix/$dir/$name"; return; fi
  done
  echo "build_wheel.sh: zig build produced no $name under $prefix" >&2
  exit 1
}

echo "==> zig build lib -Dcpu=$cpu ${target_args[*]-}"
(cd "$repo_root" && zig build lib ${target_args[@]+"${target_args[@]}"} -Dcpu="$cpu" -p "$work/release")
cp "$(built_lib "$work/release" "$lib")" "$pkg_root/zquant/$lib"

if [ "$want_portable" = 1 ]; then
  # Built under its own name (-Dlib-name), so its SONAME / install name matches the
  # file rather than claiming to be libzquant.
  echo "==> zig build lib -Dcpu=baseline -Dlib-name=zquant-portable (portable)"
  (cd "$repo_root" && zig build lib ${target_args[@]+"${target_args[@]}"} -Dcpu=baseline -Dlib-name=zquant-portable -p "$work/portable")
  cp "$(built_lib "$work/portable" "$portable")" "$pkg_root/zquant/$portable"
elif [ -f "$repo_root/zig-out/lib/$portable" ]; then
  # Someone built or downloaded one by hand; ship it.
  cp "$repo_root/zig-out/lib/$portable" "$pkg_root/zquant/$portable"
fi

echo "==> pip wheel"
mkdir -p "$wheel_dir"
"$python" -m pip wheel --no-deps --wheel-dir "$wheel_dir" "$@" "$pkg_root"

echo "==> wheel contents"
for whl in "$wheel_dir"/zquant-*.whl; do
  echo "$whl"
  "$python" -m zipfile -l "$whl"
done
