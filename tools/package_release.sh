#!/usr/bin/env bash
# Build the Python wheels and the npm package from a release's library archives.
#
#   tools/package_release.sh <dir with zquant-*.tar.gz> <output dir>
#
# The packages are made from the archives rather than from fresh builds, so every
# package holds the same library, byte for byte, as the archive for its platform.
# Nothing here compiles; Zig built every platform's library in release.yml's build job.
#
# Writes one wheel per platform and one npm tarball holding every platform's library.
# Linux and Windows packages also carry the portable library, which the bindings fall
# back to on a CPU without SSSE3 (x86_64) or dot product (arm64).
#
# Needs python3 with pip, and node with npm.
set -euo pipefail

archives="$(cd "$1" && pwd)"
mkdir -p "$2"
out="$(cd "$2" && pwd)"
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
for archive in "$archives"/zquant-*.tar.gz; do
  tar xzf "$archive" -C "$work"
done

# release name | wheel platform tag | npm platform-arch | library inside the archive
#
# The tags must describe the libraries. Linux builds target glibc 2.17 (release.yml
# checks no newer symbol version crept in), which is what manylinux_2_17 promises.
# Zig's macOS builds have a minimum of macOS 13.0.
platforms=(
  "linux-x86_64   manylinux_2_17_x86_64   linux-x64    lib/libzquant.so"
  "linux-aarch64  manylinux_2_17_aarch64  linux-arm64  lib/libzquant.so"
  "macos-x86_64   macosx_13_0_x86_64      darwin-x64   lib/libzquant.dylib"
  "macos-aarch64  macosx_13_0_arm64       darwin-arm64 lib/libzquant.dylib"
  "windows-x86_64 win_amd64               win32-x64    bin/zquant.dll"
)

prebuilds="$work/prebuilds"
for row in "${platforms[@]}"; do
  read -r name tag npm_dir file <<<"$row"
  main="$work/zquant-$name/$file"
  if [ ! -f "$main" ]; then
    echo "package_release.sh: no $file in zquant-$name.tar.gz" >&2
    exit 1
  fi
  portable_file="$(dirname "$file")/$(basename "$file" | sed 's/zquant/zquant-portable/')"
  portable="$work/zquant-$name-portable/$portable_file"
  [ -f "$portable" ] || portable=""

  echo "==> wheel for $name ($tag)${portable:+, with the portable library}"
  WHEEL_LIB="$main" WHEEL_PORTABLE_LIB="$portable" WHEEL_PLATFORM="$tag" WHEEL_DIR="$out" \
    bash "$repo/clients/python/scripts/build_wheel.sh" >/dev/null

  mkdir -p "$prebuilds/$npm_dir"
  cp "$main" "$prebuilds/$npm_dir/"
  if [ -n "$portable" ]; then cp "$portable" "$prebuilds/$npm_dir/"; fi
done

echo "==> npm package"
(
  cd "$repo/clients/typescript"
  npm install --no-audit --no-fund >/dev/null
  ZQUANT_PREBUILDS="$prebuilds" sh scripts/pack.sh
  mv zquant-*.tgz "$out/"
)

echo "==> packages"
ls -l "$out"
