#!/bin/sh
# Build an npm tarball with the native library inside it.
#
# Copies the built library into prebuilds/<platform>-<arch>/, where the binding looks
# before anywhere else, then runs `npm pack`. The tarball then works with no
# environment variable set and no repository checkout beside it.
#
#   zig build lib                      # at the repository root, first
#   sh scripts/pack.sh                 # or: npm run pack:native
#
# Environment:
#   ZQUANT_LIB_DIR       where the built library is (default: the repo's zig-out/lib)
#   ZQUANT_PORTABLE_LIB  a portable build to include as well, e.g. the libzquant.so from
#                        a release's -portable archive. It is renamed to
#                        libzquant-portable.*, the name the binding falls back to on a
#                        CPU that cannot run the release build. A libzquant-portable.*
#                        already in ZQUANT_LIB_DIR is included without this.
#
# The prebuilds/ directory is removed again afterwards, so a later `npm test` cannot
# pick up a stale copy instead of a fresh zig-out/lib.
set -eu

pkg=$(cd "$(dirname "$0")/.." && pwd)
repo=$(cd "$pkg/../.." && pwd)
plat=$(node -p 'process.platform + "-" + process.arch')

case "$plat" in
  darwin-*) main=libzquant.dylib; portable=libzquant-portable.dylib ;;
  win32-*)  main=zquant.dll;      portable=zquant-portable.dll ;;
  *)        main=libzquant.so;    portable=libzquant-portable.so ;;
esac

src=${ZQUANT_LIB_DIR:-$repo/zig-out/lib}
# Zig installs Windows DLLs to bin rather than lib.
if [ -z "${ZQUANT_LIB_DIR:-}" ] && [ ! -f "$src/$main" ] && [ -f "$repo/zig-out/bin/$main" ]; then
  src=$repo/zig-out/bin
fi
if [ ! -f "$src/$main" ]; then
  echo "pack.sh: no $main in $src. Run \`zig build lib\` at the repository root first." >&2
  exit 1
fi

dest=$pkg/prebuilds/$plat
trap 'rm -rf "$pkg/prebuilds"' EXIT
rm -rf "$pkg/prebuilds"
mkdir -p "$dest"

cp "$src/$main" "$dest/$main"
echo "pack.sh: $src/$main -> prebuilds/$plat/$main"
if [ -n "${ZQUANT_PORTABLE_LIB:-}" ]; then
  cp "$ZQUANT_PORTABLE_LIB" "$dest/$portable"
  echo "pack.sh: $ZQUANT_PORTABLE_LIB -> prebuilds/$plat/$portable"
elif [ -f "$src/$portable" ]; then
  cp "$src/$portable" "$dest/$portable"
  echo "pack.sh: $src/$portable -> prebuilds/$plat/$portable"
else
  echo "pack.sh: no portable build found; packing the release library only"
fi

cd "$pkg"
npm pack
