#!/usr/bin/env bash
#
# fsp_runtime/setup.sh -- resurrect the 2015 production `Segment` binary
# (Mach-O x86_64, g++-4.9, -ffast-math) so it runs under Rosetta 2 on this
# Apple Silicon host and can serve as the Phase 4d parity oracle.
#
# What it does (idempotent, re-runnable):
#   1. copy the read-only legacy `Segment` into bin/  (never touches legacy tree)
#   2. source-build the 6 non-gcc deps into prefix/ as x86_64
#      (era-appropriate versions -> exact SONAME + compat-version match)
#   3. extract the gcc runtime trio (libstdc++.6, libgomp.1, libgcc_s.1) from a
#      Homebrew x86_64 gcc BOTTLE (fetched from ghcr.io, NOT `brew install`)
#   4. rewrite all 9 absolute /usr/local install names to prefix/ and re-sign
#   5. self-test: argc<3 usage smoke + a solo -s run on the PRCTS wav
#
# Everything it produces (bin/, prefix/, work/) is LOCAL-ONLY and gitignored.
#
# Usage:
#   ./setup.sh            full build + self-test
#   ./setup.sh --check    fast gate (binary present, dylibs resolve, usage smoke)
#
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PREFIX="$HERE/prefix"
WORK="$HERE/work"
BIN="$HERE/bin/Segment"

LEGACY_ROOT="/Users/govit/Git/Govit/FastSpeechProcessing-legacy"
LEGACY_SEGMENT="$LEGACY_ROOT/Optimizer_V6.2.2/Executables/15-Oct-2015_BLSTM_OpenSAD15/Segment"

# The 9 absolute install names baked into the 2015 binary, mapped to prefix/.
# (/usr/lib/libz.1 and /usr/lib/libSystem.B are system libs, left untouched.)
declare -a RW_FROM=(
  "/usr/local/lib/libsndfile.1.dylib"
  "/usr/local/lib/libFLAC.8.dylib"
  "/usr/local/lib/libogg.0.dylib"
  "/usr/local/lib/libvorbisenc.2.dylib"
  "/usr/local/lib/libmatio.2.dylib"
  "/usr/local/lib/libpng16.16.dylib"
  "/usr/local/lib/gcc/4.9/libstdc++.6.dylib"
  "/usr/local/lib/gcc/4.9/libgomp.1.dylib"
  "/usr/local/lib/gcc/4.9/libgcc_s.1.dylib"
)
declare -a RW_TO=(
  "libsndfile.1.dylib" "libFLAC.8.dylib" "libogg.0.dylib" "libvorbisenc.2.dylib"
  "libmatio.2.dylib" "libpng16.16.dylib" "libstdc++.6.dylib" "libgomp.1.dylib"
  "libgcc_s.1.dylib"
)

# gcc bottle to harvest the runtime trio from (has x86_64 macOS bottles on ghcr).
GCC_VER="14.2.0_1"
GCC_PLATFORM="sonoma"   # amd64/macOS 14 bottle == x86_64

log() { printf '\n=== %s ===\n' "$*"; }

# ---------------------------------------------------------------------------
# --check: fast idempotent verification (Task 3's extractor gates on this)
# ---------------------------------------------------------------------------
do_check() {
  local ok=1
  if [ ! -x "$BIN" ]; then echo "FAIL: $BIN missing"; return 1; fi
  # every rewired dylib must exist and resolve; binary must have no /usr/local refs
  if otool -L "$BIN" | tail -n +2 | grep -q '/usr/local'; then
    echo "FAIL: binary still references /usr/local"; ok=0
  fi
  local d
  for d in "${RW_TO[@]}"; do
    [ -f "$PREFIX/lib/$d" ] || { echo "FAIL: prefix/lib/$d missing"; ok=0; }
  done
  # usage smoke: argc<3 prints usage text and exits 1
  local out
  out="$(OMP_NUM_THREADS=1 arch -x86_64 "$BIN" 2>&1 || true)"
  if ! grep -q 'Usage : Segment mode' <<<"$out"; then
    echo "FAIL: usage smoke did not print usage text"; ok=0
  fi
  if [ "$ok" = 1 ]; then echo "OK: fsp_runtime --check passed"; return 0; fi
  return 1
}

if [ "${1:-}" = "--check" ]; then do_check; exit $?; fi

# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------
arch -x86_64 /usr/bin/true 2>/dev/null || { echo "ERROR: Rosetta 2 not available (arch -x86_64 failed)"; exit 1; }
[ -f "$LEGACY_SEGMENT" ] || { echo "ERROR: legacy Segment not found at $LEGACY_SEGMENT"; exit 1; }
command -v jq >/dev/null || { echo "ERROR: jq required"; exit 1; }
mkdir -p "$HERE/bin" "$PREFIX/lib" "$PREFIX/include" "$WORK"

# ---------------------------------------------------------------------------
# 1. copy the read-only legacy binary (COPY, never modify legacy tree)
# ---------------------------------------------------------------------------
log "copy legacy Segment"
cp "$LEGACY_SEGMENT" "$BIN"
chmod u+w "$BIN"

# ---------------------------------------------------------------------------
# 2. source-build the 6 x86_64 deps into prefix/
#    era-appropriate versions give the exact SONAME + compat versions the
#    binary was linked against, so no symlink shims are needed.
# ---------------------------------------------------------------------------
export CC="clang"
export CFLAGS="-arch x86_64 -O2"
export CXXFLAGS="-arch x86_64 -O2"
export CPPFLAGS="-I$PREFIX/include"
export LDFLAGS="-arch x86_64 -L$PREFIX/lib"
export PKG_CONFIG_PATH="$PREFIX/lib/pkgconfig"

fetch() {  # url outfile
  # No sha256/checksum verification -- integrity relies solely on the version-pinned
  # upstream URL (see the dep table below) plus HTTPS transport trust.
  [ -f "$WORK/$2" ] && return 0
  echo "download $2"
  curl -fsSL -o "$WORK/$2" "$1"
}

extract() {  # tarball dir
  [ -d "$WORK/$2" ] && return 0
  echo "extract $1"
  case "$1" in
    *.tar.xz)  tar xJf "$WORK/$1" -C "$WORK" ;;
    *.tar.bz2) tar xjf "$WORK/$1" -C "$WORK" ;;
    *)         tar xzf "$WORK/$1" -C "$WORK" ;;
  esac
}

# generic autotools cross-build; extra configure flags passed after the dir.
build_dep() {  # target_dylib srcdir [configure flags...]
  local target="$1" dir="$2"; shift 2
  [ -f "$PREFIX/lib/$target" ] && { echo "skip $target (present)"; return 0; }
  echo "build $dir -> $target"
  ( cd "$WORK/$dir"
    arch -x86_64 env CC="$CC" CFLAGS="$CFLAGS" CXXFLAGS="$CXXFLAGS" \
      CPPFLAGS="$CPPFLAGS" LDFLAGS="$LDFLAGS" PKG_CONFIG_PATH="$PKG_CONFIG_PATH" \
      ./configure --prefix="$PREFIX" --disable-static --disable-dependency-tracking "$@" >/dev/null
    arch -x86_64 make -j4 >/dev/null
    make install >/dev/null )
}

log "build audio/matio/png stack (x86_64)"

# libogg 1.3.5 -> libogg.0 (compat 9.0.0)
fetch https://downloads.xiph.org/releases/ogg/libogg-1.3.5.tar.gz libogg-1.3.5.tar.gz
extract libogg-1.3.5.tar.gz libogg-1.3.5
build_dep libogg.0.dylib libogg-1.3.5

# libvorbis 1.3.7 -> libvorbisenc.2 (compat 3.0.0). Its configure injects the
# obsolete `-force_cpusubtype_ALL` linker flag for Darwin; strip it.
fetch https://downloads.xiph.org/releases/vorbis/libvorbis-1.3.7.tar.gz libvorbis-1.3.7.tar.gz
extract libvorbis-1.3.7.tar.gz libvorbis-1.3.7
if [ ! -f "$PREFIX/lib/libvorbisenc.2.dylib" ]; then
  ( cd "$WORK/libvorbis-1.3.7"
    arch -x86_64 env CC="$CC" CFLAGS="$CFLAGS" CPPFLAGS="$CPPFLAGS" LDFLAGS="$LDFLAGS" \
      PKG_CONFIG_PATH="$PKG_CONFIG_PATH" \
      ./configure --prefix="$PREFIX" --disable-static --disable-dependency-tracking >/dev/null
    find . -name Makefile -exec sed -i '' 's/-force_cpusubtype_ALL//g' {} +
    arch -x86_64 make -j4 >/dev/null
    make install >/dev/null )
fi

# flac 1.3.4 -> libFLAC.8 (compat 12.0.0); flac 1.4.x bumped to libFLAC.12.
fetch https://downloads.xiph.org/releases/flac/flac-1.3.4.tar.xz flac-1.3.4.tar.xz
extract flac-1.3.4.tar.xz flac-1.3.4
build_dep libFLAC.8.dylib flac-1.3.4 --disable-cpplibs --disable-examples --disable-doxygen-docs

# libpng 1.6.37 -> libpng16.16 (compat 54.0.0 >= required 32.0.0). pngpriv.h
# takes the classic-Mac <fp.h> branch because the modern SDK defines
# TARGET_OS_MAC; redirect that include to <math.h>.
fetch https://download.sourceforge.net/libpng/libpng-1.6.37.tar.gz libpng-1.6.37.tar.gz
extract libpng-1.6.37.tar.gz libpng-1.6.37
if [ ! -f "$PREFIX/lib/libpng16.16.dylib" ]; then
  sed -i '' 's|#      include <fp.h>|#      include <math.h>|' "$WORK/libpng-1.6.37/pngpriv.h" || true
  build_dep libpng16.16.dylib libpng-1.6.37
fi

# matio 1.5.2 -> libmatio.2 (compat 3.0.0, current 3.2.0 == the 2015 binary's
# exact libmatio.2 version). Its bundled snprintf.c uses a broken VA_COPY when
# HAVE_VA_COPY is unset; force it. Build without hdf5 (engine writes v5 .mat).
fetch https://github.com/tbeu/matio/releases/download/v1.5.2/matio-1.5.2.tar.gz matio-1.5.2.tar.gz
extract matio-1.5.2.tar.gz matio-1.5.2
if [ ! -f "$PREFIX/lib/libmatio.2.dylib" ]; then
  ( cd "$WORK/matio-1.5.2"
    arch -x86_64 env CC="$CC" CFLAGS="-arch x86_64 -O2 -DHAVE_VA_COPY=1" \
      CPPFLAGS="$CPPFLAGS" LDFLAGS="$LDFLAGS" \
      ./configure --prefix="$PREFIX" --disable-static --disable-dependency-tracking --without-hdf5 >/dev/null
    arch -x86_64 make -j4 >/dev/null
    make install >/dev/null )
fi

# libsndfile 1.0.31 -> libsndfile.1 (compat 2.0.0). Built --disable-external-libs:
# the 2015 chain's libsndfile is 1.0.25-era (WAV/FLAC/vorbis, pre-opus) and the
# oracle corpus is WAV (decoded natively). The binary's own direct load commands
# for libFLAC.8/libogg.0/libvorbisenc.2 still resolve to the dylibs built above.
fetch https://github.com/libsndfile/libsndfile/releases/download/1.0.31/libsndfile-1.0.31.tar.bz2 libsndfile-1.0.31.tar.bz2
extract libsndfile-1.0.31.tar.bz2 libsndfile-1.0.31
build_dep libsndfile.1.dylib libsndfile-1.0.31 --disable-sqlite --disable-external-libs

# ---------------------------------------------------------------------------
# 3. harvest the gcc runtime trio from an x86_64 Homebrew gcc bottle (ghcr.io).
#    libstdc++ is ABI-forward-compatible; libgomp.1/libgcc_s stay stable.
#    Modern gcc ships libgcc_s.1.1.dylib -> renamed to the libgcc_s.1.dylib
#    the 2015 binary asks for.
# ---------------------------------------------------------------------------
log "harvest gcc runtime trio (x86_64 bottle)"
if [ ! -f "$PREFIX/lib/libstdc++.6.dylib" ] || [ ! -f "$PREFIX/lib/libgomp.1.dylib" ] || [ ! -f "$PREFIX/lib/libgcc_s.1.dylib" ]; then
  TOKEN="$(curl -s "https://ghcr.io/token?service=ghcr.io&scope=repository:homebrew/core/gcc:pull" | jq -r .token)"
  # index -> the amd64/<platform> image manifest digest
  IMG_DIGEST="$(curl -s -H "Authorization: Bearer $TOKEN" \
      -H "Accept: application/vnd.oci.image.index.v1+json" \
      "https://ghcr.io/v2/homebrew/core/gcc/manifests/$GCC_VER" \
    | jq -r --arg ref "$GCC_VER.$GCC_PLATFORM" '.manifests[] | select(.annotations["org.opencontainers.image.ref.name"]==$ref) | .digest')"
  [ -n "$IMG_DIGEST" ] || { echo "ERROR: no $GCC_VER.$GCC_PLATFORM bottle on ghcr"; exit 1; }
  BLOB_DIGEST="$(curl -s -H "Authorization: Bearer $TOKEN" \
      -H "Accept: application/vnd.oci.image.manifest.v1+json" \
      "https://ghcr.io/v2/homebrew/core/gcc/manifests/$IMG_DIGEST" | jq -r '.layers[0].digest')"
  [ -f "$WORK/gcc-x86_64.tar.gz" ] || \
    curl -sL -H "Authorization: Bearer $TOKEN" "https://ghcr.io/v2/homebrew/core/gcc/blobs/$BLOB_DIGEST" -o "$WORK/gcc-x86_64.tar.gz"
  GBASE="gcc/$GCC_VER/lib/gcc/current"
  rm -rf "$WORK/gcc_extract"; mkdir -p "$WORK/gcc_extract"
  tar xzf "$WORK/gcc-x86_64.tar.gz" -C "$WORK/gcc_extract" \
    "$GBASE/libstdc++.6.dylib" "$GBASE/libgomp.1.dylib" "$GBASE/libgcc_s.1.1.dylib"
  cp "$WORK/gcc_extract/$GBASE/libstdc++.6.dylib"  "$PREFIX/lib/libstdc++.6.dylib"
  cp "$WORK/gcc_extract/$GBASE/libgomp.1.dylib"    "$PREFIX/lib/libgomp.1.dylib"
  cp "$WORK/gcc_extract/$GBASE/libgcc_s.1.1.dylib" "$PREFIX/lib/libgcc_s.1.dylib"
  chmod u+w "$PREFIX/lib/libstdc++.6.dylib" "$PREFIX/lib/libgomp.1.dylib" "$PREFIX/lib/libgcc_s.1.dylib"
  # fix the trio's own IDs + the @rpath/@@HOMEBREW deps
  install_name_tool -id "$PREFIX/lib/libstdc++.6.dylib" "$PREFIX/lib/libstdc++.6.dylib"
  install_name_tool -change "@rpath/libgcc_s.1.1.dylib" "$PREFIX/lib/libgcc_s.1.dylib" "$PREFIX/lib/libstdc++.6.dylib"
  install_name_tool -id "$PREFIX/lib/libgomp.1.dylib"  "$PREFIX/lib/libgomp.1.dylib"
  install_name_tool -id "$PREFIX/lib/libgcc_s.1.dylib" "$PREFIX/lib/libgcc_s.1.dylib"
fi

# ---------------------------------------------------------------------------
# 4. rewrite the 9 install names on the binary + re-sign everything touched
# ---------------------------------------------------------------------------
log "rewire install names + codesign"
args=()
for i in "${!RW_FROM[@]}"; do
  args+=( -change "${RW_FROM[$i]}" "$PREFIX/lib/${RW_TO[$i]}" )
done
install_name_tool "${args[@]}" "$BIN"

for f in "$PREFIX"/lib/*.dylib; do codesign -f -s - "$f" >/dev/null 2>&1 || true; done
codesign -f -s - "$BIN"

if otool -L "$BIN" | tail -n +2 | grep -q '/usr/local'; then
  echo "ERROR: binary still references /usr/local after rewire"; exit 1
fi

# ---------------------------------------------------------------------------
# 5. self-test
# ---------------------------------------------------------------------------
log "self-test: argc<3 usage smoke"
OMP_NUM_THREADS=1 arch -x86_64 "$BIN" 2>&1 | sed -n '1,3p' || true

log "self-test: solo -s run on the PRCTS wav"
S="$WORK/selftest"
rm -rf "$S"; mkdir -p "$S"
PRCTS="$LEGACY_ROOT/PRCTS_RUS_RU_0000263489_01.wav"
TUPLEA="$LEGACY_ROOT/Optimizer_V6.2.2/Executables/15-Oct-2015_BLSTM_OpenSAD15/1_worker_1.config"
WEIGHTS="$LEGACY_ROOT/Optimizer_V6.2.2/Executables/15-Oct-2015_BLSTM_OpenSAD15/NNweights_config1.bin"
LANGMAP="$LEGACY_ROOT/ConfigFiles/languagemapping.csv"
if [ -f "$PRCTS" ] && [ -f "$TUPLEA" ] && [ -f "$WEIGHTS" ] && [ -f "$LANGMAP" ]; then
  cp "$PRCTS" "$S/audio.wav"; cp "$WEIGHTS" "$S/NNweights_config1.bin"; cp "$LANGMAP" "$S/languagemapping.csv"
  : > "$S/audio.stm"
  printf '%s;%s;fax;non;1;30.0;\n' "$S/audio.wav" "$S/audio.stm" > "$S/fileslisting"
  # BLSTMSpectralSegmenter reads the weights via BinaryFile2Vector (the .mat name
  # in the committed config is a dead code path; the live loader wants the .bin).
  sed -E \
    -e "s#^Display_Output_Directory .*#Display_Output_Directory $S#" \
    -e "s#^BLSTM_weightsFile .*#BLSTM_weightsFile $S/NNweights_config1.bin#" \
    -e "s#^language2classmapping .*#language2classmapping $S/languagemapping.csv#" \
    -e "s#^fileslisting .*#fileslisting $S/fileslisting#" \
    "$TUPLEA" > "$S/tupleA_local.config"
  rc=0
  ( cd "$S"
    OMP_NUM_THREADS=1 arch -x86_64 "$BIN" -s "$S/tupleA_local.config" > "$S/stdout.txt" 2>&1 ) || rc=$?
  echo "solo exit code: $rc  (139 == post-inference save-path SIGSEGV, see README)"
  echo "VRCTS outputs written:"
  ls -1 "$S"/*.xml 2>/dev/null || echo "  (none)"
else
  echo "SKIP solo self-test (legacy fixtures not present)"
fi

log "done"
echo "final otool -L:"
otool -L "$BIN"
