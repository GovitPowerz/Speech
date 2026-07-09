# fsp_runtime -- resurrecting the 2015 `Segment` binary under Rosetta 2

Phase 4d needs the original 2015 production `fsp` binary (`Segment`) as the parity
oracle for Tasks 3/4. It is a `Mach-O x86_64` executable built in 2015 with g++-4.9
and `-ffast-math`. This directory rebuilds the runtime it needs so it runs under
Rosetta 2 on this Apple Silicon host.

Everything here that is an artifact (`bin/`, `prefix/`, `work/`) is **local-only and
gitignored**. Only `setup.sh`, this `README.md`, `.gitignore`, and
`phase4d_sources.json` are committed.

## Run it

```bash
./setup.sh            # full build + rewire + self-test (idempotent, re-runnable)
./setup.sh --check    # fast gate: binary present, dylibs resolve, usage smoke (Task 3 gates on this)

# usage smoke:
OMP_NUM_THREADS=1 arch -x86_64 bin/Segment                    # argc<3 -> prints usage, exits 1
# solo segmentation:
OMP_NUM_THREADS=1 arch -x86_64 bin/Segment -s <config>        # writes VRCTS xml per channel
```

`OMP_NUM_THREADS=1` is set for deterministic single-lane execution.

## What `setup.sh` does

1. **Copies** the read-only legacy `Segment` into `bin/` (never modifies the legacy tree).
2. **Source-builds 6 x86_64 deps** into `prefix/` with `clang -arch x86_64` under
   `arch -x86_64` (Rosetta runs the configure conftests). Era-appropriate versions
   give the **exact SONAME + compatibility version** the binary was linked against,
   so no symlink shims are needed:
   | dep | version | SONAME | compat (required / built) | build note |
   |-----|---------|--------|---------------------------|------------|
   | libogg | 1.3.5 | libogg.0 | 9.0.0 / 9.0.0 | -- |
   | libvorbis | 1.3.7 | libvorbisenc.2 | 3.0.0 / 3.0.0 | strip obsolete `-force_cpusubtype_ALL` from Makefiles |
   | flac | 1.3.4 | libFLAC.8 | 12.0.0 / 12.0.0 | flac 1.4.x bumped to libFLAC.12, so 1.3.x is required |
   | libpng | 1.6.37 | libpng16.16 | 32.0.0 / 54.0.0 | redirect the classic-Mac `<fp.h>` include to `<math.h>` (modern SDK defines `TARGET_OS_MAC`) |
   | matio | 1.5.2 | libmatio.2 | 3.0.0 / 3.0.0 (current 3.2.0, exact match) | `-DHAVE_VA_COPY=1` (bundled snprintf), `--without-hdf5` |
   | libsndfile | 1.0.31 | libsndfile.1 | 2.0.0 / 2.0.0 | `--disable-external-libs` (WAV decoded natively; see below) |
3. **Harvests the gcc runtime trio** (`libstdc++.6`, `libgomp.1`, `libgcc_s.1`) from a
   Homebrew **x86_64 gcc bottle** pulled directly from `ghcr.io/homebrew/core/gcc`
   (`14.2.0_1.sonoma`, an amd64/macOS bottle) via `curl` -- **not** `brew install`.
   libstdc++ is ABI-forward-compatible; modern gcc ships `libgcc_s.1.1.dylib`, renamed
   to the `libgcc_s.1.dylib` the binary asks for. Their `@@HOMEBREW_PREFIX@@`/`@rpath`
   IDs and inter-deps are rewritten to `prefix/`.
4. **Rewrites the 9 `/usr/local` install names** on the binary via `install_name_tool`
   and re-signs (`codesign -f -s -`) the binary plus every touched dylib.
5. **Self-tests**: argc<3 usage smoke + a solo `-s` run on the PRCTS wav.

### libsndfile note

The 2015 chain's libsndfile is 1.0.25-era (WAV/FLAC/vorbis, **pre-opus**; the binary's
recorded `libsndfile.1` current version is `2.25.0`). libsndfile 1.0.31 forces Opus
when external libs are enabled, so it is built `--disable-external-libs`. The oracle
corpus is WAV, which libsndfile decodes natively; the binary's own **direct** load
commands for `libFLAC.8`/`libogg.0`/`libvorbisenc.2` still resolve to the dylibs built
in step 2, so the whole dependency graph loads.

## Final `otool -L bin/Segment`

All 9 formerly-`/usr/local` install names now point at `prefix/lib`; only `libz.1` and
`libSystem.B` remain system libs. (The `current version` numbers below are the values
baked into the 2015 binary's load commands; dyld checks the built dylibs' actual compat
versions, all of which satisfy the requirement.)

```
	prefix/lib/libsndfile.1.dylib   (compat 2.0.0,  current 2.25.0)
	prefix/lib/libFLAC.8.dylib      (compat 12.0.0, current 12.0.0)
	prefix/lib/libogg.0.dylib       (compat 9.0.0,  current 9.2.0)
	prefix/lib/libvorbisenc.2.dylib (compat 3.0.0,  current 3.11.0)
	prefix/lib/libmatio.2.dylib     (compat 3.0.0,  current 3.2.0)
	/usr/lib/libz.1.dylib           (compat 1.0.0,  current 1.2.5)   [system]
	prefix/lib/libpng16.16.dylib    (compat 32.0.0, current 32.0.0)
	prefix/lib/libstdc++.6.dylib    (compat 7.0.0,  current 7.20.0)
	prefix/lib/libgomp.1.dylib      (compat 2.0.0,  current 2.0.0)
	/usr/lib/libSystem.B.dylib      (compat 1.0.0,  current 1225.1.1) [system]
	prefix/lib/libgcc_s.1.dylib     (compat 1.0.0,  current 1.0.0)
```

## Self-test evidence

**Usage smoke** (`arch -x86_64 bin/Segment`, argc<3) -> exit 1, prints:

```
Usage : Segment mode [config_options] config_file
    mode syntax: -s single config, -i single config with image output, ...
```

**Solo run** (`-s` on the PRCTS wav, tuple-A `1_worker_1.config` localized to a scratch
dir; weights = the real `NNweights_config1.bin`, since `BLSTMNeuralNetwork.cpp:132`
reads the weights via `BinaryFile2Vector` -- the `.mat` name in the committed config is
a dead code path):

- Corpus builds, the file is processed, the full BLSTM spectral SAD inference runs
  ("5 min 47.1 s of speech found out of 8 min 0.0 s of signal" over the 2x240 s stereo
  channels), and it writes **valid, parseable VRCTS segmentation**:
  `audio_chan_1.xml` (24 `SpeechSegment`s) and `audio_chan_2.xml`.
- The process then **SIGSEGVs (exit 139)** in the post-inference best-weight-save path
  (`BLSTMNeuralNetwork::saveWeights` -> `iof::fmtr` filename formatting), *after* the
  VRCTS output is written.

## Oracle decision: `real-binary` (with a documented save-path crash)

The binary is resurrected and produces the segmentation oracle artifact (VRCTS xml)
correctly and deterministically, so `phase4d_sources.json` records `oracle: real-binary`.

The exit-139 is a **resurrection artifact, not a 2015 bug**: the original binary links
`libstdc++.6` current `7.20.0` (gcc 4.9), and the 2015 legacy dir still contains
`weights_bestNNWeight_1_*.mat` -- proof that `saveWeights` completed on the original
toolchain. The closest obtainable x86_64-macOS libstdc++ is gcc-14's `7.33.0`; under it
the binary's old-ABI `std::string` / `iof::fmtr` interaction faults in the filename
format. gcc-11's `7.29.0` was tested and is worse (crashes *during* inference, before
VRCTS). The crash is content-independent (a short output filename crashes identically).

**Impact:** none on segmentation parity -- the VRCTS is written before the crash. The
`MultiConfigResults` `.mat` (cost/counter columns) is **not** written, because
`saveResults` runs after the crashing `saveAndUpdate`. Downstream extractors must
tolerate exit 139 and consume the VRCTS output; `setup.sh --check` therefore gates on
the (clean) usage smoke, not the solo run's exit code.

**The numeric-column complement (LIVE since Task 3):** the results `.mat` cost/counter
columns come from the `-ffast-math` REBUILD oracle -- `tools/oracle_harness/build.sh
--fastmath` -> `oracle_harness_fastmath` driving `phase4d_parity.cpp` -- which exits
cleanly and scores for real. The parity manifest records per-fixture provenance
(`oracle: real-binary` for the VRCTS legs, `oracle: fastmath-rebuild` for mcr); the two
oracles' VRCTS agree byte-for-byte on all committed combos. This binary remains the
SEGMENTATION oracle.

## Requirements

Rosetta 2 (`arch -x86_64 /usr/bin/true` must pass), `jq`, `curl`, Xcode command-line
tools (clang cross-compiles to x86_64), and network access to `downloads.xiph.org`,
`github.com`, `sourceforge.net`, and `ghcr.io`.

On another host, edit `setup.sh`'s hard-coded `LEGACY_ROOT` (the local
`FastSpeechProcessing-legacy` checkout path, outside this repo) and `GCC_VER` (the
pinned ghcr.io gcc-bottle tag) at the top of the file before running it.
