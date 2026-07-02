# Phase 1 - Features Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the legacy feature-extraction layer (audio preprocessing, exact GFFT periodogram, mel/MFCC/deltas/SDC, LTSV, TDC/pitch, streaming stats, param derivation + input assembly) bit-exactly against a strict-IEEE C++ rebuild of the vendored legacy sources.

**Architecture:** A local-only C++ oracle harness (`tools/oracle_harness/`) compiles the git-ignored legacy feature sources with strict IEEE flags and dumps every pipeline stage via the legacy's own `Matrix2BinaryFile`; committed `.bin` fixtures are consumed by Rust golden tests (`io::binary` reads them as-is). Each Rust module also gets hand-computed unit tests and (where the harness transcribes rather than compiles legacy code) an independent numpy differential oracle with committed JSON cases.

**Tech Stack:** Rust (ndarray, symphonia, existing `io::binary`/`legacy_config`/`constants`), C++ harness (Homebrew g++, Eigen, libsndfile, libmatio, boost), Python (numpy oracles, pytest).

## Global Constraints

- Branch: `feature/phase-1-features` (already created off `main`). NEVER commit to `main`. Do NOT rebase.
- Spec: `docs/superpowers/specs/2026-07-02-phase-1-features-design.md` - referenced per task as "spec S<n>". Legacy sources (git-ignored, present locally): `/Users/govit/Git/Govit/Speech/legacy/src/`.
- Commit trailer (every commit):
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`
- Legacy quirks are load-bearing. Do NOT fix them. Every reproduced quirk gets an IMPROVEMENTS.md entry (Task 12 consolidates).
- NEVER edit files under `legacy/`. Harness compile problems are solved with shim headers, `-D` defines, or extra legacy translation units - never source edits.
- Harness flags (verbatim, spec S2 d1/S10): `-O2 -std=gnu++0x -fpermissive -fno-fast-math -ffp-contract=off -DEIGEN_DONT_VECTORIZE`, NO `-fopenmp`. Use Homebrew g++ (clang has no `-fpermissive`). NOTE: gcc defaults to `-ffp-contract=fast` even without fast-math - the `-ffp-contract=off` is mandatory.
- One-time local setup: `brew install gcc eigen libsndfile libmatio boost` (harness never builds in CI; CI consumes committed fixtures only).
- Rust matrices: `ndarray::Array2<f64>`, rows = time frames, cols = bins/coeffs. All reductions are sequential left-to-right hand loops - no `.sum()` on ndarray views where iteration order is unspecified; write explicit index loops for anything the oracle computes sequentially.
- Bit-exact assertions compare `f64::to_bits()` (helper `assert_bits_eq` from Task 2). No `approx` in golden tests. Property/sanity tests may use tolerance and must say so.
- All sec->sample conversions are round-half-away-from-zero: use `f64::round()` (which matches `boost::math::round`).
- Fixtures: `tests/reference_data/phase1/` (committed). Rust tests reach them via `PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/reference_data/phase1/...")`.
- Lint gates: `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`, `./lint_code.sh` (ruff line-length 160 target py314, mypy strict). ASCII-only in all committed files: `LC_ALL=C grep -n '[^ -~]' <files>` must be empty.
- Run Rust commands from `src/rust/`; Python/uv commands from the repo root.

---

### Task 1: Oracle harness foundation + audio-stage goldens

**Files:**
- Create: `tools/oracle_harness/main.cpp`
- Create: `tools/oracle_harness/build.sh`
- Create: `tools/oracle_harness/shims/iof/io.hpp`
- Create: `tools/oracle_harness/shims/omp.h`
- Create: `scripts/extract_phase1_fixtures.py`
- Create: `tests/reference_data/phase1/excerpt_2ch_8k.wav`
- Create: `tests/reference_data/phase1/manifest.json` (generated)
- Create: `tests/test_phase1_fixtures.py`

**Interfaces:**
- Produces: committed dumps `sig_norm.bin` (2 x 16001), `sig_preemph.bin`, `sig_noise.bin` (2 x 16001), `win_hamming_257.bin`, `win_hann_257.bin`, `win_hhcw_257.bin` (1 x 257), `win_conv_norm_7.bin` (1 x 7), `synth_20x50.bin` (the closed-form synthetic matrix). All little-endian `.bin` in the phase-0a codec (i64 rows, i64 cols, f64 column-major) because the harness writes them with the legacy's own `Matrix2BinaryFile` (`Helpers.hpp`).
- Produces: `excerpt_2ch_8k.wav` = frames [240000, 264000) of `/Users/govit/Git/Govit/FastSpeechProcessing-legacy/PRCTS_RUS_RU_0000263489_01.wav` (30.0s..33.0s, 2 ch, 8 kHz, PCM16).
- Produces: harness constants ALL later tasks reuse: `OFFSET_SEC = 0.35`, `MAX_DUR_SEC = 2.0` (so the in-harness `AudioStruct` exercises offset skip and `round(rate*dur+1)` truncation: 16001 samples), `PREEMPH = 0.97`, `NOISE_RATIO = 0.001`.

- [ ] **Step 1: Cut and commit the wav excerpt**

```bash
uv run python - <<'EOF'
import soundfile as sf
data, rate = sf.read('/Users/govit/Git/Govit/FastSpeechProcessing-legacy/PRCTS_RUS_RU_0000263489_01.wav',
                     start=240000, stop=264000, dtype='int16')
assert rate == 8000 and data.shape == (24000, 2)
sf.write('tests/reference_data/phase1/excerpt_2ch_8k.wav', data, rate, subtype='PCM_16')
print('OK: excerpt', data.shape)
EOF
```
Expected: `OK: excerpt (24000, 2)`.

- [ ] **Step 2: Write the shims**

`tools/oracle_harness/shims/omp.h` (MelFilterBank.h does `#include "omp.h"`; without `-fopenmp` the header does not exist):
```c
#pragma once
static inline int omp_get_thread_num(void) { return 0; }
static inline int omp_get_max_threads(void) { return 1; }
static inline int omp_get_num_threads(void) { return 1; }
static inline void omp_set_num_threads(int n) { (void)n; }
static inline double omp_get_wtime(void) { return 0.0; }
```

`tools/oracle_harness/shims/iof/io.hpp` (compile-through only; the iof-using functions are logging paths the harness never calls):
```cpp
#pragma once
#include <ostream>
#include <string>
namespace iof {
struct fmtr {
    std::string f;
    explicit fmtr(const char* s) : f(s) {}
};
inline std::ostream& operator<<(std::ostream& os, const fmtr&) { return os; }
}
```
If the legacy usage pattern `os << iof::fmtr("...") << value` needs the fmtr to swallow subsequent `<<` values to compile, extend the shim minimally (e.g. return a proxy with a templated `operator<<`) - compile errors dictate the exact shape; keep it inert.

- [ ] **Step 3: Write build.sh**

```bash
#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
BREW="$(brew --prefix)"
GXX="$(ls "$BREW"/bin/g++-* | sort -V | tail -1)"
SRC=../../legacy/src
"$GXX" -O2 -std=gnu++0x -fpermissive -fno-fast-math -ffp-contract=off -DEIGEN_DONT_VECTORIZE \
  -I shims -I "$SRC" -I "$BREW/include" -I "$BREW/include/eigen3" \
  main.cpp "$SRC/AudioStruct.cpp" "$SRC/MelFilterBank.cpp" "$SRC/InputStatistics.cpp" "$SRC/ConfigFile.cpp" \
  -L "$BREW/lib" -lsndfile -lmatio -o oracle_harness
echo "OK: built oracle_harness with $GXX"
```
If the linker reports undefined legacy symbols (e.g. `Timer`, `tinythread`), append the corresponding `$SRC/*.cpp` translation units to the compile line - never stub legacy symbols that carry numeric behavior. Record final unit list in the manifest.

- [ ] **Step 4: Write main.cpp v1**

Responsibilities (each dump uses `Matrix2BinaryFile(path, matrix)` from `Helpers.hpp`; output dir = argv[1]):
1. `CorpusItem` + `AudioStruct audio(0.35, 2.0, 0, item)` on `excerpt_2ch_8k.wav` -> dump `audio._DataRaw` as `sig_norm.bin`. (This runs the real legacy decode+offset+truncation+normalization: spec S3.1.)
2. `audio.applyPreemph(0.97)` -> dump `audio._Data` as `sig_preemph.bin`; then `audio.applyNoise(0.001)` -> `sig_noise.bin`. (Order preemph-then-noise, spec S3.2/S3.3.)
3. `getWindowingCoefficients("hamming", false, 257, 0.83333)` -> `win_hamming_257.bin`; same for `"hann"`; `"hHCw"` with param 0.83333 -> `win_hhcw_257.bin`; `getWindowingCoefficients("hann", true, 7)` -> `win_conv_norm_7.bin`. Also assert in-harness that `"uniform"` with normalized=false returns an EMPTY matrix (abort if not - pins the quirk at the source).
4. Synthetic matrix `S(i,j) = ((i*7 + j*13) % 100)/100.0`, 20 x 50 -> `synth_20x50.bin` (later tasks reuse this exact matrix; dumping it pins codec agreement).
Each block gets a `// legacy: <file>:<lines>` provenance comment. Print `OK: <n> dumps`.

- [ ] **Step 5: Write scripts/extract_phase1_fixtures.py**

Module docstring ends with `Usage: uv run python scripts/extract_phase1_fixtures.py`. It: runs `tools/oracle_harness/build.sh`, runs `./oracle_harness tests/reference_data/phase1/`, then writes `manifest.json` with: g++ version string, brew eigen/libsndfile/libmatio versions, the exact flag string, the harness constants (offset 0.35, dur 2.0, preemph 0.97, noise 0.001), the dump inventory with shapes, and an anchor value (`sig_norm[0][0]` printed as hex via `float.hex()`). Prints `OK: ...` summary.

- [ ] **Step 6: Build, run, verify**

Run: `brew install gcc eigen libsndfile libmatio boost && uv run python scripts/extract_phase1_fixtures.py`
Expected: `OK` with 8 dumps; `sig_norm.bin` shape (2, 16001). Debug compile errors via shims/units per Steps 2-3; document any extra units in the manifest.

- [ ] **Step 7: Write the fixture-presence pytest**

`tests/test_phase1_fixtures.py`: asserts the wav has 24000 frames/2ch/8kHz (soundfile), every manifest-listed dump exists, `sig_norm.bin` parses to (2,16001) via `speech.weight_bridge.read_bin`, and the manifest flag string contains `-fno-fast-math -ffp-contract=off -DEIGEN_DONT_VECTORIZE`.

- [ ] **Step 8: Run it**

Run: `uv run pytest tests/test_phase1_fixtures.py -v`
Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add tools/oracle_harness scripts/extract_phase1_fixtures.py tests/reference_data/phase1 tests/test_phase1_fixtures.py
git commit -m "feat(phase1): oracle harness foundation + audio-stage goldens

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: audio.rs core - Audio, read_audio, normalization, preemph, noise

**Files:**
- Modify: `src/rust/src/audio.rs` (replace the stub `Audio`)
- Create: `src/rust/tests/common/mod.rs`
- Create: `src/rust/tests/phase1_audio_golden.rs`
- Modify: `src/rust/Cargo.toml` (ndarray already present; no additions expected)

**Interfaces:**
- Consumes: `speech::io::binary::read_matrix` (phase 0a), `speech::constants::random_uniform`.
- Produces:
  - `pub struct Audio { pub sample_rate: u32, pub data: Array2<f64>, pub data_raw: Array2<f64> }` (channels x samples)
  - `pub fn read_audio(path: &Path, offset_sec: f64, max_duration_sec: f64) -> anyhow::Result<Audio>`
  - `pub fn normalize_channels(data: &mut Array2<f64>)`
  - `impl Audio { pub fn apply_preemph(&mut self, ratio: f64); pub fn apply_noise(&mut self, ratio: f64); pub fn reset(&mut self); }`
  - Test helpers (`tests/common/mod.rs`): `pub fn fixture(name: &str) -> PathBuf`, `pub fn load_bin(name: &str) -> Array2<f64>` (wraps `io::binary::read_matrix`), `pub fn assert_bits_eq(a: &Array2<f64>, b: &Array2<f64>, label: &str)` (elementwise `to_bits` compare, reporting first mismatch index + hex).

- [ ] **Step 1: Write the failing golden + unit tests**

`src/rust/tests/phase1_audio_golden.rs`:
```rust
mod common;
use common::{assert_bits_eq, fixture, load_bin};
use speech::audio::read_audio;

#[test]
fn decode_normalize_matches_oracle() {
    let audio = read_audio(&fixture("excerpt_2ch_8k.wav"), 0.35, 2.0).unwrap();
    assert_eq!(audio.sample_rate, 8000);
    assert_bits_eq(&audio.data_raw, &load_bin("sig_norm.bin"), "sig_norm");
}

#[test]
fn preemph_noise_match_oracle() {
    let mut audio = read_audio(&fixture("excerpt_2ch_8k.wav"), 0.35, 2.0).unwrap();
    audio.apply_preemph(0.97);
    assert_bits_eq(&audio.data, &load_bin("sig_preemph.bin"), "sig_preemph");
    audio.apply_noise(0.001);
    assert_bits_eq(&audio.data, &load_bin("sig_noise.bin"), "sig_noise");
    audio.reset();
    assert_bits_eq(&audio.data, &load_bin("sig_norm.bin"), "reset");
}
```
Inline unit tests in `audio.rs` (hand arithmetic in comments; expression-form expectations for bit-exactness):
```rust
#[test]
fn preemph_backward_sample0_untouched() {
    // x=[1,2,3,4], r=0.5 -> backward: [1, 2-0.5*1, 3-0.5*2, 4-0.5*3] = [1, 1.5, 2, 2.5]
    let mut a = test_audio(vec![1.0, 2.0, 3.0, 4.0]);
    a.apply_preemph(0.5);
    assert_eq!(a.data.row(0).to_vec(), vec![1.0, 1.5, 2.0, 2.5]);
}
#[test]
fn noise_uses_uniform_table_same_across_channels() {
    let mut a = test_audio_2ch(vec![0.0; 3]);
    a.apply_noise(0.25);
    let expect = |k: usize| 2.0 * 0.25 * speech::constants::random_uniform(k) - 0.25;
    for k in 0..3 { assert_eq!(a.data[[0, k]], expect(k)); assert_eq!(a.data[[1, k]], expect(k)); }
}
#[test]
fn normalize_is_2rms_plus_peak_over_2_with_floor() {
    // x=[3,4]: adim=(2*sqrt(25/2/... expression below))/...
    let mut d = ndarray::arr2(&[[3.0, 4.0]]);
    normalize_channels(&mut d);
    let adim = (2.0 * ((3.0f64*3.0 + 4.0*4.0) / 2.0).sqrt() + 4.0) / 2.0;
    assert_eq!(d[[0, 0]], 3.0 / adim);
    let mut z = ndarray::arr2(&[[0.0, 0.0]]);
    normalize_channels(&mut z); // adim floors at 1e-3
    assert_eq!(z[[0, 0]], 0.0);
}
```
(`test_audio`/`test_audio_2ch` are tiny `#[cfg(test)]` constructors.)

- [ ] **Step 2: Run to verify failure**

Run: `cd src/rust && cargo test phase1_audio` - Expected: FAIL (unresolved `read_audio` etc.).

- [ ] **Step 3: Implement**

Spec S3.1-S3.3; legacy `AudioStruct.cpp:36-137,437-454`. Key requirements:
- symphonia wav decode to i16, convert `x as f64 / 32768.0`, de-interleave to channels x samples.
- `frame_offset_begin = (rate * offset_sec).round() as i64`; `frame_count_max = (rate * max_duration_sec + 1.0).round() as i64` (+1 INSIDE the round); `frames = (total - offset).clamp(0, frame_count_max)`.
- `normalize_channels`: per channel, sequential left-to-right `sum(x*x)`, `adim = (2.0*(sum/n).sqrt() + max_abs)/2.0`, floor `1e-3`, divide. `max_abs` via sequential scan (`f64::max` order-insensitive here, but keep a plain loop).
- `apply_preemph`: backward `for kk in (1..s).rev() { d[kk] -= ratio*d[kk-1] }` per channel, gate `ratio > 0` at call sites (function itself unconditional like the legacy body, which is wrapped in `if (preemphRatio > 0)` - reproduce the internal gate exactly as `AudioStruct.cpp:447`).
- `apply_noise`: `d[[c,k]] += 2.0*ratio*random_uniform(k) - ratio` - same table value for every channel at k.
- `reset`: `data = data_raw.clone()`.

- [ ] **Step 4: Run to verify pass**

Run: `cd src/rust && cargo test phase1_audio && cargo test --lib audio` - Expected: PASS. If `decode_normalize_matches_oracle` mismatches at [0][0], symphonia's PCM16 conversion differs - adapt the conversion (spec S12) and note it in the module doc.

- [ ] **Step 5: Lint + commit**

```bash
cd src/rust && cargo fmt && cargo clippy --all-targets -- -D warnings && cd ../..
git add src/rust
git commit -m "feat(phase1): audio decode, normalization, preemph, noise (bit-exact vs oracle)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Windowing coefficients + convolutions

**Files:**
- Modify: `src/rust/src/audio.rs`
- Modify: `src/rust/tests/phase1_audio_golden.rs`

**Interfaces:**
- Produces:
  - `pub fn windowing_coefficients(kind: &str, normalized: bool, size: usize, extra_param: f64) -> Option<Vec<f64>>` (None = rectangular/empty)
  - `pub fn convolution_horiz(row: &mut Array2<f64>, coeffs: &[f64])` (1 x N in place)
  - `pub fn convolution_vert(mat: &mut Array2<f64>, coeffs: &[f64])` (T x B in place, down time)

- [ ] **Step 1: Failing tests**

Golden: `windowing_coefficients("hamming", false, 257, 0.83333)` bits == `win_hamming_257.bin` row; same hann/hHCw; `("hann", true, 7)` == `win_conv_norm_7.bin`. Unit tests:
```rust
#[test]
fn uniform_unnormalized_is_rectangular() {
    assert!(windowing_coefficients("uniform", false, 7, 0.83333).is_none()); // Helpers.hpp:260 quirk
    let u = windowing_coefficients("uniform", true, 4, 0.83333).unwrap();
    assert_eq!(u, vec![0.25; 4]);
}
#[test]
fn none_unknown_or_size1_empty() {
    assert!(windowing_coefficients("none", false, 257, 0.8).is_none());
    assert!(windowing_coefficients("blackman", false, 257, 0.8).is_none());
    assert!(windowing_coefficients("hamming", false, 1, 0.8).is_none());
}
#[test]
fn hamming_formula() {
    let w = windowing_coefficients("hamming", false, 5, 0.83333).unwrap();
    for (ii, v) in w.iter().enumerate() {
        assert_eq!(*v, 0.54 - 0.46 * (2.0 * std::f64::consts::PI / 4.0 * ii as f64).cos());
    }
}
#[test]
fn convolution_edges_truncate_without_renormalization() {
    // kernel [0.25,0.5,0.25]; row [1,1,1,1]: interior=1.0, edges=0.75 (missing coeff NOT redistributed)
    let mut r = ndarray::arr2(&[[1.0, 1.0, 1.0, 1.0]]);
    convolution_horiz(&mut r, &[0.25, 0.5, 0.25]);
    assert_eq!(r.row(0).to_vec(), vec![0.75, 1.0, 1.0, 0.75]);
}
```
Plus a `convolution_vert` case on a 4x2 matrix with the same kernel (per-column expected values written out).

- [ ] **Step 2: Run, expect FAIL** - `cargo test phase1_audio windowing`

- [ ] **Step 3: Implement**

Spec S3.4/S3.6; legacy `Helpers.hpp:222-272` (windows: hamming/hann/hHCw formulas, hHCw `size1 = (extra.clamp(0,1)*size).round()`, normalize = divide by sum) and `Helpers.hpp:164-220` (convolutions: kernel `2K+1`, edge truncation without renormalization; replace the unsigned-wrap index trick with signed arithmetic yielding identical indices - the accumulation order over `jj` ascending is the bit-exact contract; `convolution_vert` copies, zeroes target, accumulates sequentially).

- [ ] **Step 4: Run, expect PASS** - `cargo test phase1_audio`

- [ ] **Step 5: Lint + commit** (message: `feat(phase1): windowing coefficients + edge-truncating convolutions`)

---

### Task 4: features/fft.rs - exact GFFT port

**Files:**
- Modify: `src/rust/src/features/fft.rs`
- Modify: `tools/oracle_harness/main.cpp` (add FFT dumps)
- Create: `src/rust/tests/phase1_fft_golden.rs`
- Modify (generated): `tests/reference_data/phase1/` (`fft_in_4.bin`/`fft_out_4.bin`, `..._8`, `..._256` - interleaved re/im as 1 x 2N rows)

**Interfaces:**
- Produces: `pub struct Gfft; impl Gfft { pub fn new(p: u32) -> Gfft; pub fn fft(&self, data: &mut [f64]); }` - in-place, interleaved `[re0,im0,...]`, len `2*(1<<p)`, unnormalized, negative exponent. Also `pub(crate) fn sin_series(b: u64, a: u64) -> f64` (exposed for the seed test).

- [ ] **Step 1: Extend harness + regenerate fixtures**

In `main.cpp`: instantiate the GFFT factory exactly as `BLSTMSpectralSegmenter.cpp:600-603` (Min=1, Max=20); **first verify against `legacy/src/BLSTMSpectralSegmenter.cpp:199-203` that the spectrum-order clamp is Max-1=19 and record the verified value in the manifest (spec S12 discrepancy)**. For N in {4, 8, 256}: build input `data[2i] = S(0,i), data[2i+1] = S(1,i)` from the synthetic matrix rows (i mod 50 indexing for N=256), dump input as `fft_in_N.bin` (1 x 2N), run `gfft->fft(data)`, dump as `fft_out_N.bin`. Run `uv run python scripts/extract_phase1_fixtures.py`.

- [ ] **Step 2: Failing tests**

```rust
#[test]
fn gfft_matches_oracle_bitexact() {
    for p in [2u32, 3, 8] {
        let n = 1usize << p;
        let mut data = common::load_bin(&format!("fft_in_{n}.bin")).row(0).to_vec();
        Gfft::new(p).fft(&mut data);
        let expect = common::load_bin(&format!("fft_out_{n}.bin"));
        for (i, v) in data.iter().enumerate() {
            assert_eq!(v.to_bits(), expect[[0, i]].to_bits(), "N={n} idx={i}");
        }
    }
}
```
Plus sanity (tolerance 1e-9, labeled as sanity-only): impulse -> flat spectrum; Parseval. Plus the seed test: `sin_series(4,1)` equals the truncated-Taylor evaluation written out inline (the S(2,34) recursion for x = PI/4, compared against a directly-coded 17-factor product loop - two independent codings of the same series must agree exactly).

- [ ] **Step 3: Run, expect FAIL** - `cargo test phase1_fft`

- [ ] **Step 4: Implement**

Spec S4.1; legacy `fft.hpp` entire file. Non-negotiables: NR bit-reversal scramble (`fft.hpp:176-190`, port the 1-based int loop faithfully); recursive DL with halves then butterflies; running twiddle recurrence `wtemp=wr; wr += wr*wpr - wi*wpi; wi += wi*wpr + wtemp*wpi;` seeded `wpr = -2*sin(pi/N)^2` and `wpi = -sin(2pi/N)` where sin comes from `sin_series` (S(M,N): `1 - ((x*x)/M)/(M+1)*next`, x = `(A as f64 * PI)/B as f64`, `A*PI` first; Sin = x * S(2,34)); DL<4> hard-coded fused `-i` butterfly in the exact statement order (`fft.hpp:107-136`); DL<2> plain butterfly. Runtime recursion on p (no const generics needed; a simple `fn dl(data: &mut [f64], n: usize, ...)` recursing to n==4/n==2 base cases is fine as long as operation order matches).

- [ ] **Step 5: Run, expect PASS** - `cargo test phase1_fft`

- [ ] **Step 6: Lint + commit** (message: `feat(phase1): exact GFFT port (Taylor twiddle seeds, NR recurrence, bit-exact)`)

---

### Task 5: Two-real periodogram + framing driver + get_sequence

**Files:**
- Modify: `src/rust/src/features/fft.rs` (`compute_two_real_periodogram`)
- Modify: `src/rust/src/audio.rs` (`get_sequence`, `compute_segment_periodogram_estimates`)
- Modify: `tools/oracle_harness/main.cpp`
- Create: `src/rust/tests/phase1_periodogram_golden.rs`

**Interfaces:**
- Consumes: `Gfft`, `windowing_coefficients`, `convolution_vert`, Task 1 audio dumps.
- Produces:
  - `pub fn compute_two_real_periodogram(gfft: &Gfft, n: usize, sig1: &[f64], sig2: &[f64], out: &mut Array2<f64>, row1: usize, row2: usize)` (sigs have n+1 samples; last dropped)
  - `pub fn get_sequence(data: &Array2<f64>, chan: usize, index: usize, half_window: usize, dc_offset: bool, coeffs: Option<&[f64]>, buf: &mut Array2<f64>)` (caller-owned 1 x (2w+1) reused buffer; index >= frames = no-op)
  - `pub fn compute_segment_periodogram_estimates(audio: &Audio, p: u32, shift: usize, chan: usize, dc_offset: bool, coeffs: Option<&[f64]>, conv: Option<&[f64]>, begin: usize, end: usize) -> Array2<f64>`

- [ ] **Step 1: Extend harness + regenerate**

In `main.cpp`, on the preemph+noise audio (order: construct fresh AudioStruct, applyPreemph(0.97), applyNoise(0.001)): call `computeSegmentPeriodogramEstimates` per channel with p=8, shift=80, DC offset TRUE, hamming-257 window, empty mel bank, begin=0, end=frames-1, once with empty temporal convolution -> `perio_p8_s80_chan{1,2}.bin`, once with the normalized hann-7 kernel -> `perio_conv_p8_s80_chan{1,2}.bin`. Also one odd-frame-count variant: end chosen so frameNb is odd (compute in harness; assert oddness) -> `perio_odd_chan1.bin`. Regenerate fixtures.

- [ ] **Step 2: Failing tests**

Golden: reproduce the same pipeline in Rust (read excerpt, preemph 0.97, noise 0.001, windowing hamming 257, p=8, shift=80) and `assert_bits_eq` against all five dumps. Unit tests for `get_sequence` (data row = 0..10, w=3, buffer pre-seeded with 7.0 sentinels):
- left edge index=1: `buf[0..2]` STILL 7.0 (no zeroing - then windowed if coeffs given: seed AFTER choosing coeffs=None here), `buf[2..7] == [0,1,2,3,4]`.
- right edge index=9 (frames=10): buffer zeroed first; `buf[0..4] == [6,7,8,9]`, rest 0.0.
- index=10: buffer untouched (all 7.0).
- DC offset: mean over the FULL 2w+1 buffer including pad; pad becomes `-mean`.
And a two-real unpack hand case: n=4, sig1=[1,0,0,0,x], sig2=[0,1,0,0,x] (x=9.0 proves the drop): expected P1 = [1/4 repeated? -- write out the DFT by hand in comments: X(k) of the packed sequence, then the exact `(a+b)^2+(c-d)^2)/16` expressions coded inline].

- [ ] **Step 3: Run, expect FAIL** - `cargo test phase1_periodogram`

- [ ] **Step 4: Implement**

Spec S4.2/S4.3/S3.5; legacy `AudioStruct.cpp:456-584`. Non-negotiables: pack only first n of n+1 samples; DC bins `data[0]^2/n`, `data[1]^2/n`; loop `ii = 2,4,..,n` with the exact four expressions over `size_1=2n`, `size_2=2n+1`, `coeff_norm=4n`; frame count = inclusive-count ceil-divide; two frames per iteration `jj += 2*shift`, rows `(jj-begin)/shift` and `+1`; odd tail packs sig1 twice writing the same row twice (second write = P2 formula); ONE pair of reused buffers zero-initialized once before the loop (stale semantics); `convolution_vert` applied iff kernel len > 1.

- [ ] **Step 5: Run, expect PASS**; **Step 6: Lint + commit** (message: `feat(phase1): two-real periodogram + framing driver (bit-exact incl. stale-buffer quirks)`)

---

### Task 6: features/mel.rs - filterbank construction + applyFilterBank + regression deltas

**Files:**
- Modify: `src/rust/src/features/mel.rs`
- Modify: `tools/oracle_harness/main.cpp`
- Create: `src/rust/tests/phase1_mel_golden.rs`
- Create: `src/python/speech/features_oracle.py`
- Create: `scripts/extract_phase1_oracle_cases.py`
- Create: `tests/test_features_oracle.py`
- Create (generated): `tests/reference_data/phase1/deltas_cases.json`

**Interfaces:**
- Consumes: `synth_20x50.bin`, periodogram dumps.
- Produces:
  - `pub fn hz_to_mel(f: f64) -> f64; pub fn mel_to_hz(m: f64) -> f64;`
  - `pub struct MelFilterBank; impl MelFilterBank { pub fn new(min_mel: f64, max_mel: f64, nb_bins: i32, min_freq: f64, max_freq: f64, rate: f64, spectrum_size: usize, is_log: bool, nb_dct: i32, ignore_first: bool, deltas_nb: i32, delta_deltas_nb: i32) -> MelFilterBank; pub fn apply_filter_bank(&self, p: &Array2<f64>) -> Array2<f64>; pub fn nb_filters(&self) -> usize; pub fn is_mel(&self) -> bool; pub fn is_dct_activated(&self) -> bool; }`
  - `pub(crate) fn regression_deltas(base: &Array2<f64>, n: i32) -> Array2<f64>` (the shared kernel; Task 7 reuses)
  - Python: `speech.features_oracle.regression_deltas_oracle(base: np.ndarray, n: int) -> np.ndarray`
  - JSON: `deltas_cases.json` rows `{base: [[..]], n: int, expected: [[..]]}`

- [ ] **Step 1: Extend harness + regenerate**

In `main.cpp`: `MelFilterBank mel(64.0, 3800.0, 26, 64.0, 3800.0, 8000.0, 128, /*isLog*/true, /*nbDCT*/0, false, /*deltas*/0, 0)` applied to `perio_p8_s80_chan1` -> `logmel_26_chan1.bin`; a deltas-no-DCT variant `(..., true, 0, false, 3, 3)` -> `logmel_deltas_chan1.bin` (pins the `[delta|delta|dd]` overwrite quirk from the REAL legacy code); a non-log variant `(..., false, 0, false, 0, 0)` -> `mel_26_chan1.bin`. Dump the synthetic-matrix filterbank too: apply the 26-filter bank to `synth_20x50.bin` reinterpreted as a 20x50 periodogram with rate 8000, spectrum_size 49 -> `logmel_synth.bin`. Regenerate.

- [ ] **Step 2: Failing tests (Rust)**

Golden tests vs the four dumps (same constructor args). Hand tests:
```rust
#[test]
fn hz_mel_roundtrip_formulas() {
    assert_eq!(hz_to_mel(700.0), 1125.0 * 2.0f64.ln());
    assert_eq!(mel_to_hz(hz_to_mel(300.0)), 700.0 * ((1125.0 * (1.0f64 + 300.0/700.0).ln())/1125.0).exp() - 700.0 + 0.0); // expression form
}
#[test]
fn grid_accumulates_sequentially() {
    // rate 8000, spectrum_size 49 -> freqStep = 81.632653...; freqs built by += must match a fold, not i*step
    let fb = MelFilterBank::new(64.0, 3800.0, 26, 64.0, 3800.0, 8000.0, 49, true, 0, false, 0, 0);
    assert!(fb.is_mel());
}
#[test]
fn empty_filter_collapses_whole_bank_to_passthrough() {
    // nb_bins so large that some filter catches no bin: spectrum_size 8 (9 bins), 26 filters
    let fb = MelFilterBank::new(64.0, 3800.0, 26, 64.0, 3800.0, 8000.0, 8, true, 0, false, 0, 0);
    assert!(!fb.is_mel()); // fallback: nb_filters = end-beg+1
}
```
Plus a triangle-coefficient case: for the 26-filter/spectrum_size-128 bank assert filter 0's first coefficient equals the rising-edge expression `(freqs[j]-mels[0])/(mels[1]-mels[0])` computed inline from `hz_to_mel`-anchored walk (the test re-derives the first two mel edges with the same sequential walk and compares).

- [ ] **Step 3: Failing tests (Python oracle)**

`tests/test_features_oracle.py::test_regression_deltas_anchor`: for base = [[0],[1],[4]] (T=3, W=1), n=2: hand arithmetic in a comment - `D[0] = (1*(x1-x0) + 2*(x2-x0)) / (2*(1+4)) = (1 + 8)/10 = 0.9`; `D[1] = (1*(x2-x0) + 2*(x2-x0))/10 = (4+8)/10 = 1.2`; `D[2] = (1*(x2-x1) + 2*(x2-x0))/10 = (3+8)/10 = 1.1`. Assert oracle returns [[0.9],[1.2],[1.1]] exactly.

- [ ] **Step 4: Run both, expect FAIL** - `cargo test phase1_mel`; `uv run pytest tests/test_features_oracle.py -v`

- [ ] **Step 5: Implement Rust + oracle + cases**

Rust: spec S5.1/S5.2; legacy `MelFilterBank.cpp:15-216,362-368`. Non-negotiables: sequential `+=` grid walks (freqs up; mels down-then-up with the exact loop conditions); melStep floor `hz_to_mel(1.0)`; `_BegFreq/_EndFreq` floor/ceil round-trip; inclusive edge membership; center bin -> falling branch (coeff exactly 1.0); whole-bank fallback on any empty filter; log = `ln(dot + 1e-24)`; the deltas-no-DCT output `[delta|delta|dd]` overwriting statics (`cpp:192-193`); `regression_deltas` implemented via the legacy block-op sequence (shifted copies + edge corrections + `D += j*deltas` ascending + final divide), T<=j branch skipping shifted copies.
Python oracle: same kernel in numpy (vectorized is fine - it must match the clamped-formula semantics; anchor test is the arbiter). Extend `scripts/extract_phase1_oracle_cases.py`: deterministic bases from `((i*7+j*13)%100)/100` at shapes (1x2, 2x3, 3x1, 10x4, 25x3) x n in {1,2,3,5}, plus the T<=n edge shapes (2x2 with n=3); write `deltas_cases.json`. Rust golden loop reads the JSON (serde_json) and `assert_eq!` on bits.

- [ ] **Step 6: Run all, expect PASS** - `cargo test phase1_mel && uv run pytest tests/test_features_oracle.py tests/test_phase1_fixtures.py -v`

- [ ] **Step 7: Lint + commit** (message: `feat(phase1): mel filterbank + log-mel + regression deltas (goldens + numpy oracle)`)

---

### Task 7: applyDCT - MFCC, deltas, SDC

**Files:**
- Modify: `src/rust/src/features/mel.rs`
- Modify: `tools/oracle_harness/main.cpp`
- Modify: `src/rust/tests/phase1_mel_golden.rs`
- Modify: `src/python/speech/features_oracle.py`, `scripts/extract_phase1_oracle_cases.py`, `tests/test_features_oracle.py`
- Create (generated): `tests/reference_data/phase1/sdc_cases.json`

**Interfaces:**
- Consumes: `regression_deltas`, Task 6 bank.
- Produces: `impl MelFilterBank { pub fn apply_dct(&self, mel: &Array2<f64>) -> Array2<f64>; pub fn nb_dct(&self) -> usize; }`; Python `sdc_oracle(mfcc: np.ndarray, nb_dct: int) -> np.ndarray`.

- [ ] **Step 1: GEMM-order verification, THEN harness extension**

The DCT apply is the feature path's one real Eigen matrix product (`melPeriodogram*_CoeffsDCT`, spec S12). In `main.cpp`, before trusting DCT dumps: compute the product both via Eigen and via an explicit triple loop (`for t / for k / accumulate over n ascending`), compare bit-for-bit in-harness, and `abort()` on mismatch, printing the first differing index. If it aborts: replace the Eigen product with the explicit loop (provenance comment `// legacy semantics: MelFilterBank.cpp:223 under EIGEN_DONT_VECTORIZE, order verified`) and record the substitution in the manifest.
Then add dumps: banks with nbDCT=13 - (a) plain `(deltas 0, dd 0, ignoreFirst false)` -> `mfcc_chan1.bin`; (b) deltas `(3, 3, false)` -> `mfcc_deltas_chan1.bin`; (c) ignoreFirst `(3, 3, true)` -> `mfcc_deltas_if_chan1.bin`; (d) SDC `(deltas -1, dd 0, ignoreFirst false)` -> `mfcc_sdc_chan1.bin`; (e) SDC+ignoreFirst -> `mfcc_sdc_if_chan1.bin` (pins the c_{N-1} clobber). All applied to the Task 6 log-mel of chan 1. Regenerate.

- [ ] **Step 2: Failing tests**

Rust goldens vs the five dumps. Unit tests: DCT matrix entry `(n,k) == (PI/26.0*(n as f64+0.5)*k as f64).cos()` for a sample of (n,k); width accessors per spec S5.2 (`nb_dct()` for deltas/SDC/ignoreFirst combinations - table-driven asserts); SDC time-offset structure on a synthetic ramp: with delta block D, assert `out[t][block kk] == D[t+10-3*kk]` or 0.0 out of range (kk in 0..7). Python anchor: `sdc_oracle` on a 4x2 MFCC with hand-verified block placement (zeros outside, offsets +10..-8 -> for T=4 only blocks with t+10-3kk in [0,4) are nonzero).

- [ ] **Step 3: Run, expect FAIL**; **Step 4: Implement**

Spec S5.2 branches A/B/C exactly; legacy `MelFilterBank.cpp:108-127,218-360`. Non-negotiables: unnormalized DCT-II; SDC hardcoded d=3/P=3/k=7, delta = regression kernel n=3 (adim 28), stack at vertical offsets 3*kk, extract at row 10, ZERO padding; ignoreFirst in SDC clobbers the LAST static column (write order: statics then `rightCols(7*NbDCT)` starting at col NbDCT-1); branch B computes into a T x (cols+1) temp then drops col 0; branch C keeps statics. `sdc_oracle` + `sdc_cases.json` (deterministic MFCC inputs, shapes covering T<21 and T>21), Rust JSON cross-check bit-exact.

- [ ] **Step 5: Run, expect PASS** - `cargo test phase1_mel && uv run pytest tests/test_features_oracle.py -v`; **Step 6: Lint + commit** (message: `feat(phase1): DCT-II MFCC, delta/delta-delta, SDC (incl. ignoreFirst clobber quirk)`)

---

### Task 8: LTSV score + column

**Files:**
- Modify: `src/rust/src/features/ltsv_tdc.rs`
- Modify: `tools/oracle_harness/main.cpp`
- Create: `src/rust/tests/phase1_ltsv_tdc_golden.rs`
- Modify: `src/python/speech/features_oracle.py`, `scripts/extract_phase1_oracle_cases.py`, `tests/test_features_oracle.py`
- Create (generated): `tests/reference_data/phase1/ltsv_cases.json`

**Interfaces:**
- Produces:
  - `pub fn ltsv_classify_sequence(p: &Array2<f64>, col: usize, freq_beg: usize, freq_end: usize, half_window: usize) -> f64`
  - `pub fn get_ltsv(p: &Array2<f64>, freq_beg: usize, freq_end: usize, half_window: usize, shift: usize) -> Vec<f64>` (len = p.nrows(); interpolation backfill; trailing zeros)
  - Python `ltsv_oracle(p, col, freq_beg, freq_end, half_window) -> float`, JSON `ltsv_cases.json`.

- [ ] **Step 1: Extend harness + regenerate**

Transcribe VERBATIM into `main.cpp` (provenance comments): `LongTermSpectralVariation::classifySequence` body (`LongTermSpectralVariation.cpp:82-128`) as a free function, and the `getLTSV` loop (`BLSTMSpectralSegmenter.cpp:316-339`). Diff-review both against the legacy source (read the legacy lines side by side; they must match modulo member->parameter renames). Dump: LTSV column over `perio_p8_s80_chan1` with `freq_beg=0, freq_end=128, R=15, shift=4` -> `ltsv_chan1.bin` (T x 1); per-frame scores over `synth_20x50.bin` (R=3, band 0..49, shift=1) -> `ltsv_synth.bin`. Regenerate.

- [ ] **Step 2: Failing tests**

Rust goldens vs both dumps. Hand test (hand arithmetic in comments, expression-coded expected value):
```rust
#[test]
fn ltsv_is_biased_variance_no_sqrt() {
    // P = [[1,2],[3,4],[5,6]] (T=3, bins=2), col=1, R=1 -> window rows 0..=2, L=3
    // bin0: mean=3; r={1/3,1,5/3}; sum r(r-1) = -2/9 + 0 + 10/9 = 8/9; dzeta0 = -(8/9)/3
    // bin1: mean=4; r={1/2,1,3/2}; sum = -1/4 + 0 + 3/4 = 1/2;      dzeta1 = -(1/2)/3
    // return ((d0-m)^2 + (d1-m)^2)/2, m = (d0+d1)/2  -- VARIANCE, no sqrt
    let p = ndarray::arr2(&[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);
    let got = ltsv_classify_sequence(&p, 1, 0, 1, 1);
    let dz = |vals: [f64; 3]| {
        let mean = ((vals[0] + vals[1] + vals[2]) / 3.0).max(1e-12);
        let mut s = 0.0;
        for v in vals { let r = v / mean; s -= r * (r - 1.0); }
        s / 3.0
    };
    let (d0, d1) = (dz([1.0, 3.0, 5.0]), dz([2.0, 4.0, 6.0]));
    let m = (d0 + d1) / 2.0;
    assert_eq!(got, ((d0 - m).powi(2) + (d1 - m).powi(2)) / 2.0);
}
```
(NOTE: the closure mirrors the legacy accumulation order - sum bins sequentially, divide, subtract - the golden dumps are the true arbiter; this pins structure: variance not std, mean-floor placement, inclusive window.) Plus: edge shrink at col=0 (window length 2), mean-floor trigger (all-zero bin -> mean floors to 1e-12, r=0, dzeta=0), `get_ltsv` interpolation (shift=2 over 5 rows: row1 = (row0+row2)/2 linear blend; trailing row 0.0), matching Python anchor test.

- [ ] **Step 3: Run, expect FAIL**; **Step 4: Implement + oracle + cases**

Spec S6.1/S6.2. `ltsv_oracle` in numpy; `ltsv_cases.json` from the closed-form matrix at (20x50) with parameter grid {R in 1,3,15} x {band (0,49),(5,20)} x cols {0, 3, 19}; Rust JSON cross-check bit-exact.

- [ ] **Step 5: Run, expect PASS** - `cargo test phase1_ltsv && uv run pytest tests/test_features_oracle.py -v`; **Step 6: Lint + commit** (message: `feat(phase1): LTSV score + interpolated column (variance-not-std, mean-only floor)`)

---

### Task 9: fmath_log + TDC + pitch + homothety

**Files:**
- Modify: `src/rust/src/features/ltsv_tdc.rs`
- Modify: `tools/oracle_harness/main.cpp`
- Modify: `src/rust/tests/phase1_ltsv_tdc_golden.rs`
- Modify: `src/python/speech/features_oracle.py`, `scripts/extract_phase1_oracle_cases.py`, `tests/test_features_oracle.py`
- Create (generated): `tests/reference_data/phase1/fmath_log_sweep.bin`, `tdc_cases.json`

**Interfaces:**
- Consumes: `get_sequence`, `windowing_coefficients`, `Segmentation` (phase 0b-i `tasks::segmentation`).
- Produces:
  - `pub fn fmath_log(x: f32) -> f32`
  - `pub fn tdc_classify_sequence(window: &[f64], min_lag: usize, max_lag: usize, balance: f64) -> f64`
  - `pub fn compute_pitch(window: &[f64], min_lag: usize, max_lag: usize, rate: f64) -> Option<f64>`
  - `pub fn get_pitch(audio: &Audio, seg: &Segmentation, chan: usize, tdc: &TdcParams, rate: f64) -> f64` (NOTE: takes the Task 11 `TdcParams`; implement here against a local struct definition placed in `features/pipeline.rs` in THIS task as a minimal `pub struct TdcParams { pub half_window: usize, pub full_window: usize, pub shift: usize, pub min_lag: usize, pub max_lag: usize, pub coeffs: Option<Vec<f64>>, pub balance: f64 }` - Task 11 keeps this exact definition)
  - `pub fn apply_homothety(p: &Array2<f64>, coeff: f64) -> Array2<f64>`
  - Python `fmath_log_oracle(x: np.float32) -> np.float32`, `tdc_oracle(window, min_lag, max_lag, balance) -> float`.

- [ ] **Step 1: Extend harness + regenerate**

Transcribe VERBATIM (provenance comments, diff-review): `TimeDomainCorrel::classifySequence` (`TimeDomainCorrel.cpp:36-91`, `_Balance` -> param), `computePitch` (`BLSTMSpectralSegmenter.cpp:172-192`), the homothety warp loop (`:762-775`). Dumps: (a) `fmath_log_sweep.bin` - x values: `{1e-24, 1e-12, 0.5, 1.0-2^-24, 1.0, 1.5, 2.0, 88.0, 1e10, 0.0}` plus `1.0 + i/2048.0` for i in {0,1,1023,2047} and `nextafterf` neighbors of table boundaries; dump as 2 x N (input f32 widened, output f64-widened f32); (b) TDC scores over the excerpt: hamming window per `TdcParams` from `TDC_window=0.032, TDC_shift=0.01, lags=(0.002,0.016), balance=0.7` derivation (transcribed param math from `TimeDomainCorrel.cpp:100-109` and spec S8.2) -> `tdc_chan1.bin` (1 x frames); (c) pitch scalar over a hand-built segmentation (single SPEECH segment covering the middle 1s) -> `pitch_chan1.bin` (1x1); (d) homothety of `perio_p8_s80_chan1` with the resulting `pitch/300` coeff -> `perio_homothety_chan1.bin`. Regenerate.

- [ ] **Step 2: Failing tests**

Rust goldens vs all four dumps. Unit tests:
```rust
#[test]
fn fmath_log_at_zero_is_finite() {
    // fmath.hpp:713-727 - no x<=0 guard; log(0f) = (0 - 127<<23) as f32 * c_log2 + app[0] + 0
    let v = fmath_log(0.0);
    assert!(v.is_finite());
    assert_eq!(v, -88.029694); // pin the exact f32; verify against the sweep dump first, adjust literal to the dump if it differs
}
#[test]
fn tdc_r0_is_one_when_min_lag_zero_and_score_finite() {
    let w: Vec<f64> = (0..9).map(|i| ((i * 7) % 5) as f64 - 2.0).collect();
    let s = tdc_classify_sequence(&w, 0, 4, 0.7);
    assert!(s.is_finite()); // MaxPeak == 1 -> fmath_log(1-1=0) finite
}
#[test]
fn tdc_fewer_than_three_crossings_zero_crosscorr() {
    // monotone R (no zero crossing): score = balance*(-fmath_log(1-MaxPeak)) exactly
    let w = vec![1.0, 1.0, 1.0, 1.0, 1.0];
    let s = tdc_classify_sequence(&w, 1, 3, 0.5);
    let r_max = (3.0f64 / 5.0).max(2.0 / 5.0); // R[1]=4/5? hand-compute in comment; assert via expressions
    assert_eq!(s, 0.5 * (-(fmath_log((1.0 - r_max) as f32) as f64)));
}
#[test]
fn homothety_interpolates_bins() {
    // coeff=0.5: out[t][2] = (1-0)*p[t][1] ... pos=(0.5*2)=1, alpha=0 -> exactly p[t][1]
    let p = ndarray::arr2(&[[1.0, 2.0, 3.0, 4.0]]);
    let out = apply_homothety(&p, 0.5);
    assert_eq!(out[[0, 2]], 2.0);
}
```
(The all-ones TDC hand case: R[k] = (5-k)/5 for k=1..3 - verify the comment arithmetic while implementing; the assertion is expression-coded so only the structure must be right.) Python: `fmath_log_oracle` built with numpy uint32/float32 bit ops replicating table build + eval; test asserts oracle == dump for every sweep entry; `tdc_oracle` anchored to the same hand case.

- [ ] **Step 3: Run, expect FAIL**; **Step 4: Implement + oracle + cases**

Spec S6.3/S6.4/S6.5. `fmath_log`: build the 2048-entry table once (lazy static or `OnceLock`), all-f32 arithmetic, table build uses f64 `ln` narrowed exactly as `fmath.hpp:204-217`. TDC: sequential R accumulation, plain max, crossing scan with `R[0]`-keyed mixed strict/non-strict polarity, third-crossing two-sided shifted products with `/2` and count-average. `compute_pitch`: argmax lag -> `rate/argmax`, accept iff strictly inside `(rate/max_lag, rate/min_lag)`. `get_pitch`: SPEECH segments only, frames stepping `tdc.shift`, zero-count -> divide by 1 (`BLSTMSpectralSegmenter.cpp:399-437`). `tdc_cases.json`: deterministic windows (closed-form + crafted: constant, alternating-sign for the `R[0]<0` polarity branch, min_lag=0) - Rust cross-check bit-exact.

- [ ] **Step 5: Run, expect PASS** - `cargo test phase1_ltsv_tdc && uv run pytest tests/test_features_oracle.py -v`; **Step 6: Lint + commit** (message: `feat(phase1): f32 table log, TDC score, pitch, homothety warp`)

---

### Task 10: features/stats.rs - InputStatistics

**Files:**
- Modify: `src/rust/src/features/stats.rs`
- Modify: `tools/oracle_harness/main.cpp`
- Create: `src/rust/tests/phase1_stats_golden.rs`
- Modify: `src/python/speech/features_oracle.py`, `scripts/extract_phase1_oracle_cases.py`, `tests/test_features_oracle.py`
- Create (generated): `tests/reference_data/phase1/stats_cases.json`

**Interfaces:**
- Produces: `pub struct InputStatistics { pub mean: Vec<f64>, pub std: Vec<f64>, pub n: i64 }` with `pub fn new() -> Self; pub fn from_matrix(m: &Array2<f64>) -> Self; pub fn update(&mut self, other: &InputStatistics);` Python `stats_merge_oracle(chunks: list[np.ndarray]) -> tuple[mean, std, n]`.

- [ ] **Step 1: Extend harness + regenerate**

Using the REAL compiled `InputStatistics.cpp`: build stats from `synth_20x50.bin` -> dump mean/std/n as `stats_batch.bin` (3 rows: mean, std, [n, 0...] - or three separate dumps `stats_batch_mean.bin`/`_std.bin`/`_n.bin`, pick separate for clarity); then split the matrix rows [0,7) and [7,20), build two InputStatistics, `update` the first with the second, dump merged -> `stats_merged_{mean,std,n}.bin`. Regenerate.

- [ ] **Step 2: Failing tests**

Goldens vs both dump sets. Unit tests: `from_matrix` on [[1],[2],[3]] - mean [2], std expression `(((1-2)^2+(2-2)^2+(3-2)^2)/3).sqrt()`; N=1 -> std exactly 0.0; `update` cases: empty-into-empty keeps n=0; nonempty.update(empty) is a no-op (bit-identical state); empty.update(nonempty) is a plain copy; merge op order pinned by the golden. Python anchor: same [[1],[2],[3]] + merge of ([1],[2]) with ([3]) equals batch of all three ONLY in exact arithmetic - assert oracle merge equals the pooled formula (NOT necessarily the batch bits; document in a comment).

- [ ] **Step 3: Run, expect FAIL**; **Step 4: Implement + oracle + cases**

Spec S7; legacy `InputStatistics.cpp:6-51`. Two-pass batch (sequential column sums); merge in the exact L42-43 op order (std stored sqrt'd, re-squared on merge); `stats_cases.json`: chains of 1-4 deterministic chunks incl. empty chunks; Rust cross-check bit-exact.

- [ ] **Step 5: Run, expect PASS**; **Step 6: Lint + commit** (message: `feat(phase1): streaming input statistics (population std, resqrt merge)`)

---

### Task 11: features/pipeline.rs - FeatureConfig, SpectralParams, assemble_input_sequence (end-to-end gate)

**Files:**
- Create: `src/rust/src/features/pipeline.rs`
- Modify: `src/rust/src/features/mod.rs` (add `pub mod pipeline;`)
- Modify: `tools/oracle_harness/main.cpp`
- Create: `src/rust/tests/phase1_pipeline_golden.rs`
- Create: `tests/reference_data/phase1/variant_{mfcc_deltas,mfcc_sdc,logmel,rawband_ltsv}.config` (hand-written legacy-format configs)

**Interfaces:**
- Consumes: everything above; `speech::legacy_config::parse_legacy_config`.
- Produces:
  - `pub struct FeatureConfig { ... typed fields per spec S8.1 ... }` with `pub fn from_legacy(map: &IndexMap<String, String>, prefix: &str) -> anyhow::Result<FeatureConfig>`
  - `pub struct TdcParams` (moved-in Task 9 definition, unchanged)
  - `pub struct SpectralParams { pub order: u32, pub window_size: usize, pub buffer_size: usize, pub bins: usize, pub shift_frames: usize, pub shift_sec: f64, pub freq_beg: usize, pub freq_end: usize, pub min_freq: f64, pub max_freq: f64, pub ltsv_half_window: usize, pub ltsv_shift: usize, pub tdc: Option<TdcParams> }` with `pub fn derive(cfg: &FeatureConfig, rate: f64) -> SpectralParams`
  - `pub fn assemble_input_sequence(p: &Array2<f64>, mel: Option<&Array2<f64>>, dct: Option<&Array2<f64>>, ltsv: Option<&[f64]>, freq_beg: usize, freq_end: usize) -> Array2<f64>`

- [ ] **Step 1: Write the four variant configs + extend harness + regenerate**

Variant configs (legacy whitespace KEY value format; shared keys: `BLSTM_spectrum_order 8`, `BLSTM_spectrum_shift 0.01`, `BLSTM_windowing_type hamming`, `BLSTM_windowing_param 0.83333`, `BLSTM_flag_DCOffset true`, `BLSTM_preemph_ratio 0.97`, `BLSTM_noise_seed 0`, `BLSTM_noise_ratio 0.001`, `BLSTM_minFreq 64`, `BLSTM_maxFreq 3800`, `BLSTM_minMelFreq 64`, `BLSTM_maxMelFreq 3800`, `BLSTM_spectrum_temporal_convolution_size 0`, `BLSTM_spectrum_temporal_convolution_type none`, `BLSTM_LTSVwindow 0.3`, `BLSTM_LTSVshift 0.04`, `BLSTM_TDCwindow 0`):
- `variant_mfcc_deltas.config`: `BLSTM_nb_bins 26`, `BLSTM_is_log_mel true`, `BLSTM_nb_DCT 13`, `BLSTM_IgnoreFirstDCT false`, `BLSTM_ComputeDeltasNb 3`, `BLSTM_ComputeDeltaDeltasNb 3`
- `variant_mfcc_sdc.config`: same but `BLSTM_ComputeDeltasNb -1`, `BLSTM_ComputeDeltaDeltasNb 0`, `BLSTM_IgnoreFirstDCT true`
- `variant_logmel.config`: `BLSTM_nb_DCT 0`, deltas 0/0, `BLSTM_nb_bins 26`
- `variant_rawband_ltsv.config`: `BLSTM_nb_bins 0`, `BLSTM_minFreq 300`, `BLSTM_maxFreq 3300`
Cross-check chosen key names against the real `/Users/govit/Git/Govit/Speech/tests/reference_data/phase0/1_worker_1.config` (same key spellings) before committing.
Harness: transcribe the param-derivation math (`BLSTMSpectralSegmenter.cpp:194-314` per spec S8.2 - order clamp, sizes, shift quantization, BLSTM-variant freq band clamps, LTSV/TDC params) and `getBLSTMInputSequence` assembly (`:561-591`, spec S8.3); for each variant config (parsed with the legacy `ConfigFile`): run the full pipeline on the excerpt (fresh AudioStruct per variant; preemph per config; noise skipped since seed 0) and dump `inputseq_<variant>_chan{1,2}.bin` plus `params_<variant>.bin` (1 x k row: [order, window_size, bins, shift_frames, freq_beg, freq_end, ltsv_half_window, ltsv_shift] as doubles). Regenerate.

- [ ] **Step 2: Failing tests**

- `FeatureConfig::from_legacy` unit tests: parse each variant config via `parse_legacy_config`, assert typed fields; sanitization cases built from synthetic maps - `minFreq=-5 -> 0` before swap; `maxMelFreq < minMelFreq` swap; span `< 2.0` widened to mean+-1.0 (assert both endpoints); `TDC_lags` shorter than 2 -> Err; negative lag entries clamped.
- `SpectralParams::derive` vs each `params_<variant>.bin` (bit-compare the k doubles).
- End-to-end: for each variant, run the full Rust pipeline (read excerpt 0.35/2.0 -> preemph 0.97 -> params -> windowing -> periodogram -> mel/dct per config -> LTSV -> assemble) and `assert_bits_eq` vs `inputseq_<variant>_chan{1,2}.bin`. THIS IS THE PHASE ACCEPTANCE GATE (spec S11 test 12).
- `assemble_input_sequence` unit tests: raw path band-limits and logs (`ln(x+1e-24)`), mel/dct paths take all columns, LTSV hcat order `[spectral..., ltsv]`.

- [ ] **Step 3: Run, expect FAIL**; **Step 4: Implement**

Spec S8; keep the Task 9 `TdcParams` unchanged. `derive`: order clamped to the Task-4-verified value; `shift_frames = (shift_sec*rate).round()`; freq band per the BLSTM variant (`floor`/`ceil` + both reclamps + snap); LTSV `R = (w*rate/2.0/shift_frames as f64).round()`, `<1 -> 0`; `shift = (s*rate/shift_frames as f64).round()`, `<1 -> 1`; TDC params per spec S8.2 (`None` when `TDCwindow == 0`).

- [ ] **Step 5: Run, expect PASS** - `cargo test phase1_pipeline` then the full suite `cargo test`; **Step 6: Lint + commit** (message: `feat(phase1): param derivation + input assembly; end-to-end feature parity gate green`)

---

### Task 12: Docs, IMPROVEMENTS.md, full verification, smart-commit

**Files:**
- Modify: `CLAUDE.md`, `README.md`, `IMPROVEMENTS.md`
- Modify: `src/rust/src/features/mod.rs` (doc-comment refresh)

- [ ] **Step 1: IMPROVEMENTS.md entries**

Add `[phase1]` entries (format: `- **[phase1] <name>** (<rust file+fn>, from <legacy file:lines>): what / why load-bearing / how ported. *Fix candidate:* ...`) for: dropped N+1th window sample; DC-bin /N vs /4N periodogram normalization; odd-tail same-row double write; get_sequence left-edge stale buffer + index>=frames no-op + DC-offset-over-padding; uniform-window unnormalized = rectangular; `round(rate*dur+1)`; LTSV variance-no-sqrt + mean-only 1e-12 floor; `[delta|delta|dd]` static overwrite; SDC asymmetric offsets {+10..-8} + ignoreFirst last-static clobber; fmath_log finite at 0 (+ f32 round-trip); InputStatistics std-resqrt merge; convolution edge truncation without renormalization. Forward-noted (Phase 2): `freq_beg/end` mel-reset before LTSV; `BLSTM_LTSVshift` uninitialized-member gate; OpenMP stats merge-order nondeterminism (Phase 3/4 threading).

- [ ] **Step 2: CLAUDE.md updates**

Architecture rows -> "Implemented (Phase 1)" for `audio.rs`, `features/fft.rs` (reword: "exact GFFT port (Taylor twiddle seeds); rustfft NOT used - library FFTs cannot match the oracle rounding"), `features/mel.rs`, `features/ltsv_tdc.rs`, `features/stats.rs`; ADD a `features/pipeline.rs` row (legacy source `BLSTMSpectralSegmenter.cpp` param derivation + input assembly); fix the conventions line "matfile lands in Phase 1" -> Phase 4; note `tools/oracle_harness/` + its local-only build in the Testing (Rust) bullet.

- [ ] **Step 3: README roadmap**

Mark Phase 1 done, one paragraph mirroring the 0b-ii style: what is bit-exact vs the strict-IEEE harness, what is oracle/hand-test-only, the -ffast-math finding and the Phase 4 deferral of original-binary comparison.

- [ ] **Step 4: Full verification**

Run: `./check_all.sh && ./lint_code.sh && uv run pytest tests -v` and `LC_ALL=C grep -rn '[^ -~]' tools/oracle_harness scripts/extract_phase1*.py src/rust/src/features src/rust/src/audio.rs docs/superpowers/specs/2026-07-02-phase-1-features-design.md docs/superpowers/plans/2026-07-02-phase-1-features.md | grep -v Binary` (expect empty).
Expected: all green. Fix anything red before proceeding.

- [ ] **Step 5: smart-commit**

Invoke the `smart-commit` skill, telling it to take the whole `feature/phase-1-features` branch into account. No push.

---

## Self-Review

**1. Spec coverage:** S3.1-3.3 -> Tasks 1-2; S3.4/S3.6 -> Task 3; S4.1 -> Task 4; S4.2/S4.3/S3.5 -> Task 5; S5.1/S5.2 -> Tasks 6-7; S6.1/S6.2 -> Task 8; S6.3-S6.5 -> Task 9; S7 -> Task 10; S8 + S11 test 12 (end-to-end gate) -> Task 11; S10 harness -> Tasks 1,4,5,6,7,8,9,10,11 incrementally; S11 tests 1-13 -> mapped in the corresponding tasks (test 13 = Task 1 Step 7); S12 risks: transcription drift (diff-review steps in 8/9/11), GEMM order (Task 7 Step 1), clamp 19-vs-14 (Task 4 Step 1), symphonia decode (Task 2 Step 4 fallback), flags slip (manifest + Task 1 Step 7 assertion); S13 -> Task 12. No gaps found.

**2. Placeholder scan:** Intricate port bodies are deliberately not pre-written (0a/0b RE-with-golden pattern): every implement step names the spec section, exact legacy file:lines, and the non-negotiable semantics, with shown tests as arbiter. Test steps show anchor tests in full; enumerated additional cases state their exact inputs and expected structure. No TBD/TODO/"handle edge cases" remain.

**3. Type consistency:** `Audio { sample_rate, data, data_raw }` consistent across Tasks 2/5/9/11; `windowing_coefficients -> Option<Vec<f64>>` consumed as `Option<&[f64]>` via `.as_deref()` in Tasks 5/9/11; `TdcParams` defined once in Task 9 (in `features/pipeline.rs`) and reused verbatim by Task 11's `SpectralParams.tdc`; `regression_deltas` (Task 6) reused by Task 7; fixture names cross-checked between producing harness steps and consuming golden tests (`sig_norm.bin`, `perio_p8_s80_chan{1,2}.bin`, `logmel_26_chan1.bin`, `mfcc_*_chan1.bin`, `ltsv_chan1.bin`, `fmath_log_sweep.bin`, `stats_batch_*.bin`/`stats_merged_*.bin`, `inputseq_<variant>_chan{1,2}.bin`, `params_<variant>.bin`). Task 9's `get_pitch` consumes phase 0b-i `Segmentation` - existing type, no new definition needed.
