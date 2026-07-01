# Phase 1 - Features Design Spec

Date: 2026-07-02
Status: approved (design), pending implementation plan
Branch: `feature/phase-1-features` (off `main`, which contains merged 0a / 0b-i / 0b-ii)

## 1. Goal and scope

Port the full feature-extraction layer: `audio.rs` (decode, amplitude normalization, pre-emphasis, additive noise, framing/windowing, periodogram driver, convolutions), `features/fft.rs` (exact GFFT port + two-real periodogram), `features/mel.rs` (mel filterbank, DCT-II MFCC, regression deltas, SDC), `features/ltsv_tdc.rs` (LTSV score, TDC score, pitch, homothety warp, `fmath::log`), `features/stats.rs` (streaming input statistics), and a new `features/pipeline.rs` (DSP-parameter derivation + NN-input assembly - the pure seam Phase 2 segmenters will consume).

**Acceptance:** every stage from decoded samples to assembled `inputSeq` matches a strict-IEEE rebuild of the vendored legacy sources **bit-for-bit** on real audio, across configs covering the three spectral variants (raw log-periodogram band, log-mel, MFCC+deltas / SDC) with and without LTSV; per-primitive synthetic goldens, hand-computed unit tests, and numpy differential oracles pin each function independently.

This closes the pure-DSP layer. Phase 2 (NN forward) consumes `pipeline.rs` outputs; comparison against the original `-ffast-math` binary is deferred to the Phase 4 end-to-end parity milestone.

## 2. Decisions (locked)

| # | Decision | Choice |
|---|----------|--------|
| 1 | Oracle | Strict-IEEE C++ harness compiling the vendored legacy feature sources (`-fno-fast-math -ffp-contract=off -DEIGEN_DONT_VECTORIZE`, no OpenMP). The original binary was built `-ffast-math -march=core2` (gcc 4.9) - bit-exact vs original is impossible; bit-exact vs strict rebuild is the target. Original-binary tolerance comparison deferred to Phase 4. |
| 2 | Branch | Single `feature/phase-1-features` (no 1a/1b split). |
| 3 | FFT | Exact port of the legacy GFFT (radix-2 DIT, NR bit-reversal, truncated-Taylor twiddle seeds, running twiddle recurrence). NOT rustfft/realfft - no library FFT reproduces the butterfly rounding. CLAUDE.md module map amended. |
| 4 | Fixtures | Harness dumps via the legacy's own `Matrix2BinaryFile` (Helpers.hpp) - the exact codec `io::binary` already reads byte-exactly. `io/matfile.rs` NOT needed for Phase 1; its CLAUDE.md note corrected. |
| 5 | Harness structure | `AudioStruct.cpp`, `MelFilterBank.cpp`, `InputStatistics.cpp` compiled directly (plus header-only `fft.hpp`, `fmath.hpp`, `Helpers.hpp` with a minimal `iof` shim). LTSV/TDC `classifySequence` + `computePitch` + `getLTSV` + homothety + param derivation are verbatim-transcribed into the harness (linking the `Segmenter` hierarchy would drag `iof`/NN sources); each transcription carries legacy file:line provenance and is diff-reviewed, and independently pinned by numpy oracle + hand tests. |
| 6 | Audio decode | symphonia (wav). PCM16 -> f64 is `x/32768` in both libsndfile and symphonia; the decoded-signal golden pins it. SPHERE/other formats deferred. |
| 7 | Matrix type | `ndarray::Array2<f64>`, rows = time frames, cols = bins/coeffs (legacy orientation). All reductions are hand-written sequential left-to-right loops - `EIGEN_DONT_VECTORIZE` makes the oracle's order identical. |
| 8 | Scope cut | `AudioStruct` file_types 1-4 excluded (external-feature corpus modes; need `.mat` reading). Dead `getTDC` feature-append excluded. Segmenter wiring, NN windowing, targets, pitch-pass orchestration excluded (Phase 2). |

## 3. Audio container + preprocessing (exact) - `audio.rs`

Legacy: `AudioStruct.{h,cpp}`, `Helpers.hpp:222-272`, `Constants.h`.

### 3.1 Decode + amplitude normalization (`AudioStruct.cpp:36-137`, file_type 0)

- Frame math (`:62-70`): `frame_offset_begin = round(rate*Audio_offset)`; `frame_count_max = round(rate*Audio_max_duration + 1)` - the **+1 is inside the round** (one extra sample); `frames = sf_frames - frame_offset_begin`, clamped to `[0, frame_count_max]`. All sec->sample roundings in this phase are `boost::math::round` = **round-half-away-from-zero**.
- Layout: `(channels x samples)`, interleaved decode.
- Per-channel normalization (`:121-126`, always on, applied to `_DataRaw`):
  `adim = sqrt(sum(x^2)/S)`; `adim = (2*adim + max|x|)/2`; `if adim < 1e-3 { adim = 1e-3 }`; `x /= adim`. Divisor is `(2*RMS + peak)/2` - not peak, not RMS. RMS sum is a sequential left-to-right reduction.
- `_Data = _DataRaw` working copy; `reset()` restores.
- Config keys (read at corpus level, `BagOfProcessors.cpp:11-13`): `Audio_offset` (default 0.0), `Audio_max_duration` (default 3.6e6), `File_Type` (default 0).

### 3.2 Pre-emphasis (`AudioStruct.cpp:446-454`)

In-place **backward** loop per channel: `for kk = S-1 down to 1: x[kk] -= ratio*x[kk-1]`; sample 0 untouched. Gated `<prefix>_preemph_ratio > 0`. Applied ONCE for the whole file, BEFORE noise.

### 3.3 Additive noise (`AudioStruct.cpp:437-444`)

`x[chan][kk] += 2*ratio*_RandomVector[kk % 100000] - ratio` - uniform table (`constants::random_uniform`), index = absolute sample index restarting at 0 per file, SAME value across channels. Gated `<prefix>_noise_seed > 0`; the seed value is never used as a seed (gate only); amplitude from `<prefix>_noise_ratio`.

### 3.4 Windowing coefficients (`Helpers.hpp:222-272`)

`windowing_coefficients(type, normalized, size, extra_param)` -> `1 x size` row, or EMPTY (= rectangular, no multiply) when `size <= 1`, type `"none"`, unknown type, or **`"uniform"` with `normalized == false`** (quirk: uniform only exists normalized).
- hamming: `0.54 - 0.46*cos(2*PI*ii/(size-1))`, ii = 0..size-1.
- hann: `0.5 - 0.5*cos(2*PI*ii/(size-1))`.
- hHCw: `extra_param` clamped [0,1] (default 0.83333); `size1 = round(extra_param*size)`, `size2 = size-size1`; ii < size1: `0.54-0.46*cos(2*PI*ii/(2*size1-1))`; ii >= size1: `cos(2*PI*(ii-size1)/(4*size2-1))`.
- `normalized == true`: divide all coeffs by their sum.
- `PI = 3.14159265358979323846264338327` (`Constants.h:15`; same double as `std::f64::consts::PI`).
Signal windows are built UNNORMALIZED; convolution kernels NORMALIZED.

### 3.5 Framing - `get_sequence` (`AudioStruct.cpp:456-485`)

Window of `2w+1` samples centered at `index` (w = half-window). Outer guard: `index >= frames` -> **no-op, caller consumes the stale previous buffer**. Branches (order matters - left-edge test first):
- left edge (`index < w`): `beg_win = w-index; nb_elem = w+index+1; beg_data = 0` - **NO zeroing**: entries `[0, beg_win)` keep the previous frame's (already windowed) content.
- right edge (`index >= frames-w`): `setZero()` then copy `nb_elem = w-index+frames` from `beg_data = index-w`.
- interior: full copy.
Then, in order: optional DC-offset removal - `mean = sum(buffer)/(2w+1)` over the FULL buffer including padding/stale cells, subtracted from the full buffer (gated `<prefix>_flag_DCOffset`); then window `cwiseProduct` iff coeffs non-empty. The port keeps the caller-owned reused buffer so stale-content semantics are reproducible.

### 3.6 Convolutions (`Helpers.hpp:164-220`)

- `Convolution` (horizontal, over row vectors) and `ConvolutionVert` (down time frames of the periodogram): kernel length `2K+1` (normalized, sums to 1), edges TRUNCATED with **no renormalization** (edge outputs attenuated). Left-edge index arithmetic uses unsigned negation wrap (`begin1 = -(K-ii)`) that re-wraps into range - port with signed arithmetic producing identical indices.
- `ConvolutionVert` transposes a copy, zeroes the target, accumulates `sum_j copy(bin, idx+j)*coeff(j)` sequentially.

## 4. FFT + periodogram (exact) - `features/fft.rs` + `audio.rs`

Legacy: `fft.hpp` (GFFT, V. Myrnyy DDJ 2007), `AudioStruct.cpp:487-584`.

### 4.1 GFFT kernel (`fft.hpp`)

- Radix-2 DIT, in-place, interleaved `[re0, im0, re1, im1, ...]`, forward with NEGATIVE exponent, **unnormalized**. Sizes strictly `N = 2^P`; the engine registers P in [1,19] and clamps `_SpectrumOrder` to 19 (`BLSTMSpectralSegmenter.cpp:199-203`; the display-only `computeStandardSpectrogram` uses Max=15 -> clamp 14, out of scope). Note: one report cited the segmenter clamp as 14 - the implementer must verify 19 vs 14 against `BLSTMSpectralSegmenter.cpp:199-203` before coding (two of three reports and the factory range `Min=1,Max=20` say 19).
- Bit-reversal scramble: Numerical-Recipes verbatim (`fft.hpp:176-190`), 1-based `int` loop over `2N` doubles.
- Danielson-Lanczos combine (`:74-104`): recurse halves, then butterfly with the RUNNING twiddle recurrence
  `wtemp = -Sin(N,1); wpr = -2*wtemp^2; wpi = -Sin(N,2); wr=1; wi=0;` and per step `wtemp = wr; wr += wr*wpr - wi*wpi; wi += wi*wpr + wtemp*wpi;` - rounding error ACCUMULATES across the loop; do NOT substitute per-bin sin/cos.
- Base cases: `DL<2>` plain butterfly; `DL<4>` hard-coded with fused `-i` twiddle (`:107-136`) - reproduce the exact statement order.
- Twiddle seeds are TRUNCATED TAYLOR SERIES, not libm: `Sin(B,A) = (A*PI/B) * S(2,34)` where `S(M,N) = 1 - ((x*x)/M)/(M+1) * S(M+2,N)`, `x = (A*PI)/B` (`A*PI` first), termination `S(N,N) = 1`. Port the polynomial with identical evaluation order (Rust `const fn` or runtime recursion; values must equal the C++ constant-folded doubles).

### 4.2 Two-real packing + periodogram (`AudioStruct.cpp:487-510`)

- Packing: `data[2i] = sig1[i]; data[2i+1] = sig2[i]` for i in [0, N) - **the windowed buffers hold N+1 samples; the last is silently dropped**.
- Unpack + normalize, with `size_1 = 2N`, `size_2 = 2N+1`, `coeff_norm = 4N`:
  - DC: `P1(0) = data[0]^2/N`, `P2(0) = data[1]^2/N`.
  - bins k = ii/2, ii = 2,4,..,N (Nyquist INCLUSIVE):
    `P1(k) = ((data[ii]+data[size_1-ii])^2 + (data[ii+1]-data[size_2-ii])^2)/(4N)`
    `P2(k) = ((data[ii+1]+data[size_2-ii])^2 + (data[size_1-ii]-data[ii])^2)/(4N)`
  Net `|S(k)|^2/N` all bins, via exactly these expressions. No window-energy compensation, no one-sided doubling, no 1/fs.

### 4.3 Framing driver (`AudioStruct.cpp:512-584`) - `audio.rs`

- Sizes: `window_size = 1<<p`; half-window `= 2^(p-1)`; buffer `= 2^p+1`; bins `= 2^(p-1)+1`.
- Frame count: `frameNb = ceil((end_frame-begin_frame+1)/shift)` (exact-divide test, integer ops).
- Loop `jj = begin; jj <= end-shift; jj += 2*shift`: TWO frames per FFT (centers jj, jj+shift) -> rows `(jj-begin)/shift`, `+1`.
- Odd `frameNb` tail: last frame packed as BOTH signals, written twice to the same row (second write is the P2 formula).
- Stale-buffer interactions: per-thread buffers zero-initialized once; `getSequence` no-op (index >= frames) re-transforms the previous window - reproduced by keeping one pair of reused buffers in the port (sequential).
- After the loop: optional `ConvolutionVert` when kernel length > 1; then mel/DCT hand-off.

## 5. Mel filterbank + MFCC/deltas/SDC (exact) - `features/mel.rs`

Legacy: `MelFilterBank.{h,cpp}`.

### 5.1 Construction (`cpp:15-127`)

- `Hz2Mel(f) = 1125*ln(1+f/700)`; `Mel2Hz(m) = 700*(exp(m/1125)-1)` (`cpp:362-368`), std double math.
- Frequency grid (`cpp:19-30`): `freqStep = rate/2/spectrum_size`; `freqs` built by **sequential accumulation** `freq += freqStep` while `freq <= rate/2` (not `i*step`); `_BegFreq = floor(max(minFreq,0)/freqStep)`, `_EndFreq = ceil(min(maxFreq,rate/2)/freqStep)` - recomputed from the caller-snapped values (double round-trip, reproduce literally).
- Mel edge grid (`cpp:32-53`): `melStep = (Hz2Mel(maxMelFreq)-Hz2Mel(minMelFreq))/(nb_bins+1)`, floored at `Hz2Mel(1.0)`; walk DOWN from `Hz2Mel(minMelFreq)` while `mel > Hz2Mel(max(minFreq,0))-melStep`, then UP pushing `Mel2Hz(mel)` while `mel < Hz2Mel(min(maxFreq,rate/2))+2*melStep` - sequential `-=`/`+=` accumulation. Filter count = `mels.len()-2`, NOT nb_bins.
- Triangles (`cpp:55-107`): filter ii spans `[mels[ii-1], mels[ii+1]]`; bin membership INCLUSIVE both edges over jj in `[_BegFreq,_EndFreq]`; rising `((f-l)/(c-l))` if `f < c` else falling `((r-f)/(r-c))` (center hits falling branch, coeff exactly 1); unit peak, NO normalization; `_IndexBegin[ii]` = first qualifying absolute bin, `_Coeffs[ii]` contiguous. **If ANY filter collects zero bins, the ENTIRE bank falls back to raw-band pass-through** (`_NbFilters = _EndFreq-_BegFreq+1; _IsMel = false`).
- DCT-II (`cpp:108-127`): `_NbDCT = min(_NbDCT, _NbFilters)`; matrix `(NbFilters x NbDCT)`, entry `(n,k) = cos(PI/NbFilters*(n+0.5)*k)` - UNNORMALIZED (no 2, no sqrt(2/N), no 1/2 on k=0).

### 5.2 Application (`cpp:130-360`)

- `applyFilterBank`: log-mel = `ln(dot + 1e-24)` (**additive** floor inside ln); non-log mel = plain dots; non-mel fallback = `ln(block + 1e-24)` or raw block copy. Dots are sequential `sum_k coeffs[ii][k]*P(t, idx_begin[ii]+k)`.
- Regression delta kernel (identical everywhere): `D[t] = sum_{j=1..n} j*( B[min(t+j,T-1)] - B[max(t-j,0)] ) / (2*sum j^2)` - HTK-style, EDGE-CLAMPED; implemented via the exact legacy block-op sequence (shifted copies + edge-row corrections, `D += j*deltas` ascending j, divide at the end); `adim` includes clamped-j terms; T <= j skips shifted copies (edge additions reproduce the clamped formula).
- **applyFilterBank no-DCT deltas quirk (`cpp:192-193`): output layout is `[delta | delta | delta-delta]` - the static log-mel block is overwritten by the delta block** (statics discarded, deltas duplicated). Contrast applyDCT layout `[static | delta | delta-delta]`. Load-bearing asymmetry.
- `applyDCT` branches: (A) SDC (`_ComputeDeltasNb < 0`): hardcoded d=3, P=3, k=7; delta = the n=3 regression kernel (adim 28), NOT the textbook two-point difference; stack 7 blocks at vertical offsets 3*kk; extract at row offset `(k*P-1)/2 = 10` -> per-block time offsets {+10,+7,+4,+1,-2,-5,-8} (asymmetric), ZERO padding out of range; with `_IgnoreFirstDCT`, SDC block 0 OVERWRITES the last static column (c0 kept, c_{N-1} lost). (B) ignoreFirst + deltas >= 0: compute into a T x (cols+1) temp, then drop exactly the first column (static c0 only; delta-of-c0 kept). (C) plain: `[c | dc | (ddc)]`, statics kept.
- Output widths (`h:32-89`): `getNbFilters` = (dd>0 ? 3 : 2)*NbFilters when deltas-no-DCT else NbFilters; `getNbDCT` per branch incl. SDC `= (ignoreFirst ? NbDCT-1 : NbDCT) + 7*NbDCT`.

## 6. LTSV + TDC (exact) - `features/ltsv_tdc.rs`

Legacy: `LongTermSpectralVariation.cpp`, `TimeDomainCorrel.cpp`, `BLSTMSpectralSegmenter.cpp:300-437,757-805`, `fmath.hpp:186-226,713-727`.

### 6.1 LTSV score - `classifySequence` (`LongTermSpectralVariation.cpp:82-128`)

Window `[max(0,col-R), min(T-1,col+R)]` inclusive, `length` SHRINKS at edges (no padding). Per freq bin in `[freq_beg, freq_end]` inclusive: `mean = sum(P(t,bin))/length`, floored `1e-12` (mean only, never the numerator); `dzeta[bin] = -(1/length) * sum_t r*(r-1)` with `r = P(t,bin)/mean` (variable named `tmp_log`, NO log anywhere); `mean_dzeta = mean over bins`; return **biased variance** `sum((dzeta-mean_dzeta)^2)/nbBins` - divide by N, **no sqrt** (comment lies).

### 6.2 LTSV column - `getLTSV` (`BLSTMSpectralSegmenter.cpp:316-339`)

Over periodogram rows: compute at `jj = 0, shift, 2*shift, ...`; backfill skipped rows by linear interpolation `(1-a)*prev + a*curr`; trailing rows stay 0. Uses the RAW periodogram with `freq_beg/freq_end` as passed (the mel-reset quirk that makes those bounds mel-sized is Phase 2 wiring; the function ports as-is).

### 6.3 TDC score - `classifySequence` (`TimeDomainCorrel.cpp:36-91`)

`adim = squaredNorm(window)` floored `1e-12`; `R[k-min_lag] = sum_{i=0}^{len-k-1} x[i]*x[i+k] / adim`, k in `[min_lag, max_lag]` inclusive (full-window-energy normalization, no length compensation); `MaxPeak = max R` (plain max, includes endpoint; `R = 1` exactly at k=0). Zero-crossing scan on R with polarity keyed on `R[0]` (mixed strict/non-strict: `>0 && <=0` vs `<0 && >=0`); from the 3rd crossing, two-sided shifted inner products between consecutive period segments, `max over shift`, `/2` (mm=0 double count), averaged by crossing-pair count (0 if fewer than 3 crossings). Return `balance*(-fmath_log(1-MaxPeak)) + (1-balance)*CrossCorr`.

### 6.4 `fmath::log` (f32 table log, `fmath.hpp:186-226,713-727`)

2048-entry table: `x = 1 + i/2048`; `app = (f32)ln(x)`; `rev = (f32)((ln(x+h-e)-ln(x))/((h-e)*2^23))` with `h = 2^-11`, `e = 2^-24`, last entry `rev = (f32)(1/(x*2^23))`; `c_log2 = ln(2.0f32)/2^23`. Eval: from the f32 bits, `a = bits & (0xFF<<23)`, `idx = (bits & (0x7FF<<12))>>12`, `b2 = bits & 0xFFF`; `f = (a - (127<<23)) as f32 * c_log2 + app[idx] + b2 as f32 * rev[idx]`. Double arg NARROWED to f32, result widened. No x<=0 guard: `fmath_log(0.0) ~= -88.02969` finite (hit when min_lag == 0 -> MaxPeak == 1). Reproduce bit-exactly, f32 arithmetic throughout.

### 6.5 Pitch + homothety (pure functions; wiring is Phase 2)

- `computePitch` (`BLSTMSpectralSegmenter.cpp:172-192`): argmax lag -> `rate/argmax_lag`, accepted iff `rate/max_lag < est < rate/min_lag`.
- `getPitch` (`:399-437`): average of accepted estimates over frames stepping TDC_window_shift inside SPEECH segments of a `Segmentation`; zero-count guard -> divide by 1.
- Homothety warp (`:762-775`): `coeff = pitch/300`; per output bin ii: `pos = (coeff*ii) as int; alpha = coeff*ii - pos; P(t,ii) = (1-alpha)*Pmem(t,pos) + alpha*Pmem(t,pos+1)` with bounds checks.

## 7. Streaming stats (exact) - `features/stats.rs`

Legacy: `InputStatistics.{h,cpp}`.

- State: `(mean: 1 x D, std: 1 x D, n: i64)`. **std is population (divide by N), finalized (sqrt'd) after EVERY batch and EVERY merge** - the accumulator stores std, not variance; merges re-square it. No epsilon anywhere inside (the `max(1e-12, std)` floor belongs to the NN normalization apply, Phase 2; the `+1e-32` floor to the self-normalization branches, Phase 2).
- Batch (`cpp:8-16`): two-pass - `mean = colsum/N` then `std = sqrt(colsum((x-mean)^2)/N)`. Sequential column sums. N=1 -> std exactly 0. No N=0 guard (callers guard).
- Merge `update` (`cpp:30-51`): self N==0 -> plain copy (empty-into-empty keeps N=0); incoming N==0 -> silent no-op; else `N = N1+N2`; `mean = (m1*N1 + m2*N2)/N`; `std = sqrt(((std1^2 + (mean-m1)^2)*N1 + (std2^2 + (mean-m2)^2)*N2)/N)` in exactly this op order.
- Cross-file merge order in the legacy is OpenMP-completion-order nondeterministic (`CorpusProcessor.cpp:173-199`) - the port documents serial order as the deterministic choice (IMPROVEMENTS.md forward-note for Phase 3/4 threading).

## 8. Param derivation + input assembly (exact) - `features/pipeline.rs` + `FeatureConfig`

Legacy: `BLSTMSpectralSegmenter.cpp:194-314`, `Segmenter.cpp:61-148`, `LongTermSpectralVariation.cpp:41-79,145-155,200-263`.

### 8.1 `FeatureConfig::from_legacy(map, prefix)`

Reads `<prefix>_spectrum_order/spectrum_shift/spectrum_temporal_convolution_{size,type}/windowing_{type,param}/flag_DCOffset/preemph_ratio/noise_{seed,ratio}/minFreq/maxFreq/minMelFreq/maxMelFreq/nb_bins/is_log_mel/nb_DCT/IgnoreFirstDCT/ComputeDeltasNb/ComputeDeltaDeltasNb/LTSVwindow/LTSVshift/TDCwindow/TDCshift/TDC_lags/TDC_balance/TDC_windowing_{type,param}`, plus corpus-level `Audio_offset/Audio_max_duration/File_Type`. Sanitization exactly as `LongTermSpectralVariation.cpp:57-79`: negative Hz clamps to 0 BEFORE the swap; swap if reversed; `< 2.0` span widened to `mean +- 1.0` - for BOTH the Hz and mel pairs. `TDC_lags` needs >= 2 entries, ALL entries clamped >= 0. The `BLSTM_LTSVshift` uninitialized-member read gate (`BLSTMSpectralSegmenter.cpp:50`) is Phase 2 wiring - `FeatureConfig` reads the key unconditionally and the quirk is forward-noted.

### 8.2 `SpectralParams::derive`

`order` clamped to 19; `window_size = 1<<order`; buffer `= 2^order+1`; bins `= 2^(order-1)+1`; `shift_frames = round(shift_sec*rate)` (the `=80` non-wav fallback is Phase 2); shift re-quantized `shift_frames/rate`. Freq band (`BLSTMSpectralSegmenter.cpp:229-239`): `freqStep = rate/2/(bins-1)`; `freq_beg = clamp(floor(minFreq/freqStep))`, `freq_end = clamp(ceil(maxFreq/freqStep))`; both clamp orders exist in the legacy (BLSTM variant has the extra `freq_end < freq_beg` reclamp; LTSV variant does not) - port the BLSTM variant, note the divergence. Snap `minFreq/maxFreq = idx*freqStep`. LTSV params (`:300-314`): `R = round(w*rate/2/shift_frames)`, `< 1 -> 0` disables (the LID-variant floor-to-1 is Phase 2); `shift = round(s*rate/shift_frames)`, `< 1 -> 1`. TDC params (`:341-370`): half-window `round(w*rate/2)`; `full = 2*hw+1`; `min_lag = round(l0*rate)` `< 1 -> 1`; `max_lag = round(l1*rate)` clamped `<= full-1`, `>= min_lag`.

### 8.3 `assemble_input_sequence` (`BLSTMSpectralSegmenter.cpp:561-591`)

Priority: DCT active -> `CepstreCoefficients`; mel active -> `FilterBankedPeriodogram`; else `ln(P.block(rows, freq_beg..=freq_end) + 1e-24)` (band-limit ONLY on the raw path). Then hcat LTSV column if present (TDC column is dead code, not ported). Output T x D, feature order `[spectral..., LTSV]`.

## 9. Module interfaces

```rust
// audio.rs
pub struct Audio { pub sample_rate: u32, pub data: Array2<f64>, pub data_raw: Array2<f64> } // channels x samples
pub fn read_audio(path: &Path, offset_sec: f64, max_duration_sec: f64) -> Result<Audio>;
pub fn normalize_channels(data: &mut Array2<f64>);                       // (2*RMS+peak)/2, floor 1e-3
impl Audio { pub fn apply_preemph(&mut self, ratio: f64); pub fn apply_noise(&mut self, ratio: f64); pub fn reset(&mut self); }
pub fn windowing_coefficients(kind: &str, normalized: bool, size: usize, extra_param: f64) -> Option<Vec<f64>>; // None = rectangular
pub fn get_sequence(data: &Array2<f64>, chan: usize, index: usize, half_window: usize,
                    dc_offset: bool, coeffs: Option<&[f64]>, buf: &mut Array2<f64>);   // caller-owned reused buffer
pub fn compute_segment_periodogram_estimates(audio: &Audio, p: u32, shift: usize, chan: usize, dc_offset: bool,
                    coeffs: Option<&[f64]>, conv: Option<&[f64]>, begin: usize, end: usize) -> Array2<f64>;
pub fn convolution_horiz(row: &mut Array2<f64>, coeffs: &[f64]);
pub fn convolution_vert(mat: &mut Array2<f64>, coeffs: &[f64]);

// features/fft.rs
pub struct Gfft { /* order p, seeds */ }
impl Gfft { pub fn new(p: u32) -> Gfft; pub fn fft(&self, data: &mut [f64]); }           // interleaved, in-place, unnormalized
pub fn compute_two_real_periodogram(gfft: &Gfft, n: usize, sig1: &[f64], sig2: &[f64],
                    out: &mut Array2<f64>, row1: usize, row2: usize);

// features/mel.rs
pub struct MelFilterBank { /* index_begin, coeffs, dct, flags, widths */ }
impl MelFilterBank {
    pub fn new(min_mel: f64, max_mel: f64, nb_bins: i32, min_freq: f64, max_freq: f64, rate: f64,
               spectrum_size: usize, is_log: bool, nb_dct: i32, ignore_first: bool,
               deltas_nb: i32, delta_deltas_nb: i32) -> MelFilterBank;
    pub fn apply_filter_bank(&self, periodogram: &Array2<f64>) -> Array2<f64>;
    pub fn apply_dct(&self, mel: &Array2<f64>) -> Array2<f64>;
    pub fn nb_filters(&self) -> usize; pub fn nb_dct(&self) -> usize; pub fn is_mel(&self) -> bool;
}
pub fn hz_to_mel(f: f64) -> f64; pub fn mel_to_hz(m: f64) -> f64;

// features/ltsv_tdc.rs
pub fn ltsv_classify_sequence(p: &Array2<f64>, col: usize, freq_beg: usize, freq_end: usize, half_window: usize) -> f64;
pub fn get_ltsv(p: &Array2<f64>, freq_beg: usize, freq_end: usize, half_window: usize, shift: usize) -> Vec<f64>;
pub fn tdc_classify_sequence(window: &[f64], min_lag: usize, max_lag: usize, balance: f64) -> f64;
pub fn compute_pitch(window: &[f64], min_lag: usize, max_lag: usize, rate: f64) -> Option<f64>;
pub fn get_pitch(audio: &Audio, seg: &Segmentation, chan: usize, params: &TdcParams) -> f64;
pub fn apply_homothety(p: &Array2<f64>, coeff: f64) -> Array2<f64>;
pub fn fmath_log(x: f32) -> f32;

// features/stats.rs
pub struct InputStatistics { pub mean: Vec<f64>, pub std: Vec<f64>, pub n: i64 }
impl InputStatistics { pub fn new() -> Self; pub fn from_matrix(m: &Array2<f64>) -> Self; pub fn update(&mut self, other: &InputStatistics); }

// features/pipeline.rs
pub struct FeatureConfig { /* typed fields */ }
impl FeatureConfig { pub fn from_legacy(map: &IndexMap<String,String>, prefix: &str) -> Result<FeatureConfig>; }
pub struct TdcParams { /* half_window, full_window, shift, min_lag, max_lag, windowing coeffs, balance */ }
pub struct SpectralParams { /* order, sizes, shift_frames, freq_beg/end, ltsv half_window/shift, pub tdc: Option<TdcParams> */ }
impl SpectralParams { pub fn derive(cfg: &FeatureConfig, rate: f64) -> SpectralParams; }
pub fn assemble_input_sequence(p: &Array2<f64>, mel: Option<&Array2<f64>>, dct: Option<&Array2<f64>>,
                    ltsv: Option<&[f64]>, freq_beg: usize, freq_end: usize) -> Array2<f64>;
```

Python oracle (tests only, `src/python/speech/`): `features_oracle.py` with `regression_deltas_oracle`, `sdc_oracle`, `ltsv_oracle`, `tdc_oracle`, `stats_merge_oracle`, `fmath_log_oracle` (numpy float32 bit ops), `mel_grid_oracle`.

## 10. Oracle harness - `tools/oracle_harness/`

- `main.cpp` + `iof_shim/iof/io.hpp` (minimal `iof::fmtr` compile-through shim) + `build.sh`, committed. Compiles the git-ignored `legacy/src/{AudioStruct,MelFilterBank,InputStatistics,ConfigFile}.cpp` (header-only: `fft.hpp`, `fmath.hpp`, `Helpers.hpp`, `Constants.h`) with brew `eigen libsndfile libmatio boost`; flags `-O2 -std=c++11 -fno-fast-math -ffp-contract=off -DEIGEN_DONT_VECTORIZE` and NO `-fopenmp` (pragmas ignored -> sequential). Builds locally only, never in CI.
- LTSV/TDC/pitch/homothety/param-derivation/`getLTSV`/input-assembly are verbatim transcriptions in `main.cpp`, each with a `// legacy: file:lines` provenance comment, diff-reviewed against the source in the plan's review step.
- Dumps via the legacy's `Matrix2BinaryFile` (i64 rows, i64 cols, f64 column-major LE - readable by `io::binary` as-is).
- Inputs: (a) `tests/reference_data/phase1/excerpt_2ch_8k.wav` - a committed ~3 s cut of `PRCTS_RUS_RU_0000263489_01.wav` (2 ch, 8 kHz, Int16); the harness reads it with a nonzero `Audio_offset`/`Audio_max_duration` to exercise the offset/truncation math; (b) synthetic matrices from the closed-form `((i*7 + j*13) % 100)/100.0` for per-primitive dumps.
- Per config variant (3-4 variants derived from vendored real configs: MFCC+deltas, MFCC+SDC, log-mel, raw-band+LTSV), stage dumps: normalized signal, post-preemph+noise signal, window coeffs (each type incl. hHCw), periodogram (with and without temporal convolution via config), mel, log-mel, MFCC/deltas/SDC, LTSV column, TDC scores over frames, pitch scalar, homothety-warped periodogram, stats (mean/std/n after batch and after a two-part merge), assembled inputSeq, plus an `fmath_log` sweep (x in a deterministic list covering table edges, subnormal-adjacent, 1-MaxPeak values) and small-N FFT input/output pairs (N = 4, 8, 256).
- Fixture extraction script `scripts/extract_phase1_fixtures.py` runs the harness, sanity-checks anchors, and the committed fixture set is what CI consumes (CI never needs the harness, brew deps, or `legacy/`).

## 11. Golden fixtures, strategy, acceptance

Vendored into `tests/reference_data/phase1/` (committed): the wav excerpt, all harness `.bin` dumps, oracle-generated JSON case files, a `manifest.json` naming the harness flags + config variants.

Tests:
1. `read_audio` + normalization: decoded/normalized signal == harness dump, bit-exact (pins symphonia-vs-libsndfile decode too).
2. preemph/noise: bit-exact vs dumps; hand-computed 5-sample cases; noise indexes `constants::random_uniform`.
3. `windowing_coefficients`: bit-exact vs dumps per type; hand-computed small sizes; `uniform`+unnormalized -> None quirk unit-tested.
4. `Gfft::fft`: bit-exact vs FFT in/out dumps (N = 4, 8, 256); property tests (impulse, Parseval) at tolerance as sanity only.
5. Periodogram (packing + normalization + framing driver + odd tail + temporal convolution): bit-exact vs dumps for both channels; hand-computed N=4 case for the two-real unpack.
6. Mel/MFCC/deltas/SDC: bit-exact vs dumps (all config variants); mel grid + triangle edges hand-computed; deltas/SDC cross-checked bit-for-bit vs numpy oracle JSON; the `[delta|delta|dd]` overwrite quirk pinned by a dedicated case.
7. LTSV: bit-exact vs dumps + numpy oracle JSON (deterministic inputs + crafted edge rows: window shrink at t=0/T-1, mean-floor trigger).
8. TDC + pitch + homothety: bit-exact vs dumps + numpy oracle JSON (crafted: min_lag=0 -> MaxPeak=1 -> fmath_log(0) finite; <3 crossings -> CrossCorr=0; polarity branch on R[0]<0).
9. `fmath_log`: bit-exact vs the sweep dump AND vs the numpy float32 oracle (double pinning; the table build itself compared entry-by-entry).
10. `InputStatistics`: bit-exact vs batch+merge dumps; numpy oracle for merge chains (empty-into-empty, N==0 no-op cases).
11. `FeatureConfig`/`SpectralParams`: hand-computed from the vendored real configs (`1_worker_1.config`, `seg.config`, `lid.config` values); sanitization order cases (negative-then-swap, span widening).
12. `assemble_input_sequence`: bit-exact vs inputSeq dumps (the end-to-end feature-pipeline acceptance gate).
13. Fixture-presence pytest (`tests/test_phase1_fixtures.py`) asserting manifest keys, dump shapes, wav properties.

## 12. Risks and pitfalls (pinned by tests / logged)

- Harness transcription drift (LTSV/TDC/params in `main.cpp`): pinned by diff-review step + numpy oracle + hand tests agreeing with the Rust port independently of the harness.
- GFFT twiddle series constant-folding: the C++ compiler folds `Sin<N,1>::value()` at compile time with strict semantics; Rust must compute the identical doubles (same nesting order). Pinned by test 4 at every registered N used in fixtures.
- Sequential-reduction assumption: `EIGEN_DONT_VECTORIZE` + no `-ffast-math` makes Eigen sums left-to-right; any harness flag slip (e.g. forgetting the define) shows up as test-1/6 mismatches, and the manifest records the exact flags.
- Eigen version drift (brew 3.4 vs 2015-era 3.2): the compiled paths are element-wise/sequential only (no matrix products in the feature path except `melPeriodogram*_CoeffsDCT` - **a real Eigen GEMM**). `EIGEN_DONT_VECTORIZE` forces the scalar kernel; the plan verifies the product matches a hand-loop dot ordering on a synthetic case before trusting DCT dumps; if Eigen's scalar GEMM accumulation order differs from the naive loop, the harness replaces that one call with an explicit loop (provenance-commented) and the deviation is logged in the manifest.
- Symphonia decode mismatch (dither-free PCM16 only expected): pinned by test 1; if symphonia normalizes differently, `read_audio` adapts (decode is not part of the bit-exact math, the samples are).
- Spectrum-order clamp 19-vs-14 report discrepancy: implementer verifies `BLSTMSpectralSegmenter.cpp:199-203` before coding `SpectralParams`.
- Stale-buffer framing semantics (left-edge no-zero, index>=frames no-op): reproduced via the caller-owned reused buffer; pinned by right-edge/left-edge hand tests and the real-audio dumps (whose first/last frames exercise both).

## 13. Definition of done

- `./check_all.sh` green (cargo test, fmt --check, clippy -D warnings, release build); `./lint_code.sh` green (ruff, mypy); `uv run pytest tests` green.
- All 13 test groups in section 11 passing; every bit-exact assertion is exact f64 (`assert_eq!` on bits or `==`), not approx.
- `audio.rs`, `features/{fft,mel,ltsv_tdc,stats,pipeline}.rs` no longer stubs; narrow `#[allow]`s removed where obsolete.
- CLAUDE.md: architecture rows flip to Implemented (Phase 1); `features/fft.rs` row notes the exact GFFT port (not rustfft); `io/matfile.rs` "lands in Phase 1" note corrected to Phase 4; `features/pipeline.rs` row added.
- README roadmap: Phase 1 marked done with validation summary.
- IMPROVEMENTS.md: new entries for every reproduced quirk in sections 3-8 (at minimum: dropped N+1th window sample; DC/4N periodogram normalization; odd-tail double write; left-edge stale buffer + DC-offset-over-padding; uniform-window quirk; `round(rate*dur+1)`; LTSV variance-no-sqrt + mean-only floor; `[delta|delta|dd]` overwrite; SDC offset asymmetry + ignoreFirst clobber; fmath_log at 0 finite; stats std-resqrt merge; OpenMP merge-order nondeterminism forward-note); `freq_beg/end` mel-reset and `BLSTM_LTSVshift` gates stay Forward-noted for Phase 2.
- Final step: `smart-commit` skill over the whole branch. No push.

## 14. Out of scope

`AudioStruct` file_types 1-4; dead `getTDC` feature append; `Segmenter` trait + Algo dispatch + NN windowing/result sizing/targets/timeStep (Phase 2); pitch-pass orchestration (Phase 2); `io/matfile.rs` (Phase 4 diagnostics); TOML config surface; resurrecting the original `fsp` binary and any tolerance comparison against it (Phase 4); `computeStandardSpectrogram`/`resizeMelSpectrogram` (display-only paths); SPHERE decode.
