# Improvements & Deferred Fixes

Tracked work to revisit once the Rust/Python port reaches end-to-end parity with the legacy
`FastSpeechProcessing` engine. Nothing here should be "fixed" mid-port - these are reproduced
bit-exactly on purpose so the goldens match.

**Maintainer note:** every phase that reproduces a legacy quirk/bug adds an entry to the Legacy
Quirks section below (what / where / why deferred / fix candidate). See CLAUDE.md for the pointer.

## Legacy Quirks (reproduced bit-exactly; revisit after parity)

- **[0a] adim bias restore vs MATLAB FP path** (`config.rs` `apply_adim`, `weight_bridge.py`): we
  restore the bias to its ORIGINAL value; legacy `config2network.m` computes `(x/adim)*adim`, which
  differs by up to 1 ULP on ~34 biases. We chose original-restore so Rust == Python. *Fix candidate:*
  once the Phase-4 inference golden lands, decide whether strict legacy parity requires replicating
  `(x/adim)*adim` on the bias.
- **[0b-i] CostLaw double-read** (`cost.rs` from `CostLaw.cpp:68-69`): `CostLawThresh{Speech,NoSpeech}`
  is read TWICE -- clamped (`1e-6..1-1e-6`) for the law coefficients, then re-read RAW/unclamped
  (defaults `10.0`/`-1.0`) for the runtime branch predicate. Almost certainly an unintended
  double-read; load-bearing for the sweep goldens. *Fix candidate:* unify to a single, clamped
  threshold after parity.
- **[0b-i] LogLaw cost/deriv Adim asymmetry** (`cost.rs` from `CostLaw.h:119-149`): `cost()` divides
  `y` by `Adim` before clamp+log; `deriv()` clamps the RAW `y` (not divided) then returns `A/y`. The
  derivative is inconsistent with the cost -- a genuine legacy bug. *Fix candidate:* make `deriv` the
  true derivative of `cost` after parity (will shift training gradients slightly).
- **[0b-i] AboveThreshCubic name-dependent coefficients** (`cost.rs` from `CostLaw.h:84-117`): the
  above-threshold cubic law switches its `A`/`B` coefficient formulas on the law-name STRING
  (`square`/`cubic` vs `linear`/`log`). Fragile and surprising. *Fix candidate:* refactor to explicit
  per-law types after parity.
- **[0b-i] Softmax `compute_deltas` ponderation-scaling asymmetry** (`cost.rs` `compute_deltas`, from
  `CostLaw.cpp:358-419`): the WER path scales EACH element by its own class's ponderation
  (`_ClassesPonderations(0,kk)`, per `kk`); the non-WER path instead captures a single ponderation
  from the frame's on-class column and scales the WHOLE frame row by it (`deltas.row(jj) *=
  ponderation`, `CostLaw.cpp:417`). The two paths disagree on granularity for no reason apparent in
  the math; reproduced verbatim (see `cost.rs` doc-comment on `compute_deltas`). *Fix candidate:*
  make the non-WER path scale per-element like the WER path, after parity.
- **[0b-i] `suppress_short` no-advance-after-erase** (`src/rust/src/tasks/segmentation.rs`
  `suppress_short`, from `Segmentation.cpp:245-282`): after any erase branch, the loop does NOT
  advance `i` -- the erased slot shifts the next segment into the current index, which must be
  re-tested (a naive `i += 1` would silently drop a merged-in short segment). Confirmed INTENDED:
  the dedicated `suppress_short_no_advance_after_erase` test locks this behavior in. *Why deferred:*
  load-bearing for legacy parity -- changing the advance semantics would change which segments get
  merged/split on real goldens. *Fix candidate:* once end-to-end parity holds, confirm/simplify the
  re-test-after-merge control flow (e.g. an explicit worklist) instead of the implicit no-advance
  loop.

- **[0b-ii] VRCTS writer byte-golden deferred; fixtures are external `vrcts_part` reference input**
  (`segmentation_io.rs` `to_vrcts_string`/`write_vrcts`, from `Segmentation.cpp:543-588`): the engine's
  `toFile_VRCTS` formats `stime`/`etime` with `iof::fmtr("%f.4s")` and `sigdur`/`spdur`/`dur` with
  `%f.2s`. `iof::fmtr`'s `%f.Ns` was resolved (by disassembling the vendored `Debug/bin/fsp` -- the `iof`
  headers are not on disk) to be exactly `std::fixed` + `precision(N)`, i.e. `%.Nf`; the fixture's own
  `sigdur="120.00"` independently confirms `%f.2s` -> 2 decimals. So the faithful writer emits
  **4-decimal** segment times. The committed fixtures (`vrcts_{1seg,empty,16seg}.xml`) instead show
  **3-decimal** times and a `spdur` (16seg: `81.79`) that is NOT the sum of their own printed durations
  (`81.772`): they are external `vrcts_part` output used as engine INPUT via `load_from_vrcts` (which
  reads only `SpeechSegment` times and ignores `spdur`), not `toFile_VRCTS` output. Legacy `spdur` is the
  sum of post-`sanitize` (4-decimal round-half-away, same-type-merged) SPEECH durations over the engine's
  full-precision internal boundaries -- unreproducible from the fixtures' rounded strings. *Validation:*
  `load_vrcts` is golden-tested for load correctness against the 3 fixtures; `to_vrcts_string` is
  unit-tested on a constructed `Segmentation` with hand-computed 4-decimal bytes. *Deferred:* the
  writer-vs-real-engine byte golden waits on the end-to-end inference-output parity milestone (README
  Roadmap Phase 4), when a real engine-produced `.xml` exists to match. *Minor:* our writer sanitizes a
  clone (non-mutating) whereas `toFile_VRCTS` mutates its `_Classification` in place; revisit if any
  caller relies on the write-time sanitize side effect.

- **[0b-ii] `compute_errors` Pass 1b reads one past the ref End sentinel (UB)** --
  `Segmentation::compute_errors` (`Segmentation.cpp:370-382`) sets
  `length = ((long)((itEndRef-1).begin)) / timeStep + 1` (the `(long)` cast is unary and binds to
  `.begin`, truncating the duration to whole seconds BEFORE dividing by `timeStep`; the port
  reproduces this via `begin.trunc()` ahead of the division), so the final `ii` makes `currentTime` land
  exactly on the last ref boundary (= `audioDuration`); the advance loop `(itRef+1 != itEndRef) && ...`
  then walks `itRef` to the End sentinel (`itEndRef-1`), after which line 382
  `durationSeg = (itRef+1).begin - itRef.begin` dereferences the **past-the-end** iterator `itEndRef`
  -- undefined behavior in C++, reading adjacent deque memory. `durationSeg` is only USED in the
  Speech/Substitution and Insertion coverage branches; when `itRef` is at End the flow falls into the
  final `else` (delay-only) branch and `durationSeg` is discarded. *Port:* our Pass 1b defers the
  `it_ref+1` read into those two branches (where `it_ref+1` is always in bounds), giving the identical
  result without the OOB panic. This is a **bounds guard beyond the legacy** (behavior-preserving, not a
  numeric change). *Revisit:* nothing to fix numerically; the guard can stay -- it merely removes UB the
  legacy relied on by luck. Related open item: the Pass-2 `(itRef+1)` / `(it+1)` accesses at
  `Segmentation.cpp:430-446` are guarded in the legacy only by the `!= END` scorable predicate (the
  sentinel makes them unreachable); the port relies on the same predicate rather than adding a redundant
  bounds check.

- **[0b-ii] `update_segmentation` tail uses `results.size()`, not the row `length`** (`segmenter.rs`
  `update_segmentation` tail, from `Segmenter.cpp:833-835`): the legacy TAIL sets `endSegment =
  timeStep*results.size()`, where `results.size()` is `rows*cols` -- correct ONLY for a pure row vector
  (the real SAD input), and wrong for any multi-row matrix. Our port takes a 1-D `results_row: &[f64]`,
  so `results.len()` equals the intended column count and the quirk is structurally avoided for the
  row-vector input; reproduced faithfully because SAD always feeds a single row (this ties to the
  row-vector `update_segmentation` vs col-vector `lid_to_segmentation` orientation asymmetry -- see spec
  3.2/3.3, also load-bearing). *Fix candidate:* if `update_segmentation` is ever fed a 2-D buffer once
  features/NN land, use the column count, not the element count.

- **[phase1] `read_audio` clip length is `round(rate*max_duration+1)`, one frame past the naive
  count** (`audio.rs` `read_audio`, from `AudioStruct.cpp:63`): `frame_count_max = (long long)
  boost::math::round(_Framerate*_DurationMax+1)`, i.e. the `+1` is INSIDE the rounded expression, not
  a post-hoc off-by-one guard -- a `max_duration` chosen to land exactly on a frame boundary still
  gets one extra frame. Reproduced verbatim (`frame_count_max = (sample_rate as f64 *
  max_duration_sec + 1.0).round() as i64`). *Fix candidate:* none needed; flagged because a naive
  re-derivation (`round(rate*dur)` then `+1`) would usually agree but can differ by a rounding
  boundary from the legacy's single combined round.
- **[phase1] `applyPreemph`/`apply_preemph` gate is whole-function, not per-sample** (`audio.rs`
  `Audio::apply_preemph`, from `AudioStruct.cpp:446-454`): the ENTIRE backward difference loop is
  wrapped in one `if (preemphRatio > 0)` -- `ratio <= 0` (including the legacy's own negative-ratio
  config convention seen in some configs) is a total no-op, not a per-sample branch or an
  absolute-value application. Reproduced verbatim. *Fix candidate:* none; flagged only because the
  gate is easy to miss when reading the loop body in isolation (it has no guard of its own).
- **[phase1] `windowing_coefficients` "uniform" is only materialized when `normalized`; otherwise
  it silently falls through to no-window (rectangular)** (`audio.rs` `windowing_coefficients`, from
  `Helpers.hpp:222-272`): the legacy `else if ((normalized)&&(windowing_type.compare("uniform")==0))`
  branch means `"uniform"` combined with `normalized == false` matches NEITHER that branch nor any
  named-window branch, so it falls to the final `else { windowing_coeff = Eigen::MatrixXd(); }` --
  an empty matrix, i.e. functionally identical to `"none"`. The port returns `None` for that same
  combination. *Fix candidate:* after parity, either make unnormalized uniform genuinely fill 1.0s
  (true rectangular window) or document the empty-matrix fallback as the intended behavior; a naive
  re-implementation would likely "fix" this silently and change every windowed-feature golden that
  uses `uniform` unnormalized.
- **[phase1] Two-real periodogram: only the first `n` of the caller's `n+1`-wide window buffer is
  packed; the DC bin divides by `n`, AC bins by `4n`** (`features/fft.rs`
  `compute_two_real_periodogram`, from `AudioStruct.cpp:487-510`): `getSequence`/`get_sequence`
  fills a `full_signal_window_size = window_size+1`-wide buffer, but
  `computeTwoRealPeriodogram`/`compute_two_real_periodogram` only reads indices `0..full_window_size`
  (`n = window_size`) when packing the two real signals into the complex FFT input -- the buffer's
  LAST sample is silently dropped every call. Separately, the unpacked periodogram normalizes the DC
  bin (`ii=0`) by `full_window_size` (`n`) while every other bin (`ii=2..=n`) normalizes by
  `coeff_norm = 2*size_1 = 4*full_window_size` (`4n`) -- a 4x scale discontinuity between the DC bin
  and the rest of the spectrum that is NOT a textbook periodogram normalization. Both reproduced
  verbatim (pinned by the periodogram goldens). *Fix candidate:* after parity, either pack all
  `n+1` samples (changing the FFT window content) or normalize DC consistently with the AC bins
  (`/4n`); both are retrain-affecting.
- **[phase1] Odd frame-count tail writes periodogram row twice, second write (P2 formula) silently
  clobbers the first (P1 formula)** (`audio.rs` `compute_segment_periodogram_estimates`, from
  `AudioStruct.cpp:551-559`): when `periodogram_frameNb` is odd, the tail packs the SAME frame as
  both "signal 1" and "signal 2" into one FFT call with `periodogram_column_1 == periodogram_column_2
  == currentFrame` -- so `computeTwoRealPeriodogram` writes that row via the P1 formula, then
  immediately overwrites it via the P2 formula (the two formulas are algebraically different
  quadratic combinations of the same FFT output, not equal in general). Only the final (P2) value
  survives; the P1 computation is pure waste. Reproduced verbatim (the port's odd-tail path performs
  the identical same-row double write). *Fix candidate:* after parity, skip the redundant P1 write
  for the self-paired tail frame (pure perf cleanup, the surviving value is unchanged).
- **[phase1] `get_sequence` left edge does not zero-pad; `index >= frames` is a silent no-op that
  leaves the caller's buffer holding the PREVIOUS frame's contents** (`audio.rs` `get_sequence`, from
  `AudioStruct.cpp:456-485`): (a) the outer guard `index < _FramesCount` wraps the entire body --
  when false, the function returns having touched nothing, so the framing driver's single reused
  buffer pair (`compute_segment_periodogram_estimates`) silently reprocesses stale data from an
  earlier call; (b) on the LEFT edge (`index < half_window`) only the in-range block-copy region is
  overwritten -- the left pad positions keep whatever was in the buffer before (never zeroed), unlike
  the RIGHT edge which zeros the whole buffer first; (c) the DC-offset mean (when enabled) is taken
  over the FULL `2*half_window+1` buffer INCLUDING any such stale/pad region, so a left-edge or
  no-op frame's DC-offset subtraction is computed over garbage-adjacent data by design. All three
  reproduced verbatim via the caller-owned reused-buffer contract (pinned by left-edge/right-edge
  hand tests plus real-audio dumps whose first/last frames exercise both paths). *Fix candidate:*
  after parity, zero the left pad explicitly and/or exclude pad positions from the DC-offset mean;
  both are retrain-affecting for segments near file boundaries.
- **[phase1] Convolution edge truncation drops taps without renormalizing** (`audio.rs`
  `convolution_horiz`/`convolution_vert`/`conv_taps`, from `Helpers.hpp:164-220`): near either edge
  of the row/column being convolved, the kernel is truncated to the taps that land in-bounds -- the
  missing taps are simply omitted from the sum, not redistributed or renormalized against the
  reduced tap weight. So edge outputs are systematically attenuated relative to interior outputs
  whenever the kernel doesn't sum to a shape-invariant constant at the truncation point. Reproduced
  verbatim (pinned by the windowing/temporal-convolution goldens). *Fix candidate:* after parity,
  renormalize by the sum of the taps actually used (or mirror/replicate-pad instead of truncating);
  retrain-affecting near segment edges.
- **[phase1] GFFT twiddles: Taylor-seed constants + Numerical-Recipes running recurrence**
  (`features/fft.rs` `sin_series`/`danielson_lanczos`, from `fft.hpp` -- V. Myrnyy, DDJ 2007): the
  butterfly twiddle factors are NOT `libm sin`/`cos`. Each stage seeds `wpr = -2*Sin(N,1)^2`,
  `wpi = -Sin(N,2)` where `Sin(B,A)` is a 16-term truncated Taylor series (`SinCosSeries<2,34>`,
  `S(M) = 1 - ((x*x)/M)/(M+1)*S(M+2)`, base `S(34)=1`), then advances `wr`/`wi` by the NR running
  recurrence `wr += wr*wpr - wi*wpi; wi += wi*wpr + wtemp*wpi;` which ACCUMULATES rounding error across
  the loop. In the legacy these are compile-time template constants; the port computes them at runtime
  with identical f64 operation order (which reproduces gcc strict-mode folding bit-for-bit). A modern
  port would use precise `sin`/`cos` and a non-accumulating twiddle table -- more accurate, but the
  goldens are pinned to this exact (lossy) scheme. *Fix candidate:* after end-to-end parity, swap to a
  precomputed twiddle table using `f64::sin`/`cos`; expect small per-bin numeric drift vs the legacy
  spectrum. Note the DL<4> base case is a hand-fused `-i` butterfly whose STATEMENT ORDER is
  load-bearing (ported verbatim), and the scramble is the 1-based NR bit-reversal.

- **[phase1] Mel deltas-no-DCT output is `[delta | delta | delta-delta]`, static log-mel OVERWRITTEN**
  (`features/mel.rs` `apply_deltas_overwrite`, from `MelFilterBank.cpp:192-193`): when `!DCT &&
  deltasNb > 0`, after computing the delta block (cols `[F,2F)`) and delta-delta block (cols `[2F,3F)`),
  the final two statements do `melPeriodogram = MFCC;` then `melPeriodogram.leftCols(F) =
  MFCC.leftCols(2F).rightCols(F)` -- i.e. the static log-mel block (cols `[0,F)`) is CLOBBERED by a copy
  of the delta block, so the emitted layout is `[delta | delta | delta-delta]`, not
  `[static | delta | delta-delta]`. Almost certainly an off-by-one authoring slip (the static features
  are silently discarded), but it is the golden contract (pinned by `logmel_deltas_chan1.bin`). *Fix
  candidate:* after end-to-end parity, keep the static block (`[static | delta | delta-delta]`); expect
  the NN input dimensionality to be unchanged but the first `F` columns to carry different (real static)
  values -- a retrain-affecting change.

- **[phase1] Mel `adim` denominator includes the clamped `j` when `T <= j`** (`features/mel.rs`
  `regression_deltas`, from `MelFilterBank.cpp:153-170`): the regression-deltas normalizer `adim += j*j`
  runs for every `j` in `1..=n` EVEN when `j > T` (rows) and the shifted-difference copy is skipped
  (`length = min(j, T)`). So on short segments the denominator `2*sum j^2` counts terms whose shifted
  contribution is fully edge-clamped rather than a true `t+/-j` difference. Reproduced exactly (the
  clamped closed form `D[t] = sum_j j*(B[min(t+j,T-1)] - B[max(t-j,0)]) / (2 sum j^2)` matches the
  block-op sequence bit-for-bit, verified against the numpy oracle across `T<=n` shapes). *Fix
  candidate:* none needed numerically; the behavior is self-consistent. Flagged only because it is a
  non-obvious edge that a naive re-derivation (denominator over unclamped active `j` only) would get
  wrong.

- **[phase1] Mel whole-bank fallback: ANY empty filter collapses the ENTIRE bank to raw-band
  pass-through** (`features/mel.rs` `new`, from `MelFilterBank.cpp:92-103`): if a single triangle
  collects zero periodogram bins (`nb_bins` too large for the spectrum resolution), the ctor sets
  `_IsMel = false` and `_NbFilters = _EndFreq - _BegFreq + 1` -- the mel projection is abandoned for the
  whole bank and `applyFilterBank` falls to a raw-band copy/log. The partially built `_IndexBegin` /
  `_Coeffs` are retained but never used. Reproduced faithfully (pinned by the
  `empty_filter_collapses_whole_bank_to_passthrough` test). *Fix candidate:* after parity, either clamp
  `nb_bins` to what the resolution supports or drop only the empty filters instead of the whole bank.

- **[phase1] SDC `ignoreFirst` clobbers the LAST static column, not c0** (`features/mel.rs` `apply_dct`
  Branch A, from `MelFilterBank.cpp:262`): unlike the deltas>=0 `ignoreFirst` path (Branch B) which drops
  the FIRST static (c0), the SDC path writes the `nb_dct` statics first, then assigns the `7*nb_dct`-wide
  SDC band into `MFCC.rightCols(7*nb_dct)`. When `ignoreFirst` the output width is `nb_dct-1 + 7*nb_dct`,
  so `rightCols` starts at column `nb_dct-1` and OVERWRITES the last static (`c_{nb_dct-1}`); c0..c_{nb_dct-2}
  survive. So the discarded static is the highest-order cepstral coeff, not the energy term -- almost
  certainly an authoring slip (the width bookkeeping subtracts 1 for "ignore first" but the write order
  clobbers the last). Pinned by `mfcc_sdc_if_chan1.bin` (c12 == SDC block-0 col-0). *Fix candidate:* after
  parity, either genuinely drop c0 (shift the SDC write right by one) or keep all statics; retrain-affecting.

- **[phase1] DCT `melPeriodogram*_CoeffsDCT` Eigen GEMM replaced by an explicit ascending triple loop**
  (`features/mel.rs` `apply_dct`/`dct_product`, from `MelFilterBank.cpp:223` etc.): the legacy computes the
  DCT projection as an Eigen matrix-matrix product on the full `T x nb_filters` log-mel. The mandatory
  Task-7 harness pre-check (Eigen GEMM vs `for t / for k / accumulate over n ascending`) DIVERGED: Eigen's
  blocked `gebp` kernel does not accumulate in ascending order even under `-DEIGEN_DONT_VECTORIZE`
  `-ffp-contract=off` (e.g. `E(0,0)` bits `...e6ea` vs ascending `...e6ec`; ~1500/2613 elements differ on
  the 201x29 * 29x13 product). Per the brief the product is done as the explicit ascending triple loop, and
  the goldens are dumped with THAT order (harness `dctProduct`), because Eigen's GEMM order is not portable
  across BLAS/arch/Eigen versions. Recorded in `manifest.json:dct_gemm_substitution`. *Fix candidate:* none
  -- the ascending loop is the deliberate portable parity target; do NOT swap in a BLAS `.dot()`.

- **[phase1] LTSV `classifySequence` returns variance, not "standard deviation"** (`features/ltsv_tdc.rs`
  `ltsv_classify_sequence`, from `LongTermSpectralVariation.cpp:118-127`): the legacy names the
  accumulator `std_dzeta`, comments the block "Compute standard deviation", and returns
  `sum((dzeta-mean_dzeta)^2)/nbBins` -- there is no `sqrt`. So the LTSV "score" is a biased variance in
  units squared, not a standard deviation; every downstream threshold tuned against this score is
  implicitly tuned against variance, not std. Reproduced exactly (pinned by `ltsv_is_biased_variance_
  no_sqrt` and the harness goldens `ltsv_chan1.bin`/`ltsv_synth.bin`). *Fix candidate:* after parity,
  either take the sqrt (changing the score's scale/units and every tuned threshold) or rename to stop
  calling it a standard deviation; retrain/re-tune-affecting either way.

- **[phase1] LTSV `ltsv_classify_sequence` mean-only 1e-12 floor quirk** (`features/ltsv_tdc.rs`
  `ltsv_classify_sequence`, from `LongTermSpectralVariation.cpp:101`): the 1e-12 epsilon floor is
  applied ONLY to the per-bin window MEAN, never to the numerator `P(t,bin)` in the ratio `r = P/mean`.
  The legacy guard `if (mean_value[row] < 1e-12) mean_value[row] = 1e-12;` prevents division by zero
  (the across-bins `mean_dzeta` is never floored); a naive
  re-derivation that floors the ratio `r` itself or the numerator `P` diverges from the goldens. Both
  the unprotected numerator path and the protected-mean ratio are reproduced verbatim (pinned by
  `ltsv_chan1.bin`/`ltsv_synth.bin`). *Fix candidate:* after parity, revisit whether a symmetric epsilon
  floor on the ratio (or numerator) is preferable to the asymmetric mean-only floor.

- **[phase1] `computePitch` returns a `long/long` INTEGER-divided pitch estimate** (`features/ltsv_tdc.rs`
  `compute_pitch`, from `BLSTMSpectralSegmenter.cpp:172-192`): the legacy signature is `computePitch(long
  min_lag, long max_lag, long frameRate, ...)` and it returns `frameRate/indiceMaxPeak` -- both `long`, so
  the division TRUNCATES (8000/35 = 228, not 228.571...) and is widened to `double` only on return. The
  accept bounds in `getPitch` (`:425`) are genuine double divisions, so the acceptance band is exact while
  the accepted estimates are quantized to integers. Every `getPitch` average (and hence the homothety
  `pitch/300` coefficient) is built from truncated estimates. Pinned by `pitch_chan1.bin` (229.9010989...
  = 20921/91, an integer sum over 91 accepted frames). *Fix candidate:* after parity, divide in `f64`
  (`rate / argmax_lag as f64`); expect the pitch estimate, homothety coefficient, and all downstream
  warped-spectrum features to shift slightly.

- **[phase1] `fmath::log` has no `x <= 0` guard; `log(0) = -88.0297f` finite** (`features/ltsv_tdc.rs`
  `fmath_log`, from `fmath.hpp:713-727`): the bit-trick table log decodes the f32 bit pattern with no
  domain check, so `fmath_log(0.0)` returns the finite `(0 - 127<<23)*c_log2 + app[0]` = -88.029694
  instead of `-inf`; the sign bit is IGNORED by all three masks, so `fmath_log(-x) == fmath_log(x)`
  (a finite wrong value, not NaN). The
  TDC score relies on this: `-fmath_log(1-MaxPeak)` at `MaxPeak == 1` (min_lag=0, or a perfectly
  periodic window) yields the finite +88.03-scaled score rather than +inf. Pinned by
  `fmath_log_sweep.bin` (x=0 entry) and `tdc_r0_is_one_when_min_lag_zero_and_score_finite`. *Fix
  candidate:* after parity, swap to `f32::ln` (or guard the domain); every tuned TDC threshold moves.

- **[phase1] TDC cross-correlation `mm=0` term double-counts, hence `/2`; crossing polarity mixes
  strict/non-strict** (`features/ltsv_tdc.rs` `tdc_classify_sequence`, from `TimeDomainCorrel.cpp:59-86`):
  (a) the shifted cross-correlation `xcor(mm) = sum_nn R[p1+nn]*R[p2+nn+mm] + R[p1+mm+nn]*R[p2+nn]` counts
  the same product twice at `mm=0`, which the `CrossCorr += tmp_xcorr/2` halving only exactly compensates
  at `mm=0` (for `mm>0` the two orders are genuinely different shifts, so `/2` averages them); (b) the
  zero-crossing predicate is keyed on `R[0]`'s sign (R at MIN lag, not lag 0) with asymmetric bounds --
  `R[ll] > 0 && R[ll+1] <= 0` for the positive branch vs `R[ll] < 0 && R[ll+1] >= 0` for the negative --
  so a sample landing exactly on 0.0 counts as a crossing in both branches but the entry condition
  differs in strictness. Also `count` is a `double`. All reproduced verbatim (pinned by `tdc_chan1.bin`
  and the `tdc_cases.json` cross-language check). *Fix candidate:* after parity, define the crossing
  predicate symmetrically and normalize the `mm=0` self-term; re-tune-affecting.

- **[phase1] `(1 - MaxPeak)` narrowed to f32 before `fmath::log`, widened back** (`features/ltsv_tdc.rs`
  `tdc_classify_sequence` return, from `TimeDomainCorrel.cpp:90`): the TDC score's peak term computes
  `1-MaxPeak` in f64, passes it through the f32-only `fmath::log`, and widens the f32 result back into
  the f64 balance blend -- two precision cliffs in the hot score path. Reproduced exactly. *Fix
  candidate:* subsumed by the `fmath::log -> f32::ln` swap above.

- **[phase1] `InputStatistics::from_matrix` has no `n == 0` guard; empty batch yields NaN**
  (`features/stats.rs` `InputStatistics::from_matrix`, from `InputStatistics.cpp:8-16`): both the mean
  and std finalization divide by `_NbOfValues` unconditionally; a zero-row input therefore produces
  `0.0/0 = NaN` mean/std rather than a guarded zero or an error. Reproduced as-is -- callers must not
  batch zero rows if they want a finite result (the merge path's `update()` handles the actual "no data
  yet" case via the `n == 0` copy branch, not via a from-empty-matrix batch). *Fix candidate:* after
  parity, either guard `n == 0` in `from_matrix` or make it return `Option`/`Result`.

- **[phase1] `InputStatistics::update` merge re-squares the STORED std (not variance) in a pinned
  op order** (`features/stats.rs` `InputStatistics::update`, from `InputStatistics.cpp:39-46`): the
  accumulator persists `std` (already sqrt'd), so every merge must square each side's std back to a
  variance-like quantity, add the squared mean-shift term, scale by that side's sample count, sum both
  sides, divide by the merged count, then `sqrt` once -- in exactly that per-element order (not
  algebraically reassociated, e.g. not `sqrt(n1)*std1` distributed differently). Because floating-point
  pooling is not exactly associative, merging in chunks generally produces DIFFERENT bits than a single
  monolithic batch over the same rows (pinned by the oracle module docstring in `features_oracle.py`
  and the Rust cross-check `stats_merge_oracle_cross_check_bit_exact`, which asserts the FORMULA, not
  batch-vs-merge bit-equality). Reproduced exactly. *Fix candidate:* none -- this is inherent to
  incremental variance/std pooling, not a bug to fix.

- **[phase1] `_LTSVWindowShift != 0.0` guards the LTSVshift read on an UNINITIALIZED member (UB); guard
  dropped** (`features/pipeline.rs` `FeatureConfig::from_legacy`, from `BLSTMSpectralSegmenter.cpp:50`):
  the legacy reads `_LTSVWindowShift` from config only `if (_LTSVWindowShift != 0.0)`, but at that point
  `_LTSVWindowShift` is a default-constructed `double` member with no in-class initializer and an empty
  ctor body -- so the branch condition reads an INDETERMINATE value (undefined behavior). The port drops
  the UB guard and reads `LTSVshift` unconditionally when the key is present (the oracle harness does the
  same, so the golden stays valid; all four variant configs supply `LTSVshift` explicitly). *Fix
  candidate:* after parity, either give `_LTSVWindowShift` a defined default or make the read
  unconditional in the legacy (already the effective behavior here).

- **[phase1] LTSV frequency band is RESET to `(0, output_dim-1)` for mel variants, diverging from the
  spectral band** (E2E composition: `src/rust/tests/phase1_pipeline_golden.rs` + the harness twin
  `tools/oracle_harness/main.cpp`, from `BLSTMSpectralSegmenter.cpp:270-271` inside the
  `nb_bins > 0` block): `initSpectralAnalysis` reuses the `freq_beg`/`freq_end` out-params, resetting them
  to `(0, periodogram_length-1)` where `periodogram_length` becomes `nbFilters` (or `nbDCT` when DCT is
  active). `getLTSV` then runs over the RAW periodogram's first `output_dim` bins, NOT the spectral band
  `[freq_beg, freq_end]` used by the raw-band assembly path. So for mel/DCT variants the LTSV score
  covers periodogram bins `0..output_dim-1` (an arbitrary low-frequency prefix), while the raw-band
  variant keeps the true spectral band. Reproduced exactly (pinned by `inputseq_{mfcc_deltas,mfcc_sdc,
  logmel}_chan{1,2}.bin`, whose last column is the LTSV over the reset band). *Fix candidate:* after
  parity, compute LTSV over the spectral band regardless of mel/DCT, or make the band an explicit
  parameter rather than an aliased out-param.

- **[phase1] `get_pitch` DC-offset flag was hardcoded `false`; now config-driven** (`features/ltsv_tdc.rs`
  `get_pitch`, from `BLSTMSpectralSegmenter.cpp:423` `getSequence(..., _FlagDCOffset, ...)`): Task 9's
  `get_pitch` passed a hardcoded `dc_offset = false` to its `get_sequence` call; the legacy value is the
  config `_FlagDCOffset`. Task 11 added a `dc_offset: bool` parameter and wires the config flag through.
  The Task 9 pitch golden was dumped with `dc_offset = false`, so the golden stays valid (both sides pass
  `false` for that dump). Not a bug per se -- a wiring gap closed; noted for provenance.

- **[phase2] LSTM `feedForwardReverse` leaves the `_Gates`/`_CellStates`/`_CellsIn` caches in
  REVERSED-input order** (`nn/layers.rs` `feed_forward_reverse`, from `LSTMLayer.cpp:415-421`): the
  reverse pass is implemented as reverse-input -> `feedForward` -> reverse-output; only `outputSeq` is
  un-reversed, so after a reverse pass the caches (which the backward pass reads) correspond to the
  reversed sequence, NOT the natural time order. Reproduced exactly: the Rust caches match, and the
  reverse gate goldens (`lstm_gates_*_rev.bin`) are dumped in reversed-input order to match. *Fix
  candidate:* none needed -- the Task 5+ backward pass must consume the caches in the same reversed
  order the legacy does; documented so the reverse-order cache state is not "fixed" into forward order.

- **[phase2] DEAD hand-unrolled `feedForwardReverse` variant with DIFFERENT peephole grouping (not
  ported)** (`LSTMLayer.cpp:423-516`, commented out): a second `feedForwardReverse` body exists fully
  commented out; it splits the two-term gates-peep expressions (`:362-363` in the live forward) into
  SEPARATE `+=` adds (e.g. `:464-465` add `P[4]` then `P[5]` in two statements) rather than the live
  code's single combined expression. Porting from it would change the FP accumulation order and break
  bit-exactness. The port follows the LIVE `feedForward` (`:312-413`) exclusively. Confirmed neutral by
  the harness NN_TOL probe: the ascending-loop reimpl matches the REAL compiled `LSTMLayer::feedForward`
  with `max_ulp=0` over the whole forward/reverse grid. *Fix candidate:* delete the dead block in a
  post-parity legacy cleanup.

- **[phase2] LSTM `feedForward` `lastLayer` parameter is UNUSED** (`nn/layers.rs` `feed_forward` /
  `feed_forward_reverse`, from `LSTMLayer.cpp:312,415`): the legacy signature carries `bool lastLayer`
  but the LSTM forward never reads it (it matters only for the output MLP `NeuronLayer`). Kept in the
  Rust signature for parity with the `Layer` dispatch and the legacy call sites; bound to `let _ =`.
  Provenance note, not a bug.

- **[phase2] `NeuronLayer` output softmax is UNSTABILIZED -- no max-subtraction overflow guard**
  (`nn/layers.rs` `NeuronLayer::feed_forward`, from `NeuronLayer.cpp:138-142`): `lastLayer && O>1`
  computes `exp(a+b)` directly on the raw pre-activation values with no `- max(row)` shift before the
  exponential, unlike a numerically-stabilized softmax. Large pre-activations can overflow `exp` to
  `inf` (and `inf/inf = NaN` in the row-sum quotient); the legacy has no guard against this and the port
  reproduces it exactly, incl. the row-sum being a SEQUENTIAL per-row loop over ascending columns
  (`:139`, not a reduction) followed by a per-COLUMN `cwiseQuotient` (`:140-142`). Confirmed neutral by
  the harness NN_TOL probe: the ascending-loop reimpl matches the REAL compiled
  `NeuronLayer::feedForward` with `max_ulp=0` over the whole dump grid. *Fix candidate:* after parity,
  add a `- rowwise().maxCoeff()` shift before the `exp` for numerical safety on unseen inputs with large
  activations; no overflow triggers on the committed fixtures or the real-net E2E run.

- **[phase2] `NeuralNetwork` copy constructor is ILL-FORMED C++ (never ported)** (`NeuralNetwork.hpp:39-51`):
  the copy ctor body writes `_NeuronNb(neuralNetwork._NeuronNb);` etc as *statements* -- calling
  `operator()` (element access) on already-constructed member vectors and discarding the result, NOT the
  member-initializer-list copies it was meant to be. It compiles only because the template is never
  copy-instantiated (dead code). The Rust port deliberately omits any `Clone`/copy path for `Network<L>`;
  callers rebuild via `new` + `set_weights`. Provenance note, not a bug to reproduce. *Fix candidate:* if a
  copy is ever needed, derive `Clone` correctly (the legacy intent was a deep copy that clears the
  `_LayersOutput`/`_LayersOutputErrors` caches).

- **[phase2] `NeuralNetwork::SubSample` DROPS the trailing `T mod R` frames (integer floor)**
  (`nn/network.rs::sub_sample`, from `NeuralNetwork.hpp:125-134`): the sub-sampled length is
  `Input.rows()/subSamplingRatio` (integer floor), so on a T not divisible by R the last `T mod R` input
  rows are silently discarded -- e.g. T=11, R=2 -> 5 output rows, frame 10 dropped. Load-bearing for the
  container output row count (`floor(T/ratio)`) and pinned by the T=11 goldens (odd length). Reproduced
  exactly; NOT a bug to fix (the recurrent nets tolerate the boundary loss). Note the nested-floor
  identity: applying `SubSample(a)` then `SubSample(b)` equals `SubSample(a*b)` in row count
  (`floor(floor(T/a)/b) == floor(T/(a*b))`), so the legacy's SEQUENTIAL per-layer division is not a
  distinguishable choice from a single product division -- ported as the legacy writes it (sequential),
  documented here rather than tested against a phantom counterexample.

- **[phase2] `feedForwardBackwardTruncateSweep` SILENTLY DROPS a trailing chunk with `lengthShort == 0`**
  (`nn/blstm.rs::feed_forward_backward_truncate_sweep`, from `BLSTMNeuralNetwork.cpp:534-542`): the
  non-overlapping window loop advances `jj += window_size`; the partial LAST window recomputes its
  post-subsampling length by SEQUENTIAL floors (LSTM ratios then Output ratios). When the trailing chunk
  is shorter than the subsampling ratio (e.g. a 1-row tail with LSTM ratio 2 -> `1/2 = 0`), the
  `if (lengthShort > 0)` guard skips it entirely -- the corresponding `outputSeq` rows keep the CALLER's
  prior contents (never written). Pinned by the T=19 window-6 golden (`blstm_truncate_out.bin`, row 9
  retains its pre-seed). Load-bearing: the caller must pre-zero (or otherwise initialize) `outputSeq` or
  the dropped rows carry garbage. Reproduced, no guard. *Fix candidate:* after parity, either process the
  short tail at reduced length or explicitly zero the dropped rows.

- **[phase2] TwoSweeps makes `_OutputForward`/`_OutputBackward` DOUBLE-WIDTH** (`nn/blstm.rs::
  feed_forward_backward_truncate`, from `BLSTMNeuralNetwork.cpp:583-586`): the two-sweeps branch runs a
  front-padded sweep and a shift-offset sweep, then HCATs each sweep's hidden-state window into the member
  matrices, so `_OutputForward` comes out `rows x 2*lstmOut` instead of `rows x lstmOut`. The
  `(window_size/2)/ratio` integer division ORDER (`window_size/2` FIRST) is load-bearing for `shiftShort`;
  sweep 1 drops the FRONT padding but KEEPS the back padding. Pinned by `blstm_twosweeps_fwd.bin` (10x4 for
  a 2-wide LSTM). Provenance quirk (a downstream consumer that assumes single-width would misread it); not
  a bug to fix pre-parity.

- **[phase2] `feedForwardBackwardOverLap` divides `0/0 -> NaN` on rows no window covers**
  (`nn/blstm.rs::feed_forward_backward_overlap`, from `BLSTMNeuralNetwork.cpp:674-679`): output + hidden
  states are accumulated per row and then divided by per-row hit COUNTS; a `window_shift > 2*window_size+1`
  leaves gap rows with count 0, so the final `cwiseQuotient` computes `0.0/0.0 = NaN` with no guard. Also
  the count-quotient loop bound is `_OutputForward.cols()` for BOTH the forward and the backward quotient
  (assumes equal widths). Window bounds are SNAPPED to the subsampling grid (begin down via
  `(begin/ratio)*ratio`, end up via `((end+1)/ratio)*ratio-1`). Pinned by `blstm_overlap_nan_out.bin`
  (rows 6-9 NaN). Reproduced exactly, no guard. *Fix candidate:* after parity, guard uncovered rows (leave
  them zero or carry the caller's value) instead of emitting NaN.

- **[phase2] `feedForwardBackwardMLPOverLap` IGNORES the passed `window_size`** (`nn/blstm.rs::
  feed_forward_backward_mlp_overlap`, from `BLSTMNeuralNetwork.cpp:686`): the method's first act is
  `window_size = getSubSamplingRatio()/2`, discarding whatever the caller passed; `length_short` is a hard
  `1`. Only EXACT windows (`length_seq == 2*window_size+1`) are processed -- edge windows are skipped, so
  those `outputSeq` rows keep the caller's contents -- and `_OutputForward`/`_OutputBackward` are EMPTIED
  at the end. Pinned by `blstm_mlpoverlap_out.bin` (row 0 edge-skipped -> stays 0) and a test that calls
  with a deliberately-wrong `window_size` and asserts the identical output. Provenance quirk; not a bug.

- **[phase2] Real-config feature width (11) does not match the trained net's input (23)** (E2E gate,
  `BLSTMNeuralNetwork.cpp:428-434` feedForward width tolerance): the REAL `1_worker_1.config` DSP keys
  (`nb_DCT 4`, `IgnoreFirstDCT true`, `ComputeDeltasNb 5`, `ComputeDeltaDeltasNb 3`) produce a `201 x 11`
  input sequence (`3*nb_DCT - 1 = 11`; `LTSVwindow 0` -> no LTSV column), but the net's `LSTMNeuronNb[0]`
  is 23. The legacy feeds the mismatched-width input anyway: the `leftCols(inputSize)` crop only fires when
  `LSTMRatios[0] > 1 && netInput < inputCols` (`11 < 23` is false, so no crop), and the per-layer input GEMM
  then silently uses `topRows(cols)` of the 23-row weight block -- i.e. the trained net consumes only the
  first 11 of its 23 input weights, leaving 12 rows of layer-0 input weights DEAD. This is almost certainly a
  stale-config artifact (the `.mat` net was trained for a 23-dim front-end that this `.config` no longer
  produces), but it is load-bearing for the golden: `e2e_out_full.bin` / `e2e_out_overlap.bin` are the real
  class's actual output under this mismatch. Pinned bit-exact by `phase2_e2e_gate.rs`. Reproduced exactly;
  *fix candidate:* once the training loop is ported, validate `feature_dim == net_input_size` at config load
  and error (or re-derive) instead of silently truncating.

- **[phase2] LSTM config-text weight branch writes only peephole rows 0-2; rows 3-11 UNINITIALIZED
  (UB); the port zero-initializes** (`LSTMLayer.cpp:21,53-124`): the ctor resizes `_PeepWeight` to
  `12 x _OutputSize` (:21) -- an Eigen resize, no zeroing -- but the per-block config-text branch
  (:53-124, taken when explicit weight keys are present rather than the 1e24 random sentinel) assigns
  ONLY rows 0/1/2 (:72, :89, :106, the cells-peephole rows). Rows 3-11 -- the gates/recurrent peephole
  rows the forward reads at :357-368 and :394 whenever `_isGatesPeepholesActive` /
  `_isGatesReccurentPeepholesActive` (both default true) -- stay INDETERMINATE, and
  `_PeepWeight /= adimCoeff` (:129) then divides the garbage. Only the random-table branch (:44-48)
  fills all 12 rows. The port deliberately deviates: `LstmLayer::new` zero-initializes every block and
  weights only ever arrive via the flat `set_weights` seam (the legacy `weightsSetExternally == true`
  path); the config-text weight-reading branch is NOT ported. Parity-neutral for the goldens (all
  fixtures set weights externally). *Fix candidate:* if the config-text branch is ever ported, zero
  rows 3-11 explicitly (the port's `new` already does).

- **[phase2] Config KEY spells "Recurrent" correctly; the C++ member is misspelled "Reccurent"**
  (`LSTMLayer.cpp:13`, `LSTMLayer.h:31`): `conf.get<bool>(prefix+"_IsGatesRecurrentPeepholesActive")`
  is stored in `_isGatesReccurentPeepholesActive`. Internally consistent (the misspelling never leaks
  into the config format), but a grep keyed on the member spelling misses every config-key site and
  vice versa -- an easy way to port the wrong default or miss a flag consumer. The port uses the
  correctly-spelled key (`config.rs`, `nn/blstm.rs`) and a conventional member name
  (`gates_rec_peep`). Provenance note, not a bug.

- **[phase2] `_MaxSaturation` config key is read and stored but ALL uses are commented out (dead)**
  (`LSTMLayer.cpp:10` read; every consumer commented out at :371-380, :401-408, :693, :888; same
  read in `SRNLayer.cpp:10` and `CWRNNLayer.cpp:10`): the ctor reads `prefix_MaxSaturation`
  (default -1) into `_MaxSaturation` and the copy ctor copies it (:142), but the forward saturation
  reset and the backward saturation counter that consume it are all commented out -- the key is
  config-surface dead weight. The port drops the key and the member entirely. *Fix candidate:*
  delete the key from the legacy config schema in a post-parity cleanup (or deliberately resurrect
  the saturation logic).

- **[phase2] `_NbOfSeqFedBackward` is never initialized by the MAIN ctor; the COPY ctor accidentally
  launders it** (`LSTMLayer.cpp:6-136` has no init vs the copy ctor's `= 0` at :155; same pattern in
  `SRNLayer.cpp:67`, `CWRNNLayer.cpp:84`, `NeuronLayer.cpp:66` -- only `ConvolutionalLayer.cpp:40`
  initializes it in the main ctor): the layer ctor leaves the backward-pass sequence counter
  indeterminate; it becomes 0 only because `NeuralNetwork.hpp:27` builds layers via
  `push_back(LayerType(...))` and the user-declared copy ctor (which suppresses the implicit move
  ctor) runs on insertion. A directly-constructed layer that fed backward would read/increment
  indeterminate memory (`getWeightsDerivatives` :260-290, `feedBackward` :720). The port initializes
  all state explicitly at construction (the counter itself lands with the Phase 3 derivative
  accumulators). *Fix candidate:* initialize the member in the legacy main ctor.

- **[phase2] `GatesFunction`/`Logistic` saturation guards: strict pass-through, boundary saturates;
  the gates guard tests the SCALED value** (`nn/activations.rs` `gates_fn`/`logistic_fn`, from
  `ActivationFunctions.h:229-238, 40-49`): both functions share the same guard shape --
  `arg < expLimit` outer, `arg > -expLimit` inner, else hard 1.0/0.0 -- so equality at the boundary
  saturates in BOTH (the spec's GatesFunction-EXCLUSIVE vs Logistic-INCLUSIVE labels, kept in the
  port docs for continuity, denote this same at-boundary saturation). The substantive quirks: (a)
  `GatesFunction` guards the pre-scaled `0.1*x`, so in x-space its saturation starts at
  `|x| >= 10*expLimit` (~7097.8), not `expLimit` (~709.78); (b) the hard 0.0 return at the negative
  boundary is OBSERVABLE, not just an overflow shield -- an unguarded sigmoid at `x == -expLimit`
  computes `1/(1+exp(+709.78))` with `exp` still finite (~1.8e308), yielding a subnormal ~5.6e-309
  instead of 0. Guard order ported exactly (pinned by the `phase2_activations_golden.rs` boundary
  tests). *Fix candidate:* none -- document-only; an unguarded or clamp-to-epsilon rewrite diverges
  at the negative boundary.

- **[phase2] LSTM `feedForward` t=0 cell update omits the forget term BY EXPRESSION SHAPE (no zero
  initial state)** (`nn/layers.rs` `feed_forward` t=0 branch, from `LSTMLayer.cpp:325-348` vs
  :351-411): the t=0 row is a structurally separate block -- `c_0 = i_0 .* g_0` (:337) with NO
  `+ c_prev .* f` term, no feedback GEMV, and no recurrent/gates peephole adds; the forget gate IS
  computed and activated at t=0 (:333) but its value is discarded. This is by-omission zero state,
  not a zeroed-previous-state multiply: materializing `c_-1 = 0` and running the general expression
  would compute `i*g + 0*f` -- a different FP expression (sign-of-zero and NaN propagation differ,
  e.g. `-0.0 + 0.0 = +0.0`). The port mirrors the two-branch structure exactly (pinned by the T=1
  and gate-cache goldens). *Fix candidate:* none -- provenance; the Phase 3 backward pass must
  mirror the same t=0 asymmetry.

- **[phase2] Layer input-width tolerance is BIDIRECTIONAL silent truncation, in the LSTM AND dense
  forwards** (`nn/layers.rs` `LstmLayer::feed_forward`/`NeuronLayer::feed_forward`, from
  `LSTMLayer.cpp:313-319` and `NeuronLayer.cpp:129-134`): a three-way branch tolerates ANY
  input-width mismatch silently. Input WIDER than the layer (`cols > I`): the extra input columns
  are DROPPED (`leftCols(I)`). Input NARROWER (`cols < I`): the weight matrix's bottom `I - cols`
  rows are silently DEAD (`_Weights.topRows(cols)`). No warning or error either way, in either
  layer type. The E2E entry above (real-config width 11 vs net input 23) is the narrow direction
  biting a real trained net; this entry records the general mechanism in both directions.
  Reproduced exactly (pinned by the width-tolerance unit tests). *Fix candidate:* covered by the
  E2E entry -- validate `feature_dim == net_input_size` at load after parity.

- **[phase2] OverLap accumulates `_Cost`/`_NbOfClassif` PER WINDOW: overlapped rows are counted
  once per covering window** (`nn/blstm.rs::feed_forward_backward_overlap` inner
  `feed_forward_backward_plain` calls, from `BLSTMNeuralNetwork.cpp:657` + :815-828): the windowed
  dispatch resets `_Cost = 0`/`_NbOfClassif = 0` once (:712-713), then the OverLap driver calls the
  plain `feedForwardBackward` per window and each inner call adds `computeCost` over the window's
  rows plus `_NbOfClassif += rows`. With `window_shift < 2*window_size+1` every overlapped output
  row contributes to the cost and the classification count once PER COVERING WINDOW, so the
  downstream per-classification normalization (cost/nbOfClassif) is over rows-times-coverage, not
  rows. Note the asymmetry with the OUTPUT itself, which IS divided by per-row hit counts (see the
  0/0 -> NaN entry above); the cost never is. Reproduced exactly -- the per-window accumulation is
  the ported control flow. *Fix candidate:* after parity, decide whether the cost should be
  coverage-normalized like the averaged output.

- **[phase2] `_TargetEnforcementStep < 0` DESTROYS the caller-visible output interior with -0.5
  before costing** (`nn/blstm.rs::feed_forward_backward` tail, from `BLSTMNeuralNetwork.cpp:
  815-828`): with targets present and `_TargetEnforcementStep < 0`, the final cost block copies the
  targets, overwrites BOTH the target copy's AND the caller-visible `outputSeq`'s interior rows
  `[1, rows-1)` with the constant -0.5, then computes the cost on the mutated pair -- the returned
  posteriors carry -0.5 in every interior row (only the two boundary rows survive the call) and the
  accumulated cost is measured against constants, not the net's output. Pinned by
  `cost_quirk_enforcement_neg_overwrites_output_interior_with_minus_half` (exact -0.5 bits + an
  expression-coded independent cost). Reproduced exactly. *Fix candidate:* after parity, operate on
  a local copy of the output (the in-place clobber looks like a debugging aid that shipped).

- **[phase2] `config.rs` flat packer implements ONLY the non-MLP layout (`LSTMNeuronNb[0] == 0`
  mode missing)** (`config.rs` `element_count`/`nnet_to_flat`/`flat_to_nnet`, vs
  `BLSTMNeuralNetwork.cpp:209-253`): the legacy `_IsMLP` mode (`LSTMNeuronNb[0] == 0`, :54-55)
  packs the `_OutputNetwork` weights only, with a mean/std tail of
  `2*_OutputNetwork.getInputSize()` (= `2*OutputNeuronNb[0]`); the Phase 0a MATLAB-seam packer in
  `config.rs` unconditionally lays out forward+backward LSTM blocks and sizes the tail
  `2*LSTMNeuronNb[0]` (`element_count`) -- an MLP config would pack a zero-length tail and phantom
  LSTM blocks. The RUNTIME seam (`nn/blstm.rs::{nb_of_weights,set_weights,get_weights}`) handles
  both modes correctly; only the config-domain packer has the gap, and no MLP golden config exists
  (the real configs are all BLSTM-mode). *Fix candidate:* add the `_IsMLP` branch to `config.rs`
  when the Python optimizer needs to drive an MLP config; until then the gap is latent.

- **[phase2] CNN is broken-as-committed: empty `_Layers` indexed on every real-config LID run (UB),
  weights orphaned, backward has no return** (`ConvolutionalNeuralNetwork.cpp`,
  `ConvolutionalLayer.cpp`, `TwinBLSTMSpectralLID.cpp` -- NOT ported, per the locked spec decision):
  (a) the ctor builds `_Layers` only when `_NeuronNb[0] > 0` (`ConvolutionalNeuralNetwork.cpp:13`),
  and `_NeuronNb` defaults to a single 0 (:8); but `feedForward`'s guard tests
  `_NeuronNb.size() > 0` (:265) -- true even when the CNN is disabled -- so `_Layers[0]` (:266) and
  `getOutputMatrix`'s `_Layers[_NeuronNb.size()-1]` (:228) index an EMPTY vector, and
  `TwinBLSTMSpectralLID.cpp:924-928` calls both UNCONDITIONALLY (`if (true)`) on every file: any
  real `lid.config` run without CNN keys executes undefined behavior (and would clobber `inputSeq`
  with the CNN output if it were live). (b) the CNN weights are ORPHANED: constructed with
  `weightsSetExternally == false` (`TwinBLSTMSpectralLID.cpp:23`, random-table init) and the LID
  weight seam (`setWeightsLID`/`getWeightsLID`/`getWeightsDerivativesLID`, :63-72) never includes
  them -- unserializable, untrainable. (c) `ConvolutionalLayer::feedBackward` (:232-287) is
  entirely commented out: a value-returning function with NO return statement (UB if ever called).
  (d) the dimension guards in `ConvolutionalLayer::feedForward` (:155-172) print to cerr WITHOUT
  exiting on two of the three failure paths, then continue into negative `outputDimensions` and
  out-of-bounds `block()` indexing. *Fix candidate:* if CNN-LID is ever wanted, fix the
  `_NeuronNb[0]` guard, wire the weights into the flat seam, and write the backward -- effectively
  a rewrite.

- **[phase2] BLSTM copy ctor DROPS `_Trainer`/`_InputStatistics`/`_OutputForward`/`_OutputBackward`**
  (`BLSTMNeuralNetwork.cpp:155-173`): the copy ctor copies 17 members (the three sub-networks, cost
  law, mode flags, normalization vectors, cost/classif counters) but omits four -- the Rprop trainer
  (the copy reverts to a default-constructed `Rprop()`, losing the configured `initDelta` and any
  accumulated step sizes), the streaming `_InputStatistics`, and both hidden-state members (empty
  matrices in the copy). The port deliberately implements no `Clone` for `BlstmNetwork` (mirroring
  the ill-formed `NeuralNetwork` copy-ctor entry above); callers rebuild from config +
  `set_weights`. Provenance note. *Fix candidate:* if a copy path is ever needed, copy or explicitly
  reset ALL members.

- **[phase2] NN product-order substitution: Eigen GEMM replaced by ascending-loop products at ALL NN
  sites; the diverging layer-0 GEMMs seed the recurrence, so the substitution propagates NON-LOCALLY**
  (`nn/layers.rs::matmul_seq` + the harness `matSeq`; extends the `[phase1]` DCT GEMM entry;
  measured in `manifest.json:nn_product_probes`, re-probed on every fixture regeneration): the
  harness probes Eigen's blocked product against explicit ascending-k accumulation at the four NN
  product shapes -- `lstm_input_gemm` (T x 23)*(23 x 96) DIVERGED on 8750/19200 elements (~1 ULP)
  and `dense_gemm` (200 x 48)*(48 x 12) DIVERGED on 1890/2400, while `lstm_recurrence_gemv`
  (1 x 24)*(24 x 96) over 150 recurrence steps and `softmax_rowsum` are bit-exact (0/14400, 0/200).
  Unlike Phase 1's DCT (a terminal projection), the diverging layer-0 input GEMM feeds the
  recurrence, so every downstream timestep/gate/cell inherits the difference. The goldens therefore
  come from a harness-local ascending-loop REIMPLEMENTATION, and the REAL compiled legacy classes
  are probed against it with recorded NN_TOL deltas (all synthetic sites 0 ULP; the real-net sites
  e.g. fullseq out 7 ULP, hidden states 735/2076 ULP at ~2e-15 abs, e2e 15/5 ULP) kept in the
  manifest as Phase 4 calibration data. *Fix candidate:* none -- the ascending loop is the portable
  parity target (same rationale as the DCT entry); do NOT swap in BLAS/library GEMMs before
  end-to-end parity re-baselines.

- **[phase2] `LSTMLayer.h:33-36` member-size comments are STALE for BOTH `_PeepWeight` and
  `_Biaises`** (`LSTMLayer.h:35-36` vs the ctor resizes at `LSTMLayer.cpp:21-22`): the header
  declares `_PeepWeight; // size 3*_OutputSize` and `_Biaises; // size _OutputSize`, but the ctor
  resizes `_PeepWeight` to `12 x _OutputSize` (and `LSTMLayer.cpp:21` repeats the stale
  `// size 3*_OutputSize` on the resize line itself) and `_Biaises` to `4*_OutputSize`. The
  comments predate the 12-row peephole-bundle / 4-gate-bias rework; sizing a port or the flat
  layout from them corrupts every weight offset after the first gate block. The port sizes from
  the RESIZES (documented in `nn/layers.rs`'s struct doc) and the flat-seam tests pin the true
  12-row/4-bias layout. *Fix candidate:* fix the two comments in a post-parity legacy cleanup.

## Toolchain deviations

- **[phase1] Oracle harness builds with -std=gnu++14, not the plan's -std=gnu++0x** (tools/oracle_harness/build.sh): Homebrew Boost 1.90 and Eigen headers require >= C++14; parity-neutral because bit-exactness is governed by -fno-fast-math -ffp-contract=off -DEIGEN_DONT_VECTORIZE, not the language standard. Also: shims/x86intrin.h redirects to sse2neon so legacy fmath.hpp parses on arm64; fmath is not odr-used by the Task-1 dumps, and the Phase-1 plan double-pins fmath::log via a numpy float32 oracle when it lands. See build.sh comments and .superpowers/sdd/task-1-report.md for full rationale.

### Forward-noted (add the entry when the phase reproduces it)

- **[3/4] `CostFunctionCalib` nnet_out-before-assignment** (Python optimizer) -- a real
  use-before-assign bug flagged in the setup analysis; fix when the Python training loop is ported.
- **[3/4] OpenMP `InputStatistics` merge-order nondeterminism** (`CorpusProcessor.cpp:171-199`):
  per-file corpus processing runs under `#pragma omp parallel for schedule(dynamic, 1)`, and each
  file's `InputStatistics::update` merge into the shared accumulator happens inside `#pragma omp
  critical` (`:192-196`) -- so the merge ORDER is whichever dynamically-scheduled thread finishes
  and grabs the critical section first, not corpus/file order. Combined with the already-landed
  `[phase1]` finding that `InputStatistics::update`'s pooled variance/std merge is not exactly
  associative in floating point, a multi-threaded legacy run is not bit-reproducible across runs
  (and a rayon `par_iter` port with a different reduction order will not match any single legacy
  run bit-for-bit either). Add the concrete `[phase3]`/`[phase4]` entry (with the Rust reduction
  strategy chosen) once `engine/corpus_processor.rs`'s parallel driver lands.
