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

- **[0b-ii] VRCTS writer byte-golden -- CLOSED in Phase 2b; fixtures are external `vrcts_part`
  reference input**
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
  unit-tested on a constructed `Segmentation` with hand-computed 4-decimal bytes. **CLOSED (Phase 2b):**
  the writer-vs-real-engine byte equivalence originally deferred here to Phase 4 is now ESTABLISHED.
  The Phase 2b oracle harness upgraded the inert `iof` shim to a FAITHFUL mini-fmtr (`%f.Ns` ==
  `std::fixed` + `setprecision(N)`, exactly the semantics pinned above; self-tested per generation via
  the manifest's `fmtr_shim.checks`, guarded by `tests/test_phase2b_fixtures.py::
  test_all_fmtr_checks_passed`), making the REAL compiled `Segmentation::toFile_VRCTS` a byte-golden
  source. Nine real-engine-produced `.xml` fixtures now exist (`tdc_vrcts_chan1.xml`,
  `ltsv_vrcts_chan1.xml`, `signal_{window0,overlap,noOverlap}_vrcts_chan1.xml`,
  `spectral_{real,overlap,noOverlap,pitch}_vrcts_chan1.xml`) and `to_vrcts_string` byte-matches every
  one: `vrcts_bytes_match_dump` (`phase2b_tdc_golden.rs`, `phase2b_ltsv_golden.rs`,
  `phase2b_spectral_golden.rs`), `pitch_vrcts_bytes_match_dump` (`phase2b_spectral_golden.rs`),
  `overlap_vrcts_matches_dump` + `vrcts_bytes_match_dump_cheap_variants` (`phase2b_signal_golden.rs`).
  Byte-exactness is asserted on the oracle environment (the boundary times upstream traverse libm; the
  standard canary-gating posture applies off-env). *Minor (still open):* our writer sanitizes a clone
  (non-mutating) whereas `toFile_VRCTS` mutates its `_Classification` in place; revisit if any caller
  relies on the write-time sanitize side effect.

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

- **[phase2] Spec erratum: S6.6 "(row 0 always)" holds only for `step == 0`** (`nn/blstm.rs`
  `feed_forward_scoring`, from `BLSTMNeuralNetwork.cpp:868-918`): the target-enforcement counter
  starts at 0 and enforces a row when `counter >= target_enforcement_step`; for `step == 0` that
  makes row 0 (and every row) enforced, but for `step >= 1` rows `0..step-1` are `-0.5` and row
  `step` is the first enforced row. The port's doc-comment previously overclaimed "row 0 ALWAYS
  enforced" as a general fact; corrected to state the `step`-dependent behavior. Code was already
  correct -- comment-only fix.

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

- **[phase2] `LSTMLayer.h:35-36` member-size comments are STALE for BOTH `_PeepWeight` and
  `_Biaises`** (`LSTMLayer.h:35-36` vs the ctor resizes at `LSTMLayer.cpp:21-22`): the header
  declares `_PeepWeight; // size 3*_OutputSize` and `_Biaises; // size _OutputSize`, but the ctor
  resizes `_PeepWeight` to `12 x _OutputSize` (and `LSTMLayer.cpp:21` repeats the stale
  `// size 3*_OutputSize` on the resize line itself) and `_Biaises` to `4*_OutputSize`. The
  comments predate the 12-row peephole-bundle / 4-gate-bias rework; sizing a port or the flat
  layout from them corrupts every weight offset after the first gate block. The port sizes from
  the RESIZES (documented in `nn/layers.rs`'s struct doc) and the flat-seam tests pin the true
  12-row/4-bias layout. *Fix candidate:* fix the two comments in a post-parity legacy cleanup.

- **[phase2b] `Segmenter::buildFromConf` reads three WRITE-ONLY cost members, plus a same-key
  DIFFERENT-DEFAULT collision with the live `CostLaw`** (`Segmenter.cpp:139-145`, NOT ported --
  `tasks/segmenter.rs::DriverConfig` omits them entirely): `_CostPonderation` (default `0.5`,
  `{prefix}_CostPonderation`), `_CostLaw` (default `"log"`, `{prefix}_CostLawSpeech`), and
  `_CostLawParam` (default `0.5`, `{prefix}_CostLawParamSpeech`) are parsed and stored on every
  `Segmenter`-derived object but never read back anywhere in the class hierarchy -- `grep` over the
  vendored source shows no live use of `_CostPonderation`/`_CostLaw`/`_CostLawParam` outside the
  copy-ctor and this assignment; the base `computeCost` that would have consumed them
  (`Segmenter.cpp:640-657`) has no live Algo 1-4 caller either (Algo 3/4 cost via the NN's own
  `getCost`; Algo 1/2 accumulate no cost at all -- confirmed by the Task 1 call-site check, S2
  decision 7), so these three members are dead weight on every driver. SEPARATE collision, same key
  family: `Segmenter.cpp:143` reads `{prefix}_CostLawParamSpeech` with default `0.5` into the
  write-only `_CostLawParam`, while `CostLaw.cpp:14` reads the IDENTICAL key
  `{prefix}_CostLawParamSpeech` with default `0.0` into `costLawParamSpeech`, which IS live (feeds
  the softmax cross-entropy cost law actually used by the NN path, ported in `cost.rs`). Two
  different classes read the same config key with two different defaults; only the `CostLaw.cpp`
  reader's value has any observable effect, so the `Segmenter`-side default is moot in practice, but
  a config that relies on the default (omits the key) would silently get `0.0` cost-law behavior
  while a naive read of `Segmenter.cpp` alone would suggest `0.5`. Not ported: `DriverConfig` has no
  field for any of the three keys. *Fix candidate:* post-parity, either delete the dead
  `Segmenter`-side reads (they do nothing) or, if `Segmenter::computeCost` is ever revived for a
  future Algo 5/6 (LID) driver, resolve the default collision explicitly rather than inheriting
  whichever class happens to read the key first.

- **[phase2b] `verifyWindowingType` double quirk: by-value no-fix AND unconditional `_LogStream`
  clobber** (`Segmenter.cpp:61-70`, ported as `tasks/segmenter.rs::verify_windowing_type`): the
  legacy function takes `windowing_type` BY VALUE and reassigns its local copy to `"none"` when the
  string is unrecognized, logging a warning -- but since the parameter is a value copy, the
  reassignment is NEVER written back to the caller's stored `_WindowingType`/`_ConvolutionType`
  field. An invalid windowing/convolution type is warned about but still used downstream as-is (fed
  straight into `getWindowingCoefficients`, whose `else` arm for an unrecognized type returns an
  empty/`None` coefficient vector anyway, so the practical effect is usually equivalent to "none" --
  but only by coincidence of that downstream fallback, not because the type was actually corrected).
  SECOND quirk, same function: `_LogStream = logStream.str()` (`:70`) runs UNCONDITIONALLY after the
  `if`, even on the valid-type path where `logStream` was never written to -- so every call, valid or
  not, overwrites `_LogStream` wholesale (not appends), clobbering whatever a prior invalid-type call
  had logged; only the LAST call's result (warning or empty string) survives on the member. The
  port's `verify_windowing_type(&str) -> Option<String>` takes a borrowed string, so there is nothing
  to mutate; it returns the warning (or `None`) for the caller to log, reproducing both the "no fix"
  behaviour and the "last call wins" semantics (the caller is expected to store/overwrite, not
  accumulate, the returned value; `DriverConfig::from_config` currently discards it via `let _ =`,
  which is a stricter -- not weaker -- reproduction since a discarded log can't diverge from "last
  call wins" either). *Fix candidate:* have the caller actually overwrite the stored type with
  `"none"` on an invalid value, and append rather than clobber the log, after parity.

- **[phase2b] `TdcSegmenter` windowing coefficients are UNNORMALIZED, unlike the convolution kernel**
  (`tasks/sad.rs::TdcSegmenter::get_segmentation`, from `TimeDomainCorrel.cpp:146`
  `getWindowingCoefficients(_WindowingType, false, full_window_size, _WindowingParam)`): the framing
  window applied inside `getSequence` before `classifySequence` passes `normalized=false`, while
  `DriverConfig::from_config`'s `{prefix}_convolution_window_size` kernel (Segmenter.cpp:113) is
  ALWAYS built with `normalized=true`. Easy to accidentally normalize both the same way when porting
  a second driver from this one; the TDC golden (`tdc_result_chan{1,2}.bin`) pins the unnormalized
  windowed-signal magnitude directly. *Fix candidate:* none needed -- this is intentional legacy
  design (the window shapes the autocorrelation without rescaling the signal energy), not a bug.

- **[phase2b] `TdcSegmenter::window_shift_sec` is a STATEFUL member re-quantized on every call using
  its OWN CURRENT VALUE, not the original config value** (`tasks/sad.rs::TdcSegmenter::
  get_segmentation`, from `TimeDomainCorrel.cpp:103-105`): `_WindowShift` is floored at `1/rate` then
  snapped to `round(x*rate)/rate` EVERY time `getSegmentation` runs, mutating the member in place; a
  second call on the same segmenter instance reads back the FIRST call's quantized value as its
  starting point, not the original `TDC_shift` config string. This quantization is PROVABLY IDEMPOTENT
  for every representable `f64` shift, at any practically realizable sample rate: writing
  `q(x) = round(x*rate)/rate`, `q(q(x)) == q(x)` always, because `q(x)` is exactly `w/rate` for some
  nonneg integer `w`, and the compounded double-rounding error of one division (`w/rate`) followed by
  one multiplication (`(w/rate)*rate`) is bounded by ~1 ULP of `w` -- for `round()` to flip its decision
  on the re-quantization the drift would need to reach 0.5, so ANALYTICALLY the idempotency holds for
  all `w` up to ~`2^52` (where 1 ULP of `w` first reaches the 0.5 tie margin), i.e. any practically
  realizable rate/duration; checked numerically over `w` up to `5*10^7` and randomized/ULP-adjacent
  probes around the `round()` half-integer boundary at rate=8000: zero mismatches. No audio file drives
  `w` (a frame count) anywhere near that magnitude. So a SECOND call on the same instance always reproduces the SAME `window_shift`
  frame count as the first call, for every config value, on- or off-grid -- there is no `TDC_shift` that
  can make the two runs' boundaries differ via this mechanism alone. Pinned by
  `two_files_in_sequence_boundaries_match_dump` (`phase2b_tdc_golden.rs`), which therefore pins
  REPEAT-CALL DETERMINISM only, not statefulness: a buggy driver that reset `window_shift_sec` from the
  config string before every call would compute the identical quantized value and pass this golden too.
  *Fix candidate:* none -- this is the documented legacy per-file mutable-state contract (spec S3.4),
  reproduced deliberately, not a bug to fix; the idempotency is a property of the quantization scheme,
  not something to "fix" either.

- **[phase2b] `LongTermSpectralVariation` freq-band clamp order DIFFERS from the BLSTM
  spectral segmenter's variant** (`features/pipeline.rs::derive_freq_band_ltsv_variant`
  vs `SpectralParams::derive`, from `LongTermSpectralVariation.cpp:200-209` vs
  `BLSTMSpectralSegmenter.cpp:229-239`): the BLSTM variant reclamps TWICE -- once
  immediately after `freq_beg` is raised (`if freq_beg > freq_end: freq_beg = freq_end`,
  using freq_end's value BEFORE the max-freq computation), and again after `freq_end` is
  computed (`if freq_end < freq_beg: freq_end = freq_beg`). The LTSV-standalone variant has
  only ONE guard, evaluated AFTER both `freq_beg` and `freq_end` are computed -- there is no
  second reclamp pulling `freq_end` back up. For a min/max pair where the intermediate
  (pre-max-freq) `freq_beg` already exceeds the periodogram's original `freq_end` (e.g.
  `minFreq` beyond Nyquist), the two variants land on DIFFERENT final band values: the
  BLSTM variant snaps to the ORIGINAL `freq_end`, the LTSV variant snaps to the
  POST-max-freq-clamped `freq_end`. Pinned by
  `ltsv_freq_band_variant_differs_from_blstm_variant` (`phase2b_ltsv_golden.rs`), a crafted
  case (`min_freq=5000.0, max_freq=100.0, rate=8000, bins=129`) where BLSTM clamps both to
  128 (4000.0 Hz, Nyquist) and LTSV clamps both to 4 (125.0 Hz). *Fix candidate:* none --
  both are faithful ports of genuinely different legacy code paths, not a bug in either.

- **[phase2b] `LongTermSpectralVariation::getSegmentation`'s periodogram `vec_size` uses
  `frameCount+1`, not `frameCount`** (`tasks/sad.rs::LtsvSegmenter::get_segmentation`, from
  `LongTermSpectralVariation.cpp:293-300`): `begin_frame=0`, `end_frame=audio.getFrameCount()`
  (i.e. the frame count itself, ONE PAST the last valid sample index) is passed into
  `computeSegmentPeriodogramEstimates`/`compute_segment_periodogram_estimates`, giving
  `vec_size = ceil((frameCount+1)/spectrum_shift)`. This is a DIFFERENT call convention from
  `features::pipeline::build_input_sequence`'s `end = audio.data.ncols() - 1` (the BLSTM
  spectral driver's own periodogram span), which does NOT carry the `+1`. Easy to
  copy-paste the wrong `end` value between the two drivers. Pinned structurally by the
  `ltsv_result_chan{1,2}.bin`/`ltsv_dct_result_chan{1,2}.bin` goldens (dumped from a real/
  reimplemented `getSegmentation` using the `+1` convention) and directly by
  `vec_size_uses_frame_count_plus_one` (`phase2b_ltsv_golden.rs`). *Fix candidate:* none --
  this is the real legacy driver's own call convention, reproduced as written.

- **[phase2b] LTSV-standalone window floor is `< 1 -> 1`, DIFFERING from the BLSTM spectral
  segmenter's `< 1 -> 0` (disables LTSV)** (`tasks/sad.rs::LtsvSegmenter::get_segmentation`,
  from `LongTermSpectralVariation.cpp:257-258` vs `BLSTMSpectralSegmenter.cpp`'s own LTSV
  param derivation, ported in `SpectralParams::derive` as `ltsv_half_window = 0` when
  `r < 1.0`): in the BLSTM spectral segmenter, an `LTSVwindow` small enough to round to 0
  half-windows means "LTSV is off" (the caller gates on `ltsv_half_window >= 1`); in the
  LTSV-standalone driver, the SAME arithmetic instead floors up to a half-window of 1 --
  LTSV always runs, never disabled by a tiny window. Pinned by `ltsv_tiny.config`
  (`LTSVwindow 0.001` -> `round(0.001*8000/2/80) == 0` -> floored to 1) and
  `ltsv_window_floors_to_one_not_zero` (`phase2b_ltsv_golden.rs`). *Fix candidate:* none --
  both are faithful ports of genuinely different legacy classes' own derivations.

- **[phase2b] `LtsvSegmenter` carries TWO stateful re-quantized members, both re-quantized
  on EVERY call using their CURRENT value** (`tasks/sad.rs::LtsvSegmenter`, from
  `LongTermSpectralVariation.cpp:154-155` and `:261-263`): `spectrum_shift_sec`
  (`_SpectrumShift`, `round(x*rate)/rate` -- the same `1/rate`-grid quantization as TDC's
  `window_shift_sec`) AND `window_shift_sec` (`_WindowShift`, `round(x*rate/spectrum_shift)
  *spectrum_shift/rate` -- quantized onto a `spectrum_shift/rate`-grid, coarser than TDC's).
  Both quantizations are `round()`-onto-a-fixed-grid operations, and both are therefore
  PROVABLY IDEMPOTENT by the same argument as the TDC `window_shift_sec` proof above (see
  the `[phase2b] TdcSegmenter::window_shift_sec ...` entry): re-quantizing an
  already-quantized value is a fixed point for any realistic frame count / shift
  combination. `two_files_in_sequence_boundaries_match_dump` (`phase2b_ltsv_golden.rs`)
  therefore pins REPEAT-CALL DETERMINISM only, not statefulness, for the SAME reason the TDC
  two-files golden does -- a driver that reset both members from the config string before
  every call would pass this golden identically to the real stateful driver. *Fix
  candidate:* none -- documented legacy per-file mutable-state contract (spec S3.4),
  reproduced deliberately.

- **[phase2b] `ltsv_classify_sequence` over log-mel-scaled input can produce astronomically
  large scores, driving the decision to "always speech" on short excerpts** (observed while
  regenerating the `ltsv.config` golden, `LTSV_is_log_mel true`): the per-bin mean floor
  (`< 1e-12 -> 1e-12`, `LongTermSpectralVariation.cpp:101`) assumes a nonnegative (power-
  scale) periodogram; fed a LOG-scale mel periodogram (whose per-bin mean can be zero or
  negative), the floor instead clamps a near-zero-or-negative mean up to `1e-12`, and the
  ratio `r = p[bin]/mean` (`:110`) then blows up to `~1e12` magnitude for any nonzero
  log-mel value, which the `-r*(r-1)` accumulation squares again (`~1e24`) before the final
  variance-of-`dzeta` squares it ONE MORE time (`~1e48-1e51`, matching the measured
  `ltsv_result_chan1.bin` magnitudes ~2e51-9e51). This is a property of applying the LTSV
  formula (designed for power-scale periodograms) to `is_log_mel=true` input, not a porting
  bug -- the REAL compiled `getSegmentation` produces the identical behavior (confirmed via
  the harness's real-classifySequence-driven `ltsv_boundaries_chan{1,2}.bin` dumps, which
  show near-full-file SPEECH coverage under `ltsv.config`'s thresholds). *Fix candidate:*
  if `is_log_mel=true` is ever paired with LTSV in a production config, consider whether the
  legacy mean-floor should be `abs(mean) < 1e-12` or similarly signed-aware -- out of scope
  for this port (bit-exact reproduction, not correction).
  **Test-coverage consequence (Task 5 review Finding 1):** this blowup makes the STANDARD
  `ltsv.config` golden (and its `ltsv_dct.config`/`ltsv_tiny.config` siblings, which inherit
  `is_log_mel=true`) an always-SPEECH single-span result on every channel -- every score sits
  far above `_DecisionThreshRising` for the whole excerpt, so `update_segmentation`/
  `smooth_segmentation` (hysteresis, area gates, `suppress_short`, `add_padding`) never see a
  real threshold crossing. A green `phase2b_ltsv_golden` suite against those three configs
  alone is BLIND to the entire segmentation decision layer -- "tests green" must not be
  mistaken for "hysteresis/smoothing verified" on this driver. `tests/reference_data/phase2b/
  ltsv_powermel.config` (`is_log_mel=false`, restoring the power-scale periodogram the LTSV
  formula's mean floor assumes, plus smaller-but-nonzero `speech_padding`/`min_speech`/
  `min_silence`) exists SPECIFICALLY to cover that gap: under it the REAL compiled
  `getSegmentation` produces a genuine rising+falling crossing pair on channel 1
  (`SPEECH[0,0.9892)/OTHER[0.9892,1.3472)/SPEECH[1.3472,2.0)`), exercised by
  `powermel_chan1_boundaries_are_non_vacuous` in `src/rust/tests/phase2b_ltsv_golden.rs`.

- **[phase2b] `BLSTMSignalSegmenter` overlap mode is BROKEN AS COMMITTED for any
  non-degenerate window/shift (out-of-bounds `outputSeq` writes)** (`BLSTMSignalSegmenter.cpp:221-240`
  sizing vs `BLSTMNeuralNetwork.cpp:664-669` OverLap write index; IMPROVEMENTS tag
  `signal-overlap-oob`): in overlapping-window mode (`window > 0`, `shift >= 1`), signal sizes
  `result_vec` to `real_vec_size = ceil(frameCount/window_shift)` and does NOT ssr-divide it (the
  ssr-division is gated on `window==0||noOverlap`, `:228`). But `setProcessingType((window>0),
  !noOverlap)` selects the OverLap FFB driver (`feedForwardBackwardOverLap`), which writes
  `outputSeq.block(begin/ssr, 0, lengthShort, cols)` with `begin` up to `~frameCount-window_size`
  and `lengthShort ~ (2*window+1)/ssr` -- so the required row count is `~frameCount/ssr +
  lengthShort`, FAR larger than `ceil(frameCount/window_shift)` for any `window_shift > ssr`. With
  the real net (ssr=4) and the brief's overlap params (`BLSTM_window 0.5, shift 0.1` -> window_size
  2000, window_shift 800), `result_vec` is 21 rows but the driver writes at rows up to ~4500 --
  out-of-bounds. VERIFIED: the REAL compiled `getSegmentation` aborts on it with an Eigen
  `Block.h:146` bounds assertion under `-DEIGEN` assertions (a standalone probe on the real config
  + weights), i.e. heap-corrupts silently under the legacy release `-DNDEBUG` build. The OverLap
  path only survives when `window_shift <= ssr` (so `ceil(frameCount/window_shift) >= frameCount/
  ssr`): the Task-6 overlap golden uses `BLSTM_window 0.01, shift 0.0005` (window_size 40,
  window_shift 4 == ssr) -- the ONLY non-broken regime, degenerate (a 4-sample = 0.5ms shift, ~4000
  windows). The brief's `0.5/0.1` overlap variant is documented, not shipped. *Fix candidate:* the
  overlap sizing should ssr-divide like `window==0`/`noOverlap`, or the OverLap driver should be
  MLPOverLap (one scalar per window). Out of scope for this port (bit-exact reproduction of the
  working regime; the broken regime is unrunnable, like the vendored CNN path). Pinned by the
  Task-6 overlap golden + the doc-comment in `src/rust/tests/phase2b_signal_golden.rs`.

- **[phase2b] `BlstmSignalSegmenter` window/shift sizing quirks: SIGNAL-sample units, the ssr-
  division GATE, and the full=1-when-window-0 windowing no-op** (`BLSTMSignalSegmenter.cpp:96-108,
  :221-240`, ported in `tasks/sad.rs::BlstmSignalSegmenter::get_segmentation`): all reproduced from
  the signal-mode reverse-engineering report's 12-divergence list vs the spectral segmenter. (1)
  window/shift are in raw SIGNAL samples with NO `/spectrum_shift` division anywhere (spectral
  divides at `:441/:445/:448/:453`). (2) When `window_size == 0`, `full_window_size` STAYS 1 --
  signal lacks spectral's `if (window_size == 0) full_window_size = 0` line (`:444`), so `:149`
  builds a windowing coefficient over a length-1 window; Rust's `windowing_coefficients` returns
  `None` for `size <= 1`, and the driver must (and does) STILL proceed -- the coeff is dump-only
  dead weight, never applied to the input (which is raw `audio.data`), matching the legacy
  empty-coeff no-op. (3) The result-vec ssr-division is GATED on `(window_size == 0 || noOverlap)`
  (`:228`); spectral's is UNCONDITIONAL (its gate is commented out, `:487`). This asymmetry is
  load-bearing, not sloppiness (overlapping signal mode emits one output per window position, ssr
  already consumed inside each window). Pinned by `result_vec_sizing_all_variants` +
  `window_zero_full_stays_one_and_driver_proceeds` in `phase2b_signal_golden.rs`.

- **[phase2b] `BlstmSignalSegmenter._WindowShift` is a GENUINELY stateful member: the noOverlap
  `= 0.0` reset makes the next file re-enter through mutated state** (`BLSTMSignalSegmenter.cpp:108`
  write-back + `:376` reset; `tasks/sad.rs::BlstmSignalSegmenter`): unlike TDC/LTSV (whose per-call
  shift re-quantization is IDEMPOTENT, so their two-files goldens only pin repeat-call determinism),
  signal's noOverlap path assigns `_WindowShift = 0.0` mid-run (`:376`, BEFORE `compute_errors`) to
  re-arm the noOverlap trigger for the NEXT file. So file 2 (same segmenter object) genuinely
  re-enters `getSegmentation` with `_WindowShift == 0.0`: `:99` rounds `0.0*rate -> 0` -> `shift < 1`
  -> noOverlap re-triggers, `:107-108` re-clamp the shift to 1 and rewrite `_WindowShift = 1/rate`.
  This is a REAL round trip through mutated state (not a fixed point). The observable OUTPUT
  boundaries nonetheless coincide across the two files (same audio -> same posteriors -> the SAME
  untouched always-Other SEED hypothesis, `[Other@0, End@dur]`: the real net's posteriors on this
  excerpt are ~0.002-0.03, far BELOW the 0.6 rising threshold, so NO speech is ever detected on
  either file), which the two-files golden asserts rather than assumes; the state DOES round-trip,
  the output does not change. Also reproduced: the reset-order divergence vs spectral (signal
  resets BEFORE `compute_errors` at `:376`; spectral AFTER the mat dump at `:885` -- functionally
  equivalent, ported as written).
  Boundary/type coincidence alone cannot distinguish "the reset fired and re-derived the same
  value" from "the reset never happened": WITHOUT it, file 2 would inherit `window_shift_sec =
  1/rate` (not `0.0`), re-deriving through the OVERLAP branch (not noOverlap) with an UNCONDITIONAL
  `ceil(frame_count/window_shift) = ceil(16001/1) = 16001`-row result vector -- vs the real
  (reset-present) `4000`. So the two-files golden additionally asserts
  `BlstmSignalSegmenter::last_result_rows()[0].len() == 4000` (not the 16001 no-reset
  counterfactual) and `window_shift_sec() == 0.0` post-call on BOTH files, making the sizing/state
  observable the actual discriminator rather than relying on the boundary coincidence. Pinned by
  `two_files_in_sequence_no_overlap_lifecycle` in `phase2b_signal_golden.rs`.

- **[phase2b] `BlstmSignalSegmenter` divergences that are dead/log-only in the signal chain**
  (`BLSTMSignalSegmenter.cpp`): `_FlagDCOffset` is LOG-ONLY (`:116`) -- never applied, since the NN
  consumes raw `audio.data` directly (`:188`, no `getSequence`/DC-removal call), unlike spectral
  which threads it into the periodogram; pinned by `dc_offset_flag_is_log_only` (flag true vs false
  -> byte-identical output). `BLSTM_TwoSweeps` is REQUIRED-but-DEAD (`:20/:28`): the only consumers
  are commented out (`:282/:297`), so it is parsed (a missing key aborts the legacy
  `conf.get<bool>`, so the parse is load-bearing) and stored, never read; pinned by
  `from_legacy_missing_two_sweeps_is_err` (missing key -> `Err`) + `from_legacy_reads_real_values`.
  The `signalRaw` (PRE-preemph, `:133-134`) vs `signal` (POST-preemph, `:174-175`) dumps GENUINELY
  DIFFER when `preemph_ratio > 0` (divergence 8 vs spectral, whose two dumps are byte-identical
  because spectral creates the .mat after preemph) -- pinned by
  `signalraw_differs_from_signal_and_both_match_dump`.

- **[phase2b] The Rust `Segmentation` container is SINGLE-CHANNEL; per-channel cost/classif live on
  the DRIVER** (spec S6; `tasks/sad.rs::BlstmSignalSegmenter` `cumulative_error`/`nb_of_classif`
  vs the legacy `Segmentation::_CumulativeError[chan]`/`_NbOfClassif[chan]`, `BLSTMSignalSegmenter.
  cpp:346-347`): the legacy `Segmentation` is channel-indexed (`_Classification.at(chan)`,
  `_CumulativeError[chan]`); the Rust container holds ONE channel's hypothesis, and the driver holds
  a `Vec<Segmentation>` (one per channel) plus the per-channel `cumulative_error`/`nb_of_classif`
  scalars. Behavior-preserving (all goldens are per-channel), a documented structural deviation.
  The NN cost path DOES fire in signal mode (unlike the cost-free TDC/LTSV): `cumulative_error[chan]
  = net.cost`, `nb_of_classif[chan] = net.nb_of_classif` after each `feed_forward_backward`.

- **[phase2b] `BlstmSpectralSegmenter` (Algo 3) result-vec sizing is UNCONDITIONALLY ssr-divided**
  (`BLSTMSpectralSegmenter.cpp:475-499`, ported in `tasks/sad.rs::get_blstm_param`): the sizing
  computes `vec_size = ceil(frameCount/_SpectrumShiftInFrames)`, then divides sequentially by every
  LSTM + output sub-sampling ratio -- guarded ONLY by `if (getSubSamplingRatio() > 1)` (`:488`). The
  `if ((BLSTM_window_size == 0)||(noOverlap))` gate that would restrict the division (`:487`) is
  COMMENTED OUT in the live source, so spectral ALWAYS decimates. This is the load-bearing CONTRAST
  with the signal driver (`BLSTMSignalSegmenter.cpp:236-240`), whose identical division IS gated on
  `(window==0 || noOverlap)`. Reproduced verbatim; pinned by
  `get_blstm_param_unconditional_ssr_division_contrast` (`phase2b_spectral_golden.rs`), which
  hand-derives both the spectral (50) and the hypothetical signal-gated (201) result at the overlap
  regime (window>0, !noOverlap). The `10*ssr` noOverlap window minimum (`:449`) and the
  window-half-to-ssr floor (`:442`) are ported too (`get_blstm_param_10ssr_floor_fires`,
  `get_blstm_param_no_overlap_floor_and_10ssr`).

- **[phase2b] `BlstmSpectralSegmenter` CROSS-CHANNEL `result_vec` REUSE** (spec-named, load-bearing;
  `BLSTMSpectralSegmenter.cpp:631/:740`): the legacy allocates `result_vec` ONCE (returned by
  `getBLSTMParam`, `:631`, BEFORE the channel loop) and REUSES the SAME buffer across channels
  (`:740` passes it to `feedForwardBackward` each iteration). The OverLap FFB accumulates INTO the
  caller's buffer in place (`:664` `noalias() +=`, then `/= count`), so channel 2 SEEDS from channel
  1's post-division contents -- channel 2's result is CONTAMINATED by channel 1's carry-over. The
  Rust driver reproduces this by holding ONE `result_buf` across the channel loop, NOT zeroed
  between channels. Pinned by `overlap_variant_and_cross_channel_reuse` (both channels' results
  byte-match the dumps) + the NON-VACUITY contrast `cross_channel_reuse_is_load_bearing` (a
  fresh-buffer chan-2 run differs from the seeded dump by up to 0.6% relative -- proving the reuse is
  pinned, not coincidental). This is the OPPOSITE convention from the signal driver, whose
  `result_vec` is FRESH per channel (`BLSTMSignalSegmenter.cpp` allocation-scope divergence).

- **[phase2b] `BlstmSpectralSegmenter` timeStep OVERLAP-branch override + noOverlap reset ORDER**
  (`BLSTMSpectralSegmenter.cpp:724-734/:885`): the overlap branch OVERRIDES `timeStep`/`timeOffset`
  with `_SpectrumShift*ssr` & `timeStep/2 - _SpectrumShift/2` (`:731-732`), NOT `_WindowShift` -- the
  asymmetry vs the signal driver, which uses `_WindowShift` in its overlap branch. Reproduced;
  pinned by `timestep_timeoffset_overlap_branch_override`. Separately, the noOverlap `_WindowShift =
  0.0` poisoning fires AFTER the dumps/`compute_errors` (`:885`), unlike signal's BEFORE-`compute_
  errors` reset (`BLSTMSignalSegmenter.cpp:376`); this reset-ORDER divergence is cosmetic (the reset
  only affects the NEXT file) but ported as written. The two-files lifecycle
  (`two_files_in_sequence_no_overlap_lifecycle`) pins the round trip: file 2 re-enters
  `get_blstm_param` with `window_shift_sec == 0.0` -> noOverlap re-triggers -> window floored to 324,
  shift clamped to 1 -- discriminated by the dumped `spectral_noOverlap_params_file2.bin` (1x4), not
  merely the coincident boundaries.

- **[phase2b] `BlstmSpectralSegmenter` non-wav `_SpectrumShiftInFrames = 80` PERSISTENCE**
  (`BLSTMSpectralSegmenter.cpp:209-210`): when `!audio.hasReadWavFile()`, `initSpectralAnalysis` sets
  `_SpectrumShiftInFrames = 80` and `_SpectrumShift = 80/rate`, and these PERSIST into the member for
  subsequent files. The Rust driver always reads a wav (there is no non-wav fixture), so the
  persistence is pinned unit-test-only via a direct state mutation
  (`BlstmSpectralSegmenter::force_non_wav_spectrum_shift` +
  `non_wav_spectrum_shift_80_fallback_persists`), exposed as a `pub` method solely for that test (no
  production caller).

- **[phase2b] The REAL `getBLSTMInputSequence` uses an Eigen `applyDCT` GEMM that diverges from the
  Rust `build_input_sequence`** (`BLSTMSpectralSegmenter.cpp:561-591` reads
  `audio._CepstreCoefficients`, filled by `MelFilterBank::applyDCT`'s Eigen `melPeriodogram*_CoeffsDCT`
  product): at the real net's DCT shapes this diverges from the ascending-loop DCT the Phase 1
  goldens + the Rust port reproduce (measured up to ~19k ULP on the assembled input). So the Task-7
  oracle harness builds the spectral input via the ASCENDING-LOOP pipeline (empty-mel periodogram +
  `applyFilterBank` + `applyDCTLoop` + `getLTSV`), matching the E2E leg, NOT the real
  `getBLSTMInputSequence` -- confirmed byte-identical to `e2e_input.bin`. This is the same
  Eigen-GEMM-vs-ascending divergence already logged for the DCT/NN sites; noted here because it
  forced the harness input-assembly choice for the spectral driver golden.

- **[phase2b] Pitch second pass rebuilds the input with the OLD (pass-1) LTSV column, NOT recomputed
  on the warped periodogram -- CODE PATH IS WIRED BUT UNEXERCISED BY ANY CURRENT FIXTURE** (
  `BLSTMSpectralSegmenter.cpp:757-805`, Task 8): the pitch pass warps `audio._Periodogram` by
  `pitch/300` (`:760-775`), re-runs the mel filterbank + DCT on the warped periodogram (`:777-787`),
  then calls `getBLSTMInputSequence(audio, melFilters, freq_beg, freq_end, LTSV, TDC)` (`:792`) with
  the SAME `LTSV` matrix computed in pass 1 over the UNWARPED periodogram -- IF an LTSV column were
  present, it would be stale relative to the warped spectral features. However, the real
  `1_worker_1.config` (and the Task 8 `pitch_map()` variant, which only overrides the `TDCwindow`/
  `TDCshift`/`TDC_lags`/`TDC_balance`/`TDC_windowing_*` keys to activate the pitch gate) both carry
  `BLSTM_LTSVwindow 0`, so `SpectralParams::derive` resolves `ltsv_half_window = 0` and NO LTSV column
  is appended in EITHER pass (`assemble_input_sequence`'s LTSV hcat is skipped entirely). The Rust
  driver still captures the pass-1 LTSV output (`build_input_sequence_parts`) and threads it through
  `assemble_from_periodogram` for the pass-2 build -- reproducing the WIRING -- but with
  `ltsv_half_window == 0` that plumbing is a no-op: `spectral_pitch_inputseq_pass2_chan1.bin` pins only
  the warp -> mel -> DCT columns, not any LTSV reuse. The stale-LTSV quirk this entry describes would
  only be exercised by a config with BOTH `TDCwindow > 0` (pitch pass active) AND `LTSVwindow > 0` (an
  LTSV column actually present) -- no such fixture exists yet. Post-parity: recompute LTSV on the
  warped periodogram (or document that the warp is only meant to affect the mel/DCT band); a fixture
  covering the LTSVwindow>0 x pitch-pass-active combination would be needed to golden-test the quirk
  itself, not just the wiring.

- **[phase2b] The externalized pitch-pass result dump preserves PASS-1, even though the final
  boundaries are PASS-2** (`BLSTMSpectralSegmenter.cpp:744,:848-850`, Task 8): `result_vec2` is set to
  `result_vec.transpose()` right after the pass-1 FFB (`:744`) and dumped by the `_IsUnitTest` `.mat`
  branch at `:848-850` -- which runs AFTER the pitch second pass has already overwritten `result_vec`
  (`:793`) and re-segmented (`:801`). But `result_vec2` is a COPY taken before the pitch pass, so the
  dumped `result_vec_chan_*` is the PASS-1 result while `seg`'s boundaries are PASS-2. The Rust
  `last_result_rows` observation point reproduces this exactly (it captures the pass-1 row and is NOT
  overwritten by the pitch pass; the pass-2 row is exposed separately as `last_result_rows_pass2` for
  test capture only). Pinned by `pitch_dump_quirk_pass1_result_preserved` (the pass-1 result is
  byte-identical to the T7 `spectral_real` result, while the boundaries differ: crossing 1.1831 ->
  1.2597). Post-parity: dump the pass-2 result if the externalized value is meant to reflect the final
  segmentation.

- **[phase2b] Pitch pass OVERWRITES the error/classif slots (not accumulated) and is guarded on
  `pitch > 0`** (`BLSTMSpectralSegmenter.cpp:791-803`, Task 8): after the pass-2 FFB, `seg.
  _CumulativeError[chan]` and `seg._NbOfClassif[chan]` are ASSIGNED (`:802-803`) from the pass-2 NN,
  discarding the pass-1 values (`:752-753`) -- an overwrite, not a `+=`. The entire re-forward +
  re-segmentation is gated on `if (pitch > 0)` (`:791`): a zero pitch (no accepted TDC estimate over
  any pass-1 SPEECH segment, or no SPEECH segment at all) leaves the pass-1 boundaries + error slots
  in place (the homothety with `coeff = 0` still runs at `:760-787` but its output is never forwarded).
  Reproduced in the Rust driver (the `pitch > 0.0` guard wraps the re-forward/clear/re-seg/overwrite).
  On the excerpt the reimpl-path cost is 0 (no targets), so the overwrite lands 0 over 0 -- the
  observable is the boundary/result change, not the cost.

- **[phase3] `CostLaw::computeUnitaryDeltas` folds the LOGISTIC output-activation
  derivative `output*(1-output)` directly into the scalar VAD delta** (`cost.rs::
  compute_unitary_delta`, from `CostLaw.cpp:343`): after selecting the per-regime law
  derivative and applying the optional `BackPropWER` scale, the final line multiplies by
  `output*(1-output)` -- the derivative of the scalar output head's `Logistic` activation
  (`NeuronLayer.cpp:144`), NOT a second law-derivative term. This is the scalar-path
  counterpart to the multiclass softmax+CE fusion (S5/S11.3): the scalar head is a plain
  logistic, not a softmax, so its chain rule is the textbook `y(1-y)` rather than the
  softmax+CE simplification, but it is likewise folded into the cost seam rather than
  applied as a separate layer-side Jacobian. The real config's own `CostLawThreshSpeech
  1`/`CostLawThreshNoSpeech 0` (`1_worker_1.config:64-65`) puts the above-thresh branch
  exactly at `output` in `{0.0, 1.0}`, where this fold is ZERO and would mask a broken
  above-thresh law derivative -- `mid_thresh_law` (both thresholds moved to 0.5) is the
  dedicated non-vacuity builder for that branch. *Why deferred:* provenance; the fold is
  the correct chain rule for the scalar head's activation, not an ad hoc addition.
  *Pinned by:* `scalar_delta_polynomial_bit_exact`/`scalar_delta_log_sqrt_canary` (the
  fold is present in every scalar delta, poly bit-exact, log/sqrt canary-gated) and
  `scalar_delta_midthresh_cubic_above_branch`/`scalar_delta_midthresh_sqrt_above_branch`
  (non-vacuous away from the fold's `output in {0,1}` zero).

- **[phase3] `BackPropWER` scales the scalar delta AND the multiclass fusion by an
  ASYMMETRIC `10x` factor keyed on target class, `10*(1-target)` (speech-on) vs
  `10*target` (other-on)** (`cost.rs::compute_unitary_delta` + `compute_deltas`, from
  `CostLaw.cpp:335-341` scalar / `:371-406` multiclass): when `_BackPropWER >= 0`
  (`isCostModified`), the delta (scalar path) or the whole fused row (multiclass path) is
  multiplied by a constant `10.0` weighted by the OPPOSITE regime's target value -- for a
  speech-on frame (`target > 0.5`) the factor is `10*(1-target)`, for an other-on frame
  it is `10*target`. At exact one-hot targets (`target` in `{0.0, 1.0}`) this reduces to
  a plain `10.0` or `0.0`, but the multiclass fixtures use SOFT targets specifically so
  the scaling is observable as a genuine non-constant factor rather than degenerating to
  0/10 (see the harness comment cited in `multiclass_deltas_wer_and_pond`). The same `10x`
  constant also gates the cost itself (`compute_unitary_cost`, `cost.rs:276-282`, from
  `CostLaw.cpp:180-186`) -- WER scaling enters BOTH the forward cost and the backward
  delta identically, per risk R4's ponderations-in-cost-AND-gradient concern. *Why
  deferred:* provenance; this is the legacy's Word-Error-Rate-oriented cost reweighting,
  reproduced verbatim. *Pinned by:* `multiclass_deltas_wer_and_pond` (`cost_deltas_wer.
  bin`/`cost_deltas_wer_pond.bin`, WER combined with and without classes_ponderations)
  and `ignore_mask_zeroes_row`'s WER-path branch (a fully-masked row stays exactly 0
  under WER too).

- **[phase3] `NeuronLayer::feedBackward` counts `InputSeq.rows()`, not `deltas.rows()`** (
  `NeuronLayer.cpp:199`): `_NbOfSeqFedBackward += InputSeq.rows()` uses the ORIGINAL input's row
  count as the frame-count denominator for the Nx2 harvest (`getWeightsDerivatives`, `:101-114`),
  not the incoming `deltas`' row count. For this layer the two coincide in every current caller
  (the dense output layer never internally decimates between its input and its deltas), so the
  distinction is currently unobservable end-to-end -- but it is the legacy's explicit choice
  (mirrored by `LSTMLayer.cpp:719`'s `linesNb` from the layer's own input, same pattern) and the
  port (`nn/layers.rs::NeuronLayer::feed_backward`) reproduces it literally rather than reusing
  `deltas.nrows()` out of convenience. Pinned by `count_is_input_rows`
  (`tests/phase3_neuron_backward_golden.rs`) and a dedicated mutation check (swapping the count
  source to `deltas.rows()` was confirmed to fail that test during Task 3 review). No fix
  candidate -- this is the correct, intentional legacy behavior, just non-obvious from the call
  site alone.

- **[phase3] `NeuronLayer::feedBackward`'s hidden-path Maxmin2 deriv fold reads the RAW `InputSeq`
  parameter, not the width-reconstructed `input` local** (`NeuronLayer.cpp:204`, `deltas_out =
  (deltas*W^T).array()*(InputSeq.unaryExpr(Maxmin2::deriv).array())`): the accumulation half
  (`:157-183`) genuinely reads the width-tolerant-reconstructed tensor (`cols > I` truncates,
  `cols < I` zero-pads, per `:157-166`), but the deriv-fold half at `:204` indexes the ORIGINAL,
  un-reconstructed `InputSeq` argument directly -- two different tensors read by the same function.
  A prior port draft (Task 3, pre-review) folded on the reconstructed tensor for both halves; caught
  in review because it is only observable at `cols != I`, which never arises from the sole legacy
  caller (`BLSTMNeuralNetwork.h:29`, `.cpp:90` always chains `NeuronLayer`s output-to-input at
  `cols == I`). At that reachable width, reconstruction is a byte-exact no-op, so the deviation was
  fully dormant against every fixture. The two out-of-range directions are NOT symmetric: a wide
  (`cols > I`) input reaching the raw fold is still IN-BOUNDS (the fold only ever indexes columns
  `0..I`, and reconstruction's wide-case truncation is exactly `leftCols(I)` -- the same first-`I`
  columns the raw tensor already has), so raw-vs-reconstructed is silently identical there and a
  wide fixture could never discriminate the two tensors even if one were built. Further, because
  `cols < I` zero-padding can only turn an out-of-bounds read into an in-bounds `0.0` (never a
  differing finite value), the ONLY way the fix is observable at all is that a genuinely narrow
  (`cols < I`) input reaching the hidden fold now panics (index out of bounds) instead of silently
  zero-filling -- which mirrors the legacy's own Eigen coefficient-wise-product shape mismatch (UB)
  at that same unreachable shape. Fixed to read
  `input[[r, c]]` (the raw parameter) directly in `nn/layers.rs::NeuronLayer::feed_backward`. Pinned
  by `hidden_fold_reads_raw_input_not_reconstructed` (arithmetic sanity at `cols == I`) and
  `hidden_fold_panics_on_narrow_raw_input` (the actual regression discriminator: reverting to the
  reconstructed tensor makes the expected panic NOT occur) in `tests/phase3_neuron_backward_golden.rs`.
  No fix candidate needed beyond the literal-fidelity correction already applied -- this entry
  documents a legacy-unreachable branch whose Rust semantics are now pinned to the literal source
  rather than silently generalized.

- **[phase3] LSTM backward mixes PRIOR-step and CURRENT-step gate deltas in the peephole
  cross-terms, via the `deltasForgetGatetmp` save** (`LSTMLayer.cpp:518-682`, port
  `nn/layers.rs::LstmLayer::feed_backward`): the reverse-time loop reuses the SAME
  `deltasInputGate`/`deltasForgetGate`/`deltasOutputGate` 1xO buffers across iterations, so when a
  block reads them for a peephole cross-term the value is whatever the PRIOR (later-time) iteration
  left there -- EXCEPT the current row's output-gate delta, which is computed first (`:585`) and is
  therefore already CURRENT by the time the state/forget/input blocks read it. The input-gate block
  (`:659-682`) additionally needs the PRIOR forget-gate delta (peep row 6, `:660`), but by then the
  forget block (`:624-657`) has already overwritten `deltasForgetGate` with the current row's value;
  the legacy saves the prior one into `deltasForgetGatetmp` at `:624` (BEFORE the overwrite) and reads
  the save at `:660`. So a single expression can legitimately mix a current-step delta (output gate)
  with prior-step deltas (input/forget) -- not a bug, but easy to "clean up" into all-current or
  all-prior and get wrong. The port reproduces the exact buffer lifetimes: `deltas_output_gate` is
  overwritten before the state/forget/input blocks read it (current), `deltas_input_gate`/
  `deltas_forget_gate` are read at their prior value inside those blocks and only then overwritten,
  and `deltas_forget_gate_tmp = deltas_forget_gate.clone()` captures the prior forget delta before the
  forget block. Pinned by the peep-row and gate-block mutation battery in
  `tests/phase3_lstm_backward_golden.rs` (the mandated peep-row-4<->5 and gate-block-i<->f swaps, plus
  the `forgettmp_use_live` mutation swapping the saved prior for the live current value, all confirmed
  to fail a golden). No fix candidate -- correct, intentional, non-obvious.

- **[phase3] LSTM backward `row==0` forget-gate branch drops the cellstate term but KEEPS the
  peephole cross-terms** (`LSTMLayer.cpp:648-657`, port `nn/layers.rs::LstmLayer::feed_backward` else
  branch): at the first time step there is no `c_{-1}`, so the `row>0` branch's
  `_CellStates.row(row-1) .* epsilonState` term is absent -- but the branch still folds the gates-peep
  (rows 4/10) and gates-rec (row 7) cross-terms and the bias derivative, and still activates through
  `GatesFunction::deriv`. It also skips the input-weight-only accumulation's feedback/peep-row-1/6/8
  half (there is no `output.col(-1)` and no `_CellStates.row(-1)`). This is standard BPTT boundary
  handling, but the asymmetry (drop ONE term, keep the rest) is a classic transcription trap -- a naive
  port either drops the whole forget block at row 0 or reads `_CellStates.row(row-1)` out of bounds.
  Pinned by `row_zero_forget_branch` (T=1, must not panic, layer-native derivs) and the
  `row0_spurious_cellstate` mutation (adding a `cs[row]` term at row 0 fails a golden). No fix
  candidate -- correct, intentional.

- **[phase3] Container backward `InvSubSample` does NOT restore the SubSample-dropped tail frames --
  the returned `deltas_out` has `floor(T/R)*R` rows, not `T`** (`NeuralNetwork.hpp:125-145`, port
  `nn/network.rs::inv_sub_sample` + `Network::feed_backward`): the forward `SubSample(R, T-rows)` FLOORS
  to `floor(T/R)` rows, silently dropping the trailing `T mod R` frames (already tracked as the forward
  decimation quirk). The backward `InvSubSample(R, .)` inflates a `K`-row sub-sampled delta block back to
  `K*R` rows by un-stacking the R column-blocks -- so a round-trip yields `floor(T/R)*R` rows, and the
  dropped tail is GONE. For the real net (`1_worker_1.config`: `LSTMNeuronNb 23,24,24` / subsample
  `4,1`; T frames, layer-0 R=4) the layer-0 `deltas_out` returned to the BLSTM wrapper is
  `floor(T/4)*4` rows wide, up to 3 frames short of the original input; the wrapper never uses these
  tail rows (it splits the output-net deltas, not the LSTM `deltas_out`), so the truncation is invisible
  downstream -- but a port that "helpfully" pads InvSubSample back to `T` rows, or that asserts
  `deltas_out.nrows() == input.nrows()`, diverges from the legacy. The Task 5 goldens are a T=11, R=2
  multi-layer net: `sub_sample` floors 11->5, `inv_sub_sample` restores 5->10 (NOT 11); fix-wave 1
  (review finding 1) added a T=7, R=2 SINGLE-layer net (`neuron_nb.len()==2`, `:255-263` -- the real
  `configs/legacy/LID_BLSTM.config` single-layer subsample shape, `LSTMNeuronNb 11,12` / subsample `4`,
  in miniature) exercising the SAME floor/expand quirk on the structurally distinct single-layer branch:
  `floor(7/2)=3` decimated rows, InvSubSample restores 3*2=6 (NOT 7).
  Pinned by `inv_sub_sample_inverts_sub_sample_on_floored_input` (round-trip == the first 10 rows of the
  11-row input, bit-exact) + the `(10, 3)`/`(10, 4)`/`(6, 2)` structural row-count assertions in
  `lstm_net_subsample_inversion_{fwd,rev}` / `dense_net_last_layer_subsample` / `single_layer_subsample_inversion`,
  and the injected mutation battery (skip-inflation, wrong-column-block, and -- fix-wave 1 -- the
  single-layer arm reading the running `deltas_out` instead of the seed `deltas`, both the subsample and
  plain arms, confirmed to fail `single_layer_subsample_inversion`/`single_layer_plain_no_subsample`/
  `single_layer_reads_seed_deltas_not_running_deltas_out`). No fix candidate -- correct, intentional; the
  floor/expand asymmetry is a direct consequence of the forward decimation and is load-bearing for the
  deltas_out shape.

- **[phase3] Container backward `NeuronLayer` layer-0 frame count is `input.rows()`, not the (smaller)
  delta rows it back-projects over** (`NeuronLayer.cpp:199`, exercised by `Network::feed_backward` over
  the dense `[4,3,2]` sub `[1,2]` net): under a downstream SubSample the incoming `deltas` have FEWER rows
  than the layer's stored input (that net's layer 0 receives 10 delta rows over an 11-row input).
  `NeuronLayer::feedBackward` accumulates its weight derivatives over `deltas.rows()` (10) but bumps
  `_NbOfSeqFedBackward` by `InputSeq.rows()` (11) -- so the Nx2 col1 (the normalizer denominator) for that
  layer is 11, while the LSTM layers count `deltas.rows()`. The mismatch is intentional (each region's
  count is whatever that layer's `:199`/`:720` line says) and load-bearing for the at-update-time
  `col0 cwiseQuotient col1` normalization. Pinned by
  `dense_net_count_reflects_input_rows_not_delta_rows` (layer-0 count == 11, layer-1 count == 5) and the
  `get_derivatives_layout_layer0_first` layout check. No fix candidate -- documented for the
  normalization audit.

- **[phase3] Container backward `net_*_deltasout` calibration: the returned `deltas_out` is a `deltas*W^T`
  GEMM (k=O) so the REAL Eigen path diverges from the ascending reimpl in the last ULP** (Task 5 harness
  `net_lstm_backward_{fwd,rev}_deltasout` / `net_dense_backward_deltasout`): the Nx2 DERIVS at these
  synthetic net shapes are k=1 outer products and match the real class bit-for-bit (`max_ulp=0`), but the
  layer-1 `deltas*W^T` back-projection (k=O=4 for the LSTM net) is a blocked-vs-ascending GEMM that
  diverges (fwd 8 ULP, rev 16 ULP, max_abs ~8.7e-19; the dense net's k=O=3 happened to be 0 ULP). The
  reimpl dump IS the golden (same ascending `matmul_seq` the Rust uses), so the Rust matches the dump; the
  divergence is recorded as calibration in `manifest.json:net_backward_tol.deltasout_calibration`, NOT
  gated to 0 like the derivs. Same non-local blocked-GEMM story as the Phase 2 forward; documented here so
  the nonzero calibration line is not mistaken for a port bug.

- **[phase3] INVERTED iRPROP- SIGN CONVENTION: `deriv>0 -> dw = -delta`, `deriv<0 -> dw = +delta`** (
  `nn/train.rs::Rprop::update_weights`, from `Rprop.cpp:14-23/:26-56`): the legacy update
  computes `dw = -delta` when `deriv > 0.0` (positive, would ascend naively) and `dw = +delta`
  when `deriv < 0.0` (negative, would descend naively), then applies `weights += dw` -- an
  inverted/descend-by-negation sign convention baked into the update rule, not a transcription
  quirk. A naive "corrected" sign (positive deriv -> positive delta -> descent via `weights -=`
  on the applied delta) would diverge from every golden immediately on step 1 (the post-first-call
  weights would flip sign, destroying all downstream deltas/step sizes). Reproduced verbatim.
  *Why deferred:* nothing to fix -- this IS the iRPROP- algorithm as the legacy implements it, and
  every trajectory golden is pinned bit-exact against it. *Pinned by:* `trajectory_a_bit_exact` +
  `trajectory_b_bit_exact` (`phase3_rprop_golden.rs`, all weights/deltas/delta_weights/prev_derivs
  bit-exact at EVERY step across two independent real-compiled golden trajectories) and
  `inverted_sign` (structural sign-direction assertion). *Mutation evidence (brief req. 2):*
  swapping the sign convention (first-call branch: `deriv>0 -> +delta`, `deriv<0 -> -delta`)
  was confirmed to fail BOTH trajectory tests at step 1, and reverting it restored the golden
  match.

- **[phase3] `Rprop::_PrevCost` is UNINITIALIZED by the legacy ctor, but this is benign:
  it is never READ before the first WRITE** (`Rprop.h:16` declares `_PrevCost` a plain
  `double` with no in-class initializer; `Rprop.cpp:6` (`Rprop(double initDelta)`) does not
  mention it in the member-init list either): the only read (`Rprop.cpp:42`, the
  cost-gated backtrack condition `_PrevCost < cost`) lives entirely inside
  `updateWeights`'s `else` branch (the "subsequent call" path, `:23-57`); the first call
  takes the `_Deltas.size() == 0` branch (`:10-22`) unconditionally and returns without
  ever touching `_PrevCost`, and that SAME first call is the only one that writes it
  (`:58`, unconditional at the end of every call). So by the time the backtrack condition
  can execute (call >= 2), `_PrevCost` always holds a real value from call 1's tail write
  -- the indeterminate ctor value is dead on every reachable path. The port
  (`nn/train.rs::Rprop`) initializes the field to `0.0` for Rust's no-uninitialized-memory
  discipline, which is a strictly SAFER default than the legacy's indeterminate value but
  provably unobservable (both are dead until the same write). *Why deferred:* nothing to
  fix in the legacy behavior itself -- documenting the reachability argument so a future
  reader does not mistake the field for a live footgun. *Pinned by:* the `deltas.
  is_empty()` first-call gate (`nn/train.rs`) structurally mirroring `_Deltas.size() == 0`,
  and `trajectory_a_bit_exact`/`trajectory_b_bit_exact` exercising calls 2+ (where
  `prev_cost` is read) bit-exact against the real class from a real call-1 write, never
  from the ctor default.

- **[phase3] COST-GATED BACKTRACK + `prev_derivs` ZEROING** (`nn/train.rs::
  Rprop::update_weights` shrink branch, from `Rprop.cpp:38-45`): on a derivative sign flip
  (`derivTimesPrev < 0`), the weight step is undone (`weights[j] -= delta_weights[j]`) ONLY if
  the cost went UP since the prior call (`prev_cost < cost`); the shrink is applied regardless
  (delta clamped to min), but the last delta_weights value is discarded/stale if the cost did
  NOT rise. Separately and critically, `prev_derivs[j]` is ZEROED (:45) unconditionally after the
  shrink, forcing the NEXT call's `deriv_times_prev` for that element to be exactly 0 regardless
  of the next step's derivative sign, routing it into the `== 0` branch (recompute + apply at
  the current, decayed delta). Do not "fix" this into an unconditional backtrack, and do not drop
  the zeroing -- the backtrack gate and the zeroing are the load-bearing iRPROP- machinery for
  handling weight reversals. Reproduced verbatim. *Why deferred:* nothing to fix -- this is the
  iRPROP- rule as specialized by Igel & Husken, 2000 (the cost-gated backtrack + prev_derivs
  zeroing on a sign flip; the original Riedmiller & Braun 1993 RPROP backtracks unconditionally
  on every sign flip, with no cost comparison -- "iRPROP-" is this project's inherited name for
  the gated variant, kept for continuity with the spec/CLAUDE.md). *Pinned by:*
  `trajectory_a_bit_exact` (step 3 backtracks on cost rise 9->12, step 4 does NOT on cost fall
  12->8, both elements' prev_derivs bit-match the zero-forced values at subsequent steps) +
  `trajectory_b_bit_exact` (alternating cost oscillations over 48 steps; element 1 hits the
  shrink branch every other step and its backtrack gate FIRES every one of those times -- cost
  rises on every even step by construction, so trajectory B never exercises a shrink WITHOUT a
  backtrack; the no-fire case (step 4's cost fall) is pinned solely by trajectory A. Element 1's
  step-boundary prev_derivs match the forced-zero value; alternating-sign delta shrinks are
  clamped correctly).
  *Mutation evidence (brief req. 3):* (a) inverting the backtrack gate (`prev_cost < cost` ->
  `prev_cost > cost`) failed both trajectory tests, the element that should backtrack at step 3
  did not. (b) Dropping the zeroing (`self.prev_derivs[j] = 0.0` removed) failed both trajectory
  tests via a `prev_derivs` mismatch at the step immediately after the shrink -- the zero-forced
  value was not present to route the next call into the `== 0` branch, so the state machine
  diverged. Both mutations reverted to verified-good state after each.

- **[phase3] BLSTM enforcement-step (`_TargetEnforcementStep < 0`) mutates the caller-visible
  `outputSeq` interior to -0.5, and does so SEPARATELY from the backward's own target rewrite**
  (`nn/blstm.rs::feed_forward_backward_plain`, from `BLSTMNeuralNetwork.cpp:788-828`, risk R7):
  the plain per-window worker, when backprop is active and `_TargetEnforcementStep < 0`, first
  builds a NEW target copy with interior rows `[1, rows-1)` set to -0.5 and feeds THAT to
  `feed_backward` (the output is NOT touched at this point, so the backward seeds its deltas from
  the raw forward output), THEN the cost block builds ANOTHER fresh target copy AND overwrites the
  caller's `output` interior with -0.5 before `computeCost`. Two independent -0.5 rewrites, and the
  ordering (backward first, on unmutated output; cost second, mutating output) is load-bearing --
  a port that shared one rewrite or reordered them would seed the backward from the -0.5-poisoned
  output. *Why deferred:* provenance; the -0.5 interior is the legacy's soft-target enforcement
  convention and the goldens are pinned against it. *Pinned by:* `enforcement_active_backward`
  (`tests/phase3_blstm_backward_golden.rs`) vs `blstm_bwd_enforce_derivs.bin`. *Mutation evidence:*
  feeding the ORIGINAL (un-rewritten) target to the backward instead of `new_target` failed the
  golden; reverted after.

- **[phase3] `updateWeights` normalizes `col0 cwiseQuotient col1` ELEMENT-WISE, so a count-0
  region divides `0/0 = NaN` (or `x/0 = +-inf`) per IEEE -- NO guard** (`nn/blstm.rs::update_weights`,
  from `BLSTMNeuralNetwork.cpp:306`, risk R1): the per-element quotient is the ONLY correct
  normalization -- a scalar `/nframes` is wrong because different deriv-object regions carry
  different frame counts (the mean/std tail count 1; the LSTM blocks accumulate frames x sweeps x
  per-window coverings). Since the downstream trainer (iRPROP-) is SIGN-based, the magnitude of the
  quotient is invisible through Rprop on a well-formed count vector -- the element-wise-vs-scalar
  distinction is observable ONLY at a count-0 element, where `0/0 = NaN` routes to Rprop's else
  (ascend) branch while a scalar `0/n = 0` takes no step. Under a real training call every count is
  > 0, so the NaN path is not hit; it is reproduced (no guard) for parity. *Why deferred:* provenance
  + the count-0 case cannot occur in real training. *Pinned by:* `update_weights_element_wise_not_scalar`
  (a count-0 zero-deriv row takes the `+init_delta` NaN-routed step). *Mutation evidence:* replacing
  `col0 / col1` with `col0 / n` (scalar) failed that test (the count-0 row took no step); reverted.

- **[phase3] BLSTM windowed backward: TwoSweeps derivs DOUBLE (both sweeps back-propagate, derivs
  NOT halved) and OverLap derivs accumulate per covering window -- the `col1` count carries the
  forward-average compensation, never the derivs** (`nn/blstm.rs` windowed drivers +
  `get_weights_derivatives`, from `BLSTMNeuralNetwork.cpp:548-590`/`:592-681`, spec S6): the
  windowed drivers call the per-window worker (forward + backward + cost) inside EVERY window, so
  the sub-network deriv accumulators (which `+=` per `feed_backward` and are reset only by
  `reset_weights_derivatives`) naturally sum across windows/sweeps. TwoSweeps runs the worker over
  two offset padded sweeps and averages the FORWARD output `/2`, but the derivs are the full
  two-sweep sum (col1 count == 34 vs the single-sweep truncate's 12); OverLap divides the forward
  output by the per-row coverage but accumulates the derivs once per covering window (col1 == 27 vs
  the single-pass 12). A "fix" that halved the derivs to match the forward /2 would corrupt the
  gradient. *Why deferred:* provenance; the count column is the legacy's compensation mechanism.
  *Pinned by:* `twosweeps_count_vector`/`overlap_count_vector` (col1 == the manifest-recorded 34/27,
  strictly > the single-pass counts) + the col0 goldens (`blstm_bwd_{twosweeps,overlap}_derivs.bin`,
  non-vacuously differing from the truncate col0 by up to 22x). *Mutation evidence:* pre-normalizing
  col0 by col1 in `get_weights_derivatives` (i.e. halving/dividing the accumulated derivs) failed
  `twosweeps_count_vector`; reverted.

- **[phase3] BLSTM `getWeightsDerivatives` mean/std tail is 4 constant blocks `[Zero | Ones | Zero
  | Ones]` -- deriv 0, count 1, stats never trained** (`nn/blstm.rs::get_weights_derivatives`, from
  `BLSTMNeuralNetwork.cpp:259,270`): the `2*inputSize` normalization-tail rows (mean then std) are
  appended to the Nx2 gradient object as `[0.0, 1.0]` each -- a zero derivative (the mean/std are
  not gradient-trained; they are folded from `InputStatistics` separately) and a count of exactly 1
  (so the `col0/col1` update-time quotient leaves them as `0/1 = 0`, i.e. no step). *Why deferred:*
  provenance. *Pinned by:* `meanstd_tail_structure` (last `2*inputSize` rows bit-exact `[0.0, 1.0]`)
  + `update_weights_changes_weights_via_rprop` (the tail weights -- mean 0 / std 1 -- are unchanged
  after a step). *Mutation evidence:* changing the tail push to `[0.0, 0.0]` failed
  `meanstd_tail_structure`; reverted.

- **[phase4a] Corpus no-listing fallback: the loop's `lang = "unk"` / `dial = "unk"` defaults are
  DEAD when the key is absent -- lang/dial come out `""`** (`engine/corpus.rs::from_config`, from
  `Corpus.cpp:44-56` + `ConfigFile.h:70-77`): `get_list<string>("reflangfiles", "", filesNames.size(), ',')`
  pads an ABSENT key to exactly `filesNames.size()` copies of the overload's OWN default `""` -- so
  `ii < refLangFiles.size()` is always true and the loop's local `"unk"` initializer is never
  reached; every item gets language/dialect `""`. `"unk"` only fires when the key is PRESENT but
  its comma-split is SHORTER than `files` (a non-empty list is never padded: `vector::resize` runs
  only in the `.empty()` branch). *Why deferred:* the `""`-vs-`"unk"` language feeds the
  class-mapping lookup and the count keys; changing it changes class indices. *Pinned by:*
  `no_listing_fallback` (absent keys -> `""`) + `no_listing_fallback_unk_when_reflangfiles_short`
  (present-but-short -> `"unk"` for the uncovered index only).

- **[phase4a] Corpus ctor "normalizing coefficients" print derefs `++begin()` -- UB when the corpus
  has fewer than 2 classes** (dropped, display-only; from `Corpus.cpp:123-130`): the legacy tail
  print walks `_ClassCount.begin(); ++it; it->second` unconditionally, dereferencing `end()` (UB)
  whenever `_ClassCount` has a single entry (e.g. every file unknown-class). The whole block is
  log-only state-free output and is dropped in the port (doc-comment on `from_config`); no
  class-balance state is computed there. *Why deferred:* nothing to port -- recorded so nobody
  "restores" the print verbatim later.

- **[phase4a] Listing weight/fileId `istringstream` extraction: hexfloat atom accumulation makes
  `"0.5abc"` FAIL but `"0.5z"` extract 0.5** (`engine/corpus.rs::iss_extract_double`/
  `iss_extract_int`, from `Corpus.cpp:80-91`): num_get stage 2 accumulates every char from the
  float atom set (digits, `a-f`/`A-F`, `x/X`, `p/P`, sign only first-or-after-exponent, at most
  one `.` -- a second dot STOPS accumulation, so `"1.2.3"` extracts 1.2), then stage 3 must consume
  the WHOLE accumulated string or the extraction fails and the 1.0/1 defaults are kept. Int atoms
  exclude `x`/`.` (`"7x"` -> 7, `"0x10"` -> 0); overflow sets failbit (C++11) -> default kept. NOT
  reproduced (cannot occur in real listings): stage-3 hexfloat (`"0x1p3"`) and the
  subnormal-underflow ERANGE corner. *Why deferred:* provenance -- the accept-on-success guard is
  the load-bearing part. *Pinned by:* `iss_extract_matches_istringstream_oracle` (31-case table
  captured from compiled `istringstream >>` probes on the oracle env).

- **[phase4a] `saveWeights` glues the `weights_`/`weightsDerivatives_` prefix to the WHOLE filename
  string, not the basename** (`nn/blstm.rs::save_weights`, from `BLSTMNeuralNetwork.cpp:319-323`):
  `buf << "weights_%s" << filename` (and `buf2 << "weightsDerivatives_%s"`) string-concatenates the
  prefix before the ENTIRE path, so a `<filename>` of `/out/epoch995.mat` yields the sibling `.bin`
  path `weights_/out/epoch995.mat` -- the prefix does NOT respect path separators. In production
  `<filename>` is a bare basename (the engine runs in the output dir), so the siblings land next to
  it; a full path would target a nonexistent `weights_<dir>/` parent. Reproduced verbatim (the port
  concatenates identically). The `<filename>` itself typically ends in `.mat` even though the two
  siblings are the custom `.bin` codec (i64 rows/cols + column-major f64), NOT MAT-files -- the
  commented-out real `.mat` weight writes (`:324-325`) are dead. *Why deferred:* provenance; the
  string-glue is load-bearing for any tooling that reads these siblings by the same rule. *Fix
  candidate:* once the driver bag is ported, prepend the prefix to the basename only (or write to a
  dedicated artifacts dir). *Pinned by:* `save_weights_writes_three_artifacts`
  (`tests/phase4a_lifecycle.rs`): the sibling paths are computed by the same whole-string glue;
  `tier2_train_epoch_weights_golden` (`tests/phase4a_train_golden.rs`): the glued-name siblings
  (`weights_bestNNWeight_1_tier2_spectral.mat`, io::binary despite the `.mat` suffix) are compared
  value-for-value against the REAL legacy `saveWeights` output from the harness train stage.

- **[phase4a] `<prefix>_weightsFile` too-many case: warning + silent head-truncation; the port drops
  the console warning** (`nn/blstm.rs::load_weights_file`, from `BLSTMNeuralNetwork.cpp:141-148`):
  a `.bin` with FEWER elements than `getNbOfWeights()` is a hard `exit(1)` (ported as `Err`); with
  MORE, the legacy prints `Warning: The number of gains given in %s is more than what's needed.` and
  STILL calls `setWeights`, which consumes only the head and ignores the tail. The port reproduces
  the truncation exactly (`set_weights` already tolerates an over-long slice) but does NOT emit the
  console warning -- there is no logging seam here yet. *Why deferred:* the numeric behavior (load
  the head, ignore the tail) is load-bearing and reproduced; the warning is a diagnostic side-effect
  with no golden. *Fix candidate:* thread a log sink through the engine drivers and restore the
  warning (or make the mismatch a hard error once configs are trusted). *Pinned by:*
  `weights_file_too_many_truncates` + `weights_file_too_few_errors` (`tests/phase4a_lifecycle.rs`).

- **[phase4a] `BagOfProcessors` ctor clears `files`/`refsegfiles`/`reflangfiles` to `""` in every
  config map but deliberately NOT `refdialfiles`** (`engine/bag_of_processors.rs::from_configs`,
  from `BagOfProcessors.cpp:21-23`): a memory-reuse quirk in the legacy ctor -- three of the four
  per-file listing keys are blanked in-place before any driver is constructed (so a later corpus
  load off the SAME map sees them empty), the fourth is left untouched with no apparent reason.
  Reproduced verbatim (asymmetric clearing, not a full reset). *Why deferred:* provenance; changing
  this would silently alter what `Corpus::from_config` (Task 5) sees on a shared map. *Fix
  candidate:* none identified -- likely an oversight in the original, revisit only if a corpus-load
  golden depends on `refdialfiles` being live post-bag-construction. *Pinned by:*
  `key_clears_applied` (`src/engine/bag_of_processors.rs`).

- **[phase4a] `isBackPropActivated`/`getWeights` non-NN-algo defaults are asymmetric (`vec![false]`
  vs empty vec)** (`engine/bag_of_processors.rs::{is_back_prop_activated, get_weights}`, from
  `BagOfProcessors.cpp:73-101`): for algo 1/2 (non-NN segmenters, no `else if` branch matches),
  `isBackPropActivated` falls through to the ctor's own `return vector<bool>(1, false)` (`:85`) --
  a ONE-element vec -- while `getWeights`/`getInputStatistics`/`getWeightsDerivatives` fall through
  to `return vector<...>()` (`:100`,`:128-129`,`:143-144`) -- an EMPTY vec. Not obviously
  intentional (the four dispatch methods otherwise mirror each other structurally), but reproduced
  as written since it is directly observable (a caller iterating `is_back_prop_activated` for a
  non-NN config sees one `false`, not zero elements). *Why deferred:* provenance; changing either
  side would need a caller-side audit of every non-NN-algo consumer once Task 5's corpus-processor
  driver lands. *Pinned by:* `dispatch_vec_shapes` (`src/engine/bag_of_processors.rs`).

- **[phase4a] `File_Type != 0` (non-wav ingestion) is unported; bag ctor bails**
  (`engine/bag_of_processors.rs::from_configs`, from `BagOfProcessors.h:44`): the legacy
  `_FileType` member documents the enum `0: wav; 1: phSeq; 2: cep`. `AudioStruct.cpp` reads
  file_type == 0 (libsndfile `sf_open`), == 1 (`.phSeq` plain binary, framerate 8000), and == 2
  (`.cep` Mel-frequency cepstral coefficient binary, framerate 8000). The Rust port reads `File_Type`
  from the config and bails with `Err` if `file_type != 0`, since the `AudioStruct` non-wav readers
  (`phSeq`/`cep` paths) are not ported. *Why deferred:* the Phase 4a parity corpora use only wav
  files (file_type 0); non-wav ingestion paths were never exercised and remain unvalidated. *Fix
  candidate:* once (if) a corpus ever requires non-wav files, port `AudioStruct`'s `.phSeq` and
  `.cep` readers and plumb the file_type enum through the audio I/O. *Pinned by:*
  `file_type_nonzero_bails` (inline, `engine/bag_of_processors.rs`).

- **[phase4a] `SegmentationFunction` lock-file block stubbed (no `LockFilesDir` -> always treat
  the file)** (`engine/bag_of_processors.rs::segmentation_function`, from `BagOfProcessors.cpp:214-250`):
  the legacy gates per-file processing on an `O_CREAT|O_EXCL` lock file under `_LockFilesDir` (a
  crude multi-worker file-claim scheme -- `<dir>/<prefix>_<jj>_<basename>.lck`, with the odd
  double-`open()` retry ladder at `:225-249`). The port drops the whole block: with `_LockFilesDir`
  empty the legacy itself always sets `treatFile = true` (`:214-215`), which is the only regime the
  Phase 4a in-process rayon-static-lane driver runs in (no cross-process lock coordination). The
  legacy `jj` argument (lane index) fed ONLY the lock filename, so it is dropped from the Rust
  signature. *Why deferred:* the lock scheme is a distributed-cluster artifact; the in-process port
  coordinates lanes via static-lane assignment (spec S), not filesystem locks. *Fix candidate:* if a
  multi-process cluster drop-in is ever needed, reintroduce the lock-file claim behind a
  `LockFilesDir`-set branch. *Pinned by:* `segmentation_function`'s always-treat behaviour across the
  scored/unscored tests (`tests/phase4a_segfn.rs`).

- **[phase4a] Unscored branch ALWAYS writes VRCTS (even with no dump dir, next to the audio)**
  (`engine/bag_of_processors.rs::segmentation_function`, from `BagOfProcessors.cpp:394-401`): the
  SCORED result branch (`:352-356`) writes the VRCTS dump ONLY when `dumpDir` is non-empty, but the
  UNSCORED branch (`:394-401`) writes it UNCONDITIONALLY -- to `dumpDir/<basename>` when a dump dir is
  set, else to `<full-audio-path-minus-4-char-extension>` right next to the source audio. So a plain
  `-s`/`-S` solo run silently drops a `<audiofile-without-ext>` VRCTS file beside every input. Both
  branches share the load-bearing basename quirk: `substr(last_slash+1, size-last_slash-1-4)` (strip
  the last path component's 4-char extension) for the dump-dir case, `substr(0, size-4)` (strip the
  extension from the FULL path) for the next-to-audio case. Reproduced verbatim. *Why deferred:*
  observable side effect on disk; changing it would diverge from the legacy's file output. *Fix
  candidate:* after end-to-end parity, gate the unscored write on an explicit opt-in flag. *Pinned by:*
  `unscored_mode_zero_columns_and_vrcts` + `dump_dir_vrcts` (`tests/phase4a_segfn.rs`).

- **[phase4a] CSV reference loads ONLY for single-channel audio (2-channel `buf` left empty ->
  no reference)** (`engine/bag_of_processors.rs::segmentation_function`, from
  `Segmentation.cpp:745-806`): `load_ref_from_csv` loops over `_ChannelNb` and builds the file-to-open
  string `buf` ONLY in the `_ChannelNb == 1` branch (`:748-749` `buf << filename`); the
  `_ChannelNb == 2` branch (`:750-752`) emits a "wrong path" LOG line and NEVER writes `buf`, so
  `ifstream(buf.str())` opens the empty string, fails, and the `else` body that pushes `_Reference` +
  parses lines is skipped for EVERY channel. Net effect: a `.csv` reference is honored only for
  mono audio; for stereo (or any `_ChannelNb != 1`) the reference is silently empty, so a scored
  stereo run with a CSV reference would hit the mandatory-reference bail (`:302-305`). The port
  reproduces this: CSV builds a reference only when `channel_count == 1`; otherwise `None`. *Why
  deferred:* load-bearing legacy bug directly affecting whether scoring runs; "fixing" it (loading
  the CSV for both channels) would diverge from the oracle. *Fix candidate:* after end-to-end parity,
  decide whether stereo CSV references should load channel 0 (or per-channel columns). *Pinned by:*
  the `channel_count == 1` guard in `segmentation_function` (the Phase 4a tests exercise the STM path
  on the 2-channel excerpt; a mono-CSV golden lands with the corpus-processor fixtures).

- **[phase4a] `.trs` reference loader unported; `segmentation_function` bails on a TRS reference**
  (`engine/bag_of_processors.rs::segmentation_function` + `extension_of`, from `Segmentation.cpp:101-109`
  `load_ref_from_trs`): the legacy `Segmentation` ctor dispatches any non-`.stm`/`.csv`/`.xml`
  reference extension (and any name too short for an extension) to `load_ref_from_trs` (a Transcriber
  `.trs` XML parser). That loader is not ported (`segmentation_io.rs` carries STM/CSV/VRCTS only), so
  the reference dispatch here `bail!`s on a TRS reference rather than silently producing an empty
  reference. `.xml` (VRCTS) reference loading is also not wired into this dispatch -- no Phase 4a
  corpus uses it -- and currently falls into the TRS bail branch. *Why deferred:* the Phase 4a parity
  corpora use STM (SAD) and CSV (WER) references only; TRS/VRCTS reference inputs were never exercised.
  *Fix candidate:* port `load_ref_from_trs` (and wire `load_vrcts` into the reference dispatch) when a
  corpus needs them. *Pinned by:* the `RefExt::Trs` bail path (inline in `segmentation_function`).

- **[phase4a] CLOSED (Task 8): multi-channel VRCTS write on the corpus path**
  (`engine/bag_of_processors.rs::segmentation_function` VRCTS write sites +
  `tasks/segmentation_io.rs::to_vrcts_string`/`write_vrcts_multichannel`, from
  `Segmentation.cpp:543-590` `Segmentation::toFile_VRCTS`): the legacy writer loops
  `for (int chan = 0 ; chan < _ChannelNb ; ++chan)`, sanitizing and emitting one
  full VRCTS document per channel -- `<basename>_chan_<n>.xml` when `_ChannelNb > 1`,
  or a single `<basename>.xml` when `_ChannelNb == 1` -- each with its own
  `Channel`/`Speaker`/`SegmentList` block (and `num`/`ch` = `chan+1`) for that
  channel's `_Classification`. Before Task 8 the port emitted channel 0 only
  (hardcoded `chan="1"`), silently dropping channel 2+ on stereo audio. *Fix (Task
  8):* `to_vrcts_string` now delegates to a per-channel `to_vrcts_string_chan`
  (channel number threaded into `num`/`ch`), and a new `write_vrcts_multichannel`
  fans out one `<base>_chan_<n>.xml` per channel (or `<base>.xml` for mono); the bag
  write sites call it with the full `seg_per_chan` slice and the legacy-derived
  `name=`/`path=` attrs (audio basename minus extension / full audio path). The
  single-channel byte goldens (0b-ii/2b) stay green (shape-generic; `chan="1"`
  unchanged for mono). *Pinned by:* `phase4a_tier1_e2e.rs::vrcts_byte_equal` (Rust
  `write_vrcts_multichannel` output byte-matches the REAL compiled `toFile_VRCTS`
  dumps for both channels of all three corpus files) + `phase4a_segfn.rs::
  {unscored_mode_zero_columns_and_vrcts,dump_dir_vrcts}` (per-channel filename
  fan-out).

- **[phase4a] CLOSED (Task 8): result-row `nb_words` column defaults to -1, not 0**
  (`engine/bag_of_processors.rs::{assemble_scored_row,assemble_unscored_row}` +
  `tasks/segmentation_io.rs::WerStats::legacy_default`, from `BagOfProcessors.cpp:
  322,373` `tmp.push_back(seg._WordErrorRate[chan]._NbWords)`): the legacy result
  row pushes the `WordErrorRate` struct's `_NbWords`, whose CONSTRUCTOR default is
  `-1` (`Segmentation.h:70`), left untouched whenever WER Pass 1 does not run (STM
  references, or no reference -- Pass 1 is CSV-only). The port emitted `0` there:
  the scored row read `report.wer.unwrap_or_default().nb_words` (Rust `WerStats`
  i64-Default `0`) and the unscored row hardcoded `0.0`. Both now use
  `WerStats::legacy_default()` (`nb_words = -1`, rest 0), matching the legacy. *Found
  by* the Task 8 tier-1 `MultiConfigResults` golden (col 10 = data col 7 = -1 in the
  real dump, 0 in the port). *Pinned by:* `phase4a_tier1_e2e.rs::{solo_tdc_matches,
  train_ltsv_two_epochs_matches,multiconfig_matches}` (MultiConfigResults col-10
  byte compare vs the real dump).

- **[phase4a] `costLID = -1.0` gate applied AFTER `saveWeights`, BEFORE `updateWeights` --
  order is load-bearing** (`engine/bag_of_processors.rs::save_and_update`, from
  `BagOfProcessors.cpp:409-471`, specifically `:464-466`): per config, `saveAndUpdate` calls
  `saveWeights` with the REAL (pre-gate) `costLID`, THEN overwrites `costLID = -1.0` when
  `totalSpeechDuration < 1e-3` (no speech segments detected in the accumulated file x channel
  rows), THEN calls `updateWeights` with the gated value. The row-slice writes
  (`costMem`/`badClassifMem`/`costLIDMem`/`badClassifLIDMem`, `:457-460`) also happen BEFORE the
  gate, so they always carry the real `costLID`, never `-1.0`. Swapping the order (gating before
  `saveWeights`, or before the row writes) would silently change algo 5/6's save criterion
  (`badClassifLID+costLID` / `cost+costLID`) in Phase 4b even though it is inert for algo 3/4 in
  4a (their save/update criteria are `cost` alone, ignoring `costLID` entirely). *Why deferred:*
  Phase 4a has no algo 5/6 bag to observe the gate on the SAVE side; only the update-side effect
  is observable, via a test-only hook. *Fix candidate:* none -- this is the correct legacy order,
  to be reproduced as-is when Phase 4b lands algo 5/6. *Pinned by:* `update_called_with_neg_costlid`
  (`tests/phase4a_save_update.rs`), which asserts the update-side value is `-1.0` while the same
  call's row-slice write (`cost_lid_row`) still carries the pre-gate value.

- **[phase4a] `saveAndUpdate`'s column means are per file x channel ROW, not per file** (from
  `BagOfProcessors.cpp:409-471`): `resultsperConf[ii].rows()` is one row per (file, channel) pair
  accumulated by `SegmentationFunction` (Task 5), not one row per file -- the legacy's own "%s
  files have been processed" print (`:449`) is therefore a misnomer (display-only, dropped in this
  port per the brief) since it prints the row count as a file count. This is not merely a cosmetic
  quibble: it is what makes `badClassif = means(2)` and `badLIDClassif = 100-means(15)` genuinely
  per-frame-class averages over every scored channel, rather than per-file averages that would
  need a further per-file channel-count weighting. *Why deferred:* not a bug, just an easy
  misreading of the legacy print; noted so a future reader does not "fix" the row semantics to
  match the dropped print's wording. *Pinned by:* `cost_mem_rows_written`
  (`tests/phase4a_save_update.rs`), which mixes single-row algo-1/algo-3 configs so
  `means == sums` and the per-config independence of the row-count basis is exercised directly.

- **[phase4a] Static-lane deterministic reduction replaces the legacy OpenMP dynamic parallel-for**
  (`engine/corpus_processor.rs::run_epoch`, from `CorpusProcessor.cpp:172-203`): the legacy runs
  `#pragma omp parallel for ... schedule(dynamic, 1)` over files and folds each file's contribution
  inside an `omp critical` block, so the FOLD ORDER is whichever thread grabs the critical section
  first -- nondeterministic even at a fixed thread count. Since the derivative `+=` and
  `InputStatistics::update` merge (`:184-199`) are float-order-sensitive, the reduced values are
  not bit-reproducible run to run. This port (spec S3, the R6 determinism fix -- the ONE deliberate
  deviation from legacy parallelism) assigns file `j` to lane `j % N` (`N = nb_of_threads` clamped
  to `[1, nb_files]`), clones the epoch-start bag once per lane (`firstprivate`, `:172`), walks each
  lane's files in ascending `j` (so genuine cross-file driver state chains within a lane), and folds
  in ASCENDING FILE INDEX after the parallel section. At `N == 1` this is byte-identical to a plain
  sequential loop (the golden-pinned parity mode); for `N > 1` the results are deterministic (fixed
  for a fixed N) but differ from the legacy's nondeterministic run AND, because state chains per
  lane, from `N == 1`. *Why deferred:* this is a deliberate FIX (determinism), not a bug to revisit;
  the entry documents the N-dependence so a reader does not expect `N > 1` to match `N == 1` or the
  legacy. This CLOSES the forward-noted `[3/4] OpenMP InputStatistics merge-order nondeterminism`
  item below. *Pinned by:* `lanes_n1_equals_sequential` (`tests/phase4a_corpus_processor.rs`),
  which asserts `run_epoch` at N=1 is bit-exact (col 6 = the wall-clock timing column masked) vs a
  hand-sequential oracle over a 3-file corpus.

- **[phase4a] NN driver `get_weights_derivatives` was shadowed by the `Segmenter` trait default
  (returned an empty matrix); added an inherent net delegate** (`tasks/sad.rs`, both
  `BlstmSignalSegmenter` and `BlstmSpectralSegmenter`): Task 3/4 added inherent net-delegating
  methods for `save_weights`/`update_weights`/`input_statistics`/`is_back_prop_activated` but NOT
  for `get_weights_derivatives`, so `BagOfProcessors::get_weights_derivatives` (which calls
  `seg.get_weights_derivatives()` with `Segmenter` in scope) resolved to the trait DEFAULT
  (`segmenter.rs:39-41`, `Array2::zeros((0,0))`) instead of the net's real Nx2 derivative matrix.
  The corpus gradient harvest (`run_epoch`) and gradCheck therefore saw an all-empty derivative
  everywhere. Task 7 adds the missing inherent `get_weights_derivatives` delegate to both NN drivers
  (mirroring the `input_statistics` delegate); Rust's inherent-over-trait method resolution makes
  the bag pick it up. *Why noted:* a cross-task gap (Task 3/4 omission) surfaced only when a caller
  actually read the harvested derivs; the fix is a pure additive delegate, no behavior change to
  existing callers (which never read a non-empty derivs before). *Pinned by:* the harvest path in
  `grad_check_synthetic` reaching a `(137, 2)` analytic derivative shape rather than `(0, 0)`
  (`tests/phase4a_corpus_processor.rs`).

- **[phase4a] CLOSED (Task 7b): corpus-level gradCheck was DEGENERATE (cost/counter both 0) under
  the Phase 2b drivers** (`engine/corpus_processor.rs::grad_check`, from `CorpusProcessor.cpp:237-340`):
  the gradCheck central difference reads result col 4 (`cumulative_error`) / col 17 (`nb_of_classif`)
  from the per-file results. The Phase 2b NN drivers (`tasks/sad.rs`,
  `BlstmSignalSegmenter`/`BlstmSpectralSegmenter` `get_segmentation`) hard-coded an EMPTY target to
  `feed_forward_backward`, which gates BOTH the backward derivative accumulation AND the cost/counter
  accumulation on `target.nrows() > 0` (`blstm.rs:1150,1171`). So with no target: `cumulative_error
  == 0`, `nb_of_classif == 0`, analytic derivs 0 with count 0, gradcheck cost `0/0 == NaN` on both
  sides -- the check was VACUOUS. **Task 7b closed this** by threading the per-channel REFERENCE
  through the `Segmenter::get_segmentation` trait (new `refs: Option<&[Segmentation]>` param) and
  calling the already-ported `get_targets` (`segmenter.rs`) inside the two NN drivers under the legacy
  gate `seg._Reference.size() > 0` (`BLSTMSignalSegmenter.cpp:255-258`,
  `BLSTMSpectralSegmenter.cpp:735-738`). The corpus bag (`bag_of_processors.rs::segmentation_function`)
  now passes its loaded references to the driver. *Pinned by:* `grad_check_synthetic`
  (`tests/phase4a_corpus_processor.rs`) now asserts mean-relative-error < 5e-4 (analytic vs numeric
  agree) plus a non-vacuity guard (>= 1 nonzero numerical deriv) and the bit-exact weight restore;
  `spectral_scored_cost_is_live` / `spectral_no_reference_zero_cost` (`tests/phase4a_segfn.rs`) pin
  the live-cost path and the no-reference gate on the REAL algo-3 config. *Mutation evidence:* the
  `refs=None` contrast test (`spectral_no_reference_zero_cost`) fails if the driver ever builds a
  target without a reference (col 4 / col 17 would go nonzero); and reverting the trait wiring so the
  bag passes `None` returns `grad_check_synthetic` to the NaN-degeneracy it had before Task 7b.

- **[phase4a] `getTargets` with `_BackPropWER < 0` uses the SPEECH `classType` arg; the drivers pass
  `SPEECH` unconditionally** (`tasks/sad.rs` NN drivers, from `BLSTMSignalSegmenter.cpp:257` /
  `BLSTMSpectralSegmenter.cpp:737`): both NN drivers call `getTargets(..., SPEECH)`. In `getTargets`
  (`Segmenter.cpp:696-704`, the `_BackPropWER < 0` branch), the reference SPEECH span -> target 1.0,
  SUBSTITUTION -> 1.0 (via `classType == SPEECH && ty == SUBSTITUTION`), EXCLUDED -> -0.5, everything
  else -> 0.0. The real `1_worker_1.config` has no `BLSTM_BackPropWER` key, so `DriverConfig`
  defaults it to -1.0 (`< 0`) -> this simple-class branch, NOT the WER-soft-target branch. The
  synthetic signal gradcheck config also sets `BLSTM_BackPropWER -1.0` explicitly. Reproduced
  verbatim; no port deviation.

- **[phase4a] The spectral PITCH second pass REUSES the pass-1 target buffer (not rebuilt on the
  warp)** (`tasks/sad.rs::BlstmSpectralSegmenter::get_segmentation` pitch block, from
  `BLSTMSpectralSegmenter.cpp:793`): the pitch pass re-forwards the WARPED input through
  `feed_forward_backward` but passes the SAME `targetSeq` built once at `:736-738` for pass 1 -- the
  legacy does NOT re-call `getTargets` for the pitch re-forward. The Rust port carries the pass-1
  `target` in scope across the pitch block and passes `&target` unchanged, matching `:793`
  element-for-element. Reproduced verbatim. (Under the real algo-3 config the pitch pass is gated on
  `TDCwindow > 0`, which the base config does not set, so this is exercised only by the pitch-variant
  spectral goldens; the target reuse is nonetheless wired for that path.)

- **[phase4a] Task 7b target flow does NOT interact with the `_TargetEnforcementStep < 0` interior
  rewrite for the ported configs** (`nn/blstm.rs::feed_forward_backward_plain`, risk R7 cross-ref):
  the R7 rewrite (interior target/output rows -> -0.5) fires only when `target_enforcement_step < 0`.
  The real `1_worker_1.config` and the synthetic gradcheck config both set/derive
  `TargetEnforcementStep = 0` (`>= 0`), so the reference-driven target flows through to `feed_backward`
  and `compute_cost` UNCHANGED (no interior overwrite). The existing `blstm.rs` R7 behaviour is thus
  unaffected by the Task 7b wiring for the ported configs; the R7 mutation of the caller-visible
  `outputSeq` remains reachable only under a `step < 0` config (none in the current test corpus).
  Verified by inspection; the existing phase3 backward goldens (which DO exercise `step < 0`) stay
  green under `cargo test`.

- **[phase4a] gradCheck `max_weights` cap is a port-only DEVIATION from the legacy full sweep**
  (`engine/corpus_processor.rs::grad_check_capped`): the legacy `gradCheck` (`:263`) checks EVERY
  weight (`kk < weightsNb`), each requiring two full corpus runs -- O(N) forwards for an N-weight
  net (33,671 for the real config). The port adds a `max_weights` cap (the public `run()` path uses
  `usize::MAX` == the full legacy sweep; the test/Task-9 path caps at 10) so the golden replay does
  not pay 33,671 * 2 corpus runs. Recorded in the Task 9 manifest as `gradcheck_max_weights`. *Why
  deferred:* a deliberate test-cost deviation, not a bug; the capped subset still spans the flat
  layout (Task 9 chooses the indices). *Pinned by:* `grad_check_synthetic` (10-weight cap) and Task
  9's `phase4a_gradcheck_golden`.

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
