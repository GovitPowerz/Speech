# Improvements & Deferred Fixes

Tracked work to revisit once the Rust/Python port reaches end-to-end parity with the legacy
`FastSpeechProcessing` engine. Nothing here should be "fixed" mid-port - these are reproduced
bit-exactly on purpose so the goldens match.

**Maintainer note:** every phase that reproduces a legacy quirk/bug adds an entry to the Legacy
Quirks section below (what / where / why deferred / fix candidate). See CLAUDE.md for the pointer.

## Phase 5 fix protocol (the golden-regime change starts here)

Phase 5 opens Roadmap 2 (`docs/superpowers/specs/2026-07-10-phase-5-unquirk-training-foundation-design.md`)
and flips the regime the rest of this file was written under: from here on, goldens assert
PORT-TRUTH, not legacy-truth, because Phase 5+ tasks deliberately FIX legacy bugs instead of
reproducing them -- the "nothing here should be fixed mid-port" rule above no longer holds
unconditionally. The bit/tolerance parity proof against the resurrected 2015 production binary
is frozen first, at the `legacy-parity-v1` tag (main@74284d6), before any fix lands (see
README.md/CLAUDE.md for the pointer). Every un-quirk task from here on follows this protocol
(design spec S2), no exceptions:

1. **RED**: the existing golden/test FAILS under the fix (proof the fix is observable; a fix
   nothing catches is either untested legacy surface -- add the pin FIRST against the old
   behavior, then fix -- or not correctness-critical, drop it back).
2. **Re-pin**: the golden regenerates/re-derives to the FIXED behavior.
3. **Mutation**: revert-the-fix breaks the new golden (recorded per fix, battery-style).
4. **IMPROVEMENTS.md**: the entry flips to "FIXED (phase 5, commit `<hash>`)" keeping the
   original legacy-behavior description for the record, plus the oracle-divergence note where an
   oracle harness still describes the legacy behavior.

Oracle harnesses (the C++ `tools/oracle_harness/`, Octave `tools/octave_harness/`, Perl
`tools/perl_oracle/`, and the resurrected-binary `tools/fsp_runtime/` families) are NOT
regenerated for fixed sites -- each fix's IMPROVEMENTS entry documents the now-deliberate
divergence (the harness still describes the LEGACY behavior; the port has left it on purpose).
Entries below with no "FIXED (phase 5, ...)" flip are unchanged: reproduced bit-exactly on
purpose, either kept-documented by the Phase 5 sweep's own adjudication or not yet swept.

## Legacy Quirks (reproduced bit-exactly; revisit after parity)

- **[0a] adim bias restore vs MATLAB FP path** (`config.rs` `apply_adim`, `weight_bridge.py`): we
  restore the bias to its ORIGINAL value; legacy `config2network.m` computes `(x/adim)*adim`, which
  differs by up to 1 ULP on ~34 biases. We chose original-restore so Rust == Python. *Fix candidate:*
  once the Phase-4 inference golden lands, decide whether strict legacy parity requires replicating
  `(x/adim)*adim` on the bias.
- **[0b-i] CostLaw double-read -- kept by decision, phase 5** (`cost.rs` from `CostLaw.cpp:68-69`): `CostLawThresh{Speech,NoSpeech}`
  is read TWICE -- clamped (`1e-6..1-1e-6`) for the law coefficients, then re-read RAW/unclamped
  (defaults `10.0`/`-1.0`) for the runtime branch predicate. Almost certainly an unintended
  double-read; load-bearing for the sweep goldens. *Fix candidate:* unify to a single, clamped
  threshold after parity. **Phase 5 Task 2 adjudication:** reviewed as a candidate against the
  S0.3 correctness-critical criterion and explicitly KEPT-documented (not correctness-critical
  in the wrong-result/UB sense; a load-bearing convention) -- see the KEEP list in
  `docs/superpowers/plans/2026-07-10-phase5-fixlist.md`.
- **[0b-i] LogLaw cost/deriv Adim asymmetry -- FIXED (phase 5, F7, commit `09151ed`)** (`cost.rs`
  `Law::deriv` from `CostLaw.h:119-149`): LEGACY behavior (recorded): `cost()` divides `y` by `Adim`
  before clamp+log (`b + A*ln(clamp(y/Adim, 1e-24, 1))`); `deriv()` clamped the RAW `y` (not divided)
  then returned `A/y` UNCONDITIONALLY. Where the forward's argument is clamped the cost is CONSTANT, so
  the true gradient is 0, but the legacy `A/y` was NONZERO there -- a genuine wrong gradient reachable on
  every real `CostLaw = log` config (the `1_worker_1.config`/`LID_BLSTM.config`/twin configs all use
  log/log). **FIX (phase 5, F7):** `deriv` for `Log` now computes the derivative of the CLAMPED forward:
  `z = y/Adim; if z < 1e-24 || z > 1.0 { 0.0 } else { a/y }`. The interior value is UNCHANGED (`a/y`
  equals the true `d/dy[b + A*ln(y/Adim)]` because the `Adim` cancels via the chain rule -- the legacy
  was only ever wrong in the saturated region, not the interior). This is a Rust seam-side fix only:
  `engine.py`'s cost assembly reads the derivatives BACK from the engine (`weights_derivatives`) and
  never reimplements `deriv` -- it is the single source of the gradient consumed by the SMORMS3 loop, so
  no Python change was needed (verified).
  **RED / re-pin:** `phase3_costlaw_backward_golden::scalar_delta_log_sqrt_canary` went RED at
  output=0 -- but ONLY on the SIGN OF ZERO (`+0.0` fixed vs the harness's `-0.0` = its bogus `A/1e-24`
  times the `+0.0` logistic fold). On the committed VAD grids (k/64 and the 1000-point sweep) the
  divergence lands ONLY at output -> {0,1}, where `compute_unitary_delta`'s logistic fold
  `output*(1-output)` zeros the contribution -- so every nonzero magnitude stays bit-exact and the only
  golden movement is a training-neutral signed zero (SMORMS3 treats +0.0/-0.0 identically). Re-pinned by
  normalizing signed zeros in the canary (`x + 0.0`, bit-identity for every other f64) and by a NEW
  off-grid directed test `phase0b_costlaw::log_deriv_consistent_with_clamped_forward` (pins `0.0` in the
  lower- and upper-clamp saturated regions with the legacy-would-be value computed inline for
  non-vacuity). `deriv_sweep_bit_exact` (linear law) and the `log_speech_cost_and_deriv_asymmetry`
  interior point are UNAFFECTED (both stay bit-exact -- the interior is unchanged).
  **Mutation:** reverting `deriv` to the unconditional `A/y` makes `log_deriv_consistent_with_clamped_forward`
  FAIL at the lower-clamp point (verified: `assertion left == right failed`, delta ~ -0.05 vs 0.0).
  **Oracle divergence:** the C++ `tools/oracle_harness/` stays LEGACY -- it still dumps `A/y` in the
  saturated region; the committed fixtures never sample it (they only reach output -> {0,1} where the
  fold zeros both), so they were NOT regenerated. Deliberate, per the S2 oracle-divergence protocol.
- **[0b-i] AboveThreshCubic name-dependent coefficients -- kept by decision, phase 5** (`cost.rs` from `CostLaw.h:84-117`): the
  above-threshold cubic law switches its `A`/`B` coefficient formulas on the law-name STRING
  (`square`/`cubic` vs `linear`/`log`). Fragile and surprising. *Fix candidate:* refactor to explicit
  per-law types after parity. **Phase 5 Task 2 adjudication:** part of the cost-law
  derivative-consistency family the sweep confirmed REACHABLE (real configs set `CostLaw=log`)
  but scoped OUT -- a retrain-affecting semantic change belonging in a dedicated Phase 6
  cost-law-correctness pass, not this sweep's narrower wrong-result/UB fix list (spec S0.3
  explicitly names the cost-law family documented-not-fixed). See
  `docs/superpowers/plans/2026-07-10-phase5-fixlist.md`'s "Plan corrections" item 5
  ("CONSIDERED, SCOPED OUT").
- **[0b-i] Softmax `compute_deltas` ponderation-scaling asymmetry -- kept by decision, phase 5** (`cost.rs` `compute_deltas`, from
  `CostLaw.cpp:358-419`): the WER path scales EACH element by its own class's ponderation
  (`_ClassesPonderations(0,kk)`, per `kk`); the non-WER path instead captures a single ponderation
  from the frame's on-class column and scales the WHOLE frame row by it (`deltas.row(jj) *=
  ponderation`, `CostLaw.cpp:417`). The two paths disagree on granularity for no reason apparent in
  the math; reproduced verbatim (see `cost.rs` doc-comment on `compute_deltas`). *Fix candidate:*
  make the non-WER path scale per-element like the WER path, after parity. **Phase 5 Task 2
  adjudication:** the third member of the cost-law derivative-consistency family the sweep
  confirmed REACHABLE but scoped OUT to a Phase 6 cost-law-correctness pass -- see
  `docs/superpowers/plans/2026-07-10-phase5-fixlist.md`'s "Plan corrections" item 5
  ("CONSIDERED, SCOPED OUT"), same adjudication as the `AboveThreshCubic` entry above.
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
  dropped -- FIXED-BY-DESIGN (phase 5, commit `53de2ca`; no code change)** (`features/pipeline.rs` `FeatureConfig::from_legacy`, from `BLSTMSpectralSegmenter.cpp:50`):
  the legacy reads `_LTSVWindowShift` from config only `if (_LTSVWindowShift != 0.0)`, but at that point
  `_LTSVWindowShift` is a default-constructed `double` member with no in-class initializer and an empty
  ctor body -- so the branch condition reads an INDETERMINATE value (undefined behavior). The port drops
  the UB guard and reads `LTSVshift` unconditionally when the key is present (the oracle harness does the
  same, so the golden stays valid; all four variant configs supply `LTSVshift` explicitly).
  **FIXED-BY-DESIGN (phase 5 adjudication; Task 2 sweep finding, folded into the F5/F6 commit):**
  the port NEVER carried this UB. The original Phase-1 guard-drop already reads `LTSVshift`
  UNCONDITIONALLY (`FeatureConfig::from_legacy`, `pipeline.rs:219`, `get_f64(...)?` -- it ERRORS if
  the key is absent, rather than branching on an indeterminate member), so there is no
  uninitialized-member read anywhere in the port and no UB-era value that could ever go RED. The
  algo-5/6 LID drivers consume the deterministic `feature_cfg.ltsv_shift` config value
  (`lid.rs:906/1313/1704/2253`); every `lid5_*` golden already pins that deterministic branch (the
  harness itself was fixed to match -- the "UB KILL" comment below). The killed Task 7 (a standalone
  "reproduce then un-quirk the LTSVshift UB" task) therefore collapsed to this documentation
  verification: no code change, no re-pin, nothing to mutate -- the port was already correct. The
  legacy-UB description is kept below for the record. *Phase 4b Task 9 addendum --
  the UB observed LIVE:* the Task-9 harness stage drives the REAL compiled `BLSTMSpectralLID`
  (whose ctor runs the real `buildFromConf`), and the process's FIRST construction landed on a
  zeroed heap page -> the `!= 0.0` gate SKIPPED the read -> `_LTSVWindowShift` stayed `0.0` ->
  the LTSV decimation floored to 1 -> a whole-file (201-row) Algo-5 SAD segmentation, while every
  LATER construction (dirty heap) read the key and produced the deterministic 34-element/0.34s
  segmentation. Run-order-dependent segmentation from uninitialized memory, empirically confirmed.
  The harness probe now re-applies the Phase-1 adjudication (assigns the key value explicitly
  after construction, `tools/oracle_harness/main.cpp` "UB KILL" comment), so every fixture pins
  the deterministic branch the port implements.

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

- **[phase2] `NeuronLayer` output softmax overflow -- FIXED (phase 5, F8, commit `05f2aed`; port-hardening,
  golden-neutral)** (`nn/layers.rs` `NeuronLayer::feed_forward`, from `NeuronLayer.cpp:138-142`): LEGACY
  behavior (recorded): `lastLayer && O>1` computed `exp(a+b)` directly on the raw pre-activation values
  with NO `- max(row)` shift, so a large pre-activation overflowed `exp` to `inf` and the `inf/inf`
  row-sum quotient was `NaN`. **FIX (phase 5, F8):** an OVERFLOW-GUARD-ONLY stabilization -- the per-row
  max is subtracted before `exp` ONLY when it exceeds `EXP_OVERFLOW_GUARD = 700.0` (`exp(x)` is finite
  iff `x <= ln(f64::MAX) ~ 709.78`; 700 leaves headroom for the O-term row sum). Below the threshold the
  code path is LITERALLY `pre_act.exp()`, byte-identical to the legacy; above it the shift makes the row
  finite and correct (softmax is shift-invariant). The row-sum stays a SEQUENTIAL ascending-column loop
  (`:139`) and the per-COLUMN `cwiseQuotient` (`:140-142`) is unchanged. This is a from-scratch-training
  NaN-proofing (the modern loop, F8's motivation): no committed fixture overflows, so the fix is
  golden-neutral -- NO RED, ZERO re-pins, the FULL `cargo test` stays green (asserted). Pinned by
  `phase2_layers_golden::dense_softmax_overflow_guarded` (a dedicated overflow-input case: a row with
  both logits > 700 that the legacy would return NaN for now yields the correct finite shift-invariant
  softmax, while a moderate row in the same call stays bit-identical to the raw path -- proving the guard
  is per-row and the sub-threshold path is untouched). The harness NN_TOL probe (ascending-loop reimpl
  == real compiled `feedForward`, `max_ulp=0`) is unaffected: it never probed the overflow regime, and
  the guard is inert below the threshold. *Fix candidate (superseded):* the earlier note proposed an
  UNCONDITIONAL `- rowwise().maxCoeff()`; the guarded form was chosen instead so no existing golden moves.

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
  production caller). [Superseded: see the [phase4b] closure entry - the phSeq corpus gives it a live
  production caller since Task 5/7.]

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
  no reference) -- FIXED (phase 5, commit `53de2ca`, F6)** (`engine/bag_of_processors.rs::segmentation_function`, from
  `Segmentation.cpp:745-806`): `load_ref_from_csv` loops over `_ChannelNb` and builds the file-to-open
  string `buf` ONLY in the `_ChannelNb == 1` branch (`:748-749` `buf << filename`); the
  `_ChannelNb == 2` branch (`:750-752`) emits a "wrong path" LOG line and NEVER writes `buf`, so
  `ifstream(buf.str())` opens the empty string, fails, and the `else` body that pushes `_Reference` +
  parses lines is skipped for EVERY channel. Net effect: a `.csv` reference is honored only for
  mono audio; for stereo (or any `_ChannelNb != 1`) the reference is silently empty, so a scored
  stereo run with a CSV reference would hit the mandatory-reference bail (`:302-305`). The port
  reproduced this until Phase 5: CSV built a reference only when `channel_count == 1`; otherwise
  `None`. A scored stereo-CSV run went silently unscored (bailed on the mandatory-reference gate),
  a wrong RESULT the Phase-5 sweep promoted to a FIX.
  **FIX (phase 5, F6):** the CSV match arm now fires for EVERY channel count. CSV is
  channel-independent (unlike STM, there is no per-channel column), so the file is parsed ONCE and
  the segmentation cloned per channel, matching how STM references already load; `nb_words` (the
  dead WER surface, kept by decision, phase 5) stays the single parse's count, shared across channels. The mono
  (`channel_count == 1`) path is byte-identical to before (`vec![seg]` == one clone). *Oracle-
  divergence note (protocol step 4):* the C++ oracle harness and the resurrected 2015 binary still
  drop the stereo reference; the port diverges by design. *Pin-old-first + re-pin:*
  `stereo_csv_reference_loads_per_channel` (`tests/phase4a_segfn.rs`) first pinned the current
  mandatory-reference BAIL on a stereo-CSV corpus (RED once the fix landed), then re-pinned to the
  fixed behavior -- the scored run succeeds on BOTH channels and its cols 0-2 (Pfa/Pmiss/global)
  match an independent `load_ref_csv` + `compute_errors` oracle per channel, with a non-vacuity
  guard that the CSV reference carries a real SPEECH span. *Mutation:* restoring the
  `channel_count == 1` gate re-introduces the bail -> RED, then restore. *Cascade:* none -- no
  committed corpus golden uses a CSV reference on stereo audio (the `.csv` fixtures under
  `tests/reference_data/` are listing/mapping files, not reference segmentations); mono-CSV goldens
  are unaffected (byte-identical path).

- **[phase4a] `.trs` reference loader unported; `segmentation_function` bails on a TRS reference**
  (`engine/bag_of_processors.rs::segmentation_function` + `extension_of`, from `Segmentation.cpp:101-109`
  `load_ref_from_trs`): the legacy `Segmentation` ctor dispatches any non-`.stm`/`.csv`/`.xml`
  reference extension (and any name too short for an extension) to `load_ref_from_trs` (a Transcriber
  `.trs` XML parser). That loader is not ported (`segmentation_io.rs` carries STM/CSV/VRCTS only), so
  the reference dispatch here `bail!`s on a TRS reference rather than silently producing an empty
  reference. *Why deferred:* the Phase 4a parity corpora use STM (SAD) and CSV (WER) references only;
  TRS reference inputs were never exercised (the `.xml`/VRCTS half of this dispatch is now wired --
  see the CLOSED (phase 6) entry directly below). *Fix candidate:* port `load_ref_from_trs` when a
  corpus needs it. *Pinned by:* the `RefExt::Trs` bail path (inline in `segmentation_function`).

- **[phase4a] CLOSED (phase 6): `.xml` (VRCTS) reference loading wired into the reference dispatch**
  (`engine/bag_of_processors.rs::segmentation_function` + `extension_of` +
  `tasks/segmentation_io.rs::load_ref_vrcts`, from `Segmentation.cpp:89-100` (the ctor `.xml` branch)
  + `:808-829` `load_ref_from_vrcts`): Phase 6's SAD track trains on the corpus `.part.xml` VRCTS
  references (`derive_sad_listings`'s wav/xml pairs); before Task 2b, `extension_of` classified `.xml`
  as `RefExt::Trs`, so every scored (`-m`/`-t`) run on a `.xml` reference hit the TRS bail above.
  *Fix (Task 2b):* `extension_of` now maps `.xml -> RefExt::Xml`, and the new `(RefExt::Xml, ...)`
  dispatch arm builds the per-channel reference via a new faithful loader
  `load_ref_vrcts(text, chan, off, dur)`. **Source-governed distinction:** the ctor `.xml` branch
  calls `load_ref_from_vrcts` (`:808-829`), the REFERENCE loader -- a DIFFERENT legacy function from
  `load_from_vrcts` (`:592-614`, the DUMP parse-back the port's `load_vrcts` mirrors for `VrctsPart`).
  The two share the line format but differ in two load-bearing ways, so `load_vrcts` was NOT reused:
  (1) the reference loader is CHANNEL-SLICED by the 1-based `ch="N"` attribute (`--chan`;
  `_Reference.at(chan)` guarded by `chan < _ChannelNb`), ported as a per-channel equality
  (`ch-1 == chan`, the same shape as `load_ref_stm`'s `line_chan == chan` gate -- calling it once per
  channel reconstructs the legacy single-pass multi-channel fill), whereas `load_vrcts` ignores `ch=`
  and dumps every segment into one caller-passed channel; (2) it seeds the segmentation `End` sentinel
  at `_AudioDuration` (the audio frame count, passed as `dur`), NOT the embedded `<Channel sigdur>` (a
  corpus `.part.xml` carries e.g. sigdur=1721.62s while the audio is `_DurationMax`-capped) --
  `load_vrcts` uses sigdur for the extent, correct for the dump round-trip but wrong for a reference
  scored against a capped-audio hyp. `nb_words` stays the -1 default (WER Pass 1 suppressed), like STM.
  The only deviations from the legacy reference loader are the same benign ones `load_ref_stm` already
  carries: the display-only `_RefCount`/percentage log (`:824-828`) is omitted, and a malformed
  `ch="0"` (`ch-1 == -1`) is dropped rather than hitting the legacy `_Reference.at(-1)` UB. *Pinned by:*
  `tests/phase6_xml_ref.rs` -- `xml_reference_scored_run_through_dispatch` (RED: the pre-wiring TRS
  bail; GREEN: a scored 2-channel `-m` run whose cols 0-2 match an independent `load_ref_vrcts` +
  `compute_errors` oracle per channel, with non-vacuity + nonzero-error guards),
  `xml_reference_is_channel_sliced` (chan 0 carries only the `ch="1"` span, chan 1 only `ch="2"`),
  `xml_reference_windows_on_audio_duration_not_sigdur` (the `End`-sentinel divergence vs `load_vrcts`),
  and a corpus-gated smoke `xml_reference_real_corpus_part_xml` (a real LRE03 `.part.xml` + wav scored
  end-to-end; skips cleanly when `data/LRE03-LRE07` is absent). *Mutation:* reverting the
  `.xml -> RefExt::Xml` arm re-introduces the TRS bail -> RED, then restore. *Cascade:* none -- no
  committed golden used a `.xml` reference (the TRS bail was the only prior behavior), and `load_vrcts`
  / `VrctsPart` are byte-untouched. *Oracle:* no C++/Octave harness covered the reference-load path
  (`load_ref_from_vrcts` was never exercised; the harness pins the writer `toFile_VRCTS` and the dump
  loader `load_from_vrcts`), so this is pinned by synthetic + corpus-gated fixtures, not a regenerated
  golden.

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
  331,373` `tmp.push_back(seg._WordErrorRate[chan]._NbWords)`): the legacy result
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
  call's row-slice write (`cost_lid_row`) still carries the pre-gate value. *Phase 4b Task 9
  addendum -- the gate is now LIVE and pinned end-to-end:* the `twin_train_ns` fixture (Mode-7
  phSeq corpus whose rising threshold 11 exceeds the constant-10 result_vec -> zero speech every
  epoch) forces `costLID = -1.0` into every real `updateWeightsLID`; the committed
  `twin_train_nsx_lid_epoch*.bin` counterfactual (same harness run with the gate DISABLED)
  diverges from the gated trajectory at the manifest-recorded epoch 5 (`corpus_lid.measured.
  ns_gate_diverge_epoch` -- the first epoch where the ungated costLID RISES, so Rprop's
  cost-gated backtrack fires only in the ungated world). *Pinned by:*
  `twin_train_ns_costlid_gate_live` (`tests/phase4b_corpus_lid.rs`): the replay must equal the
  ns goldens AND differ from nsx at epoch 5, and `CostLIDMem` must carry the PRE-gate values.
  Mutation (verified): disabling the `save_and_update` gate (`if false && ...`) makes the test
  fail at exactly `epoch 5 weight 167` -- the port then reproduces the counterfactual.

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
  spectral goldens; the target reuse is nonetheless wired for that path.) **UPDATE (Task 10):** the
  `phase2b_spectral_golden.rs` pitch goldens all pass `refs: None` (an empty target -- the reuse is
  wired but numerically inert). `phase4b_backlog.rs::pitch_pass_target_reuse_under_live_reference`
  closes that gap with a LIVE STM reference (`f1.wav`/`f1.stm`), confirming `target.nrows() > 0`
  flows into BOTH passes' `feed_forward_backward` (live cost/counter accumulation on the real
  production driver) and pinning the pass-2 result row against a new harness dump
  (`spectral_pitch_scored_result_pass2_chan1.bin`). Note: reuse-vs-rebuild is NOT an observable
  distinction on this config at all -- not via `result_vec`, and not via cost either.
  `Segmenter::get_targets` (`segmenter.rs`, from `Segmenter.cpp`) is a pure function of
  `(reference, timeStep, timeOffset, backPropWer, classType, nRows)`; `BLSTMSpectralSegmenter.cpp
  :724-733` computes `timeStep`/`timeOffset` exactly ONCE, before the `:735-738` target build and
  before the pitch block, and the pitch warp (`:760-775`) preserves the periodogram's row/column
  shape (confirmed in the ported `apply_homothety`), so `nRows` and every other argument a
  hypothetical rebuild at `:793` would pass to `getTargets` are IDENTICAL to pass 1's -- the
  rebuilt target would be bit-identical to the reused one, hence so would the resulting cost. The
  golden closes the "numerically inert target" gap (a live reference now drives real, live cost/
  counter accumulation through both passes on the production driver); the `:793` reuse-vs-rebuild
  question itself is structurally pinned by transcription (a verbatim port of the legacy line, not
  a port-invented shortcut) and is observationally indistinguishable from a rebuild for this
  driver/config -- see the Task 10 backlog entry above for the full analysis. **Correction:** the
  `0f7ab42` commit message's "All three mutation-tested (apply/run/revert)" overclaims coverage for
  this item -- item 3 (this entry) was closed by DIRECT ASSERTION (live nonzero cost/counter values
  on the production driver + the bit-exact pass-2 golden), not by a mutation. Per the analysis just
  above, an apply/run/revert mutation swapping the `:793` reuse for a rebuild would be numerically
  inert on this config (bit-identical target in, bit-identical cost out), so it could not have
  demonstrated anything; no such mutation was run, and none would be meaningful here.

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

- **[phase4a] CLI `--key=` empty-value override REMOVES the key -- a port-only DEVIATION from the
  literal (unreachable) legacy behavior** (`cli.rs::apply_override`): the legacy usage text
  (`FastSpeechProcessing.cpp:85`, `setting <variable_value> = "" removes the variable from the
  config`) documents this, but `ConfigFile::set_val` (`ConfigFile.h:34-42`) never erases -- it
  unconditionally does `_Params[name] = ss.str()` for any value including `""`. The only erase in
  `ConfigFile` is `warn_unused`'s `_Params.erase` (`ConfigFile.cpp:43`), gated on a `_Used` set whose
  single insertion site is commented out everywhere (`ConfigFile.h:39,47,56,65,75`) -- so even that
  path is dead by construction -- and `warn_unused` is never called from `main()` (`:31-75`) at all.
  Separately, `--key=` with a GENUINELY empty value can't even reach `set_val`: `split<string>(argv,
  '=', 2)` (`String.hpp:86-98`) uses `getline(ss, s, '=')`, which on `"--key="` yields a 1-element
  vector (no trailing empty field emitted at EOF) -- `argument[1]` is an out-of-bounds
  `std::vector::operator[]` read, undefined behavior, verified against a standalone repro of the exact
  `split` template. The usage string is therefore vestigial/aspirational documentation for a feature
  that is not wired to anything reachable from the legacy binary's `main()`. This port implements the
  DOCUMENTED (not the literal) behavior -- `IndexMap::shift_remove(key)` on an empty override value --
  because the task spec calls for defined, usage-text-honoring semantics rather than reproducing C++
  UB. *Why deferred:* a deliberate port-only choice (no legacy golden can pin either interpretation,
  since the real binary cannot reach this path without crashing). *Pinned by:*
  `phase4a_cli.rs::override_empty_removes_key`.

- **[phase4a] Mutation battery (Task 11): load-bearing goldens confirmed, two coverage gaps found.**
  Six targeted production mutations, each applied/run/reverted/re-run in isolation (named suite
  only, full `cargo test` once at the end): (1) fold-order reversed (`corpus_processor.rs::run_epoch`,
  `per_file.sort_by_key` ascending -> `Reverse`) against `phase4a_train_golden` -- did NOT fail (the
  tier-2 corpus has exactly 2 files and `numOuterThreads 1` clamps to one lane, so the fold is a
  2-element commutative `+=`/`InputStatistics::update` merge -- order-insensitive per the spec S7
  caveat; a 3+-file crafted-divergence golden is future work, not faked here). (2) best-cost gate
  `>` -> `>=` (`bag_of_processors.rs::save_weights`, Spectral arm) against `phase4a_train_golden`
  AND `phase4a_save_update::best_cost_gate_fires_and_skips` -- did NOT fail either: both suites only
  exercise strict improve/strict-worse costs, never an exact tie, so `>=` is behaviorally identical
  to `>` on every existing case -- a genuine golden coverage gap (no crafted-tie test exists) reported
  honestly, not faked. (3) cost column `sums[4]` -> `sums[5]` (`bag_of_processors.rs::save_and_update`)
  against `phase4a_tier1_e2e::train_ltsv_two_epochs_matches` -- FAILED as expected
  (`train CostMem[0,0]: a=12 (want) b=0 (got)`); reverted, PASS. (4) counter-normalization guard
  dropped to unconditional divide (both the cost and costLID guards) against
  `phase4a_save_update` -- FAILED as expected (`aggregation_counter_zero_guard_skips_division`:
  `left: inf, right: 7.0`, div-by-zero; 2 more tests failed downstream from the same cause);
  reverted, PASS. (5) gradCheck epsilon sign flipped (asymmetric: `+=epsilon` -> `-=epsilon` on the
  first perturbation only, since negating `epsilon` uniformly at the call site is a no-op by central-
  difference symmetry) against `phase4a_gradcheck_golden` -- FAILED as expected (`weight 0 numerical`
  bit mismatch from the broken +/- symmetry); reverted, PASS. (6) `costLID = -1.0` gate removed
  (`bag_of_processors.rs::save_and_update`, the `:465` override) against
  `phase4a_save_update::update_called_with_neg_costlid` -- FAILED as expected (`left: Some(3.0)`
  real costLID leaking through `right: Some(-1.0)` expected gate); reverted, PASS. This mutation was
  originally slated "defer to 4b" in the plan, but Task 6's `update_called_with_neg_costlid` hook
  makes it directly observable in 4a, so the stronger result (FAILS, not vacuously unreachable) is
  recorded here instead of deferred. *Lane N=1 vs N=2 measurement* (throwaway test, not committed):
  a 3-file corpus (`excerpt_2ch_8k.wav` x3) driving `signal.config` (Algo 4) with the noOverlap
  `BLSTM_window 0.5` / `BLSTM_shift 0` override at `numOuterThreads` 1 vs 2 -- outputs DIVERGED as
  expected (`MultiConfigResults` columns 6-7, the timing/cost-adjacent columns, differ across all 3
  file-rows; `CostMem`/`BadClassifMem`/etc. stayed identical since Solo mode never calls
  `saveAndUpdate`). Confirms genuine cross-file lane state chaining: N=1 chains all 3 files in one
  lane (f0->f1->f2), N=2 partitions lane0={f0,f2}/lane1={f1}, so f2 inherits poisoned
  `_WindowShift` state directly from f0 under N=2 but from f1 under N=1 -- a real, expected
  N-dependence per the `[phase4a] Static-lane deterministic reduction` entry above, not a bug.
  *Net verdict:* 4 of 6 mutations break their named golden as designed; 2 (fold-order, best-cost
  tie) expose real gaps in tie/multi-contribution coverage on the current 2-file tier-2 fixture,
  recorded honestly rather than papered over. Full detail (diffs, commands, output) in
  `.superpowers/sdd/task-11-phase4a-report.md`. **UPDATE (Task 10):** both flagged gaps are now
  CLOSED. The best-cost tie gap is closed by `phase4b_backlog.rs::best_cost_gate_skips_on_exact_tie`
  (a crafted byte-equal-cost epoch B against the same `Processor::Spectral` `save_weights` gate);
  *mutation evidence:* `best_cost[&pos] > save_criterion` -> `>=` makes it FAIL
  (`assertion failed: !weights_artifact_b.exists()`, the tie now re-saves); reverted, PASS. The
  fold-order gap is closed by `phase4b_backlog.rs::fold_order_ascending_golden` (a 3-file corpus
  whose per-file derivative fold is measurably order-sensitive, per
  `fold_order_measured_divergence_seed0`: 5865 of 67342 raw-derivative elements differ in bits
  between ascending and descending fold order); *mutation evidence:*
  `run_epoch`'s `per_file.sort_by_key(|(j, _, _, _)| *j)` -> `Reverse(*j)` makes it FAIL
  (`epoch-0 (ascending fold) raw deriv [3,0] bit mismatch`); reverted, PASS. See the Task 10
  backlog entry below for the full write-up. **Correction:** only the fold-order item pins the
  RAW folded derivative matrix, not the post-Rprop weights, since iRPROP- only reacts to
  derivative sign and this fold order's divergence is too small to flip one; the best-cost-tie
  item pins a different observable entirely (the save-gate's weights-artifact existence, via
  `!weights_artifact_b.exists()`), not the derivative matrix.

- **[phase4a] Phase 4b test backlog (from the 4a final review) -- Task 10 CLOSED all three items.**
  (1) CLOSED: a crafted best-cost TIE golden (epoch B's cost BYTE-EQUAL to epoch A's, not merely
  worse) closing the `>` vs `>=` gate coverage hole. *Pinned by:*
  `phase4b_backlog.rs::best_cost_gate_skips_on_exact_tie`. *Mutation evidence:* `save_weights`'s
  `Processor::Spectral` arm `best_cost[&pos] > save_criterion` -> `>=` makes the test FAIL
  (`assertion failed: !weights_artifact_b.exists()`, the tie now re-saves); reverted, PASS.
  (2) CLOSED: a 3-file (`f1`/`f2`/`f3`) fold-order-divergence golden -- MEASURED, no seed search
  needed (bound: 1 of a budgeted 20; the base `NNweights_config1.bin` seed diverges immediately,
  5865 of 67342 raw-derivative elements differ in bits between the ascending and descending fold).
  *Pinned by:* `phase4b_backlog.rs::fold_order_measured_divergence_seed0` (the measurement) and
  `fold_order_ascending_golden` (the locked ascending-order golden). *Non-obvious finding:* the
  golden must pin the RAW folded derivative matrix (a new test-support-only observation point,
  `CorpusProcessor::epoch_raw_derivs_trace_for_test`, captured immediately after `run_epoch`'s
  per-file fold, BEFORE `save_and_update_epoch`), NOT the post-Rprop trained weights
  (`epoch_weight_trace_for_test`) -- iRPROP- (`nn/train.rs`) reacts only to the SIGN of the
  count-normalized derivative, so this fold order's ULP-scale divergence never flips a sign and is
  therefore INVISIBLE in the trained weights; pinning `epoch_weight_trace_for_test` instead was
  tried first and did NOT fail under the mutation below (a false negative), which is why the
  raw-derivs observation point was added. *Mutation evidence:* `run_epoch`'s
  `per_file.sort_by_key(|(j, _, _, _)| *j)` -> `Reverse(*j)` makes `fold_order_ascending_golden`
  FAIL (`epoch-0 (ascending fold) raw deriv [3,0] bit mismatch`, off by 1 ULP); reverted, PASS.
  (3) CLOSED: a `TDCwindow 0.032` pitch-pass golden (`f1.wav`/`f1.stm`, the phase4a 3-file corpus)
  driven with a LIVE STM reference (`refs: Some(&refs)`), pinning `last_result_rows_pass2` against
  a new harness dump (`spectral_pitch_scored_result_pass2_chan1.bin`, `tools/oracle_harness/
  main.cpp`'s segmenter-level `transcribeSpectral` reused unmodified with a real
  `Segmentation::Segmentation(AudioStruct&, double)` STM auto-load, no lambda change). *Pinned by:*
  `phase4b_backlog.rs::pitch_pass_target_reuse_under_live_reference`, which additionally asserts
  live (nonzero) `cumulative_error`/`nb_of_classif` on the REAL Rust production driver -- the
  harness's own `cumError_chan1`/`nbClassif_chan1` read 0 by construction (its reimpl transcription
  never wires cost through the target, and `result_vec` is target-INDEPENDENT given
  `TargetEnforcementStep >= 0`; see `scripts/extract_phase4b_fixtures.py`'s `pitch_scored` manifest
  text), so the harness fixture pins the bit-exact NN forward-pass row only, while the live-cost
  claim is asserted on production code.

- **[phase4b] `PrintConfusionMatrix`'s `posTarget`/`posBestNotTarget` are STICKY across rows, NOT
  reset per row -- FIXED (phase 5, commit `53de2ca`, F5, BOTH languages)** (`engine/confusion.rs::
  confusion_from_results` + `scoring.py::confusion_matrix`, from `BagOfProcessors.cpp:
  509-510,533-534` and the sibling `confusionThresh.m`): both position variables are declared OUTSIDE the row loop and initialized to
  `0` exactly ONCE, before row 0; the end-of-row reset (`:533-534`) touches only
  `scoreTarget`/`maxScoreNotTarget`. So a row with no target sentinel at all (every column `<=
  150`) does NOT attribute its miss to `posTarget = 0` (the index-header slot) in general -- it
  inherits `posTarget` (and, unless overwritten by a competitor score that row, `posBestNotTarget`)
  from the LAST row that set them. Only a no-target row that is ALSO the very first row ever
  processed lands at index `0`. Almost certainly an oversight (the natural reading of the code is
  "reset per row"), but load-bearing for the confusion-matrix goldens as committed. Caught by a
  genuine TDD RED: the first hand-derived `sentinel_decode_and_argmax` expectation assumed a
  per-row reset and failed against both the Rust implementation (written directly from source) and
  the independently cross-validated harness dump, forcing a re-read of `:509-510` vs `:533-534`.
  PRECISION (T6 review): the quirk was LATENT on all real Algo-5/6 data -- both drivers clamp
  `targetIndex` to `[0, classNb-1]` before the `targetLID = -2.0` write, so EVERY real result row
  carries exactly one `>150` sentinel and sticky-vs-per-row decode coincide; an actual out-of-set
  file is misattributed to class 0, never rendered no-target. The fix is degenerate-case
  correctness + phase-6 insurance, not an active-corruption repair.
  *Why deferred:* provenance; the confusion matrix's row/col attribution for degenerate (no-target)
  rows is directly observable and load-bearing for any LID confusion-matrix consumer built on this
  in Phase 4b's later tasks. This corrupts phase-6 EER/DCF (row/col attribution depends spuriously
  on file order), so the Phase-5 sweep promoted it to a FIX.
  **FIX (phase 5, F5):** the four decode accumulators (`score_target`/`max_score_not_target`/
  `pos_target`/`pos_best_not_target`) are now declared INSIDE the row loop in BOTH ports, so no row
  inherits a prior row's index; the fix is IDENTICAL across the two (the Rust `PrintConfusionMatrix`
  variant and the Python `confusionThresh.m` variant differ only in their win/hit predicate, which
  is unchanged). **No-target-row semantics -- adjudicated:** the naive "reset to 0" prescription
  would credit a no-target row to `pos_target = 0`, the index-HEADER row/col (e.g. the pre-fix
  `no_target_as_first_row` test showed label `2.0 -> 3.0` pollution). Instead, a row with NO in-band
  target (`pos_target == 0`, no column `> 150`) is SKIPPED entirely: it is an out-of-set LID trial
  with no true class in the closed set, so it belongs in no cell and no total (routing it to the
  aggregate row `class_nb+1` was rejected -- the aggregate row/col index collides, double-counting
  the col-total, and fabricates a prediction attribution for an unknown true class). Skipping is
  order-independent, pollutes nothing, and is the standard closed-set treatment of out-of-set
  trials; downstream `confusion_error` reads only per-class diagonals + row totals, which skipping
  leaves correct. *Oracle-divergence note (protocol step 4):* the C++ `phase4b_confusion` harness
  stage and the Octave `confusionThresh.m` still encode the legacy sticky matrix; the port
  deliberately diverges. *Re-pinned to port-truth:* `sentinel_decode_and_argmax`,
  `no_target_row_skipped_never_pollutes_header` (renamed from `no_target_as_first_row_uses_header_
  zero_slot`), `cross_language_identity_no_target_skip` (new, mirrors the Python twin)
  (`src/engine/confusion.rs`); `confusion_matrix_matches_harness_transcription` +
  `confusion_matrix.bin` RE-DERIVED port-side (`tests/phase4b_confusion_golden.rs`);
  `confusion_error.bin` is UNCHANGED (the error aggregate is invariant -- the skipped no-target row
  only ever hit an off-diagonal cell, never a diagonal). Python: `test_confusion_matrix_no_target_
  row_skipped` (renamed from `test_confusion_matrix_sticky_pos_target_quirk`) +
  `test_confusion_cross_language_identity_no_target_skip` (new, byte-identical matrix to the Rust
  twin -- no PyO3 seam exists for confusion, so the identity is pinned by mirrored hardcoded
  matrices) (`tests/test_phase4c_scoring.py`). *Mutation battery:* (M1) drop the `pos_target == 0`
  skip -> a no-target row pollutes the header, RED in both languages; (M2) move the decls back
  outside the loop (restore stickiness) -> RED in both languages; both restore green. The LID
  DRIVER per-file confusion (`tasks/lid.rs`, `_LIDSegmentsConfusion`) is a SEPARATE path (it knows
  the target index `ti` directly, no sticky sentinel scan, no no-target case) and is UNAFFECTED;
  the all-target corpus goldens (`lid5_*`/`twin_mode*`/`mode7_*`/`twin_train_epoch_weights_golden`)
  never hit a no-target row, so they stay green.

- **[phase4b] An exact score tie sends the target to the MISS branch (strict `>` only) -- kept by decision, phase 5**
  (`engine/confusion.rs::confusion_from_results`, from `BagOfProcessors.cpp:524`): the win
  condition is `scoreTarget > maxScoreNotTarget`, so `scoreTarget == maxScoreNotTarget` (the
  decoded target score exactly equals the best competitor's raw score) falls to the
  best-competitor/miss branch, not a coin-flip or a separate tie bucket. *Why deferred:*
  provenance; a `>=` flip is a plausible "intended" reading but changes which class gets credited
  on every exact tie. *Fix candidate:* none identified without knowing the original intent. *Pinned
  by:* the row-D case (an exact 50.0/50.0 tie) in `sentinel_decode_and_argmax`
  (`src/engine/confusion.rs`); a `>` -> `>=` mutation was run and confirmed to break this test (see
  the Task 1 report). **Phase 5 Task 2 adjudication:** reviewed alongside F5's sticky-index fix
  (the SAME function) and explicitly KEPT -- matches the legacy exactly, no clear "correct"
  alternative, and distinct from the order-dependence bug F5 actually fixes; see
  `docs/superpowers/plans/2026-07-10-phase5-fixlist.md`'s KEEP list ("confusion family, non-sticky").

- **[phase4b] `Confusion2String`'s zero-row-total normalization guard leaves that row's diagonal
  cell as a RAW COUNT, not a percentage -- kept by decision, phase 5** (`engine/confusion.rs::confusion_error`, from
  `Helpers.hpp:375`): the per-row scaling only fires `if (confusion(kk+1, classNb+1) > 0)`; a class
  with zero samples in this crafted batch keeps its raw (unscaled) diagonal count feeding directly
  into the `100 - normalized(ii,ii)` deficit sum, which is only a sensible percentage-deficit for
  rows that DID get scaled. *Why deferred:* provenance; matches the legacy exactly and only matters
  for genuinely empty classes. *Fix candidate:* none identified -- an empty class arguably
  shouldn't contribute to the error average at all; revisit once a real (non-crafted) confusion
  matrix with an empty class is observed. *Pinned by:* `zero_total_row_left_unnormalized`
  (`src/engine/confusion.rs`). **Phase 5 Task 2 adjudication:** reviewed alongside F5 and
  explicitly KEPT -- see `docs/superpowers/plans/2026-07-10-phase5-fixlist.md`'s KEEP list
  ("confusion family, non-sticky").

- **[phase4b] `Confusion2String`'s ill-conditioned-matrix gate (`exit(1)` in the legacy) ported as
  a Rust panic -- kept by decision, phase 5** (`engine/confusion.rs::confusion_error`, from `Helpers.hpp:369-371`): `classNb =
  confusion.rows()-2 < 2`, or `rows-2 != cols-2`, is fatal in the legacy (`exit(1)`, no exception,
  no recovery). Rust has no direct equivalent that stays testable via `#[should_panic]`, so this
  port uses `panic!`. *Why deferred:* this branch is reachable only from a deliberately malformed
  matrix (never from `confusion_from_results`'s own construction, which always builds a valid
  square matrix or returns the `classNb <= 1` empty case without calling `confusion_error` at all);
  revisit if a future caller needs graceful-`Result` handling instead. *Fix candidate:* none
  identified. *Pinned by:* `ill_conditioned_matrix_panics` (`src/engine/confusion.rs`). **Phase 5
  Task 2 adjudication:** reviewed alongside F5 and explicitly KEPT -- see
  `docs/superpowers/plans/2026-07-10-phase5-fixlist.md`'s KEEP list ("confusion family, non-sticky").

- **[phase4b] `BagOfProcessors::PrintConfusionMatrix`'s classNb==2 binary ROC-curve variant and a
  duplicate normalization/print block are commented-out DEAD CODE, not ported**
  (`engine/confusion.rs`, from `BagOfProcessors.cpp:477-499,541-593`): `:477-499` is an entire
  `if (classNb == 2) { ... }` arm computing a 10001-step threshold-sweep negative/positive
  histogram -- entirely commented out, so `classNb == 2` falls straight into the SAME general
  `else if (classNb > 1)` accumulation as every other `classNb >= 2`, with no ROC/histogram
  behavior at all. `:541-593` duplicates (via `cout`, not `error +=`) exactly what the LIVE
  `Confusion2String` call at `:540` already computes and is never executed. *Why deferred:* N/A --
  this is a straightforward "do not port dead code" decision, not a deferred fix. *Pinned by:*
  `dead_binary_variant_not_ported` (`src/engine/confusion.rs`).

- **[phase4b] CORRECTED a Phase 4a stub doc-comment: `PrintConfusionMatrix`'s returned `error` IS
  the row-normalized `error/classNb` aggregate, not a raw off-diagonal count**
  (`engine/bag_of_processors.rs::print_confusion_matrix`): the Phase 4a stub's doc comment (written
  ahead of the real implementation, "verified against source in 4a" per that phase's task
  instructions) claimed the returned error was "the summed off-diagonal confusion (raw count, not
  yet normalized -- the normalization block is legacy dead code, commented out at `:534-596`)".
  Re-reading `BagOfProcessors.cpp:536-598` for Task 1 shows this conflated two different things:
  the LIVE `error = Confusion2String(confusion, output, "      ");` call at `:540` (NOT inside the
  commented range) IS what computes and returns the row-normalized aggregate (`Helpers.hpp:438`'s
  own `return error/classNb;`); only the DUPLICATE block at `:541-593` (a `cout`-based re-derivation
  of the same math, never executed) and the commented `return error/classNb;` at `:596` (which
  would have double-divided had it been live) are dead. *Why deferred:* N/A -- documentation
  correction, not a behavioral question. *Fix candidate:* N/A. *Pinned by:*
  `error_matches_real_confusion2string_and_printconfusionmatrix`
  (`tests/phase4b_confusion_golden.rs`), which asserts the Rust `confusion_error` return equals
  the REAL compiled `Confusion2String`'s (normalized) return.

- **[phase4b] `getCostPonderation`'s `classIndex < 0` clamp is DEAD in the legacy (unsigned
  `size_type` param), ported as a live `i64 < 0` clamp -- behavior-identical**
  (`nn/blstm.rs::get_cost_ponderation`, from `BLSTMNeuralNetwork.cpp:333-350`): the legacy
  parameter is `std::vector<double>::size_type` (UNSIGNED), so both branches' literal
  `(classIndex < 0)` sub-tests can never fire. The sole caller passes a signed `int`
  (`TwinBLSTMSpectralLID.cpp:1334`, `targetIndex`); a negative arg converts to a huge unsigned that
  trips the OTHER sub-test (`>= getOutputSize()` multiclass / `> getOutputSize()` binary) and clamps
  to 0 anyway. The port takes `class_index: i64` with an EXPLICIT `< 0` clamp, which yields the
  identical result for every input (a negative always lands on 0 either way), while being readable
  and not relying on unsigned wraparound. The multiclass guard uses `>=` and the binary guard uses
  `>` -- that asymmetry is load-bearing and reproduced verbatim. *Why deferred:* N/A -- an
  equivalence-preserving readability choice, not a behavioral deviation. *Fix candidate:* N/A.
  *Pinned by:* `cost_ponderation_clamps` (`tests/phase4b_nn_lid.rs`), which tables the fold for
  negative / equal-to-size / far-out-of-range indices in both branches.

- **[phase4b] `ponderateWeightsDerivatives` scales the deriv col0 only; the `_NbOfSeqFedBackward`
  count column and the mean/std tail are untouched** (`nn/blstm.rs`, `nn/network.rs`, `nn/layers.rs`,
  from `BLSTMNeuralNetwork.cpp:289-297` -> `NeuralNetwork.hpp:119-123` -> `LSTMLayer.cpp:305-310` /
  `NeuronLayer.cpp:122-125`): the legacy `_...Derivatives *= ponderation` touches ONLY the raw
  derivative accumulators (the harvested Nx2 col0). `_NbOfSeqFedBackward` (col1) is a separate member,
  never scaled, and the mean/std tail is assembled fresh as `[0, 1]` in `getWeightsDerivatives`, so it
  never participates. Not a bug -- a semantics note for the count-vs-deriv normalization asymmetry
  (spec S3). *Why deferred:* N/A. *Fix candidate:* N/A. *Pinned by:* `ponderation_scales_col0_not_col1`
  (`tests/phase4b_nn_lid.rs`, STRICT bits: col0 == col0*factor, col1 unchanged); mutation check --
  scaling col1 instead of col0, or scaling both, flips the strict-bit assertion.

- **[phase4b] `BLSTMSpectralLID::getSegmentation` inlines a spectral-setup COPY that DIVERGES
  from the base `BLSTMSpectralSegmenter::getSegmentation` in five load-bearing ways -- the Algo-5
  lines govern** (`tasks/lid.rs::get_segmentation`, from `BLSTMSpectralLID.cpp:27-460` diffed against
  `BLSTMSpectralSegmenter.cpp:593-887` + `getLTSVParam` `:300-314` + `getBLSTMParam` `:439-500`): (1)
  the SAD result_vec is the LTSV score (`classifySequence` over the RAW periodogram, `:271-274` -- the
  mel branch `:252-269` is COMMENTED OUT), NOT the BLSTM posterior; the BLSTM runs only per speech
  segment for LID scoring. (2) the LTSV half-window floors `< 1 -> 1` (`:157-158`), UNLIKE the base
  `getLTSVParam`'s `< 1 -> 0` (`:303`, which DISABLES LTSV) -- so LTSV SAD is always active even with
  `LTSVwindow 0` (`round(0) = 0 -> 1`); this is the STANDALONE `LtsvSegmenter`'s floor, not the
  spectral SAD's. (3) result_vec is sized `ceil(vec_size / LTSV_window_shift)` (`:227-232`), the LTSV
  decimation, NOT `getBLSTMParam`'s ssr-division (`:481-497`), and the LTSV loop writes DECIMATED
  indices `result_vec[jj/LTSV_window_shift]` with NO interpolation backfill. (4) the SAD
  `results2segmentation` timeStep is `_WindowShift` with offset 0.0 (`:302`) -- the BLSTM-derived
  window shift used as the LTSV-SAD time step, a compression quirk (the result_vec is LTSV-decimated
  but stepped by `_WindowShift`, not `_WindowShift * LTSV_window_shift`), so the SAD boundaries live on
  a compressed timeline. (5) the scoring timeStep uses the SIGNAL pattern (`:333-343`: overlap keeps
  `timeStep = _WindowShift`), NOT the base spectral `_SpectrumShift * ssr` (`:731`). *Why deferred:*
  the LID driver is a faithful port target; the divergences are load-bearing for the goldens. *Fix
  candidate:* after end-to-end LID parity, reconcile the LID SAD path with the base spectral/LTSV
  drivers (the timeline compression at (4) is the most surprising and worth a second look). *Pinned
  by:* `ltsv_sad_row_matches_dump`, `sad_boundaries_match_dump`, `lid_classification_errors_match_dump`
  (`tests/phase4b_lid5_golden.rs`), all bit-exact vs the harness `LidProbe` reimpl. Mutation: the
  `LidProbe` reimpl encodes the Algo-5 lines (not the base spectral machinery), so reverting any of
  the five divergences -- e.g. restoring the commented-out mel-branch SAD, or the base
  `getLTSVParam` `< 1 -> 0` floor -- diverges the `*_match_dump` goldens; the Task-11 battery's
  in-band `-2.0 -> -1.0` flip additionally breaks `phase4b_lid5_golden` (see the Task-11 entry
  below, item 1).

- **[phase4b] The LID driver computes TDC params + `LTSV_freq_beg`/`LTSV_freq_end` + `costLID` that
  are all DEAD, and has a `_CepstreCoefficients` writeback that never re-reads** (`tasks/lid.rs`, from
  `BLSTMSpectralLID.cpp`): TDC (`:162-173`, incl. a min_lag/max_lag derivation SIMPLER than
  `getTDCParam` -- no `min_lag < 1 -> 1`, no `max_lag < min_lag` guard, no `_MinMaxLag` writeback) is
  computed but never used (no pitch pass in the LID driver), so it is SKIPPED here. `LTSV_freq_beg`/
  `LTSV_freq_end` (`:109-110`) are captured before the mel branch but never read (the LTSV loop uses
  the post-mel `freq_beg/end`), so SKIPPED. `costLID` (`:320,376,405,410`) is accumulated but NEVER
  stored on `seg` or returned (a dead local), so NOT reproduced. The `_CepstreCoefficients` block
  writeback of the (type-1-normalized) input after each scoring call (`:364-368`) is DEAD: SAD
  segments after smoothing are separated by an OTHER span, so `rowEnd_i < rowBegin_{i+2}` always (the
  written rows are never re-read), and under `lid5.config`'s single-segment SAD it writes once and
  never re-reads -- SKIPPED. *Why deferred:* dead-code omission, not a behavioral deviation. *Fix
  candidate:* N/A. *Pinned by:* `lid_members_match_dump`/`lid_confusion_matches_dump`
  (`tests/phase4b_lid5_golden.rs`), bit-exact despite the skips. Mutation: N/A -- these are
  dead-code omissions (computing and storing the dead TDC / `costLID` / cepstre values changes
  nothing the goldens observe), so there is no live branch to break.

- **[phase4b] The LID `_IsLIDCorrect = -1` branch is DEAD (targetIndex is clamped `>= 0`), and the
  per-segment scoring-block guard adds a `rowEnd >= rowBegin` underflow check with no legacy
  counterpart** (`tasks/lid.rs::get_segmentation`, from `BLSTMSpectralLID.cpp:321-323,352,424-425`):
  `targetIndex` is clamped to `[0, classNb)` at `:321-323`, so every `if (targetIndex >= 0)` (`:375,
  :410,:413,:417`) is always true and the `else { _IsLIDCorrect = -1 }` at `:424-425` never fires --
  reproduced structurally (the accessor CAN return -1) but never reached. The `rowEnd - rowBegin + 1
  >= ssr` guard (`:352`) is unsigned in the legacy: when `rowEnd < rowBegin` (a segment shorter than
  one periodogram frame) it wraps to a huge value, passes the guard, and the subsequent `.block(...)`
  slice reads OOB (UB). The port adds an explicit `row_end >= row_begin` guard so it SKIPS such a
  segment instead of UB-slicing; this is unreachable with real min-length SPEECH segments (the harness
  would `std::abort` on the resulting structural mismatch, so no committed fixture exercises it) and is
  byte-identical on every fixture. *Why deferred:* the -1 branch is dead reproduction; the guard is a
  port-only safety over legacy UB. *Fix candidate:* N/A (the UB is unreachable). *Pinned by:*
  `lid_members_match_dump`, `sentinel_gt150_present_every_file` (`tests/phase4b_lid5_golden.rs`).
  Mutation: the Task-11 battery's in-band `-2.0 -> -1.0` flip breaks `sentinel_gt150_present_every_file`
  / `lid_classification_errors_match_dump` (Task-11 entry below, item 1); the dead `-1` branch and
  the port-only underflow guard are themselves N/A (both unreachable -- `targetIndex` is clamped
  `>= 0`, and min-length SPEECH segments never underflow the slice).

- **[phase4c] CLOSED: the lid5 (Algo 5) `LID_STRUCT` harness probe's `cost_max_abs`
  calibration recorded the MAGNITUDE of the real value, not a real-vs-reimpl delta -- unlike
  its sibling `langid_max_abs`** (`tools/oracle_harness/main.cpp`'s `runLidFile` lambda, the
  `lid5` SECONDARY real-Eigen probe added in Phase 4b Task 4): `langid_max_abs` is a genuine
  `std::fabs(langidR[chan](0,cc) - langidReal)` delta because the reimpl's per-channel `langID`
  matrix is captured through an out-param (`langidR`, filled by `transcribeLid`). The sibling
  `cost_max_abs` line instead read `std::fabs(segReal._LIDCumulativeError[chan])` -- the REAL
  value's absolute magnitude alone, because `transcribeLid` never returned its per-channel
  `NNCost` (`BLSTMSpectralLID.cpp:414`'s `seg._LIDCumulativeError[chan] = NNCost` counterpart)
  for the caller to diff against. The committed manifest values (19.47/0.1376/19.47) were
  therefore never a calibration signal at all -- they were just `|_LIDCumulativeError|`,
  masking whatever the real reimpl-vs-real forward divergence actually was. Fixed:
  `transcribeLid` gained a `cumErrOut` out-param (mirroring `langidOut`/`confusionOut`/
  `isCorrectOut`), populated with the reimpl's raw per-channel `NNCost` (no `/segmentsCount`,
  no weight scaling -- matching the real ctor's `seg._LIDCumulativeError[chan] = NNCost`
  exactly); `runLidFile` now computes `std::fabs(segReal._LIDCumulativeError[chan] -
  cumErrR[chan])`. Measured deltas after the fix: f1/f3 `3.553e-15`, f2 `1.11e-16` -- the same
  order as `langid_max_abs`'s `1.11e-16` (the expected ascending-loop-vs-real-Eigen forward
  noise floor), confirming the bug was purely a harness diagnostic defect, not a hidden
  port-vs-legacy divergence. *Pinned by:* `tests/test_phase4b_fixtures.py::
  test_lid5_calibration_cost_max_abs_is_a_true_delta` (asserts every `lid5.calibration.*.
  cost_max_abs` is `< 1e-6`; RED against the stale manifest's `19.47`/`0.1376`/`19.47`, GREEN
  after `scripts/extract_phase4b_fixtures.py` regenerated `manifest.json` with the fixed
  harness -- byte-identical across two consecutive regenerations, and no other `phase4b/`
  fixture drifted).

- **[phase4b] The phSeq reader's phoneme count carries a fixed `10 + sum(len+10)` padding
  arithmetic -- 10 phantom head phonemes plus 10 phantom gap phonemes appended after EVERY
  sentence (including the last)** (`audio.rs::read_phseq`, from `AudioStruct.cpp:164,170`):
  `numberOfPhonemes` starts at 10 (`:164`, before any line is read) and each sentence adds
  `length+10` (`:170`), so a file's "phoneme count" is inflated by `10*(nb_lines+1)` silence-gap
  slots that never correspond to any input character. Combined with the `rowBegin = 5` block-fill
  start (`:178`, see the next entry) this yields a 5-row lead margin, 10-row inter-sentence gaps,
  and -- by the identity `numberOfPhonemes - final_rowBegin == 5` (final `rowBegin` is
  `5 + sum(len+10)`) -- an exactly 5-row trailing margin, so the fill never overflows the
  `numberOfPhonemes x 38` allocation for ANY sentence lengths. The asymmetric split (5 lead vs 10
  between) is presumably a context-padding choice for the downstream BLSTM, but nothing documents
  it. *Why deferred:* provenance; the goldens encode these exact offsets. *Fix candidate:* none --
  document once a real phSeq training run validates the intent. *Pinned by:* `phseq_onehot_golden`
  (`tests/phase4b_phseq.rs`, bit-exact vs the REAL compiled `AudioStruct` `file_type==1` ctor's
  `_Periodogram`/`_ExternalFeatures` dumps) and `phseq_metadata_matches_manifest` (the
  `numberOfPhonemes`/frames-count scalars). Mutation: changing the initial 10, the per-sentence
  `+10`, or the `rowBegin = 5` seed each shifts every block placement and/or the periodogram row
  count -> `phseq_onehot_golden` fails on shape or first-row mismatch (`f1`: 52 rows, blocks at
  5/21/31). The `len==0` edge (a blank line contributes its bare `+10` with a 0-row one-hot block)
  is separately pinned by `phseq_zero_length_line_is_a_zero_row_block`.

- **[phase4b] `_FramesCount = numberOfPhonemes*0.01*_Framerate` fabricates a synthetic duration
  (10 ms per phoneme slot) with LEFT-ASSOCIATIVE float grouping, and the 0.01 is an undocumented
  magic constant** (`audio.rs::phseq_frames_count`, from `AudioStruct.cpp:173`): the zero-filled
  `_Data` gets `(numberOfPhonemes*0.01)*_Framerate` frames -- C++ `*` associates left-to-right, and
  the port isolates that exact grouping in `phseq_frames_count` because the naive regrouping
  `numberOfPhonemes*(0.01*_Framerate)` differs by 1 truncated frame where `n*0.01` rounds down
  (first divergence at `numberOfPhonemes = 803`, framerate 8000: 64239.999... -> 64239 vs 64240.0
  -> 64240; the committed fixtures' 52/46/45 all land exact, so only the synthetic in-test file
  exercises the divergence). The double -> `long long` assignment truncates toward zero (`as i64`).
  *Why deferred:* provenance; 0.01 (= 10 ms/slot at any framerate) is load-bearing for every
  downstream frame-indexed computation. *Fix candidate:* name the constant once end-to-end phSeq
  parity holds. *Pinned by:* `frames_count_arithmetic` (`tests/phase4b_phseq.rs`, a 783-char
  synthetic line -> `numberOfPhonemes` 803 -> asserts 64239 AND explicitly `!= 64240`) plus
  `phseq_metadata_matches_manifest` (fixture-file counts 4160/3680/3600). Mutation: regrouping the
  product right-associatively flips `frames_count_arithmetic`'s 64239 assert; changing 0.01 flips
  every frames-count assert in both tests.

- **[phase4b] The phSeq out-of-domain-character crash (`letterMapping.at()` ->
  uncaught `std::out_of_range` -> `std::terminate`) is ported as a recoverable `Err`, and the
  missing-file `exit(1)` as `Err` too** (`audio.rs::letter_index`/`read_phseq`, from
  `AudioStruct.cpp:168,149-152`): the legacy one-hot loop indexes `letterMapping.at(sentence[pos])`
  with no domain guard -- any character outside the 38-entry map (`AudioStruct.h:18`; note 'q' is
  absent, as are all digits and uppercase beyond the 11 phone-class letters) throws out of `.at()`
  and, uncaught anywhere in the call chain, aborts the process. This port returns a contextual
  error naming the char/line/pos instead of reproducing a crash; same `Err` treatment for the
  missing-file `exit(1)` (the established Phase 0a+ convention). Also note `sentence[pos]` is
  byte-indexed `std::string`: non-ASCII input would be consumed byte-by-byte in the legacy, while
  the port iterates `chars()`; parity-neutral because every mapped character is single-byte ASCII
  and out-of-domain input errors on both sides (differently). *Why deferred:* N/A -- an
  error-surface improvement over legacy UB/abort, behavior-identical on all valid inputs.
  *Fix candidate:* N/A. *Pinned by:* `out_of_domain_char_bails`, `missing_file_bails`
  (`tests/phase4b_phseq.rs`).

- **[phase4b] The Twin driver ports wav-modes 0/1/2/3 + Mode 7 (phSeq); modes 4/5/6 + the pitch
  pass + the CNN remain deferred/excluded** (`tasks/lid.rs::TwinBlstmSpectralLid`, from
  `TwinBLSTMSpectralLID.cpp:263-1421`): `getSegmentation` is a 1,159-line switch over 8 modes.
  This port covers the mode 0/1/2/3 scoring branch (`:1194-1310`) + the SAD FFB (`:713-764`) + the
  `abs(_Mode)==7` phSeq LID loop (`:903-1193`, `get_segmentation_mode7`); it BAILS typed on modes
  4/5/6 (the LID-first train block `:640-692` + the `:798-902` scoring branch: `getTargetsLID` ->
  LID `feedForwardBackward` train, mode-6 `LID2Segmentation`, the `_PostProcessMode` slice of the
  pre-computed `LID_result_vec`), on the Mode-7 WAV arm (`:922-963`, the CNN), and on the pitch
  second pass (`:349-614`, `TDC_window_size > 0`). **Modes-1/4 decision:** mode 1 was UNEXERCISED in
  T6 -- it is now a full GOLDEN (`twin_mode1`, the same T2 corpus + synthetic LID net, SEG_STRUCT/
  LID_STRUCT-verified vs the real compiled Twin); mode 4 is deferred WITH modes 4/5/6 (its train
  branch + reference-copy/smooth share their machinery), documented rather than probe-pinned since
  its fixture is not craftable in isolation. The Twin's pitch
  pass is REFERENCE-based (`seg._Classification = seg._Reference` then `getPitch`, `:366-369`) and
  warps the periodogram BEFORE the SAD forward -- a load-bearing DIVERGENCE from the base spectral
  driver's post-forward pitch pass (`BLSTMSpectralSegmenter.cpp:757-805`) -- so it cannot reuse that
  machinery and is deferred with its own golden. All four ported configs use `TDCwindow 0` so the
  gate is off (byte-identical to a no-pitch run). The `_LIDConvNeuralNetwork` (`:23`) is constructed
  in the legacy ctor but is dead under the port scope (only mode 7 uses it) -- NOT ported, matching
  the Phase 2 Conv exclusion (`ConvolutionalLayer` is broken-as-committed; the phSeq Mode-7 arm
  reaches the CNN only for WAV, so its typed bail is the port's whole treatment of it). Mode 1's
  synthesis (`result_vec.setConstant(10.0)`, `:726`) shares the `:1194` branch and is now golden
  (`twin_mode1`). `interestSegs` (`:1240`) + the `_LIDTrainingPruningThreshold` gate (`:1286`,
  `itInterest->_Type = OTHER`) are NOT reproduced: both feed ONLY the skipped VRCTS dump + the
  mode!=0 `interestSegs` replacement (`:1297`, dump-only), so they are observably dead (and
  config-gated on `> 0`, default -1.0, besides). *Why deferred:* the deferred modes are the next
  task's scope; the pitch divergence needs its own reference-based transcription. *Fix candidate:*
  land modes 4/5/6 + the pitch pass in a follow-up task. **UPDATE (Task 7c):** modes 4/5/6 are now
  LANDED (`get_segmentation_mode456`, full goldens, see the entry below); only the pitch second pass
  + the Mode-7 WAV/CNN arm remain deferred. *Pinned by:* `boundaries_match_dump`,
  `members_match_dump` (`tests/phase4b_twin_golden.rs`, all EIGHT wav modes 0/0_concat/1/2/3/4/5/6
  bit-exact vs the harness `TwinProbe` reimpl, itself SEG_STRUCT/LID_STRUCT-verified against the
  REAL compiled Twin) + the Mode-7 flagship goldens (`tests/phase4b_twin_mode7.rs`). Mutation:
  removing the `mode` guard makes a mode-4 config reach the mode-0 SAD-FFB path and diverge.

- **[phase4b] Task 7c: modes 4/5/6 slice `LID_result_vec` by `LIDTimeStep`/`LIDTimeOffset` with the
  offset SUBTRACTED before the divide (`:811-813`) -- NOT the `:1245` branch's bare
  `begin/_SpectrumShift`; the LID net trains on the WHOLE inputSeq ONCE, and several per-mode
  synthesis steps are DEAD** (`tasks/lid.rs::get_segmentation_mode456`, from
  `TwinBLSTMSpectralLID.cpp:640-902`): unlike modes 0/1/2/3 (per-segment `feedForward`, row index
  `begin/_SpectrumShift`), modes 4/5/6 run `_LIDBLSTMNeuralNetwork.feedForwardBackward` ONCE over the
  whole inputSeq (`:664`) into `LID_result_vec`, then the `:798-902` scoring reads blocks
  `rowBegin = ceil((begin - LIDTimeOffset)/LIDTimeStep)`, `rowEnd = floor((next - LIDTimeOffset)/
  LIDTimeStep)` (offset subtracted first -- load-bearing, since `LID_result_vec` is decimated by the
  LID net's own subsampling, not indexed in periodogram frames). The LID FFB self-normalizes inputSeq
  IN PLACE (type -1), so for mode 5 the SAD net's OWN FFB re-normalizes the already-LID-normalized
  buffer (double self-norm, reproduced by calling the two FFBs on the same `&mut input_seq` in the
  legacy order: LID first, SAD second). DEAD steps reproduced as no-ops (documented, not executed):
  (a) mode 6's `:727-732` result_vec = `1 - LID_result_vec.col(0)` + the `timeStep`/`_DecisionThresh`
  overwrites feed ONLY `results2segmentation`, which is SKIPPED for mode 6 (`_Mode != 4 && != 6`,
  `:773`) -- the port computes the result_vec ONLY for the dumped `last_result_rows` row, and drops
  the threshold overwrites; (b) `totalSpeechDuration` (`:799-806`) is never read by the `:798-902`
  post-loop (it normalizes by `segmentsCount`/`numberOfFrames`, `:888-892`) -- skipped; (c) the
  `score = segLID(targetIndex)` locals (`:860-861,:900-901`) are dead. Per-mode classification:
  mode 4 = REFERENCE smoothed (`seg._Classification = seg._Reference` then `smoothSegmentation`,
  `:779-783`, via the new `Segmentation::set_segments_from`); mode 5 = SAD VAD; mode 6 =
  `LID2Segmentation` over `1 - LID_result_vec.col(0)` (`:678-691`). The `:798-902` accumulator is the
  three `_PostProcessMode` forms shared with the mode-7 branch (`langID += isCostModified ?
  segLID/rows : segLID`; post-loop `/= segmentsCount` or `/= numberOfFrames`, `PPM != 2 -> exp`,
  row-normalize). *Why deferred:* provenance -- the offset-subtracted slice + the double-norm order +
  the dead branches are the legacy's exact behavior. *Fix candidate:* N/A. *Pinned by:*
  `sad_result_row_matches_dump`, `lid_classification_errors_match_dump`, `lid_confusion_matches_dump`,
  `members_match_dump`, `boundaries_match_dump` (all EIGHT variants bit-exact), plus the non-vacuity
  `mode6_lid2segmentation_non_vacuous` (LID2Seg produces speech spans `~[1.73,1.81]/[1.89,1.94]`
  DIFFERING from the reference `[0.4,0.9]/[1.2,1.6]`) and `mode5_runs_sad_net_modes_4_6_do_not`
  (`tests/phase4b_twin_golden.rs`). Mutation: (i) NOT subtracting `LIDTimeOffset` in the row slice
  (`begin/LIDTimeStep`) shifts the sliced blocks and diverges the confusion/langID; (ii) reversing
  the FFB order for mode 5 (SAD before LID) feeds the SAD net a differently-normalized input and
  diverges `sad_result_row`; (iii) running `results2segmentation` for mode 6 (dropping the
  `mode == 5` guard) overwrites the `LID2Segmentation` classification and breaks `boundaries_match_dump`.

- **[phase4b] Task 7c latent: the modes-4/5/6 offset-subtracted row slice saturates a negative
  index to 0 where the legacy UB-wraps to a huge unsigned** (`tasks/lid.rs::get_segmentation_mode456`,
  from `TwinBLSTMSpectralLID.cpp:811-812`): the block start is `rowBegin =
  ceil((begin - LIDTimeOffset)/LIDTimeStep)`. If a SPEECH segment's `begin` is EARLIER than
  `LIDTimeOffset`, `(begin - offset)/step` is negative; Rust's `ratio as usize` saturates the
  negative float to 0 (then the `!= ratio` bump lands `row_begin` at 1), while C++ `(size_t)` of a
  negative double is UB that on the usual targets wraps to a huge value -- which then trips the
  `rowEnd < rowBegin` skip guard (`:815`), dropping the segment entirely. So on a hypothetical
  `begin < LIDTimeOffset` input the two would DIVERGE (Rust reads a top-of-buffer block; the legacy
  skips). This is the offset-subtracted (`:811-813`) slice's contrast with the modes-0-3 `:1245`
  bare `begin/_SpectrumShift` (always `>= 0`, so no negative there). UNREACHABLE in the committed
  fixtures: every mode-4/5/6 variant derives `LIDTimeOffset = 0.0`, so `begin - 0 >= 0` for every
  segment and `ratio` is never negative. *Why deferred:* a port-only safety over legacy UB on an
  unreachable input, not a behavioral deviation on any fixture. *Fix candidate:* N/A (the negative
  case cannot occur under any committed config). *Pinned by:* `sad_result_row_matches_dump`,
  `members_match_dump` (`tests/phase4b_twin_golden.rs`, modes 4/5/6, all bit-exact -- `LIDTimeOffset`
  is 0 so the saturating cast and a UB-wrap would agree here). Mutation: N/A -- the divergence needs
  a `LIDTimeOffset > min speech begin` config, which no committed fixture provides.

- **[phase4b] The SAD FFB normalizes `inputSeq` IN PLACE (non-const `Eigen::Ref`), so the concat +
  LID scoring consume the NORMALIZED SAD input; `getBLSTMLIDInputSequence` resamples the SAD hidden
  states by nearest index when the LSTM row count differs** (`tasks/lid.rs::get_segmentation`/
  `get_blstm_lid_input_sequence`, from `TwinBLSTMSpectralLID.cpp:715,1221,139-168`): the windowed
  `feedForwardBackward(Eigen::Ref<Eigen::MatrixXd> inputSeq, ...)` (`BLSTMNeuralNetwork.cpp:711`, the
  NON-const overload, unlike the plain `:776` `const Ref`) mutates the caller's `inputSeq` via the
  type -1 self-normalization (`:737-743`) -- so for modes 0/3 the `:1221` concat guard + the
  `:1249` per-segment slice see the normalized values, NOT the raw feature. The port passes
  `&mut input_seq` to `feed_forward_backward` (which self-normalizes in place, matching), then reads
  the mutated buffer for the concat/scoring. `getBLSTMLIDInputSequence` (`:139-168`) z-normalizes
  `[_OutputForward | _OutputBackward]` per column (the `+1e-32` std floor) and, when the LSTM output
  row count `n != inputSeq.rows()` (always here: `n = T/lstm_ss`, `rows = T`), hcat's them by NEAREST
  index `round((n-1)*frame/(rows-1))` (the `+0.5` truncation, `:158`) -- an upsample, reproduced with
  a `(rows-1).max(1)` denominator (the legacy `length-1` div-by-zero at `rows==1` is unreachable).
  An EMPTY `_OutputForward` (modes 1/2, cleared at `:762-763`) returns `inputSeq` unchanged (the
  fallback). *Why deferred:* provenance -- the in-place mutation is a load-bearing Eigen-signature
  fact, and the resample is exact. *Fix candidate:* N/A. *Pinned by:* `concat_branch_matches_expected`
  (branch 0/1/2 per variant), `sad_result_row_matches_dump`, `lid_confusion_matches_dump`
  (`tests/phase4b_twin_golden.rs`). Mutation: cloning `input_seq` before the SAD FFB (leaving it raw)
  makes the concat variant's confusion/liderr diverge; forcing the direct-hcat branch (`n == rows`)
  panics on the shape mismatch.

- **[phase4b] `LID2Segmentation`'s `threshMin` argument is DEAD, `getTargetsLID` uses `counter ==
  step` (EQUALITY, not the scoring path's `>=`), and the mode 0/1/2/3 branch's `LIDTimeStep`/
  `LIDTimeOffset` locals are DEAD** (`tasks/lid.rs`, from `TwinBLSTMSpectralLID.cpp:1296,182,
  1195-1204` + `Segmenter.cpp:999-1055`): `LID2Segmentation(seg, ..., threshMax, threshMin)` (`:1296`,
  passed `_LIDDecisionThreshRising`/`_LIDDecisionThreshFalling`) reads ONLY `threshMax` -- the whole
  falling-edge block is commented out (`Segmenter.cpp:1057-1075`), so `threshMin` is inert -- ported
  via the single-threshold [`lid_to_segmentation`]. `getTargetsLID` (`:170-219`, a private fn for the
  next task's modes 4/5/6, unreached by 0/1/2/3) enforces on `counter == getTargetEnforcementStep()`
  (`:182`, strict equality), NOT the scoring `feedForward`'s `counter >= step` (`:1073`) -- a
  load-bearing asymmetry ported verbatim + unit-tested directly. The `:1195-1204` `LIDTimeStep`/
  `LIDTimeOffset` re-derivation inside the `:1194` branch is never read there (the scoring loop uses
  `_SpectrumShift` for row indices `:1245` + the SAD `timeStep` for the modifier `:1250` + `timeStep`/
  `timeOffset` for `LID2Segmentation` `:1296`) -- dead, skipped. *Why deferred:* provenance; the
  equality enforcement + the dead threshMin are the legacy's exact behavior. *Fix candidate:* N/A.
  *Pinned by:* `boundaries_match_dump` (modes 2/3 via `LID2Segmentation`,
  `tests/phase4b_twin_golden.rs`) + the inline `get_targets_lid_enforcement` unit test
  (`tasks/lid.rs`). Mutation: switching `get_targets_lid`'s `==` to `>=` flips its enforced-row
  placement for `step >= 1` (the `get_targets_lid_enforcement` test uses `step = 2`).

- **[phase4b] The phSeq (`!hasReadWavFile()`) `_SpectrumShiftInFrames = 80` override drives
  EVERY window/shift derivation** (`tasks/lid.rs` `get_segmentation_mode7`, from
  `BLSTMSpectralSegmenter.cpp:208-210`): `initSpectralAnalysis` computes
  `_SpectrumShiftInFrames = round(_SpectrumShift*frameRate)` (= `round(0.025*8000) = 200`)
  then UNCONDITIONALLY overrides it to `80` for a non-wav (phSeq) source (`:209`), and
  re-derives `_SpectrumShift = 80/frameRate = 0.01`. Load-bearing: with 200 the Mode-7 LID
  window is 10 (getLIDBLSTMParam `round(0.25*8000/2/200)=5 -> noOverlap -> 10`); with the
  correct 80 it is 25 (`round(0.25*8000/2/80)=13 -> noOverlap -> 25`), which changes the
  TwoSweeps truncate row counts (nb_of_classif 39 vs 77 for a 7-row block) and every
  posterior. The port keys the override off `audio.periodogram.is_some()` (the phSeq ctor
  populates it; the wav path leaves it `None`). *Why deferred:* faithful port; the 80 is a
  hardcoded legacy magic constant. *Fix candidate:* N/A. *Pinned by:*
  `mode7_integer_members_match_real_bitexact` + `mode7_continuous_members_match_real`
  (`tests/phase4b_twin_mode7.rs`, bit-exact vs the REAL compiled `getSegmentation`: nbclassif
  150/213/142). Mutation: dropping the `= 80` override (keeping 200) yields nb_of_classif 74
  and flips `s2`/`s3` classifications, failing every mode-7 golden. This closes the latent gap
  flagged in the Task-4 (phase2b) review entry above: the `_SpectrumShiftInFrames = 80`
  override was pinned unit-test-only via `force_non_wav_spectrum_shift` (no production caller,
  since the corpus was wav-only at the time) -- the phSeq corpus landed in Task 5/7 gives it a
  real, exercised production caller (`get_segmentation_mode7`).

- **[phase4b] The Mode-7 noise `random_init` is wall-clock -> the port fixes `randinit = 0`**
  (`tasks/lid.rs` `get_segmentation_mode7`, from `TwinBLSTMSpectralLID.cpp:311,1025-1032`):
  the noise offset is `randinit = remainder((long)(1e6*preparationElapsedSec), 100)` (`:311`
  seeds `random_init` from `Timer` wall-clock; `:1025` reduces it mod 100), so the legacy
  noise is NON-DETERMINISTIC and NON-REPRODUCIBLE across runs/machines. The table itself is
  fixed (`_RandomGaussVector[(kk*cols+ll+randinit)%_MaxRandSize]`), so only the offset is
  irreproducible. The port fixes `randinit = 0`, making the noise deterministic; the pure
  table INDEXING (magnitude * (`random_gauss(kk*cols+ll) - 0.5`)) is then bit-exact. The
  flagship configs run `_NoiseMagnitude 0` (noise off, so the real path is deterministic and
  the goldens above hold regardless). *Why deferred:* wall-clock seeding cannot be ported
  faithfully. *Fix candidate:* thread a real RNG seed through the config if noise is ever
  needed for training. *Pinned by:* `mode7_noise_table_indexing_strict`
  (`tests/phase4b_twin_mode7.rs`, STRICT bits vs the harness `_RandomGaussVector` dump with
  `randinit = 0`). Mutation: shifting the port's index by +1 (`kk*cols+ll+1`) breaks the
  strict golden.

- **[phase4b] `abs(_Mode) == 7`: the wav arm (CNN) is unported, the `_Mode < 0` sub-branches
  are dead, and the SAD net never runs** (`tasks/lid.rs`, from `TwinBLSTMSpectralLID.cpp:
  903-1193`): the `if (audio.hasReadWavFile())` block (`:922-963`) feeds `inputSeq` through
  `_LIDConvNeuralNetwork` (`:927`, the CNN -- NOT ported, dead-under-scope per the Phase 2
  Conv exclusion) then rebuilds `_ExternalFeatures`; for phSeq (`!hasReadWavFile()`) it is
  SKIPPED, so the loop iterates the ctor-loaded `_ExternalFeatures` directly. The port bails
  typed on wav Mode 7 (CNN). The `_Mode < 0 && outputSeq.cols() == 2` sub-branch (`:1040-
  1071`, an extra SAD `feedForward` when `targetIndex == 1`) is dead under `_Mode = +7` and
  skipped. The SAD `_BLSTMNeuralNetwork` is never run in Mode 7 (`result_vec` is synthesized
  constant `10.0` at `:726`, outputs cleared at `:762-763`), so the committed config's absent
  `BLSTM_weightsFile` (default-init SAD net) is parity-neutral. *Why deferred:* the CNN is
  broken-as-committed (Phase 2); the negative modes are dead. *Fix candidate:* port the CNN
  if a mode-7 wav corpus is ever needed. *Pinned by:* the mode-7 goldens above (phSeq arm) +
  the driver's typed wav bail (unit-covered by `get_segmentation_mode7`'s `periodogram.is_none()`
  guard -- the "unit-covered" claim was an overclaim: no such test existed; corrected +
  genuinely pinned by the [phase4d] complete-as-portable closure entry below). Mutation: N/A
  (dead code); the SAD-irrelevance is proven by the goldens passing with no SAD weights loaded.

- **[phase4b] Mode-7 `classNb = max(2, outputSize)`, and `_PostProcessMode 1` cancels in
  normalization; the DumpLIDInternals filename is CLOSED (Phase 4c Task 2)** (`tasks/lid.rs`,
  from `TwinBLSTMSpectralLID.cpp:624-625,906-921,1073-1179`): the committed LID net is BINARY
  (`OutputNeuronNb 48,1` -> `getOutputSize() = 1`), and `classNb` is forced to `max(2, 1) =
  2` (`:624-625`), so the brief's "3-class mapping" is superseded -- the fixtures use a
  2-class mapping. `_PostProcessMode 1` (entropy-weighted, `:1073-1089`) adds a per-row scalar
  `-sum log(entropy)` EQUALLY across every `segLID` column, which is a column-constant offset
  -> it cancels in the softmax normalization (`:1169-1172`), so ppm1's normalized `langID`
  NEAR-coincides with ppm0's (exactly in real arithmetic, ~1 ULP in floating point). That pin
  is therefore effectively ORACLE-ENV-ONLY as a mutation-catcher: `mode7_continuous_members_
  match_real`'s ppm1 golden (`tests/phase4b_twin_mode7.rs`) sits only ~1 ULP from ppm0's, well
  inside the off-oracle hybrid bound (`common::assert_oracle_eq`, `<=4` ULP / `512*eps*scale`
  absolute), so a bug that collapsed ppm1's post-process path onto ppm0's could pass
  undetected on CI glibc. *Why deferred (classNb/ppm1):* out of scope for this task, unrelated
  to the filename fix. *Fix candidate:* none identified. *Pinned by:*
  `post_process_mode_all_three_covered` (`tests/phase4b_twin_mode7.rs`; ppm2 vote DISTINCT,
  ppm1 near-coincident, all three code paths counter-asserted). Mutation: forcing `classNb =
  outputSize` (dropping the `max(2,.)`) makes the confusion 3x3 and mismatches the REAL 4x4
  dump.
  **CLOSED (Phase 4c Task 2), DumpLIDInternals filename:** `_DumpLIDInternals` derives the
  `.mat` filename from `audio.getAudioFileName()` (`:906-913`, the `_DumpDir.size() > 0`
  branch, the only one reachable here): strip the directory (the portion after the last `/`,
  or the whole string if none), then ALWAYS drop exactly 4 trailing characters from that
  basename -- NOT an extension-aware strip (a `.wav` name is cleanly de-extensioned; the
  mode-7 phSeq arm's `.phSeq`, 6 chars, leaves a partial extension, e.g. `"s1.phSeq"` ->
  `"s1.p"`); when the basename is shorter than 4 bytes, `std::string::substr`'s length-clamp
  leaves it untouched instead of underflowing. The port previously didn't thread
  `Audio::audio_file_name` (landed in 4b Task 8, populated post-hoc by
  `bag_of_processors::apply_corpus_item`, not by `read_audio`/`read_phseq`) into this dump
  path, so the dump landed at the cosmetic `<_DumpDir>/chan<c>_lid_dump.mat` instead of the
  legacy-faithful `<_DumpDir>/<basename minus 4 chars>_chan<c>_lid_dump.mat`; the VARIABLE
  names (`features_<n>` = `[_OutputForward | _OutputBackward]`, `matNb`) and values were
  always the faithful part. Now fixed: `tasks/lid.rs::mode7_dump_basename` composes the
  legacy-faithful name from `audio.audio_file_name`, threaded through
  `get_segmentation_mode7`. Verified no oracle-harness or committed-fixture filename shared
  this convention (`tools/oracle_harness/main.cpp`'s `mode7_dump_s1.mat` scipy-value probe
  writes a harness-hardcoded literal name via a hand-rolled `Mat_Create` call, never going
  through `getSegmentation`'s dump path, so no harness-side rename was needed; confirmed by a
  no-op `git status` on `tests/reference_data/phase4b/` for every file except
  `manifest.json`, whose diff is isolated to the unrelated `cost_max_abs` fix below). *Pinned
  by:* `dump_lid_internals_written_and_valued` (`tests/phase4b_twin_mode7.rs`), TDD RED/GREEN
  verified: the test's `phseq_audio` helper now sets `audio.audio_file_name` to the real
  `.phSeq` path (previously left empty), and the assertion targets `s1.p_chan0_lid_dump.mat`
  under the tempdir; reverting `tasks/lid.rs`'s driver change alone (test unchanged) fails the
  test (file not found at the new path), confirming the rename is load-bearing.

- **[phase4b] `VrctsPart` (Algo 0) hard-codes the legacy `vrcts_part` binary path, and its spawn
  failure semantics necessarily diverge from the legacy's discarded `system()` return** (`tasks/
  vrcts.rs`, from `VRCTSpart.cpp:44,46,53`): the command string embeds the absolute path
  `/usr/local/vrcts/vrcts_1_5_9/bin/vrcts_part` (not config-driven), then calls bare
  `system(command.str().c_str())` -- the shell exit status (including 127, "command not found",
  if the binary is missing) is NEVER read, so `getSegmentation` always proceeds straight to
  `load_from_vrcts` on whatever the xml file (still) contains. `std::process::Command` execs the
  binary directly with no intervening shell, so a missing binary is a SPAWN-level `io::Error`
  (`NotFound`) at the Rust API boundary, not an ignorable exit code -- there is no way to swallow
  that and still call it "faithful" (the OS never even started a process to have an exit code from).
  The port therefore surfaces ONLY the spawn-level failure as a typed error (propagated via `?`);
  a successful spawn's exit status IS still discarded, matching legacy. *Why deferred:* the binary
  does not exist on this or any CI machine, so this divergence is currently unobservable in any
  golden; revisit once a real VRCTS install is available to test the ignored-exit-code path.
  *Fix candidate:* make the binary path configurable; consider whether the exit-code-ignoring
  behavior should be preserved or fixed if VRCTS is ever run for real corpora. *Pinned by:*
  `spawn_attempted_when_missing`, `force_respawns` (`tests/phase4b_vrcts.rs`, error message
  asserted to mention the hard-coded path). Mutation: N/A (there is no legacy golden to diverge
  from -- the binary's absence IS the tested condition).

- **[phase4b] `VrctsPart::getSegmentation` composes the SAME xml path for every channel (no
  `_chan_<n>` suffix), so a multi-channel file loads an IDENTICAL segmentation into every
  channel** (`tasks/vrcts.rs`, from `VRCTSpart.cpp:34-38`): the per-channel path variant
  (`"%s_chan_%s_VRCTS_Fast.xml"`) is commented out in the legacy source; the LIVE line is
  `"%s_VRCTS_Fast.xml"` keyed only on `_RefSegFilename`, with no `chan` in it at all (the audio
  arg passed to `vrcts_part -f` is likewise the whole multi-channel file, not a per-channel
  split). Reproduced verbatim: the Rust loop recomputes the SAME path every iteration and, for
  `force=false`, the first channel's spawn (if any) makes the file exist for every subsequent
  channel too. *Why deferred:* provenance -- a straight transcription of the live (non-commented)
  line. *Fix candidate:* N/A (this is presumably intentional in the legacy: VRCTS analyzes the
  whole recording once and applies the same speech/non-speech decision to all channels). *Pinned
  by:* `same_xml_shared_across_channels` (`tests/phase4b_vrcts.rs`). Mutation: reinstating the
  `_chan_<n>` suffix in the path format breaks 3 of the 5 `phase4b_vrcts` tests (verified: the
  pre-seeded single-path fixtures are no longer found, forcing a spawn that then fails).

- **[phase4b] CLOSED: `VrctsPart::from_legacy` now replicates `Segmenter::buildFromConf`'s
  generic required-key validation** (`tasks/vrcts.rs`, from `VRCTSpart.cpp:9` +
  `Segmenter.cpp:73-148`): the legacy ctor calls `this->buildFromConf(conf, "VRCTS", ...)` BEFORE
  reading `VRCTS_isFast`/`VRCTS_force`, which requires (no default, `conf.get<T>` throws/exits on
  missing) a set of `VRCTS_*` keys -- decision thresholds, window/shift, convolution kernel,
  padding/min-speech/min-silence lists -- none of which `VRCTSPart::getSegmentation` ever reads
  (no `updateSegmentation`/`smoothSegmentation`/`results2segmentation` call site anywhere in the
  function). `from_legacy` originally skipped this validation entirely (accepting a config that
  would have failed to CONSTRUCT `VRCTSPart` in legacy); it now calls
  `SegmenterConfig::from_config`/`DriverConfig::from_config` with prefix `"VRCTS"`, same as every
  sibling driver (`TdcSegmenter`/`LtsvSegmenter`/`BlstmSignalSegmenter`/`BlstmSpectralSegmenter`
  in `tasks/sad.rs`) -- a missing key now errors, matching `buildFromConf`'s `exit(1)`, and
  `dump_dir()` now reads `DriverConfig::dump_dir` instead of a separately-parsed field. The parsed
  `SegmenterConfig`/`DriverConfig` values are otherwise still unused by `getSegmentation`, same as
  before. NOTE: this is still a PARTIAL replication of `buildFromConf` -- the windowing-type/
  preemph/noise-ratio reads and the `CostLaw` construction (`Segmenter.cpp:139-146`) remain
  unreplicated, but that gap is pre-existing and shared by every sibling driver too (none of them
  read `{prefix}_CostPonderation`/`{prefix}_CostLawSpeech`/`{prefix}_CostLawParamSpeech` either),
  so it is out of scope for this fix. *Pinned by:* `missing_required_key_rejected`
  (`tests/phase4b_vrcts.rs`): a config missing `VRCTS_min_silence` now errs naming the key.
  Mutation: reverting `from_legacy` to skip the two calls makes this test fail (the construction
  would instead succeed).

- **[phase4b] `saveAndUpdate`'s `PrintConfusionMatrix` outputs are DISPLAY-ONLY -- no
  parity-relevant storage exists** (`engine/bag_of_processors.rs::save_and_update`, from
  `BagOfProcessors.cpp:436-443,468`): the algo-5/6 confusion block slices the confusion columns
  (`block(0, 16, rows, cols-2-16)`) and calls `PrintConfusionMatrix`, but the returned
  `errorPercLID` feeds ONLY the `:443` `cout` status line and the `outputConfusion` string only
  the `:468` `cout` -- neither reaches a member, result row, mem matrix, or artifact. The port
  invokes the call for flow parity and exposes the `(error, matrix)` pair solely as a
  test-support observation (`last_confusion_for_test`); inventing storage would deviate from the
  legacy. *Pinned by:* `twin_train_epoch_weights_golden` (`tests/phase4b_corpus_lid.rs`) --
  the captured `errorPercLID` must equal the harness-measured per-epoch value (50.0 for the
  2-file 2-class corpus: one hit, one miss), computed by the REAL compiled
  `PrintConfusionMatrix` in the fixture generator. Mutation: mis-slicing the confusion block
  (e.g. starting at col 15) changes the captured error and fails the assert.

- **[phase4b] Mode-7's never-forwarded SAD net still gets `updateWeights` -> `0/0 = NaN`
  normalized derivs -> a deterministic Rprop `+delta` drift** (from `BagOfProcessors.cpp:196` +
  `BLSTMNeuralNetwork.cpp:304-308` + `Rprop.cpp:9-59`): algo 6's `updateWeights(cost)` runs
  unconditionally on the SAD net, whose Mode-7 derivs are an all-zero `Nx2` (reset per file,
  never accumulated -- the SAD net never forwards). The element-wise `col0 cwiseQuotient col1`
  yields `0/0 = NaN` for every weight row (the stats-tail rows are `0/1 = 0`), and the legacy
  Rprop's branch structure (`== 0` / `> 0` / `else`) routes NaN into the `else` arm -> `+delta`
  applied per weight row per epoch, deltas never grown (the `derivTimesPrev` NaN product also
  lands in the neither-positive-nor-negative arm). The Rust `Rprop` has the identical branch
  structure, so the NaN semantics agree without special-casing. *Pinned by:*
  `twin_train_epoch_weights_golden`'s SAD trace (`twin_train_sad_epoch{0..7}.bin` -- generated by
  the REAL compiled `updateWeights`/`Rprop`): the SAD weight rows drift by exactly `+init_delta`
  per epoch while the two stats-tail blocks stay put. Mutation: skipping the SAD update when the
  deriv counts are all zero freezes the trace and fails every epoch >= 1.

- **[phase4b] Mixed-width multi-config bags are broken-as-committed** (from
  `CorpusProcessor.cpp:342-389` + `BagOfProcessors.cpp:338-349`): algo 5/6 result rows are
  `18 + classNb` columns wide (the confusion columns), algo 0-4 rows 18 -- but
  `transformResults` sizes EVERY per-conf matrix (and `ResultsE`) from conf 0's first row's
  width (`:357`), and `_ResPerConf[cc].row(...) = res` with a wider `res` is an Eigen size
  mismatch (assert/UB). A bag mixing a LID config with a non-LID config therefore aborts in the
  legacy; the port's `transform_results_impl` panics on the same shape (out-of-bounds write).
  No committed legacy config mixes them; same-width bags (all-LID or all-non-LID with equal
  classNb) are fine. *Fix candidate:* per-conf row widths after parity. Not test-pinned
  (reaching it requires a deliberately malformed multi-config setup); documented here per the
  width-growth review in Task 9.

- **[phase4b] Mutation battery (Task 11): 6 of 7 fresh mutations break their named golden as
  designed; 1 genuine coverage gap found, recorded honestly.** Each applied/run(named suite
  only)/reverted/re-run in isolation, full `cargo test` once at the end (all foreground, no
  background test runs). (1) In-band encode `target_lid[ti] = -2.0` -> `-1.0` (`tasks/lid.rs`,
  both the `BlstmSpectralLid::get_segmentation` site and the `TwinBlstmSpectralLid::get_segmentation`
  modes-0-3 site) against `phase4b_lid5_golden`/`phase4b_twin_golden` -- FAILED as expected on
  BOTH (`sentinel_gt150_present_every_file`: "no >150 sentinel in [100.8..]"/"[143.1..]";
  `lid_classification_errors_match_dump`: target col now decodes to `100*x+100` instead of
  `100*x+200`); reverted, PASS. (2) Sentinel decode `150.0` -> `250.0`
  (`engine/confusion.rs::confusion_from_results`) against `phase4b_confusion_golden` -- FAILED
  as expected (`confusion_matrix_matches_harness_transcription` cell mismatch,
  `error_matches_real_confusion2string_and_printconfusionmatrix` bit mismatch); reverted, PASS.
  (3) PostProcessMode accumulator body swap (`2 <-> _` match arms, `get_segmentation_mode7`'s
  segLID loop) against `phase4b_twin_mode7` -- FAILED as expected (3 tests:
  `post_process_mode_all_three_covered` "ppm1 should near-coincide with ppm0" now false since ppm2
  swapped to the sum-log body pulls ppm1's near-coincidence check off; plus
  `mode7_continuous_members_match_real` and `mode7_integer_members_match_real_bitexact` bit
  mismatches); reverted, PASS. (4) `LID_` filename-prefix drop
  (`TwinBlstmSpectralLid::save_weights_lid`) against `phase4b_corpus_lid::twin_train_epoch_weights_golden`
  -- FAILED as expected, though via a different observable than the target `LID_*.exists()` assert:
  dropping the prefix makes the LID net's save collide with the SAD net's identically-named
  artifact, so the SAD weights dims check fails FIRST (`weights_bestNNWeight_1_twin_train.mat
  dims: left: (1093, 1) right: (537, 1)`) -- the corruption is caught earlier in the same test,
  confirming the mutation is load-bearing; reverted, PASS. (5) `[sad, lid]` dispatch order swap
  in `BagOfProcessors::get_weights_derivatives`'s `TwinLid` arm (now `[lid, sad]`) against
  `phase4b_corpus_lid::twin_gradcheck_golden`/`twin_train_epoch_weights_golden` -- FAILED as
  expected on both (`twin_gradcheck_golden`: "twin net 0 weight 0 backprop: got ... want ... (>
  16 ULP)"; `twin_train_epoch_weights_golden`: panics in `nn/train.rs` Rprop update, "index out
  of bounds: the len is 537 but the index is 537" -- the swapped, differently-shaped derivative
  matrix desyncs the per-weight iRPROP- state vectors); reverted, PASS. (6) `costLID = -1.0`
  no-speech gate removed (`bag_of_processors.rs::save_and_update`, the `:465` override) against
  `phase4b_corpus_lid::twin_train_ns_costlid_gate_live` -- FAILED as expected (epoch-5 weight
  mismatch, real costLID leaking into `update_weights_lid` where the gated `-1.0` should have
  fired); reverted, PASS. (7) `segmentationLID /= 56.0` -> `/= 55.0` (`tasks/lid.rs`,
  `TwinBlstmSpectralLid::get_segmentation` modes-0-3 arm) against `phase4b_twin_golden` (all 8
  variants incl. modes 1/2/3, the only ones that reach `lid_to_segmentation`) -- did NOT fail: a
  GENUINE coverage gap, measured not assumed. Diagnostic instrumentation (temporary, not
  committed) showed `segmentation_lid`'s max value on the T2 3-file corpus is `~0.0089`
  (`0.5/56`, the untouched midpoint-seed entries dominate the max) against a
  `BLSTM_LID_decision_thresh_rising` of `0.6` -- three orders of magnitude short, so neither
  `/56.0` nor `/55.0` ever crosses the rising threshold and `lid_to_segmentation` never emits a
  segment on this fixture; confirmed the gap is not merely 55-vs-56-insensitive but
  divisor-insensitive up to `/1.0` (still no crossing -- the live NN posteriors are just far from
  saturation on this synthetic corpus). Pushing further (`/0.001`, blasting every entry to `500`,
  far above threshold) surfaced a SECOND, independent reason the call site is inert here: when
  `results[0] >= thresh_max` (`tasks/segmenter.rs::lid_to_segmentation:470-472`), `begin` is
  seeded to `0.0` but `has_begun` is NOT set -- and the loop's only `has_begun = true` site
  requires `begin < 0.0` (`:478`), which is now false, so a segment that is already "open" at
  frame 0 can never close (the `:483-497` block never runs) NOR does the post-loop `if has_begun`
  finalizer (`:500-503`) fire (`has_begun` stayed false throughout) -- a same-family "already
  above threshold at frame 0" edge case never triggers `label_segment` at all. Both findings are
  measured facts about the current T2 fixture + `lid_to_segmentation`'s literal control flow, not
  a claim that the `/56.0`/`/55.0` distinction can never matter on other data. No production code
  changed to chase this gap; a future crafted fixture with LID posteriors saturated enough to
  cross 0.6 mid-sequence (not at frame 0) would be needed to close it. **Cross-reference (Task
  10, not re-run here):** the tie-gate (`>` -> `>=`) and fold-order-reversal mutations from the
  original Task 11 (4a) plan item are already executed and recorded above under the `[phase4a]
  Mutation battery` entry's `UPDATE (Task 10)` addendum and the `[phase4a] Phase 4b test backlog`
  entry, closed by `phase4b_backlog.rs::best_cost_gate_skips_on_exact_tie` and
  `fold_order_ascending_golden` respectively. *Net verdict for this task:* 6/7 fresh mutations
  break their golden as designed; 1 genuine gap (item 7) reported honestly rather than papered
  over. Full transcript (diffs, commands, exact output) in
  `.superpowers/sdd/task-11-report.md`.

- **[phase4c] SMORMS3.m header mislabeled "Sum of Functions Optimizer (SFO)"**
  (`legacy/Optimizer_V6.2.2/functions/SMORMS3.m:1-47`; ported in `src/python/speech/optimizers.py`
  `Smorms3`): the file's entire doc-comment header (title, arXiv 1311.2115 reference, the
  `obj = sfo(f_df, theta, subfunction_references, ...)` synopsis, "Author: Jascha Sohl-Dickstein")
  describes the SFO quasi-Newton optimizer, but the `classdef` at `:49` is `SMORMS3 < handle` and
  the `optimization_step` at `:324-363` implements the SMORMS3 update (`r = 1/(delta+1)`, RMS EMA,
  `min(lrate, stepRate^2/(MMS+eps))` per-param cap), NOT SFO. The SFO scaffolding (subspace
  `P`/`b`, Hessian banks, active-set growth) survives only as commented-out properties (`:55-146`).
  A copy-paste-from-sfo.m header the author never rewrote; harmless (comment only) but misleading.
  *Fix candidate:* rewrite the header to describe SMORMS3 once the optimizer zoo is settled. The
  port's docstring names the algorithm correctly and cites the real update lines.

- **[phase4c] Rprop.m dead `rand` at `:5`** (`legacy/Optimizer_V6.2.2/functions/Rprop.m:5-6`;
  ported in `src/python/speech/optimizers.py` `rprop_step`): line 5 computes
  `delta0 = 0.001*(0.9*rand+0.1)` (a random step-size seed), then line 6 UNCONDITIONALLY overwrites
  it with `delta0 = 0.01`. The `rand` draw at `:5` is discarded -- its only observable effect would
  be advancing the global RNG stream, but Rprop is a standalone leaf function that draws nothing
  else and is called fresh each step, so the discarded draw is value-neutral. The port hard-codes
  `delta0 = 0.01` and calls no RNG, matching the LIVE `:6` value bit-for-bit. *Fix candidate:*
  delete the dead `:5` line after parity. **Mutation:** restoring `:5` as the live delta0 (deleting
  `:6`) makes the init-branch `deltaweight = -sign(deriv)*delta0` random -> `rprop_sequence_bit_exact`
  fails on step 1.

- **[phase4c] Octave-compat: SMORMS3 empty-varargin lvalue cs-list miscount**
  (`tools/octave_harness/{smorms3_f_df,stage_smorms3}.m`; the .m source `SMORMS3.m:313` is
  UNCHANGED -- this is a harness-executor accommodation, not a source edit): `f_df_wrapper`
  assigns `[f, df_full, theta_local_out, obj.varargin_stored{:}] = obj.f_df(...)`. When SMORMS3 is
  constructed with no trailing varargin (`SMORMS3(f_df, theta)`), `obj.varargin_stored` is `{}` and
  the trailing `{:}` should expand to zero lvalues -- MATLAB requests 3 outputs, but Octave 11.3.0
  miscounts and raises `f_df: function called with too many outputs`. The smallest accommodation
  that keeps TIER 1 (the real classdef ctor + `optimization_step` + `f_df_wrapper` + update math all
  run unchanged): pass exactly ONE dummy varargin (`struct()`), so the cs-list is a well-defined
  single element, and have the injected `smorms3_f_df` echo it back as a 4th output. It is
  VALUE-NEUTRAL: the scripted gradient depends only on `eval_count`, and the update math never reads
  the varargin. The Python port (`Smorms3`) takes no such varargin -- its `f_df(theta, eval_count)`
  is the clean contract; the dummy exists only to work around Octave's lvalue-expansion bug in the
  oracle. Adjudicated per the CLAUDE.md rule "the .m source stays the contract; Octave the executor".

- **[phase4c] SMORMS3.m:335-336 `eps=1e-16` guards -- now pinned by a near-zero-gradient
  fixture** (`tools/octave_harness/stage_smorms3.m` eps-guard run; ported in
  `src/python/speech/optimizers.py` `Smorms3.optimization_step`): the review finding
  above (main-run coverage gap) noted `MMS >> eps` everywhere on the M=7/12-step main
  run and the round-trip run, so an `eps`-placement mutation was a no-op there. Closed
  by a third `stage_smorms3.m` sequence (M=1, 3 steps, constant `grad=1e-8`): step-1
  `MMS = 0.5*grad^2 ~= 5e-17`, the same order as `eps=1e-16` (`MMS+eps ~= 3*MMS`,
  extractor-asserted non-vacuous within 2 orders of magnitude, `scripts/
  extract_phase4c_fixtures.py::EPS_GUARD_RATIO_BAND`). *Pinned by:*
  `test_smorms3_eps_guard_active` (`tests/test_phase4c_optimizers.py`).
  **Mutation results (local, hand-verified, not committed as code):** (1) dropping
  `+eps` from `sqrt(MMS)+eps` (the dtheta denominator) changes `theta` at step 1 on
  this fixture, while leaving the main/round-trip goldens bit-identical -- confirms
  that guard is exercised HERE and only here. (2) dropping `+eps` from the `MMS+eps`
  occurrence at the delta-update site (`self.delta = 1 + self.delta * (1 -
  step_rate**2/(MMS+eps))`) changes `delta` at step 1 by ~22% (1.5 vs 1.8333) -- a
  real, non-ULP effect. (3) **honest gap, still open:** the OTHER `MMS+eps` occurrence
  -- the dtheta min-cap ratio inside `min(lrate, step_rate**2/(MMS+eps))` -- is a NO-OP
  even on this near-zero-gradient fixture, because `lrate` is still `1e-9`
  (pre-warmup) at step 1 and the `min()` saturates to `lrate` regardless of the ratio's
  value; dropping `eps` there alone leaves `dtheta`/`delta`/`theta` bit-identical.
  *Fix candidate:* none identified -- that occurrence would need a fixture where
  `step_rate^2/(MMS+eps) < lrate` AND `MMS` is eps-scale simultaneously, i.e. `lrate`
  warmed up (later step) while gradients stay near-zero (mms doesn't grow); left as a
  narrower follow-up if that call site ever needs its own golden.

- **[phase4c] Fixed-seed / injected-source determinism deviation -- REALIZED for QuantumPSO**
  (`src/python/speech/optimizers.py::quantum_pso`, `TableRng`; `QuantumPSO.m:91`): the legacy
  optimizer path is nondeterministic BY DESIGN (`QuantumPSO.m:91` `rand('state',sum(100*clock))`
  reseeds `rand` from the wall clock; no fixed seed anywhere), so no single legacy RUN is
  reproducible in principle. The port routes EVERY stochastic draw (rand/randperm/stblrnd) through
  an injected source -- a `TableRng` (sequential f64 reads of a committed table) for bit-pinning, or
  a plain numpy `Generator` for production. This preserves each operator's draw STRUCTURE/ORDER and
  makes the operator behaviour pinnable, but does not bit-match any wall-clock legacy run. The
  oracle is a MODIFIED-COPY of `QuantumPSO.m` (`tools/octave_harness/qpso_modified/`, NOT an edit to
  `legacy/`) with the clock reseed removed and rand/randperm/stblrnd substituted by the SAME table
  reads; both sides consume the identical committed `qpso_random_table.bin` (fixed numpy seed
  20260709). *Pinned by:* `test_qpso_trajectory_bit_exact_given_table` (bit-exact init +
  canary-gated trajectory + exact `cursor_end` draw count). *Fix candidate:* none -- a deliberate,
  documented divergence. SMORMS3/Rprop still draw no randomness, so their goldens remain fully
  deterministic without a table.

- **[phase4c] Exit-gate genome<->engine binding: DSP-config injection, not `config2weights`**
  (`src/python/speech/drivers/train.py`; legacy `CostFunction.m` -> `vec2struct` + `nnet2MatFile`
  + `network2config`): the legacy QPSO cost path decodes the FULL engine config AND the initial NN
  weights from each candidate genome, trains them, and writes the trained weights BACK into the
  genome (`network2config` -> the `out_param` re-encode). This port sizes/validates the search with
  the REAL vec2struct genome (`genome_length` + `masking_validation`, exactly the legacy), but binds
  it to the engine by SURGICALLY injecting only the two genome-decoded `CostPonderation` fields onto
  the byte-known-good committed base `.config` (the rest of the config stays identical to the
  committed one, so the engine never sees a malformed genome-derived config). The `[sad, lid]`
  weights are seeded from the config's committed `.bin` packs, NOT decoded from the genome, and the
  SMORMS3-trained weights are re-seeded per eval, NOT written back into the genome. *Why:* this keeps
  the exit gate a faithful full-loop DETERMINISM contract (QPSO outer + SMORMS3 inner + real engine
  via the seam) without the `config2weights`/`network2config` round trip -- and per-value legacy
  parity of the optimizer path is impossible in principle anyway (the clock-reseed deviation above).
  *Fix candidate:* wire the `weight_bridge` `config2weights` + `network2config` genome<->weight round
  trip at the end-to-end inference-output parity milestone. *Pinned by:*
  `test_full_train_loop_deterministic` (`tests/pyo3/test_exit_gate.py`, bit-identical checkpoints +
  cost history across two fixed-seed runs) + `test_twin_genome_length_and_masking`
  (`tests/test_phase4c_drivers.py`, the vec2struct sizing/gate is real). The BackPropagation.m
  normalize-tail strip (`weights(1:end-2*length(normalize.mean))`, :29/:57) IS reproduced faithfully:
  SMORMS3 steps the tail-stripped head, and the mean/std tail is folded back before every
  `Engine.set_weights` because the Rust `BLSTMNeuralNetwork::setWeights` demands the FULL vector
  (`flat.len() >= nb_of_weights()`, an `Err` below that).

- **[phase4d] CLOSED: the 4c lifecycle drivers shipped ALGO-6-ONLY -- now generalized to single-net
  algo-3 configs** (`src/python/speech/drivers/{train.py,state.py,test.py}`; cross-ref the two
  `[phase4c]` exit-gate entries above/below): the 4c drivers hard-assumed the 2-cell `[sad, lid]`
  shape everywhere -- `_backprop_inner` indexed `engine.weights(0)[1]` (IndexError on a single-net
  engine), `_ponderations` read `cfg["BLSTM_LID_CostPonderation"]` (KeyError -- an algo-3 vec2struct
  emits no LID field), `_tail_lengths` returned a `(sad, lid)` 2-tuple, `train` wrote `lid_weights.bin`
  from `final_w[1]`, and `evaluate` set `[sad, lid]` weights unconditionally. A single-net algo-3
  config therefore KeyError'd/IndexError'd (flagged as an Info finding in the 4c final review). Task 9
  drives the net count off `BackPropagation.m:11-13`'s cell contract (`weightsIni = cell(2,1)`, cell 1
  iff `PS.NS.BackPropagationActivated == 1`, cell 2 iff `(algo == 6) && LID.BackPropagationActivated`):
  `_tail_lengths(cfg, algo)` returns a `len`-1-or-2 list, `_ponderations` returns a `len`-1-or-2
  ponderation list, `_eval_config_text` injects `BLSTM_LID_*` only for algo 6, and `train`/`evaluate`
  gate the LID pack write/read on the algo. The algo-6 Twin path is behaviour-identical (the 4c exit
  gate `test_full_train_loop_deterministic` stays green untouched -- the regression sentinel). *Pinned
  by:* `tests/test_phase4d_algo3_drivers.py` (the 1-cell contract on the committed `parity_tupleA.config`
  vs the algo-6 `twin_train.config` foil) + `test_full_train_loop_algo3_deterministic`
  (`tests/pyo3/test_exit_gate.py`, the algo-3 analogue of the twin exit gate on the tier-2 spectral
  corpus, bit-identical checkpoints across two fixed-seed runs, no `lid_weights.bin` written).

- **[phase4c] Exit-gate config forces BOTH nets' backprop ON (the committed twin config has SAD off)**
  (`src/python/speech/drivers/train.py::_eval_config_text`; `tests/reference_data/phase4b/twin_train.config`):
  the committed twin config is `BLSTM_BackPropagationActivated false` / `BLSTM_LID_BackPropagationActivated
  true` -- only the LID net trains internally. `engine.forward_backward` reads
  `weights_derivatives(0)` = `[sad_deriv, lid_deriv]`; with SAD backprop OFF the SAD derivative matrix
  comes back EMPTY (`0x0`), so `average_derivs` indexes out of bounds. The exit-gate config therefore
  forces BOTH flags `true` (+ `Epochs 1` for the T9 single-eval semantics) so the 2-cell `[sad, lid]`
  SMORMS3 contract is genuinely exercised (both derivatives populated). *Fix candidate:* none needed
  -- an exit-gate config choice (train both nets), documented because it diverges from the committed
  config's flags. *Pinned by:* `test_full_train_loop_deterministic`.

- **[phase4c] `RunState` persists as JSON, not `ParamStruct.mat`**
  (`src/python/speech/drivers/state.py`; `Init_BLSTM.m:217` `save(...'ParamStruct.mat','PS')`): the
  legacy saves the free-form `PS` god-struct as a MATLAB `.mat`. The port persists a typed pydantic
  `RunState` as `run_state.json`. Safe deviation: the ENGINE consumes the flat `.config` + `.bin`
  weight packs and NEVER `ParamStruct.mat` (which is MATLAB-only optimizer bookkeeping), so the
  orchestrator's own state format is free to be a plain, diffable, deterministic JSON blob. *Fix
  candidate:* none -- a deliberate format choice. *Pinned by:* `test_run_state_json_roundtrip`
  (`tests/test_phase4c_drivers.py`, save/load equality).

- **[phase4c] `retrain` seeds `nnet_best` only, not the legacy half-population reuse**
  (`src/python/speech/drivers/retrain.py::retrain`; `ReTrain_BLSTM.m:803`
  `PSOseedparam = [nnet_best nnet_in(:,1:ceil(size(nnet_in,2)/2))]'`): the legacy resume
  seeds the QPSO population's first rows with BOTH the prior run's best genome (`nnet_best`)
  AND the top half of its final population (`nnet_in`), preserving more of the prior search's
  diversity across the resume boundary. This port's checkpoint (`RunState`/`TrainResult` JSON,
  see the entry above) stores only the gbest genome (`gbest.bin`), not the full terminal
  population, so `retrain` can only seed `nnet_best`. *Why:* consistent with the JSON-state
  deviation -- the checkpoint format was never designed to carry a population, only the winner.
  *Fix candidate:* if population diversity across resumes becomes load-bearing, checkpoint the
  QPSO's final population alongside `gbest.bin` and thread it through `seed_from_checkpoint`.
  *Pinned by:* `test_retrain_seed_from_checkpoint` (`tests/test_phase4c_drivers.py`, bit-exact
  `gbest.bin` reload) -- the single-genome seeding itself, not the missing population half.

- **[phase4c] QuantumPSO's four velocity banks are DEAD but STREAM-CONSUMING**
  (`legacy/Optimizer_V6.2.2/functions/QuantumPSO.m:375-411` vs the apply gate `:447`; ported in
  `src/python/speech/optimizers.py::quantum_pso`): each epoch computes `vel1..vel4` (Trelea sets
  1/2, Clerc type-1" with `chi`, common-PSO with linear `iwt`), drawing EIGHT `rand([ps,D])`
  matrices (`:375,376,382,383,389,390,403,404`). The only place those velocities are APPLIED to
  `pos` is inside `if (i > 20*me/2)` (`:447`) -- i.e. `i > 10*me`, which is unreachable for the
  loop `i = 1:me`. So the banks never move a particle, yet their 8*ps*D draws per epoch DO advance
  the shared RNG stream; dropping them shifts every downstream value (QDPSO update, DE/Levy gates,
  cost order). The port keeps the draws (a documented consume-and-discard) but skips the dead vel
  ARITHMETIC (parity-neutral: it feeds only the unreachable apply). The Octave modified-copy keeps
  the arithmetic verbatim for faithfulness. Same class of quirk as `:420` (a `sign(rand(ps,D)-0.5)`
  drawn then overwritten at `:442`) and the fully-dead-in-both-directions `MBest = mean(pbest)`
  (`:418`, unused since `:421` is commented). *Fix candidate:* delete the dead banks (and fix the
  `20*me/2` gate if velocity PSO was ever intended) after parity. *Pinned by:*
  `test_dead_banks_consume_stream`. **Mutation:** the `_dead_banks=False` variant (skipping the 8
  per-epoch vel draws) diverges the position trajectory from the pinned run at epoch 0 -- the
  non-vacuity proof that the dead draws are load-bearing for stream alignment.

- **[phase4c] QuantumPSO oracle substitution adjudications (Octave modified-copy)**
  (`tools/octave_harness/qpso_modified/{QuantumPSO,tbl_rand,tbl_randperm,tbl_stblrnd}.m`): the four
  RNG substitutions that make the wall-clock-reseeded `QuantumPSO.m` table-reproducible, each
  mirrored bit-for-bit by the Python `TableRng`/`levy_stable_cms`. (1) **Column-major fill**:
  MATLAB `rand([m,n])` lays its stream out column-major, so `tbl_rand`/`TableRng.rand` read `m*n`
  values and reshape with `order='F'` (Octave `reshape` default); numpy's default row-major fill
  would silently transpose every matrix draw. (2) **randperm**: MATLAB's builtin `randperm(n,k)` is
  not table-reproducible, so BOTH sides substitute an identical Durstenfeld/Fisher-Yates over 1..n
  consuming exactly `n-1` draws (`j=floor(rand*i)+1; swap p(i),p(j)`), then take the first k -- the
  selected neighbour indices match. (3) **stblrnd**: `stblrnd(1.3,1,0.5,0,1,D)` hits only the
  general `alpha!=1` Chambers-Mallows-Stuck branch (`stblrnd.m:75-82,96`), which draws V then W;
  ported as `levy_stable_cms`, table-fed in the same order. (4) **the backprop gate** (`:481`
  `(BackPropagationActivated>0) && (rand<0.5)`): with `BackPropagationActivated=0` the `&&`
  short-circuits and NO `rand<0.5` is drawn -- the port matches (a HOOK; T12 wires the real
  refinement). *Measurement:* on the oracle libm (Apple, this repo's oracle env) the ENTIRE
  trajectory including the transcendental `log(1/u)` + Levy path is bit-exact between Octave 11.3.0
  and numpy (measured 0-ULP); the `*_traj` comparators are canary-gated only for cross-libm CI. The
  copy shadows the vendored `QuantumPSO.m` via `addpath(...,'-begin')` (no other stage calls it).

- **[phase4c] printConfig.m emits the Forward_/Backward_/LID peephole-flag + MaxSaturation
  fields -- the ACTIVE inline condition DIVERGES from the commented-out `isNotExcluded`**
  (`legacy/Optimizer_V6.2.2/functions/printConfig.m:8-9` vs the dead `:45-77`; ported in
  `src/python/speech/genome.py::_printconfig_written`): the clean refactored `isNotExcluded`
  helper lists `AlgName_Forward_` / `AlgName_Backward_` / their `MaxSaturation` +
  `Is*PeepholesActive` variants ALL as excluded prefixes, so it would drop every
  `AlgName_Forward_*` field. But that helper is COMMENTED OUT at the call site (`:10`); the
  live boolean (`:8-9`) whitelists `MaxSaturation` + the three peephole-flag families (and any
  `_ActivationClocks`-suffixed field) with `~= 0` OR-arms, excluding ONLY the weight matrices
  (`Layer_*Weights` / `LSTMBlock` / `Output_Layer` / `NormalizeInput`). The port mirrors the
  LIVE condition, not the helper. *Verified against* the real `1_worker_1.config`
  (`tests/reference_data/phase0/`): `BLSTM_Forward_IsCellsPeepholesActive true` etc. ARE
  present, weight lines absent. *Pinned by:* `test_configstruct_fields_match`
  (`tests/test_phase4c_genome.py`, spectral/twin cases -- both emit the six peephole flags).
  *Mutation:* switching `_printconfig_written` to the `isNotExcluded` semantics (exclude every
  `AlgName_Forward_`/`Backward_`) drops the six peephole-flag lines -> the config golden fails.

- **[phase4c] vec2struct.m:964 LID `decision_thresh_falling` is unconditionally overwritten
  with `-decision_thresh_rising`** (`legacy/Optimizer_V6.2.2/functions/vec2struct.m:954-965`;
  ported in `src/python/speech/genome.py::_lid`): the LID falling threshold runs the full
  param-decode + mask + `> rising` clamp branch (all of which write `out_param`), then the very
  next line before `setfield` does `fieldValue = -configStruct.AlgName_LID_decision_thresh_rising`
  -- discarding the just-computed value for the STORED config (out_param keeps the branch
  result). The SAD `decision_thresh_falling` (`:50-60`) has no such negate -- a load-bearing
  asymmetry. *Pinned by:* `test_twin_lid_falling_negate_quirk` (`falling == -rising`).
  *Mutation:* dropping the `fieldValue = -rising` line makes the twin config's
  `BLSTM_LID_decision_thresh_falling` carry the clamped decode instead of `-rising` -> fails.

- **[phase4c] The calibration `CostLawParam*`/`CostLawThresh*` fields encode `out_param`
  UNCONDITIONALLY (not just under a mask), and clamp to [0,1]; `ComputeDeltasNb` rounds WITHOUT
  `abs`; `min_speech(3)` is floored at 0 only for algo < 5** (`vec2struct.m:1384-1421`, `:580`,
  `:239-241`; ported in `genome.py::_clamped01`, `_algo_spectral`, `_front_matter`): the six
  calibration clamps write `out_param(count) = fieldValue*adim` before the `isfield` mask check
  (so a sorted/clamped genome round-trips even unmasked); the law strings decode via
  `rem(round(5*|p|/adim),5) -> {log,linear,square,sqrt,cubic}`. `ComputeDeltasNb` uses
  `round(param)` (signed) unlike its `abs`-guarded neighbours. `min_speech`'s third element is
  `max(0,.)`-floored only below algo 5. *Pinned by:* `test_calib_law_decode_and_clamps`
  (sqrt/cubic + clamp-high `1`/clamp-low `0`) + `test_out_param_inverse_matches` (bit-exact).
  *Mutation:* gating the `_clamped01` out_param write behind the mask (as the string-law fields
  are) leaves the calib `out_param` == input param at those positions -> the out_param golden fails.

- **[phase4c] `count_param` (the genome length) is the 1-based cursor's FINAL value =
  `len(genome) + 1`, and is a pure function of the PS spec (algo + net sizes + balance +
  flags), independent of param values and the mask** (`vec2struct.m:30` init `count_param = 1`,
  returned at `:1`; `genome_length` in `src/python/speech/genome.py`): `Train_BLSTM_Seg.m:682`
  sets `PS.NS.ncoef = count_param` (the +1 form) then immediately overwrites it with `5e4`
  (`:683`) -- a harmless over-allocation, so the off-by-one never bites operationally. The port
  returns the identical `count_param`; `genome_length` runs the same walk over a zero genome
  (mask-independent). *Pinned by:* `test_count_param_matches` (STRICT vs Octave per case, and
  `count == len(param)+1`, and `genome_length == count`). *Mutation:* any single miscounted
  block (e.g. Cell `nbNeed` using `+2+3` instead of `+1`) shifts every downstream genome index
  -> both the count and out_param goldens fail (risk R3).

- **[phase4c] L2 regularization is wired for the LID net ONLY, never the SAD/seg net**
  (`ComputeCost.m:362` gradient `MultiDerivLID += L2_regul*(MultiWeightsLID.*(1-isBiasLID))` and
  `:554` cost `NNCostLID += L2_regul*sum((MultiWeightsLID.*(1-isBiasLID)).^2)/2`, both inside the
  `if (PS.VP.algo == 6)` LID block; the SAD deriv `MultiDeriv` is averaged at `:357` with NO L2
  term and `NNCostSeg` at `:430` gets none either; ported in `engine.py::forward_backward`): the
  seg/SAD network is left unregularized regardless of `L2_regul`. The port reproduces the
  asymmetry -- `forward_backward` applies `l2_penalty` to net index 1 (LID) only, never net 0.
  *Pinned by:* `test_l2_penalty` (the helper) + `test_forward_backward_tier2_determinism` (algo 3,
  `l2` default 0 -> no L2 on the seg net). *Mutation:* applying `l2_penalty` to `gradients[0]`
  (the SAD net) in `forward_backward` would double-count regularization the legacy never applies.

- **[phase4c] ComputeCost stage is FALLBACK-TIER (stage-local transcription of the assembly
  lines), not TIER 1** (`tools/octave_harness/stage_computecost.m`; adjudication): unlike the
  SMORMS3/Rprop/vec2struct stages (which `addpath` + CALL the vendored `.m` unchanged),
  `ComputeCost.m` cannot be driven wholesale in Octave -- its top half shells out to the engine
  (`system('python RunFsp.py ...')`, `:173-191`) after a `vec2struct`+`nnet2MatFile` config-write
  loop and then LOADS the worker `.mat`/`.bin` the shell-out wrote (`:216-282`), and the
  `!`-escape cleaning (`:37`) deletes any pre-injected worker files before the shell-out, so there
  is no injection point that leaves the vendored `.m` unmodified. The stage transcribes the PURE
  assembly lines (`:285-652`: sortrows `[1 2 3]`, deriv averaging, pooled stats, L2, the balance
  0/3/4/5/10 error + cost) line-for-line with `% legacy:` provenance and lets Octave execute the
  real MATLAB builtins (sortrows/median/hist/std/cumsum/exp/log). The engine-shelling top half is
  the seam -- covered separately by `tests/pyo3` against `speech_rs.Engine`, not by this stage.
  *Pinned by:* `tests/test_phase4c_engine_cost.py` (all groups). *Mutation:* the balance-3-vs-4
  over-90 saturation coefficient (`0*` vs `1*`, `:450`/`:458`) is checked to DIVERGE
  (`test_balance_3_vs_4_over90_saturation` + the extractor's non-vacuity guard).

- **[phase4c] The fractional-error `mean`/`mean(.^2)` cost scalars need an always-tolerant
  comparator (`close`), NOT the oracle-exact libm canary -- and NOT because of summation order**
  (`engine.py::compute_cost` `:626-651` balance cost; `tests/test_phase4c_engine_cost.py::
  _assert_close_always`). Originally misdiagnosed as a numpy-pairwise-vs-Octave-sequential
  summation-order difference; a review pass (empirical) found that diagnosis wrong: a plain
  sequential Python loop (`s=0.0; for v in x: s+=v; s/len(x)`) reproduces `np.mean`'s bit pattern
  EXACTLY on every case below, so the residual cannot be reduction order (order is invariant in
  IEEE double at these `n`, and numpy's pairwise summation only diverges from sequential at much
  larger `n` than these 3-4-element vectors anyway). The real cause is that Octave accumulates
  `mean`/`sum` internally in EXTENDED precision (x87 80-bit / long-double on this toolchain), so
  its result can land 1 ULP away from ANY double-precision loop-order variant -- a gap that is
  LOOP-ORDER-INVARIANT in double and therefore genuinely unfixable by "port the summation order
  more faithfully" (the repo's usual ascending-loop-vs-numpy-pairwise fix for this class of gap
  does not apply here). A tolerant bound is the correct and only remedy.

  Evidence (measured on the oracle env; `np.mean(x) == seq_mean(x)` bit-for-bit in every row,
  yet both differ from Octave by 1 ULP on 3 of 8 cases):

  | golden | np.mean | sequential-loop mean | == np.mean? | Octave | ULP(np vs Octave) |
  |---|---|---|---|---|---|
  | cb0_cost | 102.66666666666667 | 102.66666666666667 | yes | 102.66666666666667 | 0 |
  | cb5_cost | 45.0 | 45.0 | yes | 45.0 | 0 |
  | cb3_cost | 0.4833333333333334 | 0.4833333333333334 | yes | 0.48333333333333334 | 1 |
  | cb4_cost | 0.5666666666666668 | 0.5666666666666668 | yes | 0.5666666666666667 | 1 |
  | tier2_b0_cost | 84.0967032967033 | 84.0967032967033 | yes | 84.0967032967033 | 0 |
  | tier2_b5_cost | 36.78 | 36.78 | yes | 36.78 | 0 |
  | b10a_cost | 3.6027306048400947 | 3.6027306048400947 | yes | 3.602730604840095 | 1 |
  | b10b_cost | 2.8405808008508533 | 2.8405808008508533 | yes | 2.8405808008508533 | 0 |

  cb0/cb5 (INTEGER-error means, exact `/n`, no accumulated rounding to disagree about) and the
  3 goldens measuring 0 ULP above (tier2_b0/tier2_b5/b10b) are promoted to the oracle-exact
  `canary` comparator; only the genuinely 1-ULP-off cases (cb3/cb4/b10a) stay `close`
  (always-tolerant `<=4 ULP or 512*2^-52*max(|want|,1)`, applied on every platform including the
  oracle env, since the gap is not a libm split `canary` could gate on). *Pinned by:* the
  `close`/`canary`-classified goldens in the manifest. *Mutation:* classifying cb3_cost as
  `canary` (as first drafted) fails on the oracle env -- the extended-precision gap survives the
  libm gate, because it was never a libm gate in the first place.

- **[phase4c] The 8 `computecost_*_cpumean.bin` goldens were dumped but never asserted** (review
  follow-up to Task 9): `ComputeCost.m:432-436`'s `cpu_mean = median(Error_vad(:,4))` term feeds
  balances 0/3/4/5/10 (`engine.py::compute_cost`'s `cpu_term`/`error` formulas), and the extractor
  already dumped a `cpumean` golden alongside every `error`/`cost`/`nnseg` group, but no test read
  them back -- pure dead pinning. Now asserted in every balance-law test via `bd.cpu_mean` against
  `computecost_<group>_cpumean.bin`. All 9 cases (cb0/3/4/5, tier2_b0/b5, b10a/b10b/b10c) measure
  bit-exact numpy-vs-Octave (`np.median` vs Octave `median`) and are classified STRICT: `median` is
  a single sort + at-most-one `/2` on already-materialized values, not a multi-term reduction, so
  it never exhibits the extended-precision accumulation gap the `mean`/`mean(.^2)` cost scalars
  above do. *Pinned by:* the `_cpumean` assertions added to `test_balance_crafted`,
  `test_balance_tier2_committed`, `test_balance10_two_class_cutoff_search`,
  `test_balance10_three_class_else_branch`, `test_balance10_two_class_zero_zero_interior_cutoff`.

- **[phase4c] The balance-10 zero-zero interior-cutoff branch (`ComputeCost.m:571-572`) was
  transcribed but unpinned** (review follow-up to Task 9): `_balance10_cutoff`'s
  `if n[pos]==0 and n2[pos]==0: cutoff = (centers[first_n]+centers[last_n2])/2` branch fires only
  when the two per-class score CDFs (`n` ascending, `n2` descending) leave an interior span of the
  0-99.99 hist grid where BOTH are identically zero -- i.e. the class-1-fired and class-2-fired
  score clusters are cleanly separated with a gap between them, so the plain `argmin(|n-n2|)`
  position lands in a flat zero-zero plateau rather than on a genuine crossing, and the code
  instead bridges the gap by taking the midpoint of the two clusters' nearest edges. Neither b10a
  (crossing inside an overlapping plateau, both curves at 50) nor b10b (>2-class else branch, no
  `n`/`n2` at all) exercises it. Added a crafted 2-class case (`b10c`: class-1-fired scores
  ~290/295 -> `tmp2` max 10; class-2-fired scores ~280/285 -> `tmp` min 80, leaving the zero-zero
  span `(10,80)`) that lands the branch's midpoint cutoff at 44.995 -- bit-exact Python vs Octave,
  and far from the ~10.005 a reverted/un-guarded `cutoff = t(pos)` branch would emit instead (the
  first bin of the zero-zero plateau, since `argmin`/Octave `min` both return the first minimum).
  *Pinned by:* `test_balance10_two_class_zero_zero_interior_cutoff` (`tests/
  test_phase4c_engine_cost.py`); the extractor's non-vacuity guard SystemExits if `b10c_cutoff`
  falls outside `(30, 60)`. *Mutation:* deleting the `if` guard (always taking `cutoff = t(pos)`)
  would move b10c's cutoff to ~10.005, failing both the golden compare and the `(30, 60)` band.

- **[phase4c] `CreateBatches.m`'s non-multilingual `nbOfTargetClasses>1` branch indexes
  `Cases`/`WorstCases` by LOOP POSITION, not class VALUE -- silently clobbering the aggregate
  slot for the natural contiguous class labeling -- FIXED (phase 5, commit `1a18d18`)**
  (`CreateBatches.m:43-60`; `src/python/speech/batching.py::create_batches`). The loop is
  `for ii = 1:length(possibleValues)`, and for a
  target class (`0 < possibleValues(ii) < nbOfTargetClasses`) it writes `Cases(ii)` -- the LOOP
  COUNTER `ii`, not `possibleValues(ii)` (the class value itself). `possibleValues` is
  `unique(...)`, sorted ascending. When class values are the natural contiguous labeling
  `0, 1, ..., nbOfTargetClasses-1` (0 = non-target catch-all, 1..N-1 = targets -- the obvious
  choice), the LAST loop position (`ii = nbOfTargetClasses`) lands on `possibleValues(ii) =
  nbOfTargetClasses-1`, which is ITSELF a valid target (`0 < nbOfTargetClasses-1 <
  nbOfTargetClasses`) -- so it overwrites `Cases(nbOfTargetClasses)`, the SAME slot pre-reserved
  for the non-target aggregate, with the top target class's data. Class-0's files (assigned to
  the aggregate at `ii=1`) are silently lost; `WorstCases(nbOfTargetClasses)` (derived from
  `Cases(nbOfTargetClasses)` AFTER the loop) inherits the same clobber. Was ported faithfully:
  the Python `create_batches` used to write `cases[ii]` (the Python loop position) for target
  classes, exactly mirroring the bug. A SEPARATE non-multilingual quirk, NOT part of this fix,
  still reproduced verbatim: the aggregate's own accumulation
  branch (`Cases(nbOfTargetClasses).index = [Cases(...).index; find(...)]`, no `randperm` call)
  is NEVER shuffled, unlike every target class's pool (`create_batches`'s
  `else` branch concatenates `find`-order indices with no `shuffled(...)` call).
  *Was pinned by:* `test_create_batches_clobber_quirk` (contiguous `{0,1,2}`/`nb_classes=3` ->
  `Cases(3)` ends up as class-2's data) vs `test_create_batches_multi_nb_clean_aggregate_and_
  rotation` (non-contiguous `{1,2,5}`/`nb_classes=3` -> no clobber, clean aggregate) in `tests/
  test_phase4c_batching.py`; the extractor's non-vacuity guard SystemExits unless the clobber
  case measures class-2's data. *Old mutation record:* indexing by `possibleValues(ii)` instead
  of `ii` (the "obviously correct" fix) would change `test_create_batches_clobber_quirk`'s
  expected `Cases(3)` content and fail against the real Octave dump -- this is EXACTLY the fix
  applied below.
  **FIX (phase 5, F1):** `create_batches` now writes each target class to
  `cases[int(v) - 1]` -- the class VALUE `v` (satisfying `0 < v < nbOfTargetClasses`, so
  `v` ranges over the integers `1..nbOfTargetClasses-1`), 0-based-shifted by `-1`, never the
  loop position. `v - 1` bijects onto the non-reserved slots `0..nbOfTargetClasses-2`, so the
  reserved aggregate slot `nbOfTargetClasses-1` is never written by a target class for ANY
  labeling. RED: `test_create_batches_clobber_quirk` FAILED post-fix exactly as the old
  mutation record predicted (`cases[2].index` moved from the clobbered class-2 data `[5,4]`
  to the correct aggregate `[0,1]`); renamed and re-pinned as
  `test_create_batches_class_value_indexing`, asserting the fixed values (`cases[0]==[3,2]`
  for class value 1, `cases[1]==[5,4]` for class value 2, `cases[2]==[0,1]` for the
  aggregate) directly rather than via a golden file. ORACLE DIVERGENCE (documented, not
  regenerated per the Phase 5 fix protocol): the committed `batching_clobber_case3_index.bin`
  Octave fixture still pins the LEGACY `CreateBatches.m` output (`[4,5]`, 0-based) -- the real
  `CreateBatches.m` still has this bug, unmodified, so that fixture's value no longer matches
  the port on this one case; the fixture file is left in place, unused by the re-pinned test,
  as the legacy-oracle record. `test_create_batches_multi_nb_clean_aggregate_and_rotation` and
  `test_create_batches_single_and_rotation` were checked (per the sweep's cascade-analysis
  caution) and are UNAFFECTED -- both ran unchanged, still green, because their class values
  happen to sort into loop positions that already coincided with the class-value-derived slots
  (no non-contiguous-vs-contiguous mismatch in those fixtures); NOT blindly re-pinned.
  New mutation (revert-the-fix): reintroducing loop-position indexing (`slot = ii`) breaks
  `test_create_batches_class_value_indexing` (`cases[0].index` goes from `[3,2]` to `[]`, since
  loop position 0 corresponds to class value 0, which is never a target) -- confirmed, then
  reverted back to the fix.

- **[phase5] `create_batches`'s multilingual `nbOfTargetClasses>1` branch replaced the whole
  `CaseGroup` on a target-write collision with the aggregate slot, wiping `sub_cases` --
  PORT-INTRODUCED bug, NOT a legacy quirk -- FIXED (phase 5, commit `124d908`, F9), found in
  the Task 4 fix-wave review**
  (`src/python/speech/batching.py::create_batches`, the `else`/multilingual branch,
  `:359-364` at review time). Same loop-position collision mechanism as F1
  (`for ii, v in enumerate(possible): if 0 < v < nb_classes: cases[ii] = ...` -- `ii` is the
  LOOP POSITION, not the class value; under the natural contiguous 0-based labeling the last
  target write lands on `cases[nb_classes-1]`, the reserved aggregate slot), but a DIFFERENT
  downstream consequence: read the real vendored
  `legacy/Optimizer_V6.2.2/functions/CreateBatches.m:61-86` (the multilingual branch) and
  `GetNewBatch.m` directly to adjudicate. MATLAB's `Cases(ii).index = X` /
  `Cases(ii).currentPos = 1` are PER-FIELD struct writes -- they do not touch
  `Cases(ii).SubCases`/`.currentSubClass`, whatever those hold. And `GetNewBatch.m`'s
  multilingual branch (`:7-55`) never reads `Cases(N).index`/`.currentPos` for the aggregate
  slot `N = nbOfTargetClasses`: once `currentClass == length(Cases)`, the code only ever
  reads `.SubCases(subClass).index` and `.currentSubClass` (`:28-53`). So in the legacy, a
  target write colliding with the aggregate slot silently clobbers two DEAD `Cases` FIELDS
  specifically (`.index`/`.currentPos` there are never read again) and is benign FOR THOSE
  TWO FIELDS -- there is no `Cases`-side legacy bug to reproduce here, unlike F1's
  non-multilingual branch, where
  `GetNewBatch.m`'s non-multilingual path (`:57-81`) DOES read `Cases(currentClass).index`
  unconditionally for every slot including the aggregate.

  The Python port's pre-fix multilingual branch instead used
  `cases[ii] = CaseGroup(current_pos=0, index=idx)` -- a WHOLE-OBJECT REPLACE, not a
  per-field write. `CaseGroup` is a `@dataclass` with `sub_cases: list[SubCaseGroup] =
  field(default_factory=list)`; replacing the instance resets `sub_cases` to a FRESH EMPTY
  LIST, discarding anything appended there by an earlier non-target loop iteration. When a
  target's loop position collides with the aggregate slot (contiguous 0-based labeling, e.g.
  `fv=[0,0,1,1,2,2]`, `nb_classes=3`: `possible=[0,1,2]`, loop position 0/value 0 seeds
  `cases[2].sub_cases` via the non-target branch, then loop position 2/value 2 is itself a
  target -- `0 < 2 < 3` -- and collides with slot `nb_classes-1=2`), the whole-object replace
  wipes the just-appended `SubCaseGroup`. `get_new_batch`'s multilingual branch then reaches
  `sub = case.sub_cases[case.current_sub_class]` (`:449` at review time) against an EMPTY
  list -- `IndexError: list index out of range`, a live crash, not a silent wrong-result.
  This is a MATLAB-to-Python translation gap (mutable struct-array per-field semantics vs.
  an immutable-by-convention dataclass replace), with NO analog in the legacy MATLAB itself
  -- confirmed by the source read above, not assumed.

  Note the asymmetry with `worst`/`WorstCases`: the multilingual branch's post-loop line
  `worst[nb_classes - 1] = WorstCaseGroup(index=agg_arr, ...)` unconditionally reassigns the
  WHOLE aggregate `WorstCaseGroup` after the loop, from a separately and
  correctly-tracked `agg_worst` accumulator -- so any mid-loop collision on `worst[ii]` is
  fully repaired regardless (and `WorstCaseGroup` has no sibling field like `sub_cases` to
  lose in the first place). `cases` has no such repair -- the post-loop
  `cases[nb_classes - 1].current_sub_class = 0` line only touches `current_sub_class`, never
  reconstructs `sub_cases`.

  WORST-SIDE LEGACY-OBSERVABLE DIVERGENCE (found verifying the above by hand-simulating both
  `CreateBatches.m:61-86` and `GetNewBatch.m` against the real vendored source, T12 review):
  the port's `worst[nb_classes-1]` REPAIR mechanism (the paragraph above) is a PORT-ONLY
  design -- the raw legacy MATLAB has no `agg_worst`-style running accumulator at all;
  `WorstCases(nbOfTargetClasses).index` is a PLAIN FIELD OVERWRITE at BOTH the `else`
  (aggregate) branch (`:81`, incremental concat) and the colliding `if` (target) branch
  (`:70`, `Cases(ii).index(1:min(...))` -- the SAME statement F1 fixes for `Cases`, unfixed
  here since `Cases`/`WorstCases` are different fields with different read sites), so on a
  collision the LAST write wins: the colliding target class's OWN worst subset clobbers the
  aggregate's, exactly the F1-style mechanism, just observable here because (unlike `Cases`)
  `WorstCases(nbOfTargetClasses)` IS read downstream (`GetNewBatch.m`'s aggregate branch
  compares `sub.index` against `worst.index`). On the collision input
  `fv=[0,0,1,1,2,2]`/`nb_classes=3`/`nb_worst=1` (the same fixture
  `test_create_batches_multilingual_aggregate_slot_survives_collision` uses, shadowed
  `randperm` reversed): the PORT's `agg_worst` accumulator yields
  `worst[2].index=[1]` (0-based) so `get_new_batch` returns `[2, 0]` (matching that test);
  hand-simulating the LEGACY's field-overwrite instead yields `WorstCases(3).index=[6]`
  (1-based, i.e. `[5]` 0-based -- the colliding class-2's own worst pick, not the aggregate's);
  feeding that legacy-consistent `worst[2]` into the port's own (F9-unaffected)
  `get_new_batch` rotation, otherwise unchanged, then returns `[2, 1]` on the identical
  inputs -- a genuine, independently-reproduced VALUE divergence on the worst side,
  `port [2,0]` vs `legacy [2,1]`.
  Not part of F9's fix (F9 touches only `Cases`/`.sub_cases` field-mutation-vs-replace; this
  `worst[]` design predates Phase 5 and is UNCHANGED by F9), and not itself a bug: the port's
  `agg_worst` accumulator computes the semantically-intended aggregate worst set (files
  actually excluded from every non-target class's rotation), while the legacy's collision
  artifact is an incidental byproduct of loop order, not a "correct" alternative worth
  matching bit-for-bit -- author-intent-defensible, in the same FIXED-BY-DESIGN family as F4
  (a port choice that was already right, not a regression to chase). *Not independently
  pinned* (no dedicated test asserts the legacy-would-be `[2, 1]` value -- the divergence is
  a documentation finding, verified by hand simulation against `legacy/Optimizer_V6.2.2/
  functions/{CreateBatches,GetNewBatch}.m`, not by a committed oracle run).

  **FIX (phase 5, F9):** `create_batches`'s multilingual target-write branch now MUTATES the
  existing `CaseGroup`'s fields in place (`cases[ii].current_pos = 0; cases[ii].index = idx`)
  instead of replacing the object -- mirroring MATLAB's per-field struct-write semantics
  exactly. This is NOT F1's class-value-indexing fix (`cases[int(v) - 1]`): that mechanism
  does not transfer here, since (per the source read above) the colliding fields are dead
  regardless of which value's data ends up there -- moving the collision to a different slot
  via value-indexing would not change the outcome; only field-level mutation (matching what
  MATLAB actually does) is legacy-faithful. For non-colliding slots, field-mutation and
  whole-object-replace are behaviorally identical (both start from the same pre-seeded
  default-empty `CaseGroup` and set the same two fields), so this is a targeted fix, not a
  generalization that risks changing already-golden-tested rotation goldens.

  *RED:* no pre-existing test pinned the crash (a crash is not committed-suite "expected"
  behavior). Reproduced the reviewer's exact repro directly against the unmodified pre-fix
  code (`fv=[0,0,1,1,2,2]`, `nb_classes=3`, `multilingual=True`, `minibatch=2`,
  `nb_worst=1`): `IndexError` at `batching.py:449`, `case.sub_cases[case.current_sub_class]`
  against a wiped `[]`, matching the reviewer's citation exactly. New test added:
  `tests/test_phase4c_batching.py::test_create_batches_multilingual_aggregate_slot_survives_
  collision` -- FAILS (`IndexError`, uncaught) against the unmodified pre-fix code, PASSES
  post-fix, asserting the aggregate's `sub_cases` (length 1, `[1, 0]`) survives the collision
  and `get_new_batch` returns `[2, 0]` cleanly. Both pre-fix (crash) and post-fix (values)
  states were verified by running the actual code, not hand-derived only.

  *Mutation (revert-the-fix):* temporarily restored the whole-object replace
  (`cases[ii] = CaseGroup(current_pos=0, index=idx)`), reran the new test: fails at
  `assert len(batches.cases[2].sub_cases) == 1` (`0 == 1`, the aggregate's SubCases wiped
  again) -- confirmed the test is load-bearing on the field-mutation mechanism specifically,
  not some other incidental effect (the sibling dead-field assertion,
  `cases[2].index.tolist() == [5, 4]`, still passes under the mutation, since both
  mechanisms assign the identical `.index` value -- only `.sub_cases` discriminates, as
  designed). Reverted back to the fix; full `tests/test_phase4c_batching.py` (18 tests) and
  the fixed test rerun green.

  *Cross-reference:* `tests/test_phase4c_fixtures.py::test_batching_clobber_quirk_recorded`
  is the guard that keeps the F1 Octave-fixture divergence (`batching_clobber_case3_index.bin`,
  still pinning `CreateBatches.m`'s legacy clobbered output) honest -- it fails if that
  fixture is ever regenerated to match the fixed port output instead of staying the
  legacy-oracle record; unaffected by F9 (a different branch, no shared fixture), noted here
  since both are `create_batches` clobber-family fixes discovered/fixed one task apart.

- **[phase5] The release seam returned an all-zero gradient -- the folded derivatives were
  discarded -- FIXED (phase 5, commit `ff025ee`, F10)** -- PORT-INTRODUCED, latent since
  Phase 4c; a translation-gap/latent-seam bug in the class F9 belongs to, caught by the Task
  10 exit-gate discovery
  (`src/rust/src/engine/corpus_processor.rs::run_epoch` + `::weights_derivatives`). LEGACY
  behavior has no analog: the legacy C++ `saveWeights` writes the folded derivatives to
  `bestNNWeight_*.bin` and MATLAB reads them back, so the gradient always reached the
  optimizer. The port's PyO3 seam is a NEW surface: `forward_backward` (`engine.py`) does
  `set_weights -> run -> weights_derivatives(0)`, and the public `weights_derivatives(pos)`
  delegated to the MAIN bag's per-segmenter accumulator. But under the R6 static-lane
  determinism model (`run_epoch:355-442`), each lane CLONES the epoch-start bag, runs the
  backward on the CLONE, and folds the per-file derivatives into a LOCAL `derivs: BTreeMap`
  that `save_and_update_epoch` (Rprop) consumes and drops. The main bag's segmenters never
  run backprop, so their accumulator stays at the `set_weights` reset state (col0 = 0, only
  the normalize-tail counts present). The seam therefore returned an EXACTLY-ZERO gradient
  for every weight, and the modern SMORMS3 loop moved nothing (`||trained - init|| = 0`):
  the training foundation never trained. Invisible to every existing gate -- the seam-replay
  determinism/finiteness pins are all trivially satisfied by a zero gradient, and `grad_check`
  reads its OWN local `analytic_derivs` map (not the seam), so it was nonzero-correct the
  whole time, masking the seam bug.
  **FIX (phase 5, F10):** `run_epoch` now STASHES the folded `derivs` map on the processor
  (`seam_derivs: BTreeMap<usize, Vec<Array2<f64>>>`) at the end of the fold, before
  `save_and_update_epoch` consumes it; `weights_derivatives(pos)` returns the stash when
  present (the deterministic ascending-lane reduction -- IDENTICAL to what `grad_check`'s
  local map sees, by construction: same fold code), falling back to the bag only when no fold
  has run. `set_weights` CLEARS the stash (a weights change invalidates the cached gradient,
  so a set-without-run correctly falls back to the bag's reset state; the seam's own
  set->run->read flow repopulates it in the intervening run, so this is invisible there).
  Rust-side only: `engine.py` reads the derivatives back through the seam and never
  reimplements the fold.
  **RED / re-pin:** the always-missing pin. NEW Rust test
  `src/rust/tests/phase4c_api.rs::weights_derivatives_nonzero_through_seam` -- a backprop-on
  algo-4 corpus, Epochs 0 + Epsilon 0 (run_solo), asserts the seam returns a NONZERO col0 AND
  that its normalized `col0/col1` bit-matches `grad_check`'s analytic backprop column (the
  identity the F10 correctness claim rests on). NEW pyo3 pins
  `tests/pyo3/test_seam_replay.py::test_forward_backward_returns_nonzero_gradient` and
  `::test_forward_backward_smorms3_moves_weights` (a few SMORMS3 steps move the weights off
  init). No golden re-pinned (this surface had no prior gradient assertion).
  **Mutation (revert-the-fix):** with `weights_derivatives` reverted to the bare bag read, the
  Rust test fails at "got all-zero col0" and the pyo3 `smorms3_moves_weights` fails with
  `||trained-init||=0.0` -- the exact T10-diagnosed no-op. Reverted back.
  **Oracle divergence:** none -- the seam is a port-only surface with no C++/Octave harness
  counterpart (the legacy wrote derivs via a different mechanism); nothing regenerated.

- **[phase5] Single-eval gradient routed through the engine-internal `train()` -- Epochs=1
  was a 4c misroute -- FIXED (phase 5, commit `55bee98`, F11)** -- PORT-INTRODUCED, latent
  since Phase 4c
  (`src/python/speech/drivers/train.py::_modern_config_text`, `::_eval_config_text`). LEGACY
  behavior: the optimizer computes the gradient at theta by shelling `fsp` once per gradient
  eval (`ComputeGradient.m -> CostFunction.m -> ComputeCost.m`), with the inner Rprop loop
  living in MATLAB (`CostFunction.m:248-291` re-shells `fsp` per Rprop step). `ComputeCost.m`
  writes the config via `vec2struct/printConfig` with the base config's
  `Neural_Networks_BackPropagation_Epochs` UNCHANGED, and the real production
  `1_worker_1.config` has NEITHER `Neural_Networks_BackPropagation_Epochs` NOR
  `Neural_Networks_Gradient_Check_Epsilon` (both default 0 in the C++ ctor,
  `CorpusProcessor.cpp:60-63`), so `fsp` dispatched to `runSolo()` -- a SINGLE fold at theta.
  The port's own architecture mirrors this: `_backprop_inner` owns the inner SMORMS3 loop, so
  each `forward_backward` must be one fold at theta. But the 4c config builders set
  `Neural_Networks_BackPropagation_Epochs = 1`, which the SAME C++ dispatch
  (`corpus_processor.rs::run` mirrors `CorpusProcessor.cpp:114-135`) routes through `train()`
  (epoch-0 solo + inner epoch + final eval = 3 folds, with 2 internal Rprop updates between
  them). So `forward_backward`'s reported cost and (post-F10) stashed gradient were measured
  at engine-INTERNALLY-Rprop-moved weights, not the input theta the outer SMORMS3 owns
  (measured on the SAD fixture: Epochs=1 f=0.1278 at moved weights vs Epochs=0 f=0.3474 at
  theta). Adjudicated at the legacy source: the legacy ran a SINGLE eval (runSolo), so
  Epochs=1 was a 4c misroute, fixed in BOTH builders (not "faithful-and-kept").
  **FIX (phase 5, F11):** `_modern_config_text` and `_eval_config_text` emit
  `Neural_Networks_BackPropagation_Epochs = 0`. `Epochs 0` -> `run_solo` (one fold at theta);
  backprop is gated on `BackPropagationActivated`, not Epochs, so the fold still harvests the
  gradient into the F10 stash. Combined with F10, from-scratch SAD training now converges
  (cost-at-theta 0.348 -> ~0.300 in ~8 SMORMS3 steps, then overshoots as SMORMS3 is expected
  to past the minimum).
  **RED / re-pin:** the config-text pins in
  `tests/test_phase4d_algo3_drivers.py` (`test_eval_config_text_algo{3,6}_*` now assert
  `Epochs 0`; new `test_config_texts_single_eval_are_epochs_zero_f11` covers both builders,
  backprop ON and OFF); the `test_forward_backward_smorms3_moves_weights` pin depends on the
  theta-anchored gradient (F11) to actually move. `tests/pyo3/test_seam_replay.py`'s
  `_seed_tier2_single_epoch` helper flipped from Epochs 1 to 0 (its name is now accurate).
  RE-PINS: the 4c exit-gate determinism tests (`test_exit_gate.py`,
  `test_phase5_train_modern_smoke.py`) SHIFT values but stay deterministic (they assert
  run-twice bit-identity, not absolute bytes) -- re-verified green, no hardcoded value
  changed. `test_genome_ponderation_moves_cost` BROKE and was re-derived: under forward-only
  Epochs-0 scoring the `CostPonderation` is a BACKWARD/cost-law weighting knob that does NOT
  move the forward-scoring cost (balance-5 mode-0 zeroes `nn_cost_seg`; balance-10 scores the
  ponderation-invariant LID calibration columns -- the old cost-delta only existed via the
  Epochs=1 training misroute). Re-pinned as `test_genome_ponderation_moves_gradient`: the
  ponderation robustly moves the LID gradient (`ponderate_weights_derivatives` scales col0,
  not col1; measured LID-gradient delta 0.42 between two genomes vs 0.0 forward-cost delta).
  **Mutation (revert-the-fix):** reverting either builder to `Epochs 1` makes the config
  pins fail and re-introduces the moved-weights cost anchor; `_seed_tier2_single_epoch` at
  Epochs 1 leaves `smorms3_moves_weights` green (the fold still runs) but at the wrong theta.
  **Oracle divergence:** none -- the config-text builders are port-only orchestration
  (the legacy wrote its config via `printConfig`); nothing regenerated.

- **[phase5] The Twin's from-scratch CONVERGENCE gate is DEFERRED to Phase 6; Phase 5 ships a
  MECHANICAL twin gate only (Task 10, user-ratified 2026-07-16)** -- a SPEC DEVIATION from the
  original Task 10 brief's dual-head convergence gate, not a fix flip.
  **Why the twin convergence gate is unsatisfiable on the committed data:** the committed Twin
  corpora hold exactly ONE file per language (the Mode-7 `twin_train` phSeq corpus is `s1`
  eng/us + `s2` vie/vie; the Mode-5 `twin_gradcheck` corpus is a single vie/vie wav). Gate (b)
  of the exit-gate contract (spec S1.7) requires the trained model to beat the untrained init
  on a HELD-OUT file of the SAME distribution, but any disjoint train/held-out split of a
  1-file-per-language corpus trains on one language and validates on a DIFFERENT one -- asking a
  2-class LID net to generalize from one example of one language to a held-out example of the
  OTHER is structurally impossible (measured in the T10 discovery: held-out 0.48509 -> 0.49546,
  i.e. WORSE). Gate (a) is also near-vacuous for the Twin: the one-hot phSeq input + tiny net
  drives near-saturated posteriors and a near-zero init train cost (~0.046), so the train-cost
  improvement over a bounded run is a negligible ~2% (0.04640 -> 0.04540), well below any honest
  margin. Neither is a foundation defect -- both engine bugs the discovery surfaced (F10 zero
  gradient, F11 Epochs misroute) are FIXED, and nonzero per-net gradients demonstrably flow --
  they are a CORPUS limitation that a real >= 2-file-per-language corpus (Phase 6) resolves.
  **User ratification (2026-07-16):** SAD carries the FULL convergence gate (`test_from_scratch
  _sad_converges` -- (a) train convergence, (b) held-out generalization on the 2-file tier-2
  spectral corpus where the held-out language DOES appear in training, (c) determinism); the
  Twin keeps a MECHANICAL gate only. `test_from_scratch_twin_mechanical` asserts, from a
  seeded He init through the modern loop's `_backprop_inner` core, that BOTH nets receive a
  nonzero gradient, BOTH nets' weights move off init, and the run is bit-deterministic -- the
  foundation the Phase 6 convergence gate will ride, minus the convergence claim itself.
  **Fixture choice (Mode 5, not Mode 7):** the mechanical gate uses `twin_gradcheck.config`
  (Mode 5, both nets `BackPropagationActivated true`), NOT `twin_train.config` (Mode 7, which
  ships `BLSTM_BackPropagationActivated false` -- the SAD net is a frozen feature extractor
  feeding the LID net and takes NO gradient, measured 0/537 nonzero even when the flag is
  forced on, so a "both nets move" gate is impossible on it). The config's `Gradient_Check_
  Epsilon` is overridden to 0 so `run()` takes the run_solo fold path, not gradCheck.
  **Discrimination:** the mechanical gate fails under the F10 mutation (revert `weights_
  derivatives` to the bare bag read) -- `forward_backward` then returns an all-zero gradient
  for both nets (the nonzero-gradient assert fails) and `_backprop_inner` moves nothing
  (`||trained-init|| == 0`, the movement assert fails), the exact pre-F10 no-op the T10
  discovery diagnosed. **Oracle divergence:** none -- the twin gate is a port-only from-scratch
  training contract (the legacy trained the twin via `saveWeights`/Rprop to `.bin`, a different
  mechanism the harness does not model); nothing regenerated. **Phase 6 pointer:** re-attach the
  twin convergence gate once LRE03/07 (or a sourced SAD corpus) supplies >= 2 files per language.

- **[phase5] `train_modern`'s balance-5 forward "validation" cost is UNFIT for from-scratch
  guidance on the committed fixtures; the Phase 5 training foundation is certified PIECEWISE,
  not end-to-end (Task 10 closeout, ratified scope)** (`drivers/train.py::_make_default_validate`/
  `train_modern`). The per-epoch validation signal `train_modern` records
  (`compute_cost(..., balance, ...)`, the discrete balance-law error rate -- the role
  `valid_batch.m` never got to play in the port) is STUCK at 30.0 across the whole from-scratch
  SAD trajectory on the tier-2 spectral corpus: the net's posteriors never cross the decision
  threshold from a fresh Xavier/He init, so the signal is a flat, useless plateau. Not a bug --
  it is exactly what a discrete VAD error rate does on an undertrained net -- but not a guidance
  signal either. `test_early_stop_triggers` (`tests/pyo3/test_exit_gate.py`) EXPLOITS this
  deliberately (the stuck plateau reliably fires patience, a genuine if unglamorous use of it).
  The Phase-5 exit gates (`test_from_scratch_sad_converges`/`test_from_scratch_twin_mechanical`)
  therefore do NOT anchor on this signal at all: they call `engine.forward_backward` directly and
  measure the DIFFERENTIABLE `NNCostSeg` objective (the same objective SMORMS3 descends),
  bypassing `train_modern`'s own validation wrapper entirely.
  **What this means for certification:** the training foundation is certified PIECEWISE -- the
  core primitives (`forward_backward`'s F10/F11-fixed gradient, SMORMS3, from-scratch convergence
  itself) are proven end to end on `NNCostSeg`; the WRAPPER's state machine (epoch loop,
  early-stop, checkpoint/resume) is proven correct as MACHINERY via the Task 8 stub-injected unit
  tests (patience boundary exact, strict-`<`-best selection, resume-from-last true equivalence)
  plus one engine-backed smoke/early-stop-firing test -- but no committed fixture demonstrates the
  WRAPPER'S OWN validation-driven early-stop selecting a genuinely-better model on a signal that
  actually moves.
  **Phase-6 pointer:** once real training data makes `NNCostSeg` itself (or a comparable
  continuous signal) usable as the per-epoch validation metric -- not the discrete balance-5 error
  rate -- re-validate `train_modern`'s early-stop/checkpoint selection end to end against it before
  trusting the wrapper unsupervised on real corpora.
  **Oracle divergence:** none -- `train_modern` and its validation wrapper are port-only; the
  legacy never wired per-epoch validation into a from-scratch loop at all, so there is no legacy
  behavior to diverge from.

- **[phase4c] Octave-compat: `randperm` shadowed with a fixed reverse permutation for the
  `batching` stage** (`tools/octave_harness/batching_shadow/randperm.m`). `CreateBatches.m`
  shuffles every per-class index pool via the builtin `randperm`, which would make the
  extractor's output non-reproducible run to run (breaking the "run twice, byte-identical"
  determinism contract every other Phase 4c stage relies on). Shadowed (via `addpath` ordering,
  the shadow dir added AFTER `functions_dir` so it wins -- Octave `addpath` prepends, so the
  LATER call takes precedence; verified empirically, since this is easy to get backwards) with
  `p = n:-1:1`, a closed-form function of `n` trivial to replicate in Python (a duck-typed
  `rng.permutation` stand-in reversing its input, `tests/test_phase4c_batching.py::_ReverseRng`)
  -- so `create_batches`'s OWN shuffle output is bit-pinned against the real `CreateBatches.m`,
  not just `GetNewBatch.m`'s RNG-free rotation. `CreateBatches.m` itself is unmodified (TIER 1).

- **[phase4c] Octave-compat: `CheckGrad.m`'s real `CostFunction.m` and diagnostic plot are both
  unusable in the harness -- shadowed via `addpath` precedence, `CheckGrad.m` itself untouched**
  (`tools/octave_harness/checkgrad_shadow/`). Two independent problems: (1) the real
  `CostFunction.m` shells out to the engine (`system('python RunFsp.py ...')`) -- unusable in a
  fast, hermetic Octave-only harness; shadowed with a pure quadratic surrogate
  `f(w)=0.5*sum(c.*(w-target).^2)` over the REAL flat NN weight vector, re-derived from `param`
  via the REAL `vec2struct`+`nnet2MatFile` on every call (so `CheckGrad`'s own per-weight
  `network2config`/`weights2nnet`/`vec2struct` perturbation round trip is genuinely exercised and
  visible to the surrogate). (2) `CheckGrad.m`'s per-genome diagnostic plot (`:91-96`/`:155-159`)
  uses the old-style `subplot 211` call form, which errors ("invalid axes handle or RCN
  argument") under this Octave/FLTK combination independent of headlessness -- shadowed with
  no-op `figure`/`subplot`/`semilogy`/`hold`/`grid` stand-ins (verified empirically: the real
  calls error even with `--no-gui`, and the no-ops let the unmodified function run to completion).
  Both shadows live only in `checkgrad_shadow/`, added to the Octave path AFTER `functions_dir`.

- **[phase4c] Genuine legacy bug surfaced by driving the real `CheckGrad.m`: `weights2nnet.m`
  never writes back the normalize mean/std tail `nnet2MatFile.m` appends to `weights`, so
  CheckGrad's last `2*length(normalize.mean)` numeric derivatives are always exactly 0 --
  FIXED-BY-DESIGN (phase 5, commit `2590e48`, F4)**
  (`nnet2MatFile.m:140-141` appends `nnet.normalize.mean;nnet.normalize.std` to the flat
  `weights` vector CheckGrad iterates `kk = 1:length(weights)` over; `weights2nnet.m:150-186`
  reconstructs `nnet.output.layer(*).weights` from `weights` and then RETURNS -- it never reads
  or writes `nnet.normalize.*` at all). So perturbing weight index `kk` in the tail (`modWeights
  (kk) = modWeights(kk)+epsilon`) has NO EFFECT on the reconstructed `nnet_mod`, hence no effect
  on the config/param round trip, hence the central-diff numerator is always `PlusNNCost -
  MinusNNCost = 0` for those 2 entries -- while the analytic backprop derivative at the same
  index is whatever the (real or surrogate) `CostFunction` computed, generally nonzero. This is
  an ASYMMETRY vs the LID branch (`CheckGrad.m:109`, `weightsLID = weightsLID(1:end-2*length
  (normalizeLID.mean));`), which explicitly TRIMS that untestable tail before its own loop -- the
  SAD branch (`:45-96`) has no equivalent trim. Measured on the Task 10 golden (algo-3, tiny net,
  `Kw=53`, `2*length(normalize.mean)=2`): `MultiDeriv_Num(52:53) = [0, 0]` while `MultiDeriv_
  BackProp(52:53)` are both nonzero (~13.5 and ~5.6). NOT reproduced by the Python port:
  `speech.scoring.check_grad` is a GENERIC central-diff utility that perturbs its `weights`
  argument DIRECTLY (no config round trip), so it has no way to inherit this bug and correctly
  produces a proper nonzero numeric derivative at those indices -- a deliberate, documented,
  and tested divergence, not an oversight. *Pinned by:* `test_checkgrad_normalize_tail_quirk_
  recorded` (`tests/test_phase4c_fixtures.py`, guards the golden fixture itself) and
  `test_check_grad_normalize_tail_quirk_documented_not_reproduced` (`tests/
  test_phase4c_scoring.py`, asserts the Python port's numeric/analytic AGREE at those indices,
  the opposite of the Octave golden). *Mutation:* the extractor's non-vacuity guard SystemExits
  if the golden's last 2 numeric entries are ever nonzero (would mean the bug -- or the harness
  setup exercising it -- silently stopped firing).

  **FIXED-BY-DESIGN (phase 5, F4):** the Phase 5 sweep's own correction (`docs/superpowers/
  plans/2026-07-10-phase5-fixlist.md`, "Plan corrections" item 2) already found this entry
  states the port does NOT reproduce the bug: `weight_bridge.nnet_to_flat` appends the
  mean/std tail (`weight_bridge.py:198-` -- see `element_count`/`_pack_lstm_layer` for the
  tail-inclusive element count) and `flat_to_nnet` takes it back, so the tail round-trips
  through `pack_weights`/`unpack_weights` like every other element, not silently dropped the
  way `weights2nnet.m` drops it. Verified (Phase 5, Task 5), no code change: the directed
  tests the correction cites were re-read and confirmed genuinely VALUE-exact, not merely
  structural --
  `tests/test_phase4c_weight_bridge.py::test_pack_unpack_flat_roundtrip`/
  `test_unpack_pack_net_roundtrip` round-trip `unpack_weights(flat, spec)` ->
  `pack_weights(...)` and assert `mean`/`std` equal via `_assert_nets_equal` (`np.array_equal`,
  exact) on hypothesis-generated data, `test_unpack_weights_matches_flat_to_nnet_on_real_
  fixture` closes the same loop on a REAL committed `.bin` fixture; and this entry's own
  `test_check_grad_normalize_tail_quirk_documented_not_reproduced` (`tests/
  test_phase4c_scoring.py`) already asserts the port's `check_grad` produces a proper
  nonzero numeric derivative at the tail indices, the opposite of the legacy bug. No
  DIRECTED tail-round-trip assert was missing, so no new pin was added -- see
  `.superpowers/sdd/task-5-report.md` for the verification trace. RED/mutation do not apply
  (spec S2 step 1: nothing to fail, since the port never reproduced the bug in the first
  place -- a doc-only flip per the sweep's own adjudication, not a code fix).

- **[phase5] The QPSO genome no longer carries network weights -- a DELIBERATE DESIGN BREAK
  from the legacy genome (Task 9), NOT a fix flip** (`genome.weight_block_mask`/`_MaskTraceWalk`,
  `drivers/train.build_hyperparam_mask`/`train_hyperparam_search`)
  **Legacy behavior (still faithfully reproduced by `vec2struct`, untouched):** the legacy
  optimizer's genome is a flat vector that INTERLEAVES DSP/config hyperparameters with EVERY
  network weight/bias -- `vec2struct.m`'s `_nn_block`/`_output_neuron` arms decode the LSTM gate
  + cell matrices and the output-layer neurons straight out of the genome (`coeff_NN*(2p/adim-1)`),
  and the NormalizeInputMean/Std tail too, so QuantumPSO co-searched the weights alongside the
  hyperparameters (a candidate's net was SIZED and INITIALIZED from its own genome via the
  `config2network`/`network2config` round trip). `speech.genome.vec2struct` ports this bit-exactly
  and its 4c/4d Octave goldens are UNCHANGED by this task (the regression sentinel
  `test_phase5_genome_narrowing.py::test_vec2struct_4c_goldens_still_byte_green` re-runs them).
  **The modern-regime decision (user-locked, 2026-07-10):** network weights train by GRADIENT (the
  modern SMORMS3 loop, Task 8), so the outer QuantumPSO search is narrowed PERMANENTLY to the
  non-weight (DSP/config) genome -- `build_hyperparam_mask` pins every weight-block + normalize-tail
  FIELD out via a vec2struct field-name mask, and `train_hyperparam_search` searches only the
  remaining searchable dims (e.g. the tuple-A/tier2 algo-3 net: 265 total genome dims -> 54
  searchable; the twin: 504 -> 66). This is a DESIGN DIVERGENCE, not a legacy-bug fix, so the S2
  RED->re-pin->mutation protocol does not apply (there is no "old behavior" to fail against -- the
  legacy genome is a different, still-correct object; only its ROLE changed). **Two engine-level
  consequences, documented not-yet-closed (Phase 6 concern):** (1) the per-candidate eval scores
  the FIXED base weights forward-only (backprop OFF, `Epochs 0`) -- unlike the legacy, which
  re-init'd weights per candidate -- so a searchable genome that changes the FEATURE DIMENSION
  (nb_bins/nb_DCT/deltas) mismatches the fixed net's input size, and one that sets `TDCwindow > 0`
  hits the Twin's typed-bailed Mode-7 pitch pass; both are INVALID subregions of the DSP space that
  `train_hyperparam_search` PENALIZES (`_HYPERPARAM_PENALTY = 1e6`, caught incl. the pyo3
  `PanicException`) rather than crashing on -- a from-scratch search must steer away from invalid
  regions, but a cleaner fix (re-sizing/re-init'ing the net per candidate, or masking the
  dimension-defining keys too) is deferred to Phase 6 real training. (2) The generalized injection
  overlays the FULL decoded non-weight config (every DSP key, not just the 2 legacy
  `CostPonderation` fields) onto the byte-known-good base; the weights still reach the engine
  through the base config's committed `.bin` pack, never the genome. *Pinned by:*
  `tests/test_phase5_genome_narrowing.py` (mask carves EXACTLY the weight/normalize dims -- derived
  from the walk + an independent net-structure count, no magic numbers; injection covers exactly the
  non-weight keys; masking_validation passes; the vec2struct sentinel) + `tests/pyo3/test_exit_gate.py`
  (the narrowed 2-candidate non-vacuity: distinct DSP hyperparameters -> distinct engine configs AND
  distinct costs; and `train_hyperparam_search` runs deterministically end to end, finding a valid
  non-penalty gbest). *Oracle-divergence note:* the Octave `vec2struct`/`QuantumPSO` harnesses still
  describe the legacy weight-carrying genome; the port narrows it by design.

- **[phase4c] `MaskingValidation.m`'s FAIL case exploits a genuine encode/decode asymmetry in
  `vec2struct.m`'s `_padding_block`-family fields for negative mask values** (`vec2struct.m`
  `_padding_block`/`genome.py::_Walk._padding_block`, feeding `AlgName_speech_padding`/
  `AlgName_min_silence`/`AlgName_min_speech`). The ENCODE (mask branch) writes `out_param =
  (fv+0.1)*adim` using the RAW masked value `fv`; the DECODE (both the plain param->field path
  and the re-decode `MaskingValidation.m` performs on `out_param` with an EMPTY mask) computes
  `fv' = -0.1 + abs(p/adim)`. For `fv >= 0` these are inverses (`abs` is a no-op on a
  non-negative argument). For `fv < 0` they are NOT: encoding `fv=-5` (`adim=10`) gives
  `out_param = (-5+0.1)*10 = -49`; decoding `-49` gives `fv' = -0.1+abs(-4.9) = 4.8`, nothing like
  `-5` (mask-forced value, used by the FIRST `vec2struct` call's `configStruct`) OR `0` (the
  algo<5 floor-at-0 clamp genome.py's `_front_matter` applies to `min_speech[2]` specifically,
  also visible only in the first call's `cfg`, not in `out_param`). So `MaskingValidation.m`'s
  double round trip (`configStruct` from `vec2struct(vector,mask,...)` vs `maskedConfigStruct`
  from `vec2struct(out_param,[],...)`) genuinely diverges and correctly reports a mismatch --
  this is the validator doing its job (catching a mask value outside the field's implicit
  non-negative domain), not a false positive. Not "fixed" in the port (`speech.genome.vec2struct`
  reproduces the same asymmetric encode/decode); discovered by directly running the vendored
  `.m` with a crafted negative mask value, not by reading source alone.
  *Pinned by:* `test_masking_validation_fail_case` + `test_masking_validation_pass_case`
  (`tests/test_phase4c_scoring.py`) and `test_masking_manifest_present_and_pass_fail_split`
  (`tests/test_phase4c_fixtures.py`); the extractor's non-vacuity guard SystemExits unless
  `(pass_failed, fail_failed) == (0, 1)`. *Mutation:* clamping `fv` to `>= 0` before the mask
  encode (the "obviously correct" fix) would make the FAIL case pass too, collapsing the pinned
  contrast.

- **[phase4c] Mutation battery (Task 13): 8/8 mutations break a test as designed; one
  (item 2) via a different, adjacent test than the literally-named catcher -- recorded
  honestly, not papered over; no gap.** Each applied/run(targeted suite only, FOREGROUND)/
  reverted(`git checkout --`)/re-run in isolation; no Rust touched, so `cargo test` was
  skipped per the brief and only `uv run pytest tests` + `./lint_code.sh` ran once at the
  end. (1) SMORMS3 lrate warmup x10->x2 (`optimizers.py::Smorms3.optimization_step:94`,
  `self.lrate = min(self.lrate * 10, ...)` -> `* 2`) against
  `tests/test_phase4c_optimizers.py` -- FAILED as expected
  (`test_smorms3_lrate_warmup_x10_capped`: "lrate step 1: `0x1.12e0be826d695p-29` !=
  `0x1.5798ee2308c3ap-27`"; 4 more tests in the same file cascade-failed since `lrate`
  feeds `theta`); reverted, PASS. (2) One `eps=1e-16` placement moved: dropped `+ self.eps`
  from the dtheta line's `(np.sqrt(self.mms) + self.eps)` denominator
  (`optimizers.py:92`), leaving `eps` only at the `(self.mms + self.eps)` occurrence (the
  min-cap term, also independently present in the `self.delta` update) -- against
  `tests/test_phase4c_optimizers.py` -- the NAMED catcher `test_smorms3_trajectory_bit_exact`
  did NOT fail (the main run has `MMS >> eps` everywhere, consistent with the standing
  eps-guard finding above), but `test_smorms3_eps_guard_active` (the dedicated
  near-zero-gradient fixture) DID: "eps theta[0] step 1: `0.4999999985857864` !=
  `0.4999999985857865`" (a genuine 1-ULP divergence, not noise, per `assert_f64_close`'s
  strict-bits branch on this oracle env); the actual catcher differs from the plan's named
  one, recorded per the brief's allowance, not treated as a gap since a test DID break;
  reverted, PASS. (3) theta_out round-trip dropped: `self.theta = theta_out + dtheta` ->
  `self.theta = self.theta + dtheta` (`optimizers.py:96`, keeping the pre-`f_df` theta
  instead of the value `f_df` returned) against `tests/test_phase4c_optimizers.py` --
  FAILED as expected, ONLY `test_smorms3_theta_out_roundtrip_pinned`: "rt theta[0] step 1:
  `0.9999999985857865` != `1.0499999985857864`" (diverges by the crafted nonzero offset);
  reverted, PASS. (4) vec2struct mask-inverse write-back dropped: removed
  `self._set(0, fv * adim)` from the `AlgName_decision_thresh_rising` mask branch
  (`genome.py::_Walk._front_matter:238`) against `tests/test_phase4c_genome.py` -- FAILED
  as expected on both `test_out_param_inverse_matches[masked]` ("out_param bit mismatch")
  and `test_mask_fixes_field_and_writes_back` ("`0.5385131705458746` != `7.0`"); reverted,
  PASS. (5) `average_derivs`'s zero-count guard dropped: `d[:,0]/np.maximum(1.0, d[:,1])`
  -> `d[:,0]/d[:,1]` (`engine.py:128`) against `tests/test_phase4c_engine_cost.py` --
  FAILED as expected (`test_average_derivs`: "avg_out[1] strict: `inf` != `10.0`", plus a
  `RuntimeWarning: divide by zero"); reverted, PASS. (6) `l2_penalty` bias-exclusion
  flipped: `keep = 1.0 - is_bias` -> `keep = is_bias` (`engine.py:166`) against
  `tests/test_phase4c_engine_cost.py` -- FAILED as expected (`test_l2_penalty`:
  "l2_cost[0] strict: `0.053125000000000006` != `0.225`"); reverted, PASS. (7) QPSO
  contraction-expansion coefficient sign inverted: `pos = attractor + coef_exp_contr *
  signs * ...` -> `pos = attractor - coef_exp_contr * signs * ...`
  (`optimizers.py::quantum_pso:490`) against `tests/test_phase4c_qpso.py` -- FAILED as
  expected: `test_qpso_trajectory_bit_exact_given_table` ("pos_traj[0][0]:
  `3.0000301340132474` != `2.9999698659867526`") plus a cascading failure in
  `test_dead_banks_consume_stream` (reuses the same epoch-0 golden); reverted, PASS.
  (8) `get_new_batch` rotation off-by-one: `while len(batch) < batches.nb_cases_per_batch:`
  -> `<=` in the non-multilingual branch (`batching.py::get_new_batch:363`) against
  `tests/test_phase4c_batching.py` -- FAILED as expected on both non-multilingual rotation
  goldens: `test_create_batches_single_and_rotation` ("single step 0: batch mismatch,
  `[4, 3, 2]` == `[4, 3]`" -- one extra element pulled per call) and
  `test_create_batches_multi_nb_clean_aggregate_and_rotation` ("multi_nb step 0: batch
  mismatch, `[0, 2, 5]` == `[0, 2]`"); `test_create_batches_sub_and_rotation` is on the
  OTHER (multilingual) branch and correctly stayed green; reverted, PASS. *Process note,
  not committed as code:* two more literal "cursor advance/wrap" off-by-one variants were
  tried FIRST and discarded before ever reaching pytest -- advancing the cursor by 2
  instead of 1, and wrapping one index early (`>= case.index.size - 1`) -- each produces a
  GENUINE INFINITE LOOP on the committed `single`/`multi_nb` fixtures: both have a case
  group of size 2 whose single worst-excluded element becomes the cursor's fixed point
  under either mutation (confirmed with a bounded, alarm-guarded standalone harness, not
  committed; the runaway `pytest` processes were killed rather than left spinning). This is
  a real property of `get_new_batch`'s unbounded `while len(batch) < ...` loop -- it has no
  iteration cap, so a sufficiently-adversarial (or buggily-mutated) rotation state can spin
  forever on a small, single-worst-element case group; not a currently-shipping bug (the
  landed cursor logic is golden-pinned correct), but worth a future defensive iteration cap
  if `get_new_batch` is ever exposed to untrusted/adversarial batch configs. *Net verdict:*
  8/8 mutations break a test; item 2's catcher differs from the plan's named one (both are
  in the same suite, both true positives) -- no gap this round. Full transcript (diffs,
  commands, exact output) in `.superpowers/sdd/task-13-report.md`.

- **[phase4c/4d->phase5] `get_new_batch`'s two `while len(batch) < nbOfCasesPerBatch` loops
  had NO iteration cap -- a fully worst-excluded rotation state spins forever -- FIXED
  (phase 5, commit `1a18d18`)** (`GetNewBatch.m` has no cap anywhere; `src/python/speech/
  batching.py::get_new_batch`; hazard first surfaced as a process-note during the Phase 4d
  Task 13 mutation battery, see the entry directly above, ~line 2985). A state where EVERY
  reachable class (non-multilingual branch) or class/sub-class (multilingual branch) is
  fully worst-excluded (`case.index.size <= worst.index.size` everywhere reachable) makes
  every loop iteration take a branch that advances `current_class`/`current_sub_class`
  WITHOUT appending -- and since that branch never mutates `current_pos`, the state after
  one full non-progressing lap is IDENTICAL to the state before it, so the loop repeats
  identically forever. Not a currently-shipping bug on any committed fixture (the Task 13
  note already established this), but a genuine liveness hazard reachable once `get_new_batch`
  is wired into an untrusted/adversarial or from-scratch training loop (Phase 5 Task 8/10).
  RED (pin-old-behavior-first, per protocol -- a hang cannot be pinned directly in a
  committed test): a crafted single-class `Batches` (`index=[0,1]`, `worst=[0,1]`, i.e. the
  worst set IS the whole class) was run against the UNMODIFIED pre-fix `get_new_batch` in a
  standalone, `signal.alarm`-bounded harness (5s bound, not committed, mirroring the Task 13
  battery's own "bounded, alarm-guarded standalone harness" pattern) -- confirmed genuine
  non-termination (no return within 5s) before any fix landed.
  **FIX (phase 5, F2):** a new exported exception `BatchRotationStuck`
  (`src/python/speech/batching.py`), raised by BOTH loops after `2 * len(batches.validation)`
  CONSECUTIVE fruitless advances (an iteration that does not grow `batch`); the counter
  resets on every successful append, so a healthy rotation needing only a handful of
  worst-excluded misses per element never trips it. Cap derivation note: `len(batches.cases)`
  (the class count) was considered and REJECTED as the cap basis -- it is provably unsafe,
  since a single class can legitimately need up to `nb_worst` consecutive fruitless misses
  before succeeding (pigeonhole over its own worst-excluded positions), and the multilingual
  `SubCases` branch can legitimately need one full fruitless pass over every regular class
  PER sub-class before reaching the sub-class that finally succeeds -- both scale with corpus
  composition, not the class count (a single-class, `nb_worst>1` scenario demonstrates the
  class-count cap under-triggers-safety in exactly the direction that matters: false
  positives on a healthy rotation). `len(batches.validation)` (the corpus file count) safely
  dominates both, since every class/sub-class pool is a subset of the corpus. Pinned by two
  new tests in `tests/test_phase4c_batching.py`:
  `test_get_new_batch_raises_when_single_class_fully_excluded` (non-multilingual branch, the
  same crafted fixture as the RED harness) and
  `test_get_new_batch_raises_when_multilingual_fully_excluded` (multilingual branch: one
  regular target class plus the aggregate's one `SubCases` group, both fully excluded --
  `current_class` cycles `0 -> 1 -> 0 -> ...` forever pre-fix); plus a negative control,
  `test_get_new_batch_cap_does_not_false_trigger_on_a_healthy_rotation` (50 consecutive
  successful rotations over a non-excluded class, cap never trips). All pre-existing
  `test_phase4c_batching.py` rotation goldens (`single`, `multi_nb`, `sub`) were re-run
  unchanged post-fix and stay green -- the cap does not disturb any legitimate rotation.
  Mutation (revert-the-fix): disabling both `raise BatchRotationStuck` sites (`if fruitless
  >= cap and False:`) and re-running BOTH crafted fixtures through the same
  `signal.alarm`-bounded harness reproduces a genuine hang on EACH branch independently (5s
  bound, confirmed, not committed) -- proving the two new tests are load-bearing on the raise
  actually firing, not on some other incidental early exit; reverted back to the fix
  immediately after. No oracle-harness divergence to record: `GetNewBatch.m` itself has no
  iteration-cap concept to diverge from (there is no legacy behavior at the cap boundary to
  preserve a description of -- the legacy simply hangs, unconditionally, in this state).
  T12 nit (deliberate, not an oversight): the `BatchRotationStuck` message originally read
  "every reachable class/sub-class appears fully worst-excluded" unconditionally, which is
  not the whole truth -- the docstring's own caveat above documents a SECOND, pathological
  trigger (`nb_classes` absurdly large relative to `file_nb`, reachable only when
  `nb_worst=0` bypasses the degenerate gate's `nb_classes` term) where the cap fires as a
  non-hanging FALSE POSITIVE on a rotation that was merely slow, not stuck. Both raise sites
  now append "(or nb_classes is pathologically large relative to the corpus -- see
  get_new_batch's docstring)" so the message itself does not overclaim certainty about which
  case fired. String-only change (no test asserts on the message text, only the exception
  type via `pytest.raises(BatchRotationStuck)`), verified no test needed updating.

- **[4d] OpenSAD15 converter: NO LIVE ORACLE tier** (`src/python/speech/dataprep/opensad15.py`,
  ported from Python 2 `ProcessOpenSAD15Corpus.py:23-108`): unlike every other module in this
  repo, there is no interpreter left that can execute the legacy source -- Python 2 is EOL and
  not installed (or reasonably obtainable) in this environment, so nothing here can be
  cross-run against a live original. This is the phase's honest weakest validation tier:
  every fixture under `tests/reference_data/phase4d/opensad15/` is HAND-COMPUTED from a
  line-by-line reading of the legacy source (a second, independent-of-the-implementation
  arithmetic pass, not a dump from any run), plus one real cross-check beyond pure
  transcription -- the emitted VRCTS XML is parsed by the engine's own `load_vrcts`.
  *Pinned by:* `tests/test_phase4d_opensad15.py` (27 cases) + `src/rust/tests/phase4d_opensad15_vrcts.rs`.
- **[4d] OpenSAD15 `audiofile[:-5]` hard-coded 5-char extension strip**
  (`ProcessOpenSAD15Corpus.py:48,67,100`, ported as `_path_leaf(audio_path[:-5])`): the stem
  used for the XML `AudioDoc` name, the STM name field, and the lang fallback is derived by
  unconditionally slicing off the LAST 5 CHARACTERS of `audiofile`, not by stripping a detected
  extension. Correct only when the extension is exactly 5 chars including the dot (e.g.
  `.flac`); any other extension length silently corrupts the stem -- a 4-char `.sph` loses the
  last real character (`"rec01.sph"` -> stem `"corpus/rec0"` -> name `"rec0"`, dropping the
  `1`). Reproduced verbatim, not extension-aware. *Fix candidate:* switch to
  `Path(audiofile).stem` after parity (the legacy's own docs corpus is `.flac`-only, so this
  never manifested there). *Pinned by:*
  `test_case5_extension_length_quirk_truncates_stem` (`tests/test_phase4d_opensad15.py`).
- **[4d] OpenSAD15 `duration`/`lang` last-physical-row quirk (two instances, not one)**
  (`ProcessOpenSAD15Corpus.py:35-39`): inside the per-tab-row loop, `beg`/`endS`/`lang` are
  read from `tmp[2]`/`tmp[3]`/`tmp[8]` and `duration = endS` is assigned on EVERY row, BEFORE
  the S/RI filter `if` on line 39 -- so after the loop, both `duration` (the XML `sigdur` and
  the listing line's duration field) and `lang` (absent an explicit later fallback) hold
  whatever the LAST PHYSICAL ROW in the tab file had, whether or not that row passed the
  filter. A tab file's trailing annotation row is very often a non-speech sentinel (silence,
  end-of-file marker, etc.), so `duration`/`lang` routinely come from a row that never
  contributes a segment. Reproduced verbatim (single `duration = end_s` assignment inside the
  unconditional prefix of the loop body, ahead of the filter `if`, in `convert_tab_file`).
  *Pinned by:* `test_case1_duration_and_lang_taken_from_last_row_regardless_of_filter` and
  `test_case3_lang_from_last_row_overrides_earlier_explicit_lang`
  (`tests/test_phase4d_opensad15.py`) -- case1's last row is a filtered-out `NS` row with an
  empty lang field (exercising both the duration quirk and the lang-fallback path in one
  fixture); case3's two rows carry deliberately DIFFERENT explicit lang codes to prove the
  physically-last row wins over the only speech row.
- **[4d] OpenSAD15 `str(float)` is Python 2's 12-significant-digit format, not Python 3's
  shortest round-trip repr** (`ProcessOpenSAD15Corpus.py:53,57,61,100-102,106`, ported as
  `py2_str_float`): CPython 2.7's `float.__str__` formats via `PyOS_double_to_string(v, 'g',
  12, ...)` (C `%.12g`), while `float.__repr__` -- and Python 3's `str`, which is `repr` --
  uses the shortest decimal string that round-trips. The two diverge exactly when a float
  carries floating-point noise past 12 significant digits, which happens routinely when
  summing segment durations (e.g. case1's `spdur` lands on the double `16.366300000000003`;
  Python 2's `str()` prints `16.3663`, Python 3's plain `str()` would leak
  `16.366300000000003` verbatim into the XML/STM/listing outputs). Reproduced via
  `py2_str_float(x) = f"{x:.12g}"` plus Python 2's "always show a `.` or exponent" completion
  for bare-integral results (`f"{1.0:.12g}"` is the C-style `"1"`; Python 2's `str(1.0)` is
  `"1.0"`). Not validated against a live Python 2 interpreter (see the no-live-oracle entry
  above); derived from documented CPython 2.7 source behavior and the task brief's worked
  example. *Pinned by:* `test_py2_str_float_pinned_cases` (10 cases incl. the brief's
  `143.76000000000002` -> `"143.76"` example and both fixtures' actual noisy `spdur` values),
  `test_py2_str_float_nan_inf`, and `test_py2_str_float_differs_from_python3_str_on_noisy_case`
  (`tests/test_phase4d_opensad15.py`).
- **[4d] OpenSAD15 `Pool(20).imap` -> sequential loop (determinism deviation)**
  (`ProcessOpenSAD15Corpus.py:131-135`, ported as `process_opensad15`'s plain `for line in
  lines` loop): the legacy parallelizes `treat_file` over a 20-worker pool and writes results
  to the output listing in `imap` completion order, which need not match input order under
  uneven per-file work. The port processes the listing SEQUENTIALLY, so output order always
  equals input order -- a documented determinism deviation (an improvement, not a bug to
  preserve: nothing downstream depends on a specific non-deterministic order, and a
  reproducible order is strictly more testable). *Pinned by:*
  `test_process_opensad15_sequential_preserves_listing_order` (3-entry listing,
  deliberately out-of-alphabetical-order, asserts the output preserves exactly that order;
  `tests/test_phase4d_opensad15.py`).
- **[4d] OpenSAD15 lang-fallback `IndexError` latent bug (discovered, not exercised by the
  golden cases)** (`ProcessOpenSAD15Corpus.py:66-67`: `lang = path_leaf(audiofile[:-5]).split
  ('_')[-2]`): when the last tab row's lang field is empty AND the (possibly `[:-5]`-corrupted,
  see above) audio stem has fewer than 2 underscore-separated tokens, `split('_')[-2]` indexes
  past the result and raises -- `IndexError` in CPython 2 identically to Python 3's list
  indexing, so this needed no special handling to reproduce; Python 3's `list.__getitem__`
  raises the same way. Not a defensive addition -- the port lets it raise naturally, matching
  the legacy's crash-equivalent behavior on this input shape. None of the 5 golden fixtures
  exercise this path (all either give an explicit lang or use a 2+-token stem); recorded here
  as a discovered latent bug worth knowing about before pointing this converter at unvetted
  corpora. *Pinned by:*
  `test_lang_fallback_raises_indexerror_without_two_underscore_tokens`
  (`tests/test_phase4d_opensad15.py`).
- **[4d] `process_opensad15`'s blank-listing-line guard: added as an undocumented
  defensive divergence, removed in Task 15 for legacy parity** (`ProcessOpenSAD15Corpus.py`'s
  `treat_file(line)`, the per-listing-line loop ported as `process_opensad15`): an earlier
  pass of this port added `if not line: continue` ahead of the `line.split(";")` read, silently
  skipping blank rows in the input listing -- a defensive guard with no legacy counterpart and
  no test exercising it. The Task 5/T15 review caught this as an undocumented behavior
  divergence (ledgered in `.superpowers/sdd/progress.md`'s T5 finding) and Task 15 resolved it
  by REMOVAL, not documentation: the legacy `treat_file` has no such guard, so `tmp = line.
  split(';'); tmp[1]` on a blank line (`tmp == ['']`) raises `IndexError` in CPython 2 exactly
  as `fields = line.split(";"); fields[1]` does here in CPython 3 -- crash-equivalent, matching
  the same "let it raise naturally" posture already established for the lang-fallback
  `IndexError` above, not a fresh divergence. Removal was zero-test-breakage (no fixture ever
  fed a blank listing line). *Pinned by:*
  `test_process_opensad15_blank_listing_line_raises_indexerror` (`tests/test_phase4d_opensad15.py`).
- **[4d] Corpus augmentation: injected RNG is a DOCUMENTED CONVENTION, not a parity claim**
  (`src/python/speech/dataprep/augment.py`, ported from `AugmentCorpus.py:46-57`): the legacy
  draws `noise`/`noisetype`/`pitch`/`tempo` from `numpy.random` seeded implicitly off the
  Python 2 process's wall clock -- no legacy run is reproducible even against itself, so unlike
  every bit-exact-pinned module in this repo there is no trajectory to match, live or hand-
  computed. The port instead takes an injected `numpy.random.Generator` and preserves the
  legacy's DRAW ORDER verbatim (noise first, then noisetype, then pitch, then tempo, x5 per
  passing file) so a seeded run is at least reproducible within this port. *Pinned by:*
  `test_draw_variant_order_is_noise_then_noisetype_then_pitch_then_tempo` (an independent
  re-derivation off a freshly seeded generator, not a call into the function under test) and
  `test_draw_variant_noisetype_ladder` (4 brute-force-found seeds, one per ladder bucket)
  (`tests/test_phase4d_augment.py`).
- **[4d] Corpus augmentation: hard-coded `noise.wav`/`mod.wav` scratch-file collision hazard**
  (`AugmentCorpus.py:39,58,60,63,66`, ported as the `_NOISE_SCRATCH`/`_MOD_SCRATCH` constants in
  `augment.py`): every sox invocation -- the per-file silence-strip duration probe AND all 5
  per-variant pitch/tempo/noise/mix steps -- reads and writes the SAME two relative filenames in
  the current working directory. The legacy is safe only because it runs strictly sequentially,
  one `os.system()` call at a time, in one process/one cwd; nothing in this port changes that
  assumption (no unique temp names, no injectable scratch directory), so it inherits the same
  restriction: `augment_corpus` must not be run concurrently (multiple processes/threads sharing
  a cwd) or its scratch writes will race. Reproduced as-is rather than "fixed", since the whole
  command-string surface is the pinned artifact (see the module docstring). *Fix candidate:* a
  scratch-directory parameter with per-call-unique filenames, once nothing depends on the pinned
  literal `noise.wav`/`mod.wav` command tokens. *Pinned by:* the module docstring's collision-
  hazard note plus every `RecordingRunner`-based test in `tests/test_phase4d_augment.py` (the
  recorder's `duration()` intentionally cannot key readings by path -- only by call order --
  because the path is always the same overwritten `noise.wav`).
- **[4d] Corpus augmentation: a short listing line (fewer than 6 `;`-fields) raises an unguarded
  `IndexError`** (`AugmentCorpus.py:69` indexes `elems[1]`..`elems[5]` directly; ported as-is at
  `augment.py`'s listing-row emission): the legacy crashes identically at the same read, so the
  port reproduces rather than guards -- the same latent-crash class as opensad15's lang-fallback
  `[-2]` (its sibling entry above). Only fires on a gate-passing file whose listing row is
  malformed. *Fix candidate:* none while reproduce-bugs-exactly governs. *Pinned by:*
  `tests/test_phase4d_augment.py::test_augment_corpus_short_line_raises_indexerror_on_gate_pass`
  (this bullet was added in the T16 final-review cleanup for consistency with the sibling
  precedent; the docstring + test landed with Task 6 itself).
- **[4d] Corpus augmentation: `str(pitch)`/`str(tempo)`/`str(noise)` inside the sox COMMAND
  strings are Python 2's `str(float)`, reused from the OpenSAD15 `py2_str_float` helper**
  (`AugmentCorpus.py:60,63`, ported as `pitch_tempo_command`/`noise_synth_command` calling
  `speech.dataprep.opensad15.py2_str_float`): same CPython-2-vs-3 `str(float)` divergence
  documented for the OpenSAD15 converter (Task 5) -- `%.12g`-style 12-significant-digit
  formatting vs Python 3's shortest-round-trip `repr`-backed `str`. Only the SOX COMMAND
  arguments go through this path; the `%.3f`-formatted VARIANT FILENAME (`AugmentCorpus.py:58`,
  ported as `variant_filename`) is fixed-precision `%f`-style formatting, which does NOT diverge
  between Python 2 and 3, so it stays a plain `f"{x:.3f}"` with no `py2_str_float` involved --
  the two format paths in the same line of legacy code are NOT interchangeable and must not be
  collapsed into one helper. *Pinned by:*
  `test_pitch_tempo_command_uses_py2_str_float_not_python3_str` (a noisy-past-12-sig-figs pitch
  value) and `test_variant_filename_uses_fixed_point_not_py2_str_float`
  (`tests/test_phase4d_augment.py`).
- **[4d] `norm_stm_pk_cts_all_trans_03a.pl`'s number-normalization arm is a TYPED BAIL,
  `NormalizerUnavailable`, not a reproduction of a broken shell-out**
  (`src/python/speech/dataprep/stm_normalize.py::normalize_stm_full`, ported from
  `norm_stm_pk_cts_all_trans_03a.pl:109`): every content line (anything past the
  comment/blank checks) hits an UNCONDITIONAL backtick shell-out, `` $text=`echo "$text"
  | norm-tagger | norm-parser --lang=spa`; `` -- no branch sits between the `chomp` (:15)
  and that call. Both LIMSI binaries are LOST (not merely absent from this host; per the
  task brief they no longer exist anywhere), so this arm can never run for real. Verified
  LIVE in this environment (perl 5.34.1, binaries genuinely missing): perl's backticks do
  NOT raise on a missing command -- the shell writes "command not found" to stderr, stdout
  is empty, and the script silently continues with `$text = ""`, eventually printing
  `"$head ignore_time_segment_in_scoring\n"` for EVERY content line regardless of its
  actual content. That is an accident of this host's missing binaries, not legacy
  behavior (a host where `norm-tagger`/`norm-parser` exist would produce real tagged/
  parsed output), so the port does not reproduce it -- `normalize_stm_full` raises
  `NormalizerUnavailable(head, text)` at the equivalent point instead, carrying the
  pre-bail state for inspection. *Pinned by:*
  `test_full_content_line_raises_normalizer_unavailable`,
  `test_full_content_line_raise_carries_prefix_state`,
  `test_full_content_line_after_comments_still_raises` (`tests/test_phase4d_stm.py`).
- **[4d] `normalize_stm_full`'s pure-regex PREFIX (:17-105) has NO live-oracle backing**
  (`stm_normalize.py::_full_prefix`): unlike `normalize_stm_light` (byte-oracle-verified
  end to end against the real perl script), the prefix computation this function performs
  before bailing is NEVER observable through the real 03a script for a content line -- it
  always reaches the shell-out first (see the entry above), so there is no live run to
  diff against. Transcribed by hand from the source per the task brief ("port everything
  up to that point that is pure regex") and verified only by manual regex reasoning, not
  a live run -- the weakest-tier piece of this otherwise byte-oracle-verified module.
  *Pinned by:* `test_full_content_line_raise_carries_prefix_state` (white-box, asserts a
  specific computed value by hand-derivation, not by oracle diff).
- **[4d] `norm_stm_pk_cts_all_trans_03a.pl:49`'s `\[[rien]\]` filler rule is a BUG: a
  character class, not the literal word "rien"** (ported verbatim into
  `_FULL_PREFIX_SUBS`): `[rien]` inside the pattern is an UNESCAPED bracket expression --
  it matches any ONE of the characters r/i/e/n, not the 4-character literal string "rien"
  the author evidently intended (contrast :52's correctly-escaped `\[rire\]`, three lines
  later in the same file, which DOES match the literal word). Reproduced as-is (this whole
  rule is unreachable in practice, see the two entries above, so the bug has zero
  observable effect in this port -- recorded for completeness since the brief asked for
  every perl-vs-python semantic surprise the oracle work exposed).
- **[4d] `norm_stm_pkt_light.pl` vs `norm_stm_pk_cts_all_trans_03a.pl`: two silent
  cross-script divergences in otherwise-parallel code** (both scripts share the same
  head/text-split-then-substitution-cascade shape, but differ at two points): (1) the
  text JOIN uses a single space in `light` (`join ' ', @line[6..$#line]`, light:20) vs a
  DOUBLE space in `full` (`join '  ', @line[6..$#line]`, 03a:20); (2) the leading-space
  STRIP removes ALL leading spaces in `light` (`s/^ +//`, light:47, `+` quantifier) but
  only ONE in `full` (`s/^ //`, 03a:80, no quantifier). Both reproduced verbatim (`"
  ".join(tokens[6:])` vs `"  ".join(tokens[6:])`; `re.compile(r"^ +")` vs
  `re.compile(r"^ ")`). The `full` side of both has no live-oracle backing (see above);
  the `light` side of both IS byte-oracle-verified (every `light_*` fixture's text is
  built via the single-space join, and `light_case3`'s multi-`{fw}`-insertion cases
  exercise the ALL-leading-spaces strip after substitutions widen the leading run).
  *Pinned by:* `test_full_content_line_raise_carries_prefix_state` (documents both
  divergences inline) plus the `light_*` byte-oracle fixtures (`tests/test_phase4d_stm.py`).
- **[4d] FIX-WAVE (Task 7 review): `@line[0..5]` undef-padding vs Python slice
  truncation -- both normalizers' head split silently diverged on short lines**
  (`stm_normalize.py::normalize_stm_light` and `::normalize_stm_full`, both :19, ported
  from `norm_stm_pkt_light.pl:19` / `norm_stm_pk_cts_all_trans_03a.pl:19`): this is a
  QUIRK CLASS worth naming on its own, distinct from the two cross-script divergences
  above -- perl array slices with a FIXED literal range (`@line[0..5]`) always read six
  indices regardless of the array's actual length; an out-of-range index reads as
  `undef`, and `join`'s separator is still emitted for it (undef stringifies to `""`,
  but the separator between it and its neighbors is real). The shipped port instead used
  `" ".join(tokens[0:6])`, a Python slice, which silently TRUNCATES to `len(tokens)`
  elements for `len(tokens) < 6` and emits NO separator for the missing ones -- so every
  content line with fewer than 6 whitespace-delimited fields lost one join-separator
  space per missing field (`n` real fields, `6-n` missing: `6-n` fewer bytes in `head`,
  plus the outer `"$head $text\n"` format's own space is unaffected since it always
  fires). Lines with `n>=6` fields were never affected (`tokens[0:6]` and `@line[0..5]`
  agree exactly once the array is at least 6 long). VERIFIED LIVE against
  `/usr/bin/perl` (`tools/perl_oracle/run_norm.sh light`): a 3-field line
  (`file1 1 hi`) produces real-perl head `"file1 1 hi   "` (3 trailing pad spaces, one
  per undef slot) where the pre-fix port produced `"file1 1 hi"` (0 trailing spaces); a
  5-field line (`file1 1 spkA 0.00 hi`) produces real-perl head
  `"file1 1 spkA 0.00 hi "` (1 trailing pad space) vs the pre-fix port's 0. Fixed by a
  shared `_pad_head(tokens) -> tokens[:6] + [""] * max(0, 6 - len(tokens))` helper used
  by both callers' `:19` line, reproducing perl's join-with-undef byte output exactly
  for every `n`. `normalize_stm_full` has no live oracle for content lines (see above),
  but `head` is built at `:19`, strictly before the `:109`-equivalent
  `NormalizerUnavailable` raise, so the padded head IS observable on the exception even
  there -- pinned by hand-derivation, not by oracle diff. *Pinned by:*
  `test_pad_head_no_padding_when_six_or_more_tokens`,
  `test_pad_head_pads_short_token_lists_with_empty_strings` (direct pin on the shared
  helper), `test_light_short_line_head_matches_perl_undef_padding` +
  `test_normalize_stm_light_matches_perl_oracle[case6]` against the live-perl
  `light_case6.{in,out}` fixture (`tests/reference_data/phase4d/stm/`, added by this
  fix wave, `scripts/extract_phase4d_fixtures.py --stm-fixtures`, run twice per case
  plus twice across separate invocations to confirm determinism), and
  `test_full_short_line_head_is_padded_not_truncated` (hand-derived, exception-observed)
  (`tests/test_phase4d_stm.py`).
- **[4d] `perl s///g`'s non-overlap semantics on the filler alternation rules --
  Python's `re.sub` reproduces it with NO special-casing, verified against the live
  oracle** (`norm_stm_pkt_light.pl:29,31`, ported as `_LIGHT_SUBS`' first two entries):
  both filler rules are anchored `(^|\s)(alt1|alt2|...)(\s|$)`, and their replacement,
  `" {fw} "`, re-inserts the boundary spaces it consumed. Because perl's (and Python's)
  `s///g`/`re.sub` scan for non-overlapping matches and resume scanning from the END of
  each match (not re-offering already-consumed characters), a match's TRAILING `\s` is
  consumed as part of that match -- denying an immediately-adjacent next token its own
  REQUIRED leading `\s`. The net effect on a run of adjacent filler tokens is an
  ALTERNATING matched/unmatched pattern (1st, 3rd, 5th... replaced; 2nd, 4th... survive
  untouched), not "every filler token replaced". Confirmed identical between perl and a
  standalone Python `re.sub` on the exact same input BEFORE writing the port (not
  discovered after the fact) -- see the task-7 report for the worked comparison. No
  workaround needed: transcribing each rule as one plain `re.sub` call, in source order,
  reproduces this quirk automatically. *Pinned by:*
  `test_light_filler_class1_adjacent_tokens_alternate`,
  `test_light_filler_class2_adjacent_tokens_alternate` (`tests/test_phase4d_stm.py`),
  cross-checked against `light_case2`'s byte-oracle fixture.
- **[4d] `norm_stm_train_05a_p5_Quaero.sh` (the corpus-batch `.sh` driver around the 03a
  script) is NOT ported** (`legacy/norm_stm_train_05a_p5_Quaero.sh`, read-only reference):
  a bash wrapper that iterates a hard-coded set of `/users/vieru/...` corpus directories,
  lock-file-coordinates (`dotlockfile`) concurrent per-file processing across parallel
  invocations, detects each file's encoding via `/home/gauvain/bin/txtfile` and
  `iconv`s ISO-8859-1 files to UTF-8 before piping through
  `$script/norm_stm_pk_cts_all_trans_03a.pl am` and back to ISO-8859-1, then `sed`s out a
  couple of mojibake replacement-character bytes. Every path is a dead, personal/cluster-
  specific absolute path (mirrors the `tasks/vrcts.rs` `/usr/local/vrcts/...` shell-out
  precedent already in this repo) with no reusable logic beyond "call the 03a script with
  type=am" -- since the 03a script's content-line arm is itself a permanent typed bail
  (see above), a ported driver would have nothing left to drive. Not transcribed; recorded
  here per the task brief's explicit ask to note it.
- **[4d] `WriteListing.m`'s worker-shard `fliplr(length(index)-jj+1:-nbworker:1)`
  interleave leaves LEFTOVER files unevenly distributed across shards, front-loaded
  toward the LAST worker** (`src/python/speech/batching.py::write_listing`/
  `_worker_positions`, ported from `WriteListing.m:12-19`): when `length(index)` isn't
  evenly divisible by `nbworker`, the descending-stride range starts at
  `length(index)-jj+1` -- i.e. jj=1 starts CLOSEST to the end of the file list and steps
  backward by `nbworker`, so jj=1 is the shard most likely to pick up an extra element
  from the tail. Verified against the real vendored `.m` via Octave
  (`tools/octave_harness/stage_writelisting.m`, 7 files / 3 workers = 3+2+2): jj=1 gets
  3 files (positions 1,4,7), jj=2 and jj=3 get 2 each (3,6 and 2,5) -- the shards are a
  clean partition (every position covered exactly once) but NOT round-robin in the
  intuitive "worker 1 gets the extra" sense one might assume from `jj` ascending; it is
  "worker 1 gets the item closest to the end of the (reversed) stride". Reproduced
  exactly by `_worker_positions` (a straight 0-based port of the MATLAB range +
  `fliplr`), not rebalanced. *Pinned by:*
  `test_write_listing_worker_shards_match_golden`,
  `test_write_listing_worker_shards_partition_all_items_exactly_once`,
  `test_worker_positions_hand_derived` (`tests/test_phase4d_listing_writers.py`), byte-
  exact against `tests/reference_data/phase4d/listing/plain_worker_{1,2,3}.flst`
  (Octave TIER-1 goldens, `scripts/extract_phase4d_fixtures.py --listing-fixtures`).
- **[4d] `WriteWeightedListing.m`'s worker-shard block is dead code -- COMMENTED OUT in
  the vendored source, not ported** (`legacy/Optimizer_V6.2.2/functions/
  WriteWeightedListing.m:11-22`): unlike `WriteListing.m`, whose worker-shard loop is
  live, `WriteWeightedListing.m`'s otherwise-identical block is entirely `%`-commented.
  `write_weighted_listing` therefore has no shard variant at all -- it always writes a
  single flat file at the exact `path` given, with NO `.flst` suffix appended (contrast
  `write_listing`, which always appends `.flst`). Recorded because the asymmetry between
  the two nearly-identical sibling functions is easy to "fix" by accident when porting
  by analogy. *Pinned by:* `test_write_weighted_listing_no_suffix_appended`
  (`tests/test_phase4d_listing_writers.py`).
- **[4d] Octave's `sprintf('%g', ...)` matches Python's `f"{x:g}"` byte-for-byte at every
  probed style-switch boundary -- no port-side surprise found, but the surprise was
  worth checking before trusting it** (`write_weighted_listing`'s two numeric fields):
  both C-library-derived `%g` implementations agree on the style-switch rule (decimal
  when `-4 <= exponent < precision(6)`, else scientific with a minimum 2-digit,
  sign-forced exponent), 6-significant-figure rounding (`123456.789` -> `123457`),
  rounding CARRIES that cross the style boundary (`999999.5` rounds to 6 sig figs as
  `1.00000e+06`, printed as `1e+06`, not the decimal `1000000`), and negative zero
  (`-0.0` -> `"-0"`, not `"0"`). Verified live against Octave 11.3.0
  (aarch64-apple-darwin) across integers,
  6-sig-fig rounding, the `1e-4`/`1e-5` and `1e5`/`1e6` exponent thresholds, and
  negative numbers/zero, BEFORE committing to the plain `:g` format spec (no custom
  formatter needed). *Pinned by:* the eight `test_weighted_listing_g_format_*` boundary
  tests plus `test_write_weighted_listing_matches_golden`
  (`tests/test_phase4d_listing_writers.py`), byte-exact against
  `tests/reference_data/phase4d/listing/weighted.lst`.

- **[phase4d] Hard-example mini-batch CADENCE + the batch-listing deviations**
  (`src/python/speech/drivers/train.py` `_BatchRunner`/`_BatchStep`/`_files_values`;
  `src/python/speech/engine.py` `forward_backward`). Task 10 wires mini-batching live. Three
  things where the SOURCE overrides the plan prose or where the port simplifies, all
  deliberate:
  1. **Cadence correction (source wins over the brief's "epoch start").** `CreateBatches` is
     called EXACTLY ONCE per training run, before the optimizer (`Train_BLSTM.m:71`), NOT at
     epoch start -- there is no epoch loop around it; the `Batches` struct persists and only
     its cursors mutate. `GetNewBatch` + `WriteWeightedListing` live inside `ComputeGradient.m`
     (:53/:77) -- the ONLY live caller (Train_BLSTM.m's own GetNewBatch at :76 is a
     commented-out validation probe) -- so one fresh batch + fresh engine is drawn PER
     GRADIENT EVAL = per inner SMORMS3 step. The port matches: `create_batches` once in
     `train` (batch RNG = `seed + 2`), `next_listing`/`make_engine` per inner step.
     `WriteWeightedListing` at Train_BLSTM.m:932 (the QPSO-time write) sits in a dead `if 0`
     block for the committed algo-6 config and is NOT the live path.
  2. **WriteWeightedListing field-6 = duration, but Corpus reads field 6 as file_id.**
     `WriteWeightedListing.m` writes `[weight;duration]` as CSV fields 5/6, but
     `Corpus::from_config` (`corpus.rs:248-252`) reads field 6 as `fileId` via
     `istringstream >> int` -- so the legacy engine reads DURATION AS FILE_ID (int-truncated;
     the trailing ';' token is popped, tokens.size()==6, the read fires whenever field 5 is
     non-empty). A genuine legacy field-role mismatch, but NOT load-bearing for batching: the
     legacy fileId's ONLY consumer is `getRefFileId()` -> the LID TRAINING-TARGET class index
     (`TwinBLSTMSpectralLID.cpp:1067`, clamped `>= outputSize -> 0`); hard-example tracking
     (`getCases.m`/`GetNewBatch`) is file-INDEX-based and never touches fileId. The PORT's
     engine reads field 6 and DROPS it (`CorpusItem.file_id` is copied to no `Audio` field;
     the port's LID target sources `lang_index` -- a deliberate Phase-4b divergence from
     `getRefFileId`). Field 6 is therefore INERT in the port, and `_BatchRunner.next_listing`
     writes the record's file_id there simply as a deterministic placeholder that keeps the
     byte format identical -- writing duration would be numerically indistinguishable.
     *Fix candidate:* if the LID-target-from-fileId legacy path is ever ported for strict
     parity, BOTH the duration-as-file_id read and the `feedForward(targetIndex)` consumption
     (`BLSTMNeuralNetwork.cpp:843-918`) must land together; the listing field alone is inert.
  3. **No per-eval class-balance rescale -- FIXED (phase 5, commit `2590e48`, F3); ascending-index
     order stays a kept simplification.** `ComputeGradient.m:72-96` rescales
     `filesValues(:,2)` by `nbOfElem/sumInClassIndex` (in-class = `classNb == 1`) before
     writing; the port used to write the RAW listing weight (`_files_values` col1) verbatim --
     the `langMapConf` in-class model the port's corpus did not carry. SEPARATELY, and NOT
     part of this fix: the written index ORDER is ascending file index (`np.unique`), a
     simplification of `count_unique` + `sortrows(filesValues, -3)` (:55-58) -- the aggregate
     corpus cost is order-independent (`compute_cost` sums the config's rows;
     `aggregate_workers` re-sorts by id), so this stays a documented, deterministic
     simplification (the determinism gate is what matters for the optimizer path, S1),
     unchanged by F3. *Was pinned by:* `tests/test_phase4d_batchmode.py` (byte-exact batch
     listings pinning the RAW weight) + the batch-mode determinism exit gate
     `tests/pyo3/test_exit_gate.py::test_batch_mode_deterministic`.

     **FIX (phase 5, F3):** read `ComputeGradient.m:53-96` directly (not just the plan's
     one-line gloss). The rescale is gated by `:74-76`:
     `if ((PS.VP.algo < 5)||(PS.Corpora.Train.Batches.nbOfTargetClasses > 2))
     PS.Corpora.Train.filesValues(:,2) = 1; end` -- a HARD OVERRIDE (not a fallback) that
     discards BOTH the raw weight AND the rescale, for every non-LID algo or any >2-class LID
     split; only `algo >= 5` (LID) with `nbOfTargetClasses <= 2` (a binary target/non-target
     split) lets the rescale itself reach `WriteWeightedListing`. The rescale law (`:59-73`,
     `tmp` = the batch selection `count_unique([new_batch;worstCases])`, matching the port's
     existing `idx`): for every KEY in the mapping file (`keys(PS.Corpora.Train.langMapConf)`
     ~= `keys(PS.Corpora.langMap)` for a well-formed listing, since `langMapConf`'s only extra
     keys are listing-only-unmapped ones that would themselves error the `langMap` lookup --
     the port reads the mapping file's own keys directly, per spec R2), `classNb =
     langMap(key)`; if `classNb == 1`: `sumIn += sum(batch rows with that classNb)`, `nbOfElem
     += count(...)`; else: `sumOut += sum(...)`. Then `weight[class==1] *= nbOfElem/sumIn`,
     `weight[class!=1] *= nbOfElem/sumOut`. Iterating per KEY (not per unique classid) is
     reproduced verbatim -- a mapping file where two keys share a classid would double-count
     that classid's contribution, exactly like the legacy (not reachable by any committed
     1-key-per-classid fixture, deliberately not "cleaned up"). Division uses plain IEEE-754
     double semantics (`numpy` scalars, `0/0 -> nan`, matching MATLAB) so a batch with zero
     representation on one side produces a nan/inf factor that is provably never applied (its
     row selection is then empty).

     Ported as `speech.drivers.train.class_balance_values(listing_records, mapping_path) ->
     NDArray` (the pure `:59-73` law only) + the `:74-76` gate inlined into
     `_BatchRunner.next_listing` (needs `algo`/`Batches.nb_target_classes`, outside
     `class_balance_values`'s 2-argument contract). `_BatchRunner` gained `mapping_path`/`algo`
     fields; `_files_values`'s col1 is no longer read by `next_listing` (both branches -- the
     gate's flat-1.0 and the rescale -- re-derive the weight from `listing`'s own records, the
     same source col1 was built from).

     TEMPORAL scope (T5 review addendum): the legacy `filesValues(:,2)` persists across
     optimizer calls within one run (`SMORMS3.m:306-321` threads `PS` through
     `varargin_stored`), but `ComputeGradient.m:279-280` resets the whole column to 1 at the
     end of EVERY call where the periodic re-evaluation block (`:170-278`) does not fire --
     which is every call under `Train_BLSTM.m`'s own hardcoded defaults
     (`adjustFileImportance = -1`, permanently off). The port's recompute-from-raw-CSV on
     every call therefore reproduces the canonical every-call-behaves-like-call-1 trace. The
     `Train_BLSTM_LIDSeg.m` variant (`adjustFileImportance = 1, RunFull = 50`), where the
     reset skips every 50th call and weights COMPOUND call-over-call, is NOT modeled -- a
     documented residual for a future multi-step-training pass, not an oversight. Perf note
     (durable record of the T5 report's concern): `class_balance_values` re-parses the
     mapping CSV from disk on every gradient eval (tiny files, per the 2-arg contract);
     cache it if a profile ever shows it.

     *RED:* confirmed two ways against the pre-fix code (git-stashed just the test file,
     keeping the fix, to reconstruct the old assertions). (1) Signature-level: the OLD
     `_runner()`/`_BatchRunner(...)` call sites (no `mapping_path`/`algo`) raise `TypeError:
     _BatchRunner.__init__() missing 2 required positional arguments` against the fixed
     dataclass -- `test_batch_listing_bytes_step0`, `_rotates_across_steps`,
     `_includes_worst_cases`, `test_files_values_degenerate_gate_full_corpus` all failed this
     way. (2) Value-level (the more direct RED, isolating the fix from the signature change):
     re-ran the OLD single-class 5-file corpus (`_records`/`_files_values`) with the NEW
     required args supplied (`algo=6`, so the rescale -- not the gate -- is what runs) and
     compared against the OLD golden byte string
     `b"f0;r0;eng;us;1;1;\nf1;r1;eng;us;0.5;1;\n"`: got
     `b"f0;r0;eng;us;1.33333;1;\nf1;r1;eng;us;0.666667;1;\n"` instead (every file here is the
     corpus's only `eng_us` key, always in-class, so `factor_in = nbOfElem/sumIn = 2/1.5 =
     1.33333`) -- a genuine mismatch, not just a broken constructor.

     *Re-pin:* the two rotation goldens (`test_batch_listing_bytes_step0`,
     `_rotates_across_steps`) now use `algo=3` (a SAD algo, `_runner`'s new default) so the
     `:74-76` gate always wins and their weight field collapses to a flat `1` for every row
     (simple re-derivation: "the gate forces 1.0", no rescale arithmetic needed for tests whose
     real purpose is rotation-cursor coverage, not weight-value coverage). A crafted 2-class
     corpus (f0/f1 -> classid 1, weights 1.0/3.0; f2 -> classid 2, weight 2.0; f3 -> classid 3,
     weight 6.0; mapping file `eng;us;1` / `fra;fr;2` / `deu;de;3`) drives THREE new tests: a
     direct `class_balance_values` pin (hand-derivation in the docstring: `sumIn=4.0,
     nbOfElem=2, sumOut=8.0` -> `factor_in=0.5, factor_out=0.25` -> `[0.5, 1.5, 0.5, 1.5]`), the
     same corpus through `_BatchRunner.next_listing` end to end (algo=6, nb_classes=1, gate
     open -- byte-exact rescaled listing), and the SAME corpus with `nb_classes=3` (gate's
     other clause, `nb_target_classes > 2`) proving the rescale is computable but never
     written (flat `1` again). *Pinned by:* `tests/test_phase4d_batchmode.py::
     test_batch_listing_bytes_step0`, `::test_batch_listing_bytes_rotates_across_steps`
     (re-pinned), `::test_class_balance_values_matches_hand_derivation`, `::
     test_batch_listing_bytes_class_balance_rescale`, `::
     test_batch_listing_gate_forces_flat_weight_when_nb_target_classes_exceeds_two` (new).

     *Mutation (revert-the-fix, 3 variants, each applied then reverted):* (a) dropping the
     `nb_target_classes > 2` gate clause (`self.algo < 5` alone) breaks exactly
     `test_batch_listing_gate_forces_flat_weight_when_nb_target_classes_exceeds_two`; (b)
     deleting the whole gate (`class_balance_values` always wins) breaks exactly the THREE
     gate-dependent tests (`test_batch_listing_bytes_step0`, `_rotates_across_steps`,
     `_gate_forces_flat_weight_...`) while the two rescale-specific tests -- correctly ungated
     at `nb_classes<=2` -- stay green; (c) dropping the rescale multiply inside
     `class_balance_values` (`out = raw_weight.copy(); return out`) breaks exactly
     `test_class_balance_values_matches_hand_derivation` and
     `test_batch_listing_bytes_class_balance_rescale`, leaving the gate-dependent tests green
     (the gate short-circuits before `class_balance_values` is even called for those). Full
     `tests/test_phase4d_batchmode.py` (10 tests) reruns green after each revert. NOT
     independently mutation-tested: the per-KEY-not-per-unique-classid double-count quirk (no
     committed fixture has two keys sharing a classid).

- **[4d] `.scr` writer class order: the Phase 4c class-id divergence CLOSED -- legacy is
  `keys(langMapConf)` = ASCII-alphabetical `lang_dial` keys, and the mapping file's
  class-id column is IGNORED by the writer** (`src/python/speech/drivers/test.py::_class_keys`;
  legacy `Test_BLSTM.m:249/:263`, `processListing.m:10/:85-88`). The Phase 4c port ordered
  class keys by the `lang;dial;classid` mapping's id column and composed them WITHOUT the
  underscore (`parts[0] + parts[1]`) -- flagged as a documented divergence in the Task-12-4c
  report (item 4, "Evaluate class-key ordering divergence noted") since no `.scr` oracle
  existed then. Task 11 built the oracle (Octave `tools/octave_harness/stage_scr.m`, HYBRID
  tier: the REAL Tier-1 `processListing.m` supplies `keys(langMapConf)`; the writer loop
  Test_BLSTM.m:251-269 is FALLBACK-TIER transcription with `% legacy:` provenance --
  Test_BLSTM.m is a top-level script whose `scores_test` comes from `CostFunction.m`'s
  engine shell-out, so there is no injection point that leaves the vendored source
  unmodified, the same adjudication as `stage_computecost.m`) and the golden settled it:
  `processListing.m:85-88` OVERWRITES `langMapConf`'s values with alphabetical positions,
  so the writer labels score column ii with the ii-th alphabetical composed
  `[lang '_' dial]` key regardless of the file's ids (MATLAB/Octave `keys()` returns ASCII
  byte order -- verified live, 'Aaa_01' < 'aaa_11'; Python's code-point `sorted()` agrees
  on ASCII keys). The underscore composition is itself load-bearing: `write_scores`' dial
  slice (`tmp(end-2:end)`, Test_BLSTM.m:265) sees the underscore for a 2-char dial, so
  'aaa_11' renders as 'aaa-_11', NOT 'aaa-11' -- a genuine legacy composition artifact
  reproduced, not fixed. *Pinned by:* `tests/test_phase4d_scr.py` (byte-exact vs
  `tests/reference_data/phase4d/scr/expected.scr` on the oracle libm, canary-gated off it;
  `test_class_id_order_fails_golden` is the mutation check -- the OLD id order must FAIL
  the golden, proven non-vacuous by a mapping whose ids are deliberately not in
  alphabetical order).

- **[4d] `num2str(val,'%15.15f')` == Python `%.15f` byte-for-byte, and the `.scr` writer
  receives ALREADY-DECODED LID scores (the `>150` sentinel is applied upstream, not in
  the writer)** (`src/python/speech/drivers/test.py::write_scores`; legacy
  `Test_BLSTM.m:261-266`, `ComputeCost.m:708`, `CostFunction.m:409`). Two facts the Octave
  golden settled, no port-side change needed: (1) `num2str` applies the format via sprintf
  then trims spaces; `%15.15f`'s width 15 is strictly less than the minimum rendered length
  (every softmax output is `0.` + 15 decimals = 17+ chars, and the values are non-negative
  by construction), so the width padding NEVER fires and the trim is a no-op -- the 4c
  `%.15f` choice was already byte-correct across all probed magnitudes (a 250/250 tie pair,
  a near-zero 1.58e-6, exp(0), and a many-decimal softmax quotient). (2) The in-band
  `targetLID` sentinel (`>150 -> value-200`) is decoded at `ComputeCost.m:708` BEFORE
  `CostFunction.m:409` assigns `PS.VP.BP.LIDscoreDet`, so `scores_test` reaches the writer
  already decoded and the writer block contains no sentinel handling of its own -- the
  stage injects a raw 250.0 (> 150) and the golden shows `exp(2.5)`, not `exp(0.5)`,
  making the pass-through observable. The port mirrors this split: `_decode_lid_scores`
  decodes, `write_scores` does not. Also pinned: MATLAB `sortrows(x',-1)` descending is
  STABLE (the 250/250 tie preserves original column order; numpy `argsort(-s,
  kind="stable")` agrees). *Pinned by:* `tests/test_phase4d_scr.py::
  test_write_scores_matches_octave_golden_bytes` + `test_tie_break_is_stable_original_column_order`
  vs `tests/reference_data/phase4d/scr/expected.scr`.

- **[4d] Task 12: SIX specific vec2struct mask-VECTOR arms (per-block LSTM weight masks,
  the output-layer neuron mask, NormalizeInputMean/Std) + `_fmt_scalar` boundary
  formatting -- CLOSED for the arms actually exercised, no port fix needed (4c T8
  accepted debt).** The 4c T8 final review flagged two
  transcribed-but-uncovered items: (a) the per-block LSTM weight masks, output-layer neuron
  mask, and NormalizeInputMean/Std masks (`src/python/speech/genome.py::_nn_block`/
  `_output_neuron`/`_lstm_and_output`) were ported from `vec2struct.m` but only exercised
  through `Force_Symetry`/`Force_Identical_Rows` (which tie blocks to EACH OTHER, never hit
  a raw `isfield(maskStruct, fieldName)` vector branch); (b) `_fmt_scalar`'s `%d`/`%15.15e`
  boundary formatting had no explicit unit table, only incidental coverage via whole-config
  string comparisons. Both closed by a new `vecmask` vec2struct case (`tools/octave_harness/
  stage_vec2struct.m`) that masks six fields in isolation -- `Forward_Layer_0_LSTMBlock_0_
  InputGateWeights` (13-row gate layout), `Forward_Layer_0_LSTMBlock_1_CellWeight` (9-row
  narrow peephole-free layout, the CellWeight parity hazard), `Backward_Layer_0_LSTMBlock_0_
  OutputGateWeights` (Backward direction), `Output_Layer_1_Neuron_0_Weights` (ii=2, so the
  `Force_Identical_Rows` ii==1 repmat branch never shadows it), and `NormalizeInputMean`/
  `NormalizeInputStd` (Std's mask is deliberately signed to hit the abs() asymmetry:
  `vec2struct.m:933-938` stores `abs(mask)` in both the config field and the out_param
  write-back, unlike Mean which passes the (equally signed) mask through unmodified aside
  from the `(field+1)/2` encode) -- plus a direct real-`printConfig.m` probe
  (`fmt_scalar_boundaries.config`) over 18 boundary values (integers incl. -0 and a
  1234567890123-magnitude exact integer that must not flip to e-notation, negatives, the
  1e-5/1e+5 magnitude boundaries on both sides, a `round(v)==v` near-integer decode edge
  `2.9999999999999996`, and 15-significant-digit fractions). Every arm matched the real
  `vec2struct.m`/`printConfig.m` output bit-exact/string-exact on the first run (verified by
  independently hand-deriving each arm's expected out_param slice from the walk order --
  see `task-12-report.md` -- before wiring the golden, not by trusting the port's own
  output): no divergence, no port change. *Pinned by:* `tests/test_phase4c_genome.py::
  test_vecmask_arm_writes_expected_slice` (6 arms) + `test_vecmask_arms_are_non_vacuous_vs_
  unmasked` + `test_normalize_std_mask_applies_abs_asymmetry` + `test_fmt_scalar_boundary_
  matches_octave` (18 boundary values) + the generic `CASES`-parametrized count/config/
  out_param tests now covering `vecmask` too. *Mutation:* dropping the Std mask's `abs()`
  (`field = self._mget_v(...)` instead of `np.abs(self._mget_v(...))`) fails both
  `test_out_param_inverse_matches[vecmask]` and `test_vecmask_arm_writes_expected_slice
  [NormalizeInputStd]` -- confirming the golden is non-vacuous, not merely present.
  **Scope note (not closed by this entry):** two other vector-mask sites in `genome.py`
  stay UNEXERCISED -- `_padding_block` (`:317-327`, backing `AlgName_speech_padding`/
  `AlgName_min_silence`/`AlgName_min_speech`) and the `AlgName_TDC_lags` mask (`:552-555`,
  `np.minimum(self._mget_v(...), thresh)`). Phase 4d is the last roadmap phase, so there
  is no future phase to hand this to; it is recorded here as a permanent, honest gap
  rather than folded into the "CLOSED" claim above.

### Mutation battery (Phase 4d)

- **[phase4d] Mutation battery (Task 14): 7 of 8 mutations break their named golden as
  designed; item 7 is a targeted comparator demonstration (not an apply/revert
  production mutation), and it confirms the structural gate is load-bearing -- no gap.**
  Each production mutation applied/run (named suite only, FOREGROUND)/reverted
  (`git checkout --`)/re-run in isolation; tree confirmed clean (`git status --porcelain`)
  between every step. (1) OpenSAD15 collar `2.0`->`1.0` (`src/python/speech/dataprep/
  opensad15.py::convert_tab_file`, SIX `SegsExcl` guard-band sites, not five as first
  reported (final-review correction): the two `:149`/`:159`-style `beg_f - 2.0`
  leading-collar sites, the THREE `:151`/`:157`/`:161`-style `end_f + 2.0`
  trailing-collar sites, and the `:154` re-open threshold `beg_f - 2.0 - 0.1`. The `:157`
  occurrence (the close-then-reopen `if` arm's trailing collar) is OUTPUT-INERT on this
  fixture's corpus -- always overwritten before read, so mutating it alone produces
  byte-identical output either way; the mutation as applied changed the literal
  site-wide, so the other five sites still carried the FAILED verdict below)
  against `tests/test_phase4d_opensad15.py` -- FAILED as expected (6 of 27: all 5
  `test_convert_tab_file_matches_hand_computed_fixture` cases plus
  `test_process_opensad15_writes_xml_stm_and_listing`, each an STM excluded-region
  boundary-time mismatch, e.g. case1 `1.4118` (mutated) vs `2.4118` (golden)); reverted,
  PASS. (2) S/RI filter widened: added an `or (len(label) == 2 and label.startswith("S"))`
  arm to `convert_tab_file`'s `:113` filter (accepting 2-char `S`-prefixed labels like
  `"SP"`, previously excluded) against the same suite -- FAILED as expected, exactly the
  named catcher `test_case4_filter_excludes_non_s_non_ri_labels` (`assert 3 == 2`, the
  excluded `"SP"` row now contributes a segment) plus the case4 fixture golden; reverted,
  PASS. (3) `audio_path[:-5]` -> `audio_path[:-4]` (`convert_tab_file:117`, the XML
  `AudioDoc` stem) against the same suite -- FAILED as expected (7 of 27: the named
  `test_case5_extension_length_quirk_truncates_stem` plus all 5 fixture cases and the
  process-driver test, since every stem now truncates one character short); reverted,
  PASS. (4) Light-normalizer filler regex dropped: removed the `(euh|hm+|mm+|eh|huhum|
  hum)` cascade entry (`src/python/speech/dataprep/stm_normalize.py::_LIGHT_SUBS`, the
  `:29`-sourced tuple) against `tests/test_phase4d_stm.py` -- FAILED as expected, the
  named `test_light_filler_class1_adjacent_tokens_alternate` plus the byte-oracle
  `test_normalize_stm_light_matches_perl_oracle[case2]` (the fixture's `hm mm eh huhum
  hum` line, now left unfiltered); reverted, PASS. (5) `write_listing` fliplr stride
  off-by-one: `_worker_positions`'s `range(a, 0, -nb_workers)` -> `range(a, 0,
  -(nb_workers - 1))` (`src/python/speech/batching.py:159`) against
  `tests/test_phase4d_listing_writers.py` -- FAILED as expected, the named
  `test_worker_positions_hand_derived` (`_worker_positions(7,1,3)` now `[0,2,4,6]` vs the
  hand-derived `[0,3,6]`) plus the golden shard bytes and the partition non-vacuity check
  (now double-covers file 0 and skips file 4); reverted, PASS. (6) Augment noisetype
  ladder boundary moved: `elif noisetype < 2` -> `elif noisetype < 2.5`
  (`src/python/speech/dataprep/augment.py::draw_variant:104`, the pink/tpdf rung) against
  `tests/test_phase4d_augment.py` -- FAILED as expected, exactly the named
  `test_draw_variant_noisetype_ladder[4-tpdfnoise]` (seed 4's draw `2.0453...` now
  resolves to `"pinknoise"` instead of `"tpdfnoise"`); reverted, PASS. (8) Batch-mode
  rotation wiring bypassed: `_BatchRunner.next_listing`'s `batch, _ =
  get_new_batch(self.batches)` -> `batch = list(self.last_index) if
  self.last_index.size else get_new_batch(self.batches)[0]` (`src/python/speech/drivers/
  train.py:170`, reusing the prior step's index instead of rotating) against
  `tests/test_phase4d_batchmode.py` -- FAILED as expected, exactly the named
  `test_batch_listing_bytes_rotates_across_steps` (step 1's index stayed `[0,1]` instead
  of advancing to `[2,3]`); reverted, PASS.
  *Item 7* (parity structural assert weakened) is not a production mutation by
  construction -- the brief's own alternative framing (craft a flipped fixture COPY,
  point a temporary test-local comparison at it) was used since the assert in question
  lives in a *test* (`tests/test_phase4d_parity.py::test_vrcts_structural_and_boundaries`),
  not production code: a COPY of `tests/reference_data/phase4d/parity_tupleA_vrcts_
  chan1.xml` had its trailing `<SpeechSegment stime="56.9852" etime="59.9999">` dropped
  (a segment-type flip: Speech -> excluded/Other), written to the session scratchpad, the
  COMMITTED fixture untouched. Two throwaway test functions were appended to
  `tests/test_phase4d_parity.py` (never committed, reverted via `git checkout --` after
  the run), both driving the REAL `speech_rs.Engine` over tupleA to get the real 8-segment
  port output, then diffed against the 7-segment flipped copy: (a) the CURRENT code path
  (`:168`'s explicit `assert len(port) == len(oracle)`) raised `AssertionError: ...
  STRUCTURAL count mismatch port=8 oracle=7` as expected -- the gate FAILS on the flipped
  copy, confirmed via `pytest.raises`; (b) a WEAKENED comparison (the explicit count
  assert dropped, `zip(port, oracle, strict=True)` relaxed to non-strict `zip`) run
  against the identical port/flipped-copy inputs raised NOTHING and the test PASSED --
  non-strict zip silently truncates to the shorter (7-element) side, so the port's extra
  trailing segment (the dropped-in-legacy regression the flip simulates) goes completely
  unnoticed. Non-obvious finding: `strict=True` on the per-boundary `zip` (`:170`) is
  ALSO independently load-bearing for a count mismatch (it raises `ValueError` on its
  own, redundant with the explicit `:168` assert for a raw length difference) -- but
  ONLY the explicit assert's non-strict-zip removal in the weakened variant above was
  needed to demonstrate the pass-through, since dropping just one of the two guards
  already suffices; the two guards are not fully redundant in general (a REORDERING
  mutation with an unchanged count would slip past the `:168` count assert but still be
  caught by strict `zip`'s per-element comparison against `pinned_dt`). Both `assert
  len(port) == len(oracle)` (:168) and `zip(..., strict=True)` (:170) are therefore
  independently load-bearing, not decorative; no port-side fix needed, this is a
  demonstration, not a discovered bug. *Net verdict:* 7 of 8 fresh production mutations
  break their named golden exactly as designed (item 8 uses catcher-suite numbering `8`
  per the brief's own list, so this entry skips numeral `7` deliberately, not by
  omission); item 7's comparator-swap demonstration confirms the structural gate is
  genuinely load-bearing, closing the brief's requested check with no coverage gap
  found. Full pytest (`uv run pytest tests`) + `cd src/rust && cargo test` + `./lint_code.sh`
  ran once at the end, all green (see the Task 14 report,
  `.superpowers/sdd/task-14-report.md`, for exact commands/logs).

### Mutation battery (Phase 5)

- **[phase5] Mutation battery (Task 11): 11/11 battery items (11 apply-fail-revert-pass
  production-code cycles -- item 7 counts twice, Rust + Python -- plus item 5's one
  test-side demonstration, 12 exercises total) break/behave exactly as predicted; zero
  coverage gaps found.** [T12 phrasing fix: the prior wording's "12 production-code
  cycles ... plus one test-side demonstration" mislabeled item 5 as a production cycle
  while simultaneously counting it again as an addition -- 9 single-mutation items
  (1,2,3,4,6,8,9,10,11) + item 7's 2 (Rust+Python) = 11 production cycles; item 5 is the
  ONE non-production (test-side) exercise, for 12 total, not 13.] Every fix
  landed across Tasks 4-9 (F1/F3/F5/F7/F8/F10/F11) is RE-VERIFIED here from the committed,
  fully-integrated Phase 5 state (not merely re-trusting the per-fix land-time note), plus
  the four loop-foundation pieces (seeded init, the modern training loop's early-stop
  cadence, the narrowed QPSO genome mask, the SAD convergence margin). Each production
  mutation applied/run (named suite only, FOREGROUND)/reverted (`git checkout --`)/re-run
  in isolation; `git status --porcelain` confirmed clean between every step. Baselined
  first: every named catcher (32 pytest cases across 8 files, 5 targeted `cargo test`
  invocations, 2 pyo3 seam tests) confirmed GREEN pre-mutation.

  (1) init forget-bias flag dropped: removed the `if forget_bias_one:` write
  (`src/python/speech/init_weights.py:94-95`, `_init_lstm_gates`) against
  `tests/test_phase5_init_weights.py::test_forget_bias_one_positions` -- FAILED as
  expected on both tuple-A/B parametrizations (`assert np.all(layer["forget"][:, -1] ==
  1.0)` -> `np.False_`, the forget-gate bias column stayed all-zero); reverted, PASS.

  (2) The modern loop's early-stop cadence bypassed: `if best_epoch >= 0 and epoch -
  best_epoch >= params.patience:` -> `if False and ...`
  (`src/python/speech/drivers/train.py:830`, `train_modern`) against BOTH
  `tests/test_phase5_train_modern.py::test_early_stop_patience_triggers_at_crafted_epoch`
  (the T8 stub state-machine unit) AND `tests/pyo3/test_exit_gate.py::
  test_early_stop_triggers` (the REAL engine-backed gate) -- BOTH FAILED as expected
  (`assert res.stopped_early is True` -> `False`, the loop ran the full epoch budget on
  both the crafted-plateau stub and the real stuck-at-30.0 engine fixture); reverted,
  both PASS. Stronger than the brief's "and/or" hedge: both named catchers exist (the
  pyo3 one lives in `test_exit_gate.py`, not `test_phase5_train_modern_smoke.py`, which
  its name might suggest) and both fire.

  (3) The per-eval class-balance rescale skipped: `_BatchRunner.next_listing`'s
  `algo>=5`-and-`nb_target_classes<=2` branch collapsed to an unconditional
  `weight_col = np.ones(idx.size)` (`src/python/speech/drivers/train.py:297-300`) against
  `tests/test_phase4d_batchmode.py` -- FAILED as expected, exactly the named
  `test_batch_listing_bytes_class_balance_rescale` (1 of 10; byte mismatch at index 13,
  `1` written where the rescaled `0.5`/`1.5` was expected); the other 9 (incl. the sibling
  gate test asserting the FLAT-1.0 branch, untouched by this mutation) stayed green;
  reverted, PASS.

  (4) The genome mask widened to include a weight dim: `searchable[start:stop] = False`
  -> `searchable[start:stop - 1] = False` (`src/python/speech/genome.py:1032-1033`,
  `weight_block_mask`, leaving the last dim of every masked range wrongly searchable)
  against `tests/test_phase5_genome_narrowing.py` -- FAILED as expected, 6 of 24
  (`test_masked_dims_match_derived_net_structure` + `test_every_masked_dim_is_a_weight_
  normalize_key`, all 3 net-shape params -- twin/spectral/signal -- each: "traced
  weight/normalize ranges != mask complement" / masked-dim count below the independently
  derived net-structure expectation); the other 18 (incl. the `mask` DICT-keyed tests,
  untouched since the mutation only touches the `searchable` bool array) stayed green;
  reverted, all 24 PASS.

  (5) *Not a production mutation* (the item-7-4d pattern): the SAD convergence margin's
  bite, demonstrated test-side. A throwaway test appended to `tests/pyo3/test_exit_gate.py`
  (never committed, reverted via `git checkout --` after the run) built a STAGNANT
  "trainer" -- two `forward_backward` reads at the SAME init weights, zero SMORMS3 steps
  -- measuring EXACTLY `0.0` train-cost improvement. A zero-margin gate
  (`improvement >= 0.0`) PASSED trivially on it (proving a zero margin certifies nothing);
  the REAL `_SAD_TRAIN_MARGIN = 0.026` gate FAILED on the identical stagnant run
  (`pytest.raises(AssertionError)` fired as expected) -- confirming the margin is
  load-bearing, not cosmetic. Test removed after the run; `git status --porcelain` clean.

  (6) F1 revert (the clobber restored): `create_batches`'s non-multilingual branch
  reverted from class-VALUE indexing (`slot = int(v) - 1`) back to loop-POSITION indexing
  (`for ii, v in enumerate(possible): slot = ii`,
  `src/python/speech/batching.py:359-365`) against `tests/test_phase4c_batching.py` --
  FAILED as expected, exactly the named `test_create_batches_class_value_indexing`
  (1 of 18; `cases[0].index` went from `[3, 2]` to `[]`, matching -- bit for bit -- the
  "old mutation record" prediction written when F1 originally landed); reverted, all 18
  PASS.

  (7) F5 revert (sticky decls moved back outside the row loop), BOTH languages, one at a
  time. **Rust:** `pos_target`/`pos_best_not_target` moved out of the `for jj in
  0..results_lid.nrows()` loop (kept `score_target`/`max_score_not_target` reset per row,
  matching the legacy's own `:533-534`) in `src/rust/src/engine/confusion.rs:67-71`
  (`confusion_from_results`) -- `cargo test --lib engine::confusion::tests::` FAILED 2 of
  9 (`sentinel_decode_and_argmax`, `cross_language_identity_no_target_skip`;
  `no_target_row_skipped_never_pollutes_header` correctly stayed green -- the degenerate
  first-row case where sticky and per-row decode coincide, as documented at F5 land time)
  and `cargo test --test phase4b_confusion_golden confusion_matrix_matches_harness_
  transcription` FAILED ("mismatch at (1, 2): a=2 b=1"); reverted, all green. **Python:**
  the identical decl move in `src/python/speech/scoring.py:71-75` (`confusion_matrix`) --
  `tests/test_phase4c_scoring.py` FAILED 2 of 15, exactly the named
  `test_confusion_matrix_no_target_row_skipped` (`cm[1,2]` `2.0` vs expected `1.0`) and
  `test_confusion_cross_language_identity_no_target_skip` (matrix mismatch, the sticky row
  double-charging the competitor cell); reverted, all 15 PASS.

  (8) F7 revert (LogLaw deriv unconditional A/y): `Law::deriv`'s `Log` arm's clamp-
  consistency guard removed, back to unconditional `a / y`
  (`src/rust/src/cost.rs:99-100`) against `cargo test --test phase0b_costlaw
  log_deriv_consistent_with_clamped_forward` -- FAILED as expected ("assertion `left ==
  right` failed: left: -0.49999999999999994, right: 0.0", the lower-clamp saturated-region
  probe point); reverted, PASS.

  (9) F8 revert (the overflow guard removed): the per-row `row_max`/`EXP_OVERFLOW_GUARD`
  shift deleted, `NeuronLayer::feed_forward`'s softmax back to unconditional
  `pre_act[[t, j]].exp()` (`src/rust/src/nn/layers.rs:952-969`) against `cargo test --test
  phase2_layers_golden dense_softmax_overflow_guarded` -- FAILED as expected (panicked
  "row 0 must be finite, not NaN", the >700-logit row overflowed again); reverted, PASS.

  (10) F10 revert (`weights_derivatives` reads the bare bag): the `seam_derivs` stash
  check dropped, back to `self.processors.get_weights_derivatives(pos)` unconditionally
  (`src/rust/src/engine/corpus_processor.rs:913-918`). Rust: `cargo test --test
  phase4c_api weights_derivatives_nonzero_through_seam` FAILED as expected ("F10: the
  release seam must return the folded (nonzero) gradient; got all-zero col0"). PyO3 (the
  ONLY mutation in this battery needing an extension rebuild -- `uv run maturin develop
  --release --manifest-path src/rust/speech-py/Cargo.toml`, ~19s): both
  `tests/pyo3/test_seam_replay.py::test_forward_backward_returns_nonzero_gradient`
  ("got 0/33671 nonzero") and `::test_forward_backward_smorms3_moves_weights`
  ("||trained-init||=0.0") FAILED as expected -- the exact T10-diagnosed no-op,
  reproduced from the fully-integrated state; reverted + rebuilt, all three PASS.

  (11) F11 revert (`Epochs=1` in `_modern_config_text`, scoped to that ONE builder per the
  brief -- `_eval_config_text` deliberately left untouched):
  `cfg["Neural_Networks_BackPropagation_Epochs"] = "0"` -> `"1"`
  (`src/python/speech/drivers/train.py:647`) against `tests/test_phase4d_algo3_drivers.py`
  -- FAILED as expected, exactly the named `test_config_texts_single_eval_are_epochs_
  zero_f11` (1 of 10; `assert '1' == '0'`); the two `_eval_config_text`-only tests
  (`test_eval_config_text_algo3_no_lid_keys`, `test_eval_config_text_algo6_injects_lid`)
  correctly stayed green, confirming the mutation's scope landed on exactly the one
  builder intended; reverted, all 10 PASS. *Exit-gate impact (described, not re-run, per
  the brief):* the original F11 land-time note already recorded that reverting either
  builder to `Epochs 1` "re-introduces the moved-weights cost anchor" while a
  gradient-nonzero check alone stays green (the fold still runs, just at the wrong theta)
  -- this battery did NOT re-run the slow `test_from_scratch_sad_converges` gate to
  re-confirm that trace shift; relying on the land-time evidence rather than fresh
  evidence is the one place in this battery where the verification is secondhand.

  *Verdict:* every one of the 11 battery items produced its predicted RED, and every
  revert produced a clean, fully green re-run -- no coverage gaps, no catcher surprises
  beyond the one explicitly anticipated by the brief (item 2's "and/or", which turned out
  to be "and": both named catchers exist and both fire). Full `uv run pytest tests`
  (505 passed) + `cd src/rust && cargo test` ran once at the end, both green; working
  tree at completion contains ONLY this IMPROVEMENTS.md entry.

## Complete-as-portable closures (Phase 4d)

Task 13's declaration: these four items are PERMANENTLY blocked or deferred, not stub
placeholders waiting on a future task. Each entry names the typed bail(s) that make the
blocked behavior fail loudly (never silently), cites its pinning test, and states
concretely what would have to exist for the block to lift. Cross-referenced by the
README roadmap (T15). **Update (Phase 6 Task 1):** the fourth item (cep / `File_Type` 2)
was the one deferred "pending a corpus" -- that corpus arrived, so its block LIFTED and
it is now a real reader; the other three remain blocked. See its flipped entry below.

- **Cost balances 6-9 (the WER shell-out laws): source LOST, not deferred-portable.**
  `ComputeCost.m` computes balances 6/7/8/9 by shelling out to an external Python
  scorer at `/people/gelly/Scripts/xml2wer{,_multi,_list}.py` over `ssh` to named 2015-era
  lab workstations (`PS.VP.serversname{t.ID}`, hostnames like `UV00000123-P000`).
  *Evidence:* `legacy/Optimizer_V6.2.2/functions/ComputeCost.m:376-397` (balance==8
  pre-loop: multi-worker `ssh gelly@<server> "... python /people/gelly/Scripts/
  xml2wer_multi.py -i <xmlpart> -c <refctmdir> -l <fileList> -d <durmax> -p
  <BalanceBackProp> ..."`, plus the single-worker fallback at `:396`); `:399-427`
  (balance==-9 pre-loop, same shape over `xml2wer_list.py -l <listing>.flst -t
  <PruningThresh>`); `:469-490` (balance==6 branch, the literal shell at `:480`:
  `system(['python ~/Scripts/xml2wer.py -i xmlpart_%d -c ... -l ... -d ... -p ...'])`);
  `:491-516` (balance==7, shell at `:504`, same script, `foo.csv` output consumed at
  `:505-512`); `:517-534` (balance==8's PER-NETWORK branch: no shell of its own, it
  `load()`s `xmlpart_%d/results.csv` at `:525`, i.e. the OUTPUT the `:396` pre-loop shell
  produced); `:535-550` (balance==9: no `system()` call at all, but consumes WER-derived
  `Error_vad` columns 10-14 that only the same `xml2wer` family populates in this
  codebase -- grouped with 6-8 as "the WER shell-out laws" per this repo's own module
  map, not independently verified to have a different provenance). Confirmed lost, not
  just unvendored: `find / -iname "*xml2wer*"` and `find / -iname "*gelly*"` (repo tree
  and local filesystem, depth-bounded) both return zero hits -- neither the scripts nor
  a `/people/gelly` home directory exist anywhere on this machine, and `legacy/` is a
  source-only C++ + `.m` snapshot (no external tooling was ever vendored alongside it).
  *Typed bail:* `src/python/speech/engine.py:304`, `compute_cost`: `raise ValueError(
  f"compute_cost ports balances 0/3/4/5/10, got {balance} (6-9 are deferred to 4d)")`.
  *Pinning test:* `tests/test_phase4c_engine_cost.py::
  test_compute_cost_rejects_deferred_balances` (`:260-263`), parametrized over
  `balance in (6, 7, 8, 9)`, asserts `pytest.raises(ValueError, match="deferred to
  4d")` -- pre-existing, verified still passing.
  *What it would take:* recovering or rewriting `xml2wer.py`/`xml2wer_multi.py`/
  `xml2wer_list.py`. The CLI contract is partially reconstructable from the `system()`
  call sites above (`-i`/`-c`/`-l`/`-d`/`-p`/`-t` flags, a `foo.csv`/`results.csv`
  output with columns consumed at `ComputeCost.m:485,509,528-529`), but the actual
  WER-scoring algorithm (word alignment against a CTM/XML reference, pruning) is nowhere
  in this repo. It would also need a corpus with real word-level CTM references
  (`PS.VP.refctmdir`) -- every committed fixture carries only STM segment boundaries
  (SAD/LID labels), no word transcriptions. This is a from-scratch reimplementation
  against an unknown legacy format with no oracle to validate against; only attemptable
  if a corpus with matching WER references ever surfaces.

- **Twin pitch second pass: reference-based, structurally diverges from the (ported)
  base spectral pitch pass -- deferred, portable in principle.**
  `TwinBlstmSpectralLid`'s pitch second pass (`TDCwindow > 0`) is a DIFFERENT algorithm
  from `BlstmSpectralSegmenter`'s pitch pass (`tasks/sad.rs`, already ported and
  golden-tested), not a reusable variant of it. *Evidence:* legacy
  `TwinBLSTMSpectralLID.cpp:349-614` is the per-channel setup block, executed
  UNCONDITIONALLY for every mode including 7 (the `if (_Mode != 7)` guard at `:353` is
  commented out, dead). The pitch sub-block, `:363-403`: `seg._Classification.at(chan) =
  seg._Reference.at(chan)` (`:367`) then `smoothSegmentation` (`:368`) BEFORE `pitch =
  this->getPitch(...)` (`:369`) -- pitch is derived from the REFERENCE segmentation, not
  a forward-pass hypothesis -- then the periodogram is warped in place (`:372-401`,
  `coeff_homo = pitch/300`) BEFORE the single forward pass that follows at `:617+`.
  Contrast the ALREADY-PORTED base driver, `BLSTMSpectralSegmenter.cpp:740-804`: pass-1
  forward (`:740`) -> `results2segmentation` writes the HYPOTHESIS into `seg` (`:751`)
  -> `getPitch` over that hypothesis (`:758`) -> warp -> pass-2 forward (`:793`, gated
  `if (pitch > 0)`). The base is a genuine two-pass self-bootstrapping scheme that needs
  no ground truth (it works on unlabeled audio at inference time); the Twin's version
  requires `seg._Reference` populated before it can run at all -- a reference-dependent
  algorithm, not a code-reuse opportunity.
  *Typed bail:* three call sites in `src/rust/src/tasks/lid.rs`. (a) the shared mode
  0/1/2/3 path, `get_segmentation` `:2226-2234`: `if let Some(tdc) = s.tdc.as_ref() &&
  tdc.half_window > 0 { return Err(anyhow!("TwinBlstmSpectralLid: pitch second pass
  (TDCwindow > 0) not ported (legacy :349-614)")); }`. (b) the mode-7 path,
  `get_segmentation_mode7` `:1283-1289`, same shape: `"TwinBlstmSpectralLid mode 7:
  pitch pass (TDCwindow > 0) not ported"`. (c) the mode-4/5/6 path,
  `get_segmentation_mode456` `:1680-1686`, its OWN independent guard reached BEFORE the
  shared (a) branch ever runs (`get_segmentation`'s trait entry dispatches
  `self.mode in {4,5,6}` straight to `get_segmentation_mode456`, `:2212-2214`): `if let
  Some(tdc) = s.tdc.as_ref() && tdc.half_window > 0 { return Err(anyhow!(
  "TwinBlstmSpectralLid mode {}: pitch second pass (TDCwindow > 0) not ported",
  self.mode)); }`. Every committed twin config under
  `tests/reference_data/phase4b/twin_*.config` ships `BLSTM_TDCwindow 0` (gate off), so
  none of the three branches had ever been exercised by a golden.
  *Pinning test:* previously NONE at any site. Added in Task 13: `src/rust/tests/
  phase4b_twin_golden.rs::pitch_second_pass_bails_wav_modes` (site a),
  `src/rust/tests/phase4b_twin_mode7.rs::mode7_pitch_second_pass_bails` (site b), and
  `src/rust/tests/phase4b_twin_golden.rs::pitch_second_pass_bails_mode456` (site c,
  fix-wave) -- all three override
  `BLSTM_TDCwindow`/`_TDCshift`/`_TDC_lags`/`_TDC_balance`/`_TDC_windowing_type`/
  `_TDC_windowing_param` to the same values that activate the BASE driver's ported pitch
  pass in `phase2b_spectral_golden.rs::pitch_map` (`TDCwindow 0.032 -> TDC_window_size
  128 > 0`), then assert the `Err` fires with the expected message text. All three pass
  (`cargo test`, this task); site c's non-vacuity was mutation-checked (guarding on
  `if false && ...` makes `pitch_second_pass_bails_mode456` fail while
  `pitch_second_pass_bails_wav_modes` stays green, confirming the two tests exercise
  distinct call sites, not the same one twice).
  *What it would take:* transcribing `TwinBLSTMSpectralLID.cpp:349-614` as its own
  driver path. The low-level pieces are already ported and reusable (`get_pitch` +
  the periodogram-warp math are the identical routines `BlstmSpectralSegmenter`'s pitch
  pass already exercises), but the ORDERING is Twin-specific: seed-from-reference +
  smooth (`Segmentation::set_segments_from`, already landed for Twin mode 4) must run
  BEFORE the first and only forward pass, and the warp must land on
  `audio.periodogram`/`_FilterBankedPeriodogram`/`_CepstreCoefficients` before
  `getBLSTMInputSequence` is (re)built. Needs a new golden: a twin config with
  `TDCwindow > 0` plus a real reference segmentation wired through `refs`, and harness
  (`tools/oracle_harness` `TwinProbe`) support for the reimpl-swap under this path. No
  missing external data -- this closure is portable, just not yet scheduled.

- **Mode-7 WAV/CNN arm: broken-as-committed since Phase 2 -- deferred until the CNN
  itself is fixed (out of current scope).**
  `TwinBlstmSpectralLid`'s Mode-7 WAV arm runs `_LIDConvNeuralNetwork`, the SAME
  Convolutional net excluded since Phase 2 for being broken-as-committed (see the
  `[phase2] CNN is broken-as-committed` entry above, `~line 593`: empty `_Layers`
  indexed on every real-config LID run is UB, the weights are orphaned/unserializable,
  `feedBackward` returns nothing). *Evidence:* legacy `TwinBLSTMSpectralLID.cpp:903-1193`
  (the `abs(_Mode)==7` branch), specifically `:922-963`:
  `if (audio.hasReadWavFile()) { ... _LIDConvNeuralNetwork.feedForward(inputCNN);
  inputSeq = _LIDConvNeuralNetwork.getOutputMatrix(); ... }` (the CNN call itself at
  `:927-928`). The `[phase4b]` entries earlier in this file (`~2027-2060`, `~2195+`)
  already name this arm as deferred alongside the pitch pass.
  *Typed bail:* `src/rust/src/tasks/lid.rs`, `get_segmentation_mode7` `:1276-1280`: `if
  audio.periodogram.is_none() { return Err(anyhow!("TwinBlstmSpectralLid mode 7: wav arm
  runs the CNN (not ported); only File_Type 1 (phSeq) supported (legacy :922-963)")); }`
  -- `audio.periodogram.is_some()` is the port's proxy for `!hasReadWavFile()`, so
  `is_none()` <-> a real wav decode <-> the CNN arm.
  *Pinning test:* previously NONE -- every existing Mode-7 golden in `phase4b_twin_
  mode7.rs` uses the phSeq corpus (`corpus_phseq/s{1,2,3}.phSeq`), which never reaches
  this branch. Added in Task 13: `src/rust/tests/phase4b_twin_mode7.rs::
  mode7_wav_arm_bails_cnn_not_ported`, feeding a real wav decode (`corpus_lid/f1.wav`,
  `file_type=0` -> `periodogram: None`) into the mode-7 driver and asserting the `Err`
  mentions "CNN". Passes (`cargo test`, this task).
  *What it would take:* fixing the Phase-2-excluded CNN first (the `_Layers` ctor guard,
  the orphaned-weight serialization, the missing `feedBackward` return) -- out of scope
  per the locked Phase 2 spec decision. No real `lid.config` in this repo (including the
  vendored `configs/legacy/LID_BLSTM.config`) ever configures CNN keys; LID always runs
  via phSeq, so there is no evidence this arm was ever exercised even in production
  legacy runs. Fixing it means inventing correct CNN semantics with no working reference
  to validate against.

- **Cep ingestion (`File_Type` 2): PORTED (phase 6 Task 1, commit `this commit`) -- the
  LRE03/07 corpus arrived; `File_Type` 3/4 stay deferred (still no data).**
  *Legacy behavior (recorded, `AudioStruct.cpp:183-256`):* the cep binary is `int32
  nbRecords | int16 vectorSize | int16 magic` (little-endian), then an `int32
  vectorNb`-per-record table, then `float32` payload (`sizeof(float)`) row-major per
  record; `magic` is read but only LOGGED (never validated); records with
  `vectorSize*vectorNb <= 0` are SKIPPED (`:229`); `_FramesCount =
  numberOfFrames*0.01*_Framerate` with `numberOfFrames` seeded at 2 and accumulating
  `vectorNb+2` per kept record, and `_Periodogram` block-filled from `rowBegin = 1` with
  a 2-row gap between records; a truncated payload is SILENTLY zero-filled (the read loop
  `:234` stops on a failed read, leaving the `Eigen::Zero` remainder), excess trailing
  bytes are ignored, and a malformed header (`nbRecords <= 0`) `exit(1)`s.
  *Original deferral (Phase 4d Task 13, for the record):* no committed fixture, `legacy/`
  corpus, or `dataprep/` output carried `.cep` audio and no oracle existed, so this was
  one of the Phase 4d "four" -- deferred (NOT permanently blocked) "pending a corpus that
  actually exercises File_Type 2+".
  *What landed (Phase 6 Task 1):* that corpus is now `data/LRE03-LRE07/` (34k
  `.plp8f0mvsdd` LID feature files, gitignored + licensed, byte arithmetic exact on every
  surveyed file, `vectorSize == 23 == NNetInputSize`, `nbRecords` 1..=14).
  `src/rust/src/audio.rs::read_cep` reads the layout above into `external_features` (one
  `(vectorNb x vectorSize)` matrix per kept record -- NOT one row per record; a single
  LRE utterance is typically one record of a few thousand frames) plus the block-filled
  `periodogram`, mirroring the phSeq plumbing and reusing the pinned `phseq_frames_count`
  helper. The `read_audio` file_type==2 dispatch and the
  `BagOfProcessors::from_configs` File_Type gate both open for 2; 3/4 still bail.
  *PORT-TRUTH divergence (Roadmap 2, DELIBERATE -- not a reproduced quirk):* the port does
  NOT reproduce the legacy silent zero-fill / excess-ignore / `exit(1)`. It validates the
  total byte length against the header arithmetic EXACTLY and returns a typed `Err` on any
  mismatch (zero records, short header, short record table, short-or-excess payload,
  non-positive vectorSize) -- silent corruption becomes a loud, recoverable error. No C++
  oracle harness was ever built for the cep reader, so there is none left describing the
  legacy behavior; this entry is the record of the divergence. The layout itself is
  ground-truthed against the real corpus (the corpus-gated tests below), not a harness.
  *Pinned by (re-pins of the two former bails + new pins):* `audio::tests::
  read_audio_file_type_2_reads_not_bails` (was `read_audio_file_type_2_bails` -- File_Type
  2 now reaches the reader; a missing path yields a cep file-open error, not the unported
  bail) and `engine::bag_of_processors::tests::file_type_2_allowed` (was
  `file_type_2_bails` -- the gate now accepts 2); File_Type 3/4 stay pinned by the new
  `audio::tests::read_audio_file_type_3_bails` and `bag_of_processors::tests::
  file_type_3_bails`. Reader pins: `src/rust/tests/phase6_cep.rs` -- 8 hand-crafted
  synthetic fixtures under `tests/reference_data/phase6/cep/` (happy single/multi-record
  incl. magic-ignored + empty-record skip; typed-error edges: zero records, truncated
  header/table, short payload, excess payload, bad vectorSize) plus a `write_cep`/read
  round-trip. Corpus-gated layout confirmation (skips cleanly if the licensed corpus is
  absent): `corpus_first_file_consistency` (Rust, independent header parse + reader
  round-trip) and `test_cep_layout_byte_arithmetic` (Python, `tests/test_phase6_corpus.py`,
  independent pure-`struct` parse) -- both on the deterministic first file `ara_1.
  plp8f0mvsdd` (nbRecords=1, vectorSize=23, 3261 frames, 32.61s, max|x|=4.82).
  *Mutation:* reverting `read_cep` to the old bail breaks both re-pins; swapping the
  float payload to big-endian or column-major fill breaks the `tiny_ok`/`multi_ok`
  hardcoded-value asserts and the corpus plausibility bounds; relaxing the strict
  byte-length check to the legacy's silent zero-fill breaks `truncated_payload_errors`/
  `excess_payload_errors`.
  *Still deferred:* `File_Type` 3 (phSeq-N variant) and 4 (mat), `AudioStruct.cpp:257-412`
  -- the `data/LRE03-LRE07` corpus uses only cep (File_Type 2) for LID features and phSeq
  (File_Type 1); nothing exercises 3/4, so they keep the typed bail pending such data.
  This is now the shared Phase 6 corpus-gate infrastructure's first consumer:
  `common::corpus_root_or_skip` (Rust, `src/rust/tests/common/mod.rs`) and `CORPUS_ROOT`
  + `requires_corpus` (Python, `tests/conftest.py`), reused by every later Phase 6 task.

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
  run bit-for-bit either). CLOSED by the `[phase4a]` static-lane deterministic reduction entry
  above (the chosen Rust strategy: file `j` -> lane `j % N`, ascending-lane fold; N=1 is the
  golden-pinned legacy-sequential parity mode).
