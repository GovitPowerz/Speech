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
  `length = (long)((itEndRef-1).begin / timeStep) + 1`, so the final `ii` makes `currentTime` land
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

### Forward-noted (add the entry when the phase reproduces it)

- **[0b-ii] `updateSegmentation` tail uses `results.size()` (rows*cols), not `length`** -- correct
  only for a pure row-vector; note when 0b-ii lands.
- **[3/4] `CostFunctionCalib` nnet_out-before-assignment** (Python optimizer) -- a real
  use-before-assign bug flagged in the setup analysis; fix when the Python training loop is ported.
