# Phase 7: the efficient binary (offline) - design

Roadmap 2, phase 7. A fast inference path beside the exact one: f32, library FFT, SIMD
matmuls, allocation/memory hygiene, an RTF + peak-RSS benchmark suite. Gate: quality-parity
at tolerance vs the exact path on the phase-6 gate models + measured speed/memory budgets.

User decisions (2026-07-18, this session): PORTABLE-FIRST (pure-Rust `faer` matmuls +
`realfft` FFT, one code path, no system BLAS; benchmarked on the dev Mac); budgets
MEASURE-THEN-PIN (no absolute targets up front; measured ratios become pinned regression
bounds with headroom); gate models = the COMMITTED 2015 PACKS + the corpus-gated phase-6
SUBSET CHECKPOINTS (no dependency on the full-corpus runs, which remain post-phase user
jobs); INFERENCE-ONLY (training stays exact f64 end to end); architecture = APPROACH A,
the parallel fast module tree behind a runtime switch (the exact tree byte-untouched).

## S0. Measured and surveyed facts (2026-07-18, this box: Apple Silicon, release build)

Baseline, exact f64 path, single lane:

| run | audio | wall | RTF | peak RSS |
|---|---|---|---|---|
| SAD algo-3 spectral (real 600.96 s LRE07 wav, cap 1800) | 600.96 s | 2.40 s | 0.0040 | 387 MB |
| same, Audio_max_duration 60 / 150 / 300 / 600 | 60-600 s | 0.18 / 0.45 / 0.93 / 2.00 s | ~0.003 | 83 / 107 / 233 / 388 MB |
| LID Twin Mode-7 (real 3261-frame cep, ~32.6 s) | ~32.6 s | 0.05 s | 0.0015 | 26 MB |

- Wall-clock is ~250-300x real time single-threaded already; memory grows ~0.6 MB per
  audio-second (~9x the raw f64 samples; a 1 h file ~2.2 GB). Offline bidirectional BLSTM
  inference is inherently O(duration); phase 7 shrinks the per-second CONSTANT, phase 8
  (windowed/streaming) owns the O(1) cut.
- Code survey (Explore, file:line cites in the session ledger): every hot kernel is a
  hand-written ascending scalar loop, f64 hardcoded end to end, NO scalar-type seam
  (`Network<L>` is generic over the layer type only). `matmul_seq` (`nn/layers.rs:18-33`)
  is the SINGLE product kernel (6 call sites); `compute_two_real_periodogram`
  (`features/fft.rs:76`) the single FFT entry (GFFT radix-2, order 10 -> 1024-pt, two-real
  packing, ~30k FFTs/channel per 600 s). The LSTM recurrence is a 1xO * Ox4O `matmul_seq`
  PER TIMESTEP with a fresh allocation each call plus a `.to_owned()` per step
  (`layers.rs:249-250`). The mel bank is constructed 2x per channel (3x with the pitch
  pass) (`pipeline.rs:641-659` + `:559`, `sad.rs:1571`). Audio is held TWICE in f64
  (`data` + `data_raw`, `audio.rs:32-33`; ~154 MB for 600 s stereo). `nalgebra`,
  `rustfft`, `realfft` are declared and entirely unused. No benches/ dir, no criterion;
  release profile is `opt-level 3 + lto` only; binary 4.3 MB.
- Committed fixtures usable CI-safe: `phase4d/prcts_excerpt.wav` (2 ch, 8 kHz, 60.000 s -
  the RTF workhorse), the 3 s phase4a/4b corpus wavs, `phase4a/tier2_spectral.config`
  (algo-3), `phase4b/twin_mode7.config` (algo-6 Mode 7), the tuple-A 33,671-weight real
  2015 SAD pack, the real 12,409-weight Mode-7 LID net, committed phSeq fixtures.
- f64 is LOAD-BEARING on the exact path (golden-pinned bit-for-bit at every stage); the
  only f32 touchpoints today are the fmath_log bit-trick table, cep decode, and symphonia
  PCM widening.

## S1. Deliverables

### Track 1 - the fast tree

- **S1.1 `fast::pipeline`**: f32 feature extraction - windowing, periodogram via
  `realfft` (one plan per run at the config's window size), mel filterbank + DCT with the
  bank PRECOMPUTED ONCE per run, LTSV when the config asks. Audio decoded once into a
  single f32 buffer (no `data`/`data_raw` duplication on this path). Numeric equivalence
  to the exact features is AT TOLERANCE, never bit (library FFT + f32 by design).
- **S1.2 `fast::nn`**: f32 BLSTM forward-only - batched input projection and per-timestep
  recurrence through `faer` into PREALLOCATED workspaces (zero per-frame allocation), SoA
  gate layout, the exact activations transcribed to f32 (`asinh` cell/output,
  `sigmoid(0.1z)` gates, overflow-guarded softmax), SubSample, the dense output MLP.
  Forward-variant scope: what the gate configs exercise (plain FFB first); an unexercised
  windowed variant typed-bails (the house pattern), added only when a gate model needs it.
- **S1.3 `fast::driver`**: fast counterparts of the two inference drivers that matter -
  algo-3 spectral SAD and algo-6 Mode-7 LID (File_Type 1 phSeq and File_Type 2 cep).
  They produce posteriors/result-vecs and hand off to the SHARED, UNCHANGED f64 decision
  layer (`results_to_segmentation`, `Segmentation`, smoothing, VRCTS/`.scr` writers).
  Boundary logic is cheap and must match exactly; it is not duplicated. Out of scope:
  VrctsPart (external tool), NN-free algos 1/2, all training paths, the pitch second pass
  (typed-bail on `TDCwindow > 0` in fast mode; no gate config uses it).
- **S1.4 Weights**: the same f64 `.bin` packs, narrowed to f32 ONCE at fast-net
  construction, AFTER `adim_coeff` scaling. No format change, no packer touch, no PyO3
  signature change.
- **S1.5 Selection**: new config key `Inference_Path` (`exact` | `fast`, DEFAULT `exact`),
  TOML `[engine] inference_path`, threaded through bag construction (the `Processor` enum
  gains fast-dispatch variants); the existing CLI override mechanism gives
  `--Inference_Path=fast` for free in solo/image/unit modes. The exact tree stays
  byte-untouched by selection plumbing except the single dispatch site.

### Track 2 - proof

- **S1.6 The parity gate, CI tier**: fast vs exact on the SAME inputs, committed
  artifacts only. (a) tuple-A SAD pack + `tier2_spectral.config` + `prcts_excerpt.wav`:
  posterior tolerance (MEASURED then pinned with headroom), segment count/types
  IDENTICAL, boundary max-dt pinned (measured; the 4-decimal VRCTS rounding may absorb
  it to 0), scored-column tolerance. (b) the real Mode-7 LID net + `twin_mode7.config` +
  the committed phSeq fixtures: `.scr` score tolerance + IDENTICAL argmax decisions.
  Runs in CI forever - it is the standing drift detector for Approach A's two-trees cost.
- **S1.7 The parity gate, corpus tier (local, corpus-gated)**: the phase-6 subset
  checkpoints (retrained at gate-setup time by the committed recipe if not on disk) on
  real corpus slices - METRIC-level parity: DCF / lid_error / cavg computed from
  fast-path outputs equal the exact path's, or differ below a measured-then-pinned
  epsilon; argmax decisions compared file-by-file.
- **S1.8 The bench suite**: `criterion` micro-benches (matmul kernel, FFT, mel apply,
  activation sweep) + a `speech bench` CLI subcommand measuring end-to-end wall/RTF and
  peak RSS (ru_maxrss) per path per arm - the 60 s fixture as the CI smoke (assert only
  "fast is not slower than exact" with wide headroom in CI; exact numbers reported), the
  600 s corpus runs local. Results land in a RESULTS.md phase-7 table (exact vs fast:
  RTF, MB per audio-second, speedup, named hardware); the measured ratios become the
  pinned regression bounds. That table IS the roadmap's "measured speed/memory budgets".

### Track 3 - shared-code hygiene + docs

- **S1.9 Behaviorally-neutral exact-path fixes ONLY**, each golden-re-verified: the
  mel-bank double-construction hoist (clear); a `data`/`data_raw` adjudication (the
  restore buffer is dropped only where provably unread; else documented and kept).
  Anything that moves one bit stays out of the exact tree.
- **S1.10 Dependency hygiene**: `nalgebra` REMOVED (verifiably unused); `realfft` used;
  `rustfft` dropped if `realfft` covers the need (it is realfft's backend - keep only
  what Cargo requires); `faer` added. CLAUDE.md's "nalgebra/ndarray for linear algebra"
  convention line corrected in the docs task.
- **S1.11 Docs**: CLAUDE.md module-map rows for `fast/`; README + RESULTS.md phase-7
  sections; the mutation battery + docs refresh + final whole-branch review + smart-commit
  + finish, per the house phase shape.

## S2. Gate/oracle tiers

There is no external oracle this phase: the EXACT PATH IS THE ORACLE. Tiers: (1) CI
parity vs the exact path on committed packs/fixtures (S1.6); (2) corpus-gated metric
parity on real data (S1.7); (3) measured-then-pinned performance bounds (S1.8). The
exact path's own correctness rests on the five phases of golden/parity work behind it.

## S3. Risks

- **R1 f32 tolerance is unknown until measured**: an LSTM recurrence accumulates f32
  error over T~60k frames; the posterior delta distribution must be MEASURED first, the
  pin set with headroom, and the segment-level assertions (count/types identical) are the
  real gate - if a fixture sits knife-edge on a hysteresis threshold and flips a
  boundary, that is a FINDING to adjudicate (widen to max-dt bounds with the flip
  documented), not to hide.
- **R2 two-trees drift**: Approach A's cost. Mitigation: the CI parity gate runs forever;
  the fast tree's module docs cite the exact-path file:line they mirror; the final review
  audits the pairing.
- **R3 library behavior**: `realfft` scaling conventions and `faer` accumulation order
  differ from the hand kernels BY DESIGN - documented in module docs, absorbed by the
  tolerance gate, never "fixed" by contorting the fast path toward bit-parity.
- **R4 the pitch pass + exotic variants**: typed-bailed in fast mode until a gate model
  needs them; the bail is pinned by a test (the house pattern), so scope creep is loud.
- **R5 benchmark stability in CI**: shared runners make timing flaky - CI asserts only
  the wide not-slower-than-exact smoke; the real numbers are local, recorded in
  RESULTS.md with hardware named.

## S4. Conventions

- The exact tree is UNTOUCHABLE except S1.9's golden-re-verified neutral hoists and the
  single S1.5 dispatch site. Fast-path numeric divergence is by design: documented in
  module docs + RESULTS.md, NOT IMPROVEMENTS.md (which records legacy-engine quirks).
- Corpus gating exactly as phase 6 (skipif-local-only; CI sees committed fixtures only;
  the license bright line holds - no real corpus identifiers or per-file content values
  in tracked files).
- ASCII, mypy/ruff/clippy clean, the commit trailer, never push, feature branch only.
