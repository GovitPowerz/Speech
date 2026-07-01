# Phase 0b-i - Cost Laws + Segmentation Container Design Spec

Date: 2026-07-01
Status: approved (design), pending implementation plan
Branch: feature/phase-0b-i-cost-container

## 1. Goal and scope

Port two independent pure-logic pieces of the legacy engine to Rust: (A) the cost-law layer
(`CostLaw.cpp`/`.h`) - the 7 piecewise VAD cost primitives plus the multiclass softmax
cross-entropy; and (B) the `Segmentation` boundary-list container (`Segmentation.cpp`) - the
segment model plus its mutation primitives. Rust-only production code; a Python scipy prep extracts
the legacy `.mat` cost sweeps into committed fixtures.

**Acceptance:** (A) the scalar cost/deriv laws reproduce the real `test_costlaws.mat` /
`test_costlaws_deriv.mat` sweeps bit-for-bit; (B) the container ops (`label_segment`, `sanitize`,
`suppress_short`, `add_padding`, `modify_type`, `update_count`) pass hand-verified unit tests, with
`sanitize`'s 1e-4 round-half-away-from-zero primitive exercised.

This is Phase 0b-i. Phase 0b-ii (segmenter decision + smoothing + VRCTS/STM I/O + `compute_errors`
scoring) is the next cycle.

## 2. Decisions (locked)

| # | Decision | Choice |
|---|----------|--------|
| 1 | Cycle scope | 0b-i (cost + container) now; 0b-ii (segmenter + I/O + scoring) next. |
| 2 | Language | Rust-only production (`CostLaw`/`Segmentation` are C++ logic; the MATLAB cost files are PSO wrappers, not the NN cost). A Python scipy prep extracts the `.mat` sweeps into fixtures. |
| 3 | `.mat` I/O | `io::matfile` stays deferred; only the Python prep (scipy) reads the `.mat`. |
| 4 | Cost golden | The scalar VAD laws are golden-validated bit-for-bit against `test_costlaws.mat`. The softmax multiclass path + the non-linear laws get unit tests from the exact `CostLaw.h` coefficients. |
| 5 | WER / Pass 1-3 scoring | Out of scope (0b-ii): Pass 2 golden + Pass 1/3 unit-test-only, decided for the 0b-ii cycle. |

## 3. Cost laws (exact) - `src/rust/src/cost.rs`

Authoritative source: `CostLaw.cpp` / `CostLaw.h`. All math is IEEE-754 `f64`.

### 3.1 Parameter setup (`CostLaw.cpp:7-80`)

```
cp  = clamp(cfg "<pre>_CostPonderation"      default 0.5, 0.01, 0.99)
pS  = clamp(cfg "<pre>_CostLawParamSpeech"   default 0.0, 0.0, 1.0)
pN  = clamp(cfg "<pre>_CostLawParamNoSpeech" default 0.0, 0.0, 1.0)
thS     = clamp(cfg "<pre>_CostLawThreshSpeech"   default 1.0, 1e-6, 1-1e-6)
thN_law = 1.0 - clamp(cfg "<pre>_CostLawThreshNoSpeech" default 0.0, 1e-6, 1-1e-6)
```

Speech law objects are built with `(P=cp, q=pS, t=thS)`. No-Speech law objects are built with
`(P=1-cp, q=pN, t=thN_law)` - BOTH the ponderation and the threshold argument are inverted.

**DECISIVE DOUBLE-READ QUIRK (`CostLaw.cpp:68-69`):** after the law coefficients are built, the
runtime BRANCH selectors are re-read RAW and UNCLAMPED, overwriting the members:
`switching_thresh_speech = cfg "<pre>_CostLawThreshSpeech" default 10.0`,
`switching_thresh_no_speech = cfg "<pre>_CostLawThreshNoSpeech" default -1.0`. So the LAWS use the
clamped threshold (e.g. `1-1e-6`) while the runtime BRANCH predicate uses the raw config value (e.g.
exactly `1.0` / `0.0`), or `10.0` / `-1.0` when the key is absent.

`back_prop_wer = (cfg "<pre>_BackPropWER" default -1.0 >= 0)` is a bool. `classes_ponderations =
get_list<f64>("<pre>_classes_ponderations")` stored as a row vector iff non-empty and `list[0] > 0`,
else empty (no ponderation).

### 3.2 The 7 piecewise primitives (`CostLaw.h`), each from `(P, q, t)`

```
1) LinearLaw:            A=-P*(1-q)/t;  B=P.                    cost=B+A*y;            deriv=A (const)
2) SquareLaw:            A=P*(1-q)/t/t; B=-P*2*(1-q)/t; C=P.    cost=C+B*y+A*y*y;      deriv=B+2*A*y
3) BelowThreshCubicLaw:  A=P*2*(1-q)/t^3; B=-P*3*(1-q)/t^2; D=P. cost=D+B*y*y+A*y*y*y; deriv=2*B*y+3*A*y*y
4) AboveThreshCubicLaw:  coeffs by law NAME:
     name in {square,cubic}: A=-P*2*q/(1-t)^3;  B=P*3*q/(1-t)^2
     name in {linear,log}:   A=P*((1-q)/t/(1-t)^2 - 2*q/(1-t)^3);  B=P*(3*q/(1-t)^2 - (1-q)/t/(1-t))
     else exit(1).
     cost(y):  y := 1-y;  return B*y*y + A*y*y*y
     deriv(y): y := 1-y;  return -2*B*y - 3*A*y*y   (minus signs from the 1-y chain rule)
5) LogLaw:               A=-P*(1-q); B=P*q; Adim=t.
     cost(y):  y := y/Adim; clamp y to [1e-24, 1]; return B + A*ln(y)
     deriv(y): clamp RAW y (NOT /Adim) to [1e-24, 1]; return A/y
     ASYMMETRY: cost divides by Adim before clamp+log; deriv clamps raw y. Reproduce exactly.
6) BelowThreshSqrtLaw:   A=P*(1-q); B=P*q; Adim=t.
     cost(y):  y := y/Adim; if y>1 y=1; return B + A*sqrt(1-y)
     deriv(y): y := 1 - y/Adim; if y<1e-24 y=1e-24; return -A/2/sqrt(y)
7) AboveThreshSqrtLaw:   A=P*(-q); B=P*q; Adim=1-t.
     cost(y):  y := 1-y; y := y/Adim; if y>1 y=1; return B + A*sqrt(1-y)
     deriv(y): y := 1-y; y := 1 - y/Adim; if y<1e-24 y=1e-24; return A/2/sqrt(y)
```

**Construction routing (`CostLaw.cpp:28-63`):** if `CostLawSpeech == "sqrt"`: build both
`AboveThreshSqrtLaw` and `BelowThreshSqrtLaw` (speech uses `thS`). Else build the matching
below-thresh law (`square`->Square, `log`->Log, `linear`->Linear, `cubic`->BelowThreshCubic) AND an
`AboveThreshCubicLaw(cp, pS, thS, name)`. Unknown name -> exit(1). No-Speech mirror with
`(1-cp, pN, thN_law)`.

### 3.3 Scalar VAD path - `compute_unitary_cost` / `compute_unitary_deltas` (`CostLaw.cpp:122-189`)

Per element, `target < 0` -> cost 0, delta 0 (ignore). Else, `target > 0.5` = SPEECH regime, else
NO-SPEECH. Within a regime, `output < switching_thresh_<regime>` selects the below-thresh law
(dispatch on the law name), else the above-thresh law. Deltas apply the logistic chain-rule factor
`output*(1-output)` on the law derivative (the network output is a sigmoid). The `switching_thresh`
predicate uses the RAW double-read value (3.1).

### 3.4 Multiclass softmax path - `compute_cost` / `compute_deltas` (target > 1 column)

Softmax cross-entropy directly (the piecewise laws are NOT used): `delta = output - onehot(target)`,
`target < 0` rows ignore-masked, optional per-class ponderation from `classes_ponderations` and the
WER ponderation when `back_prop_wer`. Port the exact form from `CostLaw.cpp` (the multiclass branch).
Unit-tested (no `.mat` golden for this path).

## 4. Segmentation container (exact) - `src/rust/src/tasks/segmentation.rs`

Authoritative source: `Segmentation.cpp`.

### 4.1 Model

- `SegClass` `#[repr(i32)]` enum, codes LOAD-BEARING: `OTHER=0, SPEECH=1, RING=2, DTMF_0..9=3..12,
  DTMF_A..D=13..16, DTMF_STAR=17, DTMF_SHARP=18, INSERTION=19, SUBSTITUTION=20, EXCLUDED=21, END=22`.
- `Segment { begin: f64, ty: SegClass }`. A `Segmentation` (per channel) is a boundary list sorted by
  `begin`: element `i` covers `[seg[i].begin, seg[i+1].begin)` with type `seg[i].ty`. Always starts
  `{0.0, OTHER}` and ends with a sentinel `{audio_duration, END}`. Use an index-based `Vec<Segment>`
  (NOT deque iterator arithmetic).
- `audio_duration = (frame_count - 1) / frame_rate` (seconds).

### 4.2 Ops (port exactly; index-based)

- `label_segment(begin, end, class)` (`:147-174`): if `end <= first.begin` -> no-op; clamp `begin` up
  to `first.begin`; if `begin < sentinel.begin`: advance past boundaries `< begin` (tracking
  `previous_type`), insert `{begin, class}`, erase boundaries with `begin < end` (updating
  `previous_type` to the last erased type), then re-close by inserting `{end, previous_type}` unless
  that would duplicate the sentinel. Overlapping later labels overwrite earlier ones.
- `sanitize()` (`:176-194`): walk pairwise; round each `begin` to the 1e-4 grid via
  `round_half_away_from_zero(t * 1e4) / 1e4`; if two adjacent segments share `ty`, erase the later
  boundary (merge, keeping the earlier boundary). The terminal sentinel is NOT rounded (the loop
  exits at `it_next == end`). Use `copysign`-based half-away rounding, NOT banker's rounding.
- `suppress_short(threshold, class)` (`:245-282`): `threshold > 0` gate (else no-op, no sanitize).
  Walk; if a segment is `class` and its duration `<= threshold`: at head -> pull the next boundary
  back and erase; if next-is-sentinel -> erase; if `previous_type == (it+1).ty` -> erase both
  boundaries (full merge); else `(it+1).begin -= dur/2` then erase (half-split). NO advance after an
  erase; `previous_type` only updated on the else (kept) branch. Then `sanitize`.
- `add_padding(before, after, class)` (`:216-243`): grow each `class` segment by `before` on the
  left and `after` on the right via `label_segment`; `before == 0` / `after == 0` skip that side.
  (In the pipeline it is called twice with a 4-tuple `[p1_before, p1_after, p2_before, p2_after]` -
  the 4-tuple wiring belongs to 0b-ii's smoothing; here `add_padding` takes scalar before/after.)
- `modify_type(before, after)`: retype every non-sentinel segment `before -> after`, then `sanitize`.
- `update_count() -> [f64; 23]` (`:206`): `sanitize`, then per class accumulate
  `next.begin - cur.begin`; returns per-class total duration (index by `SegClass as usize`).

### 4.3 Testing

Hand-built boundary lists with hand-verified expected outputs for each op, covering the tricky paths:
`label_segment` mid-list insert/erase/reclose; `sanitize` merge + the terminal-not-rounded rule +
the round-half-away boundary at `x.00005`; `suppress_short` all four branches (head, pre-sentinel,
neighbor-match full-merge, neighbor-differ half-split) and the no-advance-after-erase invariant;
`add_padding` before/after zero skips; `update_count` per-class durations. A property test (proptest)
that `sanitize` is idempotent and never reorders boundaries.

## 5. Module interfaces

- `src/rust/src/cost.rs`
  - `pub struct CostLaw { ... }` with `from_config(&IndexMap<String,String>, prefix: &str) -> CostLaw`
  - `compute_unitary_cost(&self, output: f64, target: f64) -> f64`
  - `compute_unitary_delta(&self, output: f64, target: f64) -> f64`
  - `compute_cost(&self, outputs: &[f64], targets: &[f64], n_classes: usize) -> f64` (softmax path)
  - `compute_deltas(&self, outputs: &[f64], targets: &[f64], n_classes: usize, out: &mut [f64])`
- `src/rust/src/tasks/segmentation.rs`
  - `pub enum SegClass` (`#[repr(i32)]`); `pub struct Segment { pub begin: f64, pub ty: SegClass }`
  - `pub struct Segmentation { segs: Vec<Segment>, audio_duration: f64 }` with `new(audio_duration)`,
    `label_segment`, `sanitize`, `suppress_short`, `add_padding`, `modify_type`, `update_count`, and
    read accessors (`segments()`, `audio_duration()`).
- `scripts/extract_phase0b_costlaw_fixtures.py` (scipy prep): `loadmat` `test_costlaws.mat` /
  `test_costlaws_deriv.mat`; emit `tests/reference_data/phase0b/{costlaw_speech.bin,
  costlaw_nospeech.bin, costlaw_deriv_speech.bin, costlaw_deriv_nospeech.bin}` each as a `1 x N` `.bin`
  (2 columns interleaved or two files: `output` col + expected col). Also record the sweep-generating
  config in `costlaw_config.json` (the anchors imply `CostLawSpeech=linear`, `CostPonderation=0.2`,
  params 0, `ThreshSpeech=1.0`/`ThreshNoSpeech=0.0`; confirm empirically against the anchors).

## 6. Golden fixtures, strategy, acceptance

**Vendored into `tests/reference_data/phase0b/` (committed):**
- The extracted cost/deriv sweep arrays (from `test_costlaws.mat` / `_deriv.mat`), as `.bin` via the
  Phase-0a codec, plus `costlaw_config.json` (the sweep config).
- Source (local, git-ignored): `.../FastSpeechProcessing-legacy/Release/bin/test_costlaws.mat` and
  `test_costlaws_deriv.mat`.

**Tests (Rust `tests/`, + the Python prep assertion):**
1. Cost sweep (THE golden): build `CostLaw` from `costlaw_config.json`; for each of the 1000 rows,
   `compute_unitary_cost(output, target=1.0)` == the speech-sweep expected col, and `target=0.0` ==
   the no-speech col, bit-for-bit (`==` on `f64`). Same for `compute_unitary_delta` vs the deriv
   sweeps. (Anchor: `output=0.001, target=1 -> 0.1997999997999998`.)
2. Per-law unit tests: Square/BelowThreshCubic/AboveThreshCubic(name routing)/Log(asymmetry)/
   BelowThreshSqrt/AboveThreshSqrt cost+deriv at hand-computed points from the `CostLaw.h` coeffs.
3. Double-read: a config with `CostLawThreshSpeech` absent yields `switching_thresh_speech == 10.0`
   while the law uses `thS == 1-1e-6` (assert the branch predicate vs law-coeff divergence).
4. Softmax path: `compute_cost`/`compute_deltas` on a small hand-built multiclass example
   (`delta == output - onehot`, ignore-mask on `target<0`).
5. Container ops: the section 4.3 unit + property tests.

## 7. Risks and pitfalls (pinned by tests)

- The **double-read** (clamped law-thresh vs raw branch-thresh, defaults 10.0/-1.0): collapsing them
  passes the committed config by luck but breaks any config where raw != clamp.
- **LogLaw cost/deriv Adim asymmetry** (cost divides `y` by Adim before clamp; deriv clamps raw `y`):
  a "clean" symmetric version fails the log-law goldens.
- **AboveThreshCubic name-dependent coefficients** and the `1-y` substitution sign flips in its deriv.
- **`sanitize` rounding**: `round_half_away_from_zero`, NOT banker's rounding (numpy `round` / Rust
  `f64::round` semantics differ at `x.5`); the terminal sentinel is never rounded.
- **`SegClass` integer codes** are load-bearing (printed as labels, used as count indices) - `#[repr(i32)]`.
- **`suppress_short` no-advance-after-erase** and the head/pre-sentinel/merge/half-split branch
  selection - index-fragile when porting C++ deque iterators to a Rust `Vec`.

## 8. Definition of done

- `cargo build --release`, `clippy --all-targets -- -D warnings`, `cargo test`, `fmt --check` green.
- `ruff`/`mypy` clean on the prep script's touched paths; `pytest` green.
- The cost sweep golden passes bit-for-bit in Rust; the container + law unit/property tests pass.
- `cost.rs` and `tasks/segmentation.rs` no longer rely on any crate-level lint allow for their items.
- CLAUDE.md / README updated (cost + container implemented; 0b-i done).
- Final step (in the plan): invoke `smart-commit` over the branch.

## 9. Out of scope

Phase 0b-ii: `tasks::segmenter` (update_segmentation hysteresis-with-area, lid_to_segmentation,
the 7-step smooth_segmentation, get_targets), `tasks::segmentation_io` (VRCTS XML + STM/CSV loaders +
ASCII writer), `compute_errors` (Pass 2 Pfa/Pmiss golden; Pass 1/3 WER unit-test-only). Also: the
softmax multiclass path's `.mat` golden (none exists); `io::matfile`; the NN forward pass.
