# Phase 0b-ii - Segmenter Decision + I/O + Scoring Design Spec

Date: 2026-07-01
Status: approved (design), pending implementation plan
Branch: feature/phase-0b-ii-segmenter-scoring (stacked on 0b-i)

## 1. Goal and scope

Complete the pure-logic segmentation layer in Rust, on top of the 0b-i `Segmentation` container: (A)
the segmenter decision logic (`tasks/segmenter.rs`) - `update_segmentation`, `lid_to_segmentation`,
the 7-step `smooth_segmentation`, `get_targets`; (B) the segmentation I/O (`tasks/segmentation_io.rs`)
- VRCTS XML read/write, STM/CSV reference loaders, ASCII writer; (C) the `compute_errors` scoring.

**Acceptance:** the VRCTS XML round-trip is byte-exact against the real `XmlPart/*.xml` fixtures; the
segmenter decision + `compute_errors` Pass 2 are validated by hand-computed unit tests plus a Python
differential oracle (Rust == Python). Pass 1/3 (WER/coverage/delay) is unit-tested only (no `.csv` in
the tree).

This is Phase 0b-ii (one cycle: segmenter + I/O + scoring). It closes the Phase 0/0b pure-logic +
I/O parity layer. Features, NN forward, and the end-to-end inference golden are later phases.

## 2. Decisions (locked)

| # | Decision | Choice |
|---|----------|--------|
| 1 | Scope | One 0b-ii cycle: segmenter + segmentation_io + compute_errors, in dependency order. |
| 2 | Segmenter/scoring validation | No features->segments legacy golden exists (needs the NN, Phase 2). Validate by hand-computed unit tests + a Python differential oracle (numpy port), cross-checked Rust == Python. The VRCTS XML round-trip IS a real-artifact golden. |
| 3 | WER (Pass 1/3) | Unit-test-only from hand-built deques - no `.csv` reference in the tree (from the 0b-i decision). Pass 2 (Pfa/Pmiss/ErrorRate) is the golden-able scoring path. |
| 4 | Language | Rust-only production; a Python differential oracle (`speech.scoring`-side numpy) for the segmenter/compute_errors cross-check. `io::matfile` still deferred. |

## 3. Segmenter decision (exact) - `src/rust/src/tasks/segmenter.rs`

Authoritative source: `Segmenter.cpp`, `Segmenter.h`. `dt` is the guidance time step; `off` the audio
offset; `classType` the produced class (SPEECH for SAD). All comparisons preserve the exact
`>=`/`<`/`<=`/`>` directions.

### 3.1 Config (`buildFromConf`)

`thresh/area rising+falling`; the clamp `if (falling > rising) falling = rising`. `_PaddingSpeech`
requires length >= 4 (else `exit(1)`), each negative clamped to 0. `_MinSpeech` requires >= 3,
`_MinSilence` requires >= 2, each negative clamped to 0. Real golden params (`24-Feb-2014_BLSTM_Spect.config`,
last-duplicate-key wins): rising 0.6 / area 0.05, falling 0.3 / area 0.03; padding
0.2,0,0.3,0.4; min_silence 0.3,0.5; min_speech 0,0.3,0.4. LID (`LID_BLSTM.config`): LID rising 0.5,
falling -0.5 (falling ignored).

### 3.2 `update_segmentation` (hysteresis-with-area, row-vector, `Segmenter.cpp:725-845`)

Reads `r(k) = results(0, k)`, `length = results.cols()` (ROW vector). State
`begin=end=beginArea=endArea=-1`, `hasBegun=hasEnded=false`. `tR/aR` = rising thresh/area, `tF/aF` =
falling.

- INIT: `if r(0) >= tR { begin = 0; beginArea = 0; }`.
- LOOP `ii = 1..length-1`:
  - RISING (if `!hasBegun`): if `begin < 0`: if `r(ii) >= tR && r(ii-1) < tR` {
    `begin = dt*(ii - (r(ii)-tR)/(r(ii)-r(ii-1)))`; `beginArea = (ii - begin/dt)*(r(ii)-tR)/2`; if
    `beginArea >= aR` set `hasBegun`}. Else (`begin >= 0`): if `r(ii) >= tR` { `beginArea +=
    (r(ii)+r(ii-1)-tR*2)/2`; if `>= aR` set `hasBegun`} elif `r(ii) < tR && r(ii-1) >= tR` {
    `beginArea += (r(ii-1)-tR)*(tR-r(ii-1))/(r(ii)-r(ii-1))/2`; if `>= aR` set `hasBegun` else reset
    `begin/beginArea/hasBegun`} else reset `begin/beginArea/hasBegun`.
  - FALLING (if `hasBegun`): symmetric with `tF/aF` (`end`, `endArea`), see the RRopFalling formulas:
    if `end < 0`: if `r(ii) <= tF && r(ii-1) > tF` { `end = dt*(ii - (r(ii)-tF)/(r(ii)-r(ii-1)))`;
    `endArea = (ii - end/dt)*(tF-r(ii))/2`; if `>= aF` set `hasEnded`}. Else: if `r(ii) <= tF` {
    `endArea += (2*tF - r(ii) - r(ii-1))/2`; if `>= aF` set `hasEnded`} elif `r(ii) > tF && r(ii-1)
    <= tF` { `endArea += (tF-r(ii-1))*(tF-r(ii-1))/(r(ii)-r(ii-1))/2`; if `>= aF` set `hasEnded` else
    reset `end/endArea/hasEnded`} else reset.
  - if `hasEnded`: if `begin < end` { `label_segment(begin+off, end+off, classType)`; reset all 6;
    then re-run the INIT rising check at `ii` } else reset all 6.
- TAIL: if `hasBegun` { `end = dt * results.size()`; `label_segment(begin+off, end+off, classType)`;
  reset }. QUIRK: the tail uses `results.size()` (`rows*cols`), NOT `length` - equal only for a pure
  row vector. Reproduce; log in IMPROVEMENTS.md.
- FINALLY: `smooth_segmentation(seg, chan)`.

### 3.3 `lid_to_segmentation` (single-threshold, col-vector, `:999-1111`)

Reads `r(k) = results(k, 0)`, `length = results.rows()` (COL vector - the orientation asymmetry vs
3.2 is load-bearing). Only `threshMax` used (no area/hysteresis). Rising: `r(ii) >= threshMax &&
r(ii-1) < threshMax` -> linear-interp `begin`. Falling: `r(ii) <= threshMax && r(ii-1) > threshMax`
-> `end`. On `begin < end`, `label_segment`; reset. TAIL uses `results.size()`. END: `sanitize` only
(NO `smooth_segmentation`).

### 3.4 `smooth_segmentation` (the fixed pipeline, `Segmenter.cpp:709-723`)

Exactly, over the 0b-i container ops:
```
sanitize()
suppress_short(min_speech[0],  SPEECH)
add_padding(padding[0], padding[1], SPEECH)
suppress_short(min_silence[0], OTHER)
suppress_short(min_speech[1],  SPEECH)
add_padding(padding[2], padding[3], SPEECH)
suppress_short(min_silence[1], OTHER)
suppress_short(min_speech[2],  SPEECH)
```

### 3.5 `get_targets` (`:659-707`, only if `Reference` nonempty)

Per row `ii`, `t = ii*timeStep + timeOffset`; advance `itRef` while `(itRef+1).begin <= t`.
- If `back_prop_wer >= 0`: SPEECH/SUBSTITUTION -> `durSeg = max(next-cur, timeStep)`, `target = 1 -
  0.1*timeStep/durSeg`; EXCLUDED -> `-0.5`; INSERTION -> `target = 0.1*timeStep/durSeg`; else the
  neighbor-window branches (see source) or `target = timeStep/100`.
- Else (`back_prop_wer < 0`): `if itRef.ty == classType || (classType==SPEECH && itRef.ty==SUBSTITUTION)`
  -> `1.0`; EXCLUDED -> `-0.5`; else `0.0`.
Writes `targets(ii, 0)`.

## 4. Segmentation I/O (exact) - `src/rust/src/tasks/segmentation_io.rs`

Authoritative source: `Segmentation.cpp`, `VRCTSpart.cpp`. Times in files are absolute; the audio
offset is subtracted on load.

### 4.1 VRCTS XML (verified against real `XmlPart/*.xml`)

Writer emits (per channel), exact formatting:
```
<?xml version="1.0" encoding="UTF-8"?>
<AudioDoc name="<name>" path="<path>">
<ProcList>
<Proc name="vrcts_part" version="1.3"/>
</ProcList>
<ChannelList>
<Channel num="1" sigdur="<audio_duration:.2f>" spdur="<speech_duration:.2f>"/>
</ChannelList>
<SpeakerList>
<Speaker ch="1" dur="<speech_duration:.2f>" gender="1" spkid="1"/>
</SpeakerList>
<SegmentList>
<SpeechSegment ch="1" sconf="1.00" stime="<begin:.3f>" etime="<end:.3f>" spkid="1"/>
...one line per SPEECH segment...
</SegmentList>
</AudioDoc>
```
- `sigdur`/`spdur`/`dur`/`sconf` are 2-decimal; `stime`/`etime` are 3-decimal. Only SPEECH segments
  are emitted (an all-OTHER segmentation yields an empty `<SegmentList>`). The output filename carries
  a `_chan_N` suffix (the `_chan_N` rule). Match the real fixtures byte-for-byte (incl. the trailing
  newline).
- Parser: scanf-style over `SpeechSegment`; accept a segment iff the first token is `SpeechSegment`
  AND `end >= off` AND `begin < off + dur`; load into a `Segmentation` as SPEECH intervals.

### 4.2 STM loader (`load_ref_from_stm`)

`;;`-prefixed header lines skipped; 7 whitespace tokens per line. `SPEECH` iff `second.startswith(first)`
(the transcription-present test); `excluded_region` -> EXCLUDED/OTHER gated on the `exclude_nontrans`
config. Verified against the real `PRCTS_RUS_RU_..._01.stm` (2-channel).

### 4.3 CSV loader (`load_ref_from_csv`) + ASCII writer

CSV: comma->space, 4 fields, `end -= 1e-4`, `_NbWords++` for type != 'I', sets `_NbWords >= 0` (the
only loader that does - it enables Pass 1/3). ASCII writer: plain `begin end label` per segment. (No
`.csv` fixture in the tree - unit-tested only.)

## 5. Scoring (exact) - `compute_errors` in `src/rust/src/tasks/segmentation_io.rs` (or a `scoring` submodule)

Authoritative source: `Segmentation.cpp:288-476`.

- NO reference (`_Reference` empty): per channel `sanitize`, `update_count` into `_RefCount`, print
  `100*count/_AudioDuration` for classes with count > 0.
- WITH reference, per channel:
  - PASS 1 (WER/coverage/delay) - ONLY if `_NbWords >= 0` (CSV loaded). Unit-tested from hand-built
    deques; NO real golden (no `.csv`). Two-pointer ref/hyp walk (`:313-405`).
  - PASS 2 (Pfa/Pmiss/ErrorRate, `:426-476`): first `sanitize` hyp; `modify_type` ref
    SUBSTITUTION->SPEECH and INSERTION->OTHER; `update_count` ref into `_RefCount`; init
    `_ClassificationErrors[j] = 0` for `j in [OTHER, END)`. Two-pointer merge (`itRef` from `begin`,
    `it` from `begin()+1`):
    - if `itRef.begin <= it.begin`: if ref scorable (`itRef.ty != END`, `!= EXCLUDED`, `!= (it-1).ty`,
      and NOT the exemption `(it-1)==SPEECH && itRef==SUBSTITUTION`): `error = (it.begin > (itRef+1).begin
      ? (itRef+1).begin - itRef.begin : it.begin - itRef.begin)`; `Pmiss[itRef.ty] += error`;
      `ErrorRate[(it-1).ty] += error`; `Pfa[(it-1).ty] += error`. `++itRef`.
    - else: if (`it.ty != END`, `(itRef-1).ty != EXCLUDED`, `(itRef-1).ty != it.ty`): `error =
      (itRef.begin > (it+1).begin ? (it+1).begin - it.begin : itRef.begin - it.begin)`;
      `Pmiss[(itRef-1).ty] += error`; `ErrorRate[it.ty] += error`; `Pfa[it.ty] += error`. `++it`.
    - Normalization (`:456-476`): `count_excluded = _RefCount[EXCLUDED]`; per class `j in [OTHER, EXCLUDED)`:
      `count = _RefCount[j]`; `count_others = _AudioDuration - count - count_excluded`;
      `ErrorRate /= (_AudioDuration - count_excluded)` if `> 0` else 0; `Pmiss /= count` if `> 0` else
      0; `Pfa /= count_others` if `> 0` else 0. Raw accumulators in seconds; display `* 100`.
  - PASS 3 (WER printout) - display only.

## 6. Module interfaces

- `src/rust/src/tasks/segmenter.rs`
  - `pub struct SegmenterConfig` (rising/area/falling/area, `padding: [f64;4]`, `min_speech: [f64;3]`,
    `min_silence: [f64;2]`, thresholds) with `from_config(&IndexMap<String,String>, prefix) -> Self`
    (the `falling>rising` clamp + require-length checks).
  - `update_segmentation(seg: &mut Segmentation, results_row: &[f64], class: SegClass, off: f64, dt: f64, cfg: &SegmenterConfig)`
  - `lid_to_segmentation(seg: &mut Segmentation, results_col: &[f64], class: SegClass, off: f64, dt: f64, thresh_max: f64)`
  - `smooth_segmentation(seg: &mut Segmentation, cfg: &SegmenterConfig)`
  - `get_targets(seg: &Segmentation, reference: &Segmentation, time_step: f64, time_offset: f64, back_prop_wer: f64, class: SegClass) -> Vec<f64>`
- `src/rust/src/tasks/segmentation_io.rs` (replaces the scaffold stub `Segment`; consumes the 0b-i
  `tasks::segmentation::{Segmentation, Segment, SegClass}`)
  - `write_vrcts(seg: &Segmentation, name: &str, path_attr: &str, out: &Path) -> Result<()>`
  - `load_vrcts(text: &str, off: f64, dur: f64) -> Segmentation`
  - `load_ref_stm(text: &str, chan: usize, off: f64, dur: f64, exclude_nontrans: bool) -> Segmentation`
  - `load_ref_csv(text: &str, off: f64, dur: f64) -> (Segmentation, i64 /*nb_words*/)`
  - `write_ascii(seg: &Segmentation, out: &Path) -> Result<()>`
  - `compute_errors(hyp: &mut Segmentation, reference: Option<&Segmentation>, nb_words: i64) -> ScoreReport`
    (`ScoreReport { per_class: [ErrorStats;23], wer: Option<WerStats> }`, `ErrorStats { pmiss, pfa, error_rate }`)
- Python differential oracle (`src/python/speech/scoring.py` + a `segmenter` helper): numpy ports of
  `update_segmentation` and `compute_errors` Pass 2 used only in tests to cross-check Rust == Python.

## 7. Golden fixtures, strategy, acceptance

**Vendored into `tests/reference_data/phase0bii/` (committed):**
- A few real VRCTS XML: `PKCTS_ITA_IT_0000259637_02_chan_1.xml` (1 segment), an empty-list one, and a
  multi-segment one (`PRCTS_CHI_BE_0000263500_03.ed.xml`, 16 segments) - for the byte-exact round-trip.
- The real `PRCTS_RUS_RU_0000263489_01.stm` (2-channel) - for the STM loader.
- The config goldens (`24-Feb-2014_BLSTM_Spect.config`, `LID_BLSTM.config`) for the segmenter params.

**Tests:**
1. VRCTS round-trip (THE golden): `load_vrcts(real_xml)` -> `write_vrcts` == the input bytes, for the
   populated, empty, and 16-segment fixtures.
2. STM parse: `load_ref_stm(real_stm, ...)` yields the expected SPEECH intervals per channel.
3. `update_segmentation`: synthetic results rows with hand-computed expected segments (a clean
   rising+falling crossing; an area-gated non-trigger; the tail path); assert the produced
   `Segmentation`. Cross-check Rust == the numpy oracle on random results rows.
4. `lid_to_segmentation`: a col-vector single-threshold case (orientation asymmetry).
5. `smooth_segmentation`: a hand-built noisy segmentation through the 8-step pipeline with the golden
   config params.
6. `compute_errors` Pass 2: a hand-built ref+hyp pair with hand-computed Pfa/Pmiss/ErrorRate; cross-check
   Rust == the numpy oracle. Pass 1/3 (WER) unit-tested from hand-built CSV-loaded deques.
7. `get_targets`: BackPropWER-off and -on cases.

## 8. Risks and pitfalls (pinned by tests / logged)

- **Orientation asymmetry**: `update_segmentation` reads `results(0,ii)` (row), `lid_to_segmentation`
  reads `results(ii,0)` (col). Collapsing to 1D loses it.
- **`update_segmentation` tail `results.size()` vs `length`** (a legacy quirk - add to IMPROVEMENTS.md
  as `[0b-ii]`, confirmed reproduced).
- **Exact crossing/area formulas** and the `>=`/`<`/`<=`/`>` directions; the re-run-rising-after-label.
- **VRCTS decimal formatting**: `sigdur`/`spdur`/`sconf` 2-decimal, `stime`/`etime` 3-decimal; SPEECH-only;
  trailing newline; `_chan_N` filename. Byte-exact vs the real fixtures.
- **`compute_errors` pointer walk**: `it` starts at `begin()+1` (Pass 2), `(it-1)`/`(itRef-1)` neighbor
  lookups, the SPEECH-vs-SUBSTITUTION exemption, the zero-denominator guards. Off-by-one silently
  corrupts Pfa/Pmiss.
- **`get_targets`/Pass 1/3 have no real golden** (no `.csv`) - unit-tested from hand-built deques only.

## 9. Definition of done

- `cargo build --release`, `clippy --all-targets -- -D warnings`, `cargo test`, `fmt --check` green.
- Python oracle: `ruff`/`mypy` clean; `pytest` green.
- VRCTS round-trip byte-exact for the 3 fixtures; STM parse correct; segmenter + Pass 2 unit tests +
  Rust==Python oracle agree; `get_targets`/WER unit tests pass.
- `tasks/segmenter.rs` and `tasks/segmentation_io.rs` no longer stubs; the scaffold `Segment` in
  `segmentation_io.rs` is replaced by the 0b-i container's `Segment`.
- CLAUDE.md / README updated (segmenter + io + scoring implemented; 0b-ii done); IMPROVEMENTS.md gains
  the `update_segmentation` `results.size()` quirk (moved out of Forward-noted to `[0b-ii]`).
- Final step (in the plan): invoke `smart-commit` over the branch.

## 10. Out of scope

The WER/coverage/delay real golden (needs a `.csv`); `io::matfile` / MAT5 (compute_errors output to
`MultiConfigResults.mat` is a diagnostic, deferred); the audio front-end, feature extraction, the NN
forward/backward pass, and the end-to-end inference-vs-VRCTS parity milestone (later phases).
