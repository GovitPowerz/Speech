# Phase 0b-ii Segmenter + I/O + Scoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the legacy segmenter decision logic (`Segmenter.cpp`), the segmentation I/O (VRCTS XML + STM/CSV loaders), and `compute_errors` scoring to Rust, on top of the 0b-i `Segmentation` container.

**Architecture:** Rust-only production (`src/rust/src/tasks/segmenter.rs`, `segmentation_io.rs`). The VRCTS XML round-trip is a byte-exact real-artifact golden; the intricate pointer-walks (`update_segmentation`, `compute_errors` Pass 2) get hand-computed unit tests PLUS a numpy differential oracle (`src/python/speech/scoring.py`) whose Python-emitted fixtures the Rust reproduces (cross-language agreement). Builds on 0b-i (`tasks::segmentation`) and 0a (`legacy_config`).

**Tech Stack:** Rust (edition 2024), Python (numpy oracle), the 0b-i `Segmentation` container, the 0a `legacy_config` parser + `io::binary` codec.

## Global Constraints

- Branch: `feature/phase-0b-ii-segmenter-scoring` (stacked on 0b-i; do NOT rebase onto main). NEVER commit to `main`. Commit trailer `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- IEEE-754 `f64`. Preserve exact comparison directions (`>=`/`<`/`<=`/`>`).
- **Orientation asymmetry:** `update_segmentation` reads a ROW vector `results(0,ii)` (`length=cols`); `lid_to_segmentation` reads a COL vector `results(ii,0)` (`length=rows`). Do not collapse.
- **`update_segmentation` tail uses `results.size()` (rows*cols), not `length`** - reproduce the quirk; add it to IMPROVEMENTS.md as `[0b-ii]` (move it out of Forward-noted).
- VRCTS writer: `sigdur`/`spdur`/`dur`/`sconf` 2-decimal; `stime`/`etime` 3-decimal; SPEECH-only `SpeechSegment` lines; byte-exact vs the real fixtures incl. the trailing newline.
- `compute_errors` Pass 2: `it` starts at `begin()+1`, `itRef` at `begin`; the `(it-1)`/`(itRef-1)` neighbor lookups + the SPEECH-vs-SUBSTITUTION exemption + zero-denominator guards are load-bearing.
- WER Pass 1/3: unit-test-only from hand-built deques (no `.csv` in the tree).
- Rust `clippy --all-targets -- -D warnings` + `fmt --check`; Python `ruff` (160/py314) + `mypy` clean. ASCII only; no em-dashes/en-dashes/smart quotes.
- Fixtures in `tests/reference_data/phase0bii/` (committed). Legacy source (git-ignored): `/Users/govit/Git/Govit/FastSpeechProcessing-legacy/src/{Segmenter,Segmentation,VRCTSpart}.{cpp,h}`, `XmlPart/*.xml`, `PRCTS_RUS_RU_0000263489_01.stm`, `ConfigFiles/*.config`.
- Spec: `docs/superpowers/specs/2026-07-01-phase-0b-ii-segmenter-scoring-design.md` (sections referenced per task).
- Legacy Quirks backlog lives in `IMPROVEMENTS.md` (CLAUDE.md only points to it).

---

### Task 1: Fixtures (VRCTS XML + STM + configs)

**Files:**
- Create (copied, committed): `tests/reference_data/phase0bii/{vrcts_1seg.xml, vrcts_empty.xml, vrcts_16seg.xml, ref.stm, seg.config, lid.config}`
- Test: `tests/test_phase0bii_fixtures.py`

- [ ] **Step 1: Vendor the fixtures**

```bash
mkdir -p tests/reference_data/phase0bii
L=/Users/govit/Git/Govit/FastSpeechProcessing-legacy
cp "$L/XmlPart/PKCTS_ITA_IT_0000259637_02_chan_1.xml" tests/reference_data/phase0bii/vrcts_1seg.xml
cp "$L/XmlPart/PKCTS04_FRE_FR_0000023129_07_chan_1.xml" tests/reference_data/phase0bii/vrcts_empty.xml
cp "$L/XmlPart/PRCTS_CHI_BE_0000263500_03.ed.xml" tests/reference_data/phase0bii/vrcts_16seg.xml
cp "$L/PRCTS_RUS_RU_0000263489_01.stm" tests/reference_data/phase0bii/ref.stm
cp "$L/ConfigFiles/24-Feb-2014_BLSTM_Spect.config" tests/reference_data/phase0bii/seg.config
cp "$L/ConfigFiles/LID_BLSTM.config" tests/reference_data/phase0bii/lid.config
```
If `vrcts_empty.xml` (an empty `<SegmentList>`) is not at that exact name, pick any `XmlPart/*.xml` whose `<SegmentList>` has zero `SpeechSegment` lines (grep for one: `for f in "$L"/XmlPart/*.xml; do grep -qL SpeechSegment "$f" && echo "$f"; done | head`). Note the chosen file.

- [ ] **Step 2: Write the fixture presence test** `tests/test_phase0bii_fixtures.py`

```python
"""Assert the Phase-0b-ii fixtures are present and shaped as expected."""

from pathlib import Path

REF = Path("tests/reference_data/phase0bii")


def test_vrcts_fixtures_present() -> None:
    assert (REF / "vrcts_1seg.xml").read_text().count("<SpeechSegment") == 1
    assert (REF / "vrcts_empty.xml").read_text().count("<SpeechSegment") == 0
    assert (REF / "vrcts_16seg.xml").read_text().count("<SpeechSegment") == 16


def test_stm_and_configs_present() -> None:
    assert (REF / "ref.stm").read_text().startswith(";;")
    assert "BLSTM_decision_thresh_rising" in (REF / "seg.config").read_text()
```

- [ ] **Step 3: Run + commit**

```bash
uv run pytest tests/test_phase0bii_fixtures.py -q
git add tests/reference_data/phase0bii tests/test_phase0bii_fixtures.py
git commit -m "feat(phase0b-ii): vendor VRCTS/STM/config golden fixtures

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: SegmenterConfig + update_segmentation (the crux)

**Files:**
- Modify: `src/rust/src/tasks/segmenter.rs` (replace the stub)
- Test: `src/rust/tests/phase0bii_segmenter.rs`

**Interfaces:**
- Consumes: `speech::tasks::segmentation::{Segmentation, SegClass}` (0b-i), `speech::legacy_config::parse_legacy_config` (0a).
- Produces: `speech::tasks::segmenter::{SegmenterConfig, update_segmentation, smooth_segmentation}`. `SegmenterConfig { rising: f64, area_rising: f64, falling: f64, area_falling: f64, padding: [f64;4], min_speech: [f64;3], min_silence: [f64;2] }` with `from_config(&IndexMap<String,String>, prefix: &str) -> anyhow::Result<SegmenterConfig>`. `update_segmentation(seg: &mut Segmentation, results_row: &[f64], class: SegClass, off: f64, dt: f64, cfg: &SegmenterConfig)` runs the hysteresis-with-area decision and then `smooth_segmentation`.

- [ ] **Step 1: Write failing tests** `src/rust/tests/phase0bii_segmenter.rs`

```rust
//! Segmenter decision tests (hand-computed from synthetic results rows).

use speech::tasks::segmentation::{SegClass, Segmentation};
use speech::tasks::segmenter::{update_segmentation, SegmenterConfig};

fn cfg_no_smooth() -> SegmenterConfig {
    // area 0 so a single crossing triggers; padding/min all 0 so smoothing is a no-op.
    SegmenterConfig {
        rising: 0.5,
        area_rising: 0.0,
        falling: 0.5,
        area_falling: 0.0,
        padding: [0.0; 4],
        min_speech: [0.0; 3],
        min_silence: [0.0; 2],
    }
}

fn spans(s: &Segmentation) -> Vec<(f64, SegClass)> {
    s.segments().iter().map(|x| (x.begin, x.ty)).collect()
}

#[test]
fn clean_rising_falling_crossing() {
    // dt=1.0, offset=0. results cross up between idx1(0.0)->idx2(1.0) at 0.5*dt,
    // and back down between idx3(1.0)->idx4(0.0) at 0.5*dt past idx3. area_rising/falling=0.
    let mut seg = Segmentation::new(5.0);
    let results = vec![0.0, 0.0, 1.0, 1.0, 0.0];
    update_segmentation(&mut seg, &results, SegClass::Speech, 0.0, 1.0, &cfg_no_smooth());
    // rising: r(2)>=0.5 && r(1)<0.5 -> begin = 1*(2 - (1.0-0.5)/(1.0-0.0)) = 1.5
    // falling: r(4)<=0.5 && r(3)>0.5 -> end = 1*(4 - (0.0-0.5)/(0.0-1.0)) = 3.5
    let sp = spans(&seg);
    assert_eq!(sp[0], (0.0, SegClass::Other));
    assert_eq!(sp[1], (1.5, SegClass::Speech));
    assert_eq!(sp[2], (3.5, SegClass::Other));
    assert_eq!(*sp.last().unwrap(), (5.0, SegClass::End));
}

#[test]
fn config_parses_seg_golden() {
    let text = std::fs::read_to_string(
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/reference_data/phase0bii/seg.config"),
    ).unwrap();
    let m = speech::legacy_config::parse_legacy_config(&text);
    let cfg = SegmenterConfig::from_config(&m, "BLSTM").unwrap();
    // 24-Feb-2014_BLSTM_Spect.config, last-duplicate-key wins: rising 0.6/area 0.05, falling 0.3/area 0.03.
    assert_eq!(cfg.rising, 0.6);
    assert_eq!(cfg.area_rising, 0.05);
    assert_eq!(cfg.falling, 0.3);
    assert_eq!(cfg.area_falling, 0.03);
    assert_eq!(cfg.padding, [0.2, 0.0, 0.3, 0.4]);
    assert_eq!(cfg.min_silence, [0.3, 0.5]);
    assert_eq!(cfg.min_speech, [0.0, 0.3, 0.4]);
}
```

- [ ] **Step 2: Run, expect failure.** `cd src/rust && cargo test --test phase0bii_segmenter` -> FAIL.

- [ ] **Step 3: Read the source, then implement `SegmenterConfig` + `update_segmentation` in `segmenter.rs`.** Read `Segmenter.cpp:85-138` (`buildFromConf`: the `if(falling>rising)falling=rising` clamp; `_PaddingSpeech`/`_MinSpeech`/`_MinSilence` require length 4/3/2 else `exit(1)`, each negative clamped to 0) and `:725-845` (`updateSegmentation`). Transcribe design spec sections 3.1-3.2 EXACTLY into Rust: `from_config` parses `{prefix}_decision_thresh_rising`/`_decision_area_rising`/`_decision_thresh_falling`/`_decision_area_falling`, `{prefix}_speech_padding` (4, error if <4), `{prefix}_min_speech` (3), `{prefix}_min_silence` (2), applying the `falling>rising` clamp and negative->0. `update_segmentation` implements the exact RISING/FALLING/hasEnded state machine with the linear-interp `begin`/`end` and area accumulators from spec 3.2, the reset-all-6 + re-run-rising-at-ii on a completed segment, and the TAIL using `results.len()` for the row vector (equivalent to `results.size()` here since it is a row vector) - then calls `smooth_segmentation`. Use `seg.label_segment(begin+off, end+off, class)` (0b-i). Show the full implementation. The unit test's hand-computed `begin=1.5`/`end=3.5` is the arbiter for the crossing math; if it fails, re-read `Segmenter.cpp:725-845` and reconcile the exact formula.

- [ ] **Step 4: Implement `smooth_segmentation`** (spec 3.4, `Segmenter.cpp:709-723`) - the exact 8-step pipeline over the 0b-i container ops:

```rust
/// The fixed 8-step smoothing pipeline (Segmenter.cpp:709-723).
pub fn smooth_segmentation(seg: &mut Segmentation, cfg: &SegmenterConfig) {
    seg.sanitize();
    seg.suppress_short(cfg.min_speech[0], SegClass::Speech);
    seg.add_padding(cfg.padding[0], cfg.padding[1], SegClass::Speech);
    seg.suppress_short(cfg.min_silence[0], SegClass::Other);
    seg.suppress_short(cfg.min_speech[1], SegClass::Speech);
    seg.add_padding(cfg.padding[2], cfg.padding[3], SegClass::Speech);
    seg.suppress_short(cfg.min_silence[1], SegClass::Other);
    seg.suppress_short(cfg.min_speech[2], SegClass::Speech);
}
```

- [ ] **Step 5: Run, expect pass.** `cd src/rust && cargo test --test phase0bii_segmenter` -> both tests PASS.

- [ ] **Step 6: Lint + commit**

```bash
cd src/rust && cargo clippy --all-targets -- -D warnings && cargo fmt --all && cargo test --test phase0bii_segmenter && cd ../..
git add src/rust/src/tasks/segmenter.rs src/rust/tests/phase0bii_segmenter.rs
git commit -m "feat(phase0b-ii): SegmenterConfig + update_segmentation + smooth_segmentation

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Python oracle for update_segmentation + cross-language fixtures

Cross-check the Rust `update_segmentation` against an independent numpy port on random inputs (the differential oracle that caught the 0b-i softmax bug).

**Files:**
- Modify: `src/python/speech/scoring.py` (add the numpy oracle)
- Create: `scripts/extract_phase0bii_segmenter_fixtures.py`
- Create (generated, committed): `tests/reference_data/phase0bii/update_seg_cases.json`
- Test: `tests/test_segmenter_oracle.py`, `src/rust/tests/phase0bii_segmenter.rs` (extend)

**Interfaces:**
- Produces (Python): `speech.scoring.update_segmentation_oracle(results_row: list[float], rising, area_rising, falling, area_falling, dt, off) -> list[tuple[float,int]]` returning the RAW (pre-smoothing) segment boundary list as `(begin, seg_class_code)`.

- [ ] **Step 1: Implement the numpy oracle** in `src/python/speech/scoring.py` (a faithful port of spec 3.2, NO smoothing - raw boundaries only, so it isolates the decision math). Include a Python unit test `tests/test_segmenter_oracle.py` asserting the same `clean_rising_falling_crossing` hand-computed case (`begin=1.5`, `end=3.5`) as Task 2. Show the full port + test.

- [ ] **Step 2: Write the fixture-emitter** `scripts/extract_phase0bii_segmenter_fixtures.py`: generate DETERMINISTIC pseudo-random results rows (seed the sequence explicitly - use a fixed list of, e.g., 20 rows of 50 values built from a deterministic formula like `((i*7+j*13) % 100)/100.0`, NOT `random`), run the oracle with a couple of (rising, area, falling, area) settings, and write `tests/reference_data/phase0bii/update_seg_cases.json` = a list of `{"row": [...], "params": {...}, "expected": [[begin, code], ...]}`. Show the full script.

- [ ] **Step 3: Run the emitter.** `uv run python scripts/extract_phase0bii_segmenter_fixtures.py` -> writes the cases JSON. Verify it is valid JSON with >= 20 cases.

- [ ] **Step 4: Add the Rust cross-check** to `phase0bii_segmenter.rs`: load `update_seg_cases.json` (serde_json), and for each case run the Rust decision (a `update_segmentation_raw` helper WITHOUT smoothing, exposed for testing, OR run `update_segmentation` with a smooth-disabling config of all-zero padding/min so smoothing is a no-op) and assert the produced boundaries `== case.expected` bit-for-bit. This pins Rust == Python. Show the test.

- [ ] **Step 5: Run + lint + commit.** `cargo test --test phase0bii_segmenter` (incl. the cross-check) + `uv run pytest tests/test_segmenter_oracle.py` + ruff/mypy. Commit staging `src/python/speech/scoring.py scripts/extract_phase0bii_segmenter_fixtures.py tests/reference_data/phase0bii/update_seg_cases.json tests/test_segmenter_oracle.py src/rust/tests/phase0bii_segmenter.rs` with the standard trailer.

---

### Task 4: lid_to_segmentation + get_targets

**Files:**
- Modify: `src/rust/src/tasks/segmenter.rs`
- Test: `src/rust/tests/phase0bii_segmenter.rs` (extend)

**Interfaces:**
- Produces: `lid_to_segmentation(seg: &mut Segmentation, results_col: &[f64], class: SegClass, off: f64, dt: f64, thresh_max: f64)` (single-threshold, col-vector, `sanitize` only - NO smoothing); `get_targets(seg: &Segmentation, reference: &Segmentation, time_step: f64, time_offset: f64, back_prop_wer: f64, class: SegClass) -> Vec<f64>`.

- [ ] **Step 1: Write failing tests** for `lid_to_segmentation` (a col-vector single-threshold crossing, hand-computed begin/end, `sanitize` only) and `get_targets` (the `back_prop_wer < 0` plain branch: a reference with a SPEECH span -> targets 1.0 inside it, 0.0 outside, -0.5 for EXCLUDED). Show the tests with hand-computed expected `Vec<f64>`.

- [ ] **Step 2: Run, expect failure.**

- [ ] **Step 3: Implement `lid_to_segmentation`** (spec 3.3, `Segmenter.cpp:999-1111`) - col-vector `results[ii]` treated as `results(ii,0)`, single `thresh_max`, linear-interp begin/end, `sanitize` only. And **`get_targets`** (spec 3.5, `:659-707`): the `back_prop_wer < 0` plain branch fully (`itRef.ty == class || (class==SPEECH && itRef.ty==SUBSTITUTION)` -> 1.0; EXCLUDED -> -0.5; else 0.0), and the `back_prop_wer >= 0` branch per the source (SPEECH/SUBST -> `1 - 0.1*timeStep/max(next-cur,timeStep)`, EXCLUDED -> -0.5, INSERTION -> `0.1*timeStep/durSeg`, else neighbor-window / `timeStep/100`). Read `Segmenter.cpp:659-707` for the exact neighbor-window conditions. Show the full implementation.

- [ ] **Step 4: Run, expect pass. Step 5: lint + commit** (message `feat(phase0b-ii): lid_to_segmentation + get_targets`, standard trailer).

---

### Task 5: VRCTS XML write + parse (byte-exact golden)

**Files:**
- Modify: `src/rust/src/tasks/segmentation_io.rs` (replace the scaffold stub `Segment`; use the 0b-i container's `Segment`)
- Test: `src/rust/tests/phase0bii_io.rs`

**Interfaces:**
- Consumes: `speech::tasks::segmentation::{Segmentation, Segment, SegClass}`.
- Produces: `speech::tasks::segmentation_io::{write_vrcts(seg: &Segmentation, name: &str, path_attr: &str, out: &Path) -> anyhow::Result<()>, load_vrcts(text: &str, off: f64, dur: f64) -> Segmentation}`. Also a `to_vrcts_string(seg, name, path_attr) -> String` used by `write_vrcts` and testable without a file.

- [ ] **Step 1: Write the round-trip golden test** `src/rust/tests/phase0bii_io.rs`

```rust
//! VRCTS XML byte-exact round-trip vs real fixtures.

use std::path::PathBuf;

fn ref_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/reference_data/phase0bii")
}

fn roundtrip(fixture: &str) {
    let raw = std::fs::read_to_string(ref_dir().join(fixture)).unwrap();
    // extract the AudioDoc name + path attributes from the real file to reproduce the header
    let (name, path_attr) = speech::tasks::segmentation_io::parse_audiodoc_attrs(&raw);
    // Load into a Segmentation. The real files use sigdur as the audio duration.
    let seg = speech::tasks::segmentation_io::load_vrcts(&raw, 0.0, 1.0e9);
    let out = speech::tasks::segmentation_io::to_vrcts_string(&seg, &name, &path_attr);
    assert_eq!(out, raw, "round-trip mismatch for {fixture}");
}

#[test]
fn vrcts_roundtrip_1seg() { roundtrip("vrcts_1seg.xml"); }
#[test]
fn vrcts_roundtrip_empty() { roundtrip("vrcts_empty.xml"); }
#[test]
fn vrcts_roundtrip_16seg() { roundtrip("vrcts_16seg.xml"); }
```

Note: `load_vrcts` must also record the channel's `sigdur` (audio_duration) and `spdur` so the writer can reproduce them; add a `parse_audiodoc_attrs(text) -> (String, String)` helper returning the `name=` and `path=` attribute values verbatim.

- [ ] **Step 2: Run, expect failure.**

- [ ] **Step 3: Implement `to_vrcts_string`/`write_vrcts`/`load_vrcts`/`parse_audiodoc_attrs`** in `segmentation_io.rs` per spec 4.1, matching the exact fixture bytes. Read a fixture with `xxd` to confirm the exact line terminators, the `%.2f`/`%.3f` formatting, the `spdur` = sum of SPEECH durations (2-decimal), the `_chan_N` handling (the writer takes `name`/`path_attr` verbatim). The parser accepts a `SpeechSegment` iff `end >= off && begin < off + dur`. Byte-exact is the arbiter; if a decimal or whitespace differs, match the fixture. Show the full implementation. IMPORTANT: reproduce whatever rounding the legacy printf uses at the 3rd/4th decimal by matching the fixture values (they were produced by the legacy writer); if a `%.3f` in Rust differs from the fixture at a `x.xxxx5` boundary, note it and match the fixture.

- [ ] **Step 4: Run, expect pass (all 3 round-trips byte-exact). Step 5: lint + commit** (`feat(phase0b-ii): VRCTS XML write/parse byte-exact round-trip`, trailer).

---

### Task 6: STM + CSV loaders + ASCII writer

**Files:**
- Modify: `src/rust/src/tasks/segmentation_io.rs`
- Test: `src/rust/tests/phase0bii_io.rs` (extend)

**Interfaces:**
- Produces: `load_ref_stm(text: &str, chan: usize, off: f64, dur: f64, exclude_nontrans: bool) -> Segmentation`; `load_ref_csv(text: &str, off: f64, dur: f64) -> (Segmentation, i64)` (the i64 is `nb_words`, `>= 0`); `write_ascii(seg: &Segmentation, out: &Path) -> anyhow::Result<()>` + `to_ascii_string(seg) -> String`.

- [ ] **Step 1: Write tests** - `load_ref_stm` on `ref.stm` yields the expected SPEECH intervals for channel 1 (read the real file, pick a couple of known lines, assert the parsed spans); `load_ref_csv` on a hand-built CSV string (`"0,1.0,S,0.9\n1.0,2.0,I,0.5\n"` style - confirm the field order from `Segmentation.cpp`) sets `nb_words` and the intervals; `to_ascii_string` on a small segmentation. Show the tests.

- [ ] **Step 2: Run, expect failure.**

- [ ] **Step 3: Implement `load_ref_stm`/`load_ref_csv`/`write_ascii`** per spec 4.2-4.3. Read `Segmentation.cpp` for the STM 7-token parse (`;;` header skip; SPEECH iff `second.startswith(first)`; `excluded_region` -> EXCLUDED/OTHER gated on `exclude_nontrans`) and the CSV parse (comma->space, 4 fields, `end -= 1e-4`, `nb_words++` for `type != 'I'`, `nb_words >= 0`). Show the full implementation.

- [ ] **Step 4: Run, expect pass. Step 5: lint + commit** (`feat(phase0b-ii): STM/CSV loaders + ASCII writer`, trailer).

---

### Task 7: compute_errors (Pass 2 golden-able + Pass 1/3 unit)

**Files:**
- Modify: `src/rust/src/tasks/segmentation_io.rs`
- Test: `src/rust/tests/phase0bii_scoring.rs`

**Interfaces:**
- Produces: `speech::tasks::segmentation_io::{ErrorStats { pmiss: f64, pfa: f64, error_rate: f64 }, ScoreReport { per_class: [ErrorStats; 23], wer: Option<WerStats> }, compute_errors(hyp: &mut Segmentation, reference: Option<&Segmentation>, nb_words: i64) -> ScoreReport}`. `WerStats { nb_words: i64, corrects: i64, subs: i64, dels: i64, ins: i64 }`.

- [ ] **Step 1: Write hand-computed Pass 2 tests** in `src/rust/tests/phase0bii_scoring.rs`: build a ref (`[Other@0, Speech@2, Other@5, End@10]`) and a hyp (`[Other@0, Speech@3, Other@6, End@10]`), and assert the resulting `per_class[SPEECH].pmiss`/`pfa`/`error_rate` against hand-computed seconds/normalization (compute by hand from spec 5, show the arithmetic in comments). Also a no-reference test (label percentages). Show the tests.

- [ ] **Step 2: Run, expect failure.**

- [ ] **Step 3: Implement `compute_errors`** per spec 5 (`Segmentation.cpp:288-476`). Pass 2 first: `sanitize` hyp; `modify_type` ref SUBSTITUTION->SPEECH + INSERTION->OTHER (on a CLONE of the reference so the caller's ref is untouched, OR document that ref is consumed); `update_count` ref; the two-pointer walk (`it` from index 1, `itRef` from 0) with the exact scorable predicate + exemption + the `error = min-overlap` expressions + accumulation to `pmiss[ref]`/`pfa[hyp]`/`error_rate[hyp]`; then the normalization (`/count`, `/count_others`, `/(dur-excluded)` with the zero guards). Pass 1/3 (WER) ONLY when `nb_words >= 0`: implement the two-pointer WER/coverage/delay from `:313-405` (unit-tested only). Read the source for the exact pointer arithmetic. The hand-computed unit test is the arbiter for Pass 2. Show the full implementation.

- [ ] **Step 4: Run, expect pass. Step 5: lint + commit** (`feat(phase0b-ii): compute_errors Pass 2 (Pfa/Pmiss/ErrorRate) + WER`, trailer).

---

### Task 8: Python oracle for compute_errors Pass 2 + cross-check

**Files:**
- Modify: `src/python/speech/scoring.py`
- Create: `tests/reference_data/phase0bii/compute_errors_cases.json` (generated, committed)
- Modify: `scripts/extract_phase0bii_segmenter_fixtures.py` (or a second emitter) + `src/rust/tests/phase0bii_scoring.rs`, `tests/test_scoring_oracle.py`

**Interfaces:**
- Produces (Python): `speech.scoring.compute_errors_pass2_oracle(ref: list[tuple[float,int]], hyp: list[tuple[float,int]], audio_duration: float) -> dict` returning per-class pmiss/pfa/error_rate.

- [ ] **Step 1: Implement the numpy Pass-2 oracle** in `scoring.py` (independent port of spec 5 Pass 2) with a Python unit test asserting the same hand-computed ref/hyp case as Task 7. Show it.

- [ ] **Step 2: Emit deterministic cross-language cases** to `compute_errors_cases.json`: several deterministic ref/hyp boundary-list pairs + audio_duration, with the oracle's expected per-class pmiss/pfa/error_rate. Show the emitter additions.

- [ ] **Step 3: Rust cross-check** in `phase0bii_scoring.rs`: load the cases, run Rust `compute_errors`, assert `per_class` == the oracle's expected (bit-exact `f64`). Pins Rust == Python for the pointer walk.

- [ ] **Step 4: Run + lint + commit** (`feat(phase0b-ii): compute_errors Pass 2 numpy oracle + cross-language check`, trailer).

---

### Task 9: Docs + IMPROVEMENTS.md quirk + full verification + smart-commit

**Files:**
- Modify: `CLAUDE.md`, `README.md`, `IMPROVEMENTS.md`

- [ ] **Step 1: CLAUDE.md** - change the `tasks/segmenter.rs` and `tasks/segmentation_io.rs` architecture rows to "Implemented (Phase 0b-ii): <one line each>".
- [ ] **Step 2: IMPROVEMENTS.md** - MOVE the `[0b-ii] updateSegmentation results.size()` bullet out of "Forward-noted" into the live Legacy Quirks list, retagged `[0b-ii]`, pointing at `segmenter.rs` (`update_segmentation` tail, from `Segmenter.cpp:833`), confirmed reproduced (the tail uses `results.len()` for the row vector). Leave the `[3/4] CostFunctionCalib` forward-note.
- [ ] **Step 3: README.md** - mark Phase 0b-ii done: VRCTS XML byte-exact round-trip golden; segmenter + compute_errors Pass 2 unit-tested + numpy differential oracle (Rust==Python); WER unit-test-only. Note this CLOSES the Phase 0/0b pure-logic + I/O parity layer; next is Phase 1 (features). ASCII only (`LC_ALL=C grep -n '[^ -~]' CLAUDE.md README.md IMPROVEMENTS.md` returns nothing).
- [ ] **Step 4: Full verification** - `./check_all.sh` (green), `uv run ruff check src/python tests scripts`, `uv run mypy --config-file pyproject.toml src/python tests`, `uv run pytest tests -q` (all green).
- [ ] **Step 5: Commit** the docs (`docs(phase0b-ii): mark segmenter/io/scoring implemented; promote results.size() quirk`, trailer), then **invoke the `smart-commit` skill** over the whole `feature/phase-0b-ii-segmenter-scoring` branch. Do not push.

---

## Self-Review

**1. Spec coverage.** Spec s3.1 (config) -> Task 2 `SegmenterConfig`. s3.2 (update_segmentation) -> Task 2 + Task 3 oracle. s3.3 (lid) -> Task 4. s3.4 (smooth) -> Task 2 Step 4. s3.5 (get_targets) -> Task 4. s4.1 (VRCTS) -> Task 5. s4.2-4.3 (STM/CSV/ASCII) -> Task 6. s5 (compute_errors Pass 2 + Pass 1/3) -> Task 7 + Task 8 oracle. s6 (interfaces) -> task signatures. s7 (fixtures/golden): fixtures -> Task 1; VRCTS golden -> Task 5; segmenter/scoring unit+oracle -> Tasks 2/3/7/8. s8 risks pinned: orientation (Tasks 2/4), tail results.size() (Task 2 + IMPROVEMENTS Task 9), crossing formulas (Task 2 test), VRCTS decimals (Task 5 byte-exact), pointer walk (Task 7 test + Task 8 oracle). s9 DoD -> Task 9. s10 out-of-scope respected (no WER golden, matfile, NN). Covered.

**2. Placeholder scan.** No TBD/TODO. Tasks 2/4/5/6/7 Step 3 describe the implementation via the exact spec section + legacy line refs + a shown hand-computed test/golden as the arbiter, rather than pre-writing 60-line intricate ports that the tests would re-derive anyway; the tractable pieces (`smooth_segmentation`, the VRCTS test, config asserts, the round-trip harness) are shown in full. The oracle tasks (3, 8) show the numpy port is required and gated by cross-language fixtures. This matches the RE-with-golden pattern used successfully in 0a/0b-i.

**3. Type consistency.** `SegmenterConfig` fields (`rising`/`area_rising`/`falling`/`area_falling`/`padding`/`min_speech`/`min_silence`) consistent across Tasks 2-4 and tests. `update_segmentation`/`lid_to_segmentation`/`smooth_segmentation`/`get_targets` signatures consistent (Tasks 2-4). `segmentation_io::{to_vrcts_string, write_vrcts, load_vrcts, parse_audiodoc_attrs, load_ref_stm, load_ref_csv, write_ascii, to_ascii_string, compute_errors, ErrorStats, ScoreReport, WerStats}` consistent (Tasks 5-8). `SegClass`/`Segmentation`/`Segment` consumed from 0b-i `tasks::segmentation`. `parse_legacy_config`/`io::binary` from 0a. Python `scoring.{update_segmentation_oracle, compute_errors_pass2_oracle}` consistent (Tasks 3, 8). Fixture names (`vrcts_1seg/empty/16seg.xml`, `ref.stm`, `seg.config`, `lid.config`, `update_seg_cases.json`, `compute_errors_cases.json`) consistent across the tasks that produce and consume them.
