# Issue #23 -- Typed channel result: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps
> use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One Rust type owns the 18+N result-row layout. The bag builds a `ChannelResult`
per (file, config, channel); `to_row()` is the only place the row is written; every Rust
reader (`save_and_update`, `grad_check_cost`, `confusion_from_results`) reads fields; the
seam exposes names through `Engine.channel_results()`; no `[:, 4]`-style index remains under
`src/python/speech/`. The `.mat` bytes, the parity gates and every ledger number are
unchanged, and the committed suite byte-green is the proof.

**Source of truth:** the grilling outcome recorded on the issue
(https://github.com/GovitPowerz/Speech/issues/23, comment of 2026-10-07, decisions D1-D13).
Where this plan and that comment disagree, the comment wins; where either disagrees with the
tree, read the tree and say so in the task report. The vocabulary is CONTEXT.md's: *Channel
result* (the typed value), *Result row* (its wire form). Never "result record" (a Record is a
ledger entry).

**Architecture:** `engine/channel_result.rs` holds `ChannelResult`, `LidResult`, the two
constructors (`scored`, `unscored`) that replace `assemble_scored_row` / `assemble_unscored_row`,
`to_row()`, the one in-band decode (`LidResult::from_encoded`), a `test-support`-only
`from_row()` and a round-trip proptest. The bag's `results` maps carry `ChannelResult`
instead of `Vec<f64>`; `transform_results` calls `to_row()` to build `results_e` (the `.mat`
payload and the `results_matrix()` wire view, both kept). The seam adds a columnar
`channel_results()`; `engine.py` wraps it in a frozen `ChannelResults` dataclass. The only
Python layout knowledge left is `tests/_result_rows.py`, the matrix-to-names decoder the
pure-Python CI job needs for the Octave goldens, cross-pinned against Rust in `tests/pyo3`.

**Tech Stack:** Rust (edition 2024, ndarray exact tree), PyO3/`speech_rs` (`numpy` crate),
Python 3.14 + uv, pytest + hypothesis, proptest.

## Global Constraints (every task; copy into every dispatch)

- R6 MEMORY PROTOCOL (two host reboots on record): every cargo invocation `-j 4`; every
  test execution `-- --test-threads=4`; NEVER two compile-heavy commands concurrently (no
  cargo+maturin overlap); targeted single-file test runs while iterating; at most ONE
  full-suite run per task; ALL commands FOREGROUND (never `run_in_background`, never a
  Monitor; never read exit codes through a pipe: `cmd > log 2>&1; echo $?`).
- EXACT-TREE DISCIPLINE (ADR-0002): `engine/` is inside the frozen exact tree. This branch's
  edit is the touch class "the typed channel result" (D11): behaviour-free on every exact-path
  run, the committed golden suite byte-green as the proof, checked before every commit. No
  edit under `nn/`, `features/`, `tasks/`, `fast/`: the in-band encoding stays produced by
  `tasks/lid.rs` / `fast/driver.rs` (D3).
- BIT-IDENTITY, NOT CLOSENESS: every sum in `save_and_update` stays an ascending-row loop in
  the same column order; the decode `v - 200` and the re-encode `+ 200` are exact on
  [200, 300] (Sterbenz) and nothing else about the arithmetic moves. A changed `.mat` byte,
  a changed Octave golden, a changed `phase7_parity_*` / `phase9_fast_parity` number is a
  STOP (ADR-0003), never a re-pin.
- IMPROVEMENTS.md: ONE edit only (Task 6, the mixed-width note). No new quirk, no fix.
- Git: never commit to main; this branch is `feature/typed-channel-result`; never push;
  never use `gh`. Commit trailer EXACTLY:
  `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- ASCII only in all authored text (plain hyphens, straight quotes, no em dashes).
- License hygiene: nothing from `data/LRE03-LRE07/` or `data/Scoring_LRE15/` in any tracked
  file; corpus tests behind `requires_corpus` / `corpus_root_or_skip`.
- Python via `uv run`; lint gates `./lint_code.sh` (ruff + mypy) and
  `cargo fmt --all -- --check` + `cargo clippy -j 4 --all-targets --all-features` (zero
  warnings) per task.

---

### Task 1: `engine/channel_result.rs` and the bag's row site (behaviour-free)

**Files:**
- Create: `src/rust/src/engine/channel_result.rs`
- Modify: `src/rust/src/engine/mod.rs` (module line)
- Modify: `src/rust/src/engine/bag_of_processors.rs` (delete `LidRowData`,
  `assemble_scored_row`, `assemble_unscored_row`, `push_lid_block`; `lid_row_data` returns
  `Option<LidResult>`; `segmentation_function` returns `BTreeMap<usize, BTreeMap<usize,
  ChannelResult>>`; the two row tests at the bottom move to the new module)
- Modify: `src/rust/src/engine/corpus_processor.rs` (`results` map value type;
  `transform_results_impl` calls `to_row()`; the width probe reads
  `.map(|r| r.to_row().len())` so the conf-0/chan-0 quirk is kept verbatim)

**Interfaces:**
```rust
#[derive(Debug, Clone, PartialEq)]
pub struct LidResult {
    pub cost: f64,          // col 14, _LIDCumulativeError
    pub count: i64,         // col len-2, _LIDNbOfClassif (0 in the unscored row)
    pub correct: bool,      // col 15, _IsLIDCorrect: 100.0 / 0.0 on the wire
    pub target: Option<usize>, // the one column > 150; None = out-of-set row
    pub scores: Vec<f64>,   // cols 16..16+N, the target decoded (v - 200)
}
impl LidResult {
    /// The one in-band decode. Ascending scan; the LAST column > 150 is the target
    /// (today's loop semantics); every other column is kept raw.
    pub fn from_encoded(cost: f64, count: i64, correct: bool, encoded: &[f64]) -> Self;
}

#[derive(Debug, Clone, PartialEq)]
pub struct ChannelResult {
    pub pfa: f64, pub pmiss: f64, pub error_rate: f64,   // cols 0-2, already x100
    pub time_per_hour: f64,                                // col 3
    pub seg_cost: f64,                                     // col 4, _CumulativeError
    pub audio_duration: f64, pub speech_duration: f64,     // cols 5, 6
    pub wer: WerStats,                                     // cols 7-13
    pub lid: Option<LidResult>,                            // cols 14, 15, 16..16+N, len-2
    pub seg_count: i64,                                    // col len-1, _NbOfClassif
}
impl ChannelResult {
    pub fn scored(report: &ScoreReport, time_per_hour: f64, seg_cost: f64,
                  audio_duration: f64, speech_duration: f64, seg_count: i64,
                  lid: Option<LidResult>) -> Self;
    /// Zeroed errors, seg cost and both counters; real LID cost/flag/scores with
    /// count 0; `WerStats::legacy_default()`. Holds the `:358-392` asymmetry.
    pub fn unscored(time_per_hour: f64, audio_duration: f64, speech_duration: f64,
                    lid: Option<LidResult>) -> Self;
    /// The result row: 18 columns, 18 + N with LID. The ONLY writer of the layout.
    pub fn to_row(&self) -> Vec<f64>;
    #[cfg(feature = "test-support")]
    pub fn from_row(row: &[f64]) -> Self;   // for the fold tests and the proptest only
}
```

- [ ] **Step 1: Move, do not rewrite.** The module doc is the one place the 18+N layout is
  written down: carry the column table from `assemble_scored_row`'s doc, the unscored
  asymmetry (col 7 `nb_words == -1`, the two hard-0 counters, the identical LID block), and
  the in-band convention (`100*langID + 200` on the target, raw elsewhere; `(v-200)+200 == v`
  bit for bit on [200, 300]). `to_row()` is `assemble_scored_row`'s body reading fields:
  same push order, `correct` written as `100.0`/`0.0`, `count`/`seg_count` as `as f64`.
- [ ] **Step 2: `from_encoded` at the bag.** `Processor::lid_row_data` builds the
  `LidResult` from the drivers' `lid_cumulative_error`, `lid_nb_of_classif`,
  `is_lid_correct()[chan] == 100` and `lid_classification_errors()[chan]`. The unscored
  constructor overrides `count` to 0 (the `:390` hard zero). Nothing in `tasks/` or `fast/`
  changes.
- [ ] **Step 3: Tests in the module.** (a) The two existing `assemble_*` tests
  (`bag_of_processors.rs` tests `plain`/`wide`/`unscored`, ~:1667-1690) become `to_row()`
  tests, same expected vectors. (b) `from_encoded` on `[210.0, 30.0, 60.0]` gives
  `target Some(0)`, `scores [10.0, 30.0, 60.0]`; on `[20.0, 30.0]` gives `None`.
  (c) Proptest: a valid wire row (one target column drawn in [200, 300] or none, the other
  columns in [0, 150], finite f64 elsewhere, `correct` in {0, 100}, N in 0..=8) satisfies
  `from_row(to_row(from_row(row))) == from_row(row)` AND `to_row(from_row(row))` is
  bit-equal to `row`. State in the doc why the second property is the one that matters: the
  wire bytes are golden-pinned, the struct is not.
- [ ] **Step 4: Gates.** `cargo test -j 4 --test phase4a_tier1_e2e --test
  phase4a_train_golden --test phase4a_corpus_processor --test phase4b_corpus_lid --
  --test-threads=4` green; the `.mat` goldens compare byte-for-byte as before. Then
  `cargo fmt`, clippy, ONE full `cargo test -j 4 -- --test-threads=4`.
- [ ] **Step 5: Commit** `refactor(engine): ChannelResult owns the result-row layout`.

---

### Task 2: The Rust readers read fields

**Files:**
- Modify: `src/rust/src/engine/bag_of_processors.rs` (`save_and_update`,
  `print_confusion_matrix`; the `algo_types[ii] == 5 || 6` gate)
- Modify: `src/rust/src/engine/corpus_processor.rs` (`res_per_conf` becomes
  `Vec<Vec<ChannelResult>>`, filled in the same ascending (file, chan) walk;
  `grad_check_cost` reads fields; `save_and_update_epoch` passes the slices)
- Modify: `src/rust/src/engine/confusion.rs` (`confusion_from_results(&[LidResult])`)
- Modify: `src/rust/tests/phase4a_save_update.rs`, `src/rust/tests/phase4b_backlog.rs`
  (their crafted 18-wide rows go through `ChannelResult::from_row`; confirm the
  `test-support` feature is on for integration tests, as `last_confusion_for_test_from_bag`
  already requires)
- Modify: `src/rust/tests/phase4b_confusion_golden.rs` (each fixture matrix row decodes
  through `LidResult::from_encoded(0.0, 0, false, row)`; the expected matrices are
  untouched)

- [ ] **Step 1: `save_and_update(results_per_conf: &[Vec<ChannelResult>], ...)`.** Replace
  the `sums`/`means` vectors by per-field accumulators, each an ascending-row loop in the
  SAME order the column loop produced (rows outer; the per-column sequence is what matters
  and it is unchanged). `cost = sum seg_cost`, guarded by `sum seg_count`; `bad_classif =
  mean error_rate`; `cost_lid = sum lid.cost` (0.0 when None) guarded by `sum lid.count`;
  `bad_lid_classif = 100 - mean(100.0 * correct)`; the WER sums over `wer.*`;
  `total_speech_duration = mean speech_duration * rows`. The confusion block fires when
  `rows.first().is_some_and(|r| r.lid.is_some())` (D8); `print_confusion_matrix` takes
  `&[LidResult]`; the `last_confusion` hook keeps its `(f64, Array2<f64>)` type.
- [ ] **Step 2: `confusion_from_results(rows: &[LidResult])`.** `class_nb =
  rows.first().map_or(0, |r| r.scores.len())`; per row `score_target = scores[target]`,
  `pos_target = target + 1`, the best competitor is the ascending strict-`>` max over the
  other columns (same tie rule); a `None` target is skipped. The fn doc keeps the F5
  per-row-decode history and now says the decode happens once, upstream, in
  `LidResult::from_encoded`.
- [ ] **Step 3: `grad_check_cost`.** Keep `match algo0` verbatim (D8: the per-net pair choice
  is #25's); the arms read `row.seg_cost` / `row.seg_count as f64` and
  `row.lid.as_ref().map_or(0.0, |l| l.cost)` / `.map_or(0.0, |l| l.count as f64)`.
- [ ] **Step 4: Gates.** `cargo test -j 4 --test phase4a_save_update --test phase4b_backlog
  --test phase4b_confusion_golden --test phase4b_corpus_lid --test phase4a_gradcheck_golden
  -- --test-threads=4` green, then the Task 1 golden set again, fmt, clippy, ONE full run.
  `rg -n "\[4\]|\[14\]|cols - 2|len\(\) - 1|len\(\) - 2" src/rust/src/engine/` must return
  only `to_row`/`from_row` in `channel_result.rs`.
- [ ] **Step 5: Commit** `refactor(engine): the fold, gradcheck and confusion read fields`.

---

### Task 3: `tests/_result_rows.py` and the `ChannelResults` dataclass

**Files:**
- Create: `tests/_result_rows.py` (the test-side decoder, pure numpy, no `speech_rs`)
- Modify: `src/python/speech/engine.py` (add `ChannelResults`; nothing else yet)
- Create: `tests/test_result_rows.py`

**Interfaces:**
```python
@dataclass(frozen=True)
class ChannelResults:           # src/python/speech/engine.py
    file: NDArray[np.int64]     # 0-based, matching every `pos` on the seam
    conf: NDArray[np.int64]
    chan: NDArray[np.int64]
    pfa: NDArray[np.float64]; pmiss: ...; error_rate: ...; time_per_hour: ...
    seg_cost: ...; seg_count: ...; audio_duration: ...; speech_duration: ...
    lid_cost: ...; lid_count: ...
    lid_correct: NDArray[np.bool_]
    lid_target: NDArray[np.int64]          # -1 = none
    lid_scores: NDArray[np.float64]        # n x N, N = 0 without LID

    @classmethod
    def from_seam(cls, d: dict[str, NDArray]) -> ChannelResults: ...
    def for_config(self, pos: int) -> ChannelResults:
        """Rows of config `pos`. Raises ValueError on an empty selection: a silent empty
        selection is a silent 0.0 cost (D7)."""
    def __len__(self) -> int: ...

def channel_results_from_matrix(mcr: NDArray[np.float64]) -> ChannelResults:   # tests/_result_rows.py
    """MultiConfigResults (1-based id prefix + result rows) to names. The column table
    from engine.py's old module docstring lives here now."""
```

- [ ] **Step 1: Decoder.** Ids `- 1`; `res = mcr[:, 3:]`; `lid_scores = res[:, 16:-2]`
  (empty on an 18-wide row); `lid_target` = the last column `> 150` per row else -1; decode
  that one column `- 200`; `lid_correct = res[:, 15] == 100.0`; counts to int64.
  An empty `(0, 0)` matrix decodes to zero-length fields with `lid_scores` shape `(0, 0)`.
- [ ] **Step 2: Tests.** `tests/test_result_rows.py`: an 18-wide crafted row, a 20-wide
  (N = 2) row with the target in column 1 (`260.0` -> `60.0`, target `1`), a no-target
  row (`-1`), `for_config` on an absent config raises, `from_seam` round-trips the dict
  the decoder produces (`from_seam(asdict(x)) == x`).
- [ ] **Step 3: Lint** (`./lint_code.sh`), `uv run pytest tests/test_result_rows.py -q`.
- [ ] **Step 4: Commit** `test(results): the matrix-to-names decoder for the Octave goldens`.

---

### Task 4: The seam exposes names, cross-pinned

**Files:**
- Modify: `src/rust/src/engine/corpus_processor.rs` (`pub fn channel_results(&self) ->
  Vec<(usize, usize, usize, &ChannelResult)>`, ascending (file, conf, chan) over
  `self.results`; EMPTY before the first `run()`, like `results_matrix()`)
- Modify: `src/rust/speech-py/src/lib.rs` (`Engine.channel_results(self) -> dict`)
- Create: `tests/pyo3/test_channel_results.py`

- [ ] **Step 1: Seam method.** One `PyDict` with the 16 keys of `ChannelResults`, each a
  fresh numpy array (`into_pyarray`; COPY, like every other crossing). `lid_scores` is
  `n x N` with `N = max scores.len()` over the rows, zero-filled for a `None` row (a
  non-mixed bag never hits the fill); `lid_target` is `-1` for `None`; `lid_correct` is a
  bool array. Doc the contract in the method doc the way `results_matrix` does.
- [ ] **Step 2: Cross-pin.** `tests/pyo3/test_channel_results.py` seeds `tier2_spectral`
  (18 wide, 2 files x 2 chans) and `twin_train` (18 + 7, LID) exactly as
  `test_seam_replay.py` does, runs once, then asserts field by field with
  `np.array_equal` that `channel_results_from_matrix(eng.results_matrix()) ==
  ChannelResults.from_seam(eng.channel_results())` (same run, so the timing column agrees).
  Also pin: `lid_target >= 0` on every `twin_train` row and `lid_scores.shape[1] == 7`;
  `lid_scores.shape[1] == 0` on `tier2_spectral`.
- [ ] **Step 3: Build and run.** `uv run maturin develop --release --manifest-path
  src/rust/speech-py/Cargo.toml` (ALONE, no cargo in parallel), then
  `uv run pytest tests/pyo3/test_channel_results.py tests/pyo3/test_engine_smoke.py -q`.
- [ ] **Step 4: Commit** `feat(seam): Engine.channel_results() names the result row`.

---

### Task 5: Every Python reader moves to names

**Files:**
- Modify: `src/python/speech/engine.py` (`_error_vad`/`_nn_cost_seg`/`_nn_cost_lid` on
  `ChannelResults`; `_balance10` and `_balance10_cutoff` on decoded values; `compute_cost(
  results: ChannelResults, pos: int, ...)`; `forward_backward` reads
  `engine.channel_results()` with a 0-based `config_idx` default; DELETE
  `aggregate_workers`; the module docstring's column table is replaced by two sentences
  pointing at `ChannelResults` and `tests/_result_rows.py`)
- Modify: `src/python/speech/scoring.py` (`confusion_matrix(lid_target, lid_scores,
  thresh)`)
- Modify: `src/python/speech/drivers/train.py` (the three `results_matrix()` sites,
  `_confusion_error`, `validate`'s `_error_vad` use, every `compute_cost(results, 1, ...)`
  becomes `compute_cost(results, 0, ...)`)
- Modify: `src/python/speech/drivers/test.py` (`evaluate` iterates `ChannelResults` rows;
  DELETE `_decode_lid_scores`; `write_scores` takes the decoded row)
- Modify: `tests/test_phase4c_engine_cost.py` (every `_cc("*_mcr")` and the tier-2 / twin
  committed matrices pass through `channel_results_from_matrix`; `config_idx` 1 -> 0;
  DELETE `test_aggregate_workers_*` and the `agg_w1` / `agg_w2` fixtures plus their
  `computecost_manifest.json` entries)
- Modify: `tests/test_phase4c_scoring.py` (the inline encoded arrays become
  `(target, scores)` pairs: `[[151., 10.]]` -> target `[0]`, scores `[[-49., 10.]]`;
  `[[20., 30.]]` -> target `[-1]`, scores unchanged; the F5 cross-language test keeps its
  expected matrix byte-identical)
- Modify: `tests/test_phase4d_batchmode.py`, `tests/test_phase4c_drivers.py` (the fake
  engines gain `channel_results()` returning the columnar dict; the `[]` fake becomes one
  real row so `for_config(0)` has something to select; say in the report if a test relied
  on the silent-empty path)
- Modify: `tests/pyo3/test_phase9_gates.py`, `tests/pyo3/test_phase10_gates.py`
  (`per_file_cost_max` from `r.seg_cost / np.maximum(1.0, r.seg_count)` on
  `for_config(0)`; the pinned numbers MUST NOT MOVE)
- Leave: `tests/pyo3/test_seam_replay.py`, `test_engine_smoke.py`, `test_phase9_seam.py`,
  `tests/test_phase4d_fixtures.py` (they assert the WIRE and keep `results_matrix()`, D7)

- [ ] **Step 1: `_balance10` on decoded values.** With `z` the decoded score block and
  `t = lid_target`: `c2[c2 > 150] - 200` -> `z[:, 1][t == 1]`; `300 - c1[c1 > 150]` ->
  `100 - z[:, 0][t == 0]`; `(c1 > 150) * (c1 - 200 - 100 + cutoff)` ->
  `(t == 0) * (z[:, 0] - 100 + cutoff)` (the subtraction ORDER stays left to right);
  `stat_in` -> `z[:, 0]`; in the else branch `(col > 150)` -> `(t == mm)` and `tmp` is the
  decoded block with column `mm` zeroed. Each is an exact-real identity on [200, 300]; the
  arbiter is `test_balance10_*` byte-green against the Octave goldens. If any moves: STOP.
- [ ] **Step 2: The rest of `engine.py`, `train.py`, `test.py`, `scoring.py`.** Pure
  renames to fields. `evaluate`'s `.scr` loop: `file_idx = int(r.file[i])` (already
  0-based), scores `r.lid_scores[i]` (already decoded).
- [ ] **Step 3: The grep gate.** `rg -n "\[:, ?-?[0-9]+\]|\[16:|-2\]|> ?150|- ?200"
  src/python/speech/` returns NOTHING in `engine.py`, `scoring.py`, `drivers/train.py`,
  `drivers/test.py` (the `init_weights.py` / `weight_bridge.py` / `batching.py` hits are
  weight-matrix and listing slices, not result columns; list them in the report as
  reviewed).
- [ ] **Step 4: Gates.** `./lint_code.sh`; `uv run pytest tests -q -m "not slow"` (ONE
  run); rebuild the module ALONE, then `uv run pytest tests/pyo3 -q -k "channel_results or
  engine_smoke or seam_replay or phase9_gates or phase10_gates or exit_gate"`; the gate
  pins and the `phase7_parity_*` / `phase9_fast_parity` cargo legs unchanged.
- [ ] **Step 5: Commit** `refactor(python): readers take ChannelResults; aggregate_workers
  removed`.

---

### Task 6: Docs, ADR-0002, IMPROVEMENTS note, READMEs

**Files:**
- Modify: `docs/adr/0002-the-exact-tree-is-frozen.md` (status line gains
  `2026-10-07 (issue #23)`; the touch-class list gains "the typed channel result: the
  result row's layout owned by one `engine/` type, `to_row()` byte-identical, proven by the
  `.mat` goldens and the seam cross-pin")
- Modify: `IMPROVEMENTS.md` (`[phase4b] Mixed-width multi-config bags`: one sentence,
  "since issue #23 the fold reads `ChannelResult` fields and no longer depends on width; the
  panic is now only at the `MultiConfigResults` matrix edge")
- Modify: `src/rust/README.md` (new row `engine/channel_result.rs`; the
  `bag_of_processors.rs`, `corpus_processor.rs`, `confusion.rs`, `speech-py/src/lib.rs` rows
  say what moved)
- Modify: `src/python/speech/README.md` (`engine.py`, `scoring.py`, `drivers/train.py`,
  `drivers/test.py` rows)
- Modify: `docs/superpowers/README.md` (an "Issue-driven work" table with this plan)
- Modify: `docs/ARCHITECTURE.md` only if `rg -n "results_matrix|Error_vad" docs/` hits
- Modify: `CLAUDE.md` "Where things live" is untouched; the suite-size line is refreshed
  by the smart-commit pass (Task 8), not here

- [ ] **Step 1:** Make the edits above; `uv run pytest tests/test_live_docs_commands.py
  tests/test_development_md.py tests/test_license_hygiene.py -q`.
- [ ] **Step 2: Commit** `docs: the typed channel result touch class and module rows`.

---

### Task 7: The mutation battery (D12)

**Files:**
- Modify: `RESULTS.md` (new section `## Issue #23 -- the typed channel result: the mutation
  battery`, the Phase 11 catcher-map table format: `# | mutation | named catcher | verdict`)

Each mutation: apply -> run ONLY the named catcher's file (FOREGROUND, release, R6-capped)
-> record the verbatim failure line -> revert -> re-run green -> `git status` clean.
Nothing under `src/` is committed.

| # | mutation | named catcher |
|---|---|---|
| 1 | `to_row()` swaps `seg_cost` and `lid.cost` | the phase-4a `.mat` golden (`phase4a_train_golden` / `phase4a_tier1_e2e`) |
| 2 | `to_row()` drops the `+ 200` re-encode on the target column | `phase4b_confusion_golden` AND `tests/pyo3/test_channel_results.py` |
| 3 | `to_row()` writes `correct` as `1.0` | `phase4a_save_update::aggregation_bit_exact` (via `from_row`) and `phase4b_corpus_lid` |
| 4 | `channel_results_from_matrix` decodes the target as `v - 150` | `test_phase4c_scoring.py` (the F5 cross-language matrix) and `test_balance10_*` |

- [ ] **Step 1:** Run the four; a mutation caught by a DIFFERENT test than named is
  recorded as "by-other", not as caught; a mutation no test catches is recorded as a gap
  with the cheapest closer named, and the closer is NOT added in this task.
- [ ] **Step 2: Commit** `docs(results): issue #23 mutation battery`.

---

### Task 8: Whole-branch review and smart-commit

- [ ] **Step 1: Review.** Run `mattpocock-skills:code-review` since `main` on both axes
  (Standards: ADR-0002/0003, the bit-identity constraint, the vocabulary; Spec: the thirteen
  decisions on the issue). Every finding fixed or recorded as an issue comment, none
  deferred silently.
- [ ] **Step 2: Final gates.** `./check_all.sh`; `./lint_code.sh`; `uv run pytest tests -q
  -m "not slow"`; rebuild the module ALONE; `uv run pytest tests/pyo3 -q` (corpus present,
  ~20 min, ONE run); `uv run python -m speech.ledger render --check`.
- [ ] **Step 3: smart-commit.** Invoke the `smart-commit` skill over the WHOLE branch
  (`git diff main...HEAD`), so CLAUDE.md and the READMEs are synced with everything the
  eight tasks changed (the suite-size line, the module rows, the seam surface), then commit.
  The PR is opened by the user.
