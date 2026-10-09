# Phase 6 baseline results

The living record of the from-scratch SAD + LID baselines on the LRE03/07 corpus. Phase 6
delivers the baseline PROTOCOL + machinery, subset-proven; the full-corpus headline numbers
land when the user fires the offline launchers (`python -m speech.drivers.baseline <arm> ...`), and this file
receives them as they complete. Every number here is REPRODUCIBLE at a fixed seed under the
protocol below; nothing corpus-derived (no real filenames, paths, or feature tables) is
committed -- the arms synthesize listings/mappings/references at runtime from `corpus_root`.

Comparability reference (published 2015-era numbers): the IS2016 LRE15 dashboards under
`data/Scoring_LRE15/` (`Cavg_lang_eval.pdf`, `average_LER_eval_90percent_IS2016.pdf`) and the
2015 SAD `Results_Stats` dashboards. These are GRAPHICAL (visual reads only: SAD Pmiss ~0-2%,
Pfa ~15-22%, training-cost floor ~0.05-0.06); the machine-readable LID score inputs
(`refs.mat`, the 17-Apr LIDscoreDet dirs) are LOST, so both baselines are RECOMPUTED here
under this protocol rather than copied from a surviving table (design spec S0).

---

## Protocol

### Corpus + conditions

- Corpus: `data/LRE03-LRE07/` (27 GB, gitignored, licensed LDC-derived). Every real-data
  test is CORPUS-GATED (`tests.conftest.requires_corpus`) -- runs locally, skips in CI.
- LID FEATURES regime (the acoustic, 2015-comparable arm): File_Type 2 cep binaries
  (`AudioStruct.cpp:183-256`), `.plp8f0mvsdd` = PLP-8 + f0 + mean/var-scaled + delta-delta,
  vectorSize 23 (== the LID net's input size), 8 kHz, 100 fps. The LID net scores these
  precomputed features directly via Twin Mode 7 (the only mode that consumes
  `audio.external_features`); no waveform is decoded.
- LID PHONOTACTIC regime (Task 10): File_Type 1 phSeq Mode 7.
- SAD regime (Task 9): algo-3 spectral on the `train/audio` wav+xml pairs.

### R2 -- the vtln-vs-plain feature pairing (adjudicated Task 8)

The archive's TRAIN side (`train/LID_Features/plp8f0mvsdd/`) carries ONLY the plain
`plp8f0mvsdd` feature variant (LRE03 + LRE07). The EVAL side
(`eval/LID_Features/{3,10,30}/`) carries SEVEN variants per condition, including plain
`plp8f0mvsdd` AND the cmllr/vtln-adapted `plp8f0mvscmllr` / `plp8f0mvscmllrdd`. No 2015
`.config` survives anywhere in the archive (verified by exhaustive `fd`/`rg`) to state the
intended pairing.

DECISION: **plain `plp8f0mvsdd` on BOTH train and eval.** Rationale: since no cmllr variant
exists on the train side, a vtln-train pairing is structurally impossible with this archive,
and the defensible default (match train and eval variants) coincides with the only consistent
choice. A plain-train / vtln-eval pairing would be a train/eval feature mismatch the port's
fixed-dimension net cannot represent. The evidence (the 50 `*.plp8f0m.xml` metadata refs
under `train/LID_Features/` carry the 2015 absolute path pointing at `.../plp8f0mvsdd/...`,
confirming the plain feature was the training feature) is consistent with this choice.

### Metrics

- SAD: DCF = `0.75*Pmiss + 0.25*Pfa` at 5 collars (0, 1/4, 1/2, 1, 2 s), per the vendored
  NIST `scoreFile_SAD.pl` v2.1 (`speech.evaluate.dcf`, Task 4).
- LID: 12-way argmax error % (`speech.evaluate.lid_error`, per `compare_scores.m`) + the NIST
  closed-set Cavg (`speech.evaluate.cavg`, per `ComputeCavgNIST.py`). Chance argmax error for
  12 classes = 91.67%.
- The 12-class head is the port-native multiclass Twin (the FIXED F5 confusion machinery),
  NOT 2015's 12 one-vs-rest detectors -- argmax error stays directly comparable; the
  one-vs-rest replication is documented-optional legacy-comparability work (spec S1.7/R6).

### Training

- From-scratch seeded init (Xavier/He, `init_weights.py`) -> SMORMS3 epochs (`train_modern`)
  -> per-epoch forward-only validation (`val_metric="nn_cost_seg"`, the moving signal;
  balance-5 plateaus from scratch) -> early-stop -> best/last checkpoints.
- The Twin's SAD net is a FROZEN feature gate in Mode 7 (its result_vec is synthesized
  constant, whole utterance = one speech segment); ONLY the LID net trains. The cep files are
  pre-VAD'd whole-utterance speech, so a synthesized whole-file-speech STM reference satisfies
  the engine's `-m`-mode mandatory-reference gate; the LID target is the listing language.

### Seeds, determinism, N-lane parallelism

- A single fixed seed drives init + subset sampling + the split; the run metadata
  (`run_metadata.json`) records seed, N lanes, subset spec, the config hash, and the git
  SHA + dirty flag of the tree the run started on (issue #40).
- N-lane engine parallelism (`numOuterThreads`, `--lanes`): N=1 is the DETERMINISTIC PARITY
  mode (byte-identical to a sequential fold). N>1 is deterministic-but-N-dependent (the R6-4a
  static-lane model -- file `j` -> lane `j % N`, per-lane state chains diverge from N=1). The
  baseline protocol FIXES N per run and records it; all numbers below are N=1 unless noted.
- Run-twice at a fixed seed produces bit-identical checkpoints (verified in the subset gate).

---

## Results

Rows are `arm x split x metric`; cells fill in as arms + full runs complete. The `subset
gate` rows are the corpus-gated CI-proof numbers (a stratified subset, bounded epochs, this
machine); the `full run` rows are user-fired launcher outputs (post-phase).

### LID -- features regime (Algo 6, Mode 7, File_Type 2, 12-class, plain `plp8f0mvsdd`)

<!-- ledger:table phase6_lid_features -->
| split | files (train/valid/test) | LID error % | Cavg | chance % | config | seed |
|---|---|---|---|---|---|---|
| subset gate (2026-10-02) | 35 / 15 / 48 | 72.92 | 0.46 | 91.67 | `lre03_lid_features.toml`, 2 ep x 25 steps | 0 |
| subset gate -- untrained init baseline | (same test set) | 93.75 | 0.53 | 91.67 | (seed packs, no training) | 0 |
| full run | TBD | TBD | TBD | TBD | TBD | TBD |

Hosts: Apple M4 Pro (arm64, 14 cores, Darwin 25.6.0).
<!-- ledger:end -->

Subset-gate reading: the from-scratch 12-class LID net beats 12-way chance by 18.75 pt and
its own untrained init by 20.83 pt on a disjoint 48-file held-out slice, deterministically,
in ~4.7 min. On this tiny subset (~3 train files/language) the net mode-collapses toward the
majority classes -- an inherent property of from-scratch 12-way LID on little data, honestly
noted; the 48-file held-out set makes the beat-chance count (13 correct vs chance ~4) robust.
The train objective descends (2.368 -> 2.216). A general from-scratch caution (NOT this
committed run's behavior): the softmax-CE VALIDATION cost CAN rise (the net grows confident on
the classes it learns first) even as argmax accuracy improves -- observed in exploratory/longer
runs; the committed 2-epoch run's val cost actually DECREASED (best_epoch=1). Either way the gate
keys off the direction-safe TASK metric -- the held-out argmax error (beats chance AND beats
init) -- not the CE. Genuine 12-way discrimination is the full-corpus launcher's job.

### SAD (Algo 3 spectral, File_Type 0 wav) -- Task 9

From-scratch algo-3 spectral SAD on the `train/audio` wav+xml pairs (the seeded 70/15/15 split
of the 2066 pairs, `dataprep.lre.derive_sad_listings`). The held-out slice is scored END TO END
with the T4 DCF harness: the engine dumps one VRCTS hypothesis xml per test file
(`Dump_Directory`), read back via `evaluate.load_vrcts_hyp`; the `.part.xml` references via
`evaluate.load_vrcts_ref` (windowed to the capped-audio span, matching the engine's own
`_AudioDuration` reference windowing); pooled through `evaluate.dcf`. These are the FIRST REAL
DCF numbers under this protocol. DCF pooled per-collar over the whole held-out slice (Pmiss/Pfa
accumulated across a concatenated timeline, the single-number analogue of the LID arm's pooled
argmax error). This is a MICRO-average (one concatenated timeline, one `dcf()` call =
duration-weighted, `_pool_dcf`); the 2015 `ComputeDCF.py` pooling convention is not ported (and
no copy survives in the archive) -- confirming the port's micro-pooling matches it is a FULL-RUN-ERA
verification item, deferred to when a launcher run produces numbers to compare.

<!-- ledger:table phase6_sad_v1 -->
| split | files (train/valid/test) | DCF@0 | DCF@0.25 | DCF@0.5 | DCF@1 | DCF@2 | Pmiss/Pfa @0.5 | config | seed |
|---|---|---|---|---|---|---|---|---|---|
| subset gate (2026-10-02) | 10 / 8 / 24 | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 0.00 / 1.00 | `lre_sad.toml`, 3 ep x 10 steps, 20 s cap | 0 |
| subset gate -- untrained init baseline | (same test set) | 0.7500 | 0.7500 | 0.7500 | 0.7500 | 0.7500 | 1.00 / 0.00 | (seed pack, no training) | 0 |
| full run | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

Hosts: Apple M4 Pro (arm64, 14 cores, Darwin 25.6.0).
<!-- ledger:end -->

Subset-gate reading (measured seed 0, ~80 s per run, run-twice bit-identical): the from-scratch
SAD net moves the held-out DCF from the untrained init's **0.7500** to **0.2500** at every
collar (+0.50), deterministically. HONEST CAVEAT (the SAD analogue of the LID mode-collapse
note): the capped audio windows are ~53% speech, so on a 10-file subset the net mode-collapses
toward the window-majority -- the seeded init sits BELOW the rising threshold everywhere (all
non-speech, Pmiss 1.0, DCF 0.75), and ~30 SMORMS3 steps drive it ABOVE the threshold everywhere
(all speech, Pmiss 0.0, Pfa 1.0, DCF 0.25). There is no partial-discrimination sweet spot at
this scale (measured: the net jumps from all-non-speech straight to all-speech; the NNCostSeg
train objective descends into the ~0.03-0.06 floor). The direction-safe TASK metric is
therefore the trained-vs-init held-out DCF (the net LEARNED TO FIRE: trained Pmiss ~0 vs init
Pmiss ~1); a DCF below the all-speech 0.25 baseline -- genuine speech/non-speech discrimination
-- is the FULL-RUN launcher's job (more data, more epochs). Because both endpoints are
degenerate (all-one-class), the DCF is exactly 0.25/0.75 with no libm-sensitive boundary jitter,
so the numbers are rock-stable cross-machine.

### LID -- phonotactic regime (Algo 6, Mode 7, File_Type 1 phSeq) -- Task 10

The 2015 FLAGSHIP regime: the same 12-class Twin as the features arm but in Mode 7 over the
phSeq one-hot phoneme sequences (File_Type 1, the fully-ported bit-exact-lineage reader). The
LID net's input is 38 (the letterMapping width) vs the cep arm's 23. THE FROZEN-SAD CONTRACT
(the phase-5 finding, VERIFIED at the seam this task): in Mode 7 the SAD net is never run --
its result_vec is synthesized constant and its weight derivatives are reset-then-scaled but
never accumulated, so its gradient is structurally ZERO and it stays EXACTLY at its from-scratch
seed (`best_sad.bin == sad_seed.bin` byte-for-byte, asserted in the gate). ONLY the LID net
trains. Data source: `train/phSeq/*.file.phSeqbis` (598 CallFriend whole-utterance files, all 12
languages, language from the 2-letter filename prefix) -- the 2015 phSeq listings localize to 0
rows on this archive (they anchor on `eval/` but the files sit under `train/phSeq/`), so the
corpus tree is globbed directly. FULL-RUN PROVENANCE NOTE: the archive's 2015 eval phSeq
listings name 1199 `lidXXXXX` files that PHYSICALLY sit under `train/phSeq/` and carry the
listing's own language labels; they are usable for a fuller run via a RE-ANCHORED localizer
(pointing the `train`/`eval` anchor at `train/phSeq/`) instead of the 2-letter-prefix glob the
subset gate uses -- a full-run-era option, unused by the committed glob arm.

<!-- ledger:table phase6_lid_phseq -->
| split | files (train/valid/test) | LID error % | Cavg | chance % | config | seed |
|---|---|---|---|---|---|---|
| subset gate (2026-10-02) | 15 / 15 / 45 | 84.44 | 0.49 | 91.67 | `lre03_lid_phseq.toml`, 2 ep x 12 steps | 0 |
| subset gate -- untrained init baseline | (same test set) | 91.11 | 0.48 | 91.67 | (seed packs, no training) | 0 |
| full run | TBD | TBD | TBD | TBD | TBD | TBD |

Hosts: Apple M4 Pro (arm64, 14 cores, Darwin 25.6.0).
<!-- ledger:end -->

Subset-gate reading (measured seed 0, ~193-232 s per run box-dependent -- the training portion
is ~193 s, the full run_baseline call ~232 s; run-twice bit-identical): the from-scratch
12-class phonotactic LID net beats 12-way chance by **7.22 pt** and its own untrained init by
**6.67 pt** on a disjoint 45-file held-out slice, deterministically, with the SAD net PROVABLY
frozen (only the LID net moved). HONEST FRAMING (spec S1.9, no overclaim): these margins are THIN
compared to the acoustic features arm's ~20 pt, and the regime is UNSTABLE from scratch -- the
per-epoch train cost ASCENDS (2.41 -> 2.93) even as the held-out argmax improves (the T8 lesson,
sharper here), and a single-epoch training cut goes NEGATIVE (measured -3.0 pt vs init). The 2015
story was phonotactic >> acoustic, but only WITH FULL DATA; on this tiny per-language subset (~1-2
files/language) phonotactic is WEAKER than acoustic -- exactly the "needs more data than acoustic"
flip side. The direction-safe signal is the held-out argmax (beats chance + beats init), not the
CE/train cost; genuine phonotactic discrimination (and the phonotactic >> acoustic crossover) is
the full-run launcher's job (more data, more epochs).

---

## LID cell matrix (issue #21)

`--cell` / `--direction` on `lid-features` and `lid-phseq` target the LID net, the one that
trains under the frozen-SAD contract; the SAD net stays the legacy LSTM at its seed, byte-identical
across every LID row of a seed (unit-pinned), so the rows below compare to each other and to the
Phase-6 LSTM row on the same split. A forward LID net carries the same two derived keys as a
forward SAD net (`BLSTM_LID_OutputNeuronNb 24,12`, `BLSTM_LID_window 0`, the plain whole-utterance
regime). EXACT-TREE NUMBERS: every row is scored on the exact tree; since #57 the fast Twin LID runs
the same matrix (parity-pinned per pair, "Issue #57 -- the fast Twin LID matrix" below), so these
rows have fast and streamed twins but are not re-measured there. Evidence: `tests/pyo3/test_lid_cells_gates.py` -- a preflight probe over all 20
(arm x cell x direction) legs (the LID pack length the engine accepts per cell, pinned; a finite
interior init cost; a nonzero LID gradient and an exactly-zero SAD gradient), one trained
`slstm / forward` leg per arm at the Phase-6 recipe, run-twice bit-identical on both arms.

The phSeq `slstm / forward` row is recorded as measured: it beats its own init by 8.89 pt but sits
at chance (91.11% against 91.67%), the thin from-scratch phSeq regime the Phase-6 LSTM row already
documents (a 1-epoch cut went negative there). Its gate pins beat-init only; the features row pins
beat-chance (+8.33 pt) and beat-init (+10.42 pt).

### LID -- features regime, by cell

<!-- ledger:table lid_features_cells -->
| cell / direction | files (train/valid/test) | trained LID error % | init LID error % | gain (pt) | Cavg | chance % | wall |
|---|---|---|---|---|---|---|---|
| LSTM / bidirectional | 35 / 15 / 48 | 72.92 | 93.75 | +20.83 | 0.46 | 91.67 | 212 s |
| LSTM / forward | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| sLSTM / bidirectional | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| sLSTM / forward | 35 / 15 / 48 | 83.33 | 93.75 | +10.42 | 0.49 | 91.67 | 43 s |
| Mamba / bidirectional | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Mamba / forward | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| CfC / bidirectional | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| CfC / forward | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Transformer / bidirectional | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Transformer / forward | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| full-corpus runs (any cell x direction) | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

Hosts: Apple M4 Pro (arm64, 14 cores, Darwin 25.6.0).
<!-- ledger:end -->

### LID -- phonotactic regime, by cell

<!-- ledger:table lid_phseq_cells -->
| cell / direction | files (train/valid/test) | trained LID error % | init LID error % | gain (pt) | Cavg | chance % | wall |
|---|---|---|---|---|---|---|---|
| LSTM / bidirectional | 15 / 15 / 45 | 84.44 | 91.11 | +6.67 | 0.49 | 91.67 | 136 s |
| LSTM / forward | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| sLSTM / bidirectional | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| sLSTM / forward | 15 / 15 / 45 | 91.11 | 100.00 | +8.89 | 0.51 | 91.67 | 32 s |
| Mamba / bidirectional | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Mamba / forward | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| CfC / bidirectional | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| CfC / forward | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Transformer / bidirectional | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Transformer / forward | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| full-corpus runs (any cell x direction) | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

Hosts: Apple M4 Pro (arm64, 14 cores, Darwin 25.6.0).
<!-- ledger:end -->

---

## Full runs: the runners (post-phase, user-fired)

A full-corpus row is fired by a committed runner under `experiments/`, one script per `TBD`
table, each row's recipe hard-coded (seed 0, 40 epochs x 25 steps, patience 6, lanes from the
host; the recipe is the ledger row's identity, so a different budget is a different study):

| runner | rows | table |
|---|---|---|
| `experiments/00_prereqs.sh` | corpus root, release build, `runs/` ignored | -- |
| `experiments/01_phase6_full_runs.sh` | the legacy BLSTM on `lid-features`, `sad`, `lid-phseq` | `phase6_*` |
| `experiments/02_v1_cells.sh` | sLSTM, Mamba x both directions on `sad` | `phase9_v1_cells` |
| `experiments/03_v2_cells.sh` | LSTM, sLSTM, Mamba, CfC x both directions on `sad-v2` | `phase10_v2_cells` |
| `experiments/04_v2_transformer.sh` | Transformer x both directions on `sad-v2` | `phase11_v2_cells` |
| `experiments/05_lid_cells.sh` | every cell x direction on both LID arms (bar `lstm / bidirectional`) | `lid_*_cells` |

Host knobs: `SPEECH_CORPUS_ROOT` (default `data/LRE03-LRE07` in the repo; a relative path is resolved against the caller's directory) and `SPEECH_LANES` (default 1, the
parity mode; ADR-0007 puts the lane count in the record). A runner takes optional row names
(`experiments/05_lid_cells.sh lid-phseq/mamba/forward`), runs one launcher invocation per row into
`runs/<study>/<arm>_<cell>_<direction>_<UTC>/`, logs beside it, continues past a failed row, and at
the end of each row tries `python -m speech.ledger add <run>/record.json`; a refusal (dirty tree, a
live record for the recipe) prints the manual command and the runner exits nonzero, so supersession
stays a human decision. One row by hand, the same recipe:

```
uv run python -m speech.drivers.baseline lid-phseq --corpus-root data/LRE03-LRE07 --out-dir runs/lid_cells/by_hand \
    --cell mamba --direction forward --lanes 1 --seed 0 --epochs 40 --steps-per-epoch 25 --patience 6
```

Omit `--subset` to train on the whole split (SAD: the full 70% train split of the 2066 wav/xml
pairs; LID: the whole localized corpus); the SAD runners pass `--audio-max-duration 120` (the
corpus wavs are 576-1800 s CallFriend files, median ~600 s). The run writes `run_metadata.json`
(the spec as it ran plus the config hash and the start-of-run git SHA + dirty flag), `checkpoint/`
(best/last packs + `train_history.json`), the held-out scores -- LID `scores/` (`.scr`) ->
`lid_error`/`cavg`; SAD `score_trained/` (VRCTS hyp xml) -> pooled `dcf` -- and `record.json`,
the promotable record (not on `--dry-run`). A run directory that already holds a manifest is
refused unless `--resume` continues it.

Known limitation (machinery): `forget_bias_one` (the LSTM forget-gate 1.0 init, default on) is
threaded correctly through `ModernTrainParams` end to end but has NO CLI flag on `python -m speech.drivers.baseline`
-- only its default (`True`) is exercised; a `False` sweep would need the flag added.

---

## Phase 7 -- performance

The measure-then-pin exact-path baseline (Task 1): `speech bench [--repeat=N] [--path=exact]
<config>` runs a corpus config end to end (the `-i` image-mode `CorpusProcessor` path, a single
unscored forward pass -- `Neural_Networks_BackPropagation_Epochs 0` and, per config,
`*_BackPropagationActivated false`) and prints one `BENCH path=exact wall_s=<f> audio_s=<f>
rtf=<f> maxrss_mb=<f> files=<n>` line per run. `rtf = wall_s / audio_s`; `maxrss_mb` is the
process-lifetime peak RSS (`getrusage(RUSAGE_SELF)`, bytes on macOS / KiB on Linux, normalized to
MiB). UNIT NOTE (issue #54): `maxrss_mb` holds MiB; every `maxrss` figure measured in this file
(and quoted from it in `docs/ROADMAP.md`) and written as "MB" is that MiB value (43.52 MiB is
45.6 decimal MB), the ratios and bounds are unit-free. Every number below is `--repeat=1` (a
single fresh process per measurement, so `maxrss_mb` is a clean per-run peak, not inflated by
repeated in-process construction -- `--repeat>1` in one process accumulates allocator
high-water-mark across runs by design, see `bench.rs`'s doc). At Task 1 time no fast path
existed yet (`--path` accepted only `exact`); these numbers were the
baseline every later fast-path task's RTF/memory claim is measured against. **Since Task 7**,
`--path=fast` is wired (`Inference_Path` overlaid onto the config by `run_bench`, see `bench.rs`'s
doc) -- see "Exact vs fast: the measured matrix + pinned budgets" below for the comparison this
baseline exists to support.

Hardware: Apple Silicon dev box (arm64, this machine), `cargo build --release` (LTO on, the
committed release profile).

### 60 s fixture (CI workhorse)

`tests/reference_data/phase4a/tier2_spectral.config` (Algo 3 spectral SAD, the tier-2 golden
net) staged with the tuple-A weight pack (`phase0/NNweights_config1.bin`) and the committed 60 s
`phase4d/prcts_excerpt.wav` (stereo, 8 kHz) as a single-file, no-reference listing -- the exact
recipe `tests/phase7_bench.rs::stage_bench_config` builds and `bench_line_parses`/
`bench_rejects_unknown_path` pin.

| run | wall_s | audio_s | rtf | maxrss_mb | files |
|---|---|---|---|---|---|
| 1 | 0.318309 | 120.000000 | 0.002653 | 57.469 | 1 |
| 2 | 0.317524 | 120.000000 | 0.002646 | 57.641 | 1 |
| 3 | 0.312214 | 120.000000 | 0.002602 | 57.734 | 1 |

Sanity cross-check (design spec's scouting numbers, RTF ~0.004 / ~0.6 MB per audio-second):
measured RTF ~0.0026-0.0027 (same order of magnitude, ~1.5x faster here) and ~0.48 MB/audio-s
(57.6 MB / 120 s) -- consistent.

### Task 8 -- exact-path mel-bank hoist (before/after, behaviorally-neutral)

Task 8 (shared-code hygiene) removed one of the two per-channel `MelFilterBank::new`
constructions on the exact algo-3 path. `build_input_sequence_parts` used to build the bank
TWICE per channel with identical args -- once to query `nb_dct()`/`nb_filters()` for the LTSV
width, once inside `assemble_from_periodogram` to apply it; the hoist builds it once (via the new
`build_mel_bank`) and shares it through `assemble_with_bank`. `MelFilterBank::new` is pure
arithmetic of `(cfg, s, rate)` (no shared/mutable state), so the collapse is byte-identical -- the
full phase-1/2b feature goldens (`inputseq_*`, mel, spectral, LTSV) and the 4a/4b corpus goldens
all stay byte-stable. The pitch-pass third construction (`tasks/sad.rs`, warped periodogram) is
provably identical but intentionally left standalone (it never runs on this TDCwindow-0 fixture,
and threading it out would change a public return type -- see the Task 8 report).

Same 60 s stereo fixture + recipe as the CI-workhorse table above, `--path=exact`, 3 repeats x 2
fresh processes per build (Apple M4 Pro, arm64, release/LTO). The first repeat of each fresh
process is a cold-cache warm-up outlier; steady-state is the remaining five per build:

| build | wall_s (run1: r1,r2,r3 / run2: r1,r2,r3) | steady-state best | steady-state median |
|---|---|---|---|
| before (baseline) | 0.285129\*, 0.262430, 0.265041 / 0.264634, 0.263967, 0.264996 | 0.2624 | 0.2646 |
| after (hoist)     | 0.277200\*, 0.264986, 0.264126 / 0.265300, 0.267099, 0.265569 | 0.2641 | 0.2653 |

(\* = first-repeat warm-up, excluded from steady-state.) `audio_s`=120 both builds; `maxrss_mb`
~55-66 both (no measurable RSS change -- the bank is a few hundred floats). The before/after wall
delta (~0.1-1.7 ms) sits BELOW the ~2-3 ms steady-state run-to-run spread (the warm-up outliers
alone span ~8 ms): the two builds are statistically indistinguishable at this fixture, because
bank construction is a negligible fraction of the BLSTM-forward-dominated ~0.26 s wall (~1500
periodogram frames x fwd/bwd). The hoist's value is code hygiene + one fewer allocation per channel,
NOT a measurable speedup here -- recorded honestly (measure-then-pin, no delta promised beyond what
the instrument resolves).

### Corpus-gated (real LRE03/07 files, runtime sorted-first selection)

Per the license bright line, the two files below are selected at RUNTIME as the
lexicographically-FIRST match under their respective corpus subtree (`sort` over a glob, no
duration- or content-based picking) -- no filename, path, or corpus-identifying label is
recorded anywhere in this repo; only the resulting BENCH measurements are. This is a one-off
local measurement (not a committed automated test -- corpus-gated performance numbers follow the
same "user/session gathers, RESULTS.md records" convention as the Phase 6 subset-gate rows
above), reproducible by re-running the recipe below against the same locally-licensed corpus
snapshot.

**SAD leg** -- the SAME `tier2_spectral.config` recipe as the 60 s fixture above, with
`fileslisting` repointed at the sorted-first `*.wav` under `data/LRE03-LRE07/train/audio/**`
(mono, 8 kHz; `Audio_max_duration` lifted so the full file is processed uncapped). This
particular sorted-first file happens to be short (75 s) -- well inside the corpus's documented
576-1800 s range (see the SAD baseline section above) but not itself the "typical" ~600 s scale;
recorded honestly rather than re-selected to hit a target duration (measure-then-pin, no
content-based cherry-picking).

| run | wall_s | audio_s | rtf | maxrss_mb | files |
|---|---|---|---|---|---|
| 1 | 0.199811 | 75.000000 | 0.002664 | 55.062 | 1 |
| 2 | 0.211289 | 75.000000 | 0.002817 | 54.797 | 1 |
| 3 | 0.199949 | 75.000000 | 0.002666 | 54.781 | 1 |

**LID phonotactic leg (Twin, Mode 7)** -- `tests/reference_data/phase4b/twin_mode7.config` (the
Phase 4b flagship fixture net; both nets' `*_BackPropagationActivated` are already `false` in the
committed config, so no override was needed) with `fileslisting` repointed at the sorted-first
`*.phSeqbis` under `data/LRE03-LRE07/train/phSeq/**`. phSeq's "processed duration" is content-
derived (`phseq_frames_count`, NOT `Audio_max_duration`-capped -- see `audio.rs::read_phseq`), so
this leg's audio_s (42.5 s) reflects that specific file's phoneme-sequence length, not a
wav-clock duration.

| run | wall_s | audio_s | rtf | maxrss_mb | files |
|---|---|---|---|---|---|
| 1 | 0.068943 | 42.540000 | 0.001621 | 34.047 | 1 |
| 2 | 0.064122 | 42.540000 | 0.001507 | 31.609 | 1 |
| 3 | 0.058129 | 42.540000 | 0.001366 | 31.531 | 1 |

Reading: both corpus-gated legs land in the same RTF ballpark as the 60 s CI fixture (~0.0014 -
0.0028, all comfortably sub-1.0 -- i.e. the exact path already runs far faster than real time on
this hardware) and the same rough MB-per-audio-second scale (~0.7-0.75 here vs ~0.48 for the
stereo 60 s fixture -- channel count and net size both shift this, not a discrepancy). The
phonotactic (Mode 7) leg is fastest/lightest per audio-second, consistent with its much smaller
net (11,12-unit LSTM layers vs the SAD net's 23,24,24) and Mode 7's frozen-SAD contract (only the
LID net actually runs). These are exact-path numbers only -- Task 2+'s fast path is compared
against this table, not against the scouting sanity figures.

### Exact vs fast: the measured matrix + pinned budgets (Task 7)

`speech bench --path=fast <config>` (Task 7) dispatches through the SAME config `--path=exact`
uses, with `Inference_Path` overlaid to `fast` -- the ONE deliberate exception to `run_bench`'s
"no config mutation" contract (see `bench.rs`'s doc). Every leg below reuses the identical
staging recipe already established (the 60 s CI fixture's `tests/phase7_bench.rs::
stage_bench_config`; the two corpus-gated legs' sorted-first file selection, unchanged from the
Task 1 baseline above -- SAME files, confirmed by identical `audio_s` -- 75.00 s / 42.54 s -- to
the Task 1 table); the LID cep leg is NEW (Task 1 only measured phSeq), sorted-first
`*.plp8f0mvsdd` under `data/LRE03-LRE07/train/LID_Features/**` via the same `twin_mode7.config`
recipe with `File_Type 2`. Every corpus-gated number is a one-off local measurement (not a
committed automated test), same posture and license discipline as Task 1's (sorted-first
selection, no filename/path recorded). 3 independent `--repeat=1` processes per (leg, path) --
mean and [min-max] range reported; hardware: Apple M4 Pro (arm64), macOS 26.5.2, `cargo build
--release` (LTO on).

Scope note: every `exact` row runs through `nn/blstm.rs::feed_forward_backward` with
`*_BackPropagationActivated false` (forward-only, no gradient computed) -- but the underlying
`Network` container retains its per-layer `layers_output` activation cache UNCONDITIONALLY
during the forward drive itself (`network.rs`, populated inside `drive`, not gated on whether a
later `feed_backward` call happens), so the exact path's measured memory here already includes
that retained-cache cost; it is not a stripped-down "pure inference, no backward machinery"
build. This is an inference-vs-inference comparison exactly AS CONFIGURED (the same way both
paths are actually invoked elsewhere in this repo -- Image mode, backprop off), not a claim that
the exact path has no backward-shaped allocations at all.

The table below is RENDERED from the ledger (ADR-0009): every row is the latest three-process
measurement per (leg, path, host), re-taken 2026-10-02 on the same Apple M4 Pro under macOS
26.7.1 / rustc 1.99 through `python -m speech.ledger bench --leg <name>`. The reading that
follows it refers to the ORIGINAL 2026-07 measurement (SAD 4.58-4.60x, LID 3.53-3.57x, macOS
26.5.2) and is kept as history; the re-measure moved both paths' absolute walls (exact ~10-20%
slower, fast ~10% slower) and the ratios to SAD 4.93x / LID 3.43-3.63x, which the README and
ARCHITECTURE prose now quote and `speech.ledger.prose` asserts, together with the 60 s leg's
peak RSS (the `maxrss_mb` column here, against 41.95 / 56.52 in the Phase 10 Task 5 table
below, which stays as history; peak RSS is a cost figure like the walls, not a metric under
the ADR-0003 STOP rule).

<!-- ledger:table phase7_bench_matrix -->
| leg | audio_s | path | wall_s (mean [range], n) | rtf | maxrss_mib | MiB/audio-s | speedup (wall, fast vs exact) |
|---|---|---|---|---|---|---|---|
| SAD 60 s fixture (stereo) | 120.00 | exact | 0.3177 [0.3146-0.3215] (n=3) | 0.002647 | 57.182 | 0.4765 | baseline |
| SAD 60 s fixture (stereo) | 120.00 | fast | 0.0645 [0.0642-0.0648] (n=3) | 0.000537 | 43.521 | 0.3627 | 4.93x |
| SAD corpus-gated (mono) | 75.00 | exact | 0.1996 [0.1963-0.2015] (n=3) | 0.002661 | 54.062 | 0.7208 | baseline |
| SAD corpus-gated (mono) | 75.00 | fast | 0.0405 [0.0402-0.0409] (n=3) | 0.000539 | 39.281 | 0.5238 | 4.93x |
| LID phSeq corpus-gated (Twin M7) | 42.54 | exact | 0.0538 [0.0527-0.0550] (n=3) | 0.001265 | 31.010 | 0.7290 | baseline |
| LID phSeq corpus-gated (Twin M7) | 42.54 | fast | 0.0157 [0.0151-0.0167] (n=3) | 0.000368 | 20.547 | 0.4830 | 3.43x |
| LID cep corpus-gated (Twin M7) | 32.65 | exact | 0.0415 [0.0415-0.0416] (n=3) | 0.001272 | 21.979 | 0.6732 | baseline |
| LID cep corpus-gated (Twin M7) | 32.65 | fast | 0.0114 [0.0114-0.0115] (n=3) | 0.000350 | 13.495 | 0.4133 | 3.63x |

Hosts: Apple M4 Pro (arm64, 14 cores, Darwin 25.6.0).
<!-- ledger:end -->

Reading -- speedup lands ABOVE the T6 Python-level scoring range (2.6-3.6x, warm-cache `.scr`/DCF
scoring incl. PyO3 crossing + file I/O): 3.5-4.6x here, exactly the T6 report's own prediction
("your dedicated bench isolates the compute better so the ratio may be higher") -- SAD is faster
than LID relatively (4.6x vs 3.5x) because its FFT+matmul-dominated pipeline hits the biggest
per-kernel wins (see the criterion numbers below), while Mode 7's LID arms are a single small
truncate-windowed BLSTM forward over a short pre-extracted feature block, less compute to
amortize a fixed per-call overhead against.

**MEMORY FINDING, attributed not flagged as an anomaly (T3 review obligation, see `fast/
pipeline.rs`'s "MEL/DCT REUSE" doc note):** the SAD arms use MORE memory on fast (1.22-1.28x),
while the LID arms use LESS (0.59-0.66x) -- a real, documented split, not noise. `fast::
pipeline::FastPipeline::build_input_sequence_parts` allocates a FRESH f64 `Array2` every call to
widen its f32 periodogram before reusing the exact `apply_filter_bank` (deliberately NOT a
preallocated/reused scratch buffer, unlike every other buffer in that module's workspace) --
confirmed via source read (`fast/driver.rs:338`) that ONLY `FastSpectralSegmenter` (algo 3, the
SAD driver) constructs a `FastPipeline` at all; `FastTwinLid` (algo 6, Mode 7) consumes
pre-extracted phSeq/cep `external_features` directly and never builds a periodogram, so it carries
NONE of this cost -- consistent with the LID arms' fast path using genuinely LESS memory (smaller
net, f32 buffers throughout, no widening tax). Sizing the SAD arm's per-channel buffer directly, in the SAME MiB-as-"MB" unit `bench.rs::
maxrss_mb()` itself reports (`ru_maxrss bytes / 1024^2` on macOS): the 60 s fixture's
`BLSTM_spectrum_shift 0.01` at 8 kHz gives `shift_frames=80`, so a 60 s channel decodes to
`frame_nb = 480000/80 = 6000` periodogram rows; `perio64` is `6000 x 513 x 8 bytes / 1024^2 =
23.48 MB` per channel -- the right order of magnitude for the measured maxrss delta (12.05 MB on
the 120 s-audio-equivalent stereo fixture, 14.80 MB on the 75 s mono corpus leg), smaller than the
raw buffer size because `getrusage`'s peak is a WHOLE-PROCESS high-water mark, not a clean
per-buffer attribution (both paths carry other large transient buffers competing for the same
peak snapshot; channel 2's widen-buffer can also reuse channel 1's freed allocation). This is the
documented consequence of a deliberate, source-cited design decision (module doc: "the exact
`mel.rs` is UNTOUCHABLE ... an f32 re-transcription ... would be pure duplication risk with no
measurable speed benefit"), not a bug and not a target for this task to fix.

#### Pinned budgets

**CI-asserted (`tests/phase7_bench.rs::bench_fast_not_slower_than_exact_ci_smoke`, runs in CI
forever):** `fast wall_s <= exact wall_s * 1.5` on the 60 s CI fixture. WIDE headroom by design
(spec R5 -- shared CI runners make sub-second wall-clock timing flaky); the measured ratio on this
box is ~0.217x (fast is ~4.6x FASTER), so the pin carries ~6.9x headroom over measured -- it
exists to catch a catastrophic fast-path regression (e.g. an accidental fallback to a slow scalar
path), not to track the real number. Verified non-vacuous by manual mutation (temporarily set the
budget to `0.001x`, confirmed the assertion fails, reverted -- see the task report; the same
mutation is Task 10's mutation-battery item 7).

**Local-only regression bounds (NOT CI-asserted, this-box numbers, Apple M4 Pro named per spec
R5).** SUPERSEDED for the memory half -- see Phase 10's f32-mel section (the widen is deleted; the
SAD memory bound is now <= 0.85x of exact, measured 0.743x). Future local runs of this exact recipe
are expected to land within:
- SAD arms (algo 3, spectral): wall speedup >= 3.5x, memory <= 1.4x of exact (the periodogram-
  widening tax above is real and expected, not a regression signal up to this ratio). HEADROOM
  NOTE: the 1.4x ceiling sits only ~9-10% above the measured 1.22-1.28x maxrss ratios, and the
  peak is a whole-process high-water mark whose day-to-day spread exceeds the committed 3-repeat
  range (a review re-run of this recipe touched 1.288x). So a future 1.3-1.35x reading is plausible
  measurement noise, NOT a regression -- only a reading meaningfully above 1.4x is the signal.
- LID Mode-7 arms (algo 6, phSeq/cep): wall speedup >= 2.5x, memory <= 0.8x of exact (fast should
  stay LIGHTER here; memory creeping toward or above 1.0x would be the regression signal, since
  nothing in this arm's fast path should need more memory than exact).

A local run landing meaningfully below these (not within measurement noise -- the ranges above
already carry headroom under the measured 3.53-4.60x speedups and 0.367-1.276x memory ratios) is a
FINDING to investigate, the same "measure, don't silently widen" discipline the CI parity gates
(Tasks 4-6) already established for correctness; there is no automated enforcement of these bounds
(spec R5: "the real numbers are local... no CI assertion on them").

### Criterion micro-benches

`cargo bench` (`src/rust/benches/kernels.rs`) times three exact kernels and, since Task 7, their
fast-path twins, on shapes read off `tier2_spectral.config` itself: `matmul_seq_92x96` (92 frames
x `BLSTM_NNetInputSize 23` into `23 x 4*24`, the LSTM input-projection product) vs
`faer_project_92x96` (the SAME shape, f32, calling the real `fast::nn::faer_project` -- not a
bench-local reimplementation); `gfft_1024` (`BLSTM_spectrum_order 10` -> a 1024-point GFFT) vs
`realfft_1024` (one real `realfft` forward at `window_size=1024`, f32); `mel_apply_513x20` (the
resulting 513-column periodogram through a `BLSTM_nb_bins 20` log-mel filterbank) vs
`mel_apply_widened_f64` -- HONESTLY named and scoped (T3 review obligation): the fast path has NO
f32 mel kernel, so this bench measures the REAL fast-path procedure end to end (widen the f32
periodogram into a fresh f64 `Array2`, then reuse the identical `apply_filter_bank` the exact
target calls), not a fabricated f32 kernel. Measured (`cargo bench`, 100 samples/target, Apple M4
Pro, `bench` profile):

> **SUPERSEDED (Phase 10 Task 5) -- the `mel_apply_widened_f64` rows below are a PHASE-7
> RECORD, not a live bench target.** `fast::mel32` landed a real f32 mel/DCT kernel at both
> fast sites, the widen is DELETED, and the bench target was renamed `mel_apply_f32`
> accordingly (`benches/kernels.rs`). Read the three `mel_apply_widened_f64` mentions in
> this section as "what the phase-7 widen bridge cost, and why it was worth removing"; the
> live numbers are in **Phase 10 -- the full-f32 mel front-end (Task 5, spec S4)** below.
> Every other row in this table is still live.

| target | path | scope | time |
|---|---|---|---|
| `matmul_seq_92x96` | exact | 92x23 * 23x96, f64 ascending loop | 57.408 us |
| `faer_project_92x96` | fast | SAME shape, f32, real `fast::nn::faer_project` | 4.624 us |
| `gfft_1024` | exact | 1024-pt complex FFT, 2 real 1024-sample frames packed per call | 17.898 us/call (8.949 us/frame) |
| `realfft_1024` | fast | 1024-sample real FFT, 1 frame per call | 0.765 us/call (= 0.765 us/frame) |
| `mel_apply_513x20` | exact | 100x513 f64 periodogram already in hand -> log-mel | 24.135 us |
| `mel_apply_widened_f64` (SUPERSEDED, see above) | fast | SAME shape, f32->f64 widen (fresh alloc) + SAME `apply_filter_bank` | 58.309 us |

Reading: `faer_project_92x96` is **12.41x faster** than `matmul_seq_92x96` at the identical shape --
the single biggest per-kernel win, and (with the LSTM recurrence itself unbenched here, see
`fast/nn.rs`'s doc on why it stays a hand-written dot rather than a per-frame faer matvec) the main
driver of the SAD/LID wall-clock speedups above. `realfft_1024` is **23.4x faster per call**, or
**11.7x faster per FRAME** once the exact GFFT's two-real packing (one call amortizes TWO windowed
frames, not one -- see `fast/pipeline.rs`'s equivalence proof) is normalized out; either framing is
a large win.

**FINDING (reported honestly, not hidden -- exactly the brief's ask; SUPERSEDED by Phase 10
Task 5, which is what this finding motivated -- see the callout above):**
`mel_apply_widened_f64` is **2.42x SLOWER** than `mel_apply_513x20` at the identical shape, not
faster. This is the PRECISE, measured answer to "where does the fast path spend relatively more
time": the mandatory f32->f64 widen-and-fresh-allocate step (see the memory finding above) is a
real per-call cost that the reused f64 `apply_filter_bank` call alone does not carry on the exact
side (which already has its periodogram in f64, no widening needed). It does not erase the SAD
arm's overall 4.58-4.60x end-to-end win -- the FFT (11.7x/frame) and matmul (12.41x) gains
elsewhere in the same pipeline dominate the total -- but it is the one component of the fast SAD
path that is measurably, unambiguously slower than its exact counterpart, and it is the direct
mechanism behind the SAD-arm memory increase documented above. A fully-f32 mel kernel (removing
the widen step entirely) is exactly the "possible future tightening" `fast/pipeline.rs`'s module
doc already names as out of THIS task's scope.

### Metric parity: fast vs exact on the phase-6 subset checkpoints (corpus-gated)

`tests/pyo3/test_phase7_parity.py` (Task 6) lifts the Rust CI parity legs
(`phase7_parity_{sad,lid}.rs`, which pin boundary/argmax equality on committed fixtures) onto REAL
corpus data at the METRIC level: each phase-6 subset checkpoint (trained from scratch on the EXACT
f64 path via the exact `test_phase6_gates.py` recipe) is scored on its own disjoint held-out slice
under BOTH `Inference_Path` values, and the per-file DECISIONS + the reported metrics are compared.
The prediction from the CI legs -- identical decisions => identical metrics => EXACTLY 0.0 metric
delta -- holds on all three arms (measured 2026-07-19, Apple Silicon dev box, seed 0):

| arm | held-out files | per-file decision agreement | metric delta (fast - exact) | score-value max_abs |
|---|---|---|---|---|
| SAD (algo-3 spectral) | 24 | VRCTS boundaries identical (count+types+times, max_dt 0.0 s) | DCF 0.0 at every collar (0/0.25/0.5/1/2 s) | n/a (boundaries) |
| LID features (Twin M7, cep) | 48 | argmax identical, 0 flips | lid_error 0.0, cavg 0.0 | 3.5e-10 |
| LID phonotactic (Twin M7, phSeq) | 45 | argmax identical, 0 flips | lid_error 0.0, cavg 0.0 | 1.4e-10 |

`lid_error`/`cavg` are argmax-only functions (softmax is monotone), so identical per-file argmax
forces a bit-identical metric -- the 0.0 deltas are not a tolerance, they are float equality. The
SAD boundaries come from the SHARED f64 decision layer both paths hand off to, so identical
posteriors-to-the-decision => byte-identical VRCTS hyps => DCF 0.0. The `.scr` score VALUES still
carry the f32 divergence (~1e-10 after the softmax normalization compresses it; the raw pre-softmax
divergence is the ~1e-6 the Rust legs measured), reported as a diagnostic -- it is the DECISIONS
that hold, not the last bit of the score. Any future flip fails the test (the R1 drift detector);
adjudication (file count + delta distribution) is not silently absorbed.

WEIGHT-INJECTION NOTE (load-bearing for the test): the fast drivers load weights ONLY at
construction (`load_weights_file`, from `BLSTM_weightsFile`/`BLSTM_LID_weightsFile`). At T6
measurement time `BagOfProcessors::set_weights` was a SILENT NO-OP for `Processor::FastSpectral`/
`FastTwinLid`, so the parity test points BOTH configs' weight keys at the trained checkpoint (the
fast path's only injection mechanism, identical for both paths) rather than relying on the seam's
`set_weights` -- otherwise the fast path would silently score the config's SEED pack while exact
scores the injected trained pack, a 100%-divergence artefact (observed and diagnosed during Task 6,
NOT a real parity failure). **T6b (commit `2081212`, same branch) hardened this into a LOUD bail**
(`Processor::FastSpectral`/`FastTwinLid` now `bail!` naming `Inference_Path` + the
`BLSTM_weightsFile`/`BLSTM_LID_weightsFile` keys on any `set_weights` call, `bag_of_processors.rs`)
-- the config-repoint injection above is unchanged and still required (it is the fast path's ONLY
weight-injection mechanism), but the two Python callers that used to rely on the silent no-op
(`evaluate()`, `_score_sad_pack_on_test`) now SKIP their own redundant `set_weights` call under
`Inference_Path fast` instead of hitting the new bail. See `.superpowers/sdd/task-6-report.md`'s
"T6b" section for the full mechanism + the sibling-method audit. (2026-10-09, issue #62: the bail
is gone. The fast drivers take `set_weights` under the exact tree's exact-length contract, in the
same words, and the fold run injects through it on both paths; the config repoint above stays
valid and is now redundant with it.)

Runtime: warm-cache (checkpoints present) scoring is seconds -- exact 1.7-5.0 s, fast 0.6-1.4 s per
arm (the fast path is consistently ~2.5-3.5x faster to score, the RTF win these numbers exist to
prove); the whole 3-arm file re-runs in ~12 s warm. Cold-cache training (once per arm, EXACT path)
dominates at ~60 s (SAD) / ~180 s (phSeq) / ~276 s (cep). Checkpoints cache under the gitignored
`data/phase7_parity_cache/` (holds corpus-path listings -- never committed); `run_baseline` at a
fixed seed is deterministic, so a warm cache is bit-identical to a fresh run.

## Phase 8 -- online/streaming mode

### Task 1 -- the frozen-norm reference mode + the causality-cost CI leg

`Audio_fixed_gain` (S1.1, the one sanctioned exact-tree touch: `read_audio` gains
`fixed_gain: Option<f64>`, `None` = the legacy path byte-identical -- full golden suite green) plus
the fast SAD path's type-1 external normalization (`fast/nn.rs::external_normalize_f32`, the
frozen-stats input mode; the phase-7 `InputNormalizationType` bail narrowed from {-1} to {-1, 1}).
The staged frozen gate config (`common::stage_frozen_tier2`, consumed by Tasks 2/3/5/6): channel 0
of the 60 s `phase4d/prcts_excerpt.wav` extracted MONO at staging, the tuple-A pack, tier2 +
`Inference_Path fast` + `Audio_fixed_gain <measured>` + `BLSTM_InputNormalizationType 1`. The gain
is baked to the mono channel's own measured `(2*RMS+max)/2` (= 4.924708e-1), so the audio-norm
halves of the frozen and self-norm modes coincide numerically on THIS fixture and the remaining
delta isolates the type-1-vs-self-norm input-normalization change alone.

The causality-cost CI leg (S1.7, REPORTED never gated;
`phase8_frozen_norm.rs::causality_cost_frozen_vs_self_norm`, measured 2026-07-19, Apple Silicon dev
box, 60 s mono, tuple-A pack BOTH sides per R4):

| quantity | offline-frozen (type 1 + fixed gain) vs offline-self-norm (type -1) |
|---|---|
| posterior max_abs delta | 9.956e-1 |
| boundary rows | frozen 2 (the seed hypothesis -- ZERO detections) vs self-norm 17 |
| NaN-pattern mismatches | 0 (identical overlap coverage) |

MECHANISM (adjudicated before recording, exact-tree cross-run: the exact f64 path under the SAME
frozen stats reproduces the IDENTICAL 2-row collapse, inter-path posterior delta 1.28e-7 = f32
noise -- the collapse is the MODE, not a fast-path defect): the tuple-A net was TRAINED under
type -1 self-normalization (per-sequence standardize + asinh); its pack-carried type-1 tail is a
plain affine standardization with 2015-training-corpus statistics, under which this net's posterior
saturates high (~0.9995) for the whole fixture -- no rising crossing ever fires. The causality cost
for THIS pack/config pairing is therefore total on the decision level; the honest number, stated as
measured. The corpus tier (Task 8) re-measures on the phase-6 SAD checkpoint (held-out DCF, both
modes scored). The fast type-1 transcription itself is pinned two ways: the unit pin
(`fast_type1_normalization_matches_exact`, fast f32 vs the exact `nn/blstm.rs:1008-1021` branch on
the same input + tuple-A tail: measured max_rel 1.913e-7, pinned 1e-5) and the causality leg's own
end-to-end run through the frozen bag.

### Task 8 -- the corpus tier: streaming parity + the causality cost on REAL data

`tests/pyo3/test_phase8_parity.py` (corpus-gated, `slow`, local-only) lifts the Rust CI gate
(`phase8_gate.rs`, streamed == offline-frozen BIT-EQUAL on the committed 60 s tuple-A fixture) onto
REAL LRE03/07 data with the TRAINED phase-6 SAD subset checkpoint (`best_sad.bin`, the SHARED
phase-7 parity cache -- warm-cache reuse, no re-train), through the Python `speech_rs` seam. The
file is selected at RUNTIME as the lexicographically-first `*.wav` under `train/audio/**` (the
license bright line -- no filename/path recorded); it is a 75 s mono 8 kHz recording. Measured
2026-07-20, Apple M4 Pro (arm64), macOS 26.5.2, seed 0 (the phase-6 recipe).

THE TAIL FINDING (`test_frozen_tail_is_identity`): the checkpoint's normalize tail (last
`2*23=46` pack elements of the 33,671-element pack) is EXACTLY identity (`max|mean|=0`,
`max|std-1|=0`) -- `init_weights` seeds it identity and type -1 training never descends it (the
frozen tail is not the descent target). So `BLSTM_InputNormalizationType 1` is a NO-OP on this
pack and the frozen posteriors are driven by UN-normalized input. Under this net's `IgnoreFirstDCT`
+ no-LTSV/TDC config the per-file audio gain is moreover DECISION-INVARIANT (the DCT cancellation,
`phase8_frozen_norm.rs` finding 1), so the causality cost isolates the type-1-vs-self-norm INPUT
normalization alone. EVIDENTIARY SCOPE (stated explicitly): because that tail is EXACTLY identity,
`(x - 0)/1` is INDISTINGUISHABLE from skipping type-1 entirely, so this corpus equivalence does
NOT independently exercise the type-1 threading through the streaming front-end -- that coverage
lives in Task 1's fast-vs-exact unit pin (`fast_type1_normalization_matches_exact`), which runs the
type-1 branch against a NONZERO tuple-A tail (max_rel 1.913e-7, pinned 1e-5). This corpus leg pins
the streaming-vs-offline equivalence and the causality-cost regime, not the type-1 arithmetic.

**S1.9 EQUIVALENCE (`test_streaming_equivalence_on_corpus`)** -- streamed (PyO3 session, 100 ms
chunks) vs the offline-frozen fast `Engine` run on the SAME file/pack/gain (fixed_gain 5.805032e-1):

| quantity | streamed | offline-frozen | verdict |
|---|---|---|---|
| speech segments | 1 | 1 | IDENTICAL count (R1 gate) |
| speech interval | `[0, 74.999875]` | `[0, 74.9999]` (VRCTS `%f.4`) | boundary max_dt 2.500e-5 s, WITHIN the VRCTS 4-decimal write quantum (a quarter of the 1e-4 write resolution, half the 5e-5 half-quantum tolerance); at the 4-dp comparison grain the interval SETS are IDENTICAL (0.0) |
| prefix consistency | emitted set == final partition (1 == 1, no retraction/re-emission) | | PASS |
| chunk invariance | 100 ms vs 101-sample granularity -> identical segmentation + emitted set | | PASS |

The frozen SAD net collapses to ALL-SPEECH (the whole 75 s file is one speech segment) -- the same
mode-collapse the phase-6 subset checkpoint carries (RESULTS.md's SAD subset-gate reading: the net
fires everywhere, Pmiss ~0). The equivalence is EXACT: the only Python-side gap is the offline
segmentation being observable solely through the engine's `%f.4` VRCTS dump, so the streamed
full-precision boundary and the offline rounded boundary differ by LESS than one 4-decimal quantum
(2.500e-5 < 1e-4) and coincide bit-for-bit at that grain. A COUNT mismatch or a boundary delta
beyond the quantum is an R1 STOP -- neither occurred.

LATENCY (S1.8, recorded): max_lag 0.0000 s, mean_lag 0.0000 s. The all-speech collapse yields a
single segment finalized at EOS (`emitted_at == end_s`), so the measured lag is degenerately zero
-- honestly recorded, not illustrative on this file; the mid-stream latency budget is exercised on
the crafted-posterior Rust gate (`phase8_gate.rs::latency_bounds`). The DERIVED structural bound is
6.05357 s (feature_reach 0.14400 + nn_window 3.26000 + conv_delay 0.36000 + holdback 2.28957),
cited from that gate: `lre_sad.toml` is seeded VERBATIM from the SAME `1_worker_1.config` as the
tier2 gate config, so every bound-relevant key is byte-identical; the test RECOMPUTES the
value-dependent holdback from THIS config (2.28957, clamped-negatives sum of
min_speech/min_silence/speech_padding) and cross-checks it, proving the lineage. Timings:
stream 0.04 s, offline 0.04 s (75 s audio, warm cache).

**S1.7 CAUSALITY COST (REPORTED, never gated)** -- offline-frozen vs offline-self-norm:

| leg | self-norm (type -1, per-file gain) | offline-frozen (type 1 + fixed gain) | delta |
|---|---|---|---|
| single-file speech coverage | 75.000 s (1 seg) | 75.000 s (1 seg) | **0.000 s** |
| held-out pooled DCF (24 files, all collars 0/0.25/0.5/1/2 s) | 0.2500 | 0.2500 | **+0.0000** |

MECHANISM (named, honest): this from-scratch subset checkpoint mode-collapsed to ALL-SPEECH during
training (the documented phase-6 SAD behavior -- "the net jumps from all-non-speech straight to
all-speech"). An all-speech-saturated net fires everywhere under BOTH input normalizations, so the
DECISION (all speech, Pmiss ~0 / Pfa ~1, DCF 0.25) is normalization-INVARIANT here and the
causality cost on both the boundary and the held-out DCF is EXACTLY ZERO. This is the honest
contrast to the Task-1 tuple-A leg, where the selective 2015 production net's causality cost was
TOTAL (frozen collapsed to the all-non-speech seed while self-norm gave 17 boundaries): the cost is
REGIME-DEPENDENT -- zero for a collapsed subset net, total for a selective one. Both are the honest
measured number for their pack. A genuinely selective net (the full-corpus launcher's job) is where
a non-degenerate corpus causality cost would surface. The single `Audio_fixed_gain` scores all 24
held-out files consistently because the gain is decision-invariant for this config (the DCT
cancellation above), so one global gain equals a per-file gain on the decision. Sanity holds (both
DCFs finite, in [0,1], same order of magnitude -- here identical). Timings: self-norm scoring
5.51 s, frozen 5.36 s (24 files each, warm cache).

**REVISED BY PHASE 9 TASK 9 (read that section before quoting the latency numbers above).** The
phase-8 emission frontier had a third, missing term -- an OPEN (un-closed) raw segment, whose begin
sits behind the consumed frontier -- and the phase-9 corpus streaming tier caught it as a real R2
retraction on a trained causal net. The fix (`fast/stream.rs::resmooth_and_emit`, clamp the
frontier to `hyst.pending_begin()`) makes the SPEECH-class structural bound CONDITIONAL: a speech
segment whose following silence is interrupted by a raw speech segment reopening within the
holdback now waits for that segment to close, inheriting the OTHER class's commit-wait area term.
On the phase-8 GATE fixture (the crafted-posterior one the bound above is cited from -- not this
corpus file, whose all-speech collapse emits nothing mid-stream at all) that moves ONE of six
speech emissions -- the segment `[3.33710, 9.79560]`,
measured in BOTH regimes at 5.60428 -> 16.80428 s, still inside the 18.11 s commit-wait ceiling;
the other five are unmoved, and the 5.84457 s headline max above belongs to one of THOSE (the
segment `[41.35350, 43.95530]`), not to the blocked one. Nothing else in this section changes --
the equivalence, prefix, chunk-invariance, and causality-cost results are all unaffected (they
concern `finish`, which the fix never touches).

---

## Phase 9 -- new architectures

The Phase 9 record: the new recurrent cells (sLSTM, Mamba) in both directions
(bidirectional, forward/causal), trained FROM SCRATCH through the phase-5/6 machinery and
compared against the phase-6 BLSTM baseline. Two sections, both landed: **Task 8** -- the four
from-scratch subset gates, their sizing/preflight tables, and the dead-input-column finding;
**Task 9** -- the corpus tier (the phase-8 emission-frontier defect it found, the causal
streaming equivalence, the derived causal latency bound, the first non-zero causality cost,
fast-vs-exact metric parity) plus the RTF/peak-RSS bench rows. The full-corpus headline runs
stay POST-PHASE, user-fired (the launcher recipe is under Task 8).

### Task 8 -- the four from-scratch subset gates (SAD arm, corpus-gated)

`tests/pyo3/test_phase9_gates.py` (corpus-gated, `slow`, local-only). Spec S8.2, the
phase-6 SAD protocol VERBATIM -- same arm (`configs/training/lre_sad.toml`, Algo 3
spectral, File_Type 0 wav), same recipe (subset 10 / valid 8 / test 24, 3 epochs x 10
SMORMS3 steps, 20 s audio cap, seed 0, `val_metric=nn_cost_seg`), same end-to-end scorer
(engine VRCTS hyp dumps + the `.part.xml` refs -> pooled `evaluate.dcf`). ONLY the cell and
the direction differ, driven by the T4 `--cell` / `--direction` knobs on
`python -m speech.drivers.baseline sad`. Measured 2026-07-28, Apple M4 Pro (arm64), macOS 26.5.2, N=1 lane.

ONE RIDER on "only the cell and the direction differ", so the wall-time column below is not
misread as a speed result: `--direction forward` necessarily changes TWO things, because a
causal net cannot run the windowed regime meaningfully (a window boundary resets the
recurrent state, spec S1.2). `cell_overlay` therefore forces `BLSTM_window 0` on every
forward run -- the plain WHOLE-SEQUENCE regime -- while the bidirectional rows keep
`frame_window 3.25`'s windowed overlap. The forward rows are consequently faster for a
reason that is NOT the cell (no per-window recompute, half the recurrent stack, a smaller
output MLP), and none of these wall times is a controlled RTF measurement. The controlled
speed comparison is the fast-path bench (a later phase-9 task), not this table.

#### Sizing (spec S8.2 param match, +-15%) -- HOLDS AT THE S6 DEFAULTS, no resizing

The arm topology is fixed by the config (`LSTMNeuronNb 23,24,24` + `LSTMSubSampling 4,1` ->
two cell layers, layer 0 fed `4*23 = 92` stacked frames; `OutputNeuronNb 48,12,1`; a
`2*23 = 46` normalize tail). A cell swap changes only the per-layer block:

| cell | per-layer block | per-direction stack | bidirectional pack | vs BLSTM | forward pack | vs BLSTM |
|---|---|---|---|---|---|---|
| LSTM (baseline) | `4*out*(out+in) + 12*out + 4*out` | 11520 + 4992 = 16512 | **33671** | -- | **16871** | -- |
| sLSTM | `4*out*(out+in+1)` (S2.2) | 11232 + 4704 = 15936 | **32519** | **-3.42%** | **16295** | **-3.41%** |
| Mamba | S3.2 blocks, `d_inner = 2*24 = 48`, `d_state 16`, `d_conv 4`, `dt_rank auto = 2` | 8544 + 6312 = 14856 | **30359** | **-9.84%** | **15215** | **-9.82%** |

`pack = D * stack + MLP + 46`, `D = 2` bidirectional / `1` forward; MLP = 601
(bidirectional, input `2*24`) or 313 (forward -- `cell_overlay` rewrites
`OutputNeuronNb 48,12,1 -> 24,12,1` because there is no reverse half to concatenate). Both
cells land inside +-15% at the spec's own default geometry, so nothing was resized. T4
cross-pinned every length against the Rust `BlstmNetwork::nb_of_weights()`; 33671 is also
the 2015 production tuple-A pack size, an independent anchor on the unchanged LSTM path.

**LIVE-parameter caveat (surfaced by the preflight, honest):** these are NOMINAL pack
lengths, and the arm carries a block of STRUCTURALLY DEAD weights that the three cells do
NOT share proportionally -- see the dead-input-column finding below. Counting only weights
with a nonzero gradient, the ordering inverts for Mamba: LSTM 24409, sLSTM 23257 (-4.72%),
Mamba 28009 (**+14.75%**). Mamba is the smallest net nominally and the LARGEST net
effectively, because the dead columns hit a `4 gates x out x 48` block in the gate cells but
only a single `out x 48` input-projection block in Mamba. Both readings are inside a
loose param-match reading of S8.2 (+-15%), but the nominal table alone would misdescribe
the comparison -- recorded rather than papered over.

#### Preflight -- the log-law saturation hazard, measured before firing the runs

The arm trains under `CostLaw log/log`, whose forward `b + a*ln(clamp(y/adim, 1e-24, 1))`
is CONSTANT wherever the argument clamps -- so F7's consistent derivative there is EXACTLY
ZERO and a saturated from-scratch net is a PERMANENT STALL, not noisy descent. Every config
was probed at its from-scratch theta (one forward+backward through the seam, no training)
before the training runs were fired. Every value below is measured at the COMMITTED
preflight leg's own recipe (subset 2, 10 s cap, seed 0); `%clamp` is the column to its left
over the clamp constant:

| config | pack | init NNCostSeg | %clamp | worst per-file cost | %clamp | grad L2 | grad Linf | nonzero grad | init decision |
|---|---|---|---|---|---|---|---|---|---|
| lstm / bidirectional (baseline) | 33671 | 0.30034 | 0.543% | 0.42557 | 0.770% | 0.46050 | 0.20256 | 24409/33671 | all-non-speech |
| lstm / forward | 16871 | 0.26899 | 0.487% | 0.38775 | 0.702% | 0.43706 | 0.18776 | 12217/16871 | all-non-speech |
| sLSTM / bidirectional | 32519 | 0.26942 | 0.488% | 0.37362 | 0.676% | 0.35736 | 0.18957 | 23257/32519 | all-non-speech |
| sLSTM / forward | 16295 | 0.27500 | 0.498% | 0.40237 | 0.728% | 0.37443 | 0.19126 | 11641/16295 | all-non-speech |
| Mamba / bidirectional | 30359 | 0.31969 | 0.578% | 0.41293 | 0.747% | 0.72178 | 0.18946 | 28009/30359 | all-non-speech |
| Mamba / forward | 15215 | 0.31930 | 0.578% | 0.44149 | 0.799% | 0.49350 | 0.19512 | 14017/15215 | all-non-speech |

CORRECTION (review): an earlier revision of this table published a `per-frame cost` column of
0.04888 / 0.2 -- those were NOT costs. They were `100*Pmiss/counter`, read from the wrong
column of the PREFIXED `results_matrix()` (`[file+1, conf+1, chan+1, res...]`, so the res
block must be sliced off before applying the per-res convention). The tell was visible in the
published numbers themselves: a per-file max BELOW the frame-weighted mean is arithmetically
impossible, and the value was identical across three architectures. The true per-file costs
are the column above; the committed leg now slices as `engine.py::_error_vad` does and
asserts `per_file_max >= aggregate_mean` so the confusion cannot recur.

The clamp constant is `-ln(1e-24) = 55.262`; every init sits at ~0.5% of it (worst 0.58%),
every INDIVIDUAL file at <= 0.80%, i.e. deep in
the law's interior, and every epoch-0 gradient norm is far from zero. NO config started in
the zero-gradient death, and none needed a scheme-constant change (the S2.4/S3.4 constants
are spec text and were not touched). The T6 observation that a seeded Mamba fixture's
posteriors saturate the output logistic did NOT reproduce as a training hazard on this
net-sized arm: Mamba's init cost is interior and its gradient norm is the LARGEST of the
six. The init decision is all-non-speech for all six -- the same starting regime the
phase-6 BLSTM arm documented, which is why the beat-init leg is the direction-safe metric.
The preflight is a committed leg (`test_init_is_trainable`), not a one-off. EVIDENTIARY
SCOPE, stated: the seam exposes the cost and the gradient, not the posterior vector (that
is a Rust `test-support` hook), so "not saturated" here means COST-INTERIOR (in aggregate AND
per file -- the aggregate is frame-weighted, so the committed leg also pins the worst
per-file normalized cost, measured 0.374-0.442 across the six configs, i.e. <= 0.80% of the
clamp) + GRADIENT-NONZERO + the net subsequently trains -- the observables that decide
whether the log law's zero-gradient region bites -- not a directly measured posterior range.

**DEAD INPUT COLUMNS (a PRE-EXISTING arm property, found while explaining the nonzero-grad
column above; not a phase-9 regression and not touched here).** The `nonzero grad` counts
are exactly accounted for. `lre_sad.toml` sets `NNetInputSize 23`, but the DSP front-end it
also specifies produces an **11**-wide feature vector: `mel.rs`'s width law for
`compute_deltas_nb > 0` is `(dd_nb > 0 ? 3 : 2) * nb_dct - ignore_first_dct`, i.e.
`3*4 - 1 = 11` (4 statics + 4 deltas + 4 delta-deltas = 12, minus the dropped c0;
`compute_deltas_nb`/`compute_delta_deltas_nb` are regression ORDERS, not column counts --
phase-8's `reach = 5+3 = 8` is their chained temporal reach). With
`LSTMSubSampling 4`, layer 0's fan-in is sized `4*23 = 92` but only `4*11 = 44` columns
ever carry data, so the trailing 48 columns of every layer-0 fan-in row are never read.
That predicts, per direction count `D` (2 bidirectional / 1 forward), exactly:

    LSTM / sLSTM  D x 4 gates x 24 units x 48 cols + 46 (frozen normalize tail)  bi 9262 / fwd 4654
    Mamba         D x 1 input projection x 24 x 48 + 46                          bi 2350 / fwd 1198

and the measured zero counts are 9262 / 9262 / 2350 bidirectional and 4654 / 4654 / 1198
forward -- exact, with the zeros landing as a contiguous `[44, 92)` tail in every row (the
engine's documented input-width tolerance crop absorbing the mismatch silently). Index-set
checked, not just counted: for sLSTM in both directions the measured zero SET equals the
predicted set (extra 0, missing 0). One coincidence exists and is NOT structural -- at a
wider probe recipe (subset 4 / 20 s cap) sLSTM/forward reads 4655, one live weight's gradient
landing on exactly 0.0 on that data; it is not a `b_i` slot (the same recipe leaves sLSTM
bidirectional at exactly 9262), so the committed guard
(`test_init_is_trainable`) pins the count as a FLOOR plus bounded slack rather than an
equality. So ~27% of the BLSTM/sLSTM arm's weights are structurally
untrainable, and the config comment claiming `nnet_input_size 23` equals the produced
feature dimension is wrong. This is inherited VERBATIM from the 2015 production
`1_worker_1.config` (33671 is that net's pack size), so the 2015 production SAD net carried
the same dead block -- a legacy property, not a port bug. Left UNCHANGED deliberately:
correcting `NNetInputSize` would resize every pack and invalidate the phase-6 comparison
baseline this section is measured against.

#### The four gates -- HARD leg: trained held-out DCF beats own init

<!-- ledger:table phase9_v1_cells -->
| cell / direction | files (train/valid/test) | trained DCF@0.5 | Pmiss / Pfa @0.5 | trained collar range | init DCF@0.5 | init collar range | gain@0.5 | wall |
|---|---|---|---|---|---|---|---|---|
| LSTM / bidirectional | 10 / 8 / 24 | 0.250000 | 0.000000 / 1.000000 | 0.250000 (all 5) | 0.750000 | 0.750000 (all 5) | +0.500000 | 47 s |
| sLSTM / bidirectional | 10 / 8 / 24 | 0.250000 | 0.000000 / 1.000000 | 0.250000 (all 5) | 0.750000 | 0.750000 (all 5) | +0.500000 | 44 s |
| sLSTM / forward | 10 / 8 / 24 | 0.250000 | 0.000000 / 1.000000 | 0.250000 (all 5) | 0.750000 | 0.750000 (all 5) | +0.500000 | 17 s |
| Mamba / bidirectional | 10 / 8 / 24 | 0.250000 | 0.000000 / 1.000000 | 0.250000 (all 5) | 0.750000 | 0.750000 (all 5) | +0.500000 | 68 s |
| Mamba / forward | 10 / 8 / 24 | 0.249625 | 0.000000 / 0.998498 | [0.249259, 0.250000] | 0.750000 | 0.750000 (all 5) | +0.500375 | 21 s |
| full-corpus runs (all cells) | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

Hosts: Apple M4 Pro (arm64, 14 cores, Darwin 25.6.0).
<!-- ledger:end -->

All four HARD legs pass at every collar (0 / 0.25 / 0.5 / 1 / 2 s), deterministically:
run-twice at a fixed seed gives bit-identical `best_sad.bin` / `last_sad.bin` bytes and an
identical pooled DCF, on the short determinism recipe (all four cells) and on the FULL
gate recipe (spot-checked on Mamba/bidirectional, the slowest and most complex). The
causal variants converge honestly too -- the streaming artifact is not a training
regression.

**vs BLSTM -- RECORDED, NOT GATED (spec R5, verbatim): "subset vs-BLSTM numbers are
reported as-is with the thinness caveat; no architecture-superiority claim is made from
subset scale."** All four new configs TIE the phase-6 BLSTM arm's 0.2500, and that tie is
exactly what an identical mode collapse looks like: on a 10-file subset every from-scratch
SAD net jumps from all-non-speech (init, Pmiss 1.0, DCF 0.75) straight to all-speech
(trained, Pmiss 0.0, Pfa 1.0, DCF 0.25) -- the documented phase-6 behavior, with no
partial-discrimination sweet spot at this scale. A cell could tie 0.2500 by collapsing
identically, so the tie is NOT evidence of architectural equivalence and no ranking is
claimed from it. The one row that is not exactly degenerate is Mamba/forward (Pfa 0.998498
-- it correctly rejects a sliver of non-speech, landing 0.000375 below the all-speech
baseline); recorded as measured rather than rounded into the collapse story, and far too
small to read as an architectural signal. Genuine speech/non-speech discrimination -- a DCF
below the all-speech 0.25 line -- is the FULL-CORPUS user-fired launcher's job, and those
runs are the referee for any cell-vs-cell claim.

**The CE is not the signal** (the phase-6 lesson, sharper on Mamba): Mamba/bidirectional's
per-epoch train cost ASCENDS 1.090 -> 1.263 -> 2.073 across the three epochs while the
held-out DCF still lands at 0.2500, and its validation cost falls 4.675 -> 0.527 -> 0.429
over the same epochs. Only the held-out TASK metric is gated.

#### Firing the new-cell arms (post-phase, user-fired)

`experiments/02_v1_cells.sh` fires the four rows (any subset by name, e.g.
`sad/slstm/bidirectional`); each record promotes into the `full-corpus runs` row above.

`--direction forward` additionally forces `BLSTM_window 0` (the plain whole-sequence causal
regime -- a window boundary would reset the recurrent state) and resizes the output MLP's
input width; both are automatic (`drivers/baseline.py::cell_overlay`).

### Task 9 -- the corpus tier (causal streaming, causality cost, fast-vs-exact) + the bench rows

`tests/pyo3/test_phase9_parity.py` (corpus-gated, `slow`, local-only) lifts BOTH phase-9 Rust CI
gates -- `phase9_fast_parity.rs` (exact-causal vs fast-causal offline) and
`phase9_stream_causal.rs` (streamed `finish()` bit-equal to the offline fast causal run) -- onto
real LRE03/07 data with the T8 FROM-SCRATCH CAUSAL checkpoints, through the `speech_rs` seam. The
structure is the phase-7 (`test_phase7_parity.py`) + phase-8 (`test_phase8_parity.py`) corpus
tiers, on phase-9 artefacts. Checkpoints train under T8's `_GATE` recipe VERBATIM (subset 10 /
valid 8 / test 24, 3 epochs x 10 SMORMS3 steps, 20 s cap, seed 0) into the gitignored
`data/phase9_parity_cache/` (cold train 14.4 s sLSTM / 17.6 s Mamba; warm-cache whole file 10 s).
The streamed file is selected at RUNTIME as the lexicographically-first `*.wav` under
`train/audio/**` (the license bright line -- no filename recorded); it is the same 75 s mono 8 kHz
recording the phase-7/8 corpus legs used. Measured 2026-07-28, Apple M4 Pro (arm64), macOS 26.5.2.

#### A REAL DEFECT, FOUND HERE: the emission frontier ignored an OPEN raw segment (R2)

The mamba/forward streaming leg RETRACTED on its first run: `[0, 7.3051] Speech` was emitted
mid-stream at 10.1999 s, and `finish` then reported ONE `[0, 74.9999]` segment -- the emitted
boundary had moved. Adjudicated (never widened, spec R1/R2) to a genuine PHASE-8 bug in
`fast/stream.rs::resmooth_and_emit`, inherited unchanged by the phase-9 causal arm:

- The emission frontier was `max(last_raw_boundary, consumed_frontier - dt)`, and its safety
  argument concluded "future raw structure lands no earlier than the frontier".
- That is FALSE while the hysteresis is INSIDE an un-closed raw segment. Such a segment's begin
  lies BEHIND the consumed frontier and is not yet in `raw_segments`, so neither term sees it;
  when it closes, its `add_padding` `before` reach can merge it into the previous speech segment
  and MOVE a boundary the trigger already emitted. Here speech resumed ~1.3-2.6 s after the 7.3051
  end -- inside the 2.28957 s holdback -- and the final smoothing merged the two.
- FIX: a THIRD frontier term, `hyst.pending_begin()` (the open/tentative begin), clamped in with
  `min`. It costs LATENCY, never correctness -- clamping only emits LESS.
- PINNED: `phase8_stream_decision.rs::pending_open_segment_blocks_emission` + the new `reopen`
  profile (burst A, a 1.6 s merging gap, an 8 s burst B that stays open past A's emission point).
  MUTATION-VERIFIED: reverting the clamp fails three legs (`pending_open_segment_blocks_emission`,
  `prefix_consistency_holds`, `chunk_invariance`).
- COVERAGE LESSON, recorded honestly: with the clamp reverted the ENTIRE phase-8 gate stays GREEN
  (19/19) and so does the phase-9 causal gate (19/19). Neither synthetic fixture nor the crafted
  60 s tuple-A fixture ever produced the retraction -- only real data with a real trained causal
  net did. That is the corpus tier earning its keep.

**PHASE-8 LATENCY CLAIM, REVISED (a spec S1.8/S5.5 deviation surfaced, not absorbed).** The
SPEECH-class structural bound is CONDITIONAL, not universal: a speech segment whose following
silence is interrupted by a raw speech segment reopening within the holdback must wait for that
segment to close, inheriting the same data-dependent commit-wait area term the OTHER class always
carried. On the phase-8 calibrated fixture exactly one of six speech emissions is so blocked
(a raw segment opens at 12.01876, 2.2232 s after the 9.79560 end, inside the 2.28957 holdback).
That blocked emission is the speech segment `[3.33710, 9.79560]`, and BOTH REGIMES WERE
INSTRUMENTED to attribute the transition: its lag goes **5.60428 -> 16.80428 s**, still inside the
commit-wait ceiling `max_speech_dur + pipeline_forward + 1.0 = 18.11`. The other five speech
emissions are unmoved (5.39698 / 5.49408 / 5.54447 / 5.69858 / 5.84457 s, all <= the 6.05357
bound). NOTE, because an earlier revision of this section got it wrong: **5.84457 is a DIFFERENT
segment** (`[41.35350, 43.95530]`) -- it is the phase-8 headline SPEECH max and it is UNMOVED by
this fix; it is not the blocked segment's pre-fix value. `phase8_gate.rs::latency_bounds` now pins
the unblocked MAJORITY at ~bound plus the commit-wait ceiling on the max;
`settled_speech_emits_during_silence` became EXISTENTIAL over the gaps wider than the bound (the
7.50160 s gap carries the win at lag 5.69858) and records the widest gap's blocked case.

**THE HOLDBACK STAYS (T9 review, adjudicated).** The block is by 0.0664 s only, because `holdback`
(2.28957) deliberately over-covers the TRUE leftward reach (1.68510, the T4-review derivation --
the difference is exactly the rightward-only after-paddings), and the tempting move is to give the
PENDING term that tighter constant. Ruled out: the pending term is a FRONTIER bound handled exactly
like the other two (holdback is subtracted once from their `min`), so there is no per-term
conflation to fix, and a per-term constant would put a second underived, unpinned number on the
exact code path -- when it is precisely a tight-reach argument that was just proved incomplete
here. The only checkable claim is that the tighter reach would have kept THIS fixture unblocked;
whether it would also have caught the corpus retraction is NOT established (the natural comparison
mixes reference frames -- a raw-end measurement against a smoothed-end constant). If the reach is
ever tightened it is a PHASE-LEVEL change: tighten `holdback` GLOBALLY across all three frontier
terms, derive it code-side, pin it, and re-prove leftward dominance. Never a per-term second
constant.

#### The frozen-overlay contract on a `-1`-trained pack

The arm trains under `BLSTM_InputNormalizationType -1` (per-sequence self-normalization), which the
streaming session ALWAYS refuses (it needs the whole sequence before the first frame). Resolution,
evidenced rather than asserted: the trained pack's normalize tail is EXACTLY identity
(`test_frozen_tail_is_identity`, both cells: `max|mean| = 0`, `max|std-1| = 0` over the last
`2*23 = 46` elements of the 16295 / 15215 packs) because `init_weights` seeds it identity and type
-1 training never descends it -- so the streamable type-0 arm and the type-1 frozen-affine arm are
the SAME map here. `test_frozen_overlay_is_type1_equivalent` pins that end to end: same pack, same
file, same fixed gain, `InputNormalizationType 0` and `1` produce the IDENTICAL VRCTS partition on
both cells. `BLSTM_window 0` needs no overlay -- `cell_overlay` already forces it on every
`--direction forward` run. EVIDENTIARY SCOPE (the phase-8 wording, unchanged): because the tail IS
identity, this leg does NOT exercise the type-1 arithmetic through the causal front-end; that
coverage is the Rust gate's calibrated NON-identity tail
(`phase9_stream_causal.rs::stream_finish_equals_offline_causal_frozen_type1`).

#### S5.6 on real data -- streamed vs offline fast causal

Streamed (PyO3 `StreamingSession`, 100 ms chunks) vs the offline fast CAUSAL `Engine` run on the
SAME file / pack / overlaid config (`Inference_Path fast`, `Audio_fixed_gain 5.805032e-01`,
`InputNormalizationType 0`):

| cell / direction | speech segs streamed vs offline | boundary max_dt | 4-dp interval sets | prefix | chunk invariance | max_lag |
|---|---|---|---|---|---|---|
| sLSTM / forward | 1 vs 1 (IDENTICAL) | 2.500e-05 s (< the 5e-05 VRCTS half-quantum) | IDENTICAL | no retraction, no re-emission | 100 ms == 101 samples | 0.0000 s |
| Mamba / forward | 1 vs 1 (IDENTICAL) | 2.500e-05 s | IDENTICAL | no retraction, no re-emission | 100 ms == 101 samples | 0.0000 s |

Both nets collapse to ALL-SPEECH on this file (one segment spanning the whole 75 s) -- the same
mode collapse the thin subset checkpoints carry everywhere (see the T8 gate table). The residual
2.5e-05 s is purely the offline segmentation being observable only through the engine's `%f.4`
VRCTS dump; at that write grain the interval SETS are bit-identical. A count mismatch or a delta
past the quantum is an R1 STOP -- neither occurred. Both `max_lag` values are degenerately 0
(a single segment finalized at EOS), honestly recorded: post-fix nothing is emitted mid-stream on
this file at all, which is exactly the correct behaviour for a net that never stops speaking.
Timings: stream 0.03 s, offline 0.02 s per cell.

**THE DERIVED CAUSAL LATENCY BOUND on this arm (spec S5.5), recomputed from the config:**

    feature_reach 0.14400 + nn_window 0.00000 + sub_sample 0.03000 + conv_delay 0.36000
                                                      + holdback 2.28957  =  2.82357 s

versus the phase-8 WINDOWED arm's 6.05357 s on the byte-identical `1_worker_1.config` lineage --
a **2.14x structural improvement**, and the `nn_window` term (3.26000 s) DIES entirely because a
causal cell has no lookahead. The test cross-checks the three lineage-shared summands
(feature_reach / conv_delay / holdback) against the phase-8 constants bit-for-bit; only
`sub_sample` (the causal decimation buffering) is new. (The phase-9 synthetic gate's own bound is
1.73400 s on its own smaller fixture -- a different config, not comparable.)

#### The causality cost, MEASURED (REPORTED, never gated)

Same trained pack, T8 held-out slice (24 files), scored under (a) its NATIVE type -1 self-norm +
per-file `(2*RMS+max)/2` gain -- what training and the T8 gate measured, and what is NOT streamable
in principle -- and (b) the FROZEN streamable regime (`Audio_fixed_gain` + type 0):

| cell | regime | DCF @ 0 / 0.25 / 0.5 / 1 / 2 s | Pmiss / Pfa @0.5 | delta (frozen - self) @0.5 |
|---|---|---|---|---|
| sLSTM / fwd | self-norm (-1) | 0.250000 / 0.250000 / 0.250000 / 0.250000 / 0.250000 | 0.000000 / 1.000000 | -- |
| sLSTM / fwd | frozen (type 0 + gain) | 0.251607 / 0.251627 / 0.251774 / 0.252623 / 0.252623 | 0.003497 / 0.996607 | **+0.001774** |
| Mamba / fwd | self-norm (-1) | 0.249259 / 0.249367 / 0.249625 / 0.250000 / 0.250000 | 0.000000 / 0.998498 | -- |
| Mamba / fwd | frozen (type 0 + gain) | 0.248000 / 0.247357 / 0.246641 / 0.245562 / 0.245254 | 0.000000 / 0.986564 | **-0.002984** |

The two sides differ in the THREE normalization keys and nothing else: `Audio_offset` /
`Audio_max_duration` stay the arm's own (the T8 recipe's 20 s cap) on BOTH sides. A first
revision of this leg applied a whole-file overlay to the frozen side only -- it scored the FULL
held-out files against the native side's first 20 s, i.e. a duration artefact wearing a causality
cost's clothes. Caught in self-review by a 10x scoring-time asymmetry (2.76-3.63 s vs 0.31-0.36 s);
the corrected legs run 0.32 / 0.32 s and 0.40 / 0.41 s. Recorded because the tell generalizes: on
a comparison this small, a wall-clock asymmetry between two supposedly-identical workloads is the
cheapest available check that the two sides really are identical.

MECHANISM (named, honest). This is the FIRST non-zero causality cost this repo has measured: the
phase-8 corpus leg reported EXACTLY 0.0 because its checkpoint was a total all-speech collapse
(decision normalization-invariant), and the phase-8 Task-1 tuple-A leg reported a TOTAL cost
because the selective 2015 net collapsed under freezing. These two causal checkpoints sit in
between -- near-collapsed but not exactly (Mamba already rejects a sliver of non-speech, Pfa
0.9985) -- so switching the input normalization moves a handful of frame decisions, and the cost is
both TINY and SIGN-VARYING: +1.8e-3 DCF for sLSTM (frozen worse: it starts MISSING a little
speech, Pmiss 0.0000 -> 0.0035), -3.0e-3 for Mamba (frozen BETTER, rejecting more non-speech:
Pfa 0.9985 -> 0.9866). A sign-varying sub-1e-2 delta on a mode-collapsed pair is NOISE around a
degenerate operating point, NOT evidence that freezing helps; a genuinely selective full-corpus net
remains where a real causality cost would surface. The single `Audio_fixed_gain` scores all 24
files consistently because the gain is decision-invariant on this all-DCT / `IgnoreFirstDCT`
front-end (the phase-8 DCT-cancellation finding). Timings: 0.32 s / 0.32 s (sLSTM) and 0.40 s /
0.41 s (Mamba), native / frozen, 24 files each.

#### S4.3 on real data -- fast vs exact causal metric parity

Each T8 checkpoint scored on its own 24-file held-out slice under both `Inference_Path` values in
its NATIVE config (type -1, `BLSTM_window 0`, no fixed gain) -- identical configs except
`Inference_Path` and the per-path `Dump_Directory` the scorer points at its own VRCTS output (that
second key is a write TARGET, read by nothing in the compute path, and it must differ or the two
runs would overwrite each other's hyps):

| cell / direction | files | per-file decision agreement | DCF delta (fast - exact) | exact DCF@0.5 (T8-pinned) |
|---|---|---|---|---|
| sLSTM / forward | 24 | VRCTS boundaries identical (count + types + times, max_dt 0.0 s) | **0.0 at every collar** | 0.250000 |
| Mamba / forward | 24 | VRCTS boundaries identical (count + types + times, max_dt 0.0 s) | **0.0 at every collar** | 0.249625 |

Float equality, not tolerance -- the phase-7 prediction (identical decisions => identical metrics)
reproduces on the causal cells. The exact-path DCF@0.5 column doubles as the checkpoint-provenance
pin against T8's own gate numbers, so a stale cache or a recipe drift fails loudly here instead of
letting the parity comparison pass against the wrong net. Scoring: exact 0.83-0.91 s, fast
0.33-0.39 s per cell (a ~2.4x Python-level scoring speedup, consistent with the phase-7 tier).

#### S8.7 -- the bench rows (RTF + peak RSS), causal cells vs the BLSTM baseline

`speech bench --path=exact|fast` on ONE 75 s mono 8 kHz corpus wav (runtime sorted-first, no
filename recorded), each arm's config repointed at its own trained checkpoint, `Audio_max_duration`
lifted, backprop off, `numOuterThreads 1`. Local one-off measurement, same posture and license
discipline as the phase-7 corpus-gated bench legs (not a committed automated test). Apple M4 Pro
(arm64), macOS 26.5.2, `cargo build --release` (LTO on). Headline table = 3 INDEPENDENT
`--repeat=1` processes (the phase-7 recipe, so `maxrss` is a clean per-process high-water mark
rather than an accumulating one); `--repeat=3` single-process runs were taken too and agree on
wall-clock to within the ranges below.

| cell / direction | window regime | path | wall_s mean [range] | rtf | maxrss_mb | fast vs exact |
|---|---|---|---|---|---|---|
| lstm / bidirectional (phase-6 baseline) | 3.25 windowed overlap | exact | 0.163284 [0.158799-0.171087] | 0.002177 | 53.73 | baseline |
| lstm / bidirectional (phase-6 baseline) | 3.25 windowed overlap | fast | 0.035443 [0.035162-0.035613] | 0.000473 | 68.56 | **4.61x** |
| lstm / forward (control) | 0, plain | exact | 0.098132 [0.097867-0.098532] | 0.001308 | 54.84 | (was n/a; FILLED by phase-10 T8 -- **5.97x**, re-measured, see below) |
| slstm / forward | 0, plain | exact | 0.096118 [0.095678-0.096488] | 0.001282 | 56.98 | baseline |
| slstm / forward | 0, plain | fast | 0.019138 [0.018831-0.019292] | 0.000255 | 68.15 | **5.02x** |
| mamba / forward | 0, plain | exact | 0.104328 [0.104099-0.104699] | 0.001391 | 107.82 | baseline |
| mamba / forward | 0, plain | fast | 0.024225 [0.023873-0.024492] | 0.000323 | 67.98 | **4.31x** |

**THE LINEAR-TIME HYPOTHESIS, MEASURED -- AND THE CONFOUND REMOVED.** The headline comparison
(causal cell fast vs windowed-BLSTM fast) confirms the direction: sLSTM **1.85x** and Mamba
**1.46x** faster end to end than the windowed BLSTM on the fast path (1.70x / 1.57x on exact). But
`--direction forward` changes TWO things at once -- the cell AND the windowing regime
(`cell_overlay` forces `BLSTM_window 0`, plus a single-direction stack and a 313- rather than
601-weight output MLP) -- so that number is NOT a cell result. The `lstm / forward` control row
exists to separate them, and it settles the question:

- REGIME effect (same LSTM cell, windowed-bi -> plain-causal, exact path): 0.163284 -> 0.098132,
  **1.66x**. This is essentially the whole win.
- CELL effect (all forward, all window 0, exact path): LSTM 0.098132, sLSTM 0.096118 (**1.02x
  faster**), Mamba 0.104328 (**0.94x -- 6% SLOWER**). A wash.

(PHASE-10 TASK 8 FOLLOW-UP: the `lstm / forward` fast cell is no longer `n/a` -- `FastLstm`
landed and the row was RE-MEASURED end to end on the current build. The numbers live in the
Task-8 section at the end of this file rather than being back-filled here, because this
table's fast rows predate the Task-5 f32-mel front-end and mixing builds inside one table
would make its ratios meaningless. The headline: fast-vs-exact **5.97x** on that row, and the
REGIME decomposition below gains its missing FAST-tier half -- **1.95x**, against the 1.66x
measured here on the exact tier. DO NOT DERIVE A FAST WALL-CLOCK FROM THIS TABLE'S ROW: the
5.97x is the ratio of the Task-8 section's OWN re-measured PAIR (exact 0.104963 s / fast
0.017587 s on the current build), not of the 0.098132 s exact wall recorded here on the
phase-9 build. Dividing this row by 5.97 gives a number that was never measured.)

So the causal cells buy their speed by being CAUSAL (no window recompute, half the recurrent
stack), not by being cheaper per step than a peephole LSTM: every cell here is already O(T), and at
this width (24 units, `d_state 16`, `d_conv 4`) Mamba's per-step block is slightly heavier than an
LSTM's. Stated as measured rather than assumed, per spec S8.7.

**MEMORY.** On the fast path all three land at ~68 MB -- the per-call f64 periodogram widen in
`FastPipeline` dominates and is cell-independent (the documented phase-7 SAD-arm memory finding,
unchanged). On the EXACT path Mamba costs 107.8 MB vs 54.8-57.0 MB for the gate cells: ~2x, and the
one place a cell choice is visibly expensive. That is the per-timestep SSM activation cache the
`Network` container retains unconditionally during the forward drive (the phase-7 scope note: the
exact path keeps `layers_output` whether or not a backward follows). Reported, not tuned.

> **BOTH MEMORY NUMBERS ARE SUPERSEDED (Phase 10), and by different tasks.** The ~68 MB fast
> plateau became **~42 MB** once Task 5 deleted the f64 widen (see "the full-f32 mel front-end"),
> and the 107.8 MB exact-Mamba row became **58.55 MB** once Task 9 stopped retaining
> backward-only structures on an inference-only net (see "inference-only retention gating"),
> which closes this paragraph's "one place a cell choice is visibly expensive" finding: Mamba
> now sits 3.0 MB above the sLSTM control on the identical recipe. The rows above stay as the
> phase-9 record of what was measured then.

**Local-only regression bounds (NOT CI-asserted, this-box numbers, Apple M4 Pro named per spec
R5),** the phase-7 local-vs-CI split: future local runs of this recipe are expected at fast-vs-exact
wall speedup >= 3.5x on every row, causal-fast-vs-BLSTM-fast >= 1.3x, and fast maxrss within
1.3x of the ~68 MB plateau (SUPERSEDED -- see Phase 10's f32-mel section: the plateau is now
~42 MB). A reading meaningfully below these (not within the ranges above, which
already carry headroom) is a FINDING to investigate; there is no automated enforcement.

## Phase 10 -- the CfC cell (Tasks 1/2/3)

The exact-f64 CfC record. NOTHING HERE WAS RE-MEASURED FOR THIS SECTION: every value is
transcribed from the committed test that pins it, and each row carries its source, so this
section can be re-derived by reading those files rather than by re-running anything. The cell
itself is `src/rust/src/nn/cells/cfc.rs`; the tiers are the phase-9 harnesses reused unchanged
(both are cell-agnostic, which is why the phase-9 filenames carry phase-10 rows). LINE NUMBERS
BELOW ARE AS OF THIS COMMIT and every one is paired with the SYMBOL it points at -- prefer the
symbol when they disagree (the phase's own citation-drift lesson: a later task inserting lines
above a cited block silently invalidates the number, never the name).

### The FD tier -- `src/rust/tests/phase9_cell_grad.rs::cfc_backward_matches_central_difference`

Central differences vs the hand-derived analytic backward, per shape x 3 seeds, at
`CFC_EPS = 1e-5` (`phase9_cell_grad.rs:754`; the value is MEASURED -- the doc records the
`max_rel` column falling ~3 orders MONOTONICALLY as eps grows `1e-8 -> 1e-4`
(`9.9e-3 / 1.9e-3 / 7.1e-5 / 1.7e-5 / 8.3e-6`), which is the roundoff signature; a wrong
adjoint term is MULTIPLICATIVE and would be eps-INVARIANT).

| shape `(t, in, out, B, L)` | `max_rel` (worst of 3 seeds) | `rel_pin` | `max_rel_major` | `major_pin` | resolvable |
|---|---|---|---|---|---|
| `(1, 3, 2, 4, 1)` | 8.126e-9 | 8.2e-8 | 8.126e-9 | 8.2e-8 | 46 of 54 |
| `(7, 3, 2, 4, 1)` | 1.168e-7 | 1.2e-6 | 7.823e-9 | 7.9e-8 | 54 of 54 |
| `(11, 5, 4, 8, 2)` | 2.398e-7 | 2.4e-6 | 2.653e-8 | 2.7e-7 | 260 of 260 |
| `(23, 7, 3, 8, 1)` | **1.720e-5** | **1.8e-4** | **5.530e-8** | 5.6e-7 | 169 of 169 |

Sources: the measured column is the doc block at `phase9_cell_grad.rs:764-770`; the pins are
the four `CfcCase` literals at `:809-844`; the resolvable counts are the same doc block, and
the test asserts them as an EXACT equality (`resolvable_floor == resolvable_structural`,
`:868-869`) rather than a two-sided band, because nothing lands in the near-zero bucket by
magnitude alone.

- **The DISCRIMINATING bound is `max_rel_major`** (restricted to weights whose analytic
  gradient is at least `1e-4` of the pack maximum): MEASURED at **`<= 5.530e-8`** across every
  shape and seed, pinned at `5.6e-7` -- i.e. ~180x under the `1e-4` STOP threshold that the
  test asserts in-body before running anything (`:851-857`).
- **THE ONE PIN ABOVE 1e-4 IS DECLARED, NOT BURIED**: `t=23`'s `rel_pin` is `1.8e-4`, above
  the STOP threshold and deliberately so (the two mamba rows set the precedent). It is fixed
  by ONE weight on ONE seed (seed 2, `w[79]`) whose analytic derivative is `1.84e-6` -- `1e-6`
  OF the pack maximum 1.79 -- so its relative error is the central-difference floor over a
  near-zero denominator. `rel_pin` is explicitly NOT a sanctioned error budget; `major_pin` is
  what a wrong adjoint trips, and that is what the in-test STOP guards (`:851-857`).
- **THE NEAR-ZERO BUCKET IS EXACTLY 0.0, and is asserted as an EQUALITY** (`:884-897`) --
  STRONGER than sLSTM's ~1e-10 floor. At `T > 1` the bucket is EMPTY (every weight
  resolvable); at `T = 1` it holds exactly the `B*H` dead `W_bb` state columns, and perturbing
  one cannot move the loss by a single bit (it multiplies `h_{-1} = 0`), so `L(w+eps)` and
  `L(w-eps)` are BIT-IDENTICAL and the central difference is `0.0` against an analytic `0.0`.
  The grid-wide absolute pins are `1e-12` near-zero / `1.4e-9` all-weights (`:878-879`).
- **THE `T = 1` DEAD BLOCK IS PINNED SEPARATELY, IN THE CELL'S OWN UNIT TIER**:
  `nn::cells::cfc::tests::backbone_state_columns_are_gradient_dead_at_t1` (`cfc.rs:1155-1167`)
  addresses the `B*H` state slots of `W_bb0` by flat arithmetic derived HERE (row-major
  `wlo + j*(i+o) + k`, `k in [in, in+out)`), asserts `== 0.0` at `T = 1` and NON-zero at
  `T = 2` in the same body. NOTHING ELSE is dead: there is no CfC analogue of sLSTM's `b_i`
  non-identifiability, because `tanh`/`sigmoid` carry no scale invariance for a bias shift to
  be absorbed into.

### The seam tier -- `tests/pyo3/test_phase9_seam.py`

Corpus-level `grad_check` through `speech_rs.Engine` on the committed synthetic fixtures
(`tests/reference_data/phase9/`, geometry `backbone_units 6` / `backbone_layers 1` per the
manifest -- `B != H` DELIBERATELY, so a transposition cannot hide). Pack lengths from the same
manifest: `cfc_bidirectional` **1387**, `cfc_forward` **717**, and the Mode-7 Twin's LID net
**517** beside a 537-element SAD net.

| fixture | eps | measured worst scaled error | pin | source |
|---|---|---|---|---|
| `cfc_bidirectional` | 1e-5 | **3.138e-7** | 3.2e-6 | `GRAD_CHECK_PINS`, `test_phase9_seam.py:387` |
| `cfc_forward` | 1e-5 | **1.169e-8** | 1.2e-7 | `GRAD_CHECK_PINS`, `:388` |
| `twin_mode7_lid_cfc` | 1e-4 | **2.630e-9** | 2.7e-8 | `TWIN_MODE7_ROWS`, `:747` |
| `cfc_bidirectional` (block probe) | 1e-5 | 1.417e-8 | 1.5e-7 | `BLOCK_PROBE_PINS`, `:421` |
| `cfc_forward` (block probe) | 1e-5 | 1.027e-8 | 1.1e-7 | `BLOCK_PROBE_PINS`, `:422` |

Every pin is `measured * 10` and every one is under the `1e-4` STOP. The two epsilons are
MEASURED, not inherited: the CfC SAD rows reuse the shared `sad_epsilon 1e-5` on the evidence
of a 5-point sweep recorded in the file (`bi 5.92e-6 / 1.70e-7 / 3.14e-7 / 3.13e-5 / 3.13e-3`,
`fwd 2.81e-6 / 3.41e-7 / 1.17e-8 / 1.17e-6 / 1.17e-4` -- a textbook U), while the CfC Twin
takes its own `1e-4` (`manifest.json: measured.twin_mode7_cfc_epsilon`) because its LID
gradient is ~5.6e-5, ~60x the sLSTM Twin's, so its optimum sits one decade lower.

WHAT THE SEAM TIER ADDS THAT THE FD TIER CANNOT: the FD tier drives `CfcLayer` DIRECTLY, so
`CellLayer::Cfc`'s forward / backward / derivative-harvest arms were compiled-but-never-
executed; the Engine path runs a real corpus forward AND backward THROUGH the enum. Verified
by mutation: each single-arm swap fails 5 legs. The consistent DOUBLE swap that stays
self-consistent here is closed by Task 6's independent `FastCfc` implementation instead (see
the fast-parity rows), which is the note now carried in the seam file's own docstring.

### Sizing (spec S1.4) -- both closed forms, PINNED not prose

`tests/test_phase10_init.py` computes both lineages' pack lengths from the block arithmetic
and asserts the closed forms directly (`test_lineage_pack_lengths_are_the_documented_arithmetic`,
`:428-437`):

| lineage | LSTM pack | CfC closed form | at `B = 45` | vs LSTM |
|---|---|---|---|---|
| v1 (`23,24,24`, tail 46) | **33671** (the committed tuple-A length, independently reached) | `620*B + 935` | 28835 | **-14.36%** |
| v2 (`11,24,24`, tail 22) | **24431** | `524*B + 911` | **24491** | **+0.25%** |

`CFC_DEFAULT_BACKBONE_UNITS = 45` is therefore ONE default serving BOTH lineages, and both
halves of that claim are asserted rather than argued: `B = 45` is v2's OPTIMUM (`rel[45]` is
strictly below `rel[44]` and `rel[46]`, `test_the_default_matches_the_v2_lstm_pack_within_15_percent`,
`:440-449`) AND the SMALLEST integer inside v1's +-15% band (`B = 44` is asserted OUT,
`test_the_default_also_leaves_the_v1_lineage_in_band`, `:452-460`). v1's own optimum, recorded for completeness, is
`B = 53 -> 33795` (+0.37%). Forward-only runs shed one stack and the MLP's doubled input on
both sides, so the ratio barely shifts: v2 fwd LSTM 12239 vs CfC 12269 (+0.25%), v1 fwd LSTM
16871 vs CfC 14453 (-14.33%).

The Python builder emits the S1.2 flat order DIRECTLY (CfC has no structured/nnet domain to
build, spec S1.3), so the layout risk the LSTM path retires by reusing the packer is retired
here by WHOLE-PACK BLOCK-BY-BLOCK RECONSTRUCTION pins plus two shear companions -- and what
those prove is block ORDER / LENGTH / FAN, NOT orientation (unobservable for iid init; He's
asymmetric fan is what makes fans pinnable at all, which is why parametrizing over both
schemes is load-bearing).

## Phase 10 -- the `lre_sad_v2` lineage (Task 4)

The v2 record: a ONE-VARIABLE fork of the phase-6/9 SAD arm that retires the
2015-inherited dead-input-column block, its 8-gate from-scratch matrix
({LSTM, sLSTM, Mamba, CfC} x {bidirectional, forward}), the zero-dead-columns INVERSE
guard, and the like-for-like live-capacity comparison against v1. Gates:
`tests/pyo3/test_phase10_gates.py` (corpus-gated, `slow`, local-only). Measured 2026-08-01,
Apple M4 Pro (arm64), macOS 26.5.2, N=1 lane, seed 0. The full-corpus headline runs stay
POST-PHASE, user-fired (launcher recipe at the end of this section).

### The fork (spec S3.1)

`configs/training/lre_sad_v2.toml` is `configs/training/lre_sad.toml` byte-for-byte except
TWO values (verified by diffing the two files' non-comment lines: exactly these, nothing
else) and the file headers:

| key | v1 | v2 |
|---|---|---|
| `nnet_input_size` | 23 | **11** |
| `lstm_neuron_nb` | `23,24,24` | **`11,24,24`** |

plus the normalize mean/std tail those entail (`2*23 = 46` -> `2*11 = 22`), which appears in
NO config: it self-sizes off `nnet_input_size` on both sides of the seam
(`train.py::_tail_lengths`, `BLSTMNeuralNetwork::setWeights`). v1's own header gained a
one-line pointer to v2 -- COMMENT-ONLY, and verified `config_hash`-safe by parsing both
revisions through `speech_rs.load_toml_config` and asserting the maps are identical (the
hash reads the parsed map, which TOML comments never enter), so every v1 run's recorded
`config_hash` is unchanged. v1 stays FROZEN: `test_phase9_gates.py`'s asserts, pins and
numbers are untouched and green (the T4 review corrected one module-docstring sentence
about `b_i`'s computed-vs-analytic zero -- comment-only).

WHY: v1 declares a 23-wide input while its own DSP front-end emits 11 columns
(`3*nb_dct - ignore_first_dct = 3*4 - 1`), so with `lstm_sub_sampling 4` only 44 of layer 0's
92 fan-in columns carry data and the trailing 48 are structurally gradient-dead -- the
phase-9 "DEAD INPUT COLUMNS" finding. v2's declared width IS the produced width.

### Pack lengths, MEASURED (spec R4 -- no survey estimate survives)

Every number produced by `init_weights` at the arm's own overlaid config, cross-checked
in-test against the per-cell closed-form block arithmetic (`test_init_is_trainable` asserts
both, so a pinned literal and the offset arithmetic cannot drift apart silently):

| cell | v2 bidirectional | v2 forward | vs v2 LSTM (bi) | v1 bidirectional | v1 forward |
|---|---|---|---|---|---|
| LSTM (baseline) | **24431** | **12239** | -- | 33671 | 16871 |
| sLSTM | **23279** | **11663** | **-4.72%** | 32519 | 16295 |
| Mamba | **28031** | **14039** | **+14.74%** | 30359 | 15215 |
| CfC (`B = 45`, `L = 1`) | **24491** | **12269** | **+0.25%** | 28835 | 14453 |

All four land inside the S8.2-style +-15% band of the same lineage's LSTM pack. Mamba only
just (+14.74%), and the near-miss is structural, not luck: Mamba's layer-0 input projection
is a SINGLE `out x fin` width adapter, while the gate cells carry `4 x out x fin` and CfC
`B x fin`. Halving `fin` (92 -> 44) therefore shrinks the gate cells and CfC much harder
than it shrinks Mamba -- the same asymmetry that made Mamba the nominally-smallest but
effectively-LARGEST net on v1.

### The like-for-like LIVE-capacity comparison -- v2 changes ZERO trainable capacity

The sharpest v1-vs-v2 statement, and it is an identity rather than a measurement.
Counting only weights with a nonzero gradient (v1's counts are phase 9's; v2's are
`pack - 22`, the 22-element frozen normalize tail being the ONLY dead block a v2 pack has,
measured EXACTLY 22 on all eight rows):

| cell | v1 live (bi) | v2 live (bi) | v1 live (fwd) | v2 live (fwd) |
|---|---|---|---|---|
| LSTM | 24409 | **24409** | 12217 | **12217** |
| sLSTM | 23257 | **23257** | 11641 | **11641** |
| Mamba | 28009 | **28009** | 14017 | **14017** |
| CfC | 24469 | **24469** | 12247 | **12247** |

Identical in every cell, in both directions. The arithmetic: v1's layer-0 input block is
wider than v2's by exactly the dead column count (`4*out*48` for the gate cells, `out*48`
for Mamba, `B*48` for CfC, per stack), and its normalize tail is wider by exactly 24, so
`v1_pack - v2_pack = D*dead_per_stack + 24` while `v1_live = v1_pack - (D*dead_per_stack +
46)` and `v2_live = v2_pack - 22` -- the two collapse to the same number. **The fork removes
dead weight and NOTHING else.**

So v1-vs-v2 is not a capacity question. What DOES differ, measurably:

1. **Pack size / memory / I/O**: v2's LSTM pack is 27.4% smaller (33671 -> 24431). The
   COMPUTE saving is CELL-DEPENDENT, because the two width-tolerance conventions differ --
   an earlier revision of this bullet claimed a flat "2.09x narrower layer-0 matmul, 48 of
   every 92 columns multiplying constant zeros", and that is FALSE for the cell it named.
   `LstmLayer::feed_forward` (`layers.rs:194-198`) takes the `cols < i` branch and CROPS
   the WEIGHT matrix to the input's width, so v1's layer-0 product was ALREADY
   `T x 44` by `44 x 96` -- identical to v2's, and nothing ever multiplied a dead column.
   v2's only LSTM compute win is dropping the per-call `44 x 96` slice copy that crop
   makes. The three NEW cells take the opposite convention: `reconcile_input`
   (`slstm.rs:358`, `mamba.rs:723`, `cfc.rs:460`) ZERO-PADS the input up to the declared
   width, so their layer-0 products really do shrink -- sLSTM and Mamba 92 -> 44
   (**2.09x**), CfC `in+h` 116 -> 68 (**1.71x**, its fan-in carries the state
   concatenation). NONE of this was benched; it is shape arithmetic, not a measured
   speedup.
2. **The Xavier fan-in scaling of the LIVE weights.** `init_weights` sizes the layer-0
   bound off the DECLARED fan-in, so v1 seeded its live weights as though half the
   (constant-zero) fan-in were carrying signal. Measured layer-0 input-block magnitudes
   (seed 0, `xavier`, forward rows): LSTM max `0.206985 -> 0.255375` (exactly
   `sqrt(6/140) -> sqrt(6/92)`, a **1.234x** widening), sLSTM std `0.11973 -> 0.147567`
   (1.233x), Mamba `0.132364 -> 0.168974` (1.277x), CfC `0.111370 -> 0.132902` (1.193x).

**FRAMING (spec S3.4 / R5, stated so nobody reads more into this than was measured):** v1
remains the ONLY 2015-capacity-comparable lineage -- its 33671 LSTM pack IS the production
tuple-A size, and the phase-6 baseline numbers are v1's. v2 is a NEW lineage with no 2015
counterpart. The corrected Xavier scaling is a MEASURABLE difference for the full-corpus
launchers to ADJUDICATE, **not a promised win**; a wider init is not automatically a better
one, and nothing at subset scale can settle it.

### The inverse guard (spec S3.2) -- zero structurally-dead layer-0 input columns

v1's gates pin dead-count FLOORS; v2's pin the mirror image. For every cell x direction and
every stack, all 44 layer-0 input columns must carry gradient. The guard reads the layer-0
INPUT-PROJECTION block only, addressing per cell (each derived from that cell's own flat
walk; offsets are relative to the layer-0 base, `fin = 44`, `out = 24`, `B = 45`):

| cell | input block | flat position of input column `k` | entries/column |
|---|---|---|---|
| LSTM | `input_weights (fin x 4*out)`, layer's first block, COLUMN-major | `c*fin + k`, `c < 4*out` | 96 |
| sLSTM | `W_a (out x fin)` row-major, inside each gate block `[R_a \| W_a \| b_a]` | `a*out*(out+fin+1) + out*out + j*fin + k` | 96 |
| Mamba | the width adapter `P (out x fin)` row-major, first block (present iff `fin != out`) | `j*fin + k`, `j < out` | 24 |
| CfC | `W_bb0 (B x (fin+out))` row-major, first block, fan-in `z = [x \| h]` -- only `k < fin` are INPUT columns | `j*(fin+out) + k`, `j < B` | 45 |

SCOPE, by BLOCK not by slack: cell-level gradient-dead blocks that are NOT input columns
(sLSTM's non-identifiable `b_i`, Mamba's `A_log` at `T = 1`, CfC's `W_bb` STATE columns at
`T = 1`) are a separate, already-pinned phenomenon and are excluded by never being
addressed. TWO-SIDED, because "nonzero" alone is too weak: a structurally dead weight is
EXACTLY `0.0` (nothing ever accumulates into it) while an analytically-zero-but-COMPUTED
weight lands at cancellation scale -- measured, sLSTM's `b_i` reads ~1e-19 at this seam
(max 8.33e-19), not a bit-exact zero. The guard therefore asserts the exact-zero property
AND a magnitude floor of 1e-8: ~1e11 above the cancellation class it must reject, ~2.9e4
below the smallest per-column magnitude ever measured here. A third leg pins the WHOLE-PACK
zero count at exactly 22 (the frozen tail) -- so a dead block that MOVED somewhere the
offsets do not address still fails.

Precisely, so three legs are not mistaken for three independent checks: the exact-zero leg
is SUBSUMED by the floor leg (an exactly-0.0 column has max|grad| below the floor, so the
dead set is a strict subset of the weak set), and is kept for its distinct failure message
-- "structurally dead" and "gradient-inert" are different defects -- not for coverage. The
whole-pack zero count is the one genuinely independent leg: it is the only one that can see
a dead block relocated outside every per-column offset.

NON-VACUITY, proven not asserted: `test_v1_cfc_mechanical` runs the SAME machinery on the
v1 arm and finds exactly the 48 dead columns `[44, 92)` per stack. If the offset arithmetic
addressed biases, a recurrent block, or nothing, it would find 0 dead columns there and the
v2 guard would be silently vacuous.

### Preflight -- the log-law saturation hazard, measured before firing the runs

Same hazard and same discipline as phase 9 (`CostLaw log/log`, whose forward is CONSTANT
past the `1e-24` clamp, so a saturated from-scratch net is a PERMANENT STALL). Probed at the
from-scratch theta, one forward+backward through the seam, no training; the leg's own recipe
(subset 2, 10 s cap, seed 0). `%clamp` is the column to its left over `-ln(1e-24) = 55.262`;
`min col` is the smallest per-input-column max|grad| over every stack -- the inverse guard's
own margin against its 1e-8 floor:

| config | pack | init NNCostSeg | %clamp | worst per-file | %clamp | grad L2 | grad Linf | zero-grad weights | dead cols | min col |
|---|---|---|---|---|---|---|---|---|---|---|
| LSTM / bidirectional | 24431 | 0.28007 | 0.507% | 0.40669 | 0.736% | 0.39567 | 0.19439 | 22 | **0** | 2.854e-4 |
| LSTM / forward | 12239 | 0.27649 | 0.500% | 0.40178 | 0.727% | 0.37588 | 0.19127 | 22 | **0** | 3.493e-4 |
| sLSTM / bidirectional | 23279 | 0.28918 | 0.523% | 0.40714 | 0.737% | 0.38630 | 0.19917 | 22 | **0** | 6.011e-4 |
| sLSTM / forward | 11663 | 0.24322 | 0.440% | 0.33209 | 0.601% | 0.45236 | 0.17433 | 22 | **0** | 1.311e-3 |
| Mamba / bidirectional | 28031 | 0.28676 | 0.519% | 0.40283 | 0.729% | 0.54261 | 0.18486 | 22 | **0** | 2.960e-3 |
| Mamba / forward | 14039 | 0.29903 | 0.541% | 0.46646 | 0.844% | 0.50035 | 0.18487 | 22 | **0** | 3.856e-3 |
| CfC / bidirectional | 24491 | 0.29857 | 0.540% | 0.41060 | 0.743% | 0.72674 | 0.19351 | 22 | **0** | 3.269e-3 |
| CfC / forward | 12269 | 0.27922 | 0.505% | 0.39550 | 0.716% | 0.41801 | 0.19030 | 22 | **0** | 2.487e-3 |

Every init sits at ~0.5% of the clamp constant (worst 0.54%), every INDIVIDUAL file at
<= 0.84%, every epoch-0 gradient norm far from zero: no row starts in the zero-gradient
death, CfC included, and no constant needed changing. The per-file column is read by slicing
`results_matrix()`'s `[file+1, conf+1, chan+1, ...]` prefix off FIRST (`[:, 3:]`, as
`engine.py::_error_vad` does) with a `per_file_max >= aggregate_mean` self-check ahead of the
bound -- the phase-9 correction, carried forward by construction. EVIDENTIARY SCOPE
(unchanged from phase 9): the seam exposes cost and gradient, not the posterior vector, so
"not saturated" means COST-INTERIOR + GRADIENT-NONZERO + the net subsequently trains.

The v1 CfC MECHANICAL leg (spec S3.3 -- construction + its own dead-count floor, no
convergence gate and no param-match requirement on v1, run at the DEFAULT `B = 45` rather
than T2's v1-matched 53 since no fair-size comparison is being made):

| config | pack | init NNCostSeg | worst per-file | grad L2 | zero-grad weights | predicted floor | dead cols/stack |
|---|---|---|---|---|---|---|---|
| v1 CfC / bidirectional | 28835 | 0.34644 | 0.47582 | 0.97583 | 4366 | `2*45*48 + 46 = 4366` | 48 (== `[44, 92)`) |
| v1 CfC / forward | 14453 | 0.29140 | 0.41856 | 0.55960 | 2206 | `1*45*48 + 46 = 2206` | 48 (== `[44, 92)`) |

Both land EXACTLY on the derived floor. The CfC dead-block shape is `B * 48` per stack (the
`W_bb0` row width), cell-dependent exactly as the phase-9 T8 pattern predicts -- `4*out*48`
in the gate cells, `out*48` in Mamba.

### The eight gates -- HARD leg: trained held-out DCF beats own init, every collar

Phase-6 SAD protocol VERBATIM (subset 10 / valid 8 / test 24, 3 epochs x 10 SMORMS3 steps,
20 s audio cap, seed 0, `val_metric=nn_cost_seg`, end-to-end VRCTS-dump -> `evaluate.dcf`
scoring). Only the cell, the direction and the LINEAGE differ from phase 9's four rows:

<!-- ledger:table phase10_v2_cells -->
| cell / direction | files (train/valid/test) | trained DCF@0.5 | Pmiss / Pfa @0.5 | trained collar range | init DCF@0.5 | init collar range | gain@0.5 | wall |
|---|---|---|---|---|---|---|---|---|
| LSTM / bidirectional | 10 / 8 / 24 | 0.250000 | 0.000000 / 1.000000 | 0.250000 (all 5) | 0.750000 | 0.750000 (all 5) | +0.500000 | 40 s |
| LSTM / forward | 10 / 8 / 24 | 0.250000 | 0.000000 / 1.000000 | 0.250000 (all 5) | 0.750000 | 0.750000 (all 5) | +0.500000 | 18 s |
| sLSTM / bidirectional | 10 / 8 / 24 | 0.250000 | 0.000000 / 1.000000 | 0.250000 (all 5) | 0.750000 | 0.750000 (all 5) | +0.500000 | 36 s |
| sLSTM / forward | 10 / 8 / 24 | 0.252430 | 0.026415 / 0.930474 | [0.248021, 0.255545] | 0.750000 | 0.750000 (all 5) | +0.497570 | 17 s |
| Mamba / bidirectional | 10 / 8 / 24 | 0.250000 | 0.000000 / 1.000000 | 0.250000 (all 5) | 0.750000 | 0.750000 (all 5) | +0.500000 | 67 s |
| Mamba / forward | 10 / 8 / 24 | 0.248770 | 0.000000 / 0.995078 | [0.248770, 0.250000] | 0.737710 | [0.736969, 0.738062] | +0.488940 | 21 s |
| CfC / bidirectional | 10 / 8 / 24 | 0.250000 | 0.000000 / 1.000000 | 0.250000 (all 5) | 0.750000 | 0.750000 (all 5) | +0.500000 | 36 s |
| CfC / forward | 10 / 8 / 24 | 0.250000 | 0.000000 / 1.000000 | 0.250000 (all 5) | 0.750000 | 0.750000 (all 5) | +0.500000 | 17 s |
| full-corpus runs (any cell x direction) | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

Hosts: Apple M4 Pro (arm64, 14 cores, Darwin 25.6.0).
<!-- ledger:end -->

8/8 HARD legs pass at every collar (0 / 0.25 / 0.5 / 1 / 2 s) on the FIRST run, and
deterministically: run-twice at a fixed seed gives bit-identical `best_sad.bin` /
`last_sad.bin` bytes and an identical pooled DCF on all eight rows. Every DCF@0.5 above was
also reproduced in a second, independent process to the printed precision. **CfC's
convergence gate IS these two rows** (spec S3.3) and it passes both directions. The slowest
single gate is 62 s against the asserted 600 s budget. Whole-file runtime: 1.4 s preflight
(10 legs) + 227 s gates (8) + 76 s determinism (8) = ~5.1 min.

The wall column is NOT a speed comparison: `--direction forward` also forces
`BLSTM_window 0` (the plain whole-sequence causal regime -- a window boundary resets the
recurrent state), so forward rows differ from bidirectional ones in TWO ways.

**RECORDED, NOT GATED (spec R5, verbatim): "subset vs-BLSTM numbers are reported as-is with
the thinness caveat; no architecture-superiority claim is made from subset scale."** SIX of
the eight rows are degenerate at BOTH endpoints -- the documented phase-6 collapse
(all-non-speech init -> all-speech trained) -- so their six-way tie at 0.250000 is what an
identical collapse looks like, not evidence of architectural OR lineage equivalence. The
phase-9 v1 rows tie at the same 0.2500 (bar Mamba/forward's 0.249625), so this run says
nothing about v1 vs v2 either. TWO rows are honestly non-degenerate and are recorded as
measured rather than rounded into the collapse story: `sLSTM / forward` genuinely rejects
~7% of held-out non-speech at a 2.6% miss cost (its WORST collar, 0.255545, is still 0.494
below its init), and `Mamba / forward` rejects a 0.49% sliver while its INIT is also not
fully degenerate (Pmiss 0.982625 -- the one row whose init baseline is not exactly 0.75, and
the row the `init Pmiss > 0.9` pin is sized for; that margin is real but thin). Genuine
speech/non-speech discrimination, and any cell-vs-cell or v1-vs-v2 ranking, is the
FULL-CORPUS user-fired launcher's job.

**The CE is not the signal** (the standing phase-6 lesson): best-epoch validation costs
across the eight rows span 0.02659 (CfC/forward) to 0.44689 (Mamba/forward), with best
epochs at 0, 1 and 2 -- and every row still lands at the same held-out DCF. Only the
held-out TASK metric is gated.

### Firing the v2 arm (post-phase, user-fired)

`experiments/03_v2_cells.sh` fires the eight rows (any subset by name, e.g.
`sad-v2/cfc/forward`); each record promotes into the `full-corpus runs` row above.

Identical in every knob to the phase-9 `python -m speech.drivers.baseline sad` recipe -- ONLY the arm name
changes, since `sad-v2` shares the entire SAD skeleton (`SadArm`) and every size derives
from the config. To answer the lineage question, fire the SAME cell x direction on both arms
and compare; the live-capacity table above says the comparison is about init scaling and
pack size, not about capacity.

## Phase 10 -- the full-f32 mel front-end (Task 5, spec S4)

`fast/mel32.rs` replaces the phase-7 "widen the f32 periodogram to f64, run the golden f64
`MelFilterBank`, narrow the assembled sequence back" bridge with f32 kernels transcribed
op-for-op from `features/mel.rs` + `features/pipeline.rs::assemble_input_sequence`. BOTH fast
sites route through it in one commit-unit -- the offline `FastPipeline::build_input_sequence_parts`
and the streaming `assemble_perio_window` -- via ONE shared `assemble_rows` kernel, so
offline-vs-streamed stays bit-identical BY CONSTRUCTION (the phase-8/9 streaming gates compare
fast-vs-fast; they were re-run UNEDITED and stay green). The f64-widen path is DELETED, not kept
as a mode; the exact `features/mel.rs` stays byte-untouched as the transcription ORACLE.

The bank GEOMETRY is still derived in f64 (the sequential `freq += freq_step` grid, the
floor/ceil band round-trip, the inclusive triangle edges, the whole-bank fallback) and only the
coefficient VALUES narrow `as f32` once at construction: deriving the geometry in f32 could move
filter MEMBERSHIP (an `is_valid_bank` flip, a triangle-edge inclusion flip), which is a structural
change rather than a precision one. Banks are still built ONCE per pipeline.

QUIRKS CARRIED (each pinned by its own inline unit, each doc-noted against its `mel.rs` line):
the deltas-no-DCT static-block OVERWRITE layout (`mel.rs:412-417`); SDC's unconditional n=3
regression kernel (`:272-276`) and its ignoreFirst last-static clobber (`:286-294`); the
ignoreFirst branch-B FIRST-column drop (`:295-312`); `regression_deltas`' saturated-`j` denominator
(`:431-475`); `ln(dot + 1e-24)` (`:352`); the whole-bank fallback (`:138-155`). A 10-config shape
cross-check asserts `is_mel`/`is_dct_activated`/`nb_filters`/`nb_dct` agree with the exact bank
element-for-element, so a geometry drift fails BEFORE any value pin.

### The S4 re-pin sweep -- the ONE sanctioned pin re-measurement (spec S9.3)

Every fast-vs-exact number moved once, by design. Boundary count/type identity and argmax
zero-flips held EVERYWHERE -- no R1 STOP was reached. Measured on Apple M4 Pro (arm64),
macOS 26.5.2, `cargo build --release` (LTO on), `--release` test profile.

| suite / leg | metric | old measured | old pin | NEW measured | NEW pin | flips |
|---|---|---|---|---|---|---|
| `phase7_fast_pipeline::pipeline_parity_excerpt_3s` | max_rel | 3.044e-6 | 5.0e-5 | **1.807e-5** | **2.0e-4** | n/a |
| `phase7_fast_pipeline::pipeline_parity_excerpt_3s` | max_abs | 8.263e-6 | 1.0e-4 | **3.589e-5** | **4.0e-4** | n/a |
| `phase7_fast_pipeline::pipeline_parity_prcts_60s` | max_rel | 9.805e-6 | 1.5e-4 | **4.038e-5** | **6.0e-4** | n/a |
| `phase7_fast_pipeline::pipeline_parity_prcts_60s` | max_abs | 1.805e-5 | 3.0e-4 | **5.914e-5** | **9.0e-4** | n/a |
| `phase7_fast_pipeline::pipeline_parity_dc_offset_branch` | max_rel | 5.704e-4 | 6.0e-3 | 5.719e-4 | 6.0e-3 (kept) | n/a |
| `phase7_fast_pipeline::pipeline_parity_dc_offset_branch` | max_abs | 9.784e-3 | 5.0e-2 | 9.784e-3 | 5.0e-2 (kept) | n/a |
| `phase7_parity_sad::sad_parity_exact_vs_fast` | posterior max_rel | 1.398e-5 | 3.0e-4 | 1.361e-5 | 3.0e-4 (kept) | **0** |
| `phase7_parity_sad::sad_parity_exact_vs_fast` | posterior max_abs | 2.416e-6 | 5.0e-5 | 2.416e-6 | 5.0e-5 (kept) | **0** |
| `phase7_parity_sad::sad_parity_exact_vs_fast` | boundary max_dt | 0.0 | 1.0e-2 | **0.0 exactly** | 1.0e-2 (kept) | **0** |
| `phase7_parity_sad::sad_parity_scored_columns` | scored max_rel/abs | 0.0 / 0.0 | 5.0e-2 | **0.0 / 0.0** | 5.0e-2 (kept) | **0** |
| `phase7_parity_lid::lid_parity_phseq` | score max_abs / max_rel | 7.785e-7 / 1.662e-8 | 5e-4 / 5e-6 | 7.785e-7 / 1.662e-8 | unchanged | **0** |
| `phase7_parity_lid::lid_parity_cep` | score max_abs / max_rel | 2.680e-6 / 2.733e-8 | 5e-4 / 5e-6 | 2.680e-6 / 2.733e-8 | unchanged | **0** |
| `phase9_fast_parity` (worst of 4 cell x leg runs) | posterior max_rel | 4.13e-6 | 1.0e-4 | 7.740e-6 | 1.0e-4 (kept) | **0** |
| `phase9_fast_parity` (worst of 4 cell x leg runs) | boundary max_dt | 0.0 | STOP on nonzero | **0.0 exactly** | STOP on nonzero | **0** |
| `phase9_stream_causal` (19 legs, streamed-vs-offline-fast) | bit-equality | exact | exact | **exact, UNEDITED** | n/a | **0** |
| `phase8_gate` / `phase8_stream_{frontend,overlap,decision,lid}` / `phase8_frozen_norm` (59 legs) | bit-equality | exact | exact | **exact, UNEDITED** | n/a | **0** |

WHERE THE DELTAS GREW AND WHERE THEY DID NOT, mechanism-first (this is the interesting part, not
the bookkeeping):

- **The front-end pins grew ~3-6x** (3.28x to 5.94x across the four re-measured numbers) and are the only re-pinned numbers. f32 error now accumulates
  through the mel triangle dots, the DCT product AND the delta/delta-delta regressions rather than
  entering only via the periodogram -- the phase-7 bridge did the whole tail in f64.
- **The DC-offset branch did NOT move** (rel 5.704e-4 -> 5.719e-4, abs identical to 4 figures).
  There the divergence is already dominated by the f32 full-buffer DC-mean subtraction UPSTREAM of
  the mel, so the mel's own f32 error is noise against it. The one leg whose phase-7 pins survive.
- **The SAD posteriors did NOT move** (max_abs identical at 2.416e-6; max_rel 1.398e-5 -> 1.361e-5,
  i.e. DOWN, which is the max simply landing in a different cell). A ~4e-5 perturbation of a
  log-mel feature of magnitude ~10 is ~4e-6 relative -- below what the f32 LSTM recurrence already
  contributes, so the NN's own error still sets the ceiling. This is why boundary max_dt stayed
  EXACTLY 0.0 and the scored columns stayed bit-identical.
- **The LID legs are bit-for-bit unchanged, STRUCTURALLY** -- Mode 7 consumes phSeq/cep
  `external_features` with a FROZEN SAD net, so no periodogram, mel bank or DCT is ever built and
  `FastPipeline` is not even constructed. Recorded as an unchanged row precisely because an
  unexplained MOVE there would have meant the change leaked where it has no business being.
- **The phase-9 causal posteriors grew ~1.9x** (4.13e-6 -> 7.740e-6, mamba; slstm 1.40e-6), inside
  the existing 1e-4 pin at ~13x headroom, with `max_dt` still exactly 0.0.
- **The remaining fast suites are structurally unaffected**, for the same kind of reason the LID
  row is, and are named here so their absence from the table is a statement rather than an
  omission: `phase7_fast_nn.rs` drives `FastBlstm` on synthetic in-memory matrices and never
  constructs a `FastPipeline`; `phase7_bench.rs` and the phase-8 `stream` CLI legs assert
  wall-clock budgets and output FORMATTING, not feature values; and every phase-8/9 streaming
  bit-equality leg compares fast-vs-fast, so both sides move together by construction. All were
  re-run green and UNEDITED.

### The corpus metric tiers -- re-run locally, all deltas EXACTLY 0.0

Not "within tolerance": float equality on the task metrics, on real LRE03/07 data through the
phase-6/9 checkpoints. Corpus-gated + local-only as always (no filename or path recorded).

| tier | leg | files | decision disagreements | metric delta (fast - exact) |
|---|---|---|---|---|
| `test_phase7_parity.py` | lid-features (cep) | 48 | argmax 0 | lid_error 0.000e+00, cavg 0.000e+00 |
| `test_phase7_parity.py` | lid-phseq | 45 | argmax 0 | lid_error 0.000e+00, cavg 0.000e+00 |
| `test_phase7_parity.py` | sad | 24 | boundary type/count 0, max_dt 0.000e+00 s | DCF 0.000e+00 at all 5 collars |
| `test_phase9_parity.py` | slstm / forward | 24 | boundary type/count 0, max_dt 0.000e+00 s | DCF 0.000e+00 at all 5 collars |
| `test_phase9_parity.py` | mamba / forward | 24 | boundary type/count 0, max_dt 0.000e+00 s | DCF 0.000e+00 at all 5 collars |
| `test_phase8_parity.py` | streamed-vs-offline SAD | 1 (75 s) + 24 | streamed boundary_max_dt 2.500e-05 s (inside the 5.0e-05 VRCTS write quantum, unchanged) | causality DCF delta 0.000e+00 |

### RSS: the phase-7 memory finding is REVERSED

The phase-7 SAD-arm finding ("fast uses MORE memory, 1.22-1.28x -- the per-call f64
periodogram-widen") is the thing this task was aimed at, and it is now the other way round.
BEFORE/AFTER measured on IDENTICAL staging in the SAME session (the change stashed, rebuilt,
re-measured, unstashed, rebuilt) so the two columns differ in exactly one variable: 3 independent
`--repeat=1` fresh processes per (leg, path), `speech bench` on the committed 60 s stereo
`prcts_excerpt.wav` (`audio_s = 120.00`), backprop off, `Neural_Networks_BackPropagation_Epochs 0`.
Apple M4 Pro (arm64), macOS 26.5.2, `cargo build --release -j 4`. The causal row stages the
committed `phase9/slstm_forward.config` + seed pack against the SAME wav (tiny 23,4 net, plain
window-0 regime), so it isolates the front-end rather than the cell.

| leg | path | maxrss_mb BEFORE | maxrss_mb AFTER | delta | wall_s BEFORE (mean) | wall_s AFTER (mean) |
|---|---|---|---|---|---|---|
| SAD tier2 (60 s stereo) | exact | 56.484 | 56.521 | +0.04 (unchanged, exact tree untouched) | 0.27638 | 0.27304 |
| SAD tier2 (60 s stereo) | **fast** | **67.261** | **41.953** | **-25.31 MB (0.624x)** | 0.058897 | **0.055225 (1.066x faster)** |
| causal slstm/fwd (60 s stereo) | exact | 53.302 | 54.016 | +0.71 (noise, exact tree untouched) | 0.13783 | 0.14065 |
| causal slstm/fwd (60 s stereo) | **fast** | **67.026** | **41.729** | **-25.30 MB (0.623x)** | 0.023596 | **0.020763 (1.136x faster)** |

- **THE SIGN FLIP:** fast-vs-exact peak RSS on the SAD arm goes **1.191x -> 0.743x**. The fast path
  now uses roughly 26% LESS memory than exact, where phase 7 measured it using 19-28% MORE. The
  phase-9 "~68 MB plateau, cell-independent" becomes a **~42 MB plateau** (both cells land within
  0.23 MB of each other, still cell-independent -- it was never the cell).
- **THE DROP MATCHES THE PREDICTED MECHANISM TO ~0.1 MB, which is the real evidence.** Phase 7
  sized the per-channel widen buffer at `6000 x 513 x 8 bytes = 23.48 MB`; adding the other f64
  transients that died with it (`fb` 6000x20x8 = 0.92 MB, `dct`/`input64` 6000x11x8 = 0.50 MB each)
  gives ~25.4 MB against a measured 25.31 / 25.30 MB. The attribution was correct.
- **BIGGER THAN THE SPEC'S EXPECTATION** (S4 predicted ~68 -> ~56 MB, i.e. the widen buffer alone):
  the assembled/mel/DCT f64 intermediates went with it, so the measured landing is ~42 MB.
- **Wall-clock is a small bonus, not the point:** 1.07x (SAD) / 1.14x (causal) on the fast path.
  The f32 mel is cheaper than f64-widen-plus-f64-mel, but the FFT still dominates, exactly as the
  phase-7 rationale for NOT doing this predicted. The RSS was the reason to do it.

**SUPERSEDED BOUNDS.** Two local-only regression bounds recorded earlier are now stale and are
restated here rather than edited in place (those sections are the record of their own phase):
the phase-7 "SAD arms: memory <= 1.4x of exact" becomes **<= 0.85x of exact** (measured 0.743x),
and the phase-9 "fast maxrss within 1.3x of the ~68 MB plateau" becomes **within 1.3x of the
~42 MB plateau**. A future local run landing back near 67 MB on a fast SAD/causal leg means the
f64 widen has crept back in. Still NOT CI-asserted; this-box numbers, hardware named per R5.

## Phase 10 -- the bidirectional f32 cell twins (Task 7, spec S5)

`fast::bicell::FastBiCell` closes the `(cell x direction)` fast matrix on the bidirectional
side: `{slstm, mamba, cfc} x bidirectional` moved from a typed bail to a real f32 shape
(forward stack + REVERSED stack + hcat + the shared per-row `DenseRowChain`), driven by a
FRESH windowed-overlap loop over the already-shared span helpers. `FastBlstm` and its
`overlap_window_step` are BYTE-UNTOUCHED (approach A) -- the diff does not touch
`src/rust/src/fast/nn.rs` at all.

### Parity: exact f64 bidirectional vs `FastBiCell` (`tests/phase10_bicell_parity.rs`, CI)

Both regimes are live and pinned. PLAIN is the committed `BLSTM_window 0.0`; OVERLAP is an
in-test `BLSTM_window 0.5` overlay resolving `window_size 25` / `window_shift 10` at 8 kHz
(a real overlapping grid, stride 10 over a 51-frame span), which is what exercises the fresh
accumulate/average loop. `overlap_and_plain_are_distinct_regimes` is the guard that keeps the
second leg from being a copy of the first: the exact path's own plain-vs-overlap posteriors
differ on 50/50 rows, and the fast path's on 47/50.

| leg | max_abs | max_rel | boundary count/type | max_dt |
|---|---|---|---|---|
| slstm plain / overlap | 7.34e-7 / 7.51e-7 | 3.69e-6 / 3.81e-6 | IDENTICAL | **0.0** |
| mamba plain / overlap | 1.87e-6 / 1.86e-6 | **1.81e-5** / 1.80e-5 | IDENTICAL | **0.0** |
| cfc plain / overlap | 9.52e-7 / 9.43e-7 | 3.25e-6 / 3.07e-6 | IDENTICAL | **0.0** |
| crossing legs (6, see below) | worst **2.00e-6** | worst 6.39e-6 | IDENTICAL | **0.0** |

PINS: `2.0e-4` relative (`measured * 10` rounded up, set by mamba) and `1.0e-4` absolute
(~50x headroom). The relative pin is LOOSER than the causal tier's `1.0e-4` and that is a
measurement, not a concession -- the bidirectional mamba row is 2.3x the causal one's
(7.74e-6), which is what a second recurrent stack plus a twice-as-wide dense fan-in buys.
Mamba's `max_rel` is a genuine relative number, not a small-denominator artifact: it is a
~1.9e-6 absolute delta over a posterior of ~0.10, well above the comparator's 1e-2 scale
floor.

**THE CROSSING SWEEP NEEDED A SECOND KNOB, and the reason is worth recording.** Phase 9's
decision-layer leg sweeps the output MLP's BIAS until the segmentation carries interior
boundaries. That is enough for five of the six (cell x regime) rows here, but NOT for
`slstm/plain`: its posterior spans `[0.105, 0.516]`, a logit swing of 2.20 against the 1.25 a
rising/falling round trip (0.6 / 0.3) needs, leaving no offset whose crossings survive
`min_speech`/`min_silence` 0.2 s (5 rows at this 0.04 s step). MEASURED: a 129-point
bias-only sweep over `[-8, +8]` finds ZERO interior boundaries there. The sweep therefore
tries `gain 1` (bias only, the phase-9 knob) first at every offset and reaches for a gain on
the dense weight ROWS only when that fails -- an affine map on the pre-activation, so both
cell stacks stay untouched and the compared crossings are crossings of the real curve.
Settled points: slstm plain `gain 2, +2.0` (1 boundary); slstm overlap `gain 1, +0.75` (1);
mamba plain/overlap `gain 1, +0.0` -- i.e. AS COMMITTED (3 each); cfc plain `gain 1, +1.0`
(1); cfc overlap `gain 1, +0.5` (1).

### Bench: RTF + peak RSS per cell, both regimes

`speech bench --repeat=1 --path={exact,fast}`, 3 independent fresh processes per (leg, path),
the committed 60 s stereo `phase4d/prcts_excerpt.wav` (`audio_s = 120.00`), the phase-9
bidirectional fixture configs + seed packs staged against it, backprop off,
`Neural_Networks_BackPropagation_Epochs 0`. Apple M4 Pro (arm64), macOS 26.5.2,
`cargo build --release -j 4`. Means of 3.

| cell | regime | exact rtf | fast rtf | speedup | exact maxrss_mb | fast maxrss_mb | rss ratio |
|---|---|---|---|---|---|---|---|
| slstm | plain | 0.001194 | 0.000175 | **6.82x** | 54.740 | 43.865 | 0.801x |
| mamba | plain | 0.001138 | 0.000175 | **6.50x** | 57.438 | 42.453 | 0.739x |
| cfc | plain | 0.001139 | 0.000178 | **6.40x** | 57.333 | 42.448 | 0.740x |
| slstm | overlap | 0.001361 | 0.000209 | **6.50x** | 53.422 | 41.297 | 0.773x |
| mamba | overlap | 0.001254 | 0.000227 | **5.52x** | 54.125 | 42.115 | 0.778x |
| cfc | overlap | 0.001257 | 0.000224 | **5.60x** | 53.433 | 41.338 | 0.774x |

READ THESE AS A FRONT-END MEASUREMENT, not a cell comparison (R5). The committed fixtures are
TINY nets (`23,4` recurrent / `8,1` output), so the three cells' fast wall-clocks land within
2% of each other in the plain regime (0.021003 / 0.020996 / 0.021363 s) -- the FFT + mel
front-end dominates and the cell is noise against it, exactly as the phase-9 regime-vs-cell
decomposition and the Task-5 causal bench row found. What the table does establish:

- **The fast RSS lands on the ~42 MB post-mel plateau for all three cells and both regimes**
  (41.3-43.9 MB), i.e. the Task-5 plateau is cell- AND direction-independent. A second
  recurrent stack costs nothing visible in peak RSS at this width, because the plateau is set
  by the front-end, not the net.
- **The exact path is 53-57 MB across the board.** The phase-9 finding that exact-Mamba costs
  ~2x (107.8 MB) is a WIDTH effect (the per-timestep SSM activation cache `Network` retains);
  at this fixture's width the three cells are within 4 MB of each other.
- **The overlap regime costs ~20-30% more wall-clock than plain on the fast path** (0.0210 ->
  0.0251 s slstm, 0.0210 -> 0.0272 mamba, 0.0214 -> 0.0269 cfc) and slightly LESS peak RSS
  (each window is a bounded block rather than the whole sequence). Windows overlap, so rows
  are forwarded more than once; that is the regime's definition, not a fast-path artifact --
  the exact path pays the same ~10-15%.

### The Twin's frozen-SAD gate: RE-EXAMINED, STAYS CONSERVATIVE

Spec S5 says the Twin's `bail_unsupported_shape` is revisited ONLY if a covering gate exists.
At this phase none did, so it stayed, and the doc comment says so explicitly: the bidirectional
twins land behind the algo-3 SAD driver; in Mode 7 the Twin's SAD net is never run (the
frozen-SAD contract); and `drivers/baseline.py` rejected `--cell`/`--direction` on the LID
arms outright. Since issue #21 the knob targets the LID net on those arms and the exact-tree
LID cell gates exist ("LID cell matrix" above). **Issue #57 lifted the gate** ("Issue #57 --
the fast Twin LID matrix" below): the Twin's LID net dispatches through the same total
classifier as the SAD net, and `bail_unsupported_shape` is gone.

### No streaming leg, by construction

A bidirectional net is UNSTREAMABLE: the reverse stack's state at time `t` is a function of
the samples AFTER `t`, so its first frame's output depends on the last one. There is no
bounded lookahead that makes it causal -- unlike the phase-8 windowed BLSTM (bounded by the
window) or the phase-9 causal cells (no lookahead at all). `StreamingSession::new` typed-bails
the shape.

WHY THE PHASE-8/9 STREAMING LEGS STAY GREEN UNMODIFIED, stated precisely (the earlier
"moved verbatim" phrasing was wrong and is corrected here): the refusal moved out of
`classify_fast_shape` -- which now NAMES the shape, because the offline tree implements it --
into the streaming session with its LEADING CLAUSE preserved (`cell type '<x>' is not
supported on the fast inference path ... in the BIDIRECTIONAL direction`) and its TAIL
REWRITTEN. The pins survive because `phase8_gate.rs`, `phase9_stream_causal.rs::validation_bails`
and `phase7_parity_lid.rs` assert PREFIX SUBSTRINGS, not the message body. The tail had
to change: the classifier's old advice ("run this config on the exact path") is now WRONG at
the streaming site, because the offline fast path DOES implement this shape -- it is
streaming, specifically, that cannot.

## Phase 10 -- the forward-only LSTM fast twin (Task 8, spec S6)

`fast::cells::FastLstm` -- the f32 CAUSAL peephole-LSTM step kernel -- completes the causal
set `{lstm, slstm, mamba, cfc}` and empties the last shape bail out of
`fast::driver::classify_fast_shape`, which is now a TOTAL function. Measured 2026-08-01,
Apple M4 Pro (arm64), macOS 26.5.2, `cargo build --release -j 4`.

### What landed, and what it is NOT

The new kernel is the CAUSAL twin: one forward stack, per-step `dot_f32` projections, a
`step(x_t, &mut LstmState)` the offline loop and the streaming session both drive. It does
NOT replace `fast::nn::FastBlstm`, which stays the BIDIRECTIONAL twin (batched faer
projection, offline only) and is BYTE-UNTOUCHED by this task -- the phase-7/8 suites passing
without edits are the proof. Two LSTM kernels, two regimes, two parity legs, deliberately.

THE ONE RULE THAT SHAPED THE CODE (spec S6, the `DenseRowChain` lesson): every projection is
a per-step ascending `dot_f32` over a contiguous weight row, FROM DAY ONE. Reusing the
batched `lstm_layer_forward` would have made offline and streamed numbers differ in the last
f32 ULP at re-chunked granularities -- the exact drift the phase-8 gate exists to forbid --
so the batched kernel is never reachable from this path.

### Transcription fidelity (`fast::cells` unit tier, CI)

The oracle is the exact f64 `nn::layers::LstmLayer::feed_forward` per-timestep body. Same
weights, same input, all three peephole families live:

| leg | shape | max_abs | max_rel | pin |
|---|---|---|---|---|
| `lstm_matches_the_exact_cell_within_the_f32_band` | 5 -> 3, T=11 | 1.05e-7 | 1.96e-7 | `CELL_F32_PIN` 5e-6 |
| `lstm_matches_the_exact_cell_at_the_arm_geometry` | 92 -> 24, T=31 | 1.42e-7 | 1.42e-5 | 1.5e-6 abs / 1.5e-4 rel |

Both absolute deltas are ~1 f32 ULP of an O(1) output (`f32::EPSILON` = 1.19e-7), i.e. the
narrowing and nothing else. The 92x24 row's RELATIVE number is FLOOR-LIMITED and says so:
it is exactly 100x the absolute one, which is the tell that `max_rel`'s `1e-2` scale floor is
the denominator (an LSTM output is `o_t * asinh(c_t)`, and a shut output gate drives it
arbitrarily close to zero). The absolute pin is the discriminating statement there; the
relative pin is widened for that leg only rather than pretending 5e-6 means something against
a 1e-2 denominator. Same honesty the CfC cell's pins carry.

THE PEEPHOLE FAMILIES ARE PROVEN LIVE, which is what makes those two rows worth their ink.
`lstm_peephole_families_are_distinguishable` toggles each of the three flags independently
and measures the output shift from dropping it: **8.96e-2** (cells), **8.32e-2** (gates),
**3.92e-2** (gates-recurrent) -- five decades above the f32 band, so a mis-assigned peephole
row cannot hide in a dead branch. With each family off, fast still tracks exact inside
`CELL_F32_PIN`.

Also pinned: the weight count against `LstmLayer::nb_of_weights` at five shapes (and that the
`12*O` bundle is reserved regardless of the flags), the committed fixture's 1651-element pack
length against the Python packer's, wrapper-is-the-step-loop, run-twice bit-identity, and the
split-state legs (which check ALL THREE carried vectors are non-trivial at the cut).

### Driver parity: exact f64 vs fast f32 (`phase9_fast_parity.rs`, CI)

The committed `lstm_forward` fixture (new, same `scripts/extract_phase9_fixtures.py` recipe,
seed 1008001, 1651-element pack) through `BagOfProcessors` on the synthetic tier-2 excerpt:

| leg | max_abs | max_rel | max_dt | interior boundaries |
|---|---|---|---|---|
| plain (fixture as committed) | 4.40e-7 | 3.12e-6 | **0.0** | 0 |
| crossing (output gain 3, bias +3) | 1.30e-6 | 1.05e-5 | **0.0** | **3** |

Segment count and types IDENTICAL, `max_dt` EXACTLY 0.0, against the unchanged `POST_*_PIN`
of 1e-4 (~9.5x headroom). The crossing row is now the WORST of the eight (cell x leg) runs,
displacing mamba's 7.74e-6 -- structurally, not alarmingly: the gain lever sharpens the logit
3x, so the same input-side f32 delta lands on a steeper part of the logistic.

**A NEW LEVER WAS NEEDED, and the reason is measured.** Every other cell's crossing sweep
shifts the output bias. On the LSTM fixture that CANNOT work: a 129-point bias sweep over
`[-16, +16]` at 0.25 yields ZERO interior boundaries at every offset, even though the
posterior traverses the full `[0, 1]` range across the sweep. The cause is the shape of this
net's posterior on the excerpt -- its highest rows are its EARLIEST, so as the level rises the
hysteresis latches SPEECH at frame 0 and the whole file becomes one segment, and as it falls
nothing crosses at all. A pure level shift cannot manufacture an interior edge out of a
monotone-onset curve. Scaling the output layer's four WEIGHTS can: it multiplies the logit's
variable part (sharpening contrast) while the bias re-centres it, and it is the same class of
intervention -- output-layer only, post-recurrence, every cell weight untouched. A 2-D
(gain x offset) probe found gain 3 / bias +3 the richest rung (3 interior boundaries,
posterior span `[0.014, 0.842]`: sharpened, not saturated). The escalation is a SWEEP, not a
hardcoded pair, and it only runs when gain 1 finds nothing -- so the slstm/mamba/cfc rows
settle on exactly the offsets they settled on before.

### Streaming: streamed vs offline (`phase9_stream_causal.rs`, CI)

All 19 legs green with the `lstm` row added, FIRST RUN, no machinery changed -- `StreamCausal`
drives `FastCell`/`FastCellState` generically, which is the S5.2 "zero new surface" claim
being cashed again:

| leg | lstm result |
|---|---|
| crossing sweep (60 s staged fixture) | bias `+0.5` -> **4** interior boundaries (6 segment rows) |
| `finish()` vs offline fast, plain (type 0) | boundary `max_dt` EXACTLY **0.0**, posteriors bit-equal (1500 rows) |
| `finish()` vs offline fast, calibrated type 1 | boundary `max_dt` EXACTLY **0.0**, posteriors bit-equal |
| chunk invariance (20/100/1000/7 ms) | segmentation + posteriors + emission set bit-identical |
| prefix consistency | 4 mid-stream emissions, ZERO retractions |
| real-arm geometry (`23,24,24` / `4,1`, wide dense) | 143 rows -> 35 posteriors, bit-identical at chunks 1/3/4/7/143 |
| latency | bound **1.73400 s** (identical to the other three cells), speech lag 1.75688 s, tight margin **-0.07712 s** |

THE LSTM ROW IS THE ONE THIS SUITE IS BEST PLACED TO CATCH A BUG IN, and that is worth saying
plainly: the peephole LSTM carries the RICHEST state of the four cells -- `h`, `c`, AND the
previous step's POST-activation gate row, which three separate peephole families read at
`t-1`. A kernel that rolled `h` and `c` but not `gates` would compute correct whole-sequence
output and WRONG streamed output at every chunk boundary. The split-state and chunk-invariance
legs are what measure that; the `synth_net` helper's peephole flags were flipped from
`[false; 6]` to `[true; 6]` for exactly this reason (inert for the other three cells, which
have no peepholes, so those rows are bit-unchanged).

The identical 1.73400 s bound across all four cells is the point rather than a coincidence:
the fixture configs are byte-identical bar the cell keys, so the causal latency win is a
property of the REGIME, not of any cell -- now demonstrated with the legacy cell itself.

### Bench: the `n/a` cell FILLED, and the fast-tier regime control

`speech bench --repeat=1 --path={exact,fast}`, 3 independent fresh processes per (row, path),
the phase-9 corpus bench recipe UNCHANGED (one 75 s mono 8 kHz corpus wav selected at runtime,
no filename recorded; each config repointed at its own phase-9 trained checkpoint; backprop
off, `Epochs 0`, `numOuterThreads 1`). RE-MEASURED on the current build, so these four rows
are self-consistent with each other and NOT with the phase-9 table (whose fast rows predate
the Task-5 f32 mel).

| cell / direction | window regime | path | wall_s mean [range] | rtf | maxrss_mb | fast vs exact |
|---|---|---|---|---|---|---|
| lstm / bidirectional | 3.25 windowed overlap | exact | 0.167335 [0.166705-0.167773] | 0.002231 | 53.97 | baseline |
| lstm / bidirectional | 3.25 windowed overlap | fast | 0.034228 [0.034102-0.034334] | 0.000457 | 37.21 | **4.89x** |
| lstm / forward | 0, plain | exact | 0.104963 [0.099500-0.112279] | 0.001400 | 54.37 | baseline |
| lstm / forward | 0, plain | fast | 0.017587 [0.017419-0.017764] | 0.000234 | 37.42 | **5.97x** |

**THE FAST-TIER REGIME CONTROL** -- the number phase 9 could not measure, because the
`lstm / forward` fast cell did not exist. Same cell, same weights-shape lineage, only the
regime differs (windowed-bidirectional -> plain-causal):

- EXACT tier: 0.167335 -> 0.104963 = **1.59x** (phase 9 measured 1.66x on its own build).
- FAST tier: 0.034228 -> 0.017587 = **1.95x**.

So the regime win is LARGER on the fast path than on the exact one, and the direction makes
sense: the f32 tree's per-window overhead (the overlap accumulate/average pass and the
re-forwarding of overlapping rows) is a bigger share of a much smaller total once the
front-end is f32 end to end. Phase 9's conclusion is unchanged and now doubly grounded -- the
causal speedup is a REGIME effect, not a cell effect.

MEMORY: the fast path lands at 37.2-37.4 MB on both regimes (the Task-5 post-f32-mel plateau,
here on a mono 75 s file rather than the 60 s stereo fixture that measured ~42 MB), against
53.97-54.37 MB exact -- a **0.69x** ratio, i.e. the phase-7 SAD-arm memory regression stays
reversed on this cell too.

### The four sibling assertions this flip touched, and why each moved

Retiring the last shape bail invalidated four `must bail` assertions in suites this task
otherwise leaves alone. Each was INVERTED or RE-AIMED rather than deleted, and none was
weakened:

| suite / leg | before | after |
|---|---|---|
| `phase9_fast_parity::fast_dispatch_bails_and_builds_per_cell_and_direction` | `(lstm, forward)` -> shape bail | an sLSTM-sized pack under `Cell_Type lstm` must fail on LENGTH (1603 vs 1651), which is what the shape bail was protecting |
| `phase10_bicell_parity::fast_dispatch_builds_bidirectional_cells` | `(lstm, forward)` -> shape bail | must BUILD, from the matching committed pack |
| `phase7_parity_sad::fast_dispatch_per_cell_type_and_direction` | `Direction forward` -> shape bail | must BUILD (with the untagged-pack caveat that leg already documents, in its other direction: an over-long bidirectional pack is consumed head-first by design) |
| `phase8_gate::validation_bails` | `Direction forward` -> shape bail | still bails, on the WINDOWING rule instead (`causal streaming requires the plain regime`) -- tier2 carries `BLSTM_window 3.25`, and that is the refusal that actually matters for streaming |
| `phase7_parity_lid::fast_twin_bails_on_unsupported_cell_type_and_direction` | classifier's `Direction 'forward' ...` wording | the TWIN'S OWN gate's wording (`cell type 'lstm' is not supported ...` + `bidirectional twin only`), because `bail_unsupported_shape` -- deliberately conservative, re-affirmed in Task 7 -- is now what catches a causal LID net |

The Twin's frozen-SAD/LID gate is UNCHANGED in behaviour: it still refuses every shape that
is not `FastNetShape::Blstm`. Only the message a causal LID config receives changed, because
the classifier no longer produces one.

### The corpus check, reported HONESTLY as degenerate

The same 75 s corpus file, same trained forward-LSTM checkpoint, run through BOTH paths with
separate dump directories: the emitted VRCTS xml is BYTE-IDENTICAL. That is a true statement
and a WEAK one, so it is recorded as corroboration rather than evidence: the phase-9 subset
checkpoint is MODE-COLLAPSED on this file (a single all-speech segment spanning `0.0000` to
`74.9999`), exactly the degeneracy the phase-8/9 corpus tiers already document for these
subset nets. A threshold sweep confirms there is nothing to find -- rising/falling at
0.9/0.99/0.999/0.9999 all give the same one segment, and at 0.999999 both paths give zero
segments. The DISCRIMINATING parity evidence for this cell is the committed-fixture tier
above (3 interior boundaries at `max_dt` 0.0) and the streaming tier (4 boundaries, bit-equal),
not this run.

## Phase 10 -- inference-only retention gating (Task 9, spec S7)

The phase's ONE adjudicated behaviour-free touch inside `nn/` but outside `nn/cells/`. Two
backward-only retentions -- Mamba's per-timestep `MambaCache` and `Network::layers_output` --
are switched off at CONSTRUCTION for any `BlstmNetwork` whose backward can never run, closing
the phase-9 exact-Mamba memory finding.

### The condition, and the one it deliberately is not

`BlstmNetwork::from_config` sets `inference_only = !BackPropagationActivated`, once, for the
life of the net. Soundness is structural, not empirical: every backward inside `BlstmNetwork`
funnels through `feed_forward_backward_plain` / `_mlp` (the four windowed drivers call one of
those two per window), both gated on `back_propagation_activated && target.nrows() > 0`, and
`feed_backward` / `feed_backward_mlp` are PRIVATE. So `!BackPropagationActivated` proves the
backward unreachable, statically, at construction.

That is a STRICT SUBSET of spec S7's stated condition (backprop active AND references
present). Reference presence is a per-corpus-item property decided long after construction, so
it is deliberately unused: a backprop-on net with no references keeps retaining, which costs
memory and never correctness. The asymmetry is the safe one.

`Neural_Networks_BackPropagation_Epochs` is NEVER read here -- the F11 lesson. `Epochs 0` +
`BackPropagationActivated true` is exactly the modern loop's seam shape (one fold at theta,
gradient harvested), and gating on `Epochs` would silently zero every gradient
`drivers/train.py`'s SMORMS3 loop reads. `tests/pyo3/test_phase9_seam.py::
test_epochs_zero_with_backprop_on_still_returns_a_real_gradient` is the standing regression.

### Why "skip the fill" alone was not enough

Skipping the cache ASSIGNMENT moved peak RSS from 109.2 to 81.4 MB -- one layer's worth, not
the whole gap. `getrusage` reports a high-water mark, and `abar`/`h` (each
`T x (d_inner d_state)`, ~11.5 MB per layer on this recipe) are LIVE during the forward
whether or not they are kept afterwards. But neither is read by the forward at a distance:
`abar` is write-only there and `h` is read only at `t-1`. So under `retain_cache == false`
they shrink to rolling buffers (1 row / 2 alternating rows) and the recurrence runs on the
same numbers, in the same order, from the same reads. The retain path's indexing is
untouched. `nn::cells::mamba::tests::forward_is_bit_identical_without_the_cache` is the pin
(3 sequence lengths incl. `T = 1` and `T = 2`); a mutation collapsing `tp` to `th` fails it.

### The measurement (spec S10.4)

`speech bench --repeat=1 --path=exact`, 3 INDEPENDENT fresh processes per row (the phase-7/9
recipe, so `maxrss` is a clean per-process high-water mark), ONE 75 s mono 8 kHz corpus wav
(runtime sorted-first, no filename recorded), `lre_sad.toml` under the `cell_overlay(cell,
forward)` architecture overlay -- i.e. the phase-9 S8.7 geometry -- with `BackPropagationActivated
false`, `Epochs 0`, `numOuterThreads 1`. Weights are a runtime `init_weights` seed pack rather
than a trained checkpoint: peak RSS is weight-INDEPENDENT, and this keeps the row reproducible
without a checkpoint. Apple M4 Pro (arm64), macOS 26.5.2, `cargo build --release -j 4`.

| row | maxrss_mb (3 runs) | mean | wall_s mean |
|---|---|---|---|
| mamba / forward, exact -- BEFORE (base `83b56aa`) | 109.875 / 107.766 / 109.922 | **109.188** | 0.117746 |
| mamba / forward, exact -- retention skip only | 82.453 / 82.000 / 79.812 | 81.422 | 0.118049 |
| mamba / forward, exact -- AFTER (retention skip + rolling `h`/`abar`) | 59.969 / 57.844 / 57.844 | **58.552** | 0.112976 |
| slstm / forward, exact -- CONTROL, same recipe + build (untouched by T9) | 55.625 / 55.594 / 55.547 | 55.589 | 0.097868 |

(The AFTER and CONTROL rows are the FINAL re-measurement on the finished build, taken on an
otherwise idle box; the intermediate "retention skip only" row is the one-variable probe that
motivated the rolling buffers.)

**1.86x less peak RSS on the exact-Mamba row (109.19 -> 58.55 MB), and the ~2x cell penalty
is gone**: Mamba now sits 3.0 MB above the sLSTM control on the identical recipe, where before
it sat ~53 MB above. This closes the phase-9 S8.7 memory finding ("the one place a cell choice
is visibly expensive") -- with the phase-10 Task-7 caveat still standing that the ~2x was a
WIDTH effect, invisible at the tiny committed-fixture width. Wall-clock is unmoved (0.1177 ->
0.1130 s mean, inside the per-row spread); the change frees allocations, it does not add work.

The 109.19 MB BEFORE row reproduces phase 9's 107.82 on the same audio length and geometry
with a different (untrained, and irrelevant) weight pack, which is what makes the two tables
comparable. Local one-off measurement, same posture and license discipline as every other
corpus bench row here: not a committed automated test, no automated enforcement.

### The golden evidence, SPLIT -- the two halves are not equally covered

R2 names the committed golden suite as the arbiter, and it earns that name for ONE of the two
changes here. Splitting them is the honest accounting.

**The `layers_output` half IS live-golden-exercised.** Determined by PARSING each fixture
(the config parser is last-wins, and a file-level `rg` is blind to a later override -- an
error this section previously made): `BackPropagationActivated` resolves FALSE on
`phase0/1_worker_1.config` (the real 2015 production config), `phase0bii/lid.config`,
`phase2b/signal.config`, thirteen `phase4b` configs (`lid5`, `twin_e2e`, `twin_mode0` through
`twin_mode7` incl. `twin_mode0_concat` / `twin_mode7_ppm1` / `_ppm2`), all four `phase4d`
configs (`tupleA`/`tupleB_1_worker_1` + both `parity_*`), and the SAD net of both `phase9`
Mode-7 Twins. Every one now runs with the retention OFF and stays byte-identical.
CORRECTION: `phase4a/tier2_spectral.config` and `tier2_gradcheck.config` were previously
listed here and do NOT belong -- both carry a later Task-9 override block setting
`BLSTM_BackPropagationActivated true`, so they resolve TRUE and take the retaining path.

FREE PER-NET-GRANULARITY EVIDENCE, unclaimed until now: five committed configs are MIXED --
`phase4b/twin_train` + `twin_train_ns` and `phase9/twin_mode7_lid_{slstm,cfc}` (SAD false,
LID true) and `phase4c/genome_twin` (SAD true, LID false). Inside ONE bag, one net retains
and the other does not, and the goldens still match byte for byte -- so the flag is genuinely
PER NET and does not leak across the pair.

**The rolling-buffer half has NO committed golden, and that must be said plainly.** Every
backprop-false fixture above runs the legacy LSTM cell, and every committed Mamba config
(`phase9/mamba_{bidirectional,forward}`) is backprop-TRUE, so no golden ever executes the
`retain_cache == false` indexing. Its arbiters are three purpose-built legs, in ascending
strength: `nn::cells::mamba::tests::forward_is_bit_identical_without_the_cache` (bare cell,
`T = 1 / 2 / 9`); `nn::blstm::inference_only_tests::the_forward_is_bit_identical_either_way`
(the real net, all four cells); and the headline --
`tests/pyo3/test_phase9_seam.py::test_turning_backprop_off_does_not_move_the_forward`, which
runs the EXACT Mamba cell over a real corpus file through `speech_rs.Engine` with the rolling
buffers ACTIVE and asserts `results_matrix` bit-identical to the retaining run. That last leg
is the closest thing to a golden this half has, and it is the one to point at.

### What is NOT gated -- and why that is a follow-on, not a nothing

This phase gates MAMBA's cache only. The other three cells are NOT cache-free: `LstmLayer`
holds `gates`/`cells_in`/`cell_states`, `SlstmLayer` holds
`gates`/`cell_states`/`norm_states`/`m_states`, `CfcLayer` holds
`z_cache`/`backbone_pre`/`backbone_post`/`heads` -- each on the order of `T x 6-7 O`, all
backward-only, all still retained unconditionally. Gating them is a NAMED FOLLOW-ON with a
real (if smaller than Mamba's `T x d_inner d_state`) win. `CellLayer::set_retain_cache` is a
no-op on those arms because this task did not measure or pin them, NOT because there is
nothing there; a no-op is simply the safe direction, since it means "always retain". The fast
f32 path is unaffected either way: it never built any of these caches.

### The loud bails

A backward after a non-retaining forward PANICS with a named message, in both places
(`MambaLayer::feed_backward`, and `Network::drive_backward` -- the single choke point behind
`feed_backward`, `feed_backward_reverse` and `feed_backward_double`, all three named in the
message) -- never a silently wrong gradient.
Both are pinned by `#[should_panic]` legs with a retaining contrast beside them, and by the
R6 clone legs (a per-lane bag clone under the flag carries EMPTY caches, asserted at the cell,
the network and the `BlstmNetwork` level). Inverting the gating (S9.4 mutation 7) fails 7 Rust
legs and all 4 phase-10 seam legs.

## Phase 10 -- the NAMED follow-ons (nothing lost, nothing promised)

Collected at the phase closeout so each one is a tracked sentence rather than a report
paragraph nobody reads again. None is a defect; each is a place where this phase deliberately
stopped, with the reason and the shape of the work.

- **The `fast/cells.rs` <-> `fast/bicell.rs` scaffolding dedupe** (Task 7, approach A, ~55
  lines by T7's own count). `FastCausalNet` and `FastBiCell` already SHARE the parts that
  matter -- `cell_weight_count`, `build_cell`, `cell_stack_forward`, `DenseRowChain`, and the
  `window_begin`/`window_end` span helpers -- but each carries its own `element_count` walk,
  its own dense-MLP construction in `from_flat`, and its own per-row output drive. Approach A
  ratified dedupe-LATER over unification churn precisely so `FastBlstm`/`overlap_window_step`
  would stay byte-untouched; the merge is a refactor to do when a fifth shape arrives, not
  before, and the fast-vs-exact parity legs are what would keep it honest.
- **A corpus tier for `FastBiCell`.** The bidirectional twins are pinned on committed
  fixtures (`phase10_bicell_parity.rs`) but never carried onto real data, unlike the causal
  cells (`test_phase9_parity.py`). The blocker is not the machinery, it is the checkpoints:
  Task 4's bidirectional gate runs are EPHEMERAL (the gates train into a tempdir and discard),
  so a corpus leg needs a cached-checkpoint recipe first -- train once per (cell, direction)
  into a stable local path, then run the phase-9 fast-vs-exact metric comparison against it.
- **The f32 mel bank-table pin, upgraded from geometry to VALUES** (Task 5, review M3). The
  committed guard cross-checks the f32 bank's SHAPE metadata (`is_mel` / `is_dct_activated` /
  `nb_filters` / `nb_dct`) against the exact bank and then compares assembled outputs at a
  tolerance. `tests/phase1_mel_golden.rs:160-191` already shows the stronger move: drive a
  ONE-HOT periodogram through `apply_filter_bank` and the output IS a single bank coefficient,
  recoverable bit-for-bit through the public surface with no accessor. Repeating that per
  filter would pin the f32 coefficient TABLE exactly (against the exact table narrowed `as
  f32`), turning a tolerance comparison into an equality one.
- **A `Cfc_Backbone_Layers >= 2` streaming fixture** (Task 6, M4; re-confirmed by Task 11's
  rider 9). Both committed CfC fixtures are `L = 1`, so the multi-layer backbone chain is
  exercised by the exact cell's unit/FD tiers and the Python reconstruction pins but NOT by
  the fast twin or the streaming gate. No lever buys this one -- it is a FIXTURE property, so
  closing it means regenerating `cfc_forward` at `L >= 2` through
  `scripts/extract_phase9_fixtures.py` and re-measuring that row's pins.
- **Retention gating for the other three cells** (Task 9). This phase gated MAMBA's cache
  only, because that is the one the phase-9 bench measured as visibly expensive. `LstmLayer`
  (`gates`/`cells_in`/`cell_states`), `SlstmLayer`
  (`gates`/`cell_states`/`norm_states`/`m_states`) and `CfcLayer`
  (`z_cache`/`backbone_pre`/`backbone_post`/`heads`) each retain `T x 6-7 O` backward-only
  buffers unconditionally; `CellLayer::set_retain_cache`'s no-op arms are the SAFE direction
  (always retain), not an absence of work. The win is real and smaller than Mamba's.
- **Symbol/anchor citations instead of line numbers into LIVE files.** This phase hit the
  failure twice in one task: Task 7's `fast/bicell.rs` overlap table cites `nn/blstm.rs` line
  spans that Task 9 silently invalidated (it inserted 55 lines above the cited block, so every
  number there needs +55 at HEAD -- now stated in the file rather than renumbered), and Task
  12's own first draft of the CfC section above cited `test_phase9_seam.py:379` when its own
  docstring edit had already pushed those pins to `:387`. Line citations into files this repo
  keeps editing decay by construction; the pattern that does not is what the CfC section above
  uses -- name the SYMBOL (`GRAD_CHECK_PINS`, `feed_forward_backward_overlap`,
  `cfc_backward_matches_central_difference`) and treat the number as a convenience. Converting
  the existing ones tree-wide is mechanical but wide, so it is named here rather than done in
  passing; citations into `legacy/` C++ sources are NOT affected (that tree never moves).
- **A sub-unit gain rung for the streaming crossing sweep** (Task 11, recorded as a new
  observation). The `mamba` plain leg measures its 16 interior boundaries over a posterior
  span of `[0.0000, 1.0000]` at the NEUTRAL lever -- nothing swept it there, the committed
  fixture is simply saturated -- so part of that row's richness is a saturation artefact,
  where CfC's 6 at `[0.1229, 0.7759]` are not. Extending `GAIN_LADDER` BELOW 1 would let the
  sweep de-saturate such a row instead of accepting it. Nothing about the bit-equality claims
  changes either way; this is about how much the boundary comparison is worth.

---

## Interstitial -- the incremental resmooth (streaming decision layer, bounded per-push cost)

Branch `feature/stream-incremental-resmooth`, off `main@afefbb4` (phase 10 merged). Not a
phase: one gap, closed, with its proof. `fast::stream::StreamDecision::resmooth_and_emit`
re-smooths on every push, and its input -- `raw_segments` -- is APPEND-ONLY, so the phase-8/9
form replayed the whole list every time: `O(S)` `label_segment` calls each walking an `O(S)`
list, i.e. `O(S^2)` per push and quadratic-in-`S` cumulative on an unbounded stream. Memory
was never the issue (16 B/segment); COMPUTE was the online mode's scaling gap.

### What landed

A frozen prefix + a bounded retained window, spliced. `keep_from` freezes raw segments out of
the replay; `frozen` carries the smoothed entries with `begin < cut_time` verbatim; the
retained build supplies everything from `cut_time` rightward. `cut_time` advances to the SAME
`threshold` the emission uses (`frontier - holdback`), so freezing and emitting are ONE act
with one frontier -- there is no second latency-affecting quantity to keep honest, and the cut
never gates an emission.

The invariant is `raw_segments[keep_from].0 + CUT_MARGIN_HOLDBACKS * holdback <= cut_time`
with `CUT_MARGIN_HOLDBACKS = 2.0` -- expressed in HOLDBACK UNITS, never a second per-term
constant (the phase-9 T9 adjudication). The derivation (in `StreamDecision`'s docs) needs
`1 * holdback + 2e-6`; the second holdback is the slack that dominates `add_padding`'s `1e-6`
re-close epsilons and anything the eight-step induction under-counts.

### The cut argument, in one line per direction

- RIGHTWARD (nothing FUTURE changes the frozen prefix): verbatim the existing emission
  argument -- future raw segments begin at `>= frontier` (three terms, `pending_begin` clamp
  included) and the smoothing's leftward reach is `<= holdback`. Finality is ABSOLUTE (the
  future set only shrinks), which is why `cut_time` is a running max and survives the
  frontier retreating, as the `pending_begin` `min` can make it do.
- LEFTWARD (truncating the replay's left context changes nothing at or right of the cut):
  NEW. The full and retained pre-smoothing lists are identical at `>= X` (the first retained
  raw begin) including the carried type; only the leading `Other`'s DURATION differs. Each of
  the 8 steps is local in decision (`suppress_short` reads a 3-entry window plus `i == 0` and
  `i+2 == len`, with `previous_type == segs[i-1].ty` an INVARIANT rather than a walk
  accumulator) and in effect, and a difference outruns a step only by FLIPPING a straddling
  entry -- which every branch conditions on `dur <= threshold`, bounding the flipped entry's
  right end. Composing: reach `<= sum(min_speech) + sum(min_silence) + padding[1] +
  padding[3] + 2e-6 <= holdback + 2e-6`. Only the two `after` paddings enter (a `before`
  padding extends Speech LEFTWARD, away from the region at risk), so `holdback` -- which sums
  `before` too -- already over-covers.

### The oracle (`src/rust/tests/stream_incremental_resmooth.rs`, CI, 13 legs, 6.9 s)

The committed streaming gates were BLIND to the phase-9 T9 bug (19+19 green with the fix
reverted), so this change does not lean on them. The oracle runs the two paths SIDE BY SIDE
and compares the emitted sequence (all four fields, `to_bits`) and `flush()`'s `Segmentation`
(`to_bits` + types + duration). The reference is the phase-8 CODE, not a paraphrase:
`set_full_replay(true)` merely stops the cut advancing, leaving `keep_from == 0`, `frozen`
empty and the splice index 0.

| profile | why it exists | holdback | true reach |
|---|---|---|---|
| `tier2` | the committed gate config, verbatim | 2.28957 | 1.68510 |
| `silence-active` | tier2 clamps BOTH `min_silence` to ZERO, so steps 4/7 -- the derivation's LOAD-BEARING seam case -- are NO-OPS on it | 3.21953 | 1.96505 |
| `suppress-heavy` | `padding[0] = padding[2] = 0`, so the true reach EQUALS the holdback and the margin's slack is at the design's minimum | 1.90000 | 0.80000 |
| `no-conv` | no convolution: bursts map to raw segments directly, so tiny near-threshold segments and sub-reach gaps get dense | 2.68957 | 1.68510 |

MEASURED coverage across 4 profiles x 6 seeded streams x 2-4 chunkings (chunk 1 through
whole-stream): 2919 raw segments, max 296 per stream, **9418 emissions and 9490 final
segmentation entries compared bit-for-bit**, 476 gaps in the reopen band, 736 sub-reach gaps
(the merging regime that drives seam case 4), 268 near-`min_speech` segments, 527 long
segments (the `pending_begin` stall regime), 27422 pushes whose retained window STARTS on a
segment straddling the truncation limit. Every one of those is an ASSERTED floor, not a
printed hope -- and since the review, the `tiny` and `long` floors are asserted PER PROFILE
too, which is what exposed two structural holes a global sum had been covering for (see the
review-fixes subsection below).

### The mutation ladder -- what the oracle can and cannot see

| mutation | result |
|---|---|
| `CUT_MARGIN_HOLDBACKS = -3.0` | FAILS, first case (emission count 300 vs 296) |
| `-1.0` | FAILS on `tier2` |
| `-0.5` | FAILS on `silence-active` |
| `-0.25` | FAILS on `suppress-heavy` |
| `0.0` | PASSES |
| freeze at `threshold + 1.0*holdback` | FAILS on the SECOND emission of the first case (end 7.2322 vs 4.7921) |
| freeze at `threshold + 0.25*holdback` | PASSES |

Read honestly: the oracle's resolution is about a QUARTER of a holdback in both directions,
so the shipped margin of 2.0 sits two holdbacks above the empirical failure point and one
above what the derivation requires. The streams realize well under a holdback of actual
reach; the extra margin is bought by the DERIVATION, not by the measurement, which is the
correct division of labour -- the same one the holdback itself has had since phase 8.

### Bounded cost -- derived, measured, and pinned

`K_derived = 3 + floor(window/dt)` with `window = 3*holdback + conv_delay + dt`, from the
cut invariant plus "at most one raw segment is LABELED per convolved value". CONFIG-ONLY:
the review's minor 8 removed an `open_span` term that carried the generator's burst cap and
made the theorem read as experiment-conditional -- a long OPEN segment widens the window but
cannot ADD retained segments, because no raw segment CLOSES while one is open. That
TIGHTENED it from 450-553 to **155-254** across the four profiles (tier2 184 /
silence-active 254 / suppress-heavy 155 / no-conv 205), against a measured worst
of **6**. It is still DELIBERATELY loose (it assumes the hysteresis can close a segment at
every grid step, which the area gating forbids). So the derived bound is the SAFETY pin -- it is what makes bounded per-push
cost a theorem, since it does not reference the stream at all -- and a measured*10 regression
pin of 50 is the discriminating one. Tightening the derivation would mean bounding the
posterior VALUES, which are a property of the net and the kernel, not of this layer.

Measured retained window, every profile x seed x length: **3-6 segments**, flat as the raw
list doubles (e.g. `no-conv` seed 11: raw 200 -> 399, retained 6 -> 6). The doubling leg
asserts the raw list grows ~2x while `retained * 8 <= raw`.

WALL CLOCK (M4 Pro, informational -- printed by the oracle, never asserted; a wall-clock CI
assert has no place here, and the print now reports the build profile it actually ran under
rather than claiming one). A 42495-row stream = 1699.8 s of audio, 267 raw segments, 664
pushes at chunk 64:

| build | full replay | incremental | speedup | per push |
|---|---|---|---|---|
| release | 69.0 ms | 0.9 ms | **77.6x** | 103.9 us -> 1.3 us |
| debug | 511.9 ms | 12.2 ms | 41.9x | 770.9 us -> 18.4 us |

The ratio is not the point -- it GROWS with stream length, since the full path's per-push
cost rises with `S` and the incremental path's does not.

### The review fixes -- two seam/coverage findings the first pass missed

The review re-derived the whole cut argument independently and returned Spec PASS with two
Importants, both landed. Recorded because each is a general lesson, not a typo.

**A seam case whose REASON did not cover its own sub-case (the T9 shape, again).** Case 1
said "the entry at `X` is Speech preceded by Other in BOTH lists, so no merge crosses the
seam". That is FALSE when `e_{k-1} == b_k` exactly in f64: `label_segment`'s insertion walk
stops BEFORE `O@e_{k-1}` (`segs[i].begin < begin` is false at equality), inserts `S@b_k` in
front of it, and the overwrite loop erases it -- leaving adjacent Speech entries that
`sanitize` merges, DELETING the entry at `X` the retained list keeps. The CONCLUSION survived
(`D_1 = X^+`, absorbed by the slack); the reason did not. Exact arithmetic separates the two
strictly; f64 need not, when the rising interpolation's fraction rounds to 1 at the next step
index. Fixed as two sub-cases (disjoint / touching) AND -- more than documentation --
`advance_cut`'s slide is now STRICT (`< limit`), so the invariant reads `X + margin <
cut_time` and `X^+ <= cut_time` holds even at `holdback == 0`, where there is no slack at all
and a non-strict slide would have let the splice READ the one entry that seam can differ on.

**Global non-vacuity floors hid two STRUCTURALLY UNREACHABLE regimes.** The `tiny` and `long`
floors were global-only, and the two profiles that most needed them were each missing one:

| hole | measured cause |
|---|---|
| `silence-active` long = 0 on all six seeds | the old `4*holdback` criterion = 12.88 s = **322 frames > the 300-frame burst cap** -- so the `pending_begin` stall regime never occurred on the ONLY profile whose `min_silence` is live |
| `suppress-heavy` tiny = 0 on all six seeds | the 19-tap convolution imposes a **floor of 0.32089 s** on raw-segment duration (identical on every convolved profile: a property of the kernel and the rising/falling thresholds, not of the stream), and that profile's `max(min_speech)` was 0.30 -- UNDER the floor |

Diagnosed by printing the achieved per-profile duration range rather than guessing. Fixed by
making `long` holdback-RELATIVE and reachable (`>= holdback`, the delay the clamp actually
imposes), lifting `suppress-heavy`'s `min_speech[0]` 0.30 -> 0.35 above the measured floor
(its defining property `padding[0] == padding[2] == 0` stays exact: reach = holdback = 1.90),
and adding PER-PROFILE floors whose failure message names the unreachable-vs-unlucky
distinction and prints the duration range, so the next such hole diagnoses itself.
Post-fix, all four profiles hit all four regimes:

| profile | raw | gaps in band | sub-reach gaps | tiny | long | raw duration range |
|---|---|---|---|---|---|---|
| `tier2` | 640 | 68 | 155 | 20 | 150 | [0.32089, 12.62410] |
| `silence-active` | 664 | 95 | 152 | 24 | 88 | [0.32089, 11.93045] |
| `suppress-heavy` | 633 | 146 | 56 | 19 | 182 | [0.32089, 20.57045] |
| `no-conv` | 982 | 167 | 373 | 205 | 107 | [0.03552, 11.99552] |

(`no-conv`'s 0.03552 minimum is the control confirming the 0.32089 floor belongs to the
convolution: remove the kernel and it disappears.)

TWO NAMED FOLLOW-ONS, recorded in the module doc rather than done: a REALISED-REACH
diagnostic (instrument `D_8 - X` directly -- a pass/fail ladder localises the safe margin
only to a quarter-holdback, so nobody should shrink `CUT_MARGIN_HOLDBACKS` on ladder evidence
alone), and the straddle-count headline (it is PER PUSH, so fine chunkings inflate it; honest
as a floor, misleading as a headline -- distinct straddling raw-segment indices is the
quantity worth reporting).

### Posture

Behaviour-identical by construction and by oracle; INFERENCE-ONLY and MONO-first as before.
The exact f64 tree is untouched except for ONE additive port-only constructor
(`Segmentation::from_parts`, the splice's only way to hand the container a list it did not
build itself -- no existing behaviour reads it, and the committed goldens are byte-green).
The full replay stays reachable two ways: the `test-support` `set_full_replay` hook and the
flush-time sentinel guard (a caller whose `audio_duration` lands within a holdback of the cut
gets the whole-list rebuild rather than a wrong answer -- pinned, so it is not dead code).

## Phase 11 -- the transformer cell (the fifth `CellLayer` variant)

The user's named architecture list -- mamba, xLSTM/sLSTM, CfC, transformers -- closes. The
cell is a PRE-NORM block over a CAUSAL SLIDING WINDOW of bounded width `W` with ALiBi
relative positions and zero learned position parameters, landed end to end through the
phase-9 seam (exact f64 cell, both gradient tiers, both f32 fast twins, a fifth streaming
row, day-one retention gating, a 10-gate matrix, a 10-item mutation battery -- 8 of 10
breaking their spec-named leg). THE
DESIGN TRAP -- whole-sequence f64 attention is gigabytes/layer at full-corpus lengths -- is
answered by DEFINITION: the cell is windowed/bounded-KV from day one, so every backward
cache is `T x O(W + d_ff)`, never `T^2`, and the streamed KV ring is bit-identical to the
offline forward by construction. Full spec:
`docs/superpowers/specs/2026-08-12-phase-11-transformer-design.md`.

Four things distinguish this phase's record from its four predecessors', named up front so
each section below can be read as evidence for a claim already stated rather than a fact
discovered in place:

1. **AN EARLY DEDUPE WENT FIRST** (Task 1, approach A from Phase 10's own follow-on list):
   the `fast/cells.rs` / `fast/bicell.rs` scaffolding the four prior cells had begun to
   duplicate was consolidated into shared helpers BEFORE a fifth cell could multiply it
   further -- pure code motion, arbitrated by the existing phase-7/8/9/10 suites passing
   UNEDITED at their EXISTING pins.
2. **AN INIT-QUALITY ANOMALY, not a training failure.** `transformer/forward`'s from-scratch
   Xavier init does not collapse toward all-non-speech the way every gated-recurrent cell's
   does, which broke the ORIGINAL 10-gate hard-leg criterion at two collars even though the
   TRAINED model reaches the identical operating point most other rows also reach. Resolved
   by a user-ratified, two-sided criterion amendment -- see Task 9 below.
3. **THE FIRST STREAMING-GATE FIXTURE TO EXERCISE THE `pending_begin`-BLOCKED REGIME.** Every
   prior cell's streaming gate cleared the tight latency bound on every emission; the
   transformer row's does not, on two of them, and the miss is legitimate (the hysteresis
   correctly waiting out a cluster of failed rising attempts) rather than a bug -- see Task 7
   below, and the `EXPECT_TIGHT_COHORT` membership table it produced.
4. **TWO DOCTRINE SHARPENINGS from the battery** (Task 10, RESULTS-only, not re-narrated
   here -- see that section): FD is blind to forward STRUCTURE four separate times over on
   this cell alone (not just once, the way earlier phases found it), and the T4-flagged
   `[q|k|v]` matrix-half coverage question is CONFIRMED and narrowed to a single owning test,
   not left an open question.

### The exact cell + FD tier (Task 2)

`nn/cells/transformer.rs`, `TransformerLayer`. Forward per timestep `t` (`H = out`,
`A = heads`, `d = H/A`): a width adapter `x' = W_a x + b_a` -> RMSNorm_1 (the in-tree Mamba
transcription, reused not re-derived) -> ONE combined `3H x H` projection
`[q|k|v] = W_qkv u + b_qkv` (the sLSTM/CfC combined-block precedent) -> per-head logits
`(q_t . k_j)/sqrt(d) + m_h(j-t)` over the causal window `j in [max(0,t-W+1), t]`,
`m_h = 2^(-8h/A)` the ALiBi geometric slopes (constants, zero gradient) -> softmax ALWAYS
max-subtracting (a DECLARED divergence from the house F8 conditional guard -- no legacy
bytes to match here, and the f32 twin needs the headroom) -> the weighted-v fold -> residual
1 -> RMSNorm_2 -> an FFN (`silu`, reused from Mamba) -> residual 2. Flat order
`[W_a|b_a|g_1|W_qkv|b_qkv|W_o|b_o|g_2|W_1|b_1|W_2|b_2]` walked once by `for_each_slot`.

TWO dead-block classes, both derived and pinned: `b_k` (the k-third of `b_qkv`) is dead at
EVERY `T` (a uniform per-row logit shift is annihilated by softmax shift-invariance), and the
pin is TWO-CLASS -- bit-exact `== 0.0` at `T=1` (a one-element window's softmax is a literal
`1`) and a MEASURED `~1e-17` f64-cancellation floor at `T>=2` (the sLSTM `b_i` convention,
NOT the Mamba/CfC exact-zero one). A T2-review spec amendment corrected the brief's original
premise: the brief had written the `T>=2` pin as unconditional `== 0.0` by analogy to the
Mamba/CfC stored-exact-zero shape, but `b_k` at `T>=2` cancels in R without being bit-exact
in f64 -- the mechanical test the amendment records: does the adjoint terminate in a
multiplication by a stored exact zero, or in a sum that cancels? The q/k rows of `W_qkv` plus
`b_q` are ADDITIONALLY dead at `T=1` only (the Mamba-`A_log`-at-`T=1` analogue), live from
`T>=2`.

Both tiers reuse the phase-9 harnesses UNCHANGED (cell-agnostic, the seam's claim cashed a
third/fourth time).

**FD tier** (`tests/phase9_cell_grad.rs::transformer_backward_matches_central_difference`):
`eps = 1e-5` (MEASURED, an 8-point sweep bottoming there -- `max_rel_major` and the absolute
error both bottom at `1e-5`, roundoff dominating to its left and truncation's `eps^2`
signature dominating to its right), over 5 shapes (`t in {1,2,3,7,23}` at a deliberately
TINY `W=4`/`heads=2`/`d_ff=8` fixture so `t<=W` and `t>W` both appear FD-tractable) x 3 seeds:

| shape `(t, in, out)` | `max_rel` (worst of 3 seeds) | `rel_pin` | `max_rel_major` | `major_pin` | resolvable |
|---|---|---|---|---|---|
| `(1, 3, 2)` | 2.355e-8 | 2.4e-7 | 2.355e-8 | 2.4e-7 | 66 of 78 |
| `(2, 3, 2)` | 3.826e-6 | 3.9e-5 | 3.403e-8 | 3.5e-7 | 76 of 78 |
| `(3, 3, 4)` | 1.294e-5 | 1.3e-4 | 8.608e-8 | 8.7e-7 | 176 of 180 |
| `(7, 5, 4)` | 7.333e-6 | 7.4e-5 | 1.450e-7 | 1.5e-6 | 184 of 188 |
| `(23, 7, 6)` | 5.023e-6 | 5.1e-5 | **2.858e-7** | 2.9e-6 | 332 of 338 |

The DISCRIMINATING bound is `max_rel_major`: MEASURED worst **2.858e-7** at `t=23`, ~350x
under the `1e-4` STOP, pinned measured*10 per shape (worst pin `2.9e-6`, still ~34x under the
STOP). THE ONE PIN ABOVE `1e-4`, declared not buried: `t=3`'s `rel_pin` is `1.3e-4`, set by
one seed-1 weight whose analytic derivative is `2e-7` of the pack maximum -- the
central-difference floor over a near-zero denominator, not a wrong adjoint (the whole
`max_rel` column falls monotonically across five decades of eps, the roundoff signature; a
wrong adjoint term would be eps-INVARIANT). THE RESOLVABLE COUNTS CARRY THE S1.4 STRUCTURAL
CLAIMS with no slack: predicted (`transformer_total - transformer_dead`) == measured on
EVERY shape and seed (66/76/176/184/332), so the two-sided count assert degenerates to an
exact equality, matching sLSTM and CfC (unlike Mamba).

**Seam tier** (`tests/pyo3/test_phase9_seam.py`): corpus-level `grad_check` through
`speech_rs.Engine` on THREE committed synthetic fixtures (`transformer_bidirectional`,
`transformer_forward`, `twin_mode7_lid_transformer`, geometry `Transformer_Window 4` /
`Heads 2` / `D_Ff 6` -- reviewer-verified the effective cell depth is EXACTLY `T=50` by a
`W`-sweep, so `W=4` truncates 46 of 50 rows plus the start edge; `heads=2` is forced, the
only divisor of both 4 and 6):

| fixture | eps | measured worst scaled error | pin |
|---|---|---|---|
| `transformer_bidirectional` (grad_check) | 1e-5 | 3.565e-9 | 3.6e-8 |
| `transformer_forward` (grad_check) | 1e-5 | 3.193e-7 | 3.2e-6 |
| `transformer_bidirectional` (block probe) | 1e-5 | 2.716e-8 | 2.8e-7 |
| `transformer_forward` (block probe) | 1e-5 | 5.500e-8 | 5.5e-7 |
| `twin_mode7_lid_transformer` (grad_check) | 1e-4 | 5.157e-10 | 5.2e-9 |

All five measured epsilons are their OWN 5-point sweeps, not inherited defaults (the SAD
rows' minima both fall AT or beside `sad_epsilon = 1e-5`, so no fixture-specific key was
needed there; the Twin's own sweep bottoms at `1e-5` too, coincidentally matching
`sad_epsilon` -- it still gets its own named manifest key on the CfC precedent). Every pin
is measured*10 and under the `1e-4` STOP.
`test_transformer_key_bias_block_is_output_inert` adds the structural leg the block probe
alone cannot see (a `[q|k|v]` column-order swap inside the ONE named `b_qkv` block is
invisible to a probe that reads the same, possibly-swapped, order on both the analytic and
FD sides): `b_k` is bit-exact output-INVARIANT across four overwrite patterns (three
constants plus a non-uniform ramp) on BOTH stacks when bidirectional, `b_q` is LIVE at every
non-degenerate value tried. NARROWER THAN IT SOUNDS, recorded rather than hidden: this pins
the BIAS half of the `[q|k|v]` order only -- the MATRIX half (`W_qkv`) has no comparable
inertness handle (every row of `W_k` is live, since it feeds `k_j`, which varies with `j`),
so a matrix-only q<->k row-family swap with the bias order intact would pass this leg. That
gap was left KNOWINGLY OPEN for the Task-10 battery (see below) -- CONFIRMED there and
narrowed to one sentence, not left dangling.

### Python init + the dual-lineage sizing (Task 3)

`init_weights.py::init_transformer_flat` + `transformer_geometry`, on the CfC pattern: the
S1.2 flat order emitted DIRECTLY (no structured/nnet domain to build through), Xavier/He per
block with fans from the block shapes, `b_k` seeded 0 (S1.4), RMSNorm gains seeded 1.0, other
biases 0, NO magic constants (every block is plain dense/attention, no S4D-real analogue to
copy). The MANDATORY whole-pack block-by-block reconstruction pin came with TWO shear
companions (a q/k-family permutation, a deeper-FFN variant), all self-mutation-probed.

**THE SIZING** (spec S3, the CfC dual-lineage procedure): `window` and `heads` are
PARAMETER-FREE (ALiBi's slopes are constants; `heads` only reshapes `W_qkv`), so `d_ff` is
the cell's ONE sized knob. Per-layer count at hidden `H = 24` (both lineages):
`N(in) = 24*in + 2496 + 49*d_ff`. Both packs share the output MLP and normalize tail, so only
the recurrent stacks move:

| `d_ff` | v2 pack | v2 deviation | v1 pack | v1 deviation | verdict |
|---|---|---|---|---|---|
| 36 | 20927 | -14.34% | 23255 | -30.93% | v2 in / v1 OUT (v2 band floor) |
| 54 | 24455 | +0.10% | 26783 | -20.46% | v2 in / v1 OUT (v2 argmin) |
| 63 | 26219 | +7.32% | 28547 | -15.22% | v2 in / v1 OUT (v1 misses by 0.22pt) |
| **64** | **26415** | **+8.12%** | **28743** | **-14.64%** | **BOTH -- THE DEFAULT** |
| 72 | 27983 | +14.54% | 30311 | -9.98% | BOTH (v2 band ceiling) |
| 89 | 31315 | +28.18% | 33643 | -0.08% | v2 OUT / v1 in (v1 argmin) |

Bands: v2 `d_ff in [36,72]`, v1 `[64,114]`, intersection `[64,72]` -- NON-EMPTY, so the
primary rule (an integer inside BOTH bands) decides with no tiebreak needed; `64` is
simultaneously the smallest integer in the intersection AND the v2-best point of the
feasible set. `TRANSFORMER_DEFAULT_D_FF = 64` CONFIRMED the spec's design-time estimate
exactly.

**THE ONE RECORDED CONFLICT**, unlike CfC (where `B=45` was the v2 argmin AND the smallest
v1-tolerable integer at once): the v2 argmin here is `54`, and it sits OUTSIDE v1's band at
-20.46%. AND IT IS AN ARTEFACT OF COUNTING DEAD WEIGHT -- subtract v1's structurally-dead
layer-0 `W_a` columns (`2*24*48 = 2304`, the SAME 2015 dead-input-column defect every cell
inherits) and each lineage's normalize tail, and both lineages reduce to ONE live closed
form `13849 + 196*d_ff` against ONE live LSTM target `24409`: at `d_ff=64` both are
IDENTICALLY **26393** live weights, `+8.13%` both ways. The phase-10 live-count identity
extends to the transformer intact; under a live-count band there is no conflict at all (one
band `[36,72]`, one argmin `54`). PACK LENGTH stays the repo's stated sizing convention, so
it is what the default is sized against -- the two conventions happen to agree at 64 anyway.

Forward-only runs move both packs the same way (one stack, `cell_overlay` resizes the MLP
input from `2H` to `H`), so the verdict does not flip: v2 fwd LSTM 12239 vs TRANS 13231
(+8.11%), v1 fwd LSTM 16871 vs TRANS 14407 (-14.60%).

**THE FINDING, stated in the direction the T3 review corrected it to** (the spec's own
design-time estimate had it backwards): windowed attention at width 24 is parameter-CHEAP
next to a peephole LSTM (a whole cell layer costs `2496 + 49*d_ff` plus the fan-in term,
against the LSTM's `4*H*(fin+H+4)`). At the transformer-literature default `4*H = 96` the v2
pack is 32687 (+33.79%, OUTSIDE v2's band) while v1 sits +3.99% INSIDE its own -- the
textbook default fails exactly one lineage. `d_ff = 64` therefore lands BELOW the literature
default but comfortably ABOVE the naive "small cell -> small FFN" instinct.

Cross-language pins: pack-length equality vs `speech_rs.Engine` round-trip (bit-identical on
a tiny synthetic config, the phase-9/10 pattern); the three defaults
(`TRANSFORMER_DEFAULT_WINDOW`/`_HEADS`/`_D_FF`) parse-pinned against the Rust source, the
CfC `test_the_rust_and_python_defaults_agree` mechanism extended from one constant to three.
`init_transformer_flat`'s `forget_bias_one` parameter was accepted-and-ignored at this task
(call-site uniformity with the LSTM path) and REMOVED at Task 9 once that uniformity
argument was found false (no sibling builder carries it at all) -- drawn weight VALUES were
bit-unchanged either way, since the parameter was already dead. Tests: 60+4 new; pytest
949/1 skipped; lint clean.

### Seam fixtures + LID wiring (Task 4)

See "Python init" above and the FD/seam tables there -- this task authored the three
committed fixtures (`tests/reference_data/phase9/seam/`, the existing naming scheme) those
tables measure against, and closed the LID mechanical-wiring proof: an lstm/slstm/cfc/mamba
`BLSTM_LID_Cell_Type` swap against a transformer-sized LID pack is REFUSED at construction on
pack length (cell-type LOAD-BEARING, not merely read-and-ignored). The extract script's
`--check` mode confirmed 27/27 byte-identical against every PRE-EXISTING fixture (zero
regenerated), and license hygiene held (synthetic listings, no corpus tokens). Suite: 83/83.
The one coverage note this task recorded -- the `[q|k|v]` matrix-half gap -- is the item
Task 10's battery measured directly; see that section, not re-narrated here.

### The causal f32 twin (Task 5)

`fast/cells.rs::FastTransformer`, the fifth cell's causal step kernel, landed post-dedupe
(Task 1). STATE is a bounded per-head KV ring (capacity `W`, oldest-evicted,
`TransformerState` -- the Mamba private-scratch precedent, construct via `state()` only);
`step` runs adapter -> RMSNorm -> qkv row -> attend over the ring (a ring shorter than `W` at
stream start IS the offline edge-truncation, which is what makes offline-vs-streamed
bit-identity hold by construction) -> residual -> FFN -> push `(k_t, v_t)`. Op-for-op
transcription verified line-by-line (buffer indices vs `for_each_slot` per block; softmax
element-identical including division-not-reciprocal; the v-fold folds NORMALIZED
probabilities matching the exact cell; ALiBi's `j - t == jj - (len-1)` is an exact integer
identity making the ring state POSITION-FREE). Exactly TWO declared hoists, both pinned. Four
mutations (two reviewer-own) all caught with margins to 6 decades. `classify_fast_shape`
needed NO interim bail for either arm (the match is generic over `CellType`, so totality
survived construction unassisted).

Parity (`tests/phase9_fast_parity.rs`): plain `max_abs 5.536e-7` / `max_rel` **2.351e-6**,
crossing (gain 1, offset -5) `max_abs 1.602e-6` / `max_rel` **7.813e-6**, `max_dt` EXACTLY
0.0 on both with boundary count/types identical, against the UNCHANGED `1.0e-4` causal pin
(12.8x headroom on the crossing leg, second-worst of the ten cell x leg runs behind the LSTM
crossing row). THE CROSSING SEARCH, not the pin, is what widened to reach that headroom: the
committed fixture SATURATES on its 2 s tier-2 excerpt, so the row first landed at
`max_rel 1.312e-5` on the existing gain-ladder grid (which would have breached measured*10),
and the fix was a NEW `wide_offsets` fallback stage (integer offsets to `+-16`, reached ONLY
after the whole fine-grained grid is exhausted, chain order fine->wide so no sibling rung can
be preempted, verified byte-unchanged three ways). RULING A: this wide-offsets stage is a
LEGITIMATE fixture improvement, not a masked STOP -- the search is drift-blind (the rung is
selected on the EXACT path only, `max_rel` computed after), and the reviewer independently
reproduced the pre-fix `1.3120e-5` at `gain 2, offset -8` with the stage removed, confirming
the STOP was real and the widened search (not a widened pin) is what closed it. RULING B: a
BIDIRECTIONAL transformer BUILDS-BUT-UNPINNED at this task, one task wide and unreachable
from any committed artifact -- no bail needed per spec S5.3, closed by Task 6. RULING C: the
(92 -> 4) arm-geometry leg's own relative pin (`1.5e-4`) matches the phase-10 LSTM
arm-geometry precedent's form -- its absolute pin (`3.6e-6`) is the discriminating one, since
this leg's outputs sit near enough to zero that `max_rel`'s `1e-2` scale floor sits close to
the worst element's own magnitude (a NEAR-ZERO-ELEMENT-LIMITED leg, corrected wording from a
misleading "FLOOR-LIMITED" one-worder this task's review flagged and T11 fixed at the site);
the R4 STOP stays gradient-scoped, unaffected by this presentation choice. Suite:
`1269/0/2 (+10)`; zero `unimplemented!` left anywhere in `fast/`.

### The bidirectional arm (Task 6)

`fast/bicell.rs`, the `Transformer` enum arm -- `FastBiCell` is structurally `FastCausalNet`
with a SECOND stack (forward + reversed, driven by `cell_stack_forward(.., reverse = true)`,
the same causal kernel with ONE flag) -> hcat -> the shared `DenseRowChain`. No
transformer-specific bicell code beyond the enum arm and the element-count row (Task 1's
dedupe having already centralized the walk). `driver.rs` and `FastBlstm`/`overlap_window_step`
stayed UNTOUCHED -- proven byte-identical below `#[cfg(test)]` / doc-only, since Task 5's
threading had already totalized the build path.

Parity (`tests/phase10_bicell_parity.rs`, 4 new legs -- 3 cells x 2 regimes x base/crossing
grows to 4 with the fourth cell):

| leg | max_abs | max_rel |
|---|---|---|
| transformer plain | 2.86e-6 | 6.59e-6 |
| transformer overlap | 2.98e-6 | 6.84e-6 |
| transformer crossing plain (gain 1, +11) | 2.42e-6 | 1.159e-5 |
| transformer crossing overlap (gain 1, +10) | 1.57e-6 | **1.242e-5** |

`max_dt` EXACTLY 0.0 on all four, boundary count + types identical (three interior
boundaries on the plain crossing, two on overlap, both non-vacuous). Every transformer
number sits BELOW mamba's still-worst `1.81e-5`, so BOTH pins (`2.0e-4` relative / `1.0e-4`
absolute) stay UNCHANGED (transformer's worst measured*10 is `1.242e-4`, 1.6x of headroom
inside `2.0e-4` on top of what mamba already used). The crossing sweep needed a THIRD lever
transformer's two rows alone required -- not a bigger gain, a bigger OFFSET: the committed
`transformer/plain` fixture spans `[0.0000, 0.9008]` (logit span 40.21) and sits at
`interior=0` AS COMMITTED, unlike mamba's similarly wide span which happens to cross at
offset 0 -- a near-saturated curve's crossing point does not move under a small shift (the
OPPOSITE problem from `slstm/plain`'s too-flat curve), so the `+-4.0` grid that suffices for
six of the eight prior rows finds nothing, and `+11`/`+10` (still gain 1) do. THE SECOND
WIDE-OFFSET FALLBACK (bicell tier) was APPROVED on all three of Task 5's criteria: drift-blind
selection (chosen on `run_path(cell, None, ..)`, the exact path, before any fast comparison
runs), `chosen.is_none()` guarded AFTER the primary loop (sibling preemption structurally
impossible), and the six sibling rungs verified BYTE-IDENTICAL to the pre-Task-6 commit.
Reverse-stack liveness was EMPIRICALLY proven, not assumed: 488 of 518 reverse weights are
nonzero, and a reverse-only perturbation moves both the exact and fast paths by `4.34e-1`.
Suite: `1269/0/2` unchanged (no new `#[test]` functions -- the existing per-cell loops grew a
fourth array entry).

### The fifth streaming row (Task 7)

`StreamCausal` drives `FastCell`/`FastCellState` generically, so the transformer row lands
with ZERO `src/rust/src/` edits -- the S5.2 zero-new-surface claim cashed a THIRD time.
MEASURED at the fixture's committed lever (`Lever::NEUTRAL`, no search needed): 4 interior
boundaries plain / 11 type-1, `max_dt` EXACTLY 0.0, `to_bits`-equal posteriors,
chunk-invariant at 20/100/1000/7 ms, zero prefix retractions, and the SAME `1.73400 s`
derived latency bound bit-checked across all five cells now (a causal windowed-attention
cell has no lookahead either, exactly like the other four).

**THE HEADLINE FINDING**: this fixture is the FIRST streaming-gate one to exercise the
`pending_begin`-blocked regime (the phase-9 Task-9 fix) end to end rather than synthetically.
BOTH of the transformer row's mid-stream Speech emissions MISS the tight
`bound + PUSH_CHUNK_S` = `1.73400 + 0.1` = **1.834 s** bound on their own measured lag. A raw
posterior-crossing trace shows dozens of failed-area rising attempts clustering around the
fixture's two Other gaps (both under the `1.4 s` holdback) before the hysteresis's
`pending_begin` clamp finally releases the `begin` candidate -- exactly the "a segment
inherits the OTHER class's commit-wait area term" mechanism the offline gate fixture already
documented (phase 8/9's `phase8_gate.rs`), now exercised by a STREAMING fixture for the
first time. MEASURED speech lag **18.89087 s**, comfortably inside the data-dependent
`max_speech_dur + pipeline_forward + allowance` ceiling **26.22078 s** the OTHER class
already used -- 7.3 s of headroom.

This forced `latency_bounds` to check EVERY mid-stream Speech emission TIGHT-FIRST on its
OWN lag (fix round 1, F2) rather than pre-routing by a blocking-witness proxy first: an
earlier version classified emissions by a `windows(3)` proxy (is the very next entry a short
Other run, `< holdback`?) BEFORE checking either bound, which measurably OVER-classifies --
the proxy's `< holdback` threshold sums every smoothing term, over-shooting the true
`add_padding`-shrunk gap by ~0.6 s, so on the committed slstm fixture ALL THREE of its
proxy-flagged segments (`blocked_n = 3`) actually clear the tight bound on their own lag
(`missed_n = 0`) -- pre-routing them would have silently stopped testing the tight claim on
real, passing data. Tight-first makes the over-classification harmless BY CONSTRUCTION: the
witness is consulted only for an emission that ALREADY failed tight on its own lag. MEASURED
after the fix: mamba/cfc/lstm clear tight on EVERY emission (margins byte-unchanged:
-0.00533/-0.01512/-0.07712 s), slstm ALSO clears tight on every emission despite its
3-strong proxy set (margin unchanged: -0.03252 s), and `transformer` is the ONLY row with a
non-empty missed set (both emissions, backed by the witness, checked against the 26.22078 s
ceiling instead) -- a COMMITTED two-sided membership table, `EXPECT_TIGHT_COHORT`, pins this
asymmetry so a future all-blocked cell fails LOUDLY rather than silently losing its tight
claim.

Reviewer rulings: NOT retuning the crossing lever to avoid the blocked regime was the RIGHT
call -- lever-shopping to dodge an inconvenient finding would be an R1 evasion via another
knob, and rider-9's earlier lever-widening bought non-vacuity, not a license to retune away
an inconvenient result now that one showed up. The commit-wait ceiling reused here is
`phase8_gate.rs`'s established form (the SAME derivation the offline OTHER-class latency
uses), and its formal gap-term shortfall (see the follow-ons section) is INHERITED from
phase 8/9, out of this phase's scope, and recorded not fixed.

### Retention gating day one (Task 8, spec S7)

The attention cache is the largest backward cache of any cell in the tree: `attn_weights`
(`T x A*wcap`) is the widest SINGLE cached field at the default geometry (`A=4`,
`Transformer_Window=64` -> 256 columns, next-widest is `qkv` at `3H=72`) and, more to the
point, the ONLY cached field whose width is a PRODUCT of two independent config knobs
(`A*W`) rather than a fixed function of the net's own architecture -- nothing bounds it as
those knobs grow, unlike every other field here. `CellLayer::set_retain_cache`'s transformer
arm therefore GATES FROM DAY ONE, applying the phase-10 T9 Mamba mechanism proactively
rather than waiting to discover the cost later: every other cache field is computed in full
regardless of `retain_cache` (there is no cheaper way to compute it without restructuring
the projection itself) and simply left OUT of `self.cache` when not retaining; `attn_weights`
gets Mamba's `h`-treatment, an even simpler version of it, collapsing from `T x A*wcap` to
`1 x A*wcap` on a LOCAL, per-`(t, hh)` invariant (the WRITE loop and the READ loop a few
lines later share the SAME `len` range every iteration, by construction, independent of what
`window_begin` computes -- so it would survive a future non-monotone `window_begin`
unmodified too). A backward after a non-retaining forward PANICS with a named message
through the existing choke point (`Network::drive_backward`'s `Layer` funnel, which never
inspects the concrete cell type, plus a new cell-level named assert) rather than folding an
empty or stale buffer.

MEASURED on a long-T bidirectional SAD net (2 stacked layers, `H=24`, default geometry, a
60 s stereo excerpt forced through the PLAIN whole-sequence driver via `BLSTM_window 0` so
the cache scales with the full sequence, not a bounded window): `BackPropagationActivated
true` (retaining) peaks at **93.422 MB** vs `false` (this flag off) at **61.359 MB**,
`speech bench --path=exact`, one run each -- **1.52x less peak RSS (-34.3%)**. REGIME-SPECIFIC,
stated rather than hidden: the windowed-overlap driver already bounds the cache by the
window width, so this whole-sequence PLAIN-driver measurement is the MAX-BENEFIT regime, not
a universal one -- a windowed configuration would show a smaller win. The `T~6000`/`~1500`
row-count figures behind the 60 s excerpt are CONFIG-DERIVED from the frame rate, not
separately instrumented. Exact-tree touches: `nn/cells/` only (the phase-10 S7 sanctioned
class, extended to a second cell).

## Phase 11 -- the `lre_sad_v2` gate matrix grows to ten (Task 9)

`--cell transformer` joins `{lstm, slstm, mamba, cfc}` on the `sad`/`sad-v2` SAD arms
(`cell_overlay`, `drivers/baseline.py`), needing NO third derived key beyond the two the
knob already writes under `--direction forward` (`BLSTM_OutputNeuronNb` resize,
`BLSTM_window 0`): a transformer layer is just another `Layer` impl behind `CellLayer`, so
`feed_forward_backward_overlap` (the windowed driver `frame_window 3.25` selects) already
runs it bidirectionally with zero cell-type branch in that path -- verified by reading the
driver, not assumed, and empirically the SAME mechanism sLSTM/Mamba/CfC already exercised in
Task 4's eight gates. `Transformer_Heads`'s default (4) divides both lineages' hidden width
(24) at both layers, so no head-count override is needed either. `init_transformer_flat`'s
dead `forget_bias_one` parameter (Task 3's deferred F3, an inconsistency against the
`init_cfc_flat` precedent, which never carried the flag at all) is REMOVED here rather than
accepted-and-ignored -- a caller passing it now gets a `TypeError`, not a silently discarded
flag; the drawn weight VALUES are bit-unchanged (the parameter was already dead).

Gates: `tests/pyo3/test_phase10_gates.py`, extended in place (not a phase-11 sibling) --
`_CONFIGS` grows 8 -> 10, `_layer_length`/`_input_column_offsets`/`_arch` grow a
`transformer` arm (`W_a (out x fin)` row-major, the layer's first block, UNCONDITIONALLY
present unlike Mamba's `P`; same `j*fin + k` offset formula, minus the `fin != out`
precondition). Measured 2026-08-12, Apple M4 Pro (arm64), macOS 26.5.2, N=1 lane, seed 0,
same phase-6/9/10 VERBATIM recipe (subset 10 / valid 8 / test 24, 3 epochs x 10 SMORMS3
steps, 20 s audio cap).

### Pack lengths, MEASURED (spec R4)

`len(init_weights(...))` at the arm's own overlaid config, agreeing digit-for-digit with
`tests/test_phase11_init.py`'s independent closed-form derivation at the sized `d_ff = 64`:
bidirectional `13871 + 196*d_ff` (v2) / `16199 + 196*d_ff` (v1), forward `6959 + 98*d_ff`
(v2) / `8135 + 98*d_ff` (v1) -- the `d_ff`-dependent blocks (`W_1`/`b_1`/`W_2`/`b_2`) are
purely per-stack, so the coefficient exactly halves with the stack count while the constant
term does not:

| cell | v2 bidirectional | v2 forward | vs v2 LSTM (bi) | v1 bidirectional | v1 forward |
|---|---|---|---|---|---|
| Transformer (`d_ff = 64`) | **26415** | **13231** | **+8.12%** | 28743 | 14407 |

Inside the S8.2-style +-15% band on both lineages, comfortably (+8.12%, next to Mamba's
+14.74% near-miss and CfC's +0.25%). Unlike Mamba's `P` or CfC's mixed-fan-in `W_bb0`, the
transformer's `W_a` width adapter carries no v1-style dead-weight artefact to discount --
nominal == live on BOTH lineages already, so no separate live-capacity table is needed for
this cell the way Task 4's table was for the other four.

### Preflight -- the log-law saturation hazard

Same discipline as Task 4 (`CostLaw log/log`, whose forward is CONSTANT past the `1e-24`
clamp). Both rows land well inside the interior, `dead` exactly 22 (the frozen normalize
tail, nothing else) on both, zero weak columns:

| config | pack | init NNCostSeg | %clamp | worst per-file | %clamp | grad L2 | grad Linf | zero-grad weights | dead cols | min col |
|---|---|---|---|---|---|---|---|---|---|---|
| Transformer / bidirectional | 26415 | 0.40245 | 0.728% | 0.51811 | 0.938% | 1.49304 | 0.21785 | 22 | **0** | 5.692e-3 |
| Transformer / forward | 13231 | 0.47684 | 0.863% | 0.74657 | 1.351% | 2.24457 | 0.47225 | 22 | **0** | 9.819e-3 |

Both sit under 1% of the clamp (worst 0.863%, vs the other four cells' 0.44-0.54% -- higher
but still ~6-7x inside the 5% pin), `min col` an order of magnitude above every other cell's
(5.7e-3 / 9.8e-3 vs the next-highest 3.856e-3), and `dead` is exactly the 22-element frozen
tail on both rows: the inverse guard (zero structurally-dead layer-0 input columns) passes
cleanly, and `test_init_is_trainable` confirms this cell is trainable at init.

### The tenth gate -- an INIT-QUALITY ANOMALY, resolved via a user-ratified collapse floor

`transformer/bidirectional` clears the hard leg cleanly and joins the established pattern.
`transformer/forward` does not clear the ORIGINAL margin-only criterion at two collars --
recorded in full below, not argued away -- but both rows now PASS under the ratified
criterion (section below):

<!-- ledger:table phase11_v2_cells -->
| cell / direction | files (train/valid/test) | trained DCF@0.5 | Pmiss / Pfa @0.5 | trained collar range | init DCF@0.5 | init collar range | gain@0.5 | wall |
|---|---|---|---|---|---|---|---|---|
| Transformer / bidirectional | 10 / 8 / 24 | 0.250000 | 0.000000 / 1.000000 | 0.250000 (all 5) | 0.747406 | [0.747045, 0.748523] | +0.497406 | 56 s |
| Transformer / forward | 10 / 8 / 24 | 0.250000 | 0.000000 / 1.000000 | 0.250000 (all 5) | 0.455959 | [0.435836, 0.476457] | +0.205959 | 20 s |
| full-corpus runs (transformer) | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

Hosts: Apple M4 Pro (arm64, 14 cores, Darwin 25.6.0).
<!-- ledger:end -->

`transformer/forward`'s original-criterion breakdown: at collars 0.0/0.25/0.5 the
trained-vs-init margin (0.226/0.216/0.206) clears the original pinned `>= 0.2`, but at
collars 1.0/2.0 it does not (0.194262 and 0.185836 -- both measured, run-twice
bit-identical, not noise). At EVERY collar, though, the trained DCF is exactly 0.250000,
reaching the KNOWN all-speech collapse point seven of the other nine rows' trained models
also reach exactly (`slstm-forward` at 0.252430 and `mamba-forward` at 0.248770, both at
collar 0.5, are the two genuinely-non-degenerate trained exceptions -- Phase 10's "The
eight gates" section above -- and both clear their own margin comfortably regardless):

| collar | trained dcf | init dcf | margin | margin verdict | reaches floor (<= 0.2501)? |
|---|---|---|---|---|---|
| 0.0 | 0.250000 | 0.476457 | 0.226457 | OK | yes |
| 0.25 | 0.250000 | 0.465929 | 0.215929 | OK | yes |
| 0.5 | 0.250000 | 0.455959 | 0.205959 | OK | yes |
| 1.0 | 0.250000 | 0.444262 | 0.194262 | **FAIL** | yes -- **passes via the floor** |
| 2.0 | 0.250000 | 0.435836 | 0.185836 | **FAIL** | yes -- **passes via the floor** |

**The headline finding is an INIT-QUALITY ANOMALY, not a training failure.** The TRAINED
model reaches DCF 0.250000 at every collar (Pmiss 0, Pfa 1) -- the exact all-speech
collapse point SEVEN of the other nine rows' trained models also reach. (The remaining two,
`slstm-forward` at 0.252430 and `mamba-forward` at 0.248770 (both at collar 0.5), are
genuinely non-degenerate on the trained side too -- not a contradiction, since both still
clear their own untrained-vs-trained margin comfortably, 0.494 and 0.489 respectively,
without needing the collapse floor at all.) Training is not in question for
`transformer/forward` specifically: its trained result is indistinguishable from the
typical row's. The INIT side is what diverges: every other row's
from-scratch Xavier init on a GATED RECURRENT cell collapses toward the all-non-speech
baseline (Pmiss typically > 0.9, most rows exactly 1.0), which is WHY the original `-0.2`
margin (sized against a ~0.50 measured gap on those rows) carried so much headroom
elsewhere. `transformer/forward`'s untrained init does NOT collapse that way (Pmiss 0.581,
confirmed by two independent processes to the printed precision). **The likely mechanism**
(offered as the best available explanation, not independently proven by a dedicated probe):
an untrained windowed-attention layer at near-zero Xavier logits computes attention scores
`q . k / sqrt(d) + ALiBi bias` that are themselves near-zero and dominated by the ALiBi
term, so the softmax over the window is close to UNIFORM regardless of content -- the
untrained cell behaves like a fixed windowed AVERAGE of its value projections, which still
passes real signal variance through to the output. A fresh gated-recurrent cell has no such
"fall back to averaging" degenerate mode at init; empirically its output collapses hard
toward one class instead. `test_init_is_trainable[transformer-forward]` independently rules
out a saturated/dead-gradient explanation (init cost 0.863% of the log-law clamp, grad L2
2.24457, zero dead/weak columns) -- the net is trainable and DOES train, non-pathologically,
the same way every other row does; the init baseline is simply less degenerate, for
cell-architectural reasons.

**Disposition: RATIFIED (2026-08-12), then STRENGTHENED at review (fix round 1, F1).** The
user ratified an amended hard-leg criterion, mirroring the phase-9 Twin convergence-gate
deferral precedent (a hard target that could not be met exactly as originally stated was
resolved by explicit sign-off, not a unilateral agent relaxation): a collar passes if the
ORIGINAL margin holds (`>= 0.2`, UNCHANGED) **OR** the trained model reaches
`COLLAPSE_FLOOR_DCF = 0.2501` at that collar (the `+0.0001` over the mathematical 0.25
absorbing the VRCTS `%f.4s` write quantum) -- semantics: "training moved the model a lot, or
it reached the best-known subset operating point." THE NINE ALREADY-PASSING ROWS' EVIDENCE
IS UNCHANGED bit-for-bit: `margin_ok` alone is True at every one of their collars, so
`margin_ok or floor_ok` is True regardless of what `floor_ok` says -- NOT because evaluation
is "skipped" (`floor_ok` is an eagerly-assigned local, computed unconditionally on every
iteration; only the boolean `or` inside the `assert` short-circuits, and even that costs
nothing observable since `floor_ok` is already a plain bool by the time it is read) --
confirmed by re-running `cfc/forward` (the cheapest control row) post-amendment: unchanged
margin-shaped PASS.

THE TWO INIT-DEGENERACY SANITY CHECKS (`ini.pmiss > 0.9`, `ini.dcf >= 0.6`) took a
DIFFERENT path, and the honest trail is the live record here rather than a superseded
mechanism: the FIRST committed form applied the IDENTICAL `or tr.dcf <= COLLAPSE_FLOOR_DCF`
disjunct to both (a necessary extension beyond the originally-ratified single assert, found
while implementing -- pytest evaluates `ini.pmiss > 0.9` before the per-collar loop even
runs, so leaving it untouched would still fail the row before the margin-or-floor logic is
ever reached). Review (fix round 1, F1) found that form UNFALSIFIABLE on nine of ten rows
(every row's TRAINED side reaches the floor regardless of what its INIT looked like), with
ZERO detection power against a future builder bug producing an accidentally non-degenerate
init, or `transformer/forward` itself regressing back to degenerate. It was STRENGTHENED to
a two-sided COMMITTED membership table instead --
`NON_DEGENERATE_INIT_ROWS = ("transformer-forward",)`, mirroring `EXPECT_TIGHT_COHORT`
(`src/rust/tests/phase9_stream_causal.rs:1388`): listed rows MUST fail the degenerate
characterization (`Pmiss > 0.9 AND DCF >= 0.6`), unlisted rows MUST pass it, and membership
moving is an ADJUDICATION, never a silent widen. THIS is the CURRENT, committed form, at
`tests/pyo3/test_phase10_gates.py::test_subset_gate_beats_own_init` (the margin-or-floor
loop above is the ONLY site that still carries the original OR-disjunct design).

No training parameter was tuned to chase a pass at any point, across either round; the only
change is the pass CRITERION itself.

Post-amendment re-run (`uv run pytest tests/pyo3/test_phase10_gates.py -v -k transformer`):
**6/6 selected sub-legs PASSED, exit 0, 96.43 s.** Control re-run (`-k cfc-forward`, the
cheapest non-transformer row): 4/4 PASSED, exit 0, 20.58 s, unchanged margin-shaped evidence.

Whole-file runtime (transformer pair only, same box): 0.31 s preflight (2 legs) + 71.9 s
gates (2, both now PASS -- the training/scoring cost is unchanged, only the criterion moved)
+ 24.6 s determinism (2) = ~97 s, on top of Task 4's ~5.1 min for the other eight rows.

### Firing the transformer arm (post-phase, user-fired)

`experiments/04_v2_transformer.sh` fires both rows; each record promotes into the
`full-corpus runs` row above.

The full-corpus run is where the init-quality anomaly's practical consequence -- if any --
would actually surface: at subset scale it only affected which DISJUNCT a gate satisfied,
never the trained outcome itself, so whether the near-uniform-attention-at-init mechanism
helps, hurts, or is neutral to full-corpus convergence remains for the launcher to show.

## Phase 11 -- the mutation battery (Task 10): the catcher map

Spec S9's `>= 8` items, run as apply -> run ONLY the named catcher's file (FOREGROUND,
release, R6-capped) -> revert -> re-run green -> verify the tree byte-clean. **10 items, 15
applied mutations**: six items carry one mutation each (4, 5, 6, 7, 8, 9), items 1, 3 and 10
carry two, and item 2 carries three -- `6 + 2 + 3 + 2 + 2 = 15`. Item 7's `+1e3` non-vacuity
control is OUTSIDE that count (it exists to prove the mutated site live, not to be caught).

**8 of 10 items break the leg the SPEC NAMED; items 3 and 7 are the two honest gaps.** That
is the NON-GENEROUS reading and it is the right one: spec S9 names the f32 twin explicitly
for item 3, so item 3's spec-named mutation (3a) is NOT caught even though its exact-cell
variant (3b) is -- which makes that gap **f32-tier-specific**, not a hole in the convention
itself. Item 7's closer is the Task-1 dedupe rather than any test. By MUTATION rather than
by item: **12 named / 1 by-other / 2 not caught** = 15. Nothing under `src/` was committed;
the per-item evidence (diffs, commands, verbatim failure lines) is in the task ledger.

| # | mutation | named catcher | verdict |
|---|---|---|---|
| 1a | ALiBi slope sign flip, `FastTransformer::step` | `transformer_window_truncation_identity` + the (92->4) arm-geometry leg | CAUGHT (92->4, **2741x** its pin); truncation identity structurally blind |
| 1b | ALiBi slope sign flip, exact `TransformerLayer` | `multi_window_forward_matches_the_hand_transcription` | CAUGHT; **FD tier blind** (measured) |
| 2a | window off-by-one at the edge (`window_begin`) | same value pin | CAUGHT (widening variant = index panic; narrowing variant = 3 value legs); FD blind |
| 2b | KV-ring -> window rotation (`oldest + 1`) | the two fast legs | CAUGHT (92->4, **6329x** rel / **60290x** abs); truncation identity blind to a pure ROTATION |
| 3a | f32 softmax max-subtract removed | the f32 `exp` guard / parity | **NOT CAUGHT** -- 47/47 lib + 4/4 parity green |
| 3b | same removal, exact cell | `the_max_subtraction_survives_huge_logits` | CAUGHT (`the softmax overflowed`) |
| 4 | KV-ring eviction-order swap (evict the NEWEST) | `transformer_window_truncation_identity` | CAUGHT + 5 others |
| 5 | `b_k` made live (drop `- sdot`) | `the_key_bias_is_gradient_dead_at_every_length` | CAUGHT + the `T=1` pin + the FD resolvable-count CONTRACT assert |
| 6 | `init_transformer_flat` `W_1`/`W_2` emitted crosswise | `test_pack_is_reproducible_block_by_block` | CAUGHT on 10 of 10 parametrizations + both shear companions |
| 7 | re-duplicated `build_dense_tail` copy, ONE-ULP flip | the bicell parity legs at existing pins | **NOT CAUGHT** -- every printed statistic bit-identical |
| 8 | mis-scoped Jacobian `p*dp - sdot` (`T=1` zero SURVIVES) | the `T=2` contrast / FD counts | CAUGHT there; the `T=1` structural pin stays GREEN |
| 9 | retention arm folded into the no-op arm | T8's two enum pins | CAUGHT -- exactly those two, 7 siblings green |
| 10a | `W_qkv` q<->k family swap, MATRIX **and** BIAS | (the T4 open question) | CAUGHT at THREE tiers incl. the seam's `b_k` inertness leg, both stacks |
| 10b | q<->k shear, MATRIX HALF ONLY (`b_qkv` untouched) | (same) | CAUGHT at the exact tier (`flat_layout_positions_are_the_spec_order` + 2 value pins); **seam structurally blind** |

### The two honest gaps

**(1) The f32 max-subtract removal is owned by nothing** (item 3a). Softmax is exactly
shift-invariant in R, so removing the subtraction is PRECISION-ONLY at every logit magnitude
the committed fixtures reach and becomes a CORRECTNESS defect only past f32's `exp` limit
(~88.72), which none of them approach. It sits inside the ONE `step` kernel the offline and
streaming paths share, so every self-consistency leg compares two runs of the mutated kernel;
and the measured drift is **1.39x** on the `I = 5` leg (1.9019e-7 -> 2.6462e-7 against a
5.0e-6 pin) and **exactly 1.000x, bit-identical**, on the 92->4 leg, so no pin at this repo's
measured*10 convention could catch it either. This is phase-10 item 6's result reproduced on
a different cell and a different kernel. THE CHEAP CLOSER, named: an f32 twin of the exact
tier's `the_max_subtraction_survives_huge_logits` -- `fast/cells.rs`'s module doc already
argues in prose that the max-subtract is what keeps the twin precision-only; the argument is
written down, the pin is not.

**(2) A one-ULP divergence between two duplicated copies is owned by nothing** (item 7). A
re-added `build_dense_tail` copy inside `fast/bicell.rs` with `w[0]` flipped by one ULP
leaves the bicell parity legs 7/7 green with EVERY printed statistic bit-identical at 17
digits -- because the pinned quantity is a max over all elements and the perturbation is
three decades below the f32-vs-f64 gap that attains it. A `+1e3` control at the same site
flips a SEGMENT TYPE (`Other` -> `Speech`) and breaks 4 of 7, so the site is live and the
result is real. This does not weaken the Task-1 dedupe -- it IS the dedupe's justification:
only structure prevents that drift; no test tier watches for it.

### Two doctrine facts the battery measured rather than assumed

**FD is blind to forward structure, four times over.** The ALiBi sign flip (1b), the window
off-by-one (2a-ii) and the full q/k block swap (10a) all leave
`transformer_backward_matches_central_difference` GREEN, and 10b only brushes a
roundoff-set `rel_pin` (`9.295e-5` vs `7.4e-5`) while `max_rel_major` never moves and every
resolvable count stays exactly right. Central differences compare the analytic adjoint to
the numeric derivative OF THE SAME (mutated) forward, so a self-consistent forward+backward
pair is invisible by construction. **FD owns the ADJOINT; the value-pin instrument
(`multi_window_forward_matches_the_hand_transcription`, `single_step_matches_the_hand_computed_value`,
`flat_layout_positions_are_the_spec_order`) owns the FORWARD.** Conversely, the FD tier's
RESOLVABLE-COUNT assert is the sharpest catcher in the battery for a wrong Jacobian: items 5
and 8 both trip it as a CONTRACT failure ("more than the structural maximum"), no tolerance
involved.

**The T4 `[q|k|v]` matrix-half gap: CONFIRMED, and its scope narrowed to one sentence.** The
matrix-half q<->k permutation (10b) is owned by the EXACT tier's layout and forward value
pins -- three of them, all discriminating -- and has NO STRUCTURAL handle at the seam: the
`b_k` inertness leg passes on both stacks, because `b_k`'s deadness is a property of which
COLUMNS feed the key slot, not of which matrix rows were loaded there, and the shear leaves
that untouched. The seam's only reaction is a 1.6x breach of one `grad_check` pin
(`5.8205e-8` vs `3.6e-8`, measured `3.565e-9`) on one of three fixtures -- incidental, and
not something to rely on. The practical consequence: `flat_layout_positions_are_the_spec_order`
is the SOLE structural owner of the `[q | k | v]` row order, and must not be deleted as
"redundant with the forward pins". With the FULL family swapped (10a, matrix AND bias) the
seam DOES catch it, on both stacks, with a discriminating message -- so the bias half stays
closed exactly as Task 4 reported.

### One prior record that did not reproduce

The Task-5 review's parenthetical that a "ring -> window rotation does not move" the
`I = 5` -> 4 leg was NOT reproducible: the `oldest + 1` rotation moves that leg by five
decades (max_rel `9.353e-1` against a `5.0e-6` pin). The naming requirement it motivated is
unaffected and stands on the SIGN-FLIP axis, where the (92->4) leg is a measured **276x**
better catcher than the (5->4) one (2741x vs 9.93x of their respective pins). The leg that
is genuinely blind to a pure rotation is `transformer_window_truncation_identity`, and
provably so: both sides of its comparison rotate identically, which is why it owns WHICH
FRAMES are in the window (item 4, caught with a ~1e1 row delta) and never how they are
weighted.

## Phase 11 -- the NAMED follow-ons (nothing lost, nothing promised)

Collected at the phase closeout so each one is a tracked sentence rather than a report
paragraph nobody reads again. None is a defect; each is a place where this phase deliberately
stopped, with the reason and the shape of the work. ONE ITEM DELIBERATELY NOT LISTED HERE:
the T4-flagged `[q|k|v]` matrix-half coverage question (spec S1.4's inertness leg pins the
bias half only) is NOT an open follow-on -- Task 10's battery MEASURED it directly (item 10a
vs 10b) and CONFIRMED the exact tier's layout + forward value pins already own it, narrowing
it to one sentence rather than leaving it a question; see that section, not repeated here.

- **A `CellGeom` bundle for the side-by-side geometry params** (named at Task 5). `mamba`,
  `cfc` and now `transformer` each hand `build_cell` (`fast/cells.rs`/`fast/bicell.rs`) their
  own small geometry struct (`MambaParams`, `CfcParams`, `TransformerParams`) through
  parallel, near-identical plumbing. Three structs threaded the same way is a pattern, not
  yet a problem; a fourth new cell would be the natural trigger to fold them behind one
  `CellGeom`-shaped enum or trait object, the same "wait for the next shape" discipline
  Phase 10 applied to the `cells`/`bicell` scaffolding dedupe before landing it at Phase 11
  Task 1.
- **An f32 self-consistency pin for the max-subtract convention** (Task 10's named cheapest
  closer). The exact cell's `the_max_subtraction_survives_huge_logits` pins that S1.3's
  ALWAYS-subtract softmax convention survives a huge-logit input; `FastTransformer::step`
  carries the identical argument in prose (`fast/cells.rs`'s module doc) but no analogous f32
  pin exists, so the removal-of-the-subtraction mutation (item 3a) is caught by NOTHING --
  not because the convention is unimportant, but because nothing currently watches it on the
  f32 side. A same-shaped unit test (construct a `FastTransformer`, feed it a huge-logit
  input, assert the output stays finite) would close it at the cost of one test function; it
  was not added this phase because the battery's job is to MEASURE gaps, not close every one
  it finds.
- **Retention gating for the remaining three cells** (carried from Phase 10, extended not
  closed). Mamba's cache gates since Phase 10 Task 9, transformer's since this phase's Task
  8 -- two of five cells. `LstmLayer` (`gates`/`cells_in`/`cell_states`), `SlstmLayer`
  (`gates`/`cell_states`/`norm_states`/`m_states`) and `CfcLayer`
  (`z_cache`/`backbone_pre`/`backbone_post`/`heads`) still retain their `T x 6-7 O`
  backward-only buffers unconditionally; `CellLayer::set_retain_cache`'s no-op arms for them
  stay the SAFE direction (always retain), not an absence of work. Each of the two gated
  cells was gated because a measurement (Phase 9's bench for Mamba, this phase's own
  attention-cache-width argument for the transformer) named it the expensive one first --
  the remaining three have not yet had that measurement taken.
- **The `phase8_gate` commit-wait ceiling's formal gap-term shortfall** (inherited from
  phase 8/9, exercised for the first time by Task 7's streaming row, out of this phase's
  scope). The ceiling a blocked Speech emission is checked against
  (`max_speech_dur + pipeline_forward + allowance`, `phase8_gate.rs:974-982`) is an
  established form reused as-is, not re-derived for the causal/windowed-attention case; its
  own formal gap-term (blocked-Speech lag ~ gap + next_speech_dur + pipeline_forward vs the
  derived analogue max_speech_dur + holdback + pipeline_forward, ~1.4 s above the 1.0 s
  `OTHER_ALLOWANCE`) stays unexercised on every committed fixture including this phase's
  (7.3 s of headroom on the transformer row). A candidate for the streaming-endgame phase
  named in the phase-11 spec's Deferred section, not for a docs task to tighten in passing.

## Issue #23 -- the typed channel result: the mutation battery

The plan (`docs/superpowers/plans/2026-10-07-issue-23-typed-channel-result.md`, Task 7) named
four items; item 3 is run twice, once at the wire (`to_row`) and once at the fold's own
scaling, because the leg the plan named for it sits on the fold, not the wire. **Five applied
mutations**, each: apply -> run ONLY the named catcher's file(s) (FOREGROUND, R6-capped) ->
record the verbatim failure line -> revert -> re-run green -> tree clean. Nothing under
`src/` was committed; the module rebuilt through `maturin` for item 2 was rebuilt again from
the reverted tree before any later pyo3 run.

**5 of 5 caught; 3 of 5 by the leg the plan NAMED, 2 by other legs.** That is the
non-generous reading and it is the right one. By mutation: **3 named / 2 by-other / 0 not
caught**.

| # | mutation | named catcher | verdict |
|---|---|---|---|
| 1 | `to_row()` swaps `seg_cost` (col 4) and `lid.cost` (col 14) | the phase-4a `.mat` golden (`phase4a_train_golden::tier2_train_epoch_weights_golden`) | CAUGHT: `MultiConfigResults[0,7]: a=0x0 (0) b=0x403bceabbb1bb38a (27.807307905447978)`; the two twin `.mat` goldens in `phase4b_corpus_lid` fail at the same cell (4.333 vs 0; 36.386 vs 0). `phase4a_tier1_e2e` is blind by construction (TDC has no net, so col 4 is 0 either way) |
| 2 | `to_row()` drops the `+ 200` re-encode of the target column | `phase4b_confusion_golden` AND `tests/pyo3/test_channel_results.py` | `phase4b_confusion_golden` **NOT CAUGHT** (11/11 green): it feeds decoded `LidResult`s straight to `confusion_from_results`, so `to_row` is not on its path -- the plan named the wrong leg. CAUGHT by the cross-pin (`lid_target: seam array([0, 1]) != decoded wire array([-1, -1])`), by the module's three tests including the round-trip proptest, and by the twin `.mat` goldens (`MultiConfigResults[0,19]`: 65.287 vs 265.287) |
| 3a | `to_row()` writes `lid.correct` as `1.0` | `phase4a_save_update::aggregation_bit_exact` | **NOT CAUGHT** by the named catcher (5/5 green): the fold reads the struct, `to_row` is not on its path. CAUGHT by the module's three tests (`row_width_and_slots`, `from_row_inverts_to_row`, the proptest) and the twin `.mat` goldens (`MultiConfigResults[0,18]`: 1 vs 100) |
| 3b | `save_and_update` scales the flag by `1.0` instead of `100.0` | `phase4a_save_update::aggregation_bit_exact` | CAUGHT: `left: 99.5 right: 50.0`; `cost_mem_rows_written` (99.0 vs 0.0) and `update_called_with_neg_costlid` fail with it |
| 4 | `tests/_result_rows.py` decodes the target as `v - 150` | the `confusion_matrix` tests (`test_phase4c_scoring.py`) and `test_balance10_*` | `test_phase4c_scoring.py` **NOT CAUGHT** (its inputs are inline decoded `(target, scores)` pairs since this issue, and `confusionThresh.m` never had an Octave golden -- the plan misnamed it). CAUGHT by the three `test_balance10_*` Octave goldens and by `test_result_rows.py::test_lid_row_decodes_the_target_column` |

### The one fact the two by-others share

A mutation at the WIRE edge (`to_row`, the Python decoder) is owned only by the legs that
cross the wire: the `.mat` goldens, the round-trip proptest, the seam cross-pin. A leg that
consumes the typed value directly (`phase4b_confusion_golden`, `phase4a_save_update`, the
inline `confusion_matrix` unit tests) is blind to the layout BY CONSTRUCTION, which is the
module's design restated as a test fact: the readers no longer touch the layout, so no
reader can catch a layout bug. The battery therefore names the wire legs as the owners of
`to_row` / `from_encoded` / `channel_results_from_matrix`, and the fold tests as the owners
of the fold's arithmetic (3b), and nothing else should be read into a green reader test.

## Issue #22 -- the fold run: the mutation battery

The issue's acceptance named one mutation ("a mutation that removes the Epochs-0 forcing fails a
committed test"). **One applied mutation**: delete the `Neural_Networks_BackPropagation_Epochs 0`
line of `fold_run.FoldRun.__init__` (the base config's own epoch count then reaches the engine)
-> run the two named catchers, then the files the eight production sites feed (FOREGROUND) ->
record the verbatim failure line -> revert -> re-run green -> tree clean.

**Caught by both named catchers, and by four other legs.**

| # | mutation | named catcher | verdict |
|---|---|---|---|
| 1 | `FoldRun` no longer forces `Epochs 0` | `tests/test_fold_run.py::test_fold_run_forces_epochs_zero_on_every_fold` (the rendered map, no engine) and `tests/pyo3/test_fold_run.py::test_fold_run_is_one_fold_at_theta_f11` (the engine) | CAUGHT by both. At the map: `AssertionError: assert '3' == '0'`. On the engine: `a gradient fold measures the cost at theta (F11)` / `assert 0.1727108023717789 == 1.1179364603428454` -- the tier-2 fixture's `Epochs 3` routes `run()` through the engine-internal `train()` (3 folds + 2 iRPROP- steps) and the cost comes back from moved weights, the F11 misroute verbatim; the test's own foil (the same map run raw with `Epochs 3`) reports that moved cost, so the catcher cannot pass vacuously. Also caught, unasked: `tests/pyo3/test_fold_run.py::test_fold_run_fast_guard_on_the_engine` (`RuntimeError: Inference_Path fast is inference-only (training stays exact f64), but this Multi-mode run is training-shaped (Epochs 3, BackPropagationActivated SAD=false/LID=false)` -- the fast tree's training-shape bail fires on a forward-only fold once the epoch count leaks), `tests/test_phase4c_drivers.py::test_evaluate_exact_injects_resolved_packs` (`assert '6' == '0'`, the twin fixture's `Epochs 6` reaching the fake seam), and two exit-gate legs, `test_hyperparam_search_narrowed_distinct_configs_and_costs` (its `Epochs 0` text pin) and `test_from_scratch_sad_converges` (the SMORMS3 trajectory is no longer measured at theta). |

### The final tree, green

Run on the branch's final tree (2026-10-07, this box): cargo 1317 passed (`cargo test --release`,
the exact-tree goldens byte-green under the one-line lockfile bump), the pure-Python suite 907
passed + 1 skipped, `tests/pyo3` 210 passed in 19m36s with the corpus present -- the exit gates
(`test_exit_gate.py`, 9 legs), the modern-loop smoke, and every subset gate of phases 6, 9 and 10
with unchanged outcomes, the gate probes re-run on their final one-fold form.

### What the battery measured

- `tests/pyo3/test_seam_replay.py`'s four `forward_backward` legs are BLIND to this mutation by
  construction, and were before #22 too (they appended `Epochs 0` to the fixture themselves):
  determinism, finiteness, a nonzero gradient and weight movement all hold under `train()` as
  well. The "at theta" property has exactly the owners above.
- Measured in passing, then pinned rather than hidden (`test_fold_run_is_one_fold_at_theta_f11`):
  `run_solo` with backprop on ends with the bag's iRPROP- step (`save_and_update_epoch`), so a
  gradient fold's post-run pack is one engine step past theta while its cost and gradient are
  at theta. The grilling had assumed "weights before equals after"; the engine says otherwise.
  Consequence for the legacy-regime `train` driver, recorded not changed (it is byte-untouched
  by rule): the `sad_weights.bin` / `lid_weights.bin` it checkpoints are `_score_fold`'s
  post-run weights, i.e. the SMORMS3-trained pack plus one engine iRPROP- step, which the exit
  gate's run-twice bit-identity never distinguished from the trained pack itself.

## Issue #57 -- the fast Twin LID matrix: parity per (cell x direction) + the battery

The Twin's LID net (`fast::driver::FastLidNet`) now runs every `(cell, direction)` pair the
exact tree trains, through the same `classify_fast_shape` the SAD driver uses; the lifted
truncate / TwoSweeps sweep (`FastLidNet::truncate_forward`) is shared by the three shapes, and
the plain regime is admitted (TwoSweeps ignored there, as the exact tree ignores it). The
streaming LID session runs every shape too: it is utterance-granular, so ADR-0004's refusal is
the frame stream's alone (its dated amendment names the exception). What survives as a
refusal is a regime rule -- windowed causal, and overlap on every shape -- raised where the
window resolves (`check_lid_regime`), offline and streaming alike.

### Parity: exact f64 vs fast f32 (`src/rust/tests/fast_twin_lid_matrix.rs`, CI)

Over the nine phase-9 Mode-7 LID fixtures (a `12,6` LID net per cell, the SAD net frozen on the
legacy LSTM, six of them generated by this issue) and the committed phSeq trio
(`s1`/`s2`/`s3`). The `(lstm, bidirectional)` pair is the phase-7 real-net leg, untouched and
green. IDENTITY asserted on the per-utterance argmax (the confusion matrix, and a single-entry
exact run against the fast session's `scores`), the cumulative `is_lid_correct` and the
predicted language; the `classification_errors` delta MEASURED then pinned x10. Measured
2026-10-09 (M4 Pro).

THE LADDER NEEDED A SECOND LEVER, and the reason is a fixture property worth its line: the
LID nets are `12` wide over the `38`-wide phSeq one-hot, so every block whose letters all map
past column 12 (the first block of every committed file, all of `s2`) is ALL-ZERO after the
width-tolerance crop; a zero input through a zero-state cell leaves the xavier-zero output
bias, and the posterior is EXACTLY `0.5` on both trees at every gain -- a tie no scaling
breaks (`min(margin/delta) = 0` at gains 1/2/4/8). A `+0.5` bias offset on the last dense
layer (the phase-10 knob) gives those blocks a `rows * 0.5` margin against an f32 delta of
~1e-8, and every pair settles on `gain 1, offset +0.5`. The non-degenerate blocks (e.g. `s1`
block 1, `s3` block 1) carry real margins (0.09 .. 0.47 in `seg_lid` units) at the same rung.

| pair | fixture | rung | `max_abs(classification_errors)` | `min(margin / delta)` | pin |
|---|---|---|---|---|---|
| slstm / bidirectional | `twin_mode7_lid_slstm` | gain 1, +0.5 | 2.081e-6 | 5.645e6 | 2.1e-5 |
| mamba / bidirectional | `twin_mode7_lid_mamba` | gain 1, +0.5 | 2.081e-6 | 5.345e6 | 2.1e-5 |
| cfc / bidirectional | `twin_mode7_lid_cfc` | gain 1, +0.5 | 2.081e-6 | 3.952e6 | 2.1e-5 |
| transformer / bidirectional | `twin_mode7_lid_transformer` | gain 1, +0.5 | 2.081e-6 | 5.645e6 | 2.1e-5 |
| lstm / forward | `twin_mode7_lid_lstm_forward` | gain 1, +0.5 | 2.081e-6 | 5.645e6 | 2.1e-5 |
| slstm / forward | `twin_mode7_lid_slstm_forward` | gain 1, +0.5 | 2.081e-6 | 5.645e6 | 2.1e-5 |
| mamba / forward | `twin_mode7_lid_mamba_forward` | gain 1, +0.5 | 2.081e-6 | 5.645e6 | 2.1e-5 |
| cfc / forward | `twin_mode7_lid_cfc_forward` | gain 1, +0.5 | 2.081e-6 | 5.645e6 | 2.1e-5 |
| transformer / forward | `twin_mode7_lid_transformer_forward` | gain 1, +0.5 | 2.273e-6 | 5.645e6 | 2.3e-5 |

The identical `2.081e-6` on eight pairs is not a copy: the dominant term is the bias-only
blocks' f32 logistic at logit `0.5`, which every cell shares; transformer/forward's extra
`2e-7` is its own. Zero flips anywhere. The dispatch leg builds each pair from its own pack
(immediately and through the deferred `load_weights_file`, scored bit-identical on `s1`) into
its own arm and refuses the pack one element short or one element long on the in-memory seam
(the exact `set_weights` contract; the file seam keeps the legacy head-first tolerance with
its warning); on the real net (`phase7_parity_lid.rs`) the LSTM pack is refused on length
under all four cells: too short under `mamba` (14521 needed) and `transformer` (13113), too
long under `slstm` (11833) and `cfc` (12235).

The PLAIN regime (`BLSTM_LID_window 0`) on the bidirectional shapes, admitted by D2 but run
by no committed fixture, has its own legs: the four `BiCell` pairs at the same rung
(`bidirectional_plain_regime_parity_per_cell`: `2.081e-6` on all four, `min(margin/delta)`
4.23e6 .. 5.65e6, pin 2.1e-5) and the real `Blstm` net
(`phase7_parity_lid::lid_parity_phseq_plain_regime_exact_vs_fast`: score `max_abs` 5.958e-7,
pin 6.0e-6). Measured 2026-10-09 (M4 Pro), zero flips.

### Streaming (`src/rust/tests/phase8_stream_lid.rs`, CI) and the surfaces

Every one of the nine pairs: the running aggregate after each push bit-equal to the offline
fast Twin on the first `k` entries, `finish` bit-equal to the full run, on all three files. One
PyO3 `StreamingLidSession` row over `twin_mode7_lid_slstm_forward` (synthetic phSeq-shaped
one-hots, seeded): prefix-correct between an incremental and a fresh session. CLI tests
untouched.

### Real data (`tests/pyo3/test_lid_cells_gates.py`, corpus-gated)

The trained `slstm / forward` leg on both LID arms re-scores its held-out slice on the fast
tree (`Inference_Path fast`, the plain-regime causal twin) over the same listing: per-file
argmax identity (zero flips) and a LID error delta of EXACTLY `0.0` on both arms (run
2026-10-09, 3m36s for the pair).

### The mutation battery (the grilling's D13)

Apply -> run ONLY the named catcher (release) -> revert -> re-run green -> tree byte-clean.
Three items, three mutations, three catches, all by the leg the decision named.

| # | mutation | named catcher | verdict (verbatim failure) |
|---|---|---|---|
| 1 | swap the hcat order in `FastBiCell::feed_forward` (`[rev \| fwd]`), the BiCell LID arm | `fast_twin_lid_matrix::lid_parity_exact_vs_fast_per_pair`, the argmax gate | CAUGHT: `twin_mode7_lid_slstm/s3 u1: per-utterance argmax FLIP (R1 STOP)`; `min(margin/delta)` collapsed from `5.6e6` to `1.2 .. 2.0` on every rung, so no rung was interior and the fallback asserted identity at the last one |
| 2 | drop the second sweep in the lifted TwoSweeps (`output = sweep1`) | the LID parity pins | CAUGHT twice: `phase7_parity_lid`: `phseq s1 chan 0: is_lid_correct FLIP (R1 STOP: exact=100 fast=0)` (and the cep score pin); `fast_twin_lid_matrix`: `twin_mode7_lid_slstm: classification_errors delta 2.308e-1 breached the pin 2.1e-5` |
| 3 | remove the windowed-causal refusal in `check_lid_regime` | `fast_twin_lid_matrix::windowed_causal_lid_is_refused_offline_and_streaming` | CAUGHT: `the regime must be refused` |

Item 1's first run was caught by the ladder's PRECONDITION (no interior rung) rather than by
the argmax assertion; the leg now asserts identity at the last probed rung before giving up,
so a broken kernel reports as the flip it is. The strengthening is recorded here because the
battery is what found the gap.

## Issue #68 -- the MiB divisor pin: the mutation check

Issue #54 relabelled the bench peak RSS as MiB everywhere a reader sees it; nothing pinned
the divisor behind the label. The only checks on the value were the sanity ranges in
`tests/phase7_bench.rs` (`1.0 < maxrss_mb < 4096.0`), which a decimal divisor (`/ 1e6` on
macOS, `/ 1e3` on Linux) passes while moving every new record off the 24 committed ones
(+4.9% on macOS, +2.4% on Linux, neither MiB nor MB). Option 1 of the issue: the conversion
is hoisted into the pure `bench.rs::ru_maxrss_to_mib(raw)` (same `cfg` split, `maxrss_mb()`
is now `getrusage` plus one call) and unit-tested inline: one platform unit per MiB maps to
exactly `1.0` (exact in f64 on both arms, `x / x`). The pin is arithmetic only; the unit the
OS reports in (bytes on macOS, KiB on Linux) stays a documented assumption.

Apply -> run ONLY the named catcher -> revert -> re-run green -> tree byte-clean. Run on
macOS, so the verdict is the `/ 1e6` arm's; the Linux arm (`left: 1.024`) is exercised by
the CI runner, not by this battery run.

| # | mutation | named catcher | verdict (verbatim failure) |
|---|---|---|---|
| 1 | decimal divisor in `ru_maxrss_to_mib` (`raw / 1e6` resp. `raw / 1e3`) | `bench::tests::ru_maxrss_to_mib_is_binary` | CAUGHT (macOS arm): `left: 1.048576`, `right: 1.0` |
