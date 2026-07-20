# Phase 6 baseline results

The living record of the from-scratch SAD + LID baselines on the LRE03/07 corpus. Phase 6
delivers the baseline PROTOCOL + machinery, subset-proven; the full-corpus headline numbers
land when the user fires the offline launchers (`speech baseline <arm> ...`), and this file
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
  (`run_metadata.json`) records seed, N lanes, subset spec, and the config hash.
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

| split | files (train/valid/test) | LID error % | Cavg | chance % | config | seed |
|---|---|---|---|---|---|---|
| subset gate (2026-07-18, this box) | 35 / 15 / 48 | **72.92** | **0.46** | 91.67 | `lre03_lid_features.toml`, 2 ep x 25 steps | 0 |
| subset gate -- untrained init baseline | (same test set) | 93.75 | 0.53 | 91.67 | (seed packs, no training) | 0 |
| full run | TBD | TBD | TBD | 91.67 | `lre03_lid_features.toml` | TBD |

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

| split | files (train/valid/test) | DCF@0 | DCF@0.25 | DCF@0.5 | DCF@1 | DCF@2 | Pmiss/Pfa @0.5 | config | seed |
|---|---|---|---|---|---|---|---|---|---|
| subset gate (2026-07-18, this box) | 10 / 8 / 24 | **0.2500** | **0.2500** | **0.2500** | **0.2500** | **0.2500** | 0.00 / 1.00 | `lre_sad.toml`, 3 ep x 10 steps, 20 s cap | 0 |
| subset gate -- untrained init baseline | (same test set) | 0.7500 | 0.7500 | 0.7500 | 0.7500 | 0.7500 | 1.00 / 0.00 | (seed pack, no training) | 0 |
| full run | TBD | TBD | TBD | TBD | TBD | TBD | TBD | `lre_sad.toml` | TBD |

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

| split | files (train/valid/test) | LID error % | Cavg | chance % | config | seed |
|---|---|---|---|---|---|---|
| subset gate (2026-07-18, this box) | 15 / 15 / 45 | **84.44** | **0.49** | 91.67 | `lre03_lid_phseq.toml`, 2 ep x 12 steps | 0 |
| subset gate -- untrained init baseline | (same test set) | 91.11 | 0.48 | 91.67 | (seed packs, no training) | 0 |
| full run | TBD | TBD | TBD | 91.67 | `lre03_lid_phseq.toml` | TBD |

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

## Firing a full run (post-phase, user-fired)

```
# LID features (Algo 6, Mode 7, cep)
speech baseline lid-features --corpus-root data/LRE03-LRE07 --out-dir runs/lid_features_full \
    --lanes 1 --seed 0 --epochs 40 --steps-per-epoch 25

# SAD (Algo 3 spectral, wav) -- --audio-max-duration lifts the subset gate's short cap; omit
# it to score full-length recordings (the corpus wavs are 576-1800 s CallFriend files, median ~600 s).
speech baseline sad --corpus-root data/LRE03-LRE07 --out-dir runs/sad_full \
    --lanes 1 --seed 0 --epochs 40 --steps-per-epoch 25 --audio-max-duration 120

# LID phonotactic (Algo 6, Mode 7, File_Type 1 phSeq) -- the 2015 flagship regime.
speech baseline lid-phseq --corpus-root data/LRE03-LRE07 --out-dir runs/lid_phseq_full \
    --lanes 1 --seed 0 --epochs 40 --steps-per-epoch 25
```

Omit `--subset` to train on the whole split (SAD: the full 70% train split of the 2066 wav/xml
pairs; LID: the whole localized corpus); `--lanes N` sets the fold width (record N here);
`--resume` continues from `<out-dir>/checkpoint`. The run writes `run_metadata.json`
(seed/lanes/subset/config-hash/audio cap), `checkpoint/` (best/last packs + `train_history.json`),
and the held-out scores -- LID `scores/` (`.scr`) -> `lid_error`/`cavg`; SAD `score_trained/`
(VRCTS hyp xml) -> pooled `dcf`. Paste the resulting numbers into the `full run` rows above.

Known limitation (machinery): `forget_bias_one` (the LSTM forget-gate 1.0 init, default on) is
threaded correctly through `ModernTrainParams` end to end but has NO CLI flag on `speech baseline`
-- only its default (`True`) is exercised; a `False` sweep would need the flag added.

---

## Phase 7 -- performance

The measure-then-pin exact-path baseline (Task 1): `speech bench [--repeat=N] [--path=exact]
<config>` runs a corpus config end to end (the `-i` image-mode `CorpusProcessor` path, a single
unscored forward pass -- `Neural_Networks_BackPropagation_Epochs 0` and, per config,
`*_BackPropagationActivated false`) and prints one `BENCH path=exact wall_s=<f> audio_s=<f>
rtf=<f> maxrss_mb=<f> files=<n>` line per run. `rtf = wall_s / audio_s`; `maxrss_mb` is the
process-lifetime peak RSS (`getrusage(RUSAGE_SELF)`, bytes on macOS / KB on Linux, normalized to
MB). Every number below is `--repeat=1` (a single fresh process per measurement, so `maxrss_mb`
is a clean per-run peak, not inflated by repeated in-process construction -- `--repeat>1` in one
process accumulates allocator high-water-mark across runs by design, see `bench.rs`'s doc). At
Task 1 time no fast path existed yet (`--path` accepted only `exact`); these numbers were the
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

| leg | audio_s | path | wall_s (mean [range]) | rtf | maxrss_mb | MB/audio-s | speedup (wall, fast vs exact) |
|---|---|---|---|---|---|---|---|
| SAD 60 s fixture (stereo) | 120.00 | exact | 0.2638 [0.2588-0.2693] | 0.002198 | 55.641 | 0.4637 | baseline |
| SAD 60 s fixture (stereo) | 120.00 | fast | 0.0573 [0.0569-0.0580] | 0.000477 | 67.693 | 0.5641 | **4.60x** |
| SAD corpus-gated (mono) | 75.00 | exact | 0.1679 [0.1626-0.1775] | 0.002238 | 53.656 | 0.7154 | baseline |
| SAD corpus-gated (mono) | 75.00 | fast | 0.0366 [0.0364-0.0370] | 0.000489 | 68.459 | 0.9128 | **4.58x** |
| LID phSeq corpus-gated (Twin M7) | 42.54 | exact | 0.0445 [0.0441-0.0450] | 0.001045 | 29.964 | 0.7044 | baseline |
| LID phSeq corpus-gated (Twin M7) | 42.54 | fast | 0.0126 [0.0125-0.0127] | 0.000296 | 19.786 | 0.4651 | **3.53x** |
| LID cep corpus-gated (Twin M7) | 32.65 | exact | 0.0339 [0.0338-0.0340] | 0.001037 | 20.255 | 0.6204 | baseline |
| LID cep corpus-gated (Twin M7) | 32.65 | fast | 0.0095 [0.0095-0.0095] | 0.000291 | 11.979 | 0.3669 | **3.57x** |

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
R5):** future local runs of this exact recipe are expected to land within:
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

| target | path | scope | time |
|---|---|---|---|
| `matmul_seq_92x96` | exact | 92x23 * 23x96, f64 ascending loop | 57.408 us |
| `faer_project_92x96` | fast | SAME shape, f32, real `fast::nn::faer_project` | 4.624 us |
| `gfft_1024` | exact | 1024-pt complex FFT, 2 real 1024-sample frames packed per call | 17.898 us/call (8.949 us/frame) |
| `realfft_1024` | fast | 1024-sample real FFT, 1 frame per call | 0.765 us/call (= 0.765 us/frame) |
| `mel_apply_513x20` | exact | 100x513 f64 periodogram already in hand -> log-mel | 24.135 us |
| `mel_apply_widened_f64` | fast | SAME shape, f32->f64 widen (fresh alloc) + SAME `apply_filter_bank` | 58.309 us |

Reading: `faer_project_92x96` is **12.41x faster** than `matmul_seq_92x96` at the identical shape --
the single biggest per-kernel win, and (with the LSTM recurrence itself unbenched here, see
`fast/nn.rs`'s doc on why it stays a hand-written dot rather than a per-frame faer matvec) the main
driver of the SAD/LID wall-clock speedups above. `realfft_1024` is **23.4x faster per call**, or
**11.7x faster per FRAME** once the exact GFFT's two-real packing (one call amortizes TWO windowed
frames, not one -- see `fast/pipeline.rs`'s equivalence proof) is normalized out; either framing is
a large win.

**FINDING (reported honestly, not hidden -- exactly the brief's ask):**
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
"T6b" section for the full mechanism + the sibling-method audit.

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
