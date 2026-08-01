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
the direction differ, driven by the T4 `--cell-type` / `--direction` knobs on
`speech baseline sad`. Measured 2026-07-28, Apple M4 Pro (arm64), macOS 26.5.2, N=1 lane.

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

| cell / direction | files (train/valid/test) | trained DCF (all 5 collars) | Pmiss / Pfa @0.5 | init DCF | gain | wall |
|---|---|---|---|---|---|---|
| sLSTM / bidirectional | 10 / 8 / 24 | **0.250000** | 0.000 / 1.000 | 0.750000 | **+0.500000** | 42 s |
| sLSTM / forward (causal) | 10 / 8 / 24 | **0.250000** | 0.000 / 1.000 | 0.750000 | **+0.500000** | 17 s |
| Mamba / bidirectional | 10 / 8 / 24 | **0.250000** | 0.000 / 1.000 | 0.750000 | **+0.500000** | 67 s |
| Mamba / forward (causal) | 10 / 8 / 24 | **0.249625** | 0.000 / 0.998498 | 0.750000 | **+0.500375** | 20 s |
| *BLSTM / bidirectional (phase-6 Task 9, same recipe)* | *10 / 8 / 24* | *0.250000* | *0.000 / 1.000* | *0.750000* | *+0.500000* | *~80 s* |
| full-corpus runs (all cells) | TBD | TBD | TBD | TBD | TBD | TBD |

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

```
# Any {lstm,slstm,mamba} x {bidirectional,forward} combination, same launcher as phase 6.
speech baseline sad --corpus-root data/LRE03-LRE07 --out-dir runs/sad_slstm_full \
    --cell-type slstm --direction bidirectional \
    --lanes 1 --seed 0 --epochs 40 --steps-per-epoch 25 --audio-max-duration 120
```

`--direction forward` additionally forces `BLSTM_window 0` (the plain whole-sequence causal
regime -- a window boundary would reset the recurrent state) and resizes the output MLP's
input width; both are automatic (`drivers/baseline.py::cell_overlay`). Paste resulting
numbers into the `full-corpus runs` row above.

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
| lstm / forward (control) | 0, plain | exact | 0.098132 [0.097867-0.098532] | 0.001308 | 54.84 | n/a (fast typed-bails, S5.3) |
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

**Local-only regression bounds (NOT CI-asserted, this-box numbers, Apple M4 Pro named per spec
R5),** the phase-7 local-vs-CI split: future local runs of this recipe are expected at fast-vs-exact
wall speedup >= 3.5x on every row, causal-fast-vs-BLSTM-fast >= 1.3x, and fast maxrss within
1.3x of the ~68 MB plateau (SUPERSEDED -- see Phase 10's f32-mel section: the plateau is now
~42 MB). A reading meaningfully below these (not within the ranges above, which
already carry headroom) is a FINDING to investigate; there is no automated enforcement.

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

| cell / direction | trained DCF@0.5 | Pmiss / Pfa @0.5 | trained collar range | init DCF@0.5 | init collar range | gain@0.5 | wall |
|---|---|---|---|---|---|---|---|
| LSTM / bidirectional | **0.250000** | 0.000000 / 1.000000 | 0.250000 (all 5) | 0.750000 | 0.750000 (all 5) | **+0.500000** | 37 s |
| LSTM / forward | **0.250000** | 0.000000 / 1.000000 | 0.250000 (all 5) | 0.750000 | 0.750000 (all 5) | **+0.500000** | 15 s |
| sLSTM / bidirectional | **0.250000** | 0.000000 / 1.000000 | 0.250000 (all 5) | 0.750000 | 0.750000 (all 5) | **+0.500000** | 34 s |
| sLSTM / forward | **0.252430** | 0.026415 / 0.930474 | [0.248021, 0.255545] | 0.750000 | 0.750000 (all 5) | **+0.497570** | 15 s |
| Mamba / bidirectional | **0.250000** | 0.000000 / 1.000000 | 0.250000 (all 5) | 0.750000 | 0.750000 (all 5) | **+0.500000** | 62 s |
| Mamba / forward | **0.248770** | 0.000000 / 0.995078 | [0.248770, 0.250000] | 0.737710 | [0.736969, 0.738062] | **+0.488940** | 19 s |
| CfC / bidirectional | **0.250000** | 0.000000 / 1.000000 | 0.250000 (all 5) | 0.750000 | 0.750000 (all 5) | **+0.500000** | 33 s |
| CfC / forward | **0.250000** | 0.000000 / 1.000000 | 0.250000 (all 5) | 0.750000 | 0.750000 (all 5) | **+0.500000** | 15 s |
| full-corpus runs (any cell x direction) | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

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

```
# Any {lstm,slstm,mamba,cfc} x {bidirectional,forward} combination on the v2 lineage.
speech baseline sad-v2 --corpus-root data/LRE03-LRE07 --out-dir runs/sad_v2_cfc_full \
    --cell-type cfc --direction forward \
    --lanes 1 --seed 0 --epochs 40 --steps-per-epoch 25 --audio-max-duration 120
```

Identical in every knob to the phase-9 `speech baseline sad` recipe -- ONLY the arm name
changes, since `sad-v2` shares the entire SAD skeleton (`_SAD_ARMS`) and every size derives
from the config. To answer the lineage question, fire the SAME cell x direction on both arms
and compare; the live-capacity table above says the comparison is about init scaling and
pack size, not about capacity. Paste resulting numbers into the `full-corpus runs` row above.

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
