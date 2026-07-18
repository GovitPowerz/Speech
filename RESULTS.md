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
The train objective descends (2.368 -> 2.216); the softmax-CE VALIDATION cost can rise from
scratch (the net grows confident on the classes it learns first) even as argmax accuracy
improves, so the direction-safe improvement signal is the argmax error, not the CE. Genuine
12-way discrimination is the full-corpus launcher's job.

### SAD (Algo 3 spectral, File_Type 0 wav) -- Task 9

From-scratch algo-3 spectral SAD on the `train/audio` wav+xml pairs (the seeded 70/15/15 split
of the 2066 pairs, `dataprep.lre.derive_sad_listings`). The held-out slice is scored END TO END
with the T4 DCF harness: the engine dumps one VRCTS hypothesis xml per test file
(`Dump_Directory`), read back via `evaluate.load_vrcts_hyp`; the `.part.xml` references via
`evaluate.load_vrcts_ref` (windowed to the capped-audio span, matching the engine's own
`_AudioDuration` reference windowing); pooled through `evaluate.dcf`. These are the FIRST REAL
DCF numbers under this protocol. DCF pooled per-collar over the whole held-out slice (Pmiss/Pfa
accumulated across a concatenated timeline, the single-number analogue of the LID arm's pooled
argmax error).

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

### LID -- phonotactic regime (Algo 6, Mode 7, File_Type 1) -- Task 10

| split | files (train/valid/test) | LID error % | Cavg | chance % | config | seed |
|---|---|---|---|---|---|---|
| subset gate | TBD | TBD | TBD | 91.67 | `lre03_lid_phseq.toml` | TBD |
| full run | TBD | TBD | TBD | 91.67 | `lre03_lid_phseq.toml` | TBD |

---

## Firing a full run (post-phase, user-fired)

```
# LID features (Algo 6, Mode 7, cep)
speech baseline lid-features --corpus-root data/LRE03-LRE07 --out-dir runs/lid_features_full \
    --lanes 1 --seed 0 --epochs 40 --steps-per-epoch 25

# SAD (Algo 3 spectral, wav) -- --audio-max-duration lifts the subset gate's short cap; omit
# it to score full-length recordings (the corpus wavs are 1800 s CallFriend files).
speech baseline sad --corpus-root data/LRE03-LRE07 --out-dir runs/sad_full \
    --lanes 1 --seed 0 --epochs 40 --steps-per-epoch 25 --audio-max-duration 120
```

Omit `--subset` to train on the whole split (SAD: the full 70% train split of the 2066 wav/xml
pairs; LID: the whole localized corpus); `--lanes N` sets the fold width (record N here);
`--resume` continues from `<out-dir>/checkpoint`. The run writes `run_metadata.json`
(seed/lanes/subset/config-hash/audio cap), `checkpoint/` (best/last packs + `train_history.json`),
and the held-out scores -- LID `scores/` (`.scr`) -> `lid_error`/`cavg`; SAD `score_trained/`
(VRCTS hyp xml) -> pooled `dcf`. Paste the resulting numbers into the `full run` rows above.
