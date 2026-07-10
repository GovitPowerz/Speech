# Phase 4d: dataprep + end-to-end parity -- design spec

Date: 2026-07-09
Branch: `feature/phase-4d-dataprep-endtoend` (base main@07b479b, post-4c-merge)
Approach: A ("fresh-oracle-first"), user-approved.

Phase 4d is the FINAL roadmap phase. It closes the last all-stub area (`dataprep/`), executes both
deferred parity milestones (end-to-end score parity + the `-ffast-math` tolerance comparison against
the original 2015 production binary), closes the deferred Phase-0a packer/adim golden, ports the four
feasible cross-phase backlog items, and documents the four blocked items as complete-as-portable.
At the end of 4d the roadmap is declared complete.

## S0. Exploration facts the design rests on (verified 2026-07-09, do not re-derive)

- The 2015 production binary is **macOS Mach-O x86_64**, NOT Linux ELF (CLAUDE.md's "the 2015 Linux
  binary likely won't run on Darwin/arm64" was WRONG -- full-tree `file` scan found zero ELF).
  `Release/bin/fsp` (8.5 MB, 15 Oct 2015) is byte-size-identical to the deployed
  `Optimizer_V6.2.2/Executables/15-Oct-2015_BLSTM_OpenSAD15/Segment`.
- Build flags (legacy `CMakeLists.txt`, g++-4.9): Release = `-O3 -std=c++0x -fomit-frame-pointer
  -fopenmp -DNDEBUG -fno-operator-names -msse2 -mfpmath=sse -ffast-math -march=core2 -fpermissive`.
  Links: `pthread sndfile FLAC ogg vorbisenc matio z png` + the gcc-4.9 runtime.
- Rosetta 2 is installed and functional on this host (`arch -x86_64 /usr/bin/true` passes).
- All nine non-system dylibs the binary references are ABSENT (`/usr/local/lib/libsndfile.1.dylib`,
  `libFLAC.8`, `libogg.0`, `libvorbisenc.2`, `libmatio.2`, `libpng16.16`, and
  `/usr/local/lib/gcc/4.9/{libstdc++.6,libgomp.1,libgcc_s.1}`) -- the 2015 x86_64 `/usr/local`
  prefix is gone. All are rebuildable/extractable x86_64 (see S3).
- **Tuple A** (primary parity material): `Optimizer_V6.2.2/Executables/15-Oct-2015_BLSTM_OpenSAD15/`
  -- `Segment` binary, `1_worker_1.config` (Algo_choice 3, BLSTM 23/24/24, output 48/12/1),
  `NNweights_config1.bin` (269 KB), `ParamStruct.mat`, input listings (`plot`/`current`/`full`),
  expected `MultiConfigResults_worker_1.mat` (1.68 MB) + 97 `bestNNWeight_N` mats + 685 per-file
  `.mat` in the Figures run dir + `NNweights_epoch995.mat`. **Its LDC OpenSAD15 audio is ABSENT.**
- **Tuple B** (independent variant): `Optimizer_V6.2.1/Executables/14-Oct-2015_BLSTM_OpenSAD15/` --
  same shape, `NNweights_config1.bin` (343 KB), 14 Oct 2015.
- **Tuple C** (only in-tree audio+reference pair): root `PRCTS_RUS_RU_0000263489_01.wav` (35 MB,
  RIFF PCM 16-bit STEREO 8 kHz, Russian CTS) + `PRCTS_RUS_RU_0000263489_01.stm` (67 KB).
- **No `.scr` file exists anywhere** in the legacy tree -- the only `.scr` oracle is Octave driving
  `Test_BLSTM.m`'s writer block.
- **Balances 6-9 are unportable**: they shell out to `/people/gelly/Scripts/xml2wer{,_multi,_list}.py`
  over ssh to 2015 lab servers; those scripts exist nowhere locally. Source lost.
- Dataprep sources (624 LOC total): `AugmentCorpus.py` 73 (Python 2, sox/soxi shell-outs, hardcoded
  paths, 5 noise/pitch/tempo variants per file, durMin=2 gate), `ProcessOpenSAD15Corpus.py` 145
  (Python 2, PURE TEXT: NIST `.tab` -> VRCTS-style XML + STM with 2s-collar excluded_region merge +
  combined `.flst`; `Pool(20)`), `norm_stm_pkt_light.pl` 63 (self-contained regex pipeline),
  `norm_stm_pk_cts_all_trans_03a.pl` 210 (regex + a `norm-tagger | norm-parser --lang=spa`
  shell-out to lost LIMSI binaries), `norm_stm_train_05a_p5_Quaero.sh` 78 (hardcoded LIMSI infra,
  not worth porting -- driver only), `WriteListing.m` 22 (4-field flst + `fliplr` worker shards),
  `WriteWeightedListing.m` 22 (6-field weighted listing, `%g`; worker-shard block commented out),
  `processListing.m` 89 (already covered by `batching.read_listing` in 4c).
- `/usr/bin/perl` v5.34 exists on this host (and perl is preinstalled on GH runners, but the
  local-only-oracle policy stays uniform: extractor-produced committed fixtures, CI never runs perl).
- No Python 2 interpreter exists anywhere reachable -- the two legacy `.py` scripts have NO live oracle.
- The 2015 corpus dirs (`/Users/Govit/Documents/AudioFiles/...`) are absent on this machine. User
  answer on LDC audio: "Maybe, check later" -> tuple-A replay is an OPTIONAL, AUDIO-GATED milestone.
- User scope decision: port the feasible backlog four; document the blocked four
  (Twin pitch second pass, Mode-7 WAV/CNN arm, cep ingestion, balances 6-9).

## S1. Deliverables

### S1.1 dataprep/opensad15.py (port of ProcessOpenSAD15Corpus.py)
The pure-logic core is split from I/O for pinnability:
`convert_tab_file(audio_path: str, tab_lines: list[str], ...) -> (xml_bytes, stm_lines, listing_line)`
-- the field-4 `S`/`RI` segment filter, the `SegsExcl` 2.0s-collar merge with its 0.1 snap and
first/last clamp quirks, the exact XML byte format (`AudioDoc`/`vrcts_part v1.3` schema, the same
family `tasks/segmentation_io.rs::load_vrcts` already parses), the STM `excluded_region`/`toto toto`
emission, the `path_leaf(audiofile[:-5])` basename-slice quirk reproduced EXACTLY (legacy source
governs; every quirk found gets an IMPROVEMENTS entry). `process_opensad15(listing: Path, out: Path)`
wraps it: reads the `;`-listing, writes per-file `.xml`/`.stm` next to the `.tab`, appends to the
combined output listing. The legacy `multiprocessing.Pool(20)` becomes a SEQUENTIAL loop -- a
documented determinism deviation (same class as the 4a static-lane model): output-listing line order
is input order, not pool completion order.

### S1.2 dataprep/augment.py (port of AugmentCorpus.py)
`augment_corpus(listing: Path, rng: np.random.Generator, runner: SoxRunner, dur_min: float = 2.0)`.
All sox/soxi COMMAND STRINGS are built by pure pinned code preserving the legacy draw order per file
(noise = abs(0.005*randn()); noisetype = 4*rand() with the <1/<2/<3/<=4 ladder; pitch = 400*rand()-200;
tempo = 0.4*rand()+0.8; 5 variants per file) and the exact `_mod_%.3f_<type>_%.3f_%.3f.wav` naming.
Execution goes through an injected runner (default: real `subprocess`; tests: a recording stub -- the
`VrctsPart` does-it-attempt pattern). The soxi `-D` duration read gates on `dur_min`. NO legacy RNG
oracle exists (Python 2 numpy on wall-clock seed): the injected-Generator draw order is a documented
CONVENTION, not a parity claim.

### S1.3 dataprep/stm_normalize.py (port of the two .pl normalizers)
`normalize_stm_light(lines) -> lines`: full transcription of `norm_stm_pkt_light.pl` (comment
passthrough, blank-line drop, `ignore_` passthrough, head/text split at field 6, the filler/hesitation
regex battery, partial-word `(-word)` handling, punctuation strip, empty -> `ignore_time_segment_in_scoring`).
Pinned BYTE-FOR-BYTE against the REAL `.pl` executed by `/usr/bin/perl` on crafted STM fixtures --
the live-oracle tier of this phase; the extractor commits input/output fixture pairs, CI compares
fixtures only. `normalize_stm_full(lines, ...)`: the 03a regex core transcribed the same way; the
`norm-tagger | norm-parser` number-normalization arm is a TYPED BAIL (lost LIMSI binaries, same class
as xml2wer). `norm_stm_train_05a_p5_Quaero.sh` is NOT ported (hardcoded absent infra; document).

### S1.4 batching.py listing writers (port of WriteListing.m / WriteWeightedListing.m)
`write_listing(base: Path, items, nb_workers) -> None` (the `.flst` + `_worker_N.flst` shards with
the `fliplr(length-jj+1:-nbworker:1)` interleave reproduced index-for-index) and
`write_weighted_listing(path: Path, items, values) -> None` (6-field `%g` rows). Octave TIER-1
pinnable (pure sprintf); the `%g` float formatting must match Octave's exactly on the fixture values.

### S1.5 tools/fsp_runtime/ -- the resurrected 2015 binary (LOCAL-ONLY)
A setup script (`setup.sh`) builds a SELF-CONTAINED x86_64 prefix under `tools/fsp_runtime/prefix/`
(zero `/usr/local` pollution): libogg, libvorbis, FLAC, libsndfile, libpng, and matio built from
source under `arch -x86_64` (matio at a 1.3.x-era tag matching the `libmatio.2` install name; if the
modern ABI proves compatible for the symbols the binary imports, a newer tag + symlink is acceptable
-- decided empirically), plus the three gcc runtime dylibs (`libstdc++.6`, `libgomp.1`, `libgcc_s.1`)
EXTRACTED from a Homebrew x86_64 gcc bottle tarball (libstdc++ is ABI-forward-compatible; no brew
install, no compiler build). The `Segment` binary is COPIED out of the legacy tree (`legacy/` is
never modified; the copy lives under `tools/fsp_runtime/bin/`, git-ignored), its 9 install names
rewritten via `install_name_tool -change`, ad-hoc re-signed (`codesign -f -s -`), run under
`arch -x86_64`. SUCCESS CRITERION (gate for the parity tasks): prints the usage text for `argc<3`,
then COMPLETES a solo (`-s`) run on a local wav under a localized config, writing parseable outputs.
BOUNDED AT ONE TASK. FALLBACK if the ABI archaeology fails: build the vendored legacy source in the
existing C++ oracle harness with the 2015 math flags (`-O3 -ffast-math -msse2 -mfpmath=sse`, modern
g++, native arch) -- same math semantics, today's codegen -- and document the downgrade in
IMPROVEMENTS.md + the spec deviation log. CI NEVER touches any of this; it consumes committed fixtures.

### S1.6 The parity gates (the phase's headline)
An extractor (`tools/fsp_runtime/run_oracle.py`, local-only) runs the resurrected binary and commits
fixtures under `tests/reference_data/phase4d/`:
- INPUTS: a deterministic 30-60s EXCERPT of the PRCTS stereo wav (cut once by the extractor via
  soundfile, committed -- the 35 MB original stays out of the repo) + the already-committed phase-1/2
  fixture wavs; tuple-A's `1_worker_1.config` with paths localized (and a tuple-B variant); the REAL
  `NNweights_config1.bin` packs (committed -- 269/343 KB).
- ORACLE OUTPUTS: VRCTS XML bytes + `MultiConfigResults` (scipy-converted to `.bin` per the S6/4a
  precedent) + per-file result rows, from the REAL `-ffast-math` x86_64 binary under Rosetta.
- PORT GATES (pytest + cargo, run in CI against the committed fixtures): the Rust `speech` binary /
  `speech_rs.Engine` on IDENTICAL inputs asserts (a) segment counts + types EXACT (structural
  agreement is the floor), (b) boundary times within a MEASURED-THEN-PINNED bound, (c) cost/score
  columns within measured-then-pinned relative tolerances. The bounds are MEASUREMENTS recorded in a
  manifest (like the NN_TOL calibration data), not invented numbers; x86 fast-math vs arm64
  strict-IEEE cannot bit-match and the milestone has always been a tolerance comparison. A structural
  mismatch is a hard failure, not a bigger tolerance.
- THE PHASE-0A CLOSURE: real config -> `NnetSpec` -> unpack the real `NNweights_config1.bin` ->
  repack -> BYTE-IDENTICAL (both tuples); and the port consuming that pack produces the gate's
  segmentations. This closes the "bit-exact packer/adim golden against a real `.bin`" deferral from
  Phase 0a (CLAUDE.md Locked decisions).
- TUPLE-A REPLAY (OPTIONAL, AUDIO-GATED): a test that runs the port over the tuple-A `plot` listing
  and compares against the SAVED 2015 `MultiConfigResults_worker_1.mat`, skipped unless the OpenSAD15
  audio is present at a documented local path (user: "maybe, check later"). Ships as skip-by-default
  + an activation README note; NOT a phase exit criterion.

### S1.7 Backlog four (feasible deferred items)
- BATCH-WISE REBUILDS: `drivers/train.py` gains the real inner-batch mode -- `create_batches` /
  `get_new_batch` (both Octave-pinned in 4c) + `write_weighted_listing` per inner step + a per-batch
  `Engine` rebuild consuming `forward_backward`'s `listing_override` for real (closing the 4c
  "advisory" deferral and the stale-forward-promise history). Pinned by listing-byte goldens + an
  extension of the deterministic exit gate (batch mode, run twice, bit-identical).
- ALGO-3 DRIVERS: `ps_from_config` / `_tail_lengths` / `_backprop_inner` / `score_genome` /
  `evaluate` generalized to single-net (non-LID) configs -- the algo-6-only IndexError documented in
  the 4c final review. Exercised by an algo-3 train smoke on committed 4a tier-2 fixtures + the
  tuple-A-derived parity config (synergy with S1.6).
- .SCR OCTAVE GOLDEN: an octave_harness stage drives `Test_BLSTM.m`'s score-writing block
  (`:252-266`) on injected scores -> committed `.scr` golden; `evaluate` gets pinned against it.
  This FORCES the class-key-order resolution: the port switches to the legacy ALPHABETICAL
  `keys(langMapConf)` order (closing the documented T12 divergence) so the golden can be byte-exact.
- T8 DEBT: `stage_vec2struct` grows mask-vector-arm cases (per-block/output/normalize masks);
  `_fmt_scalar` gets a boundary-format unit table.

### S1.8 Blocked-four closure (documentation, not code)
Typed bails verified/added where a code path exists; IMPROVEMENTS.md gets a "complete-as-portable"
closure subsection: balances 6-9 (xml2wer source LOST -- `compute_cost` already raises a typed
ValueError; the entry records WHY it can never be ported), Twin pitch second pass (portable but
4b-deferred; stays deferred by user decision -- entry states what it would take), Mode-7 WAV/CNN arm
(CNN broken-as-committed since Phase 2), cep ingestion (`File_Type` 2+ -- no cep data exists anywhere
to validate against). The README roadmap gains the final-state statement.

## S2. Oracle tiers (weakest-to-strongest, per artifact)

| Artifact | Oracle | Tier |
|---|---|---|
| opensad15 converter | crafted .tab fixtures, hand-verified + `load_vrcts` cross-parse | transcription (NO live oracle -- py2 lost; the honest weakest tier, stated in-module) |
| augment commands | recording-runner command-string goldens | does-it-attempt (RNG convention documented, not parity) |
| stm light normalizer | REAL .pl via /usr/bin/perl, extractor fixtures | LIVE byte-oracle |
| stm full normalizer | REAL .pl regex core via perl (tagger arm bailed) | live byte-oracle, partial surface |
| listing writers | Octave Tier-1 (real .m, sprintf) | live byte-oracle |
| .scr writer | Octave stage over Test_BLSTM.m's block | live byte-oracle (transcribed block if injection impossible -- adjudicated at implementation) |
| parity gates | the REAL 2015 -ffast-math binary via Rosetta | production-binary oracle (the strongest of the whole project) |
| parity fallback | vendored source rebuilt with 2015 math flags | documented downgrade |
| packer/adim | the real NNweights_config1.bin (x2 tuples) | byte golden |

## S3. Risks

- R1 matio/libstdc++ ABI defeats the bring-up: BOUNDED at one task; fallback defined (S1.5); the
  phase does not block on it (parity gates re-point at the fallback oracle; manifest records which
  oracle produced the fixtures).
- R2 fast-math deltas exceed tight bounds on the PRCTS excerpt (8 kHz stereo CTS is unlike the
  fixture material): measure FIRST, pin honestly; structural agreement is the non-negotiable floor;
  if even structure diverges on the excerpt, shorten/re-cut the excerpt deterministically and record
  the adjudication.
- R3 the opensad15 port has no live oracle: mitigated by the pure-core split, `load_vrcts`
  cross-validation, hand-verified fixtures, and a mutation battery entry; risk accepted and stated.
- R4 the tuple-A config may depend on OpenSAD15-specific keys the localized excerpt config cannot
  exercise (e.g. reference STM semantics on CTS audio): adjudicated at implementation; the gate's
  scored columns may narrow to the unscored subset with the narrowing documented.
- R5 Octave `%g` vs Python `%g` formatting drift in the listing writers: fixture values chosen to
  probe the boundary (integers, 6-sig-fig, exponent switch); any genuine divergence is pinned to
  Octave's bytes (the port matches OCTAVE, documented if surprising).
- R6 committed-fixture size: the PRCTS excerpt + two weight packs + oracle outputs must stay in the
  low single-digit MB total; the extractor asserts a size budget.

## S4. Determinism / deviation log (documented, IMPROVEMENTS'd)

- Pool(20) -> sequential (S1.1). Injected-Generator RNG in augment (S1.2). JSON-side batch listing
  writes use the port's existing path conventions. The tuple-A replay's skip-unless-audio gate.
  The fallback-oracle downgrade if R1 fires.

## S5. Mutation battery (plan will finalize; candidates)

opensad15 collar 2.0 -> 1.0 (fixture golden breaks); the S/RI filter widened (golden breaks); the
`[:-5]` slice "fixed" to `[:-4]` (golden breaks); light-normalizer filler regex dropped (perl-oracle
fixture breaks); write_listing fliplr stride off-by-one (Octave golden breaks); augment noisetype
ladder boundary moved (command golden breaks); parity-gate structural assert weakened (the gate must
FAIL on a crafted segment-type flip); batch-mode rotation wiring bypassed (exit-gate extension breaks).

## S6. Documentation

Per-task IMPROVEMENTS entries with real Pinned-by citations (fabrication history: reviewers verify);
README Phase 4d done-bullet + ROADMAP COMPLETE statement; CLAUDE.md dataprep row -> Implemented,
project-overview de-stubbed, the "2015 Linux binary" error CORRECTED, conventions gain the
fsp_runtime narration; the scaffolding-relaxations `allow_empty_bodies` finally removable (no stub
modules remain -- verify mypy passes without it).

## S7. Finish

Subagent-driven per-task loop (fresh implementer + reviewer, opus for the bring-up/parity/RE tasks),
mutation battery task, docs task, final whole-branch review (most capable model), smart-commit
(whole branch), finishing-a-development-branch (user merges via PR; never push).
