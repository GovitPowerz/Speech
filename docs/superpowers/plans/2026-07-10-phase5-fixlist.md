# Phase 5 correctness-critical fix list (the adjudication sweep, Task 2)

Source of truth for the un-quirk track (tasks 4-7). Every IMPROVEMENTS.md entry was
walked against the user-locked criterion (spec S0.3): **fix what produces WRONG RESULTS
or UB; inert quirks (display, dead-code reproductions, working conventions like the
`>150/-200` in-band signaling, the cost-law double-read, `nb_words` -1 on the dead WER
surface) stay documented.** A fix with no observable consequence is by definition not
correctness-critical and drops back to documented.

The sweep GOVERNS task 4-7 membership over the plan's expected list. Corrections to the
plan are stated explicitly in the last section.

Line numbers are approximate anchors into `IMPROVEMENTS.md` at base `6c6acdf`.

## FIX list (correctness-critical; ordered worst-cascade-first within each task)

| # | IMPROVEMENTS anchor (~line) | Wrong-result / UB evidence (one sentence) | Golden families re-pinned | Task / order | Observable-consequence test (RED) |
|---|---|---|---|---|---|
| F1 | `[phase4c] CreateBatches.m non-multilingual nbOfTargetClasses>1 branch indexes Cases/WorstCases by LOOP POSITION` (~2793) | Under the natural contiguous class labeling `0..N-1`, the last loop slot (`ii=N`) writes the top target class into `cases[N-1]`, the slot reserved for the non-target aggregate, silently dropping class-0's files from every batch (`batching.py::create_batches:338/343`). | `tests/test_phase4c_batching.py::test_create_batches_clobber_quirk`, `::test_create_batches_multi_nb_clean_aggregate_and_rotation`, `::test_create_batches_single_and_rotation` (rotation goldens that consume the clobbered aggregate) | Task 4, first (existing RED; silently corrupts training set) | `test_create_batches_clobber_quirk` goes RED: fix = index by class VALUE, so `cases[N-1]` keeps the aggregate instead of class-(N-1)'s data. |
| F2 | `[phase4c] Mutation battery (Task 13)` process-note (~2951-2962) | `get_new_batch`'s two `while len(batch) < nb_cases_per_batch` loops (`batching.py:388/431`) have NO iteration cap: a state where every case group is fully worst-excluded (all classes have `<= nb_worst` files) advances `current_class` forever without adding an element -> infinite hang (a liveness failure, UB-adjacent). | none exists (a hang cannot be pinned directly) -> pin the NEW typed `BatchRotationStuck` on a crafted all-worst-excluded config | Task 4, second (robustness/liveness; hardening for the Phase-5 batch-mode train loop) | Pin-old-behavior-first via a bounded harness: a crafted config that would spin must, post-fix, raise `BatchRotationStuck` after a bounded number of fruitless advances (plan: `2*len(index)`). |
| F3 | `[phase4d] Hard-example mini-batch CADENCE`, item 3 (~3308) | `ComputeGradient.m:72-96` rescales the weighted-listing weight by `nbOfElem/sumInClassIndex` (in-class model) before writing; the port writes the RAW listing weight (`train.py::_files_values` col1), so per-eval class balance is wrong -> class-imbalanced gradients on multi-class corpora. | `tests/test_phase4d_batchmode.py` (byte-exact weighted-listing bytes -- currently pin the RAW weight field) | Task 5, first (consumes Task 4's fixed batching) | Existing byte-exact listing golden goes RED: the rescaled weight field differs from the raw one for a crafted 2-class corpus. Caveat (spec R2): the in-class model must be built driver-side from the mapping file; if a semantic hole appears, document-and-descope. |
| F5 | `[phase4b] PrintConfusionMatrix posTarget/posBestNotTarget are STICKY across rows` (~1801) | `pos_target`/`pos_best_not_target` are reset only ONCE before row 0 (`confusion.rs:59-60,84-85`; `scoring.py:65-66,88-89`); a no-target (out-of-set LID) row -- no column `>150` -- inherits the PRIOR row's target index, so its miss is credited to the wrong class and the confusion matrix depends spuriously on row order (corrupts phase-6 EER/DCF). | Rust: `confusion.rs::sentinel_decode_and_argmax`, `::no_target_as_first_row_uses_header_zero_slot`, `tests/phase4b_confusion_golden.rs::confusion_matrix_matches_harness_transcription` (re-pin to port-truth; harness stays legacy per S1.3). Python: `tests/test_phase4c_scoring.py::test_confusion_matrix_sticky_pos_target_quirk` | Task 6, first (fix BOTH languages identically; existing REDs) | The four sticky-quirk goldens go RED: fix = reset `pos_target`/`pos_best_not_target` to 0 per row alongside `score_target`/`max_score_not_target`. Corpus-LID golden `twin_train_epoch_weights_golden` is UNAFFECTED (its 2-file corpus is all-target -> no no-target rows). |
| F6 | `[phase4a] CSV reference loads ONLY for single-channel audio` (~1462) | `load_ref_from_csv` builds the file-to-open string only for `_ChannelNb == 1` (`bag_of_processors.rs:911` `if channel_count == 1`); stereo audio with a CSV reference loads NO reference, and `load_ref_csv` builds a real SAD segmentation (Speech/Substitution/Insertion), so a scored stereo-CSV run bails on the mandatory-reference gate / goes silently unscored. | none for stereo-CSV empty-ref (only a mono-CSV golden + stereo-STM tests exist) -> pin-old-behavior-first | Task 6, second (needs pin-old-first) | Pin-old-behavior-first: a stereo-CSV corpus test that currently pins EMPTY reference / no-score; fix = load the CSV reference for all channels. Caveat: CSV is primarily a WER (word-level) format and `nb_words` stays on the dead WER surface (KEEP); only the live SAD-reference derivation is the fix. |

Numbering keeps F1/F2/F3/F5/F6 aligned with the plan's expected-member ordering; F4 (weights2nnet
tail-drop) and F7 (LTSVshift) are NOT code fixes -- see corrections below.

### Dependencies

- F3 (rescale) consumes F1 (fixed batching) -- Task 5 after Task 4.
- F2 (cap) is independent of F1 but shares `batching.py` -> lands with F1 in Task 4.
- F5 and F6 are inference/eval-side, independent of F1-F3 and of each other; the two ports of
  F5 (Rust + Python) must land together and fix identically.
- No cross-family golden overlap: the four families (batching, rescale, confusion, reference)
  re-pin disjoint golden sets, so the plan's 4->5->6->7 order is safe.

## KEEP list (documented, not fixed -- entry + one-line reason)

Kept by the Phase-5 sweep's own adjudication. Categories, then the plan-adjacent specifics.

- **In-band signaling** (`>150`/`-200`, `-2.0 -> 100*langID+200` targetLID): working convention, correct results. KEEP.
- **`nb_words` default -1** (`[phase4a] CLOSED (Task 8): result-row nb_words ... -1`, ~1512): on the dead WER surface (xml2wer LOST); no live consumer. KEEP (brief-named).
- **CostLaw double-read** (`[0b-i]`, ~44): read-twice for coefficients vs branch predicate; load-bearing convention, brief-named KEEP.
- **Display-only** (`saveAndUpdate PrintConfusionMatrix outputs`, ~2344; `Corpus ctor ++begin() print`, ~1350; the `%s files processed` misnomer, ~1555): never reach a member/artifact. KEEP.
- **Dead-code faithfulness** (CNN broken-as-committed ~621; the classNb==2 ROC variant ~1856; QPSO dead velocity banks ~2606; the mode-4/5/6 dead synthesis steps; `_MaxSaturation` ~520; SMORMS3/Rprop dead `rand` ~2461): not ported / stream-consumed-only, no live effect. KEEP.
- **Idempotent stateful re-quantization** (TDC/LTSV/spectral `window_shift_sec`, ~732/800/965): proven fixed-point, repeat-call deterministic. KEEP.
- **Port-only safety already stricter than legacy** (compute_errors Pass-1b OOB guard ~105; the modes-4/5/6 negative-index saturate ~2125; the lid5 `rowEnd<rowBegin` guard ~1955; VRCTS spawn-failure typed error ~2286): removes legacy UB, behavior-identical on all fixtures. KEEP.
- **FP/reduction-order parity choices** (adim bias original-restore ~39; ascending-loop GEMM vs Eigen ~260/650; extended-precision `mean` tolerant comparator ~2723): deliberate Rust==Python / portability choices, not wrong results. KEEP.
- **Deliberate documented determinism deviations** (static-lane fold ~1568; Mode-7 `randinit=0` ~2206; QPSO `TableRng` ~2513; augment injected RNG ~3059; opensad15 sequential-not-Pool ~3021): reproducible-by-design improvements over wall-clock non-determinism. KEEP.
- **Genome/config/driver structural deviations** (RunState JSON not .mat ~2583; retrain seeds nnet_best only ~2592; DSP-config injection not config2weights ~2529; LID target from lang_index not fileId ~3291): documented format/binding choices, engine-neutral. KEEP.
- **Feature/NN reproduced quirks** (LTSV variance-not-std ~271; fmath::log no domain guard ~302; two-real periodogram DC 4x normalization ~158; Mel deltas-clobber-static ~220; SDC ignoreFirst last-static clobber ~250; get_sequence left-edge stale ~181; pitch integer division ~291): load-bearing golden contracts; retrain-affecting; not sweep-scoped. KEEP (revisit at a features re-baseline).
- **Confusion family, non-sticky** (exact-tie-to-miss strict `>`, ~1824; zero-row-total unnormalized diagonal, ~1835; ill-conditioned `exit(1)`->panic, ~1846): conventions/edge-cases matching legacy, no clear "correct" alternative. KEEP.
- **Broken-as-committed unrunnable regimes** (signal overlap OOB ~849; mixed-width multi-config bags ~2373; Twin pitch pass / Mode-7 CNN / cep File_Type 2+ typed bails, ~3499): fail loudly, no committed config reaches them. KEEP.

## Plan corrections (the sweep governs; stated explicitly)

1. **MAJOR -- Task 7 (LTSVshift UB) has NO code fix.** The plan and spec S1.2 assume the port
   reproduces the uninitialized-`_LTSVWindowShift` read. It does NOT: `pipeline.rs::FeatureConfig::from_legacy:219`
   reads `_LTSVshift` UNCONDITIONALLY (`get_f64(...)?`, errors if the key is absent), the Phase-1
   guard-drop the `[phase1]` entry (~350) already documents. The algo-5/6 drivers consume
   `feature_cfg.ltsv_shift` (a deterministic config value, `lid.rs:906/1313/1704/2253`); there is
   NO uninitialized-member read anywhere in the port. Every `lid5_*` golden pins the DETERMINISTIC
   branch (the harness itself was fixed to match the port -- "UB KILL" comment), so there is no
   UB-era value to go RED. **Task 7 collapses to a documentation verification**: flip the `[phase1]`
   addendum to record "the port never reproduced this UB (Phase-1 guard drop); confirmed complete
   in Phase 5." Recommend folding this doc-flip into Task 6 and dropping Task 7 as a standalone code
   task (F7 = no-op).

2. **MINOR -- Task 5's weights2nnet tail-drop (F4) is doc-only, already fixed-by-design.** The
   `[phase4c]` entry (~2847) already states the port does NOT reproduce it: `weight_bridge.nnet_to_flat`
   appends the mean/std tail (`:207-208`) and `flat_to_nnet` takes it back (`:274-275`), and
   `scoring.check_grad` perturbs `weights` directly (no config round trip). The directed tests
   ALREADY exist (`test_pack_unpack_flat_roundtrip`, `test_unpack_pack_net_roundtrip`,
   `test_check_grad_normalize_tail_quirk_documented_not_reproduced`). No RED, no code change -- just
   flip the entry to "FIXED-BY-DESIGN (phase 5)".

3. **MINOR -- F6 (stereo-CSV) needs pin-old-behavior-first.** The plan's "a stereo-CSV corpus test
   currently pins EMPTY references -- it exists per the 4a ledger, find it" is optimistic: the
   `[phase4a]` entry (~1462) itself says the Phase-4a tests exercise the STM path on the 2-channel
   excerpt and "a mono-CSV golden lands with the corpus-processor fixtures" -- there is NO stereo-CSV
   empty-reference pin. The RED must be crafted (pin the current empty/bail behavior first).

4. **MINOR -- F2 (get_new_batch cap) is a liveness/robustness fix, not a wrong-NUMBER fix.** On the
   committed fixtures and correct cursor logic it NEVER triggers (the Task-13 note: "not a
   currently-shipping bug"). Kept as a FIX because an infinite hang is a severe, observable failure
   mode reachable when every class group is fully worst-excluded (the degenerate gate does not fully
   prevent this under the F1 clobber), and Phase-5 Task 8/10 wires `get_new_batch` into the live
   `train_modern` loop. Flagged as hardening, not a data-correctness fix.

5. **CONSIDERED, SCOPED OUT (borderline -- flagged for the user/reviewer, NOT mandated Phase-5 fixes):**
   - **Cost-law derivative-consistency quirks** -- LogLaw cost/deriv Adim clamp asymmetry (`[0b-i]`, ~49),
     AboveThreshCubic name-dependent coefficients (~53), softmax `compute_deltas` ponderation-granularity
     asymmetry (~57). These are candidate WRONG-GRADIENT bugs and ARE reachable: the real
     `1_worker_1.config`/`LID_BLSTM.config`/twin configs all use `CostLawSpeech/NoSpeech = log`. But
     spec S0.3 explicitly scopes "the cost-law double-read family" as documented-not-fixed, the LogLaw
     asymmetry is a narrow clamp-boundary effect, and making the derivative consistent with the cost is
     a retrain-affecting semantic change better suited to a dedicated cost-law-correctness pass at
     Phase 6 (when real training exposes it). Recommend revisiting there.
   - **Unstabilized output softmax** (`[phase2]`, ~414): no `- rowwise().max()` before `exp` -> `inf`/`NaN`
     on large pre-activations. A latent from-scratch-training risk (Phase 5/6), but golden-neutral (no
     committed fixture overflows) and mitigated by Xavier/He init (Task 3) which bounds early activations.
     The fix is softmax-shift-invariant (bit-identical except at overflow). Recommend adding the
     max-subtraction opportunistically in Task 8's modern loop, or documenting as a known limitation --
     not a mandated un-quirk fix.

## Net verdict

**6 correctness-critical code fixes** (F1, F2, F3, F5-Rust, F5-Python, F6) across tasks 4-6;
**Task 7 is empty** (LTSVshift already de-UB'd -- doc-flip only); **F4 (tail-drop) is doc-only**.
Two borderline families (cost-law derivatives, unstabilized softmax) flagged for a Phase-6
training-correctness pass, not fixed here.
