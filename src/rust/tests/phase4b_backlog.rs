//! Phase 4b Task 10: the three 4a-backlog goldens (IMPROVEMENTS.md "[phase4a]
//! Phase 4b test backlog (from the 4a final review)"):
//!
//!   1. best-cost TIE golden: closes the `>` vs `>=` save-gate coverage gap
//!      the Task 11 mutation battery found (`phase4a_save_update.rs`'s
//!      `best_cost_gate_fires_and_skips` only ever exercises strict
//!      improve/worse costs, never a tie).
//!   2. fold-order-divergence golden: a 3-file corpus (`f1`/`f2`/`f3`) whose
//!      per-file derivative contributions sum NON-commutatively (float `+=`
//!      over 3+ terms is not associative), closing the gap the tier-2
//!      2-file fixture's commutative merge masked.
//!   3. pitch-pass target-reuse pin under LIVE STM references (see
//!      `phase2b_spectral_golden.rs`'s NN-chain-only pitch coverage +
//!      IMPROVEMENTS.md:~1608 "spectral PITCH second pass REUSES the pass-1
//!      target buffer").

mod common;

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Mutex;

use indexmap::IndexMap;
use ndarray::Array2;

use speech::cli::{Mode, ModeKind};
use speech::engine::bag_of_processors::BagOfProcessors;
use speech::engine::channel_result::ChannelResult;
use speech::engine::corpus_processor::CorpusProcessor;
use speech::features::stats::InputStatistics;

/// `set_current_dir` is process-global; serialize against `cargo test`'s
/// default parallel threads (same pattern as `phase4a_save_update.rs`/
/// `phase4a_train_golden.rs`).
static CWD_LOCK: Mutex<()> = Mutex::new(());

struct CwdGuard {
    original: PathBuf,
}

impl CwdGuard {
    fn enter(dir: &std::path::Path) -> CwdGuard {
        let original = std::env::current_dir().unwrap();
        std::env::set_current_dir(dir).unwrap();
        CwdGuard { original }
    }
}

impl Drop for CwdGuard {
    fn drop(&mut self) {
        let _ = std::env::set_current_dir(&self.original);
    }
}

fn ref_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/reference_data")
}

fn phase4a_corpus_dir() -> PathBuf {
    ref_dir().join("phase4a/corpus")
}

fn load_config(rel: &str) -> IndexMap<String, String> {
    let text = std::fs::read_to_string(ref_dir().join(rel)).unwrap();
    speech::legacy_config::parse_legacy_config(&text)
}

fn multi_mode() -> Mode {
    Mode {
        kind: ModeKind::Multi,
        verbose: false,
    }
}

// =============================================================================
// Item 1: best-cost TIE golden.
// =============================================================================
//
// `BagOfProcessors::save_weights`'s gate is strict `>` (`bag_of_processors.rs`
// `:524/532/540/549`, from `BagOfProcessors.cpp:149/156/165/172`): a save
// fires only when the running best beats the tie-breaking `save_criterion`,
// i.e. `bestCost > cost`. A worse OR a tied cost must both leave `bestCost`
// unchanged, but only a TIE distinguishes `>` from a `>=` mutation (a worse
// cost skips under either operator). `phase4a_save_update.rs`'s
// `best_cost_gate_fires_and_skips` only exercises a strict "epoch B is
// WORSE" case -- Task 11's mutation battery flagged this as a genuine
// coverage gap (IMPROVEMENTS.md "Mutation battery (Task 11)" mutation (2)).
// This test mirrors that test's structure exactly (algo-3 bag, real net,
// tempdir + mtime/content artifact check) but crafts epoch B's cost BYTE-
// EQUAL to epoch A's -- a genuine tie, not merely a non-improvement.

/// Real algo-3 bag (single config), weights overridden to the known
/// 33,671-weight net -- identical helper to `phase4a_save_update.rs::algo3_bag`.
fn algo3_bag(weights_path: &std::path::Path) -> BagOfProcessors {
    let mut cfg = load_config("phase0/1_worker_1.config");
    cfg.insert("numOuterThreads".to_string(), "1".to_string());
    cfg.insert("Algo_choice".to_string(), "3".to_string());
    cfg.insert(
        "BLSTM_weightsFile".to_string(),
        weights_path.to_str().unwrap().to_string(),
    );
    BagOfProcessors::from_configs(
        std::slice::from_mut(&mut cfg),
        Mode {
            kind: ModeKind::Solo,
            verbose: false,
        },
    )
    .unwrap()
}

/// Epoch A fires (beats the 1e20 seed); epoch B's cost is BYTE-EQUAL to
/// epoch A's (5.0 == 5.0, a genuine tie, via the identical col4/col17 row
/// values) -- the `>` gate must SKIP (no re-save): `bestCost` stays at
/// epoch A's value, and the `bestNNWeight` artifact is untouched (same mtime
/// and bytes as after epoch A). A `>` -> `>=` mutation would instead re-fire
/// on the tie (`5.0 >= 5.0` is true) and rewrite the artifact with a fresh
/// mtime -- this is the distinguishing observable the 4a battery found
/// missing.
#[test]
fn best_cost_gate_skips_on_exact_tie() {
    let _lock = CWD_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let weights_src = ref_dir().join("phase0/NNweights_config1.bin");
    let tmp = tempfile::tempdir().unwrap();
    let weights_path = tmp.path().join("weights.bin");
    std::fs::copy(&weights_src, &weights_path).unwrap();

    let mut bag = algo3_bag(&weights_path);
    let derivs_mat = bag.get_weights_derivatives(0)[0].clone();
    let stats0 = bag.get_input_statistics(0);

    let mut best_cost: BTreeMap<usize, f64> = BTreeMap::from([(0, 1e20)]);
    let derivs: BTreeMap<usize, Vec<Array2<f64>>> = BTreeMap::from([(0, vec![derivs_mat])]);
    let stats: BTreeMap<usize, Vec<InputStatistics>> = BTreeMap::from([(0, stats0)]);

    let _cwd = CwdGuard::enter(tmp.path());

    let out_a = "run_a.mat";
    // Col 4 (cost numerator) = 5.0, col 17 (denom) = 1.0 -> cost = 5.0.
    #[rustfmt::skip]
    let row_a: [f64; 18] = [0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0];
    let m_a = vec![ChannelResult::from_row(&row_a)];

    let mut cost_mem = [0.0; 1];
    let mut bad_classif = [0.0; 1];
    let mut cost_lid = [0.0; 1];
    let mut bad_classif_lid = [0.0; 1];

    bag.save_and_update(
        out_a,
        &[m_a],
        &mut best_cost,
        &derivs,
        &stats,
        &mut cost_mem,
        &mut bad_classif,
        &mut cost_lid,
        &mut bad_classif_lid,
    )
    .unwrap();

    assert_eq!(best_cost[&0], 5.0, "epoch A fires (5.0 < 1e20 seed)");
    let weights_artifact = PathBuf::from(format!("weights_bestNNWeight_1_{out_a}"));
    assert!(weights_artifact.exists(), "{weights_artifact:?} missing");

    let weights_mtime_a = std::fs::metadata(&weights_artifact)
        .unwrap()
        .modified()
        .unwrap();
    let weights_bytes_a = std::fs::read(&weights_artifact).unwrap();

    let out_b = "run_b.mat";
    // BYTE-EQUAL cost to epoch A: same col4/col17 values -> cost == 5.0 exactly
    // (not merely "close" -- both are the literal f64 5.0/1.0, so the division
    // reproduces the identical bit pattern).
    #[rustfmt::skip]
    let row_b: [f64; 18] = [0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0];
    let m_b = vec![ChannelResult::from_row(&row_b)];

    bag.save_and_update(
        out_b,
        &[m_b],
        &mut best_cost,
        &derivs,
        &stats,
        &mut cost_mem,
        &mut bad_classif,
        &mut cost_lid,
        &mut bad_classif_lid,
    )
    .unwrap();

    assert_eq!(
        cost_mem[0], 5.0,
        "epoch B's aggregated cost is byte-equal to epoch A's (genuine tie)"
    );
    assert_eq!(
        best_cost[&0], 5.0,
        "bestCost unchanged on a tie (the `>` gate does not treat == as an improvement)"
    );

    // No artifact for run_b was ever created -- the tie did not trigger a NEW
    // save under a different name.
    let weights_artifact_b = PathBuf::from(format!("weights_bestNNWeight_1_{out_b}"));
    assert!(!weights_artifact_b.exists());

    // The run_a artifact is untouched (same mtime + bytes): epoch B did NOT
    // re-save over it either. This is the observable a `>` -> `>=` mutation
    // flips (see IMPROVEMENTS.md item 1, marked CLOSED by this test).
    let weights_mtime_b = std::fs::metadata(&weights_artifact)
        .unwrap()
        .modified()
        .unwrap();
    let weights_bytes_b = std::fs::read(&weights_artifact).unwrap();
    assert_eq!(
        weights_mtime_a, weights_mtime_b,
        "tie must not touch the epoch-A artifact's mtime"
    );
    assert_eq!(
        weights_bytes_a, weights_bytes_b,
        "tie must not rewrite the epoch-A artifact's contents"
    );
}

// =============================================================================
// Item 2: fold-order-divergence golden.
// =============================================================================
//
// `run_epoch`'s derivative fold (`corpus_processor.rs:400-411`) accumulates
// each file's per-conf derivative matrices via plain `Array2` `+`. IEEE754
// addition is COMMUTATIVE for 2 operands (`a+b` is bit-identical to `b+a`),
// which is exactly why the tier-2 fixture's 2-file merge (Task 11's mutation
// battery) could not distinguish an ascending vs a reversed fold. But
// addition of 3+ terms is NOT associative: `(a+b)+c` and `(c+b)+a` can
// differ in rounding whenever the intermediate sums land at different
// magnitudes. This uses the phase4a 3-file corpus (`f1`/`f2`/`f3`, already
// committed with STM references) through the real algo-3 net with
// `BackPropagationActivated true`, so per-file derivative contributions are
// live (Task 7b), non-trivial, and (measured below) genuinely
// order-sensitive.

fn phase0_dir() -> PathBuf {
    ref_dir().join("phase0")
}

/// Seed a 3-file corpus (`f1`/`f2`/`f3` wav+STM, copied from the phase4a
/// fixture dir) + a fresh fileslisting/mapping + the real net weights, all
/// workdir-relative (mirrors `phase4a_train_golden.rs::seed_tier2`, extended
/// to 3 files under a phase4b-local fileslisting so this test does not touch
/// any phase4a fixture).
fn seed_fold_order(dir: &std::path::Path) {
    let src = phase4a_corpus_dir();
    let corpus_dst = dir.join("corpus");
    std::fs::create_dir_all(&corpus_dst).unwrap();
    for f in ["f1", "f2", "f3"] {
        std::fs::copy(
            src.join(format!("{f}.wav")),
            corpus_dst.join(format!("{f}.wav")),
        )
        .unwrap();
        std::fs::copy(
            src.join(format!("{f}.stm")),
            corpus_dst.join(format!("{f}.stm")),
        )
        .unwrap();
    }
    std::fs::write(
        dir.join("fold_order_language2classmapping.csv"),
        "eng;us;0\nunk;unk;1\n",
    )
    .unwrap();
    std::fs::write(
        dir.join("fold_order_fileslisting.csv"),
        "corpus/f1.wav;corpus/f1.stm;eng;us;0.5;2\n\
         corpus/f2.wav;corpus/f2.stm;;;0.75\n\
         corpus/f3.wav;corpus/f3.stm;;;0.6\n",
    )
    .unwrap();
    std::fs::copy(
        phase0_dir().join("NNweights_config1.bin"),
        dir.join("NNweights_config1.bin"),
    )
    .unwrap();
}

/// The real `1_worker_1.config` (algo 3) + the Task-9-style tier-2 overrides
/// (`BLSTM_BackPropagationRpropInit 0.05`, non-pathological step size) minus
/// any training-epoch key (`run_epoch_fold_probe_for_test` drives ONE pass
/// itself; `fold_order_ascending_matches_production` sets its own epoch
/// count), pointed at the 3-file fold-order corpus.
fn fold_order_config() -> IndexMap<String, String> {
    let mut m = load_config("phase0/1_worker_1.config");
    m.insert("numOuterThreads".to_string(), "1".to_string());
    m.insert("Algo_choice".to_string(), "3".to_string());
    m.insert("Audio_offset".to_string(), "0.0".to_string());
    m.insert("Audio_max_duration".to_string(), "2.0".to_string());
    m.insert("File_Type".to_string(), "0".to_string());
    m.insert(
        "BLSTM_BackPropagationActivated".to_string(),
        "true".to_string(),
    );
    m.insert(
        "BLSTM_BackPropagationRpropInit".to_string(),
        "0.05".to_string(),
    );
    m.insert(
        "BLSTM_weightsFile".to_string(),
        "NNweights_config1.bin".to_string(),
    );
    m.insert(
        "language2classmapping".to_string(),
        "fold_order_language2classmapping.csv".to_string(),
    );
    m.insert(
        "fileslisting".to_string(),
        "fold_order_fileslisting.csv".to_string(),
    );
    m.insert(
        "multiConfigResultsOutputFile".to_string(),
        "fold_order.mat".to_string(),
    );
    m
}

/// MEASUREMENT (bounded search, documented per the brief): with the base
/// `NNweights_config1.bin` seed, do the ascending vs descending derivative
/// folds over `f1,f2,f3` diverge in bits? Returns `(n_differing, n_total)`
/// over config-0's first (only, for algo 3) derivative matrix.
fn measure_fold_divergence(cfg: &IndexMap<String, String>) -> (usize, usize) {
    let ascending =
        CorpusProcessor::run_epoch_fold_probe_for_test(vec![cfg.clone()], multi_mode(), false)
            .unwrap();
    let descending =
        CorpusProcessor::run_epoch_fold_probe_for_test(vec![cfg.clone()], multi_mode(), true)
            .unwrap();
    let a = &ascending[&0][0];
    let b = &descending[&0][0];
    assert_eq!(
        a.dim(),
        b.dim(),
        "ascending/descending derivs shape must match"
    );
    let n_diff = a
        .iter()
        .zip(b.iter())
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count();
    (n_diff, a.len())
}

/// MEASURE: the ascending-vs-descending fold over `f1,f2,f3` (real algo-3
/// net, live STM targets) diverges in >=1 element with the base (unperturbed)
/// weight seed -- no seed search was needed (search bound: 1 of a budgeted
/// 20). Recorded honestly: if this ever measures 0 (e.g. after an unrelated
/// change to the net/corpus), the test fails LOUDLY here rather than the
/// golden below silently passing for the wrong reason.
#[test]
fn fold_order_measured_divergence_seed0() {
    let _lock = CWD_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let dir = tempfile::tempdir().unwrap();
    seed_fold_order(dir.path());
    let cfg = fold_order_config();
    let _cwd = CwdGuard::enter(dir.path());

    let (n_diff, n_total) = measure_fold_divergence(&cfg);
    assert!(
        n_diff > 0,
        "seed 0 (base NNweights_config1.bin): ascending vs descending fold measured 0 \
         differing elements -- the bounded search (budget 20 seeds) would need to try \
         perturbed weights, which this run did NOT do; investigate before trusting the \
         fold-order golden below"
    );
    eprintln!(
        "fold_order_measured_divergence_seed0: {n_diff} of {n_total} derivative elements \
         differ between the ascending and descending 3-file fold"
    );
}

/// GOLDEN: production `CorpusProcessor::run()` (ascending fold, the real
/// `n == 1` static-lane path) over the SAME 3-file corpus/config, one
/// training epoch, against a committed fixture -- compared via the repo's
/// canary-gated comparator (`common::assert_oracle_eq`, same family as the
/// pitch-pass golden below), NOT a strict `to_bits()` assert: the raw
/// derivatives flow through the net's ln/exp/asinh activation chain (real
/// 33,671-weight net, 67,342-element fixture), so they are NOT a
/// portable-arithmetic golden -- bit-exact on the oracle env (Apple libm),
/// hybrid ULP/absolute off it (e.g. CI glibc).
///
/// Pins `epoch_raw_derivs_trace_for_test()` (the RAW folded `[deriv, count]`
/// matrix captured immediately after `run_epoch`'s per-file fold, BEFORE
/// `save_and_update_epoch`'s Rprop update), NOT the post-Rprop trained
/// weights: iRPROP- (`nn/train.rs`) reacts only to the SIGN of the
/// (count-normalized) derivative, so the ~5865-of-67342-element bit
/// divergence this fold order produces (see
/// `fold_order_measured_divergence_seed0`) never flips a sign at this scale
/// and is therefore INVISIBLE in the trained weights -- confirmed directly:
/// pinning `epoch_weight_trace_for_test()` instead does NOT fail under a
/// `per_file.sort_by_key` ascending -> `Reverse` mutation in `run_epoch`
/// (tried first; reverted after confirming the false negative, see the task
/// report). The raw-derivs pin below DOES fail under that same mutation --
/// confirmed manually (apply/run/revert, matching the Task 11 mutation-
/// battery methodology). IMPORTANT CAVEAT: that mutation-sensitivity
/// evidence (T10/T11) was gathered ON THE ORACLE ENV ONLY, where the
/// comparator runs in `Strict` (bit-exact) mode. The fold-order divergence
/// is bit-level (~1 ULP per affected element, not a magnitude difference),
/// which sits INSIDE the hybrid ULP/absolute bound used off the oracle env
/// -- so the "this golden catches the fold-order mutation" claim is
/// oracle-env-only, not CI-verified; it was never repeated under
/// `SPEECH_ORACLE_LIBM=ulp` or on a glibc host. Off-oracle this test still
/// pins shape + hybrid-bound agreement against the fixture, just not the
/// ULP-scale mutation sensitivity the manual apply/run/revert established.
#[test]
fn fold_order_ascending_golden() {
    let _lock = CWD_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let dir = tempfile::tempdir().unwrap();
    seed_fold_order(dir.path());
    let mut cfg = fold_order_config();
    cfg.insert(
        "Neural_Networks_BackPropagation_Epochs".to_string(),
        "1".to_string(),
    );
    cfg.insert(
        "Neural_Networks_Gradient_Check_Epsilon".to_string(),
        "0.0".to_string(),
    );
    let _cwd = CwdGuard::enter(dir.path());

    let mut cp = CorpusProcessor::new(vec![cfg], multi_mode()).unwrap();
    cp.run().unwrap();

    let raw_trace = cp.epoch_raw_derivs_trace_for_test();
    // epoch 0 (solo-style log eval) + 1 inner epoch + the final eval == 3.
    assert_eq!(raw_trace.len(), 3, "epoch0 + 1 inner + final eval");
    let epoch0 = &raw_trace[0];
    assert!(
        epoch0.dim().0 > 0,
        "epoch 0 must have produced a non-empty raw derivative matrix"
    );

    let want = common::load_bin_phase4b("fold_order_ascending_epoch0_raw_derivs.bin");
    common::assert_oracle_eq(epoch0, &want, "fold_order_ascending epoch0 raw derivs");

    // NON-VACUITY: the post-Rprop trained weights also move (the fold's
    // derivatives genuinely drove a step), even though (per the doc comment
    // above) that observable is NOT what distinguishes the fold order.
    let trace = cp.epoch_weight_trace_for_test();
    let pre =
        speech::io::binary::read_weight_vector(&dir.path().join("NNweights_config1.bin")).unwrap();
    assert_eq!(pre.len(), trace[0].len());
    let differ = pre
        .iter()
        .zip(trace[0].iter())
        .any(|(a, b)| a.to_bits() != b.to_bits());
    assert!(differ, "epoch-0 update must move the weights");
}

// =============================================================================
// Item 3: pitch-pass target-reuse pin under LIVE STM references.
// =============================================================================
//
// `phase2b_spectral_golden.rs`'s pitch coverage (`pitch_both_pass_goldens`
// etc.) calls `get_segmentation(&mut audio, &mut segs, None)` -- NN-chain
// only, no reference. `BlstmSpectralSegmenter::get_segmentation`'s pitch
// second pass (`tasks/sad.rs:1548-1612`, from `BLSTMSpectralSegmenter.cpp:
// 757-805`) reuses the SAME `target` built once for pass 1 (`:1495-1509`)
// unchanged in the pass-2 `feed_forward_backward` call (`:1592`) -- the
// `:793` pass-1-target-reuse quirk. That reuse is only OBSERVABLE (as
// opposed to merely wired-but-inert on an empty target) when a live
// reference is present, which no existing golden exercises. This drives the
// SAME driver directly (matching `phase2b_spectral_golden.rs`'s pattern) on
// the phase4a 3-file corpus's `f1.wav`/`f1.stm` (real STM reference, extends
// the "4a tier-2 corpus" pattern per the task brief) with the TDC pitch
// overrides that activate the second pass, and `refs: Some(&refs)`.

use speech::io::binary::read_matrix;
use speech::tasks::sad::BlstmSpectralSegmenter;
use speech::tasks::segmentation::Segmentation;
use speech::tasks::segmentation_io::load_ref_stm;
use speech::tasks::segmenter::Segmenter;

/// The real net weights (`NNweights_config1.bin`, 33,671 f64).
fn real_weights() -> Vec<f64> {
    let (rows, cols, data) = read_matrix(&ref_dir().join("phase0/NNweights_config1.bin")).unwrap();
    assert_eq!(rows * cols, 33_671);
    data
}

/// The real spectral base (`BLSTM_shift 0.8` -> overlap) + the TDC overrides
/// that activate the pitch pass -- byte-identical values to
/// `phase2b_spectral_golden.rs::pitch_map` / the harness's
/// `setPitchOverrides` (Task 8), so the pass-1/pass-2 param derivation is the
/// SAME shape as the already-pinned NN-chain-only pitch goldens.
fn pitch_scored_map() -> IndexMap<String, String> {
    let mut m = load_config("phase0/1_worker_1.config");
    m.insert("BLSTM_shift".into(), "8.000000000000000e-01".into());
    m.insert("BLSTM_TDCwindow".into(), "0.032".into());
    m.insert("BLSTM_TDCshift".into(), "0.01".into());
    m.insert("BLSTM_TDC_lags".into(), "0.002,0.016".into());
    m.insert("BLSTM_TDC_balance".into(), "0.7".into());
    m.insert("BLSTM_TDC_windowing_type".into(), "hamming".into());
    m.insert("BLSTM_TDC_windowing_param".into(), "0.8".into());
    m
}

fn build_pitch_scored() -> BlstmSpectralSegmenter {
    BlstmSpectralSegmenter::from_legacy(&pitch_scored_map(), Some(&real_weights())).unwrap()
}

/// `f1.wav` (phase4a 3-file corpus), offset 0.0 / dur 2.0 -- matching the
/// `Audio_offset`/`Audio_max_duration` overrides the 4a tier-2 corpus uses
/// for the SAME file, so this test's audio window is the one the committed
/// `f1.stm` reference was written against.
fn f1_audio() -> speech::audio::Audio {
    speech::audio::read_audio(&phase4a_corpus_dir().join("f1.wav"), 0.0, 2.0, 0, None)
        .expect("decode f1.wav")
}

/// Per-channel reference `Segmentation`s loaded from `f1.stm` (direct port of
/// `Segmentation::load_ref_from_stm`, `Segmentation.cpp:616-635`) -- a REAL
/// committed reference file, not a synthetic hand-built span.
fn f1_refs(audio: &speech::audio::Audio) -> Vec<Segmentation> {
    let stm_text = std::fs::read_to_string(phase4a_corpus_dir().join("f1.stm")).unwrap();
    let dur = (audio.data.ncols() as f64 - 1.0) / audio.sample_rate as f64;
    (0..audio.data.nrows())
        .map(|chan| load_ref_stm(&stm_text, chan, 0.0, dur, true))
        .collect()
}

fn fresh_segs(audio: &speech::audio::Audio) -> Vec<Segmentation> {
    let dur = (audio.data.ncols() as f64 - 1.0) / audio.sample_rate as f64;
    (0..audio.data.nrows())
        .map(|_| Segmentation::new(dur))
        .collect()
}

/// The pitch pass fires (chan-0 pass-2 result exists) under a LIVE STM
/// reference, and the SAME `target` built once for pass 1 (`tasks/sad.rs
/// :1495-1509`) is reused unchanged for pass 2's FFB call (`:1592`) -- the
/// `:793` quirk, now exercised on a NON-EMPTY target for the first time
/// (every existing pitch golden, `phase2b_spectral_golden.rs`, passes
/// `refs: None`).
///
/// This golden closes the "numerically inert target" gap: a live reference
/// now genuinely drives the cost/counter accumulation end to end through
/// BOTH passes (Task 7b's target-gated `feed_forward_backward` cost block),
/// asserted below two ways -- (a) live nonzero cost/counter on THIS
/// (production) driver, and (b) the pass-2 result row bit-exact against a
/// new harness dump.
///
/// It does NOT (and structurally cannot) distinguish the `:793` reuse from
/// a hypothetical rebuild -- not via `result_vec`, and not via cost either.
/// `get_targets` (`segmenter.rs`) is a pure function of `(reference,
/// timeStep, timeOffset, backPropWer, classType, nRows)`; the legacy
/// (`BLSTMSpectralSegmenter.cpp:724-733`) computes `timeStep`/`timeOffset`
/// exactly ONCE, before the pitch block, and the pitch warp preserves the
/// periodogram's shape, so every argument a rebuild at `:793` would pass is
/// identical to pass 1's -- the rebuilt target would be bit-identical to the
/// reused one. The `:793` reuse is therefore structurally pinned by
/// transcription (a verbatim port of the legacy line), and is
/// observationally indistinguishable from a rebuild for this driver/config,
/// not merely "provable only via cost" (the version of this comment shipped
/// in `0f7ab42` overclaimed that; see IMPROVEMENTS.md's pitch-reuse UPDATE
/// addendum for the correction and the full analysis).
#[test]
fn pitch_pass_target_reuse_under_live_reference() {
    let mut sig = build_pitch_scored();
    let mut audio = f1_audio();
    let refs = f1_refs(&audio);
    let mut segs = fresh_segs(&audio);
    sig.get_segmentation(&mut audio, &mut segs, Some(&refs))
        .unwrap();

    let rows2 = sig.last_result_rows_pass2();
    assert!(
        !rows2.is_empty(),
        "pitch pass must have fired (pass-2 result rows captured) on this config/corpus"
    );

    // Live-cost non-vacuity: with a reference present, feed_forward_backward's
    // cost/counter accumulation is gated on a non-empty target (Task 7b), so
    // BOTH passes must report a live (nonzero) cumulative error/classif count.
    assert!(
        sig.cumulative_error()[0].is_finite() && sig.cumulative_error()[0] != 0.0,
        "chan-0 cumulative_error must be finite nonzero under a live reference"
    );
    assert!(
        sig.nb_of_classif()[0] > 0,
        "chan-0 nb_of_classif must be > 0 under a live reference"
    );

    // Bit-exact vs the harness's STM-driven dump (pass-2 result row, chan 1).
    let want = common::load_bin_phase4b("spectral_pitch_scored_result_pass2_chan1.bin");
    let got = Array2::from_shape_vec((1, rows2[0].len()), rows2[0].clone()).unwrap();
    assert_eq!(got.shape(), want.shape(), "pass-2 result shape");
    for (i, (&g, &w)) in got.row(0).iter().zip(want.row(0).iter()).enumerate() {
        common::assert_oracle_eq_f64(g, w, &format!("pitch_scored pass-2 result[{i}]"));
    }
}
