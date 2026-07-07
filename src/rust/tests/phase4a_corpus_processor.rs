//! Phase 4a Task 7: `CorpusProcessor` (`CorpusProcessor.cpp:48-404`) - mode
//! dispatch, the static-lane epoch loop, `transformResults`, `saveResults`, and
//! the corpus-level `gradCheck`.
//!
//! The static lane model (spec S3, the R6 determinism fix) is the ONE deliberate
//! deviation from legacy OpenMP parallelism: file `j` -> lane `j % N`; each lane
//! clones the epoch-start bag once and walks its files in ascending `j` (state
//! chains within a lane); contributions folded in ASCENDING file order after the
//! parallel section. N=1 must be byte-identical to a plain sequential loop -- that
//! is the golden-pinned parity mode (`lanes_n1_equals_sequential`).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use indexmap::IndexMap;

use speech::cli::{Mode, ModeKind};
use speech::engine::corpus_processor::CorpusProcessor;

/// `set_current_dir` is process-global; the artifact-producing runs chdir into a
/// tempdir (`save_and_update`/`save_results` write to relative filenames). Serialize
/// against `cargo test`'s parallel threads.
static CWD_LOCK: Mutex<()> = Mutex::new(());

struct CwdGuard {
    original: PathBuf,
}

impl CwdGuard {
    fn enter(dir: &Path) -> CwdGuard {
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

fn load_config(rel: &str) -> IndexMap<String, String> {
    let text = std::fs::read_to_string(ref_dir().join(rel)).unwrap();
    speech::legacy_config::parse_legacy_config(&text)
}

fn mode(kind: ModeKind) -> Mode {
    Mode {
        kind,
        verbose: false,
    }
}

/// Copy the shared 2-channel excerpt into `dir` under `name`.
fn wav_in(dir: &Path, name: &str) -> PathBuf {
    let src = ref_dir().join("phase1/excerpt_2ch_8k.wav");
    let dst = dir.join(name);
    std::fs::copy(&src, &dst).unwrap();
    dst
}

/// A 2-channel STM with one SPEECH span per channel, in-window for the 2s excerpt
/// (same layout as `phase4a_segfn.rs`).
fn write_stm(dir: &Path, name: &str) -> PathBuf {
    let p = dir.join(name);
    let body = "\
excerpt 1 excerpt 0.20 0.80 <o,f0,unk> hello world
excerpt 2 excerpt 0.30 0.90 <o,f0,unk> hello world
";
    std::fs::write(&p, body).unwrap();
    p
}

/// Write a `language2classmapping.csv` mapping `unk;unk;0` and return its path.
fn write_mapping(dir: &Path) -> PathBuf {
    let p = dir.join("mapping.csv");
    std::fs::write(&p, "unk;unk;0\n").unwrap();
    p
}

/// Build a TDC (Algo 1) config with the top-level bag keys + corpus keys pointing
/// at a `fileslisting`/`language2classmapping` pair in `dir` covering `n` copies of
/// the excerpt (each with its own STM). Returns the config map.
fn tdc_corpus_config(dir: &Path, n: usize, epochs: i32) -> IndexMap<String, String> {
    let mapping = write_mapping(dir);
    let mut listing_lines = String::new();
    for i in 0..n {
        let wav = wav_in(dir, &format!("f{i}.wav"));
        let stm = write_stm(dir, &format!("f{i}.stm"));
        listing_lines.push_str(&format!(
            "{};{};unk;unk;1.0;1\n",
            wav.to_str().unwrap(),
            stm.to_str().unwrap()
        ));
    }
    let listing = dir.join("listing.csv");
    std::fs::write(&listing, listing_lines).unwrap();

    let mut m = load_config("phase2b/tdc.config");
    m.insert("numOuterThreads".to_string(), "1".to_string());
    m.insert("Algo_choice".to_string(), "1".to_string());
    m.insert("TDC_lags".to_string(), "-0.002,0.016".to_string());
    m.insert("Audio_offset".to_string(), "0.0".to_string());
    m.insert("Audio_max_duration".to_string(), "2.0".to_string());
    m.insert(
        "language2classmapping".to_string(),
        mapping.to_str().unwrap().to_string(),
    );
    m.insert(
        "fileslisting".to_string(),
        listing.to_str().unwrap().to_string(),
    );
    m.insert(
        "Neural_Networks_BackPropagation_Epochs".to_string(),
        epochs.to_string(),
    );
    m.insert(
        "multiConfigResultsOutputFile".to_string(),
        "MultiConfigResults.mat".to_string(),
    );
    m
}

// === mode_dispatch_matrix ====================================================
// epsilon/epochs/mode-letter combinations select which private path runs. Solo
// mode never trains (mode-letter gate); a scored mode with epochs>0 trains. We
// probe the OBSERVABLE outcome: CostMem shape written to the .mat (solo -> 1 row;
// train with N epochs -> N+2 rows). A 1-file TDC corpus in a tempdir.
#[test]
fn mode_dispatch_matrix() {
    let _lock = CWD_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    // -- Solo (-s), epochs 0: runSolo -> CostMem is 1x1. --
    {
        let dir = tempfile::tempdir().unwrap();
        let cfg = tdc_corpus_config(dir.path(), 1, 0);
        let _cwd = CwdGuard::enter(dir.path());
        let mut cp = CorpusProcessor::new(vec![cfg], mode(ModeKind::Solo)).unwrap();
        cp.run().unwrap();
        let bytes = std::fs::read("MultiConfigResults.mat").unwrap();
        let (r, c) = mat_var_dims(&bytes, "CostMem").unwrap();
        assert_eq!((r, c), (1, 1), "solo CostMem 1x1");
    }

    // -- Multi (-m), epochs 2: train -> CostMem is (epochs+2)x1 = 4x1. --
    {
        let dir = tempfile::tempdir().unwrap();
        let cfg = tdc_corpus_config(dir.path(), 1, 2);
        let _cwd = CwdGuard::enter(dir.path());
        let mut cp = CorpusProcessor::new(vec![cfg], mode(ModeKind::Multi)).unwrap();
        cp.run().unwrap();
        let bytes = std::fs::read("MultiConfigResults.mat").unwrap();
        let (r, c) = mat_var_dims(&bytes, "CostMem").unwrap();
        // train() writes topRows(epoch+1); the FINAL epoch is _TrainingEpochs+1
        // == 3, so the last saveResults writes topRows(4) -> the full 4-row mem.
        assert_eq!((r, c), (4, 1), "train CostMem (epochs+2)x1");
    }

    // -- Solo mode with epochs>0: the mode-letter gate blocks training (legacy
    //    cerr + continue), so it falls through to NO run at all (train not called,
    //    solo not called -- run() only calls runSolo in the else branch when
    //    epochs==0). Observe: no .mat is written (no run happened). --
    {
        let dir = tempfile::tempdir().unwrap();
        let cfg = tdc_corpus_config(dir.path(), 1, 2);
        let _cwd = CwdGuard::enter(dir.path());
        let mut cp = CorpusProcessor::new(vec![cfg], mode(ModeKind::Solo)).unwrap();
        cp.run().unwrap();
        // The ctor truncates the output file to a single space ONLY for m/M/t/T;
        // solo mode does not, so no file exists (train was gated out, runSolo not
        // reached because epochs>0).
        assert!(
            !Path::new("MultiConfigResults.mat").exists(),
            "solo + epochs>0 -> mode gate blocks train, no run, no .mat"
        );
    }
}

// === transform_results_ordering ==============================================
// A crafted results map with a GAP file index (0 and 2, no 1) -> ascending BTreeMap
// iteration puts file 0's rows before file 2's; the ResultsE row layout is
// [file+1, conf+1, chan+1, res...]; conservative resize truncates the pre-sized
// 2*nb_files rows to the actual counter.
#[test]
fn transform_results_ordering() {
    // Two "files" (indices 0 and 2), one conf, one channel each. Crafted result
    // rows so the ordering + id columns are directly checkable.
    let mut results: BTreeMap<usize, BTreeMap<usize, BTreeMap<usize, Vec<f64>>>> = BTreeMap::new();
    let mut r0: BTreeMap<usize, BTreeMap<usize, Vec<f64>>> = BTreeMap::new();
    r0.insert(0, BTreeMap::from([(0, vec![11.0, 22.0])]));
    let mut r2: BTreeMap<usize, BTreeMap<usize, Vec<f64>>> = BTreeMap::new();
    r2.insert(0, BTreeMap::from([(0, vec![33.0, 44.0])]));
    // Insert file 2 FIRST to prove the BTreeMap re-sorts ascending.
    results.insert(2, r2);
    results.insert(0, r0);

    // nb_of_files is 2 (the corpus size), nb_of_conf 1.
    let (results_e, res_per_conf) = CorpusProcessor::transform_results_for_test(&results, 1, 2);

    // ResultsE: 2 rows, 3 id cols + 2 result cols = 5 cols.
    assert_eq!(results_e.dim(), (2, 5), "conservative resize to counter");
    // Row 0 is file 0 (ascending): [file+1=1, conf+1=1, chan+1=1, 11, 22].
    assert_eq!(results_e.row(0).to_vec(), vec![1.0, 1.0, 1.0, 11.0, 22.0]);
    // Row 1 is file 2: [file+1=3, conf+1=1, chan+1=1, 33, 44].
    assert_eq!(results_e.row(1).to_vec(), vec![3.0, 1.0, 1.0, 33.0, 44.0]);

    // Per-conf matrix (conf 0): the raw result rows, ascending file order, truncated
    // to the counter (2 rows, 2 cols).
    assert_eq!(res_per_conf.len(), 1);
    assert_eq!(res_per_conf[0].dim(), (2, 2));
    assert_eq!(res_per_conf[0].row(0).to_vec(), vec![11.0, 22.0]);
    assert_eq!(res_per_conf[0].row(1).to_vec(), vec![33.0, 44.0]);
}

// === lanes_n1_equals_sequential ==============================================
// TDC 3-file corpus: run_epoch with n=1 must equal a hand-sequential loop over the
// same corpus, STRICT bits -- EXCEPT col 3 (wall-clock timing) which is masked.
#[test]
fn lanes_n1_equals_sequential() {
    let _lock = CWD_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let dir = tempfile::tempdir().unwrap();
    let cfg = tdc_corpus_config(dir.path(), 3, 0);
    let _cwd = CwdGuard::enter(dir.path());

    // The engine path: solo run over the 3-file corpus (n = nb_of_threads clamped
    // to 1). Capture ResultsE.
    let mut cp = CorpusProcessor::new(vec![cfg.clone()], mode(ModeKind::Solo)).unwrap();
    cp.run().unwrap();
    let engine_bytes = std::fs::read("MultiConfigResults.mat").unwrap();
    let engine_results = mat_var_matrix(&engine_bytes, "MultiConfigResults").unwrap();

    // The hand-sequential oracle: for each file j in ascending order, run
    // segmentation_function directly on a fresh bag (n=1 chains state within the one
    // lane; here each file is independent for TDC, so a fresh bag per file is
    // equivalent -- TDC has no cross-file NN state). Assemble the same ResultsE row
    // layout [file+1, conf+1, chan+1, res...] and compare (col 3 = ResultsE col 6
    // masked).
    let seq_results =
        CorpusProcessor::run_epoch_sequential_oracle(vec![cfg], mode(ModeKind::Solo)).unwrap();

    assert_eq!(
        engine_results.dim(),
        seq_results.dim(),
        "ResultsE shape must match the sequential oracle"
    );
    let (rows, cols) = engine_results.dim();
    // ResultsE col 6 == result col 3 (timing) is masked; compare all others bit-exact.
    for r in 0..rows {
        for c in 0..cols {
            if c == 6 {
                continue; // timing column (result col 3), wall-clock -> masked.
            }
            assert_eq!(
                engine_results[[r, c]].to_bits(),
                seq_results[[r, c]].to_bits(),
                "ResultsE[{r},{c}] engine != sequential oracle (bit-exact, col 6 masked)"
            );
        }
    }
    // Non-vacuity: 3 files x 2 channels = 6 rows.
    assert_eq!(rows, 6, "3 files x 2 channels");
}

// === save_results_variables ==================================================
// The 5 named matrices are written in order (MultiConfigResults, CostMem,
// BadClassifMem, CostLIDMem, BadClassifLIDMem), each topRows(epoch+1). Read back
// the .mat bytes with the local parser helpers.
#[test]
fn save_results_variables() {
    let _lock = CWD_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let dir = tempfile::tempdir().unwrap();
    let cfg = tdc_corpus_config(dir.path(), 1, 0);
    let _cwd = CwdGuard::enter(dir.path());
    let mut cp = CorpusProcessor::new(vec![cfg], mode(ModeKind::Solo)).unwrap();
    cp.run().unwrap();

    let bytes = std::fs::read("MultiConfigResults.mat").unwrap();
    let order = mat_var_order(&bytes);
    assert_eq!(
        order,
        vec![
            "MultiConfigResults",
            "CostMem",
            "BadClassifMem",
            "CostLIDMem",
            "BadClassifLIDMem"
        ],
        "5 named variables in the exact legacy order"
    );
    // Solo run: each mem matrix is topRows(0+1) = 1 row, 1 conf.
    for name in ["CostMem", "BadClassifMem", "CostLIDMem", "BadClassifLIDMem"] {
        assert_eq!(mat_var_dims(&bytes, name).unwrap(), (1, 1), "{name} 1x1");
    }
    // MultiConfigResults: 1 file x 2 channels = 2 rows, 3 id + 18 result = 21 cols.
    assert_eq!(
        mat_var_dims(&bytes, "MultiConfigResults").unwrap(),
        (2, 21),
        "MultiConfigResults file x chan rows, 21 cols"
    );
}

// === grad_check_synthetic ====================================================
// The corpus-level gradCheck (`CorpusProcessor.cpp:237-340`): a synthetic signal
// (Algo 4) net on a 1-file corpus WITH an STM reference, backprop active. Task 7b
// wired the reference-driven target into the NN drivers, so `feed_forward_backward`
// now accumulates a LIVE cost/counter and the backward derivs are non-degenerate --
// the analytic gradient and the numerical central difference agree to the Task 7
// bound (mean relative error < 5e-4). Also validates the bag-snapshot / per-weight
// perturb / restore machinery and the `GradCheckReport` shape.
#[test]
fn grad_check_synthetic() {
    let _lock = CWD_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let dir = tempfile::tempdir().unwrap();
    let cfg = synthetic_signal_corpus_config(dir.path());
    let _cwd = CwdGuard::enter(dir.path());

    let mut cp = CorpusProcessor::new(vec![cfg], mode(ModeKind::UnitTest)).unwrap();

    // Seed deterministic NONZERO weights: the default-initialized net sits at a
    // degenerate operating point (near-zero gradients everywhere), so the gradcheck
    // would pass trivially on an all-zero gradient. The phase3 `synth_flat` pattern
    // (`((k*11+3)%97)/97 - 0.5`) gives a smoothly-varying cost.
    let nb = cp.get_config0_weights_for_test().len();
    let synth: Vec<f64> = (0..nb)
        .map(|k| ((k * 11 + 3) % 97) as f64 / 97.0 - 0.5)
        .collect();
    cp.set_config0_weights_for_test(&synth).unwrap();

    // Snapshot the pre-check weights (the sweep must leave them restored).
    let weights_before = cp.get_config0_weights_for_test();

    // Grad check only the first 10 weights (a full 137-weight sweep is O(N) corpus
    // runs; the brief caps it). eps = 1e-5 (the legacy default).
    let report = cp.grad_check_for_test(1e-5, 10).unwrap();

    // Orchestration mechanics: exactly 10 per-weight triples produced by the sweep.
    assert_eq!(report.per_weight.len(), 10, "10 weights checked");
    // The cost path is now LIVE (Task 7b targets): the analytic normalized deriv and
    // the numerical central difference are finite and agree. Assert the Task 7 bound.
    assert!(
        report.mean_relative_error.is_finite(),
        "grad check mean rel error must be finite (live cost path), got {}",
        report.mean_relative_error
    );
    assert!(
        report.mean_relative_error < 5e-4,
        "analytic vs numeric gradient must agree to < 5e-4, got {}",
        report.mean_relative_error
    );
    // Non-vacuity: at least one checked weight has a genuinely nonzero numerical
    // derivative (the check is not passing trivially on an all-zero gradient).
    assert!(
        report
            .per_weight
            .iter()
            .any(|(_, numerical, _)| numerical.abs() > 1e-9),
        "grad check must exercise at least one nonzero numerical derivative"
    );
    // Weight restore: after the sweep the bag is restored to its snapshot, so the
    // config-0 weights are bit-identical to before the check.
    let weights_after = cp.get_config0_weights_for_test();
    assert_eq!(weights_before.len(), weights_after.len());
    for (i, (a, b)) in weights_before.iter().zip(weights_after.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "weight {i} not restored after grad check"
        );
    }
}

/// A synthetic signal (Algo 4) config: the phase2b signal.config net overridden to
/// a small `[2,2]` LSTM + `[4,1]` output net (a 1-output Logistic head compatible
/// with the signal driver's Nx1 result buffer -- the phase3 gradcheck's two-output
/// `[4,2]` net is incompatible here, see the config body), backprop active,
/// InputNormalizationType 0, TargetEnforcementStep 0, on a 1-file corpus. The net
/// default-inits its weights; the gradcheck perturbs from there (the orchestration
/// mechanics do not depend on the specific weights).
fn synthetic_signal_corpus_config(dir: &Path) -> IndexMap<String, String> {
    let mapping = write_mapping(dir);
    let wav = wav_in(dir, "f0.wav");
    let stm = write_stm(dir, "f0.stm");
    let listing = dir.join("listing.csv");
    std::fs::write(
        &listing,
        format!(
            "{};{};unk;unk;1.0;1\n",
            wav.to_str().unwrap(),
            stm.to_str().unwrap()
        ),
    )
    .unwrap();

    let mut m = load_config("phase2b/signal.config");
    m.insert("numOuterThreads".to_string(), "1".to_string());
    m.insert("Algo_choice".to_string(), "4".to_string());
    m.insert("Audio_offset".to_string(), "0.0".to_string());
    m.insert("Audio_max_duration".to_string(), "2.0".to_string());
    // A small synthetic net compatible with the SIGNAL driver's constraints: the
    // driver feeds an Nx1 signal column and sizes the result buffer Nx1, so the net
    // must be SINGLE-output (the phase3 gradcheck's [4,2] two-output net would need
    // an Nx2 buffer -- the driver never sizes one, so it is incompatible here). A
    // [2,2] LSTM + [4,1] output net (a 1-output Logistic head). Backprop active,
    // InputNormalizationType 0, TargetEnforcementStep 0.
    m.insert("BLSTM_LSTMNeuronNb".to_string(), "2,2".to_string());
    m.insert("BLSTM_LSTMSubSampling".to_string(), "1".to_string());
    m.insert("BLSTM_OutputNeuronNb".to_string(), "4,1".to_string());
    m.insert("BLSTM_OutputSubSampling".to_string(), "1".to_string());
    m.insert("BLSTM_NNetInputSize".to_string(), "2".to_string());
    m.insert("BLSTM_InputNormalizationType".to_string(), "0".to_string());
    m.insert("BLSTM_TargetEnforcementStep".to_string(), "0".to_string());
    m.insert(
        "BLSTM_BackPropagationActivated".to_string(),
        "true".to_string(),
    );
    m.insert("BLSTM_BackPropWER".to_string(), "-1.0".to_string());
    m.insert("BLSTM_window".to_string(), "0.0".to_string());
    m.insert("BLSTM_shift".to_string(), "0.1".to_string());
    m.insert(
        "language2classmapping".to_string(),
        mapping.to_str().unwrap().to_string(),
    );
    m.insert(
        "fileslisting".to_string(),
        listing.to_str().unwrap().to_string(),
    );
    m.insert(
        "Neural_Networks_BackPropagation_Epochs".to_string(),
        "0".to_string(),
    );
    m.insert(
        "Neural_Networks_Gradient_Check_Epsilon".to_string(),
        "1e-5".to_string(),
    );
    m.insert(
        "multiConfigResultsOutputFile".to_string(),
        "MultiConfigResults.mat".to_string(),
    );
    m
}

// ============================================================================
// Minimal MAT v5 reader helpers (the writer has no reader; spec S6). Uncompressed
// v5 only: 128-byte header then a sequence of miMATRIX elements. Enough to pull a
// variable's dims / data / name-order for the assertions above.
// ============================================================================

fn u32_le(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes(b[off..off + 4].try_into().unwrap())
}

fn i32_le(b: &[u8], off: usize) -> i32 {
    i32::from_le_bytes(b[off..off + 4].try_into().unwrap())
}

fn f64_le(b: &[u8], off: usize) -> f64 {
    f64::from_le_bytes(b[off..off + 8].try_into().unwrap())
}

fn pad8(n: usize) -> usize {
    n.div_ceil(8) * 8
}

/// Walk the miMATRIX elements, returning `(name, rows, cols, col_major_data)` per
/// variable in file order.
fn mat_vars(bytes: &[u8]) -> Vec<(String, usize, usize, Vec<f64>)> {
    let mut out = Vec::new();
    let mut off = 128;
    while off + 8 <= bytes.len() {
        let tag = u32_le(bytes, off);
        let size = u32_le(bytes, off + 4) as usize;
        if tag != 14 {
            break;
        }
        let mut p = off + 8;
        // array flags sub-element (miUINT32, 8 payload): skip tag+size+8.
        p += 8 + 8;
        // dims sub-element (miINT32): tag+size then two i32.
        let dims_size = u32_le(bytes, p + 4) as usize;
        let rows = i32_le(bytes, p + 8) as usize;
        let cols = i32_le(bytes, p + 12) as usize;
        p += 8 + pad8(dims_size);
        // name sub-element (miINT8, long form): tag+size then the padded body.
        let name_size = u32_le(bytes, p + 4) as usize;
        let name = String::from_utf8(bytes[p + 8..p + 8 + name_size].to_vec()).unwrap();
        p += 8 + pad8(name_size);
        // data sub-element (miDOUBLE): tag+size then the doubles.
        let data_size = u32_le(bytes, p + 4) as usize;
        let n = data_size / 8;
        let mut data = Vec::with_capacity(n);
        for k in 0..n {
            data.push(f64_le(bytes, p + 8 + k * 8));
        }
        out.push((name, rows, cols, data));
        off += 8 + size;
    }
    out
}

fn mat_var_order(bytes: &[u8]) -> Vec<String> {
    mat_vars(bytes).into_iter().map(|(n, _, _, _)| n).collect()
}

fn mat_var_dims(bytes: &[u8], name: &str) -> Option<(usize, usize)> {
    mat_vars(bytes)
        .into_iter()
        .find(|(n, _, _, _)| n == name)
        .map(|(_, r, c, _)| (r, c))
}

fn mat_var_matrix(bytes: &[u8], name: &str) -> Option<ndarray::Array2<f64>> {
    mat_vars(bytes)
        .into_iter()
        .find(|(n, _, _, _)| n == name)
        .map(|(_, r, c, data)| {
            // The writer stored column-major; rebuild row-major Array2.
            ndarray::Array2::from_shape_fn((r, c), |(i, j)| data[j * r + i])
        })
}
