//! Phase 4a Task 8: tier-1 corpus-level e2e goldens against the REAL compiled
//! CorpusProcessor stack (`CorpusProcessor.cpp:48-404`).
//!
//! The oracle harness `phase4a_tier1` stage runs `CorpusProcessor(configs,
//! mode).run()` on the committed 3-file 2-channel corpus at a FIXED path and dumps
//! the raw `.mat` outputs (converted to `.bin` by the extractor, timing column
//! masked) + the multi-channel VRCTS xml. These tests replay the SAME committed
//! corpora/configs through the Rust engine and byte/oracle-compare.
//!
//! Comparators: TDC/LTSV chains pass through libm (fmath::log/cos, mel/log), so the
//! result matrices use the canary-gated `assert_oracle_eq` family EXCEPT structurally
//! integer values (dims, counts, id columns) which are strict. The MultiConfigResults
//! timing column (col 6) is masked (zeroed in the fixture, skipped here). VRCTS bytes
//! compare via `assert_vrcts_eq` (byte-exact on the oracle env, value-level re-parse
//! elsewhere). `listing_parse_vs_fixture` is STRICT (pure integer/float parse).

mod common;

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use indexmap::IndexMap;
use ndarray::Array2;

use common::{assert_oracle_eq_f64, assert_vrcts_eq, fixture_phase4a, load_bin_phase4a};
use speech::cli::{Mode, ModeKind};
use speech::engine::corpus::Corpus;
use speech::engine::corpus_processor::CorpusProcessor;

/// `set_current_dir` is process-global; every run chdir's into a seeded tempdir
/// (relative corpus paths + relative output filenames). Serialize against `cargo
/// test`'s parallel threads.
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

fn phase4a_src() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/reference_data/phase4a")
}

/// Seed `dir` with a byte-copy of the committed tier-1 corpus + configs, so a chdir'd
/// run resolves the relative `corpus/fN.wav`/`fileslisting.csv` paths exactly as the
/// harness did. Returns nothing; the caller chdir's into `dir`.
fn seed_corpus(dir: &Path) {
    let src = phase4a_src();
    let corpus_dst = dir.join("corpus");
    std::fs::create_dir_all(&corpus_dst).unwrap();
    for f in ["f1", "f2", "f3"] {
        std::fs::copy(
            src.join("corpus").join(format!("{f}.wav")),
            corpus_dst.join(format!("{f}.wav")),
        )
        .unwrap();
        std::fs::copy(
            src.join("corpus").join(format!("{f}.stm")),
            corpus_dst.join(format!("{f}.stm")),
        )
        .unwrap();
    }
    for f in [
        "fileslisting.csv",
        "language2classmapping.csv",
        "tier1_tdc.config",
        "tier1_ltsv_powermel.config",
    ] {
        std::fs::copy(src.join(f), dir.join(f)).unwrap();
    }
}

fn load_config(name: &str) -> IndexMap<String, String> {
    let text = std::fs::read_to_string(phase4a_src().join(name)).unwrap();
    speech::legacy_config::parse_legacy_config(&text)
}

fn mode(kind: ModeKind) -> Mode {
    Mode {
        kind,
        verbose: false,
    }
}

/// Read a variable's matrix back out of a MAT v5 file the engine wrote (the writer
/// has no reader). Row-major `Array2`.
fn read_mat_var(bytes: &[u8], name: &str) -> Array2<f64> {
    mat_var_matrix(bytes, name).unwrap_or_else(|| panic!("variable {name} not in .mat"))
}

/// Compare two matrices structurally: dims strict, then each element via
/// `assert_oracle_eq_f64` EXCEPT columns listed in `strict_cols` (id/count columns,
/// bit-exact) and `masked_cols` (skipped entirely).
fn assert_matrix(
    got: &Array2<f64>,
    want: &Array2<f64>,
    strict_cols: &[usize],
    masked_cols: &[usize],
    label: &str,
) {
    assert_eq!(got.dim(), want.dim(), "{label}: shape mismatch");
    let (rows, cols) = got.dim();
    for r in 0..rows {
        for c in 0..cols {
            if masked_cols.contains(&c) {
                continue;
            }
            let g = got[[r, c]];
            let w = want[[r, c]];
            if strict_cols.contains(&c) {
                assert_eq!(
                    g.to_bits(),
                    w.to_bits(),
                    "{label}[{r},{c}] strict: got {g} want {w}"
                );
            } else {
                assert_oracle_eq_f64(g, w, &format!("{label}[{r},{c}]"));
            }
        }
    }
}

// === solo_tdc_matches ========================================================
// CorpusProcessor::new(tier1_tdc.config with epochs->0, Mode::Solo).run(); read back
// our .mat and compare MultiConfigResults (mask col 6), CostMem/BadClassifMem vs the
// converted fixtures. TDC is libm-bearing -> canary-gated compare; the id columns
// (file/conf/chan = cols 0/1/2) are strict.
#[test]
fn solo_tdc_matches() {
    let _lock = CWD_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let dir = tempfile::tempdir().unwrap();
    seed_corpus(dir.path());
    let _cwd = CwdGuard::enter(dir.path());

    let mut cfg = load_config("tier1_tdc.config");
    cfg.insert(
        "Neural_Networks_BackPropagation_Epochs".to_string(),
        "0".to_string(),
    );
    // Match the harness set_val: solo writes solo_tdc.mat, dump dir vrcts_solo.
    cfg.insert(
        "multiConfigResultsOutputFile".to_string(),
        "solo_tdc.mat".to_string(),
    );
    cfg.insert("Dump_Directory".to_string(), "vrcts_solo".to_string());
    std::fs::create_dir_all("vrcts_solo").unwrap();

    let mut cp = CorpusProcessor::new(vec![cfg], mode(ModeKind::Solo)).unwrap();
    cp.run().unwrap();

    let bytes = std::fs::read("solo_tdc.mat").unwrap();
    let got_mcr = read_mat_var(&bytes, "MultiConfigResults");
    let want_mcr = load_bin_phase4a("solo_tdc_MultiConfigResults.bin");
    // cols 0/1/2 = file/conf/chan ids (strict); col 6 = timing (masked).
    assert_matrix(
        &got_mcr,
        &want_mcr,
        &[0, 1, 2],
        &[6],
        "solo MultiConfigResults",
    );

    for var in ["CostMem", "BadClassifMem", "CostLIDMem", "BadClassifLIDMem"] {
        let got = read_mat_var(&bytes, var);
        let want = load_bin_phase4a(&format!("solo_tdc_{var}.bin"));
        assert_matrix(&got, &want, &[], &[], &format!("solo {var}"));
    }

    // Non-vacuity: 3 files x 2 channels = 6 rows.
    assert_eq!(got_mcr.dim(), (6, 21), "solo MCR 6x21");
}

// === train_ltsv_two_epochs_matches ===========================================
// epochs=2 -> train() writes topRows(epoch+1) each epoch; the final epoch is 3 so the
// last saveResults writes topRows(4). CostMem/BadClassifMem are (epochs+2)x1 = 4x1;
// all rows compared. Non-vacuity: LTSV has NO weight updates, so the mem rows are
// EXPECTED IDENTICAL across epochs -- asserted explicitly below (tier-1 pins the
// loop/aggregation plumbing; weight evolution is tier-2/Task 9).
#[test]
fn train_ltsv_two_epochs_matches() {
    let _lock = CWD_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let dir = tempfile::tempdir().unwrap();
    seed_corpus(dir.path());
    let _cwd = CwdGuard::enter(dir.path());

    let mut cfg = load_config("tier1_ltsv_powermel.config");
    // Committed config already has epochs 2; harness kept it.
    cfg.insert(
        "multiConfigResultsOutputFile".to_string(),
        "train_ltsv.mat".to_string(),
    );
    cfg.insert("Dump_Directory".to_string(), "vrcts_train".to_string());
    std::fs::create_dir_all("vrcts_train").unwrap();

    let mut cp = CorpusProcessor::new(vec![cfg], mode(ModeKind::Multi)).unwrap();
    cp.run().unwrap();

    let bytes = std::fs::read("train_ltsv.mat").unwrap();

    let got_mcr = read_mat_var(&bytes, "MultiConfigResults");
    let want_mcr = load_bin_phase4a("train_ltsv_MultiConfigResults.bin");
    assert_matrix(
        &got_mcr,
        &want_mcr,
        &[0, 1, 2],
        &[6],
        "train MultiConfigResults",
    );

    let got_cost = read_mat_var(&bytes, "CostMem");
    let want_cost = load_bin_phase4a("train_ltsv_CostMem.bin");
    assert_eq!(got_cost.dim(), (4, 1), "train CostMem (epochs+2)x1 = 4x1");
    assert_matrix(&got_cost, &want_cost, &[], &[], "train CostMem");

    let got_bad = read_mat_var(&bytes, "BadClassifMem");
    let want_bad = load_bin_phase4a("train_ltsv_BadClassifMem.bin");
    assert_matrix(&got_bad, &want_bad, &[], &[], "train BadClassifMem");

    for var in ["CostLIDMem", "BadClassifLIDMem"] {
        let got = read_mat_var(&bytes, var);
        let want = load_bin_phase4a(&format!("train_ltsv_{var}.bin"));
        assert_matrix(&got, &want, &[], &[], &format!("train {var}"));
    }

    // Non-vacuity (Phase 3 standard): LTSV (Algo 2) is NN-free -- no weight updates
    // between epochs -- so every epoch re-evaluates the SAME hypothesis and the mem
    // rows are constant-by-construction. Assert that identity EXPLICITLY: tier-1 pins
    // the epoch loop + reduction plumbing, NOT weight evolution (that is tier-2's job,
    // Task 9). A regression that (wrongly) evolved weights, or dropped an epoch's
    // save, would break this.
    let bad0 = got_bad[[0, 0]];
    for e in 1..4 {
        assert_eq!(
            got_bad[[e, 0]].to_bits(),
            bad0.to_bits(),
            "BadClassifMem epoch {e} must equal epoch 0 (LTSV has no weight updates)"
        );
    }
    // And the mem is genuinely non-trivial (not all-zero), so the identity is a real
    // pin, not a vacuous 0==0.
    assert!(
        bad0 != 0.0,
        "train BadClassifMem must be nonzero (scored LTSV), got {bad0}"
    );
}

// === multiconfig_matches =====================================================
// Both configs (-m, epochs 0 on configs[0]): MultiConfigResults interleaves
// (file, conf, chan) = 3 files x 2 confs x 2 channels = 12 rows, ordered ascending
// by file then conf then chan (transformResults BTreeMap iteration). The id columns
// pin the interleaving; the result columns oracle-compare.
#[test]
fn multiconfig_matches() {
    let _lock = CWD_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let dir = tempfile::tempdir().unwrap();
    seed_corpus(dir.path());
    let _cwd = CwdGuard::enter(dir.path());

    let mut cfg_tdc = load_config("tier1_tdc.config");
    cfg_tdc.insert(
        "Neural_Networks_BackPropagation_Epochs".to_string(),
        "0".to_string(),
    );
    cfg_tdc.insert(
        "multiConfigResultsOutputFile".to_string(),
        "multiconfig.mat".to_string(),
    );
    cfg_tdc.insert("Dump_Directory".to_string(), "vrcts_multi_tdc".to_string());
    std::fs::create_dir_all("vrcts_multi_tdc").unwrap();

    let mut cfg_ltsv = load_config("tier1_ltsv_powermel.config");
    cfg_ltsv.insert(
        "Neural_Networks_BackPropagation_Epochs".to_string(),
        "0".to_string(),
    );
    cfg_ltsv.insert("Dump_Directory".to_string(), "vrcts_multi_ltsv".to_string());
    std::fs::create_dir_all("vrcts_multi_ltsv").unwrap();

    let mut cp = CorpusProcessor::new(vec![cfg_tdc, cfg_ltsv], mode(ModeKind::Multi)).unwrap();
    cp.run().unwrap();

    let bytes = std::fs::read("multiconfig.mat").unwrap();
    let got_mcr = read_mat_var(&bytes, "MultiConfigResults");
    let want_mcr = load_bin_phase4a("multiconfig_MultiConfigResults.bin");
    assert_eq!(got_mcr.dim(), (12, 21), "3 files x 2 confs x 2 channels");
    assert_matrix(
        &got_mcr,
        &want_mcr,
        &[0, 1, 2],
        &[6],
        "multiconfig MultiConfigResults",
    );

    // The (file, conf, chan) interleaving is the load-bearing structure: pin the id
    // triple of every row explicitly (ascending file, then conf, then chan).
    let mut expected_ids: Vec<(f64, f64, f64)> = Vec::new();
    for file in 1..=3 {
        for conf in 1..=2 {
            for chan in 1..=2 {
                expected_ids.push((file as f64, conf as f64, chan as f64));
            }
        }
    }
    for (r, (f, c, ch)) in expected_ids.iter().enumerate() {
        assert_eq!(
            (got_mcr[[r, 0]], got_mcr[[r, 1]], got_mcr[[r, 2]]),
            (*f, *c, *ch),
            "multiconfig row {r} id triple (file,conf,chan) interleaving"
        );
    }

    for var in ["CostMem", "BadClassifMem", "CostLIDMem", "BadClassifLIDMem"] {
        let got = read_mat_var(&bytes, var);
        let want = load_bin_phase4a(&format!("multiconfig_{var}.bin"));
        // 1 epoch x 2 confs.
        assert_eq!(got.dim(), (1, 2), "multiconfig {var} 1x2");
        assert_matrix(&got, &want, &[], &[], &format!("multiconfig {var}"));
    }
}

// === vrcts_byte_equal ========================================================
// The Rust write_vrcts_multichannel output (via the solo CorpusProcessor run, unscored
// branch, dump dir set) must byte-match the harness's real toFile_VRCTS dumps for both
// channels of all three files. The FIXED relative corpus path (corpus/fN.wav) makes the
// AudioDoc path= attr reproducible.
#[test]
fn vrcts_byte_equal() {
    let _lock = CWD_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let dir = tempfile::tempdir().unwrap();
    seed_corpus(dir.path());
    let _cwd = CwdGuard::enter(dir.path());

    let mut cfg = load_config("tier1_tdc.config");
    cfg.insert(
        "Neural_Networks_BackPropagation_Epochs".to_string(),
        "0".to_string(),
    );
    cfg.insert(
        "multiConfigResultsOutputFile".to_string(),
        "solo_tdc.mat".to_string(),
    );
    cfg.insert("Dump_Directory".to_string(), "vrcts_solo".to_string());
    std::fs::create_dir_all("vrcts_solo").unwrap();

    let mut cp = CorpusProcessor::new(vec![cfg], mode(ModeKind::Solo)).unwrap();
    cp.run().unwrap();

    for file in ["f1", "f2", "f3"] {
        for chan in [1, 2] {
            let name = format!("{file}_chan_{chan}.xml");
            let got = std::fs::read_to_string(format!("vrcts_solo/{name}")).unwrap();
            let want = std::fs::read_to_string(fixture_phase4a(&name)).unwrap();
            assert_vrcts_eq(&got, &want, &format!("vrcts {name}"));
        }
    }

    // Non-vacuity: the two channels of f1 must DIFFER (per-channel segmentations), so
    // the multi-channel fan-out is genuinely exercised, not two copies of channel 1.
    let c1 = std::fs::read_to_string("vrcts_solo/f1_chan_1.xml").unwrap();
    let c2 = std::fs::read_to_string("vrcts_solo/f1_chan_2.xml").unwrap();
    assert_ne!(
        c1, c2,
        "f1 channel 1 and 2 VRCTS must differ (per-channel segs)"
    );
    assert!(c2.contains("ch=\"2\""), "channel 2 doc must carry ch=\"2\"");
    assert!(
        c2.contains("num=\"2\""),
        "channel 2 doc must carry num=\"2\""
    );
}

// === listing_parse_vs_fixture ================================================
// Corpus::from_config on the committed listing == the manifest MEASURED values,
// STRICT (pure integer/float parse; no libm). Pins the weight column, defaults,
// file ids, class mapping, and per-class counts.
#[test]
fn listing_parse_vs_fixture() {
    let _lock = CWD_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let dir = tempfile::tempdir().unwrap();
    seed_corpus(dir.path());
    let _cwd = CwdGuard::enter(dir.path());

    let cfg = load_config("tier1_tdc.config");
    let corpus = Corpus::from_config(&cfg).unwrap();

    // Manifest MEASURED values (recorded by the harness's real Corpus(configs[0])).
    assert_eq!(corpus.nb_of_files(), 3, "nb_files");

    // Row 1 full: eng;us;0.5;2 -> class 0, weight 0.5, fileid 2.
    assert_eq!(corpus.item(0).language, "eng");
    assert_eq!(corpus.item(0).dialect, "us");
    assert_eq!(corpus.item(0).class_index, 0);
    assert_eq!(corpus.item(0).weight, 0.5);
    assert_eq!(corpus.item(0).file_id, 2);

    // Row 2 ;;0.75: empty lang/dial default "unk", weight 0.75, no fileid -> 1;
    // mapping unk;unk -> class 1.
    assert_eq!(corpus.item(1).language, "unk");
    assert_eq!(corpus.item(1).dialect, "unk");
    assert_eq!(corpus.item(1).class_index, 1);
    assert_eq!(corpus.item(1).weight, 0.75);
    assert_eq!(corpus.item(1).file_id, 1);

    // Row 3 defaults-only: lang/dial "unk", weight 1.0, fileid 1, class 1.
    assert_eq!(corpus.item(2).language, "unk");
    assert_eq!(corpus.item(2).dialect, "unk");
    assert_eq!(corpus.item(2).class_index, 1);
    assert_eq!(corpus.item(2).weight, 1.0);
    assert_eq!(corpus.item(2).file_id, 1);

    // Per-class / per-language counts (manifest MEASURED).
    assert_eq!(*corpus.class_count().get(&0).unwrap(), 1, "class 0 count");
    assert_eq!(*corpus.class_count().get(&1).unwrap(), 2, "class 1 count");
    assert_eq!(*corpus.language_count().get("eng").unwrap(), 1, "eng count");
    assert_eq!(*corpus.language_count().get("unk").unwrap(), 2, "unk count");
}

// ============================================================================
// Minimal MAT v5 reader (the writer has no reader; spec S6). Uncompressed v5 only:
// 128-byte header then a sequence of miMATRIX elements. Enough to pull a variable's
// matrix by name for the assertions above. (Mirrors phase4a_corpus_processor.rs.)
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
        p += 8 + 8; // array flags sub-element
        let dims_size = u32_le(bytes, p + 4) as usize;
        let rows = i32_le(bytes, p + 8) as usize;
        let cols = i32_le(bytes, p + 12) as usize;
        p += 8 + pad8(dims_size);
        let name_size = u32_le(bytes, p + 4) as usize;
        let name = String::from_utf8(bytes[p + 8..p + 8 + name_size].to_vec()).unwrap();
        p += 8 + pad8(name_size);
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

fn mat_var_matrix(bytes: &[u8], name: &str) -> Option<Array2<f64>> {
    mat_vars(bytes)
        .into_iter()
        .find(|(n, _, _, _)| n == name)
        .map(|(_, r, c, data)| {
            // The writer stored column-major; rebuild row-major Array2.
            Array2::from_shape_fn((r, c), |(i, j)| data[j * r + i])
        })
}
