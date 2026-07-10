//! Phase 4d Task 4: the port-side END-TO-END PARITY gates against the 2015
//! production `FastSpeechProcessing` binary.
//!
//! The oracle fixtures under `tests/reference_data/phase4d/` (Task 3) come from a
//! SPLIT oracle: the VRCTS xml per channel is from the resurrected 2015 x86_64
//! `-ffast-math` `Segment` binary (byte-deterministic), and the scored numeric
//! columns (`MultiConfigResults`) are from a `-ffast-math` REBUILD driving the
//! same real compiled `CorpusProcessor` stack under `-m`. These tests replay the
//! SAME committed excerpt + localized configs + weight packs through the RUST
//! engine (`CorpusProcessor`) and assert:
//!
//!   (a) STRUCTURAL EXACT per channel -- segment count + type sequence vs the
//!       oracle VRCTS xml, parsed via the crate's own `load_vrcts`. A structural
//!       mismatch is a HARD failure (the parity floor), never absorbed.
//!   (b) per-boundary |dt| <= the manifest's pinned `pinned_abs_s` (one VRCTS
//!       4-decimal display quantum; measured 0.0 -- the port's VRCTS is
//!       byte-identical to the oracle's).
//!   (c) `MultiConfigResults` columns vs `parity_*_mcr.bin` within the manifest's
//!       pinned floor: id/count columns (0/1/2/10/20) EXACT, the wall-clock timing
//!       column (6) masked, the WER/LID columns (11..=19) asserted 0, and the
//!       scored data columns (3/4/5/7/8/9) within `ULP<=4 OR |diff|<=512*eps*
//!       max(|oracle|,1)` (the phase1/4a cross-libm comparator -- the `-ffast-math`
//!       oracle is not bit-matchable even on-host, so there is no strict arm).
//!
//! Tolerances are READ FROM THE MANIFEST (`task4_measured_deltas`, filled by
//! `scripts/extract_phase4d_fixtures.py --measure-deltas`), so the committed test
//! carries no libm-fragile strict-bits assertion on any NN-chain value. Everything
//! the test needs is committed under `tests/reference_data/phase4d/`; no oracle,
//! `fsp_runtime`, fastmath build, or legacy tree is touched at test time.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use ndarray::Array2;
use serde_json::Value;

use speech::cli::{Mode, ModeKind};
use speech::engine::corpus_processor::CorpusProcessor;
use speech::legacy_config::parse_legacy_config;
use speech::tasks::segmentation::{SegClass, Segment};
use speech::tasks::segmentation_io::load_vrcts;

/// `set_current_dir` is process-global; every run chdir's into a seeded tempdir
/// (the localized configs carry relative corpus/output paths). Serialize against
/// `cargo test`'s parallel threads.
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

fn phase4d_src() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/reference_data/phase4d")
}

fn manifest() -> Value {
    let text = std::fs::read_to_string(phase4d_src().join("phase4d_parity_manifest.json")).unwrap();
    serde_json::from_str(&text).unwrap()
}

/// Read a pinned scalar out of `task4_measured_deltas.<tuple>.<leg>.<field>`, so
/// the tolerances live in the committed manifest (filled by the `--measure-deltas`
/// pass), not hardcoded in the test.
fn pinned(m: &Value, tuple: &str, leg: &str, field: &str) -> f64 {
    m["task4_measured_deltas"][tuple][leg][field]
        .as_f64()
        .unwrap_or_else(|| panic!("manifest missing task4_measured_deltas.{tuple}.{leg}.{field}"))
}

/// Weight pack + expected per-channel speech-segment counts for a tuple (the
/// manifest `fixtures.*.speech_segments`, also the extractor's non-vacuity floor).
struct Tuple {
    name: &'static str,
    weights: &'static str,
    segs: [usize; 2],
}

const TUPLE_A: Tuple = Tuple {
    name: "tupleA",
    weights: "tupleA_NNweights_config1.bin",
    segs: [8, 3],
};
const TUPLE_B: Tuple = Tuple {
    name: "tupleB",
    weights: "tupleB_NNweights_config1.bin",
    segs: [13, 15],
};

/// Seed `dir` with the COMMITTED parity fixtures (excerpt wav/stm, weight pack, aux
/// inputs) + `config_text` under the parity config name. Mirrors the extractor's
/// `_seed_parity_workdir`; the localized config resolves these relative names.
fn seed(dir: &Path, tuple: &Tuple, config_text: &str) -> String {
    let src = phase4d_src();
    std::fs::copy(src.join("prcts_excerpt.wav"), dir.join("prcts_excerpt.wav")).unwrap();
    std::fs::copy(src.join("prcts_excerpt.stm"), dir.join("prcts_excerpt.stm")).unwrap();
    std::fs::copy(src.join(tuple.weights), dir.join("NNweights_config1.bin")).unwrap();
    std::fs::write(
        dir.join("languagemapping.csv"),
        "fax;non;0\nchi;man;1\nspa;spa;2\n",
    )
    .unwrap();
    std::fs::write(
        dir.join("fileslisting"),
        "prcts_excerpt.wav;prcts_excerpt.stm;fax;non;1;30.0;\n",
    )
    .unwrap();
    let config_name = format!("parity_{}.config", tuple.name);
    std::fs::write(dir.join(&config_name), config_text).unwrap();
    config_name
}

fn config_text(tuple: &Tuple) -> String {
    std::fs::read_to_string(phase4d_src().join(format!("parity_{}.config", tuple.name))).unwrap()
}

/// Add (or overwrite) `Dump_Directory` so the unscored -s branch fans VRCTS xml out
/// to `<dir>/prcts_excerpt_chan_<n>.xml` (the committed parity configs leave it unset).
fn with_dump_dir(text: &str, dump_dir: &str) -> String {
    let mut out = String::new();
    let mut seen = false;
    for line in text.lines() {
        if line.starts_with("Dump_Directory ") {
            out.push_str(&format!("Dump_Directory {dump_dir}\n"));
            seen = true;
        } else {
            out.push_str(line);
            out.push('\n');
        }
    }
    if !seen {
        out.push_str(&format!("Dump_Directory {dump_dir}\n"));
    }
    out
}

fn mode(kind: ModeKind) -> Mode {
    Mode {
        kind,
        verbose: false,
    }
}

/// Read a committed `.bin` fixture (i64 rows, i64 cols, f64 column-major) into a
/// row-major `Array2`.
fn load_bin(name: &str) -> Array2<f64> {
    let (rows, cols, data) = speech::io::binary::read_matrix(&phase4d_src().join(name)).unwrap();
    Array2::from_shape_fn((rows, cols), |(i, j)| data[j * rows + i])
}

// === (a)+(b) VRCTS structural + boundary gate ===============================
// Run the port in Solo (-s) mode with Dump_Directory set -> per-channel VRCTS xml.
// Parse BOTH the port's xml and the committed oracle xml via `load_vrcts`, then
// require an EXACT segment-count + type-sequence match and per-boundary
// |dt| <= the pinned display quantum. The Solo unscored branch is exactly the
// path the -s real-binary oracle took (segmentation is mode-invariant).
fn run_vrcts_gate(tuple: &Tuple) {
    let _lock = CWD_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let m = manifest();
    let pinned_dt = pinned(&m, tuple.name, "port_vs_oracle_boundary_dt", "pinned_abs_s");

    let dir = tempfile::tempdir().unwrap();
    let text = with_dump_dir(&config_text(tuple), "vrcts_out");
    let config_name = seed(dir.path(), tuple, &text);
    std::fs::create_dir_all(dir.path().join("vrcts_out")).unwrap();

    let _cwd = CwdGuard::enter(dir.path());
    let cfg = parse_legacy_config(&std::fs::read_to_string(&config_name).unwrap());
    let mut cp = CorpusProcessor::new(vec![cfg], mode(ModeKind::Solo)).unwrap();
    cp.run().unwrap();

    for (chan0, &expected_segs) in tuple.segs.iter().enumerate() {
        let chan = chan0 + 1;
        let port_xml =
            std::fs::read_to_string(format!("vrcts_out/prcts_excerpt_chan_{chan}.xml")).unwrap();
        let oracle_xml = std::fs::read_to_string(
            phase4d_src().join(format!("parity_{}_vrcts_chan{chan}.xml", tuple.name)),
        )
        .unwrap();

        // Parse both via the crate's own loader (off=0, dur read from the xml).
        let port_seg = load_vrcts(&port_xml, 0.0, 60.0);
        let oracle_seg = load_vrcts(&oracle_xml, 0.0, 60.0);
        let port: &[Segment] = port_seg.segments();
        let oracle: &[Segment] = oracle_seg.segments();

        // Non-vacuity: the oracle carries the manifest-pinned speech-segment count
        // (>1, so the decision layer is genuinely exercised). VRCTS interleaves
        // Speech with Other; the boundary list has 2*speech + 1 entries.
        let speech_count = oracle.iter().filter(|s| s.ty == SegClass::Speech).count();
        assert_eq!(
            speech_count, expected_segs,
            "{} chan{chan}: oracle speech-seg count",
            tuple.name
        );
        assert!(
            expected_segs > 1,
            "{} chan{chan}: excerpt must be non-degenerate (>1 speech seg)",
            tuple.name
        );

        // (a) STRUCTURAL EXACT: identical boundary count + type sequence. HARD.
        assert_eq!(
            port.len(),
            oracle.len(),
            "{} chan{chan}: STRUCTURAL boundary-count mismatch port={} oracle={}",
            tuple.name,
            port.len(),
            oracle.len()
        );
        for (i, (p, o)) in port.iter().zip(oracle.iter()).enumerate() {
            assert_eq!(
                p.ty, o.ty,
                "{} chan{chan}: STRUCTURAL type mismatch at boundary {i}: port={:?} oracle={:?}",
                tuple.name, p.ty, o.ty
            );
        }

        // (b) per-boundary |dt| <= the pinned display quantum.
        for (i, (p, o)) in port.iter().zip(oracle.iter()).enumerate() {
            let dt = (p.begin - o.begin).abs();
            assert!(
                dt <= pinned_dt,
                "{} chan{chan}: boundary {i} |dt|={dt:.3e} > pinned {pinned_dt:.3e} (port={} oracle={})",
                tuple.name,
                p.begin,
                o.begin
            );
        }
    }
}

// === (c) MultiConfigResults numeric gate ====================================
// Run the port in Multi (-m) mode (the scored path the fastmath -m oracle took),
// read `results_matrix()` and compare vs the committed masked `mcr.bin`:
//   cols 0/1/2  (file/conf/chan ids)   : EXACT
//   col  6      (wall-clock timing)     : masked (skipped)
//   col  10     (nbWords legacy -1)     : EXACT
//   cols 11..=19 (WER/LID, 0 in fixture): asserted 0 in the port run too
//   col  20     (NbOfClassif)           : EXACT integer
//   cols 3/4/5/7/8/9 (scored data)      : ULP<=4 OR |diff|<=512*eps*max(|oracle|,1)
fn run_mcr_gate(tuple: &Tuple) {
    let _lock = CWD_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let m = manifest();
    let ulp_bound = pinned(
        &m,
        tuple.name,
        "port_vs_oracle_mcr_col_deltas",
        "pinned_ulp",
    ) as u64;
    let abs_factor = pinned(
        &m,
        tuple.name,
        "port_vs_oracle_mcr_col_deltas",
        "pinned_abs_factor",
    );
    let eps = pinned(&m, tuple.name, "port_vs_oracle_mcr_col_deltas", "eps");

    let dir = tempfile::tempdir().unwrap();
    let config_name = seed(dir.path(), tuple, &config_text(tuple));

    let _cwd = CwdGuard::enter(dir.path());
    let cfg = parse_legacy_config(&std::fs::read_to_string(&config_name).unwrap());
    let mut cp = CorpusProcessor::new(vec![cfg], mode(ModeKind::Multi)).unwrap();
    cp.run().unwrap();

    let got = cp.results_matrix();
    let want = load_bin(&format!("parity_{}_mcr.bin", tuple.name));
    assert_eq!(
        got.dim(),
        want.dim(),
        "{}: MCR shape mismatch got {:?} want {:?}",
        tuple.name,
        got.dim(),
        want.dim()
    );
    let (rows, cols) = want.dim();
    assert_eq!((rows, cols), (2, 21), "{}: MCR must be 2x21", tuple.name);

    for r in 0..rows {
        for c in 0..cols {
            let g = got[[r, c]];
            let w = want[[r, c]];
            match c {
                6 => {} // wall-clock timing: masked.
                0 | 1 | 2 | 10 | 20 => assert_eq!(
                    g.to_bits(),
                    w.to_bits(),
                    "{}: MCR[{r},{c}] id/count column must be EXACT: got {g} want {w}",
                    tuple.name
                ),
                11..=19 => {
                    // WER + LID columns are 0 in the fixture (no ASR text / live LID
                    // net wired into this excerpt run); assert the port also emits 0.
                    assert_eq!(w, 0.0, "{}: fixture MCR[{r},{c}] expected 0", tuple.name);
                    assert_eq!(
                        g, 0.0,
                        "{}: MCR[{r},{c}] WER/LID column must be 0 in the port run, got {g}",
                        tuple.name
                    );
                }
                _ => assert!(
                    hybrid_ok(g, w, ulp_bound, abs_factor * eps * w.abs().max(1.0)),
                    "{}: MCR[{r},{c}] scored column outside pinned floor: |diff|={:.3e} ULP={} got {g} want {w}",
                    tuple.name,
                    (g - w).abs(),
                    ulp_dist(g, w).map_or_else(|| "inf".to_string(), |d| d.to_string()),
                ),
            }
        }
    }
}

/// ULP distance between two finite same-sign f64s (None on NaN/inf/sign mismatch).
fn ulp_dist(a: f64, b: f64) -> Option<u64> {
    if !a.is_finite() || !b.is_finite() || a.is_sign_negative() != b.is_sign_negative() {
        return None;
    }
    Some((a.to_bits() as i64).abs_diff(b.to_bits() as i64))
}

/// Pass iff within `ulp_bound` ULP OR within `abs_tol` absolute. Same two-armed
/// cross-libm envelope as `common::hybrid_ok_f64` (mirrored here: the phase4d
/// oracle is `-ffast-math`, so unlike the `assert_oracle_eq_f64` family there is
/// NO strict-bits arm -- the fastmath columns are never bit-matchable).
fn hybrid_ok(a: f64, b: f64, ulp_bound: u64, abs_tol: f64) -> bool {
    if !a.is_finite() || !b.is_finite() {
        return false;
    }
    ulp_dist(a, b).is_some_and(|d| d <= ulp_bound) || (a - b).abs() <= abs_tol
}

#[test]
fn tuple_a_vrcts_structural_and_boundaries() {
    run_vrcts_gate(&TUPLE_A);
}

#[test]
fn tuple_a_mcr_columns() {
    run_mcr_gate(&TUPLE_A);
}

#[test]
fn tuple_b_vrcts_structural_and_boundaries() {
    run_vrcts_gate(&TUPLE_B);
}

#[test]
fn tuple_b_mcr_columns() {
    run_mcr_gate(&TUPLE_B);
}
