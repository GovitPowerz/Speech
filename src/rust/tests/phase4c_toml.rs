//! Phase 4c Task 5: TOML canonical config -- declarative flattening tests.
//!
//! Four named tests per the task brief:
//! - `toml_map_equivalence`: for every committed fixture config, `.config -> map`
//!   equals `map -> .toml -> map` (strict `IndexMap` equality, which is already
//!   order-insensitive; see `toml_config`'s module doc). Also exercises the real
//!   duplicate-key fixture (`configs/legacy/LID_BLSTM.config`) explicitly.
//! - `engine_output_bit_identity`: the tier-1 TDC fixture run through
//!   `CorpusProcessor` from its `.config` vs from its converted `.toml` produces a
//!   bit-identical `results_matrix`, modulo column 6 (wall-clock timing). A THIRD
//!   in-test control run (same `.config`-derived map, cloned) proves the col-6
//!   mask is actually justified -- two runs of the IDENTICAL config diverge only
//!   at column 6 too -- instead of relying on a one-off manual scratch check.
//! - `lid_blstm_toml_full_coverage`: converting the full legacy `LID_BLSTM.config`
//!   produces ZERO `[legacy.raw]` fallbacks -- `KEY_TABLE` covers every key it uses.
//! - `raw_escape_hatch_roundtrips`: an unmapped key (incl. one needing a quoted
//!   TOML key) round-trips losslessly through `[legacy.raw]`.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use indexmap::IndexMap;
use ndarray::Array2;

use speech::cli::{Mode, ModeKind};
use speech::engine::corpus_processor::CorpusProcessor;
use speech::legacy_config::parse_legacy_config;
use speech::toml_config::{map_to_toml, toml_to_map};

/// `set_current_dir` is process-global; serialize against `cargo test`'s parallel
/// threads (mirrors `phase4a_tier1_e2e.rs`).
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

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Every committed fixture `.config` this table was built from the union of
/// (phase0 + phase4a + phase4b + the legacy LID_BLSTM sample).
fn fixture_configs() -> Vec<PathBuf> {
    let root = repo_root();
    let mut paths = vec![
        root.join("configs/legacy/LID_BLSTM.config"),
        root.join("tests/reference_data/phase0/1_worker_1.config"),
    ];
    for dir in [
        "tests/reference_data/phase4a",
        "tests/reference_data/phase4b",
    ] {
        let d = root.join(dir);
        let mut entries: Vec<PathBuf> = std::fs::read_dir(&d)
            .unwrap_or_else(|e| panic!("read_dir {d:?}: {e}"))
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|ext| ext == "config"))
            .collect();
        entries.sort();
        paths.extend(entries);
    }
    paths
}

// === toml_map_equivalence ====================================================
#[test]
fn toml_map_equivalence() {
    let fixtures = fixture_configs();
    assert!(
        fixtures.len() >= 22,
        "expected >=22 committed fixture configs, got {}",
        fixtures.len()
    );

    for path in &fixtures {
        let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
        let want = parse_legacy_config(&text);

        let toml_text = map_to_toml(&want);
        let got = toml_to_map(&toml_text)
            .unwrap_or_else(|e| panic!("toml_to_map round-trip for {path:?} failed: {e:#}"));

        assert_eq!(
            got, want,
            "round-trip mismatch for {path:?} (.config -> map != convert -> .toml -> map)"
        );

        // Full-coverage side-channel: every key in every committed fixture must be
        // in KEY_TABLE (never fall back to [legacy.raw]) -- the table was built
        // from exactly this fixture union, so a gap here is a transcription bug.
        assert!(
            !toml_text.contains("[legacy.raw]"),
            "{path:?} produced a [legacy.raw] fallback -- KEY_TABLE is missing a key from this fixture"
        );
    }
}

/// The real `configs/legacy/LID_BLSTM.config` fixture carries GENUINE duplicate
/// keys (last-wins is the documented legacy semantics; see `legacy_config.rs` and
/// `toml_config`'s module doc) -- `language2classmapping` and `fileslisting` are
/// each repeated multiple times, and `Neural_Networks_BackPropagation_Epochs`
/// appears twice with different values. Prove the round trip preserves the
/// COLLAPSED (last-wins) value, not some other one, and that the raw duplicate
/// count in the source text is strictly greater than the post-parse map's single
/// entry -- i.e. this is a real collapse, not a vacuous single-occurrence key.
#[test]
fn duplicate_key_collapses_to_last_wins() {
    let path = repo_root().join("configs/legacy/LID_BLSTM.config");
    let text = std::fs::read_to_string(&path).unwrap();

    let raw_occurrences = text
        .lines()
        .filter(|l| l.trim_start().starts_with("fileslisting"))
        .count();
    assert!(
        raw_occurrences > 1,
        "fixture assumption: fileslisting must repeat in the raw text"
    );

    let map = parse_legacy_config(&text);
    // Last non-comment occurrence in the file wins (line 126, not the earlier
    // train_phSeq listing on line 79).
    let want_last = "/Users/Govit/Documents/QT/FastSpeechProcessing/ConfigFiles/listing_eval_minus_phSeqbis.csv";
    assert_eq!(map.get("fileslisting").unwrap(), want_last);

    let toml_text = map_to_toml(&map);
    let round_tripped = toml_to_map(&toml_text).unwrap();
    assert_eq!(round_tripped.get("fileslisting").unwrap(), want_last);
    assert_eq!(round_tripped, map);
}

// === engine_output_bit_identity ==============================================
fn phase4a_src() -> PathBuf {
    repo_root().join("tests/reference_data/phase4a")
}

fn seed_tier1_corpus(dir: &Path) {
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
    for f in ["fileslisting.csv", "language2classmapping.csv"] {
        std::fs::copy(src.join(f), dir.join(f)).unwrap();
    }
}

#[test]
fn engine_output_bit_identity() {
    let _lock = CWD_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let dir = tempfile::tempdir().unwrap();
    seed_tier1_corpus(dir.path());
    let _cwd = CwdGuard::enter(dir.path());

    let text = std::fs::read_to_string(phase4a_src().join("tier1_tdc.config")).unwrap();
    let mut cfg_map = parse_legacy_config(&text);
    // Solo-eval, no training loop -- mirrors phase4a_tier1_e2e.rs::solo_tdc_matches.
    cfg_map.insert(
        "Neural_Networks_BackPropagation_Epochs".to_string(),
        "0".to_string(),
    );
    cfg_map.insert("Dump_Directory".to_string(), "vrcts_config".to_string());
    std::fs::create_dir_all("vrcts_config").unwrap();

    // The CONTROL run: a second `CorpusProcessor` built from the SAME
    // `.config`-derived map (cloned before the primary run consumes `cfg_map`;
    // own `Dump_Directory` so its VRCTS writes cannot cross-talk with the
    // primary run's). This is what upgrades the col-6-is-timing-only claim below
    // from a one-off manual scratch check into something CI actually proves.
    let control_map = {
        let mut m = cfg_map.clone();
        m.insert("Dump_Directory".to_string(), "vrcts_control".to_string());
        m
    };
    std::fs::create_dir_all("vrcts_control").unwrap();

    let toml_map = {
        let mut m = cfg_map.clone();
        m.insert("Dump_Directory".to_string(), "vrcts_toml".to_string());
        m
    };
    std::fs::create_dir_all("vrcts_toml").unwrap();
    let toml_text = map_to_toml(&toml_map);
    std::fs::write("tier1_tdc.toml", &toml_text).unwrap();
    let reloaded_toml_map = toml_to_map(&toml_text).unwrap();
    assert_eq!(
        reloaded_toml_map, toml_map,
        "toml round trip must be lossless before feeding the engine"
    );

    fn mode(kind: ModeKind) -> Mode {
        Mode {
            kind,
            verbose: false,
        }
    }

    let mut cp_config = CorpusProcessor::new(vec![cfg_map], mode(ModeKind::Solo)).unwrap();
    cp_config.run().unwrap();

    let mut cp_control = CorpusProcessor::new(vec![control_map], mode(ModeKind::Solo)).unwrap();
    cp_control.run().unwrap();

    let mut cp_toml = CorpusProcessor::new(vec![reloaded_toml_map], mode(ModeKind::Solo)).unwrap();
    cp_toml.run().unwrap();

    let from_config = cp_config.results_matrix();
    let from_control = cp_control.results_matrix();
    let from_toml = cp_toml.results_matrix();

    // Column 6 is the wall-clock TIMING column (masked in every phase4a golden
    // comparison, e.g. `phase4a_tier1_e2e.rs::solo_tdc_matches`) -- it is expected
    // to differ between any two runs, even two runs of the IDENTICAL config in the
    // SAME process. The CONTROL pair proves exactly that claim in CI (rather than
    // a one-off manual scratch check): two `CorpusProcessor` runs from the SAME
    // `.config`-derived map must ALSO diverge only at column 6. Every other
    // column is a pure function of the (identical) config content, so both pairs
    // must be STRICT bit-identical elsewhere -- this is a same-port same-machine
    // comparison, not an oracle/libm one, so no canary-gated tolerance applies.
    assert_masked_col6_bit_identical(
        from_config,
        from_control,
        "control (.config vs .config, identical map)",
    );
    assert_masked_col6_bit_identical(from_config, from_toml, ".config vs .toml");

    // Non-vacuity: the matrix must actually contain data (3 files x 2 channels).
    assert_eq!(from_config.dim(), (6, 21));
}

/// Shared masked-compare helper for `engine_output_bit_identity`: two
/// `results_matrix()` outputs must be bit-identical on every column except 6
/// (wall-clock timing).
fn assert_masked_col6_bit_identical(a: &Array2<f64>, b: &Array2<f64>, label: &str) {
    assert_eq!(a.dim(), b.dim(), "{label}: results_matrix shape must match");
    let (rows, cols) = a.dim();
    for r in 0..rows {
        for c in 0..cols {
            if c == 6 {
                continue;
            }
            assert_eq!(
                a[[r, c]].to_bits(),
                b[[r, c]].to_bits(),
                "{label}: results_matrix[{r},{c}] must be bit-identical (got {} vs {})",
                a[[r, c]],
                b[[r, c]]
            );
        }
    }
}

// === lid_blstm_toml_full_coverage ============================================
// The COMMITTED configs/lid/lid_blstm.toml (grown from the sketch via
// `speech --convert-config`, hand-checked) must flatten to the EXACT same map as
// the legacy configs/legacy/LID_BLSTM.config -- full key coverage, identical
// values, no [legacy.raw] fallback (KEY_TABLE covers every key the legacy config
// uses).
#[test]
fn lid_blstm_toml_full_coverage() {
    let legacy_path = repo_root().join("configs/legacy/LID_BLSTM.config");
    let legacy_text = std::fs::read_to_string(&legacy_path).unwrap();
    let want = parse_legacy_config(&legacy_text);
    assert!(
        want.len() > 100,
        "sanity: LID_BLSTM.config should carry >100 keys, got {}",
        want.len()
    );

    let toml_path = repo_root().join("configs/lid/lid_blstm.toml");
    let toml_text = std::fs::read_to_string(&toml_path).unwrap();
    assert!(
        !toml_text.contains("[legacy.raw]"),
        "the committed lid_blstm.toml must not need the [legacy.raw] fallback -- \
         KEY_TABLE must cover every LID_BLSTM.config key"
    );

    let got = toml_to_map(&toml_text)
        .unwrap_or_else(|e| panic!("committed lid_blstm.toml failed to parse: {e:#}"));
    assert_eq!(
        got, want,
        "configs/lid/lid_blstm.toml must flatten to the exact legacy LID_BLSTM.config map"
    );

    // And the fresh conversion agrees with the committed file's semantics (the
    // committed file may carry extra comments, but never a semantic drift).
    let regenerated = toml_to_map(&map_to_toml(&want)).unwrap();
    assert_eq!(regenerated, got);
}

// === raw_escape_hatch_roundtrips ==============================================
#[test]
fn raw_escape_hatch_roundtrips() {
    let mut map = IndexMap::new();
    // A real mapped key, to prove raw and mapped keys coexist correctly.
    map.insert("Algo_choice".to_string(), "6".to_string());
    // An unmapped key that IS a valid bare TOML identifier.
    map.insert(
        "Some_Future_Legacy_Key".to_string(),
        "3.795068189162301e-01".to_string(),
    );
    // An unmapped key that is NOT a valid bare TOML identifier (contains a dot),
    // exercising `toml_key_repr`'s quoting path.
    map.insert(
        "weird.key with space".to_string(),
        "value, with, commas".to_string(),
    );

    let toml_text = map_to_toml(&map);
    assert!(
        toml_text.contains("[legacy.raw]"),
        "unmapped keys must land under [legacy.raw]:\n{toml_text}"
    );
    assert!(
        toml_text.contains("[engine]"),
        "mapped key must still land in its own section:\n{toml_text}"
    );
    assert!(
        toml_text.contains("\"weird.key with space\""),
        "a non-bare legacy key must be TOML-quoted:\n{toml_text}"
    );

    let round_tripped = toml_to_map(&toml_text).unwrap();
    assert_eq!(
        round_tripped, map,
        "raw escape hatch must round-trip losslessly"
    );
}
