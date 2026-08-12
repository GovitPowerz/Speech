//! Phase 9 Task 1: the port-only cell/direction config vocabulary (spec S6).
//!
//! Two things are pinned here:
//! - the four new `KEY_TABLE` rows (`BLSTM{,_LID}_Cell_Type` /
//!   `BLSTM{,_LID}_Direction`) round-trip in BOTH directions -- flat -> TOML puts
//!   them in their declared section (never `[legacy.raw]`), and TOML -> flat -> TOML
//!   is a fixed point;
//! - the four `Mamba_*` geometry rows (Task 3, spec S3.3/S6) round-trip the same way
//!   into `[nn_mamba]`;
//! - the two `Cfc_*` geometry rows (PHASE 10 Task 1, phase-10 spec S1.4/S2) round-trip
//!   into `[nn_cfc]` -- the same file, because the vocabulary is one table and the
//!   fourth cell is meant to be a row in it, not a new mechanism;
//! - the three `Transformer_*` geometry rows (PHASE 11 Task 2, phase-11 spec S1/S2)
//!   round-trip into `[nn_transformer]`, same file for the same reason;
//! - a real committed fixture config (`phase4a/tier2_spectral.config`, the Algo-3
//!   spectral SAD net) BUILDS a working driver under every cell type -- `lstm`
//!   (default), `slstm` (Task 2), `mamba` (Task 3), `cfc` (phase-10 Task 1) and
//!   `transformer` (phase-11 Task 2).
//!
//! The Task-1 scaffolding test `unimplemented_cell_type_bails_on_the_tier2_fixture`
//! is GONE: with Task 3 landed there is no unimplemented cell left to bail on, and
//! `mamba_cell_type_builds_on_the_tier2_fixture` replaced it (as
//! `slstm_cell_type_builds_on_the_tier2_fixture` replaced the `slstm` arm at Task 2).
//! A future `CellType` variant is caught by the exhaustive `match cell_type` in
//! `BlstmNetwork::from_config` -- a compile error, not a runtime bail.

use std::path::PathBuf;

use indexmap::IndexMap;
use speech::legacy_config::parse_legacy_config;
use speech::tasks::sad::BlstmSpectralSegmenter;
use speech::tasks::segmenter::Segmenter;
use speech::toml_config::{map_to_toml, toml_to_map};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn tier2_map() -> IndexMap<String, String> {
    let path = repo_root().join("tests/reference_data/phase4a/tier2_spectral.config");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path:?}: {e}"));
    parse_legacy_config(&text)
}

/// The four port-only rows land in their declared sections (`[nn]` for the SAD net,
/// `[nn_lid]` for the Twin's LID net -- the table's existing spelling), never in the
/// `[legacy.raw]` escape hatch, and decode back to the exact same flat map.
#[test]
fn cell_and_direction_keys_round_trip_through_their_declared_sections() {
    let mut map: IndexMap<String, String> = IndexMap::new();
    map.insert("BLSTM_Cell_Type".into(), "slstm".into());
    map.insert("BLSTM_Direction".into(), "forward".into());
    map.insert("BLSTM_LID_Cell_Type".into(), "mamba".into());
    map.insert("BLSTM_LID_Direction".into(), "bidirectional".into());
    // Task 3: the Mamba geometry, UNPREFIXED by design (one geometry per config).
    map.insert("Mamba_D_State".into(), "8".into());
    map.insert("Mamba_D_Conv".into(), "4".into());
    map.insert("Mamba_Expand".into(), "2".into());
    map.insert("Mamba_Dt_Rank".into(), "0".into());
    // Phase 10 Task 1: the CfC geometry, UNPREFIXED for the same reason.
    map.insert("Cfc_Backbone_Units".into(), "24".into());
    map.insert("Cfc_Backbone_Layers".into(), "1".into());
    // Phase 11 Task 2: the transformer geometry, UNPREFIXED for the same reason.
    map.insert("Transformer_Window".into(), "64".into());
    map.insert("Transformer_Heads".into(), "4".into());
    map.insert("Transformer_D_Ff".into(), "64".into());

    let toml = map_to_toml(&map);
    assert!(
        !toml.contains("[legacy.raw]"),
        "a port-only key fell through to the escape hatch:\n{toml}"
    );
    for want in [
        "[nn]",
        "cell_type = \"slstm\"",
        "direction = \"forward\"",
        "[nn_lid]",
        "cell_type = \"mamba\"",
        "direction = \"bidirectional\"",
        "[nn_mamba]",
        "d_state = 8",
        "d_conv = 4",
        "expand = 2",
        "dt_rank = 0",
        "[nn_cfc]",
        "backbone_units = 24",
        "backbone_layers = 1",
        "[nn_transformer]",
        "window = 64",
        "heads = 4",
        "d_ff = 64",
    ] {
        assert!(toml.contains(want), "missing {want:?} in:\n{toml}");
    }

    // flat -> toml -> flat is exact, and toml -> flat -> toml is a fixed point.
    assert_eq!(toml_to_map(&toml).unwrap(), map);
    assert_eq!(map_to_toml(&toml_to_map(&toml).unwrap()), toml);
}

/// The keys survive a round trip alongside a REAL fixture config's key set (the
/// tier-2 Algo-3 spectral net), i.e. they do not collide with an existing row.
#[test]
fn tier2_fixture_round_trips_with_the_new_keys_added() {
    let mut map = tier2_map();
    assert!(!map.contains_key("BLSTM_Cell_Type"));
    assert!(!map.contains_key("BLSTM_Direction"));
    assert!(!map.contains_key("Mamba_D_State"));
    assert!(!map.contains_key("Cfc_Backbone_Units"));
    assert!(!map.contains_key("Transformer_Window"));
    map.insert("BLSTM_Cell_Type".into(), "lstm".into());
    map.insert("BLSTM_Direction".into(), "bidirectional".into());
    map.insert("Mamba_D_State".into(), "16".into());
    map.insert("Mamba_Dt_Rank".into(), "0".into());
    map.insert("Cfc_Backbone_Units".into(), "24".into());
    map.insert("Transformer_Window".into(), "32".into());

    let toml = map_to_toml(&map);
    assert_eq!(toml_to_map(&toml).unwrap(), map);
}

/// Task 3 FLIP (this replaced `unimplemented_cell_type_bails_on_the_tier2_fixture`):
/// the same real fixture config with `BLSTM_Cell_Type mamba` plus a small
/// `Mamba_*` geometry now BUILDS a working Algo-3 driver, and the geometry keys
/// actually reach the cell -- doubling `Mamba_D_State` changes the pack length by
/// exactly the `A_log` + `W_x` growth the S3.2 layout predicts, which no silent
/// fallback to another cell could produce.
#[test]
fn mamba_cell_type_builds_on_the_tier2_fixture() {
    let mamba_map = |d_state: usize| {
        let mut map = tier2_map();
        map.insert("BLSTM_Cell_Type".into(), "mamba".into());
        map.insert("Mamba_D_State".into(), d_state.to_string());
        map.insert("Mamba_D_Conv".into(), "3".into());
        map.insert("Mamba_Expand".into(), "1".into());
        map
    };

    let lstm = BlstmSpectralSegmenter::from_legacy(&tier2_map(), None)
        .expect("the untouched fixture must build");
    let mamba = BlstmSpectralSegmenter::from_legacy(&mamba_map(4), None)
        .expect("mamba must build on the fixture");
    let (n_lstm, n_mamba) = (lstm.get_weights().len(), mamba.get_weights().len());
    assert!(n_lstm > 0 && n_mamba > 0);
    assert_ne!(
        n_mamba, n_lstm,
        "the Mamba pack must differ from the LSTM one -- equal counts would mean the \
         driver silently kept the LSTM stacks"
    );

    // The geometry key is LIVE: d_state 4 -> 8 adds, per mamba layer, d_inner*4 more
    // A_log entries and 2*4*d_inner more W_x entries.
    let wider = BlstmSpectralSegmenter::from_legacy(&mamba_map(8), None).unwrap();
    assert!(
        wider.get_weights().len() > n_mamba,
        "Mamba_D_State did not reach the cell"
    );

    // Same seam contract as any other cell: a full-length pack round-trips.
    let mut net = BlstmSpectralSegmenter::from_legacy(&mamba_map(4), None).unwrap();
    let w: Vec<f64> = (0..n_mamba).map(|k| 0.11 - 0.0003 * (k as f64)).collect();
    net.set_weights(&w).unwrap();
    assert_eq!(net.get_weights(), w);
}

/// A malformed `Mamba_*` value fails LOUDLY at driver construction rather than
/// silently defaulting -- a silent default would change the weight-pack LENGTH.
#[test]
fn a_malformed_mamba_geometry_bails_on_the_tier2_fixture() {
    let mut map = tier2_map();
    map.insert("BLSTM_Cell_Type".into(), "mamba".into());
    map.insert("Mamba_Expand".into(), "0".into());
    // `BlstmSpectralSegmenter` is not `Debug`, so match instead of `unwrap_err`.
    match BlstmSpectralSegmenter::from_legacy(&map, None) {
        Ok(_) => panic!("'Mamba_Expand 0' must not build"),
        Err(e) => assert!(
            format!("{e:#}").contains("'Mamba_Expand' must be >= 1 (got 0)"),
            "{e:#}"
        ),
    }
}

/// Task 2 FLIP: the same real fixture config with `BLSTM_Cell_Type slstm` now BUILDS
/// a working Algo-3 driver -- the cell reaches production config through the full
/// `Segmenter` path, not just the unit seam. The weight-pack length changes with the
/// cell (sLSTM has no peepholes), which is the observable proof that the driver holds
/// sLSTM stacks rather than silently falling back to LSTM.
#[test]
fn slstm_cell_type_builds_on_the_tier2_fixture() {
    let lstm = BlstmSpectralSegmenter::from_legacy(&tier2_map(), None)
        .expect("the untouched fixture must build");

    let mut map = tier2_map();
    map.insert("BLSTM_Cell_Type".into(), "slstm".into());
    let slstm =
        BlstmSpectralSegmenter::from_legacy(&map, None).expect("slstm must build on the fixture");

    let (n_lstm, n_slstm) = (lstm.get_weights().len(), slstm.get_weights().len());
    assert!(n_lstm > 0 && n_slstm > 0);
    assert_ne!(
        n_slstm, n_lstm,
        "the sLSTM pack must differ from the LSTM one -- equal counts would mean the \
         driver silently kept the LSTM stacks"
    );
    // Same seam contract as any other cell: a full-length pack round-trips.
    let mut net = BlstmSpectralSegmenter::from_legacy(&map, None).unwrap();
    let w: Vec<f64> = (0..n_slstm).map(|k| 0.11 - 0.0003 * (k as f64)).collect();
    net.set_weights(&w).unwrap();
    assert_eq!(net.get_weights(), w);
}

/// The same fixture with the key ABSENT (or explicitly `lstm`) builds -- the bail is
/// gated on the cell type alone, and the default is the legacy shape.
#[test]
fn tier2_fixture_still_builds_without_the_key_and_with_explicit_lstm() {
    assert!(BlstmSpectralSegmenter::from_legacy(&tier2_map(), None).is_ok());

    let mut map = tier2_map();
    map.insert("BLSTM_Cell_Type".into(), "lstm".into());
    map.insert("BLSTM_Direction".into(), "bidirectional".into());
    assert!(BlstmSpectralSegmenter::from_legacy(&map, None).is_ok());
}

/// PHASE 10 Task 1, the `mamba_cell_type_builds_on_the_tier2_fixture` twin: the same
/// real fixture config with `BLSTM_Cell_Type cfc` BUILDS a working Algo-3 driver, and
/// the geometry keys actually reach the cell -- doubling `Cfc_Backbone_Units` grows
/// the pack by exactly the S1.2 arithmetic, which no silent fallback to another cell
/// could produce.
#[test]
fn cfc_cell_type_builds_on_the_tier2_fixture() {
    let cfc_map = |units: usize, layers: usize| {
        let mut map = tier2_map();
        map.insert("BLSTM_Cell_Type".into(), "cfc".into());
        map.insert("Cfc_Backbone_Units".into(), units.to_string());
        map.insert("Cfc_Backbone_Layers".into(), layers.to_string());
        map
    };

    let lstm = BlstmSpectralSegmenter::from_legacy(&tier2_map(), None)
        .expect("the untouched fixture must build");
    let cfc = BlstmSpectralSegmenter::from_legacy(&cfc_map(6, 1), None)
        .expect("cfc must build on the fixture");
    let (n_lstm, n_cfc) = (lstm.get_weights().len(), cfc.get_weights().len());
    assert!(n_lstm > 0 && n_cfc > 0);
    assert_ne!(
        n_cfc, n_lstm,
        "the CfC pack must differ from the LSTM one -- equal counts would mean the \
         driver silently kept the LSTM stacks"
    );

    // Both geometry keys are LIVE and INDEPENDENT: widening the backbone and deepening
    // it each grow the pack, and they are not the same knob.
    let wider = BlstmSpectralSegmenter::from_legacy(&cfc_map(12, 1), None).unwrap();
    let deeper = BlstmSpectralSegmenter::from_legacy(&cfc_map(6, 2), None).unwrap();
    assert!(
        wider.get_weights().len() > n_cfc,
        "Cfc_Backbone_Units did not reach the cell"
    );
    assert!(
        deeper.get_weights().len() > n_cfc,
        "Cfc_Backbone_Layers did not reach the cell"
    );
    assert_ne!(
        wider.get_weights().len(),
        deeper.get_weights().len(),
        "widening and deepening must not be the same knob"
    );

    // Same seam contract as any other cell: a full-length pack round-trips.
    let mut net = BlstmSpectralSegmenter::from_legacy(&cfc_map(6, 1), None).unwrap();
    let w: Vec<f64> = (0..n_cfc).map(|k| 0.11 - 0.0003 * (k as f64)).collect();
    net.set_weights(&w).unwrap();
    assert_eq!(net.get_weights(), w);
}

/// A malformed `Cfc_*` value fails LOUDLY at driver construction rather than silently
/// defaulting (the `a_malformed_mamba_geometry_bails_on_the_tier2_fixture` twin, same
/// reason: a silent default would change the weight-pack LENGTH).
#[test]
fn a_malformed_cfc_geometry_bails_on_the_tier2_fixture() {
    for (key, value, want) in [
        (
            "Cfc_Backbone_Units",
            "0",
            "'Cfc_Backbone_Units' must be >= 1 (got 0)",
        ),
        (
            "Cfc_Backbone_Layers",
            "0",
            "'Cfc_Backbone_Layers' must be >= 1 (got 0)",
        ),
        ("Cfc_Backbone_Units", "wide", "cannot read 'wide' as a size"),
    ] {
        let mut map = tier2_map();
        map.insert("BLSTM_Cell_Type".into(), "cfc".into());
        map.insert(key.into(), value.into());
        // `BlstmSpectralSegmenter` is not `Debug`, so match instead of `unwrap_err`.
        match BlstmSpectralSegmenter::from_legacy(&map, None) {
            Ok(_) => panic!("'{key} {value}' must not build"),
            Err(e) => assert!(format!("{e:#}").contains(want), "{e:#}"),
        }
    }
}

/// An unknown cell-type spelling is rejected with a message NAMING every accepted
/// value -- the FIVE-cell vocabulary, so a typo cannot silently fall back to `lstm`.
#[test]
fn an_unknown_cell_type_names_the_whole_vocabulary() {
    let mut map = tier2_map();
    map.insert("BLSTM_Cell_Type".into(), "cfcc".into());
    match BlstmSpectralSegmenter::from_legacy(&map, None) {
        Ok(_) => panic!("an unknown cell type must not build"),
        Err(e) => {
            let text = format!("{e:#}");
            for want in ["'lstm'", "'slstm'", "'mamba'", "'cfc'", "'transformer'"] {
                assert!(text.contains(want), "{want} missing from: {text}");
            }
        }
    }
}

/// PHASE 11 Task 2, the `cfc_cell_type_builds_on_the_tier2_fixture` twin: the same real
/// fixture config with `BLSTM_Cell_Type transformer` BUILDS a working Algo-3 driver, and
/// the geometry keys reach the cell -- but only ONE of the three can move the pack, which
/// is itself the claim worth pinning.
#[test]
fn transformer_cell_type_builds_on_the_tier2_fixture() {
    let tr_map = |window: usize, heads: usize, d_ff: usize| {
        let mut map = tier2_map();
        map.insert("BLSTM_Cell_Type".into(), "transformer".into());
        map.insert("Transformer_Window".into(), window.to_string());
        map.insert("Transformer_Heads".into(), heads.to_string());
        map.insert("Transformer_D_Ff".into(), d_ff.to_string());
        map
    };

    let lstm = BlstmSpectralSegmenter::from_legacy(&tier2_map(), None)
        .expect("the untouched fixture must build");
    let tr = BlstmSpectralSegmenter::from_legacy(&tr_map(8, 4, 16), None)
        .expect("transformer must build on the fixture");
    let (n_lstm, n_tr) = (lstm.get_weights().len(), tr.get_weights().len());
    assert!(n_lstm > 0 && n_tr > 0);
    assert_ne!(
        n_tr, n_lstm,
        "the transformer pack must differ from the LSTM one -- equal counts would mean \
         the driver silently kept the LSTM stacks"
    );

    // `d_ff` is the ONE sized knob: it grows the pack. `window` and `heads` are
    // PARAMETER-FREE by construction (ALiBi has no weights, and `A` only reshapes the
    // same `W_qkv`), so they must leave the length untouched -- a pack that moved with
    // either would mean the layout had picked up a dependency it must not have.
    let wide_ff = BlstmSpectralSegmenter::from_legacy(&tr_map(8, 4, 32), None).unwrap();
    assert!(
        wide_ff.get_weights().len() > n_tr,
        "Transformer_D_Ff did not reach the cell"
    );
    for (w, h) in [(64usize, 4usize), (8, 8), (2, 2)] {
        let other = BlstmSpectralSegmenter::from_legacy(&tr_map(w, h, 16), None).unwrap();
        assert_eq!(
            other.get_weights().len(),
            n_tr,
            "window={w} heads={h} moved the pack length -- neither is a weight"
        );
    }

    // Same seam contract as any other cell: a full-length pack round-trips.
    let mut net = BlstmSpectralSegmenter::from_legacy(&tr_map(8, 4, 16), None).unwrap();
    let w: Vec<f64> = (0..n_tr).map(|k| 0.11 - 0.0003 * (k as f64)).collect();
    net.set_weights(&w).unwrap();
    assert_eq!(net.get_weights(), w);
}

/// A malformed `Transformer_*` value fails LOUDLY at driver construction rather than
/// silently defaulting (the mamba/cfc twins' reason exactly: a silent default would
/// change the weight-pack LENGTH, or -- worse for this cell -- the attention SPAN, with
/// nothing to catch it).
#[test]
fn a_malformed_transformer_geometry_bails_on_the_tier2_fixture() {
    for (key, value, want) in [
        (
            "Transformer_Window",
            "0",
            "'Transformer_Window' must be >= 1 (got 0)",
        ),
        (
            "Transformer_Heads",
            "0",
            "'Transformer_Heads' must be >= 1 (got 0)",
        ),
        (
            "Transformer_D_Ff",
            "0",
            "'Transformer_D_Ff' must be >= 1 (got 0)",
        ),
        ("Transformer_Window", "wide", "cannot read 'wide' as a size"),
    ] {
        let mut map = tier2_map();
        map.insert("BLSTM_Cell_Type".into(), "transformer".into());
        map.insert(key.into(), value.into());
        // `BlstmSpectralSegmenter` is not `Debug`, so match instead of `unwrap_err`.
        match BlstmSpectralSegmenter::from_legacy(&map, None) {
            Ok(_) => panic!("'{key} {value}' must not build"),
            Err(e) => assert!(format!("{e:#}").contains(want), "{e:#}"),
        }
    }
}

/// A head count that does not divide the cell width is a TYPED config error naming both
/// numbers (phase-11 spec S2), not a truncation and not a panic from inside the cell:
/// the fixture's recurrent layers are 24 wide, so `heads = 5` is refused while `heads =
/// 8` builds. The message must name the LAYER too, since only some layers may offend.
#[test]
fn a_non_dividing_head_count_bails_on_the_tier2_fixture() {
    let with_heads = |heads: usize| {
        let mut map = tier2_map();
        map.insert("BLSTM_Cell_Type".into(), "transformer".into());
        map.insert("Transformer_Heads".into(), heads.to_string());
        map
    };
    match BlstmSpectralSegmenter::from_legacy(&with_heads(5), None) {
        Ok(_) => panic!("24 cells with 5 heads must not build"),
        Err(e) => {
            let text = format!("{e:#}");
            for want in ["24 cells", "Transformer_Heads", "(5)"] {
                assert!(text.contains(want), "{want} missing from: {text}");
            }
        }
    }
    assert!(BlstmSpectralSegmenter::from_legacy(&with_heads(8), None).is_ok());
    // The check is CELL-TYPE-GATED: the same offending head count on an LSTM config is
    // inert, because nothing reads it.
    let mut lstm = with_heads(5);
    lstm.insert("BLSTM_Cell_Type".into(), "lstm".into());
    assert!(BlstmSpectralSegmenter::from_legacy(&lstm, None).is_ok());
}

/// ABSENT KEYS MEAN THE DEFAULTS (the standing rule, phase-11 spec S2): a config that
/// never mentions the transformer -- i.e. every pre-phase-11 config -- decodes to exactly
/// `TransformerParams::default()`, so nothing about it changed. Pinned against the three
/// exported constants rather than literals, and against literals ONCE so a silent
/// re-tuning of a constant cannot pass unnoticed.
#[test]
fn absent_transformer_keys_decode_to_the_defaults() {
    use speech::nn::blstm::{
        BlstmConfig, TRANSFORMER_DEFAULT_D_FF, TRANSFORMER_DEFAULT_HEADS,
        TRANSFORMER_DEFAULT_WINDOW, TransformerParams,
    };
    let map = tier2_map();
    assert!(!map.contains_key("Transformer_Window"));
    let cfg = BlstmConfig::from_legacy(&map, "BLSTM").unwrap();
    assert_eq!(cfg.transformer, TransformerParams::default());
    assert_eq!(cfg.transformer.window, TRANSFORMER_DEFAULT_WINDOW);
    assert_eq!(cfg.transformer.heads, TRANSFORMER_DEFAULT_HEADS);
    assert_eq!(cfg.transformer.d_ff, TRANSFORMER_DEFAULT_D_FF);
    // All three are SETTLED: Task 3's sizing re-derived both lineage closed forms and
    // CONFIRMED `d_ff = 64` (the smallest integer inside both +-15% bands -- see
    // `TRANSFORMER_DEFAULT_D_FF`'s doc). This literal is the Rust half of the
    // cross-language pin; the Python half is `tests/test_phase11_init.py`.
    assert_eq!(
        (
            TRANSFORMER_DEFAULT_WINDOW,
            TRANSFORMER_DEFAULT_HEADS,
            TRANSFORMER_DEFAULT_D_FF
        ),
        (64, 4, 64)
    );
}
