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
//! - a real committed fixture config (`phase4a/tier2_spectral.config`, the Algo-3
//!   spectral SAD net) BUILDS a working driver under every cell type -- `lstm`
//!   (default), `slstm` (Task 2), `mamba` (Task 3) and `cfc` (phase-10 Task 1).
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
    map.insert("BLSTM_Cell_Type".into(), "lstm".into());
    map.insert("BLSTM_Direction".into(), "bidirectional".into());
    map.insert("Mamba_D_State".into(), "16".into());
    map.insert("Mamba_Dt_Rank".into(), "0".into());
    map.insert("Cfc_Backbone_Units".into(), "24".into());

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
/// value -- the four-cell vocabulary, so a typo cannot silently fall back to `lstm`.
#[test]
fn an_unknown_cell_type_names_the_whole_vocabulary() {
    let mut map = tier2_map();
    map.insert("BLSTM_Cell_Type".into(), "cfcc".into());
    match BlstmSpectralSegmenter::from_legacy(&map, None) {
        Ok(_) => panic!("an unknown cell type must not build"),
        Err(e) => {
            let text = format!("{e:#}");
            for want in ["'lstm'", "'slstm'", "'mamba'", "'cfc'"] {
                assert!(text.contains(want), "{want} missing from: {text}");
            }
        }
    }
}
