//! Phase 9 Task 1: the port-only cell/direction config vocabulary (spec S6).
//!
//! Two things are pinned here:
//! - the four new `KEY_TABLE` rows (`BLSTM{,_LID}_Cell_Type` /
//!   `BLSTM{,_LID}_Direction`) round-trip in BOTH directions -- flat -> TOML puts
//!   them in their declared section (never `[legacy.raw]`), and TOML -> flat -> TOML
//!   is a fixed point;
//! - a real committed fixture config (`phase4a/tier2_spectral.config`, the Algo-3
//!   spectral SAD net) carrying `BLSTM_Cell_Type slstm` typed-bails at driver
//!   construction with the not-yet-implemented wording.
//!
//! REMOVE the `slstm` arm of `unimplemented_cell_type_bails_on_the_tier2_fixture`
//! when Task 2 lands `SlstmLayer` (and the `mamba` arm when Task 3 lands
//! `MambaLayer`) -- the bail is scaffolding, not a permanent contract.

use std::path::PathBuf;

use indexmap::IndexMap;
use speech::legacy_config::parse_legacy_config;
use speech::tasks::sad::BlstmSpectralSegmenter;
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
    map.insert("BLSTM_Cell_Type".into(), "lstm".into());
    map.insert("BLSTM_Direction".into(), "bidirectional".into());

    let toml = map_to_toml(&map);
    assert_eq!(toml_to_map(&toml).unwrap(), map);
}

/// A real fixture config asking for a cell Phase 9 has not landed yet fails LOUDLY
/// at construction, with the exact spec wording -- no silent fallback to LSTM.
#[test]
fn unimplemented_cell_type_bails_on_the_tier2_fixture() {
    for cell in ["slstm", "mamba"] {
        let mut map = tier2_map();
        map.insert("BLSTM_Cell_Type".into(), cell.into());
        // `BlstmSpectralSegmenter` is not `Debug`, so match instead of `unwrap_err`.
        match BlstmSpectralSegmenter::from_legacy(&map, None) {
            Ok(_) => panic!("cell type '{cell}' must not be constructible yet"),
            Err(e) => assert_eq!(
                format!("{e:#}"),
                format!("cell type '{cell}' not yet implemented")
            ),
        }
    }
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
