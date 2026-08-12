//! INTERSTITIAL (phase 11): the symmetric length guard on `BlstmNetwork::set_weights`.
//!
//! THE ADJUDICATION THIS FILE PINS, in one sentence: the over-long tolerance belongs to
//! the FILE-LOAD path (where the legacy put it, and where it warns), NOT to the general
//! weight-setting method (where the legacy has no length logic at all).
//!
//! Legacy evidence:
//! - `BLSTMNeuralNetwork.cpp:209-225` (`setWeights`) carries NO length check whatsoever --
//!   it head-eats through the sub-networks and reads the normalize tail off whatever is
//!   left. Its only two callers are the ctor branch below and `updateWeights` (`:303-310`),
//!   which passes `getWeights()` back, i.e. always exactly `getNbOfWeights()`.
//! - `BLSTMNeuralNetwork.cpp:141-149` (the ctor's `_weightsFile` branch) is where the
//!   three-way length decision lives: SHORT -> `exit(1)`; LONG -> a console WARNING and
//!   `setWeights` anyway (head-first); EXACT -> `setWeights`.
//!
//! The port had hoisted that branch's over-long tolerance INTO `set_weights`, which
//! widened it to seams the legacy never had one on: `speech_rs.Engine.set_weights`,
//! `BagOfProcessors::set_weights`, and the `from_legacy(map, Some(flat))` driver ctors.
//! A pack of the wrong architecture then loaded head-first and RAN, with no warning
//! anywhere (phase-11 T3 concern 2: the tier2 33671-element `.bin` into the 28743-weight
//! transformer net, 4928 values dropped in silence).
//!
//! So: `set_weights` is EXACT-LENGTH in both directions, and `load_weights_file` keeps the
//! legacy's documented tolerance -- now applied EXPLICITLY (it slices the head itself
//! rather than leaning on a permissive callee) and no longer silent (the legacy's `:145`
//! warning, elided by the original port, is restored on stderr). Both halves are pinned
//! here so a later reader sees the boundary rather than inferring it.

use indexmap::IndexMap;
use speech::nn::blstm::{BlstmConfig, BlstmNetwork};

/// The v1 SAD topology (`LSTMNeuronNb 23,24,24` / MLP `48,12,1` / `NNetInputSize 23`),
/// which is what makes the transformer row below land on the committed 28743.
fn tier2_map() -> IndexMap<String, String> {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/reference_data/phase4a/tier2_spectral.config");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path:?}: {e}"));
    speech::legacy_config::parse_legacy_config(&text)
}

/// The five cell families on one fixture. Geometries are kept SMALL (not the defaults)
/// so the packs stay cheap -- the guard is length arithmetic, indifferent to the values.
fn cell_map(cell: &str) -> IndexMap<String, String> {
    let mut map = tier2_map();
    map.insert("BLSTM_Cell_Type".into(), cell.into());
    match cell {
        "mamba" => {
            map.insert("Mamba_D_State".into(), "4".into());
            map.insert("Mamba_D_Conv".into(), "3".into());
            map.insert("Mamba_Expand".into(), "1".into());
        }
        "cfc" => {
            map.insert("Cfc_Backbone_Units".into(), "8".into());
            map.insert("Cfc_Backbone_Layers".into(), "1".into());
        }
        "transformer" => {
            map.insert("Transformer_Window".into(), "8".into());
            map.insert("Transformer_Heads".into(), "4".into());
            map.insert("Transformer_D_Ff".into(), "16".into());
        }
        _ => {}
    }
    map
}

fn net(cell: &str) -> BlstmNetwork {
    let cfg = BlstmConfig::from_legacy(&cell_map(cell), "BLSTM")
        .unwrap_or_else(|e| panic!("{cell}: config must parse: {e}"));
    BlstmNetwork::from_config(cfg).unwrap_or_else(|e| panic!("{cell}: net must build: {e}"))
}

/// Distinct, non-degenerate values, so a head/tail confusion is visible.
fn ramp(n: usize) -> Vec<f64> {
    (0..n).map(|i| 0.5 + (i as f64) * 1e-3).collect()
}

const CELLS: [&str; 5] = ["lstm", "slstm", "mamba", "cfc", "transformer"];

/// EXACT loads, ONE TOO MANY refused, ONE TOO FEW refused -- per cell family. The
/// too-few leg is the pre-existing behaviour (kept, now symmetric); the too-many leg is
/// the flip.
#[test]
fn set_weights_demands_the_exact_pack_length_for_every_cell() {
    for cell in CELLS {
        let mut net = net(cell);
        let n = net.nb_of_weights();
        assert!(n > 0, "{cell}: degenerate net");

        // Exact: loads, and round-trips bit-for-bit.
        let exact = ramp(n);
        net.set_weights(&exact)
            .unwrap_or_else(|e| panic!("{cell}: the exact length must load: {e}"));
        let back = net.get_weights();
        assert_eq!(back.len(), n, "{cell}: get_weights width");
        for (i, (&a, &b)) in exact.iter().zip(back.iter()).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "{cell}: round trip at {i}");
        }

        // One too many: refused, and the message names BOTH numbers.
        let err = match net.set_weights(&ramp(n + 1)) {
            Ok(()) => panic!("{cell}: an over-long pack must be refused"),
            Err(e) => e.to_string(),
        };
        assert!(
            err.contains(&(n + 1).to_string()) && err.contains(&n.to_string()),
            "{cell}: the refusal must name both lengths, got: {err}"
        );

        // One too few: still refused (unchanged), same both-numbers style.
        let err = match net.set_weights(&ramp(n - 1)) {
            Ok(()) => panic!("{cell}: a short pack must be refused"),
            Err(e) => e.to_string(),
        };
        assert!(
            err.contains(&(n - 1).to_string()) && err.contains(&n.to_string()),
            "{cell}: the short refusal must name both lengths, got: {err}"
        );
    }
}

/// A REFUSED call must leave the net exactly as it was. This is not decoration: a guard
/// placed after the first sub-network's head-eat would half-write the net and leave a
/// silently corrupt one behind -- strictly worse than the silent load being fixed.
#[test]
fn a_refused_set_weights_leaves_the_net_untouched() {
    for cell in CELLS {
        let mut net = net(cell);
        let n = net.nb_of_weights();
        let good = ramp(n);
        net.set_weights(&good).unwrap();
        let before = net.get_weights();

        let mut over = ramp(n);
        over.iter_mut().for_each(|v| *v = -*v); // different VALUES, wrong length
        over.push(7.0);
        assert!(net.set_weights(&over).is_err());
        let under = vec![-1.0; n - 1];
        assert!(net.set_weights(&under).is_err());

        let after = net.get_weights();
        assert_eq!(after.len(), before.len(), "{cell}: width moved");
        for (i, (&a, &b)) in before.iter().zip(after.iter()).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "{cell}: partial write at {i}");
        }
    }
}

/// PHASE-11 T3 CONCERN 2, VERBATIM: the committed tier2 LSTM pack (33671) handed to a
/// DEFAULT-geometry transformer net (28743). Before the guard this loaded head-first and
/// dropped 4928 values in silence; now it is a typed refusal naming both numbers.
#[test]
fn the_tier2_lstm_pack_is_refused_by_a_default_geometry_transformer_net() {
    let mut map = tier2_map();
    map.insert("BLSTM_Cell_Type".into(), "transformer".into()); // DEFAULT geometry
    let cfg = BlstmConfig::from_legacy(&map, "BLSTM").unwrap();
    let mut net = BlstmNetwork::from_config(cfg).unwrap();
    assert_eq!(
        net.nb_of_weights(),
        28743,
        "the v1 closed form 16199 + 196*d_ff at d_ff = 64"
    );

    let pack = speech::io::binary::read_weight_vector(&bin_path()).unwrap();
    assert_eq!(pack.len(), 33671, "the committed tier2 LSTM pack");

    let err = net
        .set_weights(&pack)
        .expect_err("the wrong-architecture pack must be refused")
        .to_string();
    assert!(
        err.contains("33671") && err.contains("28743"),
        "the refusal must name both lengths, got: {err}"
    );
}

fn bin_path() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/reference_data/phase0/NNweights_config1.bin")
}

/// THE OTHER HALF OF THE ADJUDICATION, pinned so the boundary is visible rather than
/// inferred: the FILE-LOAD path KEEPS the legacy's head-first tolerance
/// (`BLSTMNeuralNetwork.cpp:144-146`). Same net, same pack as the test above -- refused
/// through `set_weights`, accepted through `load_weights_file`, deliberately.
///
/// This test is also the mutation detector for the EXPLICIT head slice: with
/// `set_weights` strict, a `load_weights_file` that still delegated its tolerance to the
/// callee would fail right here.
#[test]
fn the_file_load_path_keeps_the_legacy_head_first_tolerance() {
    let mut map = tier2_map();
    map.insert("BLSTM_Cell_Type".into(), "transformer".into());
    map.insert(
        "BLSTM_weightsFile".into(),
        bin_path().to_str().unwrap().to_string(),
    );
    let cfg = BlstmConfig::from_legacy(&map, "BLSTM").unwrap();
    let mut net = BlstmNetwork::from_config(cfg).unwrap();
    let needed = net.nb_of_weights();

    net.load_weights_file(&map, "BLSTM")
        .expect("the legacy tolerance: an over-long weightsFile loads its head");

    let pack = speech::io::binary::read_weight_vector(&bin_path()).unwrap();
    let back = net.get_weights();
    assert_eq!(back.len(), needed);
    for (i, (&a, &b)) in pack[..needed].iter().zip(back.iter()).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "head mismatch at {i}");
    }
}
