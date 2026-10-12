//! Issue #27: the net-level weight layout mirrors `BlstmNetwork::set_weights`.
//!
//! The per-cell pins (`nn/cells/*`, `nn/layers.rs`) prove each layer's blocks sit where its
//! own `set_weights` lands them. This file pins the NET-level list: the stacks come in the order
//! `set_weights` threads them (forward, backward when bidirectional, output, tail), the
//! offsets tile the pack contiguously from 0, `sum(len) == nb_of_weights()` on every shape
//! (both directions, every cell, MLP mode, output sub-sampling), the tail is present in MLP
//! mode at the output net's input width, and the output layer's fans carry the
//! `OutputSubSampling` factor (the issue's battery item: dropping it is a RED here).

use indexmap::IndexMap;
use speech::nn::blstm::{BlstmConfig, BlstmNetwork};
use speech::nn::layout::{BlockKind, Stack, WeightBlock};

const IN: usize = 5;
const HIDDEN: usize = 4;
const CLASSES: usize = 3;

fn map(cell: &str, direction: &str) -> IndexMap<String, String> {
    let output_in = if direction == "forward" {
        HIDDEN
    } else {
        2 * HIDDEN
    };
    let mut m: IndexMap<String, String> = IndexMap::new();
    m.insert("X_LSTMNeuronNb".into(), format!("{IN},3,{HIDDEN}"));
    m.insert("X_LSTMSubSampling".into(), "2,1".into());
    m.insert("X_OutputNeuronNb".into(), format!("{output_in},{CLASSES}"));
    m.insert("X_OutputSubSampling".into(), "1".into());
    m.insert("X_InputNormalizationType".into(), "0".into());
    m.insert("X_TwoSweeps".into(), "false".into());
    if cell != "lstm" {
        m.insert("X_Cell_Type".into(), cell.into());
    }
    if direction != "bidirectional" {
        m.insert("X_Direction".into(), direction.into());
    }
    m.insert("Mamba_D_State".into(), "2".into());
    m.insert("Mamba_D_Conv".into(), "3".into());
    m.insert("Cfc_Backbone_Units".into(), "6".into());
    m.insert("Cfc_Backbone_Layers".into(), "2".into());
    m.insert("Transformer_Heads".into(), "1".into());
    m.insert("Transformer_D_Ff".into(), "7".into());
    m
}

fn net(m: &IndexMap<String, String>) -> BlstmNetwork {
    BlstmNetwork::from_config(BlstmConfig::from_legacy(m, "X").unwrap()).unwrap()
}

fn assert_tiles(layout: &[WeightBlock], nb: usize) {
    let mut offset = 0usize;
    for b in layout {
        assert_eq!(
            b.offset, offset,
            "{:?}/{}/{}",
            b.stack, b.layer, b.block.name
        );
        offset += b.len();
    }
    assert_eq!(offset, nb, "the layout does not tile nb_of_weights");
}

const CELLS: [&str; 5] = ["lstm", "slstm", "mamba", "cfc", "transformer"];

#[test]
fn every_cell_and_direction_tiles_nb_of_weights_in_set_weights_order() {
    for cell in CELLS {
        for direction in ["bidirectional", "forward"] {
            let m = map(cell, direction);
            let n = net(&m);
            let layout = n.weight_layout();
            assert_tiles(&layout, n.nb_of_weights());
            assert_eq!(
                layout,
                BlstmNetwork::weight_layout_of(BlstmConfig::from_legacy(&m, "X").unwrap()).unwrap(),
                "{cell}/{direction}: the from-config layout is the instance layout"
            );

            // Stack order: forward, [backward], output, tail -- each a contiguous run,
            // layers ascending inside it.
            let stacks: Vec<Stack> = layout.iter().map(|b| b.stack).collect();
            let mut runs: Vec<Stack> = Vec::new();
            for s in &stacks {
                if runs.last() != Some(s) {
                    runs.push(*s);
                }
            }
            let want = if direction == "forward" {
                vec![Stack::Forward, Stack::Output, Stack::Tail]
            } else {
                vec![Stack::Forward, Stack::Backward, Stack::Output, Stack::Tail]
            };
            assert_eq!(runs, want, "{cell}/{direction}");
            for s in &want {
                let layers: Vec<usize> = layout
                    .iter()
                    .filter(|b| b.stack == *s)
                    .map(|b| b.layer)
                    .collect();
                assert!(
                    layers.windows(2).all(|w| w[0] <= w[1]),
                    "{cell}/{direction}/{s:?}"
                );
            }
            // Two recurrent layers per stack, one output layer, the tail at layer 0.
            let max_layer = |s: Stack| {
                layout
                    .iter()
                    .filter(|b| b.stack == s)
                    .map(|b| b.layer)
                    .max()
            };
            assert_eq!(max_layer(Stack::Forward), Some(1), "{cell}/{direction}");
            assert_eq!(max_layer(Stack::Output), Some(0), "{cell}/{direction}");
            assert_eq!(max_layer(Stack::Tail), Some(0), "{cell}/{direction}");

            // The backward stack, when present, is the forward stack's twin shifted by it.
            if direction == "bidirectional" {
                let fwd: Vec<&WeightBlock> = layout
                    .iter()
                    .filter(|b| b.stack == Stack::Forward)
                    .collect();
                let bwd: Vec<&WeightBlock> = layout
                    .iter()
                    .filter(|b| b.stack == Stack::Backward)
                    .collect();
                let fwd_len: usize = fwd.iter().map(|b| b.len()).sum();
                assert_eq!(fwd.len(), bwd.len());
                for (f, b) in fwd.iter().zip(&bwd) {
                    assert_eq!(f.block, b.block, "{cell}");
                    assert_eq!(f.layer, b.layer);
                    assert_eq!(b.offset, f.offset + fwd_len);
                }
            }

            // Layer 0's first weight block reads the SUB-SAMPLED fan-in (5 * 2 = 10): the
            // net's own `Network::new` rule, not a re-derivation.
            let first = layout
                .iter()
                .find(|b| b.stack == Stack::Forward && b.block.kind == BlockKind::Weight)
                .unwrap();
            let fin = IN * 2;
            let want_fan_in = match cell {
                "lstm" | "slstm" => fin + 3, // the combined `in + out` of the gate map
                "cfc" => fin + 3,            // `[x | h]`, out = 3 at layer 0
                _ => fin,                    // the Mamba adapter / the transformer `W_a`
            };
            assert_eq!(
                first.block.fan_in, want_fan_in,
                "{cell}/{direction}: {}",
                first.block.name
            );

            // The tail: `mean` then `std`, each `IN` wide, kind `stat`.
            let tail: Vec<&WeightBlock> =
                layout.iter().filter(|b| b.stack == Stack::Tail).collect();
            assert_eq!(
                tail.iter()
                    .map(|b| (b.block.name.as_str(), b.block.kind, b.len()))
                    .collect::<Vec<_>>(),
                vec![("mean", BlockKind::Stat, IN), ("std", BlockKind::Stat, IN)],
                "{cell}/{direction}"
            );
        }
    }
}

/// The output MLP's fan-in is `OutputNeuronNb[i] * OutputSubSampling[i]` (issue #61): the
/// layout reads it off the built layer, so the shape AND the fans carry the factor, and the
/// pack still tiles.
#[test]
fn output_layer_fans_carry_the_sub_sampling_factor() {
    let mut m = map("lstm", "bidirectional");
    m.insert(
        "X_OutputNeuronNb".into(),
        format!("{},6,{CLASSES}", 2 * HIDDEN),
    );
    m.insert("X_OutputSubSampling".into(), "2,1".into());
    let n = net(&m);
    let layout = n.weight_layout();
    assert_tiles(&layout, n.nb_of_weights());
    let out: Vec<&WeightBlock> = layout.iter().filter(|b| b.stack == Stack::Output).collect();
    let rec: Vec<(usize, &str, Vec<usize>, usize, usize)> = out
        .iter()
        .map(|b| {
            (
                b.layer,
                b.block.name.as_str(),
                b.block.shape.clone(),
                b.block.fan_in,
                b.block.fan_out,
            )
        })
        .collect();
    assert_eq!(
        rec,
        vec![
            (0, "W", vec![6, 2 * HIDDEN * 2], 2 * HIDDEN * 2, 6),
            (0, "b", vec![6], 2 * HIDDEN * 2, 6),
            (1, "W", vec![CLASSES, 6], 6, CLASSES),
            (1, "b", vec![CLASSES], 6, CLASSES),
        ]
    );
}

/// MLP mode (`LSTMNeuronNb[0] == 0`): no recurrent stack, output blocks, and the tail is
/// STILL there, sized by the output net's input width -- exactly what `nb_of_weights` counts
/// and `set_weights` consumes in that mode (the issue text's "no tail" was wrong against the
/// code; the layout follows the code).
#[test]
fn mlp_mode_is_output_blocks_plus_the_tail_at_the_output_input_width() {
    let mut m = map("mamba", "bidirectional");
    m.insert("X_LSTMNeuronNb".into(), format!("0,{HIDDEN}"));
    m.insert("X_LSTMSubSampling".into(), "1".into());
    m.insert("X_OutputNeuronNb".into(), format!("7,{CLASSES}"));
    let n = net(&m);
    assert!(n.is_mlp());
    let layout = n.weight_layout();
    assert_tiles(&layout, n.nb_of_weights());
    let stacks: Vec<Stack> = layout.iter().map(|b| b.stack).collect();
    assert_eq!(
        stacks,
        vec![Stack::Output, Stack::Output, Stack::Tail, Stack::Tail]
    );
    assert_eq!(layout[0].block.shape, vec![CLASSES, 7]);
    assert_eq!(layout[2].len(), 7);
    assert_eq!(layout[3].len(), 7);
    assert_eq!(n.nb_of_weights(), CLASSES * 8 + 2 * 7);
}

/// The real phase-4d tuple-A shape (the 33671 pack): 13 blocks per LSTM layer, two layers
/// per stack, two stacks, two output layers, the tail -- and the peephole bundle carries its
/// 12 column names.
#[test]
fn tuple_a_lstm_layout_is_the_33671_pack() {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/reference_data/phase4d/tupleA_1_worker_1.config");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path:?}: {e}"));
    let m = speech::legacy_config::parse_legacy_config(&text);
    let n = net_prefixed(&m, "BLSTM");
    let layout = n.weight_layout();
    assert_tiles(&layout, 33671);
    assert_eq!(layout.len(), 2 * 2 * 13 + 2 * 2 + 2);
    let peeps: Vec<&WeightBlock> = layout
        .iter()
        .filter(|b| b.block.kind == BlockKind::Peephole)
        .collect();
    assert_eq!(peeps.len(), 4);
    for p in peeps {
        assert_eq!(p.block.shape, vec![24, 12]);
        assert_eq!(p.block.columns.as_ref().map(Vec::len), Some(12));
    }
    assert!(
        layout
            .iter()
            .filter(|b| b.block.kind != BlockKind::Peephole)
            .all(|b| b.block.columns.is_none())
    );
}

fn net_prefixed(m: &IndexMap<String, String>, prefix: &str) -> BlstmNetwork {
    BlstmNetwork::from_config(BlstmConfig::from_legacy(m, prefix).unwrap()).unwrap()
}
