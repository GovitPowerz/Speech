//! Issue #147: the fast nets' output bits across the f32 kernel change.
//!
//! The fixtures under `tests/reference_data/issue147/` hold the f32 outputs of the
//! bidirectional LSTM twin (`FastBlstm`) and of every causal cell (`FastCausalNet` and
//! its layer-0 `step`), measured on the commit BEFORE the per-output dot was replaced
//! by the k-outer block kernel, so the kernel change lands against them unmodified.
//!
//! What they own: the STEP BODIES. The kernel's own differential leg (in
//! `fast/kernels.rs`) compares one product with the per-output reference and cannot
//! see how a step combines its products (`(in + bias) + rec` against
//! `in + (bias + rec)` is a one-ULP difference), and the parity legs pin the fast tree
//! against the exact one at a resolution far above a ULP. Each case therefore records
//! the hidden rows as well as the posteriors: a posterior can round a one-ULP hidden
//! difference away.
//!
//! Inputs and packs are synthetic. Each causal cell runs twice: on its committed
//! phase9 forward config with its seed pack (recurrent `23,4`), and on a wider
//! two-layer geometry with a synthetic pack, whose projections are not a multiple of
//! the kernel's widest block. Both feed a full-width input and one narrower than the
//! net's input size (the width-tolerance crop, which sums fewer terms).
//!
//! The comparator is the standing oracle one: bit-exact on the oracle libm host, the
//! canary-gated hybrid bound elsewhere (every value read here passed through the
//! cells' transcendentals). Off the oracle host a one-ULP step-order change is
//! therefore owned by nothing.

mod common;

use std::path::PathBuf;

use common::assert_oracle_eq_f32;
use indexmap::IndexMap;

use speech::fast::cells::{CellGeometry, FastCausalNet};
use speech::fast::driver::build_aligned_spec;
use speech::fast::nn::{FastBlstm, FastMatrix};
use speech::nn::blstm::BlstmConfig;

/// Frames in one overlap window of the SAD training configs at 8 kHz.
const FRAMES: usize = 327;

/// Steps of the direct layer-0 `step` drive.
const STEPS: usize = 40;

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn fixture(name: &str) -> PathBuf {
    repo().join(format!("tests/reference_data/issue147/{name}.bin"))
}

fn load(config: &str) -> IndexMap<String, String> {
    speech::toml_config::load_config(repo().join(config).to_str().unwrap()).expect("config")
}

/// Deterministic values in `[-0.5, 0.5)`, row-major `rows x cols`, shifted by `salt`.
fn values(rows: usize, cols: usize, salt: usize) -> Vec<f32> {
    (0..rows * cols)
        .map(|idx| {
            let (t, j) = (idx / cols, idx % cols);
            (((t * 37 + j * 53 + 7 + salt * 11) % 101) as f64 / 101.0 - 0.5) as f32
        })
        .collect()
}

fn window(cols: usize, salt: usize) -> FastMatrix {
    FastMatrix {
        data: values(FRAMES, cols, salt),
        rows: FRAMES,
        cols,
    }
}

fn synthetic_pack(n: usize) -> Vec<f64> {
    (0..n)
        .map(|k| (((k * 11 + 3) % 97) as f64 / 97.0 - 0.5) * 0.2)
        .collect()
}

/// `FastBlstm` on the `prefix` net of a committed config, synthetic pack: the
/// posteriors, then the `[fwd | bwd]` hidden rows.
fn blstm_case(config: &str, prefix: &str) -> Vec<f32> {
    let map = load(config);
    let spec = build_aligned_spec(&map, prefix).expect("spec");
    let flat = synthetic_pack(speech::config::element_count(&spec));
    let mut net = FastBlstm::from_flat(&spec, &flat).expect("net");
    let hidden_rows = FRAMES / spec.lstm_subsampling.iter().product::<usize>();
    let mut out = net
        .feed_forward(&window(spec.lstm_neuron_nb[0], 0))
        .data
        .clone();
    assert!(!out.is_empty(), "{config}: empty output");
    out.extend_from_slice(net.debug_hidden(hidden_rows));
    out
}

/// The geometry override of a causal cell's wide case: two recurrent layers whose
/// projections leave every block width of the kernel a remainder. The transformer's
/// width must divide by its two heads.
fn widen(map: &mut IndexMap<String, String>, cell: &str) {
    let lstm = if cell == "transformer" {
        "23,6,10"
    } else {
        "23,7,5"
    };
    let last = lstm.rsplit(',').next().unwrap();
    map.insert("BLSTM_LSTMNeuronNb".into(), lstm.into());
    map.insert("BLSTM_LSTMSubSampling".into(), "4,1".into());
    map.insert("BLSTM_OutputNeuronNb".into(), format!("{last},3"));
    map.insert("BLSTM_OutputSubSampling".into(), "1".into());
}

/// One causal cell: the net's posteriors on a full-width window and on one three
/// columns narrower, then the layer-0 cell's `step` outputs over [`STEPS`] rows at its
/// input width and over [`STEPS`] rows three columns narrower.
fn causal_case(cell: &str, wide: bool) -> Vec<f32> {
    let base = format!("tests/reference_data/phase9/{cell}_forward");
    let mut map = load(&format!("{base}.config"));
    if wide {
        widen(&mut map, cell);
    }
    let bc = BlstmConfig::from_legacy(&map, "BLSTM").expect("net config");
    let spec = build_aligned_spec(&map, "BLSTM").expect("spec");
    let geometry = CellGeometry::from(&bc);
    let flat = if wide {
        synthetic_pack(FastCausalNet::element_count(&spec, bc.cell_type, &geometry).unwrap())
    } else {
        speech::io::binary::read_matrix(&repo().join(format!("{base}_seed.bin")))
            .expect("seed pack")
            .2
    };
    let mut net = FastCausalNet::from_flat(&spec, bc.cell_type, &geometry, &flat).expect("net");

    let in_cols = spec.lstm_neuron_nb[0];
    let mut out = Vec::new();
    for (cols, salt) in [(in_cols, 0), (in_cols - 3, 1)] {
        let post = net.feed_forward(&window(cols, salt));
        assert!(post.rows > 0, "{cell}: empty output");
        out.extend_from_slice(&post.data);
    }

    let cell0 = &net.cells()[0];
    let (i, o) = (cell0.input_size(), cell0.output_size());
    for (cols, salt) in [(i, 2), (i - 3, 3)] {
        let rows = values(STEPS, cols, salt);
        let mut state = cell0.state();
        let mut h = vec![0.0_f32; STEPS * o];
        for t in 0..STEPS {
            cell0.step(
                &rows[t * cols..(t + 1) * cols],
                &mut state,
                &mut h[t * o..(t + 1) * o],
            );
        }
        out.extend_from_slice(&h);
    }
    out
}

const CELLS: [&str; 5] = ["lstm", "slstm", "mamba", "cfc", "transformer"];

/// Every case, by fixture name.
fn cases() -> Vec<(String, Vec<f32>)> {
    let mut all = vec![
        (
            "blstm_v1".to_string(),
            blstm_case("configs/training/lre_sad.toml", "BLSTM"),
        ),
        (
            "blstm_v2".to_string(),
            blstm_case("configs/training/lre_sad_v2.toml", "BLSTM"),
        ),
        (
            "blstm_lid".to_string(),
            blstm_case("configs/lid/lid_blstm.toml", "BLSTM"),
        ),
    ];
    for cell in CELLS {
        all.push((format!("{cell}_seed"), causal_case(cell, false)));
        all.push((format!("{cell}_wide"), causal_case(cell, true)));
    }
    all
}

fn read_f32(name: &str) -> Vec<f32> {
    let bytes = std::fs::read(fixture(name)).unwrap_or_else(|e| panic!("fixture {name}: {e}"));
    assert_eq!(bytes.len() % 4, 0, "{name}: not a whole number of f32");
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}

#[test]
fn fast_net_outputs_keep_their_bits() {
    for (name, got) in cases() {
        assert!(
            got.iter().all(|v| v.is_finite()),
            "{name}: non-finite output"
        );
        let want = read_f32(&name);
        assert_eq!(got.len(), want.len(), "{name}: length");
        for (idx, (&g, &w)) in got.iter().zip(&want).enumerate() {
            assert_oracle_eq_f32(g, w, &format!("{name}[{idx}]"));
        }
    }
}

/// The cases are not degenerate: no case's output is one repeated value.
#[test]
fn the_fixtures_are_not_degenerate() {
    for (name, got) in cases() {
        let first = got[0].to_bits();
        assert!(
            got.iter().any(|v| v.to_bits() != first),
            "{name}: constant output"
        );
    }
}

/// The fixture writer. Run ONCE, on the commit the fixtures describe, with
/// `cargo test --release --test issue147_fast_bits write_fast_bit_fixtures -- --ignored`
/// from `src/rust/`; never as part of a suite run. It refuses to overwrite: delete a
/// fixture by hand to re-baseline it.
#[test]
#[ignore = "fixture writer: run once on the producing commit, see its doc"]
fn write_fast_bit_fixtures() {
    std::fs::create_dir_all(fixture("x").parent().unwrap()).unwrap();
    for (name, got) in cases() {
        let path = fixture(&name);
        assert!(
            !path.exists(),
            "{} exists; delete it by hand to re-baseline the fixture",
            path.display()
        );
        assert!(
            got.iter().all(|v| v.is_finite()),
            "{name}: non-finite output"
        );
        let bytes: Vec<u8> = got.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(&path, bytes).unwrap();
    }
}
