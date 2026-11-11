//! Issue #146: port-truth pins of the four new cells' backward, at the net.
//!
//! The legacy peephole LSTM's backward is pinned bit for bit by the training goldens
//! (`phase4a_train_golden`, the Twin replay). The sLSTM, Mamba, CfC and transformer
//! cells had no such pin: `phase9_cell_grad.rs` checks them against central
//! differences, at a tolerance. These pins hold the weight derivatives of one
//! `BlstmNetwork` forward plus backward on the eight committed phase9 seam configs
//! (cell x direction, each with its seed pack), measured on the commit BEFORE the
//! stacks' layer-0 input deltas were gated off, so the gate lands against them
//! unmodified.
//!
//! The comparator is the standing oracle one: bit-exact on the oracle libm host, the
//! canary-gated hybrid bound elsewhere (every cell's backward reads transcendentals).

mod common;

use std::path::PathBuf;

use ndarray::{Array2, ShapeBuilder};
use speech::nn::blstm::{BlstmConfig, BlstmNetwork};

/// Input rows. Not a multiple of the configs' layer-0 sub-sampling (4), so the
/// container's floor (10 steps, 2 trailing frames dropped) is on the pinned path.
const ROWS: usize = 42;

fn reference(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/reference_data")
        .join(rel)
}

fn pin_name(cell: &str, direction: &str) -> String {
    format!("issue146/{cell}_{direction}_derivatives.bin")
}

/// The `N x 2` `[derivative | count]` matrix after one plain forward plus backward of
/// the phase9 `<cell>_<direction>` net on a deterministic input and target.
fn derivatives(cell: &str, direction: &str) -> Array2<f64> {
    let name = format!("{cell}_{direction}");
    let config = reference(&format!("phase9/{name}.config"));
    let map = speech::toml_config::load_config(config.to_str().unwrap()).unwrap();
    let cfg = BlstmConfig::from_legacy(&map, "BLSTM").unwrap();
    let in_cols = cfg.lstm_neuron_nb[0];
    let steps = ROWS / cfg.lstm_sub_sampling.iter().product::<usize>();
    let out_cols = *cfg.output_neuron_nb.last().unwrap();

    let mut net = BlstmNetwork::from_config(cfg).unwrap();
    let seed = reference(&format!("phase9/{name}_seed.bin"));
    let (_, _, pack) = speech::io::binary::read_matrix(&seed).unwrap();
    net.set_weights(&pack).unwrap();
    net.set_processing_type(false, false);

    let mut input = Array2::from_shape_fn((ROWS, in_cols), |(t, j)| {
        ((t * 37 + j * 53 + 7) % 101) as f64 / 101.0 - 0.5
    });
    let target = Array2::from_shape_fn((steps, out_cols), |(r, _)| (r % 2) as f64);
    let mut output = Array2::<f64>::zeros((steps, out_cols));
    net.feed_forward_backward(&mut input, 0, 0, &mut output, &target);
    net.get_weights_derivatives()
}

fn load_pin(cell: &str, direction: &str) -> Array2<f64> {
    let path = reference(&pin_name(cell, direction));
    let (rows, cols, data) = speech::io::binary::read_matrix(&path).unwrap();
    Array2::from_shape_vec((rows, cols).f(), data).unwrap()
}

fn assert_pinned(cell: &str) {
    for direction in ["bidirectional", "forward"] {
        let pin = load_pin(cell, direction);
        assert!(
            pin.column(0).iter().any(|v| *v != 0.0),
            "{cell} {direction}: the pin holds no gradient"
        );
        common::assert_oracle_eq(
            &derivatives(cell, direction),
            &pin,
            &format!("{cell} {direction} weight derivatives"),
        );
    }
}

#[test]
fn slstm_stack_derivatives_are_port_truth_pinned() {
    assert_pinned("slstm");
}

#[test]
fn mamba_stack_derivatives_are_port_truth_pinned() {
    assert_pinned("mamba");
}

#[test]
fn cfc_stack_derivatives_are_port_truth_pinned() {
    assert_pinned("cfc");
}

#[test]
fn transformer_stack_derivatives_are_port_truth_pinned() {
    assert_pinned("transformer");
}

/// The fixture writer for the four tests above. Run ONCE, on the commit the fixtures
/// describe, with `cargo test --test stack_derivative_pins write_stack_derivative_pins
/// -- --ignored` from `src/rust/`; never as part of a suite run. It refuses to
/// overwrite: delete a fixture by hand to re-baseline it.
#[test]
#[ignore = "fixture writer: run once on the producing commit, see its doc"]
fn write_stack_derivative_pins() {
    std::fs::create_dir_all(reference("issue146")).unwrap();
    for cell in ["slstm", "mamba", "cfc", "transformer"] {
        for direction in ["bidirectional", "forward"] {
            let path = reference(&pin_name(cell, direction));
            assert!(
                !path.exists(),
                "{} exists; delete it by hand to re-baseline the pin",
                path.display()
            );
            let m = derivatives(cell, direction);
            let data: Vec<f64> = m.t().iter().copied().collect();
            speech::io::binary::write_matrix(&path, m.nrows(), m.ncols(), &data).unwrap();
        }
    }
}
