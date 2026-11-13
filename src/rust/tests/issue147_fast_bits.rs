//! Issue #147: the fast nets' output bits across the f32 kernel change.
//!
//! The fixtures under `tests/reference_data/issue147/` hold the f32 outputs of the
//! bidirectional LSTM twin (`FastBlstm`) and of every causal cell (`FastCausalNet` and
//! its layer-0 `step`), measured on the commit BEFORE the per-output dot was replaced
//! by the k-outer block kernel, so the kernel change lands against them unmodified.
//!
//! What they own: the causal STEP BODIES. The kernel's own differential leg (in
//! `fast/kernels.rs`) compares one product with the per-output reference and cannot
//! see how a step combines its products (`(in + bias) + rec` against
//! `in + (bias + rec)` is a one-ULP difference), and the parity legs pin the fast tree
//! against the exact one at a resolution far above a ULP.
//!
//! Inputs and packs are synthetic. Each causal cell runs twice: on its committed
//! phase9 forward config with its seed pack (recurrent `23,4`), and on a wider
//! two-layer geometry with a synthetic pack, whose projections are not a multiple of
//! the kernel's widest block. Both feed a full-width input and one narrower than the
//! net's input size (the width tolerance: fewer terms, or the CfC's literal zero-pad).
//!
//! Two kinds of rows, compared differently ([`Rows`]):
//!
//! - a cell's own `step` outputs read the block kernel and the f32 libm (`expf`,
//!   `tanhf`, `log1pf`, `asinhf`), nothing else. They are bit-exact on the oracle libm
//!   host when its f32 libm also reproduces the f32 canaries recorded with these
//!   fixtures (`common::f32_libm_strict`, the f64 canaries probing none of those
//!   functions), and under the hybrid bound elsewhere, so there a one-ULP step-order
//!   change is owned by nothing;
//! - every posterior, and every `FastBlstm` row, passed through a faer product, whose
//!   bits are the host's SIMD lowering as well. Those compare under the hybrid bound on
//!   EVERY host: a bit-exact pin there would pin faer's lowering, the cost a
//!   same-precision self-golden was declined for in Phase 10 (`docs/validation.md`).
//!   They guard the layout and the wiring; the `FastBlstm` recurrence's order is the
//!   kernel leg's alone.

mod common;

use std::path::PathBuf;

use common::{
    LIBM_F32_CANARIES, assert_f32_libm_eq, assert_hybrid_eq_f32, libm_f32, read_libm_f32_canaries,
};
use indexmap::IndexMap;

use speech::fast::cells::{CellGeometry, FastCausalNet};
use speech::fast::driver::build_aligned_spec;
use speech::fast::nn::{FastBlstm, FastMatrix};
use speech::nn::blstm::BlstmConfig;

/// Frames in one overlap window of the SAD training configs at 8 kHz.
const FRAMES: usize = 327;

/// Steps of the direct layer-0 `step` drive.
const STEPS: usize = 40;

/// One run of fixture values, by what its bits depend on (the module doc).
enum Rows {
    /// A cell's own `step` outputs: the block kernel and libm.
    Step(Vec<f32>),
    /// Values downstream of a faer product.
    Faer(Vec<f32>),
}

impl Rows {
    fn values(&self) -> &[f32] {
        match self {
            Rows::Step(v) | Rows::Faer(v) => v,
        }
    }
}

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
/// posteriors, then the `[fwd | bwd]` hidden rows. Its input projection is a faer
/// product, so the hidden rows are [`Rows::Faer`] too.
fn blstm_case(config: &str, prefix: &str) -> Vec<Rows> {
    let map = load(config);
    let spec = build_aligned_spec(&map, prefix).expect("spec");
    let flat = synthetic_pack(speech::config::element_count(&spec));
    let mut net = FastBlstm::from_flat(&spec, &flat).expect("net");
    let hidden_rows = FRAMES / spec.lstm_subsampling.iter().product::<usize>();
    let posteriors = net
        .feed_forward(&window(spec.lstm_neuron_nb[0], 0))
        .data
        .clone();
    assert!(!posteriors.is_empty(), "{config}: empty output");
    vec![
        Rows::Faer(posteriors),
        Rows::Faer(net.debug_hidden(hidden_rows).to_vec()),
    ]
}

/// The geometry override of a causal cell's wide case: two recurrent layers whose
/// projections are not a multiple of the kernel's widest block (`4O` is 28 then 20 for
/// the two LSTMs). The transformer's width must divide by its two heads.
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
fn causal_case(cell: &str, wide: bool) -> Vec<Rows> {
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
        out.push(Rows::Faer(post.data.clone()));
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
        out.push(Rows::Step(h));
    }
    out
}

const CELLS: [&str; 5] = ["lstm", "slstm", "mamba", "cfc", "transformer"];

/// Every case, by fixture name. A fixture file is its case's runs, concatenated.
fn cases() -> Vec<(String, Vec<Rows>)> {
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
    for (name, runs) in cases() {
        let want = read_f32(&name);
        let total: usize = runs.iter().map(|r| r.values().len()).sum();
        assert_eq!(total, want.len(), "{name}: length");
        let mut at = 0;
        for (run, rows) in runs.iter().enumerate() {
            for (idx, &g) in rows.values().iter().enumerate() {
                assert!(g.is_finite(), "{name} run {run} [{idx}]: non-finite output");
                let label = format!("{name} run {run} [{idx}]");
                match rows {
                    Rows::Step(_) => assert_f32_libm_eq(g, want[at], &label),
                    Rows::Faer(_) => assert_hybrid_eq_f32(g, want[at], &label),
                }
                at += 1;
            }
        }
    }
}

/// The cases are not degenerate: no run's output is one repeated value.
#[test]
fn the_fixtures_are_not_degenerate() {
    for (name, runs) in cases() {
        for (run, rows) in runs.iter().enumerate() {
            let first = rows.values()[0].to_bits();
            assert!(
                rows.values().iter().any(|v| v.to_bits() != first),
                "{name} run {run}: constant output"
            );
        }
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
    for (name, runs) in cases() {
        let path = fixture(&name);
        assert!(
            !path.exists(),
            "{} exists; delete it by hand to re-baseline the fixture",
            path.display()
        );
        let bytes: Vec<u8> = runs
            .iter()
            .flat_map(|r| r.values())
            .inspect(|v| assert!(v.is_finite(), "{name}: non-finite output"))
            .flat_map(|v| v.to_le_bytes())
            .collect();
        std::fs::write(&path, bytes).unwrap();
    }
}

/// The f32 canary arguments, `(function, x)`, 256 per range over what the `step` bodies
/// feed each function: `exp` the stabilized exponents (`<= 0`, down to the f32 guard)
/// and the logistic's, `tanh` and `asinh` the pre-activations and cell states, `ln_1p`
/// the softplus's `exp` outputs, log-uniform in `[1e-30, 1]`. A golden-ratio walk
/// spreads each range without round numbers.
fn libm_f32_canary_args() -> Vec<(u32, f32)> {
    let ln_tiny = 1.0e-30_f64.ln();
    let ln_mid = 1.0e-3_f64.ln();
    let ranges: [(u32, f64, f64); 8] = [
        (0, -88.0, 0.0),
        (0, -10.0, 10.0),
        (1, -10.0, 10.0),
        (1, -1.0, 1.0),
        (2, ln_tiny, ln_mid),
        (2, ln_mid, 0.0),
        (3, -10.0, 10.0),
        (3, -1.0, 1.0),
    ];
    ranges
        .iter()
        .flat_map(|&(function, lo, hi)| {
            (0..256).map(move |k| {
                let v = lo + (hi - lo) * (0.5 + k as f64 * 0.618_033_988_749_895).fract();
                (function, if function == 2 { v.exp() } else { v } as f32)
            })
        })
        .collect()
}

/// The fixture holds the drawn arguments with finite results, so a truncated or edited
/// file fails here instead of silently relaxing `common::f32_libm_strict` to hybrid.
#[test]
fn the_f32_canaries_are_the_drawn_arguments() {
    let records = read_libm_f32_canaries();
    let args = libm_f32_canary_args();
    assert_eq!(
        records.len(),
        args.len(),
        "{LIBM_F32_CANARIES}: record count"
    );
    for (&(function, x, y), &(want_function, want_x)) in records.iter().zip(&args) {
        assert_eq!((function, x), (want_function, want_x.to_bits()));
        assert!(
            f32::from_bits(y).is_finite(),
            "function {function} at {x:#010x}"
        );
    }
}

/// The f32 canary writer: run ONCE, on the oracle libm host that wrote the step fixtures,
/// with `cargo test --release --test issue147_fast_bits write_f32_libm_canaries --
/// --ignored` from `src/rust/`. It refuses to overwrite.
#[test]
#[ignore = "fixture writer: run once on the oracle libm host, see its doc"]
fn write_f32_libm_canaries() {
    let path = repo().join("tests/reference_data").join(LIBM_F32_CANARIES);
    assert!(
        !path.exists(),
        "{} exists; delete it by hand to re-record the canaries",
        path.display()
    );
    let bytes: Vec<u8> = libm_f32_canary_args()
        .into_iter()
        .flat_map(|(function, x)| {
            let y = libm_f32(function, x);
            assert!(y.is_finite(), "function {function} at {x}: {y}");
            [function, x.to_bits(), y.to_bits()]
                .into_iter()
                .flat_map(u32::to_le_bytes)
        })
        .collect();
    std::fs::write(&path, bytes).unwrap();
}
