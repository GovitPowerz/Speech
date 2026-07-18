//! Phase 7 Task 2: `fast::nn::FastBlstm` f32 forward-core parity vs the exact f64
//! `nn::blstm::BlstmNetwork::feed_forward`.
//!
//! The fast path DIVERGES from exact BY DESIGN (f32 + faer's blocked reduction order
//! vs the exact ascending-loop f64). Tolerances here are MEASURED first (the
//! `MEASURE` prints, `cargo test -- --nocapture`), then pinned with >=10x headroom;
//! the measured values are recorded in each test's docstring. argmax identity is a
//! HARD assert (a flip on committed fixtures would be R1 STOP-and-adjudicate).

use indexmap::IndexMap;
use ndarray::Array2;
use speech::config::NnetSpec;
use speech::fast::nn::{
    FastBlstm, FastMatrix, asinh_f32, gates_fn_f32, logistic_f32, softmax_row_f32,
};
use speech::nn::activations::{gates_fn, logistic_fn, maxmin2_fn};
use speech::nn::blstm::{BlstmConfig, BlstmNetwork};

// ---------------------------------------------------------------------------
// Helpers.
// ---------------------------------------------------------------------------

/// Deterministic closed-form input `x[t,j] = ((t*a + j*b + seed) % 100)/100 - 0.5`.
fn make_input(rows: usize, cols: usize, a: usize, b: usize, seed: usize) -> Array2<f64> {
    Array2::from_shape_fn((rows, cols), |(t, j)| {
        ((t * a + j * b + seed) % 100) as f64 / 100.0 - 0.5
    })
}

fn to_fast(m: &Array2<f64>) -> FastMatrix {
    let (r, c) = m.dim();
    let flat: Vec<f64> = m.iter().copied().collect();
    FastMatrix::from_f64_rows(r, c, &flat)
}

/// A synthetic flat pack sized to `spec`, filled with a deterministic pseudo-random
/// body and a nonzero mean/std tail (same closed form as the phase-2 harness).
fn synth_flat(spec: &NnetSpec) -> Vec<f64> {
    let n = speech::config::element_count(spec);
    let mut flat = vec![0.0_f64; n];
    let tail = 2 * spec.lstm_neuron_nb[0];
    for (k, v) in flat.iter_mut().enumerate().take(n - tail) {
        *v = ((k * 11 + 3) % 97) as f64 / 97.0 - 0.5;
    }
    for j in 0..spec.lstm_neuron_nb[0] {
        flat[n - tail + j] = 0.1 * (j as f64 + 1.0);
        flat[n - tail + spec.lstm_neuron_nb[0] + j] = 1.0 + 0.05 * j as f64;
    }
    flat
}

/// Build the exact `BlstmNetwork` from a config map + prefix + flat weights.
fn build_exact(map: &IndexMap<String, String>, prefix: &str, flat: &[f64]) -> BlstmNetwork {
    let cfg = BlstmConfig::from_legacy(map, prefix).unwrap();
    let mut net = BlstmNetwork::from_config(cfg).unwrap();
    net.set_weights(flat).unwrap();
    net
}

/// Run BOTH paths on the same input and return `(exact_posteriors, fast_posteriors)`.
fn run_both(
    map: &IndexMap<String, String>,
    prefix: &str,
    spec: &NnetSpec,
    flat: &[f64],
    input: &Array2<f64>,
) -> (Array2<f64>, FastMatrix) {
    let mut exact = build_exact(map, prefix, flat);
    let out_len = {
        let mut l = input.nrows();
        for &r in &spec.lstm_subsampling {
            l /= r;
        }
        l
    };
    let out_cols = *spec.output_neuron_nb.last().unwrap();
    let mut exact_out = Array2::<f64>::zeros((out_len, out_cols));
    exact.feed_forward(input, &mut exact_out);

    let mut fast = FastBlstm::from_flat(spec, flat).unwrap();
    let fast_out = fast.feed_forward(&to_fast(input)).clone();

    assert_eq!(fast_out.rows, out_len, "fast row count");
    assert_eq!(fast_out.cols, out_cols, "fast col count");
    (exact_out, fast_out)
}

/// `(max_abs, max_per_frame_rel, argmax_mismatch_count)` between exact f64 posteriors
/// and fast f32 posteriors. Per-frame rel uses the frame's max |exact| as the scale
/// (floored at 1e-6), the natural scale for softmax/logistic rows.
fn measure(exact: &Array2<f64>, fast: &FastMatrix) -> (f64, f64, usize) {
    assert_eq!(exact.nrows(), fast.rows);
    assert_eq!(exact.ncols(), fast.cols);
    let mut max_abs = 0.0_f64;
    let mut max_rel = 0.0_f64;
    let mut mism = 0usize;
    for r in 0..fast.rows {
        let mut denom = 1e-6_f64;
        let mut frame_abs = 0.0_f64;
        let (mut ex_arg, mut ex_best) = (0usize, f64::NEG_INFINITY);
        let (mut fa_arg, mut fa_best) = (0usize, f32::NEG_INFINITY);
        for c in 0..fast.cols {
            let e = exact[[r, c]];
            let f = fast.get(r, c) as f64;
            frame_abs = frame_abs.max((e - f).abs());
            denom = denom.max(e.abs());
            if e > ex_best {
                ex_best = e;
                ex_arg = c;
            }
            if fast.get(r, c) > fa_best {
                fa_best = fast.get(r, c);
                fa_arg = c;
            }
        }
        max_abs = max_abs.max(frame_abs);
        max_rel = max_rel.max(frame_abs / denom);
        if ex_arg != fa_arg {
            mism += 1;
        }
    }
    (max_abs, max_rel, mism)
}

// ---------------------------------------------------------------------------
// Synthetic net configs (mirroring the phase-2 harness prefixes).
// ---------------------------------------------------------------------------

/// LSTM [3,4,2] sub [2,1], output [4,5,O] sub [1,1]; peepholes default true.
fn synth_map(prefix: &str, out_final: usize) -> IndexMap<String, String> {
    let mut m: IndexMap<String, String> = IndexMap::new();
    m.insert(format!("{prefix}_LSTMNeuronNb"), "3,4,2".into());
    m.insert(format!("{prefix}_LSTMSubSampling"), "2,1".into());
    m.insert(
        format!("{prefix}_OutputNeuronNb"),
        format!("4,5,{out_final}"),
    );
    m.insert(format!("{prefix}_OutputSubSampling"), "1,1".into());
    m.insert(format!("{prefix}_InputNormalizationType"), "0".into());
    m.insert(format!("{prefix}_TwoSweeps"), "false".into());
    m
}

fn synth_spec(out_final: usize) -> NnetSpec {
    NnetSpec {
        lstm_neuron_nb: vec![3, 4, 2],
        lstm_subsampling: vec![2, 1],
        output_neuron_nb: vec![4, 5, out_final],
        output_subsampling: vec![1, 1],
        input_size: 3,
        peepholes: [true; 6],
    }
}

// ---------------------------------------------------------------------------
// Activation unit pins (f32 vs f64 twin).
// ---------------------------------------------------------------------------

#[test]
fn activations_match_f64_twin_at_grid() {
    // Interior grid: f32 within ~1e-6 abs of the f64 twin (smooth functions).
    let grid = [
        -50.0, -10.0, -3.0, -1.0, -0.25, 0.0, 0.25, 1.0, 3.0, 10.0, 50.0, 88.0,
    ];
    for &x in &grid {
        let xf = x as f32;
        assert!(
            (gates_fn_f32(xf) - gates_fn(x) as f32).abs() <= 1e-6,
            "gates_fn mismatch at {x}: f32={} f64={}",
            gates_fn_f32(xf),
            gates_fn(x)
        );
        assert!(
            (logistic_f32(xf) - logistic_fn(x) as f32).abs() <= 1e-6,
            "logistic mismatch at {x}"
        );
        assert!(
            (asinh_f32(xf) - maxmin2_fn(x) as f32).abs() <= 1e-5,
            "asinh mismatch at {x}"
        );
    }
}

#[test]
fn activations_saturate_at_exp_limit_edges() {
    // Beyond the f32 exp limit (~88.72): gates(0.1x) saturates when |0.1x| exceeds
    // it; logistic saturates when |x| exceeds it. All finite, no NaN.
    assert_eq!(gates_fn_f32(1.0e4), 1.0, "gates saturates high");
    assert_eq!(gates_fn_f32(-1.0e4), 0.0, "gates saturates low");
    assert_eq!(logistic_f32(1.0e4), 1.0, "logistic saturates high");
    assert_eq!(logistic_f32(-1.0e4), 0.0, "logistic saturates low");
    // asinh grows without a guard; must stay finite for a large-but-representable arg.
    assert!(asinh_f32(1.0e30).is_finite());
}

#[test]
fn softmax_guard_prevents_nan_above_threshold() {
    // A logit above the f32 guard (80.0) would overflow exp -> inf/inf = NaN without
    // the max-subtraction. With the guard the row is finite, sums to 1, and the big
    // logit dominates.
    let pre = [100.0_f32, 0.0, -5.0];
    let mut out = [0.0_f32; 3];
    softmax_row_f32(&pre, &mut out);
    assert!(out.iter().all(|v| v.is_finite()), "softmax must be finite");
    let sum: f32 = out.iter().sum();
    assert!(
        (sum - 1.0).abs() <= 1e-5,
        "softmax row sums to 1, got {sum}"
    );
    assert!(out[0] > 0.99, "the 100.0 logit dominates");

    // Below the guard: matches a plain f64 softmax within f32 epsilon.
    let pre2 = [0.5_f32, -0.5, 1.0];
    let mut out2 = [0.0_f32; 3];
    softmax_row_f32(&pre2, &mut out2);
    let ex: Vec<f64> = {
        let e: Vec<f64> = pre2.iter().map(|&v| (v as f64).exp()).collect();
        let s: f64 = e.iter().sum();
        e.iter().map(|v| v / s).collect()
    };
    for c in 0..3 {
        assert!(
            (out2[c] as f64 - ex[c]).abs() <= 1e-6,
            "softmax below guard at {c}"
        );
    }
}

// ---------------------------------------------------------------------------
// Construction (narrowing) pin.
// ---------------------------------------------------------------------------

#[test]
fn from_flat_narrows_first_block_bit_exact() {
    // The FIRST block from_flat consumes is forward layer-0 input_weights (I x 4O
    // col-major, a direct narrow copy). Each stored f32 must bit-equal flat[i] as f32.
    // Uses the REAL tuple-A pack (33,671 elements) + its config spec.
    let (map, spec, flat) = real_tuple_a();
    let _ = &map;
    let fast = FastBlstm::from_flat(&spec, &flat).unwrap();
    let stored = fast.debug_forward_l0_input_weights();
    let i0 = spec.lstm_neuron_nb[0] * spec.lstm_subsampling[0]; // 92
    let o0 = spec.lstm_neuron_nb[1]; // 24
    assert_eq!(stored.len(), i0 * 4 * o0, "layer-0 input-weight block size");
    for (k, &s) in stored.iter().enumerate() {
        assert_eq!(
            s.to_bits(),
            (flat[k] as f32).to_bits(),
            "narrowing mismatch at flat[{k}]: stored={s} expected={}",
            flat[k] as f32
        );
    }
}

#[test]
fn from_flat_output_depends_only_on_f32_narrowing() {
    // Narrowing is the ONLY lossy step: a net built from flat and one built from the
    // f32-roundtripped flat (flat[i] as f32 as f64) produce BIT-IDENTICAL output.
    let (map, spec, flat) = real_tuple_a();
    let _ = &map;
    let flat_rt: Vec<f64> = flat.iter().map(|&x| x as f32 as f64).collect();

    let input = make_input(200, 23, 31, 17, 0);
    let mut a = FastBlstm::from_flat(&spec, &flat).unwrap();
    let out_a = a.feed_forward(&to_fast(&input)).clone();
    let mut b = FastBlstm::from_flat(&spec, &flat_rt).unwrap();
    let out_b = b.feed_forward(&to_fast(&input)).clone();

    assert_eq!(out_a.data.len(), out_b.data.len());
    for (i, (&x, &y)) in out_a.data.iter().zip(out_b.data.iter()).enumerate() {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "narrowing-roundtrip diverged at {i}"
        );
    }
}

#[test]
fn from_flat_rejects_mlp_and_short_pack() {
    let mut spec = synth_spec(3);
    spec.lstm_neuron_nb[0] = 0; // MLP marker
    assert!(
        FastBlstm::from_flat(&spec, &[0.0; 10]).is_err(),
        "MLP rejected"
    );

    let spec = synth_spec(3);
    let short = vec![0.0_f64; 5];
    assert!(
        FastBlstm::from_flat(&spec, &short).is_err(),
        "short pack rejected"
    );
}

// ---------------------------------------------------------------------------
// Synthetic-net posterior parity.
// ---------------------------------------------------------------------------

#[test]
fn synthetic_multiclass_posterior_parity() {
    // MEASURED (Apple Silicon dev box): max_abs=4.700e-8, max_per_frame_rel=1.263e-7
    // over 5 seeds x {cols 3 (no crop), 5 (crop)}. Pinned at 2e-6 (~15x headroom over
    // the measured worst); argmax identical (0 mismatches).
    const REL_PIN: f64 = 2.0e-6;
    let spec = synth_spec(3);
    let map = synth_map("SYN", 3);
    let flat = synth_flat(&spec);

    let mut worst_abs = 0.0_f64;
    let mut worst_rel = 0.0_f64;
    for (seed, cols) in [(0usize, 3usize), (7, 3), (13, 5), (21, 5), (100, 3)] {
        let input = make_input(40, cols, 29, 13, seed);
        let (exact, fast) = run_both(&map, "SYN", &spec, &flat, &input);
        let (abs, rel, mism) = measure(&exact, &fast);
        worst_abs = worst_abs.max(abs);
        worst_rel = worst_rel.max(rel);
        assert_eq!(
            mism, 0,
            "argmax must be identical (seed={seed}, cols={cols})"
        );
        assert!(
            rel < REL_PIN,
            "rel {rel} exceeds pin (seed={seed}, cols={cols})"
        );
    }
    println!("MEASURE synthetic_multiclass: max_abs={worst_abs:.3e} max_rel={worst_rel:.3e}");
}

#[test]
fn synthetic_binary_posterior_parity() {
    // Binary (output_size 1, Logistic). MEASURED (Apple Silicon dev box):
    // max_abs=4.361e-8, max_per_frame_rel=1.006e-7 over 4 seeds x {cols 3, 5}. Pinned
    // at 2e-6 (~20x headroom).
    const REL_PIN: f64 = 2.0e-6;
    let spec = synth_spec(1);
    let map = synth_map("SYB", 1);
    let flat = synth_flat(&spec);

    let mut worst_abs = 0.0_f64;
    let mut worst_rel = 0.0_f64;
    for (seed, cols) in [(0usize, 3usize), (7, 3), (13, 5), (21, 5)] {
        let input = make_input(40, cols, 29, 13, seed);
        let (exact, fast) = run_both(&map, "SYB", &spec, &flat, &input);
        let (abs, rel, _) = measure(&exact, &fast);
        worst_abs = worst_abs.max(abs);
        worst_rel = worst_rel.max(rel);
        assert!(
            rel < REL_PIN,
            "rel {rel} exceeds pin (seed={seed}, cols={cols})"
        );
    }
    println!("MEASURE synthetic_binary: max_abs={worst_abs:.3e} max_rel={worst_rel:.3e}");
}

// ---------------------------------------------------------------------------
// Real tuple-A net forward parity.
// ---------------------------------------------------------------------------

fn real_tuple_a() -> (IndexMap<String, String>, NnetSpec, Vec<f64>) {
    let cfg_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/reference_data/phase0/1_worker_1.config"
    );
    let bin_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/reference_data/phase0/NNweights_config1.bin"
    );
    let text = std::fs::read_to_string(cfg_path).unwrap();
    let map = speech::legacy_config::parse_legacy_config(&text);
    let spec = NnetSpec::from_legacy(&map, "BLSTM").unwrap();
    let flat = speech::io::binary::read_weight_vector(std::path::Path::new(bin_path)).unwrap();
    assert_eq!(flat.len(), 33_671);
    (map, spec, flat)
}

#[test]
fn real_tuple_a_forward_parity() {
    // The REAL 33,671-weight SAD net (binary Logistic output) on a 200x23 input, core
    // forward, both paths. MEASURED (Apple Silicon dev box): max_abs=1.222e-8,
    // max_per_frame_rel=8.355e-7 over 3 seeds + the 30-col crop-gate case. Pinned at
    // 1e-5 (~12x headroom).
    const REL_PIN: f64 = 1.0e-5;
    let (map, spec, flat) = real_tuple_a();

    let mut worst_abs = 0.0_f64;
    let mut worst_rel = 0.0_f64;
    for seed in [0usize, 5, 17] {
        let input = make_input(200, 23, 31, 17, seed);
        let (exact, fast) = run_both(&map, "BLSTM", &spec, &flat, &input);
        let (abs, rel, _) = measure(&exact, &fast);
        worst_abs = worst_abs.max(abs);
        worst_rel = worst_rel.max(rel);
        assert!(
            rel < REL_PIN,
            "real net rel {rel} exceeds pin (seed={seed})"
        );
    }
    // Also exercise the crop gate: a 30-col input (23 < 30 -> crop to 23).
    let wide = make_input(200, 30, 31, 17, 3);
    let (exact, fast) = run_both(&map, "BLSTM", &spec, &flat, &wide);
    let (abs, rel, _) = measure(&exact, &fast);
    worst_abs = worst_abs.max(abs);
    worst_rel = worst_rel.max(rel);
    assert!(rel < REL_PIN, "real net (crop) rel {rel} exceeds pin");
    println!("MEASURE real_tuple_a: max_abs={worst_abs:.3e} max_rel={worst_rel:.3e}");
}

// ---------------------------------------------------------------------------
// Workspace hygiene.
// ---------------------------------------------------------------------------

#[test]
fn feed_forward_twice_bit_identical() {
    // Workspace reuse must be clean: two feed_forward calls on the same input give
    // BIT-IDENTICAL f32 output (no stale-buffer contamination).
    let (map, spec, flat) = real_tuple_a();
    let _ = &map;
    let mut fast = FastBlstm::from_flat(&spec, &flat).unwrap();
    let input = to_fast(&make_input(200, 23, 31, 17, 0));

    let first = fast.feed_forward(&input).clone();
    let second = fast.feed_forward(&input).clone();
    assert_eq!(first.rows, second.rows);
    assert_eq!(first.cols, second.cols);
    for (i, (&a, &b)) in first.data.iter().zip(second.data.iter()).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "repeated call diverged at {i}");
    }

    // And a DIFFERENT-length input between the two identical calls (grows/reuses the
    // workspace) must not perturb the repeat.
    let other = to_fast(&make_input(88, 23, 7, 3, 9));
    let _ = fast.feed_forward(&other);
    let third = fast.feed_forward(&input).clone();
    for (i, (&a, &b)) in first.data.iter().zip(third.data.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "post-resize repeat diverged at {i}"
        );
    }
}
