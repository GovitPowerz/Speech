//! Phase 8 Task 3: `fast::stream::StreamOverlap` -- the streaming overlap engine
//! (window firing when the input span exists, per-row accumulation + counts, row
//! finalization when the covering count completes, the EOS partial-tail flush).
//!
//! The oracle is the offline fast `FastBlstm::feed_forward_overlap` on the WHOLE
//! (already-normalized) input sequence (spec S2: same kernels, only chunking differs).
//! Four legs:
//! - `overlap_rows_bit_equal_offline`: streamed emissions concatenated == the offline
//!   overlap output, BIT-IDENTICAL incl. the 0/0 = NaN uncovered rows, across dense-
//!   overlap AND gappy (uncovered-row) configs and divisible/non-divisible lengths.
//! - `overlap_chunk_invariance`: the same input streamed at 1 / 13 / 100 / 7 (non-
//!   divisor) row granularities -> all bit-identical (chunking changes timing, never
//!   arithmetic).
//! - `finalization_bound_holds`: pushing 1 row at a time, every output row's emission
//!   input-frame index <= its input frame (`row * ssr`) + 2*window_size + ssr slack
//!   (the structural full-window lookahead), with a non-vacuity lag floor.
//! - `eos_tail_matches_offline`: the `flush()` tail (the clamped/snapped partial
//!   windows) is non-empty and matches the corresponding offline tail rows bit-for-bit.
//!
//! The net + inputs are synthetic (arbitrary f32 -- the bit-equal contract is about the
//! WINDOWING, not "nice" values; the type-1 normalization is a per-row affine the caller
//! applies before push_rows, so it is out of this layer's scope, see the module docs).

use speech::config::{self, NnetSpec};
use speech::fast::nn::{FastBlstm, FastMatrix};
use speech::fast::stream::StreamOverlap;

// ---------------------------------------------------------------------------
// Synthetic net + inputs (mirroring phase7_fast_nn.rs' shapes).
// ---------------------------------------------------------------------------

/// LSTM [3,4,2] sub [2,1] (ssr 2), output [4,5,2] sub [1,1]; peepholes default true.
fn synth_spec() -> NnetSpec {
    NnetSpec {
        lstm_neuron_nb: vec![3, 4, 2],
        lstm_subsampling: vec![2, 1],
        output_neuron_nb: vec![4, 5, 2],
        output_subsampling: vec![1, 1],
        input_size: 3,
        peepholes: [true; 6],
    }
}

/// A synthetic flat pack sized to `spec`, deterministic body + a nonzero mean/std tail.
fn synth_flat(spec: &NnetSpec) -> Vec<f64> {
    let n = config::element_count(spec);
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

fn build_net() -> FastBlstm {
    let spec = synth_spec();
    FastBlstm::from_flat(&spec, &synth_flat(&spec)).unwrap()
}

/// Deterministic f32 feature rows `x[t,j] = ((t*a + j*b + seed) % 100)/100 - 0.5`.
fn make_input(rows: usize, cols: usize, a: usize, b: usize, seed: usize) -> FastMatrix {
    let mut data = Vec::with_capacity(rows * cols);
    for t in 0..rows {
        for j in 0..cols {
            data.push(((t * a + j * b + seed) % 100) as f32 / 100.0 - 0.5);
        }
    }
    FastMatrix { data, rows, cols }
}

/// Rows `[start, end)` of a row-major FastMatrix as an owned copy (a push chunk).
fn slice_rows(m: &FastMatrix, start: usize, end: usize) -> FastMatrix {
    let c = m.cols;
    FastMatrix {
        data: m.data[start * c..end * c].to_vec(),
        rows: end - start,
        cols: c,
    }
}

/// The offline oracle: `feed_forward_overlap` on the WHOLE input into a zeroed output of
/// `input.rows / ssr` rows (the driver's `real_vec_size`; mono channel 0 starts zeroed).
fn offline_overlap(input: &FastMatrix, ws: usize, shift: usize) -> FastMatrix {
    let mut net = build_net();
    let ssr = net.sub_sampling_ratio();
    let output_size = net.output_size();
    let mut output = FastMatrix::zeros(input.rows / ssr, output_size);
    net.feed_forward_overlap(input, ws, shift, &mut output);
    output
}

/// Stream `input` through a fresh `StreamOverlap` in `chunk`-row pushes, concatenating
/// every `push_rows` return plus the `flush` tail into one FastMatrix.
fn stream_all(input: &FastMatrix, ws: usize, shift: usize, chunk: usize) -> FastMatrix {
    let mut net = build_net();
    let ssr = net.sub_sampling_ratio();
    let output_size = net.output_size();
    let mut overlap = StreamOverlap::new(ws, shift, ssr, output_size);

    let cols = output_size;
    let mut data: Vec<f32> = Vec::new();
    let mut i = 0;
    while i < input.rows {
        let end = (i + chunk).min(input.rows);
        let got = overlap.push_rows(&slice_rows(input, i, end), &mut net);
        if got.rows > 0 {
            assert_eq!(got.cols, cols, "emitted col count stable");
            data.extend_from_slice(&got.data);
        }
        i = end;
    }
    let tail = overlap.flush(&mut net);
    if tail.rows > 0 {
        assert_eq!(tail.cols, cols, "flush col count stable");
        data.extend_from_slice(&tail.data);
    }
    let rows = data.len() / cols;
    FastMatrix { data, rows, cols }
}

/// Bit-for-bit f32 compare (NaN compares equal iff same bit pattern -- both paths compute
/// the identical `+0.0 / +0.0` on uncovered rows, so the NaN bits match).
fn assert_bits_eq(got: &FastMatrix, want: &FastMatrix, label: &str) {
    assert_eq!(got.rows, want.rows, "{label}: row count");
    assert_eq!(got.cols, want.cols, "{label}: col count");
    for r in 0..got.rows {
        for c in 0..got.cols {
            let a = got.get(r, c);
            let b = want.get(r, c);
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "{label}: mismatch at [{r},{c}]: got 0x{:08x} ({a}) want 0x{:08x} ({b})",
                a.to_bits(),
                b.to_bits()
            );
        }
    }
}

/// The test config matrix: (window_size, window_shift, n_rows, input_cols). Covers
/// dense overlap (gap-free) + gappy (shift >> window -> uncovered NaN rows), divisible
/// and non-divisible lengths, and a wider input (the feed_forward crop gate).
const CONFIGS: &[(usize, usize, usize, usize)] = &[
    (6, 3, 50, 3),  // dense overlap, divisible
    (6, 3, 51, 3),  // dense overlap, non-divisible
    (2, 10, 50, 3), // gappy: trailing NaN (last window unclamped)
    (2, 10, 51, 3), // gappy: clamped tail window covers the last row
    (5, 2, 64, 3),  // heavy overlap
    (6, 3, 50, 8),  // wider input -> feed_forward crop gate
    (7, 4, 37, 3),  // odd sizes
];

// ---------------------------------------------------------------------------
// (1) streamed == offline, bit-identical (incl. NaN).
// ---------------------------------------------------------------------------

#[test]
fn overlap_rows_bit_equal_offline() {
    for &(ws, shift, n, cols) in CONFIGS {
        let input = make_input(n, cols, 31, 17, 7);
        let offline = offline_overlap(&input, ws, shift);
        assert_eq!(offline.rows, n / 2, "sanity: offline rows == n/ssr");

        // A few chunk granularities -- each must match offline bit-for-bit.
        for &chunk in &[1usize, 7, 13, n] {
            let streamed = stream_all(&input, ws, shift, chunk.max(1));
            assert_bits_eq(
                &streamed,
                &offline,
                &format!("bit-equal ws={ws} shift={shift} n={n} cols={cols} chunk={chunk}"),
            );
        }
    }

    // Non-vacuity: at least one gappy config MUST contain a NaN row (else the 0/0 = NaN
    // leg of the contract is never exercised).
    let gappy = make_input(50, 3, 31, 17, 7);
    let off = offline_overlap(&gappy, 2, 10);
    assert!(
        off.data.iter().any(|v| v.is_nan()),
        "sanity: the gappy config must yield uncovered NaN rows"
    );
}

// ---------------------------------------------------------------------------
// (2) chunk invariance: 1 / 13 / 100 / 7 (non-divisor) -> bit-identical.
// ---------------------------------------------------------------------------

#[test]
fn overlap_chunk_invariance() {
    for &(ws, shift, n, cols) in CONFIGS {
        let input = make_input(n, cols, 23, 41, 3);
        let a = stream_all(&input, ws, shift, 1);
        let b = stream_all(&input, ws, shift, 13);
        let c = stream_all(&input, ws, shift, 100);
        let d = stream_all(&input, ws, shift, 7);
        let label = format!("chunk-invariance ws={ws} shift={shift} n={n} cols={cols}");
        assert_bits_eq(&a, &b, &format!("{label} 1-vs-13"));
        assert_bits_eq(&a, &c, &format!("{label} 1-vs-100"));
        assert_bits_eq(&a, &d, &format!("{label} 1-vs-7"));
    }
}

// ---------------------------------------------------------------------------
// (3) finalization bound: emission input-frame <= row*ssr + 2*window + ssr slack.
// ---------------------------------------------------------------------------

#[test]
fn finalization_bound_holds() {
    let (ws, shift, n) = (6usize, 3usize, 80usize);
    let mut net = build_net();
    let ssr = net.sub_sampling_ratio();
    let output_size = net.output_size();
    let input = make_input(n, 3, 31, 17, 5);

    let mut overlap = StreamOverlap::new(ws, shift, ssr, output_size);
    // Push 1 row at a time; record the input-frame index (total rows pushed) at which
    // each output row is emitted.
    let mut emit_at: Vec<usize> = Vec::new();
    let mut pushed = 0usize;
    for r in 0..n {
        pushed += 1;
        let got = overlap.push_rows(&slice_rows(&input, r, r + 1), &mut net);
        for _ in 0..got.rows {
            emit_at.push(pushed);
        }
    }
    let tail = overlap.flush(&mut net);
    for _ in 0..tail.rows {
        emit_at.push(pushed); // pushed == n at flush
    }

    assert_eq!(emit_at.len(), n / ssr, "emitted output-row count == n/ssr");
    for (orow, &idx) in emit_at.iter().enumerate() {
        let input_frame = orow * ssr;
        let bound = input_frame + 2 * ws + ssr;
        assert!(
            idx <= bound,
            "output row {orow} finalized at input-frame {idx} > bound {bound} \
             (input_frame {input_frame} + 2*window {} + ssr {ssr})",
            2 * ws
        );
    }

    // Non-vacuity: the lookahead is real (a row finalizes ~a full window after its input
    // frame), so the max lag is at least one window_size, not trivially 0.
    let max_lag = emit_at
        .iter()
        .enumerate()
        .map(|(orow, &idx)| idx as i64 - (orow * ssr) as i64)
        .max()
        .unwrap();
    assert!(
        max_lag >= ws as i64,
        "sanity: max finalization lag {max_lag} should be >= window {ws} (full-window lookahead)"
    );
}

// ---------------------------------------------------------------------------
// (4) EOS tail: flush() reproduces the offline clamp+snap tail rows.
// ---------------------------------------------------------------------------

#[test]
fn eos_tail_matches_offline() {
    // A config with a clamped tail window (n=51, gappy) so flush() genuinely fires the
    // partial-window clamp+snap path and emits a non-empty tail.
    let (ws, shift, n) = (6usize, 3usize, 51usize);
    let input = make_input(n, 3, 19, 29, 11);
    let offline = offline_overlap(&input, ws, shift);

    let mut net = build_net();
    let ssr = net.sub_sampling_ratio();
    let output_size = net.output_size();
    let mut overlap = StreamOverlap::new(ws, shift, ssr, output_size);

    // Feed all but the last few rows in one push, then the rest, capturing the flush tail
    // SEPARATELY -- it must be non-empty and must equal the offline tail rows bit-for-bit.
    let split = n - 5;
    let mut emitted = overlap.push_rows(&slice_rows(&input, 0, split), &mut net);
    let mut all: Vec<f32> = emitted.data.clone();
    let mut prefix_rows = emitted.rows;
    emitted = overlap.push_rows(&slice_rows(&input, split, n), &mut net);
    all.extend_from_slice(&emitted.data);
    prefix_rows += emitted.rows;

    let tail = overlap.flush(&mut net);
    assert!(
        tail.rows > 0,
        "flush() must emit a non-empty tail (the clamp+snap partial windows)"
    );

    // The tail rows are the offline output's last `tail.rows` rows.
    let total = prefix_rows + tail.rows;
    assert_eq!(total, offline.rows, "streamed total == offline rows");
    for r in 0..tail.rows {
        let orow = prefix_rows + r;
        for c in 0..output_size {
            let a = tail.get(r, c);
            let b = offline.get(orow, c);
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "EOS tail mismatch at output row {orow} col {c}: got {a} want {b}"
            );
        }
    }

    // And the whole concatenation matches offline (prefix + tail).
    all.extend_from_slice(&tail.data);
    let streamed = FastMatrix {
        rows: total,
        cols: output_size,
        data: all,
    };
    assert_bits_eq(&streamed, &offline, "eos-full-concat");
}
