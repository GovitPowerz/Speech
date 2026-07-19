//! Phase 7 Task 1: criterion micro-benches over the EXISTING exact kernels
//! (`nn::layers::matmul_seq`, `features::fft::Gfft`, `features::mel::
//! MelFilterBank::apply_filter_bank`), on fixture-shaped inputs. Phase 7 Task 7
//! adds the three fast-path twins these target names were reserved for:
//! `faer_project_92x96` (vs `matmul_seq_92x96`), `realfft_1024` (vs
//! `gfft_1024`), `mel_apply_widened_f64` (vs `mel_apply_513x20`, HONESTLY named
//! -- see that bench's own doc comment for why it is not called
//! `mel_apply_f32`).
//!
//! The three EXACT shapes chain together, all read off the REAL
//! `tests/reference_data/phase4a/tier2_spectral.config` (Algo 3, the tier-2
//! golden net): `BLSTM_spectrum_order 10` -> a 1024-point GFFT
//! (`features/pipeline.rs::SpectralParams::derive`: `window_size = 1 <<
//! order`) -> a 513-column periodogram (`(1 << (order-1)) + 1`) -> a 20-bin
//! log-mel filterbank (`BLSTM_nb_bins 20`, `BLSTM_minMelFreq`/
//! `BLSTM_maxMelFreq`/`BLSTM_minFreq`/`BLSTM_maxFreq`). `matmul_seq_92x96` is
//! the SAME shape family `fast::nn::FastBlstm` batches through
//! `faer::linalg::matmul` ("whole-sequence frames x I * I x 4O"): 92 frames (a
//! representative per-file sequence length) x `BLSTM_NNetInputSize 23` into
//! `23 x 4*24` (4 gate blocks x the net's 24-wide hidden layer,
//! `BLSTM_LSTMNeuronNb 23,24,24`) -- the LSTM input-projection product
//! `nn/layers.rs::LstmLayer::feed_forward` computes via `matmul_seq`, and
//! `fast::nn::lstm_layer_forward` computes via `faer_project` at the identical
//! shape (see `bench_faer_project` below).
//!
//! The FAST twins call into the REAL `speech::fast` kernels (`faer_project`
//! promoted `pub` for this bench, matching Task 1's `matmul_seq` precedent; the
//! GFFT/mel twins construct the same `realfft`/`MelFilterBank` types the real
//! `fast::pipeline` module uses), never a bench-local reimplementation -- so
//! these numbers measure the actual dispatch path, not a lookalike.

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use ndarray::Array2;
use realfft::RealFftPlanner;
use speech::fast::nn::faer_project;
use speech::features::fft::Gfft;
use speech::features::mel::MelFilterBank;
use speech::nn::layers::matmul_seq;

// ---------------------------------------------------------------------------
// Exact kernels (Task 1).
// ---------------------------------------------------------------------------

fn bench_matmul_seq(c: &mut Criterion) {
    let a = Array2::from_shape_fn((92, 23), |(i, j)| (i as f64 * 0.01) + (j as f64 * 0.001));
    let b = Array2::from_shape_fn((23, 96), |(i, j)| (i as f64 * 0.002) - (j as f64 * 0.0005));
    c.bench_function("matmul_seq_92x96", |bch| {
        bch.iter(|| {
            std::hint::black_box(matmul_seq(
                std::hint::black_box(&a),
                std::hint::black_box(&b),
            ))
        })
    });
}

fn bench_gfft(c: &mut Criterion) {
    let gfft = Gfft::new(10); // p=10 -> 1 << 10 = 1024 complex points.
    let base: Vec<f64> = (0..2048).map(|i| (i as f64 * 0.0037).sin()).collect();
    c.bench_function("gfft_1024", |bch| {
        bch.iter_batched(
            || base.clone(),
            |mut data| {
                gfft.fft(std::hint::black_box(&mut data));
                std::hint::black_box(data)
            },
            criterion::BatchSize::SmallInput,
        )
    });
}

fn bench_mel_apply(c: &mut Criterion) {
    // Params read verbatim off tier2_spectral.config; spectrum_size = 512
    // (`bins - 1` where `bins == periodogram_length == 513`, the SAME
    // convention `features/pipeline.rs:544-563`'s real `build_mel_bank`
    // `MelFilterBank::new` call site uses) -- the periodogram INPUT itself stays 513 columns wide
    // (indices 0..=512), matching `mel_apply_513x20`'s name.
    let bank = MelFilterBank::new(
        186.1001763856132, // BLSTM_minMelFreq
        2502.400490895474, // BLSTM_maxMelFreq
        20,                // BLSTM_nb_bins
        100.0,             // BLSTM_minFreq
        3800.0,            // BLSTM_maxFreq
        8000.0,            // sample rate (the tier2/phase4d wav fixtures' rate)
        512,               // spectrum_size == bins - 1
        true,              // BLSTM_is_log_mel
        0,                 // nb_dct: 0 -- isolate apply_filter_bank from apply_dct
        false,             // ignore_first: irrelevant when nb_dct == 0
        0,                 // deltas_nb: 0 -- keep the log+mel branch delta-overwrite-free
        0,                 // delta_deltas_nb
    );
    let p = Array2::from_shape_fn((100, 513), |(i, j)| {
        1.0 + (i as f64 * 0.1) + (j as f64 * 0.01)
    });
    c.bench_function("mel_apply_513x20", |bch| {
        bch.iter(|| std::hint::black_box(bank.apply_filter_bank(std::hint::black_box(&p))))
    });
}

// ---------------------------------------------------------------------------
// Fast kernels (Task 7).
// ---------------------------------------------------------------------------

/// `faer_project` at the SAME 92x96 shape as `matmul_seq_92x96` (92 frames x
/// `BLSTM_NNetInputSize 23` into `23 x 4*24`), f32. Calls the real
/// `fast::nn::faer_project` -- the exact function every LSTM gate/dense
/// projection in `fast::nn` dispatches through -- not a bench-local faer call,
/// so this measures the real kernel's cost at this shape, comparable
/// one-to-one against `matmul_seq_92x96`'s f64 ascending-loop number.
fn bench_faer_project(c: &mut Criterion) {
    let t = 92usize;
    let in_cols = 23usize;
    let w_rows = 23usize;
    let w_cols = 96usize;

    // Row-major t x in_cols (in_stride == in_cols, no crop-gate padding).
    let in_data: Vec<f32> = (0..t * in_cols)
        .map(|idx| {
            let i = (idx / in_cols) as f32;
            let j = (idx % in_cols) as f32;
            (i * 0.01) + (j * 0.001)
        })
        .collect();
    // Column-major w_rows x w_cols (faer_project's mandated weight layout,
    // matching how FastLstmLayer::input_weights is stored).
    let w: Vec<f32> = (0..w_rows * w_cols)
        .map(|idx| {
            let row = (idx % w_rows) as f32;
            let col = (idx / w_rows) as f32;
            (row * 0.002) - (col * 0.0005)
        })
        .collect();
    let mut dst = vec![0.0_f32; t * w_cols];

    c.bench_function("faer_project_92x96", |bch| {
        bch.iter(|| {
            faer_project(
                std::hint::black_box(&in_data),
                t,
                in_cols,
                in_cols,
                std::hint::black_box(&w),
                w_rows,
                w_cols,
                std::hint::black_box(&mut dst),
            );
            std::hint::black_box(&dst);
        })
    });
}

/// One `realfft` real-to-complex forward transform at `window_size = 1024`
/// (the same `BLSTM_spectrum_order 10` window `gfft_1024` transforms), f32 --
/// the per-frame kernel `fast::pipeline::FastPipeline::compute_periodogram`
/// calls once per frame (`r2c.process_with_scratch`). NOTE the per-call
/// throughput asymmetry vs `gfft_1024`, not a bug: the exact GFFT packs TWO
/// real windowed frames into one 1024-POINT COMPLEX FFT (2048 f64 input
/// values), so `gfft_1024` amortizes one call over two frames; `realfft`
/// transforms ONE real 1024-SAMPLE frame per call into 513 complex bins (the
/// fast path's NO-TWO-REAL-PACKING divergence, documented in
/// `fast/pipeline.rs`'s module doc + its equivalence proof). A fair
/// per-FRAME comparison halves `gfft_1024`'s number (or doubles this one)
/// before comparing -- see RESULTS.md.
fn bench_realfft(c: &mut Criterion) {
    let mut planner = RealFftPlanner::<f32>::new();
    let r2c = planner.plan_fft_forward(1024);
    let mut input = r2c.make_input_vec();
    for (i, v) in input.iter_mut().enumerate() {
        *v = (i as f32 * 0.0037).sin();
    }
    let mut output = r2c.make_output_vec();
    let mut scratch = r2c.make_scratch_vec();

    c.bench_function("realfft_1024", |bch| {
        bch.iter_batched(
            || input.clone(),
            |mut inp| {
                r2c.process_with_scratch(
                    std::hint::black_box(&mut inp),
                    std::hint::black_box(&mut output),
                    std::hint::black_box(&mut scratch),
                )
                .expect("realfft process (fixed power-of-two length)");
                std::hint::black_box(&output);
            },
            BatchSize::SmallInput,
        )
    });
}

/// HONEST TWIN (T3 review obligation, recorded in the phase-7 ledger): the
/// fast path has NO f32 mel kernel. `fast::pipeline::FastPipeline::
/// build_input_sequence_parts` widens its f32 periodogram to a FRESH f64
/// `Array2` on EVERY call (a real per-call allocation, NOT a preallocated/
/// reused scratch buffer like the rest of that module's workspace -- see its
/// "MEL/DCT REUSE" doc note), then reuses the SAME golden f64
/// `apply_filter_bank` `mel_apply_513x20` benches. This target is deliberately
/// named `mel_apply_widened_f64`, not `mel_apply_f32` or similar -- there is
/// no fast-path-specific mel kernel to name that way, and fabricating one
/// would misrepresent the real dispatch. It measures exactly the real
/// procedure: fresh-allocate `perio64`, widen element-by-element, call
/// `apply_filter_bank`. Same bank params + 100x513 shape as `mel_apply_513x20`
/// so the delta between the two isolates the widen-and-allocate overhead,
/// nothing else (RESULTS.md's RSS reading attributes this per-call scratch
/// cost to this documented decision, not to an anomaly).
fn bench_mel_apply_widened_f64(c: &mut Criterion) {
    let bank = MelFilterBank::new(
        186.1001763856132,
        2502.400490895474,
        20,
        100.0,
        3800.0,
        8000.0,
        512,
        true,
        0,
        false,
        0,
        0,
    );
    let rows = 100_usize;
    let bins = 513_usize;
    let perio_f32: Vec<f32> = (0..rows * bins)
        .map(|idx| {
            let i = (idx / bins) as f32;
            let j = (idx % bins) as f32;
            1.0 + (i * 0.1) + (j * 0.01)
        })
        .collect();

    c.bench_function("mel_apply_widened_f64", |bch| {
        bch.iter(|| {
            let mut perio64 = Array2::<f64>::zeros((rows, bins));
            for r in 0..rows {
                let base = r * bins;
                for col in 0..bins {
                    perio64[[r, col]] = perio_f32[base + col] as f64;
                }
            }
            std::hint::black_box(bank.apply_filter_bank(std::hint::black_box(&perio64)))
        })
    });
}

criterion_group!(
    kernels,
    bench_matmul_seq,
    bench_gfft,
    bench_mel_apply,
    bench_faer_project,
    bench_realfft,
    bench_mel_apply_widened_f64
);
criterion_main!(kernels);
