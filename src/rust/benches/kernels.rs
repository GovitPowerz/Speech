//! Phase 7 Task 1: criterion micro-benches over the EXISTING exact kernels
//! (`nn::layers::matmul_seq`, `features::fft::Gfft`, `features::mel::
//! MelFilterBank::apply_filter_bank`), on fixture-shaped inputs -- no fast
//! path yet (Task 2+ add the f32/SIMD twins these three target NAMES are
//! reserved for: `matmul_seq_92x96`, `gfft_1024`, `mel_apply_513x20`).
//!
//! The three shapes chain together, all read off the REAL
//! `tests/reference_data/phase4a/tier2_spectral.config` (Algo 3, the tier-2
//! golden net): `BLSTM_spectrum_order 10` -> a 1024-point GFFT
//! (`features/pipeline.rs::SpectralParams::derive`: `window_size = 1 <<
//! order`) -> a 513-column periodogram (`(1 << (order-1)) + 1`) -> a 20-bin
//! log-mel filterbank (`BLSTM_nb_bins 20`, `BLSTM_minMelFreq`/
//! `BLSTM_maxMelFreq`/`BLSTM_minFreq`/`BLSTM_maxFreq`). `matmul_seq_92x96` is
//! the SAME shape family Task 2's `fast::nn::FastBlstm` batches through
//! `faer::linalg::matmul` ("whole-sequence frames x I * I x 4O"): 92 frames (a
//! representative per-file sequence length) x `BLSTM_NNetInputSize 23` into
//! `23 x 4*24` (4 gate blocks x the net's 24-wide hidden layer,
//! `BLSTM_LSTMNeuronNb 23,24,24`) -- the LSTM input-projection product
//! `nn/layers.rs::LstmLayer::feed_forward` computes via `matmul_seq`.

use criterion::{Criterion, criterion_group, criterion_main};
use ndarray::Array2;
use speech::features::fft::Gfft;
use speech::features::mel::MelFilterBank;
use speech::nn::layers::matmul_seq;

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
    // convention `features/pipeline.rs:559-572`'s real `MelFilterBank::new`
    // call site uses) -- the periodogram INPUT itself stays 513 columns wide
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

criterion_group!(kernels, bench_matmul_seq, bench_gfft, bench_mel_apply);
criterion_main!(kernels);
