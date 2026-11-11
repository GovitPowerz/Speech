//! Phase 7 Task 1: criterion micro-benches over the EXISTING exact kernels
//! (`nn::layers::matmul_seq`, `features::fft::Gfft`, `features::mel::
//! MelFilterBank::apply_filter_bank`), on fixture-shaped inputs. Phase 7 Task 7
//! adds the three fast-path twins these target names were reserved for:
//! `faer_project_92x96` (vs `matmul_seq_92x96`), `realfft_1024` (vs
//! `gfft_1024`), `mel_apply_f32` (vs `mel_apply_513x20`). Phase 10 Task 5
//! RETARGETED the last of those from `mel_apply_widened_f64` (the phase-7
//! widen-then-f64-apply bridge, now deleted) onto the real f32 kernel
//! `fast::mel32::FastMelBank::apply_filter_bank` -- see that bench's doc.
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
//!
//! Issue #146 adds the first target above a kernel: `blstm_fwd_bwd_v1` /
//! `blstm_fwd_bwd_v2`, one `BlstmNetwork` forward plus backward over one window at
//! the two SAD lineages' geometry (see `bench_blstm_fwd_bwd`).

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use ndarray::Array2;
use realfft::RealFftPlanner;
use speech::fast::mel32::FastMelBank;
use speech::fast::nn::faer_project;
use speech::features::fft::Gfft;
use speech::features::mel::MelFilterBank;
use speech::nn::blstm::{BlstmConfig, BlstmNetwork};
use speech::nn::layers::matmul_seq;
use speech::nn::layout::Stack;

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

/// The bank both mel benches apply (f64 as is, f32 through `FastMelBank::from_exact`), so
/// the precision comparison cannot be handed two different banks. Params read verbatim off
/// tier2_spectral.config; spectrum_size = 512 (`bins - 1` where
/// `bins == periodogram_length == 513`, the SAME convention
/// `features::pipeline::build_mel_bank`, the real `MelFilterBank::new` call site, uses) --
/// the periodogram INPUT itself stays 513 columns wide (indices 0..=512), matching
/// `mel_apply_513x20`'s name.
fn tier2_mel_bank() -> MelFilterBank {
    MelFilterBank::new(
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
    )
}

fn bench_mel_apply(c: &mut Criterion) {
    let bank = tier2_mel_bank();
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

/// The REAL f32 mel kernel the fast path runs (`fast::mel32::FastMelBank::
/// apply_filter_bank`), at the SAME bank params + 100x513 shape as
/// `mel_apply_513x20`, so the delta between the two is a straight f32-vs-f64
/// KERNEL comparison -- same triangles, same ascending dot order, same
/// `ln(x + 1e-24)`, different precision and different element width.
///
/// PHASE 10 TASK 5 RETARGET. This bench used to be `mel_apply_widened_f64`: it
/// reproduced the phase-7 bridge (fresh-allocate a `T x bins` f64 `Array2`,
/// widen element-by-element, call the golden f64 `apply_filter_bank`) and was
/// honestly named that way BECAUSE there was no f32 mel kernel to name. There
/// is one now (`fast/mel32.rs`), the widen procedure is DELETED from the fast
/// tree, and both the old name and the old doc would have asserted the
/// opposite of the dispatch. Consequence for readers of older numbers: the
/// `mel_apply_widened_f64` row measured widen-plus-f64-apply and its delta vs
/// `mel_apply_513x20` isolated the widen-and-allocate overhead; this row
/// measures the f32 apply alone, so the two rows are NOT comparable across the
/// rename.
fn bench_mel_apply_f32(c: &mut Criterion) {
    // `bench_mel_apply`'s bank narrowed (this is the point of the comparison -- only
    // the kernel precision differs).
    let bank = FastMelBank::from_exact(&tier2_mel_bank());
    let rows = 100_usize;
    let bins = 513_usize;
    let perio_f32: Vec<f32> = (0..rows * bins)
        .map(|idx| {
            let i = (idx / bins) as f32;
            let j = (idx % bins) as f32;
            1.0 + (i * 0.1) + (j * 0.01)
        })
        .collect();

    c.bench_function("mel_apply_f32", |bch| {
        bch.iter(|| {
            std::hint::black_box(bank.apply_filter_bank(
                std::hint::black_box(&perio_f32),
                rows,
                bins,
            ))
        })
    });
}

// ---------------------------------------------------------------------------
// Net level (issue #146).
// ---------------------------------------------------------------------------

/// Frames in one overlap window of the SAD training configs at 8 kHz:
/// `frame_window 3.25`, `spectrum_shift 0.01` -> `2 * round(3.25 * 8000 / 2 / 80) + 1`
/// (`tasks/sad.rs::get_blstm_param`).
const SAD_WINDOW_FRAMES: usize = 327;

/// One `BlstmNetwork` forward plus backward over one window, the unit of work a
/// gradient fold repeats per window per file. The net is built from the committed
/// training config as written (geometry, cost law, normalization type), with two
/// overrides: backprop on (the Launcher sets it per fold) and a deterministic
/// synthetic pack. The plain path is driven directly, so the windowing drivers
/// around it are not in the number.
fn blstm_fwd_bwd(c: &mut Criterion, name: &str, config: &str) {
    let path = format!(
        "{}/../../configs/training/{config}",
        env!("CARGO_MANIFEST_DIR")
    );
    let mut map = speech::toml_config::load_config(&path).expect("committed training config");
    map.insert("BLSTM_BackPropagationActivated".into(), "true".into());
    let cfg = BlstmConfig::from_legacy(&map, "BLSTM").expect("the config's BLSTM net");
    let in_cols = cfg.lstm_neuron_nb[0];
    let out_rows = SAD_WINDOW_FRAMES / cfg.lstm_sub_sampling.iter().product::<usize>();
    let out_cols = *cfg.output_neuron_nb.last().unwrap();

    let mut net = BlstmNetwork::from_config(cfg).expect("net");
    let flat: Vec<f64> = (0..net.nb_of_weights())
        .map(|k| (((k * 11 + 3) % 97) as f64 / 97.0 - 0.5) * 0.2)
        .collect();
    net.set_weights(&flat).expect("pack length");
    net.set_processing_type(false, false);

    let input = Array2::from_shape_fn((SAD_WINDOW_FRAMES, in_cols), |(t, j)| {
        ((t * 37 + j * 53 + 7) % 101) as f64 / 101.0 - 0.5
    });
    let target = Array2::from_shape_fn((out_rows, out_cols), |(r, _)| (r % 2) as f64);

    // The backward must really run down to the gated cell (the forward stack's layer
    // 0), or the number is a forward plus an output-MLP backward.
    {
        let mut x = input.clone();
        let mut out = Array2::<f64>::zeros((out_rows, out_cols));
        net.feed_forward_backward(&mut x, 0, 0, &mut out, &target);
        let d = net.get_weights_derivatives();
        assert!(
            out.iter().all(|v| v.is_finite()),
            "{name}: non-finite output"
        );
        let layer0_end = net
            .weight_layout()
            .iter()
            .filter(|b| b.stack == Stack::Forward && b.layer == 0)
            .map(|b| b.offset + b.len())
            .max()
            .expect("a forward stack");
        assert!(
            d.column(0).iter().take(layer0_end).any(|v| *v != 0.0),
            "{name}: no gradient on the forward stack's layer 0"
        );
    }

    c.bench_function(name, |bch| {
        bch.iter_batched(
            || (input.clone(), Array2::<f64>::zeros((out_rows, out_cols))),
            |(mut x, mut out)| {
                net.reset_weights_derivatives();
                net.feed_forward_backward(
                    std::hint::black_box(&mut x),
                    0,
                    0,
                    &mut out,
                    std::hint::black_box(&target),
                );
                std::hint::black_box(out)
            },
            BatchSize::SmallInput,
        )
    });
}

fn bench_blstm_fwd_bwd(c: &mut Criterion) {
    blstm_fwd_bwd(c, "blstm_fwd_bwd_v1", "lre_sad.toml");
    blstm_fwd_bwd(c, "blstm_fwd_bwd_v2", "lre_sad_v2.toml");
}

criterion_group!(
    kernels,
    bench_matmul_seq,
    bench_gfft,
    bench_mel_apply,
    bench_faer_project,
    bench_realfft,
    bench_mel_apply_f32,
    bench_blstm_fwd_bwd
);
criterion_main!(kernels);
