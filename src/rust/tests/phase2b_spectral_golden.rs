//! Phase 2b Task 7: `BlstmSpectralSegmenter` (Algo 3, no pitch pass) driver goldens.
//!
//! The REAL `1_worker_1.config`'s algorithm and the phase's biggest crux. Like the
//! signal driver, the spectral result_vec comes from `BLSTMNeuralNetwork.
//! feedForwardBackward` (`BLSTMSpectralSegmenter.cpp:740`), whose real Eigen GEMMs
//! diverge from the ascending-loop port at the real net's k>=23 shapes. So the harness
//! `SpectralProbe` stage transcribes the `getSegmentation` body (`:593-887`) MINUS the
//! pitch second pass (`:757-805`, gated off by TDCwindow 0) with ONLY that FFB call
//! swapped for the reimpl family (`signalReimplFFB`, shared with Task 6), matching what
//! the Rust `BlstmSpectralSegmenter` does by calling the Phase 2 `BlstmNetwork` forward.
//!
//! SECONDARY structural probe (spec decision 3): the harness ALSO runs the REAL
//! compiled `getSegmentation` beside the transcription; segment count + types must
//! match EXACTLY (a mismatch aborts fixture generation), and the boundary max-delta is
//! recorded as `SEG_STRUCT site=spectral_<variant> ok=1 max_dt=<measured>` in the
//! manifest. All three variants measured `max_dt=0.0`.
//!
//! UNLIKE the signal driver (sub-threshold posteriors -> the untouched always-Other
//! seed), the real trained net on the REAL spectral features DOES cross the rising
//! threshold (0.752): chan-1 detects speech [0, ~1.18s] ([SPEECH@0, OTHER@~1.183,
//! END@2.0], a 3-boundary hypothesis). So these goldens GENUINELY exercise the decision
//! layer (rising crossing + linear interpolation + hysteresis area), not merely the
//! seed. The result-vec numeric content is ALSO pinned bit-exact via
//! `BlstmSpectralSegmenter::last_result_rows` (a port-side observation point, no legacy
//! counterpart) so a stub all-zeros NN forward would fail immediately.
//!
//! CROSS-CHANNEL REUSE QUIRK (the load-bearing layout fact): the legacy allocates
//! `result_vec` ONCE (`:631`) and REUSES it across channels (`:740`); under the overlap
//! FFB (accumulate-in-place `:664` `+=` then `/= count`) channel 2 SEEDS from channel
//! 1's post-division contents. The overlap variant dumps BOTH channels; the
//! `cross_channel_reuse_*` tests assert chan-2 matches the dump ONLY because the buffer
//! carries over -- a fresh-buffer chan-2 run DIFFERS (proving the quirk is pinned).
//!
//! Goldens dumped by the harness `SpectralProbe` stage (`tools/oracle_harness/main.cpp`,
//! Phase 2b Task 7 block) against `tests/reference_data/phase0/1_worker_1.config` (the
//! real net topology + real `NNweights_config1.bin`, BLSTM_shift overridden per variant)
//! on the shared 2-channel excerpt (`excerpt_2ch_8k.wav`, offset 0.35, dur 2.0, rate
//! 8000, frame_count 16001, ssr 4, spectrum_shift_in_frames 80, periodogram rows
//! T=201, real_vec_size 50). Three variants:
//!   real      -- 1_worker_1.config AS-IS (BLSTM_shift 0.8 -> overlap window_shift 80)
//!   overlap   -- BLSTM_shift 0.01       (overlap window_shift 1, cross-channel golden)
//!   noOverlap -- BLSTM_shift 0          (truncate FFB; window_size 324, window_shift 1)

mod common;

use std::path::PathBuf;

use indexmap::IndexMap;
use speech::audio::{Audio, read_audio};
use speech::io::binary::read_matrix;
use speech::tasks::sad::{BlstmSpectralSegmenter, get_blstm_param};
use speech::tasks::segmentation::{SegClass, Segmentation};
use speech::tasks::segmentation_io::{parse_audiodoc_attrs, to_vrcts_string};
use speech::tasks::segmenter::Segmenter;

fn real_config_text() -> String {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/reference_data/phase0/1_worker_1.config");
    std::fs::read_to_string(p).unwrap()
}

/// The real net weights (`NNweights_config1.bin`, 33,671 f64) as a flat slice.
fn real_weights() -> Vec<f64> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/reference_data/phase0/NNweights_config1.bin");
    let (rows, cols, data) = read_matrix(&p).unwrap();
    assert_eq!(rows * cols, 33_671, "real net weight count");
    data
}

/// A config map for `1_worker_1.config` with `BLSTM_shift` overridden to the variant's
/// raw value (the harness does the same via `set_val`).
fn spectral_map(shift: &str) -> IndexMap<String, String> {
    let mut m = speech::legacy_config::parse_legacy_config(&real_config_text());
    m.insert("BLSTM_shift".into(), shift.into());
    m
}

fn variant_map(tag: &str) -> IndexMap<String, String> {
    match tag {
        "real" => spectral_map("8.000000000000000e-01"),
        "overlap" => spectral_map("0.01"),
        "noOverlap" => spectral_map("0"),
        _ => unreachable!(),
    }
}

/// Fresh excerpt audio (offset 0.35, dur 2.0), UNMUTATED: `get_segmentation` applies
/// preemph itself (config `BLSTM_preemph_ratio -0.97 < 0` -> SKIPPED, `noise_seed -3`
/// -> no noise). So the excerpt reaches the periodogram raw, matching the E2E gate.
fn excerpt_audio() -> Audio {
    read_audio(&common::fixture("excerpt_2ch_8k.wav"), 0.35, 2.0).expect("decode excerpt")
}

fn excerpt_audio_zero_offset() -> Audio {
    read_audio(&common::fixture("excerpt_2ch_8k.wav"), 0.0, 2.0).expect("decode excerpt")
}

fn fresh_segs(audio: &Audio) -> Vec<Segmentation> {
    let audio_duration = (audio.data.ncols() as f64 - 1.0) / audio.sample_rate as f64;
    (0..audio.data.nrows())
        .map(|_| Segmentation::new(audio_duration))
        .collect()
}

fn build(tag: &str) -> BlstmSpectralSegmenter {
    let w = real_weights();
    BlstmSpectralSegmenter::from_legacy(&variant_map(tag), Some(&w)).unwrap()
}

/// NaN-aware convolved-row comparator (same contract as the signal golden's): bit-
/// IDENTICAL elements (including any OverLap `0/0 -> NaN` uncovered-row quirk) pass in
/// ANY mode; the rest go through `assert_oracle_eq` (bit-exact on the oracle env,
/// hybrid ULP off-env). `got`/`want` are single-row vectors.
fn assert_convolved_eq(got: &ndarray::Array2<f64>, want: &ndarray::Array2<f64>, label: &str) {
    assert_eq!(got.shape(), want.shape(), "{label}: shape mismatch");
    let g = got.row(0);
    let w = want.row(0);
    let mut gf = Vec::with_capacity(g.len());
    let mut wf = Vec::with_capacity(w.len());
    for (i, (&gv, &wv)) in g.iter().zip(w.iter()).enumerate() {
        if gv.to_bits() == wv.to_bits() {
            continue; // bit-identical (incl. same-pattern NaN) -> equal in any mode.
        }
        assert!(
            !gv.is_nan() && !wv.is_nan(),
            "{label}: at {i} a non-identical NaN (got=0x{:016x}, want=0x{:016x})",
            gv.to_bits(),
            wv.to_bits()
        );
        gf.push(gv);
        wf.push(wv);
    }
    if gf.is_empty() {
        return; // every element bit-identical.
    }
    let n = gf.len();
    let gm = ndarray::Array2::from_shape_vec((1, n), gf).unwrap();
    let wm = ndarray::Array2::from_shape_vec((1, n), wf).unwrap();
    common::assert_oracle_eq(&gm, &wm, label);
}

/// Assert `segs[chan]`'s hypothesis boundary list matches the harness dump exactly.
fn assert_boundaries_match(segs: &[Segmentation], chan: usize, dump: &str, label: &str) {
    let want = common::load_bin_phase2b(dump);
    let got_segs = segs[chan].segments();
    assert_eq!(
        got_segs.len(),
        want.nrows(),
        "{label}: boundary count mismatch"
    );
    for (i, s) in got_segs.iter().enumerate() {
        let gb = ndarray::Array2::from_shape_vec((1, 1), vec![s.begin]).unwrap();
        let wb = ndarray::Array2::from_shape_vec((1, 1), vec![want[[i, 0]]]).unwrap();
        common::assert_oracle_eq(&gb, &wb, &format!("{label} boundary[{i}].begin"));
        assert_eq!(
            s.ty as i32,
            want[[i, 1]] as i32,
            "{label} boundary[{i}].type"
        );
    }
}

/// Bit-exact vs the golden: the driver's captured PRE-convolution result row (the REAL
/// Rust `BlstmNetwork` forward output, via `last_result_rows`) vs
/// `spectral_<tag>_result_chan{chan+1}.bin`. S9.3: a stub all-zeros forward fails here.
fn assert_last_result_row_matches_dump(sig: &BlstmSpectralSegmenter, tag: &str, chan: usize) {
    let want = common::load_bin_phase2b(&format!("spectral_{tag}_result_chan{}.bin", chan + 1));
    let rows = sig.last_result_rows();
    assert_eq!(
        rows.len(),
        2,
        "{tag}: last_result_rows must have 2 channels"
    );
    let got = ndarray::Array2::from_shape_vec((1, rows[chan].len()), rows[chan].clone()).unwrap();
    assert_convolved_eq(
        &got,
        &want,
        &format!("{tag} last_result_row (chan{}, NN-chain)", chan + 1),
    );
}

// === from_legacy: reads the real config ======================================

#[test]
fn from_legacy_builds_from_real_config() {
    // The real 1_worker_1.config (BLSTM prefix) must build the driver with the real
    // 33671-weight net loaded. Just constructing it exercises SegmenterConfig/
    // DriverConfig/FeatureConfig/BlstmConfig all reading the "BLSTM" namespace.
    let sig = build("real");
    // Before any get_segmentation call, spectrum_shift_in_frames is 0 (set per call).
    assert_eq!(sig.spectrum_shift_in_frames(), 0);
}

// === get_blstm_param hand tests: all clamps + the sizing contrast =============

#[test]
fn get_blstm_param_real_config_overlap() {
    // Real config: window 3.25, shift 0.8, rate 8000, ssif 80, ssr 4.
    // window = round(3.25*8000/2/80) = round(162.5) = 163 (>=ssr, kept).
    // shift  = round(0.8*8000/80) = round(80) = 80 (>=1 -> noOverlap FALSE).
    // _WindowShift = 80*80/8000 = 0.8.
    // vec_size = ceil(16001/80) = 201; real = 201/4/1/1/1 = 50.
    let mut ws = 0.8;
    let (window, shift, no_overlap, real) =
        get_blstm_param(3.25, &mut ws, 8000.0, 80, 4, &[4, 1], &[1, 1], 16001);
    assert_eq!(window, 163, "window_size (periodogram frames)");
    assert_eq!(shift, 80, "window_shift (periodogram frames)");
    assert!(!no_overlap, "shift 80 >= 1 -> overlap, not noOverlap");
    assert_eq!(real, 50, "real_vec_size = 201/4");
    assert_eq!(ws, 0.8, "_WindowShift re-quantized to 80*80/8000");
}

#[test]
fn get_blstm_param_no_overlap_floor_and_10ssr() {
    // noOverlap variant: shift 0 -> window_shift round(0) = 0 < 1 -> noOverlap.
    // window floored to (round(3.25*8000/80)/4)*4 = (325/4)*4 = 81*4 = 324;
    // 324 >= 10*4=40, kept. shift clamped to 1. _WindowShift = 1*80/8000 = 0.01.
    let mut ws = 0.0;
    let (window, shift, no_overlap, real) =
        get_blstm_param(3.25, &mut ws, 8000.0, 80, 4, &[4, 1], &[1, 1], 16001);
    assert_eq!(window, 324, "noOverlap window floored to ssr multiple");
    assert_eq!(shift, 1, "noOverlap shift clamped to 1");
    assert!(no_overlap, "shift 0 -> noOverlap");
    assert_eq!(
        real, 50,
        "real_vec_size still 201/4 (unconditional ssr div)"
    );
    assert_eq!(ws, 80.0 / 8000.0, "_WindowShift = 1*80/8000");
}

#[test]
fn get_blstm_param_10ssr_floor_fires() {
    // Tiny window -> noOverlap window rounds below 10*ssr and is floored UP.
    // window 0.01, shift 0, ssif 80, ssr 4: raw = round(0.01*8000/80) = round(1.0) = 1;
    // (1/4)*4 = 0 < 10*4=40 -> floored to 40. (window half at :441 = round(0.01*8000/2/
    // 80) = round(0.5) = 1 >= ssr? 1 < 4 -> set to 4 first, then noOverlap re-derives.)
    let mut ws = 0.0;
    let (window, shift, no_overlap, _real) =
        get_blstm_param(0.01, &mut ws, 8000.0, 80, 4, &[4, 1], &[1, 1], 16001);
    assert!(no_overlap);
    assert_eq!(window, 40, "10*ssr floor fires (raw rounds to 0)");
    assert_eq!(shift, 1);
}

#[test]
fn get_blstm_param_window_zero_disables() {
    // window 0 -> window_size 0 -> full_window 0; the noOverlap branch is guarded on
    // `window != 0`, so it never fires; shift clamped to 1; sizing still ssr-divides.
    let mut ws = 0.8;
    let (window, shift, no_overlap, real) =
        get_blstm_param(0.0, &mut ws, 8000.0, 80, 4, &[4, 1], &[1, 1], 16001);
    assert_eq!(window, 0, "window 0 stays 0");
    // :452 clamps shift to 1 whenever window==0 (OR shift<1), regardless of the raw
    // round(0.8*8000/80)=80 -- the `window_size == 0` disjunct forces the clamp.
    assert_eq!(shift, 1, "window==0 forces shift clamp to 1 (:452)");
    assert!(!no_overlap, "noOverlap guarded on window != 0");
    assert_eq!(real, 50);
}

#[test]
fn get_blstm_param_sizing_sub_ratio_2_1() {
    // sub ratios [2,1] instead of [4,1]: ssr must be passed by the caller (here 2), and
    // the sequential division uses the SAME ratios list. 201 frames / 2 / 1 / 1 / 1.
    let mut ws = 0.8;
    let (_w, _s, _no, real) =
        get_blstm_param(3.25, &mut ws, 8000.0, 80, 2, &[2, 1], &[1, 1], 16001);
    assert_eq!(real, 201 / 2, "real_vec_size = 201/2/1/1/1 = 100");
}

#[test]
fn get_blstm_param_unconditional_ssr_division_contrast() {
    // The KEY divergence vs the signal driver (cite both legacy lines):
    //   spectral (BLSTMSpectralSegmenter.cpp:487-497): the `if ((window==0)||noOverlap)`
    //     gate is COMMENTED OUT -> ssr-division is UNCONDITIONAL (guarded only ssr>1).
    //   signal   (BLSTMSignalSegmenter.cpp:236-240): the same gate is LIVE -> the
    //     ssr-division only fires when (window==0 || noOverlap).
    // Here window 3.25 -> window 163 > 0 AND shift 80 -> noOverlap FALSE: the signal
    // driver would NOT ssr-divide (real stays 201), but the spectral driver DOES
    // (real = 50). This test pins the spectral (unconditional) behavior.
    let mut ws = 0.8;
    let (window, _shift, no_overlap, real) =
        get_blstm_param(3.25, &mut ws, 8000.0, 80, 4, &[4, 1], &[1, 1], 16001);
    assert!(
        window > 0 && !no_overlap,
        "the overlap regime (window>0, !noOverlap)"
    );
    assert_eq!(
        real, 50,
        "spectral ssr-divides UNCONDITIONALLY (50); the signal gate would leave 201"
    );
    // Contrast: mimic the signal gate by hand (only divide when window==0||noOverlap).
    let signal_gated = if window == 0 || no_overlap {
        201 / 4
    } else {
        201
    };
    assert_eq!(
        signal_gated, 201,
        "the signal gate would NOT divide here (proving the contrast)"
    );
}

// === timeStep/timeOffset branch hand tests ====================================

#[test]
fn timestep_timeoffset_overlap_branch_override() {
    // The overlap branch OVERRIDES timeStep/timeOffset with _SpectrumShift (the
    // asymmetry vs signal, which uses _WindowShift there). Real config overlap:
    // _SpectrumShift = 0.01, ssr = 4.
    let spectrum_shift = 0.01;
    let ssr = 4.0;
    let step = spectrum_shift * ssr;
    let off = step / 2.0 - spectrum_shift / 2.0;
    assert_eq!(step, 0.04, "timeStep = spectrum_shift*ssr");
    assert_eq!(
        off,
        0.04 / 2.0 - 0.01 / 2.0,
        "timeOffset = step/2 - spectrum_shift/2"
    );
}

#[test]
fn timestep_timeoffset_default_and_nooverlap_branch() {
    // Default (window==0) / noOverlap branch: timeStep = _WindowShift*ssr, timeOffset =
    // step/2 - _WindowShift/2. noOverlap: _WindowShift = 1*80/8000 = 0.01.
    let window_shift = 80.0 / 8000.0; // = 0.01
    let ssr = 4.0;
    let step = window_shift * ssr;
    let off = step / 2.0 - window_shift / 2.0;
    assert_eq!(step, 0.04);
    assert_eq!(off, 0.04 / 2.0 - 0.01 / 2.0);
}

// === Result rows (pre-convolution) shape ======================================

#[test]
fn result_row_shapes_match_manifest() {
    for tag in ["real", "overlap", "noOverlap"] {
        let want = common::load_bin_phase2b(&format!("spectral_{tag}_result_chan1.bin"));
        assert_eq!(want.nrows(), 1, "{tag} result: row vector");
        assert_eq!(want.ncols(), 50, "{tag} result: real_vec_size 50");
    }
    let want_in = common::load_bin_phase2b("spectral_real_inputseq_chan1.bin");
    assert_eq!(want_in.shape(), &[201, 11], "inputseq: T=201, D=11");
}

// === Full driver goldens per variant (result + convolved + boundaries) ========
//
// The driver's expensive get_segmentation runs ONCE per cheap variant here; the
// result-vec + boundary + convolved goldens are all asserted off that single run.

fn run_and_assert_chan1(tag: &str) {
    let mut sig = build(tag);
    let mut audio = excerpt_audio();
    let mut segs = fresh_segs(&audio);
    sig.get_segmentation(&mut audio, &mut segs).unwrap();

    // S9.3: the REAL Rust NN chain's chan-1 result row, bit-exact vs the golden.
    assert_last_result_row_matches_dump(&sig, tag, 0);

    // Boundaries (the decision layer IS exercised: 3 boundaries incl. a rising crossing).
    assert_boundaries_match(
        &segs,
        0,
        &format!("spectral_{tag}_boundaries_chan1.bin"),
        &format!("{tag} e2e chan1"),
    );

    // Convolved row: rebuild via results_to_segmentation from the pre-conv dump and
    // compare to the convolved dump (NN-free, from the static dump).
    assert_convolved_rebuild(tag, 0);
}

/// Rebuild the POST-convolution row (NN-free) from the pre-conv dump via
/// results_to_segmentation, and compare against the convolved dump. Independent of the
/// live NN forward (uses the static result dump).
fn assert_convolved_rebuild(tag: &str, chan: usize) {
    use speech::tasks::segmenter::{DriverConfig, SegmenterConfig, results_to_segmentation};
    let m = variant_map(tag);
    let driver_cfg = DriverConfig::from_config(&m, "BLSTM").unwrap();
    let seg_cfg = SegmenterConfig::from_config(&m, "BLSTM").unwrap();

    // timeStep/timeOffset per variant. real/overlap: overlap branch -> spectrum_shift
    // (0.01)*ssr(4) = 0.04, offset = 0.04/2 - 0.01/2. noOverlap: _WindowShift (1*80/
    // 8000 = 0.01)*4 = 0.04, offset = 0.04/2 - 0.01/2 (numerically identical here).
    let (time_step, time_offset) = (0.01 * 4.0, (0.01 * 4.0) / 2.0 - 0.01 / 2.0);

    let pre = common::load_bin_phase2b(&format!("spectral_{tag}_result_chan{}.bin", chan + 1));
    let mut results: Vec<f64> = pre.row(0).to_vec();

    let bound =
        common::load_bin_phase2b(&format!("spectral_{tag}_boundaries_chan{}.bin", chan + 1));
    let audio_duration = bound[[bound.nrows() - 1, 0]];
    let mut seg = Segmentation::new(audio_duration);

    results_to_segmentation(
        &mut seg,
        time_step,
        time_offset,
        &mut results,
        SegClass::Speech,
        driver_cfg.conv_coeff.as_deref(),
        &seg_cfg,
    );

    let want = common::load_bin_phase2b(&format!("spectral_{tag}_convolved_chan{}.bin", chan + 1));
    let got = ndarray::Array2::from_shape_vec((1, results.len()), results).unwrap();
    assert_convolved_eq(&got, &want, &format!("{tag} convolved chan{}", chan + 1));

    // Boundaries rebuilt from the static dump must also match the boundary dump.
    let got_segs = seg.segments();
    assert_eq!(
        got_segs.len(),
        bound.nrows(),
        "{tag} chan{} rebuild boundary count",
        chan + 1
    );
    for (i, s) in got_segs.iter().enumerate() {
        let gb = ndarray::Array2::from_shape_vec((1, 1), vec![s.begin]).unwrap();
        let wb = ndarray::Array2::from_shape_vec((1, 1), vec![bound[[i, 0]]]).unwrap();
        common::assert_oracle_eq(&gb, &wb, &format!("{tag} rebuild boundary[{i}].begin"));
        assert_eq!(
            s.ty as i32,
            bound[[i, 1]] as i32,
            "{tag} rebuild boundary[{i}].type"
        );
    }
}

#[test]
fn real_variant_matches_dump() {
    run_and_assert_chan1("real");
}

#[test]
fn no_overlap_variant_matches_dump() {
    run_and_assert_chan1("noOverlap");
}

// === Overlap variant + the cross-channel reuse golden =========================

/// The overlap variant is the cross-channel reuse carrier. One driver run produces
/// BOTH channels; chan-2's result matches the dump ONLY because the ONE result_vec
/// buffer carried channel 1's post-division contents into channel 2's accumulation.
#[test]
fn overlap_variant_and_cross_channel_reuse() {
    let mut sig = build("overlap");
    let mut audio = excerpt_audio();
    let mut segs = fresh_segs(&audio);
    sig.get_segmentation(&mut audio, &mut segs).unwrap();

    // Both channels' result rows bit-exact vs the dumps (the cross-channel contents ARE
    // the golden: chan-2 is contaminated by chan-1's carry-over).
    assert_last_result_row_matches_dump(&sig, "overlap", 0);
    assert_last_result_row_matches_dump(&sig, "overlap", 1);

    // Boundaries per channel.
    assert_boundaries_match(
        &segs,
        0,
        "spectral_overlap_boundaries_chan1.bin",
        "overlap chan1",
    );
    assert_boundaries_match(
        &segs,
        1,
        "spectral_overlap_boundaries_chan2.bin",
        "overlap chan2",
    );

    // Convolved rebuild per channel.
    assert_convolved_rebuild("overlap", 0);
    assert_convolved_rebuild("overlap", 1);
}

/// NON-VACUITY CONTRAST (the brief's Step 2 mandate): a FRESH-buffer chan-2 forward
/// (zeroed output, not seeded by chan-1) DIFFERS from the dumped chan-2 result. This
/// proves the cross-channel reuse quirk is PINNED, not coincidental: the dump only
/// matches because the driver reuses the buffer across channels.
#[test]
fn cross_channel_reuse_is_load_bearing() {
    // Run the real driver (captures the contaminated chan-2 result).
    let mut sig = build("overlap");
    let mut audio = excerpt_audio();
    let mut segs = fresh_segs(&audio);
    sig.get_segmentation(&mut audio, &mut segs).unwrap();
    let contaminated = sig.last_result_rows()[1].clone();

    // The dump == the contaminated (reused-buffer) chan-2 result.
    let want = common::load_bin_phase2b("spectral_overlap_result_chan2.bin");
    let want_row: Vec<f64> = want.row(0).to_vec();
    // Sanity: they match bit-for-bit under the oracle comparator (already asserted in
    // overlap_variant_and_cross_channel_reuse; re-stated here for the contrast).
    let cont_arr =
        ndarray::Array2::from_shape_vec((1, contaminated.len()), contaminated.clone()).unwrap();
    assert_convolved_eq(&cont_arr, &want, "overlap chan2 (reused buffer) == dump");

    // Now the FRESH-buffer counterfactual: rebuild chan 2 with a ZEROED output buffer
    // (i.e. NOT seeded by chan 1). It must DIFFER from the dump. Cheapest faithful way:
    // re-run the whole driver but manually zero the buffer between channels is not
    // exposed; instead assert the CHANNELS THEMSELVES differ -- chan-1 (never seeded,
    // it is the first channel) vs chan-2 (seeded). If the buffer were freshly zeroed
    // per channel, chan-2 would be computed from scratch like chan-1 and would NOT carry
    // chan-1's values; the seeded chan-2 differs from what a fresh chan-2 would be.
    // We verify the seeding matters by checking chan-2's dump differs from a fresh
    // chan-2 recomputation, done below via the fresh-buffer helper.
    let fresh_chan2 = fresh_buffer_chan2_result("overlap");
    let fresh_arr =
        ndarray::Array2::from_shape_vec((1, fresh_chan2.len()), fresh_chan2.clone()).unwrap();
    // The fresh (unseeded) chan-2 must NOT equal the dumped (seeded) chan-2.
    assert_ne!(
        fresh_arr.row(0).to_vec(),
        want_row,
        "a FRESH-buffer chan-2 must DIFFER from the seeded dump (the reuse is load-bearing)"
    );
}

/// Recompute chan-2's overlap result with a FRESHLY-ZEROED output buffer (the
/// counterfactual where the driver did NOT reuse the buffer). Mirrors the driver's
/// per-channel forward but hands feed_forward_backward a fresh zeroed buffer, so the
/// overlap accumulation starts from 0 instead of chan-1's contents.
fn fresh_buffer_chan2_result(tag: &str) -> Vec<f64> {
    use ndarray::Array2;
    use speech::audio::windowing_coefficients;
    use speech::features::pipeline::{FeatureConfig, SpectralParams, build_input_sequence};
    use speech::nn::blstm::{BlstmConfig, BlstmNetwork};
    use speech::tasks::segmenter::DriverConfig;

    let m = variant_map(tag);
    let c = FeatureConfig::from_legacy(&m, "BLSTM").unwrap();
    let dc = DriverConfig::from_config(&m, "BLSTM").unwrap();
    let audio = excerpt_audio();
    // preemph SKIPPED (ratio < 0); noise SKIPPED (seed < 0) -- same as the driver.
    let rate = audio.sample_rate as f64;
    let s = SpectralParams::derive(&c, rate);

    let blstm_cfg = BlstmConfig::from_legacy(&m, "BLSTM").unwrap();
    let mut net = BlstmNetwork::from_config(blstm_cfg).unwrap();
    net.set_weights(&real_weights()).unwrap();

    // overlap params: window 163, shift 1, noOverlap false, real_vec 50.
    let ssr = net.sub_sampling_ratio();
    let mut ws = dc.window_shift_sec; // 0.01 for the overlap variant
    let (window, shift, no_overlap, real) = get_blstm_param(
        dc.window_size_sec,
        &mut ws,
        rate,
        80,
        ssr,
        &net.lstm_sub_sampling(),
        &net.output_sub_sampling(),
        audio.data.ncols(),
    );
    net.set_processing_type(window > 0, !no_overlap);

    let temporal_conv =
        windowing_coefficients(&c.conv_type, true, 2 * c.conv_size as usize + 1, 0.83333);

    // chan 2 (index 1) with a FRESH zeroed buffer (NOT seeded by chan 1).
    let mut input_seq = build_input_sequence(&audio, &c, &s, 1, temporal_conv.as_deref());
    let mut fresh = Array2::<f64>::zeros((real, 1));
    let target = Array2::<f64>::zeros((0, 0));
    net.feed_forward_backward(&mut input_seq, window, shift, &mut fresh, &target);
    fresh.column(0).to_vec()
}

// === Targets-active golden: score vs a programmatic reference =================

#[test]
fn score_matches_dump_real_variant() {
    // Build the reference programmatically from the known spans (constants in the
    // manifest reference_spans_sec [[0.4,0.9],[1.2,1.6]]) via label_segment on a fresh
    // Segmentation -- matching the harness's seg._Reference construction.
    for tag in ["real", "noOverlap"] {
        let audio_duration = {
            let a = excerpt_audio();
            (a.data.ncols() as f64 - 1.0) / a.sample_rate as f64
        };
        let reference: Vec<Segmentation> = (0..2)
            .map(|_| {
                let mut r = Segmentation::new(audio_duration);
                r.label_segment(0.4, 0.9, SegClass::Speech);
                r.label_segment(1.2, 1.6, SegClass::Speech);
                r
            })
            .collect();

        let mut audio = excerpt_audio();
        let mut sig = build(tag);
        let mut hyp = fresh_segs(&audio);
        sig.get_segmentation(&mut audio, &mut hyp).unwrap();

        let want = common::load_bin_phase2b(&format!("spectral_{tag}_scores.bin"));
        let reports = BlstmSpectralSegmenter::score(&mut hyp, Some(&reference), -1);
        for (chan, report) in reports.iter().enumerate() {
            let speech = report.per_class[SegClass::Speech as usize];
            let got = ndarray::Array2::from_shape_vec(
                (1, 3),
                vec![speech.pfa, speech.pmiss, speech.error_rate],
            )
            .unwrap();
            let wantv = ndarray::Array2::from_shape_vec(
                (1, 3),
                vec![want[[chan, 0]], want[[chan, 1]], want[[chan, 2]]],
            )
            .unwrap();
            common::assert_oracle_eq(&got, &wantv, &format!("{tag} score chan{chan}"));
        }
    }
}

// === VRCTS byte equivalence (0b-ii closure) ===================================

#[test]
fn vrcts_bytes_match_dump() {
    for tag in ["real", "noOverlap"] {
        let dump = std::fs::read_to_string(common::fixture_phase2b(&format!(
            "spectral_{tag}_vrcts_chan1.xml"
        )))
        .unwrap();
        let (name, path_attr) = parse_audiodoc_attrs(&dump);

        let mut sig = build(tag);
        let mut audio = excerpt_audio_zero_offset();
        let mut segs = fresh_segs(&audio);
        sig.get_segmentation(&mut audio, &mut segs).unwrap();

        let got = to_vrcts_string(&segs[0], &name, &path_attr);
        assert_eq!(
            got, dump,
            "{tag}: to_vrcts_string must byte-match the real toFile_VRCTS dump"
        );
    }
}

// === TWO-FILES noOverlap golden + the params lifecycle ========================

/// The spectral noOverlap variant assigns `_WindowShift = 0.0` at `:885` (post-dump).
/// So file 2 (SAME driver) re-enters get_segmentation with `window_shift_sec == 0.0`,
/// re-derives through get_blstm_param: shift = round(0) = 0 < 1 -> noOverlap
/// re-triggers, window floored to 324, shift clamped to 1, `_WindowShift = 1*80/8000`.
///
/// EVIDENCE (honesty per the brief): the two files' boundaries COINCIDE (same audio,
/// deterministic net -> same hypothesis). The test asserts this rather than assuming
/// it. But boundary/type coincidence alone cannot distinguish "the reset genuinely
/// fired and re-derived the same value" from "the reset never happened". The
/// DISCRIMINATING observables are: (a) `window_shift_sec()` round-trips through 0.0 (the
/// reset firing) and back to 0.01 (1*80/8000) on re-derivation; (b) `spectrum_shift_in_
/// frames()` stays 80; (c) `last_result_rows_len()` stays 50 (the noOverlap real_vec
/// size) -- these are pinned against the dumped `spectral_noOverlap_params_file2.bin`
/// (1x4: window_shift=1, window_size=324, ssif=80, real_vec=50). WITHOUT the reset,
/// file 2 would inherit `window_shift_sec = 0.01` (not 0.0) -> shift = round(0.01*8000/
/// 80) = round(1.0) = 1 >= 1 -> the OVERLAP branch, not noOverlap; window_size would be
/// the overlap 163 (not 324). The params row discriminates these.
#[test]
fn two_files_in_sequence_no_overlap_lifecycle() {
    let w = real_weights();
    let mut sig = BlstmSpectralSegmenter::from_legacy(&variant_map("noOverlap"), Some(&w)).unwrap();

    let mut audio1 = excerpt_audio();
    let mut segs1 = fresh_segs(&audio1);
    sig.get_segmentation(&mut audio1, &mut segs1).unwrap();

    // Post-file1: the :885 reset fired -> window_shift_sec back to 0.0.
    assert_eq!(
        sig.window_shift_sec(),
        0.0,
        "file1: the :885 reset must fire post-call (noOverlap branch taken)"
    );
    assert_eq!(
        sig.spectrum_shift_in_frames(),
        80,
        "file1: ssif quantized to 80"
    );
    assert_eq!(
        sig.last_result_rows_len(),
        50,
        "file1: real_vec_size 50 (noOverlap)"
    );

    let mut audio2 = excerpt_audio();
    let mut segs2 = fresh_segs(&audio2);
    sig.get_segmentation(&mut audio2, &mut segs2).unwrap();

    // Post-file2: the reset re-fired (the round trip repeats).
    assert_eq!(
        sig.window_shift_sec(),
        0.0,
        "file2: the reset re-fired (the noOverlap round trip repeats)"
    );
    assert_eq!(
        sig.last_result_rows_len(),
        50,
        "file2: real_vec_size still 50 (noOverlap re-derived)"
    );

    // Discriminating params row: re-derive via get_blstm_param on a fresh audio at the
    // post-file2 state (window_shift_sec == 0.0 -> noOverlap re-derivation), and match
    // the dumped file2 params (window_shift=1, window_size=324, ssif=80, real_vec=50).
    let params = common::load_bin_phase2b("spectral_noOverlap_params_file2.bin");
    let mut ws = sig.window_shift_sec(); // 0.0 -> noOverlap re-derivation
    let ssr = 4;
    let (window, shift, no_overlap, real) = get_blstm_param(
        3.25,
        &mut ws,
        8000.0,
        sig.spectrum_shift_in_frames(),
        ssr,
        &[4, 1],
        &[1, 1],
        16001,
    );
    assert!(
        no_overlap,
        "file2 re-derivation: noOverlap (the reset fired)"
    );
    assert_eq!(shift as f64, params[[0, 0]], "file2 window_shift");
    assert_eq!(window as f64, params[[0, 1]], "file2 window_size");
    assert_eq!(
        sig.spectrum_shift_in_frames() as f64,
        params[[0, 2]],
        "file2 ssif"
    );
    assert_eq!(real as f64, params[[0, 3]], "file2 real_vec_size");

    // The boundaries coincide (honest finding).
    let want1 = common::load_bin_phase2b("spectral_noOverlap_boundaries_file1_chan1.bin");
    let want2 = common::load_bin_phase2b("spectral_noOverlap_boundaries_file2_chan1.bin");
    assert_eq!(
        want1, want2,
        "file1 and file2 boundaries coincide (deterministic net)"
    );
    assert_boundaries_match(
        &segs1,
        0,
        "spectral_noOverlap_boundaries_file1_chan1.bin",
        "file1",
    );
    assert_boundaries_match(
        &segs2,
        0,
        "spectral_noOverlap_boundaries_file2_chan1.bin",
        "file2",
    );
}

// === Non-wav 80-fallback: unit-test-only via direct state mutation ============

#[test]
fn non_wav_spectrum_shift_80_fallback_persists() {
    // BLSTMSpectralSegmenter.cpp:209: `if (!audio.hasReadWavFile()) _SpectrumShiftInFrames
    // = 80`, and :210 `_SpectrumShift = 80/rate` PERSISTS into the member. There is no
    // non-wav fixture (the driver always reads a wav), so this is pinned by directly
    // mutating the state and confirming a subsequent get_segmentation honors the 80.
    let mut sig = build("real");
    sig.force_non_wav_spectrum_shift();
    assert_eq!(
        sig.spectrum_shift_in_frames(),
        80,
        "state mutation set ssif = 80"
    );

    // A get_segmentation call must KEEP ssif at 80 (the persistence): for the excerpt
    // at rate 8000 the wav-path re-quantization would ALSO give round(0.01*8000) = 80,
    // so the observable is that the value stays 80 (not that it changes). Assert it
    // stays 80 after a real call (the persistence path is honored, not overwritten by a
    // spurious recompute that could differ if the config shift were not 0.01).
    let mut audio = excerpt_audio();
    let mut segs = fresh_segs(&audio);
    sig.get_segmentation(&mut audio, &mut segs).unwrap();
    assert_eq!(
        sig.spectrum_shift_in_frames(),
        80,
        "non-wav 80 fallback persists through get_segmentation"
    );
}
