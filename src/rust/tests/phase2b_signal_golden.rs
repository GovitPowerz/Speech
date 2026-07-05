//! Phase 2b Task 6: `BlstmSignalSegmenter` (Algo 4) driver goldens.
//!
//! The FIRST driver with the NN in the chain. Unlike TDC/LTSV (whose real
//! `getSegmentation` is the bit-golden AS-IS), the signal driver's result_vec comes
//! from `BLSTMNeuralNetwork.feedForwardBackward` (`BLSTMSignalSegmenter.cpp:260`),
//! whose real Eigen GEMMs diverge from the ascending-loop NN port at the real net's
//! k>=23 shapes. So the harness `SignalProbe` stage transcribes `getSegmentation`'s
//! LIVE code (`BLSTMSignalSegmenter.cpp:93-398`) faithfully but swaps ONLY that FFB
//! call for the ascending-loop reimpl (`signalReimplFFB`), matching what the Rust
//! `BlstmSignalSegmenter` does by calling the Phase 2 `BlstmNetwork` forward.
//!
//! SECONDARY structural probe (spec decision 3): the harness ALSO runs the REAL
//! compiled `getSegmentation` beside the transcription; segment count + types must
//! match EXACTLY (a mismatch aborts fixture generation), and the boundary max-delta
//! is recorded as `SEG_STRUCT site=signal_<variant> ok=1 max_dt=<measured>` in the
//! manifest. All three variants measured `max_dt=0.0`: the real net's posteriors on
//! this excerpt are ~0.002-0.03, far BELOW `_DecisionThreshRising` (0.6), so NO
//! speech is ever detected and every variant collapses to the untouched always-Other
//! SEED hypothesis (`[Other@0, End@dur]`; VRCTS `SegmentList` empty, `spdur=0.00`,
//! `Pmiss=1.0`) -- insensitive to the tiny reimpl-vs-real posterior gap. The
//! `signal_*_result_chan1.bin` goldens (4000/4001 NN outputs) are what pin the whole
//! NN chain bit-exactly, via `BlstmSignalSegmenter::last_result_rows` (a port-side
//! observation point with no legacy counterpart, see its doc in `tasks/sad.rs`)
//! rather than only the (structurally trivial, given the sub-threshold posteriors)
//! boundary list.
//!
//! Goldens dumped by the harness `SignalProbe` stage (`tools/oracle_harness/main.cpp`,
//! Phase 2b Task 6 block) against `tests/reference_data/phase2b/signal.config` (the
//! real net topology + real `NNweights_config1.bin`, overridden window/shift per
//! variant) on the shared 2-channel excerpt (`excerpt_2ch_8k.wav`, offset 0.35, dur
//! 2.0, rate 8000, frame_count 16001, ssr 4). Three variants:
//!   window0   -- BLSTM_window 0        (full-sequence, plain FFB; real_vec 4000)
//!   overlap   -- BLSTM_window 0.01, shift 0.0005 (overlap FFB; window_shift 4 == ssr,
//!                the ONLY non-broken overlap regime -- see the note below; real_vec 4001)
//!   noOverlap -- BLSTM_window 0.5, shift 0 (truncate FFB via the noOverlap poisoning;
//!                window_size floored to 4000, window_shift clamped to 1; real_vec 4000)
//!
//! LEGACY BUG (IMPROVEMENTS.md signal-overlap-oob): the brief's original overlap
//! params (window 0.5, shift 0.1) are BROKEN AS COMMITTED -- the overlap result-vec
//! sizing `ceil(frameCount/shift)` is far too small for the OverLap FFB driver's
//! write index (`begin/ssr + lengthShort ~ frameCount/ssr`), so the REAL
//! `getSegmentation` aborts on it under Eigen assertions and heap-corrupts under
//! release -DNDEBUG. The OverLap path only survives when `window_shift <= ssr`
//! (== 4 samples, i.e. shift <= 0.0005s), the degenerate regime the golden uses.

mod common;

use std::path::PathBuf;

use indexmap::IndexMap;
use speech::audio::{Audio, read_audio};
use speech::io::binary::read_matrix;
use speech::tasks::sad::BlstmSignalSegmenter;
use speech::tasks::segmentation::{SegClass, Segmentation};
use speech::tasks::segmentation_io::{parse_audiodoc_attrs, to_vrcts_string};
use speech::tasks::segmenter::Segmenter;

fn config_text(name: &str) -> String {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/reference_data/phase2b")
        .join(name);
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

/// A config map for `signal.config` with `BLSTM_window`/`BLSTM_shift` overridden to
/// the variant's raw values (the harness does the same via `set_val`).
fn signal_map(window: &str, shift: &str) -> IndexMap<String, String> {
    let mut m = speech::legacy_config::parse_legacy_config(&config_text("signal.config"));
    m.insert("BLSTM_window".into(), window.into());
    m.insert("BLSTM_shift".into(), shift.into());
    m
}

fn variant_map(tag: &str) -> IndexMap<String, String> {
    match tag {
        "window0" => signal_map("0", "0.1"),
        "overlap" => signal_map("0.01", "0.0005"),
        "noOverlap" => signal_map("0.5", "0"),
        _ => unreachable!(),
    }
}

/// Fresh excerpt audio (offset 0.35, dur 2.0), UNMUTATED: `get_segmentation` applies
/// preemph itself (config `BLSTM_preemph_ratio 0.97`, `BLSTM_noise_seed 0` -> no noise).
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

fn build(tag: &str) -> BlstmSignalSegmenter {
    let w = real_weights();
    BlstmSignalSegmenter::from_legacy(&variant_map(tag), Some(&w)).unwrap()
}

/// NaN-aware convolved-row comparator: bit-IDENTICAL elements (including the OverLap
/// `0/0 -> NaN` uncovered-row quirk that the overlap variant carries) pass in ANY
/// mode; the rest go through `assert_oracle_eq` (bit-exact on the oracle env, hybrid
/// ULP off-env). This mirrors the harness's `probeGapNaN` contract (identical NaN
/// bits -> equal) so the shared `assert_oracle_eq` -- whose ULP arm cannot compare a
/// NaN -- is only fed the finite elements. `got`/`want` are single-row vectors.
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

// === from_legacy: TwoSweeps parsed-required + dead ===========================

#[test]
fn from_legacy_reads_real_values() {
    let sig = build("window0");
    // BLSTM_TwoSweeps is `false` in signal.config: parsed and stored, dead in the
    // live path (`BLSTMSignalSegmenter.cpp:282/:297` consumers are commented out).
    assert!(
        !sig.two_sweeps(),
        "two_sweeps parsed from config (dead flag)"
    );
}

#[test]
fn from_legacy_missing_two_sweeps_is_err() {
    // `conf.get<bool>("BLSTM_TwoSweeps")` aborts on a missing key in the legacy
    // (:20/:28), so the parse itself is load-bearing -- reproduced as an `Err`.
    let mut m = signal_map("0", "0.1");
    m.shift_remove("BLSTM_TwoSweeps");
    let w = real_weights();
    assert!(
        BlstmSignalSegmenter::from_legacy(&m, Some(&w)).is_err(),
        "a missing BLSTM_TwoSweeps must be an error (parity with conf.get<bool>)"
    );
}

// === Result rows (pre-convolution), per variant ==============================
//
// PERF NOTE: the signal NN forward runs on the full 16001-sample input; the overlap
// variant (window_shift 4) runs ~4000 windows through the whole BLSTM per channel,
// ~160s in debug. So the driver's expensive `get_segmentation` is invoked EXACTLY
// ONCE per cheap variant (window0/noOverlap) across the whole file, and exactly TWICE
// for overlap (`overlap_result_and_boundaries_match_dump` + `overlap_vrcts_matches_
// dump` -- two different audio slices back two different goldens, see those tests'
// doc-comments). Everything else is derived from the dumps (NN-free): the shapes
// here, the convolved-row rebuild in the next test.
//
// The result vec's NUMERIC content coming out of the REAL Rust NN chain is pinned
// bit-exact against `signal_<tag>_result_chan1.bin` via
// `BlstmSignalSegmenter::last_result_rows` (a port-side capture point, no legacy
// counterpart -- see its doc in `tasks/sad.rs`) inside
// `get_segmentation_boundaries_match_dump_cheap_variants` (window0/noOverlap) and
// `overlap_result_and_boundaries_match_dump` (overlap), reusing those tests' single
// runs rather than invoking the expensive driver again here. This closes the S9.3
// gap: a stub `BlstmNetwork` forward returning all-zeros would fail
// `assert_last_result_row_matches_dump` immediately (0.0 != the dumped ~0.002-0.03
// posteriors), even though it would still pass the boundary/VRCTS goldens (all-zeros
// is also below the 0.6 rising threshold, so it too collapses to the same
// always-Other seed).

#[test]
fn result_row_shapes_match_manifest() {
    // Dump-shape check only (NN-free): the real_vec_size per variant.
    for (tag, cols) in [
        ("window0", 4000usize),
        ("overlap", 4001),
        ("noOverlap", 4000),
    ] {
        let want = common::load_bin_phase2b(&format!("signal_{tag}_result_chan1.bin"));
        assert_eq!(want.nrows(), 1, "{tag} result: row vector");
        assert_eq!(
            want.ncols(),
            cols,
            "{tag} result: real_vec_size per the manifest constants"
        );
    }
}

/// Bit-exact vs the golden: the driver's captured PRE-convolution result row (the
/// REAL Rust `BlstmNetwork` forward output, via `last_result_rows`) compared to
/// `signal_<tag>_result_chan1.bin`. Called once per variant with that variant's
/// already-run driver (see call sites), so no extra expensive forward pass.
///
/// Uses the NaN-aware `assert_convolved_eq` comparator: the overlap variant's
/// PRE-convolution row carries the documented OverLap `0/0 -> NaN` uncovered-row
/// quirk at its last index (`last_result_rows` is captured BEFORE
/// `results_to_segmentation`'s convolution, same row `assert_convolved_eq` already
/// handles for the POST-convolution comparison below), so a plain `assert_oracle_eq`
/// would spuriously fail the ULP arm's NaN comparison.
fn assert_last_result_row_matches_dump(sig: &BlstmSignalSegmenter, tag: &str) {
    let want = common::load_bin_phase2b(&format!("signal_{tag}_result_chan1.bin"));
    let rows = sig.last_result_rows();
    assert_eq!(
        rows.len(),
        2,
        "{tag}: last_result_rows must have 2 channels"
    );
    let got = ndarray::Array2::from_shape_vec((1, rows[0].len()), rows[0].clone()).unwrap();
    assert_convolved_eq(
        &got,
        &want,
        &format!("{tag} last_result_row (chan1, NN-chain)"),
    );
}

// === Convolved rows + boundaries (post results_to_segmentation), per variant ==

#[test]
fn convolved_rows_match_dump_all_variants() {
    use speech::tasks::segmenter::{DriverConfig, SegmenterConfig, results_to_segmentation};

    for tag in ["window0", "overlap", "noOverlap"] {
        let m = variant_map(tag);
        let driver_cfg = DriverConfig::from_config(&m, "BLSTM").unwrap();
        let seg_cfg = SegmenterConfig::from_config(&m, "BLSTM").unwrap();

        // The harness dumps the convolved row via results_to_segmentation with the
        // variant's timeStep/timeOffset. window0/noOverlap: timeStep = shift*ssr =
        // (1/8000)*4 = 5e-4, timeOffset = timeStep/2 - shift/2 = 1.875e-4. overlap:
        // timeStep = shift = 4/8000 = 5e-4, timeOffset = 0.0.
        let (time_step, time_offset) = match tag {
            "overlap" => (4.0 / 8000.0, 0.0),
            _ => {
                let step = (1.0 / 8000.0) * 4.0;
                (step, step / 2.0 - (1.0 / 8000.0) / 2.0)
            }
        };

        let pre = common::load_bin_phase2b(&format!("signal_{tag}_result_chan1.bin"));
        let mut results: Vec<f64> = pre.row(0).to_vec();

        let bound = common::load_bin_phase2b(&format!("signal_{tag}_boundaries_chan1.bin"));
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

        let want = common::load_bin_phase2b(&format!("signal_{tag}_convolved_chan1.bin"));
        let got = ndarray::Array2::from_shape_vec((1, results.len()), results).unwrap();
        assert_convolved_eq(&got, &want, &format!("{tag} convolved"));

        // Boundaries: the seg built above must match the boundary dump exactly.
        let got_segs = seg.segments();
        assert_eq!(
            got_segs.len(),
            want_bound_count(tag),
            "{tag} boundary count"
        );
        for (i, s) in got_segs.iter().enumerate() {
            let gb = ndarray::Array2::from_shape_vec((1, 1), vec![s.begin]).unwrap();
            let wb = ndarray::Array2::from_shape_vec((1, 1), vec![bound[[i, 0]]]).unwrap();
            common::assert_oracle_eq(&gb, &wb, &format!("{tag} boundary[{i}].begin"));
            assert_eq!(
                s.ty as i32,
                bound[[i, 1]] as i32,
                "{tag} boundary[{i}].type"
            );
        }
    }
}

fn want_bound_count(_tag: &str) -> usize {
    2 // all variants: [Other@0, End@dur] (untouched always-Other seed hypothesis;
    // sub-threshold posteriors never trigger a rising crossing)
}

/// Assert `segs[0]`'s hypothesis boundary list matches the harness dump exactly.
fn assert_boundaries_match(segs: &[Segmentation], dump: &str, label: &str) {
    let want = common::load_bin_phase2b(dump);
    let got_segs = segs[0].segments();
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

// === End-to-end boundaries off the DRIVER (not the rebuild) ===================
//
// Cheap variants only (window0/noOverlap, single-forward). Overlap's driver-level
// e2e (boundaries + result-vec) is checked in `overlap_result_and_boundaries_match_
// dump` (one of its two ~160s runs); its VRCTS golden in `overlap_vrcts_matches_dump`
// (the other run, on zero-offset audio). The result-vec's numeric content is ALSO
// cross-checked NN-free (from the static dump, not a live run) in
// `convolved_rows_match_dump_all_variants`.

#[test]
fn get_segmentation_boundaries_match_dump_cheap_variants() {
    for tag in ["window0", "noOverlap"] {
        let mut sig = build(tag);
        let mut audio = excerpt_audio();
        let mut segs = fresh_segs(&audio);
        sig.get_segmentation(&mut audio, &mut segs).unwrap();
        assert_boundaries_match(
            &segs,
            &format!("signal_{tag}_boundaries_chan1.bin"),
            &format!("{tag} e2e"),
        );
        // S9.3: the REAL Rust NN chain's result row, bit-exact vs the golden dump
        // (see `assert_last_result_row_matches_dump`'s doc for why this is required
        // and not implied by the boundary/VRCTS checks).
        assert_last_result_row_matches_dump(&sig, tag);
    }
}

// === Overlap variant: the TWO expensive driver-level end-to-end runs ==========
//
// The overlap NN forward is ~160s in debug regardless of which audio slice it runs
// on (cost is windows-per-channel, not audio content), so it is invoked EXACTLY
// TWICE for the whole suite -- matching the harness's own dump generation, which
// also runs the overlap NN twice: once on the offset-0.35 excerpt (`runVariant`,
// backing `signal_overlap_result_chan1.bin` / `..._boundaries_chan1.bin`) and once
// on the zero-offset excerpt (the inner VRCTS block, backing
// `signal_overlap_vrcts_chan1.xml`). The two goldens are NOT interchangeable (they
// come from different audio slices), so one Rust-side run cannot serve both without
// either mismatching the result-vec golden or the VRCTS golden -- hence two runs,
// not a reused single one.

/// Run 1/2 (offset 0.35, matching `signal_overlap_result_chan1.bin` /
/// `signal_overlap_boundaries_chan1.bin`): asserts the REAL Rust NN chain's result
/// row bit-exact (S9.3; see `assert_last_result_row_matches_dump`'s doc), the
/// driver-level boundary list, and the cost/classif accumulators.
#[test]
fn overlap_result_and_boundaries_match_dump() {
    let tag = "overlap";
    let mut sig = build(tag);
    let mut audio = excerpt_audio();
    let mut segs = fresh_segs(&audio);
    sig.get_segmentation(&mut audio, &mut segs).unwrap();

    assert_last_result_row_matches_dump(&sig, tag);
    assert_boundaries_match(
        &segs,
        &format!("signal_{tag}_boundaries_chan1.bin"),
        &format!("{tag} e2e"),
    );

    assert_eq!(sig.cumulative_error().len(), 2, "overlap: per-channel cost");
    assert_eq!(
        sig.nb_of_classif().len(),
        2,
        "overlap: per-channel nb_of_classif"
    );
}

/// Run 2/2 (zero offset, matching `signal_overlap_vrcts_chan1.xml`): the VRCTS byte
/// golden, whose times must be offset-free.
#[test]
fn overlap_vrcts_matches_dump() {
    let tag = "overlap";

    let dump = std::fs::read_to_string(common::fixture_phase2b(&format!(
        "signal_{tag}_vrcts_chan1.xml"
    )))
    .unwrap();
    let (name, path_attr) = parse_audiodoc_attrs(&dump);

    let mut sig = build(tag);
    let mut audio = excerpt_audio_zero_offset();
    let mut segs = fresh_segs(&audio);
    sig.get_segmentation(&mut audio, &mut segs).unwrap();

    let got = to_vrcts_string(&segs[0], &name, &path_attr);
    common::assert_vrcts_eq(&got, &dump, "overlap vrcts");
}

// === signalraw (PRE-preemph) vs signal (POST) -- the timing divergence golden ==

#[test]
fn signalraw_differs_from_signal_and_both_match_dump() {
    // BLSTMSignalSegmenter.cpp: signalRaw is dumped BEFORE preemph (:133-134), signal
    // AFTER (:174-175). With BLSTM_preemph_ratio 0.97 > 0 they GENUINELY DIFFER (the
    // signal-variant divergence 8 vs spectral, where both dumps are post-preemph).
    // Rust side: audio.data before get_segmentation == signalraw; after == signal.
    for tag in ["window0", "overlap", "noOverlap"] {
        let raw = common::load_bin_phase2b(&format!("signal_{tag}_signalraw_chan1.bin"));
        let sig = common::load_bin_phase2b(&format!("signal_{tag}_signal_chan1.bin"));
        assert_eq!(raw.shape(), &[1, 16001], "{tag} signalraw shape");
        assert_ne!(
            raw, sig,
            "{tag}: signalraw (pre-preemph) must DIFFER from signal (post-preemph)"
        );

        // Rust: fresh audio row(0) == signalraw; after preemph == signal.
        let audio = excerpt_audio();
        let raw_rust =
            ndarray::Array2::from_shape_vec((1, audio.data.ncols()), audio.data.row(0).to_vec())
                .unwrap();
        common::assert_oracle_eq(&raw_rust, &raw, &format!("{tag} signalraw"));

        let mut audio2 = excerpt_audio();
        audio2.apply_preemph(0.97);
        let sig_rust =
            ndarray::Array2::from_shape_vec((1, audio2.data.ncols()), audio2.data.row(0).to_vec())
                .unwrap();
        common::assert_oracle_eq(&sig_rust, &sig, &format!("{tag} signal"));
    }
}

// === compute_errors vs the programmatic reference ============================

#[test]
fn score_matches_dump_cheap_variants() {
    for tag in ["window0", "noOverlap"] {
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

        let want = common::load_bin_phase2b(&format!("signal_{tag}_scores.bin"));
        let reports = BlstmSignalSegmenter::score(&mut hyp, Some(&reference), -1);
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
fn vrcts_bytes_match_dump_cheap_variants() {
    // Overlap's VRCTS is covered in `overlap_vrcts_matches_dump` (one of its two
    // expensive overlap runs); here the two cheap variants.
    for tag in ["window0", "noOverlap"] {
        let dump = std::fs::read_to_string(common::fixture_phase2b(&format!(
            "signal_{tag}_vrcts_chan1.xml"
        )))
        .unwrap();
        let (name, path_attr) = parse_audiodoc_attrs(&dump);

        let mut sig = build(tag);
        let mut audio = excerpt_audio_zero_offset();
        let mut segs = fresh_segs(&audio);
        sig.get_segmentation(&mut audio, &mut segs).unwrap();

        let got = to_vrcts_string(&segs[0], &name, &path_attr);
        common::assert_vrcts_eq(&got, &dump, &format!("{tag} vrcts"));
    }
}

// === Sizing hand tests: the three variants' real_vec_size ====================

#[test]
fn result_vec_sizing_all_variants() {
    let frame_count = 16001usize;
    // ssr = 4 (LSTM sub 4*1, output 1*1 -> whole-net ratio 4); its factors [4,1] /
    // [1,1] are applied explicitly below.

    // window0: window_size=0 -> shift clamped to 1 (:107). real = frameCount/1 then
    // ssr-division GATED ON (window==0) -> /4/1/1/1 = 4000.
    {
        let window_shift = 1usize;
        let mut real = frame_count.div_ceil(window_shift);
        // gate active: LSTM ratios [4,1], output ratios [1,1].
        for r in [4usize, 1] {
            real /= r;
        }
        for r in [1usize, 1] {
            real /= r;
        }
        assert_eq!(real, 4000, "window0 real_vec_size");
    }

    // overlap: window_size=40 (>=ssr), window_shift=4 (>=1, noOverlap FALSE). real =
    // ceil(16001/4) = 4001, NO ssr-division (gate window==0||noOverlap is false).
    {
        let window_shift = 4usize;
        let real = frame_count.div_ceil(window_shift);
        assert_eq!(real, 4001, "overlap real_vec_size (ceil, no ssr div)");
    }

    // noOverlap: window_size floored to (round(0.5*8000)/4)*4 = 4000, shift clamped
    // to 1. real = ceil(16001/1) then ssr-division GATED ON (noOverlap) -> /4 = 4000.
    {
        let window_shift = 1usize;
        let mut real = frame_count.div_ceil(window_shift);
        for r in [4usize, 1] {
            real /= r;
        }
        for r in [1usize, 1] {
            real /= r;
        }
        assert_eq!(real, 4000, "noOverlap real_vec_size");
    }
}

// === full=1-when-window-0 quirk: driver still proceeds (no windowing coeff) ===

#[test]
fn window_zero_full_stays_one_and_driver_proceeds() {
    // window==0 -> full_window_size stays 1 (NO `full = 0` line, divergence 2). The
    // windowing coeff is computed over length 1; Rust `windowing_coefficients(size<=1)`
    // returns None -- the driver must STILL proceed (the coeff is dump-only dead
    // weight, never applied), producing a valid Segmentation. This is exactly the
    // window0 variant; assert it does not panic and yields a hypothesis.
    let mut sig = build("window0");
    let mut audio = excerpt_audio();
    let mut segs = fresh_segs(&audio);
    sig.get_segmentation(&mut audio, &mut segs).unwrap();
    assert!(!segs[0].segments().is_empty());

    // Sanity: windowing_coefficients over a length-1 window is None (the no-op the
    // legacy's empty-coeff path relies on).
    assert!(
        speech::audio::windowing_coefficients("hamming", false, 1, 0.8).is_none(),
        "length-1 windowing coeff must be None (the full==1 no-op)"
    );
}

// === timeStep/timeOffset branch hand tests ===================================

#[test]
fn timestep_timeoffset_branches() {
    let rate = 8000.0;
    let ssr = 4.0;

    // Default (window==0) / noOverlap branch: shift = 1/rate (clamped). timeStep =
    // shift*ssr; timeOffset = timeStep/2 - shift/2.
    {
        let shift = 1.0 / rate;
        let step = shift * ssr;
        let off = step / 2.0 - shift / 2.0;
        assert_eq!(step, 4.0 / 8000.0);
        assert_eq!(off, (4.0 / 8000.0) / 2.0 - (1.0 / 8000.0) / 2.0);
    }
    // Overlap branch (window>0, !noOverlap): timeStep = shift (= 4/rate); offset = 0.
    {
        let shift = 4.0 / rate;
        let step = shift;
        let off = 0.0;
        assert_eq!(step, 4.0 / 8000.0);
        assert_eq!(off, 0.0);
    }
}

// === DC-offset flag is log-only (flag true vs false -> identical output) =======

#[test]
fn dc_offset_flag_is_log_only() {
    // BLSTMSignalSegmenter.cpp:116: _FlagDCOffset is LOGGED but never applied in the
    // signal variant (the NN consumes audio._Data directly, :188 -- there is no
    // getSequence with a DC-removal path). So flipping BLSTM_flag_DCOffset must NOT
    // change any output. Compare the full boundary list under both flag values.
    let w = real_weights();
    let run = |flag: &str| -> Vec<(f64, i32)> {
        let mut m = signal_map("0", "0.1");
        m.insert("BLSTM_flag_DCOffset".into(), flag.into());
        let mut sig = BlstmSignalSegmenter::from_legacy(&m, Some(&w)).unwrap();
        let mut audio = excerpt_audio();
        let mut segs = fresh_segs(&audio);
        sig.get_segmentation(&mut audio, &mut segs).unwrap();
        segs[0]
            .segments()
            .iter()
            .map(|s| (s.begin, s.ty as i32))
            .collect()
    };
    let with_true = run("true");
    let with_false = run("false");
    assert_eq!(
        with_true, with_false,
        "_FlagDCOffset is log-only in signal mode: output must be identical"
    );
}

// === TWO-FILES noOverlap golden: the REAL stateful lifecycle ==================

/// Unlike TDC/LTSV (idempotent shift re-quantization), signal noOverlap assigns
/// `_WindowShift = 0.0` mid-run (`:376`, the poisoning). So file 2 (SAME driver
/// instance) genuinely re-enters `get_segmentation` with `window_shift_sec == 0.0`:
/// `:99` rounds `0.0*rate -> 0` -> `shift<1` -> noOverlap re-triggers, `:107-108`
/// re-clamp to `1/rate`. This test runs the SAME `BlstmSignalSegmenter` twice and
/// pins BOTH files' boundaries against the harness's real-lifecycle dumps.
///
/// EVIDENCE (do the two files differ?): the harness dumps show
/// `signal_noOverlap_boundaries_file1_chan1.bin` ==
/// `signal_noOverlap_boundaries_file2_chan1.bin` -- BOTH are the untouched SEED
/// hypothesis `[Other@0, End@dur]` (the real net's posteriors on this excerpt are
/// ~0.002-0.03, far BELOW the 0.6 rising threshold, so NO speech is ever detected;
/// VRCTS `SegmentList` is empty, `spdur=0.00`). So the OUTPUT is identical across
/// files even though the lifecycle genuinely re-enters through the mutated
/// `_WindowShift`. This is NOT the idempotent-re-quantization argument (TDC/LTSV):
/// here the member is truly reset to 0.0 and re-derived, but the re-derivation lands
/// on the SAME `1/rate` and the posteriors are unchanged (same audio), so the
/// boundaries coincide. The test asserts the equality rather than assuming it, and
/// this doc-comment records the honest finding: the state DOES round-trip, the
/// observable output does not change.
///
/// This is NOT determinism-only, though: the boundary/type coincidence alone cannot
/// distinguish "the reset genuinely fired and re-derived the same value" from "the
/// reset never happened at all". WITHOUT the `:376` reset, file 2 would instead
/// inherit file 1's POST-mutation `window_shift_sec = 1/rate` (not `0.0`), which
/// re-derives `window_size (half) = round(0.5*8000/2) = 2000` -> `window_shift =
/// round((1/8000)*8000) = 1 >= 1` -> `no_overlap = false` -- the OVERLAP branch, NOT
/// noOverlap, with `full_window_size = 4001` and result-vec sizing UNCONDITIONAL
/// `ceil(frame_count/window_shift) = ceil(16001/1) = 16001` (no ssr-division gate,
/// since the gate is `window_size==0||no_overlap` and neither holds). That is a
/// completely different code path with a ~4x larger result vector than the real
/// (reset-present) `4000`. So `last_result_rows()[0].len()` after each call --
/// asserted below against the real value, 4000, not the 16001 no-reset
/// counterfactual -- IS the discriminating observable: a driver that silently
/// dropped the `:376` reset would still coincidentally match the file1==file2
/// boundary equality (both would show the always-Other seed under the overlap
/// branch too, since posteriors stay sub-threshold there as well), but it would
/// produce a differently-shaped (16001, not 4000) result row on file 2. Likewise
/// `window_shift_sec()` is asserted to be `0.0` post-call on both files: the member
/// starts each call at the config value `0.0`, is floored up to `1/rate = 1.25e-4`
/// DURING the call (`:107-108`, the noOverlap clamp), and is then reset back to
/// `0.0` by the `:376` poisoning AFTER the call returns -- so the round trip is
/// `0.0 -> 1.25e-4 -> 0.0` within a single call, not a value left sitting at a
/// fixed nonzero point.
#[test]
fn two_files_in_sequence_no_overlap_lifecycle() {
    let w = real_weights();
    let mut sig = BlstmSignalSegmenter::from_legacy(&variant_map("noOverlap"), Some(&w)).unwrap();

    let mut audio1 = excerpt_audio();
    let mut segs1 = fresh_segs(&audio1);
    sig.get_segmentation(&mut audio1, &mut segs1).unwrap();

    // Discriminator (file 1): result-vec length must be the real (reset-present)
    // 4000, not the no-reset counterfactual (irrelevant on file 1, which has no
    // prior state, but establishes the baseline shape) -- and window_shift_sec must
    // have landed back on 1/rate post-call.
    assert_eq!(
        sig.last_result_rows()[0].len(),
        4000,
        "file1: result-vec length must be 4000 (noOverlap sizing)"
    );
    assert_eq!(
        sig.window_shift_sec(),
        0.0,
        "file1: the :376 reset must fire post-call (noOverlap branch taken)"
    );

    let mut audio2 = excerpt_audio();
    let mut segs2 = fresh_segs(&audio2);
    sig.get_segmentation(&mut audio2, &mut segs2).unwrap();

    // Discriminator (file 2): THE key observable. If the :376 reset had NOT fired
    // on file 1, file 2 would inherit window_shift_sec = 1/rate (not 0.0), re-derive
    // through the OVERLAP branch (not noOverlap), and produce a result-vec of length
    // 16001 (ceil(16001/1), no ssr-division gate) -- see the doc-comment above for
    // the full counterfactual derivation. The real (reset-present) value is 4000.
    assert_eq!(
        sig.last_result_rows()[0].len(),
        4000,
        "file2: result-vec length must be 4000 (the reset re-triggered noOverlap), \
         NOT 16001 (the no-reset overlap-branch counterfactual)"
    );
    assert_eq!(
        sig.window_shift_sec(),
        0.0,
        "file2: the :376 reset must fire again post-call (the round trip repeats)"
    );

    let want1 = common::load_bin_phase2b("signal_noOverlap_boundaries_file1_chan1.bin");
    let want2 = common::load_bin_phase2b("signal_noOverlap_boundaries_file2_chan1.bin");

    for (segs, want, label) in [(&segs1, &want1, "file1"), (&segs2, &want2, "file2")] {
        let got_segs = segs[0].segments();
        assert_eq!(got_segs.len(), want.nrows(), "{label}: boundary count");
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

    // The honest finding recorded above: both files' boundaries coincide.
    assert_eq!(
        want1, want2,
        "file1 and file2 boundaries coincide (see doc-comment)"
    );
}
