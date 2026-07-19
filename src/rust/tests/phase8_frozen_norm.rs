//! Phase 8 Task 1 (spec S1.1, `2026-07-19-phase-8-streaming-design.md`): the
//! `Audio_fixed_gain` frozen-norm mechanism -- the ONE sanctioned exact-tree touch
//! of the streaming phase. `read_audio` gains a `fixed_gain: Option<f64>` param;
//! `None` (every pre-phase-8 committed config) reproduces the legacy
//! `normalize_channels` path byte-identically -- proven here directly and, more
//! broadly, by the full `cargo test` suite staying green untouched (the inertness
//! proof named in the report). `Some(g)` divides every channel by `g.max(1e-3)`
//! instead of computing the whole-file `(2*rms+max_abs)/2` statistic -- the
//! frozen-stats mode a streaming session needs, since no whole-file lookahead
//! exists online.
//!
//! Five tests: (1)/(2) `read_audio`-level RED->GREEN pins (inert when absent,
//! exact raw/gain when present); (3) a `BagOfProcessors`-level thread proof (the
//! config key genuinely reaches the internal `read_audio` call in
//! `segmentation_function`, not merely stored inertly); (4) the fast type-1
//! external-normalization unit pin (the fast-tree companion change this task
//! required: `external_normalize_f32` vs the exact `feed_forward_backward`
//! type-1 branch on the same input + the same tuple-A normalize tail,
//! measured-then-pinned); (5) the S1.7 causality-cost leg (offline-frozen vs
//! offline-self-norm on the staged mono fixture, MEASURED and RECORDED, never
//! gated -- this run also exercises the new type-1 fast path end to end).

mod common;

use std::path::Path;

use indexmap::IndexMap;
use ndarray::Array2;
use speech::audio::read_audio;
use speech::cli::{Mode, ModeKind};
use speech::config::NnetSpec;
use speech::engine::bag_of_processors::{BagOfProcessors, Processor};
use speech::engine::corpus::CorpusItem;
use speech::fast::nn::{FastBlstm, FastMatrix, external_normalize_f32};
use speech::nn::blstm::{BlstmConfig, BlstmNetwork};
use speech::tasks::segmentation::Segmentation;

use common::{assert_bits_eq, fixture, load_bin};

fn image_mode() -> Mode {
    Mode {
        kind: ModeKind::Image,
        verbose: false,
    }
}

// ---------------------------------------------------------------------------
// (1)/(2): read_audio-level pins.
// ---------------------------------------------------------------------------

#[test]
fn fixed_gain_inert_when_absent() {
    // `fixed_gain: None` must reproduce the byte-exact legacy path -- pinned
    // against the REAL Phase 1 oracle golden (`sig_norm.bin`, dumped from the
    // compiled C++ AudioStruct ctor), not merely a self-consistency check against
    // `normalize_channels`. Mirrors `phase1_audio_golden.rs::
    // decode_normalize_matches_oracle` through the new 5-arg signature.
    let audio = read_audio(&fixture("excerpt_2ch_8k.wav"), 0.35, 2.0, 0, None).unwrap();
    assert_eq!(audio.sample_rate, 8000);
    assert_bits_eq(
        &audio.data_raw,
        &load_bin("sig_norm.bin"),
        "sig_norm (fixed_gain=None)",
    );
}

#[test]
fn fixed_gain_replaces_normalization() {
    // WITH the key: normalize_channels' whole-file (2*rms+max_abs)/2 statistic must
    // NOT be computed at all -- the channel equals raw/gain EXACTLY. Ground truth
    // (raw, pre-normalization samples) comes from an independent hand-rolled WAV
    // parse (common::read_wav_pcm16), not from read_audio itself, so this is not a
    // tautological self-check. GAIN is arbitrary and unrelated to this file's own
    // self-norm adim, so a latent bug that silently kept calling
    // normalize_channels would diverge from raw/GAIN, not coincide with it.
    const GAIN: f64 = 0.5;
    const OFFSET_SEC: f64 = 0.0;
    const DURATION_SEC: f64 = 0.1;

    let (sample_rate, channels, samples) = common::read_wav_pcm16(&fixture("excerpt_2ch_8k.wav"));
    let channels = channels as usize;

    let audio = read_audio(
        &fixture("excerpt_2ch_8k.wav"),
        OFFSET_SEC,
        DURATION_SEC,
        0,
        Some(GAIN),
    )
    .unwrap();
    assert_eq!(audio.sample_rate, sample_rate);
    let (got_channels, got_frames) = audio.data_raw.dim();
    assert_eq!(got_channels, channels);
    assert!(got_frames > 0, "sanity: the 0.1s slice must be non-empty");

    for c in 0..got_channels {
        for k in 0..got_frames {
            let raw = samples[k * channels + c] as f64 / 32768.0;
            let want = raw / GAIN; // GAIN > 1e-3, the floor is inert here.
            assert_eq!(
                audio.data_raw[[c, k]].to_bits(),
                want.to_bits(),
                "chan {c} frame {k}: fixed_gain path must equal raw/gain exactly \
                 (the (2*rms+max_abs)/2 path must not be taken)"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// (3): BagOfProcessors threads the config key to its internal read_audio call.
// ---------------------------------------------------------------------------

#[test]
fn bag_threads_fixed_gain_to_internal_read_audio() {
    // Proves `Audio_fixed_gain` is not merely STORED on the bag but genuinely
    // THREADED into segmentation_function's own internal read_audio call:
    // (a) run the frozen staged config through the full corpus-style driver
    // (segmentation_function, image mode, no reference) and capture the fast
    // spectral driver's posterior rows; (b) independently read_audio the SAME
    // mono wav with the SAME measured gain (bypassing the bag's own read
    // entirely -- run_get_segmentation never touches self.fixed_gain) and feed it
    // through the SAME driver. Every other setting (InputNormalizationType 1,
    // Inference_Path fast, the weight pack, ...) is held IDENTICAL between the
    // two bags, so the audio-decode path is the only variable: if the config key
    // were read but silently dropped before reaching read_audio, (a) would fall
    // back to normalize_channels and diverge from (b).
    let dir = tempfile::tempdir().unwrap();
    let stage = common::stage_frozen_tier2(dir.path());
    let frozen_text = std::fs::read_to_string(&stage.config_path).unwrap();

    // (a) bag-internal path.
    let mut map_a = speech::legacy_config::parse_legacy_config(&frozen_text);
    let mut bag_a =
        BagOfProcessors::from_configs(std::slice::from_mut(&mut map_a), image_mode()).unwrap();
    assert_eq!(
        bag_a.fixed_gain(),
        Some(stage.fixed_gain),
        "sanity: the bag must have read Audio_fixed_gain from the config"
    );
    let item = CorpusItem {
        file_name: stage.wav_path.to_str().unwrap().to_string(),
        ref_seg: String::new(),
        language: "unk".into(),
        dialect: "unk".into(),
        class_index: 0,
        file_id: 1,
        weight: 1.0,
    };
    bag_a.segmentation_function(&item, image_mode()).unwrap();
    let rows_a = match bag_a.processor(0) {
        Processor::FastSpectral(s) => s.last_result_rows().to_vec(),
        _ => panic!("expected FastSpectral for the frozen staged config"),
    };

    // (b) directly-supplied path: read_audio called HERE with the SAME explicit
    // fixed_gain, fed through run_get_segmentation.
    let mut map_b = speech::legacy_config::parse_legacy_config(&frozen_text);
    let mut bag_b =
        BagOfProcessors::from_configs(std::slice::from_mut(&mut map_b), image_mode()).unwrap();
    let mut audio = read_audio(&stage.wav_path, 0.0, 3600.0, 0, Some(stage.fixed_gain)).unwrap();
    let dur = (audio.data.ncols() as f64 - 1.0) / audio.sample_rate as f64;
    let mut seg = vec![Segmentation::new(dur)];
    bag_b.run_get_segmentation(0, &mut audio, &mut seg).unwrap();
    let rows_b = match bag_b.processor(0) {
        Processor::FastSpectral(s) => s.last_result_rows().to_vec(),
        _ => panic!("expected FastSpectral for the frozen staged config"),
    };

    // Bit-exact per-element compare (not assert_eq! on the Vec<Vec<f64>> directly
    // -- an uncovered overlap row is a NaN in both paths, and NaN != NaN under
    // PartialEq would falsely fail an otherwise-identical pair).
    assert_eq!(rows_a.len(), rows_b.len(), "channel count");
    for (chan, (ra, rb)) in rows_a.iter().zip(rows_b.iter()).enumerate() {
        assert_eq!(ra.len(), rb.len(), "row length chan {chan}");
        for (k, (&a, &b)) in ra.iter().zip(rb.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "segmentation_function's internally-read audio must match a direct \
                 read_audio(.., Some(fixed_gain)) call at chan {chan} row {k} \
                 (a={a} b={b}): the config key must be threaded to the internal \
                 read_audio call, not just stored on the bag"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// (4): the fast type-1 external normalization vs the exact BLSTM type-1 branch.
// ---------------------------------------------------------------------------

/// Deterministic closed-form input (the `phase7_fast_nn.rs` shape).
fn make_input(rows: usize, cols: usize, a: usize, b: usize, seed: usize) -> Array2<f64> {
    Array2::from_shape_fn((rows, cols), |(t, j)| {
        ((t * a + j * b + seed) % 100) as f64 / 100.0 - 0.5
    })
}

#[test]
fn fast_type1_normalization_matches_exact() {
    // Phase 8 Task 1 fast-tree companion pin: `external_normalize_f32` (the type-1
    // frozen-stats input normalization the fast SAD driver now dispatches) vs the
    // EXACT type-1 branch (`nn/blstm.rs:1008-1021`, observed through
    // `feed_forward_backward`'s documented in-place input mutation -- not a
    // reimplementation). Both sides use the SAME tuple-A pack's normalize tail
    // (exact: f64 via `set_weights`; fast: the same tail narrowed once at
    // `from_flat`) on the SAME 200x23 input. MEASURED (Apple Silicon dev box):
    // see the MEASURE print; the pin carries >=10x headroom per the house
    // measure-then-pin pattern (expected scale: one f32 rounding of
    // (x-mean)/std, ~1e-7 relative).
    const REL_PIN: f64 = 1.0e-5;

    let cfg_path = fixture_phase0_cfg();
    let text = std::fs::read_to_string(cfg_path).unwrap();
    let mut map = speech::legacy_config::parse_legacy_config(&text);
    map.insert("BLSTM_InputNormalizationType".into(), "1".into());
    let spec = NnetSpec::from_legacy(&map, "BLSTM").unwrap();
    let flat =
        speech::io::binary::read_weight_vector(&common::fixture_phase0("NNweights_config1.bin"))
            .unwrap();
    assert_eq!(flat.len(), 33_671, "tuple-A pack length");

    // Exact side: plain forward (set_processing_type(false, false)); the type-1
    // branch normalizes `input` IN PLACE before the dispatch, so the mutated input
    // afterward IS the exact normalized sequence.
    let cfg = BlstmConfig::from_legacy(&map, "BLSTM").unwrap();
    assert_eq!(cfg.input_normalization_type, 1, "sanity: override applied");
    let mut exact = BlstmNetwork::from_config(cfg).unwrap();
    exact.set_weights(&flat).unwrap();
    exact.set_processing_type(false, false);

    let input = make_input(200, 23, 31, 17, 7);
    let mut exact_input = input.clone();
    let mut output = Array2::<f64>::zeros((input.nrows() / 4, 1)); // lstm sub 4*1
    let empty = Array2::<f64>::zeros((0, 0));
    exact.feed_forward_backward(&mut exact_input, 0, 0, &mut output, &empty);

    // Fast side: the SAME input narrowed to f32, normalized by
    // external_normalize_f32 with the FastBlstm's own (narrowed) tail.
    let fast_net = FastBlstm::from_flat(&spec, &flat).unwrap();
    let flat_in: Vec<f64> = input.iter().copied().collect();
    let mut fast_input = FastMatrix::from_f64_rows(input.nrows(), input.ncols(), &flat_in);
    external_normalize_f32(
        &mut fast_input,
        fast_net.normalize_mean(),
        fast_net.normalize_std(),
    );

    let mut max_abs = 0.0_f64;
    let mut max_rel = 0.0_f64;
    for r in 0..fast_input.rows {
        for c in 0..fast_input.cols {
            let e = exact_input[[r, c]];
            let f = fast_input.get(r, c) as f64;
            let d = (e - f).abs();
            max_abs = max_abs.max(d);
            max_rel = max_rel.max(d / e.abs().max(1e-6));
        }
    }
    println!("MEASURE fast_type1_normalization: max_abs={max_abs:.3e} max_rel={max_rel:.3e}");
    assert!(
        max_rel < REL_PIN,
        "fast type-1 normalization rel {max_rel} exceeds pin {REL_PIN}"
    );
}

fn fixture_phase0_cfg() -> std::path::PathBuf {
    common::fixture_phase0("1_worker_1.config")
}

// ---------------------------------------------------------------------------
// (5): the S1.7 causality-cost leg -- MEASURED and RECORDED, never gated.
// ---------------------------------------------------------------------------

/// tier2_spectral.config with the SAME non-normalization overrides
/// `stage_frozen_tier2` bakes (thread count, offset/duration, the tuple-A pack,
/// `Inference_Path fast`) but WITHOUT `Audio_fixed_gain`/
/// `BLSTM_InputNormalizationType 1` -- the file's own `-1` (self-normalization)
/// stays in effect. The self-norm sibling half of the causality-cost comparison.
fn self_norm_sibling_map() -> IndexMap<String, String> {
    let base = std::fs::read_to_string(common::fixture_phase4a("tier2_spectral.config")).unwrap();
    let mut m = speech::legacy_config::parse_legacy_config(&base);
    m.insert("numOuterThreads".into(), "1".into());
    m.insert("Audio_offset".into(), "0.0".into());
    m.insert("Audio_max_duration".into(), "3600".into());
    m.insert(
        "BLSTM_weightsFile".into(),
        common::fixture_phase0("NNweights_config1.bin")
            .to_str()
            .unwrap()
            .into(),
    );
    m.insert("Inference_Path".into(), "fast".into());
    m
}

/// Build a fast bag from `map`, read `wav` with `gain`, run channel 0 through
/// `run_get_segmentation`, and return (posterior row, final segmentation).
fn run_fast(
    map: &mut IndexMap<String, String>,
    wav: &Path,
    gain: Option<f64>,
) -> (Vec<f64>, Segmentation) {
    let mut bag = BagOfProcessors::from_configs(std::slice::from_mut(map), image_mode()).unwrap();
    let mut audio = read_audio(wav, 0.0, 3600.0, 0, gain).unwrap();
    let dur = (audio.data.ncols() as f64 - 1.0) / audio.sample_rate as f64;
    let mut seg = vec![Segmentation::new(dur)];
    bag.run_get_segmentation(0, &mut audio, &mut seg).unwrap();
    let row = match bag.processor(0) {
        Processor::FastSpectral(s) => s.last_result_rows()[0].clone(),
        _ => panic!("expected FastSpectral"),
    };
    (row, seg.remove(0))
}

#[test]
fn causality_cost_frozen_vs_self_norm() {
    // Spec S1.7 (REPORTED, not gated): the offline fast path under FROZEN stats
    // (Audio_fixed_gain + InputNormalizationType 1) vs the SAME fast path under
    // SELF-NORM (no Audio_fixed_gain, InputNormalizationType -1, the tier2
    // default) -- both on the SAME staged mono fixture, the SAME tuple-A pack on
    // both sides (spec R4). The gain is baked to this fixture's own self-norm
    // statistic (stage.fixed_gain), so the AUDIO-LEVEL scaling coincides between
    // the two runs by construction; the remaining delta isolates the
    // type-1-vs-self-norm INPUT NORMALIZATION change alone.
    //
    // MEASURED (Apple Silicon dev box, 60 s mono, tuple-A): fixed_gain=4.9247e-1,
    // post_max_abs=9.956e-1, boundary rows frozen=2 (the untouched seed
    // hypothesis [Other@0, End@dur] -- ZERO detections) vs self_norm=17. The
    // MECHANISM (adjudicated with an exact-tree cross-run before recording: the
    // exact path under the SAME frozen stats produces the IDENTICAL 2-row
    // collapse, inter-path posterior delta 1.28e-7 -- so this is the MODE, not a
    // fast-path defect): the tuple-A net was TRAINED under type -1
    // self-normalization (per-sequence standardize + asinh); its pack-carried
    // type-1 tail is a plain affine standardization with 2015-training-corpus
    // statistics, under which the net's posterior saturates high (~0.9995) for
    // the WHOLE file -- no rising crossing ever fires, so the hypothesis stays at
    // the seed. A large causality cost for THIS pack/config pairing is the honest
    // measurement (recorded in RESULTS.md); the corpus leg (T8) re-measures on
    // the phase-6 SAD checkpoint.
    let dir = tempfile::tempdir().unwrap();
    let stage = common::stage_frozen_tier2(dir.path());
    let frozen_text = std::fs::read_to_string(&stage.config_path).unwrap();

    let mut frozen_map = speech::legacy_config::parse_legacy_config(&frozen_text);
    let (post_frozen, seg_frozen) =
        run_fast(&mut frozen_map, &stage.wav_path, Some(stage.fixed_gain));

    let mut self_norm_map = self_norm_sibling_map();
    let (post_self, seg_self) = run_fast(&mut self_norm_map, &stage.wav_path, None);

    assert_eq!(
        post_frozen.len(),
        post_self.len(),
        "posterior row length must match (same audio, same net shape)"
    );
    let mut max_abs = 0.0_f64;
    let mut nan_mismatch = 0usize;
    let mut n_finite = 0usize;
    for (&f, &s) in post_frozen.iter().zip(post_self.iter()) {
        if f.is_nan() || s.is_nan() {
            if f.is_nan() != s.is_nan() {
                nan_mismatch += 1;
            }
            continue;
        }
        n_finite += 1;
        max_abs = max_abs.max((f - s).abs());
    }

    let se = seg_frozen.segments();
    let ss = seg_self.segments();
    let mut max_dt = 0.0_f64;
    let mut boundary_note = String::new();
    if se.len() == ss.len() {
        for (a, b) in se.iter().zip(ss.iter()) {
            max_dt = max_dt.max((a.begin - b.begin).abs());
        }
    } else {
        boundary_note =
            " (boundary-row COUNT differs -- max_dt not comparable; the mode difference IS the boundary set)"
                .to_string();
    }

    println!(
        "MEASURE phase8_causality_cost: fixed_gain={:.6e} post_max_abs={max_abs:.6e} \
         nan_pattern_mismatches={nan_mismatch} boundary_max_dt={max_dt:.6e}s \
         boundary_rows frozen={} self_norm={}{boundary_note}",
        stage.fixed_gain,
        se.len(),
        ss.len()
    );

    // Sanity only (spec S1.7: reported, never a pass/fail pin on the delta). The
    // gain is baked to coincide, so a NaN-pattern mismatch (different overlap
    // coverage on the SAME audio/windowing) or a non-finite/empty posterior
    // comparison would indicate a genuine defect, not the expected
    // type-1-vs-self-norm mode difference.
    assert_eq!(
        nan_mismatch, 0,
        "the two modes must cover the SAME overlap rows (same audio, same windowing)"
    );
    assert!(
        n_finite > 0 && max_abs.is_finite(),
        "the posterior comparison must cover finite rows"
    );
}
