//! Phase 9 Task 6: the CI parity leg for the CAUSAL f32 fast twins (spec S4.3) --
//! the exact f64 `BlstmSpectralSegmenter` (`Direction forward` + `Cell_Type
//! slstm|mamba`, driving `nn::cells::{SlstmLayer, MambaLayer}`) vs the f32
//! `FastSpectralSegmenter` (driving `fast::cells::FastCausalNet`), BOTH run through
//! `BagOfProcessors` on the committed phase-9 gate fixtures + the committed synthetic
//! tier-2 corpus audio.
//!
//! The fast path DIVERGES from exact BY DESIGN (f32 + realfft through the feature
//! front-end, f32 through the cell stack and the output MLP -- spec S4). Tolerances
//! are MEASURED first (the `MEASURE` prints, `cargo test -- --nocapture`), then pinned
//! with headroom. Segment COUNT + TYPES identical per channel is the HARD gate: a flip
//! is R1 STOP-and-adjudicate, never a silent widen.
//!
//! WINDOWING. `slstm_forward.config`/`mamba_forward.config` carry `BLSTM_window 0.0`,
//! which resolves to `window_size 0` -- the PLAIN whole-sequence forward on BOTH
//! sides, so no in-test window overlay is needed. That is not a convenience: a causal
//! cell inside a window has its state reset at every window boundary (spec S1.2's
//! "pointless-but-defined"), so window 0 is the only regime the causal fast driver
//! accepts and the only one the Task-7 streaming session can mirror.
//!
//! SUB-SAMPLING AND STACKS ARE LIVE, not bailed: the fixtures run
//! `BLSTM_LSTMSubSampling 4` (a single layer), and `causal_net_supports_stacks_and_
//! sub_sampling` in `fast::cells` covers the multi-layer chain. Both are what the
//! phase's real causal SAD arm needs (`configs/training/lre_sad.toml` is
//! `lstm_neuron_nb = "23,24,24"` / `lstm_sub_sampling = "4,1"`), which is why the fast
//! twins implement `Network::drive`'s chain rather than restricting to ratio 1.
//!
//! THE VACUITY QUESTION (the phase-2b SEG_STRUCT lesson: a decision layer exercised
//! only on non-crossing data proves nothing). MEASURED on the committed fixtures, and
//! it differs per cell:
//!
//! - `mamba_forward` IS already non-vacuous: channel 1 crosses
//!   `BLSTM_decision_thresh_rising 0.6` mid-file and the segmentation carries a real
//!   interior boundary (`Speech@1.1947`).
//! - `slstm_forward` is NOT: its posteriors span `[0.307, 0.870]`, high enough that the
//!   hysteresis latches SPEECH at frame 0 and never falls -- the segmentation reacts
//!   (the head class is Speech, not the seeded Other) but has NO interior boundary, so
//!   a boundary-time comparison on it would compare nothing.
//!
//! PHASE 10 TASK 6 adds the `cfc` rows (`cfc_forward.config` + `cfc_forward_seed.bin`,
//! the committed Task-3 fixture) to every leg below -- the suite is cell-parametric, so
//! this is new ROWS, not new machinery. It is also the INDEPENDENT-IMPLEMENTATION oracle
//! that structurally closes the T3-recorded double-swap residual: the exact
//! `CellLayer::Cfc` arms and the f32 `FastCfc` kernel are two separately-written
//! transcriptions of spec S1.1/S1.2, and a same-direction mistake in the exact tree's
//! enum delegation (a swapped arm, a mis-threaded geometry) no longer cancels out --
//! it would have to be reproduced, identically, in a kernel that shares no code with it.
//!
//! [`causal_parity_crossing_is_exercised`] closes the gap uniformly: it sweeps a
//! WEIGHT-SPACE offset on the output layer's single bias -- a pure post-recurrence
//! level shift that leaves every cell weight, and therefore the whole SHAPE of the
//! posterior curve, untouched -- until the segmentation has interior boundaries, then
//! re-asserts exact-vs-fast identity there. It picks `+0.0` for mamba (no edit needed)
//! and `-0.5` for sLSTM (which yields 3 interior boundaries across the two channels,
//! including a full Other -> Speech -> Other on channel 1).

use std::path::PathBuf;

use indexmap::IndexMap;
use speech::audio::read_audio;
use speech::cli::{Mode, ModeKind};
use speech::engine::bag_of_processors::{BagOfProcessors, Processor};
use speech::io::binary::{read_weight_vector, write_matrix};
use speech::tasks::segmentation::{SegClass, Segmentation};

fn ref_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/reference_data")
}

fn fixture(rel: &str) -> PathBuf {
    ref_dir().join(rel)
}

fn image_mode() -> Mode {
    Mode {
        kind: ModeKind::Image,
        verbose: false,
    }
}

/// One phase-9 causal fixture config as a map, pointed at the committed seed pack and
/// (optionally) `Inference_Path`. `weights_override` swaps in a locally written pack
/// (the crossing leg).
fn causal_map(
    cell: &str,
    inference: Option<&str>,
    weights_override: Option<&PathBuf>,
) -> IndexMap<String, String> {
    let text = std::fs::read_to_string(fixture(&format!("phase9/{cell}_forward.config"))).unwrap();
    let mut m = speech::legacy_config::parse_legacy_config(&text);
    let pack = match weights_override {
        Some(p) => p.clone(),
        None => fixture(&format!("phase9/{cell}_forward_seed.bin")),
    };
    m.insert(
        "BLSTM_weightsFile".into(),
        pack.to_str().unwrap().to_string(),
    );
    if let Some(p) = inference {
        m.insert("Inference_Path".into(), p.into());
    }
    m
}

/// Run one path (exact when `inference == None`, fast when `Some("fast")`) through the
/// bag on the committed 2 s tier-2 excerpt. Returns the per-channel
/// PRE-`results_to_segmentation` posterior rows + the per-channel segmentations.
fn run_path(
    cell: &str,
    inference: Option<&str>,
    weights_override: Option<&PathBuf>,
) -> (Vec<Vec<f64>>, Vec<Segmentation>) {
    let mut m = causal_map(cell, inference, weights_override);
    let mut bag =
        BagOfProcessors::from_configs(std::slice::from_mut(&mut m), image_mode()).unwrap();

    let mut audio = read_audio(&fixture("phase4a/corpus/f1.wav"), 0.0, 2.0, 0, None).unwrap();
    let dur = (audio.data.ncols() as f64 - 1.0) / audio.sample_rate as f64;
    let n_chan = audio.data.nrows();
    let mut seg: Vec<Segmentation> = (0..n_chan).map(|_| Segmentation::new(dur)).collect();
    bag.run_get_segmentation(0, &mut audio, &mut seg).unwrap();

    let posteriors = match bag.processor(0) {
        Processor::Spectral(s) => s.last_result_rows().to_vec(),
        Processor::FastSpectral(s) => s.last_result_rows().to_vec(),
        _ => panic!("unexpected processor variant for cell={cell} inference={inference:?}"),
    };
    (posteriors, seg)
}

/// Interior boundaries across every channel: segments that are neither the seeded head
/// (`begin == 0`) nor the `End` sentinel -- i.e. genuine mid-file decisions.
fn interior_boundaries(segs: &[Segmentation]) -> usize {
    segs.iter()
        .map(|s| {
            s.segments()
                .iter()
                .filter(|x| x.ty != SegClass::End && x.begin > 0.0)
                .count()
        })
        .sum()
}

/// Compare two runs' posteriors + segmentations. Returns `(max_abs, max_rel, max_dt)`;
/// PANICS on any structural mismatch (count/type/length), which is the R1 hard gate.
fn compare(
    label: &str,
    e: &(Vec<Vec<f64>>, Vec<Segmentation>),
    f: &(Vec<Vec<f64>>, Vec<Segmentation>),
) -> (f64, f64, f64) {
    let (post_e, seg_e) = e;
    let (post_f, seg_f) = f;
    assert_eq!(post_e.len(), post_f.len(), "{label}: channel count");
    assert_eq!(
        seg_e.len(),
        seg_f.len(),
        "{label}: channel count (segments)"
    );

    let mut max_abs = 0.0_f64;
    let mut max_rel = 0.0_f64;
    for chan in 0..post_e.len() {
        assert_eq!(
            post_e[chan].len(),
            post_f[chan].len(),
            "{label}: posterior row length chan {chan}"
        );
        for (&a, &b) in post_e[chan].iter().zip(post_f[chan].iter()) {
            assert!(
                a.is_finite() && b.is_finite(),
                "{label}: non-finite posterior ({a}, {b})"
            );
            let d = (a - b).abs();
            max_abs = max_abs.max(d);
            // Posteriors are logistic outputs in [0,1]; scale by max(|a|, 1e-2) so a
            // tiny posterior does not inflate a tiny absolute delta (the phase-7
            // convention, verbatim).
            max_rel = max_rel.max(d / a.abs().max(1e-2));
        }
    }

    let mut max_dt = 0.0_f64;
    for chan in 0..seg_e.len() {
        let se = seg_e[chan].segments();
        let sf = seg_f[chan].segments();
        assert_eq!(
            se.len(),
            sf.len(),
            "{label}: segment COUNT chan {chan} (R1)"
        );
        for (k, (a, b)) in se.iter().zip(sf.iter()).enumerate() {
            assert_eq!(a.ty, b.ty, "{label}: segment TYPE chan {chan} seg {k} (R1)");
            max_dt = max_dt.max((a.begin - b.begin).abs());
        }
    }
    (max_abs, max_rel, max_dt)
}

// ---------------------------------------------------------------------------
// The CI parity legs.
// ---------------------------------------------------------------------------

/// The COARSE output-bias offsets [`causal_parity_crossing_is_exercised`] tries first.
/// Phase-9's list verbatim: sLSTM settles at `-0.5` and mamba at `0.0`, so both rows are
/// byte-unchanged by the phase-10 fine-grid extension below.
const COARSE_OFFSETS: [f64; 7] = [0.0, -0.5, -1.0, 0.5, -1.5, 1.0, -2.0];

/// The FINE fallback grid (1/16 steps over `[-1, +1]`), reached only when no coarse
/// offset crosses. THE CfC ROW NEEDS IT, and the reason is a measured property of that
/// fixture worth stating plainly rather than hiding behind a longer list.
///
/// The committed CfC seed net's posterior on the 2 s tier-2 excerpt is FLAT and
/// OSCILLATORY -- it spans `[0.367, 0.705]` at offset 0, with the high region
/// (indices ~5-20) broken by 1-3 frame dips and a uniformly ~0.45 tail. The decision
/// layer's rising/falling pair (0.6 / 0.3) needs a logit swing of ~1.25 to make a round
/// trip, and `min_speech`/`min_silence` 0.2 s (5 rows at this 0.04 s time step) plus
/// 0.1 s padding then erase anything shorter. MEASURED by a 129-point sweep over
/// `[-4, +4]` at 1/16: EXACTLY ONE offset (`-0.0625`) yields an interior boundary, and
/// exactly ONE boundary there.
///
/// So this leg's CfC row is worth precisely what it claims -- ONE real interior
/// boundary, compared bit-for-bit through the shared f64 decision layer (the
/// `MIN_INTERIOR_TYPE1` posture in `phase9_stream_causal.rs`, same shape). It is NOT a
/// rich boundary-set comparison; the sLSTM row (3 boundaries, including a full
/// Other -> Speech -> Other) is. The thinness also means a front-end change that shifts
/// the posterior slightly will make the sweep find NOTHING and PANIC with that exact
/// message -- a loud failure to adjudicate, not a silent weakening.
fn fine_offsets() -> impl Iterator<Item = f64> {
    // The coarse entries are FILTERED OUT: they have already been tried and failed by the
    // time this iterator is reached, so re-running them would be wasted engine
    // invocations.
    (-16i32..=16)
        .map(|k| f64::from(k) * 0.0625)
        .filter(|o| !COARSE_OFFSETS.contains(o))
}

/// MEASURED on this box (M4 Pro / macOS 25.5 / Apple libm) -- see the per-leg prints;
/// the worst across all four (cell x leg) runs is `max_rel 7.74e-6` (mamba), leaving
/// ~13x headroom. Pinned at `measured * 10` rounded up, then widened once for
/// cross-platform realfft/faer SIMD variance (CI is x86, this box is ARM), exactly as the
/// phase-7 SAD leg documents. A FAILURE HERE MEANS RE-MEASURE AND ADJUDICATE, never widen.
///
/// RE-MEASURED by the Phase 10 Task 5 S4 sweep (full-f32 mel at both fast sites): worst
/// `max_rel` 4.13e-6 -> 7.74e-6 (mamba, unchanged leg-for-leg otherwise; slstm 1.40e-6),
/// `max_dt` still EXACTLY 0.0 with boundary count/types identical on every leg, so the
/// PINS ARE UNCHANGED -- the growth is absorbed by the existing headroom.
///
/// PHASE 10 TASK 6 adds the `cfc` rows, MEASURED: plain `max_abs 4.34e-7 / max_rel
/// 6.90e-7`, crossing `max_abs 4.11e-7 / max_rel 6.77e-7`, `max_dt` EXACTLY 0.0 on both
/// with boundary count/types identical. That is the TIGHTEST of the three cells (~145x
/// headroom against these pins, against mamba's ~13x), so the pins stay UNCHANGED here
/// too -- mamba remains the worst row and therefore what they are sized for.
const POST_REL_PIN: f64 = 1.0e-4;
const POST_ABS_PIN: f64 = 1.0e-4;

#[test]
fn causal_parity_exact_vs_fast() {
    for cell in ["slstm", "mamba", "cfc"] {
        let e = run_path(cell, None, None);
        let f = run_path(cell, Some("fast"), None);
        let (max_abs, max_rel, max_dt) = compare(cell, &e, &f);
        println!(
            "MEASURE {cell} causal exact-vs-fast: max_abs={max_abs:e} max_rel={max_rel:e} \
             max_dt={max_dt} interior={} rows={}",
            interior_boundaries(&e.1),
            e.0[0].len()
        );
        // Non-vacuity of the POSTERIOR comparison: a real sequence, not a constant.
        assert!(e.0[0].len() > 20, "{cell}: too few posterior rows");
        let first = e.0[0][0];
        assert!(
            e.0[0].iter().any(|&v| v != first),
            "{cell}: the exact posterior is constant -- the comparison is vacuous"
        );
        assert!(
            max_abs < POST_ABS_PIN && max_rel < POST_REL_PIN,
            "{cell}: posterior drift max_abs={max_abs:e} max_rel={max_rel:e}"
        );
        // Boundary times come from the SHARED f64 decision layer fed the two posterior
        // vectors; identical crossings give identical times.
        assert_eq!(max_dt, 0.0, "{cell}: boundary max_dt {max_dt} != 0 (R1)");
    }
}

/// THE DECISION-LAYER leg (see the module doc's vacuity note). Sweeps an offset on the
/// output MLP's single bias -- a post-recurrence level shift, so the causal stack still
/// produces the whole SHAPE of the posterior curve -- until the exact run's
/// segmentation carries interior boundaries, then asserts that exact and fast agree on
/// count, types and times EXACTLY on that run.
#[test]
fn causal_parity_crossing_is_exercised() {
    let tmp = tempfile::tempdir().unwrap();
    for cell in ["slstm", "mamba", "cfc"] {
        // The output layer's bias is the LAST weight before the `2*input_size`
        // normalize tail (pack layout `[stack | output MLP | mean | std]`, and the
        // fixture's output MLP is a single `4 -> 1` layer: 4 weights then 1 bias). The
        // tail length is DERIVED from the config's own `BLSTM_LSTMNeuronNb[0]`, not
        // hardcoded, so a fixture regeneration at a different input width relocates the
        // index instead of silently perturbing some interior weight.
        let input_size: usize = causal_map(cell, None, None)["BLSTM_LSTMNeuronNb"]
            .split(',')
            .next()
            .unwrap()
            .trim()
            .parse()
            .unwrap();
        let seed = fixture(&format!("phase9/{cell}_forward_seed.bin"));
        let base = read_weight_vector(&seed).unwrap();
        let bias_idx = base.len() - 2 * input_size - 1;
        let path = tmp.path().join(format!("{cell}_crossing.bin"));

        // `0.0` FIRST, so a fixture that already crosses is used AS COMMITTED and the
        // sweep is a fallback, not a default detour. COARSE steps first (slstm settles at
        // -0.5 and mamba at 0.0, both inside the original list, so those two rows are
        // byte-unchanged by the phase-10 extension), then a FINE 1/16 grid over
        // `[-1, +1]` -- see [`fine_offsets`] for why the CfC row needs it.
        let mut chosen: Option<(f64, usize)> = None;
        for offset in COARSE_OFFSETS.iter().copied().chain(fine_offsets()) {
            let mut probe = base.clone();
            probe[bias_idx] += offset;
            write_matrix(&path, probe.len(), 1, &probe).unwrap();
            let interior = interior_boundaries(&run_path(cell, None, Some(&path)).1);
            if interior > 0 {
                println!(
                    "MEASURE crossing[{cell}]: output-bias offset {offset:+} -> {interior} interior boundaries"
                );
                chosen = Some((offset, interior));
                break;
            }
        }
        let (offset, want_interior) = chosen
            .unwrap_or_else(|| panic!("{cell}: no bias offset produced an interior boundary"));

        let mut flat = base.clone();
        flat[bias_idx] += offset;
        write_matrix(&path, flat.len(), 1, &flat).unwrap();

        let e = run_path(cell, None, Some(&path));
        let f = run_path(cell, Some("fast"), Some(&path));
        let (max_abs, max_rel, max_dt) = compare(&format!("{cell}/crossing"), &e, &f);
        let interior = interior_boundaries(&e.1);
        println!(
            "MEASURE {cell} causal CROSSING (output bias {offset:+}): max_abs={max_abs:e} \
             max_rel={max_rel:e} max_dt={max_dt} interior={interior}"
        );
        assert_eq!(
            interior, want_interior,
            "{cell}: the sweep is not reproducible"
        );
        assert!(
            interior > 0,
            "{cell}: the decision-layer comparison is vacuous"
        );
        assert!(
            max_abs < POST_ABS_PIN && max_rel < POST_REL_PIN,
            "{cell}: crossing posterior drift max_abs={max_abs:e} max_rel={max_rel:e}"
        );
        assert_eq!(
            max_dt, 0.0,
            "{cell}: crossing boundary max_dt {max_dt} != 0 (R1)"
        );
    }
}

/// The dispatch table: what the fast tree now BUILDS, and the two combinations it still
/// refuses (spec S4.2). The bail legs point `BLSTM_weightsFile` at nothing, so the
/// SHAPE check is what fires -- not a pack-length mismatch downstream of it.
#[test]
fn fast_dispatch_bails_and_builds_per_cell_and_direction() {
    let build = |m: &mut IndexMap<String, String>| -> anyhow::Result<BagOfProcessors> {
        BagOfProcessors::from_configs(std::slice::from_mut(m), image_mode())
    };

    // ACCEPTED: all THREE causal cells build a fast net, from their committed packs
    // (`cfc` joined in phase-10 Task 6, which replaced its interim `element_count` bail
    // with a real `FastCfc`).
    for cell in ["slstm", "mamba", "cfc"] {
        let mut m = causal_map(cell, Some("fast"), None);
        let bag = build(&mut m).unwrap_or_else(|e| panic!("fast + causal {cell} must build: {e}"));
        assert!(
            matches!(bag.processor(0), Processor::FastSpectral(_)),
            "{cell}: expected the fast SAD driver"
        );
    }

    // BAILED (1): a new cell in the BIDIRECTIONAL direction -- the bidirectional fast
    // twin is a named follow-on (spec non-goals), so it must fail LOUDLY rather than
    // run half an architecture.
    for cell in ["slstm", "mamba", "cfc"] {
        let mut m = causal_map(cell, Some("fast"), None);
        m.insert("BLSTM_Direction".into(), "bidirectional".into());
        // A bidirectional net's output MLP consumes 2*hidden, so widen it or
        // `BlstmConfig::from_legacy` rejects the config before the fast dispatch runs.
        m.insert("BLSTM_OutputNeuronNb".into(), "8,1".into());
        m.insert("BLSTM_weightsFile".into(), String::new());
        let err = build(&mut m)
            .err()
            .unwrap_or_else(|| panic!("fast + bidirectional {cell} must bail"));
        let msg = err.to_string();
        assert!(
            msg.contains(&format!(
                "cell type '{cell}' is not supported on the fast inference path"
            )) && msg.contains("BIDIRECTIONAL"),
            "expected a bidirectional-cell bail for {cell}, got: {err}"
        );
    }

    // BAILED (2): the causal direction with the LSTM cell -- a forward-only LSTM fast
    // twin is a named follow-on (spec S5.3).
    let mut m = causal_map("slstm", Some("fast"), None);
    m.insert("BLSTM_Cell_Type".into(), "lstm".into());
    m.insert("BLSTM_weightsFile".into(), String::new());
    let err = build(&mut m)
        .err()
        .unwrap_or_else(|| panic!("fast + causal LSTM must bail"));
    let msg = err.to_string();
    assert!(
        msg.contains("Direction 'forward' is not supported on the fast inference path")
            && msg.contains("named follow-on"),
        "expected a causal-LSTM bail, got: {err}"
    );

    // ...and the EXACT path is unaffected: the same causal-LSTM config builds there
    // (the fast bail is a SCOPE statement about the f32 tree, not a config error).
    let mut exact = causal_map("slstm", Some("exact"), None);
    exact.insert("BLSTM_Cell_Type".into(), "lstm".into());
    exact.insert("BLSTM_weightsFile".into(), String::new());
    build(&mut exact).expect("Inference_Path exact + causal LSTM must build");
}

/// A causal fast net runs WINDOW 0 and nothing else: windowing resets a causal cell's
/// state at every boundary (spec S1.2), and the streaming session forbids it outright
/// (S5.3). Pinned at `get_segmentation`, where the window is resolved.
///
/// NOTE (Task 6 finding, reported not fixed -- it is exact-tree code and out of this
/// task's scope): the EXACT tree does not merely find this regime pointless, it PANICS
/// on it. `feed_forward_backward_overlap`'s final quotient loop (`nn/blstm.rs:1974`)
/// reproduces the legacy quirk of bounding BOTH the forward and backward accumulators
/// by `_OutputForward.cols()`, and a `Direction::Forward` net's backward accumulator is
/// ZERO-width -> `ndarray: index out of bounds`. Measured: `slstm_forward.config` with
/// `BLSTM_window 0.5` panics on the exact path, while the same edit on
/// `slstm_bidirectional.config` runs clean, so it is causal-specific. The fast bail
/// below is therefore strictly the better behaviour of the two, and this test asserts
/// only the fast side.
#[test]
fn fast_causal_bails_on_windowed_inference() {
    let mut m = causal_map("slstm", Some("fast"), None);
    // 0.5 s window at 8 kHz with spectrum_shift 0.01 -> window_size 25: a genuinely
    // windowed dispatch, not a degenerate one.
    m.insert("BLSTM_window".into(), "0.5".into());
    let mut bag =
        BagOfProcessors::from_configs(std::slice::from_mut(&mut m), image_mode()).unwrap();
    let mut audio = read_audio(&fixture("phase4a/corpus/f1.wav"), 0.0, 2.0, 0, None).unwrap();
    let dur = (audio.data.ncols() as f64 - 1.0) / audio.sample_rate as f64;
    let n_chan = audio.data.nrows();
    let mut seg: Vec<Segmentation> = (0..n_chan).map(|_| Segmentation::new(dur)).collect();
    let err = bag
        .run_get_segmentation(0, &mut audio, &mut seg)
        .err()
        .unwrap_or_else(|| panic!("fast causal + windowed inference must bail"));
    assert!(
        err.to_string()
            .contains("windowed inference is unsupported"),
        "expected a causal-window bail, got: {err}"
    );
}
