//! Phase 10 Task 7: the CI parity leg for the BIDIRECTIONAL f32 cell twins (spec S5) --
//! the exact f64 `BlstmSpectralSegmenter` (`Direction bidirectional` + `Cell_Type
//! slstm|mamba|cfc`, driving `nn::cells::{SlstmLayer, MambaLayer, CfcLayer}` through
//! `Network::feed_forward` + `feed_forward_reverse` + `feed_forward_double`) vs the f32
//! `FastSpectralSegmenter` driving `fast::bicell::FastBiCell`. BOTH run through
//! `BagOfProcessors` on the committed phase-9 bidirectional gate fixtures + the
//! committed synthetic tier-2 corpus audio.
//!
//! PHASE 11 TASK 6 (spec S5.3/S5.4) joined a FOURTH cell, `transformer` --
//! `nn::cells::transformer::TransformerLayer` on the exact side, `FastTransformer` (via
//! `fast::bicell::FastBiCell`, Task 5's causal kernel driven with `reverse = true` for the
//! backward stack) on the fast side. No transformer-specific `FastBiCell` code exists:
//! [`CELLS`] growing by one row is the whole diff this task made to the ARITHMETIC tier,
//! exactly as the spec predicted.
//!
//! THIS IS THE ARITHMETIC TIER for the bidirectional twins (the phase-9 battery lesson,
//! CLAUDE.md's standing rule (2)): every self-consistent leg compares two runs of the
//! SAME kernel and is structurally blind to a mutation inside it. Nothing here is
//! self-consistent -- the oracle is the exact f64 tree, a separately-written
//! implementation that shares no code with `fast::bicell` below the config layer.
//!
//! TWO REGIMES, both live and both pinned (spec S5: "BiCell runs OVERLAP or plain per
//! the config, exactly as the exact tree does"):
//!
//! - PLAIN: the committed fixtures carry `BLSTM_window 0.0` -> `window_size 0` -> the
//!   whole-sequence forward on both sides.
//! - OVERLAP: an in-test `BLSTM_window 0.5` overlay (the phase-9 pattern) -> at 8 kHz
//!   with `BLSTM_spectrum_shift 0.01` (ssif 80) that is `window_size 25`, and
//!   `BLSTM_shift 0.1` -> `window_shift 10`: a genuinely overlapping window grid
//!   (stride 10 over a 51-frame span), not a degenerate one. This is the leg that
//!   exercises `FastBiCell::feed_forward_overlap`'s FRESH accumulate/average loop
//!   against the exact `feed_forward_backward_overlap`.
//!
//! The fast path DIVERGES from exact BY DESIGN (f32 + realfft + the f32 mel front-end,
//! f32 through both cell stacks and the output MLP -- spec S4/S5). Tolerances are
//! MEASURED first (the `MEASURE` prints, `cargo test -- --nocapture`), then pinned with
//! headroom. Segment COUNT + TYPES identical per channel and `max_dt` EXACTLY 0.0 are
//! the HARD gates: a flip is R1 STOP-and-adjudicate, never a silent widen.
//!
//! VACUITY (the phase-2b SEG_STRUCT lesson). [`bicell_parity_crossing_is_exercised`]
//! sweeps an offset on the output MLP's single bias -- a pure post-recurrence level
//! shift that leaves every cell weight, and therefore the whole SHAPE of the posterior
//! curve, untouched -- until the exact run's segmentation carries interior boundaries,
//! then re-asserts exact-vs-fast identity there. The measured offsets are printed.

use std::path::PathBuf;

use indexmap::IndexMap;
use speech::audio::read_audio;
use speech::cli::{Mode, ModeKind};
use speech::engine::bag_of_processors::{BagOfProcessors, Processor};
use speech::io::binary::{read_weight_vector, write_matrix};
use speech::tasks::segmentation::{SegClass, Segmentation};

/// The four cells that now have a bidirectional f32 twin (spec S5; `transformer` joined
/// in phase-11 Task 6, spec S5.3 -- "no transformer-specific bicell code beyond the enum
/// arm", so this row is the whole diff `FastBiCell` needed). The LSTM cell is absent by
/// construction: `(lstm, bidirectional)` IS the phase-7 `FastBlstm`, which this task
/// leaves BYTE-UNTOUCHED.
const CELLS: [&str; 4] = ["slstm", "mamba", "cfc", "transformer"];

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

/// One committed phase-9 BIDIRECTIONAL fixture config as a map, pointed at its seed
/// pack and (optionally) `Inference_Path` + a `BLSTM_window` overlay.
fn bicell_map(
    cell: &str,
    inference: Option<&str>,
    weights_override: Option<&PathBuf>,
    window: Option<&str>,
) -> IndexMap<String, String> {
    let text =
        std::fs::read_to_string(fixture(&format!("phase9/{cell}_bidirectional.config"))).unwrap();
    let mut m = speech::legacy_config::parse_legacy_config(&text);
    let pack = match weights_override {
        Some(p) => p.clone(),
        None => fixture(&format!("phase9/{cell}_bidirectional_seed.bin")),
    };
    m.insert(
        "BLSTM_weightsFile".into(),
        pack.to_str().unwrap().to_string(),
    );
    if let Some(p) = inference {
        m.insert("Inference_Path".into(), p.into());
    }
    if let Some(w) = window {
        m.insert("BLSTM_window".into(), w.into());
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
    window: Option<&str>,
) -> (Vec<Vec<f64>>, Vec<Segmentation>) {
    let mut m = bicell_map(cell, inference, weights_override, window);
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
/// Transcribed from `phase9_fast_parity.rs::compare` -- deliberately the SAME
/// comparator, so the causal and bidirectional tiers report the same statistic.
///
/// NaN-AWARE, and that is not decoration here: the overlap driver's count quotient is
/// `0/0` on any output row no window covered (the reproduced legacy quirk,
/// `nn/blstm.rs:2055`), so both paths can legitimately carry NaN posteriors. A NaN must
/// appear in BOTH or NEITHER, at the same index; a NaN that appears on one side only is
/// a structural divergence, not a tolerance question.
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
        for (k, (&a, &b)) in post_e[chan].iter().zip(post_f[chan].iter()).enumerate() {
            assert_eq!(
                a.is_nan(),
                b.is_nan(),
                "{label}: NaN structure chan {chan} row {k} ({a}, {b})"
            );
            if a.is_nan() {
                continue;
            }
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
// Tolerances.
// ---------------------------------------------------------------------------

/// MEASURED on this box (M4 Pro / macOS 25.5 / Apple libm) -- see the per-leg `MEASURE`
/// prints. Worst across the original twelve (cell x regime x {base, crossing}) runs:
///
/// | leg | max_abs | max_rel |
/// |---|---|---|
/// | slstm plain / overlap | 7.34e-7 / 7.51e-7 | 3.69e-6 / 3.81e-6 |
/// | mamba plain / overlap | 1.87e-6 / 1.86e-6 | **1.81e-5** / 1.80e-5 |
/// | cfc plain / overlap | 9.52e-7 / 9.43e-7 | 3.25e-6 / 3.07e-6 |
/// | worst crossing (slstm plain, gain 2) | **2.00e-6** | 6.39e-6 |
///
/// `max_dt` is EXACTLY 0.0 on all twelve, with boundary count + types identical.
///
/// PINNED at `measured * 10` rounded up: `2.0e-4` relative (mamba's 1.81e-5 sets it) and
/// `1.0e-4` absolute (~50x headroom). The relative pin is LOOSER than the causal tier's
/// `1.0e-4` (`phase9_fast_parity.rs`) and that is a MEASUREMENT, not a concession: the
/// bidirectional mamba net's worst row is 2.3x the causal one's (7.74e-6), which is what
/// a second recurrent stack plus a twice-as-wide dense fan-in buys. Note the two
/// statistics disagree about which leg is worst -- mamba's `max_rel` comes from an
/// ordinary ~1.9e-6 absolute delta divided by a posterior of ~0.10, well ABOVE the
/// comparator's 1e-2 scale floor, so it is a genuine relative number and not a
/// small-denominator artifact.
///
/// PHASE 11 TASK 6 adds the `transformer` rows (four more runs: plain, overlap, and the
/// two crossing legs at the wide-offset rungs `+11`/`+10` -- see
/// [`bicell_parity_crossing_is_exercised`]), MEASURED:
///
/// | leg | max_abs | max_rel |
/// |---|---|---|
/// | transformer plain / overlap | 2.86e-6 / 2.98e-6 | 6.59e-6 / 6.84e-6 |
/// | transformer crossing plain (gain 1, +11) / overlap (gain 1, +10) | 2.42e-6 / 1.57e-6 | 1.159e-5 / **1.242e-5** |
///
/// `max_dt` is EXACTLY 0.0 on all four, with boundary count + types identical (three
/// interior boundaries on the plain crossing, two on overlap -- both non-vacuous). Every
/// transformer number sits BELOW mamba's, so mamba stays the worst leg overall and
/// **both pins are UNCHANGED** (transformer's worst `measured * 10` is 1.242e-4, inside
/// `2.0e-4` at 1.6x headroom on top of what mamba already used) -- the `measured * 10`
/// convention was checked and did not require adjudication (spec R1/R3).
///
/// A FAILURE HERE MEANS RE-MEASURE AND ADJUDICATE, never widen. The HARD gates
/// (boundary count/type identity, `max_dt == 0.0`) carry the decision-level claim; these
/// two are drift detectors.
const POST_REL_PIN: f64 = 2.0e-4;
const POST_ABS_PIN: f64 = 1.0e-4;

/// The `BLSTM_window` overlay that selects the OVERLAP regime (see the module doc for
/// the resolved `window_size 25` / `window_shift 10`).
const OVERLAP_WINDOW: &str = "0.5";

// ---------------------------------------------------------------------------
// The CI parity legs.
// ---------------------------------------------------------------------------

/// PLAIN regime: the committed `BLSTM_window 0.0`. Exact bidirectional cell stack
/// (`feed_forward` + `feed_forward_reverse` + `feed_forward_double`) vs `FastBiCell`.
#[test]
fn bicell_parity_exact_vs_fast_plain() {
    for cell in CELLS {
        let e = run_path(cell, None, None, None);
        let f = run_path(cell, Some("fast"), None, None);
        let (max_abs, max_rel, max_dt) = compare(&format!("{cell}/plain"), &e, &f);
        // The posterior SPAN is printed (never asserted), the `phase9_fast_parity.rs`
        // causal-tier precedent: it is what explains whether the crossing leg below
        // lands on a small offset or needs the wide fallback.
        let (lo, hi) = e.0[0]
            .iter()
            .fold((f64::MAX, f64::MIN), |(a, b), &v| (a.min(v), b.max(v)));
        println!(
            "MEASURE {cell} bicell PLAIN exact-vs-fast: max_abs={max_abs:e} max_rel={max_rel:e} \
             max_dt={max_dt} interior={} rows={} span=[{lo:.4}, {hi:.4}] logit_span={:.4}",
            interior_boundaries(&e.1),
            e.0[0].len(),
            (hi / (1.0 - hi)).ln() - (lo / (1.0 - lo)).ln()
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
            "{cell}: plain posterior drift max_abs={max_abs:e} max_rel={max_rel:e}"
        );
        assert_eq!(
            max_dt, 0.0,
            "{cell}: plain boundary max_dt {max_dt} != 0 (R1)"
        );
    }
}

/// OVERLAP regime: the `BLSTM_window 0.5` overlay. This is the leg that pins
/// `FastBiCell::feed_forward_overlap`'s fresh window-firing/accumulate/average loop
/// against the exact `feed_forward_backward_overlap`.
#[test]
fn bicell_parity_exact_vs_fast_overlap() {
    for cell in CELLS {
        let e = run_path(cell, None, None, Some(OVERLAP_WINDOW));
        let f = run_path(cell, Some("fast"), None, Some(OVERLAP_WINDOW));
        let (max_abs, max_rel, max_dt) = compare(&format!("{cell}/overlap"), &e, &f);
        let (lo, hi) = e.0[0]
            .iter()
            .fold((f64::MAX, f64::MIN), |(a, b), &v| (a.min(v), b.max(v)));
        println!(
            "MEASURE {cell} bicell OVERLAP exact-vs-fast: max_abs={max_abs:e} \
             max_rel={max_rel:e} max_dt={max_dt} interior={} rows={} span=[{lo:.4}, {hi:.4}] \
             logit_span={:.4}",
            interior_boundaries(&e.1),
            e.0[0].len(),
            (hi / (1.0 - hi)).ln() - (lo / (1.0 - lo)).ln()
        );
        assert!(e.0[0].len() > 20, "{cell}: too few posterior rows");
        let first = e.0[0][0];
        assert!(
            e.0[0].iter().any(|&v| v != first),
            "{cell}: the exact posterior is constant -- the comparison is vacuous"
        );
        assert!(
            max_abs < POST_ABS_PIN && max_rel < POST_REL_PIN,
            "{cell}: overlap posterior drift max_abs={max_abs:e} max_rel={max_rel:e}"
        );
        assert_eq!(
            max_dt, 0.0,
            "{cell}: overlap boundary max_dt {max_dt} != 0 (R1)"
        );
    }
}

/// THE REGIME IS REALLY DIFFERENT -- the guard that keeps the two legs above from being
/// two copies of the same measurement. If a `BLSTM_window` overlay were silently ignored
/// (say the fast driver fell through to the plain forward on both), every parity number
/// would still be perfect and the overlap leg would prove nothing. The exact path's own
/// plain-vs-overlap posteriors must therefore DIFFER, on both sides.
#[test]
fn overlap_and_plain_are_distinct_regimes() {
    for cell in CELLS {
        for inference in [None, Some("fast")] {
            let plain = run_path(cell, inference, None, None).0;
            let over = run_path(cell, inference, None, Some(OVERLAP_WINDOW)).0;
            let differ = plain[0]
                .iter()
                .zip(over[0].iter())
                .filter(|(a, b)| !(a.is_nan() && b.is_nan()) && a != b)
                .count();
            println!(
                "MEASURE {cell} plain-vs-overlap ({inference:?}): {differ}/{} rows differ",
                plain[0].len()
            );
            assert!(
                differ > 0,
                "{cell} ({inference:?}): the window overlay changed nothing -- the overlap \
                 leg would be vacuous"
            );
        }
    }
}

/// THE DECISION-LAYER leg: it makes the boundary comparison NON-VACUOUS. Without it the
/// two legs above would agree on segmentations that carry no interior boundary at all,
/// which is the phase-2b SEG_STRUCT lesson in its purest form.
///
/// The knob is a two-parameter AFFINE map on the output MLP's single pre-activation:
/// `logit -> gain * (w . h) + b + offset`, i.e. the dense layer's weight ROWS scaled and
/// its bias shifted. BOTH CELL STACKS ARE UNTOUCHED -- `h` is exactly what the committed
/// net produces -- so the posterior curve keeps its shape up to a monotone affine map on
/// the logit, and every crossing compared below is a crossing of the REAL curve.
///
/// WHY A GAIN AND NOT JUST THE BIAS (phase 9's knob). MEASURED on these fixtures: a
/// bias-only sweep within `GAINS[0] x offsets()` (gain 1, `|offset| <= 4.0`) is enough for
/// five of the eight (cell x regime) rows, but NOT for `slstm/plain`, whose posterior
/// spans only `[0.105, 0.516]` -- a logit swing of 2.20 against the 1.25 the
/// rising/falling pair (0.6 / 0.3) needs for a round trip, which leaves no offset whose
/// crossings survive `min_speech`/`min_silence` 0.2 s (5 rows at this 0.04 s step). A
/// 129-point bias-only sweep over `[-8, +8]` finds ZERO interior boundaries there. Scaling
/// the weight rows widens the swing, and `gain 2` at offset `+2.0` produces one. The sweep
/// therefore tries `gain 1` (the phase-9 knob, bias only) FIRST at every offset, and only
/// then reaches for a gain -- so five rows are pinned on the weaker, more faithful
/// perturbation and exactly one (of the original six) is not.
///
/// PHASE 11 TASK 6 adds `transformer`, and its two rows need a THIRD lever -- not a
/// bigger gain, a bigger OFFSET (see [`wide_offsets`]). MEASURED: `transformer/plain`
/// spans `[0.0000, 0.9008]` (logit span 40.21, printed by
/// [`bicell_parity_exact_vs_fast_plain`]) and sits at `interior=0` AS COMMITTED -- unlike
/// `mamba`'s similarly wide `[0.0000, 1.0000]` span (logit span ~41.97), which happens to
/// cross at offset 0. A near-saturated curve's crossing point does not move at all under a
/// small shift (almost every row is already pinned at 0 or the ceiling), so the `+-4.0`
/// grid that suffices for six of the eight rows finds nothing; `+11` (plain) / `+10`
/// (overlap), still at `gain 1`, do. Reaching for a WIDER offset rather than a gain keeps
/// the weaker, more faithful perturbation (the phase-9 property: every cell weight
/// untouched) instead of trading it for the sharper one -- the opposite fix from
/// `slstm/plain`'s, because the two rows are opposite problems (a curve too FLAT to reach
/// the decision band vs one too SATURATED to be moved by a small shift).
///
/// The offsets are ordered `0.0` first (a fixture that already crosses is used AS
/// COMMITTED -- mamba does) then ascending in `|offset|`, so the chosen perturbation is
/// the SMALLEST one that works, deterministically -- and the wide fallback below is tried
/// only once the entire `GAINS x offsets()` grid is exhausted, so it never displaces a
/// rung the grid above already found.
///
/// TWO CLARIFICATIONS, recorded rather than left implicit (phase-11 Task 6 minor). First,
/// "smallest wins" holds PER STAGE, not as one globally-sorted search: `gain` is the OUTER
/// loop, so EVERY offset at `gain 1` (all the way to `+-4.0`) is tried before `gain 2` is
/// touched, and all of `gain 2` before `gain 3` -- meaning `gain 3, offset +0.25` is probed
/// deep inside the primary grid, while `gain 1, offset +5.0` is not reachable there AT ALL
/// (`offsets()` stops at `4.0`) and is only tried once the WHOLE primary grid (all three
/// gains) has failed, as the very first rung of [`wide_offsets`]. So a nominally "larger"
/// primary-grid perturbation can and does fire before a nominally "smaller" fallback one.
/// Second, the half-open interval `(4.0, 5.0)` is UNPROBED by construction at every gain --
/// `offsets()` tops out at `4.0` and [`wide_offsets`] starts at `5.0` -- a documented gap
/// rather than an oversight, since no committed fixture needed anything in it.
#[test]
fn bicell_parity_crossing_is_exercised() {
    // gain 1 = bias-only. Ascending, so the least invasive perturbation wins.
    const GAINS: [f64; 3] = [1.0, 2.0, 3.0];
    // 0.0 first, then +-0.25, +-0.5, ... +-4.0.
    let offsets = || {
        std::iter::once(0.0).chain(
            (1i32..=16)
                .flat_map(|k| [f64::from(k) * 0.25, -f64::from(k) * 0.25])
                .collect::<Vec<_>>(),
        )
    };
    // WIDE FALLBACK (phase-11 Task 6, the `phase9_fast_parity.rs::wide_offsets`
    // precedent): integer offsets out to +-32, smallest magnitude first -- headroom over
    // the measured need (`transformer` settles at `+-10`/`+-11`, see the function doc
    // above for why). Reached ONLY when nothing in `GAINS x offsets()` above crosses,
    // which on the committed fixtures is the `transformer` row alone. Because the primary
    // loop below runs to completion UNCHANGED before this is ever consulted, the three
    // sibling cells' rungs are BYTE-UNCHANGED BY CONSTRUCTION -- not by re-measurement,
    // since a loop that already broke on an earlier iteration never reaches code appended
    // after it.
    let wide_offsets = || (5i32..=32).flat_map(|k| [f64::from(k), -f64::from(k)]);

    let tmp = tempfile::tempdir().unwrap();
    for cell in CELLS {
        for window in [None, Some(OVERLAP_WINDOW)] {
            let tag = if window.is_some() { "overlap" } else { "plain" };
            // Pack layout `[fwd stack | rev stack | output MLP | mean | std]`, and the
            // fixture's output MLP is a single `8 -> 1` layer: 8 weights (col-major
            // `I x O`, `O == 1`) then 1 bias. Both indices are DERIVED from the config's
            // own `BLSTM_LSTMNeuronNb[0]` / `BLSTM_OutputNeuronNb[0]`, not hardcoded, so
            // a fixture regeneration at a different width relocates them instead of
            // silently perturbing some interior weight.
            let m = bicell_map(cell, None, None, None);
            let field =
                |key: &str| -> usize { m[key].split(',').next().unwrap().trim().parse().unwrap() };
            let input_size = field("BLSTM_LSTMNeuronNb");
            let dense_in = field("BLSTM_OutputNeuronNb");
            let seed = fixture(&format!("phase9/{cell}_bidirectional_seed.bin"));
            let base = read_weight_vector(&seed).unwrap();
            let bias_idx = base.len() - 2 * input_size - 1;
            let path = tmp.path().join(format!("{cell}_{tag}_crossing.bin"));

            let perturbed = |gain: f64, offset: f64| -> Vec<f64> {
                let mut v = base.clone();
                for w in v[bias_idx - dense_in..bias_idx].iter_mut() {
                    *w *= gain;
                }
                v[bias_idx] += offset;
                v
            };

            let mut chosen: Option<(f64, f64, usize)> = None;
            'sweep: for gain in GAINS {
                for offset in offsets() {
                    let probe = perturbed(gain, offset);
                    write_matrix(&path, probe.len(), 1, &probe).unwrap();
                    let interior =
                        interior_boundaries(&run_path(cell, None, Some(&path), window).1);
                    if interior > 0 {
                        println!(
                            "MEASURE crossing[{cell}/{tag}]: dense gain {gain} + bias offset \
                             {offset:+} -> {interior} interior boundaries"
                        );
                        chosen = Some((gain, offset, interior));
                        break 'sweep;
                    }
                }
            }
            if chosen.is_none() {
                'wide: for gain in GAINS {
                    for offset in wide_offsets() {
                        let probe = perturbed(gain, offset);
                        write_matrix(&path, probe.len(), 1, &probe).unwrap();
                        let interior =
                            interior_boundaries(&run_path(cell, None, Some(&path), window).1);
                        if interior > 0 {
                            println!(
                                "MEASURE crossing[{cell}/{tag}] (wide): dense gain {gain} + bias \
                                 offset {offset:+} -> {interior} interior boundaries"
                            );
                            chosen = Some((gain, offset, interior));
                            break 'wide;
                        }
                    }
                }
            }
            let (gain, offset, want_interior) = chosen.unwrap_or_else(|| {
                panic!("{cell}/{tag}: no (gain, offset) produced an interior boundary")
            });

            let flat = perturbed(gain, offset);
            write_matrix(&path, flat.len(), 1, &flat).unwrap();

            let e = run_path(cell, None, Some(&path), window);
            let f = run_path(cell, Some("fast"), Some(&path), window);
            let (max_abs, max_rel, max_dt) = compare(&format!("{cell}/{tag}/crossing"), &e, &f);
            let interior = interior_boundaries(&e.1);
            println!(
                "MEASURE {cell} bicell {tag} CROSSING (dense gain {gain}, bias {offset:+}): \
                 max_abs={max_abs:e} max_rel={max_rel:e} max_dt={max_dt} interior={interior}"
            );
            assert_eq!(
                interior, want_interior,
                "{cell}/{tag}: the sweep is not reproducible"
            );
            assert!(
                interior > 0,
                "{cell}/{tag}: the decision-layer comparison is vacuous"
            );
            assert!(
                max_abs < POST_ABS_PIN && max_rel < POST_REL_PIN,
                "{cell}/{tag}: crossing posterior drift max_abs={max_abs:e} max_rel={max_rel:e}"
            );
            assert_eq!(
                max_dt, 0.0,
                "{cell}/{tag}: crossing boundary max_dt {max_dt} != 0 (R1)"
            );
        }
    }
}

/// THE DISPATCH FLIP (spec S5): `(slstm|mamba|cfc, bidirectional)` was a typed bail
/// through phase 9; it now BUILDS a `FastBiCell` behind the fast SAD driver.
///
/// The `(lstm, forward)` tail of this leg USED TO assert the last remaining shape bail,
/// naming Task 8 as its flipper. Task 8 flipped it (`fast::cells::FastLstm`), so the
/// assertion is inverted rather than deleted: that combination must now BUILD, from the
/// matching committed pack, which is the statement that `classify_fast_shape` is total.
#[test]
fn fast_dispatch_builds_bidirectional_cells() {
    for cell in CELLS {
        let mut m = bicell_map(cell, Some("fast"), None, None);
        let bag = BagOfProcessors::from_configs(std::slice::from_mut(&mut m), image_mode())
            .unwrap_or_else(|e| panic!("fast + bidirectional {cell} must build: {e}"));
        assert!(
            matches!(bag.processor(0), Processor::FastSpectral(_)),
            "{cell}: expected the fast SAD driver"
        );
    }

    // NO LONGER BAILED (phase-10 Task 8): the LSTM cell in the CAUSAL direction. The
    // bidirectional fixture re-declared forward is EXACTLY the `lstm_forward` fixture's
    // geometry (`23,4` recurrent / `4,1` output), so its committed pack is the right one.
    let lstm_pack = fixture("phase9/lstm_forward_seed.bin");
    let mut m = bicell_map("slstm", Some("fast"), Some(&lstm_pack), None);
    m.insert("BLSTM_Cell_Type".into(), "lstm".into());
    m.insert("BLSTM_Direction".into(), "forward".into());
    m.insert("BLSTM_OutputNeuronNb".into(), "4,1".into());
    let bag = BagOfProcessors::from_configs(std::slice::from_mut(&mut m), image_mode())
        .unwrap_or_else(|e| panic!("fast + causal LSTM must build now: {e}"));
    assert!(
        matches!(bag.processor(0), Processor::FastSpectral(_)),
        "causal LSTM: expected the fast SAD driver"
    );
}

/// A SHORT pack is still refused. `FastBiCell` consumes TWO recurrent stacks, so a pack
/// sized for the forward-only net is short by exactly one stack -- the check that stops
/// another architecture's (or another direction's) pack being consumed head-first.
#[test]
fn fast_bicell_refuses_a_forward_sized_pack() {
    for cell in CELLS {
        let fwd_pack = fixture(&format!("phase9/{cell}_forward_seed.bin"));
        let mut m = bicell_map(cell, Some("fast"), Some(&fwd_pack), None);
        let err = BagOfProcessors::from_configs(std::slice::from_mut(&mut m), image_mode())
            .err()
            .unwrap_or_else(|| {
                panic!("{cell}: a forward-sized pack must not build a bidirectional fast net")
            });
        assert!(
            err.to_string().contains("less than what's needed"),
            "{cell}: expected a pack-length bail, got: {err}"
        );
    }
}

/// The TRUNCATE (non-overlap) windowing stays unsupported on the fast path, for the
/// bidirectional cells exactly as for the phase-7 `FastBlstm`: `FastBiCell` implements
/// the plain and OVERLAP regimes only. `BLSTM_shift 0.0` resolves `window_shift < 1`
/// -> `no_overlap` at EVERY rate, so since issue #24 the plan's rate-free pre-check
/// refuses it at bag construction (`FastSadPlan::from_map`), no longer at the first
/// `get_segmentation`; the refusal itself is unchanged.
#[test]
fn fast_bicell_bails_on_truncate_windowing() {
    let mut m = bicell_map("slstm", Some("fast"), None, Some(OVERLAP_WINDOW));
    m.insert("BLSTM_shift".into(), "0.0".into());
    let err = BagOfProcessors::from_configs(std::slice::from_mut(&mut m), image_mode())
        .err()
        .unwrap_or_else(|| panic!("fast bicell + truncate windowing must bail"));
    assert!(
        err.to_string().contains("truncate"),
        "expected a truncate bail, got: {err}"
    );
}
