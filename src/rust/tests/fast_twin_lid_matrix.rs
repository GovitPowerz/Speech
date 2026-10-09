//! Issue #57: the fast Twin's LID net on the (cell x direction) matrix -- the f32
//! `fast::driver::FastTwinLid` (now dispatching its LID net through `FastLidNet`) vs the
//! exact f64 `TwinBlstmSpectralLid`, over the committed phase-9 Mode-7 LID fixtures
//! (`tests/reference_data/phase9/twin_mode7_lid_*`: a tiny `12,6` LID net per cell in both
//! directions, the SAD net frozen on the legacy LSTM) and the committed phSeq corpus
//! (`phase4b/corpus_phseq/s{1,2,3}.phSeq`). The `(lstm, bidirectional)` pair is the phase-7
//! real-net leg (`phase7_parity_lid.rs`) and is not repeated here.
//!
//! THREE LEGS PER (cell, direction) (the grilling's D11):
//!  - PARITY: argmax IDENTITY on the per-utterance decision (the confusion matrix, plus the
//!    per-utterance argmax straight from an exact single-entry run vs the fast session's
//!    `scores`), on the cumulative `is_lid_correct` and on the predicted language; the
//!    `classification_errors` delta MEASURED then pinned at x10 (`ERR_PIN`). A flip is an R1
//!    STOP, never a widening (ADR-0003: the LID tier owns argmax zero-flips).
//!  - DISPATCH: the pair builds from its OWN committed pack into its OWN `FastNetShape` arm,
//!    immediately and through the deferred `load_weights_file` (scored bit-identical), and
//!    refuses that pack one element short or one element long on the in-memory seam (the
//!    exact `set_weights` contract) -- the two length checks are what stop another shape's
//!    pack from being consumed head-first under this one's name.
//!  - the two surviving REFUSALS (D2/D10), pinned where the window resolves: a windowed
//!    CAUSAL LID net and the OVERLAP regime on any shape, both offline and streaming.
//!
//! THE DEGENERATE-ARGMAX LADDER (D5, the phase-10 crossing lesson in LID clothes). A seed net
//! can sit so close to `p == 0.5` that an argmax agreement says nothing. Interior means: the
//! EXACT tree's margin between its top two accumulated langID entries -- per utterance (the
//! `seg_lid` gap, recovered from an exact single-entry run as `rows * (ln L[a] - ln L[b])`,
//! which is the same expression under both `cost_modified` branches) and cumulative (the
//! normalized langID gap) -- exceeds TEN times the measured f32-vs-f64 delta of that same
//! quantity. The ladder scales the output MLP's LAST layer (weights + bias, located from
//! the config's own widths, not hardcoded) by 1, 2, 4, 8 from the committed seed, in
//! memory, and takes the first rung on which EVERY margin is interior. Gain 1 at offset 0
//! means the fixture is used AS COMMITTED.
//!
//! THE SECOND LEVER, MEASURED NECESSARY (the phase-10 bias knob): on these fixtures the LID
//! net is `12` wide over a `38`-wide phSeq one-hot, so every block whose letters all map
//! past column 12 is ALL-ZERO after the width-tolerance crop -- the first block of every
//! committed file and all of `s2`. A zero input through a zero-state cell is a zero hidden
//! row, the dense output is its (xavier-zero) bias, and the posterior is EXACTLY `0.5` on
//! both trees at every gain: a tie that no scaling breaks. So the ladder's second stage adds
//! a bias OFFSET (`+0.5`) to the same last layer, which gives those blocks a real logit
//! (`rows * 0.5` of margin, bias-only arithmetic, f32 delta ~1e-8) and makes the
//! per-utterance decision non-vacuous there. Gain is tried at offset 0 first, so a fixture
//! that is interior as committed stays so.

use std::path::PathBuf;

use indexmap::IndexMap;
use ndarray::Array2;
use speech::audio::{Audio, read_audio};
use speech::fast::driver::{FastNetShape, FastTwinLid};
use speech::fast::stream_lid::StreamingLidSession;
use speech::io::binary::read_weight_vector;
use speech::nn::blstm::CellType;
use speech::tasks::lid::TwinBlstmSpectralLid;
use speech::tasks::segmentation::Segmentation;
use speech::tasks::segmenter::Segmenter;

fn ref_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/reference_data")
}

/// The eight non-phase-7 pairs: `(fixture, cell, direction)`.
const MATRIX: [(&str, CellType, bool); 8] = [
    ("twin_mode7_lid_slstm", CellType::Slstm, false),
    ("twin_mode7_lid_mamba", CellType::Mamba, false),
    ("twin_mode7_lid_cfc", CellType::Cfc, false),
    ("twin_mode7_lid_transformer", CellType::Transformer, false),
    ("twin_mode7_lid_lstm_forward", CellType::Lstm, true),
    ("twin_mode7_lid_slstm_forward", CellType::Slstm, true),
    ("twin_mode7_lid_mamba_forward", CellType::Mamba, true),
    ("twin_mode7_lid_cfc_forward", CellType::Cfc, true),
];
/// The ninth pair, split out: see [`transformer_forward_is_the_ninth_pair`].
const TRANSFORMER_FORWARD: (&str, CellType, bool) = (
    "twin_mode7_lid_transformer_forward",
    CellType::Transformer,
    true,
);

/// The committed phSeq files + the language each is scored against (the phase-7 trio: a
/// hit, a hit and a miss on the real net; on these seeds whatever the ladder makes them).
const PHSEQ_FILES: [(&str, i32); 3] = [("s1", 0), ("s2", 1), ("s3", 1)];

/// The MEASURED-then-pinned `classification_errors` delta per pair (max abs over the
/// in-band `100*(langID - targetLID)` columns, over the three files, at the ladder's
/// rung), x10, rounded up. A breach is RE-MEASURE under the STOP rule, never a widening.
fn err_pin(fixture: &str) -> f64 {
    // MEASURED 2026-10-09 (M4 Pro), gain 1 / offset +0.5 on every pair: 2.081e-6 on eight
    // pairs (the dominant term is the bias-only blocks' f32 logistic, shared by all) and
    // 2.273e-6 on transformer/forward; min(margin/delta) 3.95e6 .. 5.65e6.
    match fixture {
        "twin_mode7_lid_transformer_forward" => 2.3e-5,
        "twin_mode7_lid_slstm"
        | "twin_mode7_lid_mamba"
        | "twin_mode7_lid_cfc"
        | "twin_mode7_lid_transformer"
        | "twin_mode7_lid_lstm_forward"
        | "twin_mode7_lid_slstm_forward"
        | "twin_mode7_lid_mamba_forward"
        | "twin_mode7_lid_cfc_forward" => 2.1e-5,
        other => panic!("no pin for {other}"),
    }
}

fn fixture_map(name: &str) -> IndexMap<String, String> {
    let text = std::fs::read_to_string(ref_dir().join(format!("phase9/{name}.config"))).unwrap();
    speech::legacy_config::parse_legacy_config(&text)
}

fn fixture_pack(name: &str) -> Vec<f64> {
    read_weight_vector(&ref_dir().join(format!("phase9/{name}_seed.bin"))).unwrap()
}

fn sad_pack() -> Vec<f64> {
    read_weight_vector(&ref_dir().join("phase4b/tiny_sad_seed.bin")).unwrap()
}

fn phseq_audio(f: &str, lang: i32) -> Audio {
    let path = ref_dir()
        .join("phase4b/corpus_phseq")
        .join(format!("{f}.phSeq"));
    let mut a = read_audio(&path, 0.0, 120.0, 1, None).unwrap();
    a.lang_index = lang;
    a.weight = 1.0;
    a.audio_file_name = path.to_string_lossy().into_owned();
    a
}

fn segs_for(audio: &Audio) -> Vec<Segmentation> {
    let dur = (audio.data.ncols() as f64 - 1.0) / audio.sample_rate as f64;
    (0..audio.data.nrows())
        .map(|_| Segmentation::new(dur))
        .collect()
}

/// Channel 0's LID members: `(classification_errors, is_lid_correct, confusion)`.
struct Members {
    errors: Vec<f64>,
    correct: i32,
    confusion: Array2<f64>,
}

fn run_exact(map: &IndexMap<String, String>, lid: &[f64], mut audio: Audio) -> Members {
    let sad = sad_pack();
    let mut drv = TwinBlstmSpectralLid::from_legacy(map, Some(&sad), Some(lid)).unwrap();
    let mut segs = segs_for(&audio);
    drv.get_segmentation(&mut audio, &mut segs, None).unwrap();
    Members {
        errors: drv.lid_classification_errors()[0].clone(),
        correct: drv.is_lid_correct()[0],
        confusion: drv.lid_segments_confusion()[0].clone(),
    }
}

fn run_fast(map: &IndexMap<String, String>, lid: &[f64], mut audio: Audio) -> Members {
    let mut drv = FastTwinLid::from_legacy(map, None, Some(lid)).unwrap();
    let mut segs = segs_for(&audio);
    drv.get_segmentation(&mut audio, &mut segs, None).unwrap();
    Members {
        errors: drv.lid_classification_errors()[0].clone(),
        correct: drv.is_lid_correct()[0],
        confusion: drv.lid_segments_confusion()[0].clone(),
    }
}

/// Recover the normalized langID from the in-band errors row: `err[c] = 100*(langid[c] -
/// target[c])`, `target[ti] = -2`, else 0.
fn langid_of(errors: &[f64], ti: usize) -> Vec<f64> {
    errors
        .iter()
        .enumerate()
        .map(|(c, &e)| e / 100.0 + if c == ti { -2.0 } else { 0.0 })
        .collect()
}

fn argmax(v: &[f64]) -> usize {
    let mut j = 0;
    for (c, &x) in v.iter().enumerate().skip(1) {
        if x > v[j] {
            j = c;
        }
    }
    j
}

/// `top1 - top2` of `v` (the decision margin).
fn margin(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| b.partial_cmp(a).unwrap());
    s[0] - s[1]
}

fn target_of(lang: i32, class_nb: usize) -> usize {
    if lang >= 0 && (lang as usize) < class_nb {
        lang as usize
    } else {
        0
    }
}

/// The committed pack with the output MLP's LAST layer (weights + bias) scaled by `gain`,
/// then its bias shifted by `offset`. Pack tail layout on every shape: `[... | last dense
/// layer (I*O + O) | mean | std]` with the mean/std tail `2 * LSTMNeuronNb[0]` long -- the
/// layer-0 input width, which is what BOTH trees size the tail by (`BlstmNetwork::
/// input_size` = the forward stack's layer-0 width; the fast nets take `lstm[0]`); the
/// declared `NNetInputSize` is never consulted. `I`/`O` are the config's own
/// `OutputNeuronNb` last two entries, so a regenerated fixture relocates the block.
fn perturbed_pack(
    map: &IndexMap<String, String>,
    base: &[f64],
    gain: f64,
    offset: f64,
) -> Vec<f64> {
    let ints = |key: &str| -> Vec<usize> {
        map[key]
            .split(',')
            .map(|v| v.trim().parse().unwrap())
            .collect()
    };
    let input = ints("BLSTM_LID_LSTMNeuronNb")[0];
    let outn = ints("BLSTM_LID_OutputNeuronNb");
    let (i, o) = (outn[outn.len() - 2], outn[outn.len() - 1]);
    let tail = 2 * input;
    let layer = i * o + o;
    let lo = base.len() - tail - layer;
    let mut v = base.to_vec();
    for w in v[lo..lo + layer].iter_mut() {
        *w *= gain;
    }
    for b in v[lo + i * o..lo + layer].iter_mut() {
        *b += offset;
    }
    v
}

/// One pair at one ladder rung: every per-utterance + cumulative margin, their f32 deltas,
/// and the identity assertions' inputs. `interior` is the D5 criterion over all of them.
struct RungReport {
    interior: bool,
    min_ratio: f64,
    max_err_abs: f64,
}

fn probe_rung(
    fixture: &str,
    map: &IndexMap<String, String>,
    lid: &[f64],
    assert_identity: bool,
) -> RungReport {
    let mut min_ratio = f64::INFINITY;
    let mut max_err_abs = 0.0f64;
    let mut interior = true;
    for (f, lang) in PHSEQ_FILES {
        let tag = format!("{fixture}/{f}");
        let audio = phseq_audio(f, lang);
        let feats = audio.external_features.clone();
        let e = run_exact(map, lid, phseq_audio(f, lang));
        let fa = run_fast(map, lid, phseq_audio(f, lang));
        let class_nb = e.errors.len();
        let ti = target_of(lang, class_nb);

        // Cumulative: the normalized-langID margin vs its f32 delta.
        let le = langid_of(&e.errors, ti);
        let lf = langid_of(&fa.errors, ti);
        let delta = le
            .iter()
            .zip(&lf)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max)
            .max(1e-12);
        let m = margin(&le);
        min_ratio = min_ratio.min(m / delta);
        interior &= m > 10.0 * delta;
        for (&a, &b) in e.errors.iter().zip(&fa.errors) {
            max_err_abs = max_err_abs.max((a - b).abs());
        }

        // Per utterance: the exact single-entry run vs the fast session's `scores`.
        let mut sess = StreamingLidSession::new(map, 8000.0, lang, Some(lid)).unwrap();
        for (k, feat) in feats.iter().enumerate() {
            let sc = sess.push_utterance(feat);
            let mut single = phseq_audio(f, lang);
            single.external_features = vec![feat.clone()];
            let ex = run_exact(map, lid, single);
            let Some(j_fast) = sc.argmax else {
                continue; // skipped on both sides (the `:1450` guard is shared)
            };
            let l = langid_of(&ex.errors, ti);
            let rows = feat.nrows() as f64;
            // seg_lid[c] = rows * ln L[c] + const, so the gap is rows * (ln La - ln Lb).
            let ln_l: Vec<f64> = l.iter().map(|&x| rows * x.ln()).collect();
            let gap_exact = margin(&ln_l);
            let gap_fast = margin(&sc.scores);
            let d = (gap_exact - gap_fast).abs().max(1e-12);
            min_ratio = min_ratio.min(gap_exact / d);
            interior &= gap_exact > 10.0 * d;
            if assert_identity {
                assert_eq!(
                    j_fast,
                    argmax(&l),
                    "{tag} u{k}: per-utterance argmax FLIP (R1 STOP)"
                );
            }
        }

        if assert_identity {
            assert_eq!(
                e.correct, fa.correct,
                "{tag}: is_lid_correct FLIP (R1 STOP)"
            );
            assert_eq!(e.confusion, fa.confusion, "{tag}: confusion FLIP (R1 STOP)");
            assert_eq!(
                argmax(&le),
                argmax(&lf),
                "{tag}: predicted-language FLIP (R1 STOP)"
            );
        }
    }
    RungReport {
        interior,
        min_ratio,
        max_err_abs,
    }
}

/// The ladder + the parity assertions for one pair, optionally under a `BLSTM_LID_window`
/// override.
fn parity_pair(fixture: &str, window: Option<&str>) {
    let mut map = fixture_map(fixture);
    if let Some(w) = window {
        map.insert("BLSTM_LID_window".into(), w.into());
    }
    let base = fixture_pack(fixture);
    let mut chosen: Option<((f64, f64), Vec<f64>)> = None;
    let mut last: Vec<f64> = Vec::new();
    'ladder: for offset in [0.0, 0.5] {
        for gain in [1.0, 2.0, 4.0, 8.0] {
            let lid = perturbed_pack(&map, &base, gain, offset);
            last.clone_from(&lid);
            let rep = probe_rung(fixture, &map, &lid, false);
            println!(
                "MEASURE ladder[{fixture}] gain {gain} offset {offset:+}: interior={} \
                 min(margin/delta)={:.3e}",
                rep.interior, rep.min_ratio
            );
            if rep.interior {
                chosen = Some(((gain, offset), lid));
                break 'ladder;
            }
        }
    }
    let ((gain, offset), lid) = chosen.unwrap_or_else(|| {
        // No rung interior: the f32 delta swamped every margin. Assert identity at the last
        // rung FIRST, so a broken fast kernel reports as the argmax FLIP it is (R1) rather
        // than as a ladder exhaustion (the mutation battery's item 1 is the witness).
        probe_rung(fixture, &map, &last, true);
        panic!("{fixture}: no ladder rung is interior")
    });
    let rep = probe_rung(fixture, &map, &lid, true);
    println!(
        "MEASURE parity[{fixture}] gain {gain} offset {offset:+}: max_err_abs={:.3e} \
         min(margin/delta)={:.3e} pin={:.1e}",
        rep.max_err_abs,
        rep.min_ratio,
        err_pin(fixture)
    );
    assert!(
        rep.interior,
        "{fixture}: the chosen rung is not reproducible"
    );
    assert!(
        rep.max_err_abs < err_pin(fixture),
        "{fixture}: classification_errors delta {:.3e} breached the pin {:.1e} (STOP: re-measure)",
        rep.max_err_abs,
        err_pin(fixture)
    );
}

#[test]
fn lid_parity_exact_vs_fast_per_pair() {
    for (fixture, _, _) in MATRIX {
        parity_pair(fixture, None);
    }
}

/// D2 admits the PLAIN regime on the bidirectional shapes too (bailed before #57), but every
/// committed bidirectional fixture resolves to truncate (`window 0.25`): this is the only leg
/// that runs `FastLidNet::feed_forward_scoring`'s plain branch through a `BiCell` (the `Blstm`
/// arm's is `phase7_parity_lid::lid_parity_phseq_plain_regime_exact_vs_fast`). MEASURED
/// 2026-10-09 (M4 Pro) at the ladder's rung (gain 1 / offset +0.5; offset 0 leaves 5 of 7
/// utterances an exact 0.5 tie): max_err_abs 2.081e-6 on all four, min(margin/delta)
/// 4.23e6 .. 5.65e6, so `err_pin` holds as is.
#[test]
fn bidirectional_plain_regime_parity_per_cell() {
    for (fixture, _, forward) in MATRIX {
        if !forward {
            parity_pair(fixture, Some("0"));
        }
    }
}

/// Split out of the loop so a transformer-specific failure (the one cell with a per-step
/// windowed-attention twin whose KV ring the sweep never exercised on a LID block before)
/// names itself rather than aborting the eight-way loop.
#[test]
fn transformer_forward_is_the_ninth_pair() {
    parity_pair(TRANSFORMER_FORWARD.0, None);
}

// ---------------------------------------------------------------------------
// Dispatch: each pair builds its own arm from its own pack; a wrong-length pack is refused.
// ---------------------------------------------------------------------------

#[test]
fn each_pair_builds_its_own_arm_and_refuses_a_short_pack() {
    let all = MATRIX.iter().chain(std::iter::once(&TRANSFORMER_FORWARD));
    for &(fixture, cell, forward) in all {
        let map = fixture_map(fixture);
        let pack = fixture_pack(fixture);
        let want = match (cell, forward) {
            (_, true) => FastNetShape::Causal(cell),
            (CellType::Lstm, false) => FastNetShape::Blstm,
            (_, false) => FastNetShape::BiCell(cell),
        };
        let mut drv = FastTwinLid::from_legacy(&map, None, Some(&pack))
            .unwrap_or_else(|e| panic!("{fixture}: must build from its own pack: {e}"));
        assert_eq!(drv.lid_shape(), want, "{fixture}: dispatched arm");
        // The deferred load (the bag's path) builds the same net. `lid_shape()` is the
        // ctor's field, blind to what `load_weights_file` built: score s1 (non-degenerate
        // under every committed pack) through both and demand bit-identity.
        let mut deferred = FastTwinLid::from_legacy(&map, None, None).unwrap();
        let mut m = map.clone();
        m.insert(
            "BLSTM_LID_weightsFile".into(),
            ref_dir()
                .join(format!("phase9/{fixture}_seed.bin"))
                .to_string_lossy()
                .into_owned(),
        );
        deferred.load_weights_file(&m).unwrap();
        let score = |d: &mut FastTwinLid| {
            let mut audio = phseq_audio("s1", 0);
            let mut segs = segs_for(&audio);
            d.get_segmentation(&mut audio, &mut segs, None).unwrap();
            (
                d.lid_classification_errors()[0].clone(),
                d.lid_segments_confusion()[0].clone(),
            )
        };
        assert_eq!(
            score(&mut deferred),
            score(&mut drv),
            "{fixture}: the deferred load built another net"
        );
        // One element long: refused on length too, as the exact Twin's `set_weights` does.
        let mut long = pack.clone();
        long.push(0.0);
        match FastTwinLid::from_legacy(&map, None, Some(&long)) {
            Err(e) => assert!(
                e.to_string().contains("more than what's needed"),
                "{fixture}: a long pack must fail on LENGTH, got: {e}"
            ),
            Ok(_) => panic!("{fixture}: a pack one element long must not build"),
        }
        // One element short: refused on length, whatever the shape.
        let short = &pack[..pack.len() - 1];
        match FastTwinLid::from_legacy(&map, None, Some(short)) {
            Err(e) => assert!(
                e.to_string().contains("too short"),
                "{fixture}: a short pack must fail on LENGTH, got: {e}"
            ),
            Ok(_) => panic!("{fixture}: a pack one element short must not build"),
        }
    }
}

// ---------------------------------------------------------------------------
// The two surviving refusals (D2/D10): windowed causal, overlap. Offline AND streaming.
// ---------------------------------------------------------------------------

fn offline_err(map: &IndexMap<String, String>, lid: &[f64]) -> String {
    let mut drv = FastTwinLid::from_legacy(map, None, Some(lid))
        .expect("a regime refusal is raised where the window resolves, not at construction");
    let mut audio = phseq_audio("s1", 0);
    let mut segs = segs_for(&audio);
    drv.get_segmentation(&mut audio, &mut segs, None)
        .expect_err("the regime must be refused")
        .to_string()
}

fn streaming_err(map: &IndexMap<String, String>, lid: &[f64]) -> String {
    StreamingLidSession::new(map, 8000.0, 0, Some(lid))
        .err()
        .expect("the streaming session must refuse the regime at construction")
        .to_string()
}

#[test]
fn windowed_causal_lid_is_refused_offline_and_streaming() {
    for fixture in [
        "twin_mode7_lid_lstm_forward",
        "twin_mode7_lid_slstm_forward",
        "twin_mode7_lid_transformer_forward",
    ] {
        let mut m = fixture_map(fixture);
        // The committed bidirectional value: resolves 25 periodogram frames of truncate.
        m.insert("BLSTM_LID_window".into(), "0.25".into());
        let lid = fixture_pack(fixture);
        for msg in [offline_err(&m, &lid), streaming_err(&m, &lid)] {
            assert!(
                msg.contains("windowed LID inference is unsupported for a causal LID net"),
                "{fixture}: unexpected refusal: {msg}"
            );
        }
        // And `window 0` on the same fixture RUNS (the refusal is the regime, not the shape).
        let m0 = fixture_map(fixture);
        let mut drv = FastTwinLid::from_legacy(&m0, None, Some(&lid)).unwrap();
        let mut audio = phseq_audio("s1", 0);
        let mut segs = segs_for(&audio);
        drv.get_segmentation(&mut audio, &mut segs, None).unwrap();
    }
}

#[test]
fn overlap_lid_is_refused_on_every_shape() {
    for fixture in [
        "twin_mode7_lid_slstm",
        "twin_mode7_lid_mamba",
        "twin_mode7_lid_cfc_forward",
    ] {
        let mut m = fixture_map(fixture);
        // `window 0.25 / shift 0.1`: shift resolves 10 frames (>= 1) -> overlap. On the
        // forward fixture the window is set too, so EITHER rule may fire first (the
        // windowed-causal one does); which one is not asserted -- the row pins that a
        // causal shape refuses the pair as well, not the order of the two checks.
        m.insert("BLSTM_LID_window".into(), "0.25".into());
        m.insert("BLSTM_LID_shift".into(), "0.1".into());
        let lid = fixture_pack(fixture);
        for msg in [offline_err(&m, &lid), streaming_err(&m, &lid)] {
            assert!(
                msg.contains("LID windowing is unsupported")
                    || msg.contains("windowed LID inference is unsupported"),
                "{fixture}: unexpected refusal: {msg}"
            );
        }
    }
}

#[test]
fn plain_regime_ignores_two_sweeps_like_the_exact_tree() {
    // `TwoSweeps true` (the inherited twin_train value) vs `false` must score identically on
    // BOTH trees: the plain regime never reads the flag. s1/s3, not s2: every s2 block is
    // all-zero after the 12-wide crop, so its posterior is exactly 0.5 under ANY regime.
    let fixture = "twin_mode7_lid_slstm_forward";
    let lid = fixture_pack(fixture);
    let on = fixture_map(fixture);
    let mut off = fixture_map(fixture);
    off.insert("BLSTM_LID_TwoSweeps".into(), "false".into());
    for (f, lang) in [("s1", 0), ("s3", 1)] {
        let fa = run_fast(&on, &lid, phseq_audio(f, lang));
        let fb = run_fast(&off, &lid, phseq_audio(f, lang));
        let ti = target_of(lang, fa.errors.len());
        assert!(
            margin(&langid_of(&fa.errors, ti)) > 0.0,
            "{f}: an all-tie file makes this comparison vacuous"
        );
        assert_eq!(
            fa.errors, fb.errors,
            "{f}: the fast plain regime read TwoSweeps"
        );
        assert_eq!(
            fa.confusion, fb.confusion,
            "{f}: the fast plain regime read TwoSweeps"
        );
        let ea = run_exact(&on, &lid, phseq_audio(f, lang));
        let eb = run_exact(&off, &lid, phseq_audio(f, lang));
        assert_eq!(
            ea.errors, eb.errors,
            "{f}: the exact plain regime read TwoSweeps"
        );
    }
}
