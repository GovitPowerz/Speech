//! Phase 7 Task 4: the CI parity leg (spec S1.6a) -- the f32 fast algo-3 SAD driver
//! vs the exact f64 `BlstmSpectralSegmenter`, BOTH run through `BagOfProcessors` on
//! the committed `tier2_spectral.config` + tuple-A pack (`NNweights_config1.bin`) +
//! `prcts_excerpt.wav` (2 ch, 8 kHz, 60 s). The fast path DIVERGES from exact BY
//! DESIGN (f32 + realfft + faer through the whole feature/BLSTM/overlap chain, spec
//! S4); tolerances are MEASURED first (the `MEASURE` prints, `cargo test -- --nocapture`),
//! then pinned with headroom. Segment COUNT + TYPES identical per channel is the HARD
//! gate: a flip is R1 STOP-and-adjudicate, not a silent widen.
//!
//! tier2 dispatches to the OVERLAP windowed driver (`BLSTM_window 3.25 / BLSTM_shift
//! 0.8` -> `window_size > 0`, `no_overlap false`) with `InputNormalizationType -1`, so
//! this exercises the fast overlap forward + type -1 self-normalization + the
//! cross-channel result-buffer seeding end to end.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use indexmap::IndexMap;
use speech::audio::read_audio;
use speech::cli::{Mode, ModeKind};
use speech::config::NnetSpec;
use speech::engine::bag_of_processors::{BagOfProcessors, Processor};
use speech::engine::corpus::CorpusItem;
use speech::fast::driver::build_aligned_spec;
use speech::tasks::segmentation::Segmentation;

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

/// tier2 as a config map, pointed at the committed tuple-A pack, offset 0 / duration
/// covering the whole 60 s file, and (optionally) `Inference_Path`. The base config's
/// Task-9 override tail is overwritten last-wins.
fn tier2_map(inference: Option<&str>) -> IndexMap<String, String> {
    let text = std::fs::read_to_string(fixture("phase4a/tier2_spectral.config")).unwrap();
    let mut m = speech::legacy_config::parse_legacy_config(&text);
    m.insert("numOuterThreads".into(), "1".into());
    m.insert("Audio_offset".into(), "0.0".into());
    m.insert("Audio_max_duration".into(), "3600".into());
    m.insert(
        "BLSTM_weightsFile".into(),
        fixture("phase0/NNweights_config1.bin")
            .to_str()
            .unwrap()
            .into(),
    );
    if let Some(p) = inference {
        m.insert("Inference_Path".into(), p.into());
    }
    m
}

/// Run one path (exact when `inference == None`, fast when `Some("fast")`) through the
/// bag via `run_get_segmentation` on the full 60 s prcts excerpt. Returns the
/// per-channel PRE-`results_to_segmentation` posterior rows + the per-channel final
/// segmentations.
fn run_path(inference: Option<&str>) -> (Vec<Vec<f64>>, Vec<Segmentation>) {
    let mut m = tier2_map(inference);
    let mut bag =
        BagOfProcessors::from_configs(std::slice::from_mut(&mut m), image_mode()).unwrap();

    let mut audio =
        read_audio(&fixture("phase4d/prcts_excerpt.wav"), 0.0, 3600.0, 0, None).unwrap();
    let dur = (audio.data.ncols() as f64 - 1.0) / audio.sample_rate as f64;
    let n_chan = audio.data.nrows();
    let mut seg: Vec<Segmentation> = (0..n_chan).map(|_| Segmentation::new(dur)).collect();
    bag.run_get_segmentation(0, &mut audio, &mut seg).unwrap();

    let posteriors = match bag.processor(0) {
        Processor::Spectral(s) => s.last_result_rows().to_vec(),
        Processor::FastSpectral(s) => s.last_result_rows().to_vec(),
        _ => panic!("unexpected processor variant for inference={inference:?}"),
    };
    (posteriors, seg)
}

// ---------------------------------------------------------------------------
// The CI parity leg.
// ---------------------------------------------------------------------------

#[test]
fn sad_parity_exact_vs_fast() {
    // MEASURED (Apple Silicon dev box, full 60 s x 2 ch): posterior max_abs=2.416e-6,
    // max_rel=1.361e-5; boundary max_dt=0.0 s (segment count + types IDENTICAL, 0
    // flips -- the 4-decimal-VRCTS-grade posterior agreement lands the hysteresis
    // crossings on identical f64 boundaries). Posterior pins carry ~20x headroom over
    // measured, widened for cross-platform realfft/faer SIMD variance (CI is x86, this
    // box is ARM -- the T3 pipeline test found the 60 s realfft delta grows on x86).
    //
    // RE-MEASURED by the Phase 10 Task 5 S4 sweep (full-f32 mel at both fast sites):
    // max_abs UNMOVED at 2.416e-6, max_rel 1.398e-5 -> 1.361e-5, max_dt still EXACTLY
    // 0.0, zero flips -- so the PINS ARE UNCHANGED. The front-end deltas grew ~4x
    // (phase7_fast_pipeline re-pinned), but they do not reach here: the posterior
    // divergence on this net is dominated by the f32 NN's own recurrence/matmul error,
    // and a ~4e-5 perturbation of a log-mel feature of magnitude ~10 is ~4e-6 relative
    // -- below what the LSTM's f32 arithmetic already contributes.
    // The segment COUNT + TYPES identical assert below is the HARD gate (spec S1.6a);
    // BOUNDARY_PIN (10 ms) is the soft max-dt bound over measured 0.0.
    const POST_REL_PIN: f64 = 3.0e-4;
    const POST_ABS_PIN: f64 = 5.0e-5;
    const BOUNDARY_PIN: f64 = 1.0e-2;

    let (post_e, seg_e) = run_path(None);
    let (post_f, seg_f) = run_path(Some("fast"));

    assert_eq!(post_e.len(), post_f.len(), "channel count");
    assert_eq!(seg_e.len(), seg_f.len(), "channel count (segments)");

    // --- posteriors ---
    let mut max_abs = 0.0_f64;
    let mut max_rel = 0.0_f64;
    for chan in 0..post_e.len() {
        assert_eq!(
            post_e[chan].len(),
            post_f[chan].len(),
            "posterior row length chan {chan}"
        );
        for (&e, &f) in post_e[chan].iter().zip(post_f[chan].iter()) {
            // Uncovered overlap rows are 0/0 = NaN in BOTH paths; the NaN PATTERN must
            // match (same window coverage), then those cells are skipped in the delta.
            if e.is_nan() || f.is_nan() {
                assert_eq!(e.is_nan(), f.is_nan(), "NaN pattern mismatch chan {chan}");
                continue;
            }
            let d = (e - f).abs();
            max_abs = max_abs.max(d);
            // Posteriors are logistic outputs in [0,1]; scale the relative delta by
            // max(|e|, 1e-2) so a tiny posterior does not inflate a tiny abs delta.
            max_rel = max_rel.max(d / e.abs().max(1e-2));
        }
    }

    // --- boundaries: COUNT + TYPES identical (R1), max-dt pinned ---
    let mut max_dt = 0.0_f64;
    for chan in 0..seg_e.len() {
        let se = seg_e[chan].segments();
        let sf = seg_f[chan].segments();
        assert_eq!(
            se.len(),
            sf.len(),
            "segment COUNT differs chan {chan} (R1: a count change is a boundary flip -- STOP)"
        );
        for (a, b) in se.iter().zip(sf.iter()) {
            assert_eq!(
                a.ty, b.ty,
                "segment TYPE differs chan {chan} (R1: a type change is a flip -- STOP)"
            );
            max_dt = max_dt.max((a.begin - b.begin).abs());
        }
    }

    println!(
        "MEASURE sad_parity: post max_abs={max_abs:.3e} max_rel={max_rel:.3e} boundary max_dt={max_dt:.4e}s"
    );
    assert!(
        max_abs < POST_ABS_PIN,
        "posterior abs {max_abs} exceeds pin"
    );
    assert!(
        max_rel < POST_REL_PIN,
        "posterior rel {max_rel} exceeds pin"
    );
    assert!(
        max_dt <= BOUNDARY_PIN,
        "boundary max-dt {max_dt}s exceeds pin"
    );
}

/// The scored leg's processed duration (s). Shorter than the main posterior/boundary
/// leg's full 60 s -- the scored columns derive from the SAME shared f64 decision layer
/// on the boundaries the main leg already proves identical, so a shorter slice is a
/// sufficient `compute_errors` confirmation and keeps the debug suite quicker.
const SCORED_DUR: f64 = 20.0;

/// A 2-channel STM with one in-window SPEECH span per channel (SPEECH iff token 3
/// starts with token 1; >= 7 whitespace tokens required). Spans sit inside the scored
/// window so `compute_errors` sees real speech/non-speech on both channels.
fn write_stm(dir: &Path) -> PathBuf {
    let p = dir.join("ref.stm");
    let body = "\
prcts 1 prcts 3.0 11.0 <o,f0,unk> hello world
prcts 2 prcts 5.0 16.0 <o,f0,unk> hello world
";
    std::fs::write(&p, body).unwrap();
    p
}

/// Run one path scored (image mode + a crafted STM reference) through the bag's full
/// `segmentation_function`, returning the per-channel result rows.
fn scored_rows(inference: Option<&str>, dump: &Path, stm: &Path) -> BTreeMap<usize, Vec<f64>> {
    std::fs::create_dir_all(dump).unwrap();
    let mut m = tier2_map(inference);
    m.insert("Audio_max_duration".into(), SCORED_DUR.to_string());
    m.insert("Dump_Directory".into(), dump.to_str().unwrap().into());
    let mut bag =
        BagOfProcessors::from_configs(std::slice::from_mut(&mut m), image_mode()).unwrap();

    let item = CorpusItem {
        file_name: fixture("phase4d/prcts_excerpt.wav")
            .to_str()
            .unwrap()
            .to_string(),
        ref_seg: stm.to_str().unwrap().to_string(),
        language: "unk".into(),
        dialect: "unk".into(),
        class_index: 0,
        file_id: 1,
        weight: 1.0,
    };
    let results = bag.segmentation_function(&item, image_mode()).unwrap();
    results[&0].iter().map(|(c, r)| (*c, r.to_row())).collect()
}

#[test]
fn sad_parity_scored_columns() {
    // The `compute_errors` scored columns flow through the SHARED f64 decision layer,
    // so they differ from exact ONLY via the boundary shift the f32 posteriors induce.
    // Compared: cols 0 (Pfa), 1 (Pmiss), 2 (global error rate), 5 (audio_duration), 6
    // (speech_duration). Cols 4 (NN cost) and 17 (nb_of_classif) are EXCLUDED -- the
    // fast path is forward-only (0), a documented divergence, not a parity failure.
    // MEASURED (Apple Silicon dev box, 20 s x 2 ch): max scored rel=0.0, max abs=0.0 --
    // boundaries are identical (main leg), so the shared-f64 `compute_errors` is
    // bit-identical. The pins keep headroom for a possible sub-tick x86 boundary shift
    // (error percentages move ~1e-3 %/tick), guarded by the OR (rel OR abs) below.
    // T11 note: that OR-lenient comparator is currently MOOT (both terms measured 0.0)
    // -- the assertion is effectively BOUNDARY-GATED (identical boundaries force
    // bit-identical scored columns), so the OR exists solely as headroom against a
    // future sub-tick shift, never as a live tolerance on today's fixtures.
    const SCORED_REL_PIN: f64 = 5.0e-2;
    const SCORED_ABS_PIN: f64 = 5.0e-2;

    let dir = tempfile::tempdir().unwrap();
    let stm = write_stm(dir.path());
    let rows_e = scored_rows(None, &dir.path().join("dump_e"), &stm);
    let rows_f = scored_rows(Some("fast"), &dir.path().join("dump_f"), &stm);

    assert_eq!(rows_e.len(), rows_f.len(), "channel count (scored)");
    let mut max_rel = 0.0_f64;
    let mut max_abs = 0.0_f64;
    for chan in rows_e.keys() {
        let re = &rows_e[chan];
        let rf = &rows_f[chan];
        for &col in &[0usize, 1, 2, 5, 6] {
            let d = (re[col] - rf[col]).abs();
            max_abs = max_abs.max(d);
            let rel = d / re[col].abs().max(1e-6);
            max_rel = max_rel.max(rel);
            assert!(
                rel < SCORED_REL_PIN || d < SCORED_ABS_PIN,
                "scored col {col} chan {chan}: exact={} fast={} (rel {rel}, abs {d})",
                re[col],
                rf[col]
            );
        }
    }
    println!("MEASURE sad_parity_scored: max_rel={max_rel:.3e} max_abs={max_abs:.3e}");
}

// ---------------------------------------------------------------------------
// Rider 1: peephole-default alignment.
// ---------------------------------------------------------------------------

#[test]
fn peephole_default_aligns_with_exact() {
    // The exact BlstmConfig defaults an ABSENT peephole key TRUE. build_aligned_spec must
    // produce the same so a key-omitting config does not silently give the fast net a
    // different peephole configuration than the exact net. NnetSpec::from_legacy defaulted
    // it FALSE until issue #32; it now applies the same rule.
    const KEYS: [&str; 6] = [
        "BLSTM_Forward_IsCellsPeepholesActive",
        "BLSTM_Backward_IsCellsPeepholesActive",
        "BLSTM_Forward_IsGatesPeepholesActive",
        "BLSTM_Backward_IsGatesPeepholesActive",
        "BLSTM_Forward_IsGatesRecurrentPeepholesActive",
        "BLSTM_Backward_IsGatesRecurrentPeepholesActive",
    ];

    // Omitting config: NnetSpec and aligned both -> TRUE.
    let mut omit = tier2_map(None);
    for k in KEYS {
        omit.shift_remove(k);
    }
    assert_eq!(
        NnetSpec::from_legacy(&omit, "BLSTM").unwrap().peepholes,
        [true; 6],
        "NnetSpec defaults absent peepholes TRUE, as BlstmConfig does"
    );
    assert_eq!(
        build_aligned_spec(&omit, "BLSTM").unwrap().peepholes,
        [true; 6],
        "aligned spec must match the exact BlstmConfig default (TRUE)"
    );

    // Explicit-false config: aligned reads them (not hardcoded TRUE) -> FALSE.
    let mut all_false = tier2_map(None);
    for k in KEYS {
        all_false.insert(k.into(), "false".into());
    }
    assert_eq!(
        build_aligned_spec(&all_false, "BLSTM").unwrap().peepholes,
        [false; 6],
        "aligned spec must READ explicit false flags, not hardcode TRUE"
    );

    // Mixed config: both readers put each key in its own slot (a Forward/Backward or
    // kind swap in either one passes the uniform legs above).
    let mut mixed = tier2_map(None);
    for (k, v) in KEYS.iter().zip([
        Some("false"),
        None,
        Some("true"),
        Some("false"),
        None,
        Some("false"),
    ]) {
        match v {
            Some(v) => mixed.insert((*k).into(), v.into()),
            None => mixed.shift_remove(*k),
        };
    }
    let want = [false, true, true, false, true, false];
    assert_eq!(
        NnetSpec::from_legacy(&mixed, "BLSTM").unwrap().peepholes,
        want
    );
    assert_eq!(build_aligned_spec(&mixed, "BLSTM").unwrap().peepholes, want);
}

// ---------------------------------------------------------------------------
// Typed-bail pins (unexercised fast-mode surfaces).
// ---------------------------------------------------------------------------

fn build_fast_bag(map: &mut IndexMap<String, String>) -> anyhow::Result<BagOfProcessors> {
    BagOfProcessors::from_configs(std::slice::from_mut(map), image_mode())
}

#[test]
fn fast_bails_on_pitch_pass() {
    // BLSTM_TDCwindow > 0 -> the pitch second pass (spec R4), unsupported on the fast
    // path -> construction bail.
    let mut m = tier2_map(Some("fast"));
    m.insert("BLSTM_TDCwindow".into(), "0.032".into());
    m.insert("BLSTM_TDC_lags".into(), "0.002,0.016".into());
    match build_fast_bag(&mut m) {
        Err(e) => assert!(
            e.to_string().to_lowercase().contains("pitch"),
            "expected a pitch-pass bail, got: {e}"
        ),
        Ok(_) => panic!("fast + TDCwindow > 0 must bail"),
    }
}

/// THE (cell x direction) DISPATCH TABLE from the tier2 config's point of view.
/// NARROWED TWICE, each narrowing recorded rather than rewritten away, because "which
/// combinations used to be RED" is the contract's history:
///
/// - PHASE 9 TASK 6 landed the CAUSAL new-cell twins (`slstm`/`mamba` + `Direction
///   forward`), pinned in `tests/phase9_fast_parity.rs`.
/// - PHASE 10 TASK 7 landed the BIDIRECTIONAL new-cell twins (`fast::bicell::FastBiCell`,
///   spec S5), pinned in `tests/phase10_bicell_parity.rs`. So the loop below FLIPPED from
///   "must bail" to "must build": on the tier2 config (bidirectional) a `Cell_Type slstm`
///   is now a supported shape.
///
/// WHAT THE FLIP COSTS, stated plainly rather than left implicit. The bail this loop used
/// to assert existed because `NnetSpec` carries neither cell type nor direction, so a
/// pack at least as long as the reader expects is consumed HEAD-FIRST and RUNS --
/// producing one architecture's numbers under another's name. That hazard is NOT fully
/// closed by naming the shape: the sizing check (`FastBiCell::element_count`) catches a
/// SHORT pack only, and an sLSTM bidirectional net is SHORTER than the LSTM one at this
/// geometry, so the config below builds an sLSTM net from tier2's LSTM pack. It is the
/// same inherent property the causal cells have carried since phase 9 (a `slstm_forward`
/// config pointed at an LSTM pack builds too) and it is a property of an UNTAGGED flat
/// pack, not of this dispatch: the config declares the architecture, and pointing it at
/// another architecture's `.bin` is a user error the format cannot detect. What the
/// dispatch does guarantee is that the declared architecture is the one that RUNS.
///
/// `Direction forward` with the LSTM cell FLIPPED THE SAME WAY in phase-10 Task 8
/// (`fast::cells::FastLstm`, spec S6), so the second block below asserts a BUILD too and
/// `classify_fast_shape` has no shape refusal left at all. The untagged-pack caveat above
/// applies to it verbatim, in its other direction: a forward net is SHORTER than the
/// bidirectional pack this config carries, and an over-long pack is consumed head-first by
/// design (the legacy contract, `FastBlstm::from_flat`'s docs) -- so the build below runs a
/// causal net off the head of a bidirectional pack. Again: the config declares the
/// architecture, the dispatch guarantees the declared one RUNS, and pointing a config at
/// the wrong `.bin` is a user error the format cannot detect.
#[test]
fn fast_dispatch_per_cell_type_and_direction() {
    // The tier2 config is BIDIRECTIONAL, so a new cell here selects `FastBiCell`.
    for cell in ["slstm", "mamba"] {
        let mut m = tier2_map(Some("fast"));
        m.insert("BLSTM_Cell_Type".into(), cell.into());
        let bag = build_fast_bag(&mut m)
            .unwrap_or_else(|e| panic!("fast + bidirectional {cell} must build: {e}"));
        assert!(
            matches!(bag.processor(0), Processor::FastSpectral(_)),
            "{cell}: expected the fast SAD driver"
        );
    }

    // Direction: the causal shape needs a `hidden`-wide output MLP, so the config
    // has to declare one for `BlstmConfig::from_legacy` to accept it at all -- the
    // dispatch must be exercised on a config that is otherwise VALID.
    let mut m = tier2_map(Some("fast"));
    m.insert("BLSTM_Direction".into(), "forward".into());
    let hidden: usize = m["BLSTM_LSTMNeuronNb"]
        .split(',')
        .next_back()
        .unwrap()
        .trim()
        .parse()
        .unwrap();
    // Narrow ONLY the output MLP's FIRST layer (`2*hidden -> hidden`); the rest of
    // the list must survive, or `_OutputSubSampling`'s length check fires first and
    // the test would pass for the wrong reason.
    let mut out_layers: Vec<String> = m["BLSTM_OutputNeuronNb"]
        .split(',')
        .map(|v| v.trim().to_string())
        .collect();
    out_layers[0] = hidden.to_string();
    m.insert("BLSTM_OutputNeuronNb".into(), out_layers.join(","));
    // ...and the plain regime: a causal cell under the tier2 window (3.25 s) is refused
    // at bag construction since issue #24's rate-free pre-check (it used to be refused at
    // the first `get_segmentation`, which this dispatch test never reaches).
    m.insert("BLSTM_window".into(), "0".into());
    let bag = build_fast_bag(&mut m)
        .unwrap_or_else(|e| panic!("fast + Direction forward must build (Task 8): {e}"));
    assert!(
        matches!(bag.processor(0), Processor::FastSpectral(_)),
        "causal LSTM: expected the fast SAD driver"
    );

    // The EXACT path is unaffected: the same slstm config builds there (Task 2).
    let mut exact = tier2_map(Some("exact"));
    exact.insert("BLSTM_Cell_Type".into(), "slstm".into());
    build_fast_bag(&mut exact).expect("Inference_Path exact + slstm must build");
}

#[test]
fn fast_bails_on_non_self_normalization() {
    // This bail has been NARROWED twice, each time by a phase that needed one more
    // genuinely-supported type; both narrowings are recorded here rather than
    // rewritten away, because "which types used to be RED" is the contract's history:
    //
    // - Phase 8 Task 1 admitted type 1 (pack-carried external mean/std, the
    //   frozen-stats reference mode).
    // - Phase 9 Task 6 admitted type 0 (NO normalization) -- the exact path's
    //   `_ => {}` arm (`nn/blstm.rs:1271`), a no-op on BOTH sides, which the committed
    //   phase-9 causal gate fixtures use.
    //
    // Everything else (-2, 2, ...) still typed-bails.
    for ok in ["1", "0"] {
        let mut m = tier2_map(Some("fast"));
        m.insert("BLSTM_InputNormalizationType".into(), ok.into());
        build_fast_bag(&mut m)
            .unwrap_or_else(|e| panic!("fast + InputNormalizationType {ok} must construct: {e}"));
    }

    for bad in ["-2", "2"] {
        let mut m = tier2_map(Some("fast"));
        m.insert("BLSTM_InputNormalizationType".into(), bad.into());
        match build_fast_bag(&mut m) {
            Err(e) => assert!(
                e.to_string().contains("InputNormalizationType"),
                "expected a normalization-type bail for type {bad}, got: {e}"
            ),
            Ok(_) => panic!("fast + InputNormalizationType {bad} must bail"),
        }
    }
}

/// Drive the fast driver's `get_segmentation` on the small 3 s excerpt with a forced
/// window/shift so the dispatch reaches an unsupported windowed variant.
fn run_fast_small(map: &mut IndexMap<String, String>) -> anyhow::Result<()> {
    let mut bag = BagOfProcessors::from_configs(std::slice::from_mut(map), image_mode())?;
    let mut audio = read_audio(&fixture("phase1/excerpt_2ch_8k.wav"), 0.0, 3.0, 0, None).unwrap();
    let dur = (audio.data.ncols() as f64 - 1.0) / audio.sample_rate as f64;
    let n_chan = audio.data.nrows();
    let mut seg: Vec<Segmentation> = (0..n_chan).map(|_| Segmentation::new(dur)).collect();
    bag.run_get_segmentation(0, &mut audio, &mut seg)
}

#[test]
fn fast_bails_on_plain_and_truncate() {
    // Plain (BLSTM_window 0 -> window_size 0): the non-windowed forward is unsupported.
    let mut plain = tier2_map(Some("fast"));
    plain.insert("BLSTM_window".into(), "0.0".into());
    match run_fast_small(&mut plain) {
        Err(e) => assert!(
            e.to_string().to_lowercase().contains("plain"),
            "expected a plain-path bail, got: {e}"
        ),
        Ok(_) => panic!("fast + window 0 must bail (plain unsupported)"),
    }

    // Truncate (BLSTM_shift 0 -> window_shift < 1 -> no_overlap): unsupported.
    let mut trunc = tier2_map(Some("fast"));
    trunc.insert("BLSTM_shift".into(), "0.0".into());
    match run_fast_small(&mut trunc) {
        Err(e) => assert!(
            e.to_string().to_lowercase().contains("truncate"),
            "expected a truncate-path bail, got: {e}"
        ),
        Ok(_) => panic!("fast + shift 0 must bail (truncate unsupported)"),
    }
}
