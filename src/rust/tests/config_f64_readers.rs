//! Issue #50: every f64 and list config reader in the crate goes through the shared
//! `legacy_config` family, so a non-finite value (`nan`, `inf`, `1e400`) on any of its keys
//! errors naming the key instead of flowing into the engine, and every list takes the one
//! grammar (the trailing comma, the `*` repeater). One pin per migrated module,
//! driven through that module's public constructor on a committed fixture; the shared
//! grammar itself is unit-pinned in `legacy_config::f64_reader_tests`.

use indexmap::IndexMap;
use speech::cli::{Mode, ModeKind};
use speech::config::NnetSpec;
use speech::engine::bag_of_processors::BagOfProcessors;
use speech::fast::driver::FastTwinLid;
use speech::fast::stream::StreamingSession;
use speech::features::pipeline::FeatureConfig;
use speech::legacy_config::parse_legacy_config;
use speech::tasks::lid::TwinBlstmSpectralLid;
use speech::tasks::sad::TdcSegmenter;
use speech::tasks::segmenter::{DriverConfig, SegmenterConfig};

fn load(rel: &str) -> IndexMap<String, String> {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/reference_data")
        .join(rel);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path:?}: {e}"));
    parse_legacy_config(&text)
}

fn with(mut m: IndexMap<String, String>, key: &str, value: &str) -> IndexMap<String, String> {
    m.insert(key.to_string(), value.to_string());
    m
}

/// The full anyhow chain of an `Err`; the driver types are not `Debug`, so no `unwrap_err`.
fn err_of<T>(r: anyhow::Result<T>, what: &str) -> String {
    match r {
        Ok(_) => panic!("{what}: must not build"),
        Err(e) => format!("{e:#}"),
    }
}

fn tdc_bag_config() -> IndexMap<String, String> {
    let m = load("phase2b/tdc.config");
    let m = with(m, "numOuterThreads", "1");
    let m = with(m, "Algo_choice", "1");
    with(m, "TDC_lags", "-0.002,0.016")
}

#[test]
fn bag_of_processors_rejects_non_finite_audio_offset() {
    let mut cfg = with(tdc_bag_config(), "Audio_offset", "nan");
    let mode = Mode {
        kind: ModeKind::Solo,
        verbose: false,
    };
    let err = err_of(
        BagOfProcessors::from_configs(std::slice::from_mut(&mut cfg), mode),
        "Audio_offset nan",
    );
    assert!(err.contains("Audio_offset") && err.contains("nan"), "{err}");
}

#[test]
fn feature_config_rejects_non_finite_scalar_and_lag() {
    let base = load("phase4a/tier2_spectral.config");
    let err = err_of(
        FeatureConfig::from_legacy(&with(base.clone(), "BLSTM_spectrum_shift", "inf"), "BLSTM"),
        "BLSTM_spectrum_shift inf",
    );
    assert!(
        err.contains("BLSTM_spectrum_shift") && err.contains("inf"),
        "{err}"
    );
    // The pitch second pass's lag list (read only under `TDCwindow != 0`) is the shared
    // list grammar: a trailing comma reads, a non-finite entry errors naming the key.
    let m = with(base, "BLSTM_TDCwindow", "0.02");
    let m = with(m, "BLSTM_TDCshift", "0.01");
    let err = err_of(
        FeatureConfig::from_legacy(&with(m, "BLSTM_TDC_lags", "0.001,nan"), "BLSTM"),
        "BLSTM_TDC_lags nan",
    );
    assert!(
        err.contains("BLSTM_TDC_lags") && err.contains("nan"),
        "{err}"
    );
}

#[test]
fn tdc_segmenter_rejects_non_finite_balance_and_reads_the_list_grammar() {
    let base = tdc_bag_config();
    let err = err_of(
        TdcSegmenter::from_legacy(&with(base.clone(), "TDC_balance", "1e400")),
        "TDC_balance 1e400",
    );
    assert!(
        err.contains("TDC_balance") && err.contains("1e400"),
        "{err}"
    );
    assert!(TdcSegmenter::from_legacy(&with(base.clone(), "TDC_lags", "-0.002,0.016,")).is_ok());
    let err = err_of(
        TdcSegmenter::from_legacy(&with(base, "TDC_lags", "0.016*x")),
        "TDC_lags 0.016*x",
    );
    assert!(err.contains("TDC_lags") && err.contains("0.016*x"), "{err}");
}

#[test]
fn segmenter_and_driver_configs_reject_non_finite_values() {
    let base = tdc_bag_config();
    let err = SegmenterConfig::from_config(
        &with(base.clone(), "TDC_decision_thresh_rising", "nan"),
        "TDC",
    )
    .unwrap_err()
    .to_string();
    assert!(
        err.contains("TDC_decision_thresh_rising") && err.contains("nan"),
        "{err}"
    );
    let err =
        SegmenterConfig::from_config(&with(base.clone(), "TDC_min_speech", "0.1,inf,0.1"), "TDC")
            .unwrap_err()
            .to_string();
    assert!(
        err.contains("TDC_min_speech") && err.contains("inf"),
        "{err}"
    );
    // The issue's concrete case: `BackPropWER nan` used to evaluate `nan >= 0` as false and
    // carry on here while `cost.rs` errored on the same key.
    let err = err_of(
        DriverConfig::from_config(&with(base, "TDC_BackPropWER", "nan"), "TDC"),
        "TDC_BackPropWER nan",
    );
    assert!(
        err.contains("TDC_BackPropWER") && err.contains("nan"),
        "{err}"
    );
}

#[test]
fn twin_lid_drivers_reject_non_finite_window() {
    let exact = with(load("phase4b/twin_mode0.config"), "BLSTM_LID_window", "nan");
    let err = err_of(
        TwinBlstmSpectralLid::from_legacy(&exact, None, None),
        "exact BLSTM_LID_window nan",
    );
    assert!(
        err.contains("BLSTM_LID_window") && err.contains("nan"),
        "{err}"
    );
    let fast = with(
        load("phase4b/twin_mode7.config"),
        "BLSTM_LID_NoiseMagnitude",
        "-inf",
    );
    let err = err_of(
        FastTwinLid::from_legacy(&fast, None, None),
        "fast BLSTM_LID_NoiseMagnitude -inf",
    );
    assert!(
        err.contains("BLSTM_LID_NoiseMagnitude") && err.contains("-inf"),
        "{err}"
    );
}

#[test]
fn streaming_session_rejects_non_finite_fixed_gain() {
    let m = with(
        load("phase4a/tier2_spectral.config"),
        "Audio_fixed_gain",
        "nan",
    );
    let err = err_of(StreamingSession::new(&m, 8000.0, 1), "Audio_fixed_gain nan");
    assert!(
        err.contains("Audio_fixed_gain") && err.contains("nan"),
        "{err}"
    );
}

#[test]
fn nnet_spec_reads_the_list_grammar() {
    let base = load("phase4a/tier2_spectral.config");
    // The fixture declares `23,24,24`; the repeat form spells the same list.
    let plain = NnetSpec::from_legacy(&base, "BLSTM").unwrap();
    assert_eq!(plain.lstm_neuron_nb, vec![23, 24, 24]);
    let repeated = with(base.clone(), "BLSTM_LSTMNeuronNb", "23,24*2,");
    assert_eq!(
        NnetSpec::from_legacy(&repeated, "BLSTM")
            .unwrap()
            .lstm_neuron_nb,
        plain.lstm_neuron_nb
    );
    let err = NnetSpec::from_legacy(&with(base, "BLSTM_OutputNeuronNb", "2*x"), "BLSTM")
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("BLSTM_OutputNeuronNb") && err.contains("2*x"),
        "{err}"
    );
}
