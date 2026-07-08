//! TOML canonical config: the declarative flattening between a `[section]`-grouped
//! TOML document and the flat `KEY -> value` map the engine consumes (the exact
//! shape `legacy_config::parse_legacy_config` produces).
//!
//! Port-only addition (Phase 4c Task 5) -- the legacy engine has no TOML notion at
//! all; this module exists purely so this port's config authoring story does not
//! have to stay in the legacy whitespace `.config` format forever. Semantics stay
//! IDENTICAL either way: whichever format a config arrives in, it collapses to the
//! same `IndexMap<String, String>` (last-wins on duplicate keys is a `.config`-only
//! concept -- TOML tables cannot carry a duplicate key at all, so that ambiguity is
//! structurally impossible on the TOML side; see `toml_map_equivalence` in
//! `tests/phase4c_toml.rs` for the round-trip proof through a real duplicate-key
//! fixture, `configs/legacy/LID_BLSTM.config`).
//!
//! ## The table
//!
//! [`KEY_TABLE`] is the single declarative `(legacy_key, toml_section, toml_key)`
//! list, both directions read off the SAME table (no separate encode/decode maps to
//! drift apart). It covers every key used by the committed fixture configs (the
//! phase0/phase4a/phase4b `.config` fixtures plus `configs/legacy/LID_BLSTM.config`
//! -- 175 distinct legacy keys as of this writing), grouped into sections that
//! extend the `configs/lid/lid_blstm.toml` scaffold's original four
//! (`engine`/`audio`/`decision`/`spectrum`/`preprocess`) with new ones per
//! algorithm family: `corpus`, `display`, `pitch` (the BLSTM spectral segmenter's
//! inner TDC/LTSV sub-passes used for pitch warping), `nn`/`cost` (the main BLSTM
//! SAD net), `nn_lid`/`cost_lid` (the Twin's `BLSTM_LID_*` net), `ltsv`/`tdc` (the
//! two NN-free standalone algorithms), and `cnn` (dead-per-`CLAUDE.md` but present
//! in fixtures, kept for round-trip completeness).
//!
//! A handful of legacy prefixes collide when naively suffix-stripped (e.g.
//! `BLSTM_window`/`BLSTM_shift` are the raw time-domain framing params, DISTINCT
//! from `BLSTM_spectrum_shift`, the periodogram shift); those pairs are
//! disambiguated as `frame_window`/`frame_shift` vs `order`/`shift` inside
//! `[spectrum]` (and the analogous `[ltsv]` pair) rather than picked arbitrarily.
//!
//! Any legacy key NOT in [`KEY_TABLE`] (future/unmapped keys) round-trips losslessly
//! through the `[legacy.raw]` table verbatim -- the lossless escape hatch. No key is
//! ever silently dropped: [`map_to_toml`] routes every map entry to either its
//! declared section or `[legacy.raw]`, and [`toml_to_map`] rejects any TOML key that
//! is in neither a known section nor `[legacy.raw]` (a typo'd or hand-added section
//! name is a hard error, not a silent no-op).
//!
//! ## Value fidelity
//!
//! The legacy map values are always strings (the engine parses them itself); the
//! risk is a native TOML float silently failing to reproduce a fixture's exact
//! decimal string (e.g. `3.795068189162301e-01`). This module never emits or
//! accepts a native TOML float for a legacy value: [`map_to_toml`] classifies each
//! *value* (not each key) dynamically --
//!
//! - a value that is exactly `"true"`/`"false"` becomes a native TOML boolean;
//! - a value that round-trips exactly through `i64` (`n.to_string() == value`,
//!   which rejects leading zeros, a leading `+`, etc.) becomes a native TOML
//!   integer;
//! - everything else (floats, comma-separated lists, words like `hamming`, paths)
//!   becomes a quoted TOML string, verbatim.
//!
//! [`toml_to_map`] mirrors this: TOML booleans/integers/strings decode back to the
//! legacy string trivially and exactly; a native TOML float (or array/table/
//! datetime) at a leaf position is a hard error, since this module never writes one
//! and accepting one would silently reintroduce the exact fidelity risk this scheme
//! exists to avoid.
//!
//! Per-value (not per-key) classification means the same legacy key can format
//! natively in one config and as a string in another (e.g. `Audio_max_duration 240`
//! vs `Audio_max_duration 2.0`) -- this is correct and expected, not an
//! inconsistency: [`toml_to_map`] only ever needs the value back as a string, and
//! both encodings recover it exactly.
//!
//! Serialization reuses `toml::Value`'s own `Display` impl (`value_literal`) for
//! correct TOML escaping instead of hand-rolling it, and parsing reuses
//! `toml::Table`'s `FromStr` -- this is the module that "wires" the previously
//! `toml = "0.9"` declared-but-unused dependency.

use anyhow::{Context, Result, anyhow, bail};
use indexmap::IndexMap;

/// One entry of the bidirectional legacy-key <-> TOML `section.key` table.
/// `(legacy_key, toml_section, toml_key)`.
type KeyMapping = (&'static str, &'static str, &'static str);

/// The declarative flattening table. See the module doc for the fidelity/section
/// design; see `tests/phase4c_toml.rs` for the full-coverage + round-trip proofs.
#[rustfmt::skip]
static KEY_TABLE: &[KeyMapping] = &[
    // --- engine: orchestration / top-level run params -----------------------
    ("Algo_choice",                              "engine", "algo_choice"),
    ("numOuterThreads",                          "engine", "num_outer_threads"),
    ("numInnerThreads",                          "engine", "num_inner_threads"),
    ("multiConfigResultsOutputFile",             "engine", "results_file"),
    ("Dump_Directory",                           "engine", "dump_directory"),
    ("File_Type",                                "engine", "file_type"),
    ("Neural_Networks_BackPropagation_Epochs",   "engine", "backpropagation_epochs"),
    ("Neural_Networks_Gradient_Check_Epsilon",   "engine", "gradient_check_epsilon"),
    ("exclude_nontrans",                         "engine", "exclude_nontrans"),

    // --- corpus: listing / class-mapping CSV paths ---------------------------
    ("fileslisting",                             "corpus", "fileslisting"),
    ("language2classmapping",                    "corpus", "language2classmapping"),

    // --- display: legacy image/plot dump geometry (dead-in-practice but present) --
    ("Display_Height",                           "display", "height"),
    ("Display_MillisecondsPerPixel",             "display", "milliseconds_per_pixel"),
    ("Display_Output_Directory",                 "display", "output_directory"),
    ("Display_WaveAll_Height",                   "display", "wave_all_height"),
    ("Display_WaveAll_Width",                    "display", "wave_all_width"),

    // --- audio: per-file offset/duration clamp --------------------------------
    ("Audio_offset",                             "audio", "offset"),
    ("Audio_max_duration",                       "audio", "max_duration"),

    // --- spectrum: BLSTM (main SAD net) feature extraction --------------------
    ("BLSTM_spectrum_order",                     "spectrum", "order"),
    ("BLSTM_spectrum_shift",                     "spectrum", "shift"),
    ("BLSTM_spectrum_temporal_convolution_size", "spectrum", "temporal_convolution_size"),
    ("BLSTM_spectrum_temporal_convolution_type", "spectrum", "temporal_convolution_type"),
    ("BLSTM_window",                             "spectrum", "frame_window"),
    ("BLSTM_shift",                              "spectrum", "frame_shift"),
    ("BLSTM_windowing_type",                     "spectrum", "windowing_type"),
    ("BLSTM_windowing_param",                    "spectrum", "windowing_param"),
    ("BLSTM_minFreq",                            "spectrum", "min_freq"),
    ("BLSTM_maxFreq",                            "spectrum", "max_freq"),
    ("BLSTM_minMelFreq",                         "spectrum", "min_mel_freq"),
    ("BLSTM_maxMelFreq",                         "spectrum", "max_mel_freq"),
    ("BLSTM_nb_bins",                            "spectrum", "nb_bins"),
    ("BLSTM_is_log_mel",                         "spectrum", "is_log_mel"),
    ("BLSTM_nb_DCT",                             "spectrum", "nb_dct"),
    ("BLSTM_IgnoreFirstDCT",                     "spectrum", "ignore_first_dct"),
    ("BLSTM_ComputeDeltasNb",                    "spectrum", "compute_deltas_nb"),
    ("BLSTM_ComputeDeltaDeltasNb",               "spectrum", "compute_delta_deltas_nb"),
    ("BLSTM_flag_DCOffset",                      "spectrum", "flag_dc_offset"),

    // --- preprocess: BLSTM raw-audio conditioning -----------------------------
    ("BLSTM_preemph_ratio",                      "preprocess", "preemph_ratio"),
    ("BLSTM_noise_seed",                         "preprocess", "noise_seed"),
    ("BLSTM_noise_ratio",                        "preprocess", "noise_ratio"),

    // --- decision: BLSTM SAD decision curve + segmentation cleanup ------------
    ("BLSTM_decision_thresh_rising",             "decision", "thresh_rising"),
    ("BLSTM_decision_area_rising",               "decision", "area_rising"),
    ("BLSTM_decision_thresh_falling",            "decision", "thresh_falling"),
    ("BLSTM_decision_area_falling",              "decision", "area_falling"),
    ("BLSTM_convolution_window_size",            "decision", "convolution_window_size"),
    ("BLSTM_convolution_window_type",            "decision", "convolution_window_type"),
    ("BLSTM_speech_padding",                     "decision", "speech_padding"),
    ("BLSTM_min_speech",                         "decision", "min_speech"),
    ("BLSTM_min_silence",                        "decision", "min_silence"),

    // --- pitch: the BLSTM spectral segmenter's inner TDC/LTSV pitch-pass sub-params --
    ("BLSTM_LTSVwindow",                         "pitch", "ltsv_window"),
    ("BLSTM_LTSVshift",                          "pitch", "ltsv_shift"),
    ("BLSTM_TDCwindow",                          "pitch", "tdc_window"),
    ("BLSTM_TDCshift",                           "pitch", "tdc_shift"),
    ("BLSTM_TDC_balance",                        "pitch", "tdc_balance"),
    ("BLSTM_TDC_lags",                           "pitch", "tdc_lags"),
    ("BLSTM_TDC_windowing_type",                 "pitch", "tdc_windowing_type"),
    ("BLSTM_TDC_windowing_param",                "pitch", "tdc_windowing_param"),

    // --- nn: BLSTM (main SAD net) topology / training params ------------------
    ("BLSTM_LSTMNeuronNb",                       "nn", "lstm_neuron_nb"),
    ("BLSTM_LSTMSubSampling",                    "nn", "lstm_sub_sampling"),
    ("BLSTM_NNType",                             "nn", "nn_type"),
    ("BLSTM_NNetInputSize",                      "nn", "nnet_input_size"),
    ("BLSTM_OutputNeuronNb",                     "nn", "output_neuron_nb"),
    ("BLSTM_OutputSubSampling",                  "nn", "output_subsampling"),
    ("BLSTM_PruningThresh",                      "nn", "pruning_thresh"),
    ("BLSTM_TargetEnforcementStep",              "nn", "target_enforcement_step"),
    ("BLSTM_TwoSweeps",                          "nn", "two_sweeps"),
    ("BLSTM_InputNormalizationType",             "nn", "input_normalization_type"),
    ("BLSTM_weightsFile",                        "nn", "weights_file"),
    ("BLSTM_use_cep_files",                      "nn", "use_cep_files"),
    ("BLSTM_BackPropWER",                        "nn", "backprop_wer"),
    ("BLSTM_BackPropagationActivated",           "nn", "backpropagation_activated"),
    ("BLSTM_BackPropagationRpropInit",           "nn", "backpropagation_rprop_init"),
    ("BLSTM_Backward_IsCellsPeepholesActive",    "nn", "backward_cells_peepholes_active"),
    ("BLSTM_Backward_IsGatesPeepholesActive",    "nn", "backward_gates_peepholes_active"),
    ("BLSTM_Backward_IsGatesRecurrentPeepholesActive", "nn", "backward_gates_recurrent_peepholes_active"),
    ("BLSTM_Forward_IsCellsPeepholesActive",     "nn", "forward_cells_peepholes_active"),
    ("BLSTM_Forward_IsGatesPeepholesActive",     "nn", "forward_gates_peepholes_active"),
    ("BLSTM_Forward_IsGatesRecurrentPeepholesActive",  "nn", "forward_gates_recurrent_peepholes_active"),

    // --- cost: BLSTM (main SAD net) cost law ----------------------------------
    ("BLSTM_CostLawSpeech",                      "cost", "law_speech"),
    ("BLSTM_CostLawNoSpeech",                    "cost", "law_no_speech"),
    ("BLSTM_CostLawParamSpeech",                 "cost", "law_param_speech"),
    ("BLSTM_CostLawParamNoSpeech",               "cost", "law_param_no_speech"),
    ("BLSTM_CostLawThreshSpeech",                "cost", "law_thresh_speech"),
    ("BLSTM_CostLawThreshNoSpeech",              "cost", "law_thresh_no_speech"),
    ("BLSTM_CostPonderation",                    "cost", "ponderation"),

    // --- ltsv: standalone LTSV algo (Algo_choice 2, NN-free) ------------------
    ("LTSV_spectrum_order",                      "ltsv", "spectrum_order"),
    ("LTSV_spectrum_shift",                      "ltsv", "spectrum_shift"),
    ("LTSV_spectrum_temporal_convolution_size",  "ltsv", "spectrum_temporal_convolution_size"),
    ("LTSV_spectrum_temporal_convolution_type",  "ltsv", "spectrum_temporal_convolution_type"),
    ("LTSV_window",                              "ltsv", "frame_window"),
    ("LTSV_shift",                               "ltsv", "frame_shift"),
    ("LTSV_windowing_type",                      "ltsv", "windowing_type"),
    ("LTSV_windowing_param",                     "ltsv", "windowing_param"),
    ("LTSV_minFreq",                             "ltsv", "min_freq"),
    ("LTSV_maxFreq",                             "ltsv", "max_freq"),
    ("LTSV_minMelFreq",                          "ltsv", "min_mel_freq"),
    ("LTSV_maxMelFreq",                          "ltsv", "max_mel_freq"),
    ("LTSV_nb_bins",                             "ltsv", "nb_bins"),
    ("LTSV_is_log_mel",                          "ltsv", "is_log_mel"),
    ("LTSV_nb_DCT",                              "ltsv", "nb_dct"),
    ("LTSV_IgnoreFirstDCT",                      "ltsv", "ignore_first_dct"),
    ("LTSV_ComputeDeltasNb",                     "ltsv", "compute_deltas_nb"),
    ("LTSV_ComputeDeltaDeltasNb",                "ltsv", "compute_delta_deltas_nb"),
    ("LTSV_flag_DCOffset",                       "ltsv", "flag_dc_offset"),
    ("LTSV_preemph_ratio",                       "ltsv", "preemph_ratio"),
    ("LTSV_noise_seed",                          "ltsv", "noise_seed"),
    ("LTSV_noise_ratio",                         "ltsv", "noise_ratio"),
    ("LTSV_decision_thresh_rising",              "ltsv", "decision_thresh_rising"),
    ("LTSV_decision_area_rising",                "ltsv", "decision_area_rising"),
    ("LTSV_decision_thresh_falling",             "ltsv", "decision_thresh_falling"),
    ("LTSV_decision_area_falling",               "ltsv", "decision_area_falling"),
    ("LTSV_convolution_window_size",             "ltsv", "convolution_window_size"),
    ("LTSV_convolution_window_type",             "ltsv", "convolution_window_type"),
    ("LTSV_speech_padding",                      "ltsv", "speech_padding"),
    ("LTSV_min_speech",                          "ltsv", "min_speech"),
    ("LTSV_min_silence",                         "ltsv", "min_silence"),
    ("LTSV_LTSVwindow",                          "ltsv", "ltsv_window"),
    ("LTSV_LTSVshift",                           "ltsv", "ltsv_shift"),
    ("LTSV_TDCwindow",                           "ltsv", "tdc_window"),

    // --- tdc: standalone TDC algo (Algo_choice 1, NN-free) ---------------------
    ("TDC_window",                               "tdc", "window"),
    ("TDC_shift",                                "tdc", "shift"),
    ("TDC_lags",                                 "tdc", "lags"),
    ("TDC_balance",                              "tdc", "balance"),
    ("TDC_windowing_type",                       "tdc", "windowing_type"),
    ("TDC_windowing_param",                      "tdc", "windowing_param"),
    ("TDC_flag_DCOffset",                        "tdc", "flag_dc_offset"),
    ("TDC_preemph_ratio",                        "tdc", "preemph_ratio"),
    ("TDC_noise_seed",                           "tdc", "noise_seed"),
    ("TDC_noise_ratio",                          "tdc", "noise_ratio"),
    ("TDC_decision_thresh_rising",               "tdc", "decision_thresh_rising"),
    ("TDC_decision_area_rising",                 "tdc", "decision_area_rising"),
    ("TDC_decision_thresh_falling",              "tdc", "decision_thresh_falling"),
    ("TDC_decision_area_falling",                "tdc", "decision_area_falling"),
    ("TDC_convolution_window_size",              "tdc", "convolution_window_size"),
    ("TDC_convolution_window_type",              "tdc", "convolution_window_type"),
    ("TDC_speech_padding",                       "tdc", "speech_padding"),
    ("TDC_min_speech",                           "tdc", "min_speech"),
    ("TDC_min_silence",                          "tdc", "min_silence"),

    // --- nn_lid: Twin LID net (BLSTM_LID_*) topology / training params --------
    ("BLSTM_LID_LSTMNeuronNb",                   "nn_lid", "lstm_neuron_nb"),
    ("BLSTM_LID_LSTMSubSampling",                "nn_lid", "lstm_sub_sampling"),
    ("BLSTM_LID_NNType",                         "nn_lid", "nn_type"),
    ("BLSTM_LID_NNetInputSize",                  "nn_lid", "nnet_input_size"),
    ("BLSTM_LID_OutputNeuronNb",                 "nn_lid", "output_neuron_nb"),
    ("BLSTM_LID_OutputSubSampling",              "nn_lid", "output_subsampling"),
    ("BLSTM_LID_TrainingPruningThreshold",       "nn_lid", "training_pruning_threshold"),
    ("BLSTM_LID_TwoSweeps",                      "nn_lid", "two_sweeps"),
    ("BLSTM_LID_InputNormalizationType",         "nn_lid", "input_normalization_type"),
    ("BLSTM_LID_weightsFile",                    "nn_lid", "weights_file"),
    ("BLSTM_LID_BackPropWER",                    "nn_lid", "backprop_wer"),
    ("BLSTM_LID_BackPropagationActivated",       "nn_lid", "backpropagation_activated"),
    ("BLSTM_LID_BackPropagationRpropInit",       "nn_lid", "backpropagation_rprop_init"),
    ("BLSTM_LID_Backward_IsCellsPeepholesActive", "nn_lid", "backward_cells_peepholes_active"),
    ("BLSTM_LID_Backward_IsGatesPeepholesActive", "nn_lid", "backward_gates_peepholes_active"),
    ("BLSTM_LID_Backward_IsGatesRecurrentPeepholesActive", "nn_lid", "backward_gates_recurrent_peepholes_active"),
    ("BLSTM_LID_Forward_IsCellsPeepholesActive", "nn_lid", "forward_cells_peepholes_active"),
    ("BLSTM_LID_Forward_IsGatesPeepholesActive", "nn_lid", "forward_gates_peepholes_active"),
    ("BLSTM_LID_Forward_IsGatesRecurrentPeepholesActive",  "nn_lid", "forward_gates_recurrent_peepholes_active"),
    ("BLSTM_LID_MinNbOfFrames",                  "nn_lid", "min_nb_of_frames"),
    ("BLSTM_LID_Mode",                           "nn_lid", "mode"),
    ("BLSTM_LID_NoiseMagnitude",                 "nn_lid", "noise_magnitude"),
    ("BLSTM_LID_PostProcessMode",                "nn_lid", "post_process_mode"),
    ("BLSTM_LID_decision_thresh_falling",        "nn_lid", "decision_thresh_falling"),
    ("BLSTM_LID_decision_thresh_rising",         "nn_lid", "decision_thresh_rising"),
    ("BLSTM_LID_shift",                          "nn_lid", "shift"),
    ("BLSTM_LID_window",                         "nn_lid", "window"),
    ("BLSTM_LID_DetectionThreshold",             "nn_lid", "detection_threshold"),
    ("BLSTM_LID_DumpInternals",                  "nn_lid", "dump_internals"),

    // --- cost_lid: Twin LID net cost law ---------------------------------------
    ("BLSTM_LID_CostLawSpeech",                  "cost_lid", "law_speech"),
    ("BLSTM_LID_CostLawNoSpeech",                "cost_lid", "law_no_speech"),
    ("BLSTM_LID_CostLawParamSpeech",             "cost_lid", "law_param_speech"),
    ("BLSTM_LID_CostLawParamNoSpeech",           "cost_lid", "law_param_no_speech"),
    ("BLSTM_LID_CostLawThreshSpeech",            "cost_lid", "law_thresh_speech"),
    ("BLSTM_LID_CostLawThreshNoSpeech",          "cost_lid", "law_thresh_no_speech"),
    ("BLSTM_LID_CostPonderation",                "cost_lid", "ponderation"),

    // --- cnn: dead-per-CLAUDE.md (Conv NOT ported), present in fixtures --------
    ("CNN_ConvNeuronNb",                         "cnn", "conv_neuron_nb"),
];

/// The two-level `[legacy.raw]` escape-hatch table name (a dotted TOML table).
const RAW_SECTION: &str = "legacy";
const RAW_SUBSECTION: &str = "raw";

fn find_by_legacy(key: &str) -> Option<&'static KeyMapping> {
    KEY_TABLE.iter().find(|(legacy, _, _)| *legacy == key)
}

fn find_by_toml(section: &str, key: &str) -> Option<&'static KeyMapping> {
    KEY_TABLE
        .iter()
        .find(|(_, s, k)| *s == section && *k == key)
}

/// Classify a legacy string value into the TOML literal that preserves it exactly
/// (see the module doc's "Value fidelity" section): native bool/int where the
/// round-trip is exact, a quoted string otherwise. Uses `toml::Value`'s own
/// `Display` for correct escaping.
fn value_literal(value: &str) -> String {
    let v = if value == "true" {
        toml::Value::Boolean(true)
    } else if value == "false" {
        toml::Value::Boolean(false)
    } else if let Ok(n) = value.parse::<i64>() {
        if n.to_string() == value {
            toml::Value::Integer(n)
        } else {
            toml::Value::String(value.to_string())
        }
    } else {
        toml::Value::String(value.to_string())
    };
    v.to_string()
}

/// Render a legacy key as a TOML key (bare if it is a valid bare TOML identifier,
/// quoted otherwise). Every `KEY_TABLE` `toml_key` is bare by construction; this
/// only matters for `[legacy.raw]` keys, which are arbitrary legacy key strings.
fn toml_key_repr(name: &str) -> String {
    let is_bare = !name.is_empty()
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-');
    if is_bare {
        name.to_string()
    } else {
        toml::Value::String(name.to_string()).to_string()
    }
}

/// Decode a parsed `toml::Value` leaf back to the exact legacy string. Mirrors
/// `value_literal`'s encoding; a `Float`/`Array`/`Table`/`Datetime` leaf is a hard
/// error since this module never emits one (see the module doc).
fn value_to_legacy_string(v: &toml::Value, path: &str) -> Result<String> {
    match v {
        toml::Value::Boolean(b) => Ok(if *b {
            "true".to_string()
        } else {
            "false".to_string()
        }),
        toml::Value::Integer(i) => Ok(i.to_string()),
        toml::Value::String(s) => Ok(s.clone()),
        toml::Value::Float(_) => bail!(
            "{path}: native TOML float is not supported for a legacy config value \
             (exact decimal fidelity is not guaranteed) -- write it as a quoted string"
        ),
        toml::Value::Array(_) => bail!("{path}: TOML array is not a legacy scalar value"),
        toml::Value::Table(_) => bail!("{path}: unexpected nested table"),
        toml::Value::Datetime(_) => bail!("{path}: TOML datetime is not a legacy scalar value"),
    }
}

/// Parse a canonical TOML config document into the flat legacy `KEY -> value` map
/// (the same shape `legacy_config::parse_legacy_config` produces). Every `[section]`
/// key must be in [`KEY_TABLE`] (an unknown `section.key` is a hard error -- use
/// `[legacy.raw]` for keys with no canonical mapping); `[legacy.raw]` keys pass
/// through verbatim under their own name.
pub fn toml_to_map(text: &str) -> Result<IndexMap<String, String>> {
    let doc: toml::Table = text.parse().context("invalid TOML")?;
    let mut out = IndexMap::new();

    for (section_name, section_val) in doc.iter() {
        let section_table = section_val
            .as_table()
            .ok_or_else(|| anyhow!("[{section_name}] must be a table of key = value pairs"))?;

        if section_name == RAW_SECTION {
            let raw_val = section_table.get(RAW_SUBSECTION).ok_or_else(|| {
                anyhow!("[{RAW_SECTION}] must contain a [{RAW_SECTION}.{RAW_SUBSECTION}] table")
            })?;
            let raw_table = raw_val.as_table().ok_or_else(|| {
                anyhow!("[{RAW_SECTION}.{RAW_SUBSECTION}] must be a table of key = value pairs")
            })?;
            for (key, v) in raw_table.iter() {
                let s =
                    value_to_legacy_string(v, &format!("[{RAW_SECTION}.{RAW_SUBSECTION}].{key}"))?;
                out.insert(key.clone(), s);
            }
            continue;
        }

        for (key, v) in section_table.iter() {
            let mapping = find_by_toml(section_name, key).ok_or_else(|| {
                anyhow!(
                    "unknown TOML key '{section_name}.{key}' (no legacy mapping; \
                     use [{RAW_SECTION}.{RAW_SUBSECTION}] for keys without a canonical mapping)"
                )
            })?;
            let s = value_to_legacy_string(v, &format!("{section_name}.{key}"))?;
            out.insert(mapping.0.to_string(), s);
        }
    }

    Ok(out)
}

/// Render a flat legacy `KEY -> value` map as a canonical TOML document: every key
/// present in [`KEY_TABLE`] goes to its declared `[section]`, in `KEY_TABLE`'s
/// declared order; every other key (the escape hatch) goes to `[legacy.raw]`,
/// sorted by key for a deterministic diff. `toml_to_map(&map_to_toml(m)) == m` for
/// any `m` (order-insensitively; `IndexMap`'s `PartialEq` already is) -- see
/// `toml_map_equivalence` in `tests/phase4c_toml.rs`.
pub fn map_to_toml(map: &IndexMap<String, String>) -> String {
    let mut sections: IndexMap<&'static str, Vec<(&'static str, &str)>> = IndexMap::new();
    for (legacy, section, toml_key) in KEY_TABLE {
        if let Some(v) = map.get(*legacy) {
            sections
                .entry(section)
                .or_default()
                .push((toml_key, v.as_str()));
        }
    }

    let mut raw: Vec<(&str, &str)> = map
        .iter()
        .filter(|(k, _)| find_by_legacy(k).is_none())
        .map(|(k, v)| (k.as_str(), v.as_str()))
        .collect();
    raw.sort_by_key(|(k, _)| *k);

    let mut out = String::new();
    for (section, entries) in &sections {
        out.push_str(&format!("[{section}]\n"));
        for (key, val) in entries {
            out.push_str(&format!(
                "{} = {}\n",
                toml_key_repr(key),
                value_literal(val)
            ));
        }
        out.push('\n');
    }

    if !raw.is_empty() {
        out.push_str(&format!("[{RAW_SECTION}.{RAW_SUBSECTION}]\n"));
        for (key, val) in &raw {
            out.push_str(&format!(
                "{} = {}\n",
                toml_key_repr(key),
                value_literal(val)
            ));
        }
        out.push('\n');
    }

    out
}
