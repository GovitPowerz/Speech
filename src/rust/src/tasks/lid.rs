//! Spectral LID driver (Algo 5; `BLSTMSpectralLID.{h,cpp}`) -- the FIRST LID driver.
//!
//! Ported from legacy C++: `BLSTMSpectralLID.cpp` (ctor `:10-14`, `getSegmentation`
//! `:27-460`). `BLSTMSpectralLID : BLSTMSpectralSegmenter : LongTermSpectralVariation
//! : Segmenter`; the ctor calls `BLSTMSpectralSegmenter::buildFromConf(conf, "BLSTM",
//! ...)` + a `BLSTMNeuralNetwork(conf, "BLSTM", false)`, so it reads the SAME `BLSTM_*`
//! config namespace as the spectral SAD driver (no `BLSTM_LID_*` keys).
//!
//! STRUCTURE (the crux, and why this is NOT the spectral SAD driver in disguise): the
//! LID `getSegmentation` inlines a COPY of the spectral setup (`:31-237`) that DIVERGES
//! from the base `BLSTMSpectralSegmenter::getSegmentation` in load-bearing ways -- the
//! Algo-5 lines govern where they differ (LEGACY SOURCE GOVERNS; diffed line by line):
//!
//! 1. **SAD via LTSV, not the BLSTM.** The result_vec is the LTSV spectral-variation
//!    score (`classifySequence` over the RAW periodogram, `:271-274`), NOT the NN
//!    posterior. The mel branch of that loop (`:252-269`) is COMMENTED OUT in the
//!    legacy, so the classify ALWAYS runs on `audio._Periodogram` (raw) with
//!    `freq_end` = the mel/DCT output width - 1. The BLSTM runs only PER SPEECH SEGMENT
//!    for language scoring (`:346-401`), never for the SAD result_vec.
//! 2. **LTSV window floors to 1, not 0** (`:157-158` `if (< 1) = 1`), UNLIKE the base
//!    `getLTSVParam` (`:303` `if (< 1) = 0`, disables). So LTSV SAD is ALWAYS active
//!    (even with `LTSVwindow 0`, `round(0) = 0 -> 1`). This is the STANDALONE
//!    `LtsvSegmenter`'s floor, not the spectral SAD's.
//! 3. **result_vec sized by `LTSV_window_shift`, not the ssr-division** (`:227-232`
//!    `ceil(vec_size / LTSV_window_shift)`), matching the LTSV driver -- NOT
//!    `getBLSTMParam`'s ssr-division (`:481-497`). The LTSV loop writes DECIMATED
//!    indices `result_vec[jj / LTSV_window_shift]` (NO interpolation backfill).
//! 4. **SAD `results2segmentation` timeStep = `_WindowShift`, offset 0.0** (`:302`) --
//!    the BLSTM-derived window shift used as the LTSV-SAD time step (a compression
//!    quirk: the result_vec is LTSV-decimated but stepped by `_WindowShift`, not
//!    `_WindowShift * LTSV_window_shift`). Reproduced verbatim.
//! 5. **scoring timeStep uses the SIGNAL pattern** (`:333-343`): the overlap branch
//!    keeps `timeStep = _WindowShift` (like `BlstmSignalSegmenter`), NOT
//!    `_SpectrumShift * ssr` (the base spectral driver's `:731`).
//! 6. **TDC params (`:162-173`) and `LTSV_freq_beg/end` (`:109-110`) are DEAD** --
//!    computed but never used (no pitch pass in the LID driver; the LTSV loop uses the
//!    post-mel `freq_beg/end`). Skipped here.
//! 7. **min_lag/max_lag derivation (`:170-172`) is SIMPLER than `getTDCParam`** (no
//!    `min_lag < 1 -> 1`, no `max_lag < min_lag` guard, no `_MinMaxLag` writeback) --
//!    moot since TDC is dead.
//!
//! The per-segment scoring loop (`:346-401`) drives [`BlstmNetwork::feed_forward_scoring`]
//! (the returning `feedForward` overload, `:363`): `segLID = colwise sum / rows`, the
//! `costLID`/`langID`/confusion accumulation, and the member writes (`:412-426`:
//! `targetLID(target) = -2.0`; `_LIDClassificationErrors = 100*(langID - targetLID)`;
//! `_IsLIDCorrect` 100/0/-1). `costLID` (`:320,376,405,410`) is computed but NEVER
//! stored (dead local) -- not reproduced. The `_CepstreCoefficients` writeback of the
//! normalized input (`:364-368`) is DEAD (segments read disjoint row blocks) -- skipped,
//! see IMPROVEMENTS.md.

use anyhow::{Result, anyhow};
use indexmap::IndexMap;
use ndarray::Array2;

use crate::audio::{Audio, compute_segment_periodogram_estimates, windowing_coefficients};
use crate::constants::random_gauss;
use crate::features::ltsv_tdc::ltsv_classify_sequence;
use crate::features::mel::MelFilterBank;
use crate::features::pipeline::{FeatureConfig, SpectralParams, build_input_sequence_parts};
use crate::features::stats::InputStatistics;
use crate::nn::activations::inv_logistic_fn;
use crate::nn::blstm::{BlstmConfig, BlstmNetwork};
use crate::tasks::sad::get_blstm_param;
use crate::tasks::segmentation::{SegClass, Segmentation};
use crate::tasks::segmentation_io::{ScoreReport, compute_errors};
use crate::tasks::segmenter::{
    DriverConfig, Segmenter, SegmenterConfig, get_targets, lid_to_segmentation,
    results_to_segmentation,
};

/// BLSTM spectral LID driver (Algo 5; `BLSTMSpectralLID.{h,cpp}`).
///
/// Holds the spectral/mel/LTSV config (via [`FeatureConfig`]), the SAD decision config
/// ([`SegmenterConfig`]/[`DriverConfig`]), and the LID BLSTM net. Stateful members
/// (`_SpectrumShift`, `_WindowShift`) self-quantize per call exactly like the spectral
/// driver's. The LID accessors (`lid_*`) surface the per-channel members the legacy
/// writes onto `seg._LID*[chan]`; the Rust `Segmentation` is single-channel (S6
/// restructure), so they live on the driver like the SAD driver's
/// `cumulative_error`/`nb_of_classif`.
#[derive(Clone)]
pub struct BlstmSpectralLid {
    driver_cfg: DriverConfig,
    seg_cfg: SegmenterConfig,
    feature_cfg: FeatureConfig,
    net: BlstmNetwork,
    /// Stateful `_SpectrumShift` (`:53`), self-quantizing per call.
    spectrum_shift_sec: f64,
    /// Stateful `_WindowShift` (`:187/:458`), mutated by [`get_blstm_param`] and reset
    /// to `0.0` when the noOverlap branch fired.
    window_shift_sec: f64,
    channels: usize,
    /// Port-side observation point (NO legacy counterpart, same rationale as
    /// `BlstmSpectralSegmenter::last_result_rows`): the PRE-convolution LTSV-SAD
    /// result_vec per channel, captured before `results_to_segmentation` mutates it.
    last_result_rows: Vec<Vec<f64>>,
    /// `seg._LIDCumulativeError[chan]` (`:414`): the NN cost summed over speech
    /// segments (`NNCost += getCost()` per `:370`).
    lid_cumulative_error: Vec<f64>,
    /// `seg._LIDNbOfClassif[chan]` (`:415`): `nbOfClassif += getNbOfClassif()` (`:371`).
    lid_nb_of_classif: Vec<i64>,
    /// `seg._LIDClassificationErrors[chan]` (`:416`): the `1 x classNb` row
    /// `100 * (langID - targetLID)`, with `targetLID[targetIndex] = -2.0` -- so the
    /// target entry carries the `>150` in-band sentinel (`100*langID + 200 >= 200`).
    lid_classification_errors: Vec<Vec<f64>>,
    /// `seg._LIDSegmentsConfusion[chan]` (`:418`): the `(classNb+2) x (classNb+2)`
    /// per-file confusion matrix (label header row/col 0, per-class 1..classNb, totals
    /// classNb+1).
    lid_segments_confusion: Vec<Array2<f64>>,
    /// `seg._IsLIDCorrect[chan]` (`:420-425`): 100 (target is argmax) / 0 (not) / -1
    /// (dead: `targetIndex` is clamped `>= 0`, so the -1 branch never fires).
    is_lid_correct: Vec<i32>,
}

impl BlstmSpectralLid {
    /// Port of the `BLSTMSpectralLID(ConfigFile&, ...)` ctor (`BLSTMSpectralLID.cpp:
    /// 10-14`): `BLSTMSpectralSegmenter::buildFromConf(conf, "BLSTM", ...)` (the
    /// `Segmenter` + spectral residue via [`SegmenterConfig`]/[`DriverConfig`]/
    /// [`FeatureConfig`]) + `BLSTMNeuralNetwork(conf, "BLSTM", false)`. `weights`
    /// mirrors the two ctor overloads (`Some(flat)` -> `setWeights`, `None` -> the
    /// no-weights ctor), matching [`BlstmSpectralSegmenter::from_legacy`].
    pub fn from_legacy(
        map: &IndexMap<String, String>,
        weights: Option<&[f64]>,
    ) -> Result<BlstmSpectralLid> {
        let seg_cfg = SegmenterConfig::from_config(map, "BLSTM")?;
        let driver_cfg = DriverConfig::from_config(map, "BLSTM")?;
        let feature_cfg = FeatureConfig::from_legacy(map, "BLSTM")?;

        let blstm_cfg = BlstmConfig::from_legacy(map, "BLSTM")?;
        let mut net = BlstmNetwork::from_config(blstm_cfg)?;
        if let Some(flat) = weights {
            net.set_weights(flat)?;
        }

        let spectrum_shift_sec = feature_cfg.shift_sec;
        let window_shift_sec = driver_cfg.window_shift_sec;

        Ok(BlstmSpectralLid {
            driver_cfg,
            seg_cfg,
            feature_cfg,
            net,
            spectrum_shift_sec,
            window_shift_sec,
            channels: 0,
            last_result_rows: Vec::new(),
            lid_cumulative_error: Vec::new(),
            lid_nb_of_classif: Vec::new(),
            lid_classification_errors: Vec::new(),
            lid_segments_confusion: Vec::new(),
            is_lid_correct: Vec::new(),
        })
    }

    /// `<prefix>_weightsFile` load delegate (see `BlstmSpectralSegmenter::
    /// load_weights_file`): applies a `weightsFile` key AFTER building via
    /// `from_legacy(map, None)`, matching the legacy in-ctor load.
    pub fn load_weights_file(&mut self, map: &IndexMap<String, String>) -> Result<()> {
        self.net.load_weights_file(map, "BLSTM")
    }

    /// The stateful `_WindowShift` member, POST the last `get_segmentation` call.
    pub fn window_shift_sec(&self) -> f64 {
        self.window_shift_sec
    }

    /// The stateful `_SpectrumShift` member, POST the last call.
    pub fn spectrum_shift_sec(&self) -> f64 {
        self.spectrum_shift_sec
    }

    /// The configured dump directory (see `TdcSegmenter::dump_dir`).
    pub fn dump_dir(&self) -> &str {
        &self.driver_cfg.dump_dir
    }

    /// SAD-side `seg._CumulativeError[chan]`: the LID driver NEVER writes it (the SAD is
    /// LTSV, NN-free), so it stays at the `Segmentation` ctor default 0.0 for every
    /// channel -- like `TdcSegmenter`/`LtsvSegmenter`.
    pub fn cumulative_error(&self) -> Vec<f64> {
        vec![0.0; self.channels]
    }

    /// SAD-side `seg._NbOfClassif[chan]`: never written (see [`Self::cumulative_error`]).
    pub fn nb_of_classif(&self) -> Vec<i64> {
        vec![0; self.channels]
    }

    /// PRE-convolution LTSV-SAD result rows (one per channel) from the last call.
    pub fn last_result_rows(&self) -> &[Vec<f64>] {
        &self.last_result_rows
    }

    /// `seg._LIDCumulativeError[chan]` (`:414`): the NN cost summed over speech segments.
    pub fn lid_cumulative_error(&self) -> &[f64] {
        &self.lid_cumulative_error
    }

    /// `seg._LIDNbOfClassif[chan]` (`:415`).
    pub fn lid_nb_of_classif(&self) -> &[i64] {
        &self.lid_nb_of_classif
    }

    /// `seg._LIDClassificationErrors[chan]` (`:416`): `100 * (langID - targetLID)`, one
    /// `classNb`-length row per channel. Carries the `>150` in-band sentinel at the
    /// target index.
    pub fn lid_classification_errors(&self) -> &[Vec<f64>] {
        &self.lid_classification_errors
    }

    /// `seg._LIDSegmentsConfusion[chan]` (`:418`): the per-file confusion matrix.
    pub fn lid_segments_confusion(&self) -> &[Array2<f64>] {
        &self.lid_segments_confusion
    }

    /// `seg._IsLIDCorrect[chan]` (`:420-425`): 100/0/-1 per channel.
    pub fn is_lid_correct(&self) -> &[i32] {
        &self.is_lid_correct
    }

    /// `Segmentation::compute_errors`, one call per channel (single-channel-container
    /// rationale, see `TdcSegmenter::score`).
    pub fn score(
        hyp: &mut [Segmentation],
        reference: Option<&[Segmentation]>,
        nb_words: i64,
    ) -> Vec<ScoreReport> {
        hyp.iter_mut()
            .enumerate()
            .map(|(chan, seg)| {
                let refc = reference.map(|r| &r[chan]);
                compute_errors(seg, refc, nb_words)
            })
            .collect()
    }
}

impl Segmenter for BlstmSpectralLid {
    /// Port of `BLSTMSpectralLID::getSegmentation` (`BLSTMSpectralLID.cpp:27-460`), the
    /// non-unit-test/non-plotting path (the `.mat`/PNG branches `:59-77,:207-219,
    /// :277-296,:303-305,:428-457` dropped: display/diagnostic-only, and the LID driver
    /// writes NO VRCTS in `getSegmentation`). `refs` is IGNORED: the LID scoring builds
    /// its own per-frame target internally from `audio.getRefLangIndex()` (`:363`), and
    /// the SAD is LTSV (NN-free) -- no `getTargets` call exists in this driver.
    fn get_segmentation(
        &mut self,
        audio: &mut Audio,
        seg_per_chan: &mut [Segmentation],
        _refs: Option<&[Segmentation]>,
    ) -> Result<()> {
        let rate = audio.sample_rate as f64;

        // Feature params (order clamp, pre-mel freq band, mel sizing) via the shared
        // derivation; `s.shift_frames` is not consumed (the LID uses the stateful
        // spectrum_shift below), but the band/order/bins are.
        let s = SpectralParams::derive(&self.feature_cfg, rate);

        // spectrum_shift: a LOCAL in the legacy (`:51`), but `_SpectrumShift` (`:53`)
        // persists the re-quantized seconds value. Self-quantize the stateful member.
        let spectrum_shift = f64::round(self.spectrum_shift_sec * rate) as usize;
        self.spectrum_shift_sec = spectrum_shift as f64 / rate;

        // preemph -> noise (`:78-89`). `apply_preemph` self-gates on ratio > 0.
        if self.feature_cfg.preemph_ratio > 0.0 {
            audio.apply_preemph(self.feature_cfg.preemph_ratio);
        }
        if self.feature_cfg.noise_seed > 0 {
            audio.apply_noise(self.feature_cfg.noise_ratio);
        }

        // Signal windowing coeffs (`:90`).
        let windowing_coeff = windowing_coefficients(
            &self.feature_cfg.win_type,
            false,
            s.buffer_size,
            self.feature_cfg.win_param,
        );

        // Mel bank + POST-mel freq band (`:97-146`). freq_beg/freq_end are RESET to
        // (0, mel/DCT width - 1) when mel is active (`:141-142`); else they stay the
        // pre-mel band (`:98-108`). The LTSV loop + the raw-path scoring block both read
        // these post-reset values (`:271-274`, `:360-361`).
        let (mel_bank, freq_beg, freq_end) = if self.feature_cfg.nb_bins > 0 {
            let bank = MelFilterBank::new(
                self.feature_cfg.min_mel,
                self.feature_cfg.max_mel,
                self.feature_cfg.nb_bins,
                s.min_freq,
                s.max_freq,
                rate,
                s.bins - 1,
                self.feature_cfg.is_log,
                self.feature_cfg.nb_dct,
                self.feature_cfg.ignore_first,
                self.feature_cfg.deltas_nb,
                self.feature_cfg.dd_nb,
            );
            let width = if self.feature_cfg.nb_dct > 0 {
                bank.nb_dct()
            } else {
                bank.nb_filters()
            };
            (Some(bank), 0usize, width - 1)
        } else {
            (None, s.freq_beg, s.freq_end)
        };

        // Temporal convolution of the periodogram (`:148-149`).
        let temporal_conv = windowing_coefficients(
            &self.feature_cfg.conv_type,
            true,
            2 * self.feature_cfg.conv_size as usize + 1,
            0.83333,
        );

        // LTSV window/shift (`:157-160`). FLOOR TO 1 (the divergence #2): even
        // `LTSVwindow 0 -> round(0) = 0 -> 1`, so LTSV SAD is always active.
        let mut ltsv_half_window =
            f64::round(self.feature_cfg.ltsv_window * rate / 2.0 / spectrum_shift as f64) as i64;
        if ltsv_half_window < 1 {
            ltsv_half_window = 1;
        }
        let ltsv_half_window = ltsv_half_window as usize;
        let mut ltsv_window_shift =
            f64::round(self.feature_cfg.ltsv_shift * rate / spectrum_shift as f64) as i64;
        if ltsv_window_shift < 1 {
            ltsv_window_shift = 1;
        }
        let ltsv_window_shift = ltsv_window_shift as usize;

        // TDC params (`:162-173`) are DEAD in the LID driver (no pitch pass) -- skipped.

        // getBLSTMParam window/shift derivation (`:175-190`). Mutates the stateful
        // `_WindowShift`. The returned real_vec_size (ssr-division) is IGNORED -- the
        // LID sizes result_vec by LTSV_window_shift below (divergence #3).
        let ssr = self.net.sub_sampling_ratio();
        let (window_size, window_shift, no_overlap, _real_vec_size_blstm) = get_blstm_param(
            self.driver_cfg.window_size_sec,
            &mut self.window_shift_sec,
            rate,
            spectrum_shift,
            ssr,
            &self.net.lstm_sub_sampling(),
            &self.net.output_sub_sampling(),
            audio.data.ncols(),
        );

        // vec_size / real_vec_size (`:221-232`): vec_size = ceil(frameCount /
        // spectrum_shift); real_vec_size = ceil(vec_size / LTSV_window_shift) (the LTSV
        // decimation, NOT the ssr-division).
        let frame_count = audio.data.ncols();
        let vec_size = if frame_count.is_multiple_of(spectrum_shift) {
            frame_count / spectrum_shift
        } else {
            frame_count / spectrum_shift + 1
        };
        let real_vec_size = if vec_size.is_multiple_of(ltsv_window_shift) {
            vec_size / ltsv_window_shift
        } else {
            vec_size / ltsv_window_shift + 1
        };

        // resetWeightsDerivatives ONCE (`:234`).
        self.net.reset_weights_derivatives();

        // scoring timeStep/timeOffset (`:333-343`, the SIGNAL pattern: overlap keeps
        // `_WindowShift`). window_shift_sec here is the POST-getBLSTMParam value.
        let mut time_step = self.window_shift_sec * ssr as f64;
        let mut time_offset = time_step / 2.0 - self.window_shift_sec / 2.0;
        if window_size > 0 {
            if no_overlap {
                time_step = self.window_shift_sec * ssr as f64;
                time_offset = time_step / 2.0 - self.window_shift_sec / 2.0;
            } else {
                time_step = self.window_shift_sec;
                time_offset = 0.0;
            }
        }
        let _ = time_offset; // SAD results2segmentation uses offset 0.0 (`:302`); scoring
        // needs only time_step (the ponderation modifier). timeOffset is computed for
        // parity but unused in the live path.

        // classNb = max(2, output_size) (`:309-310`).
        let class_nb = self.net.output_size().max(2);

        // setProcessingType((window > 0), !noOverlap) ONCE (`:344`).
        self.net.set_processing_type(window_size > 0, !no_overlap);

        let channels = audio.data.nrows();
        self.channels = channels;
        self.last_result_rows = Vec::with_capacity(channels);
        self.lid_cumulative_error = vec![0.0; channels];
        self.lid_nb_of_classif = vec![0; channels];
        self.lid_classification_errors = vec![Vec::new(); channels];
        self.lid_segments_confusion = vec![Array2::<f64>::zeros((0, 0)); channels];
        self.is_lid_correct = vec![0; channels];

        // Stable per-file locals for the borrow checker (self.net is borrowed mut in the
        // scoring loop; everything else must be a local read).
        let spectrum_shift_sec = self.spectrum_shift_sec;
        let window_shift_sec = self.window_shift_sec;
        let conv_coeff = self.driver_cfg.conv_coeff.clone();

        for (chan, seg) in seg_per_chan.iter_mut().enumerate().take(channels) {
            // Periodogram (`:242-244`): begin 0, end frameCount-1 (NOT the LTSV
            // driver's +1 quirk). Raw periodogram (mel is applied separately below).
            let perio = compute_segment_periodogram_estimates(
                audio,
                s.order,
                spectrum_shift,
                chan,
                self.feature_cfg.flag_dc_offset,
                windowing_coeff.as_deref(),
                temporal_conv.as_deref(),
                0,
                frame_count - 1,
            );

            // Mel/DCT for the per-segment scoring input (`:354-359`). The RAW periodogram
            // above feeds the LTSV SAD; the mel/DCT feeds the NN.
            let (fb, cep) = match &mel_bank {
                Some(bank) => {
                    let fb = bank.apply_filter_bank(&perio);
                    let cep = if self.feature_cfg.nb_dct > 0 {
                        Some(bank.apply_dct(&fb))
                    } else {
                        None
                    };
                    (Some(fb), cep)
                }
                None => (None, None),
            };

            // --- LTSV SAD (`:271-274`): DECIMATED write over the RAW periodogram, NO
            // interpolation backfill. `result_vec` is fresh per channel: the legacy
            // reuses one buffer across channels, but every index is overwritten each
            // channel (the loop covers 0..real_vec_size), so fresh-per-channel is
            // byte-identical (unlike the spectral overlap-accumulation reuse quirk).
            let mut result_vec = vec![0.0f64; real_vec_size];
            let mut jj = 0usize;
            while jj < vec_size {
                result_vec[jj / ltsv_window_shift] =
                    ltsv_classify_sequence(&perio, jj, freq_beg, freq_end, ltsv_half_window);
                jj += ltsv_window_shift;
            }

            // Capture the PRE-convolution LTSV row (port-side observation point).
            self.last_result_rows.push(result_vec.clone());

            // results2segmentation (`:300-302`): timeStep = _WindowShift, offset 0.0.
            results_to_segmentation(
                seg,
                window_shift_sec,
                0.0,
                &mut result_vec,
                SegClass::Speech,
                conv_coeff.as_deref(),
                &self.seg_cfg,
            );

            // --- LID scoring loop (`:346-401`) --------------------------------------
            let mut langid = vec![0.0f64; class_nb];
            let mut confusion = Array2::<f64>::zeros((class_nb + 2, class_nb + 2));
            for kk in 0..class_nb + 1 {
                confusion[[0, kk]] = kk as f64;
                confusion[[kk, 0]] = kk as f64;
            }
            let mut segments_count = 0i32;
            let mut nn_cost = 0.0f64;
            let mut nb_of_classif = 0i64;

            // targetIndex = clamp(audio.getRefLangIndex()) to [0, classNb) (`:321-323`).
            let mut target_index = audio.lang_index;
            if target_index >= class_nb as i32 {
                target_index = 0;
            }
            if target_index < 0 {
                target_index = 0;
            }
            let ti = target_index as usize;

            // Snapshot the SAD boundaries (the scoring loop does not mutate `seg`).
            let segs = seg.segments().to_vec();
            for i in 0..segs.len().saturating_sub(1) {
                if segs[i].ty != SegClass::Speech {
                    continue;
                }
                // rowBegin = ceil(begin / _SpectrumShift) (`:349-350`, truncate then
                // bump-if-not-exact); rowEnd = floor(next.begin / _SpectrumShift)
                // (`:351`).
                let ratio = segs[i].begin / spectrum_shift_sec;
                let mut row_begin = ratio as usize;
                if (row_begin as f64) != ratio {
                    row_begin += 1;
                }
                let row_end = (segs[i + 1].begin / spectrum_shift_sec) as usize;

                // `rowEnd - rowBegin + 1 >= ssr` (`:352`). `row_end >= row_begin` guards
                // the unsigned underflow (the legacy UB-slices there; unreachable with
                // real min-length segments -- see IMPROVEMENTS.md).
                if row_end < row_begin || row_end - row_begin + 1 < ssr {
                    continue;
                }
                let n_rows = row_end - row_begin + 1;

                // inputSeq block (`:354-361`): DCT block, else filterbank block, else the
                // raw log-periodogram band block.
                let mut input_seq = if let Some(cep) = cep.as_ref() {
                    let cols = cep.ncols();
                    let mut m = Array2::<f64>::zeros((n_rows, cols));
                    for r in 0..n_rows {
                        for c in 0..cols {
                            m[[r, c]] = cep[[row_begin + r, c]];
                        }
                    }
                    m
                } else if let Some(fb) = fb.as_ref() {
                    let cols = fb.ncols();
                    let mut m = Array2::<f64>::zeros((n_rows, cols));
                    for r in 0..n_rows {
                        for c in 0..cols {
                            m[[r, c]] = fb[[row_begin + r, c]];
                        }
                    }
                    m
                } else {
                    // raw path: ln(perio.block(rowBegin, freq_beg, n_rows, freq_end-freq_beg+1) + 1e-24).
                    let width = freq_end - freq_beg + 1;
                    let mut m = Array2::<f64>::zeros((n_rows, width));
                    for r in 0..n_rows {
                        for c in 0..width {
                            m[[r, c]] = (perio[[row_begin + r, freq_beg + c]] + 1e-24).ln();
                        }
                    }
                    m
                };

                // feedForward scoring (`:363`): modifier = timeStep / segment-duration.
                let modifier = time_step / (segs[i + 1].begin - segs[i].begin);
                let output_seq = self.net.feed_forward_scoring(
                    &mut input_seq,
                    window_size,
                    window_shift,
                    target_index as i64,
                    modifier,
                );

                // The `_CepstreCoefficients` writeback (`:364-368`) is DEAD -- skipped.

                nn_cost += self.net.cost; // NNCost += getCost() (`:370`)
                nb_of_classif += self.net.nb_of_classif; // (`:371`)

                // segLID = colwise sum / rows (`:372`).
                let out_rows = output_seq.nrows() as f64;
                let mut seg_lid = vec![0.0f64; class_nb];
                for c in 0..class_nb {
                    let mut acc = 0.0;
                    for r in 0..output_seq.nrows() {
                        acc += output_seq[[r, c]];
                    }
                    seg_lid[c] = acc / out_rows;
                }

                // costLID is a dead local (`:376`) -- not reproduced. argmax j (`:378`),
                // confusion accumulation (`:379-389`). targetIndex >= 0 always.
                let mut j = 0usize;
                let mut best = seg_lid[0];
                for (c, &v) in seg_lid.iter().enumerate().skip(1) {
                    if v > best {
                        best = v;
                        j = c;
                    }
                }
                let pos_target = ti + 1;
                if j == ti {
                    confusion[[pos_target, pos_target]] += 1.0;
                    confusion[[pos_target, class_nb + 1]] += 1.0;
                    confusion[[class_nb + 1, pos_target]] += 1.0;
                } else {
                    let pos_best = j + 1;
                    confusion[[pos_target, pos_best]] += 1.0;
                    confusion[[pos_target, class_nb + 1]] += 1.0;
                    confusion[[class_nb + 1, pos_best]] += 1.0;
                }

                for c in 0..class_nb {
                    langid[c] += seg_lid[c]; // langID += segLID (`:391`)
                }
                segments_count += 1;
            }

            // Post-loop reductions (`:403-411`).
            if segments_count > 0 {
                for v in langid.iter_mut() {
                    *v /= segments_count as f64;
                }
            } else {
                langid[0] = 1.0;
            }

            // targetLID (`:412-413`) + the member writes (`:414-426`).
            let mut target_lid = vec![0.0f64; class_nb];
            target_lid[ti] = -2.0;
            self.lid_cumulative_error[chan] = nn_cost;
            self.lid_nb_of_classif[chan] = nb_of_classif;
            let mut lid_errors = vec![0.0f64; class_nb];
            for c in 0..class_nb {
                lid_errors[c] = 100.0 * (langid[c] - target_lid[c]);
            }
            self.lid_classification_errors[chan] = lid_errors;
            self.lid_segments_confusion[chan] = confusion;
            let max_langid = langid.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            self.is_lid_correct[chan] = if langid[ti] == max_langid { 100 } else { 0 };
        }

        // seg.compute_errors() after the FULL channel loop (`:438`).
        for seg in seg_per_chan.iter_mut() {
            compute_errors(seg, None, -1);
        }

        // noOverlap `_WindowShift = 0.0` reset (`:458`).
        if no_overlap {
            self.window_shift_sec = 0.0;
        }

        Ok(())
    }

    /// `setWeights` delegates to the NN (base `BLSTMSpectralSegmenter::setWeights`).
    fn set_weights(&mut self, flat: &[f64]) -> Result<()> {
        self.net.set_weights(flat)
    }

    /// `getWeights` delegates to the NN.
    fn get_weights(&self) -> Vec<f64> {
        self.net.get_weights()
    }
}

// ===========================================================================
// Twin/Siamese LID driver (Algo 6; `TwinBLSTMSpectralLID.{h,cpp}`).
// ===========================================================================

fn twin_scalar(m: &IndexMap<String, String>, key: &str) -> Result<f64> {
    m.get(key)
        .ok_or_else(|| anyhow!("missing config key `{key}`"))?
        .trim()
        .parse::<f64>()
        .map_err(|e| anyhow!("`{key}`: cannot parse: {e}"))
}

fn twin_scalar_default(m: &IndexMap<String, String>, key: &str, default: f64) -> Result<f64> {
    match m.get(key) {
        None => Ok(default),
        Some(s) => s
            .trim()
            .parse::<f64>()
            .map_err(|e| anyhow!("`{key}`: cannot parse: {e}")),
    }
}

fn twin_i32_default(m: &IndexMap<String, String>, key: &str, default: i32) -> Result<i32> {
    match m.get(key) {
        None => Ok(default),
        Some(s) => s
            .trim()
            .parse::<i32>()
            .map_err(|e| anyhow!("`{key}`: cannot parse: {e}")),
    }
}

fn twin_bool_default(m: &IndexMap<String, String>, key: &str, default: bool) -> Result<bool> {
    match m.get(key) {
        None => Ok(default),
        Some(s) => s
            .trim()
            .parse::<bool>()
            .map_err(|e| anyhow!("`{key}`: cannot parse as bool: {e}")),
    }
}

/// Twin/Siamese spectral LID driver (Algo 6; `TwinBLSTMSpectralLID.{h,cpp}`).
///
/// UNLIKE Algo 5 (`BlstmSpectralLid`, whose SAD is LTSV-driven), Algo 6's SAD IS the
/// BLSTM spectral segmenter: `getSegmentation` (`:263-1421`) inlines the BASE
/// `BLSTMSpectralSegmenter` spectrum pipeline (`:349-619`, the same
/// `getBLSTMInputSequence` the base driver uses), runs the SAD BLSTM
/// (`_BLSTMNeuralNetwork`, `:715`) for the VAD result_vec, then a SECOND
/// (`_LIDBLSTMNeuralNetwork`) per-speech-segment for language scoring
/// (`:1243-1291`). The two nets read the `BLSTM` / `BLSTM_LID` config namespaces.
///
/// THIS DRIVER ports the wav-mode 0/2/3 paths + the hidden-state concat only. Modes
/// 4/5/6/7 (the `if ((_Mode==4)||(_Mode==5)||(_Mode==6))` LID-first block `:640-692`,
/// the `abs(_Mode)==7` CNN/noise block `:903-1193`) and the pitch second pass
/// (`:349-614` `TDC_window_size > 0`, a REFERENCE-based pre-forward warp that DIVERGES
/// from the base driver's post-forward pitch pass) BAIL typed, deferred to the next
/// task. The CNN member (`_LIDConvNeuralNetwork`, `:23`) is NOT ported (dead under the
/// port scope, only live in mode 7 -- see IMPROVEMENTS phase2 Conv exclusion).
///
/// MODE gating on the ported `:1194-1310` scoring branch (reached by modes 0/1/2/3):
/// - Mode 0/3 run the SAD net `feedForwardBackward` (`:713-717`), populating
///   `_OutputForward`/`_OutputBackward` (consumed by the concat).
/// - Mode 2 synthesizes `result_vec = LID_result_vec` (all-zero, `:719-724`) and CLEARS
///   the SAD net outputs (`:762-763`) -> the concat takes the empty fallback.
/// - Mode 1 fills `result_vec` with the constant 10.0 (`:726`) + clears outputs.
/// - The scoring loop iterates the REFERENCE for modes 2/3, the CLASSIFICATION for
///   modes 0/1 (`:1223-1239`). For modes != 0 the final classification is OVERWRITTEN by
///   `LID2Segmentation(segmentationLID)` (`:1294-1298`).
///
/// The `lid_*` accessors mirror [`BlstmSpectralLid`]'s; unlike Algo 5, the SAD-side
/// `cumulative_error`/`nb_of_classif` ARE written (`= NNCost`, then `*= LIDCostPonderation`
/// `:774,:1337`) since the SAD is a real NN here.
#[derive(Clone)]
pub struct TwinBlstmSpectralLid {
    driver_cfg: DriverConfig,
    seg_cfg: SegmenterConfig,
    feature_cfg: FeatureConfig,
    sad_net: BlstmNetwork,
    lid_net: BlstmNetwork,
    /// `_LIDWindowSize` (`:15-16`, clamped `< 0 -> 0`); constant.
    lid_window_size_sec: f64,
    /// `_LIDWindowShift` (`:17-18` seed, mutated by `getLIDBLSTMParam` `:102`, reset to
    /// `0.0` on LIDnoOverlap `:1420`); stateful.
    lid_window_shift_sec: f64,
    /// `_LIDDetectionThreshold` (`:25`, default -1.0). Unused on the 0/2/3 path.
    lid_detection_threshold: f64,
    /// `_LIDTrainingPruningThreshold` (`:26`, default -1.0). Gates `:1286` pruning.
    lid_training_pruning_threshold: f64,
    /// `_LIDDecisionThreshRising` (`:28`). Used by `LID2Segmentation` (`:1296`) + the
    /// `segmentationLID` midpoint seed (`:1242`).
    lid_decision_thresh_rising: f64,
    /// `_LIDDecisionThreshFalling` (`:29`). Midpoint seed only on the 0/2/3 path
    /// (`LID2Segmentation` ignores its `threshMin` arg -- see [`lid_to_segmentation`]).
    lid_decision_thresh_falling: f64,
    /// `_Mode` (`:33`, default 0). 0/1/2/3 ported; 4/5/6/7 bail.
    mode: i32,
    /// `_PostProcessMode` (`:35`, default 0). The 0/2/3 scoring branch always sums log
    /// posteriors via `segLID = colwise sum / rows` (the `_PostProcessMode` 1/2 variants
    /// live only in the mode 4/5/6/7 blocks) -- stored for the next task.
    post_process_mode: i32,
    /// `_MinNbOfFrames` (`:36`, default 0). Mode 7-only; stored for the next task.
    min_nb_of_frames: i32,
    /// `_NoiseMagnitude` (`:37`, default 0.0). Mode 7-only; stored for the next task.
    noise_magnitude: f64,
    /// `_DumpLIDInternals` (`:38`, default false). Mode 7-only; stored for the next task.
    dump_lid_internals: bool,
    /// Stateful `_SpectrumShift` (`:210`), self-quantizing per call.
    spectrum_shift_sec: f64,
    /// Stateful `_SpectrumShiftInFrames` (`:208-210`).
    spectrum_shift_in_frames: usize,
    /// Stateful SAD `_WindowShift` (`:453/:1419`).
    window_shift_sec: f64,
    /// Stateful `_LTSVWindowShift` (`:306`); dead for the ported configs (LTSVwindow 0).
    ltsv_shift_sec: f64,
    channels: usize,
    /// `seg._CumulativeError[chan]` (`:774`, then `*= LIDCostPonderation` `:1337`).
    cumulative_error: Vec<f64>,
    /// `seg._NbOfClassif[chan]` (`:775`).
    nb_of_classif: Vec<i64>,
    /// Port-side observation point (no legacy counterpart, same rationale as
    /// [`BlstmSpectralSegmenter::last_result_rows`]): the SAD `result_vec2` row per
    /// channel captured PRE-convolution (before `results_to_segmentation`).
    last_result_rows: Vec<Vec<f64>>,
    /// `seg._LIDCumulativeError[chan]` (`:1320`, then `*= audio._Weight` `:1336`).
    lid_cumulative_error: Vec<f64>,
    /// `seg._LIDNbOfClassif[chan]` (`:1321`).
    lid_nb_of_classif: Vec<i64>,
    /// `seg._LIDClassificationErrors[chan]` (`:1322`): `100 * (langID - targetLID)`.
    lid_classification_errors: Vec<Vec<f64>>,
    /// `seg._LIDSegmentsConfusion[chan]` (`:1324`).
    lid_segments_confusion: Vec<Array2<f64>>,
    /// `seg._IsLIDCorrect[chan]` (`:1325-1331`): 100/0 (`-1` dead, targetIndex clamped >= 0).
    is_lid_correct: Vec<i32>,
    /// Port-side observation point (no legacy counterpart): which branch of
    /// `getBLSTMLIDInputSequence` (`:139-168`) the concat guard (`:1221`) took per
    /// channel -- `0` not called (LID input <= feature width), `1` concat (SAD
    /// `_OutputForward` populated), `2` empty fallback (outputs cleared/empty). Pins the
    /// concat non-vacuity.
    concat_branch: Vec<i32>,
}

impl TwinBlstmSpectralLid {
    /// Port of the `TwinBLSTMSpectralLID(ConfigFile&, ...)` ctor (`:10-39`):
    /// `BLSTMSpectralSegmenter::buildFromConf(conf, "BLSTM", ...)` (the `Segmenter` +
    /// spectral residue) + `BLSTMNeuralNetwork(conf, "BLSTM", false)` (the SAD net) +
    /// `BLSTMNeuralNetwork(conf, "BLSTM_LID", false)` (the LID net) + the `BLSTM_LID_*`
    /// scalars. The `_LIDConvNeuralNetwork(conf, "CNN", false)` (`:23`) is NOT ported.
    /// `weights`/`lid_weights` mirror the two ctor overloads: `Some(flat)` ->
    /// `setWeights`/`setWeightsLID`, `None` -> the no-weights ctor.
    pub fn from_legacy(
        map: &IndexMap<String, String>,
        weights: Option<&[f64]>,
        lid_weights: Option<&[f64]>,
    ) -> Result<TwinBlstmSpectralLid> {
        let seg_cfg = SegmenterConfig::from_config(map, "BLSTM")?;
        let driver_cfg = DriverConfig::from_config(map, "BLSTM")?;
        let feature_cfg = FeatureConfig::from_legacy(map, "BLSTM")?;

        let sad_cfg = BlstmConfig::from_legacy(map, "BLSTM")?;
        let mut sad_net = BlstmNetwork::from_config(sad_cfg)?;
        if let Some(flat) = weights {
            sad_net.set_weights(flat)?;
        }

        let lid_cfg = BlstmConfig::from_legacy(map, "BLSTM_LID")?;
        let mut lid_net = BlstmNetwork::from_config(lid_cfg)?;
        if let Some(flat) = lid_weights {
            lid_net.set_weights(flat)?;
        }

        let mut lid_window_size_sec = twin_scalar(map, "BLSTM_LID_window")?;
        if lid_window_size_sec < 0.0 {
            lid_window_size_sec = 0.0;
        }
        let mut lid_window_shift_sec = twin_scalar(map, "BLSTM_LID_shift")?;
        if lid_window_shift_sec < 0.0 {
            lid_window_shift_sec = 0.0;
        }

        let lid_detection_threshold =
            twin_scalar_default(map, "BLSTM_LID_DetectionThreshold", -1.0)?;
        let lid_training_pruning_threshold =
            twin_scalar_default(map, "BLSTM_LID_TrainingPruningThreshold", -1.0)?;
        let lid_decision_thresh_rising = twin_scalar(map, "BLSTM_LID_decision_thresh_rising")?;
        let lid_decision_thresh_falling = twin_scalar(map, "BLSTM_LID_decision_thresh_falling")?;
        let mode = twin_i32_default(map, "BLSTM_LID_Mode", 0)?;
        let post_process_mode = twin_i32_default(map, "BLSTM_LID_PostProcessMode", 0)?;
        let min_nb_of_frames = twin_i32_default(map, "BLSTM_LID_MinNbOfFrames", 0)?;
        let noise_magnitude = twin_scalar_default(map, "BLSTM_LID_NoiseMagnitude", 0.0)?;
        let dump_lid_internals = twin_bool_default(map, "BLSTM_LID_DumpInternals", false)?;

        let spectrum_shift_sec = feature_cfg.shift_sec;
        let window_shift_sec = driver_cfg.window_shift_sec;
        let ltsv_shift_sec = feature_cfg.ltsv_shift;

        Ok(TwinBlstmSpectralLid {
            driver_cfg,
            seg_cfg,
            feature_cfg,
            sad_net,
            lid_net,
            lid_window_size_sec,
            lid_window_shift_sec,
            lid_detection_threshold,
            lid_training_pruning_threshold,
            lid_decision_thresh_rising,
            lid_decision_thresh_falling,
            mode,
            post_process_mode,
            min_nb_of_frames,
            noise_magnitude,
            dump_lid_internals,
            spectrum_shift_sec,
            spectrum_shift_in_frames: 0,
            window_shift_sec,
            ltsv_shift_sec,
            channels: 0,
            cumulative_error: Vec::new(),
            nb_of_classif: Vec::new(),
            last_result_rows: Vec::new(),
            lid_cumulative_error: Vec::new(),
            lid_nb_of_classif: Vec::new(),
            lid_classification_errors: Vec::new(),
            lid_segments_confusion: Vec::new(),
            is_lid_correct: Vec::new(),
            concat_branch: Vec::new(),
        })
    }

    /// `<prefix>_weightsFile` load delegates for BOTH nets (`BLSTM` / `BLSTM_LID`),
    /// matching the legacy in-ctor loads.
    pub fn load_weights_file(&mut self, map: &IndexMap<String, String>) -> Result<()> {
        self.sad_net.load_weights_file(map, "BLSTM")?;
        self.lid_net.load_weights_file(map, "BLSTM_LID")
    }

    // -- scalar/config accessors --------------------------------------------
    pub fn mode(&self) -> i32 {
        self.mode
    }
    pub fn post_process_mode(&self) -> i32 {
        self.post_process_mode
    }
    pub fn min_nb_of_frames(&self) -> i32 {
        self.min_nb_of_frames
    }
    pub fn noise_magnitude(&self) -> f64 {
        self.noise_magnitude
    }
    pub fn dump_lid_internals(&self) -> bool {
        self.dump_lid_internals
    }
    pub fn lid_detection_threshold(&self) -> f64 {
        self.lid_detection_threshold
    }
    pub fn lid_training_pruning_threshold(&self) -> f64 {
        self.lid_training_pruning_threshold
    }
    pub fn lid_decision_thresh_rising(&self) -> f64 {
        self.lid_decision_thresh_rising
    }
    pub fn lid_decision_thresh_falling(&self) -> f64 {
        self.lid_decision_thresh_falling
    }
    pub fn window_shift_sec(&self) -> f64 {
        self.window_shift_sec
    }
    pub fn spectrum_shift_sec(&self) -> f64 {
        self.spectrum_shift_sec
    }
    pub fn lid_window_shift_sec(&self) -> f64 {
        self.lid_window_shift_sec
    }
    pub fn dump_dir(&self) -> &str {
        &self.driver_cfg.dump_dir
    }

    // -- SAD-side outputs ----------------------------------------------------
    /// `seg._CumulativeError[chan] = NNCost` (`:774`) `*= LIDCostPonderation` (`:1337`).
    pub fn cumulative_error(&self) -> &[f64] {
        &self.cumulative_error
    }
    /// `seg._NbOfClassif[chan] = nbOfClassif` (`:775`).
    pub fn nb_of_classif(&self) -> &[i64] {
        &self.nb_of_classif
    }
    /// PRE-convolution SAD `result_vec2` rows (one per channel) from the last call.
    pub fn last_result_rows(&self) -> &[Vec<f64>] {
        &self.last_result_rows
    }

    // -- LID-side outputs (mirror Algo 5) ------------------------------------
    pub fn lid_cumulative_error(&self) -> &[f64] {
        &self.lid_cumulative_error
    }
    pub fn lid_nb_of_classif(&self) -> &[i64] {
        &self.lid_nb_of_classif
    }
    pub fn lid_classification_errors(&self) -> &[Vec<f64>] {
        &self.lid_classification_errors
    }
    pub fn lid_segments_confusion(&self) -> &[Array2<f64>] {
        &self.lid_segments_confusion
    }
    pub fn is_lid_correct(&self) -> &[i32] {
        &self.is_lid_correct
    }
    /// Per-channel concat branch taken (`0`/`1`/`2`, see the field doc) -- concat
    /// non-vacuity.
    pub fn concat_branch(&self) -> &[i32] {
        &self.concat_branch
    }

    // -- SAD net weight family (`BLSTMSpectralSegmenter` delegates) -----------
    pub fn is_back_prop_activated(&self) -> bool {
        self.sad_net.config().back_propagation_activated
    }
    pub fn get_weights_derivatives(&self) -> Array2<f64> {
        self.sad_net.get_weights_derivatives()
    }
    pub fn update_weights(&mut self, derivs: &Array2<f64>, cost: f64) {
        self.sad_net.update_weights(derivs, cost);
    }
    pub fn save_weights(
        &self,
        filename: &str,
        derivs: &Array2<f64>,
        stats: &InputStatistics,
    ) -> Result<()> {
        self.sad_net.save_weights(filename, derivs, stats)
    }
    pub fn input_statistics(&self) -> &InputStatistics {
        self.sad_net.input_statistics()
    }

    // -- LID net weight family (`:59-85` delegates) --------------------------
    /// `isBackPropActivatedLID` (`:59-61`).
    pub fn is_back_prop_activated_lid(&self) -> bool {
        self.lid_net.config().back_propagation_activated
    }
    /// `setWeightsLID` (`:63-65`).
    pub fn set_weights_lid(&mut self, flat: &[f64]) -> Result<()> {
        self.lid_net.set_weights(flat)
    }
    /// `getWeightsLID` (`:67-69`).
    pub fn get_weights_lid(&self) -> Vec<f64> {
        self.lid_net.get_weights()
    }
    /// `getWeightsDerivativesLID` (`:71-73`).
    pub fn get_weights_derivatives_lid(&self) -> Array2<f64> {
        self.lid_net.get_weights_derivatives()
    }
    /// `getInputStatisticsLID` (`:75-77`).
    pub fn get_input_statistics_lid(&self) -> &InputStatistics {
        self.lid_net.input_statistics()
    }
    /// `updateWeightsLID` (`:79-81`).
    pub fn update_weights_lid(&mut self, derivs: &Array2<f64>, cost: f64) {
        self.lid_net.update_weights(derivs, cost);
    }
    /// `saveWeightsLID` (`:83-85`): the LID net writes with the caller's filename
    /// PREFIXED by `LID_` (`_LIDBLSTMNeuralNetwork.saveWeights("LID_"+filename, ...)`).
    pub fn save_weights_lid(
        &self,
        filename: &str,
        derivs: &Array2<f64>,
        stats: &InputStatistics,
    ) -> Result<()> {
        self.lid_net
            .save_weights(&format!("LID_{filename}"), derivs, stats)
    }

    /// `Segmentation::compute_errors`, one call per channel (single-channel-container
    /// rationale, see [`BlstmSpectralLid::score`]).
    pub fn score(
        hyp: &mut [Segmentation],
        reference: Option<&[Segmentation]>,
        nb_words: i64,
    ) -> Vec<ScoreReport> {
        hyp.iter_mut()
            .enumerate()
            .map(|(chan, seg)| {
                let refc = reference.map(|r| &r[chan]);
                compute_errors(seg, refc, nb_words)
            })
            .collect()
    }

    /// Port of `getBLSTMLIDInputSequence` (`:139-168`): when the SAD net's
    /// `_OutputForward` is populated, hcat its z-normalized `[fwd | bwd]` hidden states
    /// onto `input_seq`, resampling by nearest index when the LSTM output row count
    /// differs from `input_seq`'s (`:150-163`); an EMPTY `_OutputForward` returns
    /// `input_seq` unchanged (`:164-166`). Returns `(augmented, branch)` with `branch`
    /// `1` = concat, `2` = empty fallback.
    fn get_blstm_lid_input_sequence(&self, input_seq: &Array2<f64>) -> (Array2<f64>, i32) {
        let fwd = &self.sad_net.output_forward;
        let bwd = &self.sad_net.output_backward;
        if fwd.nrows() == 0 {
            return (input_seq.clone(), 2);
        }
        // addedColumns = [fwd | bwd], then per-column z-normalize (mean/std over rows,
        // std with the +1e-32 floor), `:142-148`.
        let n = fwd.nrows();
        let fc = fwd.ncols();
        let bc = bwd.ncols();
        let added_cols = fc + bc;
        let mut added = Array2::<f64>::zeros((n, added_cols));
        for r in 0..n {
            for c in 0..fc {
                added[[r, c]] = fwd[[r, c]];
            }
            for c in 0..bc {
                added[[r, fc + c]] = bwd[[r, c]];
            }
        }
        for c in 0..added_cols {
            let mut sum = 0.0;
            for r in 0..n {
                sum += added[[r, c]];
            }
            let mean = sum / n as f64;
            let mut sq = 0.0;
            for r in 0..n {
                let d = added[[r, c]] - mean;
                added[[r, c]] = d;
                sq += d * d;
            }
            let std = ((sq + 1e-32) / n as f64).sqrt();
            for r in 0..n {
                added[[r, c]] /= std;
            }
        }

        let rows = input_seq.nrows();
        let cols = input_seq.ncols();
        let mut out = Array2::<f64>::zeros((rows, cols + added_cols));
        for r in 0..rows {
            for c in 0..cols {
                out[[r, c]] = input_seq[[r, c]];
            }
        }
        if n == rows {
            // Direct hcat (`:150-152`).
            for r in 0..rows {
                for c in 0..added_cols {
                    out[[r, cols + c]] = added[[r, c]];
                }
            }
        } else {
            // Nearest-index resample (`:154-162`): indexSpectrum = round((n-1)*frame/
            // (length-1)) via the `+0.5` truncation.
            let denom = (rows as f64 - 1.0).max(1.0);
            for frame in 0..rows {
                let idx = ((n as f64 - 1.0) * frame as f64 / denom + 0.5) as usize;
                for c in 0..added_cols {
                    out[[frame, cols + c]] = added[[idx, c]];
                }
            }
        }
        (out, 1)
    }

    /// Port of `getTargetsLID` (`:170-219`): the per-frame LID target from the
    /// (smoothed reference) classification `seg`. Multiclass (`output_size > 1`): a
    /// SPEECH row enforces `[target..]` with `target_index -> 1-target`; an OTHER row
    /// enforces `target_index -> target`, `col0 -> 1-target`; non-enforced rows are
    /// `-0.5`. Binary (`output_size <= 1`): SPEECH enforces `(target_index==1) ?
    /// 1-target : target`, OTHER is always `-0.5`. `target = isCostModified() ? 0.1/
    /// dur_seg : 0.0`. Enforcement uses `counter == step` (EQUALITY, `:182` -- NOT the
    /// scoring path's `>=`). NOT reached by modes 0/1/2/3 (only modes 4/5/6 call it,
    /// `:662`); ported for the next task, exercised by a direct unit test.
    #[allow(dead_code)]
    fn get_targets_lid(
        &self,
        seg: &Segmentation,
        time_step: f64,
        time_offset: f64,
        target_index: usize,
        n_rows: usize,
    ) -> Array2<f64> {
        let out_size = self.lid_net.output_size();
        let cols = out_size.max(1);
        let mut targets = Array2::<f64>::zeros((n_rows, cols));
        let segs = seg.segments();
        if segs.is_empty() {
            return targets;
        }
        let cost_modified = self.lid_net.is_cost_modified();
        let step = self.lid_net.target_enforcement_step();
        let mut counter = 0i32;
        let mut idx = 0usize;
        for ii in 0..n_rows {
            let t = ii as f64 * time_step + time_offset;
            while idx + 1 < segs.len() && segs[idx + 1].begin <= t {
                idx += 1;
            }
            let target = if cost_modified && idx + 1 < segs.len() {
                0.1 / (segs[idx + 1].begin - segs[idx].begin)
            } else {
                0.0
            };
            if out_size > 1 {
                if segs[idx].ty == SegClass::Speech {
                    if counter == step {
                        for c in 0..cols {
                            targets[[ii, c]] = target;
                        }
                        targets[[ii, target_index]] = 1.0 - target;
                        counter = 0;
                    } else {
                        for c in 0..cols {
                            targets[[ii, c]] = -0.5;
                        }
                        counter += 1;
                    }
                } else if counter == step {
                    targets[[ii, target_index]] = target;
                    targets[[ii, 0]] = 1.0 - target;
                    counter = 0;
                } else {
                    for c in 0..cols {
                        targets[[ii, c]] = -0.5;
                    }
                    counter += 1;
                }
            } else if segs[idx].ty == SegClass::Speech {
                if counter == step {
                    counter = 0;
                    targets[[ii, 0]] = if target_index == 1 {
                        1.0 - target
                    } else {
                        target
                    };
                } else {
                    counter += 1;
                    targets[[ii, 0]] = -0.5;
                }
            } else {
                targets[[ii, 0]] = -0.5;
            }
        }
        targets
    }

    /// Override the DumpLIDInternals output directory (`_DumpDir`). The legacy derives
    /// the `.mat` filename from `audio.getAudioFileName()` (`:906-913`); the port's
    /// `Audio` carries no source path, so the dump filename is `<dir>/chan<c>_lid_dump.mat`
    /// -- a cosmetic path deviation (IMPROVEMENTS'd); the VARIABLE names (`features_<n>`,
    /// `matNb`) and values are the faithful part.
    pub fn set_dump_dir(&mut self, dir: String) {
        self.driver_cfg.dump_dir = dir;
    }

    /// Port of the `abs(_Mode) == 7` branch (`:903-1193`), phSeq (File_Type 1) arm only.
    /// The SAD net is NEVER run (`result_vec` is synthesized constant 10.0 at `:726`);
    /// the LID net scores each `audio._ExternalFeatures[i]` block directly (`:980-1155`).
    /// The wav arm (`:922-963`) runs the CNN (`_LIDConvNeuralNetwork`, not ported) -> bail.
    fn get_segmentation_mode7(
        &mut self,
        audio: &mut Audio,
        seg_per_chan: &mut [Segmentation],
        _refs: Option<&[Segmentation]>,
    ) -> Result<()> {
        if audio.periodogram.is_none() {
            return Err(anyhow!(
                "TwinBlstmSpectralLid mode 7: wav arm runs the CNN (not ported); only File_Type 1 (phSeq) supported (legacy :922-963)"
            ));
        }
        let rate = audio.sample_rate as f64;
        let s = SpectralParams::derive(&self.feature_cfg, rate);
        if let Some(tdc) = s.tdc.as_ref()
            && tdc.half_window > 0
        {
            return Err(anyhow!(
                "TwinBlstmSpectralLid mode 7: pitch pass (TDCwindow > 0) not ported"
            ));
        }

        // Stateful spectrum-shift quantization (`initSpectralAnalysis` `:208-210`). The
        // `!hasReadWavFile()` override (`:209`) forces `_SpectrumShiftInFrames = 80` for
        // phSeq (File_Type 1) -- LOAD-BEARING: it drives every downstream window/shift
        // derivation (getLIDBLSTMParam window 25, not the 10 the raw `round(0.025*8000)=
        // 200` would give). `audio.periodogram.is_some()` is the port's `!hasReadWavFile()`.
        self.spectrum_shift_in_frames = f64::round(self.spectrum_shift_sec * rate) as usize;
        if audio.periodogram.is_some() {
            self.spectrum_shift_in_frames = 80;
        }
        self.spectrum_shift_sec = self.spectrum_shift_in_frames as f64 / rate;
        let ssif = self.spectrum_shift_in_frames;
        let spectrum_shift_sec = self.spectrum_shift_sec;

        // initSpectralAnalysis preemph/noise (harmless on the zeroed phSeq waveform).
        if self.feature_cfg.preemph_ratio > 0.0 {
            audio.apply_preemph(self.feature_cfg.preemph_ratio);
        }
        if self.feature_cfg.noise_seed > 0 {
            audio.apply_noise(self.feature_cfg.noise_ratio);
        }

        // getLTSVParam re-quantize (dead for LTSVwindow 0).
        let ltsv_ws = f64::round(self.feature_cfg.ltsv_shift * rate / ssif as f64) as i64;
        let ltsv_ws = if ltsv_ws < 1 { 1 } else { ltsv_ws };
        self.ltsv_shift_sec = (ltsv_ws * ssif as i64) as f64 / rate;

        // getBLSTMParam (SAD result_vec sizing) + getLIDBLSTMParam.
        let sad_ssr = self.sad_net.sub_sampling_ratio();
        let (window_size, _sad_window_shift, no_overlap, real_vec_size) = get_blstm_param(
            self.driver_cfg.window_size_sec,
            &mut self.window_shift_sec,
            rate,
            ssif,
            sad_ssr,
            &self.sad_net.lstm_sub_sampling(),
            &self.sad_net.output_sub_sampling(),
            audio.data.ncols(),
        );

        let lid_ssr = self.lid_net.sub_sampling_ratio();
        let ssifd = ssif as f64;
        let mut lid_window_size =
            f64::round(self.lid_window_size_sec * rate / 2.0 / ssifd) as usize;
        if lid_window_size != 0 && lid_window_size < lid_ssr {
            lid_window_size = lid_ssr;
        }
        let mut lid_window_shift = f64::round(self.lid_window_shift_sec * rate / ssifd) as i64;
        let mut lid_no_overlap = false;
        if lid_window_size != 0 && lid_window_shift < 1 {
            lid_no_overlap = true;
            let raw = f64::round(self.lid_window_size_sec * rate / ssifd) as usize;
            lid_window_size = (raw / lid_ssr) * lid_ssr;
            if lid_window_size < 10 * lid_ssr {
                lid_window_size = 10 * lid_ssr;
            }
        }
        if lid_window_size == 0 || lid_window_shift < 1 {
            lid_window_shift = 1;
        }
        self.lid_window_shift_sec = (lid_window_shift * ssif as i64) as f64 / rate;
        self.lid_net.reset_weights_derivatives();
        self.sad_net.reset_weights_derivatives();
        let lid_window_shift = lid_window_shift as usize;

        // SAD timeStep/offset (`:696-706`).
        let mut time_step = self.window_shift_sec * sad_ssr as f64;
        let mut time_offset = time_step / 2.0 - self.window_shift_sec / 2.0;
        if window_size > 0 {
            if no_overlap {
                time_step = self.window_shift_sec * sad_ssr as f64;
                time_offset = time_step / 2.0 - self.window_shift_sec / 2.0;
            } else {
                time_step = spectrum_shift_sec * sad_ssr as f64;
                time_offset = time_step / 2.0 - spectrum_shift_sec / 2.0;
            }
        }

        let class_nb = self.lid_net.output_size().max(2);
        let cost_modified = self.lid_net.is_cost_modified();

        let channels = audio.data.nrows();
        self.channels = channels;
        self.cumulative_error = vec![0.0; channels];
        self.nb_of_classif = vec![0; channels];
        self.last_result_rows = Vec::with_capacity(channels);
        self.lid_cumulative_error = vec![0.0; channels];
        self.lid_nb_of_classif = vec![0; channels];
        self.lid_classification_errors = vec![Vec::new(); channels];
        self.lid_segments_confusion = vec![Array2::<f64>::zeros((0, 0)); channels];
        self.is_lid_correct = vec![0; channels];
        self.concat_branch = vec![0; channels];

        let audio_weight = audio.weight;
        let post_process_mode = self.post_process_mode;
        let min_nb_of_frames = self.min_nb_of_frames;
        let noise_magnitude = self.noise_magnitude;
        let dump_lid_internals = self.dump_lid_internals;
        let dump_dir = self.driver_cfg.dump_dir.clone();
        let conv_coeff = self.driver_cfg.conv_coeff.clone();
        let mut lid_cost_ponderation = 1.0f64;
        // External features are shared across channels (phSeq has _ChannelsCount = 1).
        let external_features = audio.external_features.clone();

        for (chan, seg) in seg_per_chan.iter_mut().enumerate().take(channels) {
            // targetIndex = clamp(lang_index, [0, classNb)) (`:626-628`).
            let mut target_index = audio.lang_index;
            if target_index >= class_nb as i32 {
                target_index = 0;
            }
            if target_index < 0 {
                target_index = 0;
            }
            let ti = target_index as usize;

            // SAD else branch (`:725`): result_vec.setConstant(10.0); clear SAD outputs.
            let mut result_vec2 = vec![10.0f64; real_vec_size];
            self.sad_net.output_forward = Array2::<f64>::zeros((0, 0));
            self.sad_net.output_backward = Array2::<f64>::zeros((0, 0));
            self.last_result_rows.push(result_vec2.clone());
            // results2segmentation (`:773`, mode 7 runs it: _Mode != 4 && != 6).
            results_to_segmentation(
                seg,
                time_step,
                time_offset,
                &mut result_vec2,
                SegClass::Speech,
                conv_coeff.as_deref(),
                &self.seg_cfg,
            );
            self.cumulative_error[chan] = 0.0; // NNCost (SAD never run) (`:885`)
            self.nb_of_classif[chan] = 0;

            // --- abs(_Mode)==7 external-features loop (`:965-1179`) ------------
            self.lid_net
                .set_processing_type(lid_window_size > 0, !lid_no_overlap); // :971

            let mut dump = if dump_lid_internals && !dump_dir.is_empty() {
                let p = std::path::Path::new(&dump_dir).join(format!("chan{chan}_lid_dump.mat"));
                Some(crate::io::matfile::MatWriter::create(&p)?)
            } else {
                None
            };
            let mut count_dump = 0i64;

            let mut langid = vec![0.0f64; class_nb];
            let mut confusion = Array2::<f64>::zeros((class_nb + 2, class_nb + 2));
            for kk in 0..class_nb + 1 {
                confusion[[0, kk]] = kk as f64;
                confusion[[kk, 0]] = kk as f64;
            }
            let mut segments_count = 0i32;
            let mut number_of_frames = 0i32;
            let mut lid_nn_cost = 0.0f64;
            let mut lid_nb_classif = 0i64;

            for feat in &external_features {
                // _MinNbOfFrames + ssr guard (`:981`).
                if feat.nrows() < lid_ssr || (feat.nrows() as i32) < min_nb_of_frames {
                    continue;
                }
                // Gaussian noise (`:1022-1032`). PORT: random_init is wall-clock
                // (`:311`, non-deterministic/non-portable) -> the port fixes randinit = 0
                // (IMPROVEMENTS'd); table lookups are pure so the indexing is exact.
                let mut feat_noised = feat.clone();
                if noise_magnitude > 0.0 {
                    let cols = feat.ncols();
                    let randinit = 0usize;
                    for kk in 0..feat.nrows() {
                        for ll in 0..cols {
                            let m = random_gauss(kk * cols + ll + randinit) - 0.5;
                            feat_noised[[kk, ll]] += noise_magnitude * m;
                        }
                    }
                }

                // feedForward (`:1034`): modifier = 1/feat.rows().
                let modifier = 1.0 / feat.nrows() as f64;
                let output_seq = self.lid_net.feed_forward_scoring(
                    &mut feat_noised,
                    lid_window_size,
                    lid_window_shift,
                    target_index as i64,
                    modifier,
                );
                lid_nn_cost += self.lid_net.cost; // :1035
                lid_nb_classif += self.lid_net.nb_of_classif; // :1036

                // segLID accumulation per _PostProcessMode (`:1073-1108`). The legacy gate
                // `short_result_vec(kk,0) > 0.5` uses `Ones(rows, sadOutputSize)` (`:1039`)
                // -> ALWAYS true, so only `outputSeq(kk,1) >= 0` gates. The `_Mode < 0`
                // sub-branch (`:1040-1071`) is dead for _Mode = +7.
                let out_cols = output_seq.ncols();
                let mut seg_lid = vec![0.0f64; out_cols];
                match post_process_mode {
                    1 => {
                        for kk in 0..output_seq.nrows() {
                            if output_seq[[kk, 1]] >= 0.0 {
                                let mut entropy = 0.0;
                                let mut log_out = vec![0.0f64; out_cols];
                                for ll in 0..out_cols {
                                    log_out[ll] = f64::max(1e-24, output_seq[[kk, ll]]).ln();
                                    entropy -= output_seq[[kk, ll]] * log_out[ll];
                                }
                                entropy /= 2.0f64.ln();
                                if entropy < 1e-24 {
                                    entropy = 1e-24;
                                }
                                for ll in 0..out_cols {
                                    seg_lid[ll] +=
                                        f64::max(1e-24, output_seq[[kk, ll]] / entropy).ln();
                                }
                                number_of_frames += 1;
                            }
                        }
                    }
                    2 => {
                        for kk in 0..output_seq.nrows() {
                            if output_seq[[kk, 1]] >= 0.0 {
                                let mut j = 0usize;
                                let mut best = output_seq[[kk, 0]];
                                for c in 1..out_cols {
                                    if output_seq[[kk, c]] > best {
                                        best = output_seq[[kk, c]];
                                        j = c;
                                    }
                                }
                                seg_lid[j] += 1.0;
                                number_of_frames += 1;
                            }
                        }
                    }
                    _ => {
                        for kk in 0..output_seq.nrows() {
                            if output_seq[[kk, 1]] >= 0.0 {
                                for ll in 0..out_cols {
                                    seg_lid[ll] += f64::max(1e-24, output_seq[[kk, ll]]).ln();
                                }
                                number_of_frames += 1;
                            }
                        }
                    }
                }

                // argmax j (`:1113-1114`), confusion (`:1121-1130`).
                let mut j = 0usize;
                let mut best = seg_lid[0];
                for (c, &v) in seg_lid.iter().enumerate().skip(1) {
                    if v > best {
                        best = v;
                        j = c;
                    }
                }
                let pos_target = ti + 1;
                if j == ti {
                    confusion[[pos_target, pos_target]] += 1.0;
                    confusion[[pos_target, class_nb + 1]] += 1.0;
                    confusion[[class_nb + 1, pos_target]] += 1.0;
                } else {
                    let pos_best = j + 1;
                    confusion[[pos_target, pos_best]] += 1.0;
                    confusion[[pos_target, class_nb + 1]] += 1.0;
                    confusion[[class_nb + 1, pos_best]] += 1.0;
                }

                // langID (`:1131-1135`).
                let rows = output_seq.nrows() as f64;
                if cost_modified {
                    for (c, &v) in seg_lid.iter().enumerate() {
                        langid[c] += v / rows;
                    }
                } else {
                    for (c, &v) in seg_lid.iter().enumerate() {
                        langid[c] += v;
                    }
                }

                // DumpLIDInternals features_<n> = [oF | oB] (`:1136-1144`).
                if let Some(w) = dump.as_mut() {
                    let of = &self.lid_net.output_forward;
                    let ob = &self.lid_net.output_backward;
                    let n = of.nrows();
                    let (fc, bc) = (of.ncols(), ob.ncols());
                    let mut colmaj = Vec::with_capacity(n * (fc + bc));
                    for c in 0..fc {
                        for r in 0..n {
                            colmaj.push(of[[r, c]]);
                        }
                    }
                    for c in 0..bc {
                        for r in 0..n {
                            colmaj.push(ob[[r, c]]);
                        }
                    }
                    w.write_matrix(&format!("features_{count_dump}"), n, fc + bc, &colmaj)?;
                    count_dump += 1;
                }

                segments_count += 1;
            }

            if let Some(mut w) = dump.take() {
                w.write_scalar("matNb", count_dump as f64)?; // :1157-1158
                w.finish()?;
            }

            // langID normalization (`:1163-1179`).
            if segments_count > 0 && langid.iter().sum::<f64>() != 0.0 {
                if cost_modified {
                    for v in langid.iter_mut() {
                        *v /= segments_count as f64;
                    }
                } else {
                    for v in langid.iter_mut() {
                        *v /= number_of_frames as f64;
                    }
                }
                if post_process_mode != 2 {
                    for v in langid.iter_mut() {
                        *v = v.exp();
                    }
                }
                let adim: f64 = langid.iter().sum();
                for v in langid.iter_mut() {
                    *v /= adim;
                }
            } else {
                langid[0] = 1.0;
            }

            // Member writes (`:1318-1337`).
            let mut target_lid = vec![0.0f64; class_nb];
            target_lid[ti] = -2.0;
            self.lid_cumulative_error[chan] = lid_nn_cost;
            self.lid_nb_of_classif[chan] = lid_nb_classif;
            let mut lid_errors = vec![0.0f64; class_nb];
            for c in 0..class_nb {
                lid_errors[c] = 100.0 * (langid[c] - target_lid[c]);
            }
            self.lid_classification_errors[chan] = lid_errors;
            self.lid_segments_confusion[chan] = confusion;
            let max_langid = langid.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            self.is_lid_correct[chan] = if langid[ti] == max_langid { 100 } else { 0 };

            lid_cost_ponderation = self.lid_net.get_cost_ponderation(target_index as i64); // :1334
            self.lid_net.ponderate_weights_derivatives(audio_weight); // :1335
            self.lid_cumulative_error[chan] *= audio_weight; // :1336
            self.cumulative_error[chan] *= lid_cost_ponderation; // :1337
        }

        // Epilogue (`:1393-1420`).
        self.sad_net
            .ponderate_weights_derivatives(lid_cost_ponderation);
        for seg in seg_per_chan.iter_mut() {
            compute_errors(seg, None, -1);
        }
        if no_overlap {
            self.window_shift_sec = 0.0;
        }
        if lid_no_overlap {
            self.lid_window_shift_sec = 0.0;
        }
        Ok(())
    }
}

impl Segmenter for TwinBlstmSpectralLid {
    /// Port of `TwinBLSTMSpectralLID::getSegmentation` (`:263-1421`), the wav-mode
    /// 0/1/2/3 non-unit-test/non-plotting paths (the `.mat`/PNG/log branches dropped).
    /// Modes 4/5/6/7 and the pitch second pass BAIL typed (see the struct doc). `refs`
    /// maps to `seg._Reference`: required for modes 2/3 (the scoring loop iterates it)
    /// and, for any mode with a reference, builds the SAD net's `targetSeq` (`:707-711`).
    fn get_segmentation(
        &mut self,
        audio: &mut Audio,
        seg_per_chan: &mut [Segmentation],
        refs: Option<&[Segmentation]>,
    ) -> Result<()> {
        // Mode 7 (the `abs(_Mode) == 7` branch, `:903-1193`): the phSeq external-features
        // LID loop. Only the File_Type-1 (phSeq) arm is ported -- the wav arm (`:922-963`)
        // runs the CNN (`:927`, NOT ported) so it bails. Handled in its own method.
        if self.mode == 7 {
            return self.get_segmentation_mode7(audio, seg_per_chan, refs);
        }
        if !(self.mode == 0 || self.mode == 1 || self.mode == 2 || self.mode == 3) {
            return Err(anyhow!(
                "TwinBlstmSpectralLid: mode {} not ported (modes 4/5/6 deferred; legacy :640-902)",
                self.mode
            ));
        }
        let rate = audio.sample_rate as f64;

        // initSpectralAnalysis param derivation (`:281`) via the base spectral machinery.
        let s = SpectralParams::derive(&self.feature_cfg, rate);

        // The pitch second pass (`:349-614`, TDC_window_size > 0) is a REFERENCE-based
        // pre-forward warp that DIVERGES from the base driver's post-forward pitch pass --
        // deferred (bail typed). All ported configs use TDCwindow 0.
        if let Some(tdc) = s.tdc.as_ref()
            && tdc.half_window > 0
        {
            return Err(anyhow!(
                "TwinBlstmSpectralLid: pitch second pass (TDCwindow > 0) not ported (legacy :349-614)"
            ));
        }

        // Stateful spectrum-shift quantization (`:208-210`).
        self.spectrum_shift_in_frames = f64::round(self.spectrum_shift_sec * rate) as usize;
        self.spectrum_shift_sec = self.spectrum_shift_in_frames as f64 / rate;
        let spectrum_shift_in_frames = self.spectrum_shift_in_frames;
        let spectrum_shift_sec = self.spectrum_shift_sec;

        // preemph -> noise (`initSpectralAnalysis` :216-227).
        if self.feature_cfg.preemph_ratio > 0.0 {
            audio.apply_preemph(self.feature_cfg.preemph_ratio);
        }
        if self.feature_cfg.noise_seed > 0 {
            audio.apply_noise(self.feature_cfg.noise_ratio);
        }

        // getLTSVParam re-quantize `_LTSVWindowShift` (`:288/:302-306`; dead for LTSVwindow 0).
        let ltsv_ws =
            f64::round(self.feature_cfg.ltsv_shift * rate / spectrum_shift_in_frames as f64) as i64;
        let ltsv_ws = if ltsv_ws < 1 { 1 } else { ltsv_ws };
        self.ltsv_shift_sec = (ltsv_ws * spectrum_shift_in_frames as i64) as f64 / rate;

        // getBLSTMParam (`:300` -> `:439-500`): SAD window/shift + result-vec sizing;
        // mutates self.window_shift_sec.
        let sad_ssr = self.sad_net.sub_sampling_ratio();
        let (window_size, window_shift, no_overlap, real_vec_size) = get_blstm_param(
            self.driver_cfg.window_size_sec,
            &mut self.window_shift_sec,
            rate,
            spectrum_shift_in_frames,
            sad_ssr,
            &self.sad_net.lstm_sub_sampling(),
            &self.sad_net.output_sub_sampling(),
            audio.data.ncols(),
        );

        // getLIDBLSTMParam (`:306` -> `:87-137`): LID window/shift + LID_result_vec
        // sizing; mutates self.lid_window_shift_sec + resets the LID net derivs.
        let lid_ssr = self.lid_net.sub_sampling_ratio();
        let ssif = spectrum_shift_in_frames as f64;
        let mut lid_window_size = f64::round(self.lid_window_size_sec * rate / 2.0 / ssif) as usize;
        if lid_window_size != 0 && lid_window_size < lid_ssr {
            lid_window_size = lid_ssr;
        }
        let mut lid_window_shift = f64::round(self.lid_window_shift_sec * rate / ssif) as i64;
        let mut lid_no_overlap = false;
        if lid_window_size != 0 && lid_window_shift < 1 {
            lid_no_overlap = true;
            let raw = f64::round(self.lid_window_size_sec * rate / ssif) as usize;
            lid_window_size = (raw / lid_ssr) * lid_ssr;
            if lid_window_size < 10 * lid_ssr {
                lid_window_size = 10 * lid_ssr;
            }
        }
        if lid_window_size == 0 || lid_window_shift < 1 {
            lid_window_shift = 1;
        }
        self.lid_window_shift_sec =
            (lid_window_shift * spectrum_shift_in_frames as i64) as f64 / rate;
        self.lid_net.reset_weights_derivatives(); // :110
        let frame_count = audio.data.ncols();
        let vec_size = if frame_count.is_multiple_of(spectrum_shift_in_frames) {
            frame_count / spectrum_shift_in_frames
        } else {
            frame_count / spectrum_shift_in_frames + 1
        };
        let mut lid_real_vec_size = vec_size;
        if lid_ssr > 1 {
            for r in self.lid_net.lstm_sub_sampling() {
                lid_real_vec_size /= r;
            }
            for r in self.lid_net.output_sub_sampling() {
                lid_real_vec_size /= r;
            }
        }
        let lid_window_shift = lid_window_shift as usize;

        // resetWeightsDerivatives on the SAD net (`:234` via getBLSTMParam already reset,
        // but the base getBLSTMParam :473 resets -- our get_blstm_param does NOT, so reset
        // here to match).
        self.sad_net.reset_weights_derivatives();

        // SAD timeStep/timeOffset (`:696-706`): overlap OVERRIDES with _SpectrumShift.
        let mut time_step = self.window_shift_sec * sad_ssr as f64;
        let mut time_offset = time_step / 2.0 - self.window_shift_sec / 2.0;
        if window_size > 0 {
            if no_overlap {
                time_step = self.window_shift_sec * sad_ssr as f64;
                time_offset = time_step / 2.0 - self.window_shift_sec / 2.0;
            } else {
                time_step = spectrum_shift_sec * sad_ssr as f64;
                time_offset = time_step / 2.0 - spectrum_shift_sec / 2.0;
            }
        }

        let class_nb = self.lid_net.output_size().max(2);

        let temporal_conv = windowing_coefficients(
            &self.feature_cfg.conv_type,
            true,
            2 * self.feature_cfg.conv_size as usize + 1,
            0.83333,
        );

        let channels = audio.data.nrows();
        self.channels = channels;
        self.cumulative_error = vec![0.0; channels];
        self.nb_of_classif = vec![0; channels];
        self.last_result_rows = Vec::with_capacity(channels);
        self.lid_cumulative_error = vec![0.0; channels];
        self.lid_nb_of_classif = vec![0; channels];
        self.lid_classification_errors = vec![Vec::new(); channels];
        self.lid_segments_confusion = vec![Array2::<f64>::zeros((0, 0)); channels];
        self.is_lid_correct = vec![0; channels];
        self.concat_branch = vec![0; channels];

        let audio_weight = audio.weight;
        let mut lid_cost_ponderation = 1.0f64;

        for (chan, seg) in seg_per_chan.iter_mut().enumerate().take(channels) {
            // SAD input (`:359-619`): the BASE spectral pipeline (periodogram -> mel/DCT
            // -> LTSV) -- identical to `BlstmSpectralSegmenter` when the pitch pass is off.
            let (mut input_seq, _perio, _ltsv1) = build_input_sequence_parts(
                audio,
                &self.feature_cfg,
                &s,
                chan,
                temporal_conv.as_deref(),
            );

            // targetIndex = clamp(audio.getRefLangIndex()) to [0, classNb) (`:626-628`).
            let mut target_index = audio.lang_index;
            if target_index >= class_nb as i32 {
                target_index = 0;
            }
            if target_index < 0 {
                target_index = 0;
            }
            let ti = target_index as usize;

            // SAD targetSeq (`:707-711`): built when a reference exists; :710 zeroes it
            // for a BINARY LID net when targetIndex != 1.
            let has_ref = matches!(refs, Some(rs) if !rs[chan].segments().is_empty());
            let target = if has_ref {
                let rs = refs.unwrap();
                let col = get_targets(
                    seg,
                    &rs[chan],
                    time_step,
                    time_offset,
                    self.driver_cfg.back_prop_wer,
                    SegClass::Speech,
                    real_vec_size,
                );
                let mut t = Array2::from_shape_vec((real_vec_size, 1), col).unwrap();
                if self.lid_net.output_size() == 1 && target_index != 1 {
                    t.fill(0.0);
                }
                t
            } else {
                Array2::<f64>::zeros((0, 0))
            };

            // SAD result_vec: modes 0/3 run the SAD net; else (modes 1/2) synthesize.
            let mut result_vec = Array2::<f64>::zeros((real_vec_size, 1));
            let mut nn_cost = 0.0f64;
            let mut nb_of_classif = 0i64;
            if self.mode == 0 || self.mode == 3 {
                // `:713-717` SAD net feedForwardBackward (populates _OutputForward/Backward).
                self.sad_net
                    .set_processing_type(window_size > 0, !no_overlap);
                self.sad_net.feed_forward_backward(
                    &mut input_seq,
                    window_size,
                    window_shift,
                    &mut result_vec,
                    &target,
                );
                nn_cost = self.sad_net.cost;
                nb_of_classif = self.sad_net.nb_of_classif;
            } else {
                // `:718-764` else branch. Mode 2: result_vec = LID_result_vec (all-zero,
                // `:719-724`). Mode 1: result_vec.setConstant(10.0) (`:726`). Both CLEAR
                // the SAD net outputs (`:762-763`).
                if self.mode == 2 {
                    // LID_result_vec is Zero(lid_real_vec_size, lid_output_size); col
                    // (targetIndex) or the whole matrix (cols==1) -> a zero column.
                    result_vec = Array2::<f64>::zeros((lid_real_vec_size, 1));
                } else {
                    // mode 1
                    result_vec.fill(10.0);
                }
                self.sad_net.output_forward = Array2::<f64>::zeros((0, 0));
                self.sad_net.output_backward = Array2::<f64>::zeros((0, 0));
            }

            // result_vec2 = transpose -> ROW vector (`:766`).
            let mut result_vec2: Vec<f64> = result_vec.column(0).to_vec();
            self.last_result_rows.push(result_vec2.clone());

            // results2segmentation (`:773`, for modes != 4,6 -> yes for 0/1/2/3).
            results_to_segmentation(
                seg,
                time_step,
                time_offset,
                &mut result_vec2,
                SegClass::Speech,
                self.driver_cfg.conv_coeff.as_deref(),
                &self.seg_cfg,
            );
            // :774-775.
            self.cumulative_error[chan] = nn_cost;
            self.nb_of_classif[chan] = nb_of_classif;

            // ---- LID scoring branch (`:1194-1310`) --------------------------------
            self.lid_net
                .set_processing_type(lid_window_size > 0, !lid_no_overlap); // :1206

            // Concat (`:1221`): LID input size > feature width -> augment.
            self.concat_branch[chan] = if self.lid_net.input_size() > input_seq.ncols() {
                let (aug, branch) = self.get_blstm_lid_input_sequence(&input_seq);
                input_seq = aug;
                branch
            } else {
                0
            };

            // Scoring iterator: REFERENCE for modes 2/3, CLASSIFICATION for 0/1 (`:1223-1239`).
            let scoring_segs: Vec<crate::tasks::segmentation::Segment> =
                if (self.mode == 2 || self.mode == 3) && has_ref {
                    refs.unwrap()[chan].segments().to_vec()
                } else {
                    seg.segments().to_vec()
                };

            // totalSpeechDuration over the scoring iterator (`:1229-1234`).
            let mut total_speech_duration = 0.0f64;
            for i in 0..scoring_segs.len().saturating_sub(1) {
                if scoring_segs[i].ty == SegClass::Speech {
                    total_speech_duration += scoring_segs[i + 1].begin - scoring_segs[i].begin;
                }
            }

            // interestSegs (`:1240`) + the TrainingPruning gate (`:1286`, mutating
            // `itInterest->_Type`) are NOT reproduced: `interestSegs` is consumed ONLY by
            // the VRCTS dump (skipped) + the mode!=0 replacement (`:1297`, itself
            // dump-only), so both the copy and the pruning are observably dead in this
            // port. The gate is config-gated on `_LIDTrainingPruningThreshold > 0`
            // (default -1.0) besides -- see IMPROVEMENTS.md.

            // segmentationLID (`:1242`): midpoint seed.
            let mut segmentation_lid =
                vec![
                    (self.lid_decision_thresh_rising + self.lid_decision_thresh_falling) / 2.0;
                    lid_real_vec_size
                ];

            let mut langid = vec![0.0f64; class_nb];
            let mut confusion = Array2::<f64>::zeros((class_nb + 2, class_nb + 2));
            for kk in 0..class_nb + 1 {
                confusion[[0, kk]] = kk as f64;
                confusion[[kk, 0]] = kk as f64;
            }
            let mut segments_count = 0i32;
            let mut lid_nn_cost = 0.0f64;
            let mut lid_nb_classif = 0i64;

            for i in 0..scoring_segs.len().saturating_sub(1) {
                if scoring_segs[i].ty != SegClass::Speech {
                    continue;
                }
                // rowBegin = ceil(begin / _SpectrumShift), rowEnd = floor(next / _SpectrumShift)
                // (`:1245-1247`). NO offset.
                let ratio = scoring_segs[i].begin / spectrum_shift_sec;
                let mut row_begin = ratio as usize;
                if (row_begin as f64) != ratio {
                    row_begin += 1;
                }
                let row_end = (scoring_segs[i + 1].begin / spectrum_shift_sec) as usize;
                // `:1248` guard (unsigned; the row_end >= row_begin guard prevents underflow).
                if row_end < row_begin || row_end - row_begin + 1 < lid_ssr {
                    continue;
                }
                let n_rows = row_end - row_begin + 1;
                if row_begin + n_rows > input_seq.nrows() {
                    continue; // block would read past the input (legacy UB-slices; guarded).
                }

                // inputSeqLID = inputSeq.block(rowBegin, 0, n_rows, cols) (`:1249`).
                let cols = input_seq.ncols();
                let mut input_seq_lid = Array2::<f64>::zeros((n_rows, cols));
                for r in 0..n_rows {
                    for c in 0..cols {
                        input_seq_lid[[r, c]] = input_seq[[row_begin + r, c]];
                    }
                }

                // Scoring feedForward (`:1250`): modifier = timeStep / segment duration.
                let modifier = time_step / (scoring_segs[i + 1].begin - scoring_segs[i].begin);
                let output_seq = self.lid_net.feed_forward_scoring(
                    &mut input_seq_lid,
                    lid_window_size,
                    lid_window_shift,
                    target_index as i64,
                    modifier,
                );

                // segmentationLID block = InvLogistic(outputSeq.col(1)) (`:1251`).
                let start = row_begin / lid_ssr;
                for r in 0..output_seq.nrows() {
                    if start + r < segmentation_lid.len() && output_seq.ncols() > 1 {
                        segmentation_lid[start + r] = inv_logistic_fn(output_seq[[r, 1]]);
                    }
                }

                lid_nn_cost += self.lid_net.cost; // :1252
                lid_nb_classif += self.lid_net.nb_of_classif; // :1253

                // segLID = colwise sum / rows (`:1254`).
                let out_rows = output_seq.nrows() as f64;
                let mut seg_lid = vec![0.0f64; class_nb];
                for (c, sv) in seg_lid.iter_mut().enumerate() {
                    let mut acc = 0.0;
                    for r in 0..output_seq.nrows() {
                        acc += output_seq[[r, c]];
                    }
                    *sv = acc / out_rows;
                }

                // argmax j (`:1257-1258`), confusion (`:1265-1274`).
                let mut j = 0usize;
                let mut best = seg_lid[0];
                for (c, &v) in seg_lid.iter().enumerate().skip(1) {
                    if v > best {
                        best = v;
                        j = c;
                    }
                }
                let pos_target = ti + 1;
                if j == ti {
                    confusion[[pos_target, pos_target]] += 1.0;
                    confusion[[pos_target, class_nb + 1]] += 1.0;
                    confusion[[class_nb + 1, pos_target]] += 1.0;
                } else {
                    let pos_best = j + 1;
                    confusion[[pos_target, pos_best]] += 1.0;
                    confusion[[pos_target, class_nb + 1]] += 1.0;
                    confusion[[class_nb + 1, pos_best]] += 1.0;
                }

                // langID accumulation (`:1275-1279`): cost-modified += segLID; else
                // += dur*segLID.
                let dur = scoring_segs[i + 1].begin - scoring_segs[i].begin;
                if self.lid_net.is_cost_modified() {
                    for (c, &v) in seg_lid.iter().enumerate() {
                        langid[c] += v;
                    }
                } else {
                    for (c, &v) in seg_lid.iter().enumerate() {
                        langid[c] += dur * v;
                    }
                }
                segments_count += 1;
            }

            // segmentationLID /= 56 (`:1292`).
            for v in segmentation_lid.iter_mut() {
                *v /= 56.0;
            }

            if segments_count > 0 {
                // mode != 0 -> clearClassification + LID2Segmentation (`:1294-1298`).
                if self.mode != 0 {
                    seg.clear_hypothesis();
                    lid_to_segmentation(
                        seg,
                        &segmentation_lid,
                        SegClass::Speech,
                        time_offset,
                        time_step,
                        self.lid_decision_thresh_rising,
                    );
                }
                // langID normalization (`:1299-1303`).
                if self.lid_net.is_cost_modified() {
                    for v in langid.iter_mut() {
                        *v /= segments_count as f64;
                    }
                } else {
                    for v in langid.iter_mut() {
                        *v /= total_speech_duration;
                    }
                }
            } else {
                langid[0] = 1.0; // :1305
            }

            // Member writes (`:1318-1337`).
            let mut target_lid = vec![0.0f64; class_nb];
            target_lid[ti] = -2.0; // :1319 (targetIndex >= 0 always)
            self.lid_cumulative_error[chan] = lid_nn_cost; // :1320
            self.lid_nb_of_classif[chan] = lid_nb_classif; // :1321
            let mut lid_errors = vec![0.0f64; class_nb];
            for c in 0..class_nb {
                lid_errors[c] = 100.0 * (langid[c] - target_lid[c]); // :1322
            }
            self.lid_classification_errors[chan] = lid_errors;
            self.lid_segments_confusion[chan] = confusion; // :1324
            let max_langid = langid.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            self.is_lid_correct[chan] = if langid[ti] == max_langid { 100 } else { 0 }; // :1325-1329

            lid_cost_ponderation = self.lid_net.get_cost_ponderation(target_index as i64); // :1334
            self.lid_net.ponderate_weights_derivatives(audio_weight); // :1335
            self.lid_cumulative_error[chan] *= audio_weight; // :1336
            self.cumulative_error[chan] *= lid_cost_ponderation; // :1337
        }

        // Epilogue (`:1393-1420`): SAD net ponderate by the LAST channel's LIDCostPonderation.
        self.sad_net
            .ponderate_weights_derivatives(lid_cost_ponderation); // :1393
        for seg in seg_per_chan.iter_mut() {
            compute_errors(seg, None, -1); // :1394
        }
        if no_overlap {
            self.window_shift_sec = 0.0; // :1419
        }
        if lid_no_overlap {
            self.lid_window_shift_sec = 0.0; // :1420
        }

        Ok(())
    }

    /// `setWeights` delegates to the SAD NN (base `BLSTMSpectralSegmenter::setWeights`).
    fn set_weights(&mut self, flat: &[f64]) -> Result<()> {
        self.sad_net.set_weights(flat)
    }

    /// `getWeights` delegates to the SAD NN.
    fn get_weights(&self) -> Vec<f64> {
        self.sad_net.get_weights()
    }
}

#[cfg(test)]
mod twin_tests {
    use super::*;
    use crate::tasks::segmentation::Segmentation;

    /// `getTargetsLID` (`:170-219`) enforcement pattern with `TargetEnforcementStep 2`:
    /// SPEECH rows are `-0.5` until `counter == step` (EQUALITY), then the enforced row
    /// sets every column to `target` (0.0, cost NOT modified) with `target_index -> 1-target`.
    /// Pins the `==`-not-`>=` asymmetry vs the scoring path (IMPROVEMENTS'd).
    #[test]
    fn get_targets_lid_enforcement() {
        let p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/reference_data/phase4b/twin_mode0.config");
        let mut map =
            crate::legacy_config::parse_legacy_config(&std::fs::read_to_string(p).unwrap());
        map.insert(
            "BLSTM_LID_TargetEnforcementStep".to_string(),
            "2".to_string(),
        );
        let drv = TwinBlstmSpectralLid::from_legacy(&map, None, None).unwrap();

        // A single SPEECH span [0, 0.9) so all 6 rows (t = 0.0..0.5 at step 0.1) are SPEECH.
        let mut seg = Segmentation::new(1.0);
        seg.label_segment(0.0, 0.9, SegClass::Speech);
        let targets = drv.get_targets_lid(&seg, 0.1, 0.0, 1, 6);

        // step 2: rows 0,1 = -0.5; row 2 = enforced; rows 3,4 = -0.5; row 5 = enforced.
        for r in [0usize, 1, 3, 4] {
            for c in 0..3 {
                assert_eq!(targets[[r, c]], -0.5, "row {r} col {c} should be -0.5");
            }
        }
        for r in [2usize, 5] {
            assert_eq!(targets[[r, 0]], 0.0, "enforced row {r} col0");
            assert_eq!(
                targets[[r, 1]],
                1.0,
                "enforced row {r} target col (1-target)"
            );
            assert_eq!(targets[[r, 2]], 0.0, "enforced row {r} col2");
        }
    }
}
