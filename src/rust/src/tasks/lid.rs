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

use anyhow::Result;
use indexmap::IndexMap;
use ndarray::Array2;

use crate::audio::{Audio, compute_segment_periodogram_estimates, windowing_coefficients};
use crate::features::ltsv_tdc::ltsv_classify_sequence;
use crate::features::mel::MelFilterBank;
use crate::features::pipeline::{FeatureConfig, SpectralParams};
use crate::nn::blstm::{BlstmConfig, BlstmNetwork};
use crate::tasks::sad::get_blstm_param;
use crate::tasks::segmentation::{SegClass, Segmentation};
use crate::tasks::segmentation_io::{ScoreReport, compute_errors};
use crate::tasks::segmenter::{DriverConfig, Segmenter, SegmenterConfig, results_to_segmentation};

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
