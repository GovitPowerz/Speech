//! Audio container + I/O (load, normalize, dither, pre-emphasis, framing).
//!
//! Ported from legacy C++: `AudioStruct.cpp:36-137` (ctor: decode, offset/duration
//! truncation, per-channel normalization), `:433-454` (reset/applyNoise/applyPreemph).
//!
//! Decode uses `symphonia` (wav). PCM16 samples are converted `x as f64 / 32768.0`,
//! matching libsndfile's `sf_readf_double` scaling for 16-bit PCM (the legacy oracle
//! reads via libsndfile, not symphonia; this conversion was verified bit-exact against
//! the oracle's `sig_norm.bin` dump, see phase1_audio_golden.rs).

use std::fs::File;
use std::path::Path;

use anyhow::{Context, bail};
use ndarray::{Array1, Array2};
use symphonia::core::audio::{AudioBufferRef, Signal};
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

use crate::constants::random_uniform;

/// Multi-channel audio buffer (channels x frames, f64).
///
/// `data` is the working buffer (mutated by `apply_preemph`/`apply_noise`, restorable
/// via `reset`); `data_raw` is the decoded+normalized signal (legacy `_DataRaw`/`_Data`
/// right after the ctor, before any effect is applied).
pub struct Audio {
    pub sample_rate: u32,
    pub data: Array2<f64>,
    pub data_raw: Array2<f64>,
    /// `AudioStruct::_LangIndex` (AudioStruct.h:64), set from the owning
    /// `CorpusItem::getClassOfFile()` (AudioStruct.cpp:53 and its four other
    /// per-`file_type` copy sites). `read_audio` (Phase 1) has no `CorpusItem`
    /// in scope -- the bag driver sets this field right after `read_audio`
    /// returns (Phase 4b, `engine::bag_of_processors::apply_corpus_item`), a
    /// documented placement deviation, not a behavior change. Defaults to -1
    /// (the legacy default ctor leaves `_LangIndex` uninitialized; -1 is this
    /// port's explicit stand-in).
    pub lang_index: i32,
    /// `AudioStruct::_Weight` (AudioStruct.h:28), set from
    /// `CorpusItem::getWeightOfFile()` (AudioStruct.cpp:60 + siblings). Same
    /// placement deviation as `lang_index`. Defaults to 1.0, matching the
    /// legacy default ctor's `_Weight(1.0)` (AudioStruct.cpp:33).
    pub weight: f64,
    /// `AudioStruct::_ExternalFeatures` (AudioStruct.h:27): one one-hot
    /// `(sentence_len x 38)` matrix per phSeq line (`file_type == 1`,
    /// AudioStruct.cpp:163-172). Empty for the wav path (`file_type == 0`
    /// never touches this member; `_ExternalFeatures.clear()` is the only
    /// write in that branch's absence, so it stays default-constructed empty).
    pub external_features: Vec<Array2<f64>>,
    /// `AudioStruct::_Periodogram` (AudioStruct.h:24) as pre-filled by the
    /// phSeq ctor (`AudioStruct.cpp:177-182`) -- NOT the per-segmenter
    /// computed periodogram (`compute_segment_periodogram_estimates`, which
    /// returns its own `Array2` rather than writing back to `Audio`). `None`
    /// for the wav path. How a future LID/Twin driver (Phase 4b Task 6/7)
    /// should consume this pre-filled periodogram for `file_type == 1` corpora
    /// is UNRESOLVED here by design (see Task 5 report) -- this field only
    /// exposes the ctor-populated data.
    pub periodogram: Option<Array2<f64>>,
}

impl Audio {
    /// `AudioStruct::applyPreemph` (AudioStruct.cpp:446-454).
    ///
    /// Internal gate reproduced exactly: the legacy function itself is unconditional
    /// on its body, wrapped in `if (preemphRatio > 0)` -- so `ratio <= 0` is a no-op.
    pub fn apply_preemph(&mut self, ratio: f64) {
        if ratio > 0.0 {
            let (channels, frames) = self.data.dim();
            for c in 0..channels {
                for kk in (1..frames).rev() {
                    self.data[[c, kk]] -= ratio * self.data[[c, kk - 1]];
                }
            }
        }
    }

    /// `AudioStruct::applyNoise` (AudioStruct.cpp:437-444). Same table value for every
    /// channel at a given frame index `kk`.
    pub fn apply_noise(&mut self, ratio: f64) {
        let diff = 2.0 * ratio;
        let (channels, frames) = self.data.dim();
        for kk in 0..frames {
            let n = diff * random_uniform(kk) - ratio;
            for c in 0..channels {
                self.data[[c, kk]] += n;
            }
        }
    }

    /// `AudioStruct::reset` (AudioStruct.cpp:433-435).
    pub fn reset(&mut self) {
        self.data = self.data_raw.clone();
    }
}

/// Windowing coefficients (legacy `getWindowingCoefficients`, `Helpers.hpp:222-272`).
///
/// Returns `None` for `size <= 1`, `"none"`, unrecognized kinds, and the
/// `"uniform"` + `normalized == false` combination (load-bearing legacy quirk:
/// uniform is only materialized when normalized, otherwise it falls through
/// to the empty/`None` branch -- i.e. rectangular/no windowing). `normalized`
/// divides the whole vector by the coefficient sum (computed as accumulated
/// during generation, matching the legacy `adim`).
pub fn windowing_coefficients(
    kind: &str,
    normalized: bool,
    size: usize,
    extra_param: f64,
) -> Option<Vec<f64>> {
    if size <= 1 || kind == "none" {
        return None;
    }
    let mut coeff = vec![0.0_f64; size];
    let mut adim = 0.0_f64;
    match kind {
        "hamming" => {
            let constant = 2.0 * std::f64::consts::PI / (size as f64 - 1.0);
            for (ii, c) in coeff.iter_mut().enumerate() {
                let v = 0.54 - 0.46 * (constant * ii as f64).cos();
                *c = v;
                adim += v;
            }
        }
        "hann" => {
            let constant = 2.0 * std::f64::consts::PI / (size as f64 - 1.0);
            for (ii, c) in coeff.iter_mut().enumerate() {
                let v = 0.5 - 0.5 * (constant * ii as f64).cos();
                *c = v;
                adim += v;
            }
        }
        "hHCw" => {
            let extra_param = extra_param.clamp(0.0, 1.0);
            let size1 = f64::round(extra_param * size as f64) as usize;
            let size2 = size - size1;
            let constant1 = 2.0 * std::f64::consts::PI / (2.0 * size1 as f64 - 1.0);
            let constant2 = 2.0 * std::f64::consts::PI / (4.0 * size2 as f64 - 1.0);
            for (ii, c) in coeff.iter_mut().enumerate().take(size1) {
                let v = 0.54 - 0.46 * (constant1 * ii as f64).cos();
                *c = v;
                adim += v;
            }
            for (ii, c) in coeff.iter_mut().enumerate().skip(size1) {
                let v = (constant2 * (ii - size1) as f64).cos();
                *c = v;
                adim += v;
            }
        }
        "uniform" if normalized => {
            coeff.fill(1.0);
            adim = size as f64;
        }
        _ => return None,
    }
    if normalized {
        for c in coeff.iter_mut() {
            *c /= adim;
        }
    }
    Some(coeff)
}

/// Start/end taps into `coeffs` for kernel index `ii` out of `len` positions.
/// Shared edge-truncation arithmetic for `convolution_horiz`/`convolution_vert`
/// (legacy `Convolution`/`ConvolutionVert`, `Helpers.hpp:164-220`). `coeffs` has
/// `2*half_window+1` taps; the source-index math is done in `isize` (the legacy
/// unsigned-wraparound trick for the left edge, ported with signed arithmetic
/// yielding identical final indices).
fn conv_taps(ii: usize, len: usize, half_window: isize) -> (isize, usize, usize) {
    let ii = ii as isize;
    let len = len as isize;
    let (begin1, begin2) = if ii < half_window {
        let begin2 = half_window - ii;
        (-begin2, begin2)
    } else {
        (ii - half_window, 0)
    };
    let end = if ii + half_window < len {
        2 * half_window
    } else {
        len - 1 - begin1
    };
    (begin1, begin2 as usize, end as usize)
}

/// `Convolution` (`Helpers.hpp:164-191`): in-place 1xN horizontal convolution over
/// a plain slice. Edge positions truncate the kernel without renormalizing (missing
/// taps are simply dropped, not redistributed) -- a load-bearing legacy quirk. The
/// natural fit for `results_to_segmentation`/spectral/LTSV/TDC, which hold `Vec`
/// results rather than an `Array2` row.
pub fn convolution_horiz_slice(row: &mut [f64], coeffs: &[f64]) {
    let half_window = ((coeffs.len() - 1) / 2) as isize;
    let cols = row.len();
    let copy: Vec<f64> = row.to_vec();
    for (ii, out) in row.iter_mut().enumerate() {
        let (begin1, begin2, end) = conv_taps(ii, cols, half_window);
        let mut acc = 0.0;
        for jj in begin2..=end {
            acc += copy[(begin1 + jj as isize) as usize] * coeffs[jj];
        }
        *out = acc;
    }
}

/// `Convolution` (`Helpers.hpp:164-191`): in-place 1xN horizontal convolution over
/// an `Array2` row. Delegates to [`convolution_horiz_slice`] (the single
/// implementation) via a contiguous scratch copy.
pub fn convolution_horiz(row: &mut Array2<f64>, coeffs: &[f64]) {
    let mut buf: Vec<f64> = row.row(0).to_vec();
    convolution_horiz_slice(&mut buf, coeffs);
    row.row_mut(0).assign(&Array1::from(buf));
}

/// `ConvolutionVert` (`Helpers.hpp:193-220`): in-place T x B vertical convolution
/// (convolves down time/rows, independently per column). Same edge-truncation
/// contract as `convolution_horiz`.
pub fn convolution_vert(mat: &mut Array2<f64>, coeffs: &[f64]) {
    let half_window = ((coeffs.len() - 1) / 2) as isize;
    let (rows, cols) = mat.dim();
    let copy = mat.clone();
    for ii in 0..rows {
        let (begin1, begin2, end) = conv_taps(ii, rows, half_window);
        for kk in 0..cols {
            let mut acc = 0.0;
            for jj in begin2..=end {
                acc += copy[[(begin1 + jj as isize) as usize, kk]] * coeffs[jj];
            }
            mat[[ii, kk]] = acc;
        }
    }
}

/// Extract a windowed frame centred on `index` into a caller-owned reused buffer
/// (`AudioStruct::getSequence`, AudioStruct.cpp:456-485). `buf` is `1 x (2*half_window+1)`.
///
/// Load-bearing quirks ported verbatim:
/// - Outer guard `index < frames` else NO-OP: the buffer is left untouched, so the
///   caller (the framing driver) consumes whatever STALE contents it held.
/// - Branch order is LEFT edge first, then right edge, then interior.
/// - Left edge (`index < half_window`) does NOT zero the buffer -- only the block
///   copy region is overwritten, the left pad keeps its prior (stale) contents.
/// - Right edge (`index >= frames - half_window`) DOES zero the buffer first.
/// - DC offset (if `dc_offset`): mean over the FULL `2w+1` buffer INCLUDING any pad,
///   subtracted from the full buffer.
/// - Window multiply is applied LAST and only if `coeffs` is `Some`.
pub fn get_sequence(
    data: &Array2<f64>,
    chan: usize,
    index: usize,
    half_window: usize,
    dc_offset: bool,
    coeffs: Option<&[f64]>,
    buf: &mut Array2<f64>,
) {
    let frames = data.ncols();
    if index >= frames {
        return; // NO-OP: caller consumes the stale buffer.
    }
    let cols = buf.ncols();
    let (beg_win, nb_elem, beg_data) = if index < half_window {
        (half_window - index, half_window + index + 1, 0usize)
    } else if index >= frames - half_window {
        for v in buf.iter_mut() {
            *v = 0.0;
        }
        // Legacy computes `half_window - index + frames` in unsigned modular
        // arithmetic (underflows then wraps back). Reordered here to avoid the
        // intermediate underflow; the final count is identical (always positive).
        (0usize, frames + half_window - index, index - half_window)
    } else {
        (0usize, cols, index - half_window)
    };
    for k in 0..nb_elem {
        buf[[0, beg_win + k]] = data[[chan, beg_data + k]];
    }
    if dc_offset {
        let mut sum = 0.0;
        for v in buf.iter() {
            sum += *v;
        }
        let mean = sum / cols as f64;
        for v in buf.iter_mut() {
            *v -= mean;
        }
    }
    if let Some(coeffs) = coeffs {
        for (k, v) in buf.iter_mut().enumerate() {
            *v *= coeffs[k];
        }
    }
}

/// Framing driver: segment periodogram estimates over `[begin, end]`
/// (`AudioStruct::computeSegmentPeriodogramEstimates`, AudioStruct.cpp:512-584,
/// with the MelFilterBank tail dropped -- this port takes an empty bank).
///
/// `window_size = 1 << p`, `signal_window_size = window_size / 2` (the half-window
/// for `get_sequence`), `full_signal_window_size = window_size + 1` (buffer width),
/// `periodogram_length = signal_window_size + 1`.
///
/// Load-bearing quirks:
/// - `frame_nb` is an inclusive-count ceil-divide: `span = end - begin + 1`; if
///   `span` divides `shift` exactly then `span / shift` else `span / shift + 1`.
/// - ONE pair of reused buffers, zero-initialised ONCE before the loop -- the stale
///   contents carry across frames (so an `index >= frames` no-op or a left-edge pad
///   sees the previous frame's data). Matches the OMP-thread-local buffers under a
///   single-thread strict build.
/// - Main loop: `jj` from `begin` while `jj <= end - shift`, stepping `2 * shift`;
///   two frames per FFT at rows `(jj - begin) / shift` and `+1`.
/// - Odd `frame_nb` tail: the last frame is packed as BOTH signals and its row is
///   written TWICE (the second write is the P2 formula, silently clobbering P1).
/// - Temporal convolution applied AFTER the loop iff `conv` len > 1.
// Argument list mirrors the legacy `computeSegmentPeriodogramEstimates` seam (p,
// shift, chan, dc_offset, window, temporal-conv, begin, end); bundling into a
// struct would obscure the parity mapping, so keep the flat signature.
#[allow(clippy::too_many_arguments)]
pub fn compute_segment_periodogram_estimates(
    audio: &Audio,
    p: u32,
    shift: usize,
    chan: usize,
    dc_offset: bool,
    coeffs: Option<&[f64]>,
    conv: Option<&[f64]>,
    begin: usize,
    end: usize,
) -> Array2<f64> {
    let window_size = 1usize << p;
    let signal_window_size = window_size / 2;
    let full_signal_window_size = window_size + 1;
    let periodogram_length = signal_window_size + 1;

    let span = end - begin + 1;
    let frame_nb = if (span / shift) * shift == span {
        span / shift
    } else {
        span / shift + 1
    };

    let mut periodogram = Array2::<f64>::zeros((frame_nb, periodogram_length));
    let gfft = crate::features::fft::Gfft::new(p);

    // ONE pair of reused buffers, zero-initialised ONCE (stale semantics).
    let mut buf1 = Array2::<f64>::zeros((1, full_signal_window_size));
    let mut buf2 = Array2::<f64>::zeros((1, full_signal_window_size));

    // Main loop: two frames per FFT. `end - shift` may underflow if end < shift;
    // guard so the loop simply does not run (the odd tail then fills row 0). The
    // legacy computes `end_frame - window_shift` in unsigned `size_type` arithmetic
    // with no such guard, so on this same corner it would underflow-wrap to a huge
    // bound rather than skip the loop -- unreachable in practice (`end < shift` never
    // occurs with real frame counts/shifts), so this divergence is intentionally
    // unreproduced rather than load-bearing.
    if end >= shift {
        let mut jj = begin;
        while jj <= end - shift {
            get_sequence(
                &audio.data,
                chan,
                jj,
                signal_window_size,
                dc_offset,
                coeffs,
                &mut buf1,
            );
            get_sequence(
                &audio.data,
                chan,
                jj + shift,
                signal_window_size,
                dc_offset,
                coeffs,
                &mut buf2,
            );
            let current_frame = (jj - begin) / shift;
            let (b1, b2) = (buf1.row(0).to_vec(), buf2.row(0).to_vec());
            crate::features::fft::compute_two_real_periodogram(
                &gfft,
                window_size,
                &b1,
                &b2,
                &mut periodogram,
                current_frame,
                current_frame + 1,
            );
            jj += 2 * shift;
        }
    }

    // Odd number of periodogram estimates: pack the last frame as BOTH signals,
    // writing the same row twice (second write = P2, clobbering P1). The legacy
    // allocates a FRESH zeroed buffer here (NOT the loop's stale buffer), so a
    // left-edge tail keeps a genuinely zero pad -- reproduce that.
    if (frame_nb / 2) * 2 != frame_nb {
        let current_frame = frame_nb - 1;
        let jj = current_frame * shift + begin;
        let mut tail = Array2::<f64>::zeros((1, full_signal_window_size));
        get_sequence(
            &audio.data,
            chan,
            jj,
            signal_window_size,
            dc_offset,
            coeffs,
            &mut tail,
        );
        let b1 = tail.row(0).to_vec();
        crate::features::fft::compute_two_real_periodogram(
            &gfft,
            window_size,
            &b1,
            &b1,
            &mut periodogram,
            current_frame,
            current_frame,
        );
    }

    if let Some(conv) = conv
        && conv.len() > 1
    {
        convolution_vert(&mut periodogram, conv);
    }

    periodogram
}

/// Per-channel normalize: `adim = (2*rms + max_abs)/2`, floored at `1e-3`, then divide.
/// `AudioStruct.cpp:121-126`.
pub fn normalize_channels(data: &mut Array2<f64>) {
    let (channels, frames) = data.dim();
    for c in 0..channels {
        let mut sum_sq = 0.0;
        let mut max_abs = 0.0_f64;
        for k in 0..frames {
            let x = data[[c, k]];
            sum_sq += x * x;
            max_abs = max_abs.max(x.abs());
        }
        let rms = (sum_sq / frames as f64).sqrt();
        let mut adim = (2.0 * rms + max_abs) / 2.0;
        if adim < 1e-3 {
            adim = 1e-3;
        }
        for k in 0..frames {
            data[[c, k]] /= adim;
        }
    }
}

/// legacy: `static std::map<char,int> letterMapping` (AudioStruct.h:18), a 38-entry
/// char -> column bijection used by the phSeq readers (`file_type` 1 and 3, only 1
/// ported here). A `std::map` is queried only via `.at()` in the legacy (never
/// iterated), so insertion/traversal order carries no semantics here -- this is a
/// plain lookup, not an ordered table. Out-of-domain characters: the legacy `.at()`
/// throws `std::out_of_range`, UNCAUGHT -> `std::terminate` (an abort, not a clean
/// `exit(1)`); this port turns that crash into a recoverable `bail!` instead of
/// reproducing an abort (a deliberate deviation, not a reproduced quirk -- see
/// IMPROVEMENTS.md).
fn letter_index(c: char) -> Option<usize> {
    Some(match c {
        ' ' => 0,
        '&' => 1,
        'A' => 2,
        '@' => 3,
        'E' => 4,
        'I' => 5,
        'H' => 6,
        'O' => 7,
        'N' => 8,
        'S' => 9,
        'Z' => 10,
        'a' => 11,
        'c' => 12,
        'b' => 13,
        'e' => 14,
        'd' => 15,
        'g' => 16,
        'f' => 17,
        'i' => 18,
        'h' => 19,
        'k' => 20,
        'j' => 21,
        'm' => 22,
        'l' => 23,
        'o' => 24,
        'n' => 25,
        'p' => 26,
        's' => 27,
        'r' => 28,
        'u' => 29,
        't' => 30,
        'w' => 31,
        'v' => 32,
        'y' => 33,
        'x' => 34,
        'z' => 35,
        '.' => 36,
        '-' => 37,
        _ => return None,
    })
}

/// `letterMapping.size()` (AudioStruct.h:18): the one-hot width / `_Periodogram`
/// column count for the phSeq readers.
const LETTER_MAPPING_SIZE: usize = 38;

/// legacy: `AudioStruct.cpp:173` `_FramesCount = numberOfPhonemes*0.01*_Framerate;`.
/// Isolated into its own function so the LEFT-ASSOCIATIVE evaluation order
/// (`(numberOfPhonemes*0.01)*framerate`, matching C++'s left-to-right `*`
/// grouping) is directly unit-testable: it diverges from the naive
/// `numberOfPhonemes*(0.01*framerate)` regrouping by 1 truncated frame at
/// `number_of_phonemes = 803` (`framerate = 8000`) -- see `frames_count_arithmetic`
/// (IMPROVEMENTS.md). The intermediate promotions (`int -> double`, `long ->
/// double`) and the final double -> `long long` truncation-toward-zero on
/// assignment are both reproduced (`as f64` / `as i64`).
fn phseq_frames_count(number_of_phonemes: i64, framerate: i64) -> i64 {
    ((number_of_phonemes as f64 * 0.01) * framerate as f64) as i64
}

/// `AudioStruct` ctor, `file_type == 1` branch (`AudioStruct.cpp:138-182`): phoneme
/// sequence text reader. `_ChannelsCount = 1`, `_Framerate = 8000` (hardcoded, NOT
/// config-derived -- `:143-144`); `audio_offset`/`audio_max_duration` are stored on
/// the legacy `AudioStruct` (`_OffsetBegin`/`_DurationMax`) but never READ again in
/// this branch (no truncation of the zero-filled data), so this port's `Audio` has
/// no fields for them either -- both `read_audio` params are simply unused on this
/// path.
///
/// Each line of the file is a "sentence": a one-hot `(len x 38)` matrix keyed by
/// [`letter_index`]. `number_of_phonemes` accumulates `len+10` per sentence (10
/// initial, `:164`); `_FramesCount` per [`phseq_frames_count`] (`:173`).
/// `_Periodogram` is `(number_of_phonemes x 38)` zeroed (`:177`) then block-filled
/// per sentence starting at `row_begin = 5` (`:178`), stepping `len+10` rows after
/// each write (`:180-181`) -- a fixed 5-row lead margin plus a 10-row gap between
/// sentences. By construction `number_of_phonemes = 10 + sum(len_i+10)` while the
/// FINAL `row_begin` (after the loop) is `5 + sum(len_i+10)`, so
/// `number_of_phonemes - final_row_begin == 5` identically: a fixed 5-row trailing
/// margin survives regardless of sentence count/lengths, so the block-fill NEVER
/// overflows the allocation (see IMPROVEMENTS.md). `_DataRaw`/`_Data` are
/// `(1 x _FramesCount)` zeroed (`:174-175`, no real audio -- phSeq carries no
/// waveform).
fn read_phseq(path: &Path) -> anyhow::Result<Audio> {
    // legacy: :149-152 `ifstream` bool-conversion failure -> exit(1).
    let text = std::fs::read_to_string(path).with_context(|| {
        format!(
            "No .phSeq file given or wrong path for the file \"{}\".",
            path.display()
        )
    })?;
    // legacy: :157-161 getline loop, kept verbatim (no trimming beyond the newline
    // itself). Rust's `str::lines` also strips a trailing '\r' (C++ `getline` does
    // not) -- irrelevant to this port's Unix-line-ending-only fixtures.
    let lines: Vec<&str> = text.lines().collect();

    let mut external_features: Vec<Array2<f64>> = Vec::with_capacity(lines.len());
    let mut number_of_phonemes: i64 = 10; // legacy :164.
    for (line_idx, sentence) in lines.iter().enumerate() {
        let len = sentence.chars().count();
        let mut feat = Array2::<f64>::zeros((len, LETTER_MAPPING_SIZE));
        for (pos, ch) in sentence.chars().enumerate() {
            let col = letter_index(ch).with_context(|| {
                format!("phSeq char {ch:?} at line {line_idx} pos {pos} not in letterMapping")
            })?;
            feat[[pos, col]] = 1.0; // legacy :168.
        }
        number_of_phonemes += len as i64 + 10; // legacy :170.
        external_features.push(feat);
    }

    // number_of_phonemes >= 10 always (never negative), so the truncated
    // frame count is always non-negative.
    let frames_count = phseq_frames_count(number_of_phonemes, 8000) as usize;
    let data_raw = Array2::<f64>::zeros((1, frames_count));
    let data = Array2::<f64>::zeros((1, frames_count));

    let mut periodogram = Array2::<f64>::zeros((number_of_phonemes as usize, LETTER_MAPPING_SIZE));
    let mut row_begin: usize = 5; // legacy :178.
    for feat in &external_features {
        let rows = feat.nrows();
        periodogram
            .slice_mut(ndarray::s![row_begin..row_begin + rows, ..])
            .assign(feat);
        row_begin += rows + 10; // legacy :181.
    }

    Ok(Audio {
        sample_rate: 8000,
        data,
        data_raw,
        lang_index: -1,
        weight: 1.0,
        external_features,
        periodogram: Some(periodogram),
    })
}

/// `AudioStruct` ctor dispatch (`AudioStruct.cpp:36-412`). `file_type == 0`: decode a
/// wav file, apply the legacy offset/duration truncation, then normalize
/// (`:36-128`). `file_type == 1`: phSeq text reader (`:138-182`, see
/// [`read_phseq`]). Any other value is unported (`file_type` 2/3/4 -- cep/phSeq-N/
/// mat readers, AudioStruct.cpp:183-412).
pub fn read_audio(
    path: &Path,
    offset_sec: f64,
    max_duration_sec: f64,
    file_type: i32,
) -> anyhow::Result<Audio> {
    if file_type == 1 {
        return read_phseq(path);
    }
    if file_type != 0 {
        bail!(
            "read_audio: file_type {file_type} not supported (Phase 4b: 0=wav, 1=phSeq ported; 2/3/4 unported)"
        );
    }
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let probed = symphonia::default::get_probe().format(
        &hint,
        mss,
        &FormatOptions::default(),
        &MetadataOptions::default(),
    )?;
    let mut format = probed.format;

    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != symphonia::core::codecs::CODEC_TYPE_NULL)
        .context("no decodable audio track")?;
    let track_id = track.id;
    let sample_rate = track
        .codec_params
        .sample_rate
        .context("missing sample rate")?;

    let mut decoder =
        symphonia::default::get_codecs().make(&track.codec_params, &DecoderOptions::default())?;

    // Interleaved f64 samples, decoded in packet order.
    let mut interleaved: Vec<f64> = Vec::new();
    let mut channels_count = 0usize;

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(symphonia::core::errors::Error::IoError(e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(symphonia::core::errors::Error::ResetRequired) => break,
            Err(e) => return Err(e.into()),
        };
        if packet.track_id() != track_id {
            continue;
        }
        let decoded = decoder.decode(&packet)?;
        if channels_count == 0 {
            channels_count = decoded.spec().channels.count();
        }
        push_samples_as_f64(&decoded, &mut interleaved);
    }

    if channels_count == 0 {
        bail!("no audio frames decoded from {}", path.display());
    }
    let total_frames = interleaved.len() / channels_count;

    let frame_offset_begin = (sample_rate as f64 * offset_sec).round() as i64;
    let frame_count_max = (sample_rate as f64 * max_duration_sec + 1.0).round() as i64;
    let mut frames_count = total_frames as i64 - frame_offset_begin;
    if frames_count > frame_count_max {
        frames_count = frame_count_max;
    }
    if frames_count < 0 {
        frames_count = 0;
    }
    let frames_count = frames_count as usize;
    let offset = frame_offset_begin.max(0) as usize;

    let mut data_raw = Array2::<f64>::zeros((channels_count, frames_count));
    for k in 0..frames_count {
        let src_frame = offset + k;
        for c in 0..channels_count {
            data_raw[[c, k]] = interleaved[src_frame * channels_count + c];
        }
    }

    normalize_channels(&mut data_raw);
    let data = data_raw.clone();

    Ok(Audio {
        sample_rate,
        data,
        data_raw,
        lang_index: -1,
        weight: 1.0,
        external_features: Vec::new(),
        periodogram: None,
    })
}

/// Append one decoded packet's samples to `out`, interleaved, converting PCM16 as
/// `x as f64 / 32768.0` to match libsndfile's `sf_readf_double` scaling.
fn push_samples_as_f64(decoded: &AudioBufferRef, out: &mut Vec<f64>) {
    match decoded {
        AudioBufferRef::S16(buf) => {
            let channels = buf.spec().channels.count();
            let frames = buf.frames();
            for f in 0..frames {
                for c in 0..channels {
                    out.push(buf.chan(c)[f] as f64 / 32768.0);
                }
            }
        }
        AudioBufferRef::F32(buf) => {
            let channels = buf.spec().channels.count();
            let frames = buf.frames();
            for f in 0..frames {
                for c in 0..channels {
                    out.push(buf.chan(c)[f] as f64);
                }
            }
        }
        AudioBufferRef::F64(buf) => {
            let channels = buf.spec().channels.count();
            let frames = buf.frames();
            for f in 0..frames {
                for c in 0..channels {
                    out.push(buf.chan(c)[f]);
                }
            }
        }
        _ => panic!("unsupported sample format for phase1 wav decode"),
    }
}

#[cfg(test)]
fn test_audio(samples: Vec<f64>) -> Audio {
    let n = samples.len();
    let data = Array2::from_shape_vec((1, n), samples).unwrap();
    Audio {
        sample_rate: 8000,
        data: data.clone(),
        data_raw: data,
        lang_index: -1,
        weight: 1.0,
        external_features: Vec::new(),
        periodogram: None,
    }
}

#[cfg(test)]
fn test_audio_2ch(samples: Vec<f64>) -> Audio {
    let n = samples.len();
    let data = Array2::from_shape_vec((1, n), samples).unwrap();
    let data = ndarray::concatenate(ndarray::Axis(0), &[data.view(), data.view()]).unwrap();
    Audio {
        sample_rate: 8000,
        data: data.clone(),
        data_raw: data,
        lang_index: -1,
        weight: 1.0,
        external_features: Vec::new(),
        periodogram: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preemph_backward_sample0_untouched() {
        // x=[1,2,3,4], r=0.5 -> backward: [1, 2-0.5*1, 3-0.5*2, 4-0.5*3] = [1, 1.5, 2, 2.5]
        let mut a = test_audio(vec![1.0, 2.0, 3.0, 4.0]);
        a.apply_preemph(0.5);
        assert_eq!(a.data.row(0).to_vec(), vec![1.0, 1.5, 2.0, 2.5]);
    }

    #[test]
    fn noise_uses_uniform_table_same_across_channels() {
        let mut a = test_audio_2ch(vec![0.0; 3]);
        a.apply_noise(0.25);
        let expect = |k: usize| 2.0 * 0.25 * crate::constants::random_uniform(k) - 0.25;
        for k in 0..3 {
            assert_eq!(a.data[[0, k]], expect(k));
            assert_eq!(a.data[[1, k]], expect(k));
        }
    }

    #[test]
    fn normalize_is_2rms_plus_peak_over_2_with_floor() {
        // x=[3,4]: adim=(2*sqrt(25/2/... expression below))/...
        let mut d = ndarray::arr2(&[[3.0, 4.0]]);
        normalize_channels(&mut d);
        let adim = (2.0 * ((3.0f64 * 3.0 + 4.0 * 4.0) / 2.0).sqrt() + 4.0) / 2.0;
        assert_eq!(d[[0, 0]], 3.0 / adim);
        let mut z = ndarray::arr2(&[[0.0, 0.0]]);
        normalize_channels(&mut z); // adim floors at 1e-3
        assert_eq!(z[[0, 0]], 0.0);
    }
}
