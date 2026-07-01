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
use ndarray::Array2;
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

/// Decode a wav file, apply the legacy offset/duration truncation, then normalize.
/// `AudioStruct.cpp:36-128` (file_type == 0 branch).
pub fn read_audio(path: &Path, offset_sec: f64, max_duration_sec: f64) -> anyhow::Result<Audio> {
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
