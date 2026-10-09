//! PyO3 bindings exposing the `speech` engine to Python as module `speech_rs`.
//!
//! Phase 4c Task 3: the coarse corpus-level [`Engine`] -- a thin wrapper over
//! `speech::engine::corpus_processor::CorpusProcessor` (the "PyO3 seam surface"
//! landed in Task 1). This is the legacy file/subprocess contract driven
//! in-process: build from config paths or maps, `run()` the corpus fold, then read back
//! the result matrix / weights / derivatives / gradient check.
//!
//! numpy crossings COPY (spec R2): every ndarray/vec returned to Python is a
//! fresh numpy array, and every array passed in is read into an owned `Vec`.
//! The copy site is doc-commented on each method. Zero-copy is a later
//! hot-loop concern.

use indexmap::IndexMap;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use speech::cli::Mode;
use speech::engine::corpus_processor::CorpusProcessor;
use speech::fast::stream::{EmittedSegment, StreamingSession as RsStreamingSession};
use speech::fast::stream_lid::{LidAggregate, StreamingLidSession as RsStreamingLidSession};
use speech::legacy_config::parse_legacy_config as parse_legacy_config_rs;
use speech::stream_cli::class_str;
use speech::toml_config::toml_to_map as toml_to_map_rs;

/// Convert an engine error into a Python `RuntimeError`, formatting the FULL
/// chain via the alternate `Display` (`{:#}`, which `anyhow::Error` renders as
/// its cause chain) so a nested cause (e.g. a missing corpus file) survives.
/// Generic over `Display` to avoid taking a direct `anyhow` dependency.
fn to_pyerr<E: std::fmt::Display>(e: E) -> PyErr {
    PyRuntimeError::new_err(format!("{e:#}"))
}

/// Read a config FILE into the flat `IndexMap` the engine takes, dispatched by
/// extension like `cli::parse_cli`: a `.toml` path goes through
/// `toml_config::toml_to_map`, anything else through `parse_legacy_config`. The
/// one loader behind `Engine::new` and both streaming sessions (issue #22).
fn load_config_map(path: &str) -> PyResult<IndexMap<String, String>> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| PyRuntimeError::new_err(format!("cannot read config file '{path}': {e}")))?;
    if path.ends_with(".toml") {
        toml_to_map_rs(&text)
            .map_err(|e| PyRuntimeError::new_err(format!("invalid TOML config '{path}': {e:#}")))
    } else {
        Ok(parse_legacy_config_rs(&text))
    }
}

/// A Python `dict[str, str]` as the flat config map, insertion order preserved
/// into the `IndexMap` (the legacy last-wins rule has already been applied by
/// whoever built the dict). A non-`str` key or value is a `TypeError`.
fn dict_to_map(dict: &Bound<'_, PyDict>) -> PyResult<IndexMap<String, String>> {
    let mut map = IndexMap::with_capacity(dict.len());
    for (k, v) in dict.iter() {
        map.insert(k.extract::<String>()?, v.extract::<String>()?);
    }
    Ok(map)
}

/// Version of the underlying `speech` engine crate.
#[pyfunction]
fn version() -> String {
    speech::version().to_string()
}

/// Build provenance of this module (`profile`, `target`, `rustc`), baked in by
/// the core crate's `build.rs`. The ledger (issue #20) stamps it on every
/// record so a debug-profile number can be refused at promotion time.
#[pyfunction]
fn build_info(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let info = speech::build_info();
    let d = PyDict::new(py);
    d.set_item("profile", info.profile)?;
    d.set_item("target", info.target)?;
    d.set_item("rustc", info.rustc)?;
    Ok(d)
}

/// Parse a legacy whitespace `.config` text into a `dict[str, str]` (last-wins,
/// insertion order preserved). The Python mirror of the Rust
/// `legacy_config::parse_legacy_config`.
#[pyfunction]
fn parse_legacy_config<'py>(py: Python<'py>, text: &str) -> PyResult<Bound<'py, PyDict>> {
    let map = parse_legacy_config_rs(text);
    let dict = PyDict::new(py);
    for (k, v) in &map {
        dict.set_item(k, v)?;
    }
    Ok(dict)
}

/// Load a canonical TOML config FILE (Phase 4c Task 5) into a `dict[str, str]`,
/// flattened through `toml_config::toml_to_map` -- the same shape
/// `parse_legacy_config` produces, regardless of which format the config is
/// authored in. Takes a PATH (not text): unlike `parse_legacy_config`, the TOML
/// side has no meaningful "text with no file" use case in this API since the
/// canonical workflow is always a config file on disk.
#[pyfunction]
fn load_toml_config<'py>(py: Python<'py>, path: &str) -> PyResult<Bound<'py, PyDict>> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| PyRuntimeError::new_err(format!("cannot read config file '{path}': {e}")))?;
    let map = toml_to_map_rs(&text).map_err(to_pyerr)?;
    let dict = PyDict::new(py);
    for (k, v) in &map {
        dict.set_item(k, v)?;
    }
    Ok(dict)
}

/// The coarse corpus-level engine: a `CorpusProcessor` behind the legacy
/// corpus contract, driven in-process. Construct from config PATHS (`Engine(...)`)
/// or from already-flattened config MAPS (`Engine.from_map(...)`, issue #22) plus a
/// CLI mode flag, `run()`, then read the results. Either way the process cwd is what
/// relative paths in the config and the listing resolve against, as for the binary.
#[pyclass]
struct Engine {
    inner: CorpusProcessor,
}

impl Engine {
    /// The shared tail of both constructors: parse the mode flag, build the
    /// `CorpusProcessor`.
    fn build(configs: Vec<IndexMap<String, String>>, mode: &str) -> PyResult<Self> {
        let mode = Mode::from_flag(mode)
            .ok_or_else(|| PyRuntimeError::new_err(format!("unknown mode flag '{mode}'")))?;
        let inner = CorpusProcessor::new(configs, mode).map_err(to_pyerr)?;
        Ok(Engine { inner })
    }
}

#[pymethods]
impl Engine {
    /// Build from config PATHS + a CLI mode flag (`"-m"`, `"-s"`, ...; parsed
    /// via `cli::Mode::from_flag`). Each path is read through `load_config_map`
    /// (`.toml` / `.config` dispatched by extension). An empty `config_paths` is
    /// NOT checked here -- `CorpusProcessor::new` already bails with its own "at
    /// least one config is required" message, so a redundant pre-check here would
    /// just duplicate that error surface.
    #[new]
    fn new(config_paths: Vec<String>, mode: String) -> PyResult<Self> {
        let configs = config_paths
            .iter()
            .map(|p| load_config_map(p))
            .collect::<PyResult<Vec<_>>>()?;
        Self::build(configs, &mode)
    }

    /// Build from config MAPS (`list[dict[str, str]]`, the flat legacy key shape
    /// the `KEY_TABLE` produces, ADR-0006) + a CLI mode flag: the path constructor
    /// minus the file. Dict insertion order is preserved into the `IndexMap`. No
    /// file is read and nothing is written; relative paths inside the map still
    /// resolve against the process cwd. COPY: every key and value is read into an
    /// owned `String`.
    #[staticmethod]
    fn from_map(configs: Vec<Bound<'_, PyDict>>, mode: String) -> PyResult<Self> {
        let configs = configs
            .iter()
            .map(dict_to_map)
            .collect::<PyResult<Vec<_>>>()?;
        Self::build(configs, &mode)
    }

    /// Run the corpus fold / epoch loop. The engine can take minutes, so the
    /// GIL is released for its duration via `Python::detach` (pyo3 0.28's
    /// renamed `allow_threads`); the error chain is formatted (`{:#}`) inside
    /// the closure so nothing GIL-bound escapes it.
    fn run(&mut self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.inner.run().map_err(|e| format!("{e:#}")))
            .map_err(PyRuntimeError::new_err)
    }

    /// The post-`transformResults` combined result matrix (`ResultsE`): rows
    /// `[file+1, conf+1, chan+1, res...]` in ascending file/conf/chan order.
    /// EMPTY (`0x0`) before the first `run()`. COPY: `ToPyArray` clones the
    /// row-major `Array2` into a fresh C-contiguous numpy array.
    fn results_matrix<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.inner.results_matrix().to_pyarray(py)
    }

    /// The channel results by name (issue #23): one dict, one numpy array per
    /// field, one entry per (file, config, channel) in ascending order -- the
    /// typed view `results_matrix()` is rendered from, minus the WER tallies
    /// (columns 7-13, dead surface, not exposed). Ids are 0-based like
    /// every `pos` on this seam. `lid_target` is `-1` for a row with no in-band
    /// target, `lid_scores` is `n x N` with the target column already decoded
    /// (`N == 0` on a run without LID; a row without a LID block, impossible in
    /// a well-formed bag, is zero-filled), `lid_correct` is a bool array.
    /// EMPTY arrays before the first `run()`. COPY: every array is freshly
    /// allocated; `speech.engine.ChannelResults.from_seam` is the Python wrapper.
    fn channel_results<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let rows = self.inner.channel_results();
        let n = rows.len();
        let n_classes = rows
            .iter()
            .filter_map(|(_, _, _, r)| r.lid.as_ref().map(|l| l.scores.len()))
            .max()
            .unwrap_or(0);
        let mut file = Vec::with_capacity(n);
        let mut conf = Vec::with_capacity(n);
        let mut chan = Vec::with_capacity(n);
        let mut pfa = Vec::with_capacity(n);
        let mut pmiss = Vec::with_capacity(n);
        let mut error_rate = Vec::with_capacity(n);
        let mut time_per_hour = Vec::with_capacity(n);
        let mut seg_cost = Vec::with_capacity(n);
        let mut audio_duration = Vec::with_capacity(n);
        let mut speech_duration = Vec::with_capacity(n);
        let mut lid_cost = Vec::with_capacity(n);
        let mut lid_correct = Vec::with_capacity(n);
        let mut lid_target = Vec::with_capacity(n);
        let mut lid_scores = numpy::ndarray::Array2::<f64>::zeros((n, n_classes));
        let mut lid_count = Vec::with_capacity(n);
        let mut seg_count = Vec::with_capacity(n);
        for (i, (f, c, ch, r)) in rows.into_iter().enumerate() {
            file.push(f as i64);
            conf.push(c as i64);
            chan.push(ch as i64);
            pfa.push(r.pfa);
            pmiss.push(r.pmiss);
            error_rate.push(r.error_rate);
            time_per_hour.push(r.time_per_hour);
            seg_cost.push(r.seg_cost);
            audio_duration.push(r.audio_duration);
            speech_duration.push(r.speech_duration);
            seg_count.push(r.seg_count);
            lid_cost.push(r.lid_cost());
            lid_count.push(r.lid_count());
            lid_correct.push(r.lid.as_ref().is_some_and(|l| l.correct));
            lid_target.push(
                r.lid
                    .as_ref()
                    .and_then(|l| l.target)
                    .map_or(-1, |t| t as i64),
            );
            if let Some(l) = &r.lid {
                for (k, &v) in l.scores.iter().enumerate() {
                    lid_scores[[i, k]] = v;
                }
            }
        }
        let d = PyDict::new(py);
        d.set_item("file", file.into_pyarray(py))?;
        d.set_item("conf", conf.into_pyarray(py))?;
        d.set_item("chan", chan.into_pyarray(py))?;
        d.set_item("pfa", pfa.into_pyarray(py))?;
        d.set_item("pmiss", pmiss.into_pyarray(py))?;
        d.set_item("error_rate", error_rate.into_pyarray(py))?;
        d.set_item("time_per_hour", time_per_hour.into_pyarray(py))?;
        d.set_item("seg_cost", seg_cost.into_pyarray(py))?;
        d.set_item("audio_duration", audio_duration.into_pyarray(py))?;
        d.set_item("speech_duration", speech_duration.into_pyarray(py))?;
        d.set_item("lid_cost", lid_cost.into_pyarray(py))?;
        d.set_item("lid_correct", lid_correct.into_pyarray(py))?;
        d.set_item("lid_target", lid_target.into_pyarray(py))?;
        d.set_item("lid_scores", lid_scores.into_pyarray(py))?;
        d.set_item("lid_count", lid_count.into_pyarray(py))?;
        d.set_item("seg_count", seg_count.into_pyarray(py))?;
        Ok(d)
    }

    /// Config-`pos`'s per-network flat weight vectors, preserving the pairing:
    /// `[sad, lid]` for algo 6, `[flat]` for algo 3/4/5, `[]` for algo 0/1/2.
    /// COPY: each `Vec<f64>` is moved into its own numpy array.
    fn weights<'py>(&self, py: Python<'py>, pos: usize) -> Vec<Bound<'py, PyArray1<f64>>> {
        self.inner
            .weights(pos)
            .into_iter()
            .map(|v| v.into_pyarray(py))
            .collect()
    }

    /// Seed config-`pos`'s network(s): `[sad, lid]` for algo 6, `[flat]` for
    /// algo 3/4/5; algo 0/1/2 (no NN) is a no-op. COPY: each numpy array (or
    /// list) is read into an owned `Vec<f64>` on the way in.
    fn set_weights(&mut self, pos: usize, nets: Vec<Vec<f64>>) -> PyResult<()> {
        self.inner.set_weights(pos, &nets).map_err(to_pyerr)
    }

    /// Config-`pos`'s per-network `Nx2` derivative matrices (col 0 summed deriv,
    /// col 1 count; `[sad, lid]` for algo 6). COPY: each `Array2` moved into a
    /// numpy array.
    fn weights_derivatives<'py>(
        &self,
        py: Python<'py>,
        pos: usize,
    ) -> Vec<Bound<'py, PyArray2<f64>>> {
        self.inner
            .weights_derivatives(pos)
            .into_iter()
            .map(|m| m.into_pyarray(py))
            .collect()
    }

    /// Corpus-level gradient check, capped at `max_weights` per network: one
    /// `(network_index, {mean_error, mean_relative_error, per_weight})` per
    /// backprop-active network of config 0, ascending index. `per_weight` is an
    /// `Nx3` array of `(backprop_normalized, numerical, abs_diff)` rows. COPY:
    /// the report triples are packed into a fresh `Nx3` numpy array.
    fn grad_check<'py>(
        &mut self,
        py: Python<'py>,
        epsilon: f64,
        max_weights: usize,
    ) -> PyResult<Vec<(usize, Bound<'py, PyDict>)>> {
        let reports = self
            .inner
            .grad_check(epsilon, max_weights)
            .map_err(to_pyerr)?;
        let mut out = Vec::with_capacity(reports.len());
        for (net_idx, r) in reports {
            let n = r.per_weight.len();
            let mut flat = Vec::with_capacity(n * 3);
            for (bp, num, diff) in &r.per_weight {
                flat.push(*bp);
                flat.push(*num);
                flat.push(*diff);
            }
            let per_weight = numpy::ndarray::Array2::from_shape_vec((n, 3), flat)
                .expect("per_weight is Nx3 by construction")
                .into_pyarray(py);
            let dict = PyDict::new(py);
            dict.set_item("mean_error", r.mean_error)?;
            dict.set_item("mean_relative_error", r.mean_relative_error)?;
            dict.set_item("per_weight", per_weight)?;
            out.push((net_idx, dict));
        }
        Ok(out)
    }
}

/// One emitted segment as the plain Python tuple `push`/`finish` yield (aliased to keep
/// the method signatures readable, per clippy::type_complexity).
type EmissionTuple = (f64, f64, String, f64);
/// One segmentation boundary row as `(begin_s, class)`.
type SegRow = (f64, String);

/// One emitted segment as a plain `(begin_s, end_s, class, emitted_at_audio_s)` tuple.
fn seg_to_tuple(s: EmittedSegment) -> EmissionTuple {
    (s.begin_s, s.end_s, class_str(s.class), s.emitted_at_audio_s)
}

/// The chunked-input SAD streaming session (Phase 8 Task 6): a thin wrapper over
/// `speech::fast::stream::StreamingSession` (the online twin of the offline fast
/// algo-3 driver). Construct from a config PATH + the wav header's `rate`/`channels`,
/// then drive it with `push(samples)` / `finish()`. Mono, frozen-stats only -- the
/// construction validates the streaming contract (algo 3, mono, `Audio_fixed_gain`,
/// `InputNormalizationType 1`, windowed overlap, binary SAD net) and bails otherwise.
#[pyclass]
struct StreamingSession {
    inner: RsStreamingSession,
}

#[pymethods]
impl StreamingSession {
    /// Build from a config PATH (read through `load_config_map`, the same loader as
    /// `Engine::new`), the stream
    /// `rate` (Hz) and source `channels` count -- both read from the wav header by the
    /// caller (the session is mono-first; `channels != 1` bails). COPY: the config is
    /// read from disk + parsed into an owned map.
    #[new]
    fn new(config_path: &str, rate: f64, channels: usize) -> PyResult<Self> {
        let map = load_config_map(config_path)?;
        let inner = RsStreamingSession::new(&map, rate, channels).map_err(to_pyerr)?;
        Ok(StreamingSession { inner })
    }

    /// Push a chunk of RAW mono f32 samples (`i16/32768`-scaled, PRE-gain -- the
    /// front-end applies the frozen gain/preemph/noise per sample) and return the
    /// segments FINALIZED by this call as `(begin_s, end_s, class, emitted_at_audio_s)`
    /// tuples (`class` in {"Speech", "Other"}). COPY: the numpy array is read into an
    /// owned `Vec<f32>` while the GIL is held, then the GIL is RELEASED
    /// (`Python::detach`) for the streaming compute.
    fn push(
        &mut self,
        py: Python<'_>,
        samples: PyReadonlyArray1<'_, f32>,
    ) -> PyResult<Vec<EmissionTuple>> {
        let owned: Vec<f32> = samples
            .as_slice()
            .map_err(|e| {
                PyRuntimeError::new_err(format!("samples must be a contiguous 1-D f32 array: {e}"))
            })?
            .to_vec();
        let emitted = py.detach(|| self.inner.push(&owned));
        Ok(emitted.into_iter().map(seg_to_tuple).collect())
    }

    /// EOS: flush the tail and return `(final_emissions, segmentation_rows)`. The
    /// emissions are the same `(begin_s, end_s, class, emitted_at_audio_s)` tuples;
    /// `segmentation_rows` is the complete partition boundary list as `(begin_s, class)`
    /// pairs (the last row is the `End` sentinel at the audio duration). COPY + the GIL
    /// is RELEASED for the flush compute. Idempotent (a second call yields no emissions
    /// and the same segmentation rows).
    fn finish(&mut self, py: Python<'_>) -> (Vec<EmissionTuple>, Vec<SegRow>) {
        py.detach(|| {
            let (emitted, seg) = self.inner.finish();
            let ems: Vec<EmissionTuple> = emitted.into_iter().map(seg_to_tuple).collect();
            let rows: Vec<SegRow> = seg
                .segments()
                .iter()
                .map(|s| (s.begin, class_str(s.ty)))
                .collect();
            (ems, rows)
        })
    }

    /// The maximum per-emission lag (`emitted_at_audio_s - end_s`) observed so far
    /// (`0.0` if nothing has been emitted).
    fn max_lag_s(&self) -> f64 {
        self.inner.max_lag_s()
    }

    /// The mean per-emission lag observed so far (`0.0` if nothing has been emitted).
    fn mean_lag_s(&self) -> f64 {
        self.inner.mean_lag_s()
    }
}

/// The finalized/running LID aggregate as `(classification_errors, confusion,
/// is_lid_correct, predicted_language, segments_count)`
/// (`speech::fast::stream_lid::LidAggregate`'s fields, in declaration order). `confusion`
/// is a fresh numpy array (`IntoPyArray`, moved from the owned `Array2` -- no separate
/// clone needed since the aggregate is consumed here).
type LidAggTuple<'py> = (Vec<f64>, Bound<'py, PyArray2<f64>>, bool, usize, i32);

/// One utterance's streaming LID score as `(scores, argmax, is_correct,
/// running_aggregate)` (`speech::fast::stream_lid::UtteranceScore`'s fields, in
/// declaration order).
type UtteranceTuple<'py> = (Vec<f64>, Option<usize>, Option<bool>, LidAggTuple<'py>);

/// Convert an owned `LidAggregate` into its Python tuple, moving `confusion` into a
/// fresh numpy array.
fn agg_to_tuple(py: Python<'_>, a: LidAggregate) -> LidAggTuple<'_> {
    (
        a.classification_errors,
        a.confusion.into_pyarray(py),
        a.is_lid_correct != 0,
        a.predicted_language,
        a.segments_count,
    )
}

/// The per-utterance streaming LID session (Phase 10 Task 10): a thin wrapper over
/// `speech::fast::stream_lid::StreamingLidSession` (the online twin of the offline fast
/// Mode-7 LID Twin, `fast::driver::FastTwinLid`) -- the deferral that fell off the
/// phase-9 spec's own Deferred list (spec S8). Construct from a config PATH + the
/// utterances-stream `rate` (Hz -- the phSeq/cep readers hardcode 8000.0) + the eval
/// target `lang_index`, then drive it one `external_features` entry ("utterance") at a
/// time via `push_utterance` / `finish`. The LID net loads from the config's
/// `BLSTM_LID_weightsFile` (mirroring `StreamingSession`'s own config-driven weight
/// load, the T4/T5/T7 precedent `fast/stream_lid.rs`'s module doc names) -- construction
/// bails if `BLSTM_LID_Mode != 7`, the LID window resolves to a regime the net's shape
/// refuses (overlap on any shape, a window on a causal net), or no net loads.
#[pyclass]
struct StreamingLidSession {
    inner: RsStreamingLidSession,
}

#[pymethods]
impl StreamingLidSession {
    /// Build from a config PATH (read through `load_config_map`, the same loader as
    /// `Engine::new`/`StreamingSession::new`), the stream `rate` (Hz), and the eval
    /// target `lang_index`
    /// (clamped into `[0, class_nb)` by the Rust session -- a negative value targets
    /// class 0). COPY: the config is read from disk + parsed into an owned map. The LID
    /// weight pack is NOT passed explicitly here (mirrors `StreamingSession::new`'s
    /// config-driven load): `None` defers to the config's `BLSTM_LID_weightsFile`.
    #[new]
    fn new(config_path: &str, rate: f64, lang_index: i32) -> PyResult<Self> {
        let map = load_config_map(config_path)?;
        let inner = RsStreamingLidSession::new(&map, rate, lang_index, None).map_err(to_pyerr)?;
        Ok(StreamingLidSession { inner })
    }

    /// Fold ONE utterance -- an `(frames x feat_dim)` feature matrix, matching one
    /// offline `external_features` entry (a phSeq one-hot block / a cep record) -- and
    /// return `(scores, argmax, is_correct, running_aggregate)`: `scores` is this
    /// utterance's per-block LID scores (empty iff skipped, too short / below
    /// `MinNbOfFrames`), `argmax`/`is_correct` are `None` iff skipped, and
    /// `running_aggregate` is the finalized snapshot after folding this utterance. COPY:
    /// the numpy array is read into an owned `Array2<f64>` while the GIL is held, then
    /// the GIL is RELEASED (`Python::detach`) for the fold itself.
    fn push_utterance<'py>(
        &mut self,
        py: Python<'py>,
        feat: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<UtteranceTuple<'py>> {
        let owned = feat.as_array().to_owned();
        let sc = py.detach(|| self.inner.push_utterance(&owned));
        Ok((
            sc.scores,
            sc.argmax,
            sc.is_correct,
            agg_to_tuple(py, sc.running_aggregate),
        ))
    }

    /// The finalized LID result on the utterances folded so far, as
    /// `(classification_errors, confusion, is_lid_correct, predicted_language,
    /// segments_count)` -- a pure snapshot (repeatable/idempotent; the session may keep
    /// folding after a call). The GIL is RELEASED (`Python::detach`) for the read.
    fn finish<'py>(&self, py: Python<'py>) -> LidAggTuple<'py> {
        let agg = py.detach(|| self.inner.finish());
        agg_to_tuple(py, agg)
    }

    /// The number of scored (not skipped) utterances folded so far.
    fn scored_count(&self) -> i32 {
        self.inner.scored_count()
    }
}

#[pymodule]
fn speech_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(build_info, m)?)?;
    m.add_function(wrap_pyfunction!(parse_legacy_config, m)?)?;
    m.add_function(wrap_pyfunction!(load_toml_config, m)?)?;
    m.add_class::<Engine>()?;
    m.add_class::<StreamingSession>()?;
    m.add_class::<StreamingLidSession>()?;
    Ok(())
}
