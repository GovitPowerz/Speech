//! PyO3 bindings exposing the `speech` engine to Python as module `speech_rs`.
//!
//! Phase 4c Task 3: the coarse corpus-level [`Engine`] -- a thin wrapper over
//! `speech::engine::corpus_processor::CorpusProcessor` (the "PyO3 seam surface"
//! landed in Task 1). This is the legacy file/subprocess contract driven
//! in-process: build from config paths, `run()` the corpus fold, then read back
//! the result matrix / weights / derivatives / gradient check.
//!
//! numpy crossings COPY (spec R2): every ndarray/vec returned to Python is a
//! fresh numpy array, and every array passed in is read into an owned `Vec`.
//! The copy site is doc-commented on each method. Zero-copy is a later
//! hot-loop concern.

use numpy::{IntoPyArray, PyArray1, PyArray2, ToPyArray};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use speech::cli::Mode;
use speech::engine::corpus_processor::CorpusProcessor;
use speech::legacy_config::parse_legacy_config as parse_legacy_config_rs;
use speech::toml_config::toml_to_map as toml_to_map_rs;

/// Convert an engine error into a Python `RuntimeError`, formatting the FULL
/// chain via the alternate `Display` (`{:#}`, which `anyhow::Error` renders as
/// its cause chain) so a nested cause (e.g. a missing corpus file) survives.
/// Generic over `Display` to avoid taking a direct `anyhow` dependency.
fn to_pyerr<E: std::fmt::Display>(e: E) -> PyErr {
    PyRuntimeError::new_err(format!("{e:#}"))
}

/// Version of the underlying `speech` engine crate.
#[pyfunction]
fn version() -> String {
    speech::version().to_string()
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
/// corpus contract, driven in-process. Construct from config PATHS + a CLI mode
/// flag, `run()`, then read the results.
#[pyclass]
struct Engine {
    inner: CorpusProcessor,
}

#[pymethods]
impl Engine {
    /// Build from config PATHS + a CLI mode flag (`"-m"`, `"-s"`, ...; parsed
    /// via `cli::Mode::from_flag`). Dispatches by extension like `cli::parse_cli`:
    /// a `.toml` path is read + `toml_config::toml_to_map`'d, anything else is
    /// read + `parse_legacy_config`'d. An empty `config_paths` is NOT checked here
    /// -- `CorpusProcessor::new` already bails with its own "at least one config
    /// is required" message, so a redundant pre-check here would just duplicate
    /// that error surface.
    #[new]
    fn new(config_paths: Vec<String>, mode: String) -> PyResult<Self> {
        let mode = Mode::from_flag(&mode)
            .ok_or_else(|| PyRuntimeError::new_err(format!("unknown mode flag '{mode}'")))?;

        let mut configs = Vec::with_capacity(config_paths.len());
        for path in &config_paths {
            let text = std::fs::read_to_string(path).map_err(|e| {
                PyRuntimeError::new_err(format!("cannot read config file '{path}': {e}"))
            })?;
            let map = if path.ends_with(".toml") {
                toml_to_map_rs(&text).map_err(|e| {
                    PyRuntimeError::new_err(format!("invalid TOML config '{path}': {e:#}"))
                })?
            } else {
                parse_legacy_config_rs(&text)
            };
            configs.push(map);
        }

        let inner = CorpusProcessor::new(configs, mode).map_err(to_pyerr)?;
        Ok(Engine { inner })
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

#[pymodule]
fn speech_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(parse_legacy_config, m)?)?;
    m.add_function(wrap_pyfunction!(load_toml_config, m)?)?;
    m.add_class::<Engine>()?;
    Ok(())
}
