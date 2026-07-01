//! PyO3 bindings exposing the `speech` engine to Python as module `speech_rs`.
//!
//! Scaffolding: exposes `version()`. The hot-loop `forward_backward` API (Phase 4)
//! will pass numpy weight vectors in and return numpy gradients out (zero-copy).

use pyo3::prelude::*;

/// Version of the underlying `speech` engine crate.
#[pyfunction]
fn version() -> String {
    speech::version().to_string()
}

#[pymodule]
fn speech_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    Ok(())
}
