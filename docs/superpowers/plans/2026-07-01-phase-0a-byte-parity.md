# Phase 0a Byte-Parity Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the weight `.bin` codec, the legacy `.config` parser, and the `config<->nnet` `adim_coeff` seam + flat weight packer in both Rust (crate `speech`) and Python (package `speech`), validated bit-for-bit against real legacy artifacts.

**Architecture:** Two bit-exact goldens drive the work. (1) The `.bin` codec is validated by reading/re-writing the real `NNweights_config1.bin` (33,671 f64). (2) The adim+packer is validated by `BestConfigStruct` (structured config-domain weights, from the `save_net` `.mat`) -> adim -> pack == the paired `nnet_best[:33671]`. A committed scipy prep script extracts the `.mat` into binary fixtures so the tests run in CI without the `.mat`. The `.config` text file carries only sizes + a `weightsFile` pointer, not weights.

**Tech Stack:** Rust (edition 2024, `byteorder`), Python (numpy, scipy for the prep only), uv, pytest, cargo test.

## Global Constraints

- Branch: `feature/phase-0-io-parity`. NEVER commit to `main`. Every commit ends with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- `.bin` layout: `i64` LE nbrows, `i64` LE nbcols, then column-major `f64` LE payload. Hardcode little-endian. `nbrows*nbcols <= 0` is an error.
- Endianness: little-endian only; assert otherwise.
- NNType 0 (BLSTM) only. `io::matfile` stays deferred to Phase 1 (Rust reads no `.mat`; only the Python prep script uses scipy).
- `adim_coeff` lives ONLY at the config<->nnet seam, never in the packer or codec; the bias (last) column is exempt from scaling. `adim(LSTM i) = sqrt(LSTMNeuronNb[i]*LSTMSubSampling[i] + LSTMNeuronNb[i+1])`; `adim(output i) = sqrt(OutputNeuronNb[i]*OutputSubSampling[i])` (NO next term). Use `f64::sqrt` / `math.sqrt`.
- MATLAB `reshape(X',n,1)` = ROW-major flatten of `X`; in numpy use `X.reshape(-1)` (C-order), never `X.T`.
- Reference dims (`1_worker_1.config`): `LSTMNeuronNb=[23,24,24]`, `LSTMSubSampling=[4,1]`, `OutputNeuronNb=[48,12,1]`, `OutputSubSampling=[1,1]`, input feature dim 23; total element count 33,671.
- Fixtures live in `tests/reference_data/phase0/` (committed). Local source (git-ignored): `/Users/govit/Git/Govit/FastSpeechProcessing-legacy/Optimizer_V6.2.2/`.
- ASCII only in code/comments/docs: plain hyphens and straight quotes; no em-dashes/en-dashes/smart quotes.
- Ruff line 160 / py314 / select `["E","F","I","W","UP","B","SIM"]`; mypy strict. Rust `clippy --all-targets -- -D warnings`.
- Spec: `docs/superpowers/specs/2026-07-01-phase-0a-byte-parity-design.md` (sections referenced per task).

---

### Task 1: Weight `.bin` codec (Rust + Python)

The unambiguous foundation: read/write the custom `.bin` matrix format, validated against the real `NNweights_config1.bin`.

**Files:**
- Create: `tests/reference_data/phase0/NNweights_config1.bin` (copied fixture)
- Modify: `src/rust/src/io/binary.rs`
- Modify: `src/python/speech/weight_bridge.py`
- Test: `src/rust/tests/phase0_binary.rs`, `tests/test_weight_bridge_binary.py`

**Interfaces:**
- Produces (Rust): `speech::io::binary::{read_matrix(&Path) -> anyhow::Result<(usize,usize,Vec<f64>)>, write_matrix(&Path, usize, usize, &[f64]) -> anyhow::Result<()>, read_weight_vector(&Path) -> anyhow::Result<Vec<f64>>}` (data is column-major).
- Produces (Python): `speech.weight_bridge.{read_bin(path) -> tuple[int,int,NDArray[np.float64]], write_bin(rows,cols,data,path) -> None, read_weight_vector(path) -> NDArray[np.float64]}`.

- [ ] **Step 1: Vendor the fixture**

Run:
```bash
mkdir -p tests/reference_data/phase0
cp "/Users/govit/Git/Govit/FastSpeechProcessing-legacy/Optimizer_V6.2.2/Executables/15-Oct-2015_BLSTM_OpenSAD15/NNweights_config1.bin" tests/reference_data/phase0/
python3 -c "import os; print(os.path.getsize('tests/reference_data/phase0/NNweights_config1.bin'))"
```
Expected: `269384`.

- [ ] **Step 2: Write the Python failing test** `tests/test_weight_bridge_binary.py`

```python
"""Byte-parity tests for the .bin codec (Python side)."""

from pathlib import Path

import numpy as np

from speech import weight_bridge

FIX = Path("tests/reference_data/phase0/NNweights_config1.bin")


def test_header_and_size() -> None:
    rows, cols, data = weight_bridge.read_bin(FIX)
    assert (rows, cols) == (33671, 1)
    assert data.shape == (33671,)
    assert FIX.stat().st_size == 16 + 8 * rows * cols


def test_roundtrip_bytes(tmp_path: Path) -> None:
    rows, cols, data = weight_bridge.read_bin(FIX)
    out = tmp_path / "rt.bin"
    weight_bridge.write_bin(rows, cols, data, out)
    assert out.read_bytes() == FIX.read_bytes()


def test_read_weight_vector() -> None:
    v = weight_bridge.read_weight_vector(FIX)
    assert v.shape == (33671,)
    assert v.dtype == np.float64
```

- [ ] **Step 3: Run it, expect failure**

Run: `uv run pytest tests/test_weight_bridge_binary.py -q`
Expected: FAIL (functions not defined / `AttributeError`).

- [ ] **Step 4: Implement the Python codec** in `src/python/speech/weight_bridge.py` (replace the stub bodies for the `.bin` functions; keep the module docstring)

```python
"""Neural-net weight (de)serialization: canonical flat vector <-> .bin.

Ported from legacy MATLAB: WriteMatrix2Binary.m / ReadMatrixFromBinary.m and the
config2weights.m packer. The .bin is the custom format (i64 LE rows, i64 LE cols,
then column-major f64 LE), NOT matio. See design spec sections 3, 6.
"""

import struct
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def read_bin(path: Path) -> tuple[int, int, NDArray[np.float64]]:
    """Read the custom .bin matrix; returns (rows, cols, column-major f64 vector)."""
    raw = Path(path).read_bytes()
    rows, cols = struct.unpack("<qq", raw[:16])
    if rows * cols <= 0:
        raise ValueError(f"empty/invalid .bin: rows={rows} cols={cols}")
    data = np.frombuffer(raw[16 : 16 + 8 * rows * cols], dtype="<f8")
    if data.shape[0] != rows * cols:
        raise ValueError("truncated .bin payload")
    return rows, cols, np.array(data, dtype=np.float64)


def write_bin(rows: int, cols: int, data: NDArray[np.float64], path: Path) -> None:
    """Write rows/cols as i64 LE then column-major f64 LE (data is already column-major)."""
    flat = np.ascontiguousarray(data, dtype="<f8").reshape(-1)
    if flat.shape[0] != rows * cols:
        raise ValueError("data length does not match rows*cols")
    with Path(path).open("wb") as fh:
        fh.write(struct.pack("<qq", rows, cols))
        fh.write(flat.tobytes())


def read_weight_vector(path: Path) -> NDArray[np.float64]:
    """Read an N x 1 weight .bin as a flat vector (asserts cols == 1)."""
    rows, cols, data = read_bin(path)
    if cols != 1:
        raise ValueError(f"expected a column vector, got cols={cols}")
    return data
```

- [ ] **Step 5: Run Python tests, expect pass**

Run: `uv run pytest tests/test_weight_bridge_binary.py -q`
Expected: `3 passed`.

- [ ] **Step 6: Write the Rust failing test** `src/rust/tests/phase0_binary.rs`

```rust
//! Byte-parity tests for the .bin codec (Rust side).

use std::path::PathBuf;

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/reference_data/phase0/NNweights_config1.bin")
}

#[test]
fn header_and_size() {
    let (rows, cols, data) = speech::io::binary::read_matrix(&fixture()).unwrap();
    assert_eq!((rows, cols), (33671, 1));
    assert_eq!(data.len(), 33671);
    let size = std::fs::metadata(fixture()).unwrap().len();
    assert_eq!(size, 16 + 8 * rows as u64 * cols as u64);
}

#[test]
fn roundtrip_bytes() {
    let (rows, cols, data) = speech::io::binary::read_matrix(&fixture()).unwrap();
    let tmp = std::env::temp_dir().join("speech_rt_config1.bin");
    speech::io::binary::write_matrix(&tmp, rows, cols, &data).unwrap();
    assert_eq!(std::fs::read(&tmp).unwrap(), std::fs::read(fixture()).unwrap());
}

#[test]
fn empty_guard() {
    let tmp = std::env::temp_dir().join("speech_empty.bin");
    std::fs::write(&tmp, [0u8; 16]).unwrap(); // rows=0, cols=0
    assert!(speech::io::binary::read_matrix(&tmp).is_err());
}
```

- [ ] **Step 7: Run it, expect failure**

Run: `cd src/rust && cargo test --test phase0_binary`
Expected: FAIL (functions are `todo!()` stubs / do not exist).

- [ ] **Step 8: Implement the Rust codec** in `src/rust/src/io/binary.rs` (replace the stub; keep the module doc-comment)

```rust
//! Weight .bin codec: i64 LE rows, i64 LE cols, then column-major f64 LE payload.
//!
//! Ported from legacy C++: Helpers.hpp (BinaryFile2Vector / Matrix2BinaryFile).
//! The MATLAB/Python seam. Some .mat-named weight files are actually this format.

use std::io::{Read, Write};
use std::path::Path;

use anyhow::{bail, Context};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

/// Read the custom .bin matrix; returns (rows, cols, column-major f64 data).
pub fn read_matrix(path: &Path) -> anyhow::Result<(usize, usize, Vec<f64>)> {
    let mut fh = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let rows = fh.read_i64::<LittleEndian>()?;
    let cols = fh.read_i64::<LittleEndian>()?;
    if rows <= 0 || cols <= 0 {
        bail!("empty/invalid .bin: rows={rows} cols={cols}");
    }
    let n = (rows as usize).checked_mul(cols as usize).context("rows*cols overflow")?;
    let mut data = vec![0.0_f64; n];
    fh.read_f64_into::<LittleEndian>(&mut data)?;
    Ok((rows as usize, cols as usize, data))
}

/// Write rows/cols as i64 LE then column-major f64 LE (data is already column-major).
pub fn write_matrix(path: &Path, rows: usize, cols: usize, data: &[f64]) -> anyhow::Result<()> {
    if data.len() != rows * cols {
        bail!("data length {} != rows*cols {}", data.len(), rows * cols);
    }
    let mut fh = std::io::BufWriter::new(std::fs::File::create(path)?);
    fh.write_i64::<LittleEndian>(rows as i64)?;
    fh.write_i64::<LittleEndian>(cols as i64)?;
    for &x in data {
        fh.write_f64::<LittleEndian>(x)?;
    }
    fh.flush()?;
    Ok(())
}

/// Read an N x 1 weight .bin as a flat vector (asserts cols == 1).
pub fn read_weight_vector(path: &Path) -> anyhow::Result<Vec<f64>> {
    let (_rows, cols, data) = read_matrix(path)?;
    if cols != 1 {
        bail!("expected a column vector, got cols={cols}");
    }
    Ok(data)
}
```

Add `byteorder` if not present: it is already in `src/rust/Cargo.toml` `[dependencies]` from the scaffold. Confirm `use` compiles.

- [ ] **Step 9: Run Rust tests, expect pass**

Run: `cd src/rust && cargo test --test ph0_binary 2>/dev/null; cargo test --test phase0_binary`
Expected: `3 passed`.

- [ ] **Step 10: Lint + commit**

```bash
cd src/rust && cargo clippy --all-targets -- -D warnings && cargo fmt --all && cd ../..
uv run ruff check --config=pyproject.toml src/python tests && uv run ruff format --config=pyproject.toml src/python tests
uv run mypy --config-file pyproject.toml src/python tests
git add src/rust/src/io/binary.rs src/rust/tests/phase0_binary.rs src/python/speech/weight_bridge.py tests/test_weight_bridge_binary.py tests/reference_data/phase0/NNweights_config1.bin
git commit -m "feat(phase0a): weight .bin codec (rust + python), byte-parity vs NNweights_config1.bin

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Legacy `.config` parser -> NnetSpec (Rust + Python)

Parse the legacy plain-text config into the typed `NnetSpec` (sizes + peephole flags + cost keys). No weights come from here.

**Files:**
- Create: `tests/reference_data/phase0/1_worker_1.config` (copied fixture)
- Modify: `src/rust/src/legacy_config.rs`, `src/rust/src/config.rs`
- Modify: `src/python/speech/config_bridge.py`
- Test: `src/rust/tests/phase0_config.rs`, `tests/test_config_bridge.py`

**Interfaces:**
- Produces (Rust): `speech::legacy_config::parse_legacy_config(&str) -> indexmap::IndexMap<String,String>` (last-wins); `speech::config::NnetSpec { lstm_neuron_nb: Vec<usize>, lstm_subsampling: Vec<usize>, output_neuron_nb: Vec<usize>, output_subsampling: Vec<usize>, input_size: usize, peepholes: [bool;6] }` with `NnetSpec::from_legacy(&IndexMap<String,String>, prefix: &str) -> anyhow::Result<NnetSpec>`.
- Produces (Python): `speech.config_bridge.parse_legacy_config(text: str) -> dict[str,str]`; `speech.config_bridge.nnet_spec(cfg: dict, prefix: str="BLSTM") -> dict` with keys `LSTMNeuronNb`, `LSTMSubSampling`, `OutputNeuronNb`, `OutputSubSampling`, `NNetInputSize`, `peepholes` (list[bool] len 6), `CostLawSpeech`, `CostLawNoSpeech`.

- [ ] **Step 1: Vendor the config fixture**

```bash
cp "/Users/govit/Git/Govit/FastSpeechProcessing-legacy/Optimizer_V6.2.2/Executables/15-Oct-2015_BLSTM_OpenSAD15/1_worker_1.config" tests/reference_data/phase0/
```

- [ ] **Step 2: Python failing test** `tests/test_config_bridge.py`

```python
"""Legacy .config parsing tests (Python side)."""

from pathlib import Path

from speech import config_bridge

FIX = Path("tests/reference_data/phase0/1_worker_1.config")


def test_parse_last_wins() -> None:
    cfg = config_bridge.parse_legacy_config("a 1\n# c 2\na 3\n")
    assert cfg["a"] == "3"
    assert "c" not in cfg


def test_nnet_spec_from_real_config() -> None:
    cfg = config_bridge.parse_legacy_config(FIX.read_text())
    spec = config_bridge.nnet_spec(cfg, prefix="BLSTM")
    assert spec["LSTMNeuronNb"] == [23, 24, 24]
    assert spec["LSTMSubSampling"] == [4, 1]
    assert spec["OutputNeuronNb"] == [48, 12, 1]
    assert spec["OutputSubSampling"] == [1, 1]
    assert spec["NNetInputSize"] == 23
    assert spec["peepholes"] == [True, True, True, True, True, True]
    assert spec["CostLawSpeech"] == "log"
    assert spec["CostLawNoSpeech"] == "log"
```

- [ ] **Step 3: Run, expect failure.** `uv run pytest tests/test_config_bridge.py -q` -> FAIL.

- [ ] **Step 4: Implement the Python parser** in `src/python/speech/config_bridge.py` (replace stub bodies; keep the docstring)

```python
"""Config writer/reader: legacy .config <-> typed spec.

Ported from legacy MATLAB: readConfig.m / printConfig.m and C++ ConfigFile.cpp.
The .config is plain-text KEY value, last-wins, comma-vectors, # comments,
_-in-key rest-of-line continuation. See design spec section 4.
"""


def parse_legacy_config(text: str) -> dict[str, str]:
    """Parse legacy KEY value config text; last duplicate key wins, # lines skipped."""
    out: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        name = parts[0]
        if name.startswith("#"):
            continue
        out[name] = parts[1] if len(parts) > 1 else ""
    return out


def _ints(cfg: dict[str, str], key: str) -> list[int]:
    return [int(x) for x in cfg[key].split(",")]


def nnet_spec(cfg: dict[str, str], prefix: str = "BLSTM") -> dict[str, object]:
    """Extract the NNType-0 network spec from a parsed legacy config."""
    p = f"{prefix}_"
    peephole_keys = [
        f"{p}Forward_IsCellsPeepholesActive",
        f"{p}Backward_IsCellsPeepholesActive",
        f"{p}Forward_IsGatesPeepholesActive",
        f"{p}Backward_IsGatesPeepholesActive",
        f"{p}Forward_IsGatesRecurrentPeepholesActive",
        f"{p}Backward_IsGatesRecurrentPeepholesActive",
    ]
    return {
        "LSTMNeuronNb": _ints(cfg, f"{p}LSTMNeuronNb"),
        "LSTMSubSampling": _ints(cfg, f"{p}LSTMSubSampling"),
        "OutputNeuronNb": _ints(cfg, f"{p}OutputNeuronNb"),
        "OutputSubSampling": _ints(cfg, f"{p}OutputSubSampling"),
        "NNetInputSize": int(cfg[f"{p}NNetInputSize"]),
        "peepholes": [cfg[k] == "true" for k in peephole_keys],
        "CostLawSpeech": cfg[f"{p}CostLawSpeech"],
        "CostLawNoSpeech": cfg[f"{p}CostLawNoSpeech"],
    }
```

Note: the legacy `_`-continuation rule only affects multi-token values; none of the 0a keys use it, so the simple `split(None, 1)` parser above is sufficient and the test proves it against the real config. (Do NOT add continuation handling speculatively.)

- [ ] **Step 5: Run, expect pass.** `uv run pytest tests/test_config_bridge.py -q` -> `2 passed`.

- [ ] **Step 6: Rust failing test** `src/rust/tests/phase0_config.rs`

```rust
//! Legacy .config parsing tests (Rust side).

use std::path::PathBuf;

fn cfg_text() -> String {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/reference_data/phase0/1_worker_1.config");
    std::fs::read_to_string(p).unwrap()
}

#[test]
fn parse_last_wins() {
    let m = speech::legacy_config::parse_legacy_config("a 1\n# c 2\na 3\n");
    assert_eq!(m.get("a").map(String::as_str), Some("3"));
    assert!(!m.contains_key("c"));
}

#[test]
fn nnet_spec_from_real_config() {
    let m = speech::legacy_config::parse_legacy_config(&cfg_text());
    let spec = speech::config::NnetSpec::from_legacy(&m, "BLSTM").unwrap();
    assert_eq!(spec.lstm_neuron_nb, vec![23, 24, 24]);
    assert_eq!(spec.lstm_subsampling, vec![4, 1]);
    assert_eq!(spec.output_neuron_nb, vec![48, 12, 1]);
    assert_eq!(spec.output_subsampling, vec![1, 1]);
    assert_eq!(spec.input_size, 23);
    assert_eq!(spec.peepholes, [true; 6]);
}
```

- [ ] **Step 7: Run, expect failure.** `cd src/rust && cargo test --test phase0_config` -> FAIL.

- [ ] **Step 8: Implement Rust `legacy_config.rs`** (replace stub)

```rust
//! Importer for the legacy whitespace .config format.
//!
//! Ported from legacy C++: ConfigFile.cpp. KEY value, last-wins, # comments.
//! The 0a keys do not use the _-continuation rule, so a simple split suffices.

use indexmap::IndexMap;

/// Parse legacy KEY value config text; last duplicate key wins, # lines skipped.
pub fn parse_legacy_config(text: &str) -> IndexMap<String, String> {
    let mut out = IndexMap::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let mut it = line.splitn(2, char::is_whitespace);
        let name = it.next().unwrap_or("");
        if name.starts_with('#') || name.is_empty() {
            continue;
        }
        let val = it.next().unwrap_or("").trim().to_string();
        out.insert(name.to_string(), val);
    }
    out
}
```

- [ ] **Step 9: Implement Rust `config.rs` NnetSpec** (replace the minimal scaffold `Config`; keep it compiling)

```rust
//! Canonical config model + the config<->nnet adim_coeff seam + flat weight packer.
//!
//! Ported from legacy MATLAB config2network/config2weights and C++ ConfigFile.
//! See design spec sections 4-7.

use anyhow::Context;
use indexmap::IndexMap;

/// NNType-0 network spec parsed from a legacy config.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NnetSpec {
    pub lstm_neuron_nb: Vec<usize>,
    pub lstm_subsampling: Vec<usize>,
    pub output_neuron_nb: Vec<usize>,
    pub output_subsampling: Vec<usize>,
    pub input_size: usize,
    pub peepholes: [bool; 6],
}

fn ints(m: &IndexMap<String, String>, key: &str) -> anyhow::Result<Vec<usize>> {
    m.get(key)
        .with_context(|| format!("missing key {key}"))?
        .split(',')
        .map(|s| s.trim().parse::<usize>().map_err(Into::into))
        .collect()
}

impl NnetSpec {
    /// Extract the spec from a parsed legacy config using the given algName prefix (e.g. "BLSTM").
    pub fn from_legacy(m: &IndexMap<String, String>, prefix: &str) -> anyhow::Result<NnetSpec> {
        let p = format!("{prefix}_");
        let flag = |k: &str| m.get(k).map(|v| v == "true").unwrap_or(false);
        Ok(NnetSpec {
            lstm_neuron_nb: ints(m, &format!("{p}LSTMNeuronNb"))?,
            lstm_subsampling: ints(m, &format!("{p}LSTMSubSampling"))?,
            output_neuron_nb: ints(m, &format!("{p}OutputNeuronNb"))?,
            output_subsampling: ints(m, &format!("{p}OutputSubSampling"))?,
            input_size: m
                .get(&format!("{p}NNetInputSize"))
                .context("missing NNetInputSize")?
                .trim()
                .parse()?,
            peepholes: [
                flag(&format!("{p}Forward_IsCellsPeepholesActive")),
                flag(&format!("{p}Backward_IsCellsPeepholesActive")),
                flag(&format!("{p}Forward_IsGatesPeepholesActive")),
                flag(&format!("{p}Backward_IsGatesPeepholesActive")),
                flag(&format!("{p}Forward_IsGatesRecurrentPeepholesActive")),
                flag(&format!("{p}Backward_IsGatesRecurrentPeepholesActive")),
            ],
        })
    }
}
```

Remove the old scaffold `Config`/`EngineConfig`/`one()` from `config.rs` and update the smoke test in `src/rust/tests/smoke.rs` that referenced `Config::from_toml_str` (delete that one test case `config_parses_minimal_toml`, since the minimal TOML `Config` is superseded by the real model; the other smoke tests stay). Confirm `cargo test` still green.

- [ ] **Step 10: Run Rust tests, expect pass.** `cd src/rust && cargo test --test phase0_config` -> `2 passed`; `cargo test` overall green.

- [ ] **Step 11: Lint + commit**

```bash
cd src/rust && cargo clippy --all-targets -- -D warnings && cargo fmt --all && cd ../..
uv run ruff check --config=pyproject.toml src/python tests && uv run ruff format --config=pyproject.toml src/python tests && uv run mypy --config-file pyproject.toml src/python tests
git add src/rust/src/legacy_config.rs src/rust/src/config.rs src/rust/tests/phase0_config.rs src/rust/tests/smoke.rs src/python/speech/config_bridge.py tests/test_config_bridge.py tests/reference_data/phase0/1_worker_1.config
git commit -m "feat(phase0a): legacy .config parser -> NnetSpec (rust + python)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Python adim + packer and the fixture prep (the crux)

Implement `config_to_nnet` (adim) + `nnet_to_flat` (13-block packer) + `flat_to_nnet` + `element_count` in numpy, and the scipy prep script that extracts `BestConfigStruct` + `nnet_best` from the `.mat` into committed fixtures. The prep script's assertion `pack(config_to_nnet(structured)) == nnet_best[:33671]` is the ground-truth that nails the packing; the emitted fixtures let CI re-check it without scipy/the `.mat`.

**Files:**
- Modify: `src/python/speech/weight_bridge.py` (add packer functions)
- Create: `scripts/extract_phase0_fixtures.py`
- Create (generated, committed): `tests/reference_data/phase0/{best_config_domain.bin, best_config_manifest.json, best_net_flat.bin}`
- Test: `tests/test_weight_bridge_packer.py`

**Interfaces:**
- Consumes: `weight_bridge.{read_bin, read_weight_vector}` (Task 1), `config_bridge.nnet_spec` (Task 2).
- Produces (Python): `weight_bridge.element_count(spec) -> int`; `weight_bridge.load_structured(manifest_path, bin_path) -> Structured` (dict of named f64 rows); `weight_bridge.config_to_nnet(structured, spec) -> Nnet`; `weight_bridge.nnet_to_flat(nnet, spec) -> NDArray`; `weight_bridge.flat_to_nnet(flat, spec) -> Nnet`. `Structured`/`Nnet` are plain dicts documented in code. The manifest is `{"spec": {...}, "rows": [{"name": str, "len": int}, ...]}`; `best_config_domain.bin` is a `1 x sum(len)` `.bin` holding the concatenated config-domain rows in manifest order.

- [ ] **Step 1: Add `element_count` + a failing test**

Append to `tests/test_weight_bridge_packer.py`:

```python
"""adim + packer byte-parity tests (Python)."""

from pathlib import Path

import numpy as np

from speech import config_bridge, weight_bridge

REF = Path("tests/reference_data/phase0")


def _spec() -> dict:
    cfg = config_bridge.parse_legacy_config((REF / "1_worker_1.config").read_text())
    return config_bridge.nnet_spec(cfg, prefix="BLSTM")


def test_element_count() -> None:
    assert weight_bridge.element_count(_spec()) == 33671
    rows, _, _ = weight_bridge.read_bin(REF / "NNweights_config1.bin")
    assert rows == 33671
```

- [ ] **Step 2: Run, expect failure.** `uv run pytest tests/test_weight_bridge_packer.py::test_element_count -q` -> FAIL (`element_count` missing).

- [ ] **Step 3: Implement `element_count` + the packer in `weight_bridge.py`**

Add to `src/python/speech/weight_bridge.py`:

```python
import math

# A "row" is a config-domain gate/neuron vector of length ncols.
# Structured = {"forward": [layer -> {"input"/"forget"/"output"/"cell": 2D (out x ncols)}],
#               "backward": [...same...],
#               "output": [layer -> 2D (out x in+1)],
#               "mean": 1D, "std": 1D}
# Nnet has the same shape but nnet-domain (adim applied) matrices.


def element_count(spec: dict) -> int:
    lstm = spec["LSTMNeuronNb"]
    lsub = spec["LSTMSubSampling"]
    outn = spec["OutputNeuronNb"]
    total = 0
    for i in range(len(lstm) - 1):
        out = lstm[i + 1]
        fin = lstm[i] * lsub[i]
        total += 2 * (4 * out * fin + 4 * out * out + 12 * out + 4 * out)
    for i in range(len(outn) - 1):
        total += outn[i + 1] * outn[i] + outn[i + 1]
    total += 2 * lstm[0]
    return total


def _lstm_adim(spec: dict, i: int) -> float:
    return math.sqrt(spec["LSTMNeuronNb"][i] * spec["LSTMSubSampling"][i] + spec["LSTMNeuronNb"][i + 1])


def _out_adim(spec: dict, i: int) -> float:
    return math.sqrt(spec["OutputNeuronNb"][i] * spec["OutputSubSampling"][i])


def _apply_adim(mat: "np.ndarray", adim: float) -> "np.ndarray":
    # divide every column by adim EXCEPT the last (bias) column.
    out = mat / adim
    out[:, -1] = mat[:, -1]
    return out


def config_to_nnet(structured: dict, spec: dict) -> dict:
    """Config-domain structured rows -> nnet-domain matrices (adim decode)."""
    nnet: dict = {"forward": [], "backward": [], "output": [], "mean": structured["mean"], "std": structured["std"]}
    for direction in ("forward", "backward"):
        for i, layer in enumerate(structured[direction]):
            adim = _lstm_adim(spec, i)
            nnet[direction].append({g: _apply_adim(layer[g], adim) for g in ("input", "forget", "output", "cell")})
    for i, layer in enumerate(structured["output"]):
        nnet["output"].append(_apply_adim(layer, _out_adim(spec, i)))
    return nnet


def _pack_lstm_layer(gates: dict, out: int, input_size: int) -> list["np.ndarray"]:
    ncols = input_size + out + 5
    idx_in = list(range(out, out + input_size))
    idx_rec = list(range(0, out))
    iw, fw, ow, cw = gates["input"], gates["forget"], gates["output"], gates["cell"]
    parts: list = []
    # 1-4 fan-in blocks, row-major
    for m in (iw, fw, ow, cw):
        parts.append(m[:, idx_in].reshape(-1))
    # 5-8 recurrent blocks
    for m in (iw, fw, ow, cw):
        parts.append(m[:, idx_rec].reshape(-1))
    # 9 peephole bundle (out x 12), row-major
    peep = np.empty((out, 12), dtype=np.float64)
    for r in range(out):
        peep[r] = [
            iw[r, ncols - 2], fw[r, ncols - 2], ow[r, ncols - 2],
            iw[r, ncols - 5], iw[r, ncols - 4], iw[r, ncols - 3],
            fw[r, ncols - 5], fw[r, ncols - 4], fw[r, ncols - 3],
            ow[r, ncols - 5], ow[r, ncols - 4], ow[r, ncols - 3],
        ]
    parts.append(peep.reshape(-1))
    # 10-13 biases (last col)
    for m in (iw, fw, ow, cw):
        parts.append(m[:, ncols - 1].reshape(-1))
    return parts


def nnet_to_flat(nnet: dict, spec: dict) -> "np.ndarray":
    lstm = spec["LSTMNeuronNb"]
    parts: list = []
    for direction in ("forward", "backward"):
        for i, layer in enumerate(nnet[direction]):
            parts.extend(_pack_lstm_layer(layer, out=lstm[i + 1], input_size=lstm[i] * spec["LSTMSubSampling"][i]))
    for layer in nnet["output"]:
        parts.append(layer[:, :-1].reshape(-1))  # weights
        parts.append(layer[:, -1].reshape(-1))  # bias
    parts.append(np.asarray(nnet["mean"], dtype=np.float64).reshape(-1))
    parts.append(np.asarray(nnet["std"], dtype=np.float64).reshape(-1))
    return np.concatenate(parts)
```

The peephole column indices (`ncols-2` = recurrent-peephole `end-1`; `ncols-5..ncols-3` = the three input-gate-peephole `end-4:end-2`; `ncols-1` = bias `end`) follow design spec section 6. The exact 12-column order and the bias-exempt adim are the two things the golden assertion in Step 5 verifies; if it fails, bisect with the element-count and tail checks below.

- [ ] **Step 4: Run `element_count` test, expect pass.** `uv run pytest tests/test_weight_bridge_packer.py::test_element_count -q` -> `1 passed`.

- [ ] **Step 5: Write the fixture prep script** `scripts/extract_phase0_fixtures.py`

```python
"""Extract Phase-0a golden fixtures from the legacy save_net .mat (scipy).

Loads BestConfigStruct (structured config-domain weight rows) and nnet_best
(paired nnet-domain flat vector), asserts pack(config_to_nnet(structured)) ==
nnet_best[:33671] bit-for-bit, and emits committed fixtures:
  tests/reference_data/phase0/best_config_domain.bin   (1 x N config-domain rows, concatenated)
  tests/reference_data/phase0/best_config_manifest.json (spec + row names/lengths)
  tests/reference_data/phase0/best_net_flat.bin         (nnet_best[:33671], N x 1)

Usage: uv run --with scipy python scripts/extract_phase0_fixtures.py
Requires the local git-ignored .mat.
"""

import json
from pathlib import Path

import numpy as np
import scipy.io as sio

from speech import weight_bridge

MAT = Path("/Users/govit/Git/Govit/FastSpeechProcessing-legacy/Optimizer_V6.2.2/save_net/LSTM_15-Oct-2015_BLSTM_OpenSAD15.mat")
OUT = Path("tests/reference_data/phase0")
PREFIX = "AlgName"  # BestConfigStruct uses the literal AlgName_ prefix


def field(bcs: object, name: str) -> np.ndarray:
    return np.atleast_1d(np.asarray(getattr(bcs, name), dtype=np.float64)).reshape(-1)


def main() -> None:
    m = sio.loadmat(MAT, struct_as_record=False, squeeze_me=True)
    bcs = m["BestConfigStruct"]
    nnet_best = np.asarray(m["nnet_best"], dtype=np.float64).reshape(-1)

    p = f"{PREFIX}_"
    lstm = [int(x) for x in np.atleast_1d(getattr(bcs, f"{p}LSTMNeuronNb")).reshape(-1)]
    lsub = [int(x) for x in np.atleast_1d(getattr(bcs, f"{p}LSTMSubSampling")).reshape(-1)]
    outn = [int(x) for x in np.atleast_1d(getattr(bcs, f"{p}OutputNeuronNb")).reshape(-1)]
    osub = [int(x) for x in np.atleast_1d(getattr(bcs, f"{p}OutputSubSampling")).reshape(-1)]
    spec = {"LSTMNeuronNb": lstm, "LSTMSubSampling": lsub, "OutputNeuronNb": outn, "OutputSubSampling": osub, "NNetInputSize": lstm[0]}

    n = weight_bridge.element_count(spec)
    assert nnet_best[n:].sum() == 0.0, "nnet_best has nonzero padding past the real length"
    flat_golden = nnet_best[:n].copy()

    # Assemble structured config-domain rows + record the manifest.
    rows: list[dict] = []
    blob: list[np.ndarray] = []

    def emit(name: str, vec: np.ndarray) -> None:
        rows.append({"name": name, "len": int(vec.shape[0])})
        blob.append(vec)

    structured: dict = {"forward": [], "backward": [], "output": []}
    for direction in ("Forward", "Backward"):
        key = direction.lower()
        for i in range(len(lstm) - 1):
            out = lstm[i + 1]
            gates: dict = {}
            for gate, gk in (("input", "InputGateWeights"), ("forget", "ForgetGateWeights"), ("output", "OutputGateWeights"), ("cell", "CellWeight")):
                mat_rows = [field(bcs, f"{p}{direction}_Layer_{i}_LSTMBlock_{b}_{gk}") for b in range(out)]
                mat = np.stack(mat_rows, axis=0)
                gates[gate] = mat
                for b in range(out):
                    emit(f"{key}.L{i}.B{b}.{gate}", mat_rows[b])
            structured[key].append(gates)
    for i in range(len(outn) - 1):
        neurons = [field(bcs, f"{p}Output_Layer_{i}_Neuron_{nn}_Weights") for nn in range(outn[i + 1])]
        mat = np.stack(neurons, axis=0)
        structured["output"].append(mat)
        for nn in range(outn[i + 1]):
            emit(f"output.L{i}.N{nn}", neurons[nn])
    structured["mean"] = field(bcs, f"{p}NormalizeInputMean")
    structured["std"] = field(bcs, f"{p}NormalizeInputStd")
    emit("mean", structured["mean"])
    emit("std", structured["std"])

    # GROUND TRUTH: adim + pack must reproduce nnet_best exactly.
    flat = weight_bridge.nnet_to_flat(weight_bridge.config_to_nnet(structured, spec), spec)
    assert flat.shape[0] == n, f"packed length {flat.shape[0]} != {n}"
    assert np.array_equal(flat, flat_golden), "PACK MISMATCH vs nnet_best - adim/order/reshape bug"

    OUT.mkdir(parents=True, exist_ok=True)
    concat = np.concatenate(blob)
    weight_bridge.write_bin(1, int(concat.shape[0]), concat, OUT / "best_config_domain.bin")
    weight_bridge.write_bin(n, 1, flat_golden, OUT / "best_net_flat.bin")
    (OUT / "best_config_manifest.json").write_text(json.dumps({"spec": spec, "rows": rows}, indent=1))
    print(f"OK: packed {n} elems match nnet_best; wrote fixtures to {OUT}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run the prep script (the ground-truth gate)**

Run: `uv run --with scipy python scripts/extract_phase0_fixtures.py`
Expected: `OK: packed 33671 elems match nnet_best; wrote fixtures to tests/reference_data/phase0`.

If it prints `PACK MISMATCH`: the adim/order/reshape has a bug. Bisect: (a) confirm `element_count` == 33671 and `nnet_best[33671:]` is all zero; (b) check the first forward layer's first block against `nnet_best[:8832]` region size (4*24*92=8832 fan-in elems come first); (c) verify the peephole 12-col order and that adim leaves the bias column unchanged; (d) verify `reshape(-1)` (C-order) is used, never `.T`. Fix `weight_bridge.py` until the assertion passes. Do NOT relax the assertion.

- [ ] **Step 7: Add `load_structured` + `flat_to_nnet` + the golden tests**

Add to `weight_bridge.py`:

```python
def load_structured(manifest_path: Path, bin_path: Path) -> tuple[dict, dict]:
    """Load (structured, spec) from the committed manifest + concatenated-rows .bin."""
    manifest = json.loads(Path(manifest_path).read_text())
    spec = manifest["spec"]
    _, _, blob = read_bin(bin_path)
    named: dict[str, np.ndarray] = {}
    off = 0
    for r in manifest["rows"]:
        named[r["name"]] = blob[off : off + r["len"]]
        off += r["len"]
    lstm, outn = spec["LSTMNeuronNb"], spec["OutputNeuronNb"]
    structured: dict = {"forward": [], "backward": [], "output": [], "mean": named["mean"], "std": named["std"]}
    for direction in ("forward", "backward"):
        for i in range(len(lstm) - 1):
            out = lstm[i + 1]
            structured[direction].append({g: np.stack([named[f"{direction}.L{i}.B{b}.{g}"] for b in range(out)], 0) for g in ("input", "forget", "output", "cell")})
    for i in range(len(outn) - 1):
        structured["output"].append(np.stack([named[f"output.L{i}.N{nn}"] for nn in range(outn[i + 1])], 0))
    return structured, spec


def flat_to_nnet(flat: "np.ndarray", spec: dict) -> dict:
    """Inverse of nnet_to_flat: slice the flat vector back into nnet-domain matrices."""
    lstm, lsub, outn = spec["LSTMNeuronNb"], spec["LSTMSubSampling"], spec["OutputNeuronNb"]
    pos = 0
    nnet: dict = {"forward": [], "backward": [], "output": []}

    def take(k: int) -> "np.ndarray":
        nonlocal pos
        seg = flat[pos : pos + k]
        pos += k
        return seg

    for direction in ("forward", "backward"):
        for i in range(len(lstm) - 1):
            out, fin = lstm[i + 1], lstm[i] * lsub[i]
            ncols = fin + out + 5
            gates = {g: np.zeros((out, ncols)) for g in ("input", "forget", "output", "cell")}
            for g in ("input", "forget", "output", "cell"):
                gates[g][:, out : out + fin] = take(out * fin).reshape(out, fin)
            for g in ("input", "forget", "output", "cell"):
                gates[g][:, 0:out] = take(out * out).reshape(out, out)
            peep = take(12 * out).reshape(out, 12)
            for r in range(out):
                gates["input"][r, ncols - 2], gates["forget"][r, ncols - 2], gates["output"][r, ncols - 2] = peep[r, 0], peep[r, 1], peep[r, 2]
                gates["input"][r, ncols - 5 : ncols - 2] = peep[r, 3:6]
                gates["forget"][r, ncols - 5 : ncols - 2] = peep[r, 6:9]
                gates["output"][r, ncols - 5 : ncols - 2] = peep[r, 9:12]
            for g in ("input", "forget", "output", "cell"):
                gates[g][:, ncols - 1] = take(out)
            nnet[direction].append(gates)
    for i in range(len(outn) - 1):
        out, inp = outn[i + 1], outn[i]
        mat = np.zeros((out, inp + 1))
        mat[:, :inp] = take(out * inp).reshape(out, inp)
        mat[:, inp] = take(out)
        nnet["output"].append(mat)
    nnet["mean"] = take(lstm[0])
    nnet["std"] = take(lstm[0])
    return nnet
```

Add golden tests to `tests/test_weight_bridge_packer.py`:

```python
def test_full_pipeline_matches_flat() -> None:
    structured, spec = weight_bridge.load_structured(REF / "best_config_manifest.json", REF / "best_config_domain.bin")
    flat = weight_bridge.nnet_to_flat(weight_bridge.config_to_nnet(structured, spec), spec)
    golden = weight_bridge.read_weight_vector(REF / "best_net_flat.bin")
    assert np.array_equal(flat, golden)


def test_packer_bijection_on_real_bin() -> None:
    spec = _spec()
    flat = weight_bridge.read_weight_vector(REF / "NNweights_config1.bin")
    rt = weight_bridge.nnet_to_flat(weight_bridge.flat_to_nnet(flat, spec), spec)
    assert np.array_equal(rt, flat)
    assert (flat[-23:] > 0).all()  # std tail strictly positive
```

- [ ] **Step 8: Run the packer tests, expect pass.** `uv run pytest tests/test_weight_bridge_packer.py -q` -> `4 passed`.

- [ ] **Step 9: Lint + commit** (fixtures included)

```bash
uv run ruff check --config=pyproject.toml src/python tests scripts && uv run ruff format --config=pyproject.toml src/python tests scripts && uv run mypy --config-file pyproject.toml src/python
git add src/python/speech/weight_bridge.py scripts/extract_phase0_fixtures.py tests/test_weight_bridge_packer.py tests/reference_data/phase0/best_config_domain.bin tests/reference_data/phase0/best_config_manifest.json tests/reference_data/phase0/best_net_flat.bin
git commit -m "feat(phase0a): python adim+packer + fixture prep, pack==nnet_best bit-exact

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

(Note: `scripts/` is added to the ruff/format scope here; keep `mypy` scope on `src/python` since the prep script needs scipy which is not a typed dep.)

---

### Task 4: Rust adim + packer (mirror of Task 3, same goldens)

Implement `element_count`, `config_to_nnet`, `nnet_to_flat`, `flat_to_nnet`, and the `Structured`/`Nnet` types + manifest loading in `config.rs`, validated against the SAME committed fixtures. This gives the independent second implementation.

**Files:**
- Modify: `src/rust/src/config.rs`
- Test: `src/rust/tests/phase0_packer.rs`

**Interfaces:**
- Consumes: `NnetSpec` (Task 2), `io::binary::{read_matrix, read_weight_vector}` (Task 1), the committed fixtures (Task 3).
- Produces: `speech::config::{element_count(&NnetSpec) -> usize, load_structured(&Path manifest, &Path bin) -> anyhow::Result<(Structured, NnetSpec)>, config_to_nnet(&Structured, &NnetSpec) -> Nnet, nnet_to_flat(&Nnet, &NnetSpec) -> Vec<f64>, flat_to_nnet(&[f64], &NnetSpec) -> Nnet}`. `Structured`/`Nnet` model per-direction `Vec<LstmLayer{input,forget,output,cell: Vec<f64> row-major out*ncols}>` + output layers + mean/std; document the row-major matrix convention in code.

- [ ] **Step 1: Rust failing tests** `src/rust/tests/phase0_packer.rs`

```rust
//! adim + packer byte-parity tests (Rust), against the same fixtures as Python.

use std::path::PathBuf;

fn ref_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/reference_data/phase0")
}

fn spec() -> speech::config::NnetSpec {
    let text = std::fs::read_to_string(ref_dir().join("1_worker_1.config")).unwrap();
    let m = speech::legacy_config::parse_legacy_config(&text);
    speech::config::NnetSpec::from_legacy(&m, "BLSTM").unwrap()
}

#[test]
fn element_count_is_33671() {
    assert_eq!(speech::config::element_count(&spec()), 33671);
}

#[test]
fn full_pipeline_matches_flat() {
    let (structured, s) = speech::config::load_structured(
        &ref_dir().join("best_config_manifest.json"),
        &ref_dir().join("best_config_domain.bin"),
    )
    .unwrap();
    let flat = speech::config::nnet_to_flat(&speech::config::config_to_nnet(&structured, &s), &s);
    let golden = speech::io::binary::read_weight_vector(&ref_dir().join("best_net_flat.bin")).unwrap();
    assert_eq!(flat, golden);
}

#[test]
fn packer_bijection_on_real_bin() {
    let s = spec();
    let flat = speech::io::binary::read_weight_vector(&ref_dir().join("NNweights_config1.bin")).unwrap();
    let rt = speech::config::nnet_to_flat(&speech::config::flat_to_nnet(&flat, &s), &s);
    assert_eq!(rt, flat);
}
```

- [ ] **Step 2: Run, expect failure.** `cd src/rust && cargo test --test phase0_packer` -> FAIL.

- [ ] **Step 3: Implement the Rust packer in `config.rs`.** Mirror the Python numpy semantics exactly: matrices are `Vec<f64>` in ROW-MAJOR `out x ncols` layout; a block row `b` occupies `[b*ncols .. (b+1)*ncols)`. `config_to_nnet` divides every element by `adim` except column `ncols-1` of each row. `nnet_to_flat` emits, per LSTM layer, the 13 blocks in order: for each of `[input,forget,output,cell]` the `idx_in = out..out+fin` columns row-major; then the same for `idx_rec = 0..out`; then the `out x 12` peephole bundle row-major (`[iw[c-2],fw[c-2],ow[c-2], iw[c-5],iw[c-4],iw[c-3], fw[c-5],fw[c-4],fw[c-3], ow[c-5],ow[c-4],ow[c-3]]` where `c=ncols`); then the four bias columns (`ncols-1`) each `out` long; then output layers (`weights[:,:in]` row-major then bias); then mean, std. `flat_to_nnet` inverts. Use the manifest row order `{direction}.L{i}.B{b}.{gate}`, `output.L{i}.N{nn}`, `mean`, `std` (matching Task 3's prep script) to slice `best_config_domain.bin`.

Manifest JSON is read with `serde_json` (already a dep). Deserialize into `struct Manifest { spec: SpecJson, rows: Vec<RowMeta> }` where `RowMeta { name: String, len: usize }` and `SpecJson` mirrors `NnetSpec`'s vectors. Provide the full implementation of each function; keep functions small and each matrix helper (`row_major index`) a named local. (This is a direct transcription of the Python semantics in Step 3-7 of Task 3; write the equivalent Rust.)

Acceptance: the three tests pass. If `full_pipeline_matches_flat` fails while Python's equivalent passed, the Rust reshape/order diverged from numpy C-order - compare intermediate slice lengths against the Python `nnet_to_flat` `parts` lengths.

- [ ] **Step 4: Run, expect pass.** `cd src/rust && cargo test --test phase0_packer` -> `3 passed`.

- [ ] **Step 5: Lint + full Rust test + commit**

```bash
cd src/rust && cargo clippy --all-targets -- -D warnings && cargo fmt --all -- --check && cargo test && cd ../..
git add src/rust/src/config.rs src/rust/tests/phase0_packer.rs
git commit -m "feat(phase0a): rust adim+packer, byte-parity vs the same goldens

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Docs + scaffolding-allow cleanup

Update the docs to reflect that `io/binary`, `legacy_config`, and `config` are implemented, and narrow the scaffolding lints for these Rust modules.

**Files:**
- Modify: `CLAUDE.md`, `README.md`, `src/rust/src/lib.rs`

- [ ] **Step 1: Narrow the Rust scaffolding allow.** In `src/rust/src/lib.rs`, remove `dead_code`/`unused_variables` from the crate-root `#![allow(...)]` IF the crate still builds clean with `-D warnings` (the implemented modules no longer need it, but stub modules do). If the build fails without it, instead add `#![allow(dead_code)]` only (keep it) and add `#[allow(unused_variables)]` narrowly to the remaining stub modules that need it. Run `cargo clippy --all-targets -- -D warnings` and keep whichever minimal form is green. Document the choice in a one-line comment.

- [ ] **Step 2: Update `CLAUDE.md`.** In the Rust architecture table, change the `io/binary.rs`, `legacy_config.rs`, and `config.rs` rows from stub descriptions to "implemented (Phase 0a): <one line>". In the Python table, update `weight_bridge.py` and `config_bridge.py` similarly. Add a one-line note under Key Decisions that Phase 0a is complete (byte-parity vs `NNweights_config1.bin` and `nnet_best`). ASCII only; verify `LC_ALL=C grep -n '[^ -~]' CLAUDE.md` returns nothing.

- [ ] **Step 3: Update `README.md` roadmap.** Mark Phase 0a done (byte-parity foundation: `.bin` codec + legacy config parser + adim/packer, validated bit-for-bit). Leave Phase 0b and later as pending. ASCII only.

- [ ] **Step 4: Verify + commit**

```bash
grep -E '^## ' CLAUDE.md >/dev/null && LC_ALL=C grep -n '[^ -~]' CLAUDE.md; LC_ALL=C grep -n '[^ -~]' README.md
cd src/rust && cargo clippy --all-targets -- -D warnings && cd ../..
git add CLAUDE.md README.md src/rust/src/lib.rs
git commit -m "docs(phase0a): mark byte-parity foundation implemented; narrow scaffolding lints

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: Full-suite verification + smart-commit

**Files:** none (verification + commit).

- [ ] **Step 1: Run the entire toolchain**

```bash
./check_all.sh
uv run ruff check --config=pyproject.toml src/python tests && uv run ruff format --check --config=pyproject.toml src/python tests
uv run mypy --config-file pyproject.toml src/python tests
uv run pytest tests -q
```
Expected: `check_all.sh` -> `All checks passed!`; ruff/format clean; mypy `Success`; pytest all pass (the Phase-0a tests plus the prior smoke/structure/data tests).

- [ ] **Step 2: Confirm Definition of Done** (spec section 11): both bit-exact goldens pass in Rust and Python; the `.config` parse asserts the real values; element count 33671; fixtures committed under `tests/reference_data/phase0/`; docs updated.

- [ ] **Step 3: Invoke the `smart-commit` skill** over the whole `feature/phase-0-io-parity` branch (sync `CLAUDE.md`/`README.md` with the final state, commit anything outstanding). Do not push.

---

## Self-Review

**1. Spec coverage.** Spec s3 (.bin codec) -> Task 1. s4 (config parser) -> Task 2. s5-7 (adim seam, packer, element count) -> Task 3 (Python) + Task 4 (Rust). s8 (interfaces) -> Tasks 1-4 signatures. s9 (fixtures/tests/acceptance): fixture vendoring -> Tasks 1,2,3; the 9 tests map to Task 1 (tests 1-3), Task 2 (test 4), Task 3/4 (tests 5-9). s2 decision 2 (derived fixtures + prep script) -> Task 3. s10 risks pinned by the golden assertions in Tasks 3-4. s11 DoD -> Task 6. s12 out-of-scope respected (no cost/segmentation/matfile). Covered.

**2. Placeholder scan.** No TBD/TODO. Task 4 Step 3 describes the Rust packer as a transcription of the fully-shown Python semantics (Task 3 Steps 3+7) rather than repeating ~120 lines of near-identical code in the other language; the exact block order, indices, and manifest names are given, and the golden test is the gate. The prep-script assertion and the golden tests are shown in full.

**3. Type consistency.** `NnetSpec` fields (`lstm_neuron_nb`, `lstm_subsampling`, `output_neuron_nb`, `output_subsampling`, `input_size`, `peepholes`) are used identically in Tasks 2 and 4. Python `spec` dict keys (`LSTMNeuronNb`, `LSTMSubSampling`, `OutputNeuronNb`, `OutputSubSampling`, `NNetInputSize`, `peepholes`) are consistent across `config_bridge.nnet_spec`, `weight_bridge.element_count/config_to_nnet`, and the prep script. `read_bin`/`write_bin`/`read_weight_vector`/`read_matrix`/`write_matrix` signatures match between definition (Task 1) and use (Tasks 3-4). The manifest row-name scheme (`{direction}.L{i}.B{b}.{gate}`, `output.L{i}.N{nn}`, `mean`, `std`) is identical in the prep script (Task 3 Step 5), `load_structured` (Task 3 Step 7), and the Rust loader (Task 4 Step 3).
