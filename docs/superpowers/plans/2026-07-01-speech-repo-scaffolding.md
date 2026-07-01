# Speech Repo Scaffolding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the `Speech` repo as a green-compiling Rust+Python skeleton mirroring `Aerocapture`, plus `CLAUDE.md` and `README.md`, ready for the phased port of the legacy `FastSpeechProcessing` LID+SAD engine.

**Architecture:** A Rust core crate `speech` (fast engine: audio, features, NN, tasks, I/O) with a PyO3 binding crate `speech-py` (module `speech_rs`), and a Python package `speech` (optimizer/orchestrator). Every module is a typed, documented stub matching the target architecture - no ported algorithm logic. The whole thing builds clean and passes trivial tests so `check_all.sh`/`lint_code.sh`/`pytest` are green from day one.

**Tech Stack:** Rust (edition 2024, cargo workspace, PyO3/maturin), Python >=3.14 (uv, ruff, mypy, pytest, hypothesis), TOML config.

## Global Constraints

- **Branch:** all work on `feature/repo-scaffolding`. NEVER commit to `main`.
- **Commit trailer:** every commit message ends with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- **Python:** `requires-python = ">=3.14"`. Ruff line-length 160, target `py314`, lint select `["E","F","I","W","UP","B","SIM"]`. Mypy strict (`disallow_untyped_defs = true`), `mypy_path = "src/python"`. Lint scope is `src/python tests` only.
- **Rust:** edition 2024; `[profile.release] opt-level = 3, lto = true`. Core crate name `speech`; PyO3 lib name `speech_rs`; CLI binary `speech`. Crate root carries `#![allow(dead_code, unused_variables)]` with a `// scaffolding: remove as modules land` comment (skeleton only). `cargo clippy --all-targets -- -D warnings` must pass.
- **Naming:** Rust crate `speech`, PyO3 crate `speech-py` -> module `speech_rs`, Python package `speech`, CLI `speech`.
- **Legacy:** `legacy/` (C++ `src/` + `Optimizer_V6.2.2` `.m`) is git-ignored - present locally as an oracle, never committed. Source of the local copy: `/Users/govit/Git/Govit/FastSpeechProcessing-legacy`.
- **Config:** TOML is canonical; a legacy `.config` importer is a stub here (implemented in Phase 0).
- **ASCII only:** all code, comments, docstrings, and docs use plain hyphens (`-`) and straight quotes. NO em-dashes, en-dashes, or smart quotes (user hard rule; code must be copy-paste safe). Where a sample below shows a stray non-ASCII dash, render it as a plain `-`.
- **Scope:** stubs only - NO ported algorithm logic. Trivial real implementations are allowed only for CLI plumbing (`Mode` parsing), version accessors, the minimal TOML `Config` parse, and the `constants` table loader.
- **Spec:** `docs/superpowers/specs/2026-07-01-speech-repo-setup-design.md` (sections referenced per task).

---

### Task 1: Python project bootstrap

Creates the Python project so `uv sync` works and the lint/test toolchain is green on an (almost) empty package.

**Files:**
- Create: `pyproject.toml`
- Create: `src/python/speech/__init__.py`
- Create: `tests/__init__.py`, `tests/conftest.py`, `tests/test_smoke.py`
- Modify: `.gitignore` (extend the existing Rust-only file)
- Create: `setup_env.sh`, `lint_code.sh`, `upgrade_dependencies.sh`

**Interfaces:**
- Produces: package `speech` importable with `speech.__version__: str`.

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "speech"
version = "0.1.0"
description = "Speech Activity Detection (SAD) and spoken Language Identification (LID) engine - Rust engine + Python optimizer"
requires-python = ">=3.14"
dependencies = [
    "numpy>=2.4",
    "scipy>=1.17",
    "pandas>=3.0",
    "pyarrow>=19.0",
    "pydantic>=2.0",
    "matplotlib>=3.10",
    "rich>=14.3",
    "soundfile>=0.12",
    "cma>=3.3",
]

[dependency-groups]
dev = [
    "pytest>=9.0",
    "ruff>=0.15",
    "mypy>=1.9",
    "hypothesis>=6.100",
    "maturin>=1.12",
    "pandas-stubs>=3.0",
    "scipy-stubs>=1.17",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/python/speech"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src/python", "."]
markers = ["slow: marks tests as slow (deselect with '-m \"not slow\"')"]

[tool.ruff]
line-length = 160
target-version = "py314"

[tool.ruff.lint]
select = ["E", "F", "I", "W", "UP", "B", "SIM"]

[tool.mypy]
python_version = "3.14"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
mypy_path = "src/python"
exclude = ["legacy/"]

[[tool.mypy.overrides]]
module = ["speech_rs", "cma", "soundfile"]
ignore_missing_imports = true
```

- [ ] **Step 2: Write `src/python/speech/__init__.py`**

```python
"""Speech Activity Detection (SAD) + spoken Language Identification (LID).

Python optimizer/orchestrator half of the port. The fast engine lives in the
Rust crate `speech` (PyO3 module `speech_rs`). See CLAUDE.md.
"""

__version__ = "0.1.0"
```

- [ ] **Step 3: Write the smoke test `tests/test_smoke.py`**

```python
"""Smoke tests: the package and its modules import and expose a version."""

import speech


def test_package_imports() -> None:
    assert speech.__version__ == "0.1.0"
```

- [ ] **Step 4: Write `tests/__init__.py` (empty) and `tests/conftest.py`**

`tests/__init__.py`: empty file.

`tests/conftest.py`:

```python
"""Shared pytest fixtures for the speech test suite."""
```

- [ ] **Step 5: Extend `.gitignore`**

Replace the existing `.gitignore` content with (keeps the Rust rules, adds Python/data/legacy/editor sections):

```gitignore
# === macOS ===
.DS_Store

# === Rust ===
target
debug
**/*.rs.bk
*.pdb
**/mutants.out*/

# === Python ===
.venv/
__pycache__/
*.pyc
*.pyo
*.egg-info/
dist/
build/
.mypy_cache/
.pytest_cache/
.hypothesis/
.ruff_cache/

# === MATLAB / binary data ===
*.mat

# === Legacy reference (local oracle, never committed) ===
legacy/

# === Simulator/engine output (regenerated by running) ===
output/*
!output/.gitkeep

# === IDE / Editor ===
.vscode/
.idea/
*.swp
*.swo
*~

# === Claude Code ===
.claude/
.superpowers/
```

- [ ] **Step 6: Write the three Python-side scripts**

`setup_env.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
rm -rf .venv
uv sync --group dev
uv tree
cat <<'EOF' >>.venv/bin/activate

export PYTHONPATH="$PWD:$PWD/src/python"
EOF
```

`lint_code.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
echo "ruff: sorting imports..."
uv run ruff check --select I --fix --config=pyproject.toml src/python tests
echo "ruff: formatting..."
uv run ruff format --config=pyproject.toml src/python tests
echo "ruff: lint..."
uv run ruff check --config=pyproject.toml src/python tests
echo "mypy: type checking..."
uv run mypy --config-file pyproject.toml src/python tests
echo "All linters passed."
```

`upgrade_dependencies.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
echo "Updating dependencies..."
uv sync --upgrade --group dev
echo "Rebuilding PyO3 bindings..."
./build.sh
```

- [ ] **Step 7: Make scripts executable and sync the environment**

Run:
```bash
chmod +x setup_env.sh lint_code.sh upgrade_dependencies.sh
uv sync --group dev
```
Expected: resolves and installs; writes `uv.lock`.

- [ ] **Step 8: Verify lint + tests are green**

Run:
```bash
uv run ruff check --config=pyproject.toml src/python tests
uv run ruff format --check --config=pyproject.toml src/python tests
uv run mypy --config-file pyproject.toml src/python tests
uv run pytest tests -q
```
Expected: ruff clean, format clean, mypy `Success`, pytest `1 passed`.

- [ ] **Step 9: Commit**

```bash
git add pyproject.toml uv.lock src/python tests .gitignore setup_env.sh lint_code.sh upgrade_dependencies.sh
git commit -m "chore: python project bootstrap (uv, ruff, mypy, pytest)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Python package module tree

Fills in the `speech` package modules as documented, typed stubs. Establishes the architecture in code.

**Files:**
- Create: `src/python/speech/{cli,config_bridge,weight_bridge,engine,optimizers,scoring,batching,nn_reference}.py`
- Create: `src/python/speech/drivers/{__init__,init,train,retrain,test}.py`
- Create: `src/python/speech/dataprep/{__init__,augment,opensad15,stm_normalize}.py`
- Create: `tests/test_package_structure.py`

**Interfaces:**
- Produces: the importable module tree of design spec section 6. Stubs use only stdlib / numpy / pathlib types; bodies are `...`.

- [ ] **Step 1: Write the top-level module stubs**

Each file below is created verbatim. Pattern: a docstring naming its legacy MATLAB provenance + one representative typed signature with a `...` body.

`src/python/speech/config_bridge.py`:
```python
"""Config writer/reader: param-vector <-> config struct <-> `.config` text.

Ported from legacy MATLAB: vec2struct.m, printConfig.m, readConfig.m,
SanitizeConfigStruct.m, cell2str.m. See design spec section 6.
"""

from pathlib import Path


def write_config(params: dict[str, object], path: Path) -> None:
    """Serialize a config struct to legacy `.config` text (Phase 0)."""
    ...


def read_config(path: Path) -> dict[str, str]:
    """Parse a legacy `.config` file into a flat key/value map (Phase 0)."""
    ...
```

`src/python/speech/weight_bridge.py`:
```python
"""Neural-net weight (de)serialization: canonical flat vector <-> `.bin`.

Ported from legacy MATLAB: config2network*.m, network2config*.m,
config2weights*.m, nnet2MatFile.m, weights2nnet.m, Write/ReadMatrixFromBinary.m,
CombineNNets.m, ModifyOutputNetwork.m, ForceLSTMBiais.m. See design spec section 6.

High-risk: the column-major flat layout + adim_coeff scaling are the seam to the
Rust engine - round-trip property tests are mandatory when implemented (Phase 3).
"""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def pack_weights(net: dict[str, object]) -> NDArray[np.float64]:
    """Flatten a network struct into the canonical column-major vector (Phase 3)."""
    ...


def unpack_weights(flat: NDArray[np.float64], arch: dict[str, object]) -> dict[str, object]:
    """Inverse of `pack_weights` (Phase 3)."""
    ...


def write_bin(matrix: NDArray[np.float64], path: Path) -> None:
    """Write i64 LE rows, i64 LE cols, then f64 LE column-major (Phase 0)."""
    ...


def read_bin(path: Path) -> NDArray[np.float64]:
    """Inverse of `write_bin` (Phase 0)."""
    ...
```

`src/python/speech/engine.py`:
```python
"""Engine driver: invoke the Rust engine, assemble cost + gradients.

Ported from legacy MATLAB: ComputeCost.m, ComputeGradient*.m, CostFunction*.m,
and RunFsp.py (subprocess seam -> in-process PyO3 call to `speech_rs`).
See design spec section 6.
"""

from numpy.typing import NDArray


def forward_backward(config_path: str, weights: NDArray, batch: list[str]) -> tuple[float, NDArray]:
    """Run the Rust engine over a batch, returning (cost, gradient) (Phase 4)."""
    ...
```

`src/python/speech/optimizers.py`:
```python
"""Optimizer zoo for NN training and DSP-hyperparameter search.

Ported from legacy MATLAB: SMORMS3.m (wired), Adam.m, RMSprop.m, Rprop.m,
QuantumPSO.m (bespoke - hand-port, do NOT swap for pymoo), cmaes.m, sfo.m,
pso_Trelea_vectorized.m, BackPropagation.m. See design spec section 6.
"""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray


def smorms3(grad_fn: Callable[[NDArray[np.float64]], tuple[float, NDArray[np.float64]]], theta0: NDArray[np.float64]) -> NDArray[np.float64]:
    """SMORMS3 gradient descent (the wired NN optimizer) (Phase 3)."""
    ...
```

`src/python/speech/scoring.py`:
```python
"""LID confusion matrices, EER cutoff, gradient checks, masking sanity.

Ported from legacy MATLAB: confusion*.m, CheckGrad.m, MaskingValidation.m.
See design spec section 6.
"""

import numpy as np
from numpy.typing import NDArray


def confusion_matrix(scores: NDArray[np.float64], labels: NDArray[np.int64]) -> NDArray[np.int64]:
    """Per-language confusion matrix with in-band target signaling (Phase 4)."""
    ...
```

`src/python/speech/batching.py`:
```python
"""Hard-example mini-batch scheduler and listing parsing.

Ported from legacy MATLAB: CreateBatches.m, GetNewBatch.m, getCases.m,
processListing.m. See design spec section 6.
"""

from pathlib import Path


def read_listing(path: Path) -> list[dict[str, str]]:
    """Parse a ';'-separated fileslisting CSV into records (Phase 0)."""
    ...
```

`src/python/speech/nn_reference.py`:
```python
"""Pure-Python BLSTM forward/backward oracle for gradient-check fixtures only.

Ported from legacy MATLAB: BLSTM_Forward.m, BLSTM_Backward.m, Hz2Mel.m, Mel2Hz.m.
NOT a production path - used to validate the Rust engine (Phase 2-3).
"""

import numpy as np
from numpy.typing import NDArray


def blstm_forward(inputs: NDArray[np.float64], weights: dict[str, object]) -> NDArray[np.float64]:
    """Reference bidirectional-LSTM forward pass (Phase 2)."""
    ...
```

`src/python/speech/cli.py`:
```python
"""Command-line entry for the Python optimizer/orchestrator.

Ported from legacy MATLAB drivers (Init/Train/ReTrain/Test_BLSTM.m) + RunFsp.py.
See design spec section 6.
"""


def main(argv: list[str] | None = None) -> int:
    """Dispatch to the training/testing drivers (Phase 4)."""
    ...
```

- [ ] **Step 2: Write the `drivers/` subpackage**

`src/python/speech/drivers/__init__.py`:
```python
"""Run-lifecycle drivers: init, train, retrain, test."""
```

`src/python/speech/drivers/init.py`:
```python
"""Network initialization driver. Ported from legacy Init_BLSTM.m / Init_BLSTM_Seg.m."""


def init_network(config_path: str) -> None:
    """Build and initialize a BLSTM network from a config (Phase 3)."""
    ...
```

`src/python/speech/drivers/train.py`:
```python
"""Training driver. Ported from legacy Train_BLSTM*.m / valid_batch.m."""


def train(config_path: str) -> None:
    """Run the outer optimization + gradient training loop (Phase 3-4)."""
    ...
```

`src/python/speech/drivers/retrain.py`:
```python
"""Resume/retrain driver. Ported from legacy ReTrain_BLSTM.m."""


def retrain(checkpoint_path: str) -> None:
    """Resume training from a saved network (Phase 3-4)."""
    ...
```

`src/python/speech/drivers/test.py`:
```python
"""Evaluation driver. Ported from legacy Test_BLSTM.m / Test_BLSTM_multi.m."""


def evaluate(config_path: str) -> None:
    """Score a trained network on a held-out corpus (Phase 4)."""
    ...
```

- [ ] **Step 3: Write the `dataprep/` subpackage**

`src/python/speech/dataprep/__init__.py`:
```python
"""Corpus preparation: augmentation, OpenSAD15 conversion, STM normalization."""
```

`src/python/speech/dataprep/augment.py`:
```python
"""Audio corpus augmentation. Ported from legacy AugmentCorpus.py."""

from pathlib import Path


def augment_corpus(listing: Path, out_dir: Path) -> None:
    """Generate augmented copies (noise/speed/etc.) of a corpus (Phase 4)."""
    ...
```

`src/python/speech/dataprep/opensad15.py`:
```python
"""NIST OpenSAD 2015 corpus converter. Ported from legacy ProcessOpenSAD15Corpus.py."""

from pathlib import Path


def process_opensad15(root: Path, out_dir: Path) -> None:
    """Convert OpenSAD15 tab annotations to XML/STM/.flst refs (Phase 4)."""
    ...
```

`src/python/speech/dataprep/stm_normalize.py`:
```python
"""STM transcript normalization. Ported from legacy norm_stm_*.pl / .sh."""

from pathlib import Path


def normalize_stm(path: Path) -> Path:
    """Normalize an STM transcript file, returning the output path (Phase 4)."""
    ...
```

- [ ] **Step 4: Write `tests/test_package_structure.py`**

```python
"""Assert the full module tree imports (architecture is wired)."""

import importlib

import pytest

MODULES = [
    "speech.cli",
    "speech.config_bridge",
    "speech.weight_bridge",
    "speech.engine",
    "speech.optimizers",
    "speech.scoring",
    "speech.batching",
    "speech.nn_reference",
    "speech.drivers.init",
    "speech.drivers.train",
    "speech.drivers.retrain",
    "speech.drivers.test",
    "speech.dataprep.augment",
    "speech.dataprep.opensad15",
    "speech.dataprep.stm_normalize",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name: str) -> None:
    assert importlib.import_module(name) is not None
```

- [ ] **Step 5: Verify lint + tests are green**

Run:
```bash
uv run ruff check --config=pyproject.toml src/python tests
uv run ruff format --check --config=pyproject.toml src/python tests
uv run mypy --config-file pyproject.toml src/python tests
uv run pytest tests -q
```
Expected: all clean; pytest reports `16 passed` (1 smoke + 15 module-import cases).

- [ ] **Step 6: Commit**

```bash
git add src/python tests
git commit -m "feat: python package module tree (typed stubs)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Legacy vendoring, data assets, and sample configs

Copies the legacy source in as a local oracle (git-ignored), extracts the random tables from `Constants.h`, and lays down sample configs + language mapping.

**Files:**
- Create (local only, git-ignored): `legacy/`
- Create: `scripts/extract_random_tables.py`
- Create: `data/random_tables.bin` (generated, committed)
- Create: `data/languagemapping.csv`
- Create: `configs/lid/lid_blstm.toml`
- Create: `configs/legacy/LID_BLSTM.config`
- Create: `output/.gitkeep`
- Create: `tests/test_data_assets.py`

**Interfaces:**
- Produces: `data/random_tables.bin` = 200000 f64 LE = `_RandomGaussVector[100000]` then `_RandomVector[100000]` (1,600,000 bytes). Consumed by Rust `constants.rs` in Task 4.

- [ ] **Step 1: Vendor the legacy source locally (git-ignored)**

Run:
```bash
mkdir -p legacy
cp -R /Users/govit/Git/Govit/FastSpeechProcessing-legacy/src legacy/src
mkdir -p legacy/Optimizer_V6.2.2
cp -R /Users/govit/Git/Govit/FastSpeechProcessing-legacy/Optimizer_V6.2.2/functions legacy/Optimizer_V6.2.2/functions
cp /Users/govit/Git/Govit/FastSpeechProcessing-legacy/Optimizer_V6.2.2/*.m legacy/Optimizer_V6.2.2/
```
Verify it is ignored:
```bash
git check-ignore legacy/src/Constants.h
```
Expected: prints `legacy/src/Constants.h` (ignored).

- [ ] **Step 2: Write `scripts/extract_random_tables.py`**

```python
"""Extract the two fixed random tables from the legacy Constants.h.

`_RandomGaussVector[100000]` (N(0,1), weight init) and `_RandomVector[100000]`
(U[0,1), dither/augmentation) are the source of determinism in the legacy engine.
This writes them as one little-endian f64 blob (gauss then uniform) at
data/random_tables.bin so the Rust `constants` module can `include_bytes!` it.

Usage: uv run python scripts/extract_random_tables.py
Requires the git-ignored legacy/ copy present locally.
"""

import re
import struct
from pathlib import Path

CONSTANTS = Path("legacy/src/Constants.h")
OUT = Path("data/random_tables.bin")
N = 100000


def extract_array(text: str, name: str) -> list[float]:
    start = text.index(f"{name}[{N}]")
    open_brace = text.index("{", start)
    close_brace = text.index("}", open_brace)
    body = text[open_brace + 1 : close_brace]
    values = [float(tok) for tok in re.split(r"[,\s]+", body.strip()) if tok]
    if len(values) != N:
        raise ValueError(f"{name}: expected {N} values, got {len(values)}")
    return values


def main() -> None:
    text = CONSTANTS.read_text()
    gauss = extract_array(text, "_RandomGaussVector")
    uniform = extract_array(text, "_RandomVector")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("wb") as fh:
        fh.write(struct.pack(f"<{2 * N}d", *gauss, *uniform))
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run the extractor**

Run:
```bash
uv run python scripts/extract_random_tables.py
```
Expected: `wrote data/random_tables.bin (1600000 bytes)`.

- [ ] **Step 4: Lay down the sample data + configs**

`data/languagemapping.csv` (from the legacy `languagemapping.csv`; `lang;dial;classId`):
```csv
fax;non;0
chi;man;1
spa;spa;2
```

`configs/legacy/LID_BLSTM.config` - copy the legacy sample verbatim:
```bash
cp /Users/govit/Git/Govit/FastSpeechProcessing-legacy/ConfigFiles/LID_BLSTM.config configs/legacy/LID_BLSTM.config
```

`configs/lid/lid_blstm.toml` (canonical TOML mirroring a representative subset of the legacy keys; full key mapping is Phase 0):
```toml
# Canonical TOML config for a Twin-BLSTM LID run.
# Mirrors a subset of the legacy ConfigFiles/LID_BLSTM.config (Algo_choice 6).
# Full legacy-key coverage lands in Phase 0 alongside the legacy_config importer.

[engine]
algo = "twin_blstm_lid"          # legacy Algo_choice 6
num_outer_threads = 1
num_inner_threads = 1
results_file = "MultiConfigResults.mat"

[audio]
offset = 0.0                     # legacy Audio_offset
max_duration = 120.0             # legacy Audio_max_duration

[decision]
thresh_rising = 0.6              # legacy BLSTM_decision_thresh_rising
area_rising = 0.0
thresh_falling = 0.3795068189162301
area_falling = 0.0

[spectrum]
order = 9                        # legacy BLSTM_spectrum_order
shift = 0.025                    # legacy BLSTM_spectrum_shift
temporal_convolution_type = "hamming"

[preprocess]
preemph_ratio = -0.97            # legacy BLSTM_preemph_ratio
noise_seed = 3
noise_ratio = 0.02921428128844371
```

`output/.gitkeep`: empty file.

- [ ] **Step 5: Write `tests/test_data_assets.py`**

```python
"""Assert the committed data assets are well-formed."""

import struct
from pathlib import Path

N = 100000
DATA = Path("data/random_tables.bin")


def test_random_tables_size() -> None:
    assert DATA.stat().st_size == 2 * N * 8


def test_random_tables_ranges() -> None:
    raw = DATA.read_bytes()
    values = struct.unpack(f"<{2 * N}d", raw)
    uniform = values[N:]
    assert all(0.0 <= v < 1.0 for v in uniform[:1000])
```

- [ ] **Step 6: Verify tests green**

Run:
```bash
uv run pytest tests -q
```
Expected: all pass (18 total).

- [ ] **Step 7: Commit** (note: `legacy/` is git-ignored and will not be added)

```bash
git add scripts data configs output tests/test_data_assets.py
git commit -m "feat: random-table data asset + sample configs + language map

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Rust core `speech` crate skeleton

The engine skeleton: workspace + core crate, the full module tree as compiling stubs, a real minimal CLI `Mode`, TOML `Config` parse, and the `constants` table loader. Green build/clippy/test.

**Files:**
- Create: `src/rust/Cargo.toml`
- Create: `src/rust/src/{lib,main,config,legacy_config,constants,cost,audio}.rs`
- Create: `src/rust/src/engine/{mod,corpus_processor,bag_of_processors,corpus}.rs`
- Create: `src/rust/src/features/{mod,fft,mel,ltsv_tdc,stats}.rs`
- Create: `src/rust/src/nn/{mod,network,blstm,layers,activations,train}.rs`
- Create: `src/rust/src/tasks/{mod,segmenter,sad,lid,segmentation_io,vrcts}.rs`
- Create: `src/rust/src/io/{mod,binary,matfile}.rs`
- Create: `src/rust/tests/smoke.rs`

**Interfaces:**
- Consumes: `data/random_tables.bin` from Task 3 (via `include_bytes!("../../../data/random_tables.bin")` in `constants.rs`).
- Produces: crate `speech` with `speech::version() -> &'static str`, `speech::cli::Mode`, `speech::config::Config::from_toml_str`, `speech::constants::{random_gauss, random_uniform, MAX_RAND_SIZE}`.

- [ ] **Step 1: Write `src/rust/Cargo.toml`**

```toml
# NOTE: workspace members lists only "." for now; Task 5 adds "speech-py" when
# that crate is created. Listing a not-yet-existing member here would break
# `cargo build` in this task.
[workspace]
members = ["."]

[package]
name = "speech"
version = "0.1.0"
edition = "2024"
description = "SAD + spoken Language Identification engine (Rust port of FastSpeechProcessing)"

[[bin]]
name = "speech"
path = "src/main.rs"

[dependencies]
anyhow = "1"
thiserror = "2"
serde = { version = "1", features = ["derive"] }
serde_json = { version = "1", features = ["float_roundtrip"] }
toml = "0.9"
indexmap = { version = "2", features = ["serde"] }
nalgebra = "0.35"
ndarray = "0.16"
rayon = "1.10"
byteorder = "1.5"
bytemuck = "1"
rustfft = "6"
realfft = "3"
symphonia = { version = "0.5", features = ["wav", "flac", "ogg", "vorbis"] }
quick-xml = { version = "0.37", features = ["serialize"] }

[dev-dependencies]
approx = "0.5"
rstest = "0.26"
proptest = "1"
tempfile = "3"

[profile.release]
opt-level = 3
lto = true
```

Note on deferred deps: the `.mat` reader/writer crate is added in Phase 1 when `io::matfile` is implemented; `io/matfile.rs` is a signature-only stub here.

- [ ] **Step 2: Write `src/rust/src/lib.rs`**

```rust
//! `speech` - fast SAD + spoken Language Identification engine.
//!
//! Rust port of the legacy C++ `FastSpeechProcessing` engine. The Python
//! optimizer/orchestrator drives this crate via the `speech-py` PyO3 bindings.
//! See CLAUDE.md and docs/superpowers/specs/2026-07-01-speech-repo-setup-design.md.
#![allow(dead_code, unused_variables)] // scaffolding: remove as modules land

pub mod audio;
pub mod cli;
pub mod config;
pub mod constants;
pub mod cost;
pub mod engine;
pub mod features;
pub mod io;
pub mod legacy_config;
pub mod nn;
pub mod tasks;

/// Crate version, sourced from Cargo metadata.
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}
```

- [ ] **Step 3: Write `src/rust/src/cli.rs` (real `Mode` parser - CLI plumbing)**

```rust
//! CLI mode parsing, mirroring the legacy `fsp` flags.
//!
//! Ported from legacy C++: FastSpeechProcessing.cpp `main`.

/// Run mode selected by the leading CLI flag.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// `-s` / `-S`: run one config over the corpus.
    Solo,
    /// `-i` / `-I`: solo run plus image/segmentation dump.
    Image,
    /// `-t` / `-T`: solo run plus finite-difference gradient unit test.
    UnitTest,
    /// `-m` / `-M`: multi-config run.
    Multi,
}

impl Mode {
    /// Parse a leading flag into a [`Mode`]. Returns `None` for unknown flags.
    pub fn from_flag(flag: &str) -> Option<Mode> {
        match flag {
            "-s" | "-S" => Some(Mode::Solo),
            "-i" | "-I" => Some(Mode::Image),
            "-t" | "-T" => Some(Mode::UnitTest),
            "-m" | "-M" => Some(Mode::Multi),
            _ => None,
        }
    }
}
```

- [ ] **Step 4: Write `src/rust/src/config.rs` (minimal real TOML parse - plumbing)**

```rust
//! Canonical TOML configuration.
//!
//! Ported from legacy C++: ConfigFile.*, String.hpp. This is the minimal
//! scaffold; the full key schema lands in Phase 0.

use serde::Deserialize;

/// Top-level engine config (minimal scaffold subset).
#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub engine: EngineConfig,
}

/// `[engine]` section.
#[derive(Debug, Clone, Deserialize)]
pub struct EngineConfig {
    pub algo: String,
    #[serde(default = "one")]
    pub num_outer_threads: usize,
}

fn one() -> usize {
    1
}

impl Config {
    /// Parse a TOML document.
    pub fn from_toml_str(text: &str) -> Result<Config, toml::de::Error> {
        toml::from_str(text)
    }
}
```

- [ ] **Step 5: Write `src/rust/src/constants.rs` (real table loader)**

```rust
//! Fixed random tables ported from the legacy Constants.h.
//!
//! Two 100000-entry f64 tables provide the legacy engine's determinism:
//! `_RandomGaussVector` (N(0,1), weight init) and `_RandomVector` (U[0,1),
//! dither/augmentation). Embedded from data/random_tables.bin (gauss then
//! uniform) and indexed modulo [`MAX_RAND_SIZE`], exactly like the legacy.

/// Length of each random table (legacy `_MaxRandSize`).
pub const MAX_RAND_SIZE: usize = 100_000;

static RANDOM_TABLES: &[u8] = include_bytes!("../../../data/random_tables.bin");

fn table(offset_entries: usize, i: usize) -> f64 {
    let idx = offset_entries + (i % MAX_RAND_SIZE);
    let start = idx * 8;
    let bytes: [u8; 8] = RANDOM_TABLES[start..start + 8].try_into().unwrap();
    f64::from_le_bytes(bytes)
}

/// `_RandomGaussVector[i % MAX_RAND_SIZE]` - standard-normal samples.
pub fn random_gauss(i: usize) -> f64 {
    table(0, i)
}

/// `_RandomVector[i % MAX_RAND_SIZE]` - uniform [0,1) samples.
pub fn random_uniform(i: usize) -> f64 {
    table(MAX_RAND_SIZE, i)
}
```

- [ ] **Step 6: Write `src/rust/src/main.rs` (CLI entry)**

```rust
//! `speech` CLI entry - mirrors the legacy `fsp` invocation.

use speech::cli::Mode;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).and_then(|f| Mode::from_flag(f));
    match mode {
        Some(mode) => {
            // Phase 0+: build CorpusProcessor and dispatch on `mode`.
            eprintln!("speech {} - mode {:?} (engine not yet implemented)", speech::version(), mode);
        }
        None => {
            eprintln!("usage: speech <-s|-i|-t|-m> <config.toml>...");
            std::process::exit(2);
        }
    }
}
```

- [ ] **Step 7: Write the stub modules**

Each module below is created with exactly the content shown (doc-comment naming its legacy provenance + any central `pub` type). These are stubs - no logic.

`src/rust/src/audio.rs`:
```rust
//! Audio container + I/O (load, normalize, dither, pre-emphasis, framing).
//!
//! Ported from legacy C++: AudioStruct.*. Uses `symphonia` when implemented (Phase 1).

/// Multi-channel audio buffer (channels x frames, f64).
pub struct Audio {
    pub sample_rate: u32,
    pub data: Vec<Vec<f64>>,
}
```

`src/rust/src/cost.rs`:
```rust
//! Piecewise cost laws + multiclass softmax cross-entropy with ignore-mask.
//!
//! Ported from legacy C++: CostLaw.*. Implemented in Phase 0/3.
```

`src/rust/src/legacy_config.rs`:
```rust
//! Importer for the legacy whitespace `.config` format.
//!
//! Ported from legacy C++: ConfigFile.*. Reproduces the `_`-rest-of-line
//! continuation, last-value-wins, and `val*count` repeat rules. Implemented Phase 0.

use indexmap::IndexMap;

/// Read a legacy `.config` into a flat, insertion-ordered key/value map.
pub fn read_legacy_config(text: &str) -> IndexMap<String, String> {
    todo!("Phase 0: port ConfigFile parsing")
}
```

`src/rust/src/engine/mod.rs`:
```rust
//! Engine orchestration: corpus processing, processor dispatch.
//!
//! Ported from legacy C++: CorpusProcessor.*, BagOfProcessors.*, Corpus.*.

pub mod bag_of_processors;
pub mod corpus;
pub mod corpus_processor;
```

`src/rust/src/engine/corpus_processor.rs`:
```rust
//! Top-level driver: build corpus + processors, rayon `par_iter` over files,
//! epoch training loop, finite-diff grad check, result reduction + `.mat` write.
//!
//! Ported from legacy C++: CorpusProcessor.* (replaces the OpenMP parallel-for).
```

`src/rust/src/engine/bag_of_processors.rs`:
```rust
//! One `Segmenter` per config, dispatched by an `Algo` enum (replaces the legacy
//! if/else ladder); per-file dispatch, weight save/update, metric reduction.
//!
//! Ported from legacy C++: BagOfProcessors.*.
```

`src/rust/src/engine/corpus.rs`:
```rust
//! Corpus + per-file records: parse language2classmapping + fileslisting CSVs,
//! class-balance coefficients.
//!
//! Ported from legacy C++: Corpus.*, CorpusItem.*.

/// One corpus entry (audio path + refs + language/dialect/class metadata).
pub struct CorpusItem {
    pub file_name: String,
    pub ref_seg: String,
    pub language: String,
    pub dialect: String,
    pub class_index: i32,
    pub weight: f64,
}
```

`src/rust/src/features/mod.rs`:
```rust
//! Acoustic feature extraction: FFT, Mel/MFCC, LTSV, time-domain correlation.
//!
//! Ported from legacy C++: fft.hpp, MelFilterBank.*, LongTermSpectralVariation.*,
//! TimeDomainCorrel.*, InputStatistics.*.

pub mod fft;
pub mod ltsv_tdc;
pub mod mel;
pub mod stats;
```

`src/rust/src/features/fft.rs`:
```rust
//! FFT with the legacy two-reals-in-one-complex packing + Welch normalization.
//!
//! Ported from legacy C++: fft.hpp (replaces compile-time GFFT with `rustfft`). Phase 1.
```

`src/rust/src/features/mel.rs`:
```rust
//! Triangular Mel filterbank (1125/700, unit-peak), DCT-II MFCCs, deltas, SDC.
//!
//! Ported from legacy C++: MelFilterBank.*. Phase 1.
```

`src/rust/src/features/ltsv_tdc.rs`:
```rust
//! Long-Term Spectral Variation + time-domain autocorrelation pitch features.
//!
//! Ported from legacy C++: LongTermSpectralVariation.*, TimeDomainCorrel.*. Phase 1.
```

`src/rust/src/features/stats.rs`:
```rust
//! Streaming per-dimension mean/std with parallel merge; fast log/exp policy.
//!
//! Ported from legacy C++: InputStatistics.*, fmath.hpp. Phase 1.
```

`src/rust/src/nn/mod.rs`:
```rust
//! Neural network: layers, BLSTM, generic container, activations, training.
//!
//! Ported from legacy C++: NeuralNetwork.hpp, BLSTMNeuralNetwork.*, *Layer.*,
//! ActivationFunctions.h, Rprop.*, Trainer.h.

pub mod activations;
pub mod blstm;
pub mod layers;
pub mod network;
pub mod train;
```

`src/rust/src/nn/network.rs`:
```rust
//! Generic stacked-layer container over a `Layer` trait; sub-sample decimation.
//!
//! Ported from legacy C++: NeuralNetwork.hpp. Phase 2.

/// A network layer: forward (and, when trainable, backward) over a sequence.
pub trait Layer {}
```

`src/rust/src/nn/blstm.rs`:
```rust
//! Bidirectional wrapper: forward + reverse recurrent nets + output MLP, BPTT.
//!
//! Ported from legacy C++: BLSTMNeuralNetwork.*. Phase 2-3.
```

`src/rust/src/nn/layers.rs`:
```rust
//! Layer implementations: peephole LSTM, SRN, Clockwork-RNN, dense, conv.
//!
//! Ported from legacy C++: LSTMLayer.*, SRNLayer.*, CWRNNLayer.*, NeuronLayer.*,
//! Convolutional*.*. Phase 2.
```

`src/rust/src/nn/activations.rs`:
```rust
//! Activations honoring the legacy overrides: asinh cell/output, sigmoid(0.1z)
//! gates, softmax. Ported from legacy C++: ActivationFunctions.h. Phase 2.
```

`src/rust/src/nn/train.rs`:
```rust
//! iRPROP- on a flat weight vector (inverted sign, weight-backtracking).
//!
//! Ported from legacy C++: Rprop.cpp, Trainer.h. Phase 3.
```

`src/rust/src/tasks/mod.rs`:
```rust
//! Task layer: segmentation/SAD, LID, decision logic, scoring, output.
//!
//! Ported from legacy C++: Segmenter.*, Segmentation.*, BLSTM*Segmenter.*,
//! BLSTMSpectralLID.*, TwinBLSTMSpectralLID.*, VRCTSpart.*.

pub mod lid;
pub mod sad;
pub mod segmentation_io;
pub mod segmenter;
pub mod vrcts;
```

`src/rust/src/tasks/segmenter.rs`:
```rust
//! `Segmenter` trait + decision logic (hysteresis-with-area, 7-step smoothing).
//!
//! Ported from legacy C++: Segmenter.*. Pure logic - golden-tested first (Phase 0).

/// Produces a [`Segmentation`](super::segmentation_io) from features/audio.
pub trait Segmenter {}
```

`src/rust/src/tasks/sad.rs`:
```rust
//! SAD segmenters: BLSTM over signal/spectral/LTSV features -> speech posterior.
//!
//! Ported from legacy C++: BLSTMSignalSegmenter.*, BLSTMSpectralSegmenter.*,
//! BLSTMSpectralLID.*. Phase 2-4.
```

`src/rust/src/tasks/lid.rs`:
```rust
//! Twin/Siamese LID: SAD + LID BLSTM, per-segment accumulation, argmax, confusion.
//!
//! Ported from legacy C++: TwinBLSTMSpectralLID.*. Phase 4.
```

`src/rust/src/tasks/segmentation_io.rs`:
```rust
//! Segment container + primitives (sanitize, suppress-short, padding), scoring,
//! VRCTS XML + STM/TRS loaders.
//!
//! Ported from legacy C++: Segmentation.*. Phase 0.

/// A labelled time span `[start, end)` in seconds.
pub struct Segment {
    pub start: f64,
    pub end: f64,
    pub label: i32,
}
```

`src/rust/src/tasks/vrcts.rs`:
```rust
//! External-tool adapter behind the `Segmenter` trait (shell out + re-ingest XML).
//!
//! Ported from legacy C++: VRCTSpart.*. Phase 4.
```

`src/rust/src/io/mod.rs`:
```rust
//! On-disk interchange: weight `.bin` codec and `.mat` read/write.
//!
//! Ported from legacy C++: Helpers.hpp (BinaryFile2Vector/Matrix2BinaryFile/
//! Matrix2MatFile).

pub mod binary;
pub mod matfile;
```

`src/rust/src/io/binary.rs`:
```rust
//! Weight `.bin` codec: i64 LE rows, i64 LE cols, then f64 LE column-major.
//!
//! Ported from legacy C++: Helpers.hpp. The MATLAB/Python seam - implemented and
//! round-trip golden-tested in Phase 0.

use std::path::Path;

/// Read a column-major f64 matrix from the legacy `.bin` layout.
pub fn read_matrix(path: &Path) -> anyhow::Result<(usize, usize, Vec<f64>)> {
    todo!("Phase 0: port BinaryFile2Vector")
}
```

`src/rust/src/io/matfile.rs`:
```rust
//! `.mat` v5 read/write for diagnostics and legacy parity.
//!
//! Ported from legacy C++: Helpers.hpp (Matrix2MatFile). Pulls in a MAT crate in
//! Phase 1; signature-only stub for now.
```

- [ ] **Step 8: Write `src/rust/tests/smoke.rs`**

```rust
//! Skeleton smoke tests: the crate boots and the real plumbing works.

use speech::cli::Mode;
use speech::config::Config;
use speech::constants::{random_uniform, MAX_RAND_SIZE};

#[test]
fn version_is_set() {
    assert!(!speech::version().is_empty());
}

#[test]
fn mode_parses_flags() {
    assert_eq!(Mode::from_flag("-m"), Some(Mode::Multi));
    assert_eq!(Mode::from_flag("-s"), Some(Mode::Solo));
    assert_eq!(Mode::from_flag("--nope"), None);
}

#[test]
fn config_parses_minimal_toml() {
    let cfg = Config::from_toml_str("[engine]\nalgo = \"twin_blstm_lid\"\n").unwrap();
    assert_eq!(cfg.engine.algo, "twin_blstm_lid");
    assert_eq!(cfg.engine.num_outer_threads, 1);
}

#[test]
fn random_tables_load_and_index_modularly() {
    let u = random_uniform(0);
    assert!((0.0..1.0).contains(&u));
    // Modular indexing: index N wraps to index 0.
    assert_eq!(random_uniform(MAX_RAND_SIZE), random_uniform(0));
}
```

- [ ] **Step 9: Verify build, clippy, and tests are green**

Run:
```bash
cd src/rust && cargo build --release && cargo clippy --all-targets -- -D warnings && cargo test && cargo fmt --all -- --check && cd ../..
```
Expected: builds; clippy clean; all smoke tests pass; formatting clean.

- [ ] **Step 10: Commit**

```bash
git add src/rust
git commit -m "feat: rust core 'speech' crate skeleton (compiling stubs + CLI/config/constants)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: PyO3 `speech-py` crate + build/check scripts

Adds the binding crate producing module `speech_rs`, and the `build.sh`/`check_all.sh` scripts that tie Rust + Python together.

**Files:**
- Create: `src/rust/speech-py/Cargo.toml`, `src/rust/speech-py/pyproject.toml`, `src/rust/speech-py/src/lib.rs`
- Create: `build.sh`, `check_all.sh`

**Interfaces:**
- Consumes: crate `speech` (`speech::version`) from Task 4.
- Produces: importable Python module `speech_rs` with `speech_rs.version() -> str`.

- [ ] **Step 0: Register `speech-py` in the workspace**

Task 4 created `src/rust/Cargo.toml` with `members = ["."]`. Edit that line to add the new member (and drop the now-obsolete NOTE comment about it):

```toml
[workspace]
members = [".", "speech-py"]
```

- [ ] **Step 1: Write `src/rust/speech-py/Cargo.toml`**

```toml
[package]
name = "speech-py"
version = "0.1.0"
edition = "2024"
description = "PyO3 bindings for the speech engine"

[lib]
name = "speech_rs"
crate-type = ["cdylib", "lib"]

[features]
extension-module = ["pyo3/extension-module"]

[dependencies]
speech = { path = ".." }
pyo3 = { version = "0.28" }
numpy = "0.28"
```

- [ ] **Step 2: Write `src/rust/speech-py/pyproject.toml`**

```toml
[build-system]
requires = ["maturin>=1.0,<2.0"]
build-backend = "maturin"

[project]
name = "speech-rs"
requires-python = ">=3.14"
version = "0.1.0"
description = "PyO3 bindings for the speech engine"

[tool.maturin]
features = ["extension-module"]
```

- [ ] **Step 3: Write `src/rust/speech-py/src/lib.rs`**

```rust
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
```

- [ ] **Step 4: Write `build.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail

RUST_DIR="src/rust"
PYO3_DIR="src/rust/speech-py"

echo "=== Building Rust engine ==="
cargo build --release --quiet --manifest-path "$RUST_DIR/Cargo.toml"

echo "=== Building PyO3 bindings (speech_rs) ==="
uv run maturin develop --release --quiet --manifest-path "$PYO3_DIR/Cargo.toml"

echo "=== Done ==="
echo "  Binary: $RUST_DIR/target/release/speech"
echo "  PyO3:   speech_rs installed in .venv"
```

- [ ] **Step 5: Write `check_all.sh`**

```bash
#!/usr/bin/env bash
set -uo pipefail

cd src/rust
echo "Running all Rust checks..."

cargo test --quiet
test_status=$?

cargo fmt --all -- --check
fmt_status=$?

cargo clippy --all-targets --all-features --quiet -- -D warnings
clippy_status=$?

cd ../..

./build.sh
build_status=$?

echo ""
echo "=== Results ==="
[ $test_status -eq 0 ]   && echo "  Tests:      ok" || echo "  Tests:      FAILED"
[ $fmt_status -eq 0 ]    && echo "  Formatting: ok" || echo "  Formatting: FAILED"
[ $clippy_status -eq 0 ] && echo "  Clippy:     ok" || echo "  Clippy:     FAILED"
[ $build_status -eq 0 ]  && echo "  Build:      ok" || echo "  Build:      FAILED"
echo ""

if [ $test_status -ne 0 ] || [ $fmt_status -ne 0 ] || [ $clippy_status -ne 0 ] || [ $build_status -ne 0 ]; then
    echo "Some checks failed!"
    exit 1
fi
echo "All checks passed!"
```

- [ ] **Step 6: Build the bindings and verify the module imports**

Run:
```bash
chmod +x build.sh check_all.sh
./build.sh
uv run python -c "import speech_rs; print(speech_rs.version())"
```
Expected: prints `0.1.0`.

- [ ] **Step 7: Run the full check script**

Run:
```bash
./check_all.sh
```
Expected: ends with `All checks passed!`.

- [ ] **Step 8: Commit**

```bash
git add src/rust/speech-py build.sh check_all.sh
git commit -m "feat: PyO3 'speech-py' crate (module speech_rs) + build/check scripts

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: CI workflow + editor configs

**Files:**
- Create: `.github/workflows/ci.yml`
- Create: `.zed/debug.json`
- Create: `speech.code-workspace`

**Interfaces:**
- Consumes: the scripts and crate layout from Tasks 1-5.

- [ ] **Step 1: Write `.github/workflows/ci.yml`**

```yaml
name: CI

on:
  pull_request:
    branches: [main]
  workflow_dispatch:

env:
  CARGO_TERM_COLOR: always

jobs:
  rust:
    name: Rust (${{ matrix.check }})
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: src/rust
    strategy:
      fail-fast: false
      matrix:
        include:
          - check: fmt
            command: cargo fmt --all -- --check
          - check: clippy
            command: cargo clippy --all-targets -- -D warnings
          - check: test
            command: cargo test --release
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with:
          components: rustfmt, clippy
      - uses: Swatinem/rust-cache@v2
        with:
          workspaces: src/rust
      - name: ${{ matrix.check }}
        run: ${{ matrix.command }}

  python-lint:
    name: Python (${{ matrix.check }})
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        include:
          - check: lint
            command: uv run ruff check src/python tests
          - check: format
            command: uv run ruff format --check src/python tests
          - check: typecheck
            command: uv run mypy src/python tests
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
      - name: Install dependencies
        run: uv sync --group dev
      - name: ${{ matrix.check }}
        run: ${{ matrix.command }}

  python-test:
    name: Python (test)
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
      - name: Install dependencies
        run: uv sync --group dev
      - name: Run tests
        run: uv run pytest tests -v

  python-pyo3:
    name: Python (pyo3)
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - uses: Swatinem/rust-cache@v2
        with:
          workspaces: src/rust
      - uses: astral-sh/setup-uv@v5
      - name: Install dependencies
        run: uv sync --group dev
      - name: Build PyO3 module
        run: uv run maturin develop --release --manifest-path src/rust/speech-py/Cargo.toml
      - name: Import check
        run: uv run python -c "import speech_rs; print(speech_rs.version())"
```

- [ ] **Step 2: Write `.zed/debug.json`**

```json
[
  {
    "label": "Debug Lib Tests",
    "adapter": "CodeLLDB",
    "request": "launch",
    "build": {
      "command": "cargo",
      "args": ["test", "--no-run", "--lib", "--package", "speech"]
    },
    "cwd": "$ZED_WORKTREE_ROOT/src/rust",
    "sourceLanguages": ["rust"]
  },
  {
    "label": "Debug Binary (config)",
    "adapter": "CodeLLDB",
    "request": "launch",
    "build": {
      "command": "cargo",
      "args": ["build", "--package", "speech"]
    },
    "program": "$ZED_WORKTREE_ROOT/src/rust/target/debug/speech",
    "args": ["-m", "$ZED_WORKTREE_ROOT/configs/lid/lid_blstm.toml"],
    "cwd": "$ZED_WORKTREE_ROOT/src/rust",
    "sourceLanguages": ["rust"]
  }
]
```

- [ ] **Step 3: Write `speech.code-workspace`**

```json
{
    "folders": [{ "path": "." }],
    "settings": {
        "files.exclude": {
            "**/__pycache__": true,
            "**/.venv": true,
            "**/.pytest_cache": true,
            "**/.mypy_cache": true,
            "**/.ruff_cache": true,
            "**/.DS_Store": true,
            "src/rust/target": true,
            "legacy": true
        },
        "python.defaultInterpreterPath": ".venv/bin/python",
        "python.terminal.activateEnvInCurrentTerminal": true,
        "python.testing.pytestEnabled": true,
        "python.testing.pytestArgs": ["tests"],
        "python.analysis.typeCheckingMode": "off",
        "ruff.enable": true,
        "[python]": {
            "editor.defaultFormatter": "charliermarsh.ruff",
            "editor.codeActionsOnSave": {
                "source.organizeImports": "explicit",
                "source.fixAll": "explicit"
            }
        },
        "mypy-type-checker.args": ["--config-file=pyproject.toml"],
        "rust-analyzer.linkedProjects": ["src/rust/Cargo.toml"],
        "rust-analyzer.check.command": "clippy",
        "[rust]": {
            "editor.defaultFormatter": "rust-lang.rust-analyzer",
            "editor.formatOnSave": true
        },
        "editor.formatOnSave": true,
        "editor.rulers": [160]
    },
    "extensions": {
        "recommendations": [
            "charliermarsh.ruff",
            "ms-python.mypy-type-checker",
            "rust-lang.rust-analyzer",
            "anthropics.claude-code"
        ]
    }
}
```

- [ ] **Step 4: Validate the JSON/YAML parse**

Run:
```bash
uv run python -c "import json; json.load(open('.zed/debug.json')); json.load(open('speech.code-workspace')); print('json ok')"
uv run python -c "import sys; sys.exit(0)"  # yaml is validated by CI itself; ensure file is present
test -f .github/workflows/ci.yml && echo "ci.yml present"
```
Expected: `json ok` and `ci.yml present`.

- [ ] **Step 5: Commit**

```bash
git add .github .zed speech.code-workspace
git commit -m "chore: CI workflow + zed/vscode editor configs

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 7: CLAUDE.md

Write the guidance doc in the Aerocapture style, accurate to the module maps and decisions.

**Files:**
- Create: `CLAUDE.md`

- [ ] **Step 1: Write `CLAUDE.md`** with these sections (content drawn from the design spec):

1. **Header** - "This file provides guidance to Claude Code ... when working with code in this repository."
2. **Project Overview** - one dense paragraph: the LID+SAD system, the C++ engine -> Rust and MATLAB optimizer -> Python split, the file seam (`.config`/`.bin`/`.mat`) being replaced by an in-process PyO3 API, and the bit-level-validation-against-legacy goal. (Adapt design spec section 2 + interface stance.)
3. **Build & Development Commands** - the Rust (`cargo build/test` in `src/rust`, run `./src/rust/target/release/speech -m configs/lid/lid_blstm.toml`), PyO3 (`uv run maturin develop --release --manifest-path src/rust/speech-py/Cargo.toml`), Python (`uv sync --group dev`, `pytest tests`), and utility scripts (`build.sh`, `setup_env.sh`, `lint_code.sh`, `check_all.sh`, `upgrade_dependencies.sh`) blocks.
4. **Architecture** - the Rust module map (design spec section 5 table) and Python module map (section 6 table), each with responsibility + legacy provenance.
5. **Key Decisions & Pitfalls** - the parity hazards verbatim from design spec section 8: weight-layout ordering, `adim_coeff` scaling, `asinh`/`sigmoid(0.1z)` activations, iRPROP- inverted sign, FFT two-real packing, non-standard LTSV, fixed 100000-entry random tables, little-endian `.bin`/`.mat`, MATLAB 1-based vs config 0-based `Layer_%d`, the `score>150`/`val-200` in-band signaling. Plus the four locked decisions (naming, git-ignored legacy oracle, TOML-first + importer, pure-logic-first roadmap) and the scaffolding-only `#![allow(dead_code, unused_variables)]` note.
6. **Conventions** - Rust (edition 2024, nalgebra/ndarray, release LTO), Python (>=3.14, ruff 160/py314, mypy strict, uv, pytest+hypothesis), Testing, CI.
7. **Tone** - "Be a quirky friendly but critical peer reviewer ... Challenge inefficiencies." (Match Aerocapture.)

- [ ] **Step 2: Verify presence + section headers**

Run:
```bash
grep -E '^## ' CLAUDE.md
```
Expected: lists Project Overview, Build & Development Commands, Architecture, Key Decisions & Pitfalls, Conventions, Tone.

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: add CLAUDE.md (project overview, architecture, pitfalls)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 8: README.md

**Files:**
- Modify: `README.md` (currently just `# Speech`)

- [ ] **Step 1: Write `README.md`** with these sections:

1. **Title + one-paragraph what-it-is** - SAD + spoken LID; Rust engine + Python optimizer; port of the legacy `FastSpeechProcessing`; validated bit-level against it.
2. **Quick Start** - build the Rust engine, run on a TOML config, build PyO3 bindings, `uv sync --group dev`, run tests (`cargo test` + `uv run pytest tests`).
3. **Project Structure** - the tree from design spec section 4.
4. **What it does** - SAD (per-frame speech posterior -> segments via hysteresis + smoothing) and LID (TwinBLSTM consuming SAD hidden states -> per-file language posterior); the feature front-end (FFT/Mel/MFCC/LTSV/TDC) and the NN zoo (BLSTM/SRN/CWRNN/CNN).
5. **Configuration** - TOML is canonical (show the `configs/lid/lid_blstm.toml` shape); a legacy `.config` importer exists for validation runs.
6. **Validation strategy** - the layered parity ladder from design spec section (golden features -> weight roundtrip -> forward parity -> gradient parity -> optimizer-step -> decision/scoring -> end-to-end LID/SAD).
7. **Roadmap** - the five phases from design spec section 9 (Phase 0 pure-logic + I/O parity first, then features, NN forward, training/optimizers, PyO3 + end-to-end).
8. **Testing / CI** - `check_all.sh`, `lint_code.sh`, `pytest`, and the GitHub Actions jobs.

- [ ] **Step 2: Verify presence + section headers**

Run:
```bash
grep -E '^## ' README.md
```
Expected: Quick Start, Project Structure, What it does, Configuration, Validation, Roadmap, Testing (or equivalent).

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: flesh out README (quick start, structure, validation, roadmap)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 9: Full-suite verification + smart-commit

Confirms the Definition of Done end-to-end, then syncs docs and commits the branch.

**Files:** none (verification + commit).

- [ ] **Step 1: Run the entire toolchain and confirm green**

Run:
```bash
./check_all.sh
./lint_code.sh
uv run pytest tests -q
```
Expected: `check_all.sh` -> `All checks passed!`; `lint_code.sh` -> `All linters passed.`; pytest -> all passed (18).

- [ ] **Step 2: Confirm the Definition of Done items**

Verify each item in design spec section 10 is satisfied: release build + clippy clean, cargo tests pass, uv sync + ruff + mypy + pytest pass, scripts run green, `CLAUDE.md` + `README.md` present and accurate, `legacy/` git-ignored (`git status` shows it untracked-and-ignored).

- [ ] **Step 3: Invoke the `smart-commit` skill over the whole branch**

Invoke the `smart-commit` skill, instructing it to take the entire `feature/repo-scaffolding` branch into account (sync `CLAUDE.md`/`README.md` with the final state of the code and commit anything outstanding). Do not push.

---

## Self-Review

**1. Spec coverage.** Spec section 1 (green skeleton + docs) -> Tasks 1-8; section 3 decisions -> Task 1 (.gitignore/legacy), Task 3 (TOML sample + legacy config), Task 4 (naming, allow-note), Task 8 (roadmap); section 4 layout -> all tasks; section 5 Rust map -> Task 4; section 6 Python map -> Task 2; section 7 toolchain -> Tasks 1,4,5,6; section 8 docs/pitfalls -> Tasks 7,8; section 9 roadmap -> Task 8; section 10 Definition of Done -> Task 9. No gaps.

**2. Placeholder scan.** No "TBD/TODO/handle edge cases" in steps. Rust `todo!()`/Python `...` bodies are intentional stub bodies (the deliverable IS stubs), covered by the crate-level `#![allow(...)]` and mypy's `...`-body handling - not plan placeholders. Docs Tasks 7-8 enumerate exact sections + source spec sections rather than pasting the full prose; acceptable since the content is fully determined by the referenced spec sections.

**3. Type consistency.** `speech::version()` defined in Task 4 lib.rs, consumed in Task 4 smoke test + Task 5 PyO3 lib. `Mode::from_flag` defined Task 4 cli.rs, used Task 4 main.rs + smoke test. `Config::from_toml_str` / `constants::{random_gauss,random_uniform,MAX_RAND_SIZE}` defined + consumed within Task 4. `data/random_tables.bin` produced Task 3, consumed Task 4 via `include_bytes!("../../../data/random_tables.bin")`. `speech_rs.version()` defined Task 5, checked Task 5 + Task 6 CI. Python module list in Task 2's `test_package_structure.py` matches the files created in Task 2. Consistent.
