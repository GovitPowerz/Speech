# Phase 0b-i Cost Laws + Segmentation Container Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the legacy `CostLaw` (7 piecewise VAD cost laws + softmax cross-entropy) and the `Segmentation` boundary-list container to Rust, validated bit-for-bit against the real `test_costlaws.mat` sweeps and hand-verified unit tests.

**Architecture:** Rust-only production (`src/rust/src/cost.rs`, `src/rust/src/tasks/segmentation.rs`). A Python scipy prep extracts the legacy `.mat` cost sweeps into committed `.bin` fixtures (via the Phase-0a codec); the Rust cost tests reproduce them exactly. The branch builds on Phase 0a (`legacy_config`, `io::binary` are present).

**Tech Stack:** Rust (edition 2024), Python (scipy for the prep only), the Phase-0a `speech::io::binary` codec and `speech::legacy_config` parser.

## Global Constraints

- Branch: `feature/phase-0b-i-cost-container` (built ON Phase 0a; do NOT rebase onto main). NEVER commit to `main`. Commit trailer `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- All cost math is IEEE-754 `f64`; use `f64::ln`, `f64::sqrt`.
- The **double-read**: law coefficients use the CLAMPED thresholds (`thS in [1e-6, 1-1e-6]`, `thN_law = 1 - clamp(...)`); the runtime branch selectors are RAW/unclamped (`switching_thresh_speech` default `10.0`, `switching_thresh_no_speech` default `-1.0`). Never collapse them.
- Preserve the **LogLaw cost/deriv Adim asymmetry** and the **AboveThreshCubic name-dependent coefficients** exactly.
- `sanitize` rounds to the 1e-4 grid with **round-half-away-from-zero** (`(t*1e4).round()` in Rust IS half-away-from-zero for `f64` - confirm), NOT banker's rounding; the terminal sentinel is never rounded.
- `SegClass` is `#[repr(i32)]` with codes `OTHER=0, SPEECH=1, RING=2, DTMF_0..9=3..12, DTMF_A..D=13..16, DTMF_STAR=17, DTMF_SHARP=18, INSERTION=19, SUBSTITUTION=20, EXCLUDED=21, END=22`.
- Rust `clippy --all-targets -- -D warnings` + `fmt --check` green; Python `ruff` (160/py314) + `mypy` clean on touched paths.
- ASCII only in code/comments/docs; no em-dashes/en-dashes/smart quotes.
- Fixtures live in `tests/reference_data/phase0b/` (committed). Legacy source (local, git-ignored): `/Users/govit/Git/Govit/FastSpeechProcessing-legacy/Release/bin/test_costlaws.mat`, `test_costlaws_deriv.mat`; `src/CostLaw.{h,cpp}`, `src/Segmentation.{cpp,h}`.
- Spec: `docs/superpowers/specs/2026-07-01-phase-0b-i-cost-container-design.md`.
- Legacy Quirks backlog: CLAUDE.md `## Legacy Quirks & Deferred Fixes` already lists the 0b-i entries (double-read, LogLaw asymmetry, AboveThreshCubic routing). Task 6 confirms they match the implemented code.

---

### Task 1: Cost-sweep fixtures (scipy prep)

Extract the legacy `.mat` cost/deriv sweeps into committed `.bin` fixtures + the sweep config, so the Rust cost golden runs in CI without scipy/the `.mat`.

**Files:**
- Create: `scripts/extract_phase0b_costlaw_fixtures.py`
- Create (generated, committed): `tests/reference_data/phase0b/{sweep_output.bin, sweep_cost_speech.bin, sweep_cost_nospeech.bin, sweep_deriv_speech.bin, sweep_deriv_nospeech.bin, costlaw_config.json}`
- Test: `tests/test_phase0b_costlaw_fixtures.py`

**Interfaces:**
- Consumes: `speech.weight_bridge.write_bin`/`read_bin` (Phase 0a).
- Produces: five `1 x 1000` `.bin` fixtures (each the corresponding sweep column) + `costlaw_config.json` = `{"prefix": "BLSTM", "keys": {...}}` (the legacy config keys that generated the sweep).

- [ ] **Step 1: Inspect the `.mat` variables**

Run:
```bash
uv run --with scipy python -c "import scipy.io as sio; m=sio.loadmat('/Users/govit/Git/Govit/FastSpeechProcessing-legacy/Release/bin/test_costlaws.mat'); print([k for k in m if not k.startswith('__')], {k: getattr(m[k],'shape',None) for k in m if not k.startswith('__')})"
uv run --with scipy python -c "import scipy.io as sio; m=sio.loadmat('/Users/govit/Git/Govit/FastSpeechProcessing-legacy/Release/bin/test_costlaws_deriv.mat'); print([k for k in m if not k.startswith('__')])"
```
Expected: variables like `cost_speech`, `cost_other` (each `1000 x 2`: col0 = output, col1 = cost) in the first; `cost_deriv_speech`, `cost_deriv_other` in the second. Note the EXACT variable names for the script (adjust if they differ).

- [ ] **Step 2: Write `scripts/extract_phase0b_costlaw_fixtures.py`**

```python
"""Extract Phase-0b-i cost-law golden fixtures from the legacy .mat sweeps (scipy).

test_costlaws.mat holds cost_speech / cost_other (each 1000x2: output=col0, cost=col1);
test_costlaws_deriv.mat holds the derivative sweeps. This emits the five sweep columns as
1x1000 .bin fixtures plus costlaw_config.json (the legacy config that generated them), so the
Rust cost test reproduces them bit-for-bit without scipy.

Usage: uv run --with scipy python scripts/extract_phase0b_costlaw_fixtures.py
Requires the local git-ignored .mat files.
"""

import json
from pathlib import Path

import numpy as np
import scipy.io as sio

from speech import weight_bridge

BIN = Path("/Users/govit/Git/Govit/FastSpeechProcessing-legacy/Release/bin")
OUT = Path("tests/reference_data/phase0b")
# Sweep-generating config, deduced from the anchor output=0.001,target=1 -> 0.1997999997999998
# = LinearLaw(P=cp=0.2, q=0, t=1-1e-6): 0.2 + (-0.2*(1-0)/(1-1e-6))*0.001. Speech law = linear.
# The Rust cost test is the arbiter: if a value diverges, adjust these keys against the sweep.
CONFIG = {
    "prefix": "BLSTM",
    "keys": {
        "BLSTM_CostLawSpeech": "linear",
        "BLSTM_CostLawNoSpeech": "linear",
        "BLSTM_CostPonderation": "0.2",
        "BLSTM_CostLawParamSpeech": "0",
        "BLSTM_CostLawParamNoSpeech": "0",
        "BLSTM_CostLawThreshSpeech": "1",
        "BLSTM_CostLawThreshNoSpeech": "0",
    },
}


def col(mat: dict, var: str, c: int) -> np.ndarray:
    a = np.asarray(mat[var], dtype=np.float64)
    return np.ascontiguousarray(a[:, c])


def main() -> None:
    m = sio.loadmat(BIN / "test_costlaws.mat")
    md = sio.loadmat(BIN / "test_costlaws_deriv.mat")
    OUT.mkdir(parents=True, exist_ok=True)
    output = col(m, "cost_speech", 0)  # output column (same for all sweeps)
    n = output.shape[0]
    weight_bridge.write_bin(1, n, output, OUT / "sweep_output.bin")
    weight_bridge.write_bin(1, n, col(m, "cost_speech", 1), OUT / "sweep_cost_speech.bin")
    weight_bridge.write_bin(1, n, col(m, "cost_other", 1), OUT / "sweep_cost_nospeech.bin")
    weight_bridge.write_bin(1, n, col(md, "cost_deriv_speech", 1), OUT / "sweep_deriv_speech.bin")
    weight_bridge.write_bin(1, n, col(md, "cost_deriv_other", 1), OUT / "sweep_deriv_nospeech.bin")
    (OUT / "costlaw_config.json").write_text(json.dumps(CONFIG, indent=1))
    print(f"OK: wrote {n}-row sweeps + config to {OUT}; anchor cost_speech[output~0.001]={col(m,'cost_speech',1)[0]!r}")


if __name__ == "__main__":
    main()
```

If Step 1 showed different variable names (e.g. `cost_nospeech` instead of `cost_other`), use the actual names.

- [ ] **Step 3: Run the prep**

Run: `uv run --with scipy python scripts/extract_phase0b_costlaw_fixtures.py`
Expected: `OK: wrote 1000-row sweeps + config ...` and the anchor prints `0.1997999997999998` (or the sweep's first cost value).

- [ ] **Step 4: Write the fixture test** `tests/test_phase0b_costlaw_fixtures.py`

```python
"""Assert the Phase-0b-i cost-sweep fixtures are well-formed."""

from pathlib import Path

from speech import weight_bridge

REF = Path("tests/reference_data/phase0b")


def test_sweep_fixture_shapes() -> None:
    for name in ["sweep_output", "sweep_cost_speech", "sweep_cost_nospeech", "sweep_deriv_speech", "sweep_deriv_nospeech"]:
        _, cols, data = weight_bridge.read_bin(REF / f"{name}.bin")
        assert cols == 1000
        assert data.shape == (1000,)


def test_output_column_monotone() -> None:
    _, _, out = weight_bridge.read_bin(REF / "sweep_output.bin")
    assert (out[1:] >= out[:-1]).all()  # output sweep is a monotone grid
    assert 0.0 <= out.min() and out.max() <= 1.0
```

- [ ] **Step 5: Verify + commit**

```bash
uv run pytest tests/test_phase0b_costlaw_fixtures.py -q
uv run ruff check --config=pyproject.toml scripts tests && uv run ruff format --config=pyproject.toml scripts tests
git add scripts/extract_phase0b_costlaw_fixtures.py tests/test_phase0b_costlaw_fixtures.py tests/reference_data/phase0b
git commit -m "feat(phase0b-i): cost-sweep golden fixtures (scipy prep)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: The 7 cost-law primitives + scalar VAD path (the crux)

Implement `CostLaw` (params + double-read), the 7 primitives, and `compute_unitary_cost`/`compute_unitary_delta`, validated bit-for-bit against the sweep fixtures.

**Files:**
- Modify: `src/rust/src/cost.rs` (replace the stub)
- Test: `src/rust/tests/phase0b_costlaw.rs`

**Interfaces:**
- Consumes: `speech::legacy_config::parse_legacy_config` (Phase 0a), the Task-1 fixtures.
- Produces: `speech::cost::CostLaw` with `from_config(&indexmap::IndexMap<String,String>, prefix: &str) -> CostLaw`, `compute_unitary_cost(&self, output: f64, target: f64) -> f64`, `compute_unitary_delta(&self, output: f64, target: f64) -> f64`.

- [ ] **Step 1: Write the failing golden test** `src/rust/tests/phase0b_costlaw.rs`

```rust
//! Bit-exact cost-law tests vs the legacy test_costlaws.mat sweeps.

use std::path::PathBuf;

use indexmap::IndexMap;

fn ref_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/reference_data/phase0b")
}

fn read_sweep(name: &str) -> Vec<f64> {
    speech::io::binary::read_weight_vector(&ref_dir().join(format!("{name}.bin"))).unwrap()
}

fn sweep_config() -> IndexMap<String, String> {
    // Mirror tests/reference_data/phase0b/costlaw_config.json (linear, cp=0.2, thresh 1/0).
    let mut m = IndexMap::new();
    for (k, v) in [
        ("BLSTM_CostLawSpeech", "linear"),
        ("BLSTM_CostLawNoSpeech", "linear"),
        ("BLSTM_CostPonderation", "0.2"),
        ("BLSTM_CostLawParamSpeech", "0"),
        ("BLSTM_CostLawParamNoSpeech", "0"),
        ("BLSTM_CostLawThreshSpeech", "1"),
        ("BLSTM_CostLawThreshNoSpeech", "0"),
    ] {
        m.insert(k.to_string(), v.to_string());
    }
    m
}

#[test]
fn cost_sweep_bit_exact() {
    let law = speech::cost::CostLaw::from_config(&sweep_config(), "BLSTM");
    let output = read_sweep("sweep_output");
    let cost_speech = read_sweep("sweep_cost_speech");
    let cost_nospeech = read_sweep("sweep_cost_nospeech");
    for i in 0..output.len() {
        assert_eq!(law.compute_unitary_cost(output[i], 1.0), cost_speech[i], "speech cost row {i}");
        assert_eq!(law.compute_unitary_cost(output[i], 0.0), cost_nospeech[i], "nospeech cost row {i}");
    }
}

#[test]
fn deriv_sweep_bit_exact() {
    let law = speech::cost::CostLaw::from_config(&sweep_config(), "BLSTM");
    let output = read_sweep("sweep_output");
    let d_speech = read_sweep("sweep_deriv_speech");
    let d_nospeech = read_sweep("sweep_deriv_nospeech");
    for i in 0..output.len() {
        assert_eq!(law.compute_unitary_delta(output[i], 1.0), d_speech[i], "speech deriv row {i}");
        assert_eq!(law.compute_unitary_delta(output[i], 0.0), d_nospeech[i], "nospeech deriv row {i}");
    }
}
```

- [ ] **Step 2: Run, expect failure.** `cd src/rust && cargo test --test phase0b_costlaw` -> FAIL (CostLaw not defined).

- [ ] **Step 3: Implement `src/rust/src/cost.rs`** (replace the stub; keep the module doc-comment). This is the exact port of `CostLaw.{h,cpp}` sections 3.1-3.3 of the spec.

```rust
//! Piecewise VAD cost laws + multiclass softmax cross-entropy.
//!
//! Ported bit-exactly from legacy C++: CostLaw.{h,cpp}. Preserves the double-read
//! quirk, the LogLaw cost/deriv asymmetry, and the AboveThreshCubic name routing
//! (see CLAUDE.md Legacy Quirks). See design spec section 3.

use indexmap::IndexMap;

fn clamp(x: f64, lo: f64, hi: f64) -> f64 {
    x.max(lo).min(hi)
}

/// One piecewise law primitive; coefficients precomputed from (P, q, t) and name.
#[derive(Debug, Clone, Copy)]
enum Law {
    Linear { a: f64, b: f64 },
    Square { a: f64, b: f64, c: f64 },
    BelowCubic { a: f64, b: f64, d: f64 },
    AboveCubic { a: f64, b: f64 },
    Log { a: f64, b: f64, adim: f64 },
    BelowSqrt { a: f64, b: f64, adim: f64 },
    AboveSqrt { a: f64, b: f64, adim: f64 },
}

impl Law {
    fn cost(&self, y: f64) -> f64 {
        match *self {
            Law::Linear { a, b } => b + a * y,
            Law::Square { a, b, c } => c + b * y + a * y * y,
            Law::BelowCubic { a, b, d } => d + b * y * y + a * y * y * y,
            Law::AboveCubic { a, b } => {
                let y = 1.0 - y;
                b * y * y + a * y * y * y
            }
            Law::Log { a, b, adim } => {
                let mut y = y / adim;
                if y < 1e-24 {
                    y = 1e-24;
                }
                if y > 1.0 {
                    y = 1.0;
                }
                b + a * y.ln()
            }
            Law::BelowSqrt { a, b, adim } => {
                let mut y = y / adim;
                if y > 1.0 {
                    y = 1.0;
                }
                b + a * (1.0 - y).sqrt()
            }
            Law::AboveSqrt { a, b, adim } => {
                let mut y = 1.0 - y;
                y /= adim;
                if y > 1.0 {
                    y = 1.0;
                }
                b + a * (1.0 - y).sqrt()
            }
        }
    }

    fn deriv(&self, y: f64) -> f64 {
        match *self {
            Law::Linear { a, .. } => a,
            Law::Square { a, b, .. } => b + 2.0 * a * y,
            Law::BelowCubic { a, b, .. } => 2.0 * b * y + 3.0 * a * y * y,
            Law::AboveCubic { a, b } => {
                let y = 1.0 - y;
                -2.0 * b * y - 3.0 * a * y * y
            }
            Law::Log { a, .. } => {
                // ASYMMETRY: deriv clamps RAW y (not divided by adim). Legacy bug, reproduced.
                let mut y = y;
                if y < 1e-24 {
                    y = 1e-24;
                }
                if y > 1.0 {
                    y = 1.0;
                }
                a / y
            }
            Law::BelowSqrt { a, adim, .. } => {
                let mut y = 1.0 - y / adim;
                if y < 1e-24 {
                    y = 1e-24;
                }
                -a / 2.0 / y.sqrt()
            }
            Law::AboveSqrt { a, adim, .. } => {
                let mut y = 1.0 - y;
                y = 1.0 - y / adim;
                if y < 1e-24 {
                    y = 1e-24;
                }
                a / 2.0 / y.sqrt()
            }
        }
    }
}

fn below_law(name: &str, p: f64, q: f64, t: f64) -> Law {
    match name {
        "linear" => Law::Linear { a: -p * (1.0 - q) / t, b: p },
        "square" => Law::Square { a: p * (1.0 - q) / t / t, b: -p * 2.0 * (1.0 - q) / t, c: p },
        "cubic" => Law::BelowCubic { a: p * 2.0 * (1.0 - q) / t / t / t, b: -p * 3.0 * (1.0 - q) / t / t, d: p },
        "log" => Law::Log { a: -p * (1.0 - q), b: p * q, adim: t },
        other => panic!("cost law not valid: {other}"),
    }
}

fn above_cubic(name: &str, p: f64, q: f64, t: f64) -> Law {
    let (a, b) = match name {
        "square" | "cubic" => (-p * 2.0 * q / (1.0 - t).powi(3), p * 3.0 * q / (1.0 - t).powi(2)),
        "linear" | "log" => (
            p * ((1.0 - q) / t / (1.0 - t).powi(2) - 2.0 * q / (1.0 - t).powi(3)),
            p * (3.0 * q / (1.0 - t).powi(2) - (1.0 - q) / t / (1.0 - t)),
        ),
        other => panic!("above-thresh cubic name not valid: {other}"),
    };
    Law::AboveCubic { a, b }
}

/// Built law pair for one regime (below-thresh + above-thresh), plus the sqrt special case.
#[derive(Debug, Clone, Copy)]
struct RegimeLaws {
    below: Law,
    above: Law,
}

impl RegimeLaws {
    fn build(name: &str, p: f64, q: f64, t: f64) -> RegimeLaws {
        if name == "sqrt" {
            RegimeLaws {
                below: Law::BelowSqrt { a: p * (1.0 - q), b: p * q, adim: t },
                above: Law::AboveSqrt { a: p * (-q), b: p * q, adim: 1.0 - t },
            }
        } else {
            RegimeLaws { below: below_law(name, p, q, t), above: above_cubic(name, p, q, t) }
        }
    }
}

/// The VAD cost law (scalar path) + softmax config, ported from CostLaw.
#[derive(Debug, Clone)]
pub struct CostLaw {
    speech: RegimeLaws,
    no_speech: RegimeLaws,
    switching_thresh_speech: f64,
    switching_thresh_no_speech: f64,
    back_prop_wer: bool,
    classes_ponderations: Vec<f64>,
    speech_name: String,
    no_speech_name: String,
}

fn getf(m: &IndexMap<String, String>, key: &str, default: f64) -> f64 {
    m.get(key).and_then(|v| v.trim().parse::<f64>().ok()).unwrap_or(default)
}

fn gets(m: &IndexMap<String, String>, key: &str, default: &str) -> String {
    m.get(key).cloned().unwrap_or_else(|| default.to_string())
}

impl CostLaw {
    /// Build from a parsed legacy config, using the algName prefix (e.g. "BLSTM").
    pub fn from_config(m: &IndexMap<String, String>, prefix: &str) -> CostLaw {
        let p = format!("{prefix}_");
        let cp = clamp(getf(m, &format!("{p}CostPonderation"), 0.5), 0.01, 0.99);
        let ps = clamp(getf(m, &format!("{p}CostLawParamSpeech"), 0.0), 0.0, 1.0);
        let pn = clamp(getf(m, &format!("{p}CostLawParamNoSpeech"), 0.0), 0.0, 1.0);
        let th_s = clamp(getf(m, &format!("{p}CostLawThreshSpeech"), 1.0), 1e-6, 1.0 - 1e-6);
        let th_n_law = 1.0 - clamp(getf(m, &format!("{p}CostLawThreshNoSpeech"), 0.0), 1e-6, 1.0 - 1e-6);
        let speech_name = gets(m, &format!("{p}CostLawSpeech"), "linear");
        let no_speech_name = gets(m, &format!("{p}CostLawNoSpeech"), "linear");
        // DECISIVE DOUBLE-READ: branch selectors re-read raw/unclamped (defaults 10.0 / -1.0).
        let switching_thresh_speech = getf(m, &format!("{p}CostLawThreshSpeech"), 10.0);
        let switching_thresh_no_speech = getf(m, &format!("{p}CostLawThreshNoSpeech"), -1.0);
        let back_prop_wer = getf(m, &format!("{p}BackPropWER"), -1.0) >= 0.0;
        let cponds = m
            .get(&format!("{p}classes_ponderations"))
            .map(|v| v.split(',').filter_map(|s| s.trim().parse::<f64>().ok()).collect::<Vec<_>>())
            .unwrap_or_default();
        let classes_ponderations = if !cponds.is_empty() && cponds[0] > 0.0 { cponds } else { Vec::new() };
        CostLaw {
            speech: RegimeLaws::build(&speech_name, cp, ps, th_s),
            no_speech: RegimeLaws::build(&no_speech_name, 1.0 - cp, pn, th_n_law),
            switching_thresh_speech,
            switching_thresh_no_speech,
            back_prop_wer,
            classes_ponderations,
            speech_name,
            no_speech_name,
        }
    }

    /// Scalar VAD cost for one (output, target). target<0 => ignored (0).
    pub fn compute_unitary_cost(&self, output: f64, target: f64) -> f64 {
        if target < 0.0 {
            return 0.0;
        }
        if target > 0.5 {
            if output < self.switching_thresh_speech { self.speech.below.cost(output) } else { self.speech.above.cost(output) }
        } else if output < self.switching_thresh_no_speech {
            self.no_speech.below.cost(output)
        } else {
            self.no_speech.above.cost(output)
        }
    }

    /// Scalar VAD delta = law derivative * logistic chain rule output*(1-output). target<0 => 0.
    pub fn compute_unitary_delta(&self, output: f64, target: f64) -> f64 {
        if target < 0.0 {
            return 0.0;
        }
        let d = if target > 0.5 {
            if output < self.switching_thresh_speech { self.speech.below.deriv(output) } else { self.speech.above.deriv(output) }
        } else if output < self.switching_thresh_no_speech {
            self.no_speech.below.deriv(output)
        } else {
            self.no_speech.above.deriv(output)
        };
        d * output * (1.0 - output)
    }
}
```

IMPORTANT: the exact `compute_unitary_cost`/`delta` branch structure and the logistic-chain-rule factor MUST match `CostLaw.cpp:122-189`. If the sweep test fails, read `CostLaw.cpp` lines 122-189 and reconcile (esp. whether the delta multiplies by `output*(1-output)` for BOTH regimes, and the exact below/above predicate). The sweep is the arbiter; do NOT weaken the assertion. If the deduced `costlaw_config.json` produces a mismatch, adjust the config keys (Task 1) to the values that reproduce the sweep, re-run the prep, and re-test.

- [ ] **Step 4: Run the golden tests.** `cd src/rust && cargo test --test phase0b_costlaw` -> both tests PASS (bit-exact over 1000 rows x 2 regimes).

- [ ] **Step 5: Add per-law + double-read unit tests** to `phase0b_costlaw.rs`

```rust
#[test]
fn double_read_diverges_when_key_absent() {
    // With ThreshSpeech absent: law uses clamped 1-1e-6, branch uses raw default 10.0.
    let m: IndexMap<String, String> = [("BLSTM_CostLawSpeech".to_string(), "log".to_string())].into_iter().collect();
    let law = speech::cost::CostLaw::from_config(&m, "BLSTM");
    // output 0.5 < 10.0 => below-thresh log law is used (would be above-thresh if branch used 1-1e-6).
    let c = law.compute_unitary_cost(0.5, 1.0);
    assert!(c.is_finite());
}
```

(Add square/cubic/log/sqrt cost+deriv point checks computed by hand from the section-3.2 coefficients; e.g. `LinearLaw(P=0.2,q=0,t=1-1e-6).cost(0.001) == 0.1997999997999998`. Show each with its hand-computed expected value.)

- [ ] **Step 6: Lint + commit**

```bash
cd src/rust && cargo clippy --all-targets -- -D warnings && cargo fmt --all && cargo test --test phase0b_costlaw && cd ../..
git add src/rust/src/cost.rs src/rust/tests/phase0b_costlaw.rs
# if Task 1 config was adjusted, also: git add tests/reference_data/phase0b scripts/extract_phase0b_costlaw_fixtures.py
git commit -m "feat(phase0b-i): 7 cost-law primitives + scalar VAD path, sweep-golden bit-exact

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Multiclass softmax cross-entropy path

Add `compute_cost`/`compute_deltas` for the multiclass (>1 target column) path.

**Files:**
- Modify: `src/rust/src/cost.rs`
- Test: `src/rust/tests/phase0b_softmax.rs`

**Interfaces:**
- Produces: `CostLaw::compute_cost(&self, outputs: &[f64], onehot_target: &[f64], n_classes: usize) -> f64` and `compute_deltas(&self, outputs: &[f64], onehot_target: &[f64], n_classes: usize, out: &mut [f64])`.

- [ ] **Step 1: Read the legacy multiclass branch.** Read `src/CostLaw.cpp` for the `computeCost`/`computeDeltas` path taken when `targetSeq` has > 1 column (softmax cross-entropy). Note the exact ignore-mask rule (`target < 0` row), the class-ponderation application, and the WER-ponderation gate.

- [ ] **Step 2: Write the failing test** `src/rust/tests/phase0b_softmax.rs`

```rust
//! Multiclass softmax cross-entropy tests.

#[test]
fn delta_is_output_minus_onehot() {
    // 2 frames x 3 classes; frame 0 target class 1, frame 1 ignored (all target < 0).
    let law = speech::cost::CostLaw::from_config(&Default::default(), "BLSTM");
    let outputs = vec![0.2, 0.5, 0.3, 0.1, 0.6, 0.3];
    let target = vec![0.0, 1.0, 0.0, -1.0, -1.0, -1.0];
    let mut deltas = vec![0.0; 6];
    law.compute_deltas(&outputs, &target, 3, &mut deltas);
    assert_eq!(deltas[0], 0.2 - 0.0);
    assert_eq!(deltas[1], 0.5 - 1.0);
    assert_eq!(deltas[2], 0.3 - 0.0);
    assert_eq!(&deltas[3..6], &[0.0, 0.0, 0.0]); // ignored frame
}
```

- [ ] **Step 3: Run, expect failure.** `cd src/rust && cargo test --test phase0b_softmax` -> FAIL.

- [ ] **Step 4: Implement `compute_cost`/`compute_deltas` in `cost.rs`** per the `CostLaw.cpp` multiclass branch read in Step 1: cross-entropy cost `-sum onehot*ln(output)` over non-ignored frames (frame ignored when its target row is all `< 0`), `delta = output - onehot`, with the class-ponderation (`classes_ponderations`) and WER-ponderation (`back_prop_wer`) applied exactly as the legacy does. Show the full implementation.

- [ ] **Step 5: Run, expect pass.** `cd src/rust && cargo test --test phase0b_softmax` -> PASS.

- [ ] **Step 6: Lint + commit**

```bash
cd src/rust && cargo clippy --all-targets -- -D warnings && cargo fmt --all && cd ../..
git add src/rust/src/cost.rs src/rust/tests/phase0b_softmax.rs
git commit -m "feat(phase0b-i): multiclass softmax cross-entropy cost path

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Segmentation container - model + label_segment + sanitize

**Files:**
- Modify: `src/rust/src/tasks/segmentation.rs` (replace the stub `Segment`)
- Test: `src/rust/tests/phase0b_segmentation.rs`

**Interfaces:**
- Produces: `speech::tasks::segmentation::{SegClass (#[repr(i32)]), Segment{begin:f64, ty:SegClass}, Segmentation}`. `Segmentation::new(audio_duration: f64) -> Self`, `label_segment(&mut self, begin: f64, end: f64, class: SegClass)`, `sanitize(&mut self)`, `segments(&self) -> &[Segment]`, `audio_duration(&self) -> f64`.

- [ ] **Step 1: Write failing tests** `src/rust/tests/phase0b_segmentation.rs`

```rust
//! Segmentation container tests (hand-verified boundary lists).

use speech::tasks::segmentation::{SegClass, Segmentation};

fn begins(s: &Segmentation) -> Vec<f64> {
    s.segments().iter().map(|x| x.begin).collect()
}
fn types(s: &Segmentation) -> Vec<SegClass> {
    s.segments().iter().map(|x| x.ty).collect()
}

#[test]
fn new_is_other_then_end_sentinel() {
    let s = Segmentation::new(120.0);
    assert_eq!(begins(&s), vec![0.0, 120.0]);
    assert_eq!(types(&s), vec![SegClass::Other, SegClass::End]);
}

#[test]
fn label_inserts_typed_interval() {
    let mut s = Segmentation::new(10.0);
    s.label_segment(2.0, 5.0, SegClass::Speech);
    assert_eq!(begins(&s), vec![0.0, 2.0, 5.0, 10.0]);
    assert_eq!(types(&s), vec![SegClass::Other, SegClass::Speech, SegClass::Other, SegClass::End]);
}

#[test]
fn sanitize_rounds_1e4_half_away_and_merges() {
    let mut s = Segmentation::new(10.0);
    s.label_segment(2.00004, 5.0, SegClass::Speech); // 2.00004 -> 2.0000 (round to 1e-4 grid)
    s.label_segment(5.0, 7.0, SegClass::Speech); // adjacent same-type -> merge on sanitize
    s.sanitize();
    assert_eq!(types(&s), vec![SegClass::Other, SegClass::Speech, SegClass::Other, SegClass::End]);
    assert_eq!(begins(&s)[1], 2.0000);
    // terminal sentinel unchanged
    assert_eq!(*begins(&s).last().unwrap(), 10.0);
}
```

- [ ] **Step 2: Run, expect failure.** `cd src/rust && cargo test --test phase0b_segmentation` -> FAIL.

- [ ] **Step 3: Implement the model + `label_segment` + `sanitize` in `tasks/segmentation.rs`** per spec section 4. `SegClass` `#[repr(i32)]` with the exact codes. `Segmentation { segs: Vec<Segment>, audio_duration: f64 }`. `new` seeds `[{0.0, Other}, {audio_duration, End}]`. `label_segment` implements the `:147-174` insert/erase/reclose with `previous_type` (index-based; show the full logic). `sanitize` (`:176-194`): for each adjacent pair, round `begin` via `(t * 1e4).round() / 1e4` (Rust `f64::round` is half-away-from-zero - assert this in a comment), merge same-type by removing the later boundary; do NOT round the terminal sentinel. Show complete code.

- [ ] **Step 4: Run, expect pass.** `cargo test --test phase0b_segmentation` -> the three tests PASS.

- [ ] **Step 5: Lint + commit**

```bash
cd src/rust && cargo clippy --all-targets -- -D warnings && cargo fmt --all && cd ../..
git add src/rust/src/tasks/segmentation.rs src/rust/tests/phase0b_segmentation.rs
git commit -m "feat(phase0b-i): segmentation container model + label_segment + sanitize

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Segmentation container - suppress_short + add_padding + modify_type + update_count

**Files:**
- Modify: `src/rust/src/tasks/segmentation.rs`
- Test: `src/rust/tests/phase0b_segmentation.rs` (extend)

**Interfaces:**
- Produces: `Segmentation::{suppress_short(&mut self, threshold: f64, class: SegClass), add_padding(&mut self, before: f64, after: f64, class: SegClass), modify_type(&mut self, before: SegClass, after: SegClass), update_count(&mut self) -> [f64; 23]}`.

- [ ] **Step 1: Write failing tests** (append to `phase0b_segmentation.rs`)

```rust
#[test]
fn suppress_short_drops_and_merges() {
    let mut s = Segmentation::new(10.0);
    s.label_segment(2.0, 2.05, SegClass::Speech); // 0.05s speech, below threshold 0.1
    s.suppress_short(0.1, SegClass::Speech);
    // the short speech is removed; neighbors (both Other) merge -> whole thing is Other
    assert_eq!(types(&s), vec![SegClass::Other, SegClass::End]);
}

#[test]
fn suppress_short_threshold_zero_is_noop() {
    let mut s = Segmentation::new(10.0);
    s.label_segment(2.0, 2.05, SegClass::Speech);
    let before = begins(&s);
    s.suppress_short(0.0, SegClass::Speech);
    assert_eq!(begins(&s), before);
}

#[test]
fn update_count_sums_class_durations() {
    let mut s = Segmentation::new(10.0);
    s.label_segment(2.0, 5.0, SegClass::Speech);
    let counts = s.update_count();
    assert_eq!(counts[SegClass::Speech as usize], 3.0);
    assert_eq!(counts[SegClass::Other as usize], 7.0);
}
```

- [ ] **Step 2: Run, expect failure.** `cargo test --test phase0b_segmentation` -> the new tests FAIL.

- [ ] **Step 3: Implement `suppress_short` + `add_padding` + `modify_type` + `update_count`** per spec section 4.2. `suppress_short` (`:245-282`): `threshold > 0` gate; the head / pre-sentinel / neighbor-match-full-merge / neighbor-differ-half-split branches; NO advance after an erase; `previous_type` updated only on the kept branch; then `sanitize`. `add_padding` (`:216-243`): grow each `class` segment by `before`/`after` via `label_segment`, skipping a side when its value is 0. `modify_type`: retype non-sentinel `before -> after` then `sanitize`. `update_count` (`:206`): `sanitize`, then accumulate `next.begin - cur.begin` into `[f64; 23]` indexed by `ty as usize`. Show complete code.

- [ ] **Step 4: Run, expect pass.** `cargo test --test phase0b_segmentation` -> all PASS.

- [ ] **Step 5: Add a proptest** that `sanitize` is idempotent (`s.sanitize(); let a = begins(); s.sanitize(); assert_eq!(a, begins())`) and never reorders boundaries (non-decreasing). Show the proptest.

- [ ] **Step 6: Lint + commit**

```bash
cd src/rust && cargo clippy --all-targets -- -D warnings && cargo fmt --all && cargo test && cd ../..
git add src/rust/src/tasks/segmentation.rs src/rust/tests/phase0b_segmentation.rs
git commit -m "feat(phase0b-i): segmentation suppress_short/add_padding/modify_type/update_count

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: Docs + quirks backlog confirmation

**Files:**
- Modify: `CLAUDE.md`, `README.md`

- [ ] **Step 1: Update `CLAUDE.md`.** In the Rust architecture table, change the `cost.rs` and `tasks/segmentation.rs` rows from stub descriptions to "Implemented (Phase 0b-i): <one line>". Confirm the `## Legacy Quirks & Deferred Fixes` entries for `[0b-i]` (double-read, LogLaw asymmetry, AboveThreshCubic routing) match the implemented code (file/line references accurate); adjust wording if needed. ASCII only; `LC_ALL=C grep -n '[^ -~]' CLAUDE.md` returns nothing.

- [ ] **Step 2: Update `README.md`.** Mark Phase 0b-i done in the roadmap: cost laws golden-tested bit-for-bit vs `test_costlaws.mat`; the Segmentation container unit-tested. Leave 0b-ii pending. ASCII only.

- [ ] **Step 3: Verify + commit**

```bash
LC_ALL=C grep -n '[^ -~]' CLAUDE.md README.md; echo "ascii checked"
git add CLAUDE.md README.md
git commit -m "docs(phase0b-i): mark cost + container implemented; confirm quirks backlog

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 7: Full-suite verification + smart-commit

**Files:** none.

- [ ] **Step 1: Run the toolchain**

```bash
./check_all.sh
uv run ruff check --config=pyproject.toml src/python tests scripts && uv run ruff format --check --config=pyproject.toml src/python tests scripts
uv run mypy --config-file pyproject.toml src/python tests
uv run pytest tests -q
```
Expected: `check_all.sh` -> All checks passed; ruff/format/mypy clean; pytest all pass.

- [ ] **Step 2: Confirm Definition of Done** (spec section 8): cost sweep golden bit-for-bit in Rust; container + law + softmax unit tests pass; docs updated; quirks backlog current.

- [ ] **Step 3: Invoke the `smart-commit` skill** over the whole `feature/phase-0b-i-cost-container` branch. Do not push.

---

## Self-Review

**1. Spec coverage.** Spec s3.1 (params + double-read) -> Task 2 `from_config`. s3.2 (7 primitives) -> Task 2 `Law`. s3.3 (scalar path) -> Task 2 `compute_unitary_*`. s3.4 (softmax) -> Task 3. s4.1-4.2 (container model + ops) -> Tasks 4-5. s4.3 (container tests) -> Tasks 4-5. s5 (interfaces) -> Task signatures. s6 (fixtures/golden/tests): fixtures -> Task 1; cost golden -> Task 2; law/double-read/softmax/container units -> Tasks 2-5. s7 risks pinned: double-read (Task 2 test + `from_config`), LogLaw asymmetry (Task 2 `Law::deriv`), AboveThreshCubic routing (Task 2 `above_cubic`), sanitize rounding (Task 4), enum codes (Task 4), suppress_short branches (Task 5). s8 DoD -> Task 7. s9 out-of-scope respected (no segmenter/IO/compute_errors/matfile). Covered.

**2. Placeholder scan.** No TBD/TODO. Task 3 Step 4 and Task 4/5 Step 3 describe the implementation with the exact spec section + legacy line references rather than pre-writing every line, because the softmax exact form and the container insert/erase/suppress branch code must be read from `CostLaw.cpp` / `Segmentation.cpp` to be bit-exact (the tests + goldens are the gate); the tractable, formula-driven parts (the 7 `Law` primitives, `from_config`, the scalar dispatch, `new`/`label_segment`/`sanitize` structure) are shown in full. Task 2 Step 5 says "add per-law checks ... with hand-computed expected value" - the implementer computes each from the shown section-3.2 coefficients (the anchor is given in full).

**3. Type consistency.** `CostLaw::{from_config, compute_unitary_cost, compute_unitary_delta, compute_cost, compute_deltas}` consistent across Tasks 2-3 and their tests. `SegClass`/`Segment`/`Segmentation::{new,label_segment,sanitize,suppress_short,add_padding,modify_type,update_count,segments,audio_duration}` consistent across Tasks 4-5 and tests. `SegClass::{Other,Speech,End}` variant names used in tests match the `#[repr(i32)]` enum defined in Task 4. `weight_bridge.{read_bin,write_bin}` / `io::binary::read_weight_vector` consumed as defined in Phase 0a. Fixture names (`sweep_output`, `sweep_cost_speech`, `sweep_cost_nospeech`, `sweep_deriv_speech`, `sweep_deriv_nospeech`, `costlaw_config.json`) identical in the prep script (Task 1), the fixture test (Task 1), and the cost golden test (Task 2).
