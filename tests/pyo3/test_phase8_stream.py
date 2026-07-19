"""Phase 8 Task 6: `speech_rs.StreamingSession` binding replay + chunk-invariance.

`importorskip` guards the whole module: the plain `python-test` CI job never builds
`speech_rs`, so it SKIPS these; only the `python-pyo3` job (which runs `maturin develop`)
exercises them.

The binding is a THIN wrapper over `speech::fast::stream::StreamingSession` (the Rust
streaming session), so replaying the SAME staged config + samples through it reproduces the
Rust session bit-for-bit by construction -- the observable Python-side cross-checks are:
  - CHUNK-INVARIANCE: two different push granularities produce the identical emitted
    (begin/end/class) SET and the identical final segmentation (chunking changes timing,
    never arithmetic -- the S1.6b property, surfaced through the binding);
  - PREFIX CONSISTENCY: the emitted set equals the final partition (no retraction, no extra).

This uses the PRIMARY frozen tuple-A tail (thin but valid -- the tuple-A net saturates under
the frozen type-1 tail, yielding the [Other@0, End@dur] seed), mirroring the Rust gate's own
primary leg. The CALIBRATED real-segment path is Rust-gate-only (it needs `FastPipeline`,
which is not exposed to Python); the cross-language REAL-segment pin lives in the Rust CLI
test (`src/rust/tests/phase8_cli.rs`).
"""

import wave
from pathlib import Path

import numpy as np
import pytest

speech_rs = pytest.importorskip("speech_rs")

REPO_ROOT = Path(__file__).resolve().parents[2]
PHASE4A = REPO_ROOT / "tests" / "reference_data" / "phase4a"
PHASE4D = REPO_ROOT / "tests" / "reference_data" / "phase4d"
PHASE0 = REPO_ROOT / "tests" / "reference_data" / "phase0"

Triple = tuple[float, float, str]


def _read_stereo_chan0(path: Path) -> tuple[int, np.ndarray]:
    """Read a PCM16 stereo wav, returning (sample_rate, channel-0 int16 samples)."""
    with wave.open(str(path), "rb") as w:
        rate = w.getframerate()
        channels = w.getnchannels()
        assert w.getsampwidth() == 2, "fixture must be PCM16"
        frames = w.readframes(w.getnframes())
    interleaved = np.frombuffer(frames, dtype="<i2")
    assert channels == 2, "prcts_excerpt.wav is stereo (mono-extraction)"
    return rate, interleaved[0::channels].copy()


def _measure_gain(chan0_i16: np.ndarray) -> float:
    """`(2*rms + max_abs)/2` over `i16/32768` -- the exact `measure_fixed_gain` formula."""
    x = chan0_i16.astype(np.float64) / 32768.0
    rms = float(np.sqrt(np.mean(x * x)))
    max_abs = float(np.max(np.abs(x)))
    return (2.0 * rms + max_abs) / 2.0


def _stage_frozen_config(tmp_path: Path, gain: float) -> Path:
    """Mirror `common::stage_frozen_tier2`'s override tail on the committed base config +
    committed tuple-A pack (absolute paths), writing a frozen streaming config."""
    base = (PHASE4A / "tier2_spectral.config").read_text()
    pack = PHASE0 / "NNweights_config1.bin"
    dump = tmp_path / "vrcts_frozen"
    dump.mkdir(exist_ok=True)
    staged = (
        f"{base}\n"
        "# ==== Phase 8 Task 6 frozen-norm overrides (last-wins) ====\n"
        "numOuterThreads 1\n"
        "Audio_offset 0.0\n"
        "Audio_max_duration 3600\n"
        f"BLSTM_weightsFile {pack}\n"
        f"Dump_Directory {dump}\n"
        "Inference_Path fast\n"
        f"Audio_fixed_gain {gain}\n"
        "BLSTM_InputNormalizationType 1\n"
    )
    cfg = tmp_path / "frozen_tier2.config"
    cfg.write_text(staged)
    return cfg


def _run(
    cfg: Path, rate: int, samples: np.ndarray, chunk: int
) -> tuple[list[tuple[float, float, str, float]], list[tuple[float, float, str, float]], list[tuple[float, str]]]:
    """Push `samples` (f32) through a fresh session in `chunk`-sample blocks; return
    (push emissions, finish emissions, segmentation rows)."""
    sess = speech_rs.StreamingSession(str(cfg), float(rate), 1)
    push_em: list[tuple[float, float, str, float]] = []
    n = len(samples)
    i = 0
    while i < n:
        j = min(i + chunk, n)
        push_em.extend(sess.push(samples[i:j]))
        i = j
    fin_em, seg_rows = sess.finish()
    return push_em, fin_em, seg_rows


def _triples(emissions: list[tuple[float, float, str, float]]) -> set[Triple]:
    """Emission identities (begin/end/class), EXCLUDING emitted_at (timing, chunk-dependent)."""
    return {(e[0], e[1], e[2]) for e in emissions}


def _partition(seg_rows: list[tuple[float, str]]) -> set[Triple]:
    """The final partition as (begin, end, class) triples: segment i is [rows[i], rows[i+1])."""
    return {(seg_rows[i][0], seg_rows[i + 1][0], seg_rows[i][1]) for i in range(len(seg_rows) - 1)}


def test_streaming_session_chunk_invariance_and_prefix(tmp_path: Path) -> None:
    rate, chan0 = _read_stereo_chan0(PHASE4D / "prcts_excerpt.wav")
    gain = _measure_gain(chan0)
    cfg = _stage_frozen_config(tmp_path, gain)
    samples = (chan0.astype(np.float32) / np.float32(32768.0)).copy()

    chunk_a = int(round(0.1 * rate))  # 100 ms
    chunk_b = 101  # a non-divisor odd granularity

    push_a, fin_a, seg_a = _run(cfg, rate, samples, chunk_a)
    push_b, fin_b, seg_b = _run(cfg, rate, samples, chunk_b)

    # Structural sanity: each emission is (float, float, str, float), class in {Speech, Other}.
    for e in push_a + fin_a + push_b + fin_b:
        assert isinstance(e, tuple) and len(e) == 4
        assert isinstance(e[0], float) and isinstance(e[1], float) and isinstance(e[3], float)
        assert e[2] in {"Speech", "Other"}, f"unexpected class {e[2]}"

    # Non-vacuity: a real segmentation (>= the 2-row seed) with the correct span + tail. The
    # FIRST class is data-dependent (the frozen tuple-A tail can saturate to either label),
    # so we pin only the structural invariants: starts at 0.0, ends on the End sentinel.
    assert len(seg_a) >= 2, "segmentation must be non-empty (>= the seed)"
    assert seg_a[0][0] == 0.0, "the partition must start at 0.0"
    assert seg_a[0][1] in {"Speech", "Other"}, f"first segment class {seg_a[0][1]}"
    assert seg_a[-1][1] == "End", "the last segmentation row is the End sentinel"
    all_a = push_a + fin_a
    assert len(all_a) >= 1, "at least one segment must be emitted"

    # (1) CHUNK-INVARIANCE: the emitted SET and the final segmentation are bit-identical
    # across push granularities (chunking changes only timing).
    assert _triples(all_a) == _triples(push_b + fin_b), "emitted SET must be chunk-invariant"
    assert seg_a == seg_b, "the final segmentation must be chunk-invariant"

    # (2) PREFIX CONSISTENCY: the emitted set equals the final partition (no missing, no extra).
    part_a = _partition(seg_a)
    assert _triples(all_a) == part_a, "emitted set must equal the final partition"
    assert len(all_a) == len(part_a), "emitted count must equal the partition size (no re-emission)"

    # The session's running lag stats are exposed + sane (>= 0, max >= mean).
    sess = speech_rs.StreamingSession(str(cfg), float(rate), 1)
    i = 0
    while i < len(samples):
        j = min(i + chunk_a, len(samples))
        sess.push(samples[i:j])
        i = j
    sess.finish()
    assert sess.max_lag_s() >= sess.mean_lag_s() >= 0.0


def test_streaming_session_rejects_stereo(tmp_path: Path) -> None:
    # The session is mono-first: constructing with channels != 1 must raise (the Rust bail
    # surfaces as a RuntimeError via the {e:#} seam).
    rate, chan0 = _read_stereo_chan0(PHASE4D / "prcts_excerpt.wav")
    cfg = _stage_frozen_config(tmp_path, _measure_gain(chan0))
    with pytest.raises(RuntimeError, match="mono"):
        speech_rs.StreamingSession(str(cfg), float(rate), 2)
