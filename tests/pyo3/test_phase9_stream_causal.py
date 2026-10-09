"""Phase 9 Task 7: `speech_rs.StreamingSession` on a CAUSAL config.

Spec S5.2 claims the PyO3 binding gains the causal mode with ZERO new surface, because the
windowed-vs-causal dispatch lives entirely inside `fast::stream::StreamingSession::new` and
`speech-py/src/lib.rs` is untouched. This module is the cross-language proof of that claim:
the SAME binding, the SAME `push`/`finish` calls, pointed at a `Cell_Type slstm` /
`Direction forward` config.

The Rust gate (`src/rust/tests/phase9_stream_causal.rs`) owns the bit-equal-vs-offline
contract (it needs `BagOfProcessors`, which is not exposed to Python). What is observable
here -- and is what the phase-8 sibling asserts too -- is the pair of streaming properties
the binding must preserve:
  - CHUNK-INVARIANCE: two push granularities produce the identical emitted (begin/end/class)
    SET and the identical final segmentation;
  - PREFIX CONSISTENCY: the emitted set equals the final partition (no retraction, no extra).

`importorskip` guards the module (the plain `python-test` CI job never builds `speech_rs`).
"""

import wave
from pathlib import Path

import numpy as np
import pytest

speech_rs = pytest.importorskip("speech_rs")

from speech.weight_bridge import read_weight_vector, write_bin  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
PHASE4D = REPO_ROOT / "tests" / "reference_data" / "phase4d"
PHASE9 = REPO_ROOT / "tests" / "reference_data" / "phase9"

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


def _stage_causal(tmp_path: Path, gain: float, bias_offset: float, tag: str) -> Path:
    """The committed causal fixture config + the Rust gate's overlay tail, pointed at a local
    pack copy whose OUTPUT-LAYER BIAS is shifted by `bias_offset` (the crossing sweep: a pure
    post-recurrence level shift, leaving every cell weight untouched)."""
    base = (PHASE9 / "slstm_forward.config").read_text()
    # The output layer's bias is the last pack element before the `2*input_size` normalize
    # tail (layout `[stack | output MLP | mean | std]`; the fixture's MLP is a single 4 -> 1
    # layer). `input_size` is DERIVED from the config, so a regeneration relocates the index.
    input_size = next(int(line.split()[1].split(",")[0]) for line in base.splitlines() if line.startswith("BLSTM_LSTMNeuronNb "))
    flat = read_weight_vector(PHASE9 / "slstm_forward_seed.bin")
    flat[len(flat) - 2 * input_size - 1] += bias_offset
    pack = tmp_path / f"slstm_causal_{tag}.bin"
    write_bin(len(flat), 1, flat, pack)

    dump = tmp_path / f"vrcts_causal_{tag}"
    dump.mkdir(exist_ok=True)
    staged = (
        f"{base}\n"
        "# ==== Phase 9 Task 7 causal streaming overlays (last-wins) ====\n"
        "numOuterThreads 1\n"
        "Audio_offset 0.0\n"
        "Audio_max_duration 3600\n"
        f"BLSTM_weightsFile {pack}\n"
        f"Dump_Directory {dump}\n"
        "Inference_Path fast\n"
        f"Audio_fixed_gain {gain}\n"
    )
    cfg = tmp_path / f"causal_{tag}.config"
    cfg.write_text(staged)
    return cfg


def _run(
    cfg: Path, rate: int, samples: np.ndarray, chunk: int
) -> tuple[
    list[tuple[float, float, str, float]],
    list[tuple[float, float, str, float]],
    list[tuple[float, str]],
]:
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


def test_streaming_session_causal_chunk_invariance_and_prefix(tmp_path: Path) -> None:
    rate, chan0 = _read_stereo_chan0(PHASE4D / "prcts_excerpt.wav")
    gain = _measure_gain(chan0)
    samples = (chan0.astype(np.float32) / np.float32(32768.0)).astype(np.float32)

    # NON-VACUITY SWEEP (the Rust gate's `crossing_offset`, restated): a run whose posteriors
    # never cross the rising threshold collapses to the seeded [Other@0, End@dur] pair, and
    # every set comparison below would be trivially true. Sweep the output bias (0.0 FIRST, so
    # a fixture that already crosses is used as committed) until real segments appear.
    chunk = int(0.1 * rate)
    cfg = None
    seg_rows: list[tuple[float, str]] = []
    for k, offset in enumerate([0.0, -0.5, -1.0, 0.5, -1.5, 1.0]):
        candidate = _stage_causal(tmp_path, gain, offset, f"sweep{k}")
        _, _, rows = _run(candidate, rate, samples, chunk)
        if len(rows) > 2:
            cfg, seg_rows = candidate, rows
            break
    assert cfg is not None, "no output-bias offset produced a non-degenerate causal partition"
    assert len(seg_rows) > 2, f"causal partition must be non-degenerate, got {len(seg_rows)} rows"

    # (1) CHUNK-INVARIANCE through the binding: 100 ms vs a non-divisor 7 ms granularity.
    push_a, fin_a, rows_a = _run(cfg, rate, samples, chunk)
    push_b, fin_b, rows_b = _run(cfg, rate, samples, max(1, int(0.007 * rate)))
    assert rows_a == rows_b, "the final segmentation must not depend on push granularity"
    assert _triples(push_a + fin_a) == _triples(push_b + fin_b), "the emitted (begin/end/class) SET must not depend on push granularity"

    # (2) PREFIX CONSISTENCY: emitted set == final partition (no retraction, nothing extra).
    all_a = push_a + fin_a
    emitted = _triples(all_a)
    partition = _partition(rows_a)
    assert emitted == partition, f"emitted set != final partition (retraction/extra): missing={partition - emitted} extra={emitted - partition}"
    # ...and each partition segment was emitted EXACTLY ONCE. The set comparison above is
    # blind to duplicates (a re-emitted segment collapses into the same element), so without
    # this the "no re-emission" half of prefix consistency goes unasserted -- the phase-8
    # sibling's `len(seg_lines) == len(cli_set)` catch, restated through the binding.
    assert len(all_a) == len(partition), f"emissions must be unique (no re-emission): {len(all_a)} emitted vs {len(partition)} partition segments"
    assert len(push_a) > 0, "non-vacuity: the causal arm must emit MID-STREAM, not only at EOS"


def test_streaming_session_causal_rejects_stereo(tmp_path: Path) -> None:
    """The mono-first bail still fires on the causal arm (the shared validation is upstream of
    the cell dispatch)."""
    rate, chan0 = _read_stereo_chan0(PHASE4D / "prcts_excerpt.wav")
    cfg = _stage_causal(tmp_path, _measure_gain(chan0), 0.0, "stereo")
    with pytest.raises(RuntimeError, match="mono only"):
        speech_rs.StreamingSession(str(cfg), float(rate), 2)


def test_streaming_session_warns_on_an_over_long_pack_file(tmp_path: Path, capfd: pytest.CaptureFixture[str]) -> None:
    """Issue #62: the session's weight file is a FILE seam. It keeps the legacy head-first
    tolerance and, like the exact `load_weights_file`, says so on stderr (it used to load
    the head in silence): the sLSTM net (1603) given the LSTM fixture's 1651-element pack."""
    rate, chan0 = _read_stereo_chan0(PHASE4D / "prcts_excerpt.wav")
    cfg = _stage_causal(tmp_path, _measure_gain(chan0), 0.0, "overlong")
    lstm = PHASE9 / "lstm_forward_seed.bin"
    cfg.write_text(cfg.read_text() + f"BLSTM_weightsFile {lstm}\n")
    capfd.readouterr()
    speech_rs.StreamingSession(str(cfg), float(rate), 1)
    want = f"Warning: The number of gains given in {lstm} is more than what's needed (1651 > 1603); the extra 48 are ignored."
    assert want in capfd.readouterr().err
