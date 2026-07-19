"""Phase 8 Task 8: the corpus tier -- streaming SAD parity + the causality cost on REAL data.

The Rust CI gate (`src/rust/tests/phase8_gate.rs`) proved the streaming session is BIT-EQUAL
to the offline-frozen fast run on the committed 60 s tuple-A fixture (boundary max_dt EXACTLY
0.0, chunk-invariant, prefix-consistent). This tier lifts that proof onto REAL corpus data
with TRAINED weights (the phase-6 SAD subset checkpoint), through the Python `speech_rs`
seam:
  - S1.9 EQUIVALENCE: one real (sorted-first) mono train wav, streamed via the PyO3
    `StreamingSession` at 100 ms chunks, finished, and compared to the offline-frozen fast
    `Engine` run on the SAME file + SAME pack + SAME gain. Speech-interval count IDENTICAL,
    boundary max_dt within the VRCTS 4-decimal write quantum (the ONLY Python-side gap: the
    offline path's segmentation is observable only through the engine's `%f.4` VRCTS dump).
    Any COUNT mismatch or a boundary delta beyond the quantum is an R1 STOP-and-adjudicate,
    never a silent widen.
  - PREFIX consistency + CHUNK invariance surfaced through the binding (the S1.6 properties).
  - LATENCY (S1.8): the running max/mean per-emission lag recorded vs the config-derived
    structural bound (holdback computed here from the config, the full bound cited from the
    key-identical Rust gate).
  - S1.7 CAUSALITY COST (REPORTED, never gated): offline-frozen vs offline-self-norm on the
    same file (boundary delta) + on the phase-6 SAD held-out slice (pooled DCF under both
    modes, the delta reported with the mechanism named).

THE FROZEN-STATS SEAM. The streaming session requires `Audio_fixed_gain` (replaces the
whole-file `(2*RMS+max)/2` audio normalization) + `BLSTM_InputNormalizationType 1` (replaces
the type -1 per-sequence self-normalization with the pack-carried frozen tail). Both sides of
every equivalence comparison use the SAME pack + SAME gain, so the comparison is valid
regardless of whether the frozen boundary set is degenerate.

THE TAIL FINDING (investigated, stated -- `test_frozen_tail_is_identity`): the phase-6 SAD
checkpoint's normalize tail is EXACTLY identity (mean 0, std 1), because `init_weights` seeds
it identity and type -1 training never descends it (the frozen tail is not the training
target). So type-1 normalization is a NO-OP here and the frozen posteriors are driven by
UN-normalized input -- the honest consequence, recorded, not worked around. Under this net's
`IgnoreFirstDCT` + no-LTSV/TDC config the per-file audio gain is moreover DECISION-INVARIANT
(the DCT cancellation `phase8_frozen_norm.rs` finding 1 proves), so the causality cost isolates
the type-1-vs-self-norm INPUT normalization alone.

Corpus-gated (`requires_corpus`) + pyo3 (`importorskip`) + `slow`: runs locally, skips in CI.
Reuses the SHARED phase-7 checkpoint cache (`test_phase7_parity._ensure_arm`), so a warm cache
skips training. Each leg is its own test so a slow cold-cache run stays inside a single
per-test invocation (the pytest tool-timeout convention). License: the wav is selected at
RUNTIME as the lexicographically-first match under `train/audio/**` (no filename recorded).
ASCII only.
"""

from __future__ import annotations

import os
import shutil
import time
import wave
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import CORPUS_ROOT, requires_corpus

pytest.importorskip("speech_rs")

import speech_rs  # noqa: E402
from speech.config_bridge import parse_legacy_config  # noqa: E402
from speech.drivers import baseline as B  # noqa: E402
from speech.evaluate import load_vrcts_hyp  # noqa: E402
from speech.weight_bridge import read_weight_vector  # noqa: E402

from tests.pyo3.test_phase7_parity import _Arm, _ensure_arm  # noqa: E402

# The DCF collars the SAD arm pools over (== baseline._DCF_COLLARS).
_COLLARS: tuple[float, ...] = B._DCF_COLLARS

# The DERIVED structural latency bound (spec S1.8), pinned by the Rust gate
# (`phase8_gate.rs::latency_bounds`) on the tier2 staged config: feature_reach 0.14400 +
# nn_window 3.26000 + conv_delay 0.36000 + holdback 2.28957. The SAD arm's `lre_sad.toml` is
# seeded VERBATIM from the SAME `1_worker_1.config` as tier2, so every bound-relevant key
# (spectrum shift/window/order, BLSTM window/shift, convolution_window_size, deltas,
# lstm/output subsampling, min_speech/min_silence/speech_padding) is BYTE-IDENTICAL and this
# derived bound holds verbatim. `_holdback_from_config` recomputes the value-dependent holdback
# from THIS config and cross-checks it against 2.28957, proving the lineage that licenses the
# cited whole-bound constant.
_DERIVED_BOUND_S = 6.05357
_HOLDBACK_S = 2.28957

# The offline VRCTS dump writes boundary times at `%f.4` (4 decimals), so a Python-side
# comparison of the streamed full-precision boundary against the offline VRCTS boundary can
# differ by at most half a 4-decimal quantum. A delta beyond this is a REAL divergence (R1).
_VRCTS_QUANTUM_S = 5.0e-5


# --------------------------------------------------------------------------------------- #
# Corpus wav + gain (license: sorted-first at runtime, no filename recorded)
# --------------------------------------------------------------------------------------- #


def _sorted_first_train_wav() -> Path:
    """The lexicographically-FIRST `*.wav` under `train/audio/**` (a `sort` over a glob, no
    duration/content picking -- the license bright line, mirroring the phase-7 bench recipe)."""
    root = CORPUS_ROOT / "train" / "audio"
    wavs = sorted(root.glob("**/*.wav"))
    if not wavs:
        pytest.skip(f"no train/audio wavs under {root}")
    return wavs[0]


def _read_mono_wav(path: Path) -> tuple[int, np.ndarray]:
    """Read a PCM16 wav; return (rate, int16 samples). Skips (not fails) a stereo file -- the
    session is mono-first and the SAD corpus is mono, but this guards a surprise multi-channel
    file from a spurious failure."""
    with wave.open(str(path), "rb") as w:
        rate = w.getframerate()
        channels = w.getnchannels()
        if w.getsampwidth() != 2:
            pytest.skip("corpus wav is not PCM16")
        if channels != 1:
            pytest.skip(f"corpus wav is {channels}-channel; the streaming tier is mono-first")
        frames = w.readframes(w.getnframes())
    return rate, np.frombuffer(frames, dtype="<i2").copy()


def _measure_gain(chan_i16: np.ndarray) -> float:
    """`(2*rms + max_abs)/2` over `i16/32768` -- the exact `audio.rs::normalize_channels` adim
    (== `measure_fixed_gain`), computed over the mono channel."""
    x = chan_i16.astype(np.float64) / 32768.0
    rms = float(np.sqrt(np.mean(x * x)))
    max_abs = float(np.max(np.abs(x)))
    return (2.0 * rms + max_abs) / 2.0


def _holdback_from_config(cfg: dict[str, str]) -> float:
    """The smoothing holdback (spec S1.3) recomputed from THIS config: sum of the three
    decision comma-lists with each value clamped at 0 (the `SegmenterConfig::clamp0`), matching
    `StreamDecision::new`'s `min_speech + min_silence + padding`."""
    total = 0.0
    for key in ("BLSTM_min_speech", "BLSTM_min_silence", "BLSTM_speech_padding"):
        for tok in cfg[key].split(","):
            total += max(0.0, float(tok))
    return total


# --------------------------------------------------------------------------------------- #
# Config assembly (frozen streaming/offline + self-norm), on the phase-6 SAD base.config
# --------------------------------------------------------------------------------------- #


def _base_cfg(arm: _Arm) -> dict[str, str]:
    return dict(parse_legacy_config(arm.base_config.read_text()))


def _frozen_cfg(arm: _Arm, gain: float, *, fileslisting: str, dump_dir: Path | None) -> dict[str, str]:
    """The FROZEN inference config: the phase-6 SAD base config repointed at the trained
    checkpoint, with `Inference_Path fast` + `Audio_fixed_gain` + `BLSTM_InputNormalizationType 1`
    (the streaming contract) + backprop off. The weight key MUST point at the trained pack (the
    fast SAD driver loads weights only at construction; `set_weights` bails on it, T6b)."""
    cfg = _base_cfg(arm)
    cfg["fileslisting"] = fileslisting
    cfg["BLSTM_weightsFile"] = str((arm.ckpt / "best_sad.bin").resolve())
    cfg["Inference_Path"] = "fast"
    cfg["Audio_fixed_gain"] = repr(gain)
    cfg["BLSTM_InputNormalizationType"] = "1"
    cfg["Audio_offset"] = "0.0"
    cfg["Audio_max_duration"] = "3600"
    cfg["BLSTM_BackPropagationActivated"] = "false"
    cfg["Neural_Networks_BackPropagation_Epochs"] = "0"
    if dump_dir is not None:
        cfg["Dump_Directory"] = str(dump_dir.resolve())
    return cfg


def _self_norm_cfg(arm: _Arm, *, fileslisting: str, dump_dir: Path | None) -> dict[str, str]:
    """The SELF-NORM sibling: the SAME fast path + trained pack, but the base config's own
    `BLSTM_InputNormalizationType -1` (per-sequence self-normalization) and NO `Audio_fixed_gain`
    (per-file `(2*RMS+max)/2` audio normalization). The causality-cost baseline."""
    cfg = _base_cfg(arm)
    cfg["fileslisting"] = fileslisting
    cfg["BLSTM_weightsFile"] = str((arm.ckpt / "best_sad.bin").resolve())
    cfg["Inference_Path"] = "fast"
    cfg["Audio_offset"] = "0.0"
    cfg["Audio_max_duration"] = "3600"
    cfg["BLSTM_BackPropagationActivated"] = "false"
    cfg["Neural_Networks_BackPropagation_Epochs"] = "0"
    if dump_dir is not None:
        cfg["Dump_Directory"] = str(dump_dir.resolve())
    return cfg


def _config_text(cfg: dict[str, str]) -> str:
    return "\n".join(f"{k} {v}" for k, v in cfg.items()) + "\n"


def _single_file_listing(arm: _Arm, wav: Path, name: str) -> Path:
    """A one-row unscored fileslisting (empty refseg): image mode's unscored VRCTS branch
    ALWAYS writes the hyp xml when `Dump_Directory` is set, so no `.part.xml` is needed."""
    path = arm.out_dir / name
    path.write_text(f"{wav};;unk;unk;1.0;1.0;\n")
    return path


# --------------------------------------------------------------------------------------- #
# Offline-frozen segmentation via the image-mode VRCTS dump
# --------------------------------------------------------------------------------------- #


def _run_offline_image(arm: _Arm, cfg: dict[str, str], listing_name: str, dump_dir: Path, tag: str) -> list[tuple[float, float, str]]:
    """Run the offline fast SAD driver over a single-file listing in image mode (`-i`), dumping
    one VRCTS hyp xml, and read it back via `load_vrcts_hyp`. Returns the (begin, end, kind)
    partition (`kind` in {'speech', 'non-speech'})."""
    if dump_dir.exists():
        shutil.rmtree(dump_dir)
    dump_dir.mkdir(parents=True)
    cfg_path = arm.out_dir / f"_t8_off_{tag}.config"
    cfg_path.write_text(_config_text(cfg))
    prev = Path.cwd()
    os.chdir(arm.out_dir)
    try:
        engine = speech_rs.Engine([cfg_path.name], "-i")
        engine.run()
    finally:
        os.chdir(prev)
    hyps = sorted(dump_dir.glob("*.xml"))
    assert hyps, f"{tag}: image-mode run dumped no VRCTS xml"
    return load_vrcts_hyp(hyps[0])


def _speech_intervals_from_hyp(hyp: list[tuple[float, float, str]]) -> list[tuple[float, float]]:
    return [(s, e) for (s, e, k) in hyp if k == "speech"]


def _speech_intervals_from_seg_rows(seg_rows: list[tuple[float, str]]) -> list[tuple[float, float]]:
    """The streamed partition (`(begin, class)` rows, last is the End sentinel) -> Speech
    intervals `[rows[i], rows[i+1])`."""
    return [(seg_rows[i][0], seg_rows[i + 1][0]) for i in range(len(seg_rows) - 1) if seg_rows[i][1] == "Speech"]


def _boundary_max_dt(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> float:
    """Max abs boundary delta over paired speech intervals (assumes equal length -- the caller
    asserts count equality first)."""
    dt = 0.0
    for (ab, ae), (bb, be) in zip(a, b, strict=True):
        dt = max(dt, abs(ab - bb), abs(ae - be))
    return dt


# --------------------------------------------------------------------------------------- #
# Streaming drive
# --------------------------------------------------------------------------------------- #


def _stream(cfg_path: Path, rate: int, samples: np.ndarray, chunk: int) -> tuple[list, list, list, float, float, float]:
    """Push `samples` (f32) through a fresh session in `chunk`-sample blocks; return
    (push emissions, finish emissions, seg rows, max_lag, mean_lag, wall_s)."""
    t0 = time.time()
    sess = speech_rs.StreamingSession(str(cfg_path), float(rate), 1)
    push_em: list = []
    i = 0
    n = len(samples)
    while i < n:
        j = min(i + chunk, n)
        push_em.extend(sess.push(samples[i:j]))
        i = j
    fin_em, seg_rows = sess.finish()
    return push_em, fin_em, seg_rows, sess.max_lag_s(), sess.mean_lag_s(), time.time() - t0


# ======================================================================================= #
# (0) The tail finding (investigated + stated).
# ======================================================================================= #


@pytest.mark.slow
@requires_corpus
def test_frozen_tail_is_identity() -> None:
    """The phase-6 SAD checkpoint's normalize tail (last `2*input_size` pack elements) is
    EXACTLY identity (mean 0, std 1): `init_weights` seeds it identity and type -1 training
    never descends it. So `BLSTM_InputNormalizationType 1` is a NO-OP on this pack and the
    frozen posteriors are un-normalized-input-driven -- the honest consequence, stated here."""
    arm = _ensure_arm("sad")
    cfg = _base_cfg(arm)
    input_size = int(cfg["BLSTM_NNetInputSize"])
    pack = read_weight_vector(arm.ckpt / "best_sad.bin")
    tail = pack[-2 * input_size :]
    mean, std = tail[:input_size], tail[input_size:]
    max_abs_mean = float(np.max(np.abs(mean)))
    max_abs_std_dev = float(np.max(np.abs(std - 1.0)))
    print(f"\n[TAIL sad] input_size={input_size} pack_len={len(pack)} max|mean|={max_abs_mean:.3e} max|std-1|={max_abs_std_dev:.3e}")
    assert max_abs_mean == 0.0, f"normalize mean not identity (max|mean|={max_abs_mean})"
    assert max_abs_std_dev == 0.0, f"normalize std not identity (max|std-1|={max_abs_std_dev})"


# ======================================================================================= #
# (1) S1.9 EQUIVALENCE -- streamed vs offline-frozen on a real corpus file + trained weights.
# ======================================================================================= #


@pytest.mark.slow
@requires_corpus
def test_streaming_equivalence_on_corpus() -> None:
    """Stream one real mono corpus wav through the PyO3 session at 100 ms vs the offline-frozen
    fast run on the SAME file/pack/gain. Speech-interval count IDENTICAL (R1 STOP on mismatch),
    boundary max_dt within the VRCTS 4-decimal quantum, prefix-consistent, chunk-invariant, and
    the latency stats recorded against the derived bound."""
    arm = _ensure_arm("sad")
    wav = _sorted_first_train_wav()
    rate, chan = _read_mono_wav(wav)
    gain = _measure_gain(chan)
    samples = (chan.astype(np.float32) / np.float32(32768.0)).copy()
    dur = (len(chan) - 1) / rate

    listing = _single_file_listing(arm, wav, "_t8_single.flst")
    cfg = _frozen_cfg(arm, gain, fileslisting=listing.name, dump_dir=None)
    cfg_path = arm.out_dir / "_t8_frozen_stream.config"
    cfg_path.write_text(_config_text(cfg))

    # Holdback cross-check: THIS config's clamped decision sums must match the gate's 2.28957,
    # proving the 1_worker_1.config lineage that licenses the cited whole derived bound.
    holdback = _holdback_from_config(cfg)
    assert abs(holdback - _HOLDBACK_S) < 5e-5, f"holdback {holdback:.5f} != gate-pinned {_HOLDBACK_S} (config lineage drift?)"

    # --- streamed (100 ms) ---
    chunk_a = int(round(0.1 * rate))
    push_a, fin_a, seg_a, max_lag, mean_lag, stream_s = _stream(cfg_path, rate, samples, chunk_a)
    streamed_speech = _speech_intervals_from_seg_rows(seg_a)

    # --- offline-frozen (image mode, VRCTS dump) ---
    t0 = time.time()
    off_hyp = _run_offline_image(
        arm, _frozen_cfg(arm, gain, fileslisting=listing.name, dump_dir=arm.out_dir / "_t8_off_frozen"), listing.name, arm.out_dir / "_t8_off_frozen", "frozen"
    )
    offline_s = time.time() - t0
    offline_speech = _speech_intervals_from_hyp(off_hyp)

    # --- chunk invariance (a second granularity) ---
    push_b, fin_b, seg_b, _, _, _ = _stream(cfg_path, rate, samples, 101)

    print(
        f"\n[EQUIV sad] dur={dur:.3f}s gain={gain:.6e} rate={rate} "
        f"streamed_speech={len(streamed_speech)} offline_speech={len(offline_speech)} "
        f"seg_rows={len(seg_a)} push_em={len(push_a)} fin_em={len(fin_a)} "
        f"max_lag={max_lag:.4f} mean_lag={mean_lag:.4f} bound={_DERIVED_BOUND_S} holdback={holdback:.5f} "
        f"stream_s={stream_s:.2f} offline_s={offline_s:.2f}"
    )

    # (R1) speech-interval COUNT identical -- STOP-and-adjudicate on any mismatch.
    assert len(streamed_speech) == len(offline_speech), (
        f"streamed vs offline-frozen speech-segment COUNT differs ({len(streamed_speech)} vs {len(offline_speech)}) -- R1 STOP"
    )

    # (R1) boundary max_dt within the VRCTS 4-decimal write quantum (the only Python-side gap;
    # a delta beyond it is a real divergence). Degenerate (0 speech) -> trivially 0.0.
    max_dt = _boundary_max_dt(streamed_speech, offline_speech) if streamed_speech else 0.0
    print(f"[EQUIV sad] boundary_max_dt={max_dt:.3e}s (VRCTS quantum {_VRCTS_QUANTUM_S:.1e})")
    assert max_dt <= _VRCTS_QUANTUM_S, f"boundary max_dt {max_dt:.3e}s exceeds the VRCTS quantum {_VRCTS_QUANTUM_S:.1e}s -- R1 STOP"

    # At the 4-decimal VRCTS grain the two speech-interval SETS are IDENTICAL (max_dt 0.0 there,
    # the design's exact-agreement prediction surfaced through the write precision).
    streamed_4dp = {(f"{b:.4f}", f"{e:.4f}") for b, e in streamed_speech}
    offline_4dp = {(f"{b:.4f}", f"{e:.4f}") for b, e in offline_speech}
    assert streamed_4dp == offline_4dp, f"streamed vs offline speech SET differs at 4dp: {streamed_4dp} vs {offline_4dp}"

    # PREFIX consistency: the emitted set equals the final partition (no missing, no extra).
    emitted_set = {(round(e[0], 4), round(e[1], 4), e[2]) for e in push_a + fin_a}
    part_set = {(round(seg_a[i][0], 4), round(seg_a[i + 1][0], 4), seg_a[i][1]) for i in range(len(seg_a) - 1)}
    assert emitted_set == part_set, f"emitted set != final partition (retraction/extra -- R2): {len(emitted_set)} vs {len(part_set)}"
    assert len(push_a + fin_a) == len(part_set), "emitted count must equal the partition size (no re-emission)"

    # CHUNK invariance: a different push granularity yields the identical segmentation + set.
    assert seg_a == seg_b, "the final segmentation must be chunk-invariant"
    assert emitted_set == {(round(e[0], 4), round(e[1], 4), e[2]) for e in push_b + fin_b}, "the emitted set must be chunk-invariant"

    # LATENCY (S1.8, recorded): the measured max lag is within the config-derived structural
    # bound (trivially so under an all-speech/degenerate collapse; the mid-stream budget is
    # exercised on the crafted Rust gate `phase8_gate.rs::latency_bounds`).
    assert max_lag <= _DERIVED_BOUND_S, f"max_lag {max_lag:.4f}s exceeds the derived bound {_DERIVED_BOUND_S}s"
    assert max_lag >= mean_lag >= 0.0, "lag stats must be sane (max >= mean >= 0)"


# ======================================================================================= #
# (2) S1.7 CAUSALITY COST -- boundary delta on the single file (REPORTED, never gated).
# ======================================================================================= #


@pytest.mark.slow
@requires_corpus
def test_causality_cost_boundary_on_corpus() -> None:
    """Offline-frozen (type 1 + fixed gain) vs offline-self-norm (type -1, per-file gain) on the
    SAME corpus file + pack: the boundary delta is the quantified cost of causality (a MODE
    difference, not a defect). Reported, never a pass/fail pin on the delta."""
    arm = _ensure_arm("sad")
    wav = _sorted_first_train_wav()
    rate, chan = _read_mono_wav(wav)
    gain = _measure_gain(chan)
    listing = _single_file_listing(arm, wav, "_t8_single.flst")

    frozen_hyp = _run_offline_image(
        arm, _frozen_cfg(arm, gain, fileslisting=listing.name, dump_dir=arm.out_dir / "_t8_cc_frozen"), listing.name, arm.out_dir / "_t8_cc_frozen", "cc_frozen"
    )
    self_hyp = _run_offline_image(
        arm, _self_norm_cfg(arm, fileslisting=listing.name, dump_dir=arm.out_dir / "_t8_cc_self"), listing.name, arm.out_dir / "_t8_cc_self", "cc_self"
    )

    frozen_speech = _speech_intervals_from_hyp(frozen_hyp)
    self_speech = _speech_intervals_from_hyp(self_hyp)
    # Total speech coverage (seconds) is a mode-robust scalar: it summarizes the decision even
    # when the two modes' segment COUNTS differ (the interesting causality case).
    frozen_cov = sum(e - s for s, e in frozen_speech)
    self_cov = sum(e - s for s, e in self_speech)
    print(
        f"\n[CAUSALITY-boundary sad] gain={gain:.6e} "
        f"frozen: speech_segs={len(frozen_speech)} speech_cov={frozen_cov:.3f}s | "
        f"self_norm: speech_segs={len(self_speech)} speech_cov={self_cov:.3f}s | "
        f"delta_cov={abs(frozen_cov - self_cov):.3f}s"
    )
    # Sanity only (reported, not gated): both modes produced a well-formed partition.
    assert frozen_hyp and self_hyp, "both modes must produce a VRCTS partition"


# ======================================================================================= #
# (3) S1.7 CAUSALITY COST -- held-out pooled DCF under both modes (REPORTED, never gated).
# ======================================================================================= #


@pytest.mark.slow
@requires_corpus
def test_causality_cost_dcf_on_corpus() -> None:
    """Score the phase-6 SAD held-out slice (the recipe's test records) under BOTH modes via the
    established `_score_sad_pack_on_test` flow, and REPORT the pooled DCF delta. Self-norm is the
    trained-net baseline (== the phase-7 fast SAD score); frozen is the same net under the frozen
    stats. The single `Audio_fixed_gain` is DECISION-INVARIANT for this `IgnoreFirstDCT`/no-LTSV/TDC
    net (the DCT cancellation), so one global gain scores all test files consistently. REPORTED,
    not gated (sanity: both DCFs are finite and same order of magnitude)."""
    arm = _ensure_arm("sad")
    # A representative fixed gain for the frozen mode -- decision-invariant here, so the
    # sorted-first train file's gain scores every held-out file identically to a per-file gain.
    wav = _sorted_first_train_wav()
    _, chan = _read_mono_wav(wav)
    gain = _measure_gain(chan)

    self_cfg = _self_norm_cfg(arm, fileslisting=arm.test_listing.name, dump_dir=None)
    froz_cfg = _frozen_cfg(arm, gain, fileslisting=arm.test_listing.name, dump_dir=None)

    t0 = time.time()
    self_rep, _ = B._score_sad_pack_on_test(
        self_cfg, arm.out_dir, arm.ckpt / "best_sad.bin", arm.out_dir / "_t8_dcf_self", arm.test_records, arm.test_listing.name
    )
    self_s = time.time() - t0
    t0 = time.time()
    froz_rep, _ = B._score_sad_pack_on_test(
        froz_cfg, arm.out_dir, arm.ckpt / "best_sad.bin", arm.out_dir / "_t8_dcf_frozen", arm.test_records, arm.test_listing.name
    )
    froz_s = time.time() - t0
    assert self_rep is not None and froz_rep is not None, "both modes must produce held-out VRCTS hyps"

    self_dcf = {c: self_rep.by_collar(c).dcf for c in _COLLARS}
    froz_dcf = {c: froz_rep.by_collar(c).dcf for c in _COLLARS}
    deltas = {c: froz_dcf[c] - self_dcf[c] for c in _COLLARS}
    print(
        f"\n[CAUSALITY-DCF sad] files={len(arm.test_records)} "
        f"self_norm DCF={ {c: f'{v:.4f}' for c, v in self_dcf.items()} } "
        f"frozen DCF={ {c: f'{v:.4f}' for c, v in froz_dcf.items()} } "
        f"delta(frozen-self)={ {c: f'{d:+.4f}' for c, d in deltas.items()} } "
        f"self_s={self_s:.2f} frozen_s={froz_s:.2f}"
    )
    # Sanity only (reported, not gated): finite, in-range DCFs on both modes.
    for c in _COLLARS:
        assert 0.0 <= self_dcf[c] <= 1.0 and 0.0 <= froz_dcf[c] <= 1.0, f"DCF@{c} out of [0,1]"
