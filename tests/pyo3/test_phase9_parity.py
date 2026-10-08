"""Phase 9 Task 9 (spec S8.4 + S4.3/S5.6): the CORPUS tier -- causal streaming parity,
the causality cost, and fast-vs-exact metric parity on REAL data with TRAINED causal nets.

Two Rust CI gates already hold on committed synthetic fixtures: `phase9_fast_parity.rs`
(exact-causal vs fast-causal offline, boundary/argmax identity) and
`phase9_stream_causal.rs` (streamed `finish()` BIT-EQUAL to the offline fast causal run,
boundary max_dt EXACTLY 0.0, chunk-invariant, prefix-consistent). This tier lifts BOTH onto
the LRE03/07 corpus with the T8 from-scratch CAUSAL checkpoints, through the Python
`speech_rs` seam -- the phase-7 (`test_phase7_parity.py`) + phase-8
(`test_phase8_parity.py`) corpus tiers' structure, on phase-9 artefacts:

  - (S5.6 on real data) STREAMING EQUIVALENCE: one real (sorted-first) mono train wav,
    streamed via the PyO3 `StreamingSession` at 100 ms chunks, finished, and compared to the
    offline fast CAUSAL `Engine` run on the SAME file + SAME pack + SAME overlaid config.
    Speech-interval count IDENTICAL, boundary max_dt within the VRCTS 4-decimal write
    quantum (the only Python-side gap -- the offline segmentation is observable solely
    through the engine's `%f.4` dump), prefix-consistent, chunk-invariant. Any count
    mismatch or a delta beyond the quantum is an R1 STOP-and-adjudicate, never a silent
    widen.
  - CAUSALITY COST (REPORTED, never gated): the SAME trained pack scored on the T8 held-out
    slice under (a) its NATIVE type -1 per-sequence self-normalization + per-file
    `(2*RMS+max)/2` gain and (b) the frozen/streamable regime (`Audio_fixed_gain` + type 0),
    with the mechanism named. This is the phase-8 honesty pattern; the interesting number is
    (a) vs (b), measured not assumed.
  - (S4.3 on real data) FAST-VS-EXACT METRIC PARITY: each causal checkpoint scored on its own
    held-out slice under BOTH `Inference_Path` values in its NATIVE config -- per-file VRCTS
    boundaries identical, pooled DCF delta EXACTLY 0.0 at every collar (the phase-7
    prediction: identical decisions => identical metrics).

THE STREAMING OVERLAY, and why it is not a semantic change. The arm trains under
`BLSTM_InputNormalizationType -1` (per-sequence self-normalization), which the streaming
session ALWAYS refuses -- it needs the whole sequence before the first frame. The T7/T8
resolution: the trained pack's normalize tail is IDENTITY BY CONSTRUCTION (`init_weights`
seeds mean 0 / std 1 and type -1 training never descends the tail -- the phase-8 tail
finding, re-verified here per cell in `test_frozen_tail_is_identity`), so the streamable
type-0 arm (`_ => {}`, no input normalization) and the type-1 arm (the pack-carried frozen
affine, `(x - 0)/1`) are ARITHMETICALLY THE SAME MAP on this pack. The overlay therefore
changes the input normalization REGIME (-1 -> frozen), which is exactly the causality cut
the causality-cost leg measures, and nothing else. `test_frozen_overlay_is_type1_equivalent`
pins the type-0/type-1 indistinguishability directly (same pack, both overlays, offline:
identical VRCTS partition), so the choice of the type-0 arm is evidenced, not asserted.
`BLSTM_window 0` needs NO overlay: `cell_overlay` already forces the plain whole-sequence
regime on every `--direction forward` run (a window boundary would reset the causal state).

EVIDENTIARY SCOPE, stated (the phase-8 wording, unchanged): because the tail IS identity,
this corpus leg does NOT independently exercise the type-1 arithmetic through the causal
streaming front-end -- that coverage lives in the Rust gate's calibrated-tail leg
(`phase9_stream_causal.rs::stream_finish_equals_offline_causal_frozen_type1`, a NON-identity
tail). This tier pins the streaming-vs-offline equivalence and the causality-cost regime on
real data with real trained weights.

CHECKPOINTS: trained here via T8's EXACT gate recipe (`test_phase9_gates.py::_GATE`
verbatim) through `run_baseline(... cell_type=<cell>, direction="forward")`, cached under
the gitignored `data/phase9_parity_cache/` (holds corpus-path listings -- NEVER committed),
the phase-7 cache recipe. `run_baseline` at a fixed seed is deterministic (the T8
determinism legs), so a warm cache is bit-identical to a fresh run -- a pure speedup. Each
cell's cold train is < 70 s.

Corpus-gated (`requires_corpus`) + pyo3 (`importorskip`) + `slow`: runs locally, skips in
CI. License: the wav is selected at RUNTIME as the lexicographically-first match under
`train/audio/**` (no filename recorded anywhere). ASCII only.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import CORPUS_ROOT, requires_corpus

pytest.importorskip("speech_rs")

import speech_rs  # noqa: E402
from speech.batching import read_listing  # noqa: E402 -- after importorskip, matching the pyo3-suite convention
from speech.config_bridge import parse_legacy_config  # noqa: E402
from speech.drivers import baseline as B  # noqa: E402
from speech.drivers.spec import BaselineSpec  # noqa: E402
from speech.evaluate import load_vrcts_hyp  # noqa: E402
from speech.weight_bridge import read_weight_vector  # noqa: E402

# The DCF collars the SAD arm pools over (== baseline._DCF_COLLARS).
_COLLARS: tuple[float, ...] = B._DCF_COLLARS

# Session-persistent, gitignored checkpoint cache (holds corpus-path listings -> NEVER
# committed; `.gitignore` carries `data/phase9_parity_cache/`). parents[2] = repo root.
_CACHE = Path(__file__).resolve().parents[2] / "data" / "phase9_parity_cache"

# The T8 subset-gate recipe, VERBATIM (`test_phase9_gates.py::_GATE`) -- so this tier's
# checkpoints ARE the gated ones and the exact-path DCF pins below are T8's own numbers.
_GATE: dict[str, object] = dict(subset=10, valid_size=8, test_size=24, epochs=3, steps_per_epoch=10, patience=99, seed=0, audio_max_duration=20.0)

# The test listing `run_baseline` writes under out_dir for the SAD arm.
_TEST_LISTING = "sad_test_subset.flst"

_CELLS = ("slstm", "mamba")

# EXACT-path provenance pins, from T8's own gate table (`test_phase9_gates.py`'s HARD-leg
# docstring / RESULTS.md's phase-9 gate table): the trained held-out pooled DCF@0.5 for each
# causal cell at THIS recipe. They turn silent checkpoint drift (a stale cache, a recipe or
# seed edit, a training-code change) into a loud failure here, instead of a fast-vs-exact
# comparison that keeps passing against the WRONG checkpoint. Re-derive DELIBERATELY (never
# widen on a red run without checking WHY -- `rm -rf data/phase9_parity_cache/<cell>-forward`
# first, per `_ensure_causal_arm`'s cache-staleness note).
_EXPECTED_EXACT_DCF_AT_0_5: dict[str, float] = {
    "slstm": 0.250000,
    "mamba": 0.249625,
}

# The offline VRCTS dump writes boundary times at `%f.4`, so a Python-side comparison of a
# streamed full-precision boundary against the offline VRCTS boundary can differ by at most
# half a 4-decimal quantum. Beyond this is a REAL divergence (R1). Phase-8 constant, verbatim.
_VRCTS_QUANTUM_S = 5.0e-5

# The three CAUSAL-bound summands this config shares BIT-FOR-BIT with the phase-8 windowed
# arm (the same `1_worker_1.config` lineage, and the causal time base `window_shift_sec*ssr`
# coincides with the windowed `spectrum_shift*ssr` here because `window_shift` floors to 1
# when `window_size == 0`). Cited from `phase8_gate.rs::latency_bounds` /
# `test_phase8_parity.py`, cross-checked against a from-config recomputation below -- the
# match is the lineage evidence.
_PHASE8_FEATURE_REACH_S = 0.14400
_PHASE8_CONV_DELAY_S = 0.36000
_PHASE8_HOLDBACK_S = 2.28957


# --------------------------------------------------------------------------------------- #
# Checkpoint cache (the phase-7 recipe, on T8's causal arms)
# --------------------------------------------------------------------------------------- #


@dataclass
class _CausalArm:
    """A trained causal arm's on-disk artefacts (the warm-cache handle)."""

    cell: str
    out_dir: Path
    ckpt: Path
    base_config: Path
    test_listing: Path
    test_records: list[dict[str, str]]
    train_s: float
    trained_now: bool
    timings: dict[str, float] = field(default_factory=dict)


def _ensure_causal_arm(cell: str) -> _CausalArm:
    """Train `cell`'s FORWARD (causal) SAD checkpoint on the EXACT path via T8's gate recipe
    if the cache is cold, else reuse it. The `.parity_ready.json` sentinel marks a COMPLETE
    run (written only after `run_baseline` returns), so an interrupted training leaves no
    sentinel and the next call retrains from a clean dir.

    CACHE-STALENESS CAVEAT (the phase-7 note, verbatim in force here): the sentinel guards
    INTERRUPTED runs only -- it says nothing about whether `_GATE`, the training code
    (`run_baseline`/`train_modern`/the cell kernels/the arm's TOML), or
    `_EXPECTED_EXACT_DCF_AT_0_5` have drifted apart since the cache was written. A warm cache
    is reused as-is. Touching any of those requires
    `rm -rf data/phase9_parity_cache/<cell>-forward` to force a clean retrain before trusting
    this tier's pins again."""
    out_dir = (_CACHE / f"{cell}-forward").resolve()
    ckpt = out_dir / "checkpoint"
    sentinel = out_dir / ".parity_ready.json"

    trained_now = False
    if not sentinel.is_file():
        if out_dir.exists():
            shutil.rmtree(out_dir)  # clean any partial cache (this cell's subdir only)
        out_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        B.run_baseline(BaselineSpec(arm="sad", corpus_root=CORPUS_ROOT, out_dir=out_dir, cell=cell, direction="forward", **_GATE))  # type: ignore[arg-type]
        train_s = time.time() - t0
        sentinel.write_text(json.dumps({"train_s": train_s, "cell": cell, "direction": "forward", "recipe": {k: str(v) for k, v in _GATE.items()}}))
        trained_now = True
    else:
        train_s = float(json.loads(sentinel.read_text()).get("train_s", 0.0))

    test_listing = out_dir / _TEST_LISTING
    records = read_listing(test_listing)
    assert (ckpt / "best_sad.bin").is_file(), f"{cell}: checkpoint missing (training failed?)"
    return _CausalArm(cell, out_dir, ckpt, out_dir / "base.config", test_listing, records, train_s, trained_now)


# --------------------------------------------------------------------------------------- #
# Corpus wav + gain (license: sorted-first at runtime, no filename recorded)
# --------------------------------------------------------------------------------------- #


def _sorted_first_train_wav() -> Path:
    """The lexicographically-FIRST `*.wav` under `train/audio/**` (a `sort` over a glob, no
    duration- or content-based picking -- the license bright line, the phase-7/8 recipe)."""
    root = CORPUS_ROOT / "train" / "audio"
    wavs = sorted(root.glob("**/*.wav"))
    if not wavs:
        pytest.skip(f"no train/audio wavs under {root}")
    return wavs[0]


def _read_mono_wav(path: Path) -> tuple[int, np.ndarray]:
    """Read a PCM16 wav; return (rate, int16 samples). SKIPS (not fails) a non-PCM16 or
    multi-channel file -- the session is mono-first and the SAD corpus is mono."""
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


# --------------------------------------------------------------------------------------- #
# The DERIVED causal latency bound (spec S5.5), recomputed from THIS config
# --------------------------------------------------------------------------------------- #


def _derive_causal_bound(cfg: dict[str, str], rate: float) -> dict[str, float]:
    """The spec-S5.5 causal structural bound, every summand recomputed from THIS config:

        bound = feature_reach + nn_window(=0) + sub_sample + conv_delay + holdback

    UNLIKE the phase-8 Python tier (which cited the whole bound from a key-identical Rust
    gate and re-derived only the holdback), the phase-9 Rust gate runs a DIFFERENT config
    (the synthetic `{slstm,mamba}_forward.config` fixture, bound 1.73400 s), so this arm's
    bound has to be derived here. It is derived, not copied: each summand mirrors
    `fast/stream.rs::StreamingSession::new`'s own component arithmetic, and the three
    summands this config shares bit-for-bit with the PHASE-8 windowed arm (feature_reach /
    conv_delay / holdback -- same `1_worker_1.config` lineage, and the causal time base
    `window_shift_sec*ssr` coincides with the windowed `spectrum_shift*ssr` because
    `window_shift` floors to 1 at `window_size == 0`) are cross-checked against the phase-8
    constants by the caller. The ONE genuinely new term is `sub_sample` (the causal arm's
    `(ssr-1)*spectrum_shift` decimation buffering); `nn_window` DIES -- the phase-9 win."""
    ssif = round(float(cfg["BLSTM_spectrum_shift"]) * rate)
    window_size = 1 << int(cfg["BLSTM_spectrum_order"])
    deltas = int(cfg["BLSTM_ComputeDeltasNb"])
    dd = int(cfg["BLSTM_ComputeDeltaDeltasNb"])
    reach = (deltas + dd) if deltas > 0 else 0
    spectrum_shift_sec = ssif / rate
    ssr = math.prod(int(v) for v in cfg["BLSTM_LSTMSubSampling"].split(",")) * math.prod(int(v) for v in cfg["BLSTM_OutputSubSampling"].split(","))
    # `window_size == 0` (the causal plain regime, forced by `cell_overlay`): `get_blstm_param`
    # floors `window_shift` to 1, so `window_shift_sec = 1*ssif/rate` and the time base is
    # `window_shift_sec*ssr` (`tasks/sad.rs:1451-1461`, mirrored at `fast/driver.rs:530-534`).
    assert float(cfg["BLSTM_window"]) == 0.0, "the causal arm must carry BLSTM_window 0 (cell_overlay)"
    time_step = spectrum_shift_sec * ssr
    conv_half = int(cfg["BLSTM_convolution_window_size"])  # 2*n+1 taps -> half = n
    holdback = 0.0
    for key in ("BLSTM_min_speech", "BLSTM_min_silence", "BLSTM_speech_padding"):
        for tok in cfg[key].split(","):
            holdback += max(0.0, float(tok))
    comp = {
        "feature_reach": (reach * ssif + window_size // 2) / rate,
        "nn_window": 0.0,
        "sub_sample": (ssr - 1) * spectrum_shift_sec,
        "conv_delay": conv_half * time_step,
        "holdback": holdback,
    }
    comp["bound"] = sum(comp.values())
    return comp


# --------------------------------------------------------------------------------------- #
# Config assembly (native / frozen-streamable / self-norm), on the trained base.config
# --------------------------------------------------------------------------------------- #


def _base_cfg(arm: _CausalArm) -> dict[str, str]:
    return dict(parse_legacy_config(arm.base_config.read_text()))


def _inference_cfg(arm: _CausalArm, overlay: dict[str, str], *, fileslisting: str | None = None, dump_dir: Path | None = None) -> dict[str, str]:
    """The trained base config made inference-shaped and repointed at the trained pack, plus
    `overlay`. THE WEIGHT KEY IS REPOINTED (not `set_weights`-injected): the fast drivers load
    weights ONLY at construction and `BagOfProcessors::set_weights` BAILS LOUDLY on a fast
    processor (T6b), so the config repoint is the fast path's only injection mechanism -- and
    it is identical for both paths, which is what makes the fast-vs-exact comparison
    apples-to-apples."""
    cfg = _base_cfg(arm)
    cfg["BLSTM_weightsFile"] = str((arm.ckpt / "best_sad.bin").resolve())
    cfg["BLSTM_BackPropagationActivated"] = "false"
    cfg["Neural_Networks_BackPropagation_Epochs"] = "0"
    if fileslisting is not None:
        cfg["fileslisting"] = fileslisting
    if dump_dir is not None:
        cfg["Dump_Directory"] = str(dump_dir.resolve())
    cfg.update(overlay)
    return cfg


def _frozen_overlay(gain: float, norm_type: str = "0") -> dict[str, str]:
    """The STREAMABLE (causality-cut) regime, and NOTHING ELSE: the frozen per-sample
    `Audio_fixed_gain` replaces the whole-file `(2*RMS+max)/2` audio normalization, and the
    input normalization moves off type -1 (per-sequence self-normalization, un-streamable)
    onto the causal arm's type 0 / type 1. `Inference_Path fast` is the streaming session's
    own kernel family.

    DELIBERATELY NO `Audio_offset`/`Audio_max_duration` HERE. Those are a WINDOWING choice,
    not part of the causality cut, and the causality-cost leg compares this overlay against
    the arm's NATIVE config -- so if this overlay moved the duration cap, the two sides would
    score DIFFERENT spans of every held-out file and the reported delta would be a
    duration artefact rather than a normalization one. The whole-file legs (which need the
    full 75 s wav on BOTH sides) apply `_WHOLE_FILE` explicitly instead."""
    return {
        "Inference_Path": "fast",
        "Audio_fixed_gain": repr(gain),
        "BLSTM_InputNormalizationType": norm_type,
    }


# The whole-file windowing overlay for the single-file streaming legs -- applied to BOTH
# sides of every comparison there (the streamed session and the offline reference).
_WHOLE_FILE: dict[str, str] = {"Audio_offset": "0.0", "Audio_max_duration": "3600"}


def _single_file_listing(arm: _CausalArm, wav: Path, name: str) -> Path:
    """A one-row UNSCORED fileslisting (empty refseg): image mode's unscored VRCTS branch
    ALWAYS writes the hyp xml when `Dump_Directory` is set, so no `.part.xml` is needed."""
    path = arm.out_dir / name
    path.write_text(f"{wav};;unk;unk;1.0;1.0;\n")
    return path


def _run_offline_image(arm: _CausalArm, cfg: dict[str, str], dump_dir: Path, tag: str) -> list[tuple[float, float, str]]:
    """Run the offline SAD driver over a single-file listing in image mode (`-i`), dumping one
    VRCTS hyp xml, and read it back. Returns the (begin, end, kind) partition."""
    if dump_dir.exists():
        shutil.rmtree(dump_dir)
    dump_dir.mkdir(parents=True)
    cfg = dict(cfg)
    cfg["Dump_Directory"] = str(dump_dir.resolve())
    cfg_path = arm.out_dir / f"_t9_off_{tag}.config"
    cfg_path.write_text(B._config_text(cfg))
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
    """Max abs boundary delta over paired speech intervals (the caller asserts count equality
    first)."""
    dt = 0.0
    for (ab, ae), (bb, be) in zip(a, b, strict=True):
        dt = max(dt, abs(ab - bb), abs(ae - be))
    return dt


def _stream(cfg_path: Path, rate: int, samples: np.ndarray, chunk: int) -> tuple[list, list, list, float, float, float]:
    """Push `samples` (RAW `i16/32768` f32 -- the front-end applies the frozen gain/preemph
    itself) through a fresh session in `chunk`-sample blocks; return (push emissions, finish
    emissions, seg rows, max_lag, mean_lag, wall_s)."""
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
# (0) The tail finding, per cell -- what makes the type-0 streaming overlay non-semantic.
# ======================================================================================= #


@pytest.mark.slow
@requires_corpus
@pytest.mark.parametrize("cell", _CELLS)
def test_frozen_tail_is_identity(cell: str) -> None:
    """The T8 causal checkpoint's normalize tail (last `2*input_size` pack elements) is
    EXACTLY identity (mean 0, std 1): `init_weights` seeds it identity and type -1 training
    never descends it (the frozen tail is not the descent target). So the streamable type-0
    arm and the type-1 frozen-affine arm are the SAME map on this pack -- which is what makes
    the streaming overlay a REGIME change (-1 -> frozen) and not an arithmetic one."""
    arm = _ensure_causal_arm(cell)
    cfg = _base_cfg(arm)
    input_size = int(cfg["BLSTM_NNetInputSize"])
    pack = read_weight_vector(arm.ckpt / "best_sad.bin")
    tail = pack[-2 * input_size :]
    mean, std = tail[:input_size], tail[input_size:]
    max_abs_mean = float(np.max(np.abs(mean)))
    max_abs_std_dev = float(np.max(np.abs(std - 1.0)))
    print(f"\n[TAIL {cell}/forward] input_size={input_size} pack_len={len(pack)} max|mean|={max_abs_mean:.3e} max|std-1|={max_abs_std_dev:.3e}")
    assert max_abs_mean == 0.0, f"{cell}: normalize mean not identity (max|mean|={max_abs_mean})"
    assert max_abs_std_dev == 0.0, f"{cell}: normalize std not identity (max|std-1|={max_abs_std_dev})"


# ======================================================================================= #
# (1) S5.6 on real data -- streamed vs offline fast CAUSAL, same file/pack/config.
# ======================================================================================= #


@pytest.mark.slow
@requires_corpus
@pytest.mark.parametrize("cell", _CELLS)
def test_streaming_equivalence_on_corpus(cell: str) -> None:
    """Stream one real mono corpus wav through the PyO3 causal session at 100 ms vs the
    OFFLINE fast causal run on the SAME file/pack/overlaid config. Speech-interval count
    IDENTICAL (R1 STOP on mismatch), boundary max_dt within the VRCTS 4-decimal quantum,
    prefix-consistent, chunk-invariant, latency inside the config-derived causal bound."""
    arm = _ensure_causal_arm(cell)
    wav = _sorted_first_train_wav()
    rate, chan = _read_mono_wav(wav)
    gain = _measure_gain(chan)
    samples = (chan.astype(np.float32) / np.float32(32768.0)).copy()
    dur = (len(chan) - 1) / rate

    listing = _single_file_listing(arm, wav, "_t9_single.flst")
    cfg = _inference_cfg(arm, _frozen_overlay(gain) | _WHOLE_FILE, fileslisting=listing.name)
    cfg_path = arm.out_dir / f"_t9_stream_{cell}.config"
    cfg_path.write_text(B._config_text(cfg))

    # The DERIVED causal bound, recomputed from THIS config; the three summands shared with
    # the phase-8 windowed arm cross-check the config lineage, and `nn_window == 0` is the
    # phase-9 structural win made explicit.
    comp = _derive_causal_bound(cfg, float(rate))
    assert abs(comp["feature_reach"] - _PHASE8_FEATURE_REACH_S) < 5e-6, f"feature_reach {comp['feature_reach']:.5f} != phase-8 {_PHASE8_FEATURE_REACH_S}"
    assert abs(comp["conv_delay"] - _PHASE8_CONV_DELAY_S) < 5e-6, f"conv_delay {comp['conv_delay']:.5f} != phase-8 {_PHASE8_CONV_DELAY_S}"
    assert abs(comp["holdback"] - _PHASE8_HOLDBACK_S) < 5e-5, f"holdback {comp['holdback']:.5f} != phase-8 {_PHASE8_HOLDBACK_S} (config lineage drift?)"
    assert comp["nn_window"] == 0.0, "a causal session has no window lookahead"

    # --- streamed (100 ms) ---
    push_a, fin_a, seg_a, max_lag, mean_lag, stream_s = _stream(cfg_path, rate, samples, int(round(0.1 * rate)))
    streamed_speech = _speech_intervals_from_seg_rows(seg_a)

    # --- offline fast causal (image mode, VRCTS dump) ---
    t0 = time.time()
    off_hyp = _run_offline_image(arm, cfg, arm.out_dir / f"_t9_off_{cell}", f"{cell}_frozen")
    offline_s = time.time() - t0
    offline_speech = _speech_intervals_from_hyp(off_hyp)

    # --- chunk invariance (a second, deliberately non-round granularity) ---
    push_b, fin_b, seg_b, _, _, _ = _stream(cfg_path, rate, samples, 101)

    print(
        f"\n[EQUIV {cell}/forward] dur={dur:.3f}s gain={gain:.6e} rate={rate} "
        f"streamed_speech={len(streamed_speech)} offline_speech={len(offline_speech)} "
        f"seg_rows={len(seg_a)} push_em={len(push_a)} fin_em={len(fin_a)} "
        f"max_lag={max_lag:.4f} mean_lag={mean_lag:.4f} bound={comp['bound']:.5f} "
        f"(reach={comp['feature_reach']:.5f} nn={comp['nn_window']:.5f} sub={comp['sub_sample']:.5f} "
        f"conv={comp['conv_delay']:.5f} hold={comp['holdback']:.5f}) "
        f"stream_s={stream_s:.2f} offline_s={offline_s:.2f}"
    )

    # (R1) speech-interval COUNT identical -- STOP-and-adjudicate on any mismatch.
    assert len(streamed_speech) == len(offline_speech), (
        f"{cell}: streamed vs offline-fast-causal speech-segment COUNT differs ({len(streamed_speech)} vs {len(offline_speech)}) -- R1 STOP"
    )

    # (R1) boundary max_dt within the VRCTS 4-decimal write quantum (the only Python-side gap).
    max_dt = _boundary_max_dt(streamed_speech, offline_speech) if streamed_speech else 0.0
    print(f"[EQUIV {cell}/forward] boundary_max_dt={max_dt:.3e}s (VRCTS quantum {_VRCTS_QUANTUM_S:.1e})")
    assert max_dt <= _VRCTS_QUANTUM_S, f"{cell}: boundary max_dt {max_dt:.3e}s exceeds the VRCTS quantum {_VRCTS_QUANTUM_S:.1e}s -- R1 STOP"

    # At the 4-decimal VRCTS grain the two speech-interval SETS are IDENTICAL (the design's
    # exact-agreement prediction, surfaced through the write precision).
    streamed_4dp = {(f"{b:.4f}", f"{e:.4f}") for b, e in streamed_speech}
    offline_4dp = {(f"{b:.4f}", f"{e:.4f}") for b, e in offline_speech}
    assert streamed_4dp == offline_4dp, f"{cell}: streamed vs offline speech SET differs at 4dp: {streamed_4dp} vs {offline_4dp}"

    # PREFIX consistency: the emitted set equals the final partition (no missing, no extra).
    emitted_set = {(round(e[0], 4), round(e[1], 4), e[2]) for e in push_a + fin_a}
    part_set = {(round(seg_a[i][0], 4), round(seg_a[i + 1][0], 4), seg_a[i][1]) for i in range(len(seg_a) - 1)}
    assert emitted_set == part_set, f"{cell}: emitted set != final partition (retraction/extra -- phase-8 R2): {len(emitted_set)} vs {len(part_set)}"
    assert len(push_a + fin_a) == len(part_set), f"{cell}: emitted count must equal the partition size (no re-emission)"

    # CHUNK invariance: a different push granularity yields the identical segmentation + set.
    assert seg_a == seg_b, f"{cell}: the final segmentation must be chunk-invariant"
    assert emitted_set == {(round(e[0], 4), round(e[1], 4), e[2]) for e in push_b + fin_b}, f"{cell}: the emitted set must be chunk-invariant"

    # DEGENERACY PIN (T9 review minor 4) -- a TRIPWIRE, not a property claim. Both T8 subset
    # checkpoints mode-collapse to all-speech on this file, so the partition is a SINGLE speech
    # interval spanning it, nothing is emitted mid-stream, and the prefix + latency legs below
    # pass TRIVIALLY. Pinning that degeneracy means a future, genuinely selective checkpoint
    # (the full-corpus launcher's output) fails HERE, loudly, instead of quietly continuing to
    # satisfy assertions that no longer test anything -- at which point this leg must be
    # strengthened to a real interior-boundary comparison (the Rust gate's swept-bias fixtures
    # are the model). Read a failure here as "the checkpoint got better, go strengthen the
    # test", never as a streaming regression.
    assert len(streamed_speech) == 1 and abs(streamed_speech[0][0]) < 1e-9 and abs(streamed_speech[0][1] - dur) < 1e-3, (
        f"{cell}: this leg is pinned to the all-speech-collapsed regime (expected one speech interval spanning [0, {dur:.4f}], "
        f"got {streamed_speech}); a selective checkpoint makes the prefix/latency legs below non-trivial -- STRENGTHEN them"
    )
    assert not push_a and len(fin_a) == 1, f"{cell}: the collapsed regime emits nothing mid-stream (push={len(push_a)} finish={len(fin_a)})"

    # LATENCY (S5.5, recorded): the measured max lag sits inside the config-derived causal
    # bound. Trivially so under the all-speech collapse pinned just above (one segment
    # finalized at EOS -> lag 0); the mid-stream budget is exercised on the crafted Rust gate
    # (`phase9_stream_causal.rs::latency_bounds`), which sweeps the output bias until real
    # interior boundaries exist.
    assert max_lag <= comp["bound"], f"{cell}: max_lag {max_lag:.4f}s exceeds the derived causal bound {comp['bound']:.5f}s"
    assert max_lag >= mean_lag >= 0.0, f"{cell}: lag stats must be sane (max >= mean >= 0)"


# ======================================================================================= #
# (2) The overlay is not a semantic change: type 0 == type 1 on an identity tail.
# ======================================================================================= #


@pytest.mark.slow
@requires_corpus
@pytest.mark.parametrize("cell", _CELLS)
def test_frozen_overlay_is_type1_equivalent(cell: str) -> None:
    """`test_frozen_tail_is_identity` proves the pack's normalize tail is `(x-0)/1`; this leg
    proves the CONSEQUENCE end to end on real data: the same pack, same file, same fixed gain,
    scored offline under `BLSTM_InputNormalizationType 0` and `1`, produces the IDENTICAL
    VRCTS partition. So choosing the type-0 arm for the streaming leg above is evidenced, not
    assumed -- and, symmetrically, this leg does NOT exercise the type-1 arithmetic (an
    identity affine is indistinguishable from no affine; that coverage is the Rust gate's
    calibrated NON-identity tail)."""
    arm = _ensure_causal_arm(cell)
    wav = _sorted_first_train_wav()
    _, chan = _read_mono_wav(wav)
    gain = _measure_gain(chan)
    listing = _single_file_listing(arm, wav, "_t9_single.flst")

    hyp0 = _run_offline_image(
        arm, _inference_cfg(arm, _frozen_overlay(gain, "0") | _WHOLE_FILE, fileslisting=listing.name), arm.out_dir / f"_t9_n0_{cell}", f"{cell}_n0"
    )
    hyp1 = _run_offline_image(
        arm, _inference_cfg(arm, _frozen_overlay(gain, "1") | _WHOLE_FILE, fileslisting=listing.name), arm.out_dir / f"_t9_n1_{cell}", f"{cell}_n1"
    )
    print(f"\n[NORM0-vs-NORM1 {cell}/forward] segs={len(hyp0)}/{len(hyp1)} identical={hyp0 == hyp1}")
    assert hyp0 == hyp1, f"{cell}: type-0 and type-1 partitions differ on an IDENTITY tail: {hyp0} vs {hyp1}"


# ======================================================================================= #
# (3) CAUSALITY COST -- native self-norm vs the frozen streamable regime (REPORTED).
# ======================================================================================= #


@pytest.mark.slow
@requires_corpus
@pytest.mark.parametrize("cell", _CELLS)
def test_causality_cost_dcf_on_corpus(cell: str) -> None:
    """Score the T8 held-out slice with the SAME trained pack under (a) its NATIVE regime
    (type -1 per-sequence self-normalization + per-file `(2*RMS+max)/2` gain -- what training
    and the T8 gate measured) and (b) the FROZEN streamable regime (`Audio_fixed_gain` +
    type 0), and REPORT the pooled DCF delta with its mechanism. This is the price of
    causality on this checkpoint: (a) is not streamable in principle. REPORTED, never gated
    (sanity only: both DCFs finite and in range).

    THE TWO SIDES DIFFER IN THE THREE NORMALIZATION KEYS AND NOTHING ELSE. In particular the
    duration windowing (`Audio_offset` / `Audio_max_duration 20`, the T8 recipe's cap) is the
    arm's own on BOTH sides -- `_frozen_overlay` deliberately carries no windowing keys, and
    a `_WHOLE_FILE` overlay here would have scored the frozen side on the FULL held-out files
    while the native side saw only their first 20 s, turning a duration artefact into a fake
    causality cost (caught in self-review; the tell was a 10x scoring-time asymmetry)."""
    arm = _ensure_causal_arm(cell)
    # A representative fixed gain: the sorted-first train file's. On this arm's all-DCT /
    # `IgnoreFirstDCT` front-end a uniform per-file log-shift is annihilated by the AC DCT
    # basis + the temporal deltas (the phase-8 gain-inertness finding), so one global gain
    # scores every held-out file identically to a per-file gain.
    _, chan = _read_mono_wav(_sorted_first_train_wav())
    gain = _measure_gain(chan)

    native_cfg = _inference_cfg(arm, {"Inference_Path": "fast"})
    frozen_cfg = _inference_cfg(arm, _frozen_overlay(gain))

    t0 = time.time()
    native_rep, _ = B._score_sad_pack_on_test(
        native_cfg, arm.out_dir, arm.ckpt / "best_sad.bin", arm.out_dir / f"_t9_cc_native_{cell}", arm.test_records, arm.test_listing.name
    )
    native_s = time.time() - t0
    t0 = time.time()
    frozen_rep, _ = B._score_sad_pack_on_test(
        frozen_cfg, arm.out_dir, arm.ckpt / "best_sad.bin", arm.out_dir / f"_t9_cc_frozen_{cell}", arm.test_records, arm.test_listing.name
    )
    frozen_s = time.time() - t0
    assert native_rep is not None and frozen_rep is not None, f"{cell}: both regimes must produce held-out VRCTS hyps"

    native_dcf = {c: native_rep.by_collar(c).dcf for c in _COLLARS}
    frozen_dcf = {c: frozen_rep.by_collar(c).dcf for c in _COLLARS}
    deltas = {c: frozen_dcf[c] - native_dcf[c] for c in _COLLARS}
    n5 = native_rep.by_collar(0.5)
    f5 = frozen_rep.by_collar(0.5)
    print(
        f"\n[CAUSALITY-DCF {cell}/forward] files={len(arm.test_records)} "
        f"self_norm(-1) DCF={ {c: f'{v:.6f}' for c, v in native_dcf.items()} } (Pmiss={n5.pmiss:.6f} Pfa={n5.pfa:.6f}) "
        f"frozen(type0+gain) DCF={ {c: f'{v:.6f}' for c, v in frozen_dcf.items()} } (Pmiss={f5.pmiss:.6f} Pfa={f5.pfa:.6f}) "
        f"delta(frozen-self)={ {c: f'{d:+.6f}' for c, d in deltas.items()} } "
        f"native_s={native_s:.2f} frozen_s={frozen_s:.2f}"
    )
    for c in _COLLARS:
        assert 0.0 <= native_dcf[c] <= 1.0 and 0.0 <= frozen_dcf[c] <= 1.0, f"{cell}: DCF@{c} out of [0,1]"


# ======================================================================================= #
# (4) S4.3 on real data -- fast vs exact causal metric parity on the held-out slice.
# ======================================================================================= #


def _score_causal(arm: _CausalArm, inference_path: str) -> tuple[B.DcfReport, Path]:
    """Score the trained causal pack on the held-out slice under `inference_path`, in the
    arm's NATIVE config (type -1, `BLSTM_window 0`, no fixed gain) -- so exact and fast
    configs are identical except `Inference_Path` and the per-path `Dump_Directory`
    `_score_sad_pack_on_test` writes its VRCTS hyps into (a write TARGET, read by nothing in the
    compute path, and necessarily distinct or the two runs would clobber each other's output), so
    the comparison isolates the f32 causal kernels.
    `_score_sad_pack_on_test` `set_weights`-injects on exact (redundant: the config
    repoint already did it) and SKIPS on fast (T6b)."""
    cfg = _inference_cfg(arm, {"Inference_Path": inference_path})
    dump_dir = arm.out_dir / f"_t9_par_{inference_path}"
    if dump_dir.exists():
        shutil.rmtree(dump_dir)
    t0 = time.time()
    report, dump = B._score_sad_pack_on_test(cfg, arm.out_dir, arm.ckpt / "best_sad.bin", dump_dir, arm.test_records, arm.test_listing.name)
    arm.timings[f"score_{inference_path}_s"] = time.time() - t0
    assert report is not None, f"{arm.cell}/{inference_path}: no VRCTS hyps produced (empty held-out slice)"
    return report, dump


@pytest.mark.slow
@requires_corpus
@pytest.mark.parametrize("cell", _CELLS)
def test_causal_fast_vs_exact_metric_parity(cell: str) -> None:
    """The phase-7 SAD metric-parity leg, on the phase-9 CAUSAL cells: the T8 checkpoint
    scored on its own held-out slice under `Inference_Path exact` vs `fast`. Per-file VRCTS
    boundaries IDENTICAL (count + types + times); pooled DCF delta EXACTLY 0.0 at every
    collar. The exact-path DCF@0.5 is additionally pinned against T8's own gate number -- the
    checkpoint-provenance guard (see `_EXPECTED_EXACT_DCF_AT_0_5`).

    `n_scored` counts files that produced a hyp xml on BOTH paths; it is a SCORED-file count,
    not an assertion that every `test_records` entry was covered (the coverage gate is
    `n_scored > 0`) -- the phase-7 T11 note, verbatim."""
    arm = _ensure_causal_arm(cell)
    rep_e, dump_e = _score_causal(arm, "exact")
    rep_f, dump_f = _score_causal(arm, "fast")

    n_scored = 0
    type_or_count_disagree = 0
    max_dt = 0.0
    for rec in arm.test_records:
        xe = B._hyp_xml_for(rec["filename"], dump_e)
        xf = B._hyp_xml_for(rec["filename"], dump_f)
        if not (xe.is_file() and xf.is_file()):
            continue
        ie = load_vrcts_hyp(xe)
        if_ = load_vrcts_hyp(xf)
        n_scored += 1
        if [k for _, _, k in ie] != [k for _, _, k in if_]:
            type_or_count_disagree += 1
            continue
        for (se, ee, _), (sf, ef, _) in zip(ie, if_, strict=True):
            max_dt = max(max_dt, abs(se - sf), abs(ee - ef))

    dcf_deltas = {c: abs(rep_f.by_collar(c).dcf - rep_e.by_collar(c).dcf) for c in _COLLARS}
    print(
        f"\n[PARITY {cell}/forward] files={n_scored} boundary_type_or_count_disagree={type_or_count_disagree} "
        f"max_boundary_dt={max_dt:.3e}s dcf@0.5 exact={rep_e.by_collar(0.5).dcf:.6f} fast={rep_f.by_collar(0.5).dcf:.6f} "
        f"dcf_deltas={ {c: f'{d:.3e}' for c, d in dcf_deltas.items()} } "
        f"train_s={arm.train_s:.1f} (cold={arm.trained_now}) score_exact_s={arm.timings.get('score_exact_s', 0):.2f} "
        f"score_fast_s={arm.timings.get('score_fast_s', 0):.2f}"
    )

    # Provenance: the EXACT-path DCF@0.5 matches T8's gate number for this cell.
    exact_dcf = rep_e.by_collar(0.5).dcf
    expected = _EXPECTED_EXACT_DCF_AT_0_5[cell]
    assert round(exact_dcf, 6) == expected, f"{cell}: exact-path DCF@0.5={exact_dcf:.6f} != T8-pinned {expected} (stale/mismatched checkpoint cache?)"

    # (R1 drift detector): identical decisions, EXACTLY-0.0 metric delta. STOP on any move.
    assert n_scored > 0, f"{cell}: no held-out hyp xmls to compare"
    assert type_or_count_disagree == 0, f"{cell}: {type_or_count_disagree}/{n_scored} files differ in segment count/types fast-vs-exact"
    assert max_dt == 0.0, f"{cell}: boundary times moved fast-vs-exact (max_dt={max_dt:.3e}s)"
    for c in _COLLARS:
        assert rep_f.by_collar(c).dcf == rep_e.by_collar(c).dcf, f"{cell}: DCF@{c} moved {rep_e.by_collar(c).dcf} -> {rep_f.by_collar(c).dcf}"
