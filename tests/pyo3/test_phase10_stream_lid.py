"""Phase 10 Task 10: `speech_rs.StreamingLidSession` binding smoke.

`importorskip` guards the whole module: the plain `python-test` CI job never builds
`speech_rs`, so it SKIPS these; only the `python-pyo3` job (which runs `maturin develop`)
exercises them.

The binding is a THIN wrapper over `speech::fast::stream_lid::StreamingLidSession` (the
online twin of the offline fast Mode-7 LID Twin), bit-identical to the offline driver BY
CONSTRUCTION (see that module's own doc + `tests/phase8_stream_lid.rs`, which pins it
against the REAL offline `FastTwinLid`). This test does not re-derive that arithmetic; it
pins the NEW Python surface:

  - an INDEPENDENT Python reimplementation of the committed cep binary format (mirroring
    `audio.rs::read_cep`'s documented byte layout from scratch, over the SAME committed
    phase-6 cep fixtures `tests/reference_data/phase6/cep/{tiny_ok,multi_ok}.plp`) feeds
    `push_utterance` real committed feature matrices -- a genuine cross-language
    duplicate-catch: if either reader or the binding's numpy-array plumbing has a
    shape/dtype/stride bug, the fed matrix is wrong and the downstream scores go wrong too;
  - PREFIX-CORRECTNESS (the phase-8 T6 "two independent runs must agree" pattern -- the LID
    analogue of `test_phase8_stream.py`'s chunk-invariance property): a FRESH session
    pushing only the first k utterances must equal the k-th push's running aggregate of an
    incremental session, for every k;
  - a `predicted_language` RE-DERIVATION from `classification_errors` via the documented
    `langid[c] = errors[c]/100 + (-2 if c == ti else 0)` relationship (mirrors
    `tests/phase8_stream_lid.rs::offline_predicted_language`), an algebraic cross-check
    independent of the struct field the binding returns;
  - the two construction bails (mirrors
    `tests/phase8_stream_lid.rs::new_bails_on_{non_mode7,missing_lid_net}`).

T10 REVIEW STRENGTHENING (I-1): every check above compares the binding's OWN outputs
against each other (prefix runs, an algebraic identity) -- none of them can catch a
residual marshalling bug that yields a VALID-SHAPED but numerically wrong matrix (e.g. a
transposed/row-vs-column-major `confusion`, or a `classification_errors` read from the
wrong offset). `_EXPECTED` below closes that gap with LITERAL values computed OUTSIDE the
binding entirely -- a one-time `tests/phase8_stream_lid.rs`-equivalent Rust run
(`FastTwinLid::from_legacy` + `get_segmentation`, the same offline oracle
`tests/phase8_stream_lid.rs::run_offline` already exercises, invoked via a temporary
`--nocapture` test dump, transcribed here, then deleted -- see the Phase 10 Task 10 review
strengthening commit). `test_push_utterance_literal_oracle_values` asserts `finish()`'s
`classification_errors`/`confusion` against these transcribed numbers exactly.

T10 REVIEW STRENGTHENING (I-2): `scored_count()` was exposed on the pyclass but never
called by any test; `test_push_utterance_literal_oracle_values` now asserts it equals the
`finish()` aggregate's `segments_count` field.

T10 REVIEW (optional, I-3, SKIPPED as a MISS case -- stated why): a `lang != 0` MISS case
is not achievable with the committed binary-net (`class_nb == 2`) cep fixtures -- the only
non-zero target is 1, and both fixtures' raw NN posterior already favors class 1
regardless of target (confirmed by the SAME one-time oracle dump: `lang=1` on
`tiny_ok.plp` is a HIT, `is_lid_correct=100`, not a miss). Rather than skip the `ti=1` code
path entirely, `test_push_utterance_literal_oracle_values[tiny_ok.plp-1]` pins that HIT
case instead (still literal-value-pinned, still exercises the `ti=1` branch the other
tests never touch) -- a real MISS at `ti=1` would need a different net/fixture, out of
scope here; the CLI smoke's `s3` case is the committed genuine-MISS coverage this repo has.

phSeq coverage (the OTHER `File_Type`) lives in the Rust CLI smoke
(`tests/phase10_stream_lid_cli.rs`) instead of here, since building phSeq one-hot matrices
in Python would mean re-implementing the 38-entry `letterMapping` table for no independent-
reimplementation benefit (the cep format is a simple, stable, already-documented binary
layout -- worth a second implementation; the phSeq encoding is not).
"""

import struct
from pathlib import Path
from typing import Any

import numpy as np
import pytest

speech_rs = pytest.importorskip("speech_rs")

REPO_ROOT = Path(__file__).resolve().parents[2]
PHASE4B = REPO_ROOT / "tests" / "reference_data" / "phase4b"
PHASE6 = REPO_ROOT / "tests" / "reference_data" / "phase6"

CONFIG = PHASE4B / "twin_mode7.config"
LID_WEIGHTS = PHASE4B / "LID_bestNNWeight_1.bin"

CEP_FIXTURES = [("tiny_ok.plp", 0), ("multi_ok.plp", 0)]

# Literal INDEPENDENT-ORACLE values (I-1): computed OUTSIDE this binding, via a one-time
# `FastTwinLid::from_legacy` + `get_segmentation` run (the same offline oracle
# `tests/phase8_stream_lid.rs::run_offline` uses) over `twin_mode7.config` + the real
# 12409-weight LID net + these exact committed cep fixtures, dumped with a temporary
# `--nocapture` test (deleted after transcription -- not part of the committed suite).
# Keyed by `(name, lang)`; each value is `(classification_errors, confusion, is_lid_correct)`.
_EXPECTED: dict[tuple[str, int], tuple[list[float], list[list[float]], bool]] = {
    ("tiny_ok.plp", 0): (
        [207.77509335905583, 92.2249066409442],
        [[0.0, 1.0, 2.0, 0.0], [1.0, 0.0, 1.0, 1.0], [2.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
        False,
    ),
    ("multi_ok.plp", 0): (
        [201.94480557785144, 98.05519442214855],
        [[0.0, 1.0, 2.0, 0.0], [1.0, 0.0, 2.0, 2.0], [2.0, 0.0, 0.0, 0.0], [0.0, 0.0, 2.0, 0.0]],
        False,
    ),
    # ti=1 case (I-3 note): a HIT, not a MISS -- see the module docstring's I-3 entry.
    ("tiny_ok.plp", 1): (
        [7.77509335905581, 292.2249066409442],
        [[0.0, 1.0, 2.0, 0.0], [1.0, 0.0, 0.0, 0.0], [2.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 0.0]],
        True,
    ),
}


def _read_cep(path: Path) -> list[np.ndarray]:
    """Independent Python port of `audio.rs::read_cep`'s documented byte layout: int32
    nbRecords | int16 vectorSize | int16 magic (magic ignored, matching the Rust reader),
    then a per-record int32 vectorNb table, then row-major float32 payload -- one
    `(vectorNb x vectorSize)` f64 matrix per record with `vectorNb > 0` (others skipped)."""
    data = path.read_bytes()
    nb_records = struct.unpack_from("<i", data, 0)[0]
    vector_size = struct.unpack_from("<h", data, 4)[0]
    off = 8
    vector_nbs = struct.unpack_from(f"<{nb_records}i", data, off)
    off += 4 * nb_records
    feats: list[np.ndarray] = []
    for vn in vector_nbs:
        if vn <= 0:
            continue
        n = vn * vector_size
        vals = struct.unpack_from(f"<{n}f", data, off)
        off += 4 * n
        feats.append(np.array(vals, dtype=np.float64).reshape(vn, vector_size))
    return feats


def _staged_config(tmp_path: Path, *extra: str) -> Path:
    """Stage `twin_mode7.config` with an ABSOLUTE `BLSTM_LID_weightsFile` override (the
    committed config's own value is repo-root-relative, matching every other committed
    config in this repo -- see CLAUDE.md's "Run from repo root:" convention) plus any
    `extra` last-wins override lines, so construction never depends on pytest's own
    invocation CWD (the `test_phase8_stream.py::_stage_frozen_config` convention)."""
    base = CONFIG.read_text()
    lines = [base, f"BLSTM_LID_weightsFile {LID_WEIGHTS}", *extra]
    staged = tmp_path / "twin_mode7_staged.config"
    staged.write_text("\n".join(lines) + "\n")
    return staged


def _session(tmp_path: Path, lang: int) -> Any:
    return speech_rs.StreamingLidSession(str(_staged_config(tmp_path)), 8000.0, lang)


def _target_index(lang: int, class_nb: int) -> int:
    return lang if 0 <= lang < class_nb else 0


def _recompute_predicted_language(errors: list[float], ti: int) -> int:
    """`langid[c] = errors[c]/100 + (-2 if c == ti else 0)`, first-max argmax -- the same
    algebraic relationship `tests/phase8_stream_lid.rs::offline_predicted_language` uses to
    recover the offline argmax from the asserted `classification_errors` oracle."""
    langid = [e / 100.0 + (-2.0 if c == ti else 0.0) for c, e in enumerate(errors)]
    best = 0
    best_v = langid[0]
    for c, v in enumerate(langid):
        if v > best_v:
            best_v = v
            best = c
    return best


@pytest.mark.parametrize("name,lang", CEP_FIXTURES)
def test_push_utterance_self_consistent(tmp_path: Path, name: str, lang: int) -> None:
    feats = _read_cep(PHASE6 / "cep" / name)
    assert feats, f"{name}: fixture must yield at least one utterance"
    sess = _session(tmp_path, lang)

    last_agg: tuple | None = None
    for i, feat in enumerate(feats):
        scores, argmax, correct, agg = sess.push_utterance(feat)
        errors, confusion, is_lid_correct, predicted_language, segments_count = agg

        if argmax is not None:
            assert scores, f"{name} u{i}: a scored utterance must have non-empty scores"
            assert argmax == int(np.argmax(scores)), f"{name} u{i}: reported argmax vs scores argmax"
            ti = _target_index(lang, len(scores))
            assert correct == (argmax == ti), f"{name} u{i}: correct == (argmax == target)"
        else:
            assert scores == [], f"{name} u{i}: a skipped utterance must have empty scores"
            assert correct is None, f"{name} u{i}: a skipped utterance must have correct=None"

        assert confusion.ndim == 2 and confusion.shape[0] == confusion.shape[1], f"{name} u{i}: confusion must be square"
        assert is_lid_correct in (True, False)
        assert segments_count >= 0
        last_agg = agg

    assert last_agg is not None
    fin = sess.finish()
    for a, b in zip(last_agg, fin, strict=True):
        eq = (a == b).all() if isinstance(a, np.ndarray) else a == b
        assert eq, f"{name}: finish() must equal the last push's running_aggregate"

    # Idempotent finish().
    fin2 = sess.finish()
    for a, b in zip(fin, fin2, strict=True):
        eq = (a == b).all() if isinstance(a, np.ndarray) else a == b
        assert eq, f"{name}: finish() must be idempotent"

    # predicted_language re-derivation (independent algebraic cross-check).
    fin_errors, _confusion, _correct, fin_predicted, fin_count = fin
    ti = _target_index(lang, len(fin_errors))
    assert fin_predicted == _recompute_predicted_language(fin_errors, ti), (
        f"{name}: predicted_language must equal the argmax recovered from classification_errors"
    )

    # I-2: scored_count() must equal the finish() aggregate's segments_count.
    assert sess.scored_count() == fin_count, f"{name}: scored_count() must equal finish().segments_count"


@pytest.mark.parametrize("name,lang", [*CEP_FIXTURES, ("tiny_ok.plp", 1)])
def test_push_utterance_literal_oracle_values(tmp_path: Path, name: str, lang: int) -> None:
    """I-1: `finish()`'s `classification_errors`/`confusion` (plus `is_lid_correct`) must
    match LITERAL values computed OUTSIDE the binding entirely (see `_EXPECTED`'s
    provenance comment) -- the residual marshalling-bug class (a valid-shaped but
    numerically wrong matrix: transposed confusion, a wrong-offset errors read, ...) that
    the binding's own self-consistency checks structurally cannot catch."""
    feats = _read_cep(PHASE6 / "cep" / name)
    sess = _session(tmp_path, lang)
    for feat in feats:
        sess.push_utterance(feat)
    errors, confusion, is_lid_correct, _predicted, segments_count = sess.finish()

    want_errors, want_confusion, want_correct = _EXPECTED[(name, lang)]
    assert errors == pytest.approx(want_errors), f"{name} lang={lang}: classification_errors vs the independent oracle"
    assert confusion.tolist() == want_confusion, f"{name} lang={lang}: confusion vs the independent oracle"
    assert is_lid_correct == want_correct, f"{name} lang={lang}: is_lid_correct vs the independent oracle"

    # I-2 (again, on this dedicated literal-value leg): scored_count() must agree.
    assert sess.scored_count() == segments_count, f"{name} lang={lang}: scored_count() must equal segments_count"


@pytest.mark.parametrize("name,lang", CEP_FIXTURES)
def test_running_aggregate_is_prefix_correct(tmp_path: Path, name: str, lang: int) -> None:
    feats = _read_cep(PHASE6 / "cep" / name)
    incremental = _session(tmp_path, lang)

    for k in range(1, len(feats) + 1):
        *_rest, agg_k = incremental.push_utterance(feats[k - 1])

        fresh = _session(tmp_path, lang)
        for feat in feats[:k]:
            fresh.push_utterance(feat)
        fresh_agg = fresh.finish()

        for field, (a, b) in enumerate(zip(agg_k, fresh_agg, strict=True)):
            eq = (a == b).all() if isinstance(a, np.ndarray) else a == b
            assert eq, f"{name} k={k} field {field}: incremental push vs fresh-session finish"


def test_new_bails_on_non_mode7(tmp_path: Path) -> None:
    staged = _staged_config(tmp_path, "BLSTM_LID_Mode 0")
    with pytest.raises(RuntimeError, match=r"(?i)mode 7"):
        speech_rs.StreamingLidSession(str(staged), 8000.0, 0)


def test_new_bails_on_missing_lid_net(tmp_path: Path) -> None:
    # An explicit empty `BLSTM_LID_weightsFile` (last-wins, overriding the absolute
    # override `_staged_config` itself appends) -> no net loads -> construction bails.
    staged = _staged_config(tmp_path, "BLSTM_LID_weightsFile ")
    with pytest.raises(RuntimeError, match=r"(?i)net"):
        speech_rs.StreamingLidSession(str(staged), 8000.0, 0)


# --------------------------------------------------------------------------------------- #
# Issue #57: the binding streams a FORWARD (causal) LID net too -- one row over the committed
# phase-9 `twin_mode7_lid_slstm_forward` fixture (the plain regime the launcher writes on a
# forward LID net). The arithmetic is pinned against the offline Twin in Rust
# (`tests/phase8_stream_lid.rs::every_shape_streams_prefix_correct_and_finish_equals_offline`);
# this row pins the Python surface on that shape: construction, prefix-correctness between an
# incremental and a fresh session, finish == last push, and the predicted-language identity.
# The utterances are synthetic phSeq-shaped one-hots (38 wide, the `letterMapping` width;
# the 12-wide LID net crops them), drawn from a seeded rng so the row is deterministic.
# --------------------------------------------------------------------------------------- #

PHASE9 = REPO_ROOT / "tests" / "reference_data" / "phase9"
FORWARD_FIXTURE = "twin_mode7_lid_slstm_forward"


def _staged_forward_config(tmp_path: Path) -> Path:
    # Last-wins overrides for both weight files: the fixture names them bare, and since
    # issue #65 the fast Twin reads the SAD file too (phase4b holds it), not only the LID one.
    base = (PHASE9 / f"{FORWARD_FIXTURE}.config").read_text()
    staged = tmp_path / f"{FORWARD_FIXTURE}_staged.config"
    staged.write_text(base + f"BLSTM_weightsFile {PHASE4B / 'tiny_sad_seed.bin'}\n" + f"BLSTM_LID_weightsFile {PHASE9 / (FORWARD_FIXTURE + '_seed.bin')}\n")
    return staged


def _synthetic_phseq_utterances(n: int, seed: int = 57) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    out: list[np.ndarray] = []
    for _ in range(n):
        frames = int(rng.integers(2, 10))
        m = np.zeros((frames, 38), dtype=np.float64)
        m[np.arange(frames), rng.integers(0, 38, size=frames)] = 1.0
        out.append(m)
    return out


def test_forward_lid_net_streams_prefix_correct(tmp_path: Path) -> None:
    feats = _synthetic_phseq_utterances(6)
    staged = _staged_forward_config(tmp_path)
    incremental = speech_rs.StreamingLidSession(str(staged), 8000.0, 1)
    last_agg: tuple | None = None
    for k in range(1, len(feats) + 1):
        scores, argmax, correct, agg_k = incremental.push_utterance(feats[k - 1])
        assert argmax is not None and scores and correct == (argmax == 1), f"k={k}: a scored utterance"
        fresh = speech_rs.StreamingLidSession(str(staged), 8000.0, 1)
        for feat in feats[:k]:
            fresh.push_utterance(feat)
        for field, (a, b) in enumerate(zip(agg_k, fresh.finish(), strict=True)):
            eq = (a == b).all() if isinstance(a, np.ndarray) else a == b
            assert eq, f"k={k} field {field}: incremental push vs fresh-session finish"
        last_agg = agg_k
    assert last_agg is not None
    fin = incremental.finish()
    for a, b in zip(last_agg, fin, strict=True):
        eq = (a == b).all() if isinstance(a, np.ndarray) else a == b
        assert eq, "finish() must equal the last push's running_aggregate"
    errors, _confusion, _correct, predicted, count = fin
    assert count == len(feats) == incremental.scored_count()
    assert predicted == _recompute_predicted_language(errors, 1)
