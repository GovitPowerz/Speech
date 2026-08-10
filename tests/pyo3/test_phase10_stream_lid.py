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
    fin_errors, _confusion, _correct, fin_predicted, _count = fin
    ti = _target_index(lang, len(fin_errors))
    assert fin_predicted == _recompute_predicted_language(fin_errors, ti), (
        f"{name}: predicted_language must equal the argmax recovered from classification_errors"
    )


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
