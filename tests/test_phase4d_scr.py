"""Phase 4d Task 11: the `.scr` score-file writer pinned vs Octave.

No `.scr` file exists anywhere in the legacy tree, so the golden
(`tests/reference_data/phase4d/scr/expected.scr`) is the ONLY oracle: Octave driving
Test_BLSTM.m's writer block (`:249-269`) over injected scores via
`tools/octave_harness/stage_scr.m` (HYBRID tier -- the real Tier-1 `processListing.m`
supplies `keys(langMapConf)`, the writer loop is a FALLBACK-TIER transcription with
`% legacy:` provenance; see the stage header for the adjudication).

This closes the Phase 4c divergence flagged in `drivers/test.py::_class_keys` (T12-4c
report item 4): the port ordered class keys by class id, but the legacy indexes score
columns by `keys(langMapConf)` = ALPHABETICAL (ASCII byte order) `lang_dial` keys --
the mapping file's class-id column is IGNORED by the writer (processListing.m:85-88
overwrites langMapConf's values with alphabetical positions). The committed mapping
(`scr/mapping.csv`) assigns ids deliberately NOT in alphabetical order (zzz=0, aaa=1,
mmm=2, bbb=3, nnn=4), so id order vs alphabetical order VISIBLY differ -- the
non-vacuity of the order fix.

The `scores_input.json` inputs are the stage's own dump (raw scores in the SAME
alphabetical key order the stage injected them in), so the two sides' inputs match by
construction -- the same byte-oracle convention as `test_phase4d_listing_writers.py`.

The score chain traverses libm `exp` at TEST time (the golden was rendered on the
oracle env's libm), and the 0.4229... tie value sits only ~2.3 ULPs from its
15th-decimal print-rounding boundary -- so the full byte comparison is CANARY-GATED
(`tests/_libm_gate.py::oracle_strict`, the repo-standard split): byte-exact on the
oracle libm, else the LABEL/ORDER text stays byte-exact (cross-libm stable here: the
tie is exact -- identical inputs -> identical exp results -> the stable sort preserves
column order -- and the non-tie values are separated by whole percents, unmovable by a
ULP) while the numeric fields compare via `assert_f64_close` on the parsed values.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from speech.drivers.test import _class_keys, write_scores

from tests._libm_gate import assert_f64_close, oracle_strict

SCR_DIR = Path(__file__).resolve().parent / "reference_data" / "phase4d" / "scr"


def _inputs() -> dict[str, object]:
    data: dict[str, object] = json.loads((SCR_DIR / "scores_input.json").read_text())
    return data


def test_class_keys_alphabetical_matches_octave_keys_order() -> None:
    """`_class_keys` must return the composed `lang_dial` keys in the exact order the
    real containers.Map's `keys()` returned them (dumped by the stage): ASCII-lexicographic,
    NOT class-id order (processListing.m:10 composes `[lang '_' dial]`; Test_BLSTM.m:249)."""
    inp = _inputs()
    assert _class_keys(SCR_DIR / "mapping.csv") == inp["keys_alpha_order"]


def test_write_scores_matches_octave_golden_bytes() -> None:
    """Compare the full writer law -- softmax `exp(s/100)/max(1e-3,sum)`, stable
    descending sort (250.0 tie between aaa_11 and bbb_22 pins the tie-break), the
    `lang-dial` slicing (`tmp(end-2:end)` -> the leading-underscore 2-char-dial quirk:
    'aaa-_11' not 'aaa-11'), and the num2str('%15.15f') rendering -- against the Octave
    golden. Keys come through `_class_keys` (the fixed alphabetical path), scores are
    the stage's own injected row in the same key order. BYTE-exact on the oracle libm;
    canary-gated off it (see the module docstring): labels/order stay byte-exact, the
    numeric field via `assert_f64_close` on the parsed values (the parse-back
    quantization is ~half a 15th-decimal quantum, far inside the hybrid bound)."""
    inp = _inputs()
    keys = _class_keys(SCR_DIR / "mapping.csv")
    scores = np.asarray(inp["raw_scores_ordered"], dtype=np.float64)
    text = write_scores(scores, keys, str(inp["filename_stem"]))
    want_bytes = (SCR_DIR / "expected.scr").read_bytes()
    if oracle_strict():
        assert text.encode() == want_bytes
        return
    got_lines = text.splitlines()
    want_lines = want_bytes.decode().splitlines()
    assert len(got_lines) == len(want_lines)
    for i, (g, w) in enumerate(zip(got_lines, want_lines, strict=True)):
        g_head, g_num = g.rsplit(" ", 1)
        w_head, w_num = w.rsplit(" ", 1)
        assert g_head == w_head, f"line {i}: label/order mismatch: {g!r} vs {w!r}"
        assert len(g_num) == len(w_num), f"line {i}: numeric field width mismatch: {g!r} vs {w!r}"
        assert_f64_close(float(g_num), float(w_num), f"line {i} score")


def test_class_id_order_fails_golden() -> None:
    """Non-vacuity of the order fix: rendering with the OLD class-id order (the 4c
    `_class_keys` behavior -- keys sorted by the mapping's class-id column) must NOT
    reproduce the golden. Scores stay attached to their score-column position (the
    legacy writer never reorders columns either), so only the labels move -- exactly
    what the old code got wrong."""
    inp = _inputs()
    scores = np.asarray(inp["raw_scores_ordered"], dtype=np.float64)
    rows: list[tuple[int, str]] = []
    for line in (SCR_DIR / "mapping.csv").read_text().splitlines():
        lang, dial, cid = line.strip().split(";")
        rows.append((int(cid), f"{lang}_{dial}"))
    id_order_keys = [key for _cid, key in sorted(rows)]
    assert id_order_keys != inp["keys_alpha_order"]  # the crafted mapping guarantees it
    text = write_scores(scores, id_order_keys, str(inp["filename_stem"]))
    assert text.encode() != (SCR_DIR / "expected.scr").read_bytes()


def test_tie_break_is_stable_original_column_order() -> None:
    """The 250.0/250.0 tie (aaa_11 col 1, bbb_22 col 2): MATLAB sortrows(x',-1) is
    stable, so aaa_11 must precede bbb_22 in the output. Pinned independently of the
    full byte comparison so a tie-break regression is named, not just 'bytes differ'."""
    golden_lines = (SCR_DIR / "expected.scr").read_text().splitlines()
    labels = [line.split(" ")[1] for line in golden_lines]
    assert labels.index("aaa-_11") < labels.index("bbb-_22")
    keys = _class_keys(SCR_DIR / "mapping.csv")
    scores = np.asarray(_inputs()["raw_scores_ordered"], dtype=np.float64)
    text = write_scores(scores, keys, "f")
    got_labels = [line.split(" ")[1] for line in text.splitlines()]
    assert got_labels.index("aaa-_11") < got_labels.index("bbb-_22")
