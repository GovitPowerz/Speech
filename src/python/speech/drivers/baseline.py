"""Phase 6 baseline-training arms: the from-scratch SAD/LID launchers + subset gates.

Design spec `docs/superpowers/specs/2026-07-17-phase-6-baseline-training-design.md`
(S1.7) + plan Task 8. This module is the ORCHESTRATOR that stitches the landed Phase 6
seams into a runnable arm: the cep ingestion (Task 1, `audio.rs::read_cep`), the listing
tooling (Task 2, `dataprep/lre.py`), the modern training loop (Phase 5 + Task 6,
`drivers/train.py::train_modern` with `val_metric="nn_cost_seg"`), the seeded init (Task 3,
`init_weights.py`), and the LID metrics (Task 5, `evaluate.py`). It invents NO new engine
path -- every heavy step delegates to an already-golden-tested seam.

THE LID FEATURES ARM (`arm="lid-features"`): a 12-class Twin (Algo 6) in Mode 7 over the
File_Type-2 `.plp8f0mvsdd` cep features. Mode 7 is the ONLY twin mode that consumes
`audio.external_features` directly (every other mode recomputes a periodogram from the
zeroed cep waveform); the SAD net is a FROZEN feature gate there (constant result_vec), so
ONLY the LID net trains -- the same structural finding the phonotactic Mode-7 arm carries.
The cep files are pre-VAD'd whole-utterance speech (verified: the archive's own
`.plp8f0m.xml` refs carry one SpeechSegment spanning `sigdur`), so a synthesized
whole-file-speech STM reference satisfies the engine's `-m`-mode mandatory-reference gate;
the LID training target is the listing's language label, not the seg reference.

R2 (vtln vs plain): the archive's TRAIN side carries only plain `plp8f0mvsdd`; a
vtln/cmllr-train pairing is structurally impossible, so the baseline pairing is PLAIN
features on both train and eval (RESULTS.md carries the evidence).

THE LID PHONOTACTIC ARM (`arm="lid-phseq"`, Task 10): the 2015 FLAGSHIP regime -- the
same 12-class Twin (Algo 6) in Mode 7, but File_Type 1 (phSeq) instead of File_Type 2
(cep). The LID net scores the phSeq one-hot phoneme sequences (`audio.external_features`,
`read_phseq`: one row per phoneme, a 38-wide one-hot over the letterMapping) DIRECTLY, so
the LID net's input is 38 (vs the cep arm's 23). The FROZEN-SAD contract is identical (and
here VERIFIED at the seam, Task 10 report): Mode 7 never runs the SAD net -- its result_vec
is synthesized constant 10.0 and its weight derivatives are reset-then-scaled but NEVER
accumulated, so `weights_derivatives(sad)` is structurally zero; through the seam a zero
gradient leaves the SAD net EXACTLY at its from-scratch seed (`best_sad.bin == sad_seed.bin`,
asserted in the gate). Both nets are seeded; ONLY the LID net trains. The phSeq files are
sourced by globbing `train/phSeq/*.file.phSeqbis` (the whole-utterance-per-line variant, one
sequence per file) with the language from the 2-letter filename prefix -- the 2015 phSeq
listings localize to 0 rows on this archive (`derive_lid_phseq_records`). The two LID arms
share the entire training/scoring skeleton (`_LID_ARMS`); they differ only in File_Type and
the record derivation.

THE SAD ARM (`arm="sad"`, Task 9): from-scratch algo-3 spectral SAD (Algo 3, File_Type 0)
over the real corpus wav/xml pairs. The DSP front-end (periodogram -> Mel/DCT + deltas)
feeds a single trainable BLSTM whose speech-posterior output drives the hysteresis
segmenter; `train_modern` descends the NNCostSeg objective (SAD = NNCostSeg alone, algo-3
single-net). The listings come from `dataprep.lre.derive_sad_listings` (the seeded 70/15/15
split of the 2066 wav+xml pairs); the references are the corpus `.part.xml` VRCTS files
(routed through the engine's `.xml` reference dispatch, Task 2b). The held-out slice is
scored end to end with the T4 DCF harness: the engine dumps one VRCTS hypothesis xml per
test file (`Dump_Directory`), those are read back via `evaluate.load_vrcts_hyp`, the
`.part.xml` references via `evaluate.load_vrcts_ref` (windowed to the capped-audio span),
and pooled through `evaluate.dcf` -> the first real DCF numbers (trained vs its own
from-scratch init). The corpus wavs are 576-1800 s CallFriend recordings (median ~600 s), so
`Audio_max_duration` caps them; the reference is windowed to the same span (matching the engine's own
`_AudioDuration` reference windowing). NOTE (measured, honest): on a tiny subset the
from-scratch net mode-collapses toward the window-majority (all-speech) -- the same
inherent-property caveat the LID arm carries for its majority classes; the trained-vs-init
DCF improvement (init is all-non-speech, DCF ~0.75; trained fires, DCF ~0.25) is the
direction-safe task metric, and genuine speech/non-speech discrimination is the full-run
launcher's job.

THE SAD V2 ARM (`arm="sad-v2"`, Phase 10 Task 4, spec S3): the SAME arm on the v2 lineage
config (`configs/training/lre_sad_v2.toml`) -- v1 byte-for-byte except `nnet_input_size`
23 -> 11 and `lstm_neuron_nb` `23,24,24` -> `11,24,24`, which makes the declared layer-0
fan-in (`11*4 = 44`) equal the width the DSP front-end actually produces and so RETIRES the
2015-inherited dead-column block (v1 leaves 48 of its 92 fan-in columns structurally
gradient-dead). NO code path of its own: `_SAD_ARMS` covers both, every size downstream
self-derives from the config (the normalize tail, the seed pack, the output MLP width under
`--direction forward`). v1 stays FROZEN and remains the only 2015-capacity-comparable
lineage; v2 is a new lineage whose corrected Xavier fan-in scaling is a MEASURABLE
difference for the launchers to adjudicate, not a promised win (R5). Gates:
`tests/pyo3/test_phase10_gates.py` -- 10 rows (phase 11 T9 added the fifth cell), {lstm,
slstm, mamba, cfc, transformer} x {bidirectional, forward}, with the INVERSE guard (zero dead
layer-0 input columns) where v1's gates pin dead-count floors.

LICENSE HYGIENE: nothing corpus-derived is committed. The listing, mapping, reference, and
seed weight packs are all synthesized at RUNTIME under `out_dir` from `corpus_root`; the
committed TOML carries only DSP hyperparameter numbers + placeholder path strings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from rich.console import Console

from speech.batching import read_listing
from speech.config_bridge import nnet_spec
from speech.dataprep.lre import LRE03_LANGUAGES, derive_sad_listings, localize_listing
from speech.drivers.state import ModernTrainParams, RunState
from speech.drivers.train import train_modern
from speech.evaluate import DcfReport, Interval, cavg, dcf, lid_error, load_vrcts_hyp, load_vrcts_ref, read_scr_scores
from speech.init_weights import init_weights
from speech.ledger.schema import BaselinePayload, BaselineRecipe, BaselineRecord, git_state, host_info, lineage_of, now_utc, seam_build_info, write_record
from speech.ledger.schema import CollarScore as LedgerCollar
from speech.weight_bridge import read_weight_vector, write_bin

# The committed canonical TOML for each arm (relative to the repo root). Phase 6 Task 8
# ships lid-features; Task 9 adds sad; Task 10 adds lid-phseq. Phase 10 Task 4 adds
# sad-v2 (spec S3.4).
_ARM_CONFIGS: dict[str, str] = {
    "lid-features": "configs/training/lre03_lid_features.toml",
    "sad": "configs/training/lre_sad.toml",
    "sad-v2": "configs/training/lre_sad_v2.toml",
    "lid-phseq": "configs/training/lre03_lid_phseq.toml",
}

# The two LID arms share the whole Twin/Mode-7 skeleton (algo-6, `.scr` -> lid_error/cavg
# scoring, both-nets-seeded/only-LID-trains); they differ ONLY in the File_Type and the
# corpus record derivation. This set gates the shared LID dispatch below.
_LID_ARMS: frozenset[str] = frozenset({"lid-features", "lid-phseq"})

# The two SAD arms (Phase 10 spec S3.4) share the ENTIRE skeleton -- same algo 3, same
# File_Type 0, same wav+xml listings, same DCF scoring path. They differ ONLY in which
# committed TOML `_ARM_CONFIGS` hands them, and that TOML differs only in the declared
# input width (v1's 2015-inherited 23 vs v2's honest 11; `lre_sad_v2.toml`'s header).
# Everything downstream self-sizes off the config, so `sad-v2` needs no code path of its
# own -- this set is what makes every `arm == "sad"` branch below cover both.
_SAD_ARMS: frozenset[str] = frozenset({"sad", "sad-v2"})

# The five DCF collar sizes (design spec S0): no-collar + 0.25/0.5/1.0/2.0 s. The 0.5 s
# collar is the reported headline (the T4 scorer pins all five vs the NIST perl oracle).
_DCF_COLLARS: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0, 2.0)

# The single whole-file-speech STM reference the cep listings point at (synthesized under
# out_dir). One line, huge end time; `load_ref_stm` clamps it to each file's own duration
# via `Segmentation::new(dur)` + `sanitize()`, so it marks EVERY cep file as one speech
# segment regardless of length -- correct for the pre-VAD'd cep regime. The `s ... s`
# first/third tokens satisfy `load_ref_stm`'s `second.starts_with(first)` speech gate, and
# the trailing two tokens satisfy its >=7-token line requirement.
_SPEECH_STM = "s 1 s 0.0 100000 speech seg\n"

# The 12 LRE03 languages the mapping/head cover (dialect fixed "non"), alphabetical -- the
# order `dataprep.lre.write_lre_mapping` + `drivers/test.py::_class_keys` both use, so the
# mapping's class ids, the engine's lang_index, the `.scr` columns, and the argmax refs all
# share one index base.
_LANGS: tuple[str, ...] = tuple(sorted(LRE03_LANGUAGES))
_CLASS_OF: dict[str, int] = {lang: i for i, lang in enumerate(_LANGS)}


# --------------------------------------------------------------------------------------- #
# Stratified subset + split (pure, seeded, deterministic -- unit-tested with no corpus)
# --------------------------------------------------------------------------------------- #


def _by_language(records: Sequence[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for rec in records:
        groups[rec["lang"]].append(rec)
    return dict(groups)


def _per_language_counts(groups: dict[str, list[dict[str, str]]], total: int) -> dict[str, int]:
    """Per-language quota for a PROPORTIONAL draw of `total` files -- each language gets
    `round(total * n_lang / n_all)`, floored at 1 (min 1/language present) and capped at its
    own count, so the draw preserves the source's per-language proportions AND keeps every
    language (all 12 LID classes) represented even for a small `total`. Used by
    `stratified_splits` for the valid/test quotas and for train when `n_train` is an explicit
    count (the `n_train=None` full-run path derives train's quota as the remainder instead)."""
    n_all = sum(len(v) for v in groups.values())
    return {lang: min(len(items), max(1, round(total * len(items) / n_all))) for lang, items in groups.items()}


def stratified_splits(
    records: Sequence[dict[str, str]],
    n_train: int | None,
    n_valid: int,
    n_test: int,
    seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    """Draw three DISJOINT, per-language PROPORTIONAL, seeded samples (train, valid, test)
    from `records` -- by explicit target COUNTS, so training cost (~`n_train`) is decoupled
    from held-out size (a big, cheap-to-score `n_test` gives a stable LID error even though
    training stays small). Each split's size is allocated proportionally across languages
    (min 1/language present), then carved from a per-language seeded shuffle so the three
    never overlap. With an explicit `n_train` the size priority is train->valid->test;
    `n_train=None` (the full-run launcher) instead carves the valid/test quotas FIRST and
    gives train the per-language REMAINDER, so the held-out splits stay filled rather than
    train swallowing every file. A language with too few files fills the splits in priority
    order and simply runs out -- no row is fabricated or shared across splits. Deterministic
    (per-language derived seed)."""
    groups = _by_language(records)
    q_valid = _per_language_counts(groups, n_valid)
    q_test = _per_language_counts(groups, n_test)
    if n_train is None:
        # Full-run path: valid/test take their full proportional quotas FIRST, train takes
        # the per-language REMAINDER -- so a None n_train FILLS the held-out splits instead of
        # swallowing every file (pre-fix that left valid/test empty, 40/0/0). Equivalent to
        # `q_train = n - q_valid - q_test`, floored at 0.
        q_train = {lang: max(0, len(items) - q_valid[lang] - q_test[lang]) for lang, items in groups.items()}
    else:
        q_train = _per_language_counts(groups, n_train)
    train: list[dict[str, str]] = []
    valid: list[dict[str, str]] = []
    test: list[dict[str, str]] = []
    for lang in sorted(groups):
        items = groups[lang]
        n = len(items)
        rng = np.random.default_rng([seed, _string_seed(lang)])
        order = [int(i) for i in rng.permutation(n)]
        # Priority: train first (it is the training data), then valid, then test -- each
        # bounded by what remains, so the three slices are disjoint by construction.
        cut1 = min(q_train[lang], n)
        cut2 = min(cut1 + q_valid[lang], n)
        cut3 = min(cut2 + q_test[lang], n)
        train.extend(items[i] for i in order[:cut1])
        valid.extend(items[i] for i in order[cut1:cut2])
        test.extend(items[i] for i in order[cut2:cut3])
    return train, valid, test


def _string_seed(s: str) -> int:
    """A stable 32-bit seed from a string (blake2b digest), so per-language shuffles are
    reproducible across runs/platforms without depending on Python's salted `hash()`."""
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=4).digest(), "big")


# --------------------------------------------------------------------------------------- #
# Listing derivation + config assembly
# --------------------------------------------------------------------------------------- #


def derive_lid_features_records(corpus_root: Path, ref_stm: Path) -> list[dict[str, str]]:
    """Build LID features listing records straight from the corpus tree (self-contained --
    no dependency on the local-only 2015 legacy listing). Globs
    `train/LID_Features/plp8f0mvsdd/LRE03/*.plp8f0mvsdd`, taking the language from the
    filename prefix (`xxx_0001.plp8f0mvsdd` -> `xxx`), keeping only the 12 known LRE03
    languages, and pointing every row's refseg at the synthesized whole-file-speech STM.
    Every path is absolute (glob results) and every file provably exists (it was globbed).
    Sorted by filename for determinism."""
    root = (corpus_root / "train" / "LID_Features" / "plp8f0mvsdd" / "LRE03").resolve()
    records: list[dict[str, str]] = []
    for path in sorted(root.glob("*.plp8f0mvsdd")):
        lang = path.name.split("_", 1)[0]
        if lang not in _CLASS_OF:
            continue
        records.append({"filename": str(path), "refseg": str(ref_stm), "lang": lang, "dial": "non", "weight": "1.0", "file_id": "1.0"})
    return records


# The CallFriend/LRE03 2-letter phSeq filename prefix -> the 3-letter LRE03 language code
# (the `_LANGS`/mapping base). The archive's `train/phSeq/*.file.phSeqbis` files are
# CallFriend-named (the shape `<2-letter-lang>_<id>...file.phSeqbis`), carrying the language
# ONLY as this 2-letter prefix -- the 2015 phSeq listings that would supply a 3-letter `lang` column
# localize to ZERO rows on this archive (they anchor on `eval/` but the files sit under
# `train/phSeq/`; Task 10 report), so the filename prefix is the language source. The 12
# codes biject onto the 12 LRE03 languages; "ma" = Mandarin -> "chi" (the LRE03 tag for
# Chinese), corroborated by the 3 highest-count prefixes (ma/en/sp) matching the 3
# highest-count cep languages (chi/eng/spa, design spec S0 counts).
_PHSEQ_PREFIX_TO_LANG: dict[str, str] = {
    "ar": "ara",
    "ma": "chi",
    "en": "eng",
    "fa": "fas",
    "fr": "fre",
    "ge": "ger",
    "hi": "hin",
    "ja": "jap",
    "ko": "kor",
    "sp": "spa",
    "ta": "tam",
    "vi": "vie",
}


def derive_lid_phseq_records(corpus_root: Path, ref_stm: Path) -> list[dict[str, str]]:
    """Build LID phonotactic listing records straight from the corpus phSeq tree
    (self-contained -- the 2015 phSeq listings localize to 0 rows on this archive, so there
    is nothing to localize; Task 10 report). Globs `train/phSeq/*.file.phSeqbis` (the
    WHOLE-utterance-per-line variant the 2015 Mode-7 flagship used -- one sequence per file,
    the phonotactic-LID premise), takes the language from the 2-letter filename prefix via
    `_PHSEQ_PREFIX_TO_LANG`, keeps only the 12 known LRE03 languages (the eval-style
    `lidXXXXX.file.phSeqbis` files have a numeric prefix not in the map, so they drop out),
    and points every refseg at the synthesized whole-file-speech STM. Every path is absolute
    (glob results) and provably exists (it was globbed). Sorted by filename for determinism.
    Mirrors `derive_lid_features_records`'s shape exactly -- only the tree + the
    filename->language rule differ."""
    root = (corpus_root / "train" / "phSeq").resolve()
    records: list[dict[str, str]] = []
    for path in sorted(root.glob("*.file.phSeqbis")):
        prefix = path.name.split("_", 1)[0]
        lang = _PHSEQ_PREFIX_TO_LANG.get(prefix)
        if lang is None or lang not in _CLASS_OF:
            continue
        records.append({"filename": str(path), "refseg": str(ref_stm), "lang": lang, "dial": "non", "weight": "1.0", "file_id": "1.0"})
    return records


def _records_from_localized(localized_csv: Path, ref_stm: Path) -> list[dict[str, str]]:
    """Read a `localize_listing` output back into records with the refseg REPLACED by the
    synthesized whole-file-speech STM (the 2015 listing's refseg is the empty shared
    `train/seg.xml`; LID training keys off the language label, not the seg reference)."""
    from speech.batching import read_listing

    out: list[dict[str, str]] = []
    for rec in read_listing(localized_csv):
        if rec["lang"] not in _CLASS_OF:
            continue
        out.append({**rec, "refseg": str(ref_stm), "dial": "non"})
    return out


def _write_listing_rows(path: Path, records: Sequence[dict[str, str]]) -> None:
    """Write records as the 6-field `;`-separated fileslisting the engine parses
    (`filename;refseg;lang;dial;weight;file_id;`)."""
    path.write_text("".join(f"{r['filename']};{r['refseg']};{r['lang']};{r['dial']};{r.get('weight', '1.0')};{r.get('file_id', '1.0')};\n" for r in records))


def write_lre_mapping_12(path: Path) -> None:
    """The 12-class `language2classmapping` (alphabetical, class 0..11, dialect `non`) --
    identical output to `dataprep.lre.write_lre_mapping`, inlined here against the same
    `_LANGS` order so the driver has no import-order coupling to that module's constant."""
    path.write_text("".join(f"{lang};non;{i}\n" for i, lang in enumerate(_LANGS)))


def assemble_flat_config(
    flat: dict[str, str],
    *,
    fileslisting: str,
    mapping: str,
    sad_seed: str,
    lanes: int,
    lid_seed: str | None = None,
    extra: dict[str, str] | None = None,
) -> dict[str, str]:
    """Overlay the runtime corpus/weight/lane keys onto the TOML-flattened base config
    (`speech_rs.load_toml_config` output). PURE -- no engine, no I/O -- so the unit tests
    exercise it with a synthetic flat dict. `dict(flat)` preserves the base key order
    (byte-stable), and each overwritten key already exists in the base so its position is
    kept. `numOuterThreads` = `lanes` records the N-lane fold width in the config the engine
    reads (deterministic-but-N-dependent, the R6-4a model; RESULTS.md fixes N per run).

    `lid_seed` is set ONLY for the algo-6 Twin (the LID features arm); a single-net algo-3
    SAD config has no `BLSTM_LID_weightsFile` key, and adding one would leave a dangling
    weight-file reference for a net that never exists -- so `lid_seed=None` skips it. `extra`
    overlays any further arm-specific keys (the SAD arm's `Audio_max_duration` cap); a key
    already in `flat` keeps its position, a new one appends."""
    cfg = dict(flat)
    cfg["fileslisting"] = fileslisting
    cfg["language2classmapping"] = mapping
    cfg["BLSTM_weightsFile"] = sad_seed
    if lid_seed is not None:
        cfg["BLSTM_LID_weightsFile"] = lid_seed
    cfg["numOuterThreads"] = str(lanes)
    for k, v in (extra or {}).items():
        cfg[k] = v
    return cfg


def _config_text(cfg: dict[str, str]) -> str:
    return "\n".join(f"{k} {v}" for k, v in cfg.items()) + "\n"


def cell_overlay(flat: dict[str, str], cell_type: str, direction: str) -> dict[str, str]:
    """The Phase 9 (spec S7.2) architecture overlay for the SAD arm: the port-only S6 keys
    `BLSTM_Cell_Type` / `BLSTM_Direction`, plus the TWO derived keys `forward` forces.

    EMPTY at the defaults (`lstm` / `bidirectional`), so a default run's config text is
    byte-identical to today's -- the whole point of the knob being additive. The `Mamba_*`
    (and, since phase 10, `Cfc_*`; since phase 11, `Transformer_*`) geometry keys are
    deliberately NOT written: their defaults live Rust-side (`blstm.rs::MambaParams::default`
    / `CFC_DEFAULT_BACKBONE_UNITS` / `TRANSFORMER_DEFAULT_{WINDOW,HEADS,D_FF}`) and Python
    reads the same defaults (`init_weights.MambaGeometry` /
    `config_bridge.CFC_DEFAULT_BACKBONE_UNITS` / `config_bridge.TRANSFORMER_DEFAULT_*`), so
    omitting them at default values keeps the config text minimal and the two sides agreeing
    by construction.

    THE FIFTH CELL NEEDS NO THIRD DERIVED KEY (phase-11 T9, spec S8): a transformer layer is
    just another `Layer` impl behind the same `CellLayer` enum (`nn/cells/mod.rs`'s dispatch,
    the phase-9/10 precedent), so both regimes this overlay already knows about keep working
    unmodified. Bidirectional: `feed_forward_backward_overlap` (the windowed driver
    `lre_sad_v2.toml`'s `frame_window 3.25` selects) calls `feed_forward_backward_plain` once
    per driver-level window with NO cell-type branch anywhere in that path -- it is exactly as
    generic over `Network<CellLayer>` as it was for sLSTM/Mamba/CfC before it, so a
    bidirectional transformer runs the windowed overlap regime like every other cell (verified
    by reading `feed_forward_backward_overlap`: the per-window call is `self.
    feed_forward_backward_plain(&block, ...)`, unconditional on cell type). This is a
    DIFFERENT "window" than the cell's own bounded ALiBi attention span (`Transformer_Window`,
    S1.1) -- the driver-level window bounds MEMORY over a long sequence by chunking it, the
    cell-level window bounds ATTENTION cost within whatever chunk it is handed; the two never
    interact because the cell always starts its own frame index at 0 for whatever span it is
    given, exactly like every other cell's per-call state reset. Forward: DERIVED KEY 2 below
    already forces the plain whole-sequence regime for every cell, transformer included, and
    the default `Transformer_Heads 4` divides both v1's and v2's hidden width 24 (S2's `H % A
    == 0` engine-side check), so no head-count override is needed either.

    DERIVED KEY 1 (the output MLP's width): `BlstmConfig::from_legacy` requires
    `OutputNeuronNb[0] == hidden_multiplier * lstm_neuron_nb[-1]` -- `2*hidden`
    bidirectional, `hidden` forward (there is no reverse half to concatenate). Writing only
    `BLSTM_Direction forward` would therefore produce a config the engine REFUSES to build,
    so the overlay resizes the output MLP's input layer to match.

    DERIVED KEY 2 (`BLSTM_window 0`, Phase 9 Task 6): a causal net runs the PLAIN
    whole-sequence regime. `lre_sad.toml` carries `frame_window 3.25`, which resolves
    `window_size > 0` and dispatches the WINDOWED drivers -- and a window boundary RESETS the
    recurrent state, so a windowed causal run is defined-but-pointless (spec S1.2) and the
    streaming session refuses it outright (S5.3). Forcing window 0 here is what makes
    `python -m speech.drivers.baseline sad --direction forward` train the regime the phase actually targets,
    and it is also the regime the f32 fast twin implements (`fast::cells::FastCausalNet`) --
    so the exact and fast paths stay comparable arm-for-arm. Bidirectional runs are
    UNTOUCHED (they keep `frame_window`'s windowed overlap).

    Everything else in the config (the DSP front-end, the cost law, the hidden widths) is
    untouched."""
    if cell_type not in ("lstm", "slstm", "mamba", "cfc", "transformer"):
        raise ValueError(f"unknown cell type {cell_type!r} (expected lstm, slstm, mamba, cfc or transformer)")
    if direction not in ("bidirectional", "forward"):
        raise ValueError(f"unknown direction {direction!r} (expected bidirectional or forward)")
    overlay: dict[str, str] = {}
    if cell_type != "lstm":
        overlay["BLSTM_Cell_Type"] = cell_type
    if direction != "bidirectional":
        overlay["BLSTM_Direction"] = direction
        hidden = [int(x) for x in flat["BLSTM_LSTMNeuronNb"].split(",")][-1]
        outn = [int(x) for x in flat["BLSTM_OutputNeuronNb"].split(",")]
        overlay["BLSTM_OutputNeuronNb"] = ",".join(str(v) for v in [hidden, *outn[1:]])
        overlay["BLSTM_window"] = "0"
    return overlay


def _generate_seed_packs(flat: dict[str, str], out_dir: Path, seed: int, init_scheme: str, forget_bias_one: bool) -> None:
    """Write valid seed weight packs the engine loads at construction (`BLSTM_weightsFile`/
    `BLSTM_LID_weightsFile`). `train_modern` re-inits from scratch and overrides these, but
    the engine still needs a loadable, correctly-sized pack to build the net -- so this
    mirrors the phase-5 twin smoke's committed-seed-pack pattern, generated at runtime here
    (no corpus-derived bytes) via the SAME `init_weights` the training loop uses. Both nets
    are drawn from one shared `Generator` in `[sad, lid]` order (deterministic).

    `init_scheme`/`forget_bias_one` MUST match the `ModernTrainParams` the training call
    actually uses (Task 8 review fix -- previously hardcoded "xavier"/True regardless): the
    seed pack doubles as the SCORED `score_init` baseline, so a mismatched scheme here would
    silently compare a trained net (e.g. a `--init-scheme he` run) against an init baseline
    drawn under a different scheme than the one training actually started from."""
    rng = np.random.default_rng(seed)
    for prefix, name in (("BLSTM", "sad_seed.bin"), ("BLSTM_LID", "lid_seed.bin")):
        pack = init_weights(nnet_spec(flat, prefix), rng, init_scheme, forget_bias_one)[0]  # type: ignore[arg-type]
        write_bin(pack.shape[0], 1, pack, out_dir / name)


def _generate_sad_seed_pack(flat: dict[str, str], out_dir: Path, seed: int, init_scheme: str, forget_bias_one: bool) -> None:
    """The single-net (algo-3 SAD) analogue of `_generate_seed_packs`: one `BLSTM` pack
    (`sad_seed.bin`) the engine loads at construction. `train_modern` re-inits from scratch
    and overrides it, but the engine still needs a loadable, correctly-sized pack to build
    the net; the pack ALSO doubles as the scored `score_init` baseline, so it MUST be drawn
    under the SAME `init_scheme`/`forget_bias_one` the training call uses (the Task 8 review
    lesson) -- otherwise the held-out DCF would compare the trained net against an init drawn
    under a different scheme than training actually started from. One draw, deterministic."""
    rng = np.random.default_rng(seed)
    pack = init_weights(nnet_spec(flat, "BLSTM"), rng, init_scheme, forget_bias_one)[0]  # type: ignore[arg-type]
    write_bin(pack.shape[0], 1, pack, out_dir / "sad_seed.bin")


def write_sad_mapping(path: Path) -> None:
    """A minimal `language2classmapping` for the SAD arm: `unk;unk;0`. SAD carries no
    per-file language target (`derive_sad_listings` writes `lang`/`dial` = `unk`), so the
    engine's mapping lookup returns class 0 for every file and the class is never used in
    the speech/non-speech decision. The engine's `Corpus::from_config` still requires the
    `language2classmapping` key to point at a real file, so this writes the one-line stub."""
    path.write_text("unk;unk;0\n")


# --------------------------------------------------------------------------------------- #
# Scoring (delegates to evaluate.py's .scr writer + the Task-5 LID metrics)
# --------------------------------------------------------------------------------------- #


def _score_packs_on_test(
    eval_state: RunState,
    sad_pack: Path,
    lid_pack: Path,
    score_dir: Path,
    test_records: Sequence[dict[str, str]],
) -> tuple[float | None, float | None, Path]:
    """Score one `[sad, lid]` weight-pack pair on the held-out test split end to end:
    `evaluate` -> per-file `.scr` -> `read_scr_scores` -> `lid_error` + `cavg`. `evaluate`
    expects the legacy `sad_weights.bin`/`lid_weights.bin` pack names in a checkpoint dir, so
    the two packs (a trained `best_<net>.bin` or the untrained `<net>_seed.bin`) are copied
    in under those names first -- letting the SAME scorer measure both the trained model and
    its own from-scratch init on the identical test set (the direction-safe improvement).

    Returns `(lid_error_pct, cavg, scores_dir)`; the metrics are `None` only if the test
    split produced no `.scr` files (a structurally empty held-out set)."""
    from speech.drivers.test import _class_keys, evaluate

    score_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(sad_pack, score_dir / "sad_weights.bin")
    shutil.copy(lid_pack, score_dir / "lid_weights.bin")
    # Nested under score_dir (already distinct per pass -- "score_trained" vs "score_init")
    # so the trained and init `.scr` outputs never share one directory (Task 8 review fix:
    # `evaluate`'s default output dir is fixed per `eval_state`, so two calls against the
    # SAME eval_state used to clobber each other's `.scr` files on disk).
    scores_dir = evaluate(eval_state, score_dir, scores_dir=score_dir / "scores")

    mapping_path = Path(eval_state.config_path).parent / eval_state.base_config["language2classmapping"]
    class_keys = _class_keys(mapping_path)
    scores, files = read_scr_scores(scores_dir, class_keys)
    if scores.shape[0] == 0:
        return None, None, scores_dir

    # Align each scored file (`.scr` stem = the cep basename) to its true class via the test
    # records, so `refs[i]` shares the alphabetical class base with the score columns.
    lang_by_name = {Path(r["filename"]).name: r["lang"] for r in test_records}
    refs = np.array([_CLASS_OF[lang_by_name[f]] for f in files], dtype=np.int_)
    return lid_error(scores, refs), cavg(scores, refs), scores_dir


# --------------------------------------------------------------------------------------- #
# SAD scoring: the engine's VRCTS hyp dumps vs the xml-derived refs -> the T4 DCF harness
# --------------------------------------------------------------------------------------- #


def _hyp_xml_for(wav_filename: str, dump_dir: Path) -> Path:
    """The VRCTS hypothesis xml the engine writes for `wav_filename` under `dump_dir`. The
    engine's dump target is the audio basename with its 4-char extension stripped, then
    `.xml` (`bag_of_processors.rs::base_from_last_slash` -> `strip_last_4`; the corpus is
    mono so there is no `_chan_<n>` fan-out). `[:-4]` mirrors `strip_last_4` exactly (drop
    the last 4 chars of `.wav`), so `zz_0000_a.MT1.mp1.wav -> zz_0000_a.MT1.mp1.xml`."""
    return dump_dir / (Path(wav_filename).name[:-4] + ".xml")


def _windowed_vrcts_ref(ref_xml: Path, end: float) -> list[Interval]:
    """The `.part.xml` reference (`load_vrcts_ref`, kinds S/NS) clipped to `[0, end]` -- the
    span the engine's capped-audio hypothesis actually covers (`end` = the hyp's own
    `sigdur`). The engine windows its OWN reference on `_AudioDuration` identically (Task 2b),
    so this keeps the DCF scoring span aligned with the hyp. `load_vrcts_ref` yields a
    contiguous run from 0, so clipping preserves contiguity; a trailing `NS` pad guarantees
    the ref ends EXACTLY at `end` (so the pooled concatenation below has no inter-file gap),
    and an empty clip (a reference whose first segment starts past `end`) becomes one whole
    `NS` span."""
    out: list[Interval] = []
    for s, e, k in load_vrcts_ref(ref_xml):
        if s >= end:
            break
        out.append((s, min(e, end), k))
    if not out:
        return [(0.0, end, "NS")]
    if out[-1][1] < end:
        out.append((out[-1][1], end, "NS"))
    return out


def _pool_dcf(pairs: Sequence[tuple[list[Interval], list[Interval]]], collars: Sequence[float]) -> DcfReport:
    """Pool per-file (ref, hyp) pairs into ONE DCF over a concatenated timeline: each file's
    intervals are offset by the running sum of the preceding files' spans, then scored with a
    single `dcf` call (the duration-weighted pooled DCF -- the single-number analogue of the
    LID arm's pooled argmax error). Contiguity holds because every `_windowed_vrcts_ref` +
    `load_vrcts_hyp` pair spans exactly `[0, span]` and starts at 0, so file i+1 begins where
    file i ended -- no gap for `dcf`'s reference walk to reject. The only cross-file artifact
    is that a collar may straddle a file boundary, a negligible fraction of a multi-file
    timeline and identical across the trained/init passes (so the direction is unaffected)."""
    all_ref: list[Interval] = []
    all_hyp: list[Interval] = []
    offset = 0.0
    for ref, hyp in pairs:
        span = hyp[-1][1]  # == the file's capped audio span (the ref is windowed to it)
        all_ref += [(s + offset, e + offset, k) for s, e, k in ref]
        all_hyp += [(s + offset, e + offset, k) for s, e, k in hyp]
        offset += span
    return dcf(all_ref, all_hyp, collars)


def _score_sad_pack_on_test(
    base_cfg: dict[str, str],
    workdir: Path,
    pack_path: Path,
    dump_dir: Path,
    test_records: Sequence[dict[str, str]],
    test_listing_name: str,
) -> tuple[DcfReport | None, Path]:
    """Score one SAD weight pack on the held-out test split end to end: run the engine
    (scored `-m`) over the test listing with `Dump_Directory` set so it writes one VRCTS
    hypothesis xml per file, then pool `dcf` over (`.part.xml` ref windowed to the hyp span,
    engine hyp). On `Inference_Path exact` (the default), the pack (a trained `best_sad.bin`
    or the untrained `sad_seed.bin`) is loaded via `set_weights` -- the SAME scorer measures
    both the trained model and its own from-scratch init on the identical test set (the
    direction-safe DCF improvement). On `Inference_Path fast`, `set_weights` is SKIPPED
    (bag_of_processors.rs T6b: it now bails loudly on a fast conf instead of the old silent
    no-op) -- `base_cfg["BLSTM_weightsFile"]` must already point at `pack_path` for that case
    (the caller's responsibility; see `tests/pyo3/test_phase7_parity.py::_score_sad`).

    Returns `(DcfReport | None, dump_dir)`; `None` only if no hyp xml was produced (a
    structurally empty test set). Mirrors `drivers.test.evaluate`'s engine-driving shape
    (chdir into `workdir`, backprop-OFF forward-only config, `set_weights` then `run`)."""
    import speech_rs  # local: the pyo3 module is only needed on the engine path

    dump_dir.mkdir(parents=True, exist_ok=True)
    cfg = dict(base_cfg)
    cfg["fileslisting"] = test_listing_name
    cfg["Dump_Directory"] = str(dump_dir.resolve())
    cfg["BLSTM_BackPropagationActivated"] = "false"
    cfg["Neural_Networks_BackPropagation_Epochs"] = "0"
    eval_config = workdir / f"_sad_eval_{dump_dir.name}.config"
    eval_config.write_text(_config_text(cfg))

    prev = Path.cwd()
    os.chdir(workdir)
    try:
        engine = speech_rs.Engine([eval_config.name], "-m")
        # `Inference_Path fast` processors load weights ONLY at construction, from the
        # config's own BLSTM_weightsFile key; `set_weights` now bails loudly on them
        # (bag_of_processors.rs T6b) instead of the old silent no-op. Skip the call on
        # fast: `pack_path` is already the config-time-injected pack there (callers point
        # BLSTM_weightsFile at it before building this config), so the call is redundant
        # on fast and load-bearing only on exact (whose BLSTM_weightsFile is the arm's seed
        # pack, not `pack_path`).
        if cfg.get("Inference_Path", "exact") != "fast":
            engine.set_weights(0, [list(read_weight_vector(pack_path))])
        engine.run()
    finally:
        os.chdir(prev)

    pairs: list[tuple[list[Interval], list[Interval]]] = []
    for rec in test_records:
        hyp_xml = _hyp_xml_for(rec["filename"], dump_dir)
        if not hyp_xml.is_file():
            continue
        hyp = load_vrcts_hyp(hyp_xml)
        if not hyp:
            continue
        span = hyp[-1][1]
        ref = _windowed_vrcts_ref(Path(rec["refseg"]), span)
        pairs.append((ref, hyp))
    if not pairs:
        return None, dump_dir
    return _pool_dcf(pairs, _DCF_COLLARS), dump_dir


# --------------------------------------------------------------------------------------- #
# Run metadata
# --------------------------------------------------------------------------------------- #


def write_run_metadata(out_dir: Path, meta: dict[str, object]) -> Path:
    """Record the run's reproducibility metadata (seed, lanes, subset spec, config hash,
    split counts, arm) as `out_dir/run_metadata.json`. Returns the path. Pure I/O -- unit
    tested directly."""
    path = out_dir / "run_metadata.json"
    path.write_text(json.dumps(meta, indent=2, sort_keys=True))
    return path


def _config_hash(text: str) -> str:
    return hashlib.blake2b(text.encode(), digest_size=8).hexdigest()


# --------------------------------------------------------------------------------------- #
# The arm launcher
# --------------------------------------------------------------------------------------- #


@dataclass
class BaselineResult:
    """The outcome of one `run_baseline` call -- the metrics + the artefact locations the
    gate asserts on and the launcher logs."""

    arm: str
    out_dir: Path
    checkpoint_dir: Path
    metadata_path: Path
    epochs_run: int
    best_epoch: int
    best_val_cost: float
    first_val_cost: float
    val_costs: list[float] = field(default_factory=list)
    train_costs: list[float] = field(default_factory=list)
    lid_error: float | None = None
    cavg: float | None = None
    init_lid_error: float | None = None  # untrained-init held-out error (scored iff score_init)
    init_cavg: float | None = None
    scores_dir: Path | None = None
    # SAD arm (arm="sad"): the pooled held-out DCF report (per collar) for the trained model
    # and, when `score_init`, its own from-scratch init on the identical test set.
    dcf: DcfReport | None = None
    init_dcf: DcfReport | None = None
    n_train: int = 0
    n_valid: int = 0
    n_test: int = 0
    wall_s: float = 0.0

    def to_record(self, source: Literal["launcher", "gate"], test: str | None = None) -> BaselineRecord:
        """The run as a ledger record (issue #20): the recipe read back from `run_metadata.json`
        (the values that actually ran, after any dry-run rewrite), the metrics from this result,
        the provenance stamped now. `run_baseline` writes the `launcher` record into the run
        directory; a subset gate overwrites it with `source="gate"` and its pytest node id."""
        meta = json.loads(self.metadata_path.read_text())
        recipe = BaselineRecipe(
            arm=meta["arm"],
            lineage=lineage_of(meta["config_toml"]),
            cell=meta["cell_type"],
            direction=meta["direction"],
            subset=meta["subset"],
            valid_size=meta["valid_size"],
            test_size=meta["test_size"],
            audio_max_duration=meta["audio_max_duration"],
            epochs=meta["epochs"],
            steps_per_epoch=meta["steps_per_epoch"],
            patience=meta["patience"],
            minibatch=meta["minibatch"],
            init_scheme=meta["init_scheme"],
            seed=meta["seed"],
            lanes=meta["lanes"],
        )
        payload = BaselinePayload(
            source=source,
            test=test,
            config_name=meta["config_toml"],
            config_hash=meta["config_hash"],
            n_train=self.n_train,
            n_valid=self.n_valid,
            n_test=self.n_test,
            val_metric=meta["val_metric"],
            epochs_run=self.epochs_run,
            best_epoch=self.best_epoch,
            wall_s=self.wall_s,
            first_val_cost=_finite(self.first_val_cost),
            best_val_cost=_finite(self.best_val_cost),
            first_train_cost=_finite(self.train_costs[0]) if self.train_costs else None,
            last_train_cost=_finite(self.train_costs[-1]) if self.train_costs else None,
            dcf=_ledger_collars(self.dcf),
            init_dcf=_ledger_collars(self.init_dcf),
            lid_error=_finite(self.lid_error),
            cavg=_finite(self.cavg),
            init_lid_error=_finite(self.init_lid_error),
            init_cavg=_finite(self.init_cavg),
        )
        sha, dirty = git_state()
        return BaselineRecord(recorded_at=now_utc(), git_sha=sha, git_dirty=dirty, build=seam_build_info(), host=host_info(), recipe=recipe, payload=payload)


def _finite(x: float | None) -> float | None:
    return None if x is None or not np.isfinite(x) else float(x)


def _ledger_collars(rep: DcfReport | None) -> list[LedgerCollar] | None:
    return None if rep is None else [LedgerCollar(collar=s.collar, dcf=s.dcf, pmiss=s.pmiss, pfa=s.pfa) for s in rep.scores]


def _prepare_sad_listings(
    corpus_root: Path,
    out_dir: Path,
    seed: int,
    subset: int | None,
    valid_size: int,
    test_size: int,
    console: Console,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]], str, str, str, str]:
    """SAD listing prep: `derive_sad_listings` (the seeded 70/15/15 split of the wav/xml
    pairs) then a deterministic first-N SUBSET of each split (the split is already a seeded
    permutation, so `[:subset]` is a reproducible sample). Writes the subset train/valid/test
    `.flst` + a minimal `unk` mapping under out_dir; returns the records + the four listing
    filenames. `subset=None` trains on the full train split (the launcher). The refs are the
    corpus `.part.xml` paths the derived listings already carry (absolute)."""
    split = derive_sad_listings(Path(corpus_root), out_dir, seed)
    console.log(
        f"SAD corpus: {split.n_total} wav/xml pairs ({split.n_orphan_wav} orphan wav, "
        f"{split.n_orphan_xml} orphan xml); base split {split.n_train}/{split.n_valid}/{split.n_test}"
    )
    train_all = read_listing(split.train_path)
    valid_all = read_listing(split.valid_path)
    test_all = read_listing(split.test_path)
    train_rec = train_all if subset is None else train_all[:subset]
    valid_rec = valid_all[:valid_size]
    test_rec = test_all[:test_size]

    train_name, valid_name, test_name, mapping_name = "sad_train_subset.flst", "sad_valid_subset.flst", "sad_test_subset.flst", "sad_mapping.csv"
    _write_listing_rows(out_dir / train_name, train_rec)
    _write_listing_rows(out_dir / valid_name, valid_rec)
    _write_listing_rows(out_dir / test_name, test_rec)
    write_sad_mapping(out_dir / mapping_name)
    return train_rec, valid_rec, test_rec, train_name, valid_name, test_name, mapping_name


def run_baseline(
    arm: str,
    corpus_root: Path,
    out_dir: Path,
    *,
    resume: bool = False,
    lanes: int = 1,
    subset: int | None = None,
    dry_run: bool = False,
    seed: int = 0,
    epochs: int = 40,
    patience: int = 6,
    steps_per_epoch: int = 8,
    init_scheme: str = "xavier",
    lre_listing: Path | None = None,
    valid_size: int = 12,
    test_size: int = 48,
    minibatch: int = 0,
    score_init: bool = False,
    audio_max_duration: float | None = None,
    cell_type: str = "lstm",
    direction: str = "bidirectional",
    console: Console | None = None,
    _train_fn: Callable[..., object] | None = None,
) -> BaselineResult:
    """Run one baseline training arm end to end: prepare the corpus listings, assemble the
    config, seeded from-scratch init (or resume), `train_modern` with the moving NNCostSeg
    validation signal, then score a held-out slice (LID: `.scr` -> `lid_error` + `cavg`).

    `subset`: the TRAIN sample size (per-language proportional for LID, a seeded first-N of
    the SAD split for SAD) -- the CI-gate regime (< 10 min); `None` trains on the whole
    corpus (the launcher). `valid_size`/`test_size` are SEPARATE, disjoint held-out sizes --
    decoupled from `subset` so a big, cheap-to-score test set gives a stable held-out metric
    while training stays small. `dry_run`: a 1-step smoke (a tiny train subset, 1 epoch, 1
    step, and for SAD a short audio cap) still scored end to end. `resume`: continue from
    `out_dir/checkpoint`'s `last_*.bin`. `lanes`: the engine's `numOuterThreads` fold width
    (recorded in metadata; N=1 is the deterministic parity mode). `lre_listing` (LID only):
    localize this 2015 listing instead of deriving from the corpus tree. `audio_max_duration`
    (SAD only): override `Audio_max_duration` (the corpus wavs are 576-1800 s, median ~600 s; a cap
    bounds the run and the held-out DCF windows the reference to the same span). `_train_fn` injects a
    stub `train_modern` for tests.

    The arm dispatch differs in three places -- the listings (LID globs cep + a synthesized
    speech STM; SAD derives wav/xml pairs with the corpus `.part.xml` refs), the seed packs
    (LID a `[sad, lid]` Twin pair; SAD a single `[sad]` net), and the held-out scoring (LID
    `.scr` -> `lid_error` + `cavg`; SAD VRCTS hyps -> pooled `dcf`); the split/params/train/
    metadata skeleton is shared.

    `cell_type`/`direction` (Phase 9, spec S7.2): overlay the port-only S6 architecture keys
    onto the arm config (`cell_overlay`) and, since seeding reads the architecture back out of
    that config via `nnet_spec`, route the from-scratch init through the matching per-cell
    builder automatically -- both the engine-construction seed pack here and `train_modern`'s
    own re-init. Defaults are today's BLSTM arm, byte-identical."""
    if arm not in _ARM_CONFIGS:
        raise ValueError(f"unknown arm {arm!r}; known arms: {sorted(_ARM_CONFIGS)}")
    if arm not in _LID_ARMS and arm not in _SAD_ARMS:
        raise NotImplementedError(f"arm {arm!r} is not wired (known: {sorted(_ARM_CONFIGS)})")
    # The knobs target the `BLSTM_` net. On a Twin arm that net is the FROZEN SAD gate (Mode 7
    # never runs it, so it stays byte-exactly at its seed), which makes a cell/direction swap
    # there a silent no-op on everything that actually trains -- bail loudly instead. The LID
    # net's own `BLSTM_LID_Cell_Type`/`_Direction` wiring is Task 5's.
    if arm in _LID_ARMS and (cell_type != "lstm" or direction != "bidirectional"):
        raise ValueError(f"--cell-type/--direction are SAD-arm knobs; arm {arm!r} trains only its LID net (Task 5 wires BLSTM_LID_*)")

    console = console or Console()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    # __file__ = <repo>/src/python/speech/drivers/baseline.py -> parents[4] = <repo>.
    repo_root = Path(__file__).resolve().parents[4]
    toml_path = repo_root / _ARM_CONFIGS[arm]

    # dry_run overrides -- arm-aware: SAD wav ingestion is slower per file than cep, so the
    # SAD smoke draws a smaller subset AND caps the audio short (else a 1-step smoke over
    # long CallFriend recordings would blow the "fast" promise).
    if dry_run:
        if arm in _SAD_ARMS:
            subset, epochs, steps_per_epoch, patience, test_size, valid_size = (subset or 6), 1, 1, 99, min(test_size, 6), min(valid_size, 4)
            audio_max_duration = audio_max_duration if audio_max_duration is not None else 10.0
        else:
            subset, epochs, steps_per_epoch, patience, test_size, valid_size = (subset or 24), 1, 1, 99, min(test_size, 24), min(valid_size, 12)

    t0 = time.time()
    console.log(f"[bold]baseline {arm}[/bold]: corpus={corpus_root} out={out_dir} subset={subset} lanes={lanes} seed={seed} dry_run={dry_run}")

    # --- 1. listings + mapping (+ reference) synthesized under out_dir --------------------
    if arm in _SAD_ARMS:
        train_rec, valid_rec, test_rec, train_name, valid_name, test_name, mapping_name = _prepare_sad_listings(
            corpus_root, out_dir, seed, subset, valid_size, test_size, console
        )
        n_classes = 1
    else:
        ref_stm = out_dir / "ref_speech.stm"
        ref_stm.write_text(_SPEECH_STM)
        # Both LID arms share this block; only the corpus record derivation differs
        # (lid-phseq globs the phSeq tree, lid-features the cep tree). The `lre_listing`
        # override localizes a 2015 listing for either arm (unused on this archive for
        # phSeq -- those listings resolve 0 rows -- but kept symmetric with lid-features).
        if lre_listing is not None:
            localized = out_dir / "localized_lre.csv"
            rep = localize_listing(Path(lre_listing), corpus_root, localized)
            console.log(f"localized {rep.rows_found}/{rep.rows_total} rows ({rep.rows_missing} missing)")
            records = _records_from_localized(localized, ref_stm)
        elif arm == "lid-phseq":
            records = derive_lid_phseq_records(corpus_root, ref_stm)
        else:
            records = derive_lid_features_records(corpus_root, ref_stm)
        if not records:
            hint = "train/phSeq/*.file.phSeqbis" if arm == "lid-phseq" else "train/LID_Features/plp8f0mvsdd/LRE03/*.plp8f0mvsdd"
            raise RuntimeError(f"no LID records found under {corpus_root} (expected {hint})")
        console.log(f"corpus records: {len(records)}")
        train_rec, valid_rec, test_rec = stratified_splits(records, subset, valid_size, test_size, seed)
        stem = "lre03_lid_phseq" if arm == "lid-phseq" else "lre03_lid_features"
        train_name, valid_name, test_name = f"{stem}_train.flst", f"{stem}_valid.flst", f"{stem}_test.flst"
        mapping_name = "language2classmapping_lre12.csv"
        _write_listing_rows(out_dir / train_name, train_rec)
        _write_listing_rows(out_dir / valid_name, valid_rec)
        _write_listing_rows(out_dir / test_name, test_rec)
        write_lre_mapping_12(out_dir / mapping_name)
        n_classes = len(_LANGS)
    console.log(f"split: train={len(train_rec)} valid={len(valid_rec)} test={len(test_rec)}")
    if not train_rec:
        raise ValueError(f"empty train split (valid={len(valid_rec)} test={len(test_rec)}): lower --valid-size/--test-size or raise --subset")

    # --- 2. training params + config assembly + seed packs -------------------------------
    # Built here (not down in step 4) so the seed-pack init is handed the SAME
    # init_scheme/forget_bias_one the training call below will use (Task 8 review fix).
    params = ModernTrainParams(
        epochs=epochs,
        patience=patience,
        steps_per_epoch=steps_per_epoch,
        valid_listing=valid_name if valid_rec else None,
        val_metric="nn_cost_seg",
        minibatch=minibatch,
        nb_classes=n_classes,
        # NOT multilingual: that legacy layout reads class values 1..nb_classes-1 as targets and
        # sends class nb_classes-1 to the aggregate slot's never-read `.index`, so the 0..11
        # LID mapping would never draw class 11 into a batch.
        init_scheme=init_scheme,  # type: ignore[arg-type]
        init_seed=seed,
        resume_from=str(out_dir / "checkpoint") if resume else None,
    )

    import speech_rs  # local: the pyo3 module is only needed on the engine path

    flat = {k: str(v) for k, v in speech_rs.load_toml_config(str(toml_path)).items()}
    # EMPTY at the default lstm/bidirectional knobs, so `extra` -- and therefore the whole
    # config text -- is byte-identical to a pre-phase-9 run.
    extra = dict(cell_overlay(flat, cell_type, direction))
    if audio_max_duration is not None:
        extra["Audio_max_duration"] = str(audio_max_duration)
    if arm in _SAD_ARMS:
        cfg = assemble_flat_config(flat, fileslisting=train_name, mapping=mapping_name, sad_seed="sad_seed.bin", lanes=lanes, extra=extra)
        _generate_sad_seed_pack(cfg, out_dir, seed, params.init_scheme, params.forget_bias_one)
    else:
        cfg = assemble_flat_config(
            flat, fileslisting=train_name, mapping=mapping_name, sad_seed="sad_seed.bin", lid_seed="lid_seed.bin", lanes=lanes, extra=extra
        )
        _generate_seed_packs(cfg, out_dir, seed, params.init_scheme, params.forget_bias_one)
    cfg_text = _config_text(cfg)
    base_config = out_dir / "base.config"
    base_config.write_text(cfg_text)

    # --- 3. run metadata -----------------------------------------------------------------
    metadata_path = write_run_metadata(
        out_dir,
        {
            "arm": arm,
            "seed": seed,
            "lanes": lanes,
            "subset": subset,
            "dry_run": dry_run,
            "epochs": epochs,
            "patience": patience,
            "steps_per_epoch": steps_per_epoch,
            "minibatch": minibatch,
            "valid_size": valid_size,
            "test_size": test_size,
            "score_init": score_init,
            "init_scheme": init_scheme,
            "val_metric": "nn_cost_seg",
            "audio_max_duration": audio_max_duration,
            "cell_type": cell_type,
            "direction": direction,
            "config_hash": _config_hash(cfg_text),
            "config_toml": str(toml_path.relative_to(repo_root)),
            "corpus_root": str(corpus_root),
            "n_train": len(train_rec),
            "n_valid": len(valid_rec),
            "n_test": len(test_rec),
            "resume": resume,
        },
    )

    # --- 4. train (from scratch or resume) -----------------------------------------------
    state = RunState.from_config(base_config, out_dir)
    train = _train_fn if _train_fn is not None else train_modern
    console.log(f"training: {epochs} epochs x {steps_per_epoch} steps (patience {patience})")
    res = train(state, seed, params)  # type: ignore[operator]
    val_costs = [r.val_cost for r in res.history]  # type: ignore[attr-defined]
    train_costs = [r.train_cost for r in res.history]  # type: ignore[attr-defined]
    console.log(f"trained {res.epochs_run} epochs; best_epoch={res.best_epoch} best_val={res.best_val_cost:.5f}")  # type: ignore[attr-defined]

    # --- 5. score the held-out split (trained; optionally the untrained init too) ---------
    lid_err: float | None = None
    cavg_val: float | None = None
    init_err: float | None = None
    init_cavg_val: float | None = None
    dcf_rep: DcfReport | None = None
    init_dcf_rep: DcfReport | None = None
    scores_dir: Path | None = None
    ckpt = Path(res.checkpoint_dir)  # type: ignore[attr-defined]
    if test_rec and arm in _SAD_ARMS:
        dcf_rep, scores_dir = _score_sad_pack_on_test(cfg, out_dir, ckpt / "best_sad.bin", out_dir / "score_trained", test_rec, test_name)
        if dcf_rep is not None:
            c = dcf_rep.by_collar(0.5)
            console.log(f"held-out DCF@0.5={c.dcf:.4f} (Pmiss={c.pmiss:.4f} Pfa={c.pfa:.4f})")
        else:
            console.log("held-out: no VRCTS hyps produced (empty test split)")
        if score_init:
            init_dcf_rep, _ = _score_sad_pack_on_test(cfg, out_dir, out_dir / "sad_seed.bin", out_dir / "score_init", test_rec, test_name)
            if init_dcf_rep is not None and dcf_rep is not None:
                gain = init_dcf_rep.by_collar(0.5).dcf - dcf_rep.by_collar(0.5).dcf
                console.log(f"init baseline: DCF@0.5={init_dcf_rep.by_collar(0.5).dcf:.4f} (improvement {gain:+.4f})")
    elif test_rec:
        eval_base = out_dir / "eval_base.config"
        eval_cfg = dict(cfg)
        eval_cfg["fileslisting"] = test_name
        eval_base.write_text(_config_text(eval_cfg))
        eval_state = RunState.from_config(eval_base, out_dir)
        lid_err, cavg_val, scores_dir = _score_packs_on_test(eval_state, ckpt / "best_sad.bin", ckpt / "best_lid.bin", out_dir / "score_trained", test_rec)
        chance = 100.0 * (1.0 - 1.0 / len(_LANGS))
        if lid_err is not None and cavg_val is not None:
            console.log(f"held-out: lid_error={lid_err:.2f}% (chance {chance:.2f}%) cavg={cavg_val:.4f}")
        else:
            console.log("held-out: no .scr scores produced (empty test split)")
        if score_init:
            init_err, init_cavg_val, _ = _score_packs_on_test(eval_state, out_dir / "sad_seed.bin", out_dir / "lid_seed.bin", out_dir / "score_init", test_rec)
            if init_err is not None and lid_err is not None:
                console.log(f"init baseline: lid_error={init_err:.2f}% (improvement {init_err - lid_err:+.2f}pt)")

    wall_s = time.time() - t0
    console.log(f"[green]done[/green] in {wall_s:.1f}s")
    result = BaselineResult(
        arm=arm,
        out_dir=out_dir,
        checkpoint_dir=ckpt,
        metadata_path=metadata_path,
        epochs_run=res.epochs_run,  # type: ignore[attr-defined]
        best_epoch=res.best_epoch,  # type: ignore[attr-defined]
        best_val_cost=res.best_val_cost,  # type: ignore[attr-defined]
        first_val_cost=val_costs[0] if val_costs else float("nan"),
        val_costs=val_costs,
        train_costs=train_costs,
        lid_error=lid_err,
        cavg=cavg_val,
        init_lid_error=init_err,
        init_cavg=init_cavg_val,
        dcf=dcf_rep,
        init_dcf=init_dcf_rep,
        scores_dir=scores_dir,
        n_train=len(train_rec),
        n_valid=len(valid_rec),
        n_test=len(test_rec),
        wall_s=wall_s,
    )
    # The promotable record (issue #20): `python -m speech.ledger add <out_dir>/record.json`.
    write_record(result.to_record("launcher"), out_dir / "record.json")
    return result


# --------------------------------------------------------------------------------------- #
# CLI: `python -m speech.drivers.baseline lid-features --corpus-root ... --out-dir ...`
# --------------------------------------------------------------------------------------- #


def _count(text: str) -> int:
    n = int(text)
    if n < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0, got {n}")
    return n


def build_parser() -> argparse.ArgumentParser:
    """The `baseline <arm>` argument parser -- also mounted as the `baseline` subcommand in
    `speech.cli`. Unit-tested for arg wiring (no run)."""
    parser = argparse.ArgumentParser(description="Phase 6 from-scratch baseline training arms")
    parser.add_argument("arm", choices=sorted(_ARM_CONFIGS), help="the training arm")
    parser.add_argument("--corpus-root", type=Path, required=True, help="the LRE03/07 corpus root (data/LRE03-LRE07)")
    parser.add_argument("--out-dir", type=Path, required=True, help="run directory for listings/config/checkpoints/scores")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lanes", type=int, default=1, help="engine numOuterThreads fold width (N=1 is the parity mode)")
    parser.add_argument("--subset", type=_count, default=None, help="stratified subset size (omit for the full corpus)")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--steps-per-epoch", type=int, default=8)
    parser.add_argument("--init-scheme", choices=("xavier", "he"), default="xavier")
    parser.add_argument(
        "--valid-size",
        type=_count,
        default=12,
        help="held-out validation slice (files; LID allocates per language, min 1 each); a SAD 0 validates on the train listing",
    )
    parser.add_argument("--test-size", type=_count, default=48, help="held-out test slice (files; LID: min 1 per language) the final metric is scored on")
    parser.add_argument(
        "--minibatch", type=_count, default=0, help="training files drawn per SMORMS3 step, class-stratified round-robin (0 = the whole train listing)"
    )
    parser.add_argument("--score-init", action="store_true", help="also score the from-scratch seed pack on the test slice (the init-baseline row)")
    parser.add_argument(
        "--cell-type",
        choices=("lstm", "slstm", "mamba", "cfc", "transformer"),
        default="lstm",
        help="SAD arm: recurrent cell (spec S6 + phase-10 S1 + phase-11 S1/S2; default = today's peephole BLSTM)",
    )
    parser.add_argument(
        "--direction",
        choices=("bidirectional", "forward"),
        default="bidirectional",
        help="SAD arm: forward drops the backward stack (and halves the output MLP's input width)",
    )
    parser.add_argument("--lre-listing", type=Path, default=None, help="LID arm: localize this 2015 listing instead of deriving from the corpus tree")
    parser.add_argument(
        "--audio-max-duration", type=float, default=None, help="SAD arm: cap Audio_max_duration (s); the corpus wavs are 576-1800 s (median ~600 s)"
    )
    parser.add_argument("--resume", action="store_true", help="continue from out_dir/checkpoint")
    parser.add_argument("--dry-run", action="store_true", help="1-step smoke: tiny subset, 1 epoch, 1 step, still scored")
    return parser


def run_baseline_from_args(args: argparse.Namespace) -> BaselineResult:
    """Forward a `build_parser()` namespace to `run_baseline`, every dest bound by name, so an
    entry point cannot drop a flag silently (the `speech.cli` mount dropped `--cell-type`/
    `--direction` for a phase, issue #28); a dest that is not a `run_baseline` parameter fails
    here with a TypeError, and a unit test pins the alignment both ways."""
    return run_baseline(**vars(args))


def main(argv: list[str] | None = None) -> int:
    """CLI entry: parse args -> `run_baseline`. Returns a process exit code."""
    run_baseline_from_args(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
