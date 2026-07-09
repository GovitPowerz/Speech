"""Hard-example mini-batch scheduler and listing parsing.

Ported from legacy MATLAB: CreateBatches.m, GetNewBatch.m, getCases.m (the private
`getWorstAndBest` subfunction), processListing.m. See design spec section 6.

`create_batches` is the ONLY RNG consumer here: it takes an injected
`rng: np.random.Generator` (or any object exposing `.permutation(array)`) instead of
reading MATLAB's global PRNG state -- a deliberate determinism deviation (the legacy's
`randperm` calls are otherwise uncontrolled). `get_new_batch` is PURE integer-cursor
rotation with no RNG at all, matching `GetNewBatch.m` exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

# --------------------------------------------------------------------------- #
# read_listing: mirrors engine/corpus.rs's Corpus::from_config LISTING-FILE branch
# (Corpus.cpp:59-100) field-for-field -- filename/refseg/lang/dial/weight/file_id with
# the SAME sibling-not-nested `tokens.len() > N` token-count gates and defaults. This is
# the raw per-line record parse only (no language->class mapping, no corpus-wide
# aggregation); `create_batches`/`get_new_batch` consume already-numeric `files_values`
# built elsewhere from these records plus a mapping file.
# --------------------------------------------------------------------------- #


def _splitstr(line: str, delim: str) -> list[str]:
    """Legacy `splitstr(line, delim)` (String.hpp:110-114): repeated
    `getline(ss, item, delim)`. A delimiter as the LAST character is consumed to end the
    preceding field but yields no further (empty) token -- drops AT MOST ONE trailing
    empty field: "a;b;" -> ["a","b"], but "a;b;;" -> ["a","b",""]. An empty line yields
    zero tokens; a line with no delimiter yields the whole line as one token."""
    if line == "":
        return []
    out = line.split(delim)
    if line.endswith(delim):
        out.pop()
    return out


def _iss_extract_double(s: str) -> float | None:
    """`istringstream >> double` (Corpus.cpp:82-84 weight parse): accumulates every char
    from the float atom set (digits, a-f/A-F, x/X, e/E, p/P, at most one '.', a sign only
    first or right after an exponent char), then requires the WHOLE accumulated string to
    parse -- "0.5abc" fails (a/b/c accumulate, unparseable), "0.5z" extracts 0.5."""
    acc: list[str] = []
    seen_dot = False
    for c in s.lstrip():
        if c.isdigit() or c in "abcdefABCDEFxXpP":
            ok = True
        elif c in "+-":
            ok = not acc or acc[-1] in "eEpP"
        elif c == ".":
            ok = not seen_dot
        else:
            ok = False
        if not ok:
            break
        if c == ".":
            seen_dot = True
        acc.append(c)
    try:
        v = float("".join(acc))
    except ValueError:
        return None
    return v if np.isfinite(v) else None


def _iss_extract_int(s: str) -> int | None:
    """`istringstream >> int` (Corpus.cpp:87-89 fileId parse): digits plus a sign in
    first position only; whole-string parse, no partial extraction."""
    acc: list[str] = []
    for c in s.lstrip():
        if c.isdigit():
            ok = True
        elif c in "+-":
            ok = not acc
        else:
            ok = False
        if not ok:
            break
        acc.append(c)
    try:
        return int("".join(acc))
    except ValueError:
        return None


def read_listing(path: Path) -> list[dict[str, str]]:
    """Parse a ';'-separated fileslisting CSV into records (Phase 0 stub, completed here).

    Each record: `filename`, `refseg` (default ""), `lang`/`dial` (default "unk"),
    `weight` (default "1.0"), `file_id` (default "1") -- string-valued, mirroring
    `engine/corpus.rs`'s `CorpusItem` listing-branch fields exactly, including the
    SIBLING-not-nested token-count gates (a `tokens.len() > N` check is a sibling of the
    `tokens[N-1]` emptiness check one level up, not nested inside it -- an empty dial
    token does NOT block weight/file_id parsing). A blank line, or a line whose first
    token is empty, is skipped.
    """
    text = path.read_text()
    records: list[dict[str, str]] = []
    for line in _splitstr(text, "\n"):
        tokens = _splitstr(line, ";")
        if not tokens or not tokens[0]:
            continue
        filename = tokens[0]
        refseg = ""
        lang = "unk"
        dial = "unk"
        weight = "1.0"
        file_id = "1"
        if len(tokens) > 1:
            if tokens[1]:
                refseg = tokens[1]
            if len(tokens) > 2:
                if tokens[2]:
                    lang = tokens[2]
                if len(tokens) > 3:
                    if tokens[3]:
                        dial = tokens[3]
                    if len(tokens) > 4 and tokens[4]:
                        w = _iss_extract_double(tokens[4])
                        if w is not None:
                            weight = tokens[4]
                        if len(tokens) > 5 and tokens[5]:
                            fid = _iss_extract_int(tokens[5])
                            if fid is not None:
                                file_id = tokens[5]
        records.append({"filename": filename, "refseg": refseg, "lang": lang, "dial": dial, "weight": weight, "file_id": file_id})
    return records


# --------------------------------------------------------------------------- #
# WriteListing.m / WriteWeightedListing.m: the two listing writers. `items` is already
# the index-selected, ORDERED record list -- the port has no separate `liste`/`index`
# split (unlike the legacy's `PS.VP.files.liste(index(ii))` indirection), so `items[i]`
# IS `index[i]`'s record. Both functions reuse read_listing's field names (`filename`/
# `refseg`/`lang`/`dial`) rather than introducing new ones per struct; `refseg` covers
# both WriteListing's `refsegfiles.liste(...).name` and WriteWeightedListing's
# `.segfilename` (the same concept -- a reference segmentation file path).
# --------------------------------------------------------------------------- #


def _worker_positions(n: int, jj: int, nb_workers: int) -> list[int]:
    """0-based port of `worker_index(jj).index = fliplr(length(index)-jj+1:-nbworker:1)`
    (`WriteListing.m:12`). `jj` is 1-based, matching the legacy loop variable exactly
    (the caller ranges `jj` over `1..nb_workers`). The descending MATLAB range
    `a:-nbworker:1` (inclusive of the endpoint 1) becomes Python's `range(a, 0,
    -nb_workers)` (exclusive of 0) -- both stop at the same last positive term, since
    the endpoint is a plain integer arithmetic sequence, not a boundary that differs
    under MATLAB's inclusive-vs-Python's exclusive convention. `fliplr` (ascending
    order) is `[::-1]`; the final `- 1` shifts each 1-based MATLAB position to a 0-based
    Python index into `items`."""
    a = n - jj + 1
    return [p - 1 for p in range(a, 0, -nb_workers)][::-1]


def write_listing(base: Path, items: list[dict[str, str]], nb_workers: int) -> None:
    """Port of `WriteListing.m` (22 lines). Writes `<base>.flst` -- one
    `filename;refseg;lang;dial;\\n` row per item, in `items` order -- then, for
    `jj = 1..min(nb_workers, len(items))`, a round-robin shard file
    `<base>_worker_<jj>.flst` selecting `items` at `_worker_positions(len(items), jj,
    nb_workers)` (see that function for the 1-based -> 0-based translation). The shards
    partition `items` exactly once each when `len(items)` isn't evenly divisible by
    `nb_workers` -- e.g. 7 items / 3 workers splits 3+2+2, every item in exactly one
    shard (non-vacuity pinned in `tests/test_phase4d_listing_writers.py`)."""

    def _row(it: dict[str, str]) -> str:
        return f"{it['filename']};{it['refseg']};{it['lang']};{it['dial']};\n"

    (base.parent / f"{base.name}.flst").write_text("".join(_row(it) for it in items))

    nb_workers = min(nb_workers, len(items))
    for jj in range(1, nb_workers + 1):
        shard = [items[p] for p in _worker_positions(len(items), jj, nb_workers)]
        (base.parent / f"{base.name}_worker_{jj}.flst").write_text("".join(_row(it) for it in shard))


def write_weighted_listing(path: Path, items: list[dict[str, str]], values: NDArray[np.float64]) -> None:
    """Port of `WriteWeightedListing.m` (22 lines). Writes `path` AS GIVEN -- no
    `.flst` suffix appended, unlike `write_listing` -- one
    `filename;refseg;lang;dial;%g;%g;\\n` row per item: `values[i, 0]` is
    `filesValues(index(i),2)` (the per-file weight/relevance score), `values[i, 1]` is
    `listing{index(i)}.duration`. The legacy's worker-shard block (`:11-22`) is
    COMMENTED OUT in the source and is NOT ported here -- there is no
    `write_weighted_listing` shard variant.

    The two `%g` fields use Python's `:g` format spec, which is byte-identical to
    Octave's C-style `sprintf('%g', ...)` at every probed style-switch boundary
    (exponent < -4 or >= precision 6, negative zero, exponent-digit-count, rounding
    carries) -- verified against a live Octave oracle; see
    `tests/reference_data/phase4d/listing/weighted.lst` and its extractor stage
    docstring for the specific probed values."""
    lines = []
    for it, row in zip(items, values, strict=True):
        weight, duration = float(row[0]), float(row[1])
        lines.append(f"{it['filename']};{it['refseg']};{it['lang']};{it['dial']};{weight:g};{duration:g};\n")
    path.write_text("".join(lines))


# --------------------------------------------------------------------------- #
# CreateBatches.m / GetNewBatch.m / getCases.m's getWorstAndBest
# --------------------------------------------------------------------------- #


def _mround(x: float) -> float:
    """MATLAB round: half away from zero (numpy's is banker's)."""
    return float(np.sign(x) * np.floor(np.abs(x) + 0.5))


@dataclass
class SubCaseGroup:
    """`Batches.Cases(nbOfTargetClasses).SubCases(k)` (multilingual, nbOfTargetClasses>1
    only): one non-target class's shuffled index pool + rotation cursor."""

    current_pos: int
    index: NDArray[np.int64]


@dataclass
class CaseGroup:
    """`Batches.Cases(i)`: one class's shuffled file-index pool + rotation cursor.
    `sub_cases`/`current_sub_class` are populated only for the LAST (aggregate) group
    under the multilingual, nbOfTargetClasses>1 branch."""

    current_pos: int
    index: NDArray[np.int64]
    current_sub_class: int = 0
    sub_cases: list[SubCaseGroup] = field(default_factory=list)


@dataclass
class WorstCaseGroup:
    """`Batches.WorstCases(i)`: the fixed hard-example subset excluded from rotation."""

    index: NDArray[np.int64]
    score: NDArray[np.float64]


@dataclass
class Batches:
    """Port of the `Corpora.Batches` struct built by `CreateBatches.m` and consumed/
    mutated by `GetNewBatch.m`. `current_class is None` <=> the degenerate full-batch
    case (`CreateBatches.m:15-17`): `degenerate_index` holds `(1:file_nb)'` and
    `cases`/`worst_cases` are empty."""

    nb_cases_per_batch: int
    nb_worst_cases: int
    is_multilingual: bool
    nb_target_classes: int
    validation: NDArray[np.int64]
    current_class: int | None
    cases: list[CaseGroup]
    worst_cases: list[WorstCaseGroup]
    degenerate_index: NDArray[np.int64] | None = None
    count_pass: float = 0.0


def create_batches(
    files_values: NDArray[np.float64],
    minibatch: int,
    nb_worst: int,
    multilingual: bool,
    nb_classes: int,
    rng: np.random.Generator,
) -> Batches:
    """Port of `CreateBatches.m`. `files_values[:, 0]` is the per-file class index
    (mirrors `Corpora.filesValues(:,1)`); `rng.permutation(array)` replaces every
    `randperm`-driven shuffle (the ONLY RNG use in this module).

    Reproduces two legacy quirks verbatim (documented in IMPROVEMENTS.md):
      * degenerate gate (`:15-17`): `nb_classes*nb_worst+minibatch > file_nb` skips all
        shuffling -- every file is one flat, unshuffled batch.
      * the non-multilingual `nb_classes>1` branch (`:43-60`) assigns
        `cases[ii]` using the LOOP POSITION `ii` over `sorted(unique(class values))`, NOT
        the class VALUE itself. When class values are the natural contiguous labeling
        `0..nb_classes-1`, the LAST loop position collides with the aggregate slot
        (`nb_classes-1`) reserved for non-target files, silently OVERWRITING it with the
        top target class's data and losing the aggregate.
    """
    file_nb = files_values.shape[0]
    validation = np.zeros(file_nb, dtype=np.int64)

    if nb_classes * nb_worst + minibatch > file_nb:
        return Batches(
            nb_cases_per_batch=minibatch,
            nb_worst_cases=nb_worst,
            is_multilingual=multilingual,
            nb_target_classes=nb_classes,
            validation=validation,
            current_class=None,
            cases=[],
            worst_cases=[],
            degenerate_index=np.arange(file_nb, dtype=np.int64),
        )

    def shuffled(mask: NDArray[np.bool_]) -> NDArray[np.int64]:
        base = np.flatnonzero(mask).astype(np.int64)
        return np.asarray(rng.permutation(base), dtype=np.int64)

    if nb_classes <= 1:
        if not multilingual:
            idx = shuffled(np.ones(file_nb, dtype=np.bool_))
            cases = [CaseGroup(current_pos=0, index=idx)]
            worst = [WorstCaseGroup(index=idx[:nb_worst].copy(), score=np.zeros(nb_worst))]
        else:
            possible = np.unique(files_values[:, 0])
            cases = []
            worst = []
            for v in possible:
                idx = shuffled(files_values[:, 0] == v)
                cases.append(CaseGroup(current_pos=0, index=idx))
                k = min(nb_worst, idx.size)
                worst.append(WorstCaseGroup(index=idx[:k].copy(), score=np.zeros(k)))
        return Batches(
            nb_cases_per_batch=minibatch,
            nb_worst_cases=nb_worst,
            is_multilingual=multilingual,
            nb_target_classes=nb_classes,
            validation=validation,
            current_class=0,
            cases=cases,
            worst_cases=worst,
        )

    possible = np.unique(files_values[:, 0])
    cases = [CaseGroup(current_pos=0, index=np.array([], dtype=np.int64)) for _ in range(nb_classes)]
    worst = [WorstCaseGroup(index=np.array([], dtype=np.int64), score=np.array([])) for _ in range(nb_classes)]

    if not multilingual:
        for ii, v in enumerate(possible):
            if 0 < v < nb_classes:
                idx = shuffled(files_values[:, 0] == v)
                cases[ii] = CaseGroup(current_pos=0, index=idx)  # legacy quirk: slot ii, not value v
                k = min(nb_worst, idx.size)
                worst[ii] = WorstCaseGroup(index=idx[:k].copy(), score=np.zeros(k))
            else:
                extra = np.flatnonzero(files_values[:, 0] == v).astype(np.int64)
                cases[nb_classes - 1].index = np.concatenate([cases[nb_classes - 1].index, extra])
        agg = cases[nb_classes - 1].index
        k = min(nb_worst, agg.size)
        worst[nb_classes - 1] = WorstCaseGroup(index=agg[:k].copy(), score=np.zeros(k))
    else:
        agg_worst: list[int] = []
        for ii, v in enumerate(possible):
            if 0 < v < nb_classes:
                idx = shuffled(files_values[:, 0] == v)
                cases[ii] = CaseGroup(current_pos=0, index=idx)
                k = min(nb_worst, idx.size)
                worst[ii] = WorstCaseGroup(index=idx[:k].copy(), score=np.zeros(k))
            else:
                sub_idx = shuffled(files_values[:, 0] == v)
                cases[nb_classes - 1].sub_cases.append(SubCaseGroup(current_pos=0, index=sub_idx))
                remaining = nb_worst - len(agg_worst)
                k = min(remaining, sub_idx.size)
                agg_worst.extend(int(x) for x in sub_idx[:k])
        cases[nb_classes - 1].current_sub_class = 0
        agg_arr = np.array(agg_worst, dtype=np.int64)
        worst[nb_classes - 1] = WorstCaseGroup(index=agg_arr, score=np.zeros(agg_arr.size))

    return Batches(
        nb_cases_per_batch=minibatch,
        nb_worst_cases=nb_worst,
        is_multilingual=multilingual,
        nb_target_classes=nb_classes,
        validation=validation,
        current_class=0,
        cases=cases,
        worst_cases=worst,
    )


def get_new_batch(batches: Batches) -> tuple[list[int], Batches]:
    """Port of `GetNewBatch.m`. PURE integer-cursor rotation -- no RNG. Mutates and
    returns `batches` (matching the legacy's in/out struct semantics); `batches.validation`
    and `batches.count_pass` are updated in place before returning.
    """
    if batches.current_class is None:
        assert batches.degenerate_index is not None
        batch = [int(x) for x in batches.degenerate_index]
    else:
        batch = []
        if batches.is_multilingual and batches.nb_target_classes > 1:
            while len(batch) < batches.nb_cases_per_batch:
                cc = batches.current_class
                if cc < len(batches.cases) - 1:
                    case = batches.cases[cc]
                    worst = batches.worst_cases[cc]
                    if case.index.size > worst.index.size:
                        elem = int(case.index[case.current_pos])
                        has_changed = False
                        if worst.index.size == 0 or not np.any(worst.index == elem):
                            batch.append(elem)
                            has_changed = True
                        case.current_pos += 1
                        if case.current_pos >= case.index.size:
                            case.current_pos = 0
                        if has_changed:
                            batches.current_class = cc + 1
                    else:
                        batches.current_class = cc + 1
                else:
                    case = batches.cases[cc]
                    worst = batches.worst_cases[cc]
                    sub = case.sub_cases[case.current_sub_class]
                    diff_sub = np.setdiff1d(sub.index, worst.index)
                    if diff_sub.size > 0:
                        elem = int(sub.index[sub.current_pos])
                        has_changed = False
                        if worst.index.size == 0 or not np.any(worst.index == elem):
                            batch.append(elem)
                            has_changed = True
                        sub.current_pos += 1
                        if sub.current_pos >= sub.index.size:
                            sub.current_pos = 0
                        case.current_sub_class += 1
                        if case.current_sub_class >= len(case.sub_cases):
                            case.current_sub_class = 0
                        if has_changed:
                            batches.current_class = 0
                    else:
                        case.current_sub_class += 1
                        if case.current_sub_class >= len(case.sub_cases):
                            case.current_sub_class = 0
                        batches.current_class = 0
        else:
            while len(batch) < batches.nb_cases_per_batch:
                cc = batches.current_class
                case = batches.cases[cc]
                worst = batches.worst_cases[cc]
                if case.index.size > worst.index.size:
                    elem = int(case.index[case.current_pos])
                    has_changed = False
                    if worst.index.size == 0 or not np.any(worst.index == elem):
                        batch.append(elem)
                        has_changed = True
                    case.current_pos += 1
                    if case.current_pos >= case.index.size:
                        case.current_pos = 0
                    if has_changed:
                        nxt = cc + 1
                        batches.current_class = nxt if nxt < len(batches.cases) else 0
                else:
                    nxt = cc + 1
                    batches.current_class = nxt if nxt < len(batches.cases) else 0

    batch_arr = np.array(batch, dtype=np.int64)
    if batch_arr.size:
        batches.validation[batch_arr] += 1
    if batches.worst_cases:
        worst_flat = np.concatenate([w.index for w in batches.worst_cases])
        if worst_flat.size:
            batches.validation[worst_flat] += 1

    vmin = int(batches.validation.min())
    batches.count_pass = vmin + float(np.sum(batches.validation != vmin)) / batches.validation.size
    return batch, batches


@dataclass
class WorstAndBest:
    """`[indBest,indMiddle,indWorst,scoreBest,scoreMiddle,scoreWorst]` from
    `getCases.m`'s private `getWorstAndBest` subfunction."""

    ind_best: list[int]
    ind_middle: list[int]
    ind_worst: list[int]
    score_best: list[float]
    score_middle: list[float]
    score_worst: list[float]


def get_worst_and_best(sorting_table: NDArray[np.float64], nb_worst: int, nb_best: int, nb_middle: int) -> WorstAndBest:
    """Port of `getCases.m:116-165`'s private `getWorstAndBest` subfunction (unreachable
    standalone in the legacy -- it is a MATLAB file-local helper of `getCases.m`, which
    itself requires the full `ComputeCost`/`CostFunction` engine chain to invoke).

    `sorting_table` is `[cost, index]`, PRE-SORTED ascending by cost (mirrors the
    caller's `sortrows` contract in `getCases.m`); `index` values are opaque (typically
    0-based file indices) and are returned as-is, deduplicated by value. Worst = the
    `nb_worst` highest-cost rows (scanned from the tail, first-seen-wins dedup, order
    restored ascending-by-original-position via a final reverse); best = the `nb_best`
    lowest-cost rows excluding anything already claimed by worst; middle = `nb_middle`
    rows evenly step-spaced between best and worst (`round` half-away-from-zero, min
    step 1), excluding anything already claimed. `nb_worst<=0` widens worst to cover the
    whole table; `nb_best<=0` still reserves exactly one "best" slot count (row 1) even
    though nothing is appended -- both mirror the legacy's `else` branches verbatim,
    including their effect on the `step`/`count` bookkeeping the middle scan reuses.

    Internal bookkeeping (`count`/`nb_worst`/`nb_best`) is kept 1-based, mirroring the
    legacy row-position arithmetic exactly; only array reads are translated to 0-based.
    """
    file_nb = sorting_table.shape[0]

    ind_worst: list[int] = []
    score_worst: list[float] = []
    if nb_worst > 0:
        count = file_nb
        while len(ind_worst) < nb_worst and count > 0:
            idx = int(sorting_table[count - 1, 1])
            if not ind_worst or idx not in ind_worst:
                ind_worst.append(idx)
                score_worst.append(float(sorting_table[count - 1, 0]))
            count -= 1
        nb_worst = count + 1
    else:
        nb_worst = file_nb
    ind_worst.reverse()
    score_worst.reverse()

    ind_best: list[int] = []
    score_best: list[float] = []
    if nb_best > 0:
        count = 1
        while len(ind_best) < nb_best and count <= file_nb:
            idx = int(sorting_table[count - 1, 1])
            not_worst = idx not in ind_worst
            if (not ind_best and not_worst) or (idx not in ind_best and not_worst):
                ind_best.append(idx)
                score_best.append(float(sorting_table[count - 1, 0]))
            count += 1
        nb_best = count - 1
    else:
        nb_best = 1

    ind_middle: list[int] = []
    score_middle: list[float] = []
    if nb_middle > 0:
        step = int(max(1.0, _mround((nb_worst - nb_best + 1) / (nb_middle + 1))))
        count = nb_best + step
        while len(ind_middle) < nb_middle and count <= nb_worst:
            idx = int(sorting_table[count - 1, 1])
            not_taken = idx not in ind_best and idx not in ind_worst
            if (not ind_middle and not_taken) or (idx not in ind_middle and not_taken):
                ind_middle.append(idx)
                score_middle.append(float(sorting_table[count - 1, 0]))
            count += step

    return WorstAndBest(ind_best, ind_middle, ind_worst, score_best, score_middle, score_worst)
