#!/usr/bin/env bash
# Shared by the runners (issue #21): one launcher invocation per row into a fresh timestamped
# run directory under runs/<study>/, a log beside it, then a promotion attempt through
# `python -m speech.ledger add`. A row's recipe is hard-coded in its runner (the recipe is the
# ledger row's identity); only the host varies:
#   SPEECH_CORPUS_ROOT  the LRE03/07 corpus root (default data/LRE03-LRE07)
#   SPEECH_LANES        the engine's fold width (default 1, the parity mode; ADR-0007 records it)
# A failed row never stops the others; a refused promotion prints the manual command. The
# runner exits nonzero if any row failed or was not promoted.
set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CORPUS_ROOT="${SPEECH_CORPUS_ROOT:-data/LRE03-LRE07}"
LANES="${SPEECH_LANES:-1}"
BUDGET=(--seed 0 --epochs 40 --steps-per-epoch 25 --patience 6)
FAILED=()

# run_row <study> <arm> <cell> <direction> [extra launcher flags...]
run_row() {
  local study=$1 arm=$2 cell=$3 direction=$4
  shift 4
  local stamp out
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  out="$REPO/runs/$study/${arm}_${cell}_${direction}_${stamp}"
  mkdir -p "$REPO/runs/$study"
  echo "== $study: $arm $cell/$direction -> $out"
  if ! (cd "$REPO" && uv run python -m speech.drivers.baseline "$arm" --corpus-root "$CORPUS_ROOT" --out-dir "$out" \
        --lanes "$LANES" --cell "$cell" --direction "$direction" "${BUDGET[@]}" "$@" 2>&1 | tee "$out.log"); then
    echo "!! $arm $cell/$direction failed; log: $out.log"
    FAILED+=("$arm $cell/$direction")
    return 0
  fi
  if (cd "$REPO" && uv run python -m speech.ledger add "$out/record.json"); then
    echo "== promoted $out/record.json"
  else
    echo "!! $arm $cell/$direction ran but was not promoted (dirty tree, or a live record for this recipe)."
    echo "   When the tree is clean, or with --supersede REASON / --allow-dirty:"
    echo "   uv run python -m speech.ledger add $out/record.json"
    FAILED+=("promote $arm $cell/$direction")
  fi
}

# select_rows <rows...> -- <args...>: the rows named by the args (arm/cell/direction), or all of them.
select_rows() {
  local rows=() args=() seen_sep=0
  for token in "$@"; do
    if [[ $token == "--" ]]; then seen_sep=1; continue; fi
    if ((seen_sep)); then args+=("$token"); else rows+=("$token"); fi
  done
  if ((${#args[@]} == 0)); then printf '%s\n' "${rows[@]}"; return 0; fi
  for want in "${args[@]}"; do
    local hit=0
    for row in "${rows[@]}"; do [[ $row == "$want" ]] && hit=1; done
    if ((hit == 0)); then echo "!! unknown row $want (rows: ${rows[*]})" >&2; exit 2; fi
    echo "$want"
  done
}

finish() {
  if ((${#FAILED[@]})); then
    printf '!! %s\n' "${FAILED[@]}"
    exit 1
  fi
  echo "== all rows ran and were promoted"
}
