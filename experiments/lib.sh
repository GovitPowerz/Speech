#!/usr/bin/env bash
# Shared by the runners (issue #21): one launcher invocation per row into a fresh timestamped
# run directory under runs/<study>/, a log beside it, then a promotion attempt through
# `python -m speech.ledger add`. A row's recipe is hard-coded here (the recipe is the ledger
# row's identity; a SAD row adds the 120 s audio cap the full-run protocol fixes); only the host
# varies:
#   SPEECH_CORPUS_ROOT  the LRE03/07 corpus root (default data/LRE03-LRE07 in the repo; a relative
#                       path is the caller's, resolved here before any row changes directory)
#   SPEECH_LANES        the engine's fold width (default 1, the parity mode; ADR-0007 records it)
# A failed row never stops the others; a refused promotion prints the manual command. The
# runner exits nonzero if any row failed or was not promoted. Written for the bash 3.2 macOS
# ships: no empty-array expansion under `set -u`, no `mapfile`.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
case ${SPEECH_CORPUS_ROOT:-} in "" | /*) ;; *) SPEECH_CORPUS_ROOT="$PWD/$SPEECH_CORPUS_ROOT" ;; esac
CORPUS_ROOT="${SPEECH_CORPUS_ROOT:-data/LRE03-LRE07}"
LANES="${SPEECH_LANES:-1}"
BUDGET="--seed 0 --epochs 40 --steps-per-epoch 25 --patience 6"
FAILED=""

# run_row <study> <arm> <cell> <direction>
run_row() {
  local study=$1 arm=$2 cell=$3 direction=$4
  local stamp out cap=""
  case $arm in sad*) cap="--audio-max-duration 120" ;; esac
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  out="$REPO/runs/$study/${arm}_${cell}_${direction}_${stamp}"
  mkdir -p "$REPO/runs/$study"
  echo "== $study: $arm $cell/$direction -> $out"
  # shellcheck disable=SC2086 -- BUDGET and cap are flag lists on purpose
  if ! (cd "$REPO" && uv run python -m speech.drivers.baseline "$arm" --corpus-root "$CORPUS_ROOT" --out-dir "$out" \
        --lanes "$LANES" --cell "$cell" --direction "$direction" $BUDGET $cap 2>&1 | tee "$out.log"); then
    echo "!! $arm $cell/$direction failed; log: $out.log"
    FAILED="$FAILED
!! $arm $cell/$direction failed"
    return 0
  fi
  if (cd "$REPO" && uv run python -m speech.ledger add "$out/record.json"); then
    echo "== promoted $out/record.json"
  else
    echo "!! $arm $cell/$direction ran but was not promoted (dirty tree, or a live record for this recipe)."
    echo "   When the tree is clean, or with --supersede REASON / --allow-dirty:"
    echo "   uv run python -m speech.ledger add $out/record.json"
    FAILED="$FAILED
!! $arm $cell/$direction not promoted"
  fi
}

# run_rows <study> <rows...> -- <args...>: run the rows the args name (arm/cell/direction), or
# all of them; an unknown name stops the runner before any row runs.
run_rows() {
  local study=$1
  shift
  local rows="" args="" seen_sep=0 token want row hit
  for token in "$@"; do
    if [[ $token == "--" ]]; then seen_sep=1; continue; fi
    if ((seen_sep)); then args="$args $token"; else rows="$rows $token"; fi
  done
  local selected="$rows"
  if [[ -n $args ]]; then
    selected=""
    for want in $args; do
      hit=0
      for row in $rows; do [[ $row == "$want" ]] && hit=1; done
      if ((hit == 0)); then echo "!! unknown row $want (rows:$rows)" >&2; exit 2; fi
      selected="$selected $want"
    done
  fi
  for row in $selected; do
    IFS=/ read -r arm cell direction <<<"$row"
    run_row "$study" "$arm" "$cell" "$direction"
  done
  if [[ -n $FAILED ]]; then
    echo "$FAILED"
    exit 1
  fi
  echo "== all rows ran and were promoted"
}
