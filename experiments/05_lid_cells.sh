#!/usr/bin/env bash
# 05: the LID (cell x direction) full-run rows (RESULTS.md tables lid_features_cells and
# lid_phseq_cells): every cell in both directions on both LID arms, except lstm/bidirectional
# which 01 fires. The knob targets the LID net; the SAD net stays the legacy LSTM at its seed.
# Exact-tree numbers: the fast Twin LID is BLSTM-only (#57).
# Usage: experiments/05_lid_cells.sh [arm/cell/direction ...]   (no args = every row)
set -u
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
ROWS=()
for arm in lid-features lid-phseq; do
  ROWS+=("$arm/lstm/forward")
  for cell in slstm mamba cfc transformer; do for direction in bidirectional forward; do ROWS+=("$arm/$cell/$direction"); done; done
done
for row in $(select_rows "${ROWS[@]}" -- "$@"); do
  IFS=/ read -r arm cell direction <<<"$row"
  run_row lid_cells "$arm" "$cell" "$direction"
done
finish
