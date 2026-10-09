#!/usr/bin/env bash
# 05: the LID (cell x direction) full-run rows (RESULTS.md tables lid_features_cells and
# lid_phseq_cells): every cell in both directions on both LID arms, except lstm/bidirectional
# which 01 fires. The knob targets the LID net; the SAD net stays the legacy LSTM at its seed.
# Exact-tree numbers (the fast Twin LID runs the same matrix since #57, parity-pinned in CI).
# Usage: experiments/05_lid_cells.sh [arm/cell/direction ...]   (no args = every row)
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
rows=""
for arm in lid-features lid-phseq; do
  rows="$rows $arm/lstm/forward"
  for cell in slstm mamba cfc transformer; do for direction in bidirectional forward; do rows="$rows $arm/$cell/$direction"; done; done
done
# shellcheck disable=SC2086 -- rows is a word list on purpose
run_rows lid_cells $rows -- "$@"
