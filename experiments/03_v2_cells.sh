#!/usr/bin/env bash
# 03: the v2-lineage (cell x direction) full-run rows (RESULTS.md table phase10_v2_cells):
# LSTM, sLSTM, Mamba and CfC in both directions on `sad-v2`.
# Usage: experiments/03_v2_cells.sh [arm/cell/direction ...]   (no args = every row)
set -u
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
ROWS=()
for cell in lstm slstm mamba cfc; do for direction in bidirectional forward; do ROWS+=("sad-v2/$cell/$direction"); done; done
for row in $(select_rows "${ROWS[@]}" -- "$@"); do
  IFS=/ read -r arm cell direction <<<"$row"
  run_row v2_cells "$arm" "$cell" "$direction" --audio-max-duration 120
done
finish
