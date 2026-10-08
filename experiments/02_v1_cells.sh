#!/usr/bin/env bash
# 02: the v1-lineage (cell x direction) full-run rows (RESULTS.md table phase9_v1_cells):
# sLSTM and Mamba in both directions on `sad`; lstm/bidirectional is 01's row.
# Usage: experiments/02_v1_cells.sh [arm/cell/direction ...]   (no args = every row)
set -u
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
ROWS=(sad/slstm/bidirectional sad/slstm/forward sad/mamba/bidirectional sad/mamba/forward)
for row in $(select_rows "${ROWS[@]}" -- "$@"); do
  IFS=/ read -r arm cell direction <<<"$row"
  run_row v1_cells "$arm" "$cell" "$direction" --audio-max-duration 120
done
finish
