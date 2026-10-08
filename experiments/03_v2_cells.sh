#!/usr/bin/env bash
# 03: the v2-lineage (cell x direction) full-run rows (RESULTS.md table phase10_v2_cells):
# LSTM, sLSTM, Mamba and CfC in both directions on `sad-v2`.
# Usage: experiments/03_v2_cells.sh [arm/cell/direction ...]   (no args = every row)
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
rows=""
for cell in lstm slstm mamba cfc; do for direction in bidirectional forward; do rows="$rows sad-v2/$cell/$direction"; done; done
# shellcheck disable=SC2086 -- rows is a word list on purpose
run_rows v2_cells $rows -- "$@"
