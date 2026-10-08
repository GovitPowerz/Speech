#!/usr/bin/env bash
# 04: the v2-lineage transformer full-run rows (RESULTS.md table phase11_v2_cells), both directions.
# Usage: experiments/04_v2_transformer.sh [arm/cell/direction ...]   (no args = both rows)
set -u
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
ROWS=(sad-v2/transformer/bidirectional sad-v2/transformer/forward)
for row in $(select_rows "${ROWS[@]}" -- "$@"); do
  IFS=/ read -r arm cell direction <<<"$row"
  run_row v2_transformer "$arm" "$cell" "$direction" --audio-max-duration 120
done
finish
