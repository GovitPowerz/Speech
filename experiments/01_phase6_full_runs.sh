#!/usr/bin/env bash
# 01: the three Phase-6 full-run rows (RESULTS.md tables phase6_lid_features, phase6_sad_v1,
# phase6_lid_phseq): the legacy BLSTM on each arm, whole split, 40 epochs x 25 steps.
# Usage: experiments/01_phase6_full_runs.sh [arm/cell/direction ...]   (no args = every row)
set -u
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
ROWS=(lid-features/lstm/bidirectional sad/lstm/bidirectional lid-phseq/lstm/bidirectional)
for row in $(select_rows "${ROWS[@]}" -- "$@"); do
  IFS=/ read -r arm cell direction <<<"$row"
  extra=()
  [[ $arm == sad* ]] && extra=(--audio-max-duration 120)
  run_row phase6_full "$arm" "$cell" "$direction" "${extra[@]}"
done
finish
