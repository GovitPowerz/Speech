#!/usr/bin/env bash
# 01: the three Phase-6 full-run rows (RESULTS.md tables phase6_lid_features, phase6_sad_v1,
# phase6_lid_phseq): the legacy BLSTM on each arm, whole split, 40 epochs x 25 steps.
# Usage: experiments/01_phase6_full_runs.sh [arm/cell/direction ...]   (no args = every row)
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
run_rows phase6_full lid-features/lstm/bidirectional sad/lstm/bidirectional lid-phseq/lstm/bidirectional -- "$@"
