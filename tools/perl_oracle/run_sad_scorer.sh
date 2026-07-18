#!/usr/bin/env bash
#
# perl_oracle/run_sad_scorer.sh -- pipe crafted ref/hyp TAB files through the REAL
# vendored NIST OpenSAD scorer (Optimizer_V6.2.2/scoreFile_SAD.pl, v2.1, Greg Sanders,
# public domain; read-only legacy tree, never modified/copied). This is the live
# value-oracle Phase 6 Task 4 pins `src/python/speech/evaluate.py`'s `dcf()` against:
# `scripts/extract_phase6_fixtures.py`'s DCF stage stages crafted ref/hyp tab pairs and
# shells out to this runner to capture the scorer's per-collar Prob_Miss/Prob_FalseAlarm/
# DCF table (committed as tests/reference_data/phase6/dcf/*.json goldens).
#
# The column args are the ComputeDCF.py convention (Optimizer_V6.2.{1,2}/ComputeDCF.py:83,
# the only in-tree caller of this scorer): ref start/end/type at zero-based cols 2/3/4,
# hyp start/end/type at cols 5/6/7. -v (verbose STDERR dumps) is deliberately NOT passed,
# so STDOUT carries only the header + the five per-collar score blocks.
#
# LOCAL-ONLY, like run_norm.sh: needs /usr/bin/perl (v5.34 vendored with macOS) and the
# legacy tree checked out at FSP_LEGACY_TREE. CI never runs this -- it consumes only the
# committed goldens.
#
# Usage:
#   run_sad_scorer.sh <ref.tab> <hyp.tab>    # -> the scorer's STDOUT table on stdout
#
set -euo pipefail

LEGACY_TREE="${FSP_LEGACY_TREE:-/Users/govit/Git/Govit/FastSpeechProcessing-legacy}"
PERL="${PERL_BIN:-/usr/bin/perl}"
SCORER="${SAD_SCORER:-$LEGACY_TREE/Optimizer_V6.2.2/scoreFile_SAD.pl}"

usage() {
  echo "usage: $(basename "$0") <ref.tab> <hyp.tab>" >&2
  exit 2
}

[[ $# -eq 2 ]] || usage
ref_file="$1"
hyp_file="$2"

exec "$PERL" "$SCORER" \
  -r "$ref_file" -h "$hyp_file" \
  -s 2 -e 3 -g 4 \
  -t 5 -f 6 -u 7
