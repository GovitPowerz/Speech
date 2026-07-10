#!/usr/bin/env bash
#
# perl_oracle/run_norm.sh -- pipe fixture stdin through the REAL vendored perl STM
# normalizer scripts (read-only legacy tree, never modified/copied). This is the live
# byte-oracle Phase 4d Task 7 pins `src/python/speech/dataprep/stm_normalize.py`
# against: `scripts/extract_phase4d_fixtures.py`'s STM stage shells out to this runner
# to produce the committed tests/reference_data/phase4d/stm/*.in/*.out pairs.
#
# LOCAL-ONLY, like the other tools/ oracles: needs /usr/bin/perl (v5.34 vendored with
# macOS) and the legacy tree checked out at FSP_LEGACY_TREE. CI never runs this.
#
# Usage:
#   run_norm.sh light           < in.stm > out.stm    # norm_stm_pkt_light.pl
#   run_norm.sh full [type]     < in.stm > out.stm     # norm_stm_pk_cts_all_trans_03a.pl
#     [type] is the script's $ARGV[0] (its "am" vs non-"am" punctuation branch); it is
#     ONLY read after the norm-tagger|norm-parser shell-out (line ~109), which never
#     completes here (the LIMSI binaries are lost) -- so for every real content line the
#     real script bails via a failed backtick before [type] is ever consulted. Defaults
#     to "cts" (a non-"am" placeholder) since it is unreachable either way.
#
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
LEGACY_TREE="${FSP_LEGACY_TREE:-/Users/govit/Git/Govit/FastSpeechProcessing-legacy}"
PERL="${PERL_BIN:-/usr/bin/perl}"

usage() {
  echo "usage: $(basename "$0") {light|full} [type] < in.stm > out.stm" >&2
  exit 2
}

[[ $# -ge 1 ]] || usage
script_name="$1"
shift || true

case "$script_name" in
  light)
    script_path="$LEGACY_TREE/norm_stm_pkt_light.pl"
    exec "$PERL" "$script_path"
    ;;
  full)
    script_path="$LEGACY_TREE/norm_stm_pk_cts_all_trans_03a.pl"
    type_arg="${1:-cts}"
    exec "$PERL" "$script_path" "$type_arg"
    ;;
  *)
    usage
    ;;
esac
