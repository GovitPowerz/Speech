#!/usr/bin/env bash
# 04: the v2-lineage transformer full-run rows (RESULTS.md table phase11_v2_cells), both directions.
# Usage: experiments/04_v2_transformer.sh [arm/cell/direction ...]   (no args = both rows)
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
run_rows v2_transformer sad-v2/transformer/bidirectional sad-v2/transformer/forward -- "$@"
