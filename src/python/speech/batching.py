"""Hard-example mini-batch scheduler and listing parsing.

Ported from legacy MATLAB: CreateBatches.m, GetNewBatch.m, getCases.m,
processListing.m. See design spec section 6.
"""

from pathlib import Path


def read_listing(path: Path) -> list[dict[str, str]]:
    """Parse a ';'-separated fileslisting CSV into records (Phase 0)."""
    ...
