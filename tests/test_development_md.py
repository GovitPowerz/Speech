"""DEVELOPMENT.md quotes the agent deny list verbatim and completely."""

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_deny_list_quote_matches_settings() -> None:
    deny = json.loads((REPO / ".claude/settings.json").read_text(encoding="utf-8"))["permissions"]["deny"]
    quoted = re.findall(r"`(Bash\([^`]*\))`", (REPO / "DEVELOPMENT.md").read_text(encoding="utf-8"))
    assert quoted == deny


def test_sections_and_length() -> None:
    text = (REPO / "DEVELOPMENT.md").read_text(encoding="utf-8")
    headings = re.findall(r"^## (.+)$", text, flags=re.MULTILINE)
    assert headings == ["Division of labour", "Evidence rule", "Enforcement", "Worked examples"]
    assert len(text.splitlines()) <= 100
