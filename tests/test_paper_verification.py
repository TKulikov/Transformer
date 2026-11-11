"""Paper-vs-code verification as part of the default CPU test-suite.

Each formula/claim checked by ``scripts/verify_paper.py`` becomes a separate
test item, so a drift between the article and the code fails loudly here.
Run standalone with ``python scripts/verify_paper.py`` for a PASS/FAIL table.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import verify_paper as VP  # noqa: E402


@pytest.mark.parametrize("name,fn", VP._CHECKS, ids=[n for n, _ in VP._CHECKS])
def test_paper_matches_code(name, fn):
    try:
        fn()
    except VP._Skip as e:
        pytest.skip(str(e))
