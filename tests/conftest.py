import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """In CI, a test skipped because Quarto is missing is a failure, not a skip.

    The figure and filter fixture tests render small pages with Quarto. Locally they
    skip without it; in CI (where the workflow installs Quarto) a silent skip would
    hide a broken filter.
    """
    outcome = yield
    rep = outcome.get_result()
    if not (os.environ.get("CI") and rep.skipped and isinstance(rep.longrepr, tuple)):
        return
    if "quarto not on PATH" in str(rep.longrepr[-1]):
        rep.outcome = "failed"
        rep.longrepr = "CI must run the Quarto fixture tests, but quarto is not on PATH."
