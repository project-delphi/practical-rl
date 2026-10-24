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


def _experiment(steps, seeds=(0, 1, 2), **overrides):
    """A valid experiment record with a 4-point grid in its budget and one curve per seed."""
    grid = [steps // 4 * k for k in range(1, 5)]
    rec = {
        "schema": 1,
        "kind": "experiment",
        "experiment": "m99-bands",
        "script": "scripts/m99_bands.py",
        "args": {},
        "prl_version": "0.1.0",
        "date": "2026-10-09",
        "platform": "local",
        "env": "local",
        "hardware": {"cpu": "x", "n_cpu": 2, "ram_gb": 8.0, "accel": "cpu", "gpu": None},
        "torch_threads": 2,
        "python": "3.13.2",
        "packages": {"numpy": "2.1.3"},
        "budget": {"steps": steps, "grid": grid, "lr": 0.001},
        "seeds": list(seeds),
        "per_seed": {"return": [[float(s + k) for k in range(4)] for s in seeds]},
        "seconds": 1.5,
    }
    rec.update(overrides)
    return rec


@pytest.fixture
def experiment_record():
    """Factory: experiment_record(steps, seeds=(0, 1, 2), **overrides) -> a valid record."""
    return _experiment
