"""Generated files must match their generators (the drift gate, also run in CI)."""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "script",
    [
        "gen_includes.py",
        "build_notebooks.py",
        "gen_readiness.py",
        "gen_tokens.py",
        "make_expected.py",
    ],
)
def test_generated_files_are_current(script):
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), "--check"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert result.returncode == 0, result.stdout + result.stderr
