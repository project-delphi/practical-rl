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


def test_fixture_check_allows_float_rounding_only():
    import numpy as np

    import make_expected as me

    base = {"V": np.array([-152.87, 0.5]), "k": np.array(140)}
    blob = me.save_npz(**base)
    rounded = me.save_npz(V=base["V"] * (1 + 4e-16), k=base["k"])
    assert blob != rounded and me.same_content(blob, rounded)
    assert not me.same_content(blob, me.save_npz(V=base["V"], k=np.array(141)))
    assert not me.same_content(blob, me.save_npz(V=base["V"] + 1e-6, k=base["k"]))
    assert not me.same_content(blob, me.save_npz(V=base["V"]))
