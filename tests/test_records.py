import json
from pathlib import Path

import pytest

import add_run_record
import variables

ROOT = Path(__file__).resolve().parents[1]


def committed():
    return sorted((ROOT / "runs").glob("*.json"))


@pytest.mark.parametrize("path", committed(), ids=lambda p: p.name)
def test_committed_records_are_valid(path):
    rec = json.loads(path.read_text())
    assert add_run_record.check(rec, variables.load()) == []


def test_colab_env_requires_colab_platform():
    rec = {"kind": "notebook", "platform": "local", "env": "colab-cpu", "notebook": "00-setup"}
    problems = add_run_record.check(rec, variables.load())
    assert any("only runs on Colab" in p for p in problems)
