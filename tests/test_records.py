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


def test_filing_never_overwrites_a_different_record(tmp_path, monkeypatch):
    monkeypatch.setattr(add_run_record, "RUNS", tmp_path)
    v = variables.load()
    rec = json.loads(committed()[-1].read_text())
    first = add_run_record.file_record(rec, v)
    assert add_run_record.file_record(rec, v) == first  # the same record again: a no-op
    second = add_run_record.file_record({**rec, "seconds": rec["seconds"] + 1}, v)
    third = add_run_record.file_record({**rec, "seconds": rec["seconds"] + 2}, v)
    assert second.name == f"{first.stem}-2.json" and third.name == f"{first.stem}-3.json"
    assert json.loads(first.read_text()) == rec
