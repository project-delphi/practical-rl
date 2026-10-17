import datetime as dt
import json

from prl import record


def good_record(**overrides):
    rec = {
        "schema": 1,
        "kind": "notebook",
        "notebook": "00-setup",
        "content_sha": "0123456789abcdef",
        "prl_version": "0.1.0",
        "date": "2026-10-09",
        "platform": "local",
        "env": "local",
        "hardware": {"cpu": "x", "n_cpu": 2, "ram_gb": 8.0, "accel": "cpu", "gpu": None},
        "python": "3.13.2",
        "packages": {"numpy": "2.1.3"},
        "quick": False,
        "quick_reason": "r",
        "test_doubles": [],
        "shims": [],
        "solutions_bound": True,
        "scope": "notebook",
        "seeds": [0],
        "settings": {},
        "seconds": 1.5,
        "checkpoints": [{"label": "1", "pass": True, "whose": "reference"}],
        "metrics": {},
        "status": "pass",
        "source": "tool",
    }
    rec.update(overrides)
    return rec


def test_valid_record_has_no_problems():
    assert record.validate(good_record(), today=dt.date(2026, 10, 9)) == []


def test_invalid_records_are_caught():
    today = dt.date(2026, 10, 9)
    cases = {
        "missing": {k: v for k, v in good_record().items() if k != "status"},
        "unknown": good_record(extra=1),
        "secret": good_record(note="token hf_ABCDEFGHIJKLMNOPQRSTUV"),
        "aws key": good_record(note="AKIAABCDEFGHIJKLMNOP"),
        "path": good_record(note="/Users/alice/secret"),
        "future": good_record(date="2030-01-01"),
        "bad sha": good_record(content_sha="xyz"),
        "bad platform": good_record(platform="laptop"),
        "partial without note": good_record(scope="partial"),
        "bool seconds": good_record(seconds=True),
    }
    for name, rec in cases.items():
        assert record.validate(rec, today=today), name


def test_ci_runner_paths_are_allowed():
    rec = good_record(note="/home/runner/work/x")
    assert record.validate(rec, today=dt.date(2026, 10, 9)) == []


def test_content_sha_is_stable_and_normalized():
    a = record.content_sha([("ex01-stub", "x = 1\r\ny = 2   \n\n")])
    b = record.content_sha([("ex01-stub", "x = 1\ny = 2")])
    c = record.content_sha([("ex01-stub", "x = 1\ny = 3")])
    d = record.content_sha([("ex02-stub", "x = 1\ny = 2")])
    assert a == b and a != c and a != d and len(a) == 16


def test_emit_and_parse_roundtrip(tmp_path, capsys):
    rec = good_record()
    path = record.emit(rec, inbox=tmp_path)
    assert path is not None and json.loads(path.read_text()) == rec
    printed = capsys.readouterr().out
    assert record.parse_printed(printed) == [rec]


def test_room_code_roundtrip():
    import numpy as np

    code = record.room_code(
        "m03",
        seat=7,
        passed=8,
        total=9,
        quick=True,
        metrics={
            "td rms": 1.234567891,
            "edge_cells": 0,
            "converged": True,
            "per_seed": [1.0, 2.0],  # not a scalar: left out
            "seat": 3.5,  # collides with a fixed field
            "gap": np.float64(26.75),
        },
    )
    assert code == (
        "ROOM m03 seat=7 checks=8/9 quick=1 converged=1 edge_cells=0 gap=26.75 m_seat=3.5 td_rms=1.235"
    )
    pasted = f"Seat 7 says: {code}\nnoise\nROOM m03 seat=- checks=0/9 quick=0\n"
    rows = record.parse_room_codes(pasted)
    assert rows[0] == {
        "module": "m03",
        "seat": 7,
        "passed": 8,
        "total": 9,
        "quick": True,
        "metrics": {
            "converged": 1.0,
            "edge_cells": 0.0,
            "gap": 26.75,
            "m_seat": 3.5,
            "td_rms": 1.235,
        },
    }
    assert rows[1]["seat"] is None and rows[1]["metrics"] == {}


def test_source_commit_reads_the_checkout(tmp_path):
    import subprocess

    (tmp_path / "_variables.yml").write_text("x: 1\n")
    (tmp_path / "prl").mkdir()
    run = lambda *a: subprocess.run(["git", *a], cwd=tmp_path, check=True, capture_output=True)  # noqa: E731
    run("init", "-q")
    run("-c", "user.email=t@example.com", "-c", "user.name=t", "add", ".")
    run("-c", "user.email=t@example.com", "-c", "user.name=t", "commit", "-q", "-m", "x")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=tmp_path, capture_output=True, text=True, check=True
    ).stdout.strip()
    # The editable prl in this repo has no vcs_info, so the checkout's HEAD is used.
    sha, dirty = record.source_commit(tmp_path)
    assert (sha, dirty) == (head, False)
    (tmp_path / "_variables.yml").write_text("x: 2\n")
    assert record.source_commit(tmp_path) == (head, True)
