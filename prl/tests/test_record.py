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
