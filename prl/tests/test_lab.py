import pytest

from prl import lab as harness
from prl.checks import CheckFailed, assert_close


def new_ns(monkeypatch, worked=False):
    monkeypatch.setenv("PRL_WORKED", "1" if worked else "0")
    monkeypatch.delenv("PRL_QUICK", raising=False)
    ns = {}
    lab = harness.init(
        "01-test", "0123456789abcdef", designed="colab-cpu", api=1, ns=ns, checkpoints=["1"]
    )
    return ns, lab


def reference_double(x):
    return 2 * x


def test_solution_never_replaces_participant_code(monkeypatch):
    ns, lab = new_ns(monkeypatch)

    def double(x):  # participant's (wrong) code
        return x + 2

    ns["double"] = double
    reference_double.__name__ = "double"
    bound = lab.solution(1)(reference_double)
    assert bound is double


def test_missing_todo_gives_standin(monkeypatch):
    ns, lab = new_ns(monkeypatch)
    reference_double.__name__ = "double"
    bound = lab.solution(1)(reference_double)
    with pytest.raises(NotImplementedError, match="TODO 1"):
        bound(3)


def test_worked_mode_binds_reference(monkeypatch):
    ns, lab = new_ns(monkeypatch, worked=True)
    ns["double"] = lambda x: x
    reference_double.__name__ = "double"
    assert lab.solution(1)(reference_double) is reference_double


def test_check_reports_whose_and_status(monkeypatch, capsys):
    ns, lab = new_ns(monkeypatch)
    reference_double.__name__ = "double"
    ns["double"] = lambda x: x + 2
    ns["double"] = lab.solution(1)(reference_double)

    def checkpoint(fn):
        assert_close(fn(3), 6, what="double(3)")

    with pytest.raises(CheckFailed):
        lab.check(1, checkpoint, ns["double"])
    assert lab.status() == "fail"
    lab.use_reference(1)
    lab.check(1, checkpoint, ns["double"])
    out = capsys.readouterr().out
    assert "REFERENCE" in out and "passed" in out
    assert lab.status() == "pass"
    assert lab._s["checks"][-1]["whose"] == "reference"


def test_not_implemented_becomes_check_failed(monkeypatch):
    ns, lab = new_ns(monkeypatch)

    def stub(x):
        raise NotImplementedError("TODO 1: write me")

    with pytest.raises(CheckFailed, match="TODO 1"):
        lab.check(1, lambda f: f(1), stub)


def test_rerunning_init_keeps_references(monkeypatch):
    ns, lab = new_ns(monkeypatch)
    reference_double.__name__ = "double"
    lab.solution(1)(reference_double)
    lab2 = harness.init("01-test", "0123456789abcdef", designed="colab-cpu", api=1, ns=ns)
    assert lab2._s["refs"][1]["double"] is reference_double


def test_api_mismatch_refuses(monkeypatch):
    with pytest.raises(harness.APIVersionError):
        harness.init("x", "0" * 16, designed="colab-cpu", api=999, ns={})


def test_record_is_valid(monkeypatch):
    from prl import record

    ns, lab = new_ns(monkeypatch)
    lab.check(1, lambda: None)
    rec = lab.build_record()
    assert rec["status"] == "pass"
    assert record.validate(rec) == []


def test_solution_cell_says_it_does_not_replace_code(monkeypatch, capsys):
    ns, lab = new_ns(monkeypatch)
    ns["double"] = lambda x: x + 2
    reference_double.__name__ = "double"
    lab.solution(1)(reference_double)
    out = capsys.readouterr().out
    assert "not used" in out and "lab.use_reference(1)" in out
