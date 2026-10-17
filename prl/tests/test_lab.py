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
    assert lab2._s["refs"]["1"]["double"] is reference_double


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


def test_reference_provenance_survives_rerunning_the_solution_cell(monkeypatch):
    ns, lab = new_ns(monkeypatch)

    def ref(x):
        return 2 * x

    ref.__name__ = "double"
    ns["double"] = lambda x: x + 2
    ns["double"] = lab.solution(1)(ref)
    lab.use_reference(1)
    ns["double"] = lab.solution(1)(ref)  # rerun the solution cell
    assert lab._whose(1) == "reference"


def test_use_reference_can_pick_one_name(monkeypatch):
    ns, lab = new_ns(monkeypatch)

    def residual(x):
        return x

    def suspect(x):
        return x

    ns["residual"], ns["suspect"] = (lambda x: 0), (lambda x: 1)
    lab.solution(6)(residual)
    lab.solution(6)(suspect)
    keep = ns["suspect"]
    lab.use_reference(6, "residual")
    assert ns["residual"] is residual and ns["suspect"] is keep
    lab.use_reference("6")  # string and int labels are the same exercise
    assert ns["suspect"] is suspect


def test_messages_show_valid_python_for_lettered_exercises(monkeypatch, capsys):
    ns, lab = new_ns(monkeypatch)

    def spec(c):
        return c

    lab.solution("3a")(spec)
    assert "lab.use_reference('3a')" in capsys.readouterr().out


def test_fixtures_are_copies():
    from prl.checks import expected

    a = expected("m01_eval")["rand_R"]
    a[:] = 0
    assert expected("m01_eval")["rand_R"].any()


# ---- verify mode: mutants, stubs and stochastic checkpoints ---------------------------
def _stochastic(thresholds):
    from prl.checks import StochasticCheck

    return StochasticCheck(
        ex=1,
        functions="double",
        metric=lambda fn, *, seed, n: 0.0,
        live={"n": 10},
        quick={"n": 2},
        higher_is_better=False,
        min_gap=1.0,
        thresholds=thresholds,
    )


def _exact_rejects(fn):
    raise CheckFailed("wrong")


def _exact_passes(fn):
    pass


def test_mutant_mode_skips_stochastic_checkpoints_and_reports_the_rest(monkeypatch, capsys):
    ns, lab = new_ns(monkeypatch)  # live budget: PRL_QUICK is unset
    check = _stochastic({"live": (1.0, "runs/x.json")})
    lab._mutant_begin("1", "wrong", {"double": "def double(x):\n    return x"})
    lab.check(1, _exact_rejects, ns["double"], label="1.1")
    lab.check(1, check, ns["double"], label="1.2")
    assert lab._s["mutant"]["results"][1] == {"label": "1.2", "skipped": True, "threshold": True}
    lab._mutant_end("1")
    assert "double" not in ns  # restored: the name did not exist before
    assert "rejected by 1 checkpoint(s), 1 skipped as stochastic" in capsys.readouterr().out


def test_an_exercise_with_only_stochastic_checkpoints_is_covered_by_the_protocol(
    monkeypatch, capsys
):
    ns, lab = new_ns(monkeypatch)
    check = _stochastic({"live": (1.0, "runs/x.json")})
    lab._mutant_begin("1", "wrong", {"double": "def double(x):\n    return x"})
    lab.check(1, check, ns["double"])
    lab._mutant_end("1")
    assert "covered by the threshold protocol (1 stochastic checkpoint(s) skipped)" in (
        capsys.readouterr().out
    )


def test_a_skipped_stochastic_checkpoint_needs_a_threshold_for_this_budget(monkeypatch):
    ns, lab = new_ns(monkeypatch)
    check = _stochastic({"quick": (1.0, "runs/x.json")})  # live is missing
    lab._mutant_begin("1", "wrong", {"double": "def double(x):\n    return x"})
    lab.check(1, _exact_rejects, ns["double"], label="1.1")
    lab.check(1, check, ns["double"], label="1.2")
    with pytest.raises(RuntimeError, match="1.2 are stochastic and have no threshold"):
        lab._mutant_end("1")


def test_an_exact_only_mutant_must_meet_an_exact_checkpoint(monkeypatch):
    ns, lab = new_ns(monkeypatch)
    check = _stochastic({"live": (1.0, "runs/x.json")})
    src = {"double": "def double(x):\n    return x"}
    lab._mutant_begin("1", "fine-grained", src, exact_only=True)
    lab.check(1, check, ns["double"])
    with pytest.raises(AssertionError, match="exact_only"):
        lab._mutant_end("1")
    lab._mutant_begin("1", "fine-grained", src, exact_only=True)
    lab.check(1, _exact_passes, ns["double"], label="1.1")
    lab.check(1, check, ns["double"], label="1.2")
    with pytest.raises(AssertionError, match="checkpoint 1.1 PASSED"):
        lab._mutant_end("1")


def test_mutant_decorator_records_exact_only():
    @harness.mutant(ex=1, replaces="f", id="a", why="plain")
    def f():
        pass

    @harness.mutant(ex=1, replaces="g", id="b", why="fine", exact_only=True)
    def g():
        pass

    assert f.__prl_mutant__["exact_only"] is False
    assert g.__prl_mutant__ == {
        "ex": "1",
        "replaces": "g",
        "id": "b",
        "why": "fine",
        "exact_only": True,
    }
