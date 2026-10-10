"""scripts/threshold_protocol.py on a synthetic lab: a noisy mean estimator and its mutants.

The lab estimates the mean of n samples from N(1, 1). Its solution's error is about
|N(0, 1/n)|; the "biased" and "halved" mutants are off by about 0.5, and "first-only"
(one sample) overlaps the solution, so no threshold separates them.
"""

import json
import textwrap

import nbformat
import numpy as np
import pytest

import add_run_record
import threshold_protocol as tp
import variables
from prl import API, record
from prl.checks import StochasticCheck

TRUE_MEAN = 1.0

CELLS = [
    ("gen-header", "markdown", "# 99 · A synthetic lab"),
    ("gen-setup", "code", "raise RuntimeError('the protocol replaces the setup cell')"),
    (
        "gen-init",
        "code",
        "import prl.lab\n\n"
        f'lab = prl.lab.init("99-synthetic", "0123456789abcdef", designed="colab-cpu", api={API}, '
        "ns=globals(), seat=SEED, badge_ref=PRL_REF, worked=WORKED or None)",
    ),
    ("c001", "code", "import numpy as np\n\nSHIFT = 0.5  # a notebook global that a mutant uses"),
    (
        "ex1-stub",
        "code",
        "def total(samples):\n    raise NotImplementedError('TODO 1, or lab.use_reference(1)')",
    ),
    ("ex1-sol", "code", "@lab.solution(1)\ndef total(samples):\n    return float(np.sum(samples))"),
    ("ex1-run", "code", "RAN_EX1 = total([1.0, 2.0])"),
    ("ex1-chk", "code", "raise RuntimeError('checkpoint cells must not run')"),
    (
        "ex2-stub",
        "code",
        "def mean_estimate(samples):\n    raise NotImplementedError('TODO 2, or lab.use_reference(2)')",
    ),
    (
        "ex2-sol",
        "code",
        "#@title Solution 2\n@lab.solution(2)\ndef mean_estimate(samples):\n"
        "    return total(samples) / len(samples)",
    ),
    ("ex2-run", "code", "raise RuntimeError('the target exercise must stop before its run cell')"),
    ("ex2-chk", "code", "raise RuntimeError('checkpoint cells must not run')"),
    ("gen-finish", "code", "lab.finish()"),
]

GOOD_MUTANTS = """
import numpy as np

from prl.lab import mutant


@mutant(ex=1, replaces="total", id="drops-last", why="an exercise-1 mutant; ignored for ex 2")
def total(samples):
    return float(np.sum(samples[:-1]))


@mutant(ex=2, replaces="mean_estimate", id="biased", why="adds a constant")
def mean_estimate(samples):
    return total(samples) / len(samples) + SHIFT


@mutant(ex=2, replaces="mean_estimate", id="halved", why="divides by 2n")
def mean_estimate(samples):
    return total(samples) / (2 * len(samples))
"""

OVERLAPPING_MUTANT = """
@mutant(ex=2, replaces="mean_estimate", id="first-only", why="uses only the first sample")
def mean_estimate(samples):
    return float(samples[0])
"""

CRASHING_MUTANT = """
@mutant(ex=2, replaces="mean_estimate", id="crashes", why="a broken mutant")
def mean_estimate(samples):
    return undefined_name
"""


def abs_error(mean_estimate, *, seed, n):
    """Lower is better."""
    rng = np.random.default_rng(seed)
    return abs(mean_estimate(rng.normal(TRUE_MEAN, 1.0, size=n)) - TRUE_MEAN)


def closeness(mean_estimate, *, seed, n):
    """Higher is better."""
    return 1.0 - abs_error(mean_estimate, seed=seed, n=n)


def make_check(metric=abs_error, higher_is_better=False, min_gap=0.1):
    return StochasticCheck(
        ex=2,
        functions="mean_estimate",
        metric=metric,
        live={"n": 1600},
        quick={"n": 400},
        higher_is_better=higher_is_better,
        min_gap=min_gap,
        what="error of the mean",
    )


@pytest.fixture
def lab(tmp_path):
    """A built synthetic notebook and a mutants folder; returns (notebook, mutants_dir)."""
    nb = nbformat.v4.new_notebook()
    for cid, kind, src in CELLS:
        new = nbformat.v4.new_code_cell if kind == "code" else nbformat.v4.new_markdown_cell
        cell = new(src)
        cell["id"] = cid
        nb.cells.append(cell)
    nb.metadata["prl"] = {"module": "m99", "content_sha": "0123456789abcdef"}
    notebook = tmp_path / "99-synthetic.ipynb"
    nbformat.write(nb, notebook)
    mutants = tmp_path / "mutants"
    mutants.mkdir()
    (mutants / "m99.py").write_text(textwrap.dedent(GOOD_MUTANTS))
    return notebook, mutants


def run(lab, check, *, budget="quick", n_seeds=100):
    notebook, mutants = lab
    return tp.run_protocol(
        check,
        check_name="check_mean",
        notebook=notebook,
        budget=budget,
        n_seeds=n_seeds,
        mutants_dir=mutants,
        progress=lambda _line: None,
    )


def test_namespace_stops_before_the_exercise_and_skips_checkpoints(lab):
    ns, executed = tp.solution_namespace(lab[0], "2")
    ids = [cid for cid, _ in executed]
    assert ids == ["gen-setup", "gen-init", "c001", "ex1-stub", "ex1-sol", "ex1-run"] + [
        "ex2-stub",
        "ex2-sol",
    ]
    assert ns["RAN_EX1"] == 3.0  # earlier run cells run
    assert ns["mean_estimate"].__prl_ref__ == ("2", "mean_estimate")  # the bound solution


def test_lower_is_better(lab):
    res = run(lab, make_check())
    a = res.analysis
    assert a.ok, a.problems
    assert set(res.per_seed) == {"solution", "mutant:biased", "mutant:halved"}
    assert a.strongest == "mutant:biased"  # its 1st percentile is the lowest
    # The solution's error is |N(0, 0.05^2)| (n = 400), so its 99th percentile is near
    # 2.58 x 0.05 = 0.129. Over 100 seeds the sample p99 falls in (0.08, 0.18) with
    # probability above 0.999 (binomial tail counts beyond 1.6 and 3.6 sd).
    assert 0.08 < a.solution_tail < 0.18
    assert a.solution_tail < a.threshold < a.tails["mutant:biased"]
    assert abs(a.threshold - (a.solution_tail + a.tails[a.strongest]) / 2) <= a.gap / 10
    assert a.solution_failures == [] and all(not s for s in a.escapes.values())
    assert a.flake_bound == pytest.approx(0.03)


def test_higher_is_better(lab):
    a = run(lab, make_check(metric=closeness, higher_is_better=True)).analysis
    assert a.ok, a.problems
    assert a.strongest == "mutant:biased"  # its 99th percentile is the highest
    assert a.tails["mutant:biased"] < a.threshold < a.solution_tail
    # The mirror image of the lower-is-better case.
    lower = run(lab, make_check()).analysis
    assert a.threshold == pytest.approx(1 - lower.threshold, abs=max(a.gap, lower.gap) / 5)


def test_gap_below_the_stated_minimum_fails(lab):
    a = run(lab, make_check(min_gap=0.5)).analysis
    assert not a.ok and a.gap < 0.5
    assert any("below the stated minimum 0.5" in p for p in a.problems)


def test_overlapping_mutant_fails_loudly(lab):
    notebook, mutants = lab
    with (mutants / "m99.py").open("a") as fh:
        fh.write(textwrap.dedent(OVERLAPPING_MUTANT))
    a = run(lab, make_check()).analysis
    assert not a.ok and a.strongest == "mutant:first-only" and a.gap < 0
    assert any("passes the threshold" in p for p in a.problems)


def test_seed_floor(lab):
    with pytest.raises(tp.ProtocolError, match="at least 100 seeds at the quick budget"):
        run(lab, make_check(), budget="quick", n_seeds=99)
    with pytest.raises(tp.ProtocolError, match="at least 20 seeds at the live budget"):
        run(lab, make_check(), budget="live", n_seeds=19)
    res = run(lab, make_check(), budget="live", n_seeds=20)
    assert res.analysis.ok and res.settings == {"n": 1600}


def test_a_crashing_mutant_is_not_a_rejection(lab):
    with (lab[1] / "m99.py").open("a") as fh:
        fh.write(textwrap.dedent(CRASHING_MUTANT))
    with pytest.raises(tp.ProtocolError, match="A crash is not a rejection"):
        run(lab, make_check())


def test_no_wrong_version_is_refused(lab):
    (lab[1] / "m99.py").write_text("")
    with pytest.raises(tp.ProtocolError, match="at least one wrong version"):
        run(lab, make_check())


def test_record_validates_and_rederives(lab, tmp_path, capsys):
    res = run(lab, make_check())
    rec = tp.build_record(res)
    assert record.validate(rec) == []
    assert add_run_record.check(rec, variables.load()) == []
    assert rec["kind"] == "experiment" and rec["seeds"] == list(range(100))
    assert set(rec["per_seed"]) == {"solution", "mutant:biased", "mutant:halved"}
    assert rec["args"]["notebook"] == "99-synthetic.ipynb"  # never an absolute path
    assert rec["budget"]["settings"] == {"n": 400} and rec["budget"]["quick"] is True
    path = tp.write_record(rec, tmp_path / "inbox")
    assert path.name == record.record_filename(rec)
    assert json.loads(path.read_text()) == rec
    capsys.readouterr()
    assert tp.main(["--from-record", str(path)]) == 0
    out = capsys.readouterr().out
    assert f'"quick": ({res.analysis.threshold!r}, "threshold protocol, 100 seeds: runs/' in out
    assert "N = 100 seeds" in out and "3/N = 3.0%" in out


def test_cli_refuses_a_check_that_is_not_stochastic(capsys):
    assert tp.main(["m01", "check_discounted_return", "--budget", "quick"]) == 1
    assert "not a StochasticCheck" in capsys.readouterr().err
    assert tp.main(["m01", "check_discounted_return", "--budget", "live", "--seeds", "5"]) == 1
    assert "at least 20 seeds" in capsys.readouterr().err


def test_cli_end_to_end(lab, tmp_path, monkeypatch, capsys):
    notebook, mutants = lab
    monkeypatch.setattr(tp, "load_check", lambda module, name: make_check())
    monkeypatch.setattr(tp, "notebook_for", lambda module: notebook)
    monkeypatch.setattr(tp, "ensure_current", lambda nb: None)
    monkeypatch.setattr(tp.check_notebooks, "MUTANTS", mutants)
    inbox = tmp_path / "inbox"
    argv = ["m99", "check_mean", "--budget", "quick", "--inbox", str(inbox)]
    assert tp.main([*argv, "--dry-run"]) == 0
    assert "Dry run" in capsys.readouterr().out and not inbox.exists()
    assert tp.main(argv) == 0
    out = capsys.readouterr().out
    (path,) = inbox.glob("*.json")
    assert path.name.endswith("-threshold-m99-check_mean-quick-n100-exp.json")
    assert "Paste into prl/checks/m99.py" in out and f"runs/{path.name}" in out


def test_ensure_current_and_notebook_for(tmp_path):
    pytest.importorskip("jupytext")  # colab-sim has none; the protocol then skips this check
    notebook = tp.notebook_for("m01")
    tp.ensure_current(notebook)  # the drift gate keeps committed notebooks current
    stale = tmp_path / notebook.name
    stale.write_text(notebook.read_text().replace("discounted_return", "discounted_sum", 1))
    with pytest.raises(tp.ProtocolError, match="out of date"):
        tp.ensure_current(stale)
    with pytest.raises(tp.ProtocolError, match="unknown module"):
        tp.notebook_for("m98")


def test_namespace_of_a_real_lab():
    # M1's run cells call plt.show(). An earlier test may have imported pyplot without
    # MPLBACKEND (macOS then picks an interactive backend, and show() blocks). The namespace
    # build must switch to Agg whatever pyplot already has.
    import matplotlib.pyplot as plt

    with tp._environment(PRL_QUICK="1", PRL_WORKED="1"):
        ns, executed = tp.solution_namespace(tp.notebook_for("m01"), "4")
    assert plt.get_backend().lower() == "agg"
    ids = [cid for cid, _ in executed]
    assert ids[-2:] == ["ex4-stub", "ex4-sol"] and not any("-chk" in i for i in ids)
    assert ns["evaluate_exact"].__prl_ref__ == ("4", "evaluate_exact")
    assert ns["lab"]._s["checks"] == []  # no checkpoint ran
