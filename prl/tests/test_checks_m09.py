"""prl.checks.m09: the fixtures' consistency, the checks on the lab's reference solutions and on
every wrong version in labs/mutants/m09.py, and checkpoint 6.2's declaration.

The lab's solutions are read from labs/src/09-debugging-evaluation.py (its solution cells), so
the tests check the code participants see. Exercises 1, 2, 5 and 6 need NumPy only; Exercise 4
and the stretch need gymnasium; nothing here needs torch (checkpoint 6.2's training metric is
measured by scripts/threshold_protocol.py, and the notebook jobs run the lab itself).
"""

import re
import sys
from pathlib import Path

import numpy as np
import pytest

import prl.data
from prl.checks import CheckFailed, StochasticCheck, expected, m09

ROOT = Path(__file__).resolve().parents[2]
LAB = ROOT / "labs" / "src" / "09-debugging-evaluation.py"
MUTANTS = ROOT / "labs" / "mutants" / "m09.py"
SOLUTION_ROLES = ("ex1-sol", "ex2-sol", "ex3-sol", "ex4-sol", "ex5-sol", "ex6-sol", "stretch-sol")
RECORD_CHECKS = {"gather_check", "target_lag", "q_drift", "td_error_scale"}


def _has_broken_runs() -> bool:
    return "m09_broken_runs" in prl.data.manifest()


needs_broken_runs = pytest.mark.skipif(
    not _has_broken_runs(), reason="the dataset m09_broken_runs is not built on this branch yet"
)


# ------------------------------------------------------------------ fixtures (NumPy only)
def test_gather_fixture_has_one_wrong_row_of_eight():
    fx = expected("m09_gather")
    want = np.take_along_axis(fx["snap_q"], fx["snap_actions"][..., None], axis=-1)[..., 0]
    off = np.abs(fx["snap_gathered"] - want)
    assert (off > 0.5).sum() == 1 and ((off > 0) & (off < 1e-6)).sum() == 1
    assert float(fx["answer"]) == 1 / 8


def test_lag_fixture_target_changes_twice_and_online_always():
    fx = expected("m09_lag")
    changes = [np.any(fx["q_probe_target"][i] != fx["q_probe_target"][i - 1]) for i in range(1, 6)]
    assert changes == [False, True, False, False, True]
    assert all(np.any(fx["q_probe"][i] != fx["q_probe"][i - 1]) for i in range(1, 6))
    assert fx["answer"].tolist() == [0, 500, 0, 500, 1000, 0]
    assert np.issubdtype(fx["answer"].dtype, np.integer)


def test_drift_fixture_separates_discounted_from_undiscounted():
    fx = expected("m09_drift")
    assert not np.allclose(fx["ep_return"], fx["ep_disc_return"])
    assert fx["windows"].tolist() == [600, 50]
    assert np.isnan(fx["answers"][1]).sum() == 3


def test_td_scale_fixture_has_a_negative_q_scale():
    fx = expected("m09_td_scale")
    assert (fx["q_max_mean"] < 0).any()
    assert np.allclose(fx["answer"], fx["td_abs_mean"] / np.abs(fx["q_max_mean"]))


def test_diagnoses_fixture_uses_the_table_keys():
    fx = expected("m09_diagnoses")
    assert set(fx["answers"].tolist()) <= set(fx["causes"].tolist())
    assert len(set(fx["answers"].tolist())) == 4  # one cause per run


def test_probe_fixture_values_are_exact():
    fx = expected("m09_probes")
    assert np.isclose(float(fx["q1_unmasked"]), 1 / (1 - float(fx["gamma"])))
    assert float(fx["q3_first"]) == float(fx["gamma"])


def test_iqm_fixture_is_the_trimmed_mean_on_the_page():
    fx = expected("m09_iqm")
    start = 0
    for n, want in zip(fx["lengths"], fx["answers"], strict=True):
        x = np.sort(fx["x"][start : start + n])
        start += n
        k = int(n) // 4
        assert np.isclose(x[k : n - k].mean(), want, rtol=1e-12)
    assert 11 in fx["lengths"].tolist()  # floor(11/4) = 2, round(11/4) = 3


def test_band_position_fixture_has_ties_at_both_ends():
    fx = expected("m09_band_position")
    ties = [v in (r.min(), r.max()) for v, r in zip(fx["values"], fx["recorded"], strict=True)]
    assert sum(ties) == 2 and all(a == 0 for a, t in zip(fx["answers"], ties, strict=True) if t)


# ------------------------------------------------------------------ 6.2's declaration
def test_live_return_check_is_a_stochastic_check_of_the_lab_budgets():
    c = m09.check_live_return
    assert isinstance(c, StochasticCheck)
    assert c.functions == ("train_dqn", "CONFIG") and c.higher_is_better and c.ex == "6"
    assert c.live == {"steps": 50_000, "eval_every": 5_000, "eval_episodes": 10}
    assert c.quick == {"steps": 10_000, "eval_every": 1_000, "eval_episodes": 10}


def test_live_return_thresholds_name_filed_protocol_records():
    c = m09.check_live_return
    assert set(c.thresholds) == {"live", "quick"}  # one metric at both budgets (decision 16)
    for budget, (threshold, provenance) in c.thresholds.items():
        record = re.search(r"runs/(\S+\.json)", provenance).group(1)
        assert (ROOT / "runs" / record).exists(), f"{budget}: {record} is not in runs/"
        assert f"-{budget}-" in record and c.min_gap < threshold < 500


def test_live_return_metric_is_the_best_computed_evaluation_without_training():
    budget = dict(m09.check_live_return.live)  # 10 evaluations, every 5,000 of 50,000 steps
    evals = np.array([20.0, 150.0, 480.0, 210.0, 90.0, 12.0, 300.0, 250.0, 260.0, 12.3])
    assert m09._live_return(evals, seed=0, **budget) == 480.0  # the best, not the last
    for bad in (evals[:9], np.append(evals[:9], np.nan), "123", None, 123.4):
        with pytest.raises(CheckFailed):
            m09._live_return(bad, seed=0, **budget)


# ------------------------------------------------------------------ the lab's solutions
def _lab_cells() -> dict[str, str]:
    cells, role, body = {}, None, []
    for line in LAB.read_text().splitlines(keepends=True):
        if line.startswith("# %%"):
            if role:
                cells[role] = "".join(body)
            m = re.search(r'role="([^"]+)"', line)
            role, body = (m.group(1) if m else None), []
        else:
            body.append(line)
    if role:
        cells[role] = "".join(body)
    return cells


@pytest.fixture(scope="module")
def lab():
    """The lab's solution cells, exec'd with a stand-in `lab` whose solution() is identity."""
    gym = pytest.importorskip("gymnasium")
    if not LAB.exists():
        pytest.skip("labs/src is not next to prl")
    cells = _lab_cells()
    ns: dict = {"np": np, "gym": gym}
    ns["lab"] = type("Lab", (), {"solution": staticmethod(lambda n: lambda f: f)})()
    for role in SOLUTION_ROLES:
        exec(compile(cells[role], f"<lab {role}>", "exec"), ns)  # noqa: S102
    return ns


CHECKS = {
    "gather_check": lambda ns: m09.check_gather_check(ns["gather_check"]),
    "target_lag": lambda ns: m09.check_target_lag(ns["target_lag"]),
    "q_drift": lambda ns: m09.check_q_drift(ns["q_drift"]),
    "td_error_scale": lambda ns: m09.check_td_error_scale(ns["td_error_scale"]),
    "diagnoses": lambda ns: [m09.check_diagnosis(ns["diagnoses"], r) for r in "ABCD"],
    "ConstantRewardEnv": lambda ns: m09.check_constant_reward_env(ns["ConstantRewardEnv"]),
    "ObsRewardEnv": lambda ns: m09.check_obs_reward_env(ns["ObsRewardEnv"]),
    "iqm": lambda ns: m09.check_iqm(ns["iqm"]),
    "bootstrap_ci": lambda ns: m09.check_bootstrap_ci(ns["bootstrap_ci"]),
    "band_position": lambda ns: m09.check_band_position(ns["band_position"]),
    "TwoStepEnv": lambda ns: m09.check_two_step_env(ns["TwoStepEnv"]),
}


@pytest.mark.parametrize(
    "name",
    [pytest.param(n, marks=needs_broken_runs) if n in RECORD_CHECKS else n for n in sorted(CHECKS)],
)
def test_exact_checks_accept_the_reference(lab, name):
    CHECKS[name](lab)


def _mutants():
    if not MUTANTS.exists():
        return []
    sys.path.insert(0, str(ROOT / "scripts"))
    import check_notebooks

    return [
        pytest.param(m, id=f"{m['ex']}-{m['id']}")
        for m in check_notebooks.load_mutants("m09", MUTANTS.parent)
        if set(m["sources"]) <= set(CHECKS)
    ]


def test_every_mutant_has_an_exact_check_and_the_protocol_has_its_wrong_versions():
    sys.path.insert(0, str(ROOT / "scripts"))
    import check_notebooks

    names = {n for m in check_notebooks.load_mutants("m09", MUTANTS.parent) for n in m["sources"]}
    assert names <= set(CHECKS)
    # 6.2's wrong versions replace provided code, so they are the protocol's alone.
    proto = check_notebooks.load_mutants("m09", MUTANTS.parent / "protocol")
    assert {(m["ex"], m["id"], tuple(m["sources"])) for m in proto} == {
        ("6", "random", ("train_dqn",)),
        ("6", "untrained", ("train_dqn",)),
    }


@pytest.mark.parametrize("mutant", _mutants())
def test_exact_checks_reject_every_mutant(lab, mutant):
    ns = dict(lab)
    for src in mutant["sources"].values():
        exec(src, ns)  # noqa: S102
    (name,) = mutant["sources"]
    with pytest.raises(CheckFailed):
        CHECKS[name](ns)


@pytest.mark.parametrize("run", ["A", "B", "C", "D"])
def test_a_wrong_diagnosis_names_the_run_and_not_the_answer(run):
    fx = expected("m09_diagnoses")
    answer = str(fx["answers"][list(fx["runs"]).index(run)])
    wrong = "seed" if answer != "seed" else "lr"
    out = {r: "seed" for r in "ABCD"} | {run: wrong}
    with pytest.raises(CheckFailed) as err:
        m09.check_diagnosis(lambda: out, run)
    msg = str(err.value)
    assert f"Run {run}" in msg and repr(answer) not in msg


def test_diagnoses_must_be_a_dict_of_the_four_runs():
    with pytest.raises(CheckFailed, match="exactly the keys"):
        m09.check_diagnosis(lambda: {"A": "lr"}, "A")
    with pytest.raises(CheckFailed, match="not one of the table"):
        m09.check_diagnosis(lambda: dict.fromkeys("ABCD", "bug"), "A")


@needs_broken_runs
def test_record_fixture_matches_the_lab_signals_on_the_dataset(lab):
    fx = expected("m09_record")
    d = prl.data.load("m09_broken_runs")
    assert fx["runs"].tolist() == list(m09.RUNS)
    for i, run in enumerate(fx["runs"]):
        for j, seed in enumerate(fx["seeds"]):
            log = m09._run_log(d, str(run), int(seed))
            assert lab["gather_check"](log) == fx["gather_check"][i, j]
            assert np.array_equal(lab["target_lag"](log), fx["target_lag"][i, j])
