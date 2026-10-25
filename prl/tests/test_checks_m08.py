"""prl.checks.m08: the fixtures' consistency, the checks on the lab's reference code and on
wrong versions, and checkpoint 4.2's declaration.

CI's test job has no torch, so the tests that call a check skip there; the notebook jobs
(worked, learner and verify modes, with torch) run the checks on the lab itself, and
scripts/threshold_protocol.py measures checkpoint 4.2.
"""

import dataclasses
import re
import sys
from pathlib import Path

import numpy as np
import pytest

from prl.checks import CheckFailed, StochasticCheck, expected, m08

ROOT = Path(__file__).resolve().parents[2]
SHARED = ROOT / "labs" / "shared"
MUTANTS = ROOT / "labs" / "mutants" / "m08.py"


# ------------------------------------------------------------------ fixtures (NumPy only)
def test_buffer_fixture_is_the_ring_after_seven_writes():
    fx = expected("m08_buffer")
    assert fx["obs0"].tolist() == [5, 6, 2, 3, 4]
    assert (int(fx["ptr"]), int(fx["size"])) == (2, 5)
    assert fx["terminated"].tolist() == [0, 1, 0, 0, 0]
    assert fx["truncated"].tolist() == [0, 0, 0, 1, 0]
    assert np.array_equal(fx["next_obs0"], fx["obs0"] + 1)


def test_td_target_fixture_bootstraps_truncated_rows_only():
    fx = expected("m08_td_target")
    for gamma, y in zip(fx["gammas"], fx["y"], strict=True):
        assert np.allclose(y, fx["rew"] + gamma * (1 - fx["terminated"]) * fx["q_next"])
    cut = (fx["truncated"] > 0) & (fx["terminated"] < 1)
    both = (fx["truncated"] > 0) & (fx["terminated"] > 0)
    assert cut.sum() == 2 and both.sum() == 1  # rows 2 and 4 bootstrap; row 5 does not


def test_double_dqn_fixture_differs_from_each_wrong_target():
    fx = expected("m08_double_dqn")
    assert np.allclose(fx["y"], [3.7, 6.4, 1.8, 2.0])
    for key in ("y_dqn", "y_online", "y_at_s"):
        assert np.abs(fx[key] - fx["y"]).max() > 0.5


def test_mc_return_fixture_is_the_backward_recursion():
    fx = expected("m08_mc_return")
    starts = np.concatenate([[0], np.cumsum(fx["lengths"])])
    for i, gamma in enumerate(fx["gammas"]):
        r = fx["rewards"][starts[i] : starts[i + 1]]
        g = fx["returns"][starts[i] : starts[i + 1]]
        assert np.isclose(g[-1], r[-1])
        assert np.allclose(g[:-1], r[:-1] + gamma * g[1:])


def test_chain_fixture_snapshots_follow_the_first_gradient_step():
    fx = expected("m08_chain")
    assert fx["snap_at"].tolist() == [int(fx["learning_starts"]) + 1, 40, 60]
    assert int(fx["steps"]) % int(fx["episode_length"]) == 0
    assert int(fx["target_sync"]) == int(fx["log_every"])  # every log point follows a copy


def test_sb3_call_fixture_has_a_verdict_per_budget():
    fx = expected("m08_sb3_call")
    assert set(fx) >= {"live", "quick"}
    assert str(fx["live"]) in m08.CALLS and str(fx["quick"]) in m08.CALLS


# ------------------------------------------------------------------ 4.2's declaration
def test_train_dqn_check_is_declared_with_both_thresholds():
    c = m08.check_train_dqn
    assert isinstance(c, StochasticCheck)
    assert c.functions == ("train_dqn", "make_cartpole", "CONFIG") and c.higher_is_better
    assert c.live == c.quick == {"steps": 20_000}  # C33
    assert set(c.thresholds) == {"live", "quick"}
    for budget, (threshold, provenance) in c.thresholds.items():
        record = re.search(r"runs/(\S+\.json)", provenance).group(1)
        assert (ROOT / "runs" / record).exists(), f"{budget}: {record} is not in runs/"
        assert f"-{budget}-" in record
        assert c.min_gap < threshold < 500  # an improvement in greedy return, in steps


# ------------------------------------------------------------------ the checks (need torch)
@pytest.fixture(scope="module")
def lab():
    """The lab's provided code and reference solutions (labs/shared), as one namespace."""
    torch = pytest.importorskip("torch")
    gym = pytest.importorskip("gymnasium")
    if not (SHARED / "dqn.py").exists():
        pytest.skip("labs/shared is not next to prl")
    ns: dict = {}
    exec((SHARED / "dqn_setup.py").read_text(), ns)  # noqa: S102
    exec((SHARED / "dqn.py").read_text(), ns)  # noqa: S102
    exec("import numpy as np\nimport torch\nfrom torch import nn", ns)  # noqa: S102
    exec(  # noqa: S102
        "class ColleagueBuffer(ReplayBuffer):\n"
        "    def sample(self, batch_size, rng):\n"
        "        return to_batch(self, rng.integers(0, self.size, size=batch_size))\n",
        ns,
    )
    ns["buffer_diagnosis"] = lambda: "b"
    ns["make_cartpole"] = lambda: gym.make("CartPole-v1")
    torch.manual_seed(0)
    return ns


CHECKS = {
    "ReplayBuffer": lambda ns: m08.check_replay_buffer(ns["ReplayBuffer"]),
    "buffer_diagnosis": lambda ns: m08.check_buffer_diagnosis(ns["buffer_diagnosis"]),
    "ColleagueBuffer": lambda ns: m08.check_colleague_buffer(ns["ColleagueBuffer"]),
    "td_target": lambda ns: m08.check_td_target(ns["td_target"]),
    "train_dqn": lambda ns: m08.check_train_dqn_masks(ns["train_dqn"], ns["CONFIG"]),
    "double_dqn_target": lambda ns: m08.check_double_dqn_target(ns["double_dqn_target"]),
    "mc_return": lambda ns: m08.check_mc_return(ns["mc_return"]),
}


@pytest.mark.parametrize("name", sorted(CHECKS))
def test_exact_checks_accept_the_reference(lab, name):
    CHECKS[name](lab)


def _mutants():
    if not MUTANTS.exists():
        return []
    sys.path.insert(0, str(ROOT / "scripts"))
    import check_notebooks

    return [
        pytest.param(m, id=m["id"])
        for m in check_notebooks.load_mutants("m08", MUTANTS.parent)
        if set(m["sources"]) <= set(CHECKS)
    ]


@pytest.mark.parametrize("mutant", _mutants())
def test_exact_checks_reject_every_mutant(lab, mutant):
    ns = dict(lab)
    for src in mutant["sources"].values():
        exec(src, ns)  # noqa: S102
    (name,) = mutant["sources"]
    with pytest.raises(CheckFailed):
        CHECKS[name](ns)


def test_train_dqn_masks_names_a_missing_zero_grad(lab):
    ns = dict(lab)
    src = (SHARED / "dqn.py").read_text()
    body = re.search(r"^def train_dqn\(.*?(?=^def |\Z)", src, re.S | re.M).group(0)
    exec(body.replace("p.opt.zero_grad()\n", "pass\n"), ns)  # noqa: S102
    with pytest.raises(CheckFailed, match="zero_grad"):
        m08.check_train_dqn_masks(ns["train_dqn"], ns["CONFIG"])


def test_train_dqn_masks_names_a_hard_coded_step_size(lab):
    ns = dict(lab)
    real = ns["train_dqn"]

    def fixed_lr(make_env, config, seed, steps, **kw):
        return real(make_env, dataclasses.replace(config, lr=1e-3), seed, steps, **kw)

    with pytest.raises(CheckFailed, match="config.lr"):
        m08.check_train_dqn_masks(fixed_lr, ns["CONFIG"])


def test_buffer_diagnosis_wants_one_letter():
    with pytest.raises(CheckFailed, match="one letter"):
        m08.check_buffer_diagnosis(lambda: "bc")
    with pytest.raises(CheckFailed, match="returned 'd'"):
        m08.check_buffer_diagnosis(lambda: "D")
    m08.check_buffer_diagnosis(lambda: " b ")


def test_sb3_call_rejects_a_word_outside_the_rule(monkeypatch):
    monkeypatch.setenv("PRL_QUICK", "1")
    with pytest.raises(CheckFailed, match="should return"):
        m08.check_sb3_call(lambda: "SB3")
    want = str(expected("m08_sb3_call")["quick"])
    m08.check_sb3_call(lambda: want)
    with pytest.raises(CheckFailed, match="not what the rule says"):
        m08.check_sb3_call(lambda: next(c for c in m08.CALLS if c != want))


def test_greedy_improvement_needs_the_logged_evaluations(lab):
    cfg = lab["CONFIG"]
    with pytest.raises(CheckFailed, match="greedy evaluations"):
        m08._greedy_improvement(lambda *a, **k: (None, {}), None, cfg, seed=0, steps=2000)

    def wrong_grid(make_env, config, seed, steps, **kw):
        return None, {"eval_step": np.array([5, 10]), "eval_return": np.array([9.0, 9.0])}

    with pytest.raises(CheckFailed, match="should be at steps"):
        m08._greedy_improvement(wrong_grid, None, cfg, seed=0, steps=2000)


def test_greedy_improvement_is_the_best_evaluation_above_the_first(lab):
    def fake(make_env, config, seed, steps, *, eval_every, **kw):
        assert eval_every == config.learning_starts  # the first one precedes any gradient step
        grid = np.arange(eval_every, steps + 1, eval_every)
        returns = np.full(len(grid), 40.0)
        returns[0], returns[7] = 130.0, 200.0  # a lucky untrained network, then learning
        return None, {"eval_step": grid, "eval_return": returns}

    assert m08._greedy_improvement(fake, None, lab["CONFIG"], seed=0, steps=20_000) == 70.0
