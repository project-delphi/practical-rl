"""prl.checks.m04: the provided plumbing, the fixtures' consistency, and the checks' guards.

The checks' behavior on the lab's solutions and mutants is tested where those live:
scripts/check_notebooks.py --mode verify (exact checks) and scripts/threshold_protocol.py
(checkpoint 3.3). Here: the parts that do not need the lab.
"""

import ast
from pathlib import Path

import numpy as np
import pytest

from prl.checks import CheckFailed, StochasticCheck, expected, m04

LAB = Path(__file__).resolve().parents[2] / "labs" / "src" / "04-bandits.py"


def test_argmax_tie_takes_the_first_maximizer_within_rounding():
    assert m04.argmax_tie([0.2, 0.5, 0.5]) == 1
    assert m04.argmax_tie([0.5, 0.5 + 1e-12, 0.1]) == 0  # within 1e-9 (1 + max|x|): a tie
    assert m04.argmax_tie([0.5, 0.5 + 1e-6, 0.1]) == 1  # a real difference
    assert m04.argmax_tie([3.0, np.inf, np.inf]) == 1  # an infinite entry wins at once
    assert m04.argmax_tie(np.array([0, 0, 1], dtype=bool)) == 2


def test_reward_stack_is_seeded_apart_from_the_policy_and_extends_by_rows():
    a = m04.reward_stack(m04.R_MEAN, 200, 3)
    assert a.shape == (200, 3) and set(np.unique(a)) <= {0.0, 1.0}
    assert np.array_equal(a, m04.reward_stack(m04.R_MEAN, 200, 3))
    assert not np.array_equal(a, m04.reward_stack(m04.R_MEAN, 200, 4))
    # A longer stack starts with the shorter one, so the first 5000 rounds of a 20 000-round
    # run are the 5000-round run (the figure script relies on this).
    assert np.array_equal(m04.reward_stack(m04.R_MEAN, 500, 3)[:200], a)
    assert np.array_equal(m04.reward_stack(m04.R_MEAN, 200, 3, scale=10.0), 10 * a)
    # Each column's mean is near its rate (2000 draws: 5 standard errors is about 0.055).
    means = m04.reward_stack(m04.R_MEAN, 2000, 11).mean(axis=0)
    assert np.all(np.abs(means - m04.R_MEAN) < 0.055)


def _functions(path: Path) -> dict[str, str]:
    """Each top-level function's AST without its docstring and annotations."""
    out = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.FunctionDef):
            node.returns = None
            for arg in node.args.args:
                arg.annotation = None
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                node.body = body[1:]
            out[node.name] = ast.dump(node)
    return out


@pytest.mark.skipif(not LAB.exists(), reason="the lab source is not next to prl")
def test_the_plumbing_equals_the_labs_provided_cells():
    lab, here = _functions(LAB), _functions(Path(m04.__file__))
    for name in ("argmax_tie", "reward_stack"):
        assert lab[name] == here[name], f"{name} differs from the lab's provided cell"


@pytest.mark.parametrize(
    "name, seeds", [("m04_greedy_replay", (1, 17)), ("m04_ucb1_replay", (None,))]
)
def test_replays_are_consistent_runs_on_their_stacks(name, seeds):
    fx = expected(name)
    for seed in seeds:
        sfx = "" if seed is None else f"_{seed}"
        arms, counts, sums = (fx[f"{k}{sfx}"] for k in ("arms", "counts_before", "sums_before"))
        stack = m04.reward_stack(
            m04.R_MEAN, len(arms), seed if seed is not None else int(fx["seed"])
        )
        assert not counts[0].any() and not sums[0].any()
        for t in range(len(arms) - 1):
            a = arms[t]
            step = np.eye(m04.N_ARMS, dtype=int)[a]
            assert np.array_equal(counts[t + 1], counts[t] + step)
            assert np.allclose(sums[t + 1], sums[t] + step * stack[counts[t, a], a])
    if name == "m04_ucb1_replay":
        assert fx["final_counts"].tolist() == [126, 111, 763]
        assert np.array_equal(fx["final_counts"], np.bincount(fx["arms"], minlength=3))
        assert np.allclose(fx["rewards"].sum(), fx["final_sums"].sum())


def test_exact_shares_and_reference_shares_agree():
    fx = expected("m04_ts_shares")
    assert np.allclose(fx["exact"].sum(axis=1), 1.0)
    assert np.all(np.abs(fx["reference"] - fx["exact"]) < 0.02)
    # Every wrong version the check names is far outside its 0.04 tolerance on some state.
    for wrong in ("posterior_mean", "sample_reward"):
        assert np.max(np.abs(fx[wrong] - fx["exact"])) > 0.3


def test_the_scale_reference_run_gives_the_directions():
    fx = expected("m04_scale")
    want = dict(zip(fx["keys"].tolist(), fx["directions"].tolist(), strict=True))
    assert want == {"eps_greedy": "same", "ucb1": "changes", "ucb1_rescaled": "same"}
    assert fx["same_arms_seeds"].tolist() == [20, 0, 20]
    assert np.allclose(fx["eps_greedy_x1"], fx["eps_greedy_x10"], rtol=0, atol=1e-9)
    assert np.allclose(fx["ucb1_x1"], fx["ucb1_rescaled_x10"], rtol=0, atol=1e-9)


def test_ucb1_for_range_fixture_is_ucb1_scaled_to_the_range():
    fx = expected("m04_ucb1_for_range")
    b1 = int(np.flatnonzero(fx["B"] == 1.0)[0])  # B = 1 is UCB1's own index
    assert np.allclose(fx["index"][b1], expected("m04_ucb1")["index"][3], rtol=0, atol=1e-12)
    for i, (c, s, t, B) in enumerate(zip(fx["counts"], fx["sums"], fx["t"], fx["B"], strict=True)):
        seen = c > 0
        assert np.all(np.isposinf(fx["index"][i][~seen]))
        if seen.any():
            bonus = B * np.sqrt(2 * np.log(t) / c[seen])
            assert np.allclose(fx["index"][i][seen], s[seen] / c[seen] + bonus, rtol=1e-12)


def _loop(arms_out=None, regret_out=None, t_shift=0, new_rng=False):
    """A stand-in run_bandit with one deliberate fault, for the guards (not a solution)."""

    def run(policy, stack, r_mean, seed):
        rng = np.random.default_rng(seed)
        T, n = stack.shape
        counts, sums, arms = np.zeros(n, dtype=int), np.zeros(n), np.zeros(T, dtype=int)
        for t in range(T):
            g = np.random.default_rng(seed) if new_rng else rng
            a = policy(counts, sums, t + t_shift, g)
            sums[a] += stack[counts[a], a]
            counts[a] += 1
            arms[t] = a
        gaps = np.max(r_mean) - np.asarray(r_mean)
        out_arms = arms if arms_out is None else arms_out(arms)
        return out_arms, (np.cumsum(gaps[arms]) if regret_out is None else regret_out(arms))

    return run


@pytest.mark.parametrize(
    "run, match",
    [
        (lambda *a: None, "returned None"),
        (lambda *a: (np.zeros(3, dtype=int),), "pair"),
        (_loop(arms_out=lambda a: a[:-1]), "shape"),
        (_loop(arms_out=lambda a: a.astype(float)), "integers"),
        (_loop(t_shift=1), "number of rounds played"),
        (_loop(new_rng=True), "different generator"),
        (_loop(regret_out=lambda a: np.zeros(len(a))), "gap of the arm pulled"),
    ],
)
def test_check_run_bandit_names_the_fault(run, match):
    with pytest.raises(CheckFailed, match=match):
        m04.check_run_bandit(run)


def test_check_run_bandit_accepts_a_correct_loop():
    m04.check_run_bandit(_loop())  # the stand-in without a fault is the loop the lab describes


@pytest.mark.parametrize(
    "check, bad",
    [
        (m04.check_eps_greedy, lambda *a, **k: None),
        (m04.check_eps_greedy, lambda *a, **k: 1.0),
        (m04.check_eps_greedy, lambda *a, **k: 3),
        (m04.check_ucb1_index, lambda *a: None),
        (m04.check_ucb1_index, lambda *a: np.full(3, np.nan)),
        (m04.check_ucb1_index, lambda *a: np.zeros(2)),
        (m04.check_ucb1_counts, lambda *a: np.full(3, np.nan)),
        (m04.check_beta_posterior, lambda *a: None),
        (m04.check_beta_posterior, lambda *a: (np.ones(3),)),
        (m04.check_thompson_choices, lambda *a: True),
        (m04.check_thompson_choices, lambda c, s, t, rng: int(rng.integers(3))),
        (m04.check_scale_directions, lambda: None),
        (m04.check_scale_directions, lambda: {"eps_greedy": "same"}),
        (
            m04.check_scale_directions,
            lambda: {"eps_greedy": "same", "ucb1": "up", "ucb1_rescaled": "same"},
        ),
        (
            m04.check_scale_directions,
            lambda: {"eps_greedy": "same", "ucb1": "same", "ucb1_rescaled": "same"},
        ),
        (m04.check_ucb1_for_range, lambda *a: None),
        (m04.check_ucb1_for_range, lambda c, s, t, B: np.full(3, np.nan)),
        (
            m04.check_ucb1_for_range,
            lambda c, s, t, B: np.where(c > 0, s / np.maximum(c, 1), np.inf),
        ),
        (m04.check_gaussian_posterior, lambda *a: None),
    ],
)
def test_exact_checks_reject_wrong_answers(check, bad):
    with pytest.raises(CheckFailed):
        check(bad)


def test_check_scale_directions_does_not_grade_up_or_down():
    with pytest.raises(CheckFailed, match="grades only whether the choices change"):
        m04.check_scale_directions(
            lambda: {"eps_greedy": "same", "ucb1": "up", "ucb1_rescaled": "same"}
        )
    with pytest.raises(CheckFailed, match="pulled the same arms on 0 of 20 seeds") as err:
        m04.check_scale_directions(
            lambda: {"eps_greedy": "same", "ucb1": "same", "ucb1_rescaled": "same"}
        )
    assert " up" not in str(err.value) and "down" not in str(err.value)


def test_check_verdict_checks_the_late_pair_when_given():
    def rule(a, b):  # compare()'s winner: the rule itself, for this test only
        from prl.evaluate import compare

        return compare(a, b, higher_is_better=False).winner

    rng = np.random.default_rng(3)
    final = {m: rng.normal(mu, 5, 20) for m, mu in (("TS", 20), ("UCB1", 110), ("eps-greedy", 60))}
    late = {"UCB1": rng.normal(180, 10, 20), "eps-greedy": rng.normal(210, 10, 20)}
    m04.check_verdict(rule, final, late)
    flips = {"n": 0}

    def wrong_late(a, b):  # right on the fixed cases and on final, wrong on the late pair
        if np.mean(a) > 150 or np.mean(b) > 150:
            flips["n"] += 1
            return "tie"
        return rule(a, b)

    m04.check_verdict(wrong_late, final)  # without late it cannot see the mistake
    with pytest.raises(CheckFailed, match="late"):
        m04.check_verdict(wrong_late, final, late)
    assert flips["n"] >= 1


def test_check_verdict_rejects_a_word_outside_the_rule():
    with pytest.raises(CheckFailed, match="one of"):
        m04.check_verdict(lambda a, b: "maybe", {})
    with pytest.raises(CheckFailed, match="higher|lower"):
        m04.check_verdict(lambda a, b: "single seed" if min(len(a), len(b)) == 1 else "B", {})


class _Frozen:
    """The planted bug's shape: counts that never grow."""

    def __init__(self, n_arms):
        self.counts = np.zeros(n_arms)
        self.sums = np.zeros(n_arms)

    def act(self, t):
        return t if t < len(self.counts) else 0

    def update(self, arm, reward):
        self.sums[arm] += reward


def test_check_ucb1_suspect_explains_counts_that_never_grow():
    with pytest.raises(CheckFailed, match="never grow"):
        m04.check_ucb1_suspect(_Frozen)


def _fake_run(arm):
    def run(policy, stack, r_mean, seed):
        T = len(stack)
        policy(np.zeros(3, dtype=int), np.zeros(3), 0, np.random.default_rng(seed))
        return np.full(T, arm), np.full(T, -1.0)  # the regret array is ignored

    return run


def test_ts_regret_metric_takes_the_regret_from_the_arms():
    ts = lambda c, s, t, rng: 0  # noqa: E731
    check = m04.check_thompson_regret
    assert check.measure(ts, _fake_run(2), seed=3, budget={"T": 50, "runs": 2}) == 0.0
    assert check.measure(ts, _fake_run(1), seed=3, budget={"T": 50, "runs": 2}) == pytest.approx(
        10.0
    )
    with pytest.raises(CheckFailed, match="shape"):
        check.measure(
            ts, lambda *a: (np.zeros(3, dtype=int), None), seed=0, budget={"T": 50, "runs": 1}
        )


def test_ts_regret_check_is_declared_with_both_thresholds():
    check = m04.check_thompson_regret
    assert isinstance(check, StochasticCheck)
    assert check.functions == ("thompson_step", "run_bandit") and check.ex == "3"
    assert not check.higher_is_better and check.min_gap == 20.0
    assert set(check.thresholds) == {"live", "quick"}
    for threshold, provenance in check.thresholds.values():
        assert threshold > 0 and "runs/" in provenance and "threshold-m04" in provenance
