"""prl.checks.m03: the helpers the stochastic checks use, and the exact checks' guards.

The checks' behavior on the lab's solutions and mutants is tested where those live:
scripts/check_notebooks.py --mode verify (exact checks) and scripts/threshold_protocol.py
(stochastic checks). Here: the parts that do not need the lab.
"""

import numpy as np
import pytest

from prl.checks import CheckFailed, StochasticCheck, expected, m03


def test_route_episodes_are_seeded_fresh_and_well_formed():
    a, b = m03.route_episodes(7, 50), m03.route_episodes(7, 50)
    assert a == b and a is not b and a[0].states is not b[0].states  # same data, fresh lists
    assert a != m03.route_episodes(8, 50)
    for ep in a:
        assert ep.states[0] == m03.START and ep.states[-1] == m03.GOAL and ep.terminated
        assert len(ep.states) == len(ep.rewards) + 1 == len(ep.actions) + 1
        assert not ep.truncated


def test_route_returns_average_to_the_exact_value():
    # The discounted return from the start is an unbiased sample of V_pi(start). Over 2000
    # seeded episodes the sample mean must lie within 4 standard errors of the fixture's
    # exact value (a correct sampler fails this with probability below 1e-4).
    returns = []
    for ep in m03.route_episodes(123, 2000):
        g = 0.0
        for r in reversed(ep.rewards):
            g = r + m03.GAMMA * g
        returns.append(g)
    returns = np.array(returns)
    exact = expected("m03_prediction")["V_pi"][m03.START]
    assert exact == pytest.approx(-15.366, abs=5e-4)  # the value the page states
    se = returns.std(ddof=1) / np.sqrt(len(returns))
    assert abs(returns.mean() - exact) < 4 * se


def test_greedy_path_follows_argmax_and_detects_loops():
    Q = np.zeros((m03.S, m03.A))
    path, reached = m03.greedy_path(Q)  # argmax 0 = up: up the left column, then stuck
    assert not reached and path[:4] == [36, 24, 12, 0] and path[-1] == 0
    Q[36, 0] = 1  # up once,
    Q[24, 1] = Q[25:35, 1] = 1  # right along row 2,
    Q[35, 2] = 1  # then down into the goal: the 13-move edge route
    path, reached = m03.greedy_path(Q)
    assert reached and len(path) - 1 == m03.OPTIMAL_MOVES


@pytest.mark.parametrize(
    "check, bad",
    [
        (m03.check_mc_prediction, lambda *a: None),
        (m03.check_mc_prediction, lambda *a: np.zeros(4)),
        (m03.check_td0_prediction, lambda *a, **k: np.full(5, np.nan)),
        (m03.check_n_step_target, lambda *a: None),
        (m03.check_n_step_td_prediction, lambda *a: [0.0] * 5),
        (m03.check_bias2_variance, lambda *a: None),
        (m03.check_bias2_variance, lambda *a: (0.0, 0.0, 0.0)),
        (m03.check_bias2_variance, lambda *a: (float("inf"), 0.0)),
    ],
)
def test_exact_checks_reject_wrong_answers(check, bad):
    with pytest.raises(CheckFailed):
        check(bad)


def _learner(Q, online_len):
    def learn(env, episodes, alpha, epsilon, gamma, seed, q0=0.0):
        return Q, np.zeros(online_len if online_len is not None else episodes)

    return learn


@pytest.mark.parametrize(
    "learner, match",
    [
        (lambda *a, **k: None, "pair"),
        (_learner(np.zeros((48, 3)), None), "shape"),
        (_learner(np.zeros((48, 4)), 7), "one reward sum per episode"),
    ],
)
def test_control_metrics_validate_the_return_value(learner, match):
    with pytest.raises(CheckFailed, match=match):
        m03.check_sarsa.measure(
            learner, seed=0, budget={"control_episodes": 3, "control_alpha": 0.1}
        )


def test_control_metric_scores_a_learner_that_never_reaches_the_goal(capsys):
    value = m03.check_q_learning_path.measure(
        _learner(np.zeros((48, 4)), None),
        seed=0,
        budget={"control_episodes": 2, "control_alpha": 0.1},
    )
    assert value == m03.S and "does not reach the goal" in capsys.readouterr().out


def test_stochastic_checks_are_declared_with_both_thresholds():
    checks = {k: v for k, v in vars(m03).items() if isinstance(v, StochasticCheck)}
    assert set(checks) == {
        "check_mc_rms",
        "check_td0_rms",
        "check_sarsa",
        "check_q_learning_path",
        "check_cliff_contrast",
        "check_q_learning_suspect",
    }
    for name, check in checks.items():
        assert set(check.thresholds) == {"live", "quick"}, name
        for threshold, provenance in check.thresholds.values():
            assert np.isfinite(threshold) and provenance.startswith("threshold protocol"), name
