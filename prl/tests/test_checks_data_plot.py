import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from prl import data, envs, plot
from prl.checks import (
    CheckFailed,
    assert_close,
    assert_simplex,
    assert_threshold,
    expected,
    m00,
    m01,
)


def correct_return(rewards, gamma):
    g = 0.0
    for r in reversed(rewards):
        g = r + gamma * g
    return g


def test_m01_check_passes_on_solution_and_fails_on_mutants():
    m01.check_discounted_return(correct_return)
    mutants = [
        lambda rs, g: sum(r * g ** (k + 1) for k, r in enumerate(rs)),  # discounts the first reward
        lambda rs, g: sum(rs),  # ignores gamma
        lambda rs, g: None,  # forgot return
    ]
    for mutant in mutants:
        with pytest.raises(CheckFailed):
            m01.check_discounted_return(mutant)


def test_assert_helpers():
    assert_close([1.0, 2.0], [1.0, 2.0])
    with pytest.raises(CheckFailed, match="off by"):
        assert_close([1.0], [1.1], atol=1e-3)
    assert_simplex(np.full((2, 3), 1 / 3))
    with pytest.raises(CheckFailed):
        assert_simplex([0.5, 0.6])
    with pytest.raises(CheckFailed, match="threshold from"):
        assert_threshold(10, 20, provenance="runs/x.json")


def test_m00_checks():
    m00.check_episode_returns([10, 20, 500], 3)
    with pytest.raises(CheckFailed):
        m00.check_episode_returns([0, 20, 501], 3)
    m00.check_seed_bands(np.array([[1, 2], [3, 4]]), 2, 2)
    with pytest.raises(CheckFailed):
        m00.check_seed_bands(np.array([[1, 2], [1, 2]]), 2, 2)


def test_expected_fixture_loads():
    fx = expected("m01_returns")
    assert set(fx) == {"rewards", "lengths", "gammas", "returns"}


def test_generate_logs_has_propensities():
    g = envs.CliffGridworld(3, 4)
    behavior = np.full((g.n_states, g.n_actions), 0.25)
    logs = data.generate_logs(g, behavior, episodes=5, horizon=30, seed=0)
    assert len(logs) > 0 and np.allclose(logs.behavior_prob, 0.25)
    assert logs.truncated.sum() + logs.terminated.sum() == 5


def test_unknown_dataset():
    with pytest.raises(KeyError, match="Unknown dataset"):
        data.load("nope")


def test_plot_helpers_draw():
    import matplotlib.pyplot as plt

    plot.style()
    fig, ax = plt.subplots()
    plot.curves(np.arange(5), {"a": np.random.default_rng(0).normal(size=(3, 5))}, ax=ax)
    band = {"x": np.arange(5), "runs": np.ones((3, 5)), "budget": {"steps": 5}, "record": "r"}
    plot.with_recorded(band, np.ones(5), x=np.arange(5), your_budget={"steps": 5}, ax=ax)
    with pytest.raises(ValueError, match="Budgets differ"):
        plot.with_recorded(band, np.ones(5), x=np.arange(5), your_budget={"steps": 1}, ax=ax)
    plt.close(fig)
