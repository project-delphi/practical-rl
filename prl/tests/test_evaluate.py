import numpy as np
import pytest

from prl import envs
from prl.evaluate import Budget, bootstrap_ci, compare, iqm, rollout_returns, stratified_bootstrap


def test_iqm_ignores_outer_quartiles():
    assert iqm([1, 2, 3, 4, 5, 6, 7, 1000]) == pytest.approx(np.mean([3, 4, 5, 6]))


def test_bootstrap_ci_coverage():
    # Seeded statistical test: a 95% percentile bootstrap CI for a normal mean (n=30)
    # should cover the true mean in roughly 95% of repetitions. Tolerance: >= 88% of 200.
    rng = np.random.default_rng(0)
    covered = 0
    for rep in range(200):
        x = rng.normal(1.0, 2.0, size=30)
        lo, hi = bootstrap_ci(x, np.mean, n_boot=1000, seed=rep)
        covered += lo <= 1.0 <= hi
    assert covered >= 176


def test_stratified_bootstrap_single_task_is_ordinary():
    x = np.arange(20.0)
    assert stratified_bootstrap([x], np.mean, seed=1) == pytest.approx(
        bootstrap_ci(x, np.mean, seed=1), abs=0.5
    )


def test_compare_rules():
    assert compare([1.0], [2.0, 3.0]).winner == "single seed"
    assert compare([1, 2, 3], [2.5, 4, 5]).winner == "tie"  # ranges overlap with < 5 seeds
    assert compare([1, 2, 3], [10, 11, 12]).winner == "B"
    v = compare(
        [1, 2, 3], [10, 11, 12], higher_is_better=False, budget=Budget(steps=50000, seeds=3)
    )
    assert v.winner == "A" and "50,000" in v.sentence
    rng = np.random.default_rng(0)
    assert compare(rng.normal(0, 1, 10), rng.normal(0.1, 1, 10)).winner == "tie"
    assert compare(rng.normal(0, 0.1, 10), rng.normal(5, 0.1, 10)).winner == "B"


def test_rollout_returns_on_tabular_env():
    g = envs.CliffGridworld(3, 4, gamma=1.0)
    env_fn = lambda: envs.TabularEnv(g, max_episode_steps=50)  # noqa: E731
    up_right_down = {g.start: 0}

    # Policy: go up once, then right until above goal, then down.
    def policy(s):
        r, c = g.coords(s)
        if s in up_right_down:
            return 0
        if c < g.cols - 1:
            return 1
        return 2

    returns = rollout_returns(env_fn, policy, episodes=3, seed=0)
    assert np.allclose(returns, -(1 + (g.cols - 1) + 1))
