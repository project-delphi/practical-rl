import gymnasium as gym
import numpy as np
import pytest

from prl import envs


@pytest.mark.parametrize(
    "mdp",
    [
        envs.CliffGridworld(),
        envs.CliffGridworld(3, 4),
        envs.CliffGridworld(4, 6, slip=0.1),
        envs.InventoryMDP(),
        envs.InventoryMDP(capacity=5, max_demand=6),
        envs.RiverSwim(),
        envs.RiverSwim(n=10),
    ],
    ids=lambda m: type(m).__name__,
)
def test_tabular_envs_validate(mdp):
    mdp.validate()


def test_validate_rejects_bad_rows():
    m = envs.CliffGridworld(3, 4)
    m.P[0, 0, 0] += 0.1
    with pytest.raises(ValueError, match="sums to"):
        m.validate()


def test_cliff_layout():
    g = envs.CliffGridworld(4, 12)
    assert g.start == g.index(3, 0) and g.goal == g.index(3, 11)
    assert len(g.cliff) == 10
    # Stepping right from the start falls into the cliff: back to start, -100.
    assert g.P[g.start, 1, g.start] == 1.0 and g.R[g.start, 1] == -100.0
    # Moving up from the start costs -1.
    assert g.R[g.start, 0] == -1.0
    assert g.terminal[g.goal] and not g.terminal[g.start]


def test_inventory_sampled_reward_matches_expectation():
    # Seeded statistical test: mean of n samples within 5 standard errors of R[s, a].
    m = envs.InventoryMDP()
    rng = np.random.default_rng(0)
    for s, a in [(0, 10), (5, 0), (20, 3)]:
        rs = np.array([m.sample_step(s, a, rng)[1] for _ in range(4000)])
        assert abs(rs.mean() - m.R[s, a]) < 5 * rs.std() / np.sqrt(rs.size) + 1e-9


def test_inventory_order_cap():
    m = envs.InventoryMDP(capacity=5, max_demand=6)
    # Ordering more than capacity allows behaves like ordering up to capacity.
    assert np.allclose(m.P[3, 5], m.P[3, 2]) and np.isclose(m.R[3, 5], m.R[3, 2])


def test_riverswim_uniform_random_drift_ratio():
    rs = envs.RiverSwim()
    pi = np.full((rs.n_states, 2), 0.5)
    P_pi = np.einsum("sa,sat->st", pi, rs.P)
    p_right, p_left = P_pi[2, 3], P_pi[2, 1]
    assert np.isclose(p_left / p_right, 3.0)  # (0.5*1 + 0.5*0.05) / (0.5*0.35)


def test_riverswim_sampled_reward_matches_expectation():
    rs = envs.RiverSwim()
    rng = np.random.default_rng(1)
    s = rs.n - 1
    samples = np.array([rs.sample_step(s, 1, rng)[1] for _ in range(5000)])
    assert abs(samples.mean() - rs.R[s, 1]) < 5 * samples.std() / np.sqrt(samples.size)


def test_recsim_properties():
    sim = envs.RecSim(n_items=5, d=4, seed=3)
    x = sim.context()
    mu = sim.expected_reward(x)
    assert mu.shape == (5,) and np.all((mu >= 0.02) & (mu <= 0.98))
    assert sim.step(int(np.argmax(mu))) in (0.0, 1.0)
    uniform = lambda x, rng: np.full(5, 0.2)  # noqa: E731
    logs = sim.log(uniform, 200, seed=0)
    assert len(logs) == 200 and np.allclose(logs.P, 0.2) and np.allclose(logs.probs.sum(1), 1)
    v = sim.value(uniform, n=2000)
    assert 0.02 <= v <= 0.98


def test_recsim_misspec_and_drift_change_rewards():
    x = np.ones(4) / 2
    base = envs.RecSim(5, 4, seed=0).expected_reward(x, t=0)
    mis = envs.RecSim(5, 4, misspec=1.0, seed=0).expected_reward(x, t=0)
    drift = envs.RecSim(5, 4, drift=0.01, seed=0)
    assert not np.allclose(base, mis)
    assert not np.allclose(drift.expected_reward(x, t=0), drift.expected_reward(x, t=300))


def test_recsim_rejects_bad_policy():
    sim = envs.RecSim(3, 2)
    with pytest.raises(ValueError):
        sim.log(lambda x, rng: np.array([0.5, 0.5, 0.5]), 3)


def test_baird_structure():
    b = envs.Baird()
    assert b.phi.shape == (7, 8)
    assert np.linalg.matrix_rank(b.phi) == 7  # features can represent any value function exactly
    assert np.allclose(b.P_behavior.sum(1), 1) and np.allclose(b.P_target[:, 6], 1)
    assert np.isclose(b.behavior_probs().sum(), 1)


def test_gymnasium_registration():
    ids = envs.register()
    env = gym.make(ids[0])
    obs, _ = env.reset(seed=0)
    obs2, r, term, trunc, _ = env.step(0)
    assert env.observation_space.contains(obs2) and isinstance(r, float)
