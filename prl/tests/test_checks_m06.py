"""prl.checks.m06: the exact checks against independent implementations, their guards, and
the helpers the stochastic checks use.

The checks' behavior on the lab's solutions and mutants is tested where those live:
scripts/check_notebooks.py --mode verify (exact checks) and scripts/threshold_protocol.py
(stochastic checks). Here: the parts that do not need the lab. The implementations below are
written for the tests (tests may hold algorithms; prl may not).
"""

import numpy as np
import pytest

from prl import envs
from prl.checks import CheckFailed, StochasticCheck, expected, m06
from prl.evaluate import compare


# ------------------------------------------------------------------ independent implementations
def policy_transition(P, pi):
    S, A, _ = P.shape
    out = np.zeros((S, S))
    for s in range(S):
        for a in range(A):
            out[s] += pi[s, a] * P[s, a]
    return out


def hitting_times(P_pi, target):
    """The fundamental matrix (I - M)^-1 times 1 (not the lab's np.linalg.solve)."""
    n = len(P_pi)
    keep = np.arange(n) != target
    h = np.zeros(n)
    h[keep] = np.linalg.inv(np.eye(n - 1) - P_pi[keep][:, keep]).sum(axis=1)
    return h


def visitation(states, n_states):
    v = np.zeros(n_states)
    for s in states:
        v[int(s)] += 1
    return v / len(states)


def optimistic_q0(r_max, gamma, n_states, n_actions):
    return np.full((n_states, n_actions), sum(r_max * gamma**k for k in range(4000)))


def mbie_eb_backup(P_hat, R_hat, counts, gamma, c, tol):
    R = R_hat + c / np.sqrt(np.maximum(counts, 1))
    Q = np.zeros_like(R)
    while True:
        new = R + gamma * P_hat @ Q.max(axis=1)
        if np.max(np.abs(new - Q)) < tol:
            return new
        Q = new


def verdict(a, b):
    return {"A": "A", "B": "B", "tie": "tie"}[compare(a, b, higher_is_better=False).winner]


def psrl_rewards_first(P_hat, R_hat, counts, rng):
    """A correct sampler that draws in another order than the lab's reference."""
    n_s, n_a = counts.shape
    R = rng.normal((1 + counts * R_hat) / (counts + 1), 1 / np.sqrt(counts + 1))
    N_next = P_hat * counts[:, :, None]
    P = np.empty_like(P_hat)
    for a in range(n_a):  # actions in the outer loop
        for s in range(n_s):
            P[s, a] = rng.dirichlet(1.0 / n_s + N_next[s, a])
    return P, R


def psrl_by_gamma(P_hat, R_hat, counts, rng):
    """A correct sampler that builds each Dirichlet draw from Gamma draws."""
    n_s = counts.shape[0]
    alpha = 1.0 / n_s + P_hat * counts[:, :, None]
    g = rng.gamma(alpha)
    R = (1 + counts * R_hat) / (counts + 1) + rng.standard_normal(counts.shape) / np.sqrt(
        counts + 1
    )
    return g / g.sum(axis=2, keepdims=True), R


@pytest.mark.parametrize(
    "check, fn",
    [
        (m06.check_policy_transition, policy_transition),
        (m06.check_hitting_times, hitting_times),
        (m06.check_visitation, visitation),
        (m06.check_optimistic_q0, optimistic_q0),
        (m06.check_mbie_eb_backup, mbie_eb_backup),
        (m06.check_verdict, verdict),
        (m06.check_psrl_sample, psrl_rewards_first),
        (m06.check_psrl_sample, psrl_by_gamma),
    ],
    ids=lambda x: getattr(x, "__name__", ""),
)
def test_exact_checks_accept_independent_implementations(check, fn):
    check(fn)


# ------------------------------------------------------------------ guards
@pytest.mark.parametrize(
    "check, bad",
    [
        (m06.check_policy_transition, lambda P, pi: None),
        (m06.check_policy_transition, lambda P, pi: P.mean(axis=1)),
        (m06.check_policy_transition, lambda P, pi: np.full((len(pi),) * 2, np.nan)),
        (m06.check_hitting_times, lambda P, t: np.ones(len(P))),
        (m06.check_hitting_times, lambda P, t: hitting_times(P, t)[:-1]),
        (m06.check_visitation, lambda s, n: np.bincount(np.asarray(s, dtype=int)) / len(s)),
        (m06.check_visitation, lambda s, n: [1.0 / n] * n),
        (m06.check_optimistic_q0, lambda r, g, n, a: np.full((n, a), r / (1 - g))[:, 0]),
        (m06.check_optimistic_q0, lambda r, g, n, a: np.full((n, a), 10.0)),
        (
            m06.check_mbie_eb_backup,
            lambda P, R, N, g, c, tol: mbie_eb_backup(P, R, N, g, c, tol)[:, 0],
        ),
        (m06.check_verdict, lambda a, b: "a"),
        (m06.check_verdict, lambda a, b: None),
        (m06.check_psrl_sample, lambda P, R, N, rng: (P, R)),
        (m06.check_psrl_sample, lambda P, R, N, rng: P),
    ],
)
def test_exact_checks_reject_wrong_answers(check, bad):
    with pytest.raises(CheckFailed):
        check(bad)


def test_a_crash_in_the_participants_code_is_a_check_failure_not_chained():
    def singular(P_pi, target):
        return np.linalg.solve(np.zeros((2, 2)), np.ones(2))

    with pytest.raises(CheckFailed, match="LinAlgError") as info:
        m06.check_hitting_times(singular)
    assert info.value.__cause__ is None  # verify mode counts it as a rejection


def test_hints_name_the_usual_mistakes():
    with pytest.raises(CheckFailed, match="minlength"):
        m06.check_visitation(lambda s, n: np.bincount(np.asarray(s, dtype=int)) / len(s))
    with pytest.raises(CheckFailed, match="plain average"):
        m06.check_policy_transition(lambda P, pi: P.mean(axis=1))
    with pytest.raises(CheckFailed, match="after\\s+solving"):
        fx = expected("m06_mbie")
        cases = {
            (float(fx[f"gamma_{i}"]), float(fx[f"c_{i}"]), fx[f"N_{i}"].tobytes()): i
            for i in range(int(fx["n_cases"]))
        }
        m06.check_mbie_eb_backup(
            lambda P, R, N, g, c, tol: fx[f"after_{cases[(g, c, N.tobytes())]}"].copy()
        )
    with pytest.raises(CheckFailed, match="MEAN"):
        from prl.evaluate import bootstrap_ci

        def mean_rule(a, b):
            lo_a, hi_a = bootstrap_ci(a)
            lo_b, hi_b = bootstrap_ci(b)
            if hi_a >= lo_b and hi_b >= lo_a:
                return "tie"
            return "A" if np.mean(a) < np.mean(b) else "B"

        m06.check_verdict(mean_rule)


# ------------------------------------------------------------------ fixtures
def test_fixtures_hold_the_numbers_the_page_quotes():
    chain = expected("m06_chain")
    assert np.allclose(chain["hit_h_0"], [48, 44, 31, 0])
    assert chain["hit_h_5"][0] == pytest.approx(936.428571, abs=1e-6)  # RiverSwim(6)
    assert chain["hit_h_9"][0] == pytest.approx(77296.428571, abs=1e-5)  # RiverSwim(10)
    assert float(chain["ratio_12_11"]) == pytest.approx(3.00026, abs=1e-5)
    Q_star = expected("m06_q0")["Q_star"]
    assert round(float(Q_star.max()), 2) == 16.68 and np.all(Q_star.argmax(axis=1) == 1)
    mbie = expected("m06_mbie")
    assert np.allclose(mbie["Q_0"], [[19.505, 20.0]]) and np.allclose(mbie["Q_1"], [[5.8, 6.0]])


def test_simulated_walks_agree_with_the_solve():
    # The fixture's 20000 walks (default_rng(0)) against the exact h(0) of RiverSwim(5):
    # within 3 standard errors (a correct simulation fails this with probability 0.27%;
    # this draw is fixed, so the test is deterministic).
    fx = expected("m06_chain")
    h0 = float(fx[f"hit_h_{int(fx['sim_case'])}"][0])
    assert abs(float(fx["sim_mean"]) - h0) < 3 * float(fx["sim_se"])


# ------------------------------------------------------------------ the stochastic checks' helpers
def _fixed_learner(action):
    """Stands in for q_learning: one episode of a fixed action, seeded like the lab's."""

    def learn(env, episodes, alpha, epsilon, gamma, seed, q0=0.0):
        env.reset(seed=seed)
        for _ in range(episodes):
            env.reset()
            done = False
            while not done:
                _, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
        return np.zeros((m06.S, m06.A)), np.zeros(episodes)

    return learn


def test_recorder_keeps_one_trajectory_per_episode():
    states = m06._q_learning_run(_fixed_learner(1), np.zeros((10, 2)), seed=3, steps=200)
    assert len(states) == 201 and states[0] == 0
    assert states == m06._q_learning_run(_fixed_learner(1), np.zeros((10, 2)), seed=3, steps=200)
    assert all(s == 0 for s in m06._q_learning_run(_fixed_learner(0), None, seed=3, steps=50))


def test_metrics_take_all_randomness_from_the_seed_and_validate():
    left, right = _fixed_learner(0), _fixed_learner(1)
    assert m06._eps_greedy_end(visitation, left, seed=0, steps=500) == 0.0
    end = m06._eps_greedy_end(visitation, right, seed=0, steps=500)
    assert end == m06._eps_greedy_end(visitation, right, seed=0, steps=500) and end > 0.5
    with pytest.raises(CheckFailed, match="sum to 1"):
        m06._eps_greedy_end(lambda s, n: np.ones(n), right, seed=0, steps=100)
    with pytest.raises(CheckFailed, match="ran"):
        m06._q_learning_run(lambda *a, **k: None, np.zeros((10, 2)), seed=0, steps=10)
    q0 = lambda r, g, n, a: np.zeros((n, a))  # noqa: E731
    assert m06._optimistic_reach(q0, left, seed=0, steps=200, runs=3) == 0.0
    assert m06._optimistic_reach(q0, right, seed=0, steps=200, runs=3) == 3.0
    with pytest.raises(CheckFailed, match="shape"):
        m06._optimistic_reach(lambda *a: np.zeros(3), right, seed=0, steps=10, runs=1)


def test_suspect_metric_validates_its_output():
    def run(action):
        def agent(env, steps, c, gamma, seed):
            env.reset(seed=seed)
            states = [0]
            for _ in range(steps):
                states.append(int(env.step(action)[0]))
            return np.array(states), np.full(steps, action), np.zeros((10, 2))

        return agent

    assert m06._suspect_end(run(0), seed=0, steps=300, runs=2) == 0.0
    assert m06._suspect_end(run(1), seed=0, steps=300, runs=2) > 0.5
    with pytest.raises(CheckFailed, match="states"):
        m06._suspect_end(lambda env, *a: (np.zeros(3), np.zeros(2), None), seed=0, steps=5, runs=1)
    with pytest.raises(CheckFailed, match="return"):
        m06._suspect_end(lambda env, *a: None, seed=0, steps=5, runs=1)


@pytest.mark.parametrize(
    "name",
    ["check_eps_greedy_end", "check_optimistic_reach", "check_mbie_eb_suspect"],
)
def test_stochastic_checks_have_thresholds_for_both_budgets(name):
    check = getattr(m06, name)
    assert isinstance(check, StochasticCheck)
    assert set(check.thresholds) == {"live", "quick"}
    assert check.live == check.quick  # every run is cheap: QUICK is the live budget


def test_river_is_the_labs():
    env = m06._river(7)
    assert isinstance(env.unwrapped.mdp, envs.RiverSwim) and env.unwrapped.mdp.n == m06.N_CHAIN
    assert env.unwrapped.mdp.gamma == m06.GAMMA and env.max_episode_steps == 7
