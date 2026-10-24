"""Plausible wrong versions of Module 6's functions (academic director; technical expert).

scripts/check_notebooks.py --mode verify swaps each one into the notebook and requires
every exact checkpoint that calls the function to reject it; stochastic checkpoints are
skipped there, because scripts/threshold_protocol.py has already required every mutant of
a stochastic checkpoint's functions to fail its threshold at every seed, with a stated
gap. Each mutant is a mistake a participant could plausibly make; `why` says which.
`id="random"` is the never-learns (or never-looks) version the protocol needs.

`exact_only=True` marks a wrong version that a budgeted metric cannot separate from the
solution but an exact checkpoint rejects by construction: an optimistic start of r_max
instead of r_max / (1 - gamma) still reaches RiverSwim's right end, and so does
gamma r_max / (1 - gamma) = 19, which is above every Q* of the river (16.68 at most):
checkpoint 4.1 rejects both; 4.2 cannot.

Names the notebook defines (estimate_model, value_iteration, greedy, bootstrap_ci, iqm,
TOL) are used as the notebook has them.
"""

import numpy as np

from prl.lab import mutant

# ---- Exercise 1: policy_transition -------------------------------------------------------


@mutant(ex=1, replaces="policy_transition", id="ignores-pi", why="averages over actions")
def policy_transition(P, pi):
    return P.mean(axis=1)


@mutant(
    ex=1,
    replaces="policy_transition",
    id="sums-over-states",
    why="sums over the current state instead of the action",
)
def policy_transition(P, pi):  # noqa: F811
    return (pi[:, :, None] * P).sum(axis=0)


@mutant(ex=1, replaces="policy_transition", id="transposed", why="returns P_pi transposed")
def policy_transition(P, pi):  # noqa: F811
    return np.einsum("sa,sat->st", pi, P).T


# ---- Exercise 2: hitting_times -----------------------------------------------------------


@mutant(
    ex=2,
    replaces="hitting_times",
    id="pairs-indexing",
    why="P_pi[others, others] pairs the two lists: a vector of diagonal entries, not a matrix",
)
def hitting_times(P_pi, target):
    n = len(P_pi)
    others = [s for s in range(n) if s != target]
    h = np.zeros(n)
    M = P_pi[others, others]
    h[others] = np.linalg.solve(np.eye(n - 1) - M, np.ones(n - 1))
    return h


@mutant(ex=2, replaces="hitting_times", id="transposed", why="solves with P_pi transposed")
def hitting_times(P_pi, target):  # noqa: F811
    n = len(P_pi)
    others = [s for s in range(n) if s != target]
    h = np.zeros(n)
    M = P_pi.T[np.ix_(others, others)]
    h[others] = np.linalg.solve(np.eye(n - 1) - M, np.ones(n - 1))
    return h


@mutant(
    ex=2,
    replaces="hitting_times",
    id="last-target",
    why="assumes the target is the last state: solves over states 0..n-2 whatever target is",
)
def hitting_times(P_pi, target):  # noqa: F811
    n = len(P_pi)
    others = list(range(n - 1))
    h = np.zeros(n)
    M = P_pi[np.ix_(others, others)]
    h[others] = np.linalg.solve(np.eye(n - 1) - M, np.ones(n - 1))
    return h


@mutant(ex=2, replaces="hitting_times", id="random", why="never looks at P_pi")
def hitting_times(P_pi, target):  # noqa: F811
    h = np.random.default_rng(0).uniform(1, 100, size=len(P_pi))
    h[target] = 0.0
    return h


# ---- Exercise 3: visitation --------------------------------------------------------------


@mutant(
    ex=3,
    replaces="visitation",
    id="no-minlength",
    why="np.bincount without minlength drops the states past the last one visited",
)
def visitation(states, n_states):
    counts = np.bincount(np.asarray(states, dtype=int))
    return counts / counts.sum()


@mutant(ex=3, replaces="visitation", id="counts", why="returns counts, not fractions")
def visitation(states, n_states):  # noqa: F811
    return np.bincount(np.asarray(states, dtype=int), minlength=n_states).astype(float)


@mutant(
    ex=3,
    replaces="visitation",
    id="reversed",
    why="numbers the states from the right end",
)
def visitation(states, n_states):  # noqa: F811
    counts = np.bincount(np.asarray(states, dtype=int), minlength=n_states)
    return counts[::-1] / counts.sum()


@mutant(ex=3, replaces="visitation", id="random", why="never looks at the states")
def visitation(states, n_states):  # noqa: F811
    return np.random.default_rng(0).dirichlet(np.ones(n_states))


# ---- Exercise 4: optimistic_q0 -----------------------------------------------------------


@mutant(ex=4, replaces="optimistic_q0", id="zero", why="starts at 0: no optimism")
def optimistic_q0(r_max, gamma, n_states, n_actions):
    return np.zeros((n_states, n_actions))


@mutant(
    ex=4,
    replaces="optimistic_q0",
    id="times-one-minus-gamma",
    why="multiplies by 1 - gamma instead of dividing",
)
def optimistic_q0(r_max, gamma, n_states, n_actions):  # noqa: F811
    return np.full((n_states, n_actions), r_max * (1 - gamma))


@mutant(
    ex=4,
    replaces="optimistic_q0",
    id="r-max",
    why="one step's reward, not the discounted sum",
    exact_only=True,
)
def optimistic_q0(r_max, gamma, n_states, n_actions):  # noqa: F811
    return np.full((n_states, n_actions), float(r_max))


@mutant(
    ex=4,
    replaces="optimistic_q0",
    id="gamma-over-one-minus-gamma",
    why="starts the geometric series at gamma: gamma r_max / (1 - gamma), 19 on the river",
    exact_only=True,
)
def optimistic_q0(r_max, gamma, n_states, n_actions):  # noqa: F811
    return np.full((n_states, n_actions), gamma * r_max / (1 - gamma))


@mutant(ex=4, replaces="optimistic_q0", id="random", why="a random start near 0")
def optimistic_q0(r_max, gamma, n_states, n_actions):  # noqa: F811
    return np.random.default_rng(0).normal(size=(n_states, n_actions))


# ---- Exercise 5: mbie_eb_backup ----------------------------------------------------------


@mutant(
    ex=5,
    replaces="mbie_eb_backup",
    id="bonus-after-solve",
    why="solves the plain model, then adds the bonus: it never propagates",
)
def mbie_eb_backup(P_hat, R_hat, counts, gamma, c, tol):
    bonus = c / np.sqrt(np.maximum(counts, 1))
    return value_iteration(P_hat, R_hat, gamma, tol)[1] + bonus  # noqa: F821


@mutant(ex=5, replaces="mbie_eb_backup", id="bonus-over-n", why="c / N instead of c / sqrt(N)")
def mbie_eb_backup(P_hat, R_hat, counts, gamma, c, tol):  # noqa: F811
    bonus = c / np.maximum(counts, 1)
    return value_iteration(P_hat, R_hat + bonus, gamma, tol)[1]  # noqa: F821


@mutant(
    ex=5,
    replaces="mbie_eb_backup",
    id="untried-no-bonus",
    why="gives a pair never tried no bonus, so its self-loop is worth 0",
)
def mbie_eb_backup(P_hat, R_hat, counts, gamma, c, tol):  # noqa: F811
    bonus = np.where(counts > 0, c / np.sqrt(np.maximum(counts, 1)), 0.0)
    return value_iteration(P_hat, R_hat + bonus, gamma, tol)[1]  # noqa: F821


@mutant(
    ex=5,
    replaces="mbie_eb_backup",
    id="lab-tol",
    why="solves to the lab's TOL = 0.01, not the tol it is given (off by up to 0.19)",
)
def mbie_eb_backup(P_hat, R_hat, counts, gamma, c, tol):  # noqa: F811
    bonus = c / np.sqrt(np.maximum(counts, 1))
    return value_iteration(P_hat, R_hat + bonus, gamma, TOL)[1]  # noqa: F821


@mutant(ex=5, replaces="mbie_eb_backup", id="returns-v", why="returns V, not Q")
def mbie_eb_backup(P_hat, R_hat, counts, gamma, c, tol):  # noqa: F811
    bonus = c / np.sqrt(np.maximum(counts, 1))
    return value_iteration(P_hat, R_hat + bonus, gamma, tol)[0]  # noqa: F821


# ---- Exercise 6: regret ------------------------------------------------------------------


@mutant(
    ex=6, replaces="regret", id="per-step", why="returns each step's regret, not the running sum"
)
def regret(states, actions, R, opt_rewards):
    states, actions = np.asarray(states), np.asarray(actions)
    return opt_rewards - R[states[:-1], actions]


@mutant(ex=6, replaces="regret", id="sign", why="subtracts the optimum from what the learner got")
def regret(states, actions, R, opt_rewards):  # noqa: F811
    states, actions = np.asarray(states), np.asarray(actions)
    return np.cumsum(R[states[:-1], actions] - opt_rewards)


@mutant(
    ex=6,
    replaces="regret",
    id="next-states",
    why="pairs each action with the state it led to, states[1:]",
)
def regret(states, actions, R, opt_rewards):  # noqa: F811
    states, actions = np.asarray(states), np.asarray(actions)
    return np.cumsum(opt_rewards - R[states[1:], actions])


@mutant(
    ex=6,
    replaces="regret",
    id="best-action-here",
    why="compares with the best action in the state the learner was in, not with the optimal policy",
)
def regret(states, actions, R, opt_rewards):  # noqa: F811
    states, actions = np.asarray(states), np.asarray(actions)
    here = R[states[:-1]]
    return np.cumsum(here.max(axis=1) - here[np.arange(len(actions)), actions])


@mutant(ex=6, replaces="regret", id="final-only", why="returns only the total, Reg_T")
def regret(states, actions, R, opt_rewards):  # noqa: F811
    states, actions = np.asarray(states), np.asarray(actions)
    return np.sum(opt_rewards - R[states[:-1], actions])


@mutant(ex=6, replaces="regret", id="average", why="divides the running sum by the number of steps")
def regret(states, actions, R, opt_rewards):  # noqa: F811
    states, actions = np.asarray(states), np.asarray(actions)
    total = np.cumsum(opt_rewards - R[states[:-1], actions])
    return total / np.arange(1, len(total) + 1)


@mutant(ex=6, replaces="regret", id="clipped", why="counts no step as negative regret")
def regret(states, actions, R, opt_rewards):  # noqa: F811
    states, actions = np.asarray(states), np.asarray(actions)
    return np.cumsum(np.maximum(0.0, opt_rewards - R[states[:-1], actions]))


# ---- Exercise 7: the planted bug ---------------------------------------------------------


@mutant(
    ex=7,
    replaces="mbie_eb_suspect",
    id="still-at-selection",
    why="leaves the colleague's bug in place: the bonus only at action selection",
)
def mbie_eb_suspect(env, steps, c, gamma, seed):
    n_s, n_a = env.observation_space.n, env.action_space.n
    log, states, actions = [], [], []
    counts = np.zeros((n_s, n_a), dtype=int)
    s, _ = env.reset(seed=seed)
    stale = True
    for _ in range(steps):
        if stale:
            P_hat, R_hat, N = estimate_model(log, n_s, n_a)  # noqa: F821
            Q = value_iteration(P_hat, R_hat, gamma, TOL)[1]  # noqa: F821
        bonus = c / np.sqrt(np.maximum(counts, 1))
        a = int(greedy(Q[[s]] + bonus[[s]])[0])  # noqa: F821
        s2, r, terminated, truncated, _ = env.step(a)
        log.append((s, a, r, s2))
        states.append(s)
        actions.append(a)
        counts[s, a] += 1
        stale = counts[s, a] & (counts[s, a] - 1) == 0
        s = s2
    states.append(s)
    return np.array(states), np.array(actions), Q


@mutant(
    ex=7,
    replaces="mbie_eb_suspect",
    id="bonus-deleted",
    why="deletes the selection bonus without moving it into the backup: no bonus anywhere",
)
def mbie_eb_suspect(env, steps, c, gamma, seed):  # noqa: F811
    n_s, n_a = env.observation_space.n, env.action_space.n
    log, states, actions = [], [], []
    counts = np.zeros((n_s, n_a), dtype=int)
    s, _ = env.reset(seed=seed)
    stale = True
    for _ in range(steps):
        if stale:
            P_hat, R_hat, N = estimate_model(log, n_s, n_a)  # noqa: F821
            Q = value_iteration(P_hat, R_hat, gamma, TOL)[1]  # noqa: F821
        a = int(greedy(Q[[s]])[0])  # noqa: F821
        s2, r, terminated, truncated, _ = env.step(a)
        log.append((s, a, r, s2))
        states.append(s)
        actions.append(a)
        counts[s, a] += 1
        stale = counts[s, a] & (counts[s, a] - 1) == 0
        s = s2
    states.append(s)
    return np.array(states), np.array(actions), Q


@mutant(ex=7, replaces="mbie_eb_suspect", id="random", why="acts at random and never plans")
def mbie_eb_suspect(env, steps, c, gamma, seed):  # noqa: F811
    rng = np.random.default_rng(seed)
    n_s, n_a = env.observation_space.n, env.action_space.n
    s, _ = env.reset(seed=seed)
    states, actions = [s], []
    for _ in range(steps):
        a = int(rng.integers(n_a))
        s, *_ = env.step(a)
        states.append(s)
        actions.append(a)
    return np.array(states), np.array(actions), np.zeros((n_s, n_a))


# ---- Stretch: psrl_sample ----------------------------------------------------------------


@mutant(
    ex="stretch",
    replaces="psrl_sample",
    id="posterior-mean",
    why="returns the posterior means: no sampling, so no exploration from uncertainty",
)
def psrl_sample(P_hat, R_hat, counts, rng):
    n_s = counts.shape[0]
    alpha = 1.0 / n_s + P_hat * counts[:, :, None]
    return alpha / alpha.sum(axis=2, keepdims=True), (1 + counts * R_hat) / (counts + 1)
