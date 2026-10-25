"""Plausible wrong versions of Module 7's functions (academic director's first draft,
completed by the technical expert).

scripts/check_notebooks.py --mode verify swaps each one into the notebook and requires
every exact checkpoint of its exercise that calls the function to reject it; stochastic
checkpoints are skipped there, because scripts/threshold_protocol.py has already required
every mutant of a stochastic checkpoint's functions to fail its threshold at every seed,
with a stated gap. Each mutant is a mistake a participant could plausibly make; `why` says
which. `id="random"` is the never-learns version the protocol needs.

Checkpoints 6.2 and 7.2 are stochastic (training on CartPole). A mutant that trains about as
well as the solution cannot be told from it by a threshold, so it is `exact_only=True` and an
exact checkpoint rejects it: 6.1 (fixed arrays, with a terminated and a truncated transition)
for `linear_q_update`, 7.1 (seven fixed observations) for `colleague_features`. These are
exact_only because a scan of 20 seeds at the QUICK budget (laptop, not a record; the
technical expert's notes) put their greedy returns inside the solution's range:
mask-on-truncated, no-mask, swapped-flags, next-features, no-discount and squared-norm. The
others go to scripts/threshold_protocol.py, whose records show each one rejected at every
seed. A run whose weight norm diverges (sign, unfixed, max-normalized) is rejected by the
checks' metric itself.

Checkpoint 7.2 checks both functions, so the protocol scores the mutants of both exercises
together; mutant ids must therefore differ across Exercises 6 and 7 (the protocol keys its
per-seed values by id). Exercise 6's `random` is the never-learns version for both checks;
Exercise 7's never-learns featurizer is `constant-features`.
"""

import numpy as np

from prl.lab import mutant


# ------------------------------------------------------------------ 1: scale_spec
@mutant(ex=1, replaces="scale_spec", id="add-not-multiply", why="adds levels across products")
def scale_spec(n_products, capacity):
    levels = capacity + 1
    n_states = n_products * levels
    return {
        "n_states": n_states,
        "n_actions": n_states,
        "q_entries": n_states * n_states,
        "feature_dim": n_products * levels,
    }


@mutant(ex=1, replaces="scale_spec", id="capacity-not-levels", why="forgets stock level 0")
def scale_spec(n_products, capacity):  # noqa: F811
    n_states = capacity**n_products
    return {
        "n_states": n_states,
        "n_actions": n_states,
        "q_entries": n_states * n_states,
        "feature_dim": n_products * capacity,
    }


@mutant(
    ex=1,
    replaces="scale_spec",
    id="joint-one-hot",
    why="one-hot of the joint state: the table again, not one indicator per product",
)
def scale_spec(n_products, capacity):  # noqa: F811
    levels = capacity + 1
    n_states = levels**n_products
    return {
        "n_states": n_states,
        "n_actions": n_states,
        "q_entries": n_states * n_states,
        "feature_dim": n_states,
    }


@mutant(ex=1, replaces="scale_spec", id="q-no-actions", why="a Q table with one entry per state")
def scale_spec(n_products, capacity):  # noqa: F811
    levels = capacity + 1
    n_states = levels**n_products
    return {
        "n_states": n_states,
        "n_actions": n_states,
        "q_entries": n_states,
        "feature_dim": n_products * levels,
    }


# ------------------------------------------------------------------ 2: semi_gradient_td0
@mutant(
    ex=2,
    replaces="semi_gradient_td0",
    id="residual-gradient",
    why="differentiates the target too: the full gradient of ½δ², not the semi-gradient",
)
def semi_gradient_td0(w, phi, r, phi_next, gamma, alpha):
    delta = r + gamma * w @ phi_next - w @ phi
    return w + alpha * delta * (phi - gamma * phi_next)


@mutant(ex=2, replaces="semi_gradient_td0", id="next-features", why="steps along φ(s')")
def semi_gradient_td0(w, phi, r, phi_next, gamma, alpha):  # noqa: F811
    delta = r + gamma * w @ phi_next - w @ phi
    return w + alpha * delta * phi_next


@mutant(ex=2, replaces="semi_gradient_td0", id="no-discount", why="drops γ from the target")
def semi_gradient_td0(w, phi, r, phi_next, gamma, alpha):  # noqa: F811
    delta = r + w @ phi_next - w @ phi
    return w + alpha * delta * phi


@mutant(ex=2, replaces="semi_gradient_td0", id="sign", why="subtracts the step")
def semi_gradient_td0(w, phi, r, phi_next, gamma, alpha):  # noqa: F811
    delta = r + gamma * w @ phi_next - w @ phi
    return w - alpha * delta * phi


@mutant(
    ex=2,
    replaces="semi_gradient_td0",
    id="in-place",
    why="w += ... changes the caller's array (the checkpoint keeps a copy and compares)",
)
def semi_gradient_td0(w, phi, r, phi_next, gamma, alpha):  # noqa: F811
    delta = r + gamma * w @ phi_next - w @ phi
    w += alpha * delta * phi
    return w


# ------------------------------------------------------------------ 3: td_matrix
@mutant(ex=3, replaces="td_matrix", id="transposed", why="returns Aᵀ")
def td_matrix(phi, mu, P, gamma):
    S = len(mu)
    return (phi.T @ np.diag(mu) @ (np.eye(S) - gamma * P) @ phi).T


@mutant(ex=3, replaces="td_matrix", id="no-weights", why="leaves out D = diag(mu)")
def td_matrix(phi, mu, P, gamma):  # noqa: F811
    S = len(mu)
    return phi.T @ (np.eye(S) - gamma * P) @ phi


@mutant(ex=3, replaces="td_matrix", id="P-transposed", why="uses Pᵀ: where states come from")
def td_matrix(phi, mu, P, gamma):  # noqa: F811
    S = len(mu)
    return phi.T @ np.diag(mu) @ (np.eye(S) - gamma * P.T) @ phi


@mutant(ex=3, replaces="td_matrix", id="symmetrized", why="(A + Aᵀ)/2, as if it were a Hessian")
def td_matrix(phi, mu, P, gamma):  # noqa: F811
    S = len(mu)
    A = phi.T @ np.diag(mu) @ (np.eye(S) - gamma * P) @ phi
    return (A + A.T) / 2


@mutant(ex=3, replaces="td_matrix", id="plus-gamma", why="I + γP: a sign slip in the TD error")
def td_matrix(phi, mu, P, gamma):  # noqa: F811
    S = len(mu)
    return phi.T @ np.diag(mu) @ (np.eye(S) + gamma * P) @ phi


# ------------------------------------------------------------------ 4: baird_expected_update
@mutant(ex=4, replaces="baird_expected_update", id="unweighted", why="sums states without mu")
def baird_expected_update(w, phi, mu, P, r, gamma, alpha):
    v = phi @ w
    return w + alpha * phi.T @ (r + gamma * P @ v - v)


@mutant(ex=4, replaces="baird_expected_update", id="P-transposed", why="uses Pᵀ @ v")
def baird_expected_update(w, phi, mu, P, r, gamma, alpha):  # noqa: F811
    v = phi @ w
    return w + alpha * phi.T @ (mu * (r + gamma * P.T @ v - v))


@mutant(
    ex=4,
    replaces="baird_expected_update",
    id="sequential",
    why="updates w after each state in turn: asynchronous, not one expected sweep",
)
def baird_expected_update(w, phi, mu, P, r, gamma, alpha):  # noqa: F811
    w = w.copy()
    for s in range(len(mu)):
        v = phi @ w
        w = w + alpha * mu[s] * (r[s] + gamma * P[s] @ v - v[s]) * phi[s]
    return w


@mutant(ex=4, replaces="baird_expected_update", id="random", why="never moves")
def baird_expected_update(w, phi, mu, P, r, gamma, alpha):  # noqa: F811
    return w.copy()


# ------------------------------------------------------------------ 5: remove_one_leg
# Verify mode inserts only the decorated function into the notebook, so every mutant is
# self-contained (it may use the lab's own names, such as GRID_POINTS in Exercise 7).
@mutant(ex=5, replaces="remove_one_leg", id="behavior-weights", why="keeps the uniform mu")
def remove_one_leg(env):
    S, P = env.n_states, env.P_target
    v_pi = np.linalg.solve(np.eye(S) - env.gamma * P, env.rewards)
    return {
        "on-policy": {"mu": np.full(S, 1.0 / S)},
        "tabular": {"phi": np.eye(S)},
        "monte-carlo": {"gamma": 0.0, "r": v_pi},
    }


@mutant(
    ex=5,
    replaces="remove_one_leg",
    id="hard-coded-state-7",
    why="types in all weight on the last state; right for Baird, wrong for another target",
)
def remove_one_leg(env):  # noqa: F811
    S, P = env.n_states, env.P_target
    v_pi = np.linalg.solve(np.eye(S) - env.gamma * P, env.rewards)
    return {
        "on-policy": {"mu": np.eye(S)[S - 1]},
        "tabular": {"phi": np.eye(S)},
        "monte-carlo": {"gamma": 0.0, "r": v_pi},
    }


@mutant(
    ex=5,
    replaces="remove_one_leg",
    id="mc-rewards",
    why="uses the one-step rewards as the Monte Carlo target (equal to v_pi only if all are 0)",
)
def remove_one_leg(env):  # noqa: F811
    S = env.n_states
    mu = np.full(S, 1.0 / S)
    for _ in range(1000):
        mu = mu @ env.P_target
    return {
        "on-policy": {"mu": mu},
        "tabular": {"phi": np.eye(S)},
        "monte-carlo": {"gamma": 0.0, "r": np.asarray(env.rewards, dtype=float)},
    }


@mutant(
    ex=5,
    replaces="remove_one_leg",
    id="mc-keeps-gamma",
    why="sets r = v_pi but keeps gamma, so the target still bootstraps",
)
def remove_one_leg(env):  # noqa: F811
    S, P = env.n_states, env.P_target
    mu = np.full(S, 1.0 / S)
    for _ in range(1000):
        mu = mu @ P
    v_pi = np.linalg.solve(np.eye(S) - env.gamma * P, env.rewards)
    return {
        "on-policy": {"mu": mu},
        "tabular": {"phi": np.eye(S)},
        "monte-carlo": {"gamma": env.gamma, "r": v_pi},
    }


@mutant(
    ex=5,
    replaces="remove_one_leg",
    id="tabular-in-weights",
    why="an identity over the 8 weights, not one feature per state",
)
def remove_one_leg(env):  # noqa: F811
    S, P = env.n_states, env.P_target
    mu = np.full(S, 1.0 / S)
    for _ in range(1000):
        mu = mu @ P
    v_pi = np.linalg.solve(np.eye(S) - env.gamma * P, env.rewards)
    return {
        "on-policy": {"mu": mu},
        "tabular": {"phi": np.eye(env.n_features)},
        "monte-carlo": {"gamma": 0.0, "r": v_pi},
    }


# ------------------------------------------------------------------ 6: linear_q_update
@mutant(
    ex=6,
    replaces="linear_q_update",
    id="mask-on-truncated",
    why="stops bootstrapping on truncated as well (done = terminated or truncated)",
    exact_only=True,
)
def linear_q_update(W, phi, a, r, phi_next, terminated, truncated, gamma, alpha):
    done = terminated or truncated
    bootstrap = 0.0 if done else gamma * np.max(W @ phi_next)
    W_new = W.copy()
    W_new[a] = W[a] + alpha * (r + bootstrap - W[a] @ phi) * phi
    return W_new


@mutant(
    ex=6,
    replaces="linear_q_update",
    id="no-mask",
    why="always bootstraps, even from a terminal state",
    exact_only=True,
)
def linear_q_update(W, phi, a, r, phi_next, terminated, truncated, gamma, alpha):  # noqa: F811
    bootstrap = gamma * np.max(W @ phi_next)
    W_new = W.copy()
    W_new[a] = W[a] + alpha * (r + bootstrap - W[a] @ phi) * phi
    return W_new


@mutant(ex=6, replaces="linear_q_update", id="all-rows", why="updates every action's row")
def linear_q_update(W, phi, a, r, phi_next, terminated, truncated, gamma, alpha):  # noqa: F811
    bootstrap = 0.0 if terminated else gamma * np.max(W @ phi_next)
    delta = r + bootstrap - W[a] @ phi
    return W + alpha * delta * phi[None, :]


@mutant(
    ex=6,
    replaces="linear_q_update",
    id="action-0-target",
    why="bootstraps from Q(s', 0) instead of the max over actions",
)
def linear_q_update(W, phi, a, r, phi_next, terminated, truncated, gamma, alpha):  # noqa: F811
    bootstrap = 0.0 if terminated else gamma * (W[0] @ phi_next)
    W_new = W.copy()
    W_new[a] = W[a] + alpha * (r + bootstrap - W[a] @ phi) * phi
    return W_new


@mutant(
    ex=6,
    replaces="linear_q_update",
    id="swapped-flags",
    why="stops the bootstrap on truncated and bootstraps on terminated: the flags swapped",
    exact_only=True,
)
def linear_q_update(W, phi, a, r, phi_next, terminated, truncated, gamma, alpha):  # noqa: F811
    bootstrap = 0.0 if truncated else gamma * np.max(W @ phi_next)
    W_new = W.copy()
    W_new[a] = W[a] + alpha * (r + bootstrap - W[a] @ phi) * phi
    return W_new


@mutant(
    ex=6,
    replaces="linear_q_update",
    id="next-features",
    why="steps along phi(s') instead of phi(s); successive CartPole states are close",
    exact_only=True,
)
def linear_q_update(W, phi, a, r, phi_next, terminated, truncated, gamma, alpha):  # noqa: F811
    bootstrap = 0.0 if terminated else gamma * np.max(W @ phi_next)
    W_new = W.copy()
    W_new[a] = W[a] + alpha * (r + bootstrap - W[a] @ phi) * phi_next
    return W_new


@mutant(
    ex=6,
    replaces="linear_q_update",
    id="no-discount",
    why="drops gamma from the target; CartPole's episodes end, so it still learns",
    exact_only=True,
)
def linear_q_update(W, phi, a, r, phi_next, terminated, truncated, gamma, alpha):  # noqa: F811
    bootstrap = 0.0 if terminated else np.max(W @ phi_next)
    W_new = W.copy()
    W_new[a] = W[a] + alpha * (r + bootstrap - W[a] @ phi) * phi
    return W_new


@mutant(ex=6, replaces="linear_q_update", id="sign", why="subtracts the step: ‖W‖ diverges")
def linear_q_update(W, phi, a, r, phi_next, terminated, truncated, gamma, alpha):  # noqa: F811
    bootstrap = 0.0 if terminated else gamma * np.max(W @ phi_next)
    W_new = W.copy()
    W_new[a] = W[a] - alpha * (r + bootstrap - W[a] @ phi) * phi
    return W_new


@mutant(ex=6, replaces="linear_q_update", id="random", why="never learns")
def linear_q_update(W, phi, a, r, phi_next, terminated, truncated, gamma, alpha):  # noqa: F811
    return W.copy()


# ------------------------------------------------------------------ 7: colleague_features
@mutant(
    ex=7,
    replaces="colleague_features",
    id="unfixed",
    why="the colleague's version: bumps in [0, 1], ‖φ(s)‖² near 10",
)
def colleague_features(obs):
    z = np.asarray(obs, dtype=float) / OBS_SCALE
    f = np.exp(-((GRID_POINTS - z) ** 2).sum(axis=1) / (2 * WIDTH**2))
    return f


@mutant(
    ex=7,
    replaces="colleague_features",
    id="sum-normalized",
    why="divides by the sum: stable, but ‖φ(s)‖² near 0.007, so alpha = 0.3 barely learns",
)
def colleague_features(obs):  # noqa: F811
    z = np.asarray(obs, dtype=float) / OBS_SCALE
    f = np.exp(-((GRID_POINTS - z) ** 2).sum(axis=1) / (2 * WIDTH**2))
    return f / f.sum()


@mutant(
    ex=7,
    replaces="colleague_features",
    id="max-normalized",
    why="divides by the largest bump, which is near 1: ‖φ(s)‖² stays near 10",
)
def colleague_features(obs):  # noqa: F811
    z = np.asarray(obs, dtype=float) / OBS_SCALE
    f = np.exp(-((GRID_POINTS - z) ** 2).sum(axis=1) / (2 * WIDTH**2))
    return f / f.max()


@mutant(
    ex=7,
    replaces="colleague_features",
    id="squared-norm",
    why="divides by the squared length: ‖φ(s)‖² near 0.1, stable but slow",
    exact_only=True,
)
def colleague_features(obs):  # noqa: F811
    z = np.asarray(obs, dtype=float) / OBS_SCALE
    f = np.exp(-((GRID_POINTS - z) ** 2).sum(axis=1) / (2 * WIDTH**2))
    return f / (f @ f)


@mutant(
    ex=7,
    replaces="colleague_features",
    id="sqrt-d",
    why="divides by sqrt(625), the largest length 625 bumps in [0, 1] can have: ‖φ(s)‖² near "
    "0.016, so alpha = 0.3 barely learns",
)
def colleague_features(obs):  # noqa: F811
    z = np.asarray(obs, dtype=float) / OBS_SCALE
    f = np.exp(-((GRID_POINTS - z) ** 2).sum(axis=1) / (2 * WIDTH**2))
    return f / np.sqrt(len(f))


@mutant(
    ex=7,
    replaces="colleague_features",
    id="constant-features",
    why="features that carry no state (Exercise 7's never-learns version)",
)
def colleague_features(obs):  # noqa: F811
    return np.full(len(GRID_POINTS), 1.0 / np.sqrt(len(GRID_POINTS)))


# ------------------------------------------------------------------ stretch
@mutant(
    ex="stretch",
    replaces="tdc_expected_update",
    id="no-correction",
    why="drops the gradient correction: semi-gradient TD again",
)
def tdc_expected_update(w, v, phi, mu, P, r, gamma, alpha, beta):
    e = phi.T @ (mu * (r + gamma * P @ phi @ w - phi @ w))
    gram = phi.T @ (mu[:, None] * phi)
    return w + alpha * e, v + beta * (e - gram @ v)


@mutant(
    ex="stretch",
    replaces="tdc_expected_update",
    id="correction-sign",
    why="adds the correction instead of subtracting it",
)
def tdc_expected_update(w, v, phi, mu, P, r, gamma, alpha, beta):  # noqa: F811
    e = phi.T @ (mu * (r + gamma * P @ phi @ w - phi @ w))
    cross = (P @ phi).T @ (mu[:, None] * phi)
    gram = phi.T @ (mu[:, None] * phi)
    return w + alpha * (e + gamma * cross @ v), v + beta * (e - gram @ v)
