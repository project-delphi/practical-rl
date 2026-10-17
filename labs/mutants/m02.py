"""Plausible wrong versions of Module 2's functions.

scripts/check_notebooks.py --mode verify swaps each one into the notebook and requires
every checkpoint that calls the function to reject it. Each mutant is a mistake a
participant could plausibly make; `why` says which. Names the notebook defines
(evaluate_exact, policy_matrix, greedy, estimate_model, value_iteration,
sample_transitions) are used as the notebook has them.
"""

import numpy as np

from prl.lab import mutant

# ---- Exercise 1: value_iteration -------------------------------------------------------


@mutant(ex=1, replaces="value_iteration", id="in-place", why="updates V state by state")
def value_iteration(P, R, gamma, tol):
    S = P.shape[0]
    V, k = np.zeros(S), 0
    while True:
        change = 0.0
        for s in range(S):
            v = (R[s] + gamma * P[s] @ V).max()
            change = max(change, abs(v - V[s]))
            V[s] = v
        k += 1
        if change < tol:
            return V, R + gamma * P @ V, k


@mutant(ex=1, replaces="value_iteration", id="mean-stop", why="stops on the mean change")
def value_iteration(P, R, gamma, tol):  # noqa: F811
    V, k = np.zeros(P.shape[0]), 0
    while True:
        V_new = (R + gamma * P @ V).max(axis=1)
        k += 1
        if np.mean(np.abs(V_new - V)) < tol:
            return V_new, R + gamma * P @ V_new, k
        V = V_new


@mutant(ex=1, replaces="value_iteration", id="min-stop", why="stops on the smallest change")
def value_iteration(P, R, gamma, tol):  # noqa: F811
    V, k = np.zeros(P.shape[0]), 0
    while True:
        V_new = (R + gamma * P @ V).max(axis=1)
        k += 1
        if np.min(np.abs(V_new - V)) < tol:
            return V_new, R + gamma * P @ V_new, k
        V = V_new


@mutant(ex=1, replaces="value_iteration", id="no-abs", why="tests the change without abs()")
def value_iteration(P, R, gamma, tol):  # noqa: F811
    V, k = np.zeros(P.shape[0]), 0
    while True:
        V_new = (R + gamma * P @ V).max(axis=1)
        k += 1
        if np.max(V_new - V) < tol:
            return V_new, R + gamma * P @ V_new, k
        V = V_new


@mutant(
    ex=1, replaces="value_iteration", id="max-axis-0", why="takes the max over states, not actions"
)
def value_iteration(P, R, gamma, tol):  # noqa: F811
    V, k = np.zeros(P.shape[0]), 0
    while True:
        V_new = (R + gamma * P @ V).max(axis=0)
        k += 1
        if np.max(np.abs(V_new - V)) < tol:
            return V_new, R + gamma * P @ V_new, k
        V = V_new


@mutant(
    ex=1,
    replaces="value_iteration",
    id="stale-Q",
    why="returns the Q of the last sweep, built from the previous V",
)
def value_iteration(P, R, gamma, tol):  # noqa: F811
    V, k = np.zeros(P.shape[0]), 0
    while True:
        Q = R + gamma * P @ V
        V_new = Q.max(axis=1)
        k += 1
        if np.max(np.abs(V_new - V)) < tol:
            return V_new, Q, k
        V = V_new


@mutant(
    ex=1, replaces="value_iteration", id="no-gamma", why="forgets the discount inside the backup"
)
def value_iteration(P, R, gamma, tol):  # noqa: F811
    V, k = np.zeros(P.shape[0]), 0
    while k < 10_000:
        V_new = (R + P @ V).max(axis=1)
        k += 1
        if np.max(np.abs(V_new - V)) < tol:
            break
        V = V_new
    return V_new, R + gamma * P @ V_new, k


# ---- Exercise 1: greedy -----------------------------------------------------------------


@mutant(ex=1, replaces="greedy", id="argmax-axis-0", why="takes the argmax over states")
def greedy(Q):
    return np.argmax(Q, axis=0)


@mutant(ex=1, replaces="greedy", id="argmin", why="picks the smallest Q")
def greedy(Q):  # noqa: F811
    return np.argmin(Q, axis=1)


@mutant(ex=1, replaces="greedy", id="one-hot", why="returns an (S, A) one-hot matrix")
def greedy(Q):  # noqa: F811
    pi = np.zeros_like(Q)
    pi[np.arange(Q.shape[0]), Q.argmax(axis=1)] = 1.0
    return pi


@mutant(ex=1, replaces="greedy", id="max-value", why="returns the best value, not the action")
def greedy(Q):  # noqa: F811
    return Q.max(axis=1)


# ---- Exercise 2: policy_iteration -------------------------------------------------------


@mutant(
    ex=2,
    replaces="policy_iteration",
    id="one-improvement",
    why="returns after the first evaluate-improve step",
)
def policy_iteration(P, R, gamma, pi0):
    S, A = R.shape
    pi = np.asarray(pi0, dtype=int).copy()
    V = evaluate_exact(P, R, gamma, policy_matrix(pi, A))  # noqa: F821
    Q = R + gamma * P @ V
    keep = Q[np.arange(S), pi] >= Q.max(axis=1)
    pi = np.where(keep, pi, greedy(Q))  # noqa: F821
    return evaluate_exact(P, R, gamma, policy_matrix(pi, A)), pi, 1  # noqa: F821


@mutant(
    ex=2, replaces="policy_iteration", id="no-tie-rule", why="improves to greedy(Q) even on ties"
)
def policy_iteration(P, R, gamma, pi0):  # noqa: F811
    S, A = R.shape
    pi, k = np.asarray(pi0, dtype=int).copy(), 0
    while k < 1000:
        V = evaluate_exact(P, R, gamma, policy_matrix(pi, A))  # noqa: F821
        k += 1
        pi_new = greedy(R + gamma * P @ V)  # noqa: F821
        if np.array_equal(pi_new, pi):
            break
        pi = pi_new
    return V, pi, k


@mutant(
    ex=2,
    replaces="policy_iteration",
    id="greedy-of-R",
    why="improves with argmax R, ignoring gamma P V",
)
def policy_iteration(P, R, gamma, pi0):  # noqa: F811
    S, A = R.shape
    pi, k = np.asarray(pi0, dtype=int).copy(), 0
    while k < 1000:
        V = evaluate_exact(P, R, gamma, policy_matrix(pi, A))  # noqa: F821
        k += 1
        keep = R[np.arange(S), pi] >= R.max(axis=1)
        pi_new = np.where(keep, pi, greedy(R))  # noqa: F821
        if np.array_equal(pi_new, pi):
            break
        pi = pi_new
    return V, pi, k


@mutant(ex=2, replaces="policy_iteration", id="argmax-axis-0", why="improves with Q.argmax(axis=0)")
def policy_iteration(P, R, gamma, pi0):  # noqa: F811
    S, A = R.shape
    pi, k = np.asarray(pi0, dtype=int).copy(), 0
    while k < 1000:
        V = evaluate_exact(P, R, gamma, policy_matrix(pi, A))  # noqa: F821
        k += 1
        Q = R + gamma * P @ V
        keep = Q[np.arange(S), pi] >= Q.max(axis=1)
        pi_new = np.where(keep, pi, Q.argmax(axis=0))
        if np.array_equal(pi_new, pi):
            break
        pi = pi_new
    return V, pi, k


@mutant(
    ex=2,
    replaces="policy_iteration",
    id="returns-new-policy",
    why="returns the improved policy with the value of the previous one, counting one short",
)
def policy_iteration(P, R, gamma, pi0):  # noqa: F811
    S, A = R.shape
    pi, k = np.asarray(pi0, dtype=int).copy(), 0
    while True:
        V = evaluate_exact(P, R, gamma, policy_matrix(pi, A))  # noqa: F821
        Q = R + gamma * P @ V
        keep = Q[np.arange(S), pi] >= Q.max(axis=1)
        pi_new = np.where(keep, pi, greedy(Q))  # noqa: F821
        if np.array_equal(pi_new, pi):
            return V, pi, k
        k += 1
        pi = pi_new


# ---- Exercise 3: estimate_model ---------------------------------------------------------


@mutant(
    ex=3,
    replaces="estimate_model",
    id="no-self-loop",
    why="leaves unvisited rows of P_hat as zeros",
)
def estimate_model(log, n_states, n_actions):
    counts = np.zeros((n_states, n_actions))
    P_hat = np.zeros((n_states, n_actions, n_states))
    R_hat = np.zeros((n_states, n_actions))
    for s, a, r, s2 in log:
        s, a, s2 = int(s), int(a), int(s2)
        counts[s, a] += 1
        P_hat[s, a, s2] += 1
        R_hat[s, a] += r
    seen = counts > 0
    P_hat[seen] /= counts[seen][:, None]
    R_hat[seen] /= counts[seen]
    return P_hat, R_hat, counts


@mutant(
    ex=3,
    replaces="estimate_model",
    id="nan-division",
    why="divides every row by its count, so unvisited rows are 0/0",
)
def estimate_model(log, n_states, n_actions):  # noqa: F811
    counts = np.zeros((n_states, n_actions))
    P_hat = np.zeros((n_states, n_actions, n_states))
    R_hat = np.zeros((n_states, n_actions))
    for s, a, r, s2 in log:
        s, a, s2 = int(s), int(a), int(s2)
        counts[s, a] += 1
        P_hat[s, a, s2] += 1
        R_hat[s, a] += r
    with np.errstate(invalid="ignore", divide="ignore"):
        return P_hat / counts[:, :, None], R_hat / counts, counts


@mutant(ex=3, replaces="estimate_model", id="reward-sum", why="sums rewards instead of averaging")
def estimate_model(log, n_states, n_actions):  # noqa: F811
    counts = np.zeros((n_states, n_actions))
    P_hat = np.zeros((n_states, n_actions, n_states))
    R_hat = np.zeros((n_states, n_actions))
    for s, a, r, s2 in log:
        s, a, s2 = int(s), int(a), int(s2)
        counts[s, a] += 1
        P_hat[s, a, s2] += 1
        R_hat[s, a] += r
    for s in range(n_states):
        for a in range(n_actions):
            if counts[s, a] > 0:
                P_hat[s, a] /= counts[s, a]
            else:
                P_hat[s, a, s] = 1.0
    return P_hat, R_hat, counts


@mutant(
    ex=3,
    replaces="estimate_model",
    id="normalize-axis-0",
    why="normalizes P_hat over the wrong axis (column totals)",
)
def estimate_model(log, n_states, n_actions):  # noqa: F811
    counts = np.zeros((n_states, n_actions))
    P_hat = np.zeros((n_states, n_actions, n_states))
    R_hat = np.zeros((n_states, n_actions))
    for s, a, r, s2 in log:
        s, a, s2 = int(s), int(a), int(s2)
        counts[s, a] += 1
        P_hat[s, a, s2] += 1
        R_hat[s, a] += r
    totals = P_hat.sum(axis=0, keepdims=True)
    P_hat = np.divide(P_hat, totals, out=np.zeros_like(P_hat), where=totals > 0)
    for s in range(n_states):
        for a in range(n_actions):
            if counts[s, a] > 0:
                R_hat[s, a] /= counts[s, a]
            else:
                P_hat[s, a, s] = 1.0
    return P_hat, R_hat, counts


@mutant(
    ex=3,
    replaces="estimate_model",
    id="counts-of-next-state",
    why="tallies (s2, a) instead of (s, a)",
)
def estimate_model(log, n_states, n_actions):  # noqa: F811
    counts = np.zeros((n_states, n_actions))
    P_hat = np.zeros((n_states, n_actions, n_states))
    R_hat = np.zeros((n_states, n_actions))
    for s, a, r, s2 in log:
        s, a, s2 = int(s), int(a), int(s2)
        counts[s2, a] += 1
        P_hat[s, a, s2] += 1
        R_hat[s, a] += r
    for s in range(n_states):
        for a in range(n_actions):
            n = P_hat[s, a].sum()
            if n > 0:
                P_hat[s, a] /= n
                R_hat[s, a] /= n
            else:
                P_hat[s, a, s] = 1.0
    return P_hat, R_hat, counts


# ---- Exercise 4: plan_and_evaluate (each returns a correct eps_R unless that is the bug) --


@mutant(
    ex=4,
    replaces="plan_and_evaluate",
    id="evaluate-in-P-hat",
    why="computes both values in the estimate, so the plan always looks optimal",
)
def plan_and_evaluate(mdp, N, seed, tol=1e-6):
    rng = np.random.default_rng(seed)
    log = sample_transitions(mdp, N, rng)  # noqa: F821
    P_hat, R_hat, _ = estimate_model(log, mdp.n_states, mdp.n_actions)  # noqa: F821
    _, Q_hat, _ = value_iteration(P_hat, R_hat, mdp.gamma, tol)  # noqa: F821
    pi_hat = policy_matrix(greedy(Q_hat), mdp.n_actions)  # noqa: F821
    V_best = evaluate_exact(P_hat, R_hat, mdp.gamma, pi_hat)  # noqa: F821
    V_plan = evaluate_exact(P_hat, R_hat, mdp.gamma, pi_hat)  # noqa: F821
    return (
        float(np.max(V_best - V_plan)),
        float(np.abs(mdp.P - P_hat).sum(axis=2).max()),
        float(np.abs(mdp.R - R_hat).max()),
    )


@mutant(
    ex=4,
    replaces="plan_and_evaluate",
    id="plan-in-truth",
    why="plans with mdp.P, mdp.R instead of the estimate",
)
def plan_and_evaluate(mdp, N, seed, tol=1e-6):  # noqa: F811
    rng = np.random.default_rng(seed)
    log = sample_transitions(mdp, N, rng)  # noqa: F821
    P_hat, R_hat, _ = estimate_model(log, mdp.n_states, mdp.n_actions)  # noqa: F821
    _, Q, _ = value_iteration(mdp.P, mdp.R, mdp.gamma, tol)  # noqa: F821
    pi = policy_matrix(greedy(Q), mdp.n_actions)  # noqa: F821
    V_star = evaluate_exact(mdp.P, mdp.R, mdp.gamma, pi)  # noqa: F821
    V_plan = evaluate_exact(mdp.P, mdp.R, mdp.gamma, pi)  # noqa: F821
    return (
        float(np.max(V_star - V_plan)),
        float(np.abs(mdp.P - P_hat).sum(axis=2).max()),
        float(np.abs(mdp.R - R_hat).max()),
    )


@mutant(
    ex=4, replaces="plan_and_evaluate", id="ignores-seed", why="always samples with default_rng(0)"
)
def plan_and_evaluate(mdp, N, seed, tol=1e-6):  # noqa: F811
    rng = np.random.default_rng(0)
    log = sample_transitions(mdp, N, rng)  # noqa: F821
    P_hat, R_hat, _ = estimate_model(log, mdp.n_states, mdp.n_actions)  # noqa: F821
    _, Q_hat, _ = value_iteration(P_hat, R_hat, mdp.gamma, tol)  # noqa: F821
    _, Q_true, _ = value_iteration(mdp.P, mdp.R, mdp.gamma, tol)  # noqa: F821
    A = mdp.n_actions
    V_star = evaluate_exact(mdp.P, mdp.R, mdp.gamma, policy_matrix(greedy(Q_true), A))  # noqa: F821
    V_plan = evaluate_exact(mdp.P, mdp.R, mdp.gamma, policy_matrix(greedy(Q_hat), A))  # noqa: F821
    return (
        float(np.max(V_star - V_plan)),
        float(np.abs(mdp.P - P_hat).sum(axis=2).max()),
        float(np.abs(mdp.R - R_hat).max()),
    )


@mutant(
    ex=4,
    replaces="plan_and_evaluate",
    id="eps-max-norm",
    why="measures eps_P as the largest single entry, not the l1 norm of a row",
)
def plan_and_evaluate(mdp, N, seed, tol=1e-6):  # noqa: F811
    rng = np.random.default_rng(seed)
    log = sample_transitions(mdp, N, rng)  # noqa: F821
    P_hat, R_hat, _ = estimate_model(log, mdp.n_states, mdp.n_actions)  # noqa: F821
    _, Q_hat, _ = value_iteration(P_hat, R_hat, mdp.gamma, tol)  # noqa: F821
    V_true, Q_true, _ = value_iteration(mdp.P, mdp.R, mdp.gamma, tol)  # noqa: F821
    A = mdp.n_actions
    V_star = evaluate_exact(mdp.P, mdp.R, mdp.gamma, policy_matrix(greedy(Q_true), A))  # noqa: F821
    V_plan = evaluate_exact(mdp.P, mdp.R, mdp.gamma, policy_matrix(greedy(Q_hat), A))  # noqa: F821
    return (
        float(np.max(V_star - V_plan)),
        float(np.abs(mdp.P - P_hat).max()),
        float(np.abs(mdp.R - R_hat).max()),
    )


@mutant(
    ex=4,
    replaces="plan_and_evaluate",
    id="loss-sign",
    why="subtracts the wrong way round: V^pi_hat - V*",
)
def plan_and_evaluate(mdp, N, seed, tol=1e-6):  # noqa: F811
    rng = np.random.default_rng(seed)
    log = sample_transitions(mdp, N, rng)  # noqa: F821
    P_hat, R_hat, _ = estimate_model(log, mdp.n_states, mdp.n_actions)  # noqa: F821
    _, Q_hat, _ = value_iteration(P_hat, R_hat, mdp.gamma, tol)  # noqa: F821
    V_true, Q_true, _ = value_iteration(mdp.P, mdp.R, mdp.gamma, tol)  # noqa: F821
    A = mdp.n_actions
    V_star = evaluate_exact(mdp.P, mdp.R, mdp.gamma, policy_matrix(greedy(Q_true), A))  # noqa: F821
    V_plan = evaluate_exact(mdp.P, mdp.R, mdp.gamma, policy_matrix(greedy(Q_hat), A))  # noqa: F821
    return (
        float(np.max(V_plan - V_star)),
        float(np.abs(mdp.P - P_hat).sum(axis=2).max()),
        float(np.abs(mdp.R - R_hat).max()),
    )


@mutant(
    ex=4,
    replaces="plan_and_evaluate",
    id="vi-value-as-V-star",
    why="uses value iteration's V as V* instead of the exact value of its greedy policy",
)
def plan_and_evaluate(mdp, N, seed, tol=1e-6):  # noqa: F811
    rng = np.random.default_rng(seed)
    log = sample_transitions(mdp, N, rng)  # noqa: F821
    P_hat, R_hat, _ = estimate_model(log, mdp.n_states, mdp.n_actions)  # noqa: F821
    _, Q_hat, _ = value_iteration(P_hat, R_hat, mdp.gamma, tol)  # noqa: F821
    V_true, Q_true, _ = value_iteration(mdp.P, mdp.R, mdp.gamma, tol)  # noqa: F821
    A = mdp.n_actions
    V_star = V_true
    V_plan = evaluate_exact(mdp.P, mdp.R, mdp.gamma, policy_matrix(greedy(Q_hat), A))  # noqa: F821
    return (
        float(np.max(V_star - V_plan)),
        float(np.abs(mdp.P - P_hat).sum(axis=2).max()),
        float(np.abs(mdp.R - R_hat).max()),
    )


@mutant(
    ex=4,
    replaces="plan_and_evaluate",
    id="eps-R-signed",
    why="takes the max of R - R_hat without abs()",
)
def plan_and_evaluate(mdp, N, seed, tol=1e-6):  # noqa: F811
    rng = np.random.default_rng(seed)
    log = sample_transitions(mdp, N, rng)  # noqa: F821
    P_hat, R_hat, _ = estimate_model(log, mdp.n_states, mdp.n_actions)  # noqa: F821
    _, Q_hat, _ = value_iteration(P_hat, R_hat, mdp.gamma, tol)  # noqa: F821
    V_true, Q_true, _ = value_iteration(mdp.P, mdp.R, mdp.gamma, tol)  # noqa: F821
    A = mdp.n_actions
    V_star = evaluate_exact(mdp.P, mdp.R, mdp.gamma, policy_matrix(greedy(Q_true), A))  # noqa: F821
    V_plan = evaluate_exact(mdp.P, mdp.R, mdp.gamma, policy_matrix(greedy(Q_hat), A))  # noqa: F821
    return (
        float(np.max(V_star - V_plan)),
        float(np.abs(mdp.P - P_hat).sum(axis=2).max()),
        float((mdp.R - R_hat).max()),
    )


@mutant(
    ex=4,
    replaces="plan_and_evaluate",
    id="eps-R-mean",
    why="averages |R - R_hat| instead of taking the max",
)
def plan_and_evaluate(mdp, N, seed, tol=1e-6):  # noqa: F811
    rng = np.random.default_rng(seed)
    log = sample_transitions(mdp, N, rng)  # noqa: F821
    P_hat, R_hat, _ = estimate_model(log, mdp.n_states, mdp.n_actions)  # noqa: F821
    _, Q_hat, _ = value_iteration(P_hat, R_hat, mdp.gamma, tol)  # noqa: F821
    V_true, Q_true, _ = value_iteration(mdp.P, mdp.R, mdp.gamma, tol)  # noqa: F821
    A = mdp.n_actions
    V_star = evaluate_exact(mdp.P, mdp.R, mdp.gamma, policy_matrix(greedy(Q_true), A))  # noqa: F821
    V_plan = evaluate_exact(mdp.P, mdp.R, mdp.gamma, policy_matrix(greedy(Q_hat), A))  # noqa: F821
    return (
        float(np.max(V_star - V_plan)),
        float(np.abs(mdp.P - P_hat).sum(axis=2).max()),
        float(np.abs(mdp.R - R_hat).mean()),
    )


# ---- Exercise 4: planning_loss_bound ----------------------------------------------------


@mutant(
    ex=4,
    replaces="planning_loss_bound",
    id="one-horizon",
    why="one horizon factor in the transition term: (1 - gamma), not (1 - gamma)^2",
)
def planning_loss_bound(eps_P, eps_R, r_range, gamma):
    return 2 * eps_R / (1 - gamma) + 2 * gamma * eps_P * r_range / (1 - gamma)


@mutant(ex=4, replaces="planning_loss_bound", id="no-range", why="drops the reward range")
def planning_loss_bound(eps_P, eps_R, r_range, gamma):  # noqa: F811
    return 2 * eps_R / (1 - gamma) + 2 * gamma * eps_P / (1 - gamma) ** 2


@mutant(ex=4, replaces="planning_loss_bound", id="no-reward-term", why="drops the eps_R term")
def planning_loss_bound(eps_P, eps_R, r_range, gamma):  # noqa: F811
    return 2 * gamma * eps_P * r_range / (1 - gamma) ** 2


@mutant(
    ex=4,
    replaces="planning_loss_bound",
    id="no-factor-2",
    why="uses the value-error bound (eq-sim-bound) instead of the loss bound",
)
def planning_loss_bound(eps_P, eps_R, r_range, gamma):  # noqa: F811
    return eps_R / (1 - gamma) + gamma * eps_P * r_range / (1 - gamma) ** 2


@mutant(ex=4, replaces="planning_loss_bound", id="no-gamma", why="drops gamma from the numerator")
def planning_loss_bound(eps_P, eps_R, r_range, gamma):  # noqa: F811
    return 2 * eps_R / (1 - gamma) + 2 * eps_P * r_range / (1 - gamma) ** 2


@mutant(
    ex=4,
    replaces="planning_loss_bound",
    id="reward-term-squared",
    why="two horizon factors under eps_R as well",
)
def planning_loss_bound(eps_P, eps_R, r_range, gamma):  # noqa: F811
    return 2 * eps_R / (1 - gamma) ** 2 + 2 * gamma * eps_P * r_range / (1 - gamma) ** 2


# ---- Exercise 5: optimality_residual ----------------------------------------------------


@mutant(ex=5, replaces="optimality_residual", id="no-abs", why="takes the max without abs()")
def optimality_residual(P, R, gamma, V):
    return float(np.max((R + gamma * P @ V).max(axis=1) - V))


@mutant(
    ex=5,
    replaces="optimality_residual",
    id="mean-over-states",
    why="averages over states instead of taking the max",
)
def optimality_residual(P, R, gamma, V):  # noqa: F811
    return float(np.mean(np.abs((R + gamma * P @ V).max(axis=1) - V)))


@mutant(
    ex=5,
    replaces="optimality_residual",
    id="action-0",
    why="backs up action 0, no max over actions",
)
def optimality_residual(P, R, gamma, V):  # noqa: F811
    return float(np.max(np.abs((R + gamma * P @ V)[:, 0] - V)))


@mutant(
    ex=5,
    replaces="optimality_residual",
    id="mean-over-actions",
    why="averages over actions (a uniform policy's Bellman residual)",
)
def optimality_residual(P, R, gamma, V):  # noqa: F811
    return float(np.max(np.abs((R + gamma * P @ V).mean(axis=1) - V)))


@mutant(ex=5, replaces="optimality_residual", id="no-gamma", why="forgets the discount")
def optimality_residual(P, R, gamma, V):  # noqa: F811
    return float(np.max(np.abs((R + P @ V).max(axis=1) - V)))


@mutant(
    ex=5, replaces="optimality_residual", id="returns-vector", why="forgets the max over states"
)
def optimality_residual(P, R, gamma, V):  # noqa: F811
    return np.abs((R + gamma * P @ V).max(axis=1) - V)


@mutant(
    ex=5,
    replaces="optimality_residual",
    id="max-axis-0",
    why="takes the max over states, not actions",
)
def optimality_residual(P, R, gamma, V):  # noqa: F811
    return float(np.max(np.abs((R + gamma * P @ V).max(axis=0) - V)))


# ---- Exercise 5: value_iteration_suspect (the stub is the planted mean-stop bug) -------


@mutant(
    ex=5, replaces="value_iteration_suspect", id="min-stop", why="'fixes' it to the smallest change"
)
def value_iteration_suspect(P, R, gamma, tol):
    V, k, log = np.zeros(P.shape[0]), 0, []
    while True:
        V_new = (R + gamma * P @ V).max(axis=1)
        k += 1
        change = np.abs(V_new - V)
        log.append(float(change.max()))
        if change.min() < tol:
            return V_new, R + gamma * P @ V_new, k, log
        V = V_new


@mutant(
    ex=5,
    replaces="value_iteration_suspect",
    id="two-norm-stop",
    why="'fixes' it to the 2-norm of the change",
)
def value_iteration_suspect(P, R, gamma, tol):  # noqa: F811
    V, k, log = np.zeros(P.shape[0]), 0, []
    while True:
        V_new = (R + gamma * P @ V).max(axis=1)
        k += 1
        change = np.abs(V_new - V)
        log.append(float(change.max()))
        if np.linalg.norm(change) < tol:
            return V_new, R + gamma * P @ V_new, k, log
        V = V_new


@mutant(
    ex=5, replaces="value_iteration_suspect", id="sum-stop", why="'fixes' it to the total change"
)
def value_iteration_suspect(P, R, gamma, tol):  # noqa: F811
    V, k, log = np.zeros(P.shape[0]), 0, []
    while True:
        V_new = (R + gamma * P @ V).max(axis=1)
        k += 1
        change = np.abs(V_new - V)
        log.append(float(change.max()))
        if change.sum() < tol:
            return V_new, R + gamma * P @ V_new, k, log
        V = V_new


@mutant(
    ex=5,
    replaces="value_iteration_suspect",
    id="stops-on-mean-logs-mean",
    why="makes the log agree with the bug instead of fixing the test",
)
def value_iteration_suspect(P, R, gamma, tol):  # noqa: F811
    V, k, log = np.zeros(P.shape[0]), 0, []
    while True:
        V_new = (R + gamma * P @ V).max(axis=1)
        k += 1
        change = np.abs(V_new - V)
        log.append(float(change.mean()))
        if change.mean() < tol:
            return V_new, R + gamma * P @ V_new, k, log
        V = V_new


# ---- Stretch: modified_policy_iteration ------------------------------------------------


@mutant(
    ex="stretch",
    replaces="modified_policy_iteration",
    id="no-eval-sweeps",
    why="skips the m - 1 evaluation sweeps (value iteration for every m)",
)
def modified_policy_iteration(P, R, gamma, m, tol):
    V, rounds = np.zeros(P.shape[0]), 0
    while True:
        Q = R + gamma * P @ V
        V_new = Q.max(axis=1)
        rounds += 1
        if np.max(np.abs(V_new - V)) < tol:
            return V_new, greedy(R + gamma * P @ V_new), rounds  # noqa: F821
        V = V_new


@mutant(
    ex="stretch",
    replaces="modified_policy_iteration",
    id="counts-sweeps",
    why="returns the number of sweeps (rounds times m) as the round count",
)
def modified_policy_iteration(P, R, gamma, m, tol):  # noqa: F811
    S = P.shape[0]
    V, rounds = np.zeros(S), 0
    while True:
        Q = R + gamma * P @ V
        pi = greedy(Q)  # noqa: F821
        V_new = Q.max(axis=1)
        P_pi, r_pi = P[np.arange(S), pi], R[np.arange(S), pi]
        for _ in range(m - 1):
            V_new = r_pi + gamma * P_pi @ V_new
        rounds += m
        if np.max(np.abs(V_new - V)) < tol:
            return V_new, greedy(R + gamma * P @ V_new), rounds  # noqa: F821
        V = V_new
