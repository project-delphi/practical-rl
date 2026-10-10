"""Plausible wrong versions of Module 1's functions.

scripts/check_notebooks.py --mode verify swaps each one into the notebook and requires
every checkpoint that calls the function to reject it. Each mutant is a mistake a
participant could plausibly make; `why` says which.
"""

import numpy as np

from prl.lab import mutant


@mutant(
    ex=1, replaces="discounted_return", id="discount-first", why="discounts the first reward too"
)
def discounted_return(rewards, gamma):
    return sum(r * gamma ** (k + 1) for k, r in enumerate(rewards))


@mutant(ex=1, replaces="discounted_return", id="no-gamma", why="forgets the discount")
def discounted_return(rewards, gamma):  # noqa: F811
    return float(sum(rewards))


@mutant(
    ex=2, replaces="gridworld_arrays", id="cliff-costs-one", why="the cliff costs -1, like any move"
)
def gridworld_arrays(rows, cols):
    S, A = rows * cols, 4
    P, R = np.zeros((S, A, S)), np.zeros((S, A))
    start, goal = index(rows - 1, 0, cols), index(rows - 1, cols - 1, cols)  # noqa: F821
    cliff = [index(rows - 1, c, cols) for c in range(1, cols - 1)]  # noqa: F821
    for s in range(S):
        r, c = divmod(s, cols)
        for a in range(A):
            if s == goal:
                P[s, a, goal] = 1.0
            elif s in cliff:
                P[s, a, start] = 1.0
            else:
                s2 = index(*move(r, c, a, rows, cols), cols)  # noqa: F821
                P[s, a, start if s2 in cliff else s2] = 1.0
                R[s, a] = -1.0
    return P, R


@mutant(
    ex=2, replaces="gridworld_arrays", id="goal-not-absorbing", why="treats the goal like any cell"
)
def gridworld_arrays(rows, cols):  # noqa: F811
    S, A = rows * cols, 4
    P, R = np.zeros((S, A, S)), np.zeros((S, A))
    start = index(rows - 1, 0, cols)  # noqa: F821
    cliff = [index(rows - 1, c, cols) for c in range(1, cols - 1)]  # noqa: F821
    for s in range(S):
        r, c = divmod(s, cols)
        for a in range(A):
            if s in cliff:
                P[s, a, start] = 1.0
                continue
            s2 = index(*move(r, c, a, rows, cols), cols)  # noqa: F821
            if s2 in cliff:
                P[s, a, start], R[s, a] = 1.0, -100.0
            else:
                P[s, a, s2], R[s, a] = 1.0, -1.0
    return P, R


@mutant(
    ex="3a",
    replaces="inventory_spec",
    id="forgets-zero",
    why="forgets stock level 0 and ordering nothing",
)
def inventory_spec(capacity):
    return {"n_states": capacity, "n_actions": capacity}


@mutant(ex="3a", replaces="period_reward", id="no-cap", why="lets the order exceed capacity")
def period_reward(stock, order, demand, p):
    y = stock + order
    sales = min(y, demand)
    cost = (p["fixed_cost"] if order > 0 else 0.0) + p["unit_cost"] * order
    return (
        p["price"] * sales
        - cost
        - p["holding_cost"] * (y - sales)
        - p["lost_sales_penalty"] * (demand - sales)
    )


@mutant(
    ex="3a",
    replaces="period_reward",
    id="fixed-cost-always",
    why="charges the fixed cost even with no order",
)
def period_reward(stock, order, demand, p):  # noqa: F811
    q = min(order, p["capacity"] - stock)
    y = stock + q
    sales = min(y, demand)
    cost = p["fixed_cost"] + p["unit_cost"] * q
    return (
        p["price"] * sales
        - cost
        - p["holding_cost"] * (y - sales)
        - p["lost_sales_penalty"] * (demand - sales)
    )


@mutant(
    ex="3b",
    replaces="inventory_arrays",
    id="last-demand-only",
    why="assigns R instead of accumulating the expectation",
)
def inventory_arrays(capacity, pmf, p):
    S = A = capacity + 1
    P, R = np.zeros((S, A, S)), np.zeros((S, A))
    for s in range(S):
        for a in range(A):
            y = s + min(a, capacity - s)
            for d, prob in enumerate(pmf):
                P[s, a, y - min(y, d)] += prob
                R[s, a] = period_reward(s, a, d, p)  # noqa: F821
    return P, R


@mutant(
    ex="3b",
    replaces="inventory_arrays",
    id="no-cap",
    why="lets stock exceed capacity, clipped at the end",
)
def inventory_arrays(capacity, pmf, p):  # noqa: F811
    S = A = capacity + 1
    P, R = np.zeros((S, A, S)), np.zeros((S, A))
    for s in range(S):
        for a in range(A):
            y = s + a
            for d, prob in enumerate(pmf):
                P[s, a, min(y - min(y, d), capacity)] += prob
                R[s, a] += prob * period_reward(s, a, d, p)  # noqa: F821
    return P, R


@mutant(ex=4, replaces="evaluate_exact", id="no-discount", why="drops gamma from the solve")
def evaluate_exact(P, R, gamma, pi):
    P_pi = np.einsum("sa,sat->st", pi, P)
    r_pi = (pi * R).sum(axis=1)
    return np.linalg.lstsq(np.eye(len(r_pi)) - 0.999999 * P_pi, r_pi, rcond=None)[0]


@mutant(
    ex=4,
    replaces="evaluate_exact",
    id="ignores-policy",
    why="averages rewards over actions uniformly",
)
def evaluate_exact(P, R, gamma, pi):  # noqa: F811
    P_pi = P.mean(axis=1)
    r_pi = R.mean(axis=1)
    return np.linalg.solve(np.eye(len(r_pi)) - gamma * P_pi, r_pi)


@mutant(ex=5, replaces="evaluate_iterative", id="no-abs", why="tests the change without abs()")
def evaluate_iterative(P, R, gamma, pi, tol):
    P_pi = np.einsum("sa,sat->st", pi, P)
    r_pi = (pi * R).sum(axis=1)
    V, k = np.zeros(len(r_pi)), 0
    while True:
        V_new = r_pi + gamma * P_pi @ V
        k += 1
        if np.max(V_new - V) < tol:
            return V_new, k
        V = V_new


@mutant(ex=5, replaces="evaluate_iterative", id="fixed-iterations", why="runs a fixed 50 updates")
def evaluate_iterative(P, R, gamma, pi, tol):  # noqa: F811
    P_pi = np.einsum("sa,sat->st", pi, P)
    r_pi = (pi * R).sum(axis=1)
    V = np.zeros(len(r_pi))
    for _ in range(50):
        V = r_pi + gamma * P_pi @ V
    return V, 50


@mutant(
    ex=6, replaces="bellman_residual", id="mean-not-max", why="averages the residual over states"
)
def bellman_residual(P, R, gamma, pi, V):
    P_pi = np.einsum("sa,sat->st", pi, P)
    r_pi = (pi * R).sum(axis=1)
    return float(np.mean(np.abs(r_pi + gamma * P_pi @ V - V)))


@mutant(ex=6, replaces="bellman_residual", id="no-gamma", why="forgets the discount in T^pi")
def bellman_residual(P, R, gamma, pi, V):  # noqa: F811
    P_pi = np.einsum("sa,sat->st", pi, P)
    r_pi = (pi * R).sum(axis=1)
    return float(np.max(np.abs(r_pi + P_pi @ V - V)))


@mutant(
    ex=6,
    replaces="evaluate_suspect",
    id="still-transposed",
    why="leaves the colleague's bug in place",
)
def evaluate_suspect(P, R, gamma, pi):
    P_pi = np.einsum("sa,sat->st", pi, P)
    r_pi = (pi * R).sum(axis=1)
    return np.linalg.solve(np.eye(len(r_pi)) - gamma * P_pi.T, r_pi)
