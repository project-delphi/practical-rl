"""Write the checkpoint fixtures in prl/src/prl/_expected/.

Fixtures are precomputed expected values, so prl ships answers to checks,
never solver code. This script may compute them however it likes; it is
not part of prl. Deterministic: rerunning it must produce identical files
(CI's drift gate checks this).
"""

from __future__ import annotations

import sys
from fractions import Fraction
from math import comb
from pathlib import Path

import numpy as np

from package_data import same_content, save_npz

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "prl" / "src" / "prl" / "_expected"


def _save(name: str, **arrays: np.ndarray) -> bytes:
    return save_npz(**arrays)


def m01_returns() -> bytes:
    cases = [
        ([1.0, 1.0, 1.0], 0.9),
        ([0.0, 0.0, 10.0], 0.5),
        ([-1.0, -1.0, -1.0, -1.0, 0.0], 1.0),
        ([5.0], 0.99),
        ([1.0, -2.0, 3.0, -4.0], 0.8),
    ]
    width = max(len(r) for r, _ in cases)
    rewards = np.zeros((len(cases), width))
    lengths = np.zeros(len(cases), dtype=np.int64)
    gammas = np.zeros(len(cases))
    answers = np.zeros(len(cases))
    for i, (rs, g) in enumerate(cases):
        rewards[i, : len(rs)] = rs
        lengths[i] = len(rs)
        gammas[i] = g
        answers[i] = sum(r * g**k for k, r in enumerate(rs))
    return _save("m01_returns", rewards=rewards, lengths=lengths, gammas=gammas, returns=answers)


# Module 1 lab parameters (also printed in the notebook; see the Module 1 contract).
M01_INVENTORY = dict(
    capacity=10,
    demand_mean=4.0,
    max_demand=10,
    price=4.0,
    unit_cost=2.0,
    fixed_cost=5.0,
    holding_cost=0.2,
    lost_sales_penalty=1.0,
    gamma=0.95,
)


def _policy_eval(P, R, gamma, pi):
    """Exact V^pi by linear solve (fixture code; never shipped in prl)."""
    S = P.shape[0]
    P_pi = np.einsum("sa,sat->st", pi, P)
    r_pi = (pi * R).sum(axis=1)
    return np.linalg.solve(np.eye(S) - gamma * P_pi, r_pi), P_pi, r_pi


def _iterate(P_pi, r_pi, gamma, tol):
    V = np.zeros_like(r_pi)
    k = 0
    while True:
        V_new = r_pi + gamma * P_pi @ V
        k += 1
        if np.max(np.abs(V_new - V)) < tol:
            return V_new, k
        V = V_new


def m01_eval() -> bytes:
    sys.path.insert(0, str(ROOT / "prl" / "src"))
    from prl.envs import CliffGridworld, InventoryMDP

    out: dict[str, np.ndarray] = {}
    # Case "grid": 3x4 cliff gridworld, uniform random policy, gamma 0.9.
    g = CliffGridworld(3, 4, gamma=0.9)
    pi_g = np.full((g.n_states, 4), 0.25)
    V_g, _, _ = _policy_eval(g.P, g.R, 0.9, pi_g)
    out.update(grid_P=g.P, grid_R=g.R, grid_pi=pi_g, grid_gamma=np.array(0.9), grid_V=V_g)
    _, Pg_pi, rg_pi = _policy_eval(g.P, g.R, 0.9, pi_g)
    out.update(grid_k=np.array(_iterate(Pg_pi, rg_pi, 0.9, 1e-6)[1], dtype=np.int64))
    # Case "inv": the lab's inventory MDP, base-stock policy "order up to 7".
    inv = InventoryMDP(**M01_INVENTORY)
    pi_i = np.zeros((inv.n_states, inv.n_actions))
    for s in range(inv.n_states):
        pi_i[s, max(0, 7 - s)] = 1.0
    V_i, _, _ = _policy_eval(inv.P, inv.R, 0.95, pi_i)
    out.update(inv_pi=pi_i, inv_V=V_i)
    # Case "rand": a seeded random MDP (5 states, 3 actions) and a stochastic policy.
    rng = np.random.default_rng(20261009)
    P = rng.dirichlet(np.ones(5), size=(5, 3))
    R = rng.normal(size=(5, 3))
    pi = rng.dirichlet(np.ones(3), size=5)
    gammas = np.array([0.5, 0.9, 0.99])
    out.update(rand_P=P, rand_R=R, rand_pi=pi, gammas=gammas)
    Vs, ks = [], []
    for gm in gammas:
        V, P_pi, r_pi = _policy_eval(P, R, gm, pi)
        Vs.append(V)
        ks.append(_iterate(P_pi, r_pi, gm, 1e-6)[1])
    out.update(rand_V=np.array(Vs), rand_k=np.array(ks, dtype=np.int64), tol=np.array(1e-6))
    # Bellman residual cases: V^pi plus known perturbations, gamma 0.9.
    V9, P_pi9, r_pi9 = _policy_eval(P, R, 0.9, pi)
    deltas = np.array([np.zeros(5), np.full(5, 1.0), rng.normal(size=5)])
    deltas = np.vstack([deltas, 3 * np.eye(5)[0]])
    Vtest = V9 + deltas
    resid = np.array([np.max(np.abs(r_pi9 + 0.9 * P_pi9 @ v - v)) for v in Vtest])
    out.update(res_V=Vtest, res_expected=resid)

    # Period rewards: (stock, order, demand) -> reward, with the lab parameters.
    def reward(p, stock, order, demand):
        q = min(order, p["capacity"] - stock)
        y = stock + q
        sales = min(y, demand)
        left = y - sales
        cost = (p["fixed_cost"] if q > 0 else 0.0) + p["unit_cost"] * q
        return (
            p["price"] * sales
            - cost
            - p["holding_cost"] * left
            - p["lost_sales_penalty"] * (demand - sales)
        )

    # A second parameter set catches hard-coded prices.
    p2 = dict(
        M01_INVENTORY,
        capacity=6,
        price=5.0,
        unit_cost=1.5,
        fixed_cost=3.0,
        holding_cost=0.5,
        lost_sales_penalty=2.0,
    )
    keys = ["capacity", "price", "unit_cost", "fixed_cost", "holding_cost", "lost_sales_penalty"]
    cases2 = np.array([[4, 5, 3], [0, 6, 6], [6, 2, 1]], dtype=np.int64)
    out.update(
        p2_keys=np.array(keys),
        p2_values=np.array([float(p2[k]) for k in keys]),
        reward2_cases=cases2,
        reward2_expected=np.array([reward(p2, *map(int, c)) for c in cases2]),
    )
    p = M01_INVENTORY
    # (10, 3, 2): a full shelf caps the order to 0, so no fixed cost is due.
    cases = np.array(
        [[0, 7, 4], [3, 0, 5], [6, 1, 2], [9, 5, 10], [10, 0, 0], [10, 3, 2]], dtype=np.int64
    )
    rewards = []
    for stock, order, demand in cases:
        q = min(order, p["capacity"] - stock)
        y = stock + q
        sales = min(y, demand)
        left = y - sales
        cost = (p["fixed_cost"] if q > 0 else 0.0) + p["unit_cost"] * q
        rewards.append(
            p["price"] * sales
            - cost
            - p["holding_cost"] * left
            - p["lost_sales_penalty"] * (demand - sales)
        )
    out.update(reward_cases=cases, reward_expected=np.array(rewards))
    return _save("m01_eval", **out)


# ---------------------------------------------------------------- Module 2 (planning)
# Solver code for the fixtures: value iteration, policy iteration, modified policy
# iteration and the counts estimator. It lives here, never in prl.


def _value_iteration(P, R, gamma, tol):
    """V <- max_a [R + gamma P V] from V = 0 until max|V_new - V| < tol; (V, Q, k, log)."""
    V, k, log = np.zeros(P.shape[0]), 0, []
    while True:
        Q = R + gamma * P @ V
        V_new = Q.max(axis=1)
        k += 1
        log.append(float(np.max(np.abs(V_new - V))))
        if log[-1] < tol:
            return V_new, R + gamma * P @ V_new, k, log
        V = V_new


def _exact_value(P, R, gamma, pi):
    """V^pi of a deterministic policy by a linear solve."""
    S = P.shape[0]
    P_pi, r_pi = P[np.arange(S), pi], R[np.arange(S), pi]
    return np.linalg.solve(np.eye(S) - gamma * P_pi, r_pi)


#: Ties. Orders the capacity caps to the same delivery are identical actions (bitwise equal
#: rows of P and R), so the shop's Q ties exactly on paper. A BLAS may still split such a tie
#: by a few units in the last place: OpenBLAS computes some rows of P @ V along a different
#: kernel path. Every tie decision here therefore compares within _tie_tol(Q) =
#: TIE_TOL * (1 + max|Q|), about 1.06e-7 on the shop: six orders of magnitude above rounding
#: (about 1e-13 at |Q| near 100) and six below the smallest real gap between distinct
#: deliveries (0.032 in value iteration's Q_0 = R, 0.125 along policy iteration's path).
TIE_TOL = 1e-9


def _tie_tol(Q):
    return TIE_TOL * (1.0 + float(np.abs(Q).max()))


def _greedy(Q):
    """The FIRST action within _tie_tol of each row's max: the same choice on every platform."""
    return (Q >= Q.max(axis=1, keepdims=True) - _tie_tol(Q)).argmax(axis=1)


def _policy_iteration(P, R, gamma, pi0, *, tie_tol=True):
    """Exact evaluation and greedy improvement, keeping the current action when it is within
    _tie_tol of the best (tie_tol=False: the exact comparison, for an assertion only)."""
    S = P.shape[0]
    pi, k = np.asarray(pi0, dtype=int).copy(), 0
    while True:
        V = _exact_value(P, R, gamma, pi)
        k += 1
        Q = R + gamma * P @ V
        keep = Q[np.arange(S), pi] >= Q.max(axis=1) - (_tie_tol(Q) if tie_tol else 0.0)
        pi_new = np.where(keep, pi, _greedy(Q))
        if np.array_equal(pi_new, pi):
            return V, pi, k
        pi = pi_new


def _modified_policy_iteration(P, R, gamma, m, tol):
    S = P.shape[0]
    V, rounds = np.zeros(S), 0
    while True:
        Q = R + gamma * P @ V
        pi = _greedy(Q)
        V_new = Q.max(axis=1)
        P_pi, r_pi = P[np.arange(S), pi], R[np.arange(S), pi]
        for _ in range(m - 1):
            V_new = r_pi + gamma * P_pi @ V_new
        rounds += 1
        if np.max(np.abs(V_new - V)) < tol:
            return V_new, _greedy(R + gamma * P @ V_new), rounds
        V = V_new


def _optimal(P, R, gamma, tol=1e-6):
    """(V*, pi*, Q*, k_vi): pi* from value iteration, V* by the exact solve of pi*."""
    _, Q_vi, k, _ = _value_iteration(P, R, gamma, tol)
    pi_star = _greedy(Q_vi)
    V_star = _exact_value(P, R, gamma, pi_star)
    return V_star, pi_star, R + gamma * P @ V_star, k


def _loud_mdp(S=100, gamma=0.95):
    """The 'mostly settled' MDP: every state absorbing, R = 0 except R[0] = [1, 0.5]."""
    P = np.zeros((S, 2, S))
    P[np.arange(S), :, np.arange(S)] = 1.0
    R = np.zeros((S, 2))
    R[0] = [1.0, 0.5]
    return P, R, gamma


def _estimate_model(log, n_states, n_actions):
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
                R_hat[s, a] /= counts[s, a]
            else:
                P_hat[s, a, s] = 1.0
    return P_hat, R_hat, counts


DUP_DELTA = 1e-13


def m02_plan() -> bytes:
    sys.path.insert(0, str(ROOT / "prl" / "src"))
    from prl.envs import CliffGridworld, InventoryMDP

    out: dict[str, np.ndarray] = {}
    tol = 1e-6
    # The shop (Module 1's numbers), gamma 0.95.
    inv = InventoryMDP(**M01_INVENTORY)
    S = inv.n_states
    V_star, pi_star, Q_star, k_vi = _optimal(inv.P, inv.R, 0.95, tol)
    assert k_vi == 299 and pi_star.tolist() == [10, 9, 8, 7, 0, 0, 0, 0, 0, 0, 0]
    pi7 = np.maximum(0, 7 - np.arange(S))
    out.update(
        inv_V_star=V_star,
        inv_pi_star=pi_star.astype(np.int64),
        inv_Q_star=Q_star,
        inv_pi7=pi7.astype(np.int64),
        inv_k_vi=np.array(k_vi, dtype=np.int64),
        inv_k_pi_zero=np.array(_policy_iteration(inv.P, inv.R, 0.95, np.zeros(S, int))[2]),
        inv_k_pi_7=np.array(_policy_iteration(inv.P, inv.R, 0.95, pi7)[2]),
        inv_k_mpi_1=np.array(_modified_policy_iteration(inv.P, inv.R, 0.95, 1, tol)[2]),
        inv_k_mpi_5=np.array(_modified_policy_iteration(inv.P, inv.R, 0.95, 5, tol)[2]),
    )
    # A seeded random MDP (5 states, 3 actions), gamma 0.9, and its negated-rewards twin
    # (the optimality backup does not commute with a sign flip, so both get fixtures).
    rng = np.random.default_rng(20261017)
    P = rng.dirichlet(np.ones(5), size=(5, 3))
    R = rng.normal(size=(5, 3))
    out.update(rand_P=P, rand_R=R, rand_gamma=np.array(0.9))
    for prefix, sign in (("rand", 1.0), ("rand_neg", -1.0)):
        V, pi, _, k = _optimal(P, sign * R, 0.9, tol)
        out[f"{prefix}_V_star"] = V
        out[f"{prefix}_pi_star"] = pi.astype(np.int64)
        out[f"{prefix}_k_vi"] = np.array(k, dtype=np.int64)
    out["rand_k_pi"] = np.array(_policy_iteration(P, R, 0.9, np.zeros(5, int))[2])
    # The 3x4 cliff gridworld at gamma 0.9 (in-place updates stop earlier).
    g = CliffGridworld(3, 4, gamma=0.9)
    V_g, _, _, k_g = _optimal(g.P, g.R, 0.9, tol)
    out.update(grid_V_star=V_g, grid_k_vi=np.array(k_g, dtype=np.int64))
    # The 'mostly settled' MDP: stopping on the mean change stops at sweep 181 of 271.
    P_l, R_l, gamma_l = _loud_mdp()
    V_l, _, _, k_l = _optimal(P_l, R_l, gamma_l, tol)
    assert k_l == 271 and np.allclose(V_l, np.r_[1 / (1 - gamma_l), np.zeros(99)])
    out.update(
        loud_S=np.array(100, dtype=np.int64),
        loud_gamma=np.array(gamma_l),
        loud_V_star=V_l,
        loud_k_vi=np.array(k_l, dtype=np.int64),
    )
    # The 4x12 cliff with slip 0.2 at gamma 0.95: the mean-stop bug shows (72 of 76 sweeps).
    c = CliffGridworld(4, 12, slip=0.2, gamma=0.95)
    V_c, _, _, k_c = _optimal(c.P, c.R, 0.95, tol)
    assert k_c == 76
    out.update(cliff_V_star=V_c, cliff_k_vi=np.array(k_c, dtype=np.int64))
    # Greedy cases: random (6, 4) arrays with distinct row maxima, and one with an exact tie.
    Qs = rng.normal(size=(4, 6, 4)).round(3)
    for Q in Qs:
        assert all(np.sum(row == row.max()) == 1 for row in Q)
    Q_tie = rng.normal(size=(6, 4)).round(3)
    Q_tie[2, 1] = Q_tie[2, 3] = Q_tie[2].max() + 1.0
    out.update(
        greedy_Q=Qs,
        greedy_expected=Qs.argmax(axis=2).astype(np.int64),
        greedy_tie_Q=Q_tie,
        greedy_tie_row=np.array(2, dtype=np.int64),
    )
    # A near-duplicate action: action 3 is a copy of the action pi* uses most, with its
    # reward lowered by DUP_DELTA = 1e-13, the size of a difference rounding can make. Started
    # from 'always 3', a tie rule with a tolerance keeps 3 where it is optimal; an exact
    # comparison (or argmax alone) switches to the original, on every platform: 1e-13 is
    # about 25 ulps of the random MDP's Q (|Q| < 10), far above the few ulps a BLAS adds.
    # Action 4, a copy of action 0 with a reward 1 lower, is never chosen; it makes A = S = 5,
    # so a mistaken max over states (axis 0) gives a wrong policy rather than a shape error.
    _, pi_r, _, _ = _optimal(P, R, 0.9, tol)
    dup_of = int(np.bincount(pi_r, minlength=3).argmax())
    P_d = np.concatenate([P, P[:, dup_of : dup_of + 1], P[:, :1]], axis=1)
    R_d = np.concatenate([R, R[:, dup_of : dup_of + 1] - DUP_DELTA, R[:, :1] - 1.0], axis=1)
    pi0_d = np.full(5, 3)
    V_d, pi_d, k_d = _policy_iteration(P_d, R_d, 0.9, pi0_d)
    assert np.any(pi_d == 3) and not np.any(pi_d == dup_of) and not np.any(pi_d == 4)
    exact_pi = _policy_iteration(P_d, R_d, 0.9, pi0_d, tie_tol=False)[1]
    assert not np.any(exact_pi == 3) and np.abs(R_d).max() < 10
    out.update(
        dup_P=P_d,
        dup_R=R_d,
        dup_pi0=pi0_d.astype(np.int64),
        dup_V_star=V_d,
        dup_pi_star=pi_d.astype(np.int64),
        dup_k_pi=np.array(k_d, dtype=np.int64),
    )
    return _save("m02_plan", **out)


def _random_log(mdp, days, rng):
    """random_log of the Module 2 lab, reproduced exactly: uniform orders from stock 0."""
    rows, s = [], 0
    for _ in range(days):
        a = int(rng.integers(0, mdp.n_actions))
        s2, r, _ = mdp.sample_step(s, a, rng)
        rows.append((s, a, r, s2))
        s = s2
    return np.array(rows, dtype=float)


def m02_model() -> bytes:
    sys.path.insert(0, str(ROOT / "prl" / "src"))
    from prl.envs import InventoryMDP

    inv = InventoryMDP(**M01_INVENTORY)
    log = _random_log(inv, 200, np.random.default_rng(0))
    P_hat, R_hat, counts = _estimate_model(log, inv.n_states, inv.n_actions)
    assert int((counts > 0).sum()) == 83
    # The five-row hand log of knowledge check 2.5: 2 states, 2 actions.
    hand = np.array(
        [[0, 1, -2, 1], [1, 0, 3, 1], [1, 0, 3, 0], [0, 1, -2, 1], [1, 0, 3, 1]], dtype=float
    )
    hP, hR, hc = _estimate_model(hand, 2, 2)
    assert np.allclose(hP[1, 0], [1 / 3, 2 / 3]) and hR[1, 0] == 3
    return _save(
        "m02_model",
        log=log,
        P_hat=P_hat,
        R_hat=R_hat,
        counts=counts,
        hand_log=hand,
        hand_P_hat=hP,
        hand_R_hat=hR,
        hand_counts=hc,
    )


def _sample_transitions(mdp, N, rng):
    """sample_transitions of the Module 2 lab, reproduced exactly: N draws at every (s, a)."""
    rows = []
    for s in range(mdp.n_states):
        for a in range(mdp.n_actions):
            for _ in range(N):
                s2, r, _ = mdp.sample_step(s, a, rng)
                rows.append((s, a, r, s2))
    return np.array(rows, dtype=float)


def m02_plan_eval() -> bytes:
    """Exercise 4's loss and eps_P for seeds 0-4 at N = 2 and N = 100 (the lab's sampler).

    The loss is max_s [V*(s) - V^pi_hat(s)] in the true shop. Orders that the capacity caps
    to the same delivery are identical actions, so a different tie-break in greedy changes
    pi_hat but not the loss. The assert below checks that no other near-tie (within 1e-6)
    decides pi_hat, so a correct solution reproduces these losses to 1e-6.
    """
    sys.path.insert(0, str(ROOT / "prl" / "src"))
    from prl.envs import InventoryMDP

    inv = InventoryMDP(**M01_INVENTORY)
    S, A, cap = inv.n_states, inv.n_actions, inv.capacity
    V_star, _, _, _ = _optimal(inv.P, inv.R, 0.95)
    eff = np.minimum(np.arange(A)[None, :], cap - np.arange(S)[:, None])  # delivered units
    Ns = np.array([2, 100], dtype=np.int64)
    seeds = np.arange(5, dtype=np.int64)
    loss = np.zeros((len(Ns), len(seeds)))
    eps_P = np.zeros_like(loss)
    eps_R = np.zeros_like(loss)
    for i, N in enumerate(Ns):
        for j, seed in enumerate(seeds):
            log = _sample_transitions(inv, int(N), np.random.default_rng(int(seed)))
            P_hat, R_hat, _ = _estimate_model(log, S, A)
            _, Q_hat, _, _ = _value_iteration(P_hat, R_hat, 0.95, 1e-6)
            pi_hat = _greedy(Q_hat)
            for s in range(S):
                rivals = Q_hat[s][eff[s] != eff[s, pi_hat[s]]]
                assert rivals.size == 0 or Q_hat[s].max() - rivals.max() > 1e-6, (N, seed, s)
            loss[i, j] = np.max(V_star - _exact_value(inv.P, inv.R, 0.95, pi_hat))
            eps_P[i, j] = np.abs(inv.P - P_hat).sum(axis=2).max()
            eps_R[i, j] = np.abs(inv.R - R_hat).max()
    assert abs(eps_P[0].mean() - 1.869) < 5e-4 and abs(loss[0].mean() - 31.54) < 5e-3
    assert np.allclose(
        eps_R.round(2), [[12.58, 12.62, 18.92, 14.68, 14.87], [2.54, 2.33, 2.29, 2.16, 2.45]]
    )
    # Checkpoint 4.2: the planning-loss bound, rescaled to the reward range (eq-plan-loss).
    r_range = float(inv.R.max() - inv.R.min())
    bound = _planning_loss_bound(eps_P, eps_R, r_range, 0.95)
    assert np.all(bound >= loss)
    bound_cases = np.array(
        [[0.1, 0.0, 1.0, 0.95], [0.2, 0.0, 1.0, 0.9], [0.5, 3.0, 25.0, 0.95], [0.0, 1.0, 1.0, 0.9]]
    )
    bound_expected = np.array([_planning_loss_bound(*c) for c in bound_cases])
    assert np.allclose(bound_expected, [76.0, 36.0, 9620.0, 20.0], rtol=1e-12)
    return _save(
        "m02_plan_eval",
        Ns=Ns,
        seeds=seeds,
        loss=loss,
        eps_P=eps_P,
        eps_R=eps_R,
        r_range=np.array(r_range),
        bound=bound,
        bound_cases=bound_cases,
        bound_expected=bound_expected,
    )


def _planning_loss_bound(eps_P, eps_R, r_range, gamma):
    """eq-plan-loss rescaled to rewards in a range r_range."""
    return 2 * eps_R / (1 - gamma) + 2 * gamma * eps_P * r_range / (1 - gamma) ** 2


def _optimality_residual(P, R, gamma, V):
    return float(np.max(np.abs((R + gamma * P @ V).max(axis=1) - V)))


def m02_residual() -> bytes:
    """Checkpoint 5.1: value vectors and their optimality residuals, on the shop and the
    random MDP of m02_plan (negative rewards)."""
    sys.path.insert(0, str(ROOT / "prl" / "src"))
    from prl.envs import InventoryMDP

    inv = InventoryMDP(**M01_INVENTORY)
    V_star = _optimal(inv.P, inv.R, 0.95)[0]
    e5 = np.eye(inv.n_states)[5]
    shop_V = np.array([V_star, V_star + 1, V_star - 2, V_star + 3 * e5, np.zeros_like(V_star)])
    shop_res = np.array([_optimality_residual(inv.P, inv.R, 0.95, V) for V in shop_V])
    assert shop_res[0] < 1e-9 and np.allclose(shop_res[1:3], [0.05, 0.1])
    assert round(shop_res[3], 4) == 2.9478 and round(shop_res[4], 4) == 15.0293
    rng = np.random.default_rng(20261017)
    P = rng.dirichlet(np.ones(5), size=(5, 3))
    R = rng.normal(size=(5, 3))
    rand_V_star = _optimal(P, R, 0.9)[0]
    rand_V = np.array(
        [
            rand_V_star,
            rand_V_star + rng.normal(size=5),
            np.zeros(5),
            rand_V_star - 1.5 * np.eye(5)[2],
        ]
    )
    rand_res = np.array([_optimality_residual(P, R, 0.9, V) for V in rand_V])
    return _save(
        "m02_residual",
        shop_V=shop_V,
        shop_residual=shop_res,
        rand_P=P,
        rand_R=R,
        rand_gamma=np.array(0.9),
        rand_V=rand_V,
        rand_residual=rand_res,
    )


# ---------------------------------------------------------------- Module 3 (model-free)
# Hand-made episodes: (states s_0..s_T, rewards r_0..r_{T-1}, terminated, truncated).
# Repeated states within an episode (so every-visit differs from first-visit) and an
# episode of length 1. n_states = 5; state 4 is the terminal state.
M03_MC_EPISODES = [
    ([0, 1, 0, 4], [-1.0, -1.0, -1.0], True, False),  # knowledge check Q3.1 (gamma 0.5)
    ([0, 2, 2, 4], [-1.0, -3.0, 2.0], True, False),  # a repeat of state 2 in a row
    ([1, 0, 4], [5.0, -1.0], True, False),
    ([3, 4], [1.0], True, False),  # length 1
]
# TD(0) sees the same episodes in order, then one cut by a step limit: its last transition
# (1 -> 3) must bootstrap from V[3], which the episode before has already moved.
M03_TD_EPISODES = M03_MC_EPISODES + [([2, 1, 3], [-1.0, 0.5], False, True)]
# The page's worked example: corridor A -> B -> goal, two identical episodes.
M03_CORRIDOR = [([0, 1, 2], [-1.0, -1.0], True, False)] * 2
M03_MC_GAMMAS = (0.5, 0.9)
# (episode set, alpha, gamma, n_states, v0); v0 != 0 makes a bootstrap through the terminal show.
M03_TD_CASES = [("corridor", 0.5, 0.9, 3, 0.0), ("td", 0.5, 0.9, 5, 0.0), ("td", 0.5, 0.9, 5, 1.0)]
M03_TD_CASES.append(("td", 0.5, 0.5, 5, 1.0))
M03_NSTEP_TD = [(0.5, 0.9, 2), (0.5, 0.9, 3)]  # stretch: (alpha, gamma, n)


def _returns(rewards, gamma):
    G, out = 0.0, [0.0] * len(rewards)
    for t in reversed(range(len(rewards))):
        G = rewards[t] + gamma * G
        out[t] = G
    return out


def _mc(episodes, gamma, n_states, *, first_visit=True):
    """Monte Carlo averages of the returns after visits (fixture code; never shipped in prl)."""
    total, count = np.zeros(n_states), np.zeros(n_states)
    for states, rewards, _term, _trunc in episodes:
        seen = set()
        for s, G in zip(states[:-1], _returns(rewards, gamma), strict=True):
            if first_visit and s in seen:
                continue
            seen.add(s)
            total[s] += G
            count[s] += 1
    return np.divide(total, count, out=np.zeros(n_states), where=count > 0)


def _td0(episodes, alpha, gamma, n_states, v0):
    V = np.full(n_states, float(v0))
    for states, rewards, terminated, _trunc in episodes:
        T = len(rewards)
        for t in range(T):
            s, s2 = states[t], states[t + 1]
            boot = 0.0 if (terminated and t == T - 1) else V[s2]
            V[s] += alpha * (rewards[t] + gamma * boot - V[s])
    return V


def _nstep_target(rewards, states, t, n, V, gamma, terminated):
    T = len(rewards)
    end = min(t + n, T)
    G = sum(gamma ** (k - t) * rewards[k] for k in range(t, end))
    if end < T or not terminated:
        G += gamma ** (end - t) * V[states[end]]
    return G


def _nstep_td(episodes, alpha, gamma, n_states, n):
    """Offline n-step TD in time order: each target uses V as it stands when t is processed."""
    V = np.zeros(n_states)
    for states, rewards, terminated, _trunc in episodes:
        for t in range(len(rewards)):
            G = _nstep_target(rewards, states, t, n, V, gamma, terminated)
            V[states[t]] += alpha * (G - V[states[t]])
    return V


def _pack(episodes) -> dict[str, np.ndarray]:
    width = max(len(r) for _, r, _, _ in episodes)
    states = np.full((len(episodes), width + 1), -1, dtype=np.int64)
    rewards = np.zeros((len(episodes), width))
    for i, (s, r, _, _) in enumerate(episodes):
        states[i, : len(s)] = s
        rewards[i, : len(r)] = r
    return {
        "states": states,
        "rewards": rewards,
        "lengths": np.array([len(r) for _, r, _, _ in episodes], dtype=np.int64),
        "terminated": np.array([e[2] for e in episodes]),
        "truncated": np.array([e[3] for e in episodes]),
    }


def m03_safe_route(gamma: float = 0.95, epsilon: float = 0.1):
    """The lab's epsilon-soft safe route on the 4x12 cliff: (mdp, pi, route, V_pi)."""
    sys.path.insert(0, str(ROOT / "prl" / "src"))
    from prl.envs import CliffGridworld

    mdp = CliffGridworld(4, 12, gamma=gamma)
    S, A, cols = mdp.n_states, mdp.n_actions, mdp.cols
    pi = np.full((S, A), epsilon / A)
    for s in range(S):
        r, c = divmod(s, cols)
        pi[s, 2 if c == cols - 1 else (1 if r == 0 else 0)] += 1.0 - epsilon  # down, right, up
    s, route = mdp.start, [mdp.start]
    while s != mdp.goal:
        s = int(np.argmax(mdp.P[s, int(np.argmax(pi[s]))]))
        route.append(s)
    V_pi, _, _ = _policy_eval(mdp.P, mdp.R, gamma, pi)
    return mdp, pi, np.array(route[:-1], dtype=np.int64), V_pi


def m03_prediction() -> bytes:
    out: dict[str, np.ndarray] = {}
    out.update({f"mc_{k}": v for k, v in _pack(M03_MC_EPISODES).items()})
    out.update({f"td_{k}": v for k, v in _pack(M03_TD_EPISODES).items()})
    out.update({f"corridor_{k}": v for k, v in _pack(M03_CORRIDOR).items()})
    gammas = np.array(M03_MC_GAMMAS)
    out["mc_gammas"] = gammas
    out["mc_first_V"] = np.array([_mc(M03_MC_EPISODES, g, 5) for g in gammas])
    out["mc_every_V"] = np.array([_mc(M03_MC_EPISODES, g, 5, first_visit=False) for g in gammas])
    out["mc_one_V"] = _mc(M03_MC_EPISODES[:1], 0.5, 5)  # the first episode alone, gamma 0.5
    out["mc_one_every_V"] = _mc(M03_MC_EPISODES[:1], 0.5, 5, first_visit=False)
    sets = {"corridor": M03_CORRIDOR, "td": M03_TD_EPISODES}
    out["td_case_set"] = np.array([c[0] for c in M03_TD_CASES])
    out["td_case_alpha"] = np.array([c[1] for c in M03_TD_CASES])
    out["td_case_gamma"] = np.array([c[2] for c in M03_TD_CASES])
    out["td_case_n_states"] = np.array([c[3] for c in M03_TD_CASES], dtype=np.int64)
    out["td_case_v0"] = np.array([c[4] for c in M03_TD_CASES])
    width = max(c[3] for c in M03_TD_CASES)
    td_V = np.full((len(M03_TD_CASES), width), np.nan)
    for i, (name, alpha, gamma, n_states, v0) in enumerate(M03_TD_CASES):
        td_V[i, :n_states] = _td0(sets[name], alpha, gamma, n_states, v0)
    out["td_case_V"] = td_V
    out["nstep_td_alpha"] = np.array([c[0] for c in M03_NSTEP_TD])
    out["nstep_td_gamma"] = np.array([c[1] for c in M03_NSTEP_TD])
    out["nstep_td_n"] = np.array([c[2] for c in M03_NSTEP_TD], dtype=np.int64)
    out["nstep_td_V"] = np.array(
        [_nstep_td(M03_TD_EPISODES, a, g, 5, n) for a, g, n in M03_NSTEP_TD]
    )
    _mdp, pi, route, V_pi = m03_safe_route()
    out.update(route=route, V_pi=V_pi, pi=pi, gamma=np.array(0.95), epsilon=np.array(0.1))
    return _save("m03_prediction", **out)


def m03_nstep() -> bytes:
    # Trajectory 0: T = 6, a revisit of state 1, non-constant V with V[s_T] != 0 (so a
    # bootstrap from the end shows). Trajectory 1: the page's corridor, V = -2 everywhere.
    trajs = [
        (
            [0, 1, 2, 3, 1, 4, 5],
            [-1.0, 2.0, -0.5, 3.0, -2.0, 1.0],
            [0.3, -1.2, 2.5, -0.7, 1.1, 4.0],
        ),
        ([0, 1, 2, 3, 4], [-1.0] * 4, [-2.0] * 5 + [0.0]),
    ]
    states = np.full((2, 7), -1, dtype=np.int64)
    rewards = np.zeros((2, 6))
    V = np.array([v for _, _, v in trajs])
    for i, (s, r, _) in enumerate(trajs):
        states[i, : len(s)] = s
        rewards[i, : len(r)] = r
    cases = []
    for gamma in (0.9, 0.5):
        for terminated in (True, False):
            for t in (0, 1, 3, 5):
                for n in (1, 2, 3, 6, 100):
                    cases.append((0, t, n, gamma, terminated))
    for terminated in (True, False):
        for t in range(4):
            cases.append((1, t, 2, 0.9, terminated))
    expected = []
    for k, t, n, gamma, terminated in cases:
        s, r, v = trajs[k]
        expected.append(_nstep_target(r, s, t, n, np.array(v), gamma, terminated))
    return _save(
        "m03_nstep",
        states=states,
        rewards=rewards,
        lengths=np.array([len(r) for _, r, _ in trajs], dtype=np.int64),
        V=V,
        case_traj=np.array([c[0] for c in cases], dtype=np.int64),
        case_t=np.array([c[1] for c in cases], dtype=np.int64),
        case_n=np.array([c[2] for c in cases], dtype=np.int64),
        case_gamma=np.array([c[3] for c in cases]),
        case_terminated=np.array([c[4] for c in cases]),
        expected=np.array(expected),
    )


def m03_decomposition() -> bytes:
    """Exercise 2b: bias^2 and variance over seeds, plus the values of the usual mistakes."""
    rng = np.random.default_rng(20261017)
    cases = [
        # The knowledge check's one-state row: 3 seeds, so dividing by seeds - 1 is 1.5x off.
        (np.array([[-15.0], [-16.0], [-15.5]]), np.array([-15.37])),
    ]
    # 4 seeds x 5 states: per-state errors of both signs that cancel on average, and unequal
    # spreads.
    v = rng.normal(-10.0, 3.0, size=5)
    offsets = np.array([1.5, -1.5, 0.5, -0.5, 0.0])
    scales = np.array([0.1, 0.5, 1.0, 2.0, 0.3])
    cases.append((v + offsets + scales * rng.normal(size=(4, 5)), v))
    # The lab's shape: 10 seeds x 17 route states.
    v = rng.normal(-12.0, 2.0, size=17)
    cases.append((v + rng.normal(0.3, 0.6, size=17) + rng.normal(size=(10, 17)), v))
    out: dict[str, np.ndarray] = {"n_cases": np.array(len(cases), dtype=np.int64)}
    for i, (est, v_true) in enumerate(cases):
        mean = est.mean(axis=0)
        out[f"est_{i}"], out[f"v_{i}"] = est, v_true
        out[f"bias2_{i}"] = np.array(np.mean((mean - v_true) ** 2))
        out[f"var_{i}"] = np.array(np.mean(est.var(axis=0)))
        out[f"mse_{i}"] = np.array(np.mean((est - v_true) ** 2))
        out[f"var_ddof1_{i}"] = np.array(np.mean(est.var(axis=0, ddof=1)))
        out[f"bias2_pooled_{i}"] = np.array(np.mean(mean - v_true) ** 2)
    return _save("m03_decomposition", **out)


# ------------------------------------------------------------------ Module 4: bandits
# Fixture code may implement the lab's policies and its loop; prl.checks.m04 only replays these
# runs and compares. The functions below equal the lab's solutions statement for statement.
M04_R = np.array([0.4, 0.3, 0.5])  # click rates of headlines 0, 1, 2
M04_T_SCALE = 5000  # the lab's live T; checkpoint 5's reference run
M04_SCALE_SEEDS = list(range(20))


def _m04_argmax_tie(x):
    x = np.asarray(x, dtype=float)
    top = x.max()
    if np.isinf(top):
        return int(np.flatnonzero(x == top)[0])
    tol = 1e-9 * (1 + np.abs(x).max())
    return int(np.flatnonzero(x >= top - tol)[0])


def _m04_stack(r_mean, T, seed, scale=1.0):
    rng = np.random.default_rng([seed, 1])
    r_mean = np.asarray(r_mean, dtype=float)
    return scale * (rng.random((T, len(r_mean))) < r_mean)


def _m04_run(policy, stack, r_mean, seed):
    """The lab's run_bandit, also returning the counts and sums before every round."""
    rng = np.random.default_rng(seed)
    T, n_arms = stack.shape
    counts = np.zeros(n_arms, dtype=np.int64)
    sums = np.zeros(n_arms)
    arms = np.zeros(T, dtype=np.int64)
    before_c = np.zeros((T, n_arms), dtype=np.int64)
    before_s = np.zeros((T, n_arms))
    rewards = np.zeros(T)
    for t in range(T):
        before_c[t], before_s[t] = counts, sums
        a = policy(counts, sums, t, rng)
        reward = stack[counts[a], a]
        counts[a] += 1
        sums[a] += reward
        arms[t], rewards[t] = a, reward
    gaps = np.max(r_mean) - np.asarray(r_mean)
    return {
        "arms": arms,
        "regret": np.cumsum(gaps[arms]),
        "counts_before": before_c,
        "sums_before": before_s,
        "rewards": rewards,
        "final_counts": counts.copy(),
        "final_sums": sums.copy(),
    }


def _m04_eps_greedy(counts, sums, t, rng, epsilon=0.1):
    if np.any(counts == 0):
        return int(np.argmin(counts))
    if rng.random() < epsilon:
        return int(rng.integers(len(counts)))
    return _m04_argmax_tie(sums / counts)


def _m04_ucb1_index(counts, sums, t, *, log=np.log, plus=0, sqrt=np.sqrt, unpulled=np.inf):
    """UCB1's index; the keywords give the usual mistakes' values for the check's hints."""
    index = np.full(len(counts), unpulled)
    seen = counts > 0
    if seen.any():
        index[seen] = sums[seen] / counts[seen] + sqrt(2 * log(t + plus) / counts[seen])
    return index


def _m04_ucb1(counts, sums, t, rng, B=1.0):
    index = np.full(len(counts), np.inf)
    seen = counts > 0
    if seen.any():
        index[seen] = sums[seen] / counts[seen] + B * np.sqrt(2 * np.log(t) / counts[seen])
    return _m04_argmax_tie(index)


def m04_bandit() -> bytes:
    """Checkpoint 1.1: a spy policy that pulls rng.integers(n_arms) from the run's generator.

    The arms are what a loop that makes default_rng(seed) once shows the spy; the regret adds
    the gap of each arm pulled. Case 1 has two equal best arms, so every gap is 0.
    """
    cases = [(M04_R, 30, 3), (np.array([0.6, 0.6]), 12, 4), (np.array([0.2, 0.5, 0.1, 0.4]), 25, 5)]
    out: dict[str, np.ndarray] = {"n_cases": np.array(len(cases), dtype=np.int64)}
    for i, (r_mean, T, seed) in enumerate(cases):
        n = len(r_mean)
        run = _m04_run(
            lambda c, s, t, rng, n=n: int(rng.integers(n)),
            _m04_stack(r_mean, T, seed),
            r_mean,
            seed,
        )
        rng = np.random.default_rng(seed)
        assert run["arms"].tolist() == [int(rng.integers(n)) for _ in range(T)]
        assert len(set(run["arms"].tolist())) == n  # every arm is pulled at least once
        out.update(
            {
                f"r_mean_{i}": r_mean,
                f"seed_{i}": np.array(seed, dtype=np.int64),
                f"stack_{i}": _m04_stack(r_mean, T, seed),
                f"arms_{i}": run["arms"],
                f"regret_{i}": run["regret"],
            }
        )
    assert not out["regret_1"].any()
    return _save("m04_bandit", **out)


def m04_policy() -> bytes:
    """Checkpoint 1.2: epsilon-greedy's choices on fixed states, and its exact shares."""
    choice_counts = np.array([[1, 0, 1], [0, 0, 0], [20, 4, 10], [10, 10, 10]], dtype=np.int64)
    choice_sums = np.array([[1.0, 0, 0], [0, 0, 0], [8, 3, 4], [5, 5, 3]])
    want = np.array([1, 0, 1, 0], dtype=np.int64)  # an unpulled arm; the best mean; the tie
    for c, s, w in zip(choice_counts, choice_sums, want, strict=True):
        assert _m04_eps_greedy(c, s, 0, np.random.default_rng(0), epsilon=0.0) == w
    share_counts = np.array([10, 10, 10], dtype=np.int64)
    share_sums = np.array([5.0, 3.0, 7.0])
    eps = np.array([1.0, 0.2])
    n_arms = 3
    shares = np.array([[e / n_arms + (1 - e) * (a == 2) for a in range(n_arms)] for e in eps])
    ref = []
    for e in eps:
        rng = np.random.default_rng(0)
        picks = [_m04_eps_greedy(share_counts, share_sums, 30, rng, epsilon=e) for _ in range(4000)]
        ref.append(np.bincount(picks, minlength=n_arms) / 4000)
    ref = np.array(ref)
    assert np.all(np.abs(ref - shares) < 0.02)
    return _save(
        "m04_policy",
        choice_counts=choice_counts,
        choice_sums=choice_sums,
        choice_want=want,
        share_counts=share_counts,
        share_sums=share_sums,
        share_eps=eps,
        share_exact=shares,
        share_reference=ref,
        share_n=np.array(4000, dtype=np.int64),
    )


def m04_greedy_replay() -> bytes:
    """Checkpoint 1.2: greedy (epsilon = 0) on the lab's headlines, replayed round by round.

    Seed 1's first rewards are (1, 0, 0): headline 2's first showing pays 0 and greedy never
    shows it again (regret 0.3 + 0.1 x 197 = 20.0). Seed 17's are (0, 1, 0): greedy locks on
    headline 1 (0.3 + 0.2 x 197 = 39.7).
    """
    out: dict[str, np.ndarray] = {}
    for seed, first, total in ((1, [1.0, 0.0, 0.0], 20.0), (17, [0.0, 1.0, 0.0], 39.7)):
        stack = _m04_stack(M04_R, 200, seed)
        assert stack[0].tolist() == first
        run = _m04_run(lambda c, s, t, rng: _m04_eps_greedy(c, s, t, rng, 0.0), stack, M04_R, seed)
        assert run["arms"][:3].tolist() == [0, 1, 2] and not np.any(run["arms"][3:] == 2)
        assert abs(run["regret"][-1] - total) < 1e-9
        for key in ("arms", "counts_before", "sums_before"):
            out[f"{key}_{seed}"] = run[key]
        out[f"regret_{seed}"] = run["regret"]
    out["seeds"] = np.array([1, 17], dtype=np.int64)
    return _save("m04_greedy_replay", **out)


def m04_ucb1() -> bytes:
    """Checkpoint 2.1: UCB1's index on fixed counts and sums, and the usual mistakes' values."""
    cases = [
        ([0, 0, 0], [0.0, 0, 0], 0),
        ([1, 0, 0], [1.0, 0, 0], 1),
        ([1, 1, 1], [0.0, 1, 1], 3),
        ([90, 9, 1], [54.0, 4, 0], 100),  # the briefing's Check yourself 3.1
        ([50, 500, 4], [20.0, 260, 3], 554),
    ]
    counts = np.array([c for c, _, _ in cases], dtype=np.int64)
    sums = np.array([s for _, s, _ in cases])
    t = np.array([x for _, _, x in cases], dtype=np.int64)
    out = {"counts": counts, "sums": sums, "t": t}
    with np.errstate(divide="ignore"):
        variants = {
            "index": {},
            "log10": {"log": np.log10},
            "no_sqrt": {"sqrt": lambda x: x},
            "log_t_plus_one": {"plus": 1},
            "unpulled_zero": {"unpulled": 0.0},
        }
        for name, kw in variants.items():
            out[name] = np.array(
                [_m04_ucb1_index(c, s, x, **kw) for c, s, x in zip(counts, sums, t, strict=True)]
            )
    seen = counts > 0
    sums_not_means = np.where(seen, 0.0, np.inf)
    for i in range(len(cases)):
        if seen[i].any():
            sums_not_means[i, seen[i]] = sums[i, seen[i]] + np.sqrt(
                2 * np.log(t[i]) / counts[i, seen[i]]
            )
    out["sums_not_means"] = sums_not_means
    assert np.allclose(out["index"][3], [0.91990, 1.45606, 3.03485], atol=5e-6)
    assert np.allclose(out["index"][2], [1.4823, 2.4823, 2.4823], atol=5e-5)
    return _save("m04_ucb1", **out)


def m04_ucb1_replay() -> bytes:
    """Checkpoints 2.2 and 6: UCB1 on reward_stack(R, 1000, seed=2), round by round.

    UCB1 uses no randomness, so the run is fixed by the stack. The smallest nonzero gap between
    the largest index and another finite one is asserted far above argmax_tie's tolerance, so
    the arms are the same on every platform.
    """
    stack = _m04_stack(M04_R, 1000, 2)
    run = _m04_run(_m04_ucb1, stack, M04_R, 2)
    assert run["final_counts"].tolist() == [126, 111, 763]
    assert round(float(run["regret"][-1]), 4) == 34.8
    smallest = np.inf
    for c, s, t in zip(run["counts_before"], run["sums_before"], range(1000), strict=True):
        x = _m04_ucb1_index(c, s, t)
        if np.isinf(x.max()):
            continue
        d = np.sort((x.max() - x) / (1 + np.abs(x).max()))[1:]
        if d[d > 0].size:
            smallest = min(smallest, float(d[d > 0].min()))
    assert smallest > 1e-6, smallest  # far from the tie tolerance 1e-9
    return _save(
        "m04_ucb1_replay",
        seed=np.array(2, dtype=np.int64),
        **{k: run[k] for k in ("arms", "counts_before", "sums_before", "rewards")},
        final_counts=run["final_counts"],
        final_sums=run["final_sums"],
    )


def m04_beta() -> bytes:
    """Checkpoint 3.1: Beta(1 + S, 1 + F) on fixed counts, and the usual mistakes' values."""
    counts = np.array([[0, 0, 0], [10, 4, 1], [50, 10, 4]], dtype=np.int64)
    sums = np.array([[0.0, 0, 0], [3, 4, 0], [20, 3, 3]])
    shape_s, shape_f = 1 + sums, 1 + (counts - sums)
    assert shape_s[1].tolist() == [4, 5, 1] and shape_f[1].tolist() == [8, 1, 2]
    assert shape_s[2].tolist() == [21, 4, 4] and shape_f[2].tolist() == [31, 8, 2]
    return _save(
        "m04_beta",
        counts=counts,
        sums=sums,
        shape_s=shape_s,
        shape_f=shape_f,
        no_prior_s=sums,
        no_prior_f=counts - sums,
        counts_f=1.0 + counts,
    )


def _poly_mul(p, q):
    out = [Fraction(0)] * (len(p) + len(q) - 1)
    for i, a in enumerate(p):
        if a:
            for j, b in enumerate(q):
                out[i + j] += a * b
    return out


def _poly_integral(p):
    return [Fraction(0)] + [c / (k + 1) for k, c in enumerate(p)]


def m04_prob_best(posteriors):
    """P(arm i has the largest rate) for independent Beta(u, v) posteriors with integer u, v.

    Each density x^(u-1) (1-x)^(v-1) / B(u, v) is a polynomial, so the integral over [0, 1] of
    pdf_i(x) times the product of the other cdfs is computed exactly, in rationals, and is the
    same on every platform. (Also used by scripts/make_figures_m04.py.)
    """
    pdfs = []
    for u, v in posteriors:
        u, v = int(u), int(v)
        norm = (u + v - 1) * comb(u + v - 2, u - 1)  # 1 / B(u, v)
        coef = [Fraction(0)] * (u + v - 1)
        for k in range(v):
            coef[u - 1 + k] += Fraction(norm * comb(v - 1, k) * (-1) ** k)
        pdfs.append(coef)
    cdfs = [_poly_integral(p) for p in pdfs]
    out = []
    for i, f in enumerate(pdfs):
        for j, cdf in enumerate(cdfs):
            if j != i:
                f = _poly_mul(f, cdf)
        out.append(sum(_poly_integral(f)))
    assert sum(out) == 1
    return [float(x) for x in out]


def m04_ts_shares() -> bytes:
    """Checkpoint 3.2: the exact probability that each arm is best on fixed posteriors.

    Also the reference thompson_step's shares over 4000 calls from default_rng(0) at t = 64,
    for the record (the check compares with the exact values).
    """
    counts = np.array([[50, 10, 4], [20, 20, 2], [0, 0, 0]], dtype=np.int64)
    sums = np.array([[20.0, 3, 3], [10, 8, 1], [0, 0, 0]])
    exact = np.array(
        [
            m04_prob_best(list(zip(1 + s, 1 + c - s, strict=True)))
            for c, s in zip(counts, sums, strict=True)
        ]
    )
    assert np.allclose(exact[0], [0.0771, 0.0543, 0.8686], atol=5e-5)
    assert np.allclose(exact[1], [0.4011, 0.1338, 0.4651], atol=5e-5)
    ref = []
    for c, s in zip(counts, sums, strict=True):
        rng = np.random.default_rng(0)
        picks = [_m04_argmax_tie(rng.beta(1 + s, 1 + (c - s))) for _ in range(4000)]
        ref.append(np.bincount(picks, minlength=3) / 4000)
    ref = np.array(ref)
    assert np.all(np.abs(ref - exact) < 0.02)
    # The mistakes' shares: the posterior mean (greedy) and a 0/1 draw per arm.
    means = (1 + sums) / (2 + counts)
    greedy = np.zeros_like(exact)
    for i, m in enumerate(means):
        greedy[i, _m04_argmax_tie(m)] = 1.0
    p = means
    sample_reward = np.stack(
        [
            p[:, 0] + (1 - p[:, 0]) * (1 - p[:, 1]) * (1 - p[:, 2]),
            (1 - p[:, 0]) * p[:, 1],
            (1 - p[:, 0]) * (1 - p[:, 1]) * p[:, 2],
        ],
        axis=1,
    )
    assert np.allclose(sample_reward[0], [0.536, 0.199, 0.265], atol=5e-4)
    return _save(
        "m04_ts_shares",
        counts=counts,
        sums=sums,
        t=np.array(64, dtype=np.int64),
        n=np.array(4000, dtype=np.int64),
        exact=exact,
        reference=ref,
        posterior_mean=greedy,
        sample_reward=sample_reward,
    )


# Checkpoint 4's fixed cases (academic director's handoff, §2a): (name, a, b, expected winner).
M04_VERDICT_CASES = [
    (
        "single",
        [5.0],
        [
            7.2,
            12.5,
            8.3,
            9.5,
            9.8,
            8.5,
            7.3,
            11.3,
            10.7,
            6.1,
            14.7,
            11.9,
            8.5,
            11.8,
            9.1,
            9.9,
            11.6,
            7.5,
            11.2,
            12.8,
        ],
        "single seed",
    ),
    ("ranges-3-disjoint", [1.0, 2.0, 3.0], [5.0, 6.0, 7.0], "A"),
    ("ranges-3-overlap", [1.0, 1.2, 5.0], [4.8, 6.0, 6.2], "tie"),
    (
        "clear-20",
        [
            24.0,
            19.1,
            22.7,
            15.1,
            19.5,
            21.3,
            16.0,
            19.8,
            25.2,
            27.9,
            22.3,
            22.5,
            17.1,
            16.4,
            15.8,
            21.6,
            22.3,
            18.0,
            16.3,
            20.8,
        ],
        [
            61.6,
            59.3,
            66.3,
            59.5,
            59.7,
            54.5,
            60.7,
            66.7,
            60.3,
            60.4,
            62.2,
            61.4,
            62.7,
            62.7,
            63.1,
            56.0,
            61.5,
            52.0,
            61.3,
            53.7,
        ],
        "A",
    ),
    (
        "overlap-20",
        [
            49.3,
            54.7,
            45.9,
            51.0,
            33.6,
            41.4,
            56.9,
            38.5,
            56.5,
            36.1,
            40.9,
            39.0,
            50.1,
            55.3,
            39.3,
            48.2,
            66.2,
            46.8,
            41.8,
            53.9,
        ],
        [
            50.8,
            46.0,
            35.0,
            61.2,
            47.3,
            53.0,
            42.4,
            66.0,
            60.5,
            62.8,
            51.9,
            57.7,
            61.9,
            63.2,
            56.1,
            52.4,
            49.4,
            45.5,
            43.3,
            56.6,
        ],
        "tie",
    ),
    (
        "outlier-range-overlap",
        [
            60.0,
            24.0,
            29.5,
            33.2,
            29.2,
            24.4,
            29.6,
            32.4,
            30.6,
            28.7,
            35.5,
            35.7,
            29.7,
            32.4,
            31.2,
            32.3,
            34.4,
            32.5,
            30.3,
            28.0,
        ],
        [
            42.6,
            44.9,
            48.5,
            45.9,
            45.2,
            45.8,
            45.1,
            46.6,
            41.6,
            39.1,
            43.7,
            41.6,
            49.8,
            44.5,
            44.2,
            40.4,
            45.8,
            43.1,
            48.4,
            47.5,
        ],
        "A",
    ),
    (
        "outliers-mean-vs-iqm",
        [
            200.0,
            180.0,
            160.0,
            28.8,
            30.1,
            25.3,
            29.6,
            27.5,
            28.5,
            28.9,
            29.4,
            30.8,
            32.1,
            29.7,
            32.7,
            28.7,
            30.7,
            31.8,
            30.2,
            28.5,
        ],
        [
            46.9,
            48.2,
            45.8,
            47.4,
            47.5,
            48.9,
            48.2,
            48.5,
            46.5,
            47.5,
            49.3,
            50.8,
            45.3,
            50.8,
            50.5,
            49.3,
            48.3,
            47.1,
            50.7,
            51.7,
        ],
        "A",
    ),
]


def m04_verdict() -> bytes:
    """Checkpoint 4: compare()'s verdict on fixed per-seed regrets (lower is better).

    Each decided case's intervals are at least 1 apart and each tie's overlap at least 1, so
    no rounding in the bootstrap can change a verdict.
    """
    sys.path.insert(0, str(ROOT / "prl" / "src"))
    from prl.evaluate import compare

    width = max(max(len(a), len(b)) for _, a, b, _ in M04_VERDICT_CASES)
    n = len(M04_VERDICT_CASES)
    a_arr, b_arr = np.full((n, width), np.nan), np.full((n, width), np.nan)
    n_a, n_b = np.zeros(n, dtype=np.int64), np.zeros(n, dtype=np.int64)
    summary = np.zeros((n, 2, 3))
    for i, (name, a, b, want) in enumerate(M04_VERDICT_CASES):
        v = compare(a, b, higher_is_better=False)
        assert v.winner == want, (name, v.winner)
        a_arr[i, : len(a)], b_arr[i, : len(b)] = a, b
        n_a[i], n_b[i] = len(a), len(b)
        summary[i] = v.summary_a, v.summary_b
        if want in ("A", "B"):
            sep = max(v.summary_b[1] - v.summary_a[2], v.summary_a[1] - v.summary_b[2])
            assert sep >= 1.0, (name, sep)
        elif want == "tie":
            ov = min(v.summary_a[2], v.summary_b[2]) - max(v.summary_a[1], v.summary_b[1])
            assert ov >= 0.2, (name, ov)
    return _save(
        "m04_verdict",
        names=np.array([c[0] for c in M04_VERDICT_CASES]),
        a=a_arr,
        b=b_arr,
        n_a=n_a,
        n_b=n_b,
        winner=np.array([c[3] for c in M04_VERDICT_CASES]),
        summary=summary,
    )


def _m04_ucb1_for_range(counts, sums, t, B):
    """The lab's ucb1_for_range: rescale the rewards to [0, 1], take UCB1's index, scale back."""
    return B * _m04_ucb1_index(counts, sums / B, t)


def m04_ucb1_for_range() -> bytes:
    """Checkpoint 5.1: UCB1's index for rewards in [0, B], and the usual mistakes' values.

    The B = 1 case equals ucb1_index; the others separate `bonus-not-scaled`
    (ucb1_index(counts, sums, t)) and `mean-scaled-twice` (B * ucb1_index(counts, sums, t)).
    """
    cases = [
        ([0, 0, 0], [0.0, 0, 0], 0, 10.0),
        ([1, 0, 1], [10.0, 0, 0], 2, 10.0),
        ([90, 9, 1], [540.0, 40, 0], 100, 10.0),  # ten times Check yourself 3.1
        ([90, 9, 1], [54.0, 4, 0], 100, 1.0),  # B = 1: ucb1_index itself
        ([4, 10, 2], [5.0, 10, 0], 16, 2.5),
    ]
    counts = np.array([c for c, _, _, _ in cases], dtype=np.int64)
    sums = np.array([x for _, x, _, _ in cases])
    t = np.array([c[2] for c in cases], dtype=np.int64)
    B = np.array([c[3] for c in cases])
    rows = list(zip(counts, sums, t, B, strict=True))
    with np.errstate(divide="ignore"):
        index = np.array([_m04_ucb1_for_range(c, x, int(k), b) for c, x, k, b in rows])
        not_scaled = np.array([_m04_ucb1_index(c, x, int(k)) for c, x, k, b in rows])
        twice = np.array([b * _m04_ucb1_index(c, x, int(k)) for c, x, k, b in rows])
    assert np.all(np.isposinf(index[0]))
    assert np.allclose(index[1][[0, 2]], [21.7741, 11.7741], atol=5e-5) and np.isinf(index[1][1])
    assert np.allclose(index[2], [9.19902, 14.56063, 30.34854], atol=5e-6)
    assert np.allclose(index[3], _m04_ucb1_index(counts[3], sums[3], 100), rtol=0, atol=1e-12)
    assert np.allclose(index[4], [4.19353, 2.86165, 4.16277], atol=5e-6)
    for wrong in (not_scaled, twice):  # each case but B = 1 (and t = 0) separates both
        for i in (1, 2, 4):
            fin = np.isfinite(index[i])
            assert np.max(np.abs(wrong[i][fin] - index[i][fin])) > 0.5
    return _save(
        "m04_ucb1_for_range",
        counts=counts,
        sums=sums,
        t=t,
        B=B,
        index=index,
        bonus_not_scaled=not_scaled,
        mean_scaled_twice=twice,
    )


def m04_scale_runs(seeds=M04_SCALE_SEEDS, T=M04_T_SCALE) -> dict[str, np.ndarray]:
    """Checkpoint 5.2's reference run: final pseudo-regret in the original units, and the arms,
    per seed. UCB1 with the bonus x 10 is the lab's ucb1_for_range at B = 10."""

    def for_range(c, s, t, rng, B=10.0):
        return _m04_argmax_tie(_m04_ucb1_for_range(c, s, t, B))

    configs = {
        "eps_greedy_x1": (lambda c, s, t, rng: _m04_eps_greedy(c, s, t, rng, 0.1), 1.0),
        "eps_greedy_x10": (lambda c, s, t, rng: _m04_eps_greedy(c, s, t, rng, 0.1), 10.0),
        "ucb1_x1": (_m04_ucb1, 1.0),
        "ucb1_x10": (_m04_ucb1, 10.0),
        "ucb1_rescaled_x10": (for_range, 10.0),
    }
    out = {}
    for name, (policy, scale) in configs.items():
        regrets, arms = [], []
        for k in seeds:
            run = _m04_run(policy, _m04_stack(M04_R, T, k, scale), scale * M04_R, k)
            regrets.append(run["regret"][-1] / scale)
            arms.append(run["arms"])
        out[name] = np.array(regrets)
        out[f"{name}_arms"] = np.array(arms)
    return out


def m04_scale() -> bytes:
    """Checkpoint 5.2: "same" or "changes" per method in the reference run (seeds 0-19,
    T = 5000), from the arms: "same" means identical arms on every seed."""
    runs = m04_scale_runs()
    directions, same_seeds = {}, {}
    for key, base in (
        ("eps_greedy", "eps_greedy_x1"),
        ("ucb1", "ucb1_x1"),
        ("ucb1_rescaled", "ucb1_x1"),
    ):
        same = [
            np.array_equal(x, y)
            for x, y in zip(runs[f"{base}_arms"], runs[f"{key}_x10_arms"], strict=True)
        ]
        same_seeds[key] = sum(same)
        directions[key] = "same" if all(same) else "changes"
        if all(same):
            assert np.allclose(runs[base], runs[f"{key}_x10"], rtol=0, atol=1e-9)
    assert directions == {"eps_greedy": "same", "ucb1": "changes", "ucb1_rescaled": "same"}
    assert same_seeds["ucb1"] == 0  # UCB1 x 10 changes its arms on every seed
    return _save(
        "m04_scale",
        keys=np.array(list(directions)),
        directions=np.array(list(directions.values())),
        same_arms_seeds=np.array(list(same_seeds.values()), dtype=np.int64),
        seeds=np.array(M04_SCALE_SEEDS, dtype=np.int64),
        T=np.array(M04_T_SCALE, dtype=np.int64),
        **{k: v for k, v in runs.items() if not k.endswith("_arms")},
    )


def m04_gauss() -> bytes:
    """Stretch: the Gaussian posterior of each arm's mean, and the usual mistakes' values."""
    cases = [
        ([0, 0, 0], [0.0, 0, 0], 0.5, 1.0),
        ([4, 10, 1], [2.0, 3, 1], 0.5, 1.0),
        ([4, 10, 1], [20.0, 30, 10], 5.0, 10.0),  # 10 x the row above: the scale
    ]
    out: dict[str, np.ndarray] = {
        "counts": np.array([c for c, _, _, _ in cases], dtype=np.int64),
        "sums": np.array([s for _, s, _, _ in cases]),
        "noise_sd": np.array([c[2] for c in cases]),
        "prior_sd": np.array([c[3] for c in cases]),
    }
    mean, sd = [], []
    for c, s, noise, prior in cases:
        precision = 1 / prior**2 + np.asarray(c) / noise**2
        mean.append((np.asarray(s) / noise**2) / precision)
        sd.append(np.sqrt(1 / precision))
    out["mean"], out["sd"] = np.array(mean), np.array(sd)
    assert np.allclose(out["mean"][1], [8 / 17, 12 / 41, 0.8]) and np.allclose(
        out["mean"][2], 10 * out["mean"][1]
    )
    assert np.allclose(out["sd"][1] ** 2, [1 / 17, 1 / 41, 1 / 5]) and np.allclose(
        out["sd"][2], 10 * out["sd"][1]
    )
    out["sample_mean"] = out["sums"] / np.maximum(out["counts"], 1)
    return _save("m04_gauss", **out)


# ---------------------------------------------------------------- Module 5: contextual bandits
# The lab's constants (labs/src/05-contextual-bandits.py): K items, contexts of dimension 5,
# the catalog seed. Checkpoint seeds are 51-57, clear of the seats' (SEED + 1) * 1000 + i.
M05_K, M05_D_RAW, M05_CATALOG = 10, 5, 0


def _m05_features(x):
    return np.append(x, 1.0)


def _m05_ridge_closed_form(X, r, lam):
    """Ridge by the closed form (fixture code; never shipped in prl)."""
    A = lam * np.eye(X.shape[1]) + X.T @ X
    b = X.T @ r
    return A, b, np.linalg.inv(A), np.linalg.solve(A, b)


def _m05_eps_greedy(scores, eps):
    """The workshop's epsilon-greedy as probabilities: eps / K everywhere, 1 - eps on the best."""
    probs = np.full(len(scores), eps / len(scores))
    probs[int(np.argmax(scores))] += 1.0 - eps
    return probs


def m05_reward() -> bytes:
    """Exercise 1: r(x, a) = clip(0.5 + scale theta_a^T x, 0.02, 0.98), by hand and on RecSim."""
    sys.path.insert(0, str(ROOT / "prl" / "src"))
    from prl.envs import RecSim

    hand = [
        ([[1, 0], [0, 2], [-3, 3]], [0.5, 0.5], 0.15),  # interior: (0.575, 0.65, 0.5)
        ([[2, 0], [0, -4], [10, 10]], [0.5, 0.5], 0.15),  # top clip: 2.0 -> 0.98
        ([[1, 0], [0, 1], [-3, 0]], [2.0, 1.0], 0.3),  # both clips: (0.98, 0.8, 0.02)
    ]
    theta = np.array([h[0] for h in hand], dtype=float)
    x = np.array([h[1] for h in hand], dtype=float)
    scale = np.array([h[2] for h in hand], dtype=float)
    lin = np.einsum("ckd,cd->ck", theta, x)
    want = np.clip(0.5 + scale[:, None] * lin, 0.02, 0.98)
    by_hand = [[0.575, 0.65, 0.5], [0.65, 0.2, 0.98], [0.98, 0.8, 0.02]]
    assert np.allclose(want, by_hand, atol=1e-15), want
    env = RecSim(M05_K, M05_D_RAW, seed=M05_CATALOG)
    recsim_x = env.sample_context(np.random.default_rng(55))
    return _save(
        "m05_reward",
        hand_theta=theta,
        hand_x=x,
        hand_scale=scale,
        hand_expected=want,
        # The usual mistakes, for the checkpoint's hints.
        hand_no_clip=0.5 + scale[:, None] * lin,
        hand_no_intercept=np.clip(scale[:, None] * lin, 0.02, 0.98),
        hand_no_scale=np.clip(0.5 + lin, 0.02, 0.98),
        recsim_x=recsim_x,
        recsim_expected=env.expected_reward(recsim_x),
    )


def m05_ridge() -> bytes:
    """Exercise 2: seven updates of a 3-feature Ridge, for two penalties, against the closed form."""
    rng = np.random.default_rng(54)
    X = np.round(rng.normal(0.0, 1.5, size=(7, 3)), 2)  # rows far from unit length
    r = np.array([1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0])
    lams = np.array([1.0, 0.5])
    probes = np.round(rng.normal(size=(3, 3)), 2)
    A, A_inv, theta, pred = [], [], [], []
    for lam in lams:
        a, b, a_inv, th = _m05_ridge_closed_form(X, r, float(lam))
        A.append(a)
        A_inv.append(a_inv)
        theta.append(th)
        pred.append(probes @ th)
    assert np.all(np.linalg.norm(X, axis=1) > 1.2), "non-unit rows"
    return _save(
        "m05_ridge",
        X=X,
        r=r,
        lams=lams,
        probes=probes,
        A=np.array(A),
        b=X.T @ r,
        A_inv=np.array(A_inv),
        pred=np.array(pred),
        # The usual mistakes, for the hints: b = sum x, and predict with A in place of A^-1.
        b_without_reward=X.sum(axis=0),
        pred_with_A=np.array([probes @ (a @ (X.T @ r)) for a in A]),
    )


def m05_linucb() -> bytes:
    """Exercise 3: LinUCB scores on fixed models, and the width of a unit x seen n times."""
    rng = np.random.default_rng(56)
    A, A_inv, b, n = [], [], [], []
    for k in range(3):
        M = rng.normal(size=(3, 4 + 2 * k))
        a = np.eye(3) + M @ M.T  # a Gram matrix: lam I + sum x x^T, not diagonal
        A.append(a)
        A_inv.append(np.linalg.inv(a))
        b.append(rng.normal(size=3))
        n.append(M.shape[1])
    A, A_inv, b = np.array(A), np.array(A_inv), np.array(b)
    x = np.array([0.6, -0.3, 1.1])
    cs = np.array([0.5, 2.0])
    means = np.einsum("kij,kj->ki", A_inv, b) @ x
    quad = np.einsum("i,kij,j->k", x, A_inv, x)
    quad_A = np.einsum("i,kij,j->k", x, A, x)
    scores = np.array([means + c * np.sqrt(quad) for c in cs])
    wrong = {
        "variance": np.array([means + c * quad for c in cs]),
        "A_not_inverse": np.array([means + c * np.sqrt(quad_A) for c in cs]),
        "width_only": np.array([c * np.sqrt(quad) for c in cs]),
        "c_inside_sqrt": np.array([means + np.sqrt(c * quad) for c in cs]),
    }
    for name, w in wrong.items():  # every mistake is visibly wrong at one c at least
        assert np.abs(w - scores).max() > 1e-3, name
    u = np.array([1.0, 2.0, 2.0]) / 3.0
    ns = np.array([0, 1, 10, 100])
    unit_A = np.array([np.eye(3) + k * np.outer(u, u) for k in ns])
    return _save(
        "m05_linucb",
        A=A,
        A_inv=A_inv,
        b=b,
        n=np.array(n, dtype=np.int64),
        x=x,
        cs=cs,
        scores=scores,
        **{f"wrong_{k}": v for k, v in wrong.items()},
        u=u,
        unit_ns=ns,
        unit_A=unit_A,
        unit_A_inv=np.array([np.linalg.inv(a) for a in unit_A]),
        unit_scores=np.array([c / np.sqrt(1.0 + ns) for c in cs]),
    )


def m05_lints() -> bytes:
    """Exercise 4: a fixed model for the sampler's moments, and the reference's margins.

    A's correlations are strong enough that every wrong covariance is far from v^2 A^-1 once
    whitened: the upper Cholesky factor (L^T L) by 1.44, v^2 I by 2.0, v A^-1 by 1.0 and
    v^2 A by 11, against the checkpoint's 0.2 (a correct sampler's entries have sd 0.022).
    """
    A = np.array([[3.0, 1.5, 0.0], [1.5, 2.0, 0.9], [0.0, 0.9, 1.5]])
    A_inv = np.linalg.inv(A)
    b = np.array([1.0, -0.5, 0.25])
    v, n_draws = 0.5, 4000
    mean = A_inv @ b
    # The reference sampler (the lab's solution) on the checkpoint's draw, to show its margin:
    # the checkpoint allows 6 standard errors for each mean and 0.2 for each entry of the
    # whitened covariance.
    rng = np.random.default_rng(11)
    L = np.linalg.cholesky(A_inv)
    draws = np.array([mean + v * L @ rng.standard_normal(3) for _ in range(n_draws)])
    se = v * np.sqrt(np.diag(A_inv) / n_draws)
    z = (draws.mean(axis=0) - mean) / se
    Li = np.linalg.inv(L)
    white = Li @ (np.cov(draws.T) / v**2) @ Li.T
    dev = np.abs(white - np.eye(3)).max()
    assert np.abs(z).max() < 3 and dev < 0.1, (z, dev)
    wrong = {"upper": L.T @ L, "isotropic": np.eye(3), "sqrt-v": A_inv / v, "A": A}
    for name, cov in wrong.items():
        assert np.abs(Li @ cov @ Li.T - np.eye(3)).max() > 0.95, name
    return _save(
        "m05_lints",
        A=A,
        A_inv=A_inv,
        b=b,
        v=np.array(v),
        n_draws=np.array(n_draws, dtype=np.int64),
        estimate=mean,
        ref_max_z=np.array(np.abs(z).max()),
        ref_max_dev=np.array(dev),
    )


def m05_verdicts() -> bytes:
    """Exercise 5: four tables of final regrets and prl.evaluate.compare's verdicts on them."""
    sys.path.insert(0, str(ROOT / "prl" / "src"))
    from prl.evaluate import Budget, compare

    tables = {
        "clear": ([200, 210, 190, 205, 220], [100, 105, 110, 95, 102], [150, 260, 180, 230, 200]),
        "reverse": (
            [150, 160, 155, 145, 170],
            [300, 310, 305, 290, 320],
            [152, 158, 150, 149, 166],
        ),
        "three-seeds": ([200, 220, 210], [100, 120, 110], [190, 230, 205]),
        "one-seed": ([200], [100], [150]),
    }
    by_hand = {
        "clear": ("LinUCB", "tie"),
        "reverse": ("greedy", "tie"),
        "three-seeds": ("LinUCB", "tie"),
        "one-seed": ("single seed", "single seed"),
    }
    methods = ("greedy", "LinUCB", "LinTS")
    budget = Budget(steps=3000, seeds=5)
    regrets = np.full((len(tables), 3, 5), np.nan)
    n_seeds = np.zeros(len(tables), dtype=np.int64)
    words = []
    for i, (regime, rows) in enumerate(tables.items()):
        n_seeds[i] = len(rows[0])
        for j, row in enumerate(rows):
            regrets[i, j, : len(row)] = row
        got = []
        for k, method in ((1, "LinUCB"), (2, "LinTS")):
            v = compare(
                rows[k], rows[0], names=(method, "greedy"), budget=budget, higher_is_better=False
            )
            got.append({"A": method, "B": "greedy"}.get(v.winner, v.winner))
            if v.winner in ("A", "B"):  # a clear call: the intervals are well apart
                lo_a, hi_a = v.summary_a[1:]
                lo_b, hi_b = v.summary_b[1:]
                assert max(lo_a - hi_b, lo_b - hi_a) > 10, (regime, method)
        assert tuple(got) == by_hand[regime], (regime, got)
        words.append(got)
    return _save(
        "m05_verdicts",
        regimes=np.array(list(tables)),
        methods=np.array(methods),
        regrets=regrets,
        n_seeds=n_seeds,
        steps=np.array(budget.steps, dtype=np.int64),
        seeds=np.array(budget.seeds, dtype=np.int64),
        expected=np.array(words),
    )


def m05_logging() -> bytes:
    """Exercises 6 and 7: epsilon-greedy probabilities, and RecSim logs with known propensities."""
    sys.path.insert(0, str(ROOT / "prl" / "src"))
    from prl.envs import RecSim

    scores4, eps4 = np.array([0.1, 0.5, 0.3, 0.2]), 0.1
    scores10 = np.array([0.3, 0.1, 0.72, 0.2, 0.5, 0.65, 0.0, 0.4, 0.45, 0.6])
    eps10 = np.array([0.0, 0.1, 0.3, 1.0])
    expected4 = _m05_eps_greedy(scores4, eps4)
    assert np.allclose(expected4, [0.025, 0.925, 0.025, 0.025], atol=1e-15)
    # Fixed score weights for the logging checkpoints: scores(x) = W @ phi(x). Not a learner's.
    W = np.round(np.random.default_rng(51).normal(size=(M05_K, M05_D_RAW + 1)), 3)
    K = M05_K

    def scores(x):
        return W @ _m05_features(x)

    # Exercise 6's checkpoint logs with env.log and the reference log_policy at epsilon 0.1.
    log_eps, log_n, log_seed = 0.1, 4000, 52
    env = RecSim(K, M05_D_RAW, seed=M05_CATALOG)
    log = env.log(lambda x, rng: _m05_eps_greedy(scores(x), log_eps), log_n, seed=log_seed)
    w = 1.0 / (K * log.P)
    log_z = (w.mean() - 1.0) / (w.std(ddof=1) / np.sqrt(log_n))
    # Exercise 7's checkpoint: the fixed log_suspect (the lab's solution) with its own
    # epsilon-greedy policy at epsilon 0.2, on RecSim's stream `seed`, method coin seed + 1.
    sus_eps, sus_n, sus_seed = 0.2, 3000, 53
    env = RecSim(K, M05_D_RAW, seed=M05_CATALOG)
    env.reset(seed=sus_seed)
    rng = np.random.default_rng(sus_seed + 1)
    A, P, best = [], [], []
    for _ in range(sus_n):
        probs = _m05_eps_greedy(scores(env.context()), sus_eps)
        a = int(rng.choice(K, p=probs))
        env.step(a)
        A.append(a)
        P.append(probs[a])
        best.append(int(np.argmax(probs)))
    A, P, best = np.array(A), np.array(P), np.array(best)
    share = float(np.mean(A != best))
    p_other = sus_eps * (K - 1) / K
    share_z = (share - p_other) / np.sqrt(p_other * (1 - p_other) / sus_n)
    w = (1.0 / K) / P
    sus_z = (w.mean() - 1.0) / (w.std(ddof=1) / np.sqrt(sus_n))
    # The checkpoints allow 4 (weights) and 5 (share) standard errors; the reference is well inside.
    assert max(abs(log_z), abs(share_z), abs(sus_z)) < 3, (log_z, share_z, sus_z)
    return _save(
        "m05_logging",
        scores4=scores4,
        eps4=np.array(eps4),
        expected4=expected4,
        scores10=scores10,
        eps10=eps10,
        expected10=np.array([_m05_eps_greedy(scores10, e) for e in eps10]),
        score_weights=W,
        log_eps=np.array(log_eps),
        log_n=np.array(log_n, dtype=np.int64),
        log_seed=np.array(log_seed, dtype=np.int64),
        ref_log_z=np.array(log_z),
        suspect_eps=np.array(sus_eps),
        suspect_n=np.array(sus_n, dtype=np.int64),
        suspect_seed=np.array(sus_seed, dtype=np.int64),
        ref_suspect_share=np.array(share),
        ref_suspect_share_z=np.array(share_z),
        ref_suspect_z=np.array(sus_z),
    )


# ---------------------------------------------------------------- Module 6 (exploration in MDPs)
# Solver code for the fixtures: the policy chain, the hitting-time solve, simulated walks,
# value iteration with the MBIE-EB bonus and the counts estimator (Module 2's, above). It
# lives here, never in prl.
M06_GAMMA = 0.95
M06_SEED = 0  # every draw below; the lab's seats use (SEED + 1) * 1000 and up
M06_SIM_N, M06_SIM_WALKS = 5, 20000  # the run cell's chain (n = 5), ten times its 2000 walks


def _riverswim(n: int, gamma: float = M06_GAMMA):
    sys.path.insert(0, str(ROOT / "prl" / "src"))
    from prl.envs import RiverSwim

    return RiverSwim(n, gamma=gamma)


def _policy_chain(P, pi):
    return np.einsum("sa,sat->st", pi, P)


def _hitting(P_pi, target):
    n = len(P_pi)
    others = [s for s in range(n) if s != target]
    h = np.zeros(n)
    h[others] = np.linalg.solve(np.eye(n - 1) - P_pi[np.ix_(others, others)], np.ones(n - 1))
    return h


def _simulate_hitting(P_pi, target, walks, rng, start=0):
    """The lab's provided simulate_hitting, step for step."""
    cum = np.cumsum(P_pi, axis=1)
    s = np.full(walks, start)
    steps = np.zeros(walks, dtype=int)
    active = s != target
    while active.any():
        u = rng.random(int(active.sum()))
        nxt = (u[:, None] > cum[s[active]]).sum(axis=1)
        s[active] = np.minimum(nxt, len(P_pi) - 1)
        steps[active] += 1
        active = s != target
    return steps


def m06_chain() -> bytes:
    """Exercises 1 and 2: policy chains, hitting times, the growth ratio and simulated walks."""
    rng = np.random.default_rng(M06_SEED)
    out: dict[str, np.ndarray] = {}
    # ---- Exercise 1: P_pi = sum_a pi(a|s) P(s'|s, a)
    river6 = _riverswim(6)
    random_P = rng.dirichlet(np.ones(4), size=(4, 3))  # 4 states, 3 actions
    pt_cases = [
        ("RiverSwim(6), uniform random play", river6.P, np.full((6, 2), 0.5)),
        ("RiverSwim(6), a policy that is not uniform", river6.P, rng.dirichlet([1, 1], size=6)),
        ("a random MDP with 4 states and 3 actions", random_P, rng.dirichlet(np.ones(3), size=4)),
    ]
    for i, (label, P, pi) in enumerate(pt_cases):
        want = _policy_chain(P, pi)
        assert np.allclose(want.sum(axis=1), 1.0, atol=1e-12)
        # The case that rejects 'ignores pi' (P.mean(axis=1)) must differ from it clearly.
        if i > 0:
            assert np.abs(P.mean(axis=1) - want).max() > 0.05, label
        out.update({f"pt_P_{i}": P, f"pt_pi_{i}": pi, f"pt_want_{i}": want})
        out[f"pt_label_{i}"] = np.array(label)
    out["pt_n_cases"] = np.array(len(pt_cases), dtype=np.int64)
    # ---- Exercise 2: h = 1 + P_pi h off the target, h(target) = 0
    kc = np.zeros((4, 4))  # knowledge check 6.1's chain; the target, 3, absorbs
    kc[0, [0, 1]] = [0.75, 0.25]
    for s in (1, 2):
        kc[s, [s - 1, s, s + 1]] = [0.4, 0.4, 0.2]
    kc[3, 3] = 1.0
    hit_cases = [("knowledge check 6.1's 4-state chain", kc, 3)]
    for n in range(2, 13):
        P_pi = _policy_chain(_riverswim(n).P, np.full((n, 2), 0.5))
        hit_cases.append((f"RiverSwim({n}) under uniform random play", P_pi, n - 1))
    ergodic = rng.dirichlet(np.ones(5), size=5)  # every entry > 0
    hit_cases.append(("a random 5-state chain, target state 2 (not the last)", ergodic, 2))
    for i, (label, P_pi, target) in enumerate(hit_cases):
        h = _hitting(P_pi, target)
        out.update({f"hit_P_{i}": P_pi, f"hit_target_{i}": np.array(target, dtype=np.int64)})
        out[f"hit_h_{i}"], out[f"hit_label_{i}"] = h, np.array(label)
    out["hit_n_cases"] = np.array(len(hit_cases), dtype=np.int64)
    assert np.allclose(out["hit_h_0"], [48.0, 44.0, 31.0, 0.0], atol=1e-9)
    h0 = {n: float(out[f"hit_h_{n - 1}"][0]) for n in range(2, 13)}  # case i is RiverSwim(i + 1)
    assert [round(h0[n], 3) for n in (2, 3, 4, 6, 10, 12)] == [
        5.0,
        25.714,
        93.571,
        936.429,
        77296.429,
        695899.286,
    ], h0
    out["ratio_cases"] = np.array([10, 11], dtype=np.int64)  # RiverSwim(11) and RiverSwim(12)
    out["ratio_12_11"] = np.array(h0[12] / h0[11])
    assert abs(h0[12] / h0[11] - 3.00026) < 1e-5
    # ---- simulated walks on RiverSwim(5), as the run cell does (with more walks)
    i5 = M06_SIM_N - 1
    walks = _simulate_hitting(out[f"hit_P_{i5}"], M06_SIM_N - 1, M06_SIM_WALKS, rng)
    mean, se = float(walks.mean()), float(walks.std() / np.sqrt(len(walks)))
    assert abs(mean - h0[M06_SIM_N]) < 3 * se, (mean, se, h0[M06_SIM_N])  # this draw agrees
    out.update(sim_case=np.array(i5, dtype=np.int64), sim_walks=np.array(len(walks)))
    out.update(sim_mean=np.array(mean), sim_se=np.array(se), sim_max=np.array(walks.max()))
    return _save("m06_chain", **out)


def m06_q0() -> bytes:
    """Exercise 4: Q* of the lab's 10-state river (pi* by value iteration, Q* exactly)."""
    river = _riverswim(10)
    _, pi_star, Q_star, _ = _optimal(river.P, river.R, M06_GAMMA, tol=1e-12)
    assert np.all(pi_star == 1), "pi* swims right in every state"
    assert round(float(Q_star.max()), 2) == 16.68 and Q_star.argmax() == 19
    assert Q_star.max() < 1.0 / (1 - M06_GAMMA)
    return _save("m06_q0", Q_star=Q_star)


def _mbie(P_hat, R_hat, counts, gamma, c, *, bonus_untried=True):
    """MBIE-EB's Q~: value iteration on R_hat + c / sqrt(max(N, 1)), to 1e-12."""
    bonus = c / np.sqrt(np.maximum(counts, 1))
    if not bonus_untried:
        bonus = np.where(counts > 0, bonus, 0.0)
    return _value_iteration(P_hat, R_hat + bonus, gamma, 1e-12)[1]


def m06_mbie() -> bytes:
    """Exercise 5: the MBIE-EB backup on fixed models with pairs never tried."""
    one = np.ones((1, 2, 1))  # one state; both actions are self-loops
    cases = [
        (
            "Check yourself 4.1: left tried 4 times, right never",
            one,
            [[0.005, 0.0]],
            [[4, 0]],
            0.95,
            1.0,
        ),
        ("knowledge check 6.4: a tried 9 times, b never", one, [[0.2, 0.0]], [[9, 0]], 0.9, 0.6),
    ]
    river = _riverswim(10)
    rng = np.random.default_rng(M06_SEED)
    s, log = river.start, []
    for _ in range(300):  # uniform random play from the left end
        a = int(rng.integers(river.n_actions))
        s2, r, _ = river.sample_step(s, a, rng)
        log.append((s, a, r, s2))
        s = s2
    P_hat, R_hat, counts = _estimate_model(log, river.n_states, river.n_actions)
    assert (counts == 0).any() and counts[0].min() > 0, "untried pairs, away from the start"
    for c in (1.0, 0.4):
        cases.append(
            (
                f"300 steps of uniform play on RiverSwim(10), gamma 0.95, c {c:g}",
                P_hat,
                R_hat,
                counts,
                0.95,
                c,
            )
        )
    out: dict[str, np.ndarray] = {"n_cases": np.array(len(cases), dtype=np.int64)}
    for i, (label, P, R, N, gamma, c) in enumerate(cases):
        P, R, N = (np.asarray(x, dtype=float) for x in (P, R, N))
        want = _mbie(P, R, N, gamma, c)
        after = _value_iteration(P, R, gamma, 1e-12)[1] + c / np.sqrt(np.maximum(N, 1))
        no_bonus = _mbie(P, R, N, gamma, c, bonus_untried=False)
        # eq-untried: every state with an untried action is worth at least c / (1 - gamma).
        untried = (N == 0).any(axis=1)
        assert np.all(want.max(axis=1)[untried] >= c / (1 - gamma) - 1e-9)
        assert np.abs(after - want).max() > 1.0 and np.abs(no_bonus - want).max() > 0.5, label
        out.update({f"P_{i}": P, f"R_{i}": R, f"N_{i}": N, f"Q_{i}": want})
        out.update({f"after_{i}": after, f"no_bonus_{i}": no_bonus})
        out.update(
            {f"gamma_{i}": np.array(gamma), f"c_{i}": np.array(c), f"label_{i}": np.array(label)}
        )
    assert np.allclose(out["Q_0"], [[19.505, 20.0]], atol=1e-9)
    assert np.allclose(out["Q_1"], [[5.8, 6.0]], atol=1e-9)
    return _save("m06_mbie", **out)


def _expected_rewards(mdp, pi, steps):
    """The lab's expected_rewards: pi's exact expected reward at each step from the start."""
    P_pi, r_pi = _policy_chain(mdp.P, pi), (pi * mdp.R).sum(axis=1)
    d = np.zeros(mdp.n_states)
    d[mdp.start] = 1.0
    out = np.empty(steps)
    for t in range(steps):
        out[t] = d @ r_pi
        d = d @ P_pi
    return out


def _rollout(mdp, act, steps, rng):
    """States s_0..s_T and actions a_0..a_{T-1} from the start; act(s, rng) picks the action."""
    s, states, actions = mdp.start, [mdp.start], []
    for _ in range(steps):
        a = act(s, rng)
        s, _, _ = mdp.sample_step(s, a, rng)
        states.append(s)
        actions.append(a)
    return np.array(states, dtype=np.int64), np.array(actions, dtype=np.int64)


def m06_regret() -> bytes:
    """Exercise 6: Reg_t after every step on fixed trajectories, and the usual mistakes' values."""
    sys.path.insert(0, str(ROOT / "prl" / "src"))
    from prl.envs import RiverSwim

    sure = RiverSwim(4, p_right=1.0, p_stay=0.0, p_left=0.0, gamma=M06_GAMMA)  # deterministic
    river6, river10 = _riverswim(6), _riverswim(10)
    pi6 = _optimal(river6.P, river6.R, M06_GAMMA, tol=1e-12)[1]
    pi10 = _optimal(river10.P, river10.R, M06_GAMMA, tol=1e-12)[1]
    pi_sure = _optimal(sure.P, sure.R, M06_GAMMA, tol=1e-12)[1]
    assert np.all(pi6 == 1) and np.all(pi10 == 1) and np.all(pi_sure == 1)
    right = sure.policy_matrix(pi_sure)
    cases = [
        (
            "pi*'s own run of a deterministic 4-state chain",
            sure,
            right,
            np.array([0, 1, 2, 3, 3, 3, 3, 3]),
            np.ones(7, dtype=np.int64),
        ),
        (
            "the same chain: left, left, then right five times",
            sure,
            right,
            np.array([0, 0, 0, 1, 2, 3, 3, 3]),
            np.array([0, 0, 1, 1, 1, 1, 1]),
        ),
        (
            "300 steps of uniform random play on RiverSwim(10)",
            river10,
            river10.policy_matrix(pi10),
            *_rollout(river10, lambda s, rng: int(rng.integers(2)), 300, np.random.default_rng(0)),
        ),
        (
            "200 steps of the optimal policy on RiverSwim(6), a lucky run",
            river6,
            river6.policy_matrix(pi6),
            *_rollout(river6, lambda s, rng: int(pi6[s]), 200, np.random.default_rng(1)),
        ),
        ("one step: left at the left end", sure, right, np.array([0, 0]), np.array([0])),
    ]
    out: dict[str, np.ndarray] = {"n_cases": np.array(len(cases), dtype=np.int64)}
    for i, (label, mdp, pi_star, states, actions) in enumerate(cases):
        T = len(actions)
        opt = _expected_rewards(mdp, pi_star, T)
        here = mdp.R[states[:-1]]
        r = here[np.arange(T), actions]
        assert np.all(mdp.P[states[:-1], actions, states[1:]] > 0), label  # a possible run
        steps = opt - r
        out.update(
            {f"states_{i}": states.astype(np.int64), f"actions_{i}": actions.astype(np.int64)}
        )
        out.update({f"R_{i}": mdp.R, f"opt_{i}": opt, f"want_{i}": np.cumsum(steps)})
        out[f"label_{i}"] = np.array(label)
        # The usual mistakes, for the check's hints (labs/mutants/m06.py has each as a mutant).
        out[f"per_step_{i}"] = steps
        out[f"next_{i}"] = np.cumsum(opt - mdp.R[states[1:], actions])
        out[f"best_{i}"] = np.cumsum(here.max(axis=1) - r)
        out[f"average_{i}"] = np.cumsum(steps) / np.arange(1, T + 1)
        out[f"clipped_{i}"] = np.cumsum(np.maximum(0.0, steps))
    assert np.all(out["want_0"] == 0.0), "pi*'s own deterministic run has no regret at all"
    assert np.allclose(out["want_1"], [-0.005, -0.01, -0.01, 0.99, 1.99, 1.99, 1.99], atol=1e-12)
    w3 = out["want_3"]
    assert w3.min() < -3 and w3[-1] > 2, "the lucky run goes below 0 and ends above it"
    assert np.allclose(out["want_4"], [-0.005], atol=1e-12)
    # Every mistake differs from the answer on case 1 (the left-left-right run).
    for key in ("per_step", "next", "best", "average", "clipped"):
        assert np.abs(out[f"{key}_1"] - out["want_1"]).max() > 0.004, key
    return _save("m06_regret", **out)


def m06_psrl() -> bytes:
    """The stretch: a 3-state, 2-action model with untried pairs, and its posterior moments."""
    N_next = np.zeros((3, 2, 3))
    N_next[0, 0] = [5, 0, 0]
    N_next[0, 1] = [2, 3, 0]
    N_next[1, 0] = [7, 0, 0]
    N_next[1, 1] = [0, 1, 6]
    N_next[2, 1] = [0, 1, 0]  # (2, 0) never tried
    counts = N_next.sum(axis=2)
    rewards = np.array([[0.0, 0.0], [0.0, 0.25], [0.0, 0.7]])
    P_hat = np.zeros_like(N_next)
    for s in range(3):
        for a in range(2):
            if counts[s, a] > 0:
                P_hat[s, a] = N_next[s, a] / counts[s, a]
            else:
                P_hat[s, a, s] = 1.0  # Module 2's default: a self-loop with reward 0
    R_hat = np.where(counts > 0, rewards, 0.0)
    S = 3
    return _save(
        "m06_psrl",
        P_hat=P_hat,
        R_hat=R_hat,
        counts=counts,
        P_mean=(1.0 / S + N_next) / (1.0 + counts)[:, :, None],
        R_mean=(1.0 + counts * R_hat) / (counts + 1.0),
        R_var=1.0 / (counts + 1.0),
        n_draws=np.array(4000, dtype=np.int64),
    )


# ------------------------------------------------------------------ Module 7
# Function approximation (labs/src/07-function-approximation.py; contracts in
# _notes/m07-handoff.md). Each builder also stores the outputs of the usual mistakes (the
# mutants in labs/mutants/m07.py), so a checkpoint can name the one it sees.
M07_SCALE_CASES = [(3, 50), (2, 5), (1, 20), (4, 9)]  # (n_products, capacity)
M07_OBS_SCALE = np.array([2.4, 3.0, 0.2095, 3.5])  # the lab's OBS_SCALE (ex6-run)
M07_CENTERS = np.linspace(-1.0, 1.0, 5)
M07_WIDTH = 0.5


def _m07_baird():
    """Sutton and Barto's Figure 11.1, as prl.envs.Baird builds it."""
    from prl.envs import Baird

    b = Baird()
    return b.phi, b.mu_behavior, b.P_target, b.rewards, b.w0, b.gamma


def _m07_small_mdp(seed: int, S: int = 4, d: int = 3):
    """A random model (the handoff's small_mdp): normal features, Dirichlet mu and rows of P."""
    rng = np.random.default_rng(seed)
    phi = rng.normal(size=(S, d))
    mu = rng.dirichlet(np.ones(S))
    P = rng.dirichlet(np.ones(S), size=S)
    r = rng.normal(size=S)
    return phi, mu, P, r


def _m07_sweep(w, phi, mu, P, r, gamma, alpha):
    v = phi @ w
    return w + alpha * phi.T @ (mu * (r + gamma * P @ v - v))


def m07_scale() -> bytes:
    n = np.array([c[0] for c in M07_SCALE_CASES], dtype=np.int64)
    capacity = np.array([c[1] for c in M07_SCALE_CASES], dtype=np.int64)
    levels = [c + 1 for _, c in M07_SCALE_CASES]  # Python ints: exact
    states = [lv**k for (k, _), lv in zip(M07_SCALE_CASES, levels, strict=True)]
    out = {
        "n_products": n,
        "capacity": capacity,
        "n_states": np.array(states, dtype=np.int64),
        "n_actions": np.array(states, dtype=np.int64),
        "q_entries": np.array([s * s for s in states], dtype=np.int64),
        "feature_dim": np.array(
            [k * lv for (k, _), lv in zip(M07_SCALE_CASES, levels, strict=True)], dtype=np.int64
        ),
    }
    want = [
        (132_651, 132_651, 17_596_287_801, 153),
        (36, 36, 1296, 12),
        (21, 21, 441, 21),
        (10_000, 10_000, 100_000_000, 40),
    ]
    got = list(
        zip(
            out["n_states"].tolist(),
            out["n_actions"].tolist(),
            out["q_entries"].tolist(),
            out["feature_dim"].tolist(),
            strict=True,
        )
    )
    assert got == want, got
    return _save("m07_scale", **out)


def m07_td0() -> bytes:
    """Checkpoint 2: Baird's 1 -> 7, a terminal transition, three random d = 5 cases."""
    phi_b, _, _, _, w0, gamma_b = _m07_baird()
    rng = np.random.default_rng(70)
    cases = [
        (w0.copy(), phi_b[0], 0.0, phi_b[6], gamma_b, 0.01),
        (np.array([0.5, -1.0, 2.0]), np.array([1.0, 0.0, 2.0]), 1.0, np.zeros(3), 0.9, 0.1),
    ]
    for _ in range(3):
        cases.append(
            (
                rng.normal(size=5),
                rng.normal(size=5),
                float(rng.normal()),
                rng.normal(size=5),
                0.95,
                0.05,
            )
        )
    out: dict[str, np.ndarray] = {"n_cases": np.array(len(cases), dtype=np.int64)}
    for i, (w, phi, r, phi_next, g, a) in enumerate(cases):
        delta = r + g * w @ phi_next - w @ phi
        out |= {
            f"w_{i}": w,
            f"phi_{i}": phi,
            f"r_{i}": np.array(r),
            f"phi_next_{i}": phi_next,
            f"gamma_{i}": np.array(g),
            f"alpha_{i}": np.array(a),
            f"want_{i}": w + a * delta * phi,
            # the usual mistakes (mutants residual-gradient, next-features, no-discount, sign)
            f"residual_{i}": w + a * delta * (phi - g * phi_next),
            f"next_{i}": w + a * delta * phi_next,
            f"no_discount_{i}": w + a * (r + w @ phi_next - w @ phi) * phi,
            f"sign_{i}": w - a * delta * phi,
        }
    assert np.allclose(out["want_0"], [1.1776, 1, 1, 1, 1, 1, 10, 1.0888], atol=1e-12)
    assert np.allclose(out["want_1"], [0.15, -1.0, 1.3], atol=1e-12)
    for i in range(len(cases)):  # every mistake is visible in every case but the terminal one
        for key in ("residual", "no_discount", "next", "sign"):
            if i == 1 and key in ("residual", "no_discount"):
                continue  # phi(s') = 0: these equal the right answer there
            assert np.abs(out[f"{key}_{i}"] - out[f"want_{i}"]).max() > 1e-3, (key, i)
    return _save("m07_td0", **out)


def _m07_A(phi, mu, P, gamma):
    return phi.T @ np.diag(mu) @ (np.eye(len(mu)) - gamma * P) @ phi


def m07_td_matrix() -> bytes:
    """Checkpoint 3: A on Baird and on a random S = 4, d = 3 model (default_rng(71))."""
    phi_b, mu_b, P_b, _, _, gamma_b = _m07_baird()
    phi_r, mu_r, P_r, _ = _m07_small_mdp(71)
    out: dict[str, np.ndarray] = {}
    for tag, (phi, mu, P, g) in {
        "baird": (phi_b, mu_b, P_b, gamma_b),
        "random": (phi_r, mu_r, P_r, 0.9),
    }.items():
        A = _m07_A(phi, mu, P, g)
        out |= {
            f"{tag}_phi": phi,
            f"{tag}_mu": mu,
            f"{tag}_P": P,
            f"{tag}_gamma": np.array(g),
            f"{tag}_A": A,
            f"{tag}_transposed": A.T,
            f"{tag}_no_weights": phi.T @ (np.eye(len(mu)) - g * P) @ phi,
            f"{tag}_P_transposed": _m07_A(phi, mu, P.T, g),
            f"{tag}_symmetrized": (A + A.T) / 2,
            f"{tag}_plus_gamma": phi.T @ np.diag(mu) @ (np.eye(len(mu)) + g * P) @ phi,
        }
    sevenA = 7 * out["baird_A"]
    upper = np.zeros(8)
    upper[6:] = [-1.98, -1.96]
    for i in range(6):
        row = upper.copy()
        row[i] = 4.0
        assert np.allclose(sevenA[i], row, atol=1e-12)
    assert np.allclose(sevenA[6], [0, 0, 0, 0, 0, 0, 0.01, 0.02], atol=1e-12)
    assert np.allclose(sevenA[7], [2, 2, 2, 2, 2, 2, -5.92, -5.84], atol=1e-12)
    return _save("m07_td_matrix", **out)


M07_SWEEPS = (1, 10, 100, 1000)


def m07_expected_update() -> bytes:
    """Checkpoint 4: one sweep on a random model with rewards; Baird's path from w0."""
    phi, mu, P, r = _m07_small_mdp(72)
    w = np.array([0.3, -0.2, 1.0])
    g, a = 0.9, 0.1
    v = phi @ w
    seq = w.copy()
    for s in range(len(mu)):  # mutant 'sequential': one state at a time
        vs = phi @ seq
        seq = seq + a * mu[s] * (r[s] + g * P[s] @ vs - vs[s]) * phi[s]
    out: dict[str, np.ndarray] = {
        "phi": phi,
        "mu": mu,
        "P": P,
        "r": r,
        "w": w,
        "gamma": np.array(g),
        "alpha": np.array(a),
        "want": _m07_sweep(w, phi, mu, P, r, g, a),
        "unweighted": w + a * phi.T @ (r + g * P @ v - v),
        "P_transposed": w + a * phi.T @ (mu * (r + g * P.T @ v - v)),
        "sequential": seq,
    }
    phi_b, mu_b, P_b, r_b, w0, gamma_b = _m07_baird()
    wk = w0.copy()
    for k in range(1, M07_SWEEPS[-1] + 1):
        wk = _m07_sweep(wk, phi_b, mu_b, P_b, r_b, gamma_b, 0.01)
        if k in M07_SWEEPS:
            out[f"baird_{k}"] = wk.copy()
    out["sweeps"] = np.array(M07_SWEEPS, dtype=np.int64)
    out["alpha_baird"] = np.array(0.01)
    norms = [round(float(np.linalg.norm(out[f"baird_{k}"])), 3) for k in M07_SWEEPS]
    assert round(float(np.linalg.norm(w0)), 3) == 10.344
    assert norms == [10.366, 10.609, 16.826, 436.856], norms
    w1000 = np.round(out["baird_1000"], 3)
    assert np.all(w1000[:6] == 114.434) and w1000[6] == 6.850 and w1000[7] == 335.001, w1000
    for key in ("unweighted", "P_transposed", "sequential"):
        assert np.abs(out[key] - out["want"]).max() > 1e-3, key
    return _save("m07_expected_update", **out)


def _m07_stationary(P):
    """The stationary distribution of P: the left eigenvector for eigenvalue 1, normalized."""
    S = len(P)
    M = np.vstack([P.T - np.eye(S), np.ones(S)])
    rhs = np.zeros(S + 1)
    rhs[-1] = 1.0
    mu, *_ = np.linalg.lstsq(M, rhs, rcond=None)
    mu = np.where(np.abs(mu) < 1e-14, 0.0, mu)  # rounding noise, printed as -0. otherwise
    return mu / mu.sum()


def m07_legs() -> bytes:
    """Checkpoint 5: Baird, and a second target policy with nonzero rewards."""
    _, _, P_b, r_b, _, gamma = _m07_baird()
    P_m = np.zeros((7, 7))
    P_m[:, 6] = 0.5
    P_m[:, :6] = 0.5 / 6
    r_m = np.array([1.0, 0, 0, 0, 0, 0, 2.0])
    out: dict[str, np.ndarray] = {"gamma": np.array(gamma)}
    for tag, (P, r) in {"baird": (P_b, r_b), "modified": (P_m, r_m)}.items():
        mu = _m07_stationary(P)
        assert np.allclose(mu @ P, mu, atol=1e-14) and abs(mu.sum() - 1) < 1e-14
        out |= {
            f"{tag}_P": P,
            f"{tag}_rewards": r,
            f"{tag}_mu": mu,
            f"{tag}_v_pi": np.linalg.solve(np.eye(7) - gamma * P, r),
        }
    assert np.allclose(out["baird_mu"], np.eye(7)[6], atol=1e-14)
    assert np.allclose(out["modified_mu"], [1 / 12] * 6 + [0.5], atol=1e-14)
    assert np.all(out["baird_v_pi"] == 0)
    assert np.abs(out["modified_v_pi"] - r_m).max() > 1  # mutant mc-rewards fails here
    return _save("m07_legs", **out)


M07_Q_CASES = [(0, False, False), (1, True, False), (0, False, True), (1, True, True)]


def m07_linear_q() -> bytes:
    """Checkpoint 6.1: four transitions (a, terminated, truncated) on fixed arrays."""
    rng = np.random.default_rng(76)
    W = rng.normal(size=(2, 6))
    phi = rng.random(6)
    phi_next = rng.random(6)
    r, g, alpha = 1.0, 0.99, 0.3

    def update(a, target):
        W_new = W.copy()
        W_new[a] = W[a] + alpha * (target - W[a] @ phi) * phi
        return W_new

    out: dict[str, np.ndarray] = {
        "W": W,
        "phi": phi,
        "phi_next": phi_next,
        "r": np.array(r),
        "gamma": np.array(g),
        "alpha": np.array(alpha),
        "cases": np.array(M07_Q_CASES, dtype=np.int64),
    }
    q_max = np.max(W @ phi_next)
    for i, (a, term, _trunc) in enumerate(M07_Q_CASES):
        out[f"want_{i}"] = update(a, r if term else r + g * q_max)
        out[f"always_{i}"] = update(a, r + g * q_max)  # bootstraps from a terminal state
        out[f"never_{i}"] = update(a, r)  # stops the bootstrap on truncation as well
        out[f"action0_{i}"] = update(a, r if term else r + g * W[0] @ phi_next)
        out[f"no_discount_{i}"] = update(a, r if term else r + q_max)
        W_next = W.copy()  # steps along phi(s') instead of phi(s)
        W_next[a] = W[a] + alpha * ((r if term else r + g * q_max) - W[a] @ phi) * phi_next
        out[f"next_{i}"] = W_next
    assert W[1] @ phi_next > W[0] @ phi_next  # the max is action 1, so 'action0' differs
    assert not np.allclose(out["want_2"], out["never_2"])  # truncated: must bootstrap
    assert not np.allclose(out["want_1"], out["always_1"])  # terminated: must not
    return _save("m07_linear_q", **out)


def _m07_bumps(obs):
    """The 625 Gaussian bumps of the lab's rbf_raw: a product over the four scaled numbers."""
    z = np.asarray(obs, dtype=float) / M07_OBS_SCALE
    per_dim = np.exp(-((z[:, None] - M07_CENTERS) ** 2) / (2 * M07_WIDTH**2))
    return np.einsum("i,j,k,l->ijkl", *per_dim).ravel()


def m07_features() -> bytes:
    """Checkpoint 7.1: unit-length features at seven fixed observations (default_rng(77))."""
    rng = np.random.default_rng(77)
    obs = np.vstack(
        [
            np.zeros(4),
            [2.4, -3.0, 0.2, 3.5],
            rng.uniform(-1, 1, (5, 4)) * np.array([2.4, 3.0, 0.21, 3.5]),
        ]
    )
    raw = np.array([_m07_bumps(o) for o in obs])
    out = {"obs": obs, "features": raw / np.linalg.norm(raw, axis=1, keepdims=True)}
    assert round(float(raw[0] @ raw[0]), 3) == 9.868  # the colleague's upright state
    return _save("m07_features", **out)


def _m07_tdc(w, v, phi, mu, P, r, g, alpha, beta, sign=-1.0):
    e = phi.T @ (mu * (r + g * P @ phi @ w - phi @ w))
    cross = (P @ phi).T @ (mu[:, None] * phi)
    gram = phi.T @ (mu[:, None] * phi)
    return w + alpha * (e + sign * g * cross @ v), v + beta * (e - gram @ v), e


def m07_tdc() -> bytes:
    """The stretch: one expected TDC sweep on Baird and on a random model (default_rng(78))."""
    phi_b, mu_b, P_b, r_b, w0, gamma_b = _m07_baird()
    phi_r, mu_r, P_r, r_r = _m07_small_mdp(78)
    cases = {
        "baird": (w0, np.linspace(-1, 1, 8), phi_b, mu_b, P_b, r_b, gamma_b, 0.005, 0.05),
        "random": (
            np.array([0.3, -0.2, 1.0]),
            np.array([0.1, 0.2, -0.3]),
            phi_r,
            mu_r,
            P_r,
            r_r,
            0.9,
            0.1,
            0.2,
        ),
    }
    out: dict[str, np.ndarray] = {}
    names = ("w", "v", "phi", "mu", "P", "r", "gamma", "alpha", "beta")
    for tag, args in cases.items():
        w_new, v_new, e = _m07_tdc(*args)
        w_plus, _, _ = _m07_tdc(*args, sign=1.0)
        out |= {f"{tag}_{k}": np.asarray(x, dtype=float) for k, x in zip(names, args, strict=True)}
        out |= {
            f"{tag}_want_w": w_new,
            f"{tag}_want_v": v_new,
            f"{tag}_no_correction_w": args[0] + args[7] * e,
            f"{tag}_correction_sign_w": w_plus,
        }
        assert np.abs(out[f"{tag}_no_correction_w"] - w_new).max() > 1e-4, tag
        assert np.abs(w_plus - w_new).max() > 1e-4, tag
    return _save("m07_tdc", **out)


FIXTURES = {
    "m01_returns": m01_returns,
    "m01_eval": m01_eval,
    "m02_plan": m02_plan,
    "m02_model": m02_model,
    "m02_plan_eval": m02_plan_eval,
    "m02_residual": m02_residual,
    "m03_prediction": m03_prediction,
    "m03_nstep": m03_nstep,
    "m03_decomposition": m03_decomposition,
    "m04_bandit": m04_bandit,
    "m04_policy": m04_policy,
    "m04_greedy_replay": m04_greedy_replay,
    "m04_ucb1": m04_ucb1,
    "m04_ucb1_replay": m04_ucb1_replay,
    "m04_ucb1_for_range": m04_ucb1_for_range,
    "m04_beta": m04_beta,
    "m04_ts_shares": m04_ts_shares,
    "m04_verdict": m04_verdict,
    "m04_scale": m04_scale,
    "m04_gauss": m04_gauss,
    "m05_reward": m05_reward,
    "m05_ridge": m05_ridge,
    "m05_linucb": m05_linucb,
    "m05_lints": m05_lints,
    "m05_verdicts": m05_verdicts,
    "m05_logging": m05_logging,
    "m06_chain": m06_chain,
    "m06_q0": m06_q0,
    "m06_mbie": m06_mbie,
    "m06_regret": m06_regret,
    "m06_psrl": m06_psrl,
    "m07_scale": m07_scale,
    "m07_td0": m07_td0,
    "m07_td_matrix": m07_td_matrix,
    "m07_expected_update": m07_expected_update,
    "m07_legs": m07_legs,
    "m07_linear_q": m07_linear_q,
    "m07_features": m07_features,
    "m07_tdc": m07_tdc,
}


def main(argv: list[str]) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    stale = []
    for name, build in FIXTURES.items():
        blob = build()
        path = OUT / f"{name}.npz"
        if "--check" in argv:
            if not path.exists() or not same_content(path.read_bytes(), blob):
                stale.append(name)
        else:
            path.write_bytes(blob)
            print(f"wrote {path.relative_to(ROOT)}")
    if stale:
        print(f"Fixtures out of date: {', '.join(stale)}. Run scripts/make_expected.py.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
