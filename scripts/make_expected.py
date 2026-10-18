"""Write the checkpoint fixtures in prl/src/prl/_expected/.

Fixtures are precomputed expected values, so prl ships answers to checks,
never solver code. This script may compute them however it likes; it is
not part of prl. Deterministic: rerunning it must produce identical files
(CI's drift gate checks this).
"""

from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "prl" / "src" / "prl" / "_expected"


def save_npz(**arrays: np.ndarray) -> bytes:
    """An .npz that is byte-identical across runs (fixed zip timestamps, sorted keys)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as zf:
        for key in sorted(arrays):
            info = zipfile.ZipInfo(f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            arr_buf = io.BytesIO()
            np.lib.format.write_array(arr_buf, np.asarray(arrays[key]), allow_pickle=False)
            zf.writestr(info, arr_buf.getvalue())
    return buf.getvalue()


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
    "m05_reward": m05_reward,
    "m05_ridge": m05_ridge,
    "m05_linucb": m05_linucb,
    "m05_lints": m05_lints,
    "m05_verdicts": m05_verdicts,
    "m05_logging": m05_logging,
}


def same_content(old: bytes, new: bytes) -> bool:
    """True if two fixture files hold the same arrays.

    Byte-identical files match. Otherwise keys, dtypes, shapes and every non-float value must
    match exactly, and floats to 1e-12: LAPACK's linear solve differs in the last bits between
    platforms (Accelerate on macOS, OpenBLAS on Linux), so a fixture written on one machine is
    not byte-identical to the same fixture built on another.
    """
    if old == new:
        return True
    a, b = np.load(io.BytesIO(old)), np.load(io.BytesIO(new))
    if sorted(a.files) != sorted(b.files):
        return False
    for key in a.files:
        x, y = a[key], b[key]
        if x.dtype != y.dtype or x.shape != y.shape:
            return False
        if np.issubdtype(x.dtype, np.floating):
            if not np.allclose(x, y, rtol=1e-12, atol=1e-12, equal_nan=True):
                return False
        elif not np.array_equal(x, y):
            return False
    return True


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
