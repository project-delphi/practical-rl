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


def _policy_iteration(P, R, gamma, pi0):
    """Exact evaluation and greedy improvement (keep the current action on ties); (V, pi, k)."""
    S = P.shape[0]
    pi, k = np.asarray(pi0, dtype=int).copy(), 0
    while True:
        V = _exact_value(P, R, gamma, pi)
        k += 1
        Q = R + gamma * P @ V
        keep = Q[np.arange(S), pi] >= Q.max(axis=1)
        pi_new = np.where(keep, pi, Q.argmax(axis=1))
        if np.array_equal(pi_new, pi):
            return V, pi, k
        pi = pi_new


def _modified_policy_iteration(P, R, gamma, m, tol):
    S = P.shape[0]
    V, rounds = np.zeros(S), 0
    while True:
        Q = R + gamma * P @ V
        pi = Q.argmax(axis=1)
        V_new = Q.max(axis=1)
        P_pi, r_pi = P[np.arange(S), pi], R[np.arange(S), pi]
        for _ in range(m - 1):
            V_new = r_pi + gamma * P_pi @ V_new
        rounds += 1
        if np.max(np.abs(V_new - V)) < tol:
            return V_new, (R + gamma * P @ V_new).argmax(axis=1), rounds
        V = V_new


def _optimal(P, R, gamma, tol=1e-6):
    """(V*, pi*, Q*, k_vi): pi* from value iteration, V* by the exact solve of pi*."""
    _, Q_vi, k, _ = _value_iteration(P, R, gamma, tol)
    pi_star = Q_vi.argmax(axis=1)
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
    # Duplicate actions: action 3 is a copy of the action pi* uses most. Started from
    # 'always 3', the tie rule keeps 3 where it is optimal; argmax alone returns the copy's
    # original index.
    _, pi_r, _, _ = _optimal(P, R, 0.9, tol)
    dup_of = int(np.bincount(pi_r, minlength=3).argmax())
    P_d = np.concatenate([P, P[:, dup_of : dup_of + 1]], axis=1)
    R_d = np.concatenate([R, R[:, dup_of : dup_of + 1]], axis=1)
    pi0_d = np.full(5, 3)
    V_d, pi_d, k_d = _policy_iteration(P_d, R_d, 0.9, pi0_d)
    assert np.any(pi_d == 3) and not np.any(pi_d == dup_of)
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
    for i, N in enumerate(Ns):
        for j, seed in enumerate(seeds):
            log = _sample_transitions(inv, int(N), np.random.default_rng(int(seed)))
            P_hat, R_hat, _ = _estimate_model(log, S, A)
            _, Q_hat, _, _ = _value_iteration(P_hat, R_hat, 0.95, 1e-6)
            for s in range(S):
                best = Q_hat[s].max()
                rivals = Q_hat[s][eff[s] != eff[s, Q_hat[s].argmax()]]
                assert rivals.size == 0 or best - rivals.max() > 1e-6, (N, seed, s)
            pi_hat = Q_hat.argmax(axis=1)
            loss[i, j] = np.max(V_star - _exact_value(inv.P, inv.R, 0.95, pi_hat))
            eps_P[i, j] = np.abs(inv.P - P_hat).sum(axis=2).max()
    assert abs(eps_P[0].mean() - 1.869) < 5e-4 and abs(loss[0].mean() - 31.54) < 5e-3
    return _save("m02_plan_eval", Ns=Ns, seeds=seeds, loss=loss, eps_P=eps_P)


FIXTURES = {
    "m01_returns": m01_returns,
    "m01_eval": m01_eval,
    "m02_plan": m02_plan,
    "m02_model": m02_model,
    "m02_plan_eval": m02_plan_eval,
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
            if not np.allclose(x, y, rtol=1e-12, atol=1e-12):
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
