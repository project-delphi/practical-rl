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


FIXTURES = {"m01_returns": m01_returns, "m01_eval": m01_eval}


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
