"""Checkpoints for Module 1 (MDPs, returns and the Bellman equation).

Expected values come from fixtures written by scripts/make_expected.py, or
from the environments themselves (an environment's arrays are its definition,
not an algorithm). Every check takes the participant's function and calls it,
so it can tell a correct function from a stub or a plausible mistake.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from ..envs import CliffGridworld, InventoryMDP, demand_pmf
from ..envs.gridworld import ACTION_NAMES
from . import CheckFailed, assert_close, assert_shape, expected

#: The inventory parameters used in Module 1 (the notebook shows the same numbers).
INVENTORY = dict(
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


def _two(out: Any, what: str) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(out, tuple) or len(out) != 2:
        raise CheckFailed(f"{what} should return a pair (P, R).")
    return np.asarray(out[0], dtype=float), np.asarray(out[1], dtype=float)


def check_discounted_return(fn: Callable[[list[float], float], float]) -> None:
    fx = expected("m01_returns")
    rewards, lengths, gammas, answers = fx["rewards"], fx["lengths"], fx["gammas"], fx["returns"]
    for i in range(len(lengths)):
        rs = [float(r) for r in rewards[i, : lengths[i]]]
        got = fn(rs, float(gammas[i]))
        if got is None:
            raise CheckFailed("discounted_return returned None. Did you forget `return`?")
        assert_close(
            got,
            answers[i],
            atol=1e-9,
            what=f"discounted_return({rs}, gamma={gammas[i]})",
            hint="G_0 = r_0 + gamma r_1 + gamma^2 r_2 + ... The first reward is not discounted.",
        )
    if fn([], 0.9) != 0:
        raise CheckFailed("discounted_return([], gamma) should be 0: no rewards, no return.")


def check_gridworld_arrays(fn: Callable[[int, int], tuple]) -> None:
    for rows, cols in [(2, 3), (3, 4), (4, 6)]:
        P, R = _two(fn(rows, cols), "gridworld_arrays")
        ref = CliffGridworld(rows, cols)
        assert_shape(P, ref.P.shape, what=f"P for a {rows}x{cols} grid")
        assert_shape(R, ref.R.shape, what=f"R for a {rows}x{cols} grid")
        sums = P.sum(axis=2)
        if not np.allclose(sums, 1.0):
            s, a = np.unravel_index(np.argmax(np.abs(sums - 1)), sums.shape)
            raise CheckFailed(
                f"P[{s}, {a}, :] sums to {sums[s, a]:.3g}; every row of P must sum to 1."
            )
        bad = np.argwhere(~np.isclose(P, ref.P).all(axis=2) | ~np.isclose(R, ref.R))
        if bad.size:
            s, a = (int(x) for x in bad[0])
            r, c = divmod(s, cols)
            where = (
                "the goal"
                if s == ref.goal
                else "a cliff cell"
                if s in ref.cliff
                else "the start"
                if s == ref.start
                else f"row {r}, column {c}"
            )
            want = int(np.argmax(ref.P[s, a]))
            raise CheckFailed(
                f"In a {rows}x{cols} grid, state {s} ({where}), action {ACTION_NAMES[a]}: expected next "
                f"state {want} with reward {ref.R[s, a]:g}, got next-state probabilities "
                f"{np.round(P[s, a], 3).tolist()} and reward {R[s, a]:g}."
            )


def check_inventory_spec(fn: Callable[[int], dict]) -> None:
    for capacity in (5, 10):
        spec = fn(capacity)
        if not isinstance(spec, dict) or not {"n_states", "n_actions"} <= set(spec):
            raise CheckFailed(
                "inventory_spec should return a dict with keys 'n_states' and 'n_actions'."
            )
        if spec["n_states"] != capacity + 1:
            raise CheckFailed(
                f"With capacity {capacity}, how many stock levels are possible, counting 0? "
                f"You said {spec['n_states']}."
            )
        if spec["n_actions"] != capacity + 1:
            raise CheckFailed(
                f"With capacity {capacity}, how many order sizes are possible, counting "
                f"'order nothing'? You said {spec['n_actions']}."
            )


def check_period_reward(fn: Callable[..., float]) -> None:
    fx = expected("m01_eval")
    p = dict(INVENTORY)
    for (stock, order, demand), want in zip(fx["reward_cases"], fx["reward_expected"], strict=True):
        got = fn(int(stock), int(order), int(demand), p)
        assert_close(
            got,
            want,
            atol=1e-9,
            what=f"period_reward(stock={stock}, order={order}, demand={demand})",
            hint="Revenue counts only units sold; you pay the fixed cost only if you order; "
            "orders are capped so stock never exceeds capacity.",
        )


def check_inventory_arrays(fn: Callable[..., tuple]) -> None:
    for params in (dict(INVENTORY), dict(INVENTORY, capacity=4, max_demand=6, demand_mean=2.0)):
        pmf = demand_pmf(params["demand_mean"], params["max_demand"])
        P, R = _two(fn(params["capacity"], pmf, params), "inventory_arrays")
        ref = InventoryMDP(**params)
        assert_shape(P, ref.P.shape, what="P")
        assert_shape(R, ref.R.shape, what="R")
        assert_close(
            P.sum(axis=2),
            np.ones(ref.P.shape[:2]),
            atol=1e-9,
            what="row sums of P",
            hint="Every P[s, a, :] is a probability distribution.",
        )
        assert_close(
            P,
            ref.P,
            atol=1e-9,
            what="P",
            hint="After ordering, stock is y = s + q; next stock is y - min(y, demand).",
        )
        assert_close(
            R,
            ref.R,
            atol=1e-9,
            what="R",
            hint="R[s, a] is the EXPECTED reward: sum over demand of pmf[d] * period reward.",
        )


def check_evaluate_exact(fn: Callable[..., np.ndarray]) -> None:
    fx = expected("m01_eval")
    cases = [
        (
            "the 3x4 gridworld (uniform policy)",
            fx["grid_P"],
            fx["grid_R"],
            float(fx["grid_gamma"]),
            fx["grid_pi"],
            fx["grid_V"],
        ),
        (
            "the random MDP",
            fx["rand_P"],
            fx["rand_R"],
            float(fx["gammas"][1]),
            fx["rand_pi"],
            fx["rand_V"][1],
        ),
    ]
    inv = InventoryMDP(**INVENTORY)
    cases.append(
        ("the inventory MDP (order up to 7)", inv.P, inv.R, 0.95, fx["inv_pi"], fx["inv_V"])
    )
    for name, P, R, gamma, pi, V in cases:
        got = fn(P, R, gamma, pi)
        assert_close(
            got,
            V,
            atol=1e-8,
            rtol=1e-8,
            what=f"V for {name}",
            hint="Solve (I - gamma P_pi) V = r_pi with np.linalg.solve; P_pi and r_pi average over pi.",
        )


def check_evaluate_iterative(fn: Callable[..., tuple]) -> None:
    fx = expected("m01_eval")
    tol = float(fx["tol"])
    cases = list(zip(fx["gammas"], fx["rand_V"], fx["rand_k"], [1.0] * 3, strict=True))
    # The same MDP with rewards negated: V^pi flips sign (it is linear in r) and the iterates
    # now decrease, which catches a stopping test that forgets abs().
    cases += [(fx["gammas"][1], -fx["rand_V"][1], fx["rand_k"][1], -1.0)]
    for gamma, V_exact, k_ref, sign in cases:
        out = fn(fx["rand_P"], sign * fx["rand_R"], float(gamma), fx["rand_pi"], tol)
        if not isinstance(out, tuple) or len(out) != 2:
            raise CheckFailed("evaluate_iterative should return a pair (V, k).")
        V, k = np.asarray(out[0], dtype=float), out[1]
        err = float(np.max(np.abs(V - V_exact)))
        bound = gamma * tol / (1 - gamma)
        if err > bound * (1 + 1e-6) + 1e-12:
            raise CheckFailed(
                f"With gamma={gamma}, your V is {err:.3g} from V^pi, but stopping when "
                f"max|V_(k+1) - V_k| < tol guarantees at most {bound:.3g}. Check the stopping rule."
            )
        if not isinstance(k, (int, np.integer)) or abs(int(k) - int(k_ref)) > 1:
            raise CheckFailed(
                f"With gamma={gamma} and tol={tol:g}, starting from V = 0, the reference takes "
                f"{int(k_ref)} updates; you report k = {k}. Count one per application of T^pi."
            )


def check_bellman_residual(fn: Callable[..., float]) -> None:
    fx = expected("m01_eval")
    for V, want in zip(fx["res_V"], fx["res_expected"], strict=True):
        got = fn(fx["rand_P"], fx["rand_R"], 0.9, fx["rand_pi"], V)
        assert_close(
            got,
            want,
            atol=1e-9,
            what="bellman_residual",
            hint="The residual is max over states of |r_pi + gamma P_pi V - V| (the infinity norm).",
        )


def check_evaluate_suspect(fn: Callable[..., np.ndarray]) -> None:
    fx = expected("m01_eval")
    got = fn(fx["rand_P"], fx["rand_R"], 0.9, fx["rand_pi"])
    V, _ = fx["rand_V"][1], None
    if np.allclose(got, V, atol=1e-8):
        return
    raise CheckFailed(
        "The repaired evaluator still disagrees with V^pi (largest error "
        f"{np.max(np.abs(np.asarray(got) - V)):.3g}). Compute its Bellman residual: which line builds P_pi?"
    )


def check_operator_gains(fn: Callable[..., tuple]) -> None:
    """Stretch: ||gamma P_pi||_inf <= gamma, while the 2-norm gain can exceed 1."""
    P_pi = np.array([[0.0, 1.0], [0.0, 1.0]])
    inf_gain, two_gain = fn(P_pi, 0.9)
    assert_close(
        inf_gain,
        0.9,
        atol=1e-9,
        what="infinity-norm gain",
        hint="The infinity-norm of a matrix is its largest absolute row sum.",
    )
    assert_close(
        two_gain,
        0.9 * np.sqrt(2),
        atol=1e-9,
        what="2-norm gain",
        hint="The 2-norm of a matrix is its largest singular value (np.linalg.norm(A, 2)).",
    )
