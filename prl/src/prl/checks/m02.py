"""Checkpoints for Module 2 (planning: value iteration, policy iteration, model error).

Expected values come from fixtures written by scripts/make_expected.py (exact solves and
sweep counts on known models). A check may compute one backup, R + gamma * P @ V, to
compare the returned Q with the returned V; it never runs a solver of its own.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from ..envs import CliffGridworld, InventoryMDP
from . import CheckFailed, assert_close, assert_shape, assert_true, expected
from .m01 import INVENTORY

TOL = 1e-6


def _stop_bound(gamma: float, tol: float = TOL) -> float:
    """What stopping at max|V_new - V| < tol guarantees: ||V - V*||_inf <= gamma tol / (1 - gamma)."""
    return gamma * tol / (1 - gamma)


def _int(value: Any, what: str) -> int:
    try:
        k = int(value)
    except (TypeError, ValueError):
        raise CheckFailed(f"{what} should be a whole number; you returned {value!r}.") from None
    if k != value:
        raise CheckFailed(f"{what} should be a whole number; you returned {value!r}.")
    return k


def _unpack(out: Any, n: int, what: str, names: str) -> tuple:
    if not isinstance(out, (tuple, list)) or len(out) != n:
        raise CheckFailed(f"{what} should return {names}, a tuple of {n}.")
    return tuple(out)


def _first_non_greedy(pi: np.ndarray, Q_star: np.ndarray, atol: float = 1e-9) -> int | None:
    """The first state where pi does not pick a maximizer of Q*, or None.

    The shop has exact ties: orders the capacity caps to the same delivery are the same
    action (at stock 1, orders 9 and 10 both fill the shelf), so any maximizer is optimal.
    """
    S = Q_star.shape[0]
    if np.any(pi < 0) or np.any(pi >= Q_star.shape[1]):
        return int(np.flatnonzero((pi < 0) | (pi >= Q_star.shape[1]))[0])
    bad = Q_star[np.arange(S), pi] < Q_star.max(axis=1) - atol
    return int(np.flatnonzero(bad)[0]) if bad.any() else None


def _loud_mdp(S: int, gamma: float) -> tuple[np.ndarray, np.ndarray, float]:
    """The 'mostly settled' MDP: every state absorbing, R = 0 except R[0] = [1, 0.5].

    Its arrays are a definition, not an algorithm; the fixture holds V* and the sweep count.
    """
    P = np.zeros((S, 2, S))
    P[np.arange(S), :, np.arange(S)] = 1.0
    R = np.zeros((S, 2))
    R[0] = [1.0, 0.5]
    return P, R, gamma


def _vi_cases(fx: dict[str, np.ndarray]) -> list[tuple]:
    """(name, P, R, gamma, V*, k) for value iteration from V = 0 at tol 1e-6."""
    inv = InventoryMDP(**INVENTORY)
    P_l, R_l, g_l = _loud_mdp(int(fx["loud_S"]), float(fx["loud_gamma"]))
    g = float(fx["rand_gamma"])
    grid = CliffGridworld(3, 4, gamma=0.9)
    return [
        ("the shop", inv.P, inv.R, 0.95, fx["inv_V_star"], int(fx["inv_k_vi"])),
        # Every reward is <= 0, so V falls from 0: a test without abs() stops at once.
        (
            "the 3x4 cliff gridworld",
            grid.P,
            grid.R,
            0.9,
            fx["grid_V_star"],
            int(fx["grid_k_vi"]),
        ),
        (
            "the mostly settled MDP (100 absorbing states, reward only in state 0)",
            P_l,
            R_l,
            g_l,
            fx["loud_V_star"],
            int(fx["loud_k_vi"]),
        ),
        ("the random MDP", fx["rand_P"], fx["rand_R"], g, fx["rand_V_star"], int(fx["rand_k_vi"])),
        (
            "the random MDP with negated rewards",
            fx["rand_P"],
            -fx["rand_R"],
            g,
            fx["rand_neg_V_star"],
            int(fx["rand_neg_k_vi"]),
        ),
    ]


def _check_vi_result(
    name: str,
    P: np.ndarray,
    R: np.ndarray,
    gamma: float,
    V: Any,
    Q: Any,
    k: Any,
    V_star: np.ndarray,
    k_ref: int,
    *,
    fn_name: str,
) -> None:
    S, A = R.shape
    V = np.asarray(V, dtype=float)
    assert_shape(V, (S,), what=f"V for {name}")
    if not np.all(np.isfinite(V)):
        raise CheckFailed(f"V for {name} contains NaN or infinity.")
    err = float(np.max(np.abs(V - V_star)))
    bound = _stop_bound(gamma)
    if err > bound * (1 + 1e-6) + 1e-12:
        raise CheckFailed(
            f"On {name} with gamma={gamma}, your V is {err:.3g} from V*, but stopping when "
            f"max|V_new - V| < tol guarantees at most {bound:.3g}. Check the stopping rule: it "
            "tests the LARGEST absolute change over all states, and the backup must be "
            "max over actions of R + gamma * P @ V, built from the whole old V."
        )
    Q = np.asarray(Q, dtype=float)
    assert_shape(Q, (S, A), what=f"Q for {name}")
    assert_close(
        Q,
        R + gamma * P @ V,
        atol=1e-9,
        rtol=0.0,
        what=f"Q for {name}",
        hint="Return the Q computed from the V you return (R + gamma * P @ V with the final V), "
        "not the Q of the sweep before it.",
    )
    k_int = _int(k, "k (the number of sweeps)")
    if abs(k_int - k_ref) > 1:
        msg = (
            f"On {name} with gamma={gamma} and tol={TOL:g}, starting from V = 0, the reference "
            f"takes {k_ref} sweeps; your {fn_name} reports k = {k_int}. Count one per backup."
        )
        if k_int < k_ref - 1:
            msg += (
                " Fewer sweeps means the loop stopped early (a stopping statistic other than "
                "the max change?) or V was updated in place, state by state: build all of "
                "V_new from the old V."
            )
        else:
            msg += " More sweeps means the stopping test fires too late: compare max|V_new - V| with tol."
        raise CheckFailed(msg)


def check_value_iteration(fn: Callable[..., tuple]) -> None:
    fx = expected("m02_plan")
    for name, P, R, gamma, V_star, k_ref in _vi_cases(fx):
        V, Q, k = _unpack(fn(P, R, gamma, TOL), 3, "value_iteration", "(V, Q, k)")
        _check_vi_result(name, P, R, gamma, V, Q, k, V_star, k_ref, fn_name="value_iteration")


def check_greedy(fn: Callable[[np.ndarray], np.ndarray]) -> None:
    fx = expected("m02_plan")
    cases = [
        (f"random Q number {i + 1}", Q, pi)
        for i, (Q, pi) in enumerate(zip(fx["greedy_Q"], fx["greedy_expected"], strict=True))
    ]
    for name, Q, want in cases:
        got = _greedy_array(fn(Q.copy()), Q, name)
        if not np.array_equal(got, want):
            s = int(np.flatnonzero(got != want)[0])
            raise CheckFailed(
                f"greedy({name}) picks action {got[s]} in state {s}, but the largest entry of "
                f"row {s} is at action {want[s]}. One argmax per state: over the action axis "
                "(axis=1), and the largest value, not the smallest."
            )
    # Ties: the shop's Q* (capped orders are identical actions) and a Q with one tied row.
    for name, Q in (("the shop's Q*", fx["inv_Q_star"]), ("a Q with a tie", fx["greedy_tie_Q"])):
        got = _greedy_array(fn(Q.copy()), Q, name)
        s = _first_non_greedy(got, Q, atol=0.0)
        if s is not None:
            raise CheckFailed(
                f"greedy({name}) picks action {got[s]} in state {s}, which is not a maximizer of "
                f"row {s} (the best is action {int(Q[s].argmax())}). Where a row ties, any "
                "maximizer is accepted."
            )


def _greedy_array(out: Any, Q: np.ndarray, name: str) -> np.ndarray:
    if out is None:
        raise CheckFailed("greedy returned None. Did you forget `return`?")
    got = np.asarray(out)
    S = Q.shape[0]
    if got.shape != (S,):
        raise CheckFailed(
            f"greedy({name}) should return one action per state, shape ({S},), but has shape "
            f"{got.shape}. Q is (S, A): the argmax goes over the action axis (axis=1)."
        )
    if not np.issubdtype(got.dtype, np.integer):
        raise CheckFailed(
            f"greedy({name}) should return integer action indices (argmax), but its dtype is "
            f"{got.dtype}. Return the index of the largest entry, not the entry."
        )
    return got


def check_policy_iteration(fn: Callable[..., tuple]) -> None:
    fx = expected("m02_plan")
    inv = InventoryMDP(**INVENTORY)
    S = inv.n_states
    cases = [
        (
            "the shop from 'never order'",
            inv.P,
            inv.R,
            0.95,
            np.zeros(S, dtype=int),
            fx["inv_V_star"],
            fx["inv_pi_star"],
            int(fx["inv_k_pi_zero"]),
        ),
        (
            "the shop from 'order up to 7'",
            inv.P,
            inv.R,
            0.95,
            fx["inv_pi7"],
            fx["inv_V_star"],
            fx["inv_pi_star"],
            int(fx["inv_k_pi_7"]),
        ),
        (
            "the random MDP from action 0",
            fx["rand_P"],
            fx["rand_R"],
            float(fx["rand_gamma"]),
            np.zeros(5, dtype=int),
            fx["rand_V_star"],
            fx["rand_pi_star"],
            int(fx["rand_k_pi"]),
        ),
        (
            "an MDP whose action 3 duplicates another action, from 'always 3'",
            fx["dup_P"],
            fx["dup_R"],
            float(fx["rand_gamma"]),
            fx["dup_pi0"],
            fx["dup_V_star"],
            fx["dup_pi_star"],
            int(fx["dup_k_pi"]),
        ),
    ]
    for name, P, R, gamma, pi0, V_star, pi_star, k_ref in cases:
        V, pi, k = _unpack(fn(P, R, gamma, pi0.copy()), 3, "policy_iteration", "(V, pi, k)")
        pi = np.asarray(pi)
        assert_shape(pi, pi_star.shape, what=f"pi for {name}")
        if not np.issubdtype(pi.dtype, np.integer):
            raise CheckFailed(
                f"pi for {name} should hold integer actions; its dtype is {pi.dtype}."
            )
        duplicate = name.startswith("an MDP whose action 3")
        if duplicate:
            wrong = np.flatnonzero(pi != pi_star)
            s = int(wrong[0]) if wrong.size else None
        else:
            s = _first_non_greedy(pi, R + gamma * P @ V_star)
        if s is not None:
            msg = (
                f"On {name}, your policy picks action {pi[s]} in state {s}; the optimal policy "
                f"picks {pi_star[s]}."
            )
            if duplicate:
                msg += (
                    f" Actions {pi[s]} and {pi_star[s]} are identical here, and the loop started "
                    "on an optimal action: keep the current action when it ties for the best, "
                    "and change it only where a strictly better action exists."
                )
            else:
                msg += (
                    " Improve with Q = R + gamma * P @ V of the policy you just evaluated, and "
                    "loop until the greedy policy is the one you already have."
                )
            raise CheckFailed(msg)
        assert_close(
            V,
            V_star,
            atol=1e-8,
            rtol=1e-8,
            what=f"V for {name}",
            hint="Return the exact value of the policy you return (the last evaluation).",
        )
        k_int = _int(k, "k (the number of evaluations)")
        if k_int != k_ref:
            raise CheckFailed(
                f"On {name}, the reference needs {k_ref} evaluations; you report k = {k_int}. "
                "Count one per linear solve, including the last one, whose improvement changes "
                "nothing."
            )


def check_estimate_model(fn: Callable[..., tuple]) -> None:
    fx = expected("m02_model")
    cases = [
        ("the 200-day log", fx["log"], 11, 11, fx["P_hat"], fx["R_hat"], fx["counts"]),
        (
            "the five-row log",
            fx["hand_log"],
            2,
            2,
            fx["hand_P_hat"],
            fx["hand_R_hat"],
            fx["hand_counts"],
        ),
    ]
    for name, log, S, A, P_want, R_want, c_want in cases:
        P_hat, R_hat, counts = _unpack(
            fn(log.copy(), S, A), 3, "estimate_model", "(P_hat, R_hat, counts)"
        )
        P_hat, R_hat, counts = (np.asarray(x, dtype=float) for x in (P_hat, R_hat, counts))
        assert_shape(P_hat, (S, A, S), what=f"P_hat for {name}")
        assert_shape(R_hat, (S, A), what=f"R_hat for {name}")
        assert_shape(counts, (S, A), what=f"counts for {name}")
        if not np.array_equal(counts, c_want):
            s, a = (int(i) for i in np.argwhere(counts != c_want)[0])
            raise CheckFailed(
                f"For {name}, counts[{s}, {a}] is {counts[s, a]:g}; the log has {c_want[s, a]:g} "
                f"rows with stock {s} and order {a}. Tally the (s, a) of each row, where s is the "
                "state the row starts in (its first column)."
            )
        for arr, what in ((P_hat, "P_hat"), (R_hat, "R_hat")):
            if not np.all(np.isfinite(arr)):
                s, a = (
                    int(i) for i in np.argwhere(~np.isfinite(arr).reshape(S, A, -1).all(axis=2))[0]
                )
                raise CheckFailed(
                    f"For {name}, {what}[{s}, {a}] is not finite: the pair was visited "
                    f"{c_want[s, a]:g} times. Divide only where counts[s, a] > 0."
                )
        unvisited = np.argwhere(c_want == 0)
        for s, a in unvisited:
            s, a = int(s), int(a)
            if P_hat[s, a, s] != 1.0 or R_hat[s, a] != 0.0:
                raise CheckFailed(
                    f"For {name}, (stock {s}, order {a}) is never in the log, so P_hat[{s}, {a}, :] "
                    f"should be a self-loop (1 at state {s}) with R_hat[{s}, {a}] = 0; you have "
                    f"P_hat[{s}, {a}, {s}] = {P_hat[s, a, s]:g} and R_hat = {R_hat[s, a]:g}."
                )
        sums = P_hat.sum(axis=2)
        if not np.allclose(sums, 1.0, atol=1e-12):
            s, a = (int(i) for i in np.unravel_index(np.argmax(np.abs(sums - 1)), sums.shape))
            raise CheckFailed(
                f"For {name}, P_hat[{s}, {a}, :] sums to {sums[s, a]:.4g}; every row must sum to 1. "
                f"Divide the row's tallies by counts[{s}, {a}], not by a column total."
            )
        assert_close(
            P_hat,
            P_want,
            atol=1e-12,
            rtol=0.0,
            what=f"P_hat for {name}",
            hint="P_hat[s, a, s2] is the fraction of the rows at (s, a) whose next state is s2.",
        )
        assert_close(
            R_hat,
            R_want,
            atol=1e-12,
            rtol=0.0,
            what=f"R_hat for {name}",
            hint="R_hat[s, a] is the MEAN of the rewards seen at (s, a): the sum divided by counts.",
        )


#: How far a mean eps_P at N = 2 may sit from the fixture's. A correct solution matches it
#: exactly (eps_P depends only on the samples); the max-norm version gives about half.
EPS_P_TOLERANCE = 0.05


def check_plan_and_evaluate(fn: Callable[..., tuple]) -> None:
    fx = expected("m02_plan_eval")
    mdp = InventoryMDP(**INVENTORY)
    seeds = [int(x) for x in fx["seeds"]]
    results: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for N in (int(x) for x in fx["Ns"]):
        losses, eps = [], []
        for seed in seeds:
            loss, eps_P = _unpack(fn(mdp, N, seed), 2, "plan_and_evaluate", "(loss, eps_P)")
            loss, eps_P = float(loss), float(eps_P)
            if not np.isfinite(loss) or loss < -1e-9:
                raise CheckFailed(
                    f"At N = {N}, seed {seed}: loss = {loss:.4g}. The loss max_s [V*(s) - V^pi_hat(s)] "
                    "is a finite number >= 0 when V* is the exact value of the true model's "
                    "greedy policy and V^pi_hat is evaluated in the TRUE model."
                )
            if not 0.0 <= eps_P <= 2.0:
                raise CheckFailed(
                    f"At N = {N}, seed {seed}: eps_P = {eps_P:.4g}, outside [0, 2]. eps_P is the "
                    "largest l1 distance between a row of P and the same row of P_hat."
                )
            losses.append(loss)
            eps.append(eps_P)
        results[N] = (np.array(losses), np.array(eps))
    (n_small, n_large) = (int(x) for x in fx["Ns"])
    loss_s, eps_s = results[n_small]
    loss_l, eps_l = results[n_large]
    if np.ptp(loss_s) == 0.0:
        raise CheckFailed(
            f"All five seeds at N = {n_small} give the same loss ({loss_s[0]:.4g}). Five seeds "
            "should sample five different models: pass `seed` to np.random.default_rng, plan in "
            "the ESTIMATE, and evaluate that plan in the truth (mdp.P, mdp.R)."
        )
    spread = float(loss_l.max() - loss_l.min())
    gap = float(loss_s.mean() - loss_l.mean())
    if not gap > spread:
        raise CheckFailed(
            f"Mean loss at N = {n_small} is {loss_s.mean():.3g} and at N = {n_large} is "
            f"{loss_l.mean():.3g}: the difference ({gap:.3g}) should exceed the spread of the five "
            f"N = {n_large} losses ({spread:.3g}). With {n_small} samples per pair the plan should "
            f"lose tens of units in the true model; with {n_large} it should lose almost nothing."
        )
    if not eps_l.mean() < eps_s.mean():
        raise CheckFailed(
            f"Mean eps_P is {eps_s.mean():.3g} at N = {n_small} and {eps_l.mean():.3g} at "
            f"N = {n_large}; more data should shrink the model error. Compute eps_P from the "
            "P_hat you planned with."
        )
    ref = float(fx["eps_P"][0].mean())
    if abs(eps_s.mean() - ref) > EPS_P_TOLERANCE:
        raise CheckFailed(
            f"Mean eps_P over seeds {seeds[0]}-{seeds[-1]} at N = {n_small} is {eps_s.mean():.3f}; "
            f"the reference gives {ref:.3f}. eps_P is the max over (s, a) of the l1 norm of a row "
            "difference, np.abs(mdp.P - P_hat).sum(axis=2).max(), not the largest single entry."
        )
    for i, N in enumerate((n_small, n_large)):
        got_loss, got_eps = results[N]
        assert_close(
            got_eps,
            fx["eps_P"][i],
            atol=1e-9,
            rtol=0.0,
            what=f"eps_P for seeds {seeds[0]}-{seeds[-1]} at N = {N}",
            hint="Draw the log with the provided sample_transitions(mdp, N, rng), "
            "rng = np.random.default_rng(seed), and estimate the model from all of it.",
        )
        assert_close(
            got_loss,
            fx["loss"][i],
            atol=1e-6,
            rtol=0.0,
            what=f"the loss for seeds {seeds[0]}-{seeds[-1]} at N = {N}",
            hint="V* is the exact value of the TRUE model's greedy policy; V^pi_hat is the exact "
            "value of the estimate's greedy policy, also in the TRUE model.",
        )


def check_value_iteration_suspect(fn: Callable[..., tuple]) -> None:
    fx = expected("m02_plan")
    inv = InventoryMDP(**INVENTORY)

    cliff = CliffGridworld(4, 12, slip=0.2, gamma=0.95)
    P_l, R_l, g_l = _loud_mdp(int(fx["loud_S"]), float(fx["loud_gamma"]))
    cases = [
        ("the shop", inv.P, inv.R, 0.95, fx["inv_V_star"], int(fx["inv_k_vi"])),
        (
            "the 4x12 cliff with slip 0.2",
            cliff.P,
            cliff.R,
            0.95,
            fx["cliff_V_star"],
            int(fx["cliff_k_vi"]),
        ),
        (
            "the mostly settled MDP (100 absorbing states, reward only in state 0)",
            P_l,
            R_l,
            g_l,
            fx["loud_V_star"],
            int(fx["loud_k_vi"]),
        ),
    ]
    for name, P, R, gamma, V_star, k_ref in cases:
        V, Q, k, log = _unpack(fn(P, R, gamma, TOL), 4, "value_iteration_suspect", "(V, Q, k, log)")
        k_int = _int(k, "k (the number of sweeps)")
        log = np.asarray(log, dtype=float)
        if log.shape != (k_int,):
            raise CheckFailed(
                f"On {name}, the log has {log.shape} entries for k = {k_int} sweeps; log one "
                "max|V_new - V| per sweep."
            )
        if not log[-1] < TOL:
            raise CheckFailed(
                f"On {name}, the suspect stopped after {k_int} sweeps with its last logged "
                f"max|V_new - V| = {log[-1]:.3g}, which is not below tol = {TOL:g}: the loop stops "
                "on a different statistic than the one it logs. The stopping rule needs the max."
            )
        _check_vi_result(
            name, P, R, gamma, V, Q, k, V_star, k_ref, fn_name="value_iteration_suspect"
        )


def check_modified_policy_iteration(fn: Callable[..., tuple]) -> None:
    """Stretch: on the shop, m = 1 is value iteration (299 rounds) and m = 5 takes 67 rounds."""
    fx = expected("m02_plan")
    inv = InventoryMDP(**INVENTORY)
    for m, k_ref, v_tol in ((1, int(fx["inv_k_mpi_1"]), None), (5, int(fx["inv_k_mpi_5"]), 1e-3)):
        V, pi, rounds = _unpack(
            fn(inv.P, inv.R, 0.95, m, TOL), 3, "modified_policy_iteration", "(V, pi, rounds)"
        )
        pi = np.asarray(pi)
        assert_shape(pi, fx["inv_pi_star"].shape, what=f"pi with m = {m}")
        s = _first_non_greedy(pi, fx["inv_Q_star"])
        assert_true(
            s is None,
            f"With m = {m}, the final greedy policy {pi.tolist()} is not optimal (state {s}); "
            f"pi* is {fx['inv_pi_star'].tolist()}. Return greedy(R + gamma * P @ V) of the final V.",
        )
        r = _int(rounds, "rounds")
        if abs(r - k_ref) > 1:
            raise CheckFailed(
                f"With m = {m} on the shop at tol = {TOL:g}, the reference takes {k_ref} rounds; "
                f"you report {r}. Each round is one optimality sweep plus m - 1 sweeps of T^pi "
                "for the current greedy policy; with m = 1 the count equals value iteration's."
            )
        if v_tol is not None:
            assert_close(
                V, fx["inv_V_star"], atol=v_tol, rtol=0.0, what=f"V with m = {m}", hint=None
            )
