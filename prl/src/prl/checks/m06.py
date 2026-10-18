"""Checkpoints for Module 6 (exploration in MDPs).

Exact checks compare the participant's function with fixtures written by
scripts/make_expected.py (fixed chains and policies, the exact hitting times and Q* of the
lab's river, fixed models with untried pairs, fixed trajectories for regret). Stochastic
checks are StochasticCheck declarations: a metric of the participant's function at a budget,
with thresholds set by scripts/threshold_protocol.py.

No learning algorithm lives here. The metrics run the participant's (or the lab's
reference) learner on a fresh river, record the states it visited, and measure where it
went; the learning, the planning and the bonus are the participant's code.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np

from .. import envs
from . import CheckFailed, StochasticCheck, assert_close, assert_simplex, expected

#: The lab's river (Exercises 3-7): 10 states, discount 0.95, rewards in [0, 1].
N_CHAIN = 10
S, A = N_CHAIN, 2
END = N_CHAIN - 1  # the right end
GAMMA = 0.95
R_MAX = 1.0
EPSILON = ALPHA = 0.1  # Lab 3's exploration rate and step size
C_BONUS = 1.0

#: A participant's code that raises one of these gets a CheckFailed with a hint, not a crash
#: (verify mode does not count a crash as a rejection).
_CRASHES = (np.linalg.LinAlgError, ValueError, IndexError)


# ------------------------------------------------------------------ helpers
def _call(fn: Callable[..., Any], *args: Any, what: str, hint: str = "") -> Any:
    try:
        return fn(*args)
    except _CRASHES as exc:
        raise CheckFailed(
            f"{what} raised {type(exc).__name__}: {exc}." + (f" Hint: {hint}" if hint else "")
        ) from None


def _array(out: Any, shape: tuple[int, ...], what: str, hint: str = "") -> np.ndarray:
    if out is None:
        raise CheckFailed(f"{what} returned None. Did you forget `return`?")
    try:
        arr = np.asarray(out, dtype=float)
    except (TypeError, ValueError):
        raise CheckFailed(f"{what} should return an array of shape {shape}.") from None
    if arr.shape != tuple(shape):
        raise CheckFailed(
            f"{what} should return shape {tuple(shape)}; got {arr.shape}."
            + (f" Hint: {hint}" if hint else "")
        )
    if not np.all(np.isfinite(arr)):
        raise CheckFailed(f"{what} contains NaN or infinity.")
    return arr


class _Recorder(gym.Wrapper):
    """The lab's Recorder: a reset starts a new trajectory; each step adds the next state."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.states: list[int] = []

    def reset(self, **kwargs: Any):
        s, info = self.env.reset(**kwargs)
        self.states = [int(s)]
        return s, info

    def step(self, action: Any):
        out = self.env.step(action)
        self.states.append(int(out[0]))
        return out


def _river(steps: int) -> gym.Env:
    """The lab's make_river(10, steps): one episode of `steps` steps from the left end."""
    return envs.TabularEnv(envs.RiverSwim(N_CHAIN, gamma=GAMMA), max_episode_steps=steps)


def _q_learning_run(q_learning: Callable[..., Any], q0: np.ndarray, seed: int, steps: int):
    """One episode of the notebook's q_learning (Lab 3's reference) on a recorded river."""
    rec = _Recorder(_river(steps))
    q_learning(rec, 1, ALPHA, EPSILON, GAMMA, seed, q0=q0)
    if len(rec.states) != steps + 1:
        raise CheckFailed(
            f"q_learning ran {len(rec.states) - 1} steps; one episode of the river lasts "
            f"{steps}. The checkpoint calls q_learning(env, 1, {ALPHA}, {EPSILON}, {GAMMA}, "
            "seed, q0=...)."
        )
    return rec.states


# ------------------------------------------------------------------ 1 the policy chain (exact)
def check_policy_transition(fn: Callable[..., Any]) -> None:
    fx = expected("m06_chain")
    for i in range(int(fx["pt_n_cases"])):
        P, pi, want = fx[f"pt_P_{i}"], fx[f"pt_pi_{i}"], fx[f"pt_want_{i}"]
        n_s, n_a = pi.shape
        what = f"policy_transition(P, pi) on {fx[f'pt_label_{i}']}"
        out = _call(fn, P.copy(), pi.copy(), what=what, hint="P has shape (S, A, S), pi (S, A).")
        P_pi = _array(
            out,
            (n_s, n_s),
            what,
            hint="sum over the action index a, the middle axis of P, weighted by pi[s, a].",
        )
        if np.allclose(P_pi, P.mean(axis=1), atol=1e-12) and not np.allclose(P_pi, want):
            raise CheckFailed(
                f"{what}: your P_pi is the plain average of P over actions, which equals the "
                "answer only under uniform random play. Weight each row P[s, a, :] by pi[s, a]."
            )
        if np.allclose(P_pi.T, want, atol=1e-12) and not np.allclose(P_pi, want):
            raise CheckFailed(
                f"{what}: your P_pi is transposed. Row s should be where s goes in one step: "
                "P_pi[s, s'] = sum over a of pi[s, a] P[s, a, s']."
            )
        assert_simplex(P_pi, axis=1, atol=1e-9, what=f"{what}: the rows of P_pi")
        assert_close(
            P_pi,
            want,
            atol=1e-12,
            rtol=0,
            what=what,
            hint="np.einsum('sa,sat->st', pi, P).",
        )


# ------------------------------------------------------------------ 2 hitting times (exact)
_HIT_HINT = (
    "M = P_pi[np.ix_(others, others)] keeps the rows AND columns of the other states; "
    "solve (I - M) x = 1 and put x into h[others]."
)


def check_hitting_times(fn: Callable[..., Any]) -> None:
    fx = expected("m06_chain")
    got: dict[int, np.ndarray] = {}
    for i in range(int(fx["hit_n_cases"])):
        P_pi, target, want = fx[f"hit_P_{i}"], int(fx[f"hit_target_{i}"]), fx[f"hit_h_{i}"]
        n = len(P_pi)
        what = f"hitting_times(P_pi, {target}) on {fx[f'hit_label_{i}']}"
        h = _array(_call(fn, P_pi.copy(), target, what=what, hint=_HIT_HINT), (n,), what)
        if abs(h[target]) > 1e-12:
            raise CheckFailed(
                f"{what}: h[{target}] = {h[target]:.6g}. At the target nothing is left to "
                "spend, so h[target] = 0; it is not one of the unknowns."
            )
        if np.any(h < -1e-9):
            raise CheckFailed(
                f"{what}: your h has negative entries ({np.round(h, 4).tolist()}), but an "
                "expected number of steps is at least 0. P_pi[others, others] pairs the two "
                "lists and returns the diagonal entries as a vector, which np.eye(n - 1) - M "
                "then broadcasts into a matrix. " + _HIT_HINT
            )
        assert_close(h, want, atol=1e-9, rtol=1e-9, what=what, hint=_HIT_HINT)
        got[i] = h
    # The growth ratio of (crossing): each extra state multiplies h(0) by about 3.
    i11, i12 = (int(k) for k in fx["ratio_cases"])
    ratio = float(got[i12][0] / got[i11][0])
    if abs(ratio - float(fx["ratio_12_11"])) > 1e-6:
        raise CheckFailed(
            f"Your h(0) grows by {ratio:.6f} from 11 to 12 states; the solve gives "
            f"{float(fx['ratio_12_11']):.6f}."
        )
    # The linear solve against simulated walks (the fixture's walks, with a fixed seed).
    i5 = int(fx["sim_case"])
    h0, mean, se = float(got[i5][0]), float(fx["sim_mean"]), float(fx["sim_se"])
    if abs(h0 - mean) > 4 * se:
        raise CheckFailed(
            f"On RiverSwim(5) your h(0) = {h0:.2f}, but {int(fx['sim_walks'])} simulated walks "
            f"take {mean:.2f} steps on average (standard error {se:.2f})."
        )


# ------------------------------------------------------------------ 3.1 visitation (exact)
_VISITS: list[tuple[str, Any, int, list[float]]] = [
    ("knowledge check 6.5's list", [0, 0, 1, 0, 1, 2, 2, 2], 4, [0.375, 0.25, 0.375, 0.0]),
    ("the same states as a NumPy array", np.array([0, 0, 1, 0, 1, 2, 2, 2]), 4, []),
    ("a run that never leaves states 0 and 1", [0] * 7 + [1, 0, 1] * 3, 10, []),
    ("a single step in state 3", [3], 5, [0.0, 0.0, 0.0, 1.0, 0.0]),
    ("states given as NumPy integers", [np.int64(2), np.int64(2), np.int64(0)], 3, []),
]


def check_visitation(fn: Callable[..., Any]) -> None:
    for label, states, n, want in _VISITS:
        seq = list(states) if isinstance(states, list) else np.array(states)
        k = len(seq)
        want_arr = np.asarray(want, dtype=float) if want else None
        if want_arr is None:
            counts = np.zeros(n)
            for s in seq:
                counts[int(s)] += 1
            want_arr = counts / k
        what = f"visitation(states, {n}) on {label}"
        out = _call(fn, seq, n, what=what, hint="np.bincount(states, minlength=n_states).")
        if out is not None and np.ndim(out) == 1 and 0 < len(out) < n:
            raise CheckFailed(
                f"{what} returned {len(out)} entries, not {n}: np.bincount stops at the "
                "largest state that occurs, so the states never reached are missing from the "
                "end. Pass minlength=n_states."
            )
        v = _array(out, (n,), what)
        if np.allclose(v, want_arr * k, atol=1e-9) and k > 1:
            raise CheckFailed(
                f"{what}: your result sums to {v.sum():g}, the number of steps: those are "
                "counts. Divide by the number of entries."
            )
        if np.allclose(v, want_arr[::-1], atol=1e-12) and not np.allclose(v, want_arr):
            raise CheckFailed(
                f"{what}: your fractions are in reverse order. Entry s is the fraction of "
                "steps in state s, with state 0 the left end."
            )
        assert_close(
            v,
            want_arr,
            atol=1e-12,
            rtol=0,
            what=what,
            hint="count each state 0..n_states-1, then divide by the number of entries.",
        )


# ------------------------------------------------------------------ 3.2 where eps-greedy goes
def _eps_greedy_end(
    visitation: Callable[..., Any], q_learning: Callable[..., Any], *, seed: int, steps: int
) -> float:
    states = _q_learning_run(q_learning, np.zeros((S, A)), seed, steps)
    what = f"visitation(states, {S}) on {steps} steps of eps-greedy Q-learning"
    v = _array(visitation(states[:-1], S), (S,), what)
    if np.any(v < 0) or abs(v.sum() - 1.0) > 1e-9:
        raise CheckFailed(f"{what} should be fractions that sum to 1; they sum to {v.sum():g}.")
    return float(v[END])


check_eps_greedy_end = StochasticCheck(
    ex=3,
    functions=("visitation", "q_learning"),
    metric=_eps_greedy_end,
    live={"steps": 5000},
    quick={"steps": 5000},
    higher_is_better=False,
    min_gap=0.2,
    what=(
        f"fraction of eps-greedy Q-learning's steps at the right end (state {END}), "
        "from your visitation"
    ),
    thresholds={
        # threshold_protocol.py, live budget {"steps": 5000}, N = 100 seeds (0-99), 2026-10-18,
        # local (Apple M1 Pro, arm64). Solution p99 0; strongest wrong version mutant 'random'
        # p1 0.4264; gap 0.4264 (minimum 0.2); threshold: the midpoint 0.213221, rounded to
        # 0.21. 0 of 100 solution seeds fail: flake rate at most about 3/N = 3.0% (rule of
        # three).
        "live": (
            0.21,
            "threshold protocol, 100 seeds: runs/2026-10-18-local-threshold-m06-check_eps_greedy_end-live-n100-exp.json",
        ),
        # threshold_protocol.py, quick budget {"steps": 5000}, N = 100 seeds (0-99), 2026-10-18,
        # local (Apple M1 Pro, arm64). Solution p99 0; strongest wrong version mutant 'random'
        # p1 0.4264; gap 0.4264 (minimum 0.2); threshold: the midpoint 0.213221, rounded to
        # 0.21. 0 of 100 solution seeds fail: flake rate at most about 3/N = 3.0% (rule of
        # three).
        "quick": (
            0.21,
            "threshold protocol, 100 seeds: runs/2026-10-18-local-threshold-m06-check_eps_greedy_end-quick-n100-exp.json",
        ),
    },
)


# ------------------------------------------------------------------ 4.1 the optimistic start
def check_optimistic_q0(fn: Callable[..., Any]) -> None:
    for r_max, gamma, n_s, n_a, want in (
        (1.0, 0.95, 10, 2, 20.0),
        (2.0, 0.9, 3, 4, 20.0),
        (5.0, 0.8, 2, 3, 25.0),
    ):
        what = f"optimistic_q0({r_max:g}, {gamma:g}, {n_s}, {n_a})"
        Q0 = _array(_call(fn, r_max, gamma, n_s, n_a, what=what), (n_s, n_a), what)
        if np.allclose(Q0, r_max):
            hint = (
                "that is one step's largest reward. The start must bound every discounted "
                "return: r_max + gamma r_max + gamma^2 r_max + ... = r_max / (1 - gamma)."
            )
        elif np.allclose(Q0, r_max * (1 - gamma)):
            hint = "divide r_max by 1 - gamma; do not multiply."
        elif np.allclose(Q0, 0.0):
            hint = "a start of 0 is not optimistic. Use r_max / (1 - gamma)."
        else:
            hint = "the sum of the geometric series r_max (1 + gamma + gamma^2 + ...), everywhere."
        assert_close(Q0, np.full((n_s, n_a), want), atol=1e-9, rtol=1e-9, what=what, hint=hint)
    Q_star = expected("m06_q0")["Q_star"]
    Q0 = np.asarray(fn(R_MAX, GAMMA, S, A), dtype=float)
    if not np.all(Q0 >= Q_star - 1e-9):
        s, a = np.unravel_index(np.argmax(Q_star - Q0), Q_star.shape)
        raise CheckFailed(
            f"optimistic_q0({R_MAX:g}, {GAMMA:g}, {S}, {A}) is below Q* of the river at "
            f"state {s}, action {a}: {Q0[s, a]:.4g} < {Q_star[s, a]:.4g}."
        )


# ------------------------------------------------------------------ 4.2 the right end reached
def _optimistic_reach(
    optimistic_q0: Callable[..., Any],
    q_learning: Callable[..., Any],
    *,
    seed: int,
    steps: int,
    runs: int,
) -> float:
    what = f"optimistic_q0({R_MAX:g}, {GAMMA:g}, {S}, {A})"
    q0 = _array(optimistic_q0(R_MAX, GAMMA, S, A), (S, A), what)
    reached = 0
    for i in range(runs):
        states = _q_learning_run(q_learning, q0.copy(), seed * runs + i, steps)
        reached += END in states
    return float(reached)


check_optimistic_reach = StochasticCheck(
    ex=4,
    functions=("optimistic_q0", "q_learning"),
    metric=_optimistic_reach,
    live={"steps": 5000, "runs": 10},
    quick={"steps": 5000, "runs": 10},
    higher_is_better=True,
    min_gap=4.0,
    what="runs of Q-learning from your start that reach the right end (of 10)",
    thresholds={
        # threshold_protocol.py, live budget {"runs": 10, "steps": 5000}, N = 100 seeds (0-99),
        # 2026-10-18, local (Apple M1 Pro, arm64). Solution p1 10; strongest wrong version
        # mutant 'random' p99 5.01; gap 4.99 (minimum 4); threshold: the midpoint 7.505, rounded
        # to 7.5. 0 of 100 solution seeds fail: flake rate at most about 3/N = 3.0% (rule of
        # three).
        "live": (
            7.5,
            "threshold protocol, 100 seeds: runs/2026-10-18-local-threshold-m06-check_optimistic_reach-live-n100-exp.json",
        ),
        # threshold_protocol.py, quick budget {"runs": 10, "steps": 5000}, N = 100 seeds (0-99),
        # 2026-10-18, local (Apple M1 Pro, arm64). Solution p1 10; strongest wrong version
        # mutant 'random' p99 5.01; gap 4.99 (minimum 4); threshold: the midpoint 7.505, rounded
        # to 7.5. 0 of 100 solution seeds fail: flake rate at most about 3/N = 3.0% (rule of
        # three).
        "quick": (
            7.5,
            "threshold protocol, 100 seeds: runs/2026-10-18-local-threshold-m06-check_optimistic_reach-quick-n100-exp.json",
        ),
    },
)


# ------------------------------------------------------------------ 5 the MBIE-EB backup (exact)
def check_mbie_eb_backup(fn: Callable[..., Any]) -> None:
    fx = expected("m06_mbie")
    for i in range(int(fx["n_cases"])):
        P, R, N = fx[f"P_{i}"], fx[f"R_{i}"], fx[f"N_{i}"]
        gamma, c = float(fx[f"gamma_{i}"]), float(fx[f"c_{i}"])
        want = fx[f"Q_{i}"]
        what = f"mbie_eb_backup(P_hat, R_hat, counts, {gamma:g}, {c:g}, tol=1e-10) on {fx[f'label_{i}']}"
        out = _call(
            fn,
            P.copy(),
            R.copy(),
            N.copy(),
            gamma,
            c,
            1e-10,
            what=what,
            hint="value_iteration(P_hat, R_hat + bonus, gamma, tol) returns (V, Q, sweeps).",
        )
        Q = _array(
            out,
            R.shape,
            what,
            hint="return the Q from value_iteration, one value per (state, action), not V.",
        )
        if np.allclose(Q, fx[f"after_{i}"], atol=1e-6) and not np.allclose(Q, want, atol=1e-6):
            hint = (
                "your Q equals the plain model's Q plus the bonus: the bonus was added after "
                "solving, so it never travels to the states that lead to it. Put it in the "
                "reward that value iteration backs up: value_iteration(P_hat, R_hat + bonus, ...)."
            )
        elif np.allclose(Q, fx[f"no_bonus_{i}"], atol=1e-6):
            hint = (
                "your Q gives a pair never tried no bonus. A pair with count 0 gets the bonus "
                "of a pair tried once: c / sqrt(max(N, 1))."
            )
        else:
            hint = (
                "the bonus is c / sqrt(max(N, 1)), added to R_hat inside the backup; pass the "
                "checkpoint's tol to value_iteration (not the lab's TOL)."
            )
        assert_close(Q, want, atol=1e-6, rtol=0, what=what, hint=hint)


# ------------------------------------------------------------------ 6 regret (exact)
_REGRET_HINTS = (
    ("per_step", "these are the per-step regrets. Reg_t is their running sum: np.cumsum."),
    (
        "next",
        "the reward of step t comes from s_t and a_t. states holds one more entry than "
        "actions (s_T has no action): use states[:-1].",
    ),
    (
        "best",
        "you compared with the best action in the state the learner was in. Regret compares "
        "with the optimal policy's expected reward at each step from the start: opt_rewards.",
    ),
    ("average", "that is the running average. Reg_t is the running sum, not divided by t."),
    (
        "clipped",
        "a step can have negative regret (the learner beat the optimal policy's expectation "
        "at that step); keep it in the sum.",
    ),
)


def check_regret(fn: Callable[..., Any]) -> None:
    fx = expected("m06_regret")
    for i in range(int(fx["n_cases"])):
        states, actions = fx[f"states_{i}"], fx[f"actions_{i}"]
        R, opt, want = fx[f"R_{i}"], fx[f"opt_{i}"], fx[f"want_{i}"]
        T = len(actions)
        args = (states.tolist(), actions.tolist()) if T == 1 else (states.copy(), actions.copy())
        what = f"regret(states, actions, R, opt_rewards) on {fx[f'label_{i}']} (T = {T})"
        out = _call(fn, *args, R.copy(), opt.copy(), what=what)
        if out is not None and np.ndim(out) == 0:
            raise CheckFailed(
                f"{what} returned one number, {float(out):.6g}. Return Reg_t after every "
                f"step: {T} entries, the last of which is Reg_T."
            )
        got = _array(out, (T,), what, hint=f"one entry per step: {T} entries, Reg_1..Reg_T.")
        hint = "Reg_t = sum over k < t of (opt_rewards[k] - R[s_k, a_k]): np.cumsum."
        if not np.allclose(got, want, atol=1e-9):
            if np.allclose(got, -want, atol=1e-9):
                hint = "your signs are reversed: regret is the optimum minus what the learner got."
            for key, words in _REGRET_HINTS:
                if np.allclose(got, fx[f"{key}_{i}"], atol=1e-9):
                    hint = words
                    break
        assert_close(
            got, want, atol=1e-12 if not want.any() else 1e-9, rtol=0, what=what, hint=hint
        )


# ------------------------------------------------------------------ 7 the planted bug
def _suspect_end(mbie_eb_suspect: Callable[..., Any], *, seed: int, steps: int, runs: int) -> float:
    fractions = []
    for i in range(runs):
        out = mbie_eb_suspect(_river(steps), steps, C_BONUS, GAMMA, seed * runs + i)
        if not isinstance(out, (tuple, list)) or len(out) != 3:
            raise CheckFailed("mbie_eb_suspect should return (states, actions, Q).")
        states, actions = np.asarray(out[0]), np.asarray(out[1])
        if states.shape != (steps + 1,) or actions.shape != (steps,):
            raise CheckFailed(
                f"mbie_eb_suspect(env, {steps}, ...) should return {steps + 1} states "
                f"(s_0..s_T) and {steps} actions; got {states.shape} and {actions.shape}."
            )
        fractions.append(float(np.mean(states[:-1] == END)))
    return float(np.mean(fractions))


@dataclass(frozen=True, eq=False)
class _HintedCheck(StochasticCheck):
    """A StochasticCheck whose threshold failure ends with a hint for the participant."""

    hint: str = ""

    def __call__(self, *fns: Any) -> None:
        try:
            super().__call__(*fns)
        except CheckFailed as exc:
            if not str(exc).startswith(f"{self.what} ="):
                raise  # a wrong shape, say: its own message says what to fix
            raise CheckFailed(f"{exc} Hint: {self.hint}") from None


check_mbie_eb_suspect = _HintedCheck(
    ex=7,
    functions="mbie_eb_suspect",
    metric=_suspect_end,
    live={"steps": 5000, "runs": 3},
    quick={"steps": 5000, "runs": 3},
    higher_is_better=True,
    min_gap=0.3,
    what=f"fraction of your agent's steps at the right end (state {END}), mean of 3 runs",
    hint=(
        "if your agent no longer adds the bonus when it picks an action, check that the bonus "
        "moved into the plan: an agent with no bonus anywhere never explores. The bonus must go "
        "into the reward that value iteration backs up (mbie_eb_backup), not vanish."
    ),
    thresholds={
        # threshold_protocol.py, live budget {"runs": 3, "steps": 5000}, N = 100 seeds (0-99),
        # 2026-10-18, local (Apple M1 Pro, arm64). Solution p1 0.8077; strongest wrong version
        # mutant 'random' p99 0.0004673; gap 0.8073 (minimum 0.3); threshold: the midpoint
        # 0.404094, rounded to 0.4. 0 of 100 solution seeds fail: flake rate at most about 3/N =
        # 3.0% (rule of three).
        "live": (
            0.4,
            "threshold protocol, 100 seeds: runs/2026-10-18-local-threshold-m06-check_mbie_eb_suspect-live-n100-exp.json",
        ),
        # threshold_protocol.py, quick budget {"runs": 3, "steps": 5000}, N = 100 seeds (0-99),
        # 2026-10-18, local (Apple M1 Pro, arm64). Solution p1 0.8077; strongest wrong version
        # mutant 'random' p99 0.0004673; gap 0.8073 (minimum 0.3); threshold: the midpoint
        # 0.404094, rounded to 0.4. 0 of 100 solution seeds fail: flake rate at most about 3/N =
        # 3.0% (rule of three).
        "quick": (
            0.4,
            "threshold protocol, 100 seeds: runs/2026-10-18-local-threshold-m06-check_mbie_eb_suspect-quick-n100-exp.json",
        ),
    },
)


# ------------------------------------------------------------------ stretch: PSRL's sample
def check_psrl_sample(fn: Callable[..., Any]) -> None:
    """Moments of many draws with a fixed generator: means within 4.5 standard errors,
    the reward variance within 15%. A correct sampler that draws in another order fails
    with probability below about 2e-4 (24 means at 4.5 SE; 6 variances at about 6.7 SE)."""
    fx = expected("m06_psrl")
    P_hat, R_hat, counts = fx["P_hat"], fx["R_hat"], fx["counts"]
    n_s, n_a = counts.shape
    k = int(fx["n_draws"])
    rng = np.random.default_rng(0)
    Ps, Rs = np.empty((k, n_s, n_a, n_s)), np.empty((k, n_s, n_a))
    what = "psrl_sample(P_hat, R_hat, counts, rng)"
    for j in range(k):
        out = _call(fn, P_hat.copy(), R_hat.copy(), counts.copy(), rng, what=what)
        if not isinstance(out, (tuple, list)) or len(out) != 2:
            raise CheckFailed(f"{what} should return a pair (P_tilde, R_tilde).")
        Ps[j] = _array(out[0], (n_s, n_a, n_s), f"{what}: P_tilde")
        Rs[j] = _array(out[1], (n_s, n_a), f"{what}: R_tilde")
        if j == 0:
            assert_simplex(Ps[0], axis=2, atol=1e-9, what=f"{what}: the rows of P_tilde")
    assert_simplex(Ps, axis=3, atol=1e-9, what=f"{what}: the rows of P_tilde")
    if np.any(Rs.std(axis=0) == 0) or np.any(Ps.std(axis=0) == 0):
        raise CheckFailed(
            f"{what} returns the same model on every call: return a draw from the posterior "
            "(rng.dirichlet, rng.normal), not its mean."
        )
    var = Rs.var(axis=0, ddof=1)
    worst = np.unravel_index(np.argmax(np.abs(var / fx["R_var"] - 1)), var.shape)
    if abs(var[worst] / fx["R_var"][worst] - 1) > 0.15:
        raise CheckFailed(
            f"Over {k} draws, the variance of R_tilde{list(worst)} is {var[worst]:.4f}; the "
            f"posterior's is 1/(N + 1) = {fx['R_var'][worst]:.4f}. The standard deviation is "
            "1 / sqrt(N + 1)."
        )
    for name, draws, mean in (("R_tilde", Rs, fx["R_mean"]), ("P_tilde", Ps, fx["P_mean"])):
        se = draws.std(axis=0, ddof=1) / np.sqrt(k)
        z = np.abs(draws.mean(axis=0) - mean) / se
        if np.any(z > 4.5):
            idx = np.unravel_index(np.argmax(z), z.shape)
            raise CheckFailed(
                f"Over {k} draws, the mean of {name}{list(idx)} is "
                f"{draws.mean(axis=0)[idx]:.4f} (standard error {se[idx]:.4f}); the posterior's "
                f"is {mean[idx]:.4f}. Dirichlet parameters 1/S + N(s, a, s'); reward mean "
                "(1 + N r_hat) / (N + 1)."
            )


__all__ = [
    "check_eps_greedy_end",
    "check_hitting_times",
    "check_mbie_eb_backup",
    "check_mbie_eb_suspect",
    "check_optimistic_q0",
    "check_optimistic_reach",
    "check_policy_transition",
    "check_psrl_sample",
    "check_regret",
    "check_visitation",
]
