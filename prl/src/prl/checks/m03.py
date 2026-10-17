"""Checkpoints for Module 3 (model-free prediction and control).

Exact checks compare the participant's function with fixtures written by
scripts/make_expected.py (fixed episodes, a fixed trajectory, and the exact V^pi of the
lab's policy). Stochastic checks are StochasticCheck declarations: a metric of the
participant's function at a budget, with thresholds set by scripts/threshold_protocol.py.

No learning algorithm lives here. The metrics collect episodes under a fixed policy, call
the participant's function, follow the greedy path of the Q it returns, and measure an
error; the learning is the participant's code.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import cache, lru_cache
from typing import Any, NamedTuple

import gymnasium as gym
import numpy as np

from .. import envs
from ..envs import CliffGridworld
from . import CheckFailed, StochasticCheck, assert_close, expected

#: The lab's cliff: 4 x 12, discounted (the lab builds the same with gym.make).
GRID = dict(rows=4, cols=12, gamma=0.95)
GAMMA = 0.95
EPSILON = 0.1
S, A = 48, 4
START, GOAL = 36, 47
ABOVE_GOAL = 35
DOWN = 2
OPTIMAL_MOVES = 13  # the edge route
V_STAR_START = -(1 - GAMMA**OPTIMAL_MOVES) / (1 - GAMMA)
Q0 = 10.0  # the optimistic start of the planted-bug checkpoint


class Episode(NamedTuple):
    """The lab's episode record (same field names)."""

    states: list  # s_0, ..., s_T
    actions: list  # a_0, ..., a_{T-1}
    rewards: list  # r_0, ..., r_{T-1}
    terminated: bool
    truncated: bool


# ------------------------------------------------------------------ helpers
def _episodes(fx: dict[str, np.ndarray], prefix: str) -> list[Episode]:
    out = []
    for i, T in enumerate(fx[f"{prefix}_lengths"]):
        states = [int(s) for s in fx[f"{prefix}_states"][i, : T + 1]]
        rewards = [float(r) for r in fx[f"{prefix}_rewards"][i, :T]]
        out.append(
            Episode(
                states,
                [0] * int(T),  # the actions play no part in prediction
                rewards,
                bool(fx[f"{prefix}_terminated"][i]),
                bool(fx[f"{prefix}_truncated"][i]),
            )
        )
    return out


def _vector(out: Any, n: int, what: str) -> np.ndarray:
    if out is None:
        raise CheckFailed(f"{what} returned None. Did you forget `return V`?")
    try:
        V = np.asarray(out, dtype=float)
    except (TypeError, ValueError):
        raise CheckFailed(f"{what} should return an array of {n} values.") from None
    if V.shape != (n,):
        raise CheckFailed(f"{what} should return shape ({n},), one value per state; got {V.shape}.")
    if not np.all(np.isfinite(V)):
        raise CheckFailed(f"{what} contains NaN or infinity.")
    return V


def _describe(ep: Episode) -> str:
    end = "terminated" if ep.terminated else "truncated"
    return f"states {ep.states}, rewards {ep.rewards}, {end}"


def _make_env() -> gym.Env:
    envs.register()
    return gym.make("prl/CliffGridworld-v0", **GRID)


@cache
def _mdp() -> CliffGridworld:
    return CliffGridworld(**GRID)


# ------------------------------------------------------------------ 1.1 first-visit MC (exact)
def check_mc_prediction(fn: Callable[..., Any]) -> None:
    fx = expected("m03_prediction")
    cases = [(1, 0.5, fx["mc_one_V"], fx["mc_one_every_V"])]  # knowledge check Q3.1's episode
    cases += [
        (len(fx["mc_lengths"]), float(g), first, every)
        for g, first, every in zip(fx["mc_gammas"], fx["mc_first_V"], fx["mc_every_V"], strict=True)
    ]
    for k, gamma, want, every in cases:
        episodes = _episodes(fx, "mc")[:k]  # fresh lists for every call
        what = f"mc_prediction on {k} episode(s), gamma={gamma:g}"
        V = _vector(fn(episodes, gamma, 5), 5, what)
        if np.allclose(V, every, atol=1e-9) and not np.allclose(V, want, atol=1e-9):
            raise CheckFailed(
                f"{what}: your V matches every-visit MC, {np.round(V, 4).tolist()}. The episode "
                f"({_describe(episodes[0])}) visits state 0 twice; first-visit MC averages only "
                "the return after the first visit to a state in each episode."
            )
        assert_close(
            V,
            want,
            atol=1e-9,
            what=what,
            hint="Per episode: returns backward with G = r_t + gamma G; then, for each state's "
            "first visit, add that return to the state's running mean. Never update s_T.",
        )


def _route_rms(V: np.ndarray) -> float:
    fx = expected("m03_prediction")
    route = fx["route"]
    return float(np.sqrt(np.mean((V[route] - fx["V_pi"][route]) ** 2)))


@lru_cache(maxsize=128)
def _route_episodes(seed: int, n: int) -> tuple[tuple, ...]:
    """n episodes of the epsilon-soft safe route, as the lab's `collect(env, PI, n, seed)`."""
    pi = expected("m03_prediction")["pi"]
    env = _make_env()
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        s, _ = env.reset()
        states, actions, rewards = [int(s)], [], []
        while True:
            a = int(rng.choice(A, p=pi[s]))
            s, r, terminated, truncated, _ = env.step(a)
            actions.append(a)
            rewards.append(float(r))
            states.append(int(s))
            if terminated or truncated:
                out.append((tuple(states), tuple(actions), tuple(rewards), terminated, truncated))
                break
    return tuple(out)


def route_episodes(seed: int, n: int) -> list[Episode]:
    """Fresh Episode records (lists) for a seed; the cache keeps the sampling cheap."""
    return [
        Episode(list(s), list(a), list(r), bool(te), bool(tr))
        for s, a, r, te, tr in _route_episodes(int(seed), int(n))
    ]


# ------------------------------------------------------------------ 1.2 MC at the budget
def _mc_rms(mc_prediction: Callable[..., Any], *, seed: int, episodes: int) -> float:
    V = mc_prediction(route_episodes(seed, episodes), GAMMA, S)
    return _route_rms(_vector(V, S, f"mc_prediction(episodes, {GAMMA}, {S})"))


check_mc_rms = StochasticCheck(
    ex=1,
    functions="mc_prediction",
    metric=_mc_rms,
    live={"episodes": 1000},
    quick={"episodes": 200},
    higher_is_better=False,
    min_gap=1.0,
    what="RMS error on the route",
    thresholds={
        # threshold_protocol.py, live budget {"episodes": 1000}, N = 100 seeds (0-99),
        # 2026-10-17, local (Apple M1 Pro, arm64). Solution p99 0.3473; strongest wrong version
        # mutant 'no-discount' p1 3.238; gap 2.89 (minimum 1); threshold: the midpoint 1.79257,
        # rounded to 1.8. 0 of 100 solution seeds fail: flake rate at most about 3/N = 3.0%
        # (rule of three).
        "live": (
            1.8,
            "threshold protocol, 100 seeds: runs/2026-10-17-local-threshold-m03-check_mc_rms-live-n100-exp.json",
        ),
        # threshold_protocol.py, quick budget {"episodes": 200}, N = 100 seeds (0-99),
        # 2026-10-17, local (Apple M1 Pro, arm64). Solution p99 0.862; strongest wrong version
        # mutant 'no-discount' p1 2.976; gap 2.114 (minimum 1); threshold: the midpoint 1.91905,
        # rounded to 1.9. 0 of 100 solution seeds fail: flake rate at most about 3/N = 3.0%
        # (rule of three).
        "quick": (
            1.9,
            "threshold protocol, 100 seeds: runs/2026-10-17-local-threshold-m03-check_mc_rms-quick-n100-exp.json",
        ),
    },
)


# ------------------------------------------------------------------ 2a.1 TD(0) (exact)
def check_td0_prediction(fn: Callable[..., Any]) -> None:
    fx = expected("m03_prediction")
    for i, name in enumerate(fx["td_case_set"]):
        name = str(name)
        alpha, gamma = float(fx["td_case_alpha"][i]), float(fx["td_case_gamma"][i])
        n, v0 = int(fx["td_case_n_states"][i]), float(fx["td_case_v0"][i])
        episodes = _episodes(fx, name)
        label = (
            "the corridor A -> B -> goal, twice"
            if name == "corridor"
            else "five fixed episodes (the last one truncated)"
        )
        what = f"td0_prediction on {label}, alpha={alpha:g}, gamma={gamma:g}, v0={v0:g}"
        out = fn(episodes, alpha, gamma, n) if v0 == 0 else fn(episodes, alpha, gamma, n, v0=v0)
        V = _vector(out, n, what)
        assert_close(
            V,
            fx["td_case_V"][i, :n],
            atol=1e-9,
            what=what,
            hint="Process transitions in time order. The target is r + gamma V[s2], except on "
            "the last transition of a TERMINATED episode, where it is r alone. A truncated "
            "episode's last transition still bootstraps. Start every state at v0.",
        )


# ------------------------------------------------------------------ 2a.2 TD(0) at the budget
def _td0_rms(
    td0_prediction: Callable[..., Any], *, seed: int, episodes: int, alpha: float
) -> float:
    V = td0_prediction(route_episodes(seed, episodes), alpha, GAMMA, S)
    return _route_rms(_vector(V, S, f"td0_prediction(episodes, {alpha}, {GAMMA}, {S})"))


check_td0_rms = StochasticCheck(
    ex="2a",
    functions="td0_prediction",
    metric=_td0_rms,
    live={"episodes": 1000, "alpha": 0.05},
    quick={"episodes": 200, "alpha": 0.05},
    higher_is_better=False,
    min_gap=2.0,
    what="RMS error on the route",
    thresholds={
        # threshold_protocol.py, live budget {"alpha": 0.05, "episodes": 1000}, N = 100 seeds
        # (0-99), 2026-10-17, local (Apple M1 Pro, arm64). Solution p99 1.638; strongest wrong
        # version mutant 'no-bootstrap' p1 7.197; gap 5.558 (minimum 2); threshold: the midpoint
        # 4.41746, rounded to 4.4. 0 of 100 solution seeds fail: flake rate at most about 3/N =
        # 3.0% (rule of three).
        "live": (
            4.4,
            "threshold protocol, 100 seeds: runs/2026-10-17-local-threshold-m03-check_td0_rms-live-n100-exp.json",
        ),
        # threshold_protocol.py, quick budget {"alpha": 0.05, "episodes": 200}, N = 100 seeds
        # (0-99), 2026-10-17, local (Apple M1 Pro, arm64). Solution p99 2.934; strongest wrong
        # version mutant 'no-bootstrap' p1 7.286; gap 4.352 (minimum 2); threshold: the midpoint
        # 5.11018, rounded to 5.1. 0 of 100 solution seeds fail: flake rate at most about 3/N =
        # 3.0% (rule of three).
        "quick": (
            5.1,
            "threshold protocol, 100 seeds: runs/2026-10-17-local-threshold-m03-check_td0_rms-quick-n100-exp.json",
        ),
    },
)


# ------------------------------------------------------------------ 2b bias^2 and variance (exact)
def check_bias2_variance(fn: Callable[..., Any]) -> None:
    fx = expected("m03_decomposition")
    for i in range(int(fx["n_cases"])):
        est, v_true = fx[f"est_{i}"], fx[f"v_{i}"]
        seeds, states = est.shape
        what = f"bias2_variance on {seeds} seeds x {states} state(s)"
        out = fn(est.copy(), v_true.copy())
        if out is None:
            raise CheckFailed(f"{what} returned None. Did you forget `return bias2, variance`?")
        if not isinstance(out, (tuple, list)) or len(out) != 2:
            raise CheckFailed(f"{what} should return a pair (bias2, variance).")
        try:
            bias2, var = float(out[0]), float(out[1])
        except (TypeError, ValueError):
            raise CheckFailed(f"{what} should return two numbers (bias2, variance).") from None
        if not (np.isfinite(bias2) and np.isfinite(var)):
            raise CheckFailed(f"{what} returned NaN or infinity.")
        mse = float(fx[f"mse_{i}"])
        if np.isclose(var, float(fx[f"var_ddof1_{i}"]), rtol=1e-9, atol=1e-12):
            hint = (
                f"your variance divides by seeds - 1 = {seeds - 1}. The decomposition needs the "
                f"spread divided by the number of seeds, {seeds}: estimates.var(axis=0)."
            )
        elif np.isclose(var, mse, rtol=1e-9, atol=1e-12) and not np.isclose(var, fx[f"var_{i}"]):
            hint = (
                "your variance is the spread around v_true, which is the MSE itself. The "
                "variance is the spread around the mean over seeds."
            )
        elif np.isclose(bias2, float(fx[f"bias2_pooled_{i}"]), rtol=1e-9, atol=1e-12) and (
            not np.isclose(bias2, fx[f"bias2_{i}"])
        ):
            hint = (
                "your bias^2 averages the errors over states before squaring, so errors of "
                "opposite sign cancel. Square each state's error, then average."
            )
        else:
            hint = (
                "bias^2 = mean over states of (estimates.mean(axis=0) - v_true)^2; variance = "
                "mean over states of estimates.var(axis=0)."
            )
        if not np.isclose(bias2 + var, mse, rtol=1e-9, atol=1e-12):
            raise CheckFailed(
                f"{what}: bias^2 + variance = {bias2:.6g} + {var:.6g} = {bias2 + var:.6g}, but "
                f"the mean squared error over seeds and states is {mse:.6g}; the two must be "
                f"equal (decomposition). Hint: {hint}"
            )
        assert_close(bias2, fx[f"bias2_{i}"], atol=1e-12, what=f"{what}: bias2", hint=hint)
        assert_close(var, fx[f"var_{i}"], atol=1e-12, what=f"{what}: variance", hint=hint)


# ------------------------------------------------------------------ 3 the n-step target (exact)
def check_n_step_target(fn: Callable[..., Any]) -> None:
    fx = expected("m03_nstep")
    for i, want in enumerate(fx["expected"]):
        k, t, n = int(fx["case_traj"][i]), int(fx["case_t"][i]), int(fx["case_n"][i])
        gamma, terminated = float(fx["case_gamma"][i]), bool(fx["case_terminated"][i])
        T = int(fx["lengths"][k])
        rewards = [float(r) for r in fx["rewards"][k, :T]]
        states = [int(s) for s in fx["states"][k, : T + 1]]
        V = fx["V"][k].copy()
        got = fn(list(rewards), list(states), t, n, V, gamma, terminated)
        if got is None:
            raise CheckFailed("n_step_target returned None. Did you forget `return G`?")
        if t + n < T:
            hint = f"t + n = {t + n} < T = {T}: n rewards, then gamma^n V[s_(t+n)]."
        elif terminated:
            hint = (
                f"t + n = {t + n} >= T = {T} and the episode TERMINATED: sum the rewards to the "
                "end and add nothing; s_T is terminal, its value is 0 whatever V says."
            )
        else:
            hint = (
                f"t + n = {t + n} >= T = {T} and the episode was TRUNCATED: sum the rewards to "
                "the end, then bootstrap with gamma^(T - t) V[s_T]."
            )
        assert_close(
            got,
            want,
            atol=1e-9,
            what=(
                f"n_step_target(rewards={rewards}, states={states}, t={t}, n={n}, V, "
                f"gamma={gamma:g}, terminated={terminated})"
            ),
            hint=hint + " The first reward, r_t, is not discounted.",
        )


# ------------------------------------------------------------------ control: shared pieces
def _train(fn: Callable[..., Any], name: str, *, seed: int, episodes: int, alpha: float, **kw):
    """Call the participant's learner on a fresh env; validate (Q, online)."""
    out = fn(_make_env(), episodes, alpha, EPSILON, GAMMA, seed, **kw)
    if not isinstance(out, (tuple, list)) or len(out) != 2:
        raise CheckFailed(f"{name} should return a pair (Q, online).")
    try:
        Q = np.asarray(out[0], dtype=float)
        online = np.asarray(out[1], dtype=float)
    except (TypeError, ValueError):
        raise CheckFailed(f"{name} should return (Q, online): two arrays of numbers.") from None
    if Q.shape != (S, A):
        raise CheckFailed(f"{name}: Q should have shape ({S}, {A}); got {Q.shape}.")
    if not np.all(np.isfinite(Q)):
        raise CheckFailed(f"{name}: Q contains NaN or infinity.")
    if online.shape != (episodes,):
        raise CheckFailed(
            f"{name}: online should hold one reward sum per episode ({episodes}); "
            f"got shape {online.shape}."
        )
    return Q, online


def greedy_path(Q: np.ndarray) -> tuple[list[int], bool]:
    """From the start, argmax Q (first index on ties) until the goal, a revisit or S steps."""
    P = _mdp().P
    s, path = START, [START]
    for _ in range(S):
        if s == GOAL:
            return path, True
        s = int(np.argmax(P[s, int(np.argmax(Q[s]))]))
        if s in path:
            return path + [s], False
        path.append(s)
    return path, s == GOAL


def _no_goal(name: str, path: list[int], penalty: float) -> float:
    print(
        f"{name}'s greedy path does not reach the goal ({len(path) - 1} moves, then a loop "
        f"or the step limit); the checkpoint scores that as {penalty:g}."
    )
    return penalty


# ------------------------------------------------------------------ 4 SARSA
def _sarsa_edge(
    sarsa: Callable[..., Any], *, seed: int, control_episodes: int, control_alpha: float
) -> float:
    Q, _ = _train(sarsa, "sarsa", seed=seed, episodes=control_episodes, alpha=control_alpha)
    path, reached = greedy_path(Q)
    if not reached:
        return _no_goal("SARSA", path, 12.0)
    cols = GRID["cols"]
    return float(
        sum(1 for s in path if s // cols == GRID["rows"] - 2 and 1 <= s % cols <= cols - 2)
    )


check_sarsa = StochasticCheck(
    ex=4,
    functions="sarsa",
    metric=_sarsa_edge,
    live={"control_episodes": 1000, "control_alpha": 0.1},
    quick={"control_episodes": 1000, "control_alpha": 0.1},
    higher_is_better=False,
    min_gap=4.0,
    what="cells of SARSA's greedy path directly above the cliff",
    thresholds={
        # threshold_protocol.py, live budget {"control_alpha": 0.1, "control_episodes": 1000}, N
        # = 100 seeds (0-99), 2026-10-17, local (Apple M1 Pro, arm64). Solution p99 2; strongest
        # wrong version mutant 'max-target' p1 10; gap 8 (minimum 4); threshold: the midpoint,
        # 6.0. 0 of 100 solution seeds fail: flake rate at most about 3/N = 3.0% (rule of
        # three).
        "live": (
            6.0,
            "threshold protocol, 100 seeds: runs/2026-10-17-local-threshold-m03-check_sarsa-live-n100-exp.json",
        ),
        # threshold_protocol.py, quick budget {"control_alpha": 0.1, "control_episodes": 1000},
        # N = 100 seeds (0-99), 2026-10-17, local (Apple M1 Pro, arm64). Solution p99 2;
        # strongest wrong version mutant 'max-target' p1 10; gap 8 (minimum 4); threshold: the
        # midpoint, 6.0. 0 of 100 solution seeds fail: flake rate at most about 3/N = 3.0% (rule
        # of three).
        "quick": (
            6.0,
            "threshold protocol, 100 seeds: runs/2026-10-17-local-threshold-m03-check_sarsa-quick-n100-exp.json",
        ),
    },
)


# ------------------------------------------------------------------ 5.1 Q-learning's path
def _q_excess(
    q_learning: Callable[..., Any], *, seed: int, control_episodes: int, control_alpha: float
) -> float:
    Q, _ = _train(
        q_learning, "q_learning", seed=seed, episodes=control_episodes, alpha=control_alpha
    )
    path, reached = greedy_path(Q)
    if not reached:
        return _no_goal("Q-learning", path, float(S))
    return float(len(path) - 1 - OPTIMAL_MOVES)


check_q_learning_path = StochasticCheck(
    ex=5,
    functions="q_learning",
    metric=_q_excess,
    live={"control_episodes": 1000, "control_alpha": 0.1},
    quick={"control_episodes": 1000, "control_alpha": 0.1},
    higher_is_better=False,
    min_gap=1.0,
    what=f"moves of Q-learning's greedy path beyond the optimal {OPTIMAL_MOVES}",
    thresholds={
        # threshold_protocol.py, live budget {"control_alpha": 0.1, "control_episodes": 1000}, N
        # = 100 seeds (0-99), 2026-10-17, local (Apple M1 Pro, arm64). Solution p99 0; strongest
        # wrong version mutant 'taken-action-target' p1 2; gap 2 (minimum 1); threshold: the
        # midpoint, 1.0. 0 of 100 solution seeds fail: flake rate at most about 3/N = 3.0% (rule
        # of three).
        "live": (
            1.0,
            "threshold protocol, 100 seeds: runs/2026-10-17-local-threshold-m03-check_q_learning_path-live-n100-exp.json",
        ),
        # threshold_protocol.py, quick budget {"control_alpha": 0.1, "control_episodes": 1000},
        # N = 100 seeds (0-99), 2026-10-17, local (Apple M1 Pro, arm64). Solution p99 0;
        # strongest wrong version mutant 'taken-action-target' p1 2; gap 2 (minimum 1);
        # threshold: the midpoint, 1.0. 0 of 100 solution seeds fail: flake rate at most about
        # 3/N = 3.0% (rule of three).
        "quick": (
            1.0,
            "threshold protocol, 100 seeds: runs/2026-10-17-local-threshold-m03-check_q_learning_path-quick-n100-exp.json",
        ),
    },
)


# ------------------------------------------------------------------ 5.2 the cliff contrast
def _contrast(
    sarsa: Callable[..., Any],
    q_learning: Callable[..., Any],
    *,
    seed: int,
    control_episodes: int,
    control_alpha: float,
    window: int,
) -> float:
    _, on_s = _train(sarsa, "sarsa", seed=seed, episodes=control_episodes, alpha=control_alpha)
    _, on_q = _train(
        q_learning, "q_learning", seed=seed, episodes=control_episodes, alpha=control_alpha
    )
    g_s, g_q = float(on_s[-window:].mean()), float(on_q[-window:].mean())
    # The second term fails a q_learning that never learns (about -4500 per episode under
    # random play), for which the first term alone would be hugely positive.
    return min(g_s - g_q, g_q + 500.0)


check_cliff_contrast = StochasticCheck(
    ex=5,
    functions=("sarsa", "q_learning"),
    metric=_contrast,
    live={"control_episodes": 1000, "control_alpha": 0.1, "window": 500},
    quick={"control_episodes": 1000, "control_alpha": 0.1, "window": 500},
    higher_is_better=True,
    min_gap=5.0,
    what="SARSA's online-return lead over Q-learning (mean of the last window episodes)",
    thresholds={
        # threshold_protocol.py, live budget {"control_alpha": 0.1, "control_episodes": 1000,
        # "window": 500}, N = 100 seeds (0-99), 2026-10-17, local (Apple M1 Pro, arm64).
        # Solution p1 20.05; strongest wrong version mutant 'max-target' p99 6.765; gap 13.29
        # (minimum 5); threshold: the midpoint 13.4091, rounded to 13.0. 0 of 100 solution seeds
        # fail: flake rate at most about 3/N = 3.0% (rule of three).
        "live": (
            13.0,
            "threshold protocol, 100 seeds: runs/2026-10-17-local-threshold-m03-check_cliff_contrast-live-n100-exp.json",
        ),
        # threshold_protocol.py, quick budget {"control_alpha": 0.1, "control_episodes": 1000,
        # "window": 500}, N = 100 seeds (0-99), 2026-10-17, local (Apple M1 Pro, arm64).
        # Solution p1 20.05; strongest wrong version mutant 'max-target' p99 6.765; gap 13.29
        # (minimum 5); threshold: the midpoint 13.4091, rounded to 13.0. 0 of 100 solution seeds
        # fail: flake rate at most about 3/N = 3.0% (rule of three).
        "quick": (
            13.0,
            "threshold protocol, 100 seeds: runs/2026-10-17-local-threshold-m03-check_cliff_contrast-quick-n100-exp.json",
        ),
    },
)


# ------------------------------------------------------------------ 6 the planted bug
def _suspect_error(
    q_learning_suspect: Callable[..., Any],
    *,
    seed: int,
    control_episodes: int,
    control_alpha: float,
) -> float:
    Q, _ = _train(
        q_learning_suspect,
        "q_learning_suspect",
        seed=seed,
        episodes=control_episodes,
        alpha=control_alpha,
        q0=Q0,
    )
    return abs(float(Q[ABOVE_GOAL, DOWN]) + 1.0) + abs(float(Q[START].max()) - V_STAR_START)


check_q_learning_suspect = StochasticCheck(
    ex=6,
    functions="q_learning_suspect",
    metric=_suspect_error,
    live={"control_episodes": 1000, "control_alpha": 0.1},
    quick={"control_episodes": 1000, "control_alpha": 0.1},
    higher_is_better=False,
    min_gap=5.0,
    what=(
        f"|Q(above goal, down) + 1| + |max Q(start) - V*(start)| from Q0 = {Q0:g} "
        "(both terms are 0 for a correct fix)"
    ),
    thresholds={
        # threshold_protocol.py, live budget {"control_alpha": 0.1, "control_episodes": 1000}, N
        # = 100 seeds (0-99), 2026-10-17, local (Apple M1 Pro, arm64). Solution p99 4.485e-14;
        # strongest wrong version mutant 'still-bootstraps' p1 14.63; gap 14.63 (minimum 5);
        # threshold: the midpoint 7.31671, rounded to 7.3. 0 of 100 solution seeds fail: flake
        # rate at most about 3/N = 3.0% (rule of three).
        "live": (
            7.3,
            "threshold protocol, 100 seeds: runs/2026-10-17-local-threshold-m03-check_q_learning_suspect-live-n100-exp.json",
        ),
        # threshold_protocol.py, quick budget {"control_alpha": 0.1, "control_episodes": 1000},
        # N = 100 seeds (0-99), 2026-10-17, local (Apple M1 Pro, arm64). Solution p99 4.485e-14;
        # strongest wrong version mutant 'still-bootstraps' p1 14.63; gap 14.63 (minimum 5);
        # threshold: the midpoint 7.31671, rounded to 7.3. 0 of 100 solution seeds fail: flake
        # rate at most about 3/N = 3.0% (rule of three).
        "quick": (
            7.3,
            "threshold protocol, 100 seeds: runs/2026-10-17-local-threshold-m03-check_q_learning_suspect-quick-n100-exp.json",
        ),
    },
)


# ------------------------------------------------------------------ stretch: n-step TD (exact)
def check_n_step_td_prediction(fn: Callable[..., Any]) -> None:
    fx = expected("m03_prediction")
    for alpha, gamma, n, want in zip(
        fx["nstep_td_alpha"], fx["nstep_td_gamma"], fx["nstep_td_n"], fx["nstep_td_V"], strict=True
    ):
        what = (
            f"n_step_td_prediction on five fixed episodes, alpha={alpha:g}, gamma={gamma:g}, n={n}"
        )
        V = _vector(fn(_episodes(fx, "td"), float(alpha), float(gamma), 5, int(n)), 5, what)
        assert_close(
            V,
            want,
            atol=1e-9,
            what=what,
            hint="Start at V = 0. For each episode and each t in order, compute G_t^(n) with "
            "n_step_target and the values as they stand, then move V[s_t] a fraction alpha.",
        )


__all__ = [
    "Episode",
    "check_bias2_variance",
    "check_cliff_contrast",
    "check_mc_prediction",
    "check_mc_rms",
    "check_n_step_target",
    "check_n_step_td_prediction",
    "check_q_learning_path",
    "check_q_learning_suspect",
    "check_sarsa",
    "check_td0_prediction",
    "check_td0_rms",
    "greedy_path",
    "route_episodes",
]
