"""Checkpoints for Module 7 (function approximation and why it can diverge).

Exact checks (1 to 5, 6.1, 7.1 and the stretch) compare the participant's function with
fixtures written by scripts/make_expected.py on fixed arrays: Baird's counterexample, small
random models, four CartPole-like transitions and seven fixed observations. The fixtures also
hold the outputs of the usual mistakes, so a failure can name the one it sees. Nothing here
computes a TD matrix, an update or a stationary distribution of its own.

Checkpoints 6.2 and 7.2 are StochasticChecks with thresholds from
scripts/threshold_protocol.py. Each trains the participant's linear_q_update (with the
provided rbf_features in 6.2, with the participant's colleague_features in 7.2) on the
checkpoint's seed and scores the greedy policy's mean return over 10 episodes. A run whose
logged weight norm becomes non-finite or passes NORM_BOUND fails with its own message,
whatever its return (OPEN_QUESTIONS C31): the greedy return of a diverged run can be high by
luck.

The training plumbing below (the features, the loop and the greedy evaluation) is the lab's
provided ex6-run code, copied here so that the checks train exactly as the lab does, without
needing the notebook's namespace; prl/tests/test_checks_m07.py checks that the copies equal
the lab's cells. The learning update is always the participant's.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np

from ..envs import Baird
from . import CheckFailed, StochasticCheck, assert_close, expected

#: A participant's code that raises one of these gets a CheckFailed with a hint, not a crash
#: (verify mode does not count a crash as a rejection).
_CRASHES = (np.linalg.LinAlgError, ValueError, IndexError, TypeError, KeyError, AttributeError)

#: How large ‖W‖ may get in the training checkpoints before the run counts as diverged. CartPole
#: pays 1 per step, so no discounted return exceeds 1 / (1 - gamma) = 100, and with unit-length
#: features a healthy run's ‖W‖ levels off in the hundreds. The bound is 100 times that scale.
#: The threshold protocol fails if the solution passes it at any seed; see the provenance of
#: each threshold and _notes/m07-te-notes.md for the margins on both sides.
NORM_BOUND = 1e4
EVAL_EPISODES = 10  # the lab's budget.eval_episodes


# ------------------------------------------------------------------ the lab's provided plumbing
OBS_SCALE = np.array([2.4, 3.0, 0.2095, 3.5])  # cart x and pole angle: where CartPole ends
CENTERS = np.linspace(-1.0, 1.0, 5)  # five bumps per scaled number
WIDTH = CENTERS[1] - CENTERS[0]  # sigma = the spacing, 0.5
GRID_POINTS = np.stack(np.meshgrid(*[CENTERS] * 4, indexing="ij"), axis=-1).reshape(-1, 4)
ALPHA_Q, GAMMA_Q, EPS_END = 0.3, 0.99, 0.05


def rbf_raw(obs):
    """Provided: 625 Gaussian bumps on a 5^4 grid of the scaled observation, each in [0, 1]."""
    z = np.asarray(obs, dtype=float) / OBS_SCALE
    per_dim = np.exp(-((z[:, None] - CENTERS) ** 2) / (2 * WIDTH**2))  # (4, 5)
    return np.einsum("i,j,k,l->ijkl", *per_dim).ravel()


def rbf_features(obs):
    """Provided: the bumps scaled to unit length (RBF features)."""
    f = rbf_raw(obs)
    return f / np.linalg.norm(f)


def train_linear_q(update, featurize, steps, eps_steps, seed, log_every=1000):
    """Provided: epsilon-greedy linear Q-learning on CartPole-v1 with your update.

    Returns (W, log). Every `log_every` steps the log records the step, ‖W‖ and the mean
    return of the last 10 finished episodes; it also keeps every episode's return."""
    env = gym.make("CartPole-v1")
    rng = np.random.default_rng([seed, 7])
    obs, _ = env.reset(seed=seed)
    phi = featurize(obs)
    W = np.zeros((env.action_space.n, len(phi)))
    log = {"step": [], "w_norm": [], "recent_return": [], "returns": []}
    episode_return, start, shown = 0.0, time.perf_counter(), 0
    for t in range(steps):
        epsilon = max(EPS_END, 1.0 - (1.0 - EPS_END) * t / eps_steps)
        with np.errstate(over="ignore", invalid="ignore"):
            q = W @ phi
        explore = rng.random() < epsilon or not np.all(np.isfinite(q))
        a = int(rng.integers(len(W))) if explore else int(np.argmax(q))
        obs, r, terminated, truncated, _ = env.step(a)
        phi_next = featurize(obs)
        with np.errstate(over="ignore", invalid="ignore"):
            W = update(W, phi, a, r, phi_next, terminated, truncated, GAMMA_Q, ALPHA_Q)
        episode_return += r
        if terminated or truncated:
            log["returns"].append(episode_return)
            episode_return = 0.0
            obs, _ = env.reset()
            phi_next = featurize(obs)
        phi = phi_next
        if (t + 1) % log_every == 0:
            recent = log["returns"][-10:]
            log["step"].append(t + 1)
            log["w_norm"].append(float(np.linalg.norm(W)))
            log["recent_return"].append(float(np.mean(recent)) if recent else np.nan)
            if time.perf_counter() - start > 10 * (shown + 1):  # a line about every 10 s
                shown += 1
                print(f"  step {t + 1:6d}: last 10 returns {log['recent_return'][-1]:6.1f}")
    env.close()
    return W, log


def greedy_return(W, featurize, seed, episodes):
    """Provided: the mean return of the greedy policy over `episodes` seeded episodes."""
    env = gym.make("CartPole-v1")
    total = 0.0
    for i in range(episodes):
        obs, _ = env.reset(seed=seed + 10_000 + i)
        done = False
        while not done:
            with np.errstate(over="ignore", invalid="ignore"):
                q = W @ featurize(obs)
            a = int(np.argmax(q)) if np.all(np.isfinite(q)) else 0
            obs, r, terminated, truncated, _ = env.step(a)
            total += r
            done = terminated or truncated
    env.close()
    return total / episodes


# ------------------------------------------------------------------ helpers
def _call(fn: Callable[..., Any], *args: Any, what: str) -> Any:
    try:
        return fn(*args)
    except _CRASHES as exc:
        raise CheckFailed(f"{what} raised {type(exc).__name__}: {exc}.") from None


def _array(out: Any, shape: tuple[int, ...], what: str) -> np.ndarray:
    if out is None:
        raise CheckFailed(f"{what} returned None. Did you forget `return`?")
    try:
        arr = np.asarray(out, dtype=float)
    except (TypeError, ValueError):
        raise CheckFailed(f"{what} should return an array of shape {shape}.") from None
    if arr.shape != tuple(shape):
        raise CheckFailed(f"{what} should return shape {tuple(shape)}; got {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise CheckFailed(f"{what} contains NaN or infinity.")
    return arr


def _copies(args: tuple[Any, ...]) -> tuple[Any, ...]:
    return tuple(np.array(x, copy=True) if isinstance(x, np.ndarray) else x for x in args)


def _unchanged(before: tuple[Any, ...], after: tuple[Any, ...], names: str, fn: str) -> None:
    for name, b, a in zip(names.split(), before, after, strict=True):
        if isinstance(b, np.ndarray) and not np.array_equal(a, b):
            raise CheckFailed(
                f"{fn} changed its argument `{name}` in place. Return a new array and leave "
                f"`{name}` as it was: `{name} + ...` makes a new array, `{name} += ...` changes "
                "the caller's."
            )


def _matches(got: np.ndarray, fx: Mapping[str, np.ndarray], key: str) -> bool:
    if key not in fx:
        return False
    want = fx[key]
    return got.shape == want.shape and bool(np.allclose(got, want, rtol=1e-6, atol=1e-8))


def _close_or_hint(
    got: np.ndarray,
    want: np.ndarray,
    fx: Mapping[str, np.ndarray],
    hints: Mapping[str, str],
    *,
    what: str,
    rtol: float = 1e-6,
    atol: float = 1e-8,
) -> None:
    """assert_close, with a hint naming a known mistake when the answer matches one."""
    if got.shape == want.shape and np.allclose(got, want, rtol=rtol, atol=atol):
        return
    hint = next((h for key, h in hints.items() if _matches(got, fx, key)), None)
    assert_close(got, want, rtol=rtol, atol=atol, what=what, hint=hint)


def _integer(x: Any) -> int | None:
    """x as an int if it is an integral number (int, numpy integer or integral float)."""
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)) and float(x).is_integer():
        return int(x)
    return None


# ------------------------------------------------------------------ 1 scale_spec
_SCALE_KEYS = ("n_states", "n_actions", "q_entries", "feature_dim")


def check_scale_spec(scale_spec: Callable[..., Any]) -> None:
    """Checkpoint 1: the four counts, exactly, for four shop sizes."""
    fx = expected("m07_scale")
    for i, (n, c) in enumerate(zip(fx["n_products"], fx["capacity"], strict=True)):
        n, c = int(n), int(c)
        call = f"scale_spec({n}, {c})"
        out = _call(scale_spec, n, c, what=call)
        if not isinstance(out, Mapping):
            raise CheckFailed(f"{call} should return a dict, not {type(out).__name__}.")
        missing = [k for k in _SCALE_KEYS if k not in out]
        if missing:
            raise CheckFailed(f"{call} has no key {missing[0]!r}; it needs {list(_SCALE_KEYS)}.")
        levels = c + 1
        for key in _SCALE_KEYS:
            want = int(fx[key][i])
            got = _integer(out[key])
            if got == want:
                continue
            hint = {
                "n_states": "a product has capacity + 1 stock levels (0 to capacity), and the "
                "products' levels combine: the counts multiply, levels ** n_products.",
                "n_actions": "an order is 0 to capacity units of each product, so joint orders "
                "multiply like joint stock levels.",
                "q_entries": "a table Q[s, a] has one entry per state and action: "
                "n_states * n_actions.",
                "feature_dim": "one indicator per product and stock level: the lengths add up, "
                "n_products * (capacity + 1).",
            }[key]
            if key == "n_states" and got == n * levels:
                hint = "you added the levels; joint stock levels multiply: levels ** n_products."
            elif key == "n_states" and got == c**n:
                hint = "stock 0 is a level too: a product has capacity + 1 levels."
            elif key == "feature_dim" and got == levels**n:
                hint = (
                    "a one-hot vector of the joint state is the table again; the briefing's "
                    "features have one indicator per product and stock level, side by side."
                )
            elif key == "q_entries" and got == levels**n:
                hint = "a Q table has an entry for every action at every state."
            shown = f"{out[key]!r}" if got is None else f"{got:,}"
            raise CheckFailed(f"{call}[{key!r}] should be {want:,}; got {shown}. Hint: {hint}")


# ------------------------------------------------------------------ 2 semi_gradient_td0
_TD0_HINTS = {
    "residual": "you differentiated the target too (the full gradient of ½δ², a residual-"
    "gradient step). Semi-gradient TD(0) holds the target fixed: the step is along φ(s) only.",
    "next": "the step goes along φ(s), the features of the state being updated, not φ(s').",
    "no_discount": "the target is r + γ w·φ(s'): multiply the next value by gamma.",
    "sign": "the update adds alpha * delta * phi; a positive TD error must raise v̂(s).",
}


def check_semi_gradient_td0(semi_gradient_td0: Callable[..., Any]) -> None:
    """Checkpoint 2: exact on Baird's 1 -> 7, a terminal transition and three random cases."""
    fx = expected("m07_td0")
    for i in range(int(fx["n_cases"])):
        args = tuple(
            fx[f"{k}_{i}"] if k.startswith(("w", "phi")) else float(fx[f"{k}_{i}"])
            for k in ("w", "phi", "r", "phi_next", "gamma", "alpha")
        )
        given = _copies(args)
        what = f"semi_gradient_td0 (case {i + 1})"
        out = _call(semi_gradient_td0, *given, what=what)
        _unchanged(args, given, "w phi r phi_next gamma alpha", "semi_gradient_td0")
        got = _array(out, args[0].shape, what)
        _close_or_hint(
            got,
            fx[f"want_{i}"],
            fx,
            {f"{k}_{i}": h for k, h in _TD0_HINTS.items()},
            what=f"the new w (case {i + 1})",
        )


# ------------------------------------------------------------------ 3 td_matrix
_A_HINTS = {
    "transposed": "this is Aᵀ: the rows of A belong to the weights being updated, "
    "A = Φᵀ D (I − γP) Φ.",
    "no_weights": "weight each state's row by mu: D = diag(mu) sits between Φᵀ and (I − γP).",
    "P_transposed": "use P as given, P[s, s'] = P(s' | s): the target of s looks at its "
    "successors, (P @ phi)[s] = Σ_s' P(s, s') φ(s').",
    "symmetrized": "do not symmetrize A. It is not a Hessian, and its asymmetry is the point "
    "of the exercise.",
    "plus_gamma": "it is I − γP: the TD error subtracts the current value from the target, "
    "δ = r + γ v(s') − v(s).",
}


def check_td_matrix(td_matrix: Callable[..., Any]) -> None:
    """Checkpoint 3: A = Φᵀ D (I − γP) Φ on Baird and on a random model."""
    fx = expected("m07_td_matrix")
    for tag, name in (("random", "a random 4-state model"), ("baird", "Baird's counterexample")):
        args = (fx[f"{tag}_phi"], fx[f"{tag}_mu"], fx[f"{tag}_P"], float(fx[f"{tag}_gamma"]))
        given = _copies(args)
        out = _call(td_matrix, *given, what=f"td_matrix on {name}")
        _unchanged(args, given, "phi mu P gamma", "td_matrix")
        d = args[0].shape[1]
        got = _array(out, (d, d), f"td_matrix on {name}")
        hints = {f"{tag}_{k}": h for k, h in _A_HINTS.items()}
        _close_or_hint(got, fx[f"{tag}_A"], fx, hints, what=f"A on {name}")


# ------------------------------------------------------------------ 4 baird_expected_update
_SWEEP_HINTS = {
    "unweighted": "weight each state's expected TD error by mu(s) before summing: "
    "phi.T @ (mu * delta).",
    "P_transposed": "the expected next value of state s is (P @ v)[s] = Σ_s' P(s, s') v(s').",
    "sequential": "compute every state's TD error from the same w, then update once: one "
    "expected sweep, not one state after another.",
}


def check_baird_expected_update(baird_expected_update: Callable[..., Any]) -> None:
    """Checkpoint 4: one sweep on a random model; Baird's path after 1, 10, 100, 1000 sweeps."""
    fx = expected("m07_expected_update")
    names = "w phi mu P r gamma alpha"
    args = (fx["w"], fx["phi"], fx["mu"], fx["P"], fx["r"], float(fx["gamma"]), float(fx["alpha"]))
    given = _copies(args)
    what = "baird_expected_update on a random 4-state model"
    out = _call(baird_expected_update, *given, what=what)
    _unchanged(args, given, names, "baird_expected_update")
    got = _array(out, args[0].shape, what)
    if np.array_equal(got, args[0]):
        raise CheckFailed(
            f"{what} returned w unchanged. One sweep adds alpha * phi.T @ (mu * delta), with "
            "delta = r + gamma P v - v and v = phi @ w."
        )
    _close_or_hint(got, fx["want"], fx, _SWEEP_HINTS, what="w after one sweep (random model)")
    b = Baird()
    alpha = float(fx["alpha_baird"])
    w = b.w0.copy()
    for k in range(1, int(fx["sweeps"][-1]) + 1):
        out = _call(
            baird_expected_update,
            w,
            b.phi.copy(),
            b.mu_behavior.copy(),
            b.P_target.copy(),
            b.rewards.copy(),
            b.gamma,
            alpha,
            what=f"baird_expected_update on Baird, sweep {k}",
        )
        w = _array(out, (b.n_features,), f"baird_expected_update on Baird, sweep {k}")
        if k in fx["sweeps"]:
            assert_close(
                w, fx[f"baird_{k}"], rtol=1e-9, atol=1e-9, what=f"Baird's w after {k} sweeps"
            )
    if not np.linalg.norm(w) > 40 * np.linalg.norm(b.w0):
        raise CheckFailed("On Baird, ‖w‖ should grow over 1000 sweeps.")


# ------------------------------------------------------------------ 5 remove_one_leg
_LEGS = {"on-policy": ("mu",), "tabular": ("phi",), "monte-carlo": ("gamma", "r")}


def _modified_baird(fx: Mapping[str, np.ndarray]) -> Baird:
    env = Baird()
    env.P_target = fx["modified_P"].copy()
    env.rewards = fx["modified_rewards"].copy()
    return env


def check_remove_one_leg(remove_one_leg: Callable[..., Any]) -> None:
    """Checkpoint 5: the changed arguments for each leg, on Baird and on a modified Baird."""
    fx = expected("m07_legs")
    for tag, env, name in (
        ("baird", Baird(), "Baird"),
        ("modified", _modified_baird(fx), "a second target policy with nonzero rewards"),
    ):
        what = f"remove_one_leg on {name}"
        out = _call(remove_one_leg, env, what=what)
        if not isinstance(out, Mapping):
            raise CheckFailed(f"{what} should return a dict, not {type(out).__name__}.")
        if set(out) != set(_LEGS):
            raise CheckFailed(
                f"{what} should have exactly the keys {sorted(_LEGS)}; got {sorted(out)}."
            )
        for leg, fields in _LEGS.items():
            change = out[leg]
            if not isinstance(change, Mapping) or set(change) != set(fields):
                got = sorted(change) if isinstance(change, Mapping) else type(change).__name__
                raise CheckFailed(
                    f"{what}[{leg!r}] should change exactly {list(fields)}; got {got}."
                )
        S = env.n_states
        mu = _array(out["on-policy"]["mu"], (S,), f"{what}['on-policy']['mu']")
        if not np.allclose(mu, fx[f"{tag}_mu"], rtol=0, atol=1e-10):
            if np.allclose(mu, np.full(S, 1 / S)):
                hint = (
                    "this is the behavior's uniform weighting. On-policy weights are the target "
                    "policy's stationary distribution: mu with mu @ P_target = mu."
                )
            elif tag == "modified":
                hint = (
                    "your mu fits Baird but not this target policy: compute the stationary "
                    "distribution from env.P_target, not from Baird's numbers."
                )
            else:
                hint = "mu must satisfy mu @ P_target = mu and sum to 1."
            assert_close(
                mu, fx[f"{tag}_mu"], rtol=0, atol=1e-10, what="the on-policy mu", hint=hint
            )
        phi = _array(out["tabular"]["phi"], (S, S), f"{what}['tabular']['phi']")
        if not np.array_equal(phi, np.eye(S)):
            raise CheckFailed(
                f"{what}['tabular']['phi'] should be the {S} x {S} identity: one feature per state."
            )
        gamma = out["monte-carlo"]["gamma"]
        if _integer(gamma) != 0:
            raise CheckFailed(
                f"{what}['monte-carlo']['gamma'] should be 0, got {gamma!r}. A Monte Carlo "
                "target does not bootstrap: with r = v_pi the update needs no next value."
            )
        r = _array(out["monte-carlo"]["r"], (S,), f"{what}['monte-carlo']['r']")
        hint = None
        if np.allclose(r, env.rewards) and not np.allclose(r, fx[f"{tag}_v_pi"]):
            hint = (
                "these are the one-step rewards. The expected Monte Carlo target is v_pi, the "
                "solution of v = r + gamma P_target v (they agree only when every reward is 0)."
            )
        assert_close(r, fx[f"{tag}_v_pi"], what=f"the Monte Carlo r (v_pi) on {name}", hint=hint)


# ------------------------------------------------------------------ 6.1 linear_q_update
def check_linear_q_update(linear_q_update: Callable[..., Any]) -> None:
    """Checkpoint 6.1: four transitions; only row a changes; truncation still bootstraps."""
    fx = expected("m07_linear_q")
    W, phi, phi_next = fx["W"], fx["phi"], fx["phi_next"]
    r, g, alpha = float(fx["r"]), float(fx["gamma"]), float(fx["alpha"])
    for i, (a, term, trunc) in enumerate(fx["cases"].tolist()):
        term, trunc = bool(term), bool(trunc)
        args = (W, phi, int(a), r, phi_next, term, trunc, g, alpha)
        given = _copies(args)
        case = f"a = {a}, terminated = {term}, truncated = {trunc}"
        what = f"linear_q_update ({case})"
        out = _call(linear_q_update, *given, what=what)
        _unchanged(
            args, given, "W phi a r phi_next terminated truncated gamma alpha", "linear_q_update"
        )
        got = _array(out, W.shape, what)
        other = 1 - int(a)
        if not np.allclose(got[other], W[other], rtol=0, atol=1e-12):
            raise CheckFailed(
                f"{what} changed row {other} of W. Only the row of the action taken moves: the "
                "gradient of W[a] @ phi with respect to every other row is 0."
            )
        if term and trunc:
            words = "terminated: the target is r alone, whatever truncated says."
        elif term:
            words = "terminated: the next state's value is 0, so the target is r alone."
        elif trunc:
            words = (
                "truncated but not terminated: the episode was cut off by the time limit, so "
                "the target still bootstraps, r + γ max_b W[b] @ phi_next."
            )
        else:
            words = "the target is r + γ max_b W[b] @ phi_next."
        hints = {
            f"always_{i}": f"this transition {words}",
            f"never_{i}": f"this transition {words}",
            f"action0_{i}": "the target uses the max over actions of Q(s', b), "
            "np.max(W @ phi_next), not one action's value.",
            f"no_discount_{i}": "the target is r + γ max_b W[b] @ phi_next: multiply the next "
            "value by gamma.",
            f"next_{i}": "row a moves along phi, the features of s, not phi_next.",
        }
        _close_or_hint(got, fx[f"want_{i}"], fx, hints, what=f"the new W ({case})")


# ------------------------------------------------------------------ 6.2 and 7.2 training
def _trained_return(
    update: Callable[..., Any],
    featurize: Callable[..., Any],
    *,
    seed: int,
    steps: int,
    eps_steps: int,
    log_every: int,
) -> float:
    """Train with the lab's loop; the greedy return, or CheckFailed if ‖W‖ diverged."""
    W, log = train_linear_q(update, featurize, steps, eps_steps, seed, log_every=log_every)
    norms = np.asarray(log["w_norm"], dtype=float)
    bad = ~np.isfinite(norms) | (norms > NORM_BOUND)
    if bad.any():
        i = int(np.argmax(bad))
        finite = norms[np.isfinite(norms)]
        largest = f"; its largest finite value was {finite.max():.3g}" if finite.size else ""
        raise CheckFailed(
            f"While training on the checkpoint's seed, ‖W‖ reached {norms[i]:.3g} at step "
            f"{log['step'][i]:,}{largest}. A healthy run of this lab levels off in the hundreds; "
            f"above {NORM_BOUND:g} the weights have diverged, whatever the return. Read the "
            "weight-norm log, and compute alpha * ‖φ(s)‖² for a few states (step-size rule): "
            "above 2, each update makes a state's error larger."
        )
    return float(greedy_return(W, featurize, seed, EVAL_EPISODES))


def _linear_q_return(linear_q_update, *, seed, steps, eps_steps, log_every):
    return _trained_return(
        linear_q_update,
        rbf_features,
        seed=seed,
        steps=steps,
        eps_steps=eps_steps,
        log_every=log_every,
    )


def _colleague_return(linear_q_update, colleague_features, *, seed, steps, eps_steps, log_every):
    return _trained_return(
        linear_q_update,
        colleague_features,
        seed=seed,
        steps=steps,
        eps_steps=eps_steps,
        log_every=log_every,
    )


@dataclass(frozen=True, eq=False)
class _HintedCheck(StochasticCheck):
    """A StochasticCheck whose threshold failure ends with a hint for the participant."""

    hint: str = ""

    def __call__(self, *fns: Any) -> None:
        try:
            super().__call__(*fns)
        except CheckFailed as exc:
            if not str(exc).startswith(f"{self.what} ="):
                raise  # a diverged run: its own message says what to look at
            raise CheckFailed(f"{exc} Hint: {self.hint}") from None


#: The lab's live budget (its settings cell), in the keys train_linear_q takes. The training
#: checkpoints use it under QUICK too: at the lab's QUICK budget (10,000 steps, epsilon over
#: 4,000) the threshold protocol found no threshold (100 seeds: the solution's p1 140.8
#: against action-0-target's p99 122.1, a gap of 18.6, below the stated 20, and one solution
#: seed under the midpoint). One checkpoint call at the live budget takes about 1 s on a
#: laptop (Apple M1 Pro; not a Colab time).
LIVE = {"steps": 30_000, "eps_steps": 10_000, "log_every": 1000}
_WHAT = "greedy return after training on the checkpoint's seed (mean of 10 episodes)"

check_linear_q_training = _HintedCheck(
    ex=6,
    functions="linear_q_update",
    metric=_linear_q_return,
    live=LIVE,
    quick=LIVE,
    higher_is_better=True,
    min_gap=20.0,
    what=_WHAT,
    hint=(
        "the target is r + γ max_b W[b] @ phi(s') (r alone if terminated), and only row a "
        "moves, by alpha * delta * phi(s). If ‖W‖ at the end of the run cell is 0, your update "
        "returns W unchanged."
    ),
    thresholds={},
)

check_colleague_training = _HintedCheck(
    ex=7,
    functions=("linear_q_update", "colleague_features"),
    metric=_colleague_return,
    live=LIVE,
    quick=LIVE,
    higher_is_better=True,
    min_gap=20.0,
    what=_WHAT,
    hint=(
        "with the fixed features, training should go as in Exercise 6. Features of another "
        "length change the effective step alpha * ‖φ(s)‖² (step-size rule): divide the bumps "
        "by their Euclidean length, not by their sum or their largest entry."
    ),
    thresholds={},
)


# ------------------------------------------------------------------ 7.1 colleague_features
def check_colleague_features(colleague_features: Callable[..., Any]) -> None:
    """Checkpoint 7.1: the unit-length bumps at seven fixed observations."""
    fx = expected("m07_features")
    for obs, want in zip(fx["obs"], fx["features"], strict=True):
        shown = np.array2string(obs, precision=3)
        what = f"colleague_features({shown})"
        got = _array(_call(colleague_features, obs.copy(), what=what), want.shape, what)
        if np.allclose(got, want, rtol=1e-9, atol=1e-12):
            continue
        length2, total = float(got @ got), float(got.sum())
        if np.allclose(got, got[0]):
            hint = "these features are the same for every bump: they carry no state."
        elif abs(total - 1) < 1e-9:
            hint = (
                f"these sum to 1, so ‖φ(s)‖² = {length2:.4f}: you divided by the sum. Divide by "
                "the Euclidean length, np.linalg.norm."
            )
        elif length2 > 1.5:
            hint = (
                f"‖φ(s)‖² = {length2:.3f} here, so alpha * ‖φ(s)‖² = {ALPHA_Q * length2:.2f}. "
                "Each bump is in [0, 1], but the vector's length is what multiplies the step. "
                "Divide the bumps by their Euclidean length (dividing by the largest bump, "
                "which is near 1, leaves the length near 3)."
            )
        elif np.allclose(got / np.sqrt(length2), want, rtol=1e-6, atol=1e-9):
            hint = (
                f"your features point the right way but have length {np.sqrt(length2):.4f}. "
                "Divide the bumps by their Euclidean length, np.linalg.norm, so that "
                "‖φ(s)‖ = 1 at every state."
            )
        else:
            hint = "the features should be the 625 bumps divided by their Euclidean length."
        assert_close(got, want, rtol=1e-9, atol=1e-12, what=f"φ({shown})", hint=hint)


# ------------------------------------------------------------------ stretch: TDC
def check_tdc_expected_update(tdc_expected_update: Callable[..., Any]) -> None:
    """Stretch: one expected TDC sweep on Baird and on a random model."""
    fx = expected("m07_tdc")
    names = ("w", "v", "phi", "mu", "P", "r", "gamma", "alpha", "beta")
    for tag, name in (("baird", "Baird"), ("random", "a random 4-state model")):
        args = tuple(
            fx[f"{tag}_{k}"] if k in ("w", "v", "phi", "mu", "P", "r") else float(fx[f"{tag}_{k}"])
            for k in names
        )
        what = f"tdc_expected_update on {name}"
        out = _call(tdc_expected_update, *_copies(args), what=what)
        if not (isinstance(out, (tuple, list)) and len(out) == 2):
            raise CheckFailed(f"{what} should return a pair (new w, new v).")
        w_new = _array(out[0], args[0].shape, f"{what}: the new w")
        v_new = _array(out[1], args[1].shape, f"{what}: the new v")
        hints = {
            f"{tag}_no_correction_w": "the w update needs the correction − α γ (P Φ)ᵀ D Φ v; "
            "without it this is semi-gradient TD again.",
            f"{tag}_correction_sign_w": "subtract the correction: w + α(e − γ (P Φ)ᵀ D Φ v).",
        }
        _close_or_hint(w_new, fx[f"{tag}_want_w"], fx, hints, what=f"the new w on {name}")
        assert_close(
            v_new,
            fx[f"{tag}_want_v"],
            what=f"the new v on {name}",
            hint="v moves by beta * (e − Φᵀ D Φ v), with e = b − A w.",
        )
