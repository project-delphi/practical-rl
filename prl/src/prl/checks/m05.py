"""Checkpoints for Module 5 (contextual bandits: ridge, LinUCB, linear TS, logging).

Every check is exact: its expected values come from fixtures written by
scripts/make_expected.py (m05_*), computed once on fixed inputs. Nothing here fits a model.
Exercises 3 and 4 get fixed containers (`A`, `A_inv`, `b`, `n` and `predict(x) =
(A_inv @ b) @ x`), not learners. Exercise 4's sampler is checked on the moments of a
fixed-seed draw: the outcome is a fixed function of the participant's code, and the
tolerances (6 standard errors per mean, 0.2 per whitened-covariance entry, about 9 standard
errors) pass any correct sampler, whatever order it draws in. Exercises 6 and 7 log from
RecSim with a fixed table of score weights (`score_weights`); Exercise 7's behavior policy is the
epsilon-greedy written below, not the participant's `log_policy`.

Every check turns a wrong shape, None, NaN or an invalid log into CheckFailed itself, so
verify mode (which counts only CheckFailed) sees each wrong version rejected.
"""

from __future__ import annotations

import contextlib
import copy
import io
import math
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from ..envs import RecSim
from ..evaluate import Budget, compare
from . import CheckFailed, _fmt, expected

K, D_RAW, CATALOG = 10, 5, 0  # the lab's catalog: RecSim(10, 5, seed=0)
LINTS_MEAN_SE = 6.0  # each draw mean within 6 standard errors of A_inv b
LINTS_COV_TOL = 0.2  # each whitened-covariance entry within 0.2 of I (about 9 sd)
WEIGHT_SE = 4.0  # the mean importance weight within 4 standard errors of 1
SHARE_SE = 5.0  # Exercise 7: the share of rows not showing the best item, within 5 SE


def _features(x: np.ndarray) -> np.ndarray:
    return np.append(x, 1.0)


class _Model:
    """One item's ridge statistics, fixed (a container, not a learner)."""

    def __init__(self, A: Any, A_inv: Any, b: Any, n: int):
        self.A = np.array(A, dtype=float)
        self.A_inv = np.array(A_inv, dtype=float)
        self.b = np.array(b, dtype=float)
        self.n = int(n)

    def predict(self, x: np.ndarray) -> float:
        return float(self.A_inv @ self.b @ x)


# ------------------------------------------------------------------ helpers
def _array(out: Any, shape: tuple[int, ...], what: str, *, unit: str = "") -> np.ndarray:
    """A finite float array of `shape`, or CheckFailed (None, ragged, NaN are rejected)."""
    if out is None:
        raise CheckFailed(f"{what} is None. Did you forget `return`?")
    try:
        arr = np.asarray(out, dtype=float)
    except (TypeError, ValueError):
        raise CheckFailed(f"{what} should be numbers of shape {shape}; got {out!r:.80}.") from None
    if arr.shape != tuple(shape):
        raise CheckFailed(
            f"{what} should have shape {tuple(shape)}{unit}, but has shape {arr.shape}."
        )
    if not np.all(np.isfinite(arr)):
        raise CheckFailed(f"{what} contains NaN or infinity: {_fmt(arr)}.")
    return arr


def _number(out: Any, what: str) -> float:
    """One finite real number (a float or a 0-d array), or CheckFailed."""
    if out is None:
        raise CheckFailed(f"{what} returned None. Did you forget `return`?")
    if np.ndim(out) != 0:
        raise CheckFailed(
            f"{what} should return one number, but returned shape {np.shape(out)}. "
            "The prediction is one number: wrap it in float(...)."
        )
    try:
        value = float(out)
    except (TypeError, ValueError):
        raise CheckFailed(f"{what} should return a number; it returned {out!r:.80}.") from None
    if not math.isfinite(value):
        raise CheckFailed(f"{what} returned {value}, not a finite number.")
    return value


def _like(got: np.ndarray, pattern: np.ndarray, atol: float = 1e-9) -> bool:
    return got.shape == pattern.shape and bool(np.allclose(got, pattern, atol=atol, rtol=0.0))


# ------------------------------------------------------------------ 1 reward_fn
def check_reward_fn(fn: Callable[..., Any]) -> None:
    """Exercise 1: clip(0.5 + scale theta_a^T x, 0.02, 0.98) on three hand cases and RecSim."""
    fx = expected("m05_reward")
    for i, scale in enumerate(fx["hand_scale"]):
        theta, x, s = fx["hand_theta"][i], fx["hand_x"][i], float(scale)
        want = fx["hand_expected"][i]
        what = f"reward_fn(theta={theta.tolist()}, x={x.tolist()}, scale={s:g})"
        got = _array(fn(theta.copy(), x.copy(), s), want.shape, what, unit=", one per item")
        if np.allclose(got, want, atol=1e-12, rtol=0.0):
            continue
        if _like(got, fx["hand_no_clip"][i]):
            hint = (
                "These are the unclipped values 0.5 + scale theta_a^T x: clip them to "
                "[0.02, 0.98] with np.clip."
            )
        elif _like(got, fx["hand_no_intercept"][i]):
            hint = "Every item's click probability starts from 0.5: add it before the clip."
        elif _like(got, fx["hand_no_scale"][i]):
            hint = "Multiply theta @ x by `scale` before adding 0.5 (RecSim's is 0.15)."
        else:
            hint = (
                "theta is (K, d) and x is (d,), so theta @ x has one entry per item; scale it, "
                "add 0.5 and clip to [0.02, 0.98]."
            )
        raise CheckFailed(f"{what} = {_fmt(got)}; the click model gives {_fmt(want)}. {hint}")
    env = RecSim(K, D_RAW, seed=CATALOG)
    x, want = fx["recsim_x"], fx["recsim_expected"]
    what = "reward_fn(env.theta, x, env.scale) on RecSim's catalog (seed 0)"
    got = _array(fn(env.theta.copy(), x.copy(), env.scale), (K,), what, unit=", one per item")
    if not np.allclose(got, want, atol=1e-12, rtol=0.0):
        raise CheckFailed(
            f"{what} is off by up to {np.abs(got - want).max():.3g} from env.expected_reward(x) "
            f"for one user. Expected {_fmt(want)}."
        )


# ------------------------------------------------------------------ 2 Ridge
def check_ridge(cls: Callable[..., Any]) -> None:
    """Exercise 2: Ridge's A, b, A_inv and predict after seven updates, for two penalties."""
    fx = expected("m05_ridge")
    X, r, probes = fx["X"], fx["r"], fx["probes"]
    n, d = X.shape
    for j, lam in enumerate(float(v) for v in fx["lams"]):
        name = f"Ridge({d}, lam={lam:g})"
        m = cls(d, lam)
        p0 = _number(m.predict(probes[0].copy()), f"{name}.predict(x) before any update")
        if p0 != 0.0:
            raise CheckFailed(
                f"{name}.predict(x) is {p0:.6g} before any update; with b = 0 the prediction "
                "is 0. predict returns (A_inv @ b) @ x: the estimate A_inv b times x."
            )
        for x, reward in zip(X, r, strict=True):
            m.update(x.copy(), float(reward))
        what = f"after {n} updates of {name}"
        count = getattr(m, "n", None)
        if count != n:
            raise CheckFailed(f"n is {count!r} {what}; update should add 1 to n each time.")
        A = _array(getattr(m, "A", None), (d, d), f"A {what}")
        if not np.allclose(A, fx["A"][j], atol=1e-12, rtol=0.0):
            raise CheckFailed(
                f"A {what} is off by up to {np.abs(A - fx['A'][j]).max():.3g} from "
                "lam I + sum x x^T. update adds np.outer(x, x) to A."
            )
        b = _array(getattr(m, "b", None), (d,), f"b {what}")
        if not np.allclose(b, fx["b"], atol=1e-12, rtol=0.0):
            hint = (
                "That is the sum of the x's: add r * x, the click times the features."
                if _like(b, fx["b_without_reward"])
                else "update adds r * x to b."
            )
            raise CheckFailed(f"b {what} is {_fmt(b)}; sum r x is {_fmt(fx['b'])}. {hint}")
        A_inv = _array(getattr(m, "A_inv", None), (d, d), f"A_inv {what}")
        if not np.allclose(A_inv, fx["A_inv"][j], atol=1e-9, rtol=0.0):
            resid = float(np.abs(A_inv @ fx["A"][j] - np.eye(d)).max())
            if _like(A_inv, np.eye(d) / lam, atol=1e-12):
                hint = "A_inv has not changed since __init__: update must keep it current."
            else:
                hint = (
                    "With Ax = A_inv @ x, Sherman-Morrison subtracts np.outer(Ax, Ax) / "
                    "(1 + x @ Ax) from A_inv: a minus, and 1 plus x^T A^-1 x below."
                )
            raise CheckFailed(
                f"A_inv {what} is not the inverse of A: max|A_inv @ A - I| = {resid:.3g} "
                f"(a correct inverse gives about 1e-15). {hint}"
            )
        for k, p in enumerate(probes):
            want = float(fx["pred"][j, k])
            got = _number(m.predict(p.copy()), f"{name}.predict(x) {what}")
            if abs(got - want) > 1e-9:
                hint = (
                    "That is (A @ b) @ x: the estimate is A_inv @ b, not A @ b."
                    if abs(got - float(fx["pred_with_A"][j, k])) < 1e-9
                    else "predict returns (A_inv @ b) @ x, the estimate A_inv b times x."
                )
                raise CheckFailed(
                    f"{name}.predict({p.tolist()}) {what} is {got:.6g}; the closed form "
                    f"(lam I + X^T X)^-1 X^T r gives {want:.6g}. {hint}"
                )


# ------------------------------------------------------------------ 3 linucb_scores
_LINUCB_HINTS = {
    "variance": "Your scores add c x^T A^-1 x: that is a variance, not a width. Take its "
    "square root.",
    "A_not_inverse": "Your width uses A, not A^-1: the width is sqrt(x @ m.A_inv @ x).",
    "width_only": "Your scores are the widths alone: add each model's prediction, m.predict(x).",
    "c_inside_sqrt": "Your scores take sqrt(c x^T A^-1 x): c multiplies the width, outside the "
    "square root.",
}


def check_linucb_scores(fn: Callable[..., Any]) -> None:
    """Exercise 3: m.predict(x) + c sqrt(x^T A_a^-1 x) on fixed models, and c / sqrt(1 + n)."""
    fx = expected("m05_linucb")
    x = fx["x"]
    for i, c in enumerate(float(v) for v in fx["cs"]):
        models = [
            _Model(A, A_inv, b, n)
            for A, A_inv, b, n in zip(fx["A"], fx["A_inv"], fx["b"], fx["n"], strict=True)
        ]
        what = f"linucb_scores(models, x, c={c:g}) for three fixed models"
        got = _array(fn(models, x.copy(), c), (3,), what, unit=", one score per model")
        want = fx["scores"][i]
        if np.allclose(got, want, atol=1e-12, rtol=0.0):
            continue
        hint = next(
            (h for key, h in _LINUCB_HINTS.items() if _like(got, fx[f"wrong_{key}"][i])),
            "The score of item a is m.predict(x) + c * np.sqrt(x @ m.A_inv @ x).",
        )
        raise CheckFailed(
            f"{what} = {_fmt(got)}; m.predict(x) + c sqrt(x^T A_a^-1 x) gives {_fmt(want)}. {hint}"
        )
    u, ns = fx["u"], fx["unit_ns"]
    for i, c in enumerate(float(v) for v in fx["cs"]):
        models = [
            _Model(A, A_inv, np.zeros(3), n)
            for A, A_inv, n in zip(fx["unit_A"], fx["unit_A_inv"], ns, strict=True)
        ]
        what = f"linucb_scores(models, u, c={c:g})"
        got = _array(fn(models, u.copy(), c), (len(ns),), what, unit=", one score per model")
        want = fx["unit_scores"][i]
        if not np.allclose(got, want, atol=1e-12, rtol=0.0):
            hint = (
                " Yours grow with n: the width uses A^-1, which shrinks as data arrive, not A."
                if np.all(np.diff(got) > 0)
                else ""
            )
            raise CheckFailed(
                f"{what} = {_fmt(got)} for a unit u seen n = {ns.tolist()} times with no clicks "
                "(A = I + n u u^T, b = 0). The width is 1/sqrt(1 + n), so the scores are "
                f"c / sqrt(1 + n) = {_fmt(want)}.{hint}"
            )


# ------------------------------------------------------------------ 4 lints_sample
def check_lints_sample(fn: Callable[..., Any]) -> None:
    """Exercise 4: one draw from N(A^-1 b, v^2 A^-1), checked on a fixed-seed sample.

    (a) shape and finite values; (b) the same seed gives the same draw; (c) two seeds give
    two draws; (d) n_draws consecutive draws from default_rng(11): each mean within 6
    standard errors of the estimate A^-1 b, and the covariance, whitened by A^-1's Cholesky factor
    (L^-1 (S / v^2) L^-T), within 0.2 of I in every entry. A correct sampler's entries have
    a standard deviation of about sqrt(2 / n_draws) = 0.022 (diagonal) or 0.016, so 0.2 is
    about 9 of them, whatever order the sampler draws in.
    """
    fx = expected("m05_lints")
    A, A_inv, b, mean = fx["A"], fx["A_inv"], fx["b"], fx["estimate"]
    v, n_draws, d = float(fx["v"]), int(fx["n_draws"]), len(fx["b"])

    def draw(rng: np.random.Generator) -> np.ndarray:
        out = fn(_Model(A, A_inv, b, 12), v, rng)
        return _array(out, (d,), f"lints_sample(model, v={v:g}, rng)", unit=", one draw")

    one = draw(np.random.default_rng(7))
    if not np.array_equal(one, draw(np.random.default_rng(7))):
        raise CheckFailed(
            "Two calls with np.random.default_rng(7) gave different draws. Draw z from the "
            "`rng` you are given (rng.standard_normal), not from np.random or a new generator, "
            "so every run can be repeated."
        )
    if np.array_equal(one, draw(np.random.default_rng(8))):
        hint = (
            "It returns the estimate A_inv @ b itself, which makes LinTS greedy: add v L z."
            if np.allclose(one, mean, atol=1e-12, rtol=0.0)
            else "It ignores `rng`: draw z with rng.standard_normal(d)."
        )
        raise CheckFailed(
            f"np.random.default_rng(7) and default_rng(8) gave the same draw, {_fmt(one)}. {hint}"
        )
    rng = np.random.default_rng(11)
    draws = np.array([draw(rng) for _ in range(n_draws)])
    se = v * np.sqrt(np.diag(A_inv) / n_draws)
    z = (draws.mean(axis=0) - mean) / se
    if np.abs(z).max() > LINTS_MEAN_SE:
        raise CheckFailed(
            f"The mean of {n_draws} draws is {_fmt(draws.mean(axis=0))}; the estimate "
            f"A_inv @ b is {_fmt(mean)}, {np.abs(z).max():.1f} standard errors away "
            f"(a correct sampler is within {LINTS_MEAN_SE:g}). Center the draw at "
            "model.A_inv @ model.b."
        )
    L = np.linalg.cholesky(A_inv)
    Li = np.linalg.inv(L)
    white = Li @ (np.cov(draws.T) / v**2) @ Li.T
    dev = float(np.abs(white - np.eye(d)).max())
    if dev > LINTS_COV_TOL:
        wrong = {
            "v A^-1 (z scaled by sqrt(v), not v)": Li @ (A_inv / v) @ Li.T,
            "v^2 I (z not multiplied by the Cholesky factor of A_inv)": Li @ Li.T,
            "v^2 A (the Cholesky factor of A, not of A_inv)": Li @ A @ Li.T,
            "v^2 L^T L (the upper factor L^T: np.linalg.cholesky returns the lower L, "
            "scipy.linalg.cholesky the upper one unless lower=True)": Li @ L.T @ L @ Li.T,
        }
        # A wrong covariance's sampling error grows with its entries: compare relatively.
        near = [
            k
            for k, w in wrong.items()
            if np.abs(white - w).max() < LINTS_COV_TOL * max(1.0, float(np.abs(w).max()))
        ]
        hint = f" It looks like {near[0]}." if near else ""
        raise CheckFailed(
            f"The covariance of {n_draws} draws, whitened by A_inv (L^-1 (S / v^2) L^-T with "
            f"L L^T = A_inv), should be close to the identity; its worst entry is off by "
            f"{dev:.2f} (a correct sampler: below {LINTS_COV_TOL:g}).{hint} Draw "
            "A_inv @ b + v L z with L = np.linalg.cholesky(model.A_inv) and z = rng.standard_normal(d)."
        )


# ------------------------------------------------------------------ 5 verdicts
def check_verdicts(fn: Callable[..., Any]) -> None:
    """Exercise 5: verdicts on four fixed tables equal prl.evaluate.compare's (lower is better)."""
    fx = expected("m05_verdicts")
    methods = [str(m) for m in fx["methods"]]
    tables: dict[str, dict[str, list[float]]] = {}
    for i, regime in enumerate(str(r) for r in fx["regimes"]):
        k = int(fx["n_seeds"][i])
        tables[regime] = {
            m: [float(v) for v in fx["regrets"][i, j, :k]] for j, m in enumerate(methods)
        }
    budget = Budget(steps=int(fx["steps"]), seeds=int(fx["seeds"]))
    with contextlib.redirect_stdout(io.StringIO()):
        out = fn(copy.deepcopy(tables), budget)
    if out is None:
        raise CheckFailed("verdicts returned None. Did you forget `return`?")
    if not isinstance(out, Mapping):
        raise CheckFailed(
            "verdicts should return a dict keyed (regime, method); it returned "
            f"{type(out).__name__}."
        )
    for i, regime in enumerate(tables):
        for k, method in enumerate(("LinUCB", "LinTS")):
            key = (regime, method)
            if key not in out:
                raise CheckFailed(
                    f"verdicts' dict has no key {key!r}. It needs one entry per regime and "
                    "method, keyed (regime, method), for LinUCB and for LinTS."
                )
            want, got = str(fx["expected"][i, k]), out[key]
            if got == want:
                continue
            rows = tables[regime]
            v = compare(
                rows[method],
                rows["greedy"],
                names=(method, "greedy"),
                budget=budget,
                higher_is_better=False,
            )
            if got not in (method, "greedy", "tie", "single seed"):
                hint = (
                    "The word is the better method's name, 'tie' or 'single seed': map compare's "
                    f"winner 'A' to {method!r} and 'B' to 'greedy'."
                )
            elif want == "tie":
                hint = "Overlapping intervals are a tie, whatever the centers say."
            elif want == "single seed":
                hint = "With one seed per side there is no ranking: compare says 'single seed'."
            elif got == "tie":
                hint = "These intervals do not overlap, so one method is better."
            else:
                hint = "Regret is a cost, so lower is better: higher_is_better=False."
            raise CheckFailed(
                f"For the table {regime!r} ({len(rows['greedy'])} seeds per method), "
                f"{method} against greedy, verdicts says {got!r}; the workshop's rule says "
                f"{want!r}. {hint} compare's sentence: {v.sentence}"
            )


# ------------------------------------------------------------------ 6 log_policy
def _probabilities(out: Any, n: int, what: str) -> np.ndarray:
    probs = _array(out, (n,), what, unit=", one probability per item")
    if np.any(probs < 0):
        raise CheckFailed(f"{what} has a negative entry: {_fmt(probs)}.")
    total = float(probs.sum())
    if abs(total - 1.0) > 1e-12:
        raise CheckFailed(
            f"{what} sums to {total:.6g}, not 1: {_fmt(probs)}. Start from epsilon / K on every "
            "item and ADD 1 - epsilon to the best one (do not set it to 1 or to 1 - epsilon)."
        )
    return probs


def check_log_policy(fn: Callable[..., Any]) -> None:
    """Exercise 6: epsilon-greedy probabilities, exactly, then a RecSim log through env.log."""
    fx = expected("m05_logging")
    cases = [(fx["scores4"], float(fx["eps4"]), fx["expected4"])]
    cases += [
        (fx["scores10"], float(e), w) for e, w in zip(fx["eps10"], fx["expected10"], strict=True)
    ]
    for scores, eps, want in cases:
        n = len(scores)
        what = f"log_policy({_fmt(scores) if n <= 8 else f'{n} scores'}, epsilon={eps:g})"
        got = _probabilities(fn(scores.copy(), eps), n, what)
        if np.allclose(got, want, atol=1e-12, rtol=0.0):
            continue
        best = int(np.argmax(scores))
        others = np.delete(got, best)
        if eps > 0 and got[best] == 1.0:
            hint = "It ignores epsilon: every item needs epsilon / K, so the log can show it."
        elif eps > 0 and np.isclose(got[best], 1 - eps) and np.allclose(others, eps / (n - 1)):
            hint = (
                "It spreads epsilon over the other K - 1 items. The workshop's epsilon-greedy "
                "spreads epsilon / K over all K items, the best included, so the best gets "
                "1 - epsilon + epsilon / K."
            )
        elif int(np.argmax(got)) != best:
            hint = f"The best-scoring item is {best}: it gets 1 - epsilon more than the others."
        else:
            hint = "Every item gets epsilon / K, and the best-scoring item 1 - epsilon more."
        raise CheckFailed(f"{what} = {_fmt(got)}; epsilon-greedy gives {_fmt(want)}. {hint}")
    # In context: a logged run on RecSim, with fixed scores score_weights @ phi(x).
    W, eps, n_log = fx["score_weights"], float(fx["log_eps"]), int(fx["log_n"])
    env = RecSim(K, D_RAW, seed=CATALOG)

    def policy(x: np.ndarray, _rng: Any = None) -> Any:
        return fn(W @ _features(x), eps)

    x0 = env.sample_context(np.random.default_rng(57))
    _probabilities(policy(x0), K, f"log_policy(scores of one RecSim user, epsilon={eps:g})")
    try:
        log = env.log(policy, n_log, seed=int(fx["log_seed"]))
    except ValueError as exc:
        raise CheckFailed(f"env.log refused log_policy's output on RecSim: {exc}.") from None
    w = 1.0 / (K * log.P)
    se = w.std(ddof=1) / math.sqrt(n_log)
    if abs(w.mean() - 1.0) > WEIGHT_SE * se:
        raise CheckFailed(
            f"On a {n_log}-row RecSim log made through env.log with your log_policy, the mean "
            f"weight for the uniform target is {w.mean():.3f}; on a correct log it is 1 "
            f"(within {WEIGHT_SE:g} standard errors, {WEIGHT_SE * se:.3f})."
        )


# ------------------------------------------------------------------ 7 log_suspect
def _suspect_policy(W: np.ndarray, eps: float) -> Callable[..., np.ndarray]:
    """The checkpoint's behavior policy: epsilon-greedy over the fixed scores W @ phi(x)."""

    def policy(x: np.ndarray, _rng: Any = None) -> np.ndarray:
        probs = np.full(K, eps / K)
        probs[int(np.argmax(W @ _features(x)))] += 1.0 - eps
        return probs

    return policy


def check_log_suspect(fn: Callable[..., Any]) -> None:
    """Exercise 7: a log whose every propensity is the shown item's probability."""
    fx = expected("m05_logging")
    eps, n = float(fx["suspect_eps"]), int(fx["suspect_n"])
    policy = _suspect_policy(fx["score_weights"], eps)
    env = RecSim(K, D_RAW, seed=CATALOG)
    log = fn(env, policy, n, int(fx["suspect_seed"]))
    if log is None:
        raise CheckFailed("log_suspect returned None. Did you forget `return`?")
    fields = {"X": (n, D_RAW), "A": (n,), "R": (n,), "P": (n,), "probs": (n, K)}
    got = {}
    for name, shape in fields.items():
        if not hasattr(log, name):
            raise CheckFailed(
                f"log_suspect's result has no field {name!r}; return envs.LoggedBandit(X=..., "
                "A=..., R=..., P=..., probs=..., T=...)."
            )
        got[name] = _array(getattr(log, name), shape, f"log.{name}", unit=f" for {n} rounds")
    X, A, R, P, probs = (got[k] for k in fields)
    if not np.all((A == np.round(A)) & (A >= 0) & (A < K)):
        raise CheckFailed(f"log.A should hold item indices 0 to {K - 1} on every row.")
    A = A.astype(int)
    if not np.all((R == 0) | (R == 1)):
        raise CheckFailed("log.R should hold clicks, 0 or 1, on every row.")
    want_probs = np.array([policy(x) for x in X])
    if not np.allclose(probs, want_probs, atol=1e-12, rtol=0.0):
        raise CheckFailed(
            "log.probs[i] should be the behavior policy's probabilities for row i's user, "
            "policy(X[i], rng), on every row."
        )
    shown = probs[np.arange(n), A]
    bad = np.abs(P - shown) > 1e-12
    if bad.any():
        k = int(bad.sum())
        if np.allclose(P[bad], probs[bad].max(axis=1), atol=1e-12, rtol=0.0):
            msg = f"On {k} rows P is the best item's probability, but another item was shown."
        elif np.allclose(P, 1.0 / K, atol=1e-12, rtol=0.0):
            msg = "P is 1/K on every row: the uniform policy's probability, not the logger's."
        elif np.allclose(P, 1.0, atol=1e-12, rtol=0.0):
            msg = "P is 1 on every row, as if the shown item were certain."
        else:
            msg = f"On {k} of {n} rows P differs from the probability of the item shown."
        raise CheckFailed(
            f"{msg} The propensity is the probability that the logger showed the item it "
            "showed: probs[a] for the sampled a, row by row."
        )
    best = probs.argmax(axis=1)
    share, p_other = float(np.mean(A != best)), eps * (K - 1) / K
    se = math.sqrt(p_other * (1 - p_other) / n)
    if abs(share - p_other) > SHARE_SE * se:
        raise CheckFailed(
            f"Another item than the best was shown on {share:.1%} of the rows; an "
            f"epsilon-greedy logger with epsilon = {eps:g} and K = {K} shows one on about "
            f"{p_other:.0%} (epsilon (K - 1) / K). Sample the item from the policy's "
            "probabilities with rng.choice(len(probs), p=probs), using the rng you made."
        )
    w = (1.0 / K) / P
    se_w = w.std(ddof=1) / math.sqrt(n)
    if abs(w.mean() - 1.0) > WEIGHT_SE * se_w:
        raise CheckFailed(
            f"The mean weight for the uniform target is {w.mean():.3f} on this log; on a "
            f"correct log it is 1 (within {WEIGHT_SE:g} standard errors, {WEIGHT_SE * se_w:.3f})."
        )


__all__ = [
    "check_lints_sample",
    "check_linucb_scores",
    "check_log_policy",
    "check_log_suspect",
    "check_reward_fn",
    "check_ridge",
    "check_verdicts",
]
