"""Checkpoints for Module 4 (bandits from first principles).

Exact checks compare the participant's function with fixtures written by
scripts/make_expected.py: fixed counts and sums, exact probabilities, and recorded runs
replayed round by round. A replay hands the participant's function the counts and sums
that the reference run had before each round and requires the same arm. Nothing here
plays a bandit, keeps a policy's statistics or computes an index, a posterior or a
verdict of its own: the fixture code does that, and this module compares.

The one stochastic checkpoint, 3.3, is a StochasticCheck: the participant's thompson_step,
played by the participant's run_bandit, with thresholds from scripts/threshold_protocol.py.

`argmax_tie` and `reward_stack` are the lab's provided cells (the tie rule and the reward
stack), copied here as plumbing so that the checks use exactly the lab's rule.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from ..evaluate import compare
from . import CheckFailed, StochasticCheck, expected

#: The lab's click rates r(a) of headlines 0, 1 and 2 (headline 2 is the best).
R_MEAN = np.array([0.4, 0.3, 0.5])
GAPS = R_MEAN.max() - R_MEAN  # 0.1, 0.2 and 0
N_ARMS = len(R_MEAN)
VERDICTS = ("A", "B", "tie", "single seed")
DIRECTIONS = ("same", "changes")


# ------------------------------------------------------------------ the lab's provided plumbing
def argmax_tie(x: Any) -> int:
    """The first index within rounding of the max. An infinite entry wins at once."""
    x = np.asarray(x, dtype=float)
    top = x.max()
    if np.isinf(top):
        return int(np.flatnonzero(x == top)[0])
    tol = 1e-9 * (1 + np.abs(x).max())
    return int(np.flatnonzero(x >= top - tol)[0])


def reward_stack(r_mean: Any, T: int, seed: int, scale: float = 1.0) -> np.ndarray:
    """Bernoulli rewards drawn up front, 0 or `scale`. stack[n, a] is the reward of the
    (n + 1)-th pull of arm a. Its stream is seeded by [seed, 1], apart from the policy's."""
    rng = np.random.default_rng([seed, 1])
    r_mean = np.asarray(r_mean, dtype=float)
    return scale * (rng.random((T, len(r_mean))) < r_mean)


# ------------------------------------------------------------------ helpers
def _ints(x: Any) -> list:
    return np.asarray(x).astype(int).tolist()


def _arm(out: Any, n_arms: int, what: str) -> int:
    """A policy's answer: one arm, as an integer in range."""
    if out is None:
        raise CheckFailed(f"{what} returned None. Did you forget to `return` the arm?")
    if isinstance(out, (bool, np.bool_)) or not isinstance(out, (int, np.integer)):
        raise CheckFailed(
            f"{what} returned {out!r} ({type(out).__name__}); it should return one arm, an "
            "integer such as int(np.argmin(counts)) or argmax_tie(...)."
        )
    if not 0 <= int(out) < n_arms:
        raise CheckFailed(f"{what} returned arm {int(out)}; the arms are 0 to {n_arms - 1}.")
    return int(out)


def _vector(out: Any, n: int, what: str, *, allow_inf: bool = False) -> np.ndarray:
    if out is None:
        raise CheckFailed(f"{what} returned None. Did you forget to `return` it?")
    try:
        x = np.asarray(out, dtype=float)
    except (TypeError, ValueError):
        raise CheckFailed(f"{what} should return an array of {n} numbers.") from None
    if x.shape != (n,):
        raise CheckFailed(f"{what} should return shape ({n},), one value per arm; got {x.shape}.")
    if np.any(np.isnan(x)) or (not allow_inf and not np.all(np.isfinite(x))):
        raise CheckFailed(f"{what} contains NaN or infinity: {np.round(x, 4).tolist()}.")
    return x


def _pair(out: Any, n: int, what: str, names: str) -> tuple[np.ndarray, np.ndarray]:
    if out is None:
        raise CheckFailed(f"{what} returned None. Did you forget `return {names}`?")
    if not isinstance(out, (tuple, list)) or len(out) != 2:
        raise CheckFailed(f"{what} should return a pair: {names}.")
    return _vector(out[0], n, f"{what}[0]"), _vector(out[1], n, f"{what}[1]")


def _run_output(out: Any, T: int, n_arms: int) -> tuple[np.ndarray, Any]:
    """run_bandit's (arms, regret): arms an integer array of length T with values in range."""
    if out is None:
        raise CheckFailed("run_bandit returned None. Did you forget `return arms, regret`?")
    if not isinstance(out, (tuple, list)) or len(out) != 2:
        raise CheckFailed("run_bandit should return a pair (arms, regret).")
    arms = np.asarray(out[0])
    if arms.shape != (T,):
        raise CheckFailed(
            f"run_bandit's arms should have shape ({T},), one arm per round of the stack; "
            f"got {arms.shape}."
        )
    if not np.issubdtype(arms.dtype, np.integer):
        raise CheckFailed(
            f"run_bandit's arms should be integers (np.zeros(T, dtype=int)); got dtype {arms.dtype}."
        )
    if arms.size and (arms.min() < 0 or arms.max() >= n_arms):
        raise CheckFailed(f"run_bandit's arms should lie in 0 to {n_arms - 1}.")
    return arms.astype(np.int64), out[1]


# ------------------------------------------------------------------ 1.1 the bandit loop (exact)
class _Spy:
    """A policy that checks what run_bandit shows it in every round, then pulls a random arm
    with the generator it was given (rng.integers(n_arms))."""

    def __init__(self, stack: np.ndarray):
        self.stack = stack
        self.n = stack.shape[1]
        self.tally = np.zeros(self.n, dtype=np.int64)
        self.arms: list[int] = []
        self.rng: Any = None
        self.problem: CheckFailed | None = None

    def __call__(self, counts: Any, sums: Any, t: Any, rng: Any) -> int:
        try:
            self._inspect(counts, sums, t, rng)
        except CheckFailed as exc:
            self.problem = self.problem or exc
            raise
        a = int(rng.integers(self.n))
        self.tally[a] += 1
        self.arms.append(a)
        return a

    def _inspect(self, counts: Any, sums: Any, t: Any, rng: Any) -> None:
        r = len(self.arms)  # this round's index
        if not isinstance(rng, np.random.Generator):
            raise CheckFailed(
                f"round {r}: run_bandit passed the policy {type(rng).__name__}, not a NumPy "
                "Generator. Make rng = np.random.default_rng(seed) once, before the loop."
            )
        if self.rng is None:
            self.rng = rng
        elif rng is not self.rng:
            raise CheckFailed(
                f"round {r}: the policy got a different generator than in round 0. Make "
                "rng = np.random.default_rng(seed) once, before the loop: a new generator in "
                "every round repeats the same coin flips."
            )
        if isinstance(t, (bool, np.bool_)) or not isinstance(t, (int, np.integer)) or t != r:
            raise CheckFailed(
                f"round {r}: the policy was told t = {t!r}; t is the number of rounds played "
                "so far, starting at 0."
            )
        c = np.asarray(counts)
        if c.shape != (self.n,) or not np.array_equal(c, self.tally):
            raise CheckFailed(
                f"round {r}: the policy was shown counts {np.asarray(counts).tolist()}, but the "
                f"arms pulled so far give {self.tally.tolist()}. counts[a] must grow by 1 each "
                "time arm a is pulled."
            )
        want = np.array([self.stack[: self.tally[a], a].sum() for a in range(self.n)])
        s = np.asarray(sums, dtype=float)
        if s.shape != (self.n,) or not np.allclose(s, want, rtol=0, atol=1e-9):
            raise CheckFailed(
                f"round {r}: the policy was shown sums {np.asarray(sums).tolist()}, but the "
                f"rewards of the pulls so far add up to {want.tolist()}. A pull of arm a with "
                "counts[a] = n so far must take stack[n, a], the top of its own column: read "
                "the reward before you increment counts[a]."
            )


def check_run_bandit(run_bandit: Callable[..., Any]) -> None:
    fx = expected("m04_bandit")
    for i in range(int(fx["n_cases"])):
        r_mean, seed, stack = fx[f"r_mean_{i}"], int(fx[f"seed_{i}"]), fx[f"stack_{i}"]
        T, n = stack.shape
        what = f"run_bandit(policy, stack, r_mean={r_mean.tolist()}, seed={seed}) on {T} rounds"
        spy = _Spy(stack)
        out = run_bandit(spy, stack.copy(), r_mean.copy(), seed)
        if spy.problem is not None:
            raise spy.problem
        arms, regret = _run_output(out, T, n)
        if len(spy.arms) != T:
            raise CheckFailed(
                f"{what}: the policy was called {len(spy.arms)} times; play one round per row "
                f"of the stack, T = {T}."
            )
        if arms.tolist() != spy.arms:
            t = int(np.flatnonzero(arms != np.array(spy.arms))[0])
            raise CheckFailed(
                f"{what}: arms[{t}] is {arms[t]}, but the policy chose arm {spy.arms[t]} in "
                f"round {t}. arms[t] is the arm the policy returned in round t."
            )
        want = fx[f"arms_{i}"]
        if not np.array_equal(arms, want):
            t = int(np.flatnonzero(arms != want)[0])
            raise CheckFailed(
                f"{what}: the policy's coin flips differ from those of "
                f"np.random.default_rng({seed}), made once before the loop (first difference "
                f"in round {t}). Seed the run's generator with `seed`, and give the policy "
                "that same generator in every round."
            )
        got = _vector(regret, T, f"{what}: regret")
        want = fx[f"regret_{i}"]
        if not np.allclose(got, want, rtol=0, atol=1e-9):
            t = int(np.flatnonzero(~np.isclose(got, want, rtol=0, atol=1e-9))[0])
            raise CheckFailed(
                f"{what}: regret[{t}] is {got[t]:.4g}; expected {want[t]:.4g} (the first "
                "difference). Pseudo-regret adds the gap of the arm pulled, max(r_mean) - "
                "r_mean[a], in every round, and regret[t] is the running total after round t: "
                "np.cumsum(gaps[arms]). It is not r* minus the reward received."
            )


# ------------------------------------------------------------------ 1.2 epsilon-greedy (exact)
def _shares(fn: Callable[..., Any], counts, sums, t, n: int, seed: int, what: str, **kw):
    rng = np.random.default_rng(seed)
    picks = [
        _arm(fn(counts.copy(), sums.copy(), t, rng, **kw), len(counts), what) for _ in range(n)
    ]
    return np.bincount(picks, minlength=len(counts)) / n


def check_eps_greedy(eps_greedy: Callable[..., Any]) -> None:
    fx = expected("m04_policy")
    # 1. Fixed states: an unpulled arm comes first, before any coin flip; then the best mean.
    for counts, sums, want in zip(
        fx["choice_counts"], fx["choice_sums"], fx["choice_want"], strict=True
    ):
        t = int(counts.sum())
        unpulled = bool(np.any(counts == 0))
        for epsilon in (0.0, 1.0) if unpulled else (0.0,):
            what = f"eps_greedy(counts={_ints(counts)}, sums={_ints(sums)}, t={t}, rng, epsilon={epsilon:g})"
            rng = np.random.default_rng(7)
            for _ in range(12):  # 12 calls: a random arm cannot pass by luck
                got = _arm(eps_greedy(counts.copy(), sums.copy(), t, rng, epsilon=epsilon), 3, what)
                if got == want:
                    continue
                if unpulled:
                    hint = (
                        f"arm {want} has never been pulled (counts 0). Pull it first, before "
                        "any coin flip: return the first such arm, int(np.argmin(counts))."
                    )
                elif got == argmax_tie(sums):
                    hint = (
                        f"arm {got} has the largest total reward, but arm {want} has the best "
                        f"mean, {sums[want]:g}/{counts[want]} (greedy compares sums / counts)."
                    )
                elif np.isclose(sums[got] / counts[got], sums[want] / counts[want]):
                    hint = (
                        f"the means {np.round(sums / counts, 3).tolist()} tie; ties go to the "
                        "lowest arm: argmax_tie(sums / counts)."
                    )
                else:
                    hint = (
                        "with epsilon = 0 it never explores: draw u = rng.random() and explore "
                        "only when u < epsilon; otherwise return argmax_tie(sums / counts)."
                    )
                raise CheckFailed(f"{what} returned arm {got}; expected arm {want}. Hint: {hint}")
    # 2. The share of each arm over 4000 calls from default_rng(0), against the exact share.
    counts, sums, n = fx["share_counts"], fx["share_sums"], int(fx["share_n"])
    greedy = int(np.argmax(fx["share_exact"][-1]))  # the arm with the best mean
    for epsilon, exact in zip(fx["share_eps"], fx["share_exact"], strict=True):
        what = f"eps_greedy(counts={_ints(counts)}, sums={_ints(sums)}, epsilon={epsilon:g})"
        got = _shares(eps_greedy, counts, sums, 30, n, 0, what, epsilon=float(epsilon))
        tol = 0.04 if epsilon == 1 else 0.03
        if np.all(np.abs(got - exact) <= tol):
            continue
        if got[greedy] < exact[greedy] - tol and abs(got[greedy] - (1 - epsilon)) <= tol:
            hint = (
                "the random pull must choose among all arms, rng.integers(len(counts)), the "
                f"greedy one included, so arm {greedy}'s share is 1 - epsilon + epsilon / 3."
            )
        elif epsilon == 1:
            hint = (
                "with epsilon = 1 every pull after the first ones is random, so each arm "
                "should take about 1/3. Draw u = rng.random() and explore when u < epsilon."
            )
        else:
            hint = (
                f"explore with probability epsilon = {epsilon:g}: when u = rng.random() is "
                "below epsilon, return rng.integers(len(counts)); otherwise the best mean."
            )
        raise CheckFailed(
            f"{what}, {n} calls with one generator: the arms' shares are "
            f"{np.round(got, 3).tolist()}; expected about {np.round(exact, 3).tolist()} "
            f"(within {tol:g}). Hint: {hint}"
        )
    # 3. Greedy on the trap instance, replayed round by round.
    gx = expected("m04_greedy_replay")
    for seed in gx["seeds"]:
        seed = int(seed)
        arms, c_all, s_all = (gx[f"{k}_{seed}"] for k in ("arms", "counts_before", "sums_before"))
        rng = np.random.default_rng(seed)
        for t, (c, s, want) in enumerate(zip(c_all, s_all, arms, strict=True)):
            what = f"eps_greedy(counts={_ints(c)}, sums={_ints(s)}, t={t}, rng, epsilon=0.0)"
            got = _arm(eps_greedy(c.copy(), s.copy(), t, rng, epsilon=0.0), 3, what)
            if got != want:
                raise CheckFailed(
                    f"Greedy (epsilon = 0) replayed on reward_stack(R_MEAN, {len(arms)}, "
                    f"seed={seed}): in round {t}, {what} returned arm {got}; greedy pulls arm "
                    f"{int(want)}."
                )
    r1, r17 = float(gx["regret_1"][-1]), float(gx["regret_17"][-1])
    print("Greedy, replayed on two recorded stacks: headline 2 is never shown after round 2.")
    print(
        f"  The regret grows by 0.1 per round to {r1:.1f} (seed 1), by 0.2 to {r17:.1f} (seed 17)."
    )


# ------------------------------------------------------------------ 2.1 the UCB1 index (exact)
_UCB1_HINTS = {
    "log10": "you used log base 10; UCB1's log is the natural log, np.log.",
    "no_sqrt": "the bonus is the square root, np.sqrt(2 * np.log(t) / counts).",
    "log_t_plus_one": "use ln t, with t the number of rounds played so far, not t + 1.",
    "unpulled_zero": "an arm never pulled gets +inf, not 0, so it is pulled first.",
    "sums_not_means": "the bonus goes on the mean, sums / counts, not on the total reward.",
}


def _same_index(got: np.ndarray, want: np.ndarray) -> bool:
    inf_g, inf_w = np.isposinf(got), np.isposinf(want)
    if not np.array_equal(inf_g, inf_w) or np.any(np.isneginf(got)) or np.any(np.isnan(got)):
        return False
    return bool(np.allclose(got[~inf_g], want[~inf_w], rtol=0, atol=1e-12))


def _index_hint(got: np.ndarray, fx: Mapping[str, np.ndarray], i: int) -> str:
    for key, hint in _UCB1_HINTS.items():
        if _same_index(got, fx[key][i]):
            return hint
    return (
        "U = sums / counts + sqrt(2 ln t / counts) for every arm with counts > 0; +inf otherwise."
    )


def check_ucb1_index(ucb1_index: Callable[..., Any]) -> None:
    fx = expected("m04_ucb1")
    for i, (counts, sums, t) in enumerate(zip(fx["counts"], fx["sums"], fx["t"], strict=True)):
        t = int(t)
        what = f"ucb1_index(counts={_ints(counts)}, sums={_ints(sums)}, t={t})"
        with np.errstate(divide="ignore", invalid="ignore"):
            out = ucb1_index(counts.copy(), sums.copy(), t)
        got = _vector(out, len(counts), what, allow_inf=True)
        want = fx["index"][i]
        if np.any(np.isnan(got)) or np.any(np.isneginf(got)):
            raise CheckFailed(
                f"{what} returned {got.tolist()}. At t = 0 no arm has been pulled and "
                "log(0) = -inf: compute the bonus only for arms with counts > 0, and only "
                "when there is one."
            )
        if _same_index(got, want):
            continue
        raise CheckFailed(
            f"{what} returned {np.round(got, 5).tolist()}; expected "
            f"{np.round(want, 5).tolist()}. Hint: {_index_hint(got, fx, i)}"
        )


# ------------------------------------------------------------------ 2.2 UCB1's recorded run
def check_ucb1_counts(ucb1_index: Callable[..., Any]) -> None:
    fx = expected("m04_ucb1_replay")
    arms, seed = fx["arms"], int(fx["seed"])
    T = len(arms)
    for t, (c, s, want) in enumerate(
        zip(fx["counts_before"], fx["sums_before"], arms, strict=True)
    ):
        what = f"ucb1_index(counts={_ints(c)}, sums={_ints(s)}, t={t})"
        with np.errstate(divide="ignore", invalid="ignore"):
            index = _vector(ucb1_index(c.copy(), s.copy(), t), len(c), what, allow_inf=True)
        got = argmax_tie(index)
        if got != want:
            raise CheckFailed(
                f"UCB1's recorded run on reward_stack(R_MEAN, {T}, seed={seed}), round {t}: "
                f"counts {_ints(c)}, sums {_ints(s)}. UCB1 pulls arm {int(want)}; your index "
                f"{np.round(index, 4).tolist()} picks arm {got}. Hint: compare one round by "
                "hand: sums / counts + sqrt(2 ln t / counts), with t the rounds played so far."
            )
    print(
        f"UCB1 replayed for {T} rounds on reward_stack(R_MEAN, {T}, seed={seed}): "
        f"final counts {_ints(fx['final_counts'])}."
    )


# ------------------------------------------------------------------ 3.1 the Beta posterior
def check_beta_posterior(beta_posterior: Callable[..., Any]) -> None:
    fx = expected("m04_beta")
    for i, (counts, sums) in enumerate(zip(fx["counts"], fx["sums"], strict=True)):
        what = f"beta_posterior(counts={_ints(counts)}, sums={_ints(sums)})"
        got_s, got_f = _pair(beta_posterior(counts.copy(), sums.copy()), 3, what, "(1 + S, 1 + F)")
        want_s, want_f = fx["shape_s"][i], fx["shape_f"][i]
        if np.allclose(got_s, want_s, rtol=0, atol=1e-12) and np.allclose(
            got_f, want_f, rtol=0, atol=1e-12
        ):
            continue
        if np.allclose(got_s, want_f) and np.allclose(got_f, want_s):
            hint = "the two are swapped: the first parameter counts clicks, 1 + S."
        elif np.allclose(got_s, fx["no_prior_s"][i]) and np.allclose(got_f, fx["no_prior_f"][i]):
            hint = "add the prior's 1 to both: the uniform prior is Beta(1, 1)."
        elif np.allclose(got_f, fx["counts_f"][i]):
            hint = "F is the number of non-clicks, counts - sums, not the number of pulls."
        else:
            hint = "S = sums (the clicks) and F = counts - sums (the non-clicks)."
        raise CheckFailed(
            f"{what} returned ({got_s.tolist()}, {got_f.tolist()}); expected "
            f"({want_s.tolist()}, {want_f.tolist()}), Beta(1 + S, 1 + F). Hint: {hint}"
        )


# ------------------------------------------------------------------ 3.2 Thompson's choices
def check_thompson_choices(thompson_step: Callable[..., Any]) -> None:
    fx = expected("m04_ts_shares")
    t, n, tol = int(fx["t"]), int(fx["n"]), 0.04
    for i, (counts, sums) in enumerate(zip(fx["counts"], fx["sums"], strict=True)):
        what = f"thompson_step(counts={_ints(counts)}, sums={_ints(sums)}, t={t}, rng)"
        got = _shares(thompson_step, counts, sums, t, n, 0, what)
        exact = fx["exact"][i]
        if np.all(np.abs(got - exact) <= tol):
            continue
        if np.all(np.abs(got - fx["posterior_mean"][i]) <= tol):
            hint = (
                "you pulled the arm with the largest posterior mean every time: that is "
                "greedy. Draw one rate per arm, rng.beta(1 + S, 1 + F), and take the largest."
            )
        elif np.all(np.abs(got - fx["sample_reward"][i]) <= tol):
            hint = (
                "you drew a 0/1 outcome per arm, so most draws tie. Draw a rate from each "
                "posterior, rng.beta(1 + S, 1 + F), and pull the largest."
            )
        elif np.all(np.abs(got - 1 / 3) <= tol):
            hint = "your choices do not depend on the counts. Draw from each arm's posterior."
        elif int(np.argmax(got)) == int(np.argmin(exact)):
            hint = (
                "the arm least likely to be best wins most often: check the order of the shape "
                "parameters, rng.beta(1 + S, 1 + F), clicks first."
            )
        else:
            hint = "draw one rate per arm, rng.beta(*beta_posterior(counts, sums)); argmax_tie."
        raise CheckFailed(
            f"{what}, {n} calls with one generator: the arms' shares are "
            f"{np.round(got, 3).tolist()}. Thompson sampling pulls each arm with its posterior "
            f"probability of being the best, {np.round(exact, 3).tolist()} (within {tol:g}). "
            f"Hint: {hint}"
        )


# ------------------------------------------------------------------ 3.3 Thompson's regret
def _ts_regret(
    thompson_step: Callable[..., Any],
    run_bandit: Callable[..., Any],
    *,
    seed: int,
    T: int,
    runs: int,
) -> float:
    """Mean pseudo-regret over `runs` runs of T rounds, run i on stack and policy seed
    100 * seed + i. The regret is taken from the arms, so a wrong regret array cannot help."""
    total = 0.0
    for i in range(runs):
        k = 100 * seed + i
        arms, _ = _run_output(
            run_bandit(thompson_step, reward_stack(R_MEAN, T, k), R_MEAN, k), T, N_ARMS
        )
        total += float(GAPS[arms].sum())
    return total / runs


check_thompson_regret = StochasticCheck(
    ex=3,
    functions=("thompson_step", "run_bandit"),
    metric=_ts_regret,
    live={"T": 2000, "runs": 10},
    quick={"T": 1000, "runs": 5},
    higher_is_better=False,
    min_gap=20.0,
    what="mean pseudo-regret of Thompson sampling over the checkpoint's runs (from the arms)",
    thresholds={
        # threshold_protocol.py, live budget {"T": 2000, "runs": 10}, N = 100 seeds (0-99),
        # 2026-10-24, local (Apple M1 Pro, arm64). Solution p99 38.4; strongest wrong version
        # mutant 'sample-reward' p1 189.7; gap 151.3 (minimum 20); threshold: the midpoint
        # 114.068, rounded to 110.0. 0 of 100 solution seeds fail: flake rate at most about 3/N
        # = 3.0% (rule of three).
        "live": (
            110.0,
            "threshold protocol, 100 seeds: runs/2026-10-24-local-threshold-m04-check_thompson_regret-live-n100-exp.json",
        ),
        # threshold_protocol.py, quick budget {"T": 1000, "runs": 5}, N = 100 seeds (0-99),
        # 2026-10-24, local (Apple M1 Pro, arm64). Solution p99 38.07; strongest wrong version
        # mutant 'sample-reward' p1 92.31; gap 54.24 (minimum 20); threshold: the midpoint
        # 65.1901, rounded to 65.0. 0 of 100 solution seeds fail: flake rate at most about 3/N =
        # 3.0% (rule of three).
        "quick": (
            65.0,
            "threshold protocol, 100 seeds: runs/2026-10-24-local-threshold-m04-check_thompson_regret-quick-n100-exp.json",
        ),
    },
)


# ------------------------------------------------------------------ 4 the tie rule (exact)
_VERDICT_HINTS = {
    "single": 'with one seed on either side the rule makes no ranking: return "single seed".',
    "ranges-3-disjoint": "lower regret is better: the side with the lower center wins.",
    "ranges-3-overlap": (
        "with 2 to 4 seeds on the smaller side the rule compares the [min, max] ranges; these "
        'overlap, so the verdict is "tie".'
    ),
    "clear-20": "lower regret is better: the side with the lower IQM wins.",
    "overlap-20": (
        'the two 95% bootstrap intervals of the IQM overlap, so the verdict is "tie", '
        "whatever the centers say."
    ),
    "outlier-range-overlap": (
        "with 5 or more seeds compare the IQMs' intervals, bootstrap_ci(x, iqm), not the "
        "[min, max] ranges: one outlier seed makes the ranges overlap, not the intervals."
    ),
    "outliers-mean-vs-iqm": (
        "bootstrap the IQM, bootstrap_ci(x, iqm), not the mean: three outlier seeds widen "
        "the mean's interval until it overlaps."
    ),
}


def _word(out: Any, what: str) -> str:
    if out not in VERDICTS:
        raise CheckFailed(f"{what} returned {out!r}; it should return one of {list(VERDICTS)}.")
    return str(out)


def _seeds(x: np.ndarray) -> str:
    if x.size <= 6:
        return str(x.tolist())
    return f"{x.size} seeds from {x.min():g} to {x.max():g}"


def _summary_words(sa: Any, sb: Any, n: int) -> str:
    kind = "IQM" if n >= 5 else ("mean" if n >= 2 else "value")
    span = "95% bootstrap interval" if n >= 5 else "range"
    return (
        f"{kind} a {sa[0]:.2f} [{sa[1]:.2f}, {sa[2]:.2f}] vs {kind} b {sb[0]:.2f} "
        f"[{sb[1]:.2f}, {sb[2]:.2f}], {span}s"
    )


def _check_pair(verdict: Callable[..., Any], runs: Mapping, a_name: str, b_name: str, src: str):
    """verdict on the lab's own per-seed regrets, both orders, against compare()."""
    if a_name not in runs or b_name not in runs:
        raise CheckFailed(f"{src} has no {a_name!r} or {b_name!r}: rerun the run cells above.")
    xa = np.asarray(runs[a_name], dtype=float).ravel()
    xb = np.asarray(runs[b_name], dtype=float).ravel()
    rule = compare(xa, xb, higher_is_better=False)
    swap = {"A": "B", "B": "A", "tie": "tie", "single seed": "single seed"}
    for (p, q), (pn, qn), want, (sa, sb) in (
        ((xa, xb), (a_name, b_name), rule.winner, (rule.summary_a, rule.summary_b)),
        ((xb, xa), (b_name, a_name), swap[rule.winner], (rule.summary_b, rule.summary_a)),
    ):
        call = f"verdict({src}[{pn!r}], {src}[{qn!r}])"
        got = _word(verdict(p.copy(), q.copy()), call)
        if got != want:
            raise CheckFailed(
                f"on your own runs, {call} is {got!r}; the rule (prl.evaluate.compare) gives "
                f"{want!r} ({_summary_words(sa, sb, min(p.size, q.size))})."
            )


def check_verdict(verdict: Callable[..., Any], final: Any, late: Any = None) -> None:
    """The fixed cases, then the lab's own runs: `final` (5000 rounds) and, if given, `late`
    (epsilon-greedy and UCB1 after the late horizon)."""
    fx = expected("m04_verdict")
    for i, name in enumerate(fx["names"]):
        name = str(name)
        a, b = fx["a"][i, : int(fx["n_a"][i])], fx["b"][i, : int(fx["n_b"][i])]
        what = f"verdict(a, b) with {a.size} vs {b.size} seeds"
        got, want = _word(verdict(a.copy(), b.copy()), what), str(fx["winner"][i])
        if got != want:
            sa, sb = fx["summary"][i]
            raise CheckFailed(
                f"for the per-seed regrets a = {_seeds(a)} and b = {_seeds(b)}, your verdict "
                f"is {got!r}; the rule gives {want!r} ({_summary_words(sa, sb, min(a.size, b.size))}). "
                f"Hint: {_VERDICT_HINTS[name]}"
            )
    if not isinstance(final, Mapping):
        raise CheckFailed("final should be the lab's dict of per-seed regrets per method.")
    for a_name, b_name in (("TS", "UCB1"), ("TS", "eps-greedy"), ("UCB1", "eps-greedy")):
        _check_pair(verdict, final, a_name, b_name, "final")
    if late is not None:
        if not isinstance(late, Mapping):
            raise CheckFailed(
                "late should be the lab's dict of per-seed regrets at the late horizon."
            )
        _check_pair(verdict, late, "UCB1", "eps-greedy", "late")


# ------------------------------------------------------------------ 5 reward scale (exact)
_RANGE_HINTS = {
    "bonus_not_scaled": (
        "your index is ucb1_index on the raw sums: the bonus must grow with the range. Divide the "
        "sums by B, take ucb1_index, and multiply the result by B."
    ),
    "mean_scaled_twice": (
        "the mean came out B times too large: divide the sums by B before ucb1_index, so that "
        "multiplying the index by B brings the mean back to reward units."
    ),
}


def check_ucb1_for_range(ucb1_for_range: Callable[..., Any]) -> None:
    """Exact: UCB1's index for rewards in [0, B] on fixed counts, sums, t and B."""
    fx = expected("m04_ucb1_for_range")
    rows = zip(fx["counts"], fx["sums"], fx["t"], fx["B"], strict=True)
    for i, (counts, sums, t, B) in enumerate(rows):
        t, B = int(t), float(B)
        what = f"ucb1_for_range(counts={_ints(counts)}, sums={_ints(sums)}, t={t}, B={B:g})"
        with np.errstate(divide="ignore", invalid="ignore"):
            out = ucb1_for_range(counts.copy(), sums.copy(), t, B)
        got = _vector(out, len(counts), what, allow_inf=True)
        if np.any(np.isnan(got)) or np.any(np.isneginf(got)):
            raise CheckFailed(
                f"{what} returned {got.tolist()}. An arm never pulled keeps the index +inf, and "
                "at t = 0 no log may be taken: build on your ucb1_index."
            )
        want = fx["index"][i]
        inf = np.isposinf(want)
        if np.array_equal(np.isposinf(got), inf) and np.allclose(
            got[~inf], want[~inf], rtol=1e-12, atol=1e-10
        ):
            continue
        hint = "B * ucb1_index(counts, sums / B, t): rescale, take the index, scale back."
        for key, text in _RANGE_HINTS.items():
            alt = fx[key][i]
            ainf = np.isposinf(alt)
            if np.array_equal(np.isposinf(got), ainf) and np.allclose(got[~ainf], alt[~ainf]):
                hint = text
        raise CheckFailed(
            f"{what} returned {np.round(got, 5).tolist()}; expected {np.round(want, 5).tolist()}, "
            f"sums / counts + B sqrt(2 ln t / counts). Hint: {hint}"
        )


_SCALE_WHY = {
    "eps_greedy": (
        "multiplying every mean by 10 changes no comparison between means, and the coin flips "
        "are the same"
    ),
    "ucb1_rescaled": (
        "ucb1_for_range at B = 10 multiplies the mean and the bonus by 10, so every comparison "
        "is the same"
    ),
}


def check_scale_directions(scale_directions: Callable[[], Any]) -> None:
    """Exact: does each method pull the same arms on every seed when rewards are x 10? The
    direction of the regret is the Explain's question, not the checkpoint's."""
    fx = expected("m04_scale")
    want = dict(zip(fx["keys"].tolist(), fx["directions"].tolist(), strict=True))
    same_seeds = dict(zip(fx["keys"].tolist(), fx["same_arms_seeds"].tolist(), strict=True))
    out = scale_directions()
    if not isinstance(out, Mapping):
        raise CheckFailed(
            f"scale_directions() should return a dict with the keys {list(want)}; got {out!r}."
        )
    if set(out) != set(want):
        raise CheckFailed(
            f"scale_directions() has the keys {sorted(map(str, out))}; it should have exactly "
            f"{list(want)}."
        )
    for key in want:
        if out[key] not in DIRECTIONS:
            raise CheckFailed(
                f'scale_directions()[{key!r}] is {out[key]!r}; write "same" (identical pulls on '
                'every seed) or "changes". Put "up" or "down" in a comment: the checkpoint '
                "grades only whether the choices change."
            )
    seeds, T = fx["seeds"], int(fx["T"])
    ref = f"the reference run (seeds {int(seeds[0])}-{int(seeds[-1])}, {T} rounds, rewards x 10)"
    for key, direction in want.items():
        if out[key] == direction:
            continue
        if direction == "changes":
            raise CheckFailed(
                f"scale_directions()[{key!r}] is 'same'; UCB1's choices change: in {ref} it "
                f"pulled the same arms on {same_seeds[key]} of {len(seeds)} seeds. Its bonus "
                "sqrt(2 ln t / N) was solved for rewards in [0, 1], so it stays the same while the "
                "means and gaps are 10 times larger. Read the lines your run cell printed, and the "
                "Explain, for what that does to the regret."
            )
        raise CheckFailed(
            f"scale_directions()[{key!r}] is 'changes'; in {ref} it pulled identical arms on all "
            f"{same_seeds[key]} seeds, because {_SCALE_WHY[key]}. See the briefing's section 5 and "
            "the lines your run cell printed."
        )


# ------------------------------------------------------------------ 6 the planted bug (exact)
def check_ucb1_suspect(UCB1Suspect: Callable[..., Any]) -> None:
    fx = expected("m04_ucb1_replay")
    arms, rewards, seed = fx["arms"], fx["rewards"], int(fx["seed"])
    T = len(arms)
    after = np.vstack([fx["counts_before"][1:], fx["final_counts"][None, :]])
    agent = UCB1Suspect(N_ARMS)
    prev = np.zeros(N_ARMS)
    for t in range(T):
        got = _arm(agent.act(t), N_ARMS, f"agent.act({t})")
        if got != arms[t]:
            raise CheckFailed(
                f"UCB1's recorded run on reward_stack(R_MEAN, {T}, seed={seed}), round {t}: "
                f"UCB1 pulls arm {int(arms[t])}, the agent pulls arm {got} (agent.counts "
                f"{_ints(agent.counts)}, agent.sums {_ints(agent.sums)})."
            )
        agent.update(got, float(rewards[t]))
        counts = np.asarray(getattr(agent, "counts", None), dtype=float)
        want = after[t]
        if counts.shape == want.shape and np.array_equal(counts, want):
            prev = counts
            continue
        where = f"after round {t} (arm {got} pulled), agent.counts is {counts.tolist()}"
        if counts.shape == want.shape and np.array_equal(counts, prev):
            raise CheckFailed(
                f"{where}; it should be {want.tolist()}. The counts never grow, so every "
                "arm's bonus stays sqrt(2 ln t) and the arm that leads in clicks takes every "
                "pull after round 2. update must count the pull."
            )
        if counts.shape == want.shape and np.all(counts - prev >= 1):
            raise CheckFailed(
                f"{where}; it should be {want.tolist()}. Counts grow for arms not pulled: "
                "update must add 1 to the pulled arm's count only."
            )
        if counts.shape == want.shape and np.array_equal(counts, 2 * want):
            raise CheckFailed(
                f"{where}; it should be {want.tolist()}. Each pull is counted twice: count it "
                "in one place, update."
            )
        raise CheckFailed(
            f"{where}; it should be {want.tolist()}, the number of pulls of each arm so far."
        )
    print(f"The agent replayed UCB1's recorded run for {T} rounds, the same arm in every round;")
    print(f"  final counts {_ints(agent.counts)}.")


# ------------------------------------------------------------------ stretch: Gaussian posterior
def check_gaussian_posterior(gaussian_posterior: Callable[..., Any]) -> None:
    fx = expected("m04_gauss")
    for i, (counts, sums) in enumerate(zip(fx["counts"], fx["sums"], strict=True)):
        noise, prior = float(fx["noise_sd"][i]), float(fx["prior_sd"][i])
        what = (
            f"gaussian_posterior(counts={_ints(counts)}, sums={sums.tolist()}, "
            f"noise_sd={noise:g}, prior_sd={prior:g})"
        )
        mean, sd = _pair(
            gaussian_posterior(counts.copy(), sums.copy(), noise, prior), 3, what, "(mean, sd)"
        )
        want_m, want_sd = fx["mean"][i], fx["sd"][i]
        if np.allclose(mean, want_m, rtol=0, atol=1e-12) and np.allclose(
            sd, want_sd, rtol=0, atol=1e-12
        ):
            continue
        if np.allclose(sd, want_sd**2) and not np.allclose(sd, want_sd):
            hint = "you returned the variance; the standard deviation is np.sqrt(1 / precision)."
        elif np.allclose(mean, fx["sample_mean"][i]) and not np.allclose(mean, want_m):
            hint = (
                "the posterior mean is not the sample mean: the prior pulls it toward 0, "
                "mean = (sums / noise_sd**2) / precision."
            )
        else:
            hint = (
                "precision = 1 / prior_sd**2 + counts / noise_sd**2; mean = (sums / noise_sd**2) "
                "/ precision; sd = sqrt(1 / precision)."
            )
        raise CheckFailed(
            f"{what} returned mean {np.round(mean, 5).tolist()} and sd {np.round(sd, 5).tolist()}; "
            f"expected {np.round(want_m, 5).tolist()} and {np.round(want_sd, 5).tolist()}. "
            f"Hint: {hint}"
        )


__all__ = [
    "GAPS",
    "N_ARMS",
    "R_MEAN",
    "argmax_tie",
    "check_beta_posterior",
    "check_eps_greedy",
    "check_gaussian_posterior",
    "check_run_bandit",
    "check_scale_directions",
    "check_thompson_choices",
    "check_thompson_regret",
    "check_ucb1_for_range",
    "check_ucb1_counts",
    "check_ucb1_index",
    "check_ucb1_suspect",
    "check_verdict",
    "reward_stack",
]
