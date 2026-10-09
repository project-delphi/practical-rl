"""Evaluation discipline: seeds, robust aggregates, confidence intervals and honest comparisons.

The comparison rule used across the workshop (PLAN.md §3):
- 1 seed per side: "single seed, not a ranking".
- 2-4 seeds per side: show every seed; overlapping min-max ranges are a tie; no CI.
- 5+ seeds per side: IQM with percentile-bootstrap CIs; overlapping CIs are a tie.
IQM and the stratified bootstrap follow Agarwal et al. (2021, s4.1, s4.3);
rliable itself is archived, so they are reimplemented here with NumPy/SciPy.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats


def run_seeds(fn: Callable[[int], float], seeds: Iterable[int]) -> np.ndarray:
    """Call fn(seed) for each seed; return the results as an array."""
    return np.array([float(fn(int(s))) for s in seeds])


def iqm(x: Sequence[float] | np.ndarray) -> float:
    """Interquartile mean: the mean of the middle 50% of values."""
    arr = np.asarray(x, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError("iqm of an empty array")
    return float(stats.trim_mean(arr, 0.25))


def bootstrap_ci(
    x: Sequence[float] | np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
    *,
    level: float = 0.95,
    n_boot: int = 5000,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for statistic(x)."""
    arr = np.asarray(x, dtype=float).ravel()
    if arr.size < 2:
        raise ValueError("need at least 2 values for a bootstrap CI")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    boots = np.array([statistic(arr[i]) for i in idx])
    lo, hi = np.quantile(boots, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


def stratified_bootstrap(
    scores: dict[str, Sequence[float]] | Sequence[Sequence[float]],
    statistic: Callable[[np.ndarray], float] = iqm,
    *,
    level: float = 0.95,
    n_boot: int = 5000,
    seed: int = 0,
) -> tuple[float, float]:
    """Bootstrap CI that resamples runs within each task (stratum), then pools.

    With a single task this is an ordinary bootstrap over runs.
    """
    groups = list(scores.values()) if isinstance(scores, dict) else list(scores)
    arrays = [np.asarray(g, dtype=float).ravel() for g in groups]
    if any(a.size < 2 for a in arrays):
        raise ValueError("each task needs at least 2 runs")
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        pooled = np.concatenate([a[rng.integers(0, a.size, a.size)] for a in arrays])
        boots[b] = statistic(pooled)
    lo, hi = np.quantile(boots, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


@dataclass(frozen=True)
class Budget:
    """What a comparison cost. Every comparison sentence states it."""

    steps: int | None = None
    seeds: int | None = None
    wall_clock_s: float | None = None
    hardware: str | None = None

    def sentence(self) -> str:
        parts = []
        if self.steps is not None:
            parts.append(f"{self.steps:,} environment steps per run")
        if self.seeds is not None:
            parts.append(f"{self.seeds} seed{'s' if self.seeds != 1 else ''} per method")
        if self.wall_clock_s is not None:
            parts.append(f"{self.wall_clock_s:.0f} s wall-clock")
        if self.hardware:
            parts.append(f"on {self.hardware}")
        return "Budget: " + ", ".join(parts) + "." if parts else "Budget: not stated."


@dataclass(frozen=True)
class Verdict:
    winner: str  # "A", "B", "tie" or "single seed"
    sentence: str
    summary_a: tuple[float, float, float]  # (center, low, high)
    summary_b: tuple[float, float, float]


def _summary(x: np.ndarray, use_ci: bool, seed: int) -> tuple[float, float, float]:
    if use_ci:
        lo, hi = bootstrap_ci(x, iqm, seed=seed)
        return iqm(x), lo, hi
    return float(np.mean(x)), float(np.min(x)), float(np.max(x))


def compare(
    a: Sequence[float],
    b: Sequence[float],
    *,
    names: tuple[str, str] = ("A", "B"),
    budget: Budget | None = None,
    higher_is_better: bool = True,
    seed: int = 0,
) -> Verdict:
    """Compare two methods' per-seed results using the workshop rule."""
    xa, xb = np.asarray(a, dtype=float).ravel(), np.asarray(b, dtype=float).ravel()
    if xa.size == 0 or xb.size == 0:
        raise ValueError("both sides need at least one result")
    tail = f" {budget.sentence()}" if budget else ""
    if min(xa.size, xb.size) == 1:
        sa, sb = _summary(xa, False, seed), _summary(xb, False, seed)
        return Verdict("single seed", f"Single seed, not a ranking.{tail}", sa, sb)
    use_ci = min(xa.size, xb.size) >= 5
    sa, sb = _summary(xa, use_ci, seed), _summary(xb, use_ci, seed)
    what = "IQM [95% bootstrap CI]" if use_ci else "mean [min, max over seeds]"
    overlap = not (sa[2] < sb[1] or sb[2] < sa[1])
    desc = (
        f"{names[0]} {sa[0]:.3g} [{sa[1]:.3g}, {sa[2]:.3g}] vs "
        f"{names[1]} {sb[0]:.3g} [{sb[1]:.3g}, {sb[2]:.3g}] ({what}; "
        f"{xa.size} vs {xb.size} seeds)"
    )
    if overlap:
        return Verdict("tie", f"Tie: the intervals overlap. {desc}.{tail}", sa, sb)
    a_better = (sa[0] > sb[0]) == higher_is_better
    winner, name = ("A", names[0]) if a_better else ("B", names[1])
    return Verdict(winner, f"{name} is better at this budget: {desc}.{tail}", sa, sb)


def rollout_returns(
    env_fn: Callable[[], object],
    policy: Callable[[object], int],
    *,
    episodes: int,
    seed: int,
    gamma: float = 1.0,
    max_steps: int = 10_000,
) -> np.ndarray:
    """Discounted return of `policy` over `episodes` fresh episodes of env_fn()."""
    env = env_fn()
    out = np.empty(episodes)
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        g, disc = 0.0, 1.0
        for _ in range(max_steps):
            obs, r, terminated, truncated, _ = env.step(policy(obs))
            g += disc * float(r)
            disc *= gamma
            if terminated or truncated:
                break
        out[ep] = g
    close = getattr(env, "close", None)
    if callable(close):
        close()
    return out
