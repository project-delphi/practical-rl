"""Checkpoint helpers.

Every check raises CheckFailed with a message that says what was expected,
what was found, and what to look at next. Expected values come from
precomputed fixtures in prl/_expected (written by scripts/make_expected.py),
never from solver code shipped in this package.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from functools import cache
from importlib import resources
from typing import Any

import numpy as np

from .. import runtime


class CheckFailed(AssertionError):
    """A checkpoint failed. The message is meant for the participant."""


def expected(name: str) -> dict[str, np.ndarray]:
    """Load the fixture prl/_expected/<name>.npz as a dict of arrays."""
    return {k: v.copy() for k, v in _load_expected(name)}  # copies: a check must not see edits


@cache
def _load_expected(name: str) -> tuple[tuple[str, np.ndarray], ...]:
    ref = resources.files("prl") / "_expected" / f"{name}.npz"
    with resources.as_file(ref) as path, np.load(path, allow_pickle=False) as data:
        return tuple((k, data[k].copy()) for k in data.files)


def _fmt(x: Any) -> str:
    arr = np.asarray(x)
    if arr.size <= 8:
        return np.array2string(arr, precision=6, suppress_small=True)
    return f"array of shape {arr.shape}"


def assert_shape(x: Any, shape: tuple[int, ...], what: str = "result") -> None:
    arr = np.asarray(x)
    if arr.shape != tuple(shape):
        raise CheckFailed(f"{what} should have shape {tuple(shape)}, but has shape {arr.shape}.")


def assert_close(
    actual: Any,
    expected_value: Any,
    *,
    atol: float = 1e-8,
    rtol: float = 1e-6,
    what: str = "result",
    hint: str | None = None,
) -> None:
    a = np.asarray(actual, dtype=float)
    e = np.asarray(expected_value, dtype=float)
    if a.shape != e.shape:
        raise CheckFailed(f"{what} should have shape {e.shape}, but has shape {a.shape}.")
    if not np.all(np.isfinite(a)):
        raise CheckFailed(f"{what} contains NaN or infinity.")
    if not np.allclose(a, e, atol=atol, rtol=rtol):
        diff = np.max(np.abs(a - e))
        if a.size > 8:
            i = np.unravel_index(np.argmax(np.abs(a - e)), a.shape)
            where = ", ".join(map(str, i))
            msg = (
                f"{what} is off by up to {diff:.3g} (tolerance {atol:g} + {rtol:g} x |expected|). "
                f"Worst entry: {what}[{where}] should be {e[i]:.6g}, got {a[i]:.6g}."
            )
        else:
            msg = (
                f"{what} is off by up to {diff:.3g} (tolerance {atol:g} + {rtol:g} x |expected|). "
                f"Expected {_fmt(e)}, got {_fmt(a)}."
            )
        if hint:
            msg += f" Hint: {hint}"
        raise CheckFailed(msg)


def assert_simplex(
    p: Any, *, axis: int = -1, atol: float = 1e-8, what: str = "probabilities"
) -> None:
    arr = np.asarray(p, dtype=float)
    if np.any(arr < -atol):
        raise CheckFailed(f"{what} contain negative entries (min {arr.min():.3g}).")
    sums = arr.sum(axis=axis)
    if not np.allclose(sums, 1.0, atol=atol):
        worst = np.max(np.abs(sums - 1.0))
        raise CheckFailed(
            f"{what} should sum to 1 along axis {axis}; worst sum is off by {worst:.3g}."
        )


def assert_threshold(
    value: float,
    threshold: float,
    *,
    provenance: str,
    higher_is_better: bool = True,
    what: str = "result",
) -> None:
    """Compare a stochastic outcome with a threshold set by threshold_protocol.

    `provenance` names the experiment record the threshold came from.
    """
    ok = value >= threshold if higher_is_better else value <= threshold
    if not ok:
        word = "at least" if higher_is_better else "at most"
        raise CheckFailed(
            f"{what} = {value:.4g}; a correct solution reaches {word} {threshold:.4g} "
            f"(threshold from {provenance}). Check the Diagnostics cell before rerunning."
        )


def assert_true(cond: bool, message: str) -> None:
    if not cond:
        raise CheckFailed(message)


@dataclass(frozen=True, eq=False)
class StochasticCheck:
    """A checkpoint whose outcome depends on the seed: a metric compared with a threshold.

    Declare it once in prl/checks/mNN.py. It is used two ways:

    - as the checkpoint: ``lab.check(n, mNN.check_x, fn)`` calls it, which computes
      ``metric(fn, seed=self.seed, **budget)`` at the live or QUICK budget (from
      ``runtime.settings``) and compares the value with that budget's threshold;
    - by scripts/threshold_protocol.py, which computes the same metric over many seeds
      for the reference solution and for every mutant of `functions`, and prints the
      ``thresholds`` entry to paste here, with its provenance comment.

    `metric` takes the participant's functions positionally, in the order of
    `functions`, plus ``seed=`` and the budget's keys, and returns a float. It must take
    all of its randomness from `seed`. It may raise CheckFailed for an answer that is
    wrong whatever the seed (a wrong shape, say). `min_gap` is the smallest acceptable
    distance, in the metric's units, between the solution and the strongest wrong
    version. `thresholds` maps "live" and "quick" to (threshold, provenance).
    """

    ex: int | str
    functions: str | tuple[str, ...]
    metric: Callable[..., float]
    live: Mapping[str, Any]
    quick: Mapping[str, Any]
    higher_is_better: bool
    min_gap: float
    what: str = "result"
    designed: str = "colab-cpu"
    seed: int = 0
    thresholds: Mapping[str, tuple[float, str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        names = (self.functions,) if isinstance(self.functions, str) else tuple(self.functions)
        object.__setattr__(self, "functions", names)
        object.__setattr__(self, "ex", str(self.ex))
        if not names:
            raise ValueError("a StochasticCheck needs at least one function name")
        if not self.min_gap > 0:
            raise ValueError("min_gap must be a positive number in the metric's units")
        if set(self.live) != set(self.quick):
            raise ValueError("live and quick budgets must have the same keys")
        if "seed" in self.live:
            raise ValueError("'seed' cannot be a budget key; the check passes it separately")
        unknown = set(self.thresholds) - {"live", "quick"}
        if unknown:
            raise ValueError(f"thresholds keys must be 'live' or 'quick', not {sorted(unknown)}")

    def settings(self) -> runtime.Settings:
        """The budget this runtime uses: live, or QUICK (same keys, smaller numbers)."""
        return runtime.settings(dict(self.live), dict(self.quick), designed=self.designed)

    def measure(self, *fns: Any, seed: int, budget: Mapping[str, Any]) -> float:
        """The metric of `fns` at one seed and budget (the protocol calls this too)."""
        return float(self.metric(*fns, seed=seed, **budget))

    def __call__(self, *fns: Any) -> None:
        budget = self.settings()
        if budget.source not in self.thresholds:
            raise RuntimeError(
                f"This checkpoint has no threshold for the {budget.source} budget yet. "
                "(Authors: run scripts/threshold_protocol.py and paste its entry.)"
            )
        threshold, provenance = self.thresholds[budget.source]
        value = self.measure(*fns, seed=self.seed, budget=budget)
        assert_threshold(
            value,
            threshold,
            provenance=provenance,
            higher_is_better=self.higher_is_better,
            what=self.what,
        )


__all__ = [
    "CheckFailed",
    "StochasticCheck",
    "assert_close",
    "assert_shape",
    "assert_simplex",
    "assert_threshold",
    "assert_true",
    "expected",
]
