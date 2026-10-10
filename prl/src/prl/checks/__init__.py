"""Checkpoint helpers.

Every check raises CheckFailed with a message that says what was expected,
what was found, and what to look at next. Expected values come from
precomputed fixtures in prl/_expected (written by scripts/make_expected.py),
never from solver code shipped in this package.
"""

from __future__ import annotations

from functools import cache
from importlib import resources
from typing import Any

import numpy as np


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


__all__ = [
    "CheckFailed",
    "assert_close",
    "assert_shape",
    "assert_simplex",
    "assert_threshold",
    "assert_true",
    "expected",
]
