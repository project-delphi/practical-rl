"""Checkpoints for Module 0 (setup and "hello RL")."""

from __future__ import annotations

from typing import Any

import numpy as np

from . import CheckFailed, assert_shape, assert_true


def check_runtime(rt: Any) -> None:
    d = rt.as_dict() if hasattr(rt, "as_dict") else dict(rt)
    for key in ("platform", "accel", "cpu", "n_cpu", "python"):
        assert_true(
            key in d and d[key] not in (None, ""),
            f"runtime is missing {key!r}; rerun the setup cell.",
        )
    assert_true(
        d["platform"] in {"colab", "local", "aws", "ci"}, f"unknown platform {d['platform']!r}"
    )
    assert_true(int(d["n_cpu"]) >= 1, "n_cpu should be at least 1")


def check_episode_returns(returns: Any, n_episodes: int) -> None:
    arr = np.asarray(returns, dtype=float)
    assert_shape(arr, (n_episodes,), what="returns")
    assert_true(bool(np.all(np.isfinite(arr))), "returns contain NaN or infinity")
    # CartPole-v1 pays +1 per step and truncates at 500 steps.
    if np.any(arr < 1) or np.any(arr > 500):
        raise CheckFailed(
            f"CartPole-v1 returns must lie in [1, 500]; got min {arr.min():g}, max {arr.max():g}. "
            "Did you add the reward of every step, and stop when terminated or truncated?"
        )


def check_seed_bands(returns_by_seed: Any, n_seeds: int, n_episodes: int) -> None:
    arr = np.asarray(returns_by_seed, dtype=float)
    assert_shape(arr, (n_seeds, n_episodes), what="returns_by_seed")
    if n_seeds > 1 and np.all(arr == arr[0]):
        raise CheckFailed("Every seed gave identical returns. Pass a different seed to each run.")


def check_throughput(steps_per_second: float) -> None:
    assert_true(
        np.isfinite(steps_per_second) and steps_per_second > 0,
        "throughput should be a positive number of steps per second",
    )


def check_record(rec: dict) -> None:
    from ..record import validate

    problems = validate(rec)
    if problems:
        raise CheckFailed("Your run record is not valid: " + "; ".join(problems))
