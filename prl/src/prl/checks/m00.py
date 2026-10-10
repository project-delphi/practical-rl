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


def check_run_episode(fn) -> None:
    """Calls the participant's run_episode on fresh CartPole environments."""
    import gymnasium as gym

    class StepCounter(gym.Wrapper):
        """Counts the steps since the last reset; CartPole pays +1 for each one."""

        def reset(self, **kwargs):
            self.steps = 0
            return self.env.reset(**kwargs)

        def step(self, action):
            self.steps += 1
            return self.env.step(action)

    env = StepCounter(gym.make("CartPole-v1"))
    returns = []
    for seed in range(5):
        ret = fn(env, seed)
        returns.append(ret)
        if np.isfinite(ret) and float(ret) != getattr(env, "steps", None):
            raise CheckFailed(
                f"Seed {seed}: your episode took {getattr(env, 'steps', 0)} steps but returned "
                f"{ret:g}. CartPole-v1 pays +1 per step, so they must match: add up the reward "
                "of every step, not just the last one."
            )
    check_episode_returns(returns, 5)
    if fn(env, 3) != fn(env, 3):
        raise CheckFailed(
            "The same seed gave different returns. Seed both env.reset and env.action_space."
        )
    env.close()


def check_returns_by_seed(fn) -> None:
    """Calls the participant's returns_by_seed with a small budget."""
    arr = fn(3, 4, 0)
    check_seed_bands(arr, 3, 4)
