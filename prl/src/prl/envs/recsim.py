"""A small contextual-bandit recommender simulator.

Each round a user context x in R^d arrives; you recommend one of n_items; the
user clicks (reward 1) with probability mu(x, a), else 0.

    mu(x, a) = clip(0.5 + scale * theta_a(t)^T x + misspec * g(x, a), 0.02, 0.98)

- With misspec = 0 the click probability is linear in x (inside the clip),
  so linear models (LinUCB, linear TS) are well specified.
- misspec > 0 adds a centered quadratic term g(x, a) = (u_a^T x)^2 - |u_a|^2/d
  that linear models cannot represent.
- drift > 0 rotates each item's parameter over time:
  theta_a(t) = cos(drift * t) * theta_a + sin(drift * t) * theta'_a.

Logged mode (`log`) records contexts, actions, rewards and the behavior
policy's propensities, so off-policy evaluation can be checked against the
exact value (`value`), computed from mu without reward noise.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

Policy = Callable[[np.ndarray, np.random.Generator], np.ndarray]
"""A policy maps (context x, rng) to a probability vector over items."""


@dataclass
class LoggedBandit:
    X: np.ndarray  # (n, d) contexts
    A: np.ndarray  # (n,) logged actions
    R: np.ndarray  # (n,) observed rewards
    P: np.ndarray  # (n,) behavior propensity of the logged action
    probs: np.ndarray  # (n, K) full behavior distribution at each round
    T: np.ndarray  # (n,) round index (time) of each context

    def __len__(self) -> int:
        return int(self.A.shape[0])


class RecSim:
    def __init__(
        self,
        n_items: int = 10,
        d: int = 5,
        *,
        misspec: float = 0.0,
        drift: float = 0.0,
        scale: float = 0.15,
        seed: int = 0,
    ):
        self.n_items, self.d = n_items, d
        self.misspec, self.drift, self.scale = misspec, drift, scale
        world = np.random.default_rng(seed)  # the world's parameters are fixed by `seed`
        self.theta = world.normal(size=(n_items, d))
        self.theta_alt = world.normal(size=(n_items, d))
        self.u = world.normal(size=(n_items, d))
        self.rng = np.random.default_rng(seed + 1)
        self.t = 0
        self._x: np.ndarray | None = None

    def reset(self, seed: int | None = None) -> None:
        """Restart time and the user stream (the world's item parameters stay fixed)."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.t = 0
        self._x = None

    def _theta_at(self, t: int) -> np.ndarray:
        if self.drift == 0.0:
            return self.theta
        return np.cos(self.drift * t) * self.theta + np.sin(self.drift * t) * self.theta_alt

    def sample_context(self, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or self.rng
        return rng.normal(size=self.d) / np.sqrt(self.d)

    def expected_reward(self, x: np.ndarray, t: int | None = None) -> np.ndarray:
        """Click probability of every item for context x at time t (default: now)."""
        t = self.t if t is None else t
        lin = 0.5 + self.scale * (self._theta_at(t) @ x)
        quad = (self.u @ x) ** 2 - (self.u**2).sum(axis=1) / self.d
        return np.clip(lin + self.misspec * self.scale * quad, 0.02, 0.98)

    def context(self) -> np.ndarray:
        """Draw the next user's context."""
        self._x = self.sample_context()
        return self._x

    def step(self, action: int) -> float:
        """Show item `action` to the current user; return the click (0 or 1) and advance time."""
        if self._x is None:
            raise RuntimeError("call context() before step()")
        mu = self.expected_reward(self._x)[int(action)]
        reward = float(self.rng.random() < mu)
        self.t += 1
        self._x = None
        return reward

    def log(self, policy: Policy, n: int, *, seed: int = 0) -> LoggedBandit:
        """Run a stochastic behavior policy for n rounds and record propensities."""
        rng = np.random.default_rng(seed)
        X = np.empty((n, self.d))
        A = np.empty(n, dtype=int)
        R = np.empty(n)
        P = np.empty(n)
        probs_all = np.empty((n, self.n_items))
        T = np.empty(n, dtype=int)
        for i in range(n):
            x = self.sample_context(rng)
            probs = np.asarray(policy(x, rng), dtype=float)
            if (
                probs.shape != (self.n_items,)
                or not np.isclose(probs.sum(), 1.0)
                or np.any(probs < 0)
            ):
                raise ValueError("policy must return a probability vector over items")
            a = int(rng.choice(self.n_items, p=probs))
            mu = self.expected_reward(x, t=self.t)[a]
            X[i], A[i], P[i], probs_all[i], T[i] = x, a, probs[a], probs, self.t
            R[i] = float(rng.random() < mu)
            self.t += 1
        return LoggedBandit(X=X, A=A, R=R, P=P, probs=probs_all, T=T)

    def value(
        self, policy: Policy, *, n: int = 20000, seed: int = 12345, t: int | None = None
    ) -> float:
        """Expected click rate of `policy`, by Monte Carlo over contexts with exact mu (no click noise)."""
        rng = np.random.default_rng(seed)
        total = 0.0
        for _ in range(n):
            x = self.sample_context(rng)
            probs = np.asarray(policy(x, rng), dtype=float)
            total += float(probs @ self.expected_reward(x, t=t))
        return total / n
