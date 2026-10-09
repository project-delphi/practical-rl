"""Finite MDPs as arrays, and a Gymnasium wrapper for sampling from them."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np


@dataclass
class TabularMDP:
    """A finite MDP.

    P[s, a, s'] is the transition probability, R[s, a] the expected reward of
    taking a in s, and gamma the discount. `terminal[s]` marks absorbing
    terminal states: every action keeps you there with reward 0.
    """

    P: np.ndarray
    R: np.ndarray
    gamma: float
    terminal: np.ndarray | None = None
    start: int = 0
    state_names: list[str] | None = None
    action_names: list[str] | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_states(self) -> int:
        return int(self.P.shape[0])

    @property
    def n_actions(self) -> int:
        return int(self.P.shape[1])

    def validate(self, atol: float = 1e-8) -> None:
        """Raise ValueError if the arrays do not describe a valid MDP."""
        P, R = np.asarray(self.P), np.asarray(self.R)
        if P.ndim != 3 or P.shape[0] != P.shape[2]:
            raise ValueError(f"P must have shape (S, A, S); got {P.shape}")
        if R.shape != P.shape[:2]:
            raise ValueError(f"R must have shape (S, A) = {P.shape[:2]}; got {R.shape}")
        if not np.all(np.isfinite(P)) or not np.all(np.isfinite(R)):
            raise ValueError("P and R must be finite")
        if np.any(P < -atol):
            raise ValueError(f"P has negative entries (min {P.min():.3g})")
        sums = P.sum(axis=2)
        if not np.allclose(sums, 1.0, atol=atol):
            s, a = np.unravel_index(np.argmax(np.abs(sums - 1.0)), sums.shape)
            raise ValueError(f"P[{s}, {a}, :] sums to {sums[s, a]:.6g}, not 1")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError(f"gamma must be in [0, 1]; got {self.gamma}")
        if not 0 <= self.start < P.shape[0]:
            raise ValueError(f"start state {self.start} is out of range")
        if self.terminal is not None:
            term = np.asarray(self.terminal, dtype=bool)
            if term.shape != (P.shape[0],):
                raise ValueError("terminal must have shape (S,)")
            for s in np.flatnonzero(term):
                if not np.allclose(P[s, :, s], 1.0, atol=atol):
                    raise ValueError(f"terminal state {s} must be absorbing (P[{s}, a, {s}] = 1)")
                if not np.allclose(R[s], 0.0, atol=atol):
                    raise ValueError(f"terminal state {s} must have reward 0")

    def sample_step(self, s: int, a: int, rng: np.random.Generator) -> tuple[int, float, bool]:
        """Sample (next state, reward, terminated). Rewards are the expected R[s, a]."""
        s2 = int(rng.choice(self.n_states, p=self.P[s, a]))
        terminated = bool(self.terminal is not None and self.terminal[s2])
        return s2, float(self.R[s, a]), terminated

    def policy_matrix(self, policy: np.ndarray) -> np.ndarray:
        """Turn a deterministic policy (S,) or stochastic policy (S, A) into (S, A) probabilities."""
        pol = np.asarray(policy)
        if pol.ndim == 1:
            probs = np.zeros((self.n_states, self.n_actions))
            probs[np.arange(self.n_states), pol.astype(int)] = 1.0
            return probs
        if pol.shape != (self.n_states, self.n_actions):
            raise ValueError(f"policy must have shape (S,) or (S, A); got {pol.shape}")
        return pol


class TabularEnv(gym.Env):
    """Gymnasium view of a TabularMDP: Discrete observations and actions."""

    metadata = {"render_modes": []}

    def __init__(self, mdp: TabularMDP, max_episode_steps: int = 200):
        mdp.validate()
        self.mdp = mdp
        self.observation_space = gym.spaces.Discrete(mdp.n_states)
        self.action_space = gym.spaces.Discrete(mdp.n_actions)
        self.max_episode_steps = max_episode_steps
        self._s = mdp.start
        self._t = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._s = self.mdp.start
        self._t = 0
        return self._s, {}

    def step(self, action: int):
        s2, r, terminated = self.mdp.sample_step(self._s, int(action), self.np_random)
        self._s = s2
        self._t += 1
        truncated = (not terminated) and self._t >= self.max_episode_steps
        return s2, r, terminated, truncated, {}
