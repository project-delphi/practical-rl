"""Baird's counterexample, as presented in Sutton & Barto (2018, s11.2, Fig. 11.1).

Seven states, eight features, all rewards zero. The "dashed" action moves to
one of the six upper states uniformly; the "solid" action moves to the lower
state (state 7, index 6). The behavior policy takes dashed with probability
6/7 and solid with 1/7, so the next state is uniform over all seven. The
target policy always takes solid. Feature vectors: upper state i has value
2 w_i + w_8; the lower state has w_7 + 2 w_8.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DASHED, SOLID = 0, 1


@dataclass
class Baird:
    gamma: float = 0.99
    n_states: int = 7
    n_features: int = 8
    phi: np.ndarray = field(init=False)
    P_behavior: np.ndarray = field(init=False)
    P_target: np.ndarray = field(init=False)
    mu_behavior: np.ndarray = field(init=False)
    w0: np.ndarray = field(init=False)

    def __post_init__(self):
        S, d = self.n_states, self.n_features
        phi = np.zeros((S, d))
        for i in range(S - 1):
            phi[i, i] = 2.0
            phi[i, d - 1] = 1.0
        phi[S - 1, S - 1] = 1.0
        phi[S - 1, d - 1] = 2.0
        self.phi = phi
        self.P_behavior = np.full(
            (S, S), 1.0 / S
        )  # dashed 6/7 spread over 6 upper states + solid 1/7
        self.P_target = np.zeros((S, S))
        self.P_target[:, S - 1] = 1.0
        self.mu_behavior = np.full(S, 1.0 / S)
        self.w0 = np.array([1, 1, 1, 1, 1, 1, 10, 1], dtype=float)
        self.rewards = np.zeros(S)

    def behavior_probs(self) -> np.ndarray:
        """Probability of (dashed, solid) under the behavior policy."""
        return np.array([6.0 / 7.0, 1.0 / 7.0])

    def target_probs(self) -> np.ndarray:
        return np.array([0.0, 1.0])

    def next_state(self, action: int, rng: np.random.Generator) -> int:
        if action == SOLID:
            return self.n_states - 1
        return int(rng.integers(0, self.n_states - 1))
