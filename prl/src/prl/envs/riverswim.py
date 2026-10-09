"""RiverSwim: a chain where the good reward is far upstream.

The layout follows Strehl & Littman (2008, s6, Fig. 1): a chain of states,
a small reward at the left end, a large one at the right end, and a current
that makes swimming right unreliable. The transition probabilities and
rewards below are OUR choices, stated here and on the Module 6 page; the
paper's figure has not been read for its exact numbers (OPEN_QUESTIONS B11).

Actions: 0 = swim left (always succeeds), 1 = swim right.
Swimming right in an interior state: right with p_right, stay with p_stay,
left with p_left. At the left end: right with p_right + p_left, else stay.
At the right end: stay with p_stay + p_right (and collect r_right), else left.
"""

from __future__ import annotations

import numpy as np

from .tabular import TabularMDP

LEFT, RIGHT = 0, 1


class RiverSwim(TabularMDP):
    def __init__(
        self,
        n: int = 6,
        *,
        p_right: float = 0.35,
        p_stay: float = 0.6,
        p_left: float = 0.05,
        r_left: float = 0.005,
        r_right: float = 1.0,
        gamma: float = 0.95,
    ):
        if n < 2:
            raise ValueError("need at least 2 states")
        if not np.isclose(p_right + p_stay + p_left, 1.0):
            raise ValueError("p_right + p_stay + p_left must be 1")
        P = np.zeros((n, 2, n))
        R = np.zeros((n, 2))
        for s in range(n):
            P[s, LEFT, max(s - 1, 0)] = 1.0
            if s == 0:
                P[s, RIGHT, 1] += p_right + p_left
                P[s, RIGHT, 0] += p_stay
            elif s == n - 1:
                P[s, RIGHT, s] += p_stay + p_right
                P[s, RIGHT, s - 1] += p_left
            else:
                P[s, RIGHT, s + 1] += p_right
                P[s, RIGHT, s] += p_stay
                P[s, RIGHT, s - 1] += p_left
        R[0, LEFT] = r_left
        R[n - 1, RIGHT] = r_right * (p_stay + p_right)
        super().__init__(
            P=P,
            R=R,
            gamma=gamma,
            start=0,
            action_names=["left", "right"],
            meta=dict(
                n=n, p_right=p_right, p_stay=p_stay, p_left=p_left, r_left=r_left, r_right=r_right
            ),
        )
        self.n = n
        self.r_right = r_right

    def sample_step(self, s: int, a: int, rng: np.random.Generator) -> tuple[int, float, bool]:
        s2 = int(rng.choice(self.n, p=self.P[s, a]))
        if a == LEFT:
            r = self.R[s, LEFT]
        else:
            # The right-end reward is paid when you hold your position there.
            r = self.r_right if (s == self.n - 1 and s2 == s) else 0.0
        return s2, float(r), False
