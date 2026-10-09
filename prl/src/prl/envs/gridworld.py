"""A parametric cliff gridworld (the layout of Sutton & Barto 2018, Example 6.6).

States are grid cells, indexed row * cols + col, with row 0 at the top. The
agent starts in the bottom-left cell; the goal is the bottom-right cell; the
cells between them on the bottom row are the cliff. Each move costs -1.
Stepping into the cliff costs -100 and sends you back to the start (you are
never *in* a cliff cell: those states only lead back to the start). The goal
is absorbing with reward 0.
"""

from __future__ import annotations

import numpy as np

from .tabular import TabularMDP

UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
ACTION_NAMES = ["up", "right", "down", "left"]
_MOVES = {UP: (-1, 0), RIGHT: (0, 1), DOWN: (1, 0), LEFT: (0, -1)}


class CliffGridworld(TabularMDP):
    def __init__(
        self,
        rows: int = 4,
        cols: int = 12,
        *,
        gamma: float = 0.99,
        step_reward: float = -1.0,
        cliff_reward: float = -100.0,
        slip: float = 0.0,
    ):
        if rows < 2 or cols < 3:
            raise ValueError("need rows >= 2 and cols >= 3")
        if not 0.0 <= slip < 1.0:
            raise ValueError("slip must be in [0, 1)")
        self.rows, self.cols, self.slip = rows, cols, slip
        S, A = rows * cols, 4
        start = self.index(rows - 1, 0)
        goal = self.index(rows - 1, cols - 1)
        cliff = [self.index(rows - 1, c) for c in range(1, cols - 1)]
        P = np.zeros((S, A, S))
        R = np.zeros((S, A))
        terminal = np.zeros(S, dtype=bool)
        terminal[goal] = True
        for s in range(S):
            for a in range(A):
                if s == goal:
                    P[s, a, s] = 1.0
                    continue
                if s in cliff:
                    P[s, a, start] = 1.0
                    continue
                # With probability `slip` the move goes in a uniformly random other direction.
                outcomes = [(a, 1.0 - slip)] + [
                    (b, slip / 3) for b in range(A) if b != a and slip > 0
                ]
                for b, p in outcomes:
                    s2, r = self._move(s, b, start, goal, cliff, step_reward, cliff_reward)
                    P[s, a, s2] += p
                    R[s, a] += p * r
        super().__init__(
            P=P,
            R=R,
            gamma=gamma,
            terminal=terminal,
            start=start,
            action_names=list(ACTION_NAMES),
            meta={"rows": rows, "cols": cols, "goal": goal, "cliff": cliff},
        )
        self.goal, self.cliff = goal, cliff

    def index(self, row: int, col: int) -> int:
        return row * self.cols + col

    def coords(self, s: int) -> tuple[int, int]:
        return divmod(int(s), self.cols)

    def _move(self, s, a, start, goal, cliff, step_reward, cliff_reward):
        r, c = self.coords(s)
        dr, dc = _MOVES[a]
        r2 = min(max(r + dr, 0), self.rows - 1)
        c2 = min(max(c + dc, 0), self.cols - 1)
        s2 = self.index(r2, c2)
        if s2 in cliff:
            return start, cliff_reward
        return s2, step_reward
