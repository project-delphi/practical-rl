"""Checkpoints for Module 1 (MDPs, returns and the Bellman equation).

Phase 1 stub: only Exercise 1 (discounted return). Phase 2 adds the rest.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from . import CheckFailed, assert_close, expected


def check_discounted_return(fn: Callable[[list[float], float], float]) -> None:
    fx = expected("m01_returns")
    rewards, lengths, gammas, answers = fx["rewards"], fx["lengths"], fx["gammas"], fx["returns"]
    for i in range(len(lengths)):
        rs = [float(r) for r in rewards[i, : lengths[i]]]
        got = fn(rs, float(gammas[i]))
        if got is None:
            raise CheckFailed("discounted_return returned None. Did you forget `return`?")
        assert_close(
            got,
            answers[i],
            atol=1e-9,
            what=f"discounted_return({rs}, gamma={gammas[i]})",
            hint="G_0 = r_1 + gamma r_2 + gamma^2 r_3 + ... The first reward is not discounted.",
        )
    if not np.isfinite(fn([], 0.9)):
        raise CheckFailed("discounted_return([]) should be 0.")
