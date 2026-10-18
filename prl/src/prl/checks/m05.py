"""Checkpoints for Module 5: a placeholder until the technical expert builds them.

It exists only so that Lab 5's `from prl.checks import m05` imports and the notebook runs
in learner mode. Every check fails with "not built yet". The contracts are in the academic
director's handoff; the technical expert replaces this file.
"""

from __future__ import annotations

from typing import Any

from . import CheckFailed


def _not_built(*_args: Any) -> None:
    raise CheckFailed("This checkpoint is not built yet (prl.checks.m05 is a placeholder).")


check_reward_fn = _not_built
check_ridge = _not_built
check_linucb_scores = _not_built
check_lints_sample = _not_built
check_verdicts = _not_built
check_log_policy = _not_built
check_log_suspect = _not_built
