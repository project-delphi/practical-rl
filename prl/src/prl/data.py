"""Logged datasets that ship inside prl, and the containers labs use for logged data.

Datasets live in prl/datasets/ with a manifest (name -> file, sha256, bytes).
They are built by scripts/make_datasets.py from known behavior policies, so
off-policy evaluation and offline RL can be checked against ground truth.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from importlib import resources
from typing import Any

import numpy as np

from .envs.recsim import LoggedBandit
from .envs.tabular import TabularMDP


@dataclass
class Transitions:
    """Logged transitions from episodes of a behavior policy."""

    obs: np.ndarray
    act: np.ndarray
    rew: np.ndarray
    next_obs: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    episode: np.ndarray
    behavior_prob: np.ndarray | None = (
        None  # pi_b(a_t | s_t), when the behavior policy is stochastic and known
    )

    def __len__(self) -> int:
        return int(self.act.shape[0])


def manifest() -> dict[str, Any]:
    ref = resources.files("prl") / "datasets" / "manifest.json"
    try:
        return json.loads(ref.read_text())
    except FileNotFoundError:
        return {}


def load(name: str) -> dict[str, np.ndarray]:
    """Load dataset `name`, checking its sha256 against the manifest."""
    entries = manifest()
    if name not in entries:
        known = ", ".join(sorted(entries)) or "none yet"
        raise KeyError(f"Unknown dataset {name!r}. Available: {known}.")
    entry = entries[name]
    ref = resources.files("prl") / "datasets" / entry["file"]
    blob = ref.read_bytes()
    digest = hashlib.sha256(blob).hexdigest()
    if digest != entry["sha256"]:
        raise ValueError(
            f"Dataset {name!r} is corrupted: sha256 {digest[:12]} != {entry['sha256'][:12]}."
        )
    with resources.as_file(ref) as path, np.load(path, allow_pickle=False) as data:
        return {k: data[k].copy() for k in data.files}


def generate_logs(
    mdp: TabularMDP, behavior: np.ndarray, *, episodes: int, horizon: int, seed: int
) -> Transitions:
    """Roll out a stochastic tabular behavior policy (S, A) and log transitions with propensities."""
    probs = mdp.policy_matrix(behavior)
    rng = np.random.default_rng(seed)
    rows: dict[str, list] = {
        k: [] for k in ("obs", "act", "rew", "next_obs", "terminated", "truncated", "episode", "p")
    }
    for ep in range(episodes):
        s = mdp.start
        for t in range(horizon):
            a = int(rng.choice(mdp.n_actions, p=probs[s]))
            s2, r, terminated = mdp.sample_step(s, a, rng)
            truncated = (not terminated) and t == horizon - 1
            for k, v in (
                ("obs", s),
                ("act", a),
                ("rew", r),
                ("next_obs", s2),
                ("terminated", terminated),
                ("truncated", truncated),
                ("episode", ep),
                ("p", probs[s, a]),
            ):
                rows[k].append(v)
            s = s2
            if terminated:
                break
    return Transitions(
        obs=np.array(rows["obs"]),
        act=np.array(rows["act"]),
        rew=np.array(rows["rew"], dtype=float),
        next_obs=np.array(rows["next_obs"]),
        terminated=np.array(rows["terminated"]),
        truncated=np.array(rows["truncated"]),
        episode=np.array(rows["episode"]),
        behavior_prob=np.array(rows["p"], dtype=float),
    )


__all__ = ["LoggedBandit", "Transitions", "generate_logs", "load", "manifest"]
