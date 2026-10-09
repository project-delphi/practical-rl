"""Workshop environments. Each one is small enough to solve exactly, so every
learned result can be checked against ground truth."""

from __future__ import annotations

import gymnasium as gym

from .baird import Baird
from .gridworld import CliffGridworld
from .inventory import InventoryMDP, demand_pmf
from .recsim import LoggedBandit, RecSim
from .riverswim import RiverSwim
from .tabular import TabularEnv, TabularMDP

_IDS = {
    "prl/CliffGridworld-v0": lambda **kw: TabularEnv(CliffGridworld(**kw), max_episode_steps=500),
    "prl/RiverSwim-v0": lambda **kw: TabularEnv(RiverSwim(**kw), max_episode_steps=1000),
    "prl/Inventory-v0": lambda **kw: TabularEnv(InventoryMDP(**kw), max_episode_steps=200),
}


def register() -> list[str]:
    """Register the tabular environments with Gymnasium; return their IDs."""
    for env_id, entry in _IDS.items():
        if env_id not in gym.registry:
            gym.register(id=env_id, entry_point=entry)
    return list(_IDS)


__all__ = [
    "Baird",
    "CliffGridworld",
    "InventoryMDP",
    "LoggedBandit",
    "RecSim",
    "RiverSwim",
    "TabularEnv",
    "TabularMDP",
    "demand_pmf",
    "register",
]
