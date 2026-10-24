"""Datasets and recorded bands that ship inside prl, and the containers labs use for logged data.

Both live in prl/datasets/ with one manifest (entry id -> file, sha256, bytes, kind, ...):
- datasets (kind "dataset"), built by scripts/make_datasets.py from known behavior policies,
  so off-policy evaluation and offline RL can be checked against ground truth; load(name);
- recorded multi-seed bands (kind "band", in prl/datasets/bands/), copied from experiment
  records in runs/ by scripts/export_bands.py; band(name, budget).
A Colab install of prl never sees the repository's runs/ folder, so these copies are what a
lab reads. Every read checks the file's sha256 against the manifest.
"""

from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import resources
from typing import Any

import numpy as np

from ._budget import differences, plain
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


def _root():
    """The folder that holds the manifest, the datasets and bands/ (tests point it elsewhere)."""
    return resources.files("prl") / "datasets"


def manifest() -> dict[str, Any]:
    """Every shipped dataset and band: entry id -> {"kind", "file", "sha256", "bytes", ...}."""
    try:
        return json.loads((_root() / "manifest.json").read_text())
    except FileNotFoundError:
        return {}


def _read_checked(what: str, entry: Mapping[str, Any]) -> dict[str, np.ndarray]:
    blob = (_root() / entry["file"]).read_bytes()
    digest = hashlib.sha256(blob).hexdigest()
    if digest != entry["sha256"]:
        raise ValueError(f"{what} is corrupted: sha256 {digest[:12]} != {entry['sha256'][:12]}.")
    with np.load(io.BytesIO(blob), allow_pickle=False) as data:
        return {k: data[k].copy() for k in data.files}


def load(name: str) -> dict[str, np.ndarray]:
    """Load dataset `name`, checking its sha256 against the manifest."""
    entries = {k: e for k, e in manifest().items() if e.get("kind", "dataset") == "dataset"}
    if name not in entries:
        known = ", ".join(sorted(entries)) or "none yet"
        raise KeyError(f"Unknown dataset {name!r}. Available: {known}.")
    return _read_checked(f"Dataset {name!r}", entries[name])


def band(name: str, budget: Mapping[str, Any]) -> dict[str, Any]:
    """The recorded multi-seed band `name` at your budget, ready for prl.plot.with_recorded.

    Pass the lab's settings, `prl.runtime.settings(...)`. A band is recorded at each of the
    lab's budgets (live and QUICK), and this returns the recording whose budget matches yours,
    so lab code never branches on QUICK. A recorded budget matches when each of its keys is in
    `budget` with an equal value; other keys of `budget` are ignored. If none matches, the error
    lists the recorded budgets.

    Returns a dict:
        x           the grid, shape (n,)
        runs        float array (n_seeds, n); runs[i] is seeds[i]'s curve
        seeds       the recorded seeds (list of int)
        budget      the recorded budget (dict)
        metric      what runs holds, in words
        record      id of the experiment record in runs/ it was copied from
        record_sha  sha256 of that record (see scripts/export_bands.py)
        hardware    where the record ran: {"platform", "env", "cpu", "n_cpu", "accel",
                    "gpu", "torch_threads"}; platform "local" is a laptop, not Colab
        seconds     its wall-clock: {"total": the whole record's seconds (every run in it,
                    not only this band's), "per_seed": a list like seeds, or None}
    Together with budget and seeds, these give the budget sentence (steps, seeds,
    wall-clock, hardware) without runs/. Any value the record lacks is None.
    A per-seed scalar is a band with one x value. The file's sha256 is checked.
    """
    bands = [e for e in manifest().values() if e.get("kind") == "band"]
    entries = sorted((e for e in bands if e["name"] == name), key=lambda e: e["label"])
    if not entries:
        known = ", ".join(sorted({e["name"] for e in bands})) or "none yet"
        raise KeyError(f"Unknown band {name!r}. Available: {known}.")
    matches = [e for e in entries if not differences(e["budget"], budget)]
    if len(matches) != 1:
        keys = sorted({k for e in entries for k in e["budget"]})
        yours = {k: plain(budget[k]) if k in budget else "<missing>" for k in keys}
        recorded = "; ".join(f"{e['label']} {e['budget']}" for e in entries)
        if not matches:
            raise ValueError(
                f"No recorded band {name!r} at your budget {yours}. Recorded budgets: {recorded}."
            )
        raise ValueError(f"Band {name!r}: {len(matches)} recordings match {yours}: {recorded}.")
    entry = matches[0]
    arrays = _read_checked(f"Band {name!r} ({entry['label']})", entry)
    x, runs, seeds = arrays["x"], arrays["runs"], [int(s) for s in arrays["seeds"]]
    if runs.shape != (len(seeds), x.shape[0]) or seeds != list(entry["seeds"]):
        raise ValueError(f"Band {name!r} ({entry['label']}) does not match its manifest entry.")
    return {
        "x": x,
        "runs": runs,
        "seeds": seeds,
        "budget": dict(entry["budget"]),
        "metric": entry["metric"],
        "record": entry["record"],
        "record_sha": entry["record_sha"],
        "hardware": dict(entry["hardware"]),
        "seconds": dict(entry["seconds"]),
    }


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


__all__ = ["LoggedBandit", "Transitions", "band", "generate_logs", "load", "manifest"]
